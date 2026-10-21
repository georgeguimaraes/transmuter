defmodule Transmuter.Featurizer do
  @moduledoc """
  Turns images into the `pixel_values` input of a vision model, with the `preprocessor_config.json` of a model repository.

      featurizer = Transmuter.Featurizer.load!({:hf, "Xenova/vit-base-patch16-224"})
      image = StbImage.read_file!("cat.jpg")
      Transmuter.Featurizer.process(featurizer, [image], backend: EMLX.Backend)
      #=> #Nx.Tensor<f32[1][3][224][224]>

  An image is a `StbImage` struct or a tensor of shape `{height, width, channels}`, usually of type `:u8`. To read image files, add `{:stb_image, "~> 1.0"}` to your deps.

  The steps and their order are those of the image processors of Python transformers:

    1. `do_convert_rgb` - the image gets 3 channels. A gray image repeats its channel, and the alpha channel is removed.
    2. `do_resize` - the image gets the `size` of the configuration with the `resample` filter of Pillow. See `resize/3`. A `size` with `height` and `width` gives that size. A `size` with `shortest_edge` keeps the aspect ratio, and `longest_edge` limits the long side. The DETR and YOLOS image processors compute this size with their own rounding, and YOLOS rounds each side down to a multiple of 16. The featurizer reads `image_processor_type` and does the same.
    3. `do_center_crop` - the center region of `crop_size` stays. A configuration with `crop_pct`, such as ConvNeXt, resizes the short side to `shortest_edge / crop_pct` and crops a square of `shortest_edge`. When `shortest_edge` is 384 or more, it resizes to a square of `shortest_edge` and does not crop.
    4. `do_rescale` - the values are multiplied by `rescale_factor`.
    5. `do_normalize` - the values are normalized with `image_mean` and `image_std`.

  The result has the shape `{batch, channels, height, width}` and the type `:f32`.
  """

  @compile {:no_warn_undefined, StbImage}

  alias Transmuter.{Hub, Repo}

  defstruct [:resize, :crop, :rescale, :normalize, resample: :bilinear, rgb: true]

  # The resample codes of Pillow.
  @resamples %{
    0 => :nearest,
    1 => :lanczos,
    2 => :bilinear,
    3 => :bicubic,
    4 => :box,
    5 => :hamming
  }

  @doc """
  Loads the featurizer of a `Transmuter.Repo` or of a source (`{:hf, repo}` or a local directory) from its `preprocessor_config.json`.

  Options go to `Transmuter.Repo.load!/2` when the argument is a source.
  """
  def load!(repo_or_source, opts \\ [])

  def load!(repo = %Repo{}, _opts) do
    config =
      Hub.json(repo.source, "preprocessor_config.json", repo.opts) ||
        raise ArgumentError, "#{inspect(repo.source)} has no preprocessor_config.json"

    new(config)
  end

  def load!(source, opts), do: load!(Repo.load!(source, opts))

  @doc """
  Builds a featurizer from the map of a `preprocessor_config.json`.
  """
  def new(config) do
    crop = if config["do_center_crop"], do: square(config["crop_size"])
    size = config_size(config["size"], crop != nil or is_number(config["crop_pct"]))
    {size, crop} = crop_pct(config["crop_pct"], size, crop)
    size = detr_size(config["image_processor_type"], size)

    %__MODULE__{
      resize: if(Map.get(config, "do_resize", true), do: size),
      resample: Map.fetch!(@resamples, config["resample"] || 2),
      crop: crop,
      rescale: if(Map.get(config, "do_rescale", true), do: config["rescale_factor"] || 1 / 255),
      normalize: if(Map.get(config, "do_normalize", true), do: mean_std(config)),
      rgb: Map.get(config, "do_convert_rgb", true)
    }
  end

  defp config_size(%{"height" => height, "width" => width}, _crops?), do: {height, width}

  defp config_size(size = %{}, _crops?)
       when is_map_key(size, "shortest_edge") or is_map_key(size, "longest_edge"),
       do: {:edges, size["shortest_edge"], size["longest_edge"]}

  # Older configurations give an integer. The image processors that crop, such as CLIP and
  # ConvNeXt, read it as the short side, and the others as a square.
  defp config_size(size, true) when is_integer(size), do: {:edges, size, nil}
  defp config_size(size, false) when is_integer(size), do: {size, size}
  defp config_size(nil, _crops?), do: nil

  defp square(%{"height" => height, "width" => width}), do: {height, width}
  defp square(size) when is_integer(size), do: {size, size}

  # As ConvNextImageProcessor of Python transformers.
  defp crop_pct(pct, {:edges, edge, nil}, _crop) when is_number(pct) and edge < 384,
    do: {{:edges, trunc(edge / pct), nil}, {edge, edge}}

  defp crop_pct(pct, {:edges, edge, nil}, _crop) when is_number(pct), do: {{edge, edge}, nil}
  defp crop_pct(_pct, size, crop), do: {size, crop}

  # The DETR and YOLOS image processors compute the size with get_size_with_aspect_ratio.
  # YOLOS also rounds each side down to a multiple of 16.
  defp detr_size(type, {:edges, shortest, longest}) when is_binary(type) do
    cond do
      String.starts_with?(type, "Yolos") -> {:detr, shortest, longest, 16}
      String.starts_with?(type, "Detr") -> {:detr, shortest, longest, 1}
      true -> {:edges, shortest, longest}
    end
  end

  defp detr_size(_type, size), do: size

  defp mean_std(%{"image_mean" => mean, "image_std" => std}) when is_list(mean) and is_list(std),
    do: {mean, std}

  defp mean_std(_config),
    do:
      raise(
        ArgumentError,
        "preprocessor_config.json sets do_normalize without image_mean and image_std"
      )

  @doc """
  Returns the `{height, width}` of the images that `process/2` gives, or `nil` when the size depends on the image, for example with `shortest_edge` and no crop.
  """
  def size(%__MODULE__{crop: {_, _} = crop}), do: crop
  def size(%__MODULE__{resize: {height, width}}) when is_integer(height), do: {height, width}
  def size(%__MODULE__{}), do: nil

  @doc """
  Returns the `{height, width}` that `process/2` gives for an image of size `{height, width}`.
  """
  def image_size(featurizer, size) do
    cond do
      featurizer.crop -> featurizer.crop
      featurizer.resize -> output_size(featurizer.resize, size)
      true -> size
    end
  end

  @doc """
  Turns a list of images into one tensor of shape `{batch, channels, height, width}` and type `:f32`. The images must give the same size.

  ## Options

    * `:backend` - the backend that does the steps, such as `EMLX.Backend`. The default is the backend of each image. The resize is slow on `Nx.BinaryBackend`.
  """
  def process(featurizer, images, opts \\ []) do
    images
    |> Enum.map(&process_image(featurizer, &1, opts[:backend]))
    |> Nx.stack()
  end

  defp process_image(featurizer, image, backend) do
    image =
      case tensor(image) do
        {:ok, image} when backend == nil -> image
        {:ok, image} -> Nx.backend_transfer(image, backend)
        {:error, message} -> raise ArgumentError, message
      end

    image = if featurizer.rgb, do: rgb(image), else: image
    {height, width, _channels} = Nx.shape(image)

    image =
      if featurizer.resize,
        do: resize(image, output_size(featurizer.resize, {height, width}), featurizer.resample),
        else: image

    image = if featurizer.crop, do: center_crop(image, featurizer.crop), else: image
    image = Nx.as_type(image, :f32)
    image = if featurizer.rescale, do: Nx.multiply(image, featurizer.rescale), else: image
    image = if featurizer.normalize, do: normalize(image, featurizer.normalize), else: image
    Nx.transpose(image, axes: [2, 0, 1])
  end

  @doc """
  Returns `{:ok, tensor}` for an image, a `StbImage` struct or a tensor of shape `{height, width, channels}`. Returns `{:error, message}` for other values.
  """
  def tensor(image = %{__struct__: StbImage}), do: {:ok, StbImage.to_nx(image)}
  def tensor(image = %Nx.Tensor{shape: {_, _, _}}), do: {:ok, image}

  def tensor(image),
    do:
      {:error,
       "expected an StbImage or a tensor of shape {height, width, channels}, got: #{inspect(image)}"}

  @doc """
  Converts an image of shape `{height, width, channels}` to 3 channels, as `Image.convert("RGB")` of Pillow. A gray image repeats its channel, and the alpha channel is removed.
  """
  def rgb(image) do
    {height, width, channels} = Nx.shape(image)

    if channels < 3,
      do: image |> Nx.slice_along_axis(0, 1, axis: 2) |> Nx.broadcast({height, width, 3}),
      else: Nx.slice_along_axis(image, 0, 3, axis: 2)
  end

  # As get_resize_output_image_size of Python transformers. The short side gets
  # shortest_edge, and then the long side gets at most longest_edge.
  defp output_size({:edges, shortest, longest}, {height, width}) do
    {short, long} = {min(height, width), max(height, width)}
    {short, long} = if shortest, do: {shortest, div(shortest * long, short)}, else: {short, long}

    {short, long} =
      if longest && long > longest, do: {div(longest * short, long), longest}, else: {short, long}

    if height <= width, do: {short, long}, else: {long, short}
  end

  # As get_size_with_aspect_ratio of the DETR and YOLOS image processors of Python
  # transformers. When the long side is larger than longest_edge, the short side gets
  # longest_edge times the aspect ratio, rounded half to even as round of Python.
  defp output_size({:detr, size, max_size, multiple}, {height, width}) do
    {short, long} = {min(height, width), max(height, width)}
    raw = if max_size && long / short * size > max_size, do: max_size * short / long
    size = if raw, do: round_even(raw), else: size
    scale = raw || size

    {out_height, out_width} =
      cond do
        min(height, width) == size -> {height, width}
        width < height -> {trunc(scale * height / width), size}
        true -> {size, trunc(scale * width / height)}
      end

    {out_height - rem(out_height, multiple), out_width - rem(out_width, multiple)}
  end

  defp output_size(size, _image_size), do: size

  defp round_even(x) do
    floor = Float.floor(x)
    low = trunc(floor)

    cond do
      x - floor < 0.5 -> low
      x - floor > 0.5 -> low + 1
      rem(low, 2) == 0 -> low
      true -> low + 1
    end
  end

  # As center_crop of Python transformers: an image smaller than the crop is padded with zeros.
  defp center_crop(image, {height, width}) do
    image |> crop_axis(0, height) |> crop_axis(1, width)
  end

  defp crop_axis(image, axis, size) do
    case Nx.axis_size(image, axis) do
      ^size ->
        image

      in_size when in_size > size ->
        Nx.slice_along_axis(image, div(in_size - size, 2), size, axis: axis)

      in_size ->
        low = div(size - in_size + 1, 2)

        padding =
          for a <- 0..2, do: if(a == axis, do: {low, size - in_size - low, 0}, else: {0, 0, 0})

        Nx.pad(image, Nx.tensor(0, type: Nx.type(image)), padding)
    end
  end

  defp normalize(image, {mean, std}) do
    mean = Nx.tensor(mean, type: :f32, backend: Nx.BinaryBackend)
    std = Nx.tensor(std, type: :f32, backend: Nx.BinaryBackend)
    image |> Nx.subtract(mean) |> Nx.divide(std)
  end

  @doc """
  Resizes an image of shape `{height, width, channels}` to `{height, width}` as `Image.resize/2` of Pillow.

  `method` is `:nearest`, `:box`, `:bilinear`, `:hamming`, `:bicubic` or `:lanczos`. The filters other than `:nearest` widen when the image gets smaller, so they also remove aliasing. The width changes first and the height second.

  A `:u8` image gives a `:u8` image. As in Pillow, the weights are fixed-point numbers with 22 fraction bits, and each pass rounds to `:u8`. An image of another type is resized in `:f32`.
  """
  def resize(image, {height, width}, method) do
    image = if Nx.type(image) == {:u, 8}, do: image, else: Nx.as_type(image, :f32)
    image |> resize_axis(1, width, method) |> resize_axis(0, height, method)
  end

  defp resize_axis(image, axis, size, method) do
    in_size = Nx.axis_size(image, axis)

    if in_size == size do
      image
    else
      taps = taps(in_size, size, method)
      width = taps |> Enum.map(&length/1) |> Enum.max()
      taps = Enum.map(taps, &(&1 ++ List.duplicate({0, 0.0}, width - length(&1))))
      indices = for row <- taps, {j, _} <- row, do: j
      weights = for row <- taps, do: Enum.map(row, &elem(&1, 1))

      shape =
        image |> Nx.shape() |> put_elem(axis, size) |> Tuple.insert_at(axis + 1, width)

      weight_shape = for(a <- 0..3, do: if(a in [axis, axis + 1], do: elem(shape, a), else: 1))

      gathered =
        image
        |> Nx.take(Nx.tensor(indices, backend: Nx.BinaryBackend), axis: axis)
        |> Nx.reshape(shape)

      weighted_sum(gathered, weights, List.to_tuple(weight_shape), axis + 1, Nx.type(image))
    end
  end

  @precision_bits 22

  # Pillow rounds the weights of 8-bit images to integers with 22 fraction bits, adds half
  # of the last bit and clips the sum to 0..255.
  defp weighted_sum(gathered, weights, shape, axis, {:u, 8}) do
    one = 2 ** @precision_bits

    weights =
      weights
      |> Enum.map(fn row -> Enum.map(row, &trunc(&1 * one + if(&1 < 0, do: -0.5, else: 0.5))) end)
      |> Nx.tensor(type: :s32, backend: Nx.BinaryBackend)
      |> Nx.reshape(shape)

    gathered
    |> Nx.as_type(:s32)
    |> Nx.multiply(weights)
    |> Nx.sum(axes: [axis])
    |> Nx.add(div(one, 2))
    |> Nx.right_shift(@precision_bits)
    |> Nx.clip(0, 255)
    |> Nx.as_type(:u8)
  end

  defp weighted_sum(gathered, weights, shape, axis, type) do
    weights = weights |> Nx.tensor(type: type, backend: Nx.BinaryBackend) |> Nx.reshape(shape)
    gathered |> Nx.multiply(weights) |> Nx.sum(axes: [axis])
  end

  # {input index, weight} pairs of each output element. For :nearest, Pillow adds the scale
  # to the position at each step, so the rounding errors of the sum decide the pixel at the
  # boundaries.
  defp taps(in_size, out_size, :nearest) do
    scale = in_size / out_size

    (scale * 0.5)
    |> Stream.iterate(&(&1 + scale))
    |> Enum.take(out_size)
    |> Enum.map(&[{min(trunc(&1), in_size - 1), 1.0}])
  end

  # As precompute_coeffs of Pillow.
  defp taps(in_size, out_size, method) do
    {support, filter} = filter(method)
    scale = in_size / out_size
    stretch = max(scale, 1.0)
    support = support * stretch
    inverse = 1.0 / stretch

    for i <- 0..(out_size - 1) do
      center = (i + 0.5) * scale
      low = max(trunc(center - support + 0.5), 0)
      high = min(trunc(center + support + 0.5), in_size)
      taps = for j <- low..(high - 1)//1, do: {j, filter.((j - center + 0.5) * inverse)}
      total = taps |> Enum.map(&elem(&1, 1)) |> Enum.sum()
      if total == 0, do: taps, else: Enum.map(taps, fn {j, w} -> {j, w / total} end)
    end
  end

  defp filter(:box), do: {0.5, &if(&1 > -0.5 and &1 <= 0.5, do: 1.0, else: 0.0)}
  defp filter(:bilinear), do: {1.0, &max(1 - abs(&1), 0.0)}
  defp filter(:hamming), do: {1.0, &hamming/1}
  defp filter(:bicubic), do: {2.0, &bicubic/1}

  defp filter(:lanczos),
    do: {3.0, &if(&1 >= -3 and &1 < 3, do: sinc(&1) * sinc(&1 / 3), else: 0.0)}

  defp hamming(x) do
    x = abs(x)

    cond do
      x == 0 ->
        1.0

      x >= 1 ->
        0.0

      true ->
        :math.sin(x * :math.pi()) / (x * :math.pi()) * (0.54 + 0.46 * :math.cos(x * :math.pi()))
    end
  end

  # Pillow uses a = -0.5.
  defp bicubic(x) do
    a = -0.5
    x = abs(x)

    cond do
      x < 1 -> ((a + 2) * x - (a + 3)) * x * x + 1
      x < 2 -> (((x - 5) * x + 8) * x - 4) * a
      true -> 0.0
    end
  end

  defp sinc(x) when x == 0, do: 1.0
  defp sinc(x), do: :math.sin(:math.pi() * x) / (:math.pi() * x)
end
