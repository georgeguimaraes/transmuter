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
    2. `do_resize` - the image gets the `size` of the configuration with the `resample` filter of Pillow. See `resize/4`. For an `image_processor_type` that ends in `Fast`, the weights are those of torchvision, so a pixel can be one step away from Pillow. A `size` with `height` and `width` gives that size. A `size` with `shortest_edge` keeps the aspect ratio, and `longest_edge` limits the long side. With `ensure_multiple_of`, each side becomes the nearest multiple of it, as in the DPT image processor. With `keep_aspect_ratio`, both sides take the scale of `size` that is nearer to 1, so the size depends on the image. The DETR, YOLOS and Grounding DINO image processors compute this size with their own rounding, and YOLOS rounds each side down to a multiple of 16. The SAM image processor gives the long side `longest_edge`, also when the image is smaller. The featurizer reads `image_processor_type` and does the same.
    3. `do_center_crop` - the center region of `crop_size` stays. A configuration with `crop_pct`, such as ConvNeXt, resizes the short side to `shortest_edge / crop_pct` and crops a square of `shortest_edge`. When `shortest_edge` is 384 or more, it resizes to a square of `shortest_edge` and does not crop.
    4. `do_rescale` - the values are multiplied by `rescale_factor`.
    5. `do_normalize` - the values are normalized with `image_mean` and `image_std`.
    6. `do_pad` of the SAM image processor - zeros on the bottom and the right give the image the `pad_size`, 1024x1024 by default.

  The OWLv2 image processor (`Owlv2ImageProcessor`) does the steps in another order, and the featurizer reads `image_processor_type` and does the same. It rescales the image, pads it to a square on the bottom and the right with the value 0.5 (`do_pad`), resizes it with the `:gaussian_linear` method of `resize/4`, and normalizes it.

  The Swin2SR image processor (`Swin2SRImageProcessor`) only rescales and pads. With `do_pad`, the bottom and the right get the `"symmetric"` padding of numpy (the edge pixel repeats: `a b c | c b a`) to the next multiple of `size_divisor` (`pad_size` in older configurations). As in Python transformers, a side that is already a multiple gets `size_divisor` more pixels.

  The result has the shape `{batch, channels, height, width}` and the type `:f32`.
  """

  @compile {:no_warn_undefined, StbImage}

  alias Transmuter.{Hub, Repo}

  defstruct [
    :resize,
    :crop,
    :rescale,
    :normalize,
    :pad,
    resample: :bilinear,
    weights: :pillow,
    rgb: true
  ]

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
  # Swin2SRImageProcessor of Python transformers only rescales and pads.
  def new(config = %{"image_processor_type" => "Swin2SRImageProcessor"}) do
    multiple = config["size_divisor"] || config["pad_size"] || 8

    %__MODULE__{
      rescale: if(Map.get(config, "do_rescale", true), do: config["rescale_factor"] || 1 / 255),
      pad: if(Map.get(config, "do_pad", true), do: {{:next_multiple, multiple}, :symmetric})
    }
  end

  def new(config) do
    crop = if config["do_center_crop"], do: square(config["crop_size"])
    size = config_size(config["size"], crop != nil or is_number(config["crop_pct"]))
    {size, crop} = crop_pct(config["crop_pct"], size, crop)
    size = detr_size(config["image_processor_type"], size)
    size = dpt_size(config, size)

    owlv2(config, %__MODULE__{
      resize: if(Map.get(config, "do_resize", true), do: size),
      resample: Map.fetch!(@resamples, config["resample"] || 2),
      weights: weights(config["image_processor_type"]),
      crop: crop,
      rescale: if(Map.get(config, "do_rescale", true), do: config["rescale_factor"] || 1 / 255),
      normalize: if(Map.get(config, "do_normalize", true), do: mean_std(config)),
      pad: sam_pad(config),
      rgb: Map.get(config, "do_convert_rgb", true)
    })
  end

  # The fast image processors of Python transformers resize with torchvision.
  defp weights(type) when is_binary(type),
    do: if(String.ends_with?(type, "Fast"), do: :torchvision, else: :pillow)

  defp weights(_type), do: :pillow

  # SamImageProcessor of Python transformers pads to pad_size with zeros after it normalizes.
  defp sam_pad(config = %{"image_processor_type" => "SamImageProcessor" <> _}) do
    if Map.get(config, "do_pad", true),
      do: {square(config["pad_size"] || %{"height" => 1024, "width" => 1024}), 0}
  end

  defp sam_pad(_config), do: nil

  # Owlv2ImageProcessor of Python transformers ignores resample.
  defp owlv2(config = %{"image_processor_type" => "Owlv2ImageProcessor"}, featurizer) do
    pad = if Map.get(config, "do_pad", true), do: {:square, 0.5}
    %{featurizer | resample: :gaussian_linear, pad: pad}
  end

  defp owlv2(_config, featurizer), do: featurizer

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

  # The DETR, YOLOS and Grounding DINO image processors compute the size with
  # get_size_with_aspect_ratio.
  # YOLOS also rounds each side down to a multiple of 16. The SAM image processor gives the
  # long side longest_edge, also for a smaller image, and rounds the short side.
  defp detr_size(type, {:edges, shortest, longest}) when is_binary(type) do
    cond do
      String.starts_with?(type, "Yolos") -> {:detr, shortest, longest, 16}
      String.starts_with?(type, ["Detr", "GroundingDino"]) -> {:detr, shortest, longest, 1}
      String.starts_with?(type, "SamImageProcessor") -> {:sam, longest}
      true -> {:edges, shortest, longest}
    end
  end

  defp detr_size(_type, size), do: size

  # As get_resize_output_image_size of the DPT image processor of Python transformers. Each
  # side becomes a multiple of ensure_multiple_of. With keep_aspect_ratio, both sides take the
  # scale that is nearer to 1, so the size depends on the image.
  defp dpt_size(config, {height, width})
       when is_map_key(config, "keep_aspect_ratio") or is_map_key(config, "ensure_multiple_of") do
    if config["do_pad"],
      do:
        raise(ArgumentError, "the featurizer does not support do_pad of preprocessor_config.json")

    multiple = config["ensure_multiple_of"] || 1

    if config["keep_aspect_ratio"],
      do: {:dpt, height, width, multiple},
      else: {multiple_of(height, multiple), multiple_of(width, multiple)}
  end

  defp dpt_size(_config, size), do: size

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
  def size(%__MODULE__{pad: {{height, width}, _value}}) when is_integer(height),
    do: {height, width}

  def size(%__MODULE__{crop: {_, _} = crop}), do: crop
  def size(%__MODULE__{resize: {height, width}}) when is_integer(height), do: {height, width}
  def size(%__MODULE__{}), do: nil

  @doc """
  Returns the `{height, width}` of the image in the result of `process/2` for an image of size `{height, width}`. A `pad_size` (as in SAM) adds padding on the bottom and the right after this size, and `size/1` gives the size with the padding.
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

    image =
      if featurizer.resample == :gaussian_linear do
        image
        |> rescale(featurizer)
        |> pad(featurizer.pad)
        |> resize_step(featurizer)
        |> normalize(featurizer.normalize)
      else
        image
        |> resize_step(featurizer)
        |> center_crop(featurizer.crop)
        |> rescale(featurizer)
        |> normalize(featurizer.normalize)
        |> pad(featurizer.pad)
      end

    Nx.transpose(image, axes: [2, 0, 1])
  end

  defp resize_step(image, %{resize: nil}), do: image

  defp resize_step(image, featurizer) do
    {height, width, _channels} = Nx.shape(image)
    size = output_size(featurizer.resize, {height, width})
    resize(image, size, featurizer.resample, featurizer.weights)
  end

  defp rescale(image, featurizer) do
    image = Nx.as_type(image, :f32)
    if featurizer.rescale, do: Nx.multiply(image, featurizer.rescale), else: image
  end

  # Pads the bottom and the right of the image to a square of its long side
  # (Owlv2ImageProcessor), to a size (pad_size of SamImageProcessor) or to the next multiple
  # of a size (Swin2SRImageProcessor). The padding is a value, or :symmetric, the mode of
  # numpy.pad that reflects the image with its edge pixel: a b c | c b a.
  defp pad(image, nil), do: image

  defp pad(image, {target, value}) do
    {height, width, _channels} = Nx.shape(image)
    {to_height, to_width} = pad_size(target, {height, width})

    if value == :symmetric do
      image |> symmetric(0, to_height) |> symmetric(1, to_width)
    else
      padding = [{0, to_height - height, 0}, {0, to_width - width, 0}, {0, 0, 0}]
      Nx.pad(image, Nx.tensor(value, type: Nx.type(image)), padding)
    end
  end

  defp pad_size(:square, {height, width}), do: {max(height, width), max(height, width)}

  # Swin2SRImageProcessor also adds `multiple` pixels to a side that is already a multiple.
  defp pad_size({:next_multiple, multiple}, {height, width}),
    do: {(div(height, multiple) + 1) * multiple, (div(width, multiple) + 1) * multiple}

  defp pad_size({to_height, to_width}, _size), do: {to_height, to_width}

  defp symmetric(image, axis, to_size) do
    size = Nx.axis_size(image, axis)

    indices =
      for i <- 0..(to_size - 1) do
        i = rem(i, 2 * size)
        if i < size, do: i, else: 2 * size - 1 - i
      end

    Nx.take(image, Nx.tensor(indices, backend: Nx.BinaryBackend), axis: axis)
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

  defp output_size({:dpt, out_height, out_width, multiple}, {height, width}) do
    {scale_height, scale_width} = {out_height / height, out_width / width}

    scale =
      if abs(1 - scale_width) < abs(1 - scale_height), do: scale_width, else: scale_height

    {multiple_of(scale * height, multiple), multiple_of(scale * width, multiple)}
  end

  # As _get_preprocess_shape of SamImageProcessor.
  defp output_size({:sam, longest}, {height, width}) do
    scale = longest / max(height, width)
    {trunc(height * scale + 0.5), trunc(width * scale + 0.5)}
  end

  defp output_size(size, _image_size), do: size

  defp multiple_of(value, multiple), do: round_even(value / multiple) * multiple

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
  defp center_crop(image, nil), do: image

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

  defp normalize(image, nil), do: image

  defp normalize(image, {mean, std}) do
    mean = Nx.tensor(mean, type: :f32, backend: Nx.BinaryBackend)
    std = Nx.tensor(std, type: :f32, backend: Nx.BinaryBackend)
    image |> Nx.subtract(mean) |> Nx.divide(std)
  end

  @doc """
  Resizes an image of shape `{height, width, channels}` to `{height, width}` as `Image.resize/2` of Pillow.

  `method` is `:nearest`, `:box`, `:bilinear`, `:hamming`, `:bicubic` or `:lanczos`. The filters other than `:nearest` widen when the image gets smaller, so they also remove aliasing. The width changes first and the height second.

  A `:u8` image gives a `:u8` image. As in Pillow, the weights are fixed-point numbers with 22 fraction bits, and each pass rounds to `:u8`. An image of another type is resized in `:f32`.

  `weights` is `:pillow`, the default, or `:torchvision`, which the featurizer uses for the fast image processors of Python transformers (an `image_processor_type` that ends in `Fast`). On Apple silicon and on x86 with AVX2, torchvision resizes a `:u8` image with the filters and the taps of Pillow, but its weights are 16-bit integers. So for `:bilinear`, `:bicubic` and `:lanczos`, the weights of an axis get the most fraction bits, up to 22, that keep the largest weight as an integer below 2^15. With fewer bits, a pixel can be one step away from Pillow when the image gets smaller. The other methods and the images of other types resize as with `:pillow`.

  `:gaussian_linear` is the resize of the OWLv2 image processor of Python transformers, which is `skimage.transform.resize` with anti-aliasing. On each axis, a gaussian blur with sigma `max(0, (scale - 1) / 2)` comes first, where `scale` is the input size divided by the output size. Then a linear interpolation between the pixel centers gives the output. Both steps reflect the image at its edges, as the `"mirror"` mode of `scipy.ndimage`. The result is clipped to the minimum and the maximum of the image.
  """
  def resize(image, size, method, weights \\ :pillow)

  def resize(image, size, :gaussian_linear, weights) do
    image = Nx.as_type(image, :f32)
    resized = resize_pillow(image, size, :gaussian_linear, weights)
    Nx.clip(resized, Nx.reduce_min(image), Nx.reduce_max(image))
  end

  def resize(image, size, method, weights) do
    image = if Nx.type(image) == {:u, 8}, do: image, else: Nx.as_type(image, :f32)
    resize_pillow(image, size, method, weights)
  end

  defp resize_pillow(image, {height, width}, method, weights) do
    image
    |> resize_axis(1, width, method, weights)
    |> resize_axis(0, height, method, weights)
  end

  defp resize_axis(image, axis, size, method, mode) do
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

      bits = precision_bits(weights, method, mode)
      weighted_sum(gathered, weights, List.to_tuple(weight_shape), axis + 1, Nx.type(image), bits)
    end
  end

  # Pillow rounds the weights of 8-bit images to integers with 22 fraction bits. torchvision
  # takes the most bits that keep the weights in 16 bits, see
  # _compute_index_ranges_int16_weights in aten/src/ATen/native/cpu/UpSampleKernel.cpp of
  # PyTorch, which its bilinear, bicubic and lanczos filters use for 8-bit images.
  defp precision_bits(weights, method, :torchvision)
       when method in [:bilinear, :bicubic, :lanczos] do
    largest = weights |> List.flatten() |> Enum.max()
    Enum.find(0..21, 22, &(trunc(0.5 + largest * 2 ** (&1 + 1)) >= 2 ** 15))
  end

  defp precision_bits(_weights, _method, _mode), do: 22

  # The weights of 8-bit images are integers with `bits` fraction bits. The sum gets half of
  # the last bit and is clipped to 0..255.
  defp weighted_sum(gathered, weights, shape, axis, {:u, 8}, bits) do
    one = 2 ** bits

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
    |> Nx.right_shift(bits)
    |> Nx.clip(0, 255)
    |> Nx.as_type(:u8)
  end

  defp weighted_sum(gathered, weights, shape, axis, type, _bits) do
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

  # As the gaussian_filter and the zoom with grid_mode of scipy.ndimage, in the "mirror" mode.
  # The zoom interpolates at (i + 0.5) * scale - 0.5 between the blurred pixels, and each
  # blurred pixel is a sum of the pixels around it, so the taps of both steps multiply.
  defp taps(in_size, out_size, :gaussian_linear) do
    scale = in_size / out_size
    blur = gaussian(max(0, (scale - 1) / 2))

    for i <- 0..(out_size - 1) do
      x = (i + 0.5) * scale - 0.5
      low = floor(x)
      t = x - low

      taps =
        for {j, weight} <- [{low, 1 - t}, {low + 1, t}],
            {offset, blur_weight} <- blur,
            do: {mirror(mirror(j, in_size) + offset, in_size), weight * blur_weight}

      taps
      |> Enum.group_by(&elem(&1, 0), &elem(&1, 1))
      |> Enum.map(fn {j, weights} -> {j, Enum.sum(weights)} end)
    end
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

  # As _gaussian_kernel1d of scipy.ndimage, with a radius of 4 sigma. gaussian_filter does not
  # filter an axis whose sigma is not more than 1.0e-15.
  defp gaussian(sigma) when sigma <= 1.0e-15, do: [{0, 1.0}]

  defp gaussian(sigma) do
    radius = trunc(4 * sigma + 0.5)
    weights = for x <- -radius..radius, do: {x, :math.exp(-0.5 / (sigma * sigma) * x * x)}
    total = weights |> Enum.map(&elem(&1, 1)) |> Enum.sum()
    Enum.map(weights, fn {x, weight} -> {x, weight / total} end)
  end

  # The "mirror" mode of scipy.ndimage reflects about the edge pixels: d c b | a b c d | c b a.
  defp mirror(_index, 1), do: 0

  defp mirror(index, size) do
    index = Integer.mod(index, 2 * (size - 1))
    if index < size, do: index, else: 2 * (size - 1) - index
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
