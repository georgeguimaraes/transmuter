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
    2. `do_resize` - the image gets the `size` of the configuration with the `resample` filter of Pillow. Without `resample`, the filter is the default of the class of `image_processor_type` in Python transformers, such as bicubic for `CLIPImageProcessor` and lanczos for `Idefics3ImageProcessor`, else bilinear. A fast image processor resizes with bicubic in place of lanczos, because torchvision has no lanczos for tensors. See `resize/4`. For an `image_processor_type` that ends in `Fast`, the weights are those of torchvision, so a pixel can be one step away from Pillow. A `size` with `height` and `width` gives that size. A `size` with `shortest_edge` keeps the aspect ratio, and `longest_edge` limits the long side. With `ensure_multiple_of`, each side becomes the nearest multiple of it, as in the DPT image processor. With `keep_aspect_ratio`, both sides take the scale of `size` that is nearer to 1, so the size depends on the image. The DETR, YOLOS and Grounding DINO image processors compute this size with their own rounding, and YOLOS rounds each side down to a multiple of 16. The SAM image processor gives the long side `longest_edge`, also when the image is smaller. The featurizer reads `image_processor_type` and does the same.
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

  # The default resample of the image processors of Python transformers 4.57, also the
  # deprecated ones, whose default is not bilinear, by the name of the class without
  # "ImageProcessor" and "Fast". The slow and the fast class of a model have the same default.
  @default_resamples Map.merge(
                       Map.new(
                         ~w(Aria Beit Bit Blip BridgeTower CLIP ChineseCLIP Cohere2Vision DPT
                            DeepseekVL DeepseekVLHybrid DeiT EfficientFormer Emu3 Flava Glm4v
                            GotOcr2 InstructBlipVideo Janus Levit Llava LlavaNext LlavaNextVideo
                            LlavaOnevision Ovis2 OwlViT Perceiver PerceptionLM Phi4Multimodal
                            Pixtral PoolFormer PromptDepthAnything Qwen2VL SegGpt Siglip
                            ViTHybrid VideoLlava Vilt),
                         &{&1, 3}
                       ),
                       %{"EfficientNet" => 0, "Chameleon" => 1, "Idefics3" => 1, "SmolVLM" => 1}
                     )

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

  An older configuration names a `feature_extractor_type`, such as `ViTFeatureExtractor`, in place of an `image_processor_type`. As `AutoImageProcessor` of Python transformers, the featurizer reads it as the image processor of the same name, such as `ViTImageProcessor`.
  """
  def new(config = %{"feature_extractor_type" => type})
      when is_binary(type) and not is_map_key(config, "image_processor_type") do
    type = String.replace(type, "FeatureExtractor", "ImageProcessor")
    new(Map.put(config, "image_processor_type", type))
  end

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
      resample: resample(config),
      weights: weights(config["image_processor_type"]),
      crop: crop,
      rescale: if(Map.get(config, "do_rescale", true), do: config["rescale_factor"] || 1 / 255),
      normalize: if(Map.get(config, "do_normalize", true), do: mean_std(config)),
      pad: sam_pad(config),
      rgb: Map.get(config, "do_convert_rgb", true)
    })
  end

  # A missing resample takes the default of the class of image_processor_type, as in Python.
  # torchvision has no lanczos for tensors, so the fast image processors of Python transformers
  # resize with bicubic in its place, as Idefics3ImageProcessorFast does.
  defp resample(config) do
    type = config["image_processor_type"] || ""

    class =
      type |> String.replace_suffix("Fast", "") |> String.replace_suffix("ImageProcessor", "")

    resample = Map.fetch!(@resamples, config["resample"] || Map.get(@default_resamples, class, 2))
    if resample == :lanczos and weights(type) == :torchvision, do: :bicubic, else: resample
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

  The steps of an image run in one compiled function. The host computes the sizes of the steps and the taps of each resize and crop from the size of the image, and the function gets the image and the taps padded to buckets of size. So images of many sizes compile few functions (see "Buckets" below).

  ## Options

    * `:backend` - the backend that runs the steps, such as `EMLX.Backend` or `{EXLA.Backend, client: :cuda}`. The default is the backend of each image. The steps are slow on `Nx.BinaryBackend`.

  ## Buckets

  Each side of the image and of the result of each step rounds up to the next size of the form 2^k or 3 * 2^(k - 1) (..., 256, 384, 512, 768, 1024, ...), so the function sees at most 1.5 times each side. The taps of each output pixel of a resize round up to a power of two, with the weight 0 for the added taps. The taps read only the pixels of the real image, so the padding of a bucket does not change the result: an added tap adds an exact 0. The host cuts the real result from the bucket at the end.
  """
  def process(featurizer, images, opts \\ []) do
    {:ok, first} = images |> hd() |> tensor()
    backend = opts[:backend] || first.data.__struct__
    featurizer |> process_on_host(images, opts) |> Nx.backend_transfer(backend)
  end

  @doc false
  # process/3 with the result on Nx.BinaryBackend, where the host cuts it from its bucket, for
  # the callers that keep the result on the host.
  def process_on_host(featurizer, images, opts \\ []) do
    images |> Enum.map(&process_image(featurizer, &1, opts[:backend])) |> stack()
  end

  defp process_image(featurizer, image, backend) do
    image =
      case tensor(image) do
        {:ok, image} -> image
        {:error, message} -> raise ArgumentError, message
      end

    compiler = Transmuter.compiler_options(backend || image.data.__struct__)
    image = Nx.backend_copy(image, Nx.BinaryBackend)

    {height, width, channels} = Nx.shape(image)
    {steps, args, {out_height, out_width}} = plan(featurizer, {height, width}, Nx.type(image))
    image = pad_bucket(image, {bucket(height), bucket(width), channels})
    run = Nx.Defn.jit(&run_steps(steps, &1, &2), compiler)
    image |> run.(List.to_tuple(args)) |> cut({out_height, out_width})
  end

  # The steps of an image in the order of the image processor, as {steps, args, size}: each
  # step is static (its kind and the static values that decide the compiled function), the
  # args are the tensors of the steps, such as the taps, and size is the real size of the
  # result. The steps run in run_steps/3. The OWLv2 image processor rescales and pads before
  # it resizes.
  defp plan(featurizer, size, type) do
    order =
      if featurizer.resample == :gaussian_linear,
        do: [:rgb, :rescale, :pad, :resize, :normalize],
        else: [:rgb, :resize, :crop, :rescale, :normalize, :pad]

    {steps, {size, _type}} =
      Enum.flat_map_reduce(order, {size, type}, &plan_step(featurizer, &1, &2))

    {steps, args} =
      Enum.map_reduce(steps, [], fn {step, step_args}, args ->
        {{step, length(args)}, args ++ step_args}
      end)

    {steps, args, size}
  end

  defp plan_step(%{rgb: true}, :rgb, state), do: {[{:rgb, []}], state}

  defp plan_step(%{resize: resize} = featurizer, :resize, {size, type}) when resize != nil do
    {height, width} = output_size(resize, size)

    type =
      if type == {:u, 8} and featurizer.resample != :gaussian_linear, do: type, else: {:f, 32}

    width_step = resize_taps(featurizer, elem(size, 1), width, 1, type)
    height_step = resize_taps(featurizer, elem(size, 0), height, 0, type)
    steps = width_step ++ height_step

    # The resize of the OWLv2 image processor clips to the minimum and the maximum of its
    # input, which the mask of the real pixels takes without the bucket.
    steps =
      if featurizer.resample == :gaussian_linear do
        real = Nx.tensor(Tuple.to_list(size), type: :s32, backend: Nx.BinaryBackend)
        [{:bounds, [real]}] ++ steps ++ [{:clip, []}]
      else
        steps
      end

    {steps, {{height, width}, type}}
  end

  defp plan_step(%{crop: {height, width}}, :crop, {size, type}) do
    steps = crop_taps(elem(size, 0), height, 0) ++ crop_taps(elem(size, 1), width, 1)
    {steps, {{height, width}, type}}
  end

  defp plan_step(featurizer, :rescale, {size, _type}),
    do: {[{{:rescale, featurizer.rescale}, []}], {size, {:f, 32}}}

  defp plan_step(%{normalize: {mean, std}}, :normalize, state) do
    mean = Nx.tensor(mean, type: :f32, backend: Nx.BinaryBackend)
    std = Nx.tensor(std, type: :f32, backend: Nx.BinaryBackend)
    {[{:normalize, [mean, std]}], state}
  end

  defp plan_step(%{pad: {target, value}}, :pad, {{height, width}, type}) do
    {to_height, to_width} = pad_size(target, {height, width})
    steps = pad_taps(height, to_height, 0, value) ++ pad_taps(width, to_width, 1, value)
    {steps, {{to_height, to_width}, type}}
  end

  defp plan_step(_featurizer, _step, state), do: {[], state}

  # The resize of one axis as a gather: the input pixels and the weights of each output pixel,
  # as resize/4 computes them. A u8 image gets the weights as integers with fraction bits.
  defp resize_taps(_featurizer, size, size, _axis, _type), do: []

  defp resize_taps(featurizer, in_size, size, axis, type) do
    taps = taps(in_size, size, featurizer.resample)
    weights = Enum.map(taps, fn row -> Enum.map(row, &elem(&1, 1)) end)

    case type do
      {:u, 8} ->
        bits = precision_bits(weights, featurizer.resample, featurizer.weights)
        one = 2 ** bits

        fixed =
          Enum.map(taps, fn row ->
            Enum.map(row, fn {j, w} -> {j, trunc(w * one + if(w < 0, do: -0.5, else: 0.5))} end)
          end)

        [gather_step(fixed, size, axis, {:s, 32}, {:u8, bits})]

      _f32 ->
        [gather_step(taps, size, axis, {:f, 32}, :f32)]
    end
  end

  # A crop keeps the center of an axis, and a crop larger than the axis pads it with zeros on
  # both sides, as center_crop of Python transformers.
  defp crop_taps(size, size, _axis), do: []

  defp crop_taps(in_size, size, axis) do
    indices =
      if in_size > size do
        offset = div(in_size - size, 2)
        Enum.map(0..(size - 1), &[{offset + &1, 1}])
      else
        low = div(size - in_size + 1, 2)

        Enum.map(
          0..(size - 1),
          &if(&1 >= low and &1 < low + in_size, do: [{&1 - low, 1}], else: [])
        )
      end

    [take_step(indices, size, axis)]
  end

  # A pad on the bottom and the right: the symmetric padding of numpy reflects the image with
  # its edge pixel (a b c | c b a), and a value pads with that value.
  defp pad_taps(size, size, _axis, _value), do: []

  defp pad_taps(in_size, size, axis, :symmetric) do
    indices =
      for i <- 0..(size - 1) do
        i = rem(i, 2 * in_size)
        [{if(i < in_size, do: i, else: 2 * in_size - 1 - i), 1}]
      end

    [take_step(indices, size, axis)]
  end

  defp pad_taps(in_size, size, axis, value) do
    taps = for i <- 0..(size - 1), do: if(i < in_size, do: [{i, 1.0}], else: [])
    bias = for i <- 0..(size - 1), do: if(i < in_size, do: 0.0, else: value / 1)
    bias = bias ++ List.duplicate(0.0, bucket(size) - size)
    {{:gather, ^axis, :f32}, args} = gather_step(taps, size, axis, {:f, 32}, :f32)
    [{{:gather, axis, :bias}, args ++ [Nx.tensor(bias, type: :f32, backend: Nx.BinaryBackend)]}]
  end

  # A gather with weights on one axis: the indices and the weights of each output pixel, padded
  # to the bucket of the size and to a power of two of taps with the index 0 and the weight 0.
  defp gather_step(taps, size, axis, type, path) do
    width = taps |> Enum.map(&length/1) |> Enum.max() |> max(1) |> power_of_two()
    rows = bucket(size)
    empty = List.duplicate({0, 0}, width)
    taps = Enum.map(taps, &(&1 ++ Enum.drop(empty, length(&1))))
    taps = taps ++ List.duplicate(empty, rows - size)
    indices = for row <- taps, {j, _w} <- row, do: j
    weights = for row <- taps, {_j, w} <- row, do: w

    indices =
      indices |> Nx.tensor(type: :s32, backend: Nx.BinaryBackend) |> Nx.reshape({rows, width})

    weights =
      weights |> Nx.tensor(type: type, backend: Nx.BinaryBackend) |> Nx.reshape({rows, width})

    {{:gather, axis, path}, [indices, weights]}
  end

  # A gather without weights on one axis: the input pixel of each output pixel, or no pixel,
  # which gives 0.
  defp take_step(taps, size, axis) do
    taps = taps ++ List.duplicate([], bucket(size) - size)
    index = Enum.map(taps, fn tap -> if tap == [], do: 0, else: elem(hd(tap), 0) end)
    mask = Enum.map(taps, &if(&1 == [], do: 0, else: 1))

    {{:take, axis},
     [
       Nx.tensor(index, type: :s32, backend: Nx.BinaryBackend),
       Nx.tensor(mask, type: :u8, backend: Nx.BinaryBackend)
     ]}
  end

  # The compiled function of the steps. The steps are static, so the function compiles once
  # for each set of steps and of bucket shapes. The bounds of a :bounds step stay for the
  # :clip step after the resize.
  defp run_steps(steps, image, args) do
    {image, _bounds} =
      Enum.reduce(steps, {image, nil}, fn {step, at}, {image, bounds} ->
        run_step(step, {image, bounds}, args, at)
      end)

    Nx.transpose(image, axes: [2, 0, 1])
  end

  defp run_step(:rgb, {image, bounds}, _args, _at) do
    {height, width, channels} = Nx.shape(image)

    image =
      if channels < 3,
        do: image |> Nx.slice_along_axis(0, 1, axis: 2) |> Nx.broadcast({height, width, 3}),
        else: Nx.slice_along_axis(image, 0, 3, axis: 2)

    {image, bounds}
  end

  defp run_step({:rescale, factor}, {image, bounds}, _args, _at) do
    image = Nx.as_type(image, :f32)
    {if(factor, do: Nx.multiply(image, factor), else: image), bounds}
  end

  defp run_step(:normalize, {image, bounds}, args, at),
    do: {image |> Nx.subtract(elem(args, at)) |> Nx.divide(elem(args, at + 1)), bounds}

  defp run_step(:bounds, {image, nil}, args, at) do
    {height, width, _channels} = shape = Nx.shape(image)
    real = elem(args, at)
    rows = Nx.less(Nx.iota({height, 1, 1}), real[0])
    columns = Nx.less(Nx.iota({1, width, 1}), real[1])
    mask = Nx.broadcast(Nx.logical_and(rows, columns), shape)
    low = Nx.reduce_min(Nx.select(mask, image, Nx.Constants.max_finite(:f32)))
    high = Nx.reduce_max(Nx.select(mask, image, Nx.Constants.min_finite(:f32)))
    {image, {low, high}}
  end

  defp run_step(:clip, {image, {low, high}}, _args, _at), do: {Nx.clip(image, low, high), nil}

  defp run_step({:take, axis}, {image, bounds}, args, at) do
    mask = elem(args, at + 1) |> Nx.as_type(Nx.type(image)) |> along(axis)
    {image |> Nx.take(elem(args, at), axis: axis) |> Nx.multiply(mask), bounds}
  end

  defp run_step({:gather, axis, path}, {image, bounds}, args, at) do
    {indices, weights} = {elem(args, at), elem(args, at + 1)}
    {rows, width} = Nx.shape(indices)
    shape = image |> Nx.shape() |> put_elem(axis, rows) |> Tuple.insert_at(axis + 1, width)
    weight_shape = for(a <- 0..3, do: if(a in [axis, axis + 1], do: elem(shape, a), else: 1))
    weights = Nx.reshape(weights, List.to_tuple(weight_shape))
    gathered = image |> Nx.take(Nx.flatten(indices), axis: axis) |> Nx.reshape(shape)

    image =
      case path do
        {:u8, bits} ->
          gathered
          |> Nx.as_type(:s32)
          |> Nx.multiply(weights)
          |> Nx.sum(axes: [axis + 1])
          |> Nx.add(div(2 ** bits, 2))
          |> Nx.right_shift(bits)
          |> Nx.clip(0, 255)
          |> Nx.as_type(:u8)

        :f32 ->
          gathered |> Nx.as_type(:f32) |> Nx.multiply(weights) |> Nx.sum(axes: [axis + 1])

        :bias ->
          gathered
          |> Nx.multiply(weights)
          |> Nx.sum(axes: [axis + 1])
          |> Nx.add(along(elem(args, at + 2), axis))
      end

    {image, bounds}
  end

  # A vector of one value for each pixel of an axis, shaped to broadcast along that axis of an
  # image {height, width, channels}.
  defp along(vector, 0), do: Nx.reshape(vector, {Nx.size(vector), 1, 1})
  defp along(vector, 1), do: Nx.reshape(vector, {1, Nx.size(vector), 1})

  # The bucket of a size: the next size of the form 2^k or 3 * 2^(k - 1).
  @doc false
  def bucket(size) when size <= 2, do: size

  def bucket(size) do
    power = power_of_two(size)
    if div(power * 3, 4) >= size, do: div(power * 3, 4), else: power
  end

  defp power_of_two(size), do: 2 ** length(Integer.digits(size - 1, 2))

  # The image {height, width, channels} on the host, with zeros after it to the bucket shape.
  defp pad_bucket(image, {height, width, channels}) do
    {rows, columns, ^channels} = Nx.shape(image)
    {_kind, bits} = type = Nx.type(image)
    row = div(columns * channels * bits, 8)
    binary = Nx.to_binary(image)
    zeros = :binary.copy(<<0>>, div((width - columns) * channels * bits, 8))

    padded = [
      for(r <- 0..(rows - 1), do: [binary_part(binary, r * row, row), zeros]),
      :binary.copy(<<0>>, div((height - rows) * width * channels * bits, 8))
    ]

    padded
    |> IO.iodata_to_binary()
    |> Nx.from_binary(type, backend: Nx.BinaryBackend)
    |> Nx.reshape({height, width, channels})
  end

  # The real result {channels, height, width} of a result in its bucket, on the host.
  defp cut(image, {height, width}) do
    {channels, rows, columns} = Nx.shape(image)
    binary = Nx.to_binary(image)
    {_kind, bits} = type = Nx.type(image)
    row = div(columns * bits, 8)
    size = div(width * bits, 8)

    for(
      c <- 0..(channels - 1),
      r <- 0..(height - 1),
      do: binary_part(binary, (c * rows + r) * row, size)
    )
    |> IO.iodata_to_binary()
    |> Nx.from_binary(type, backend: Nx.BinaryBackend)
    |> Nx.reshape({channels, height, width})
  end

  # The images {channels, height, width} on the host as one tensor with a batch axis.
  defp stack(images = [first | _]) do
    images
    |> Enum.map(&Nx.to_binary/1)
    |> IO.iodata_to_binary()
    |> Nx.from_binary(Nx.type(first), backend: Nx.BinaryBackend)
    |> Nx.reshape(Tuple.insert_at(Nx.shape(first), 0, length(images)))
  end

  defp pad_size(:square, {height, width}), do: {max(height, width), max(height, width)}

  # Swin2SRImageProcessor also adds `multiple` pixels to a side that is already a multiple.
  defp pad_size({:next_multiple, multiple}, {height, width}),
    do: {(div(height, multiple) + 1) * multiple, (div(width, multiple) + 1) * multiple}

  defp pad_size({to_height, to_width}, _size), do: {to_height, to_width}

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
