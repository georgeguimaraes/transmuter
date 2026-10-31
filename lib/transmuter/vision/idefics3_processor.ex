defmodule Transmuter.Vision.Idefics3Processor do
  @moduledoc """
  Prepares the images and the prompt of Idefics3 models, such as SmolVLM, as `Idefics3ImageProcessor` and `Idefics3Processor` of Python transformers do. `Transmuter.Vision.image_text_to_text/2` uses it.

  The steps for each image are:

    1. The image gets 3 channels, as in `Transmuter.Featurizer.rgb/1`.
    2. The long side of the image gets `size["longest_edge"]` of `preprocessor_config.json`, and the short side keeps the aspect ratio. A small image also gets larger.
    3. Each side gets the next multiple of `max_image_size["longest_edge"]`, the size of a tile.
    4. The image splits into tiles, from left to right and from top to bottom. The image resized to one tile comes after them. Without `do_image_splitting`, the image resized to one tile is the only tile.
    5. Each tile is rescaled and normalized, as in `Transmuter.Featurizer`.

  The resizes use the `resample` filter of the configuration on the 8-bit image, as Pillow does, or with the weights of torchvision for `Idefics3ImageProcessorFast`, which uses bicubic in place of lanczos. See `Transmuter.Featurizer.resize/4`.

  In the prompt, each image is a placeholder `<image>`, and `process/3` gives the text that replaces it: `<fake_token_around_image><row_r_col_c>` and `image_seq_len` image tokens for each tile, a newline after each row, and then `\\n<fake_token_around_image><global-img>`, the image tokens of the last tile and `<fake_token_around_image>`.
  """

  alias Transmuter.{Featurizer, Hub}
  alias Transmuter.Vision.ImageTextToText

  defstruct [:featurizer, :longest_edge, :tile, :split?, :image_seq_len]

  @placeholder "<image>"

  @doc """
  Loads the processor of a `Transmuter.Repo` from its `preprocessor_config.json` and `processor_config.json`.
  """
  def load!(repo) do
    config =
      Hub.json(repo.source, "preprocessor_config.json", repo.opts) ||
        raise ArgumentError, "#{inspect(repo.source)} has no preprocessor_config.json"

    processor = Hub.json(repo.source, "processor_config.json", repo.opts) || %{}

    # The defaults are those of Python transformers.
    %__MODULE__{
      featurizer: Featurizer.new(Map.put(config, "do_resize", false)),
      longest_edge:
        if(Map.get(config, "do_resize", true),
          do: get_in(config, ["size", "longest_edge"]) || 1456
        ),
      tile: get_in(config, ["max_image_size", "longest_edge"]) || 364,
      split?: Map.get(config, "do_image_splitting", true),
      image_seq_len: processor["image_seq_len"] || 169
    }
  end

  @doc """
  The text of an image in the prompt before `process/3` expands it.
  """
  def placeholder(_processor), do: @placeholder

  @doc """
  Turns an image into tiles and gives the text of the image in the prompt.

  Returns `{tensors, text}`. `tensors` has `"pixel_values"` of shape `{tiles, 3, tile, tile}` and type `:f32`, on `backend`.
  """
  def process(processor, image, backend),
    do: process_with(processor, image, backend, &Featurizer.process_image/3)

  @doc false
  # process/3 with the tensors on the host.
  def process_on_host(processor, image, backend),
    do: process_with(processor, image, backend, &Featurizer.process_image_on_host/3)

  # The resizes, the steps of the featurizer and the tiles with the global tile run in one
  # compiled function for each bucket of image size, see
  # Transmuter.Featurizer.process_image_on_host/3.
  defp process_with(processor, image, backend, process) do
    {:ok, tensor} = Featurizer.tensor(image)
    {height, width, _channels} = Nx.shape(tensor)
    tile = processor.tile

    resizes =
      if processor.longest_edge,
        do: [longest_edge({height, width}, processor.longest_edge)],
        else: []

    size = List.last(resizes, {height, width})

    resizes =
      if processor.split?,
        do: resizes ++ [tile_multiple(size, tile)],
        else: resizes ++ [{tile, tile}]

    {height, width} = List.last(resizes)
    {rows, columns} = {div(height, tile), div(width, tile)}

    # An image of one tile does not split, and it has no global tile.
    {layout, rows, columns} =
      if rows * columns > 1, do: {{:tiles, tile}, rows, columns}, else: {:image, 0, 0}

    before = Enum.map(resizes, &{:resize, &1})

    pixels =
      process.(processor.featurizer, image, backend: backend, before: before, layout: layout)

    pixels = if layout == :image, do: Nx.new_axis(pixels, 0), else: pixels
    {%{"pixel_values" => pixels}, text(processor, rows, columns)}
  end

  @doc """
  Runs the vision encoder `encode`, a function from `Transmuter.compile/3`, on the `"pixel_values"` of `tensors`, of shape `{batch, tiles, 3, tile, tile}` on the host, and gives the inputs of `Transmuter.Generation.generate/5`.

  The tiles of zeros do not run, as `Idefics3Model.get_image_features` of Python transformers removes them. Each real tile runs in a call of its own, from the bytes of the pixel values, so no operation compiles for each count of tiles, and the tiles of zeros do not go to the device. So the features are those of the real tiles of all rows, in order. Returns `%{image_features: features}`, a tensor of shape `{real_tiles * image_seq_len, hidden}` on the host. The prompts take the default positions.
  """
  def encode(_processor, encode, %{"pixel_values" => pixels}, _prompts, _backend) do
    {batch, tiles, channels, height, width} = Nx.shape(pixels)
    pixels = Nx.reshape(pixels, {batch * tiles, 1, channels, height, width})
    zeros = :binary.copy(<<0>>, div(Nx.byte_size(pixels), batch * tiles))
    mask = mask(height, width)

    features =
      for tile <- 0..(batch * tiles - 1),
          pixels = ImageTextToText.host_rows(pixels, tile, 1),
          Nx.to_binary(pixels) != zeros do
        encode.(%{"pixel_values" => pixels, "pixel_attention_mask" => mask})["image_features"]
      end

    %{image_features: ImageTextToText.join_rows(features)}
  end

  # Each tile is a whole image, so the mask has no padding.
  defp mask(height, width) do
    <<1>>
    |> :binary.copy(height * width)
    |> Nx.from_binary(:u8, backend: Nx.BinaryBackend)
    |> Nx.reshape({1, 1, height, width})
  end

  # As _resize_output_size_rescale_to_max_len of Python transformers: the long side gets
  # max, and an odd short side gets one more pixel.
  defp longest_edge({height, width}, max) do
    aspect = width / height

    {height, width} =
      if width >= height,
        do: {even(trunc(max / aspect)), max},
        else: {max, even(trunc(max * aspect))}

    {max(height, 1), max(width, 1)}
  end

  defp even(size), do: size + rem(size, 2)

  # As resize_for_vision_encoder of Python transformers. The short side comes from the new
  # long side and the aspect ratio, and then gets the next multiple of a tile.
  defp tile_multiple({height, width}, tile) do
    aspect = width / height

    if width >= height do
      width = ceil(width / tile) * tile
      {ceil(trunc(width / aspect) / tile) * tile, width}
    else
      height = ceil(height / tile) * tile
      {height, ceil(trunc(height * aspect) / tile) * tile}
    end
  end

  defp text(processor, rows, columns) do
    tokens = String.duplicate(@placeholder, processor.image_seq_len)
    global = "<fake_token_around_image><global-img>" <> tokens <> "<fake_token_around_image>"

    grid =
      for row <- 1..rows//1, into: "" do
        for(column <- 1..columns, into: "", do: tile_text(row, column, tokens)) <> "\n"
      end

    if rows == 0, do: global, else: grid <> "\n" <> global
  end

  defp tile_text(row, column, tokens),
    do: "<fake_token_around_image><row_#{row}_col_#{column}>" <> tokens
end
