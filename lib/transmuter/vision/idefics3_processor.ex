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
  def process(processor, image, backend) do
    {:ok, image} = Featurizer.tensor(image)
    %{resample: resample, weights: weights} = processor.featurizer
    resize = &Featurizer.resize(&1, &2, resample, weights)
    image = image |> Nx.backend_transfer(backend) |> Featurizer.rgb()
    {height, width, _channels} = Nx.shape(image)

    image =
      if processor.longest_edge,
        do: resize.(image, longest_edge({height, width}, processor.longest_edge)),
        else: image

    {tiles, rows, columns} =
      if processor.split? do
        {height, width, _channels} = Nx.shape(image)
        image = resize.(image, tile_multiple({height, width}, processor.tile))
        split(image, processor.tile, resize)
      else
        {[resize.(image, {processor.tile, processor.tile})], 0, 0}
      end

    pixels = Featurizer.process(processor.featurizer, tiles, backend: backend)
    {%{"pixel_values" => pixels}, text(processor, rows, columns)}
  end

  @doc """
  The inputs of the vision encoder for one tile, on `Nx.BinaryBackend`. The encoder compiles for them. The encoder takes one tile in each call.
  """
  def template(processor) do
    zero = Nx.tensor(0.0, backend: Nx.BinaryBackend)

    %{
      "pixel_values" => Nx.broadcast(zero, {1, 1, 3, processor.tile, processor.tile}),
      "pixel_attention_mask" => mask(processor.tile, processor.tile)
    }
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

  # An image of one tile does not split. The sides are multiples of a tile, so all tiles
  # have the same size.
  defp split(image, tile, resize) do
    {height, width, channels} = Nx.shape(image)

    if height > tile or width > tile do
      {rows, columns} = {div(height, tile), div(width, tile)}

      tiles =
        for row <- 0..(rows - 1),
            column <- 0..(columns - 1),
            do: Nx.slice(image, [row * tile, column * tile, 0], [tile, tile, channels])

      {tiles ++ [resize.(image, {tile, tile})], rows, columns}
    else
      {[image], 0, 0}
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
