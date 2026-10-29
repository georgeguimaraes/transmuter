defmodule Transmuter.Vision.PixtralProcessor do
  @moduledoc """
  Prepares the images and the prompt of Pixtral and Mistral 3 models, such as Ministral 3, as `PixtralImageProcessor` and `PixtralProcessor` of Python transformers do. `Transmuter.Vision.image_text_to_text/2` uses it.

  The steps for each image are:

    1. The image gets 3 channels, as in `Transmuter.Featurizer.rgb/1`. For `PixtralImageProcessor`, the slow image processor, an image with an alpha channel goes on a white background first, as `Image.paste` of Pillow does. `PixtralImageProcessorFast` removes the alpha channel.
    2. When a side is longer than `size["longest_edge"]` of `preprocessor_config.json`, both sides get smaller by the same ratio and are rounded down. Then each side gets the next multiple of `patch_size * spatial_merge_size` of `processor_config.json` (28 for Ministral 3). The resize uses the `resample` filter of the configuration on the 8-bit image, with the weights of torchvision for a fast image processor, as in `Transmuter.Featurizer`. Without `do_resize`, the image keeps its size, and the blocks of step 4 start at the top left corner. The pixels to the right of the last full block and below it are not used, as in Python transformers, where the processor gives `height // block` rows of tokens. A side with no full block raises an error.
    3. The image is rescaled and normalized, as in `Transmuter.Featurizer`.
    4. The image splits into blocks of `patch_size * spatial_merge_size` pixels on each side, from left to right and from top to bottom. Each block is one row of `3 * 28 * 28` values. The vision encoder merges the patches of a block into one feature, so the image has one row for each feature.

  In the prompt, each image is a placeholder, the `image_token` of `processor_config.json` (`[IMG]` by default). It becomes one row of these tokens for each row of blocks, with one token for each block. `[IMG_BREAK]` comes after each row, and `[IMG_END]` replaces the last `[IMG_BREAK]`. The features replace the `[IMG]` tokens, and the prompt takes the default positions.
  """

  alias Transmuter.{Featurizer, Hub}
  alias Transmuter.Vision.ImageTextToText

  defstruct [
    :featurizer,
    :white?,
    :max_size,
    :block,
    :image_token,
    :break_token,
    :end_token
  ]

  # The defaults of PixtralImageProcessor and PixtralImageProcessorFast of Python
  # transformers, which are the same. They also take the place of a null value. The
  # featurizer gives the bicubic resample of these classes.
  @defaults %{
    "image_mean" => [0.48145466, 0.4578275, 0.40821073],
    "image_std" => [0.26862954, 0.26130258, 0.27577711]
  }

  @doc """
  Loads the processor of a `Transmuter.Repo` from its `preprocessor_config.json` and `processor_config.json`.
  """
  def load!(repo) do
    config =
      Hub.json(repo.source, "preprocessor_config.json", repo.opts) ||
        raise ArgumentError, "#{inspect(repo.source)} has no preprocessor_config.json"

    new(config, Hub.json(repo.source, "processor_config.json", repo.opts) || %{})
  end

  @doc """
  Builds the processor from the maps of a `preprocessor_config.json` and a `processor_config.json`.
  """
  def new(config, processor) do
    config = Map.merge(@defaults, config, fn _key, default, value -> value || default end)

    # The defaults are those of Python transformers. PixtralProcessor gives the image
    # processor its own patch size times the merge size.
    %__MODULE__{
      featurizer: config |> Map.put("do_resize", false) |> Map.delete("size") |> Featurizer.new(),
      white?: config["image_processor_type"] == "PixtralImageProcessor",
      max_size: if(Map.get(config, "do_resize", true), do: max_size(config["size"])),
      block: (processor["patch_size"] || 16) * (processor["spatial_merge_size"] || 1),
      image_token: processor["image_token"] || "[IMG]",
      break_token: processor["image_break_token"] || "[IMG_BREAK]",
      end_token: processor["image_end_token"] || "[IMG_END]"
    }
  end

  defp max_size(%{"longest_edge" => edge}), do: {edge, edge}
  defp max_size(%{"height" => height, "width" => width}), do: {height, width}
  defp max_size(nil), do: {1024, 1024}

  @doc """
  The text of an image in the prompt before `process/3` expands it.
  """
  def placeholder(processor), do: processor.image_token

  @doc """
  Turns an image into blocks and gives the text of the image in the prompt.

  Returns `{tensors, text}`. `tensors` has `"pixel_values"` of shape `{blocks, 3 * block * block}` and type `:f32` on `backend`, and `"image_sizes"` of shape `{1, 2}` and type `:s64`, the height and the width of the image after the resize.
  """
  def process(processor, image, backend) do
    {:ok, image} = Featurizer.tensor(image)
    image = Nx.backend_transfer(image, backend)
    image = if processor.white?, do: on_white(image), else: Featurizer.rgb(image)
    %{featurizer: featurizer, block: block} = processor
    {height, width, _channels} = shape = Nx.shape(image)

    {height, width} =
      if processor.max_size,
        do: resize_size({height, width}, processor.max_size, block),
        else: {div(height, block) * block, div(width, block) * block}

    if height == 0 or width == 0 do
      raise ArgumentError,
            "the image of shape #{inspect(shape)} gives no block of #{block}x#{block} pixels on a side"
    end

    image =
      if processor.max_size,
        do: Featurizer.resize(image, {height, width}, featurizer.resample, featurizer.weights),
        else: Nx.slice(image, [0, 0, 0], [height, width, 3])

    pixels = Featurizer.process(featurizer, [image], backend: backend)
    {rows, columns} = {div(height, block), div(width, block)}

    tensors = %{
      "pixel_values" => blocks(pixels, block),
      "image_sizes" => Nx.tensor([[height, width]], type: :s64, backend: Nx.BinaryBackend)
    }

    {tensors, text(processor, rows, columns)}
  end

  @doc """
  Runs the vision encoder `encode`, a function from `Transmuter.compile/3`, on the images of a batch, and gives the inputs of `Transmuter.Generation.generate/5`.

  `tensors` has the `"pixel_values"` of each row, of shape `{batch, blocks, 3 * block * block}`, and the `"image_sizes"` of each row, of shape `{batch, rows, 2}`. Rows of zeros follow the blocks and the sizes of the images.

  The export of the vision encoder computes the rotary positions of the patches from the size of the image, so each image runs alone. Returns `%{image_features: features}`: the features of the images of all rows, in order. The prompts take the default positions.
  """
  def encode(processor, encode, tensors, _prompts, backend) do
    %{"pixel_values" => pixels, "image_sizes" => sizes} = tensors

    features =
      for row <- ImageTextToText.images(pixels, sizes, &count(processor, &1)),
          {[_height, width], blocks} <- row do
        image =
          blocks
          |> Nx.backend_transfer(backend)
          |> image(processor.block, div(width, processor.block))

        encode.(%{"pixel_values" => image})["image_features"]
      end

    %{image_features: ImageTextToText.join_rows(features)}
  end

  defp count(processor, [height, width]),
    do: div(height, processor.block) * div(width, processor.block)

  # As get_resize_output_image_size of Python transformers: a larger image gets the ratio
  # of its larger side, and each side gets the next multiple of the block.
  defp resize_size({height, width}, {max_height, max_width}, block) do
    ratio = max(height / max_height, width / max_width)

    {height, width} =
      if ratio > 1, do: {floor(height / ratio), floor(width / ratio)}, else: {height, width}

    {(Integer.floor_div(height - 1, block) + 1) * block,
     (Integer.floor_div(width - 1, block) + 1) * block}
  end

  # As convert_to_rgb of PixtralImageProcessor: an image with alpha goes on white, as
  # Image.paste of Pillow with the image as the mask. Each channel is
  # in * alpha / 255 + 255 * (255 - alpha) / 255, with the rounding of Pillow.
  defp on_white(image) do
    {_height, _width, channels} = Nx.shape(image)

    if channels in [2, 4] do
      image = Nx.as_type(image, :s32)
      alpha = Nx.slice_along_axis(image, channels - 1, 1, axis: 2)
      color = image |> Nx.slice_along_axis(0, channels - 1, axis: 2) |> Featurizer.rgb()

      color
      |> mul_div_255(alpha)
      |> Nx.add(mul_div_255(255, Nx.subtract(255, alpha)))
      |> Nx.as_type(:u8)
    else
      Featurizer.rgb(image)
    end
  end

  defp mul_div_255(a, b) do
    product = Nx.add(Nx.multiply(a, b), 128)
    product |> Nx.right_shift(8) |> Nx.add(product) |> Nx.right_shift(8)
  end

  # {1, channels, height, width} to one row of {channels, block, block} values for each block.
  defp blocks(pixels, block) do
    {1, channels, height, width} = Nx.shape(pixels)
    {rows, columns} = {div(height, block), div(width, block)}

    pixels
    |> Nx.reshape({channels, rows, block, columns, block})
    |> Nx.transpose(axes: [1, 3, 0, 2, 4])
    |> Nx.reshape({rows * columns, channels * block * block})
  end

  # The inverse of blocks/2, for an image of `columns` blocks in each row.
  defp image(blocks, block, columns) do
    rows = div(Nx.axis_size(blocks, 0), columns)

    blocks
    |> Nx.reshape({rows, columns, :auto, block, block})
    |> Nx.transpose(axes: [2, 0, 3, 1, 4])
    |> Nx.reshape({1, :auto, rows * block, columns * block})
  end

  defp text(processor, rows, columns) do
    tokens = String.duplicate(processor.image_token, columns)
    String.duplicate(tokens <> processor.break_token, rows - 1) <> tokens <> processor.end_token
  end
end
