defmodule Transmuter.Vision.PixtralProcessor do
  @moduledoc """
  Prepares the images and the prompt of Pixtral and Mistral 3 models, such as Ministral 3, as `PixtralImageProcessor` and `PixtralProcessor` of Python transformers do. `Transmuter.Vision.image_text_to_text/2` uses it.

  The steps for each image are:

    1. The image gets 3 channels, as in `Transmuter.Featurizer.rgb/1`.
    2. When a side is longer than `size["longest_edge"]` of `preprocessor_config.json`, both sides get smaller by the same ratio and are rounded down. Then each side gets the next multiple of `patch_size * spatial_merge_size` of `processor_config.json` (28 for Ministral 3). The resize uses the `resample` filter of the configuration on the 8-bit image, with the weights of torchvision for a fast image processor, as in `Transmuter.Featurizer`.
    3. The image is rescaled and normalized, as in `Transmuter.Featurizer`.
    4. The image splits into blocks of `patch_size * spatial_merge_size` pixels on each side, from left to right and from top to bottom. Each block is one row of `3 * 28 * 28` values. The vision encoder merges the patches of a block into one feature, so the image has one row for each feature.

  In the prompt, each image is a placeholder `[IMG]`. It becomes one row of `[IMG]` tokens for each row of blocks, with one token for each block. `[IMG_BREAK]` comes after each row, and `[IMG_END]` replaces the last `[IMG_BREAK]`. The features replace the `[IMG]` tokens, and the prompt takes the default positions.
  """

  alias Transmuter.{Featurizer, Hub}
  alias Transmuter.Vision.ImageTextToText

  defstruct [
    :featurizer,
    :max_size,
    :block,
    :image_token,
    :break_token,
    :end_token
  ]

  @placeholder "[IMG]"

  # The defaults of PixtralImageProcessor and PixtralImageProcessorFast of Python
  # transformers, which are the same. They also take the place of a null value.
  @defaults %{
    "resample" => 3,
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

    config = Map.merge(@defaults, config, fn _key, default, value -> value || default end)
    processor = Hub.json(repo.source, "processor_config.json", repo.opts) || %{}

    # The defaults are those of Python transformers. PixtralProcessor gives the image
    # processor its own patch size times the merge size.
    %__MODULE__{
      featurizer: config |> Map.put("do_resize", false) |> Map.delete("size") |> Featurizer.new(),
      max_size: if(Map.get(config, "do_resize", true), do: max_size(config["size"])),
      block: (processor["patch_size"] || 16) * (processor["spatial_merge_size"] || 1),
      image_token: processor["image_token"] || @placeholder,
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
  def placeholder, do: @placeholder

  @doc """
  Turns an image into blocks and gives the text of the image in the prompt.

  Returns `{tensors, text}`. `tensors` has `"pixel_values"` of shape `{blocks, 3 * block * block}` and type `:f32` on `backend`, and `"image_sizes"` of shape `{1, 2}` and type `:s64`, the height and the width of the image after the resize.
  """
  def process(processor, image, backend) do
    {:ok, image} = Featurizer.tensor(image)
    image = image |> Nx.backend_transfer(backend) |> Featurizer.rgb()
    {height, width, _channels} = Nx.shape(image)

    {height, width} =
      if processor.max_size,
        do: resize_size({height, width}, processor.max_size, processor.block),
        else: {height, width}

    %{featurizer: featurizer} = processor
    image = Featurizer.resize(image, {height, width}, featurizer.resample, featurizer.weights)
    pixels = Featurizer.process(featurizer, [image], backend: backend)
    {rows, columns} = {div(height, processor.block), div(width, processor.block)}

    tensors = %{
      "pixel_values" => blocks(pixels, processor.block),
      "image_sizes" => Nx.tensor([[height, width]], type: :s64, backend: Nx.BinaryBackend)
    }

    {tensors, text(processor, rows, columns)}
  end

  @doc """
  The inputs of the vision encoder for an image of one block, on `Nx.BinaryBackend`. The encoder compiles for them. The encoder takes one image in each call, so the template does not depend on `batch_size`.
  """
  def template(processor, _batch_size) do
    zero = Nx.tensor(0.0, backend: Nx.BinaryBackend)
    %{"pixel_values" => Nx.broadcast(zero, {1, 3, processor.block, processor.block})}
  end

  @doc """
  Runs the vision encoder `encode`, a function from `Transmuter.compile/3`, on the images of a batch, and gives the inputs of `Transmuter.Generation.generate/5`.

  `tensors` has the `"pixel_values"` of each row, of shape `{batch, blocks, 3 * block * block}`, and the `"image_sizes"` of each row, of shape `{batch, rows, 2}`. Rows of zeros follow the blocks and the sizes of the images.

  The export of the vision encoder computes the rotary positions of the patches from the size of the image, so each image runs alone. Returns `%{image_features: features}`: the features of the images of all rows, in order. The prompts take the default positions.
  """
  def encode(processor, encode, tensors, _prompts, _backend) do
    %{"pixel_values" => pixels, "image_sizes" => sizes} = tensors

    features =
      for row <- ImageTextToText.images(pixels, sizes, &count(processor, &1)),
          {[_height, width], blocks} <- row do
        image = image(blocks, processor.block, div(width, processor.block))
        encode.(%{"pixel_values" => image})["image_features"] |> Nx.squeeze(axes: [0])
      end

    %{image_features: Nx.concatenate(features)}
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
