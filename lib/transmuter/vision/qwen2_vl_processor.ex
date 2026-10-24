defmodule Transmuter.Vision.Qwen2VLProcessor do
  @moduledoc """
  Prepares the images and the prompt of Qwen2-VL and Qwen3-VL, as `Qwen2VLImageProcessor`, `Qwen2VLImageProcessorFast`, `Qwen2VLProcessor` and `Qwen3VLProcessor` of Python transformers do for images. `Transmuter.Vision.image_text_to_text/2` uses it.

  The steps for each image are:

    1. The image gets 3 channels, as in `Transmuter.Featurizer.rgb/1`.
    2. Each side gets the nearest multiple of `patch_size * merge_size` pixels: 28 for Qwen2-VL and 32 for Qwen3-VL. When the image then has more than `max_pixels` or fewer than `min_pixels` pixels, the sides get smaller or larger with the same aspect ratio, as `smart_resize` of Python transformers. The resize uses the `resample` filter of the configuration, bicubic by default, on the 8-bit image. The weights are those of Pillow, or of torchvision for `Qwen2VLImageProcessorFast`, see `Transmuter.Featurizer.resize/4`.
    3. The image is rescaled and normalized, as in `Transmuter.Featurizer`.
    4. The image is repeated `temporal_patch_size` (2) times, as a video of 2 frames, and splits into patches of `patch_size` pixels (14x14 for Qwen2-VL). Each patch is one row of `3 * 2 * patch_size * patch_size` values. Each group of 2x2 patches, which the vision encoder merges into one feature, comes in consecutive rows.

  The grid of an image is `[1, height / patch_size, width / patch_size]`, as `image_grid_thw` of Python transformers. In the prompt, each image is a placeholder `<|image_pad|>`, which becomes one `<|image_pad|>` for each feature: the number of patches divided by 4.

  The image tokens of the prompt take 3-D positions for M-RoPE, as `get_rope_index` of Python transformers. The text tokens have the same position on the 3 axes. The image tokens of an image have the positions of their frame, row and column after the text before them, and the text after the image continues from the largest of these positions.
  """

  alias Transmuter.{Featurizer, Hub, Repo}
  alias Transmuter.Vision.ImageTextToText

  defstruct [
    :featurizer,
    :resize?,
    :min_pixels,
    :max_pixels,
    :patch_size,
    :merge_size,
    :temporal_patch_size,
    :image_token,
    :grid_input
  ]

  @placeholder "<|image_pad|>"

  @doc """
  Loads the processor of a `Transmuter.Repo` from its `preprocessor_config.json` and `config.json`.

  ## Options

    * `:inputs` - the input names of the vision encoder, such as `model.inputs` of the `vision_encoder` component. The input whose name ends in `grid_thw` takes the grid: `"grid_thw"` in the export of Qwen2-VL, and `"image_grid_thw"` in the export of Qwen3-VL. The default is `["pixel_values", "grid_thw"]`.
  """
  def load!(repo, opts \\ []) do
    inputs = Keyword.validate!(opts, inputs: ["pixel_values", "grid_thw"])[:inputs]

    grid_input =
      Enum.find(inputs, &String.ends_with?(&1, "grid_thw")) ||
        raise ArgumentError,
              "the vision encoder has no input whose name ends in grid_thw, got: #{inspect(inputs)}"

    config =
      Hub.json(repo.source, "preprocessor_config.json", repo.opts) ||
        raise ArgumentError, "#{inspect(repo.source)} has no preprocessor_config.json"

    # The defaults are those of Python transformers. The featurizer gives the bicubic resample
    # of Qwen2VLImageProcessor. min_pixels and max_pixels take precedence over the size.
    size = config["size"] || %{}

    %__MODULE__{
      featurizer: config |> Map.put("do_resize", false) |> Map.delete("size") |> Featurizer.new(),
      resize?: Map.get(config, "do_resize", true),
      min_pixels: config["min_pixels"] || size["shortest_edge"] || 56 * 56,
      max_pixels: config["max_pixels"] || size["longest_edge"] || 28 * 28 * 1280,
      patch_size: config["patch_size"] || 14,
      merge_size: config["merge_size"] || 2,
      temporal_patch_size: config["temporal_patch_size"] || 2,
      image_token: Repo.image_token_id(repo),
      grid_input: grid_input
    }
  end

  @doc """
  The text of an image in the prompt before `process/3` expands it.
  """
  def placeholder(_processor), do: @placeholder

  @doc """
  Turns an image into patches and gives the text of the image in the prompt.

  Returns `{tensors, text}`. `tensors` has `"pixel_values"` of shape `{patches, 3 * 2 * patch_size * patch_size}` (1176 for Qwen2-VL) and type `:f32` on `backend`, and `"image_grid_thw"` of shape `{1, 3}` and type `:s64`.
  """
  def process(processor, image, backend) do
    {:ok, image} = Featurizer.tensor(image)
    image = image |> Nx.backend_transfer(backend) |> Featurizer.rgb()
    {height, width, _channels} = Nx.shape(image)
    factor = processor.patch_size * processor.merge_size

    {height, width} =
      if processor.resize?,
        do: smart_resize(height, width, factor, processor.min_pixels, processor.max_pixels),
        else: {height, width}

    %{resample: resample, weights: weights} = processor.featurizer
    image = Featurizer.resize(image, {height, width}, resample, weights)
    frames = Featurizer.process(processor.featurizer, [image], backend: backend)
    {patches, grid} = patches(processor, frames)
    count = div(Enum.product(grid), processor.merge_size ** 2)

    tensors = %{
      "pixel_values" => patches,
      "image_grid_thw" => Nx.tensor([grid], type: :s64, backend: Nx.BinaryBackend)
    }

    {tensors, String.duplicate(@placeholder, count)}
  end

  @doc """
  The inputs of the vision encoder for an image of 2x2 patches, on `Nx.BinaryBackend`. The encoder compiles for them. The encoder takes one image in each call, so the template does not depend on `batch_size`.
  """
  def template(processor, _batch_size) do
    %{patch_size: patch, temporal_patch_size: temporal, merge_size: merge} = processor
    zero = Nx.tensor(0.0, backend: Nx.BinaryBackend)

    %{
      "pixel_values" => Nx.broadcast(zero, {merge * merge, 3 * temporal * patch * patch}),
      processor.grid_input =>
        Nx.tensor([[1, merge, merge]], type: :s64, backend: Nx.BinaryBackend)
    }
  end

  @doc """
  Runs the vision encoder `encode`, a function from `Transmuter.compile/3`, on the images of a batch, and gives the inputs of `Transmuter.Generation.generate/5`.

  `tensors` has the `"pixel_values"` of each row, of shape `{batch, rows, values}`, and the `"image_grid_thw"` of each row, of shape `{batch, rows, 3}`. Rows of zeros follow the patches and the grids of the images. `prompts` are the token ids of each row.

  The export of the vision encoder of Qwen2-VL computes the rotary positions of the patches for one image, so each image runs alone. Returns `%{image_features: features, position_ids: positions}`: the features of the images of all rows, in order, and the 3-D positions of each prompt.
  """
  def encode(processor, encode, tensors, prompts, _backend) do
    %{"pixel_values" => pixels, "image_grid_thw" => grids} = tensors
    images = ImageTextToText.images(pixels, grids, &Enum.product/1)

    features =
      for row <- images, {grid, pixels} <- row do
        inputs = %{
          "pixel_values" => pixels,
          processor.grid_input => Nx.tensor([grid], type: :s64, backend: Nx.BinaryBackend)
        }

        encode.(inputs)["image_features"]
      end

    grids = Enum.map(images, fn row -> Enum.map(row, &elem(&1, 0)) end)

    %{
      image_features: Nx.concatenate(features),
      position_ids: Enum.zip_with(prompts, grids, &positions(processor, &1, &2))
    }
  end

  @doc """
  The 3-D positions of the tokens `ids` of a prompt with the images of grids `grids`, as `get_rope_index` of Python transformers. Returns a list of the positions on each of the 3 axes.
  """
  def positions(processor, ids, grids) do
    {done, rest, start} =
      Enum.reduce(grids, {[], ids, 0}, fn [frames, rows, columns], {done, rest, start} ->
        {text, rest} = Enum.split_while(rest, &(&1 != processor.image_token))
        {rows, columns} = {div(rows, processor.merge_size), div(columns, processor.merge_size)}
        offset = start + length(text)

        image =
          for t <- 0..(frames - 1),
              h <- 0..(rows - 1),
              w <- 0..(columns - 1),
              do: {offset + t, offset + h, offset + w}

        rest = Enum.drop(rest, frames * rows * columns)

        {[done, text_positions(start, length(text)), image], rest,
         offset + Enum.max([frames, rows, columns])}
      end)

    positions = List.flatten([done, text_positions(start, length(rest))])
    for axis <- 0..2, do: Enum.map(positions, &elem(&1, axis))
  end

  defp text_positions(start, count), do: for(i <- start..(start + count - 1)//1, do: {i, i, i})

  # As _preprocess of Qwen2VLImageProcessor: the frames {temporal, 3, height, width} split
  # into a grid of patches, with the 2x2 patches of each merged feature in consecutive rows.
  defp patches(processor, frame) do
    %{patch_size: patch, temporal_patch_size: temporal, merge_size: merge} = processor
    {1, channels, height, width} = Nx.shape(frame)
    {rows, columns} = {div(height, patch), div(width, patch)}
    frames = Nx.concatenate(List.duplicate(frame, temporal))

    patches =
      frames
      |> Nx.reshape(
        {1, temporal, channels, div(rows, merge), merge, patch, div(columns, merge), merge, patch}
      )
      |> Nx.transpose(axes: [0, 3, 6, 4, 7, 2, 1, 5, 8])
      |> Nx.reshape({rows * columns, channels * temporal * patch * patch})

    {patches, [1, rows, columns]}
  end

  # As smart_resize of Python transformers. round/1 of Python rounds half to even.
  defp smart_resize(height, width, factor, min_pixels, max_pixels) do
    if max(height, width) / min(height, width) > 200 do
      raise ArgumentError,
            "the aspect ratio of the image must be smaller than 200, got: #{max(height, width) / min(height, width)}"
    end

    {h, w} = {round_even(height / factor) * factor, round_even(width / factor) * factor}

    cond do
      h * w > max_pixels ->
        beta = :math.sqrt(height * width / max_pixels)

        {max(factor, floor(height / beta / factor) * factor),
         max(factor, floor(width / beta / factor) * factor)}

      h * w < min_pixels ->
        beta = :math.sqrt(min_pixels / (height * width))
        {ceil(height * beta / factor) * factor, ceil(width * beta / factor) * factor}

      true ->
        {h, w}
    end
  end

  defp round_even(x) do
    down = floor(x)

    cond do
      x - down > 0.5 -> down + 1
      x - down < 0.5 -> down
      rem(down, 2) == 0 -> down
      true -> down + 1
    end
  end
end
