defmodule Transmuter.Vision.LlavaProcessor do
  @moduledoc """
  Prepares the images and the prompt of LLaVA models, such as LLaVA-Interleave, as `LlavaProcessor` of Python transformers does. `Transmuter.Vision.image_text_to_text/2` uses it.

  The image processor of `preprocessor_config.json` turns each image into pixel values of one size, as in `Transmuter.Featurizer`: `SiglipImageProcessor` resizes to `size`, and `CLIPImageProcessor` resizes the short side and crops the center.

  In the prompt, each image is a placeholder, the `image_token` of the tokenizer configuration, else of `processor_config.json` (`<image>` by default). It becomes one image token for each feature: `(height // patch_size) * (width // patch_size) + num_additional_image_tokens` tokens of `processor_config.json`, and one token less when its `vision_feature_select_strategy` is `"default"`. The height and the width are those of the pixel values. The vision encoder gives one feature for each of these tokens.
  """

  alias Transmuter.{Featurizer, Hub, Repo}
  alias Transmuter.Vision.ImageTextToText

  defstruct [:featurizer, :size, :image_seq_len, :image]

  @doc """
  Loads the processor of a `Transmuter.Repo` from its `preprocessor_config.json`, `processor_config.json` and tokenizer configuration.

  The image token is that of the tokenizer, as `LlavaProcessor` of Python transformers reads it, else the `image_token` of `processor_config.json`, else the `:image_token` option, the text of the token of `image_token_index` in `config.json`.
  """
  def load!(repo, opts \\ []) do
    opts = Keyword.validate!(opts, [:image_token])

    config =
      Hub.json(repo.source, "preprocessor_config.json", repo.opts) ||
        raise ArgumentError, "#{inspect(repo.source)} has no preprocessor_config.json"

    processor = Hub.json(repo.source, "processor_config.json", repo.opts) || %{}
    featurizer = Featurizer.new(config)

    {height, width} =
      size =
      Featurizer.size(featurizer) ||
        raise ArgumentError,
              "the LLaVA processor needs a size of preprocessor_config.json that does not depend on the image"

    patch =
      processor["patch_size"] ||
        raise ArgumentError, "processor_config.json of #{inspect(repo.source)} has no patch_size"

    count =
      div(height, patch) * div(width, patch) + (processor["num_additional_image_tokens"] || 0)

    count =
      if processor["vision_feature_select_strategy"] == "default", do: count - 1, else: count

    %__MODULE__{
      featurizer: featurizer,
      size: size,
      image_seq_len: count,
      image:
        Repo.special_token(repo, "image") || processor["image_token"] || opts[:image_token] ||
          raise(ArgumentError, "#{inspect(repo.source)} names no image token")
    }
  end

  @doc """
  The text of an image in the prompt before `process/3` expands it.
  """
  def placeholder(processor), do: processor.image

  @doc """
  Turns an image into pixel values and gives the text of the image in the prompt.

  Returns `{tensors, text}`. `tensors` has `"pixel_values"` of shape `{1, 3, height, width}` and type `:f32` on `backend`, and `"image_sizes"` of shape `{1, 2}` and type `:s64`, the height and the width of the image.
  """
  def process(processor, image, backend) do
    tensors = ImageTextToText.image_tensors(processor.featurizer, processor.size, image, backend)
    {tensors, String.duplicate(processor.image, processor.image_seq_len)}
  end

  @doc """
  Runs the vision encoder `encode`, a function from `Transmuter.compile/3`, on each image of a batch alone, and gives the inputs of `Transmuter.Generation.generate/5`.

  `tensors` has the `"pixel_values"` of each row, of shape `{batch, rows, 3, height, width}`, and the `"image_sizes"` of each row, of shape `{batch, rows, 2}`. Rows of zeros follow the images of a row, and they do not run. Returns `%{image_features: features}`: the features of the images of all rows, in order. The prompts take the default positions.
  """
  def encode(processor, encode, tensors, _prompts, _backend),
    do: %{image_features: ImageTextToText.encode_each(encode, tensors, processor.image_seq_len)}
end
