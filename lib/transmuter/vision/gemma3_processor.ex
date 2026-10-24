defmodule Transmuter.Vision.Gemma3Processor do
  @moduledoc """
  Prepares the images and the prompt of Gemma 3, as `Gemma3ImageProcessor` and `Gemma3Processor` of Python transformers do. `Transmuter.Vision.image_text_to_text/2` uses it.

  Each image gets 3 channels, the `size` of `preprocessor_config.json` (896x896 for Gemma 3) without its aspect ratio, and the rescale and the normalization of the configuration, as in `Transmuter.Featurizer`. Python transformers does pan and scan only when the caller asks for it, so the processor does not do it.

  The image tokens come from the tokenizer configuration, as in `Gemma3Processor`: `boi_token` (`<start_of_image>`), `image_token` (`<image_soft_token>`) and `eoi_token` (`<end_of_image>`). In the prompt, each image is a placeholder `boi_token`, and `process/3` gives the text that replaces it: `\\n\\n`, `boi_token`, `image_seq_length` image tokens, `eoi_token` and `\\n\\n`. `image_seq_length` comes from `processor_config.json`, else from `mm_tokens_per_image` of `config.json`, else it is 256. The vision encoder gives one feature for each image token.
  """

  alias Transmuter.{Featurizer, Hub, Repo}
  alias Transmuter.Vision.ImageTextToText

  defstruct [:featurizer, :size, :image_seq_len, :boi, :image, :eoi]

  @doc """
  Loads the processor of a `Transmuter.Repo` from its `preprocessor_config.json`, `processor_config.json`, `config.json` and tokenizer configuration.
  """
  def load!(repo) do
    config =
      Hub.json(repo.source, "preprocessor_config.json", repo.opts) ||
        raise ArgumentError, "#{inspect(repo.source)} has no preprocessor_config.json"

    processor = Hub.json(repo.source, "processor_config.json", repo.opts) || %{}

    # Gemma3Processor of Python transformers always converts the images to RGB.
    featurizer = Featurizer.new(Map.put(config, "do_convert_rgb", true))

    %__MODULE__{
      featurizer: featurizer,
      size:
        Featurizer.size(featurizer) ||
          raise(
            ArgumentError,
            "the Gemma 3 processor needs a size of preprocessor_config.json that does not depend on the image"
          ),
      image_seq_len: processor["image_seq_length"] || repo.config["mm_tokens_per_image"] || 256,
      boi: token!(repo, "boi"),
      image: token!(repo, "image"),
      eoi: token!(repo, "eoi")
    }
  end

  defp token!(repo, name) do
    Repo.special_token(repo, name) ||
      raise ArgumentError,
            "the tokenizer configuration of #{inspect(repo.source)} has no #{name}_token"
  end

  @doc """
  The text of an image in the prompt before `process/3` expands it.
  """
  def placeholder(processor), do: processor.boi

  @doc """
  Turns an image into pixel values and gives the text of the image in the prompt.

  Returns `{tensors, text}`. `tensors` has `"pixel_values"` of shape `{1, 3, height, width}` and type `:f32` on `backend`, and `"image_sizes"` of shape `{1, 2}` and type `:s64`, the height and the width of the image. Each pixel value row of a batch has 3 * height * width values (9.6 MB for 896x896), also a row of zeros that fills a chat to its batch key.
  """
  def process(processor, image, backend) do
    pixels = Featurizer.process(processor.featurizer, [image], backend: backend)
    {1, _channels, height, width} = Nx.shape(pixels)
    sizes = Nx.tensor([[height, width]], type: :s64, backend: Nx.BinaryBackend)
    tokens = String.duplicate(processor.image, processor.image_seq_len)
    text = "\n\n" <> processor.boi <> tokens <> processor.eoi <> "\n\n"
    {%{"pixel_values" => pixels, "image_sizes" => sizes}, text}
  end

  @doc """
  The inputs of the vision encoder for one image, on `Nx.BinaryBackend`. The encoder compiles for them. The encoder takes one image in each call, so the template does not depend on `batch_size`.
  """
  def template(processor, _batch_size) do
    {height, width} = processor.size
    zero = Nx.tensor(0.0, backend: Nx.BinaryBackend)
    %{"pixel_values" => Nx.broadcast(zero, {1, 3, height, width})}
  end

  @doc """
  Runs the vision encoder `encode`, a function from `Transmuter.compile/3`, on the images of a batch, and gives the inputs of `Transmuter.Generation.generate/5`.

  `tensors` has the `"pixel_values"` of each row, of shape `{batch, rows, 3, height, width}`, and the `"image_sizes"` of each row, of shape `{batch, rows, 2}`. Rows of zeros follow the images of a row. Each image runs alone, so the encoder has one compiled shape and the memory of one image, and the rows of zeros do not run. Returns `%{image_features: features}`: the features of the images of all rows, in order, of shape `{images * image_seq_length, hidden}`. The prompts take the default positions.
  """
  def encode(processor, encode, tensors, _prompts, _backend) do
    %{"pixel_values" => pixels, "image_sizes" => sizes} = tensors

    features =
      for row <- ImageTextToText.images(pixels, sizes, fn _size -> 1 end),
          {_size, image} <- row do
        %{"image_features" => features} = encode.(%{"pixel_values" => image})
        {1, tokens, hidden} = Nx.shape(features)

        if tokens != processor.image_seq_len do
          raise ArgumentError,
                "the vision encoder gives #{tokens} features for each image, and image_seq_length is #{processor.image_seq_len}"
        end

        Nx.reshape(features, {tokens, hidden})
      end

    %{image_features: Nx.concatenate(features)}
  end
end
