defmodule Transmuter.Vision.Gemma3Processor do
  @moduledoc """
  Prepares the images and the prompt of Gemma 3, as `Gemma3ImageProcessor` and `Gemma3Processor` of Python transformers do. `Transmuter.Vision.image_text_to_text/2` uses it.

  Each image gets 3 channels, the `size` of `preprocessor_config.json` (896x896 for Gemma 3) without its aspect ratio, and the rescale and the normalization of the configuration, as in `Transmuter.Featurizer`. Python transformers does pan and scan only when the caller asks for it, so the processor does not do it.

  In the prompt, each image is a placeholder `<start_of_image>`, and `process/3` gives the text that replaces it: `\\n\\n<start_of_image>`, `image_seq_length` (256) tokens `<image_soft_token>`, and `<end_of_image>\\n\\n`. The vision encoder gives one feature for each of these image tokens.
  """

  alias Transmuter.{Featurizer, Hub, Repo}

  defstruct [:featurizer, :image_seq_len, :image_token]

  @placeholder "<start_of_image>"

  @doc """
  Loads the processor of a `Transmuter.Repo` from its `preprocessor_config.json`, `processor_config.json` and `config.json`.
  """
  def load!(repo) do
    config =
      Hub.json(repo.source, "preprocessor_config.json", repo.opts) ||
        raise ArgumentError, "#{inspect(repo.source)} has no preprocessor_config.json"

    processor = Hub.json(repo.source, "processor_config.json", repo.opts) || %{}

    # Gemma3Processor of Python transformers always converts the images to RGB.
    %__MODULE__{
      featurizer: Featurizer.new(Map.put(config, "do_convert_rgb", true)),
      image_seq_len: processor["image_seq_length"] || 256,
      image_token: Repo.image_token_id(repo)
    }
  end

  @doc """
  The text of an image in the prompt before `process/3` expands it.
  """
  def placeholder(_processor), do: @placeholder

  @doc """
  Turns an image into pixel values and gives the text of the image in the prompt.

  Returns `{tensors, text}`. `tensors` has `"pixel_values"` of shape `{1, 3, height, width}` and type `:f32`, on `backend`.
  """
  def process(processor, image, backend) do
    pixels = Featurizer.process(processor.featurizer, [image], backend: backend)
    tokens = String.duplicate("<image_soft_token>", processor.image_seq_len)
    {%{"pixel_values" => pixels}, "\n\n" <> @placeholder <> tokens <> "<end_of_image>\n\n"}
  end

  @doc """
  The inputs of the vision encoder for one image, on `Nx.BinaryBackend`. The encoder compiles for them. The encoder takes one image in each call, so the template does not depend on `batch_size`.
  """
  def template(processor, _batch_size) do
    {height, width} = processor.featurizer.resize
    zero = Nx.tensor(0.0, backend: Nx.BinaryBackend)
    %{"pixel_values" => Nx.broadcast(zero, {1, 3, height, width})}
  end

  @doc """
  Runs the vision encoder `encode`, a function from `Transmuter.compile/3`, on the images of a batch, and gives the inputs of `Transmuter.Generation.generate/5`.

  `tensors` has the `"pixel_values"` of each row, of shape `{batch, rows, 3, height, width}`. Rows of zeros follow the images of a row. `prompts` are the token ids of each row, and the images of a row are its image tokens divided by `image_seq_length`. The encoder runs on each image alone, so the rows of zeros do not run. Returns `%{image_features: features}`: the features of the images of all rows, in order, of shape `{images * image_seq_length, hidden}`. The prompts take the default positions.
  """
  def encode(processor, encode, %{"pixel_values" => pixels}, prompts, _backend) do
    features =
      for {prompt, row} <- Enum.with_index(prompts),
          image <- 0..(images(processor, prompt) - 1)//1 do
        %{"image_features" => features} =
          encode.(%{"pixel_values" => Nx.slice_along_axis(pixels[row], image, 1, axis: 0)})

        {1, tokens, hidden} = Nx.shape(features)
        Nx.reshape(features, {tokens, hidden})
      end

    %{image_features: Nx.concatenate(features)}
  end

  defp images(processor, prompt),
    do: div(Enum.count(prompt, &(&1 == processor.image_token)), processor.image_seq_len)
end
