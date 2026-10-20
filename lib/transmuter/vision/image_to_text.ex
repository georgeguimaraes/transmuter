defmodule Transmuter.Vision.ImageToText do
  @moduledoc """
  The image-to-text task of encoder-decoder models, such as `VisionEncoderDecoderModel` of ViT-GPT2 and TrOCR. See `Transmuter.Vision.image_to_text/2`.

  The client turns each image into `pixel_values` with `Transmuter.Featurizer`, as the other image tasks do. The serving compiles the encoder for the batch size and the image size of the featurizer. The decoder generates from `decoder_start_token_id` and reads the output of the encoder (`encoder_hidden_states`) in each step.
  """

  alias Transmuter.{Featurizer, Generation, Serving, Tokenizer}

  @doc false
  def image_to_text(source, opts) do
    opts =
      Keyword.validate!(opts, [
        :compile,
        :variant,
        :cache_dir,
        :tokenizer,
        :featurizer,
        :backend,
        :compute_type,
        :max_new_tokens,
        :strategy,
        :temperature,
        :top_k,
        :top_p,
        :repetition_penalty,
        :seed,
        stream: false,
        stream_done: false,
        defn_options: []
      ])

    compile = Keyword.validate!(opts[:compile] || [], batch_size: 1)
    {repo, encoder, decoder} = Serving.load_encoder_decoder!(source, opts)
    tokenizer = opts[:tokenizer] || Tokenizer.load!(repo)
    featurizer = opts[:featurizer] || Featurizer.load!(repo)
    {height, width} = Serving.image_size!(featurizer)
    backend = opts[:backend] || Nx.default_backend()

    prompt = [
      repo.generation_config["decoder_start_token_id"] || repo.config["decoder_start_token_id"]
    ]

    opts = Keyword.put(opts, :max_new_tokens, Generation.max_new_tokens(repo, opts, 1))
    ctx = Generation.context(repo, decoder, opts, ["encoder_hidden_states"])

    init = fn defn_options ->
      shape = {compile[:batch_size], 3, height, width}
      template = Nx.broadcast(Nx.tensor(0.0, backend: Nx.BinaryBackend), shape)
      encode = Transmuter.compile(encoder, %{"pixel_values" => template}, defn_options)

      fn %{"pixel_values" => pixels}, prompts, generate ->
        %{"last_hidden_state" => hidden} = encode.(%{"pixel_values" => pixels})
        generate.(%{"encoder_hidden_states" => hidden}, prompts)
      end
    end

    preprocess = fn %{image: image, seed: seed} ->
      pixels =
        featurizer
        |> Featurizer.process([image], backend: backend)
        |> Nx.backend_transfer(Nx.BinaryBackend)

      tensors = %{"pixel_values" => pixels[0]}
      seed = seed || opts[:seed] || :rand.uniform(2 ** 32)
      summary = %{input: length(prompt), padding: 0}
      [%{prompt: prompt, seed: seed, tensors: tensors, summary: summary}]
    end

    postprocess = fn [ids], _runs -> %{results: [%{text: Tokenizer.decode(tokenizer, ids)}]} end

    task = %{validate: &validate/1, preprocess: preprocess, init: init, postprocess: postprocess}
    Serving.generation(ctx, tokenizer, task, Keyword.put(opts, :compile, compile))
  end

  defp validate(input = %{image: image}), do: validate(image, input[:seed])
  defp validate(image), do: validate(image, nil)

  defp validate(image, seed) do
    case Featurizer.tensor(image) do
      {:ok, image} ->
        {:ok, %{image: image, seed: seed}}

      {:error, _message} ->
        {:error,
         "expected an StbImage, a tensor of shape {height, width, channels} or %{image: image, seed: integer}, got: #{inspect(image)}"}
    end
  end
end
