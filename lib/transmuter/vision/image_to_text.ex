defmodule Transmuter.Vision.ImageToText do
  @moduledoc """
  The image-to-text task of encoder-decoder models, such as `VisionEncoderDecoderModel` of ViT-GPT2 and TrOCR. See `Transmuter.Vision.image_to_text/2`.

  The client turns each image into `pixel_values` with `Transmuter.Featurizer`, as the other image tasks do. The serving compiles the encoder for the batch size and the image size of the featurizer. The decoder generates from `decoder_start_token_id` and reads the output of the encoder (`encoder_hidden_states`) in each step.

  An encoder that takes `inputs_embeds` in place of `pixel_values`, as in the Florence-2 exports, reads the features of the image and the embeddings of a prompt. The repository then also has `vision_encoder.onnx` and `embed_tokens.onnx`. `Transmuter.Vision.Florence2Processor` gives the prompt of the task, and the tokenizer pads it to a sequence length of the `:compile` option, as in `Transmuter.Text.text_to_text/2`. The encoder takes the image features and then the embeddings of the prompt, with an attention mask that hides the padding. The decoder reads this mask in `encoder_attention_mask`, so the padding has no effect on the result. The sequence length is the batch key, so inputs with different tasks generate in one batch when their prompts pad to the same length.
  """

  alias Transmuter.{Featurizer, Generation, Hub, Repo, Serving, Tokenizer}
  alias Transmuter.Vision.Florence2Processor

  @doc false
  def image_to_text(source, opts) do
    opts =
      Keyword.validate!(
        opts,
        [
          :compile,
          :variant,
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
          :num_beams,
          :length_penalty,
          :early_stopping,
          :task,
          :seed,
          token_logprobs: false,
          stream: false,
          stream_done: false,
          defn_options: []
        ] ++ Hub.options()
      )
      |> Serving.generation_options!()

    {repo, encoder, decoder} = Serving.load_encoder_decoder!(source, opts)
    tokenizer = opts[:tokenizer] || Tokenizer.load!(repo)
    featurizer = opts[:featurizer] || Featurizer.load!(repo)
    opts = Keyword.put(opts, :max_new_tokens, Generation.max_new_tokens(repo, opts, 1))
    models = %{repo: repo, encoder: encoder, decoder: decoder}

    {ctx, task, compile} =
      if "inputs_embeds" in encoder.inputs,
        do: embeds_task(models, tokenizer, featurizer, opts),
        else: pixels_task(models, tokenizer, featurizer, opts)

    Serving.generation(ctx, tokenizer, task, Keyword.put(opts, :compile, compile))
  end

  defp pixels_task(%{repo: repo, encoder: encoder, decoder: decoder}, tokenizer, featurizer, opts) do
    if opts[:task] do
      raise ArgumentError,
            ":task applies only to an encoder that takes inputs_embeds, such as Florence-2"
    end

    compile = Keyword.validate!(opts[:compile] || [], batch_size: 1)
    {height, width} = Serving.image_size!(featurizer)
    backend = opts[:backend] || Nx.default_backend()
    prompt = [start_token(repo)]
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

    preprocess = fn input ->
      tensors = %{"pixel_values" => pixels(featurizer, input.image, backend)}
      seed = input.seed || opts[:seed] || :rand.uniform(2 ** 32)
      summary = %{input: length(prompt), padding: 0}
      [%{prompt: prompt, seed: seed, tensors: tensors, summary: summary}]
    end

    task = %{
      validate: &validate(&1, false),
      preprocess: preprocess,
      init: init,
      postprocess: Serving.text_postprocess(tokenizer, ctx.token_logprobs?)
    }

    {ctx, task, compile}
  end

  # Florence-2: the features of the image and the embeddings of the prompt go into the encoder.
  # The prompt pads to a sequence length of :compile, as in Transmuter.Text.text_to_text/2, and
  # the attention mask hides the padding.
  defp embeds_task(models, tokenizer, featurizer, opts) do
    %{repo: repo, encoder: encoder, decoder: decoder} = models
    model_opts = Keyword.take(opts, [:backend, :compute_type])
    vision = Repo.load_component!(repo, "vision_encoder", :fp32, model_opts)
    embed = Repo.load_component!(repo, "embed_tokens", :fp32, model_opts)
    processor = Florence2Processor.load!(repo)
    given = ~w(encoder_hidden_states encoder_attention_mask)
    ctx = Generation.context(repo, decoder, [embed: embed] ++ opts, given)

    # The image tokens and the prompt share the positions of the encoder.
    limit = repo.config["text_config"]["max_position_embeddings"] - processor.image_tokens

    compile =
      Keyword.validate!(opts[:compile] || [],
        batch_size: 1,
        sequence_length: Serving.sequence_lengths(repo, limit)
      )

    lengths = List.wrap(compile[:sequence_length])
    text_tokenizer = Tokenizer.configure(tokenizer, length: lengths)
    {height, width} = Serving.image_size!(featurizer)
    backend = opts[:backend] || Nx.default_backend()
    prompt = [start_token(repo)]
    default_task = opts[:task] || "<CAPTION>"

    init = fn defn_options ->
      rows = compile[:batch_size]
      cache = [cache_size: length(lengths)] ++ defn_options
      %{inputs: [ids_name], outputs: [embeds_name]} = embed
      [features_name] = vision.outputs
      {embeds_type, [_, _, hidden]} = encoder.input_types["inputs_embeds"]
      {mask_type, _dims} = encoder.input_types["attention_mask"]
      sequence = processor.image_tokens + hd(lengths)

      see =
        Transmuter.compile(
          vision,
          %{"pixel_values" => zeros({rows, 3, height, width}, :f32)},
          defn_options
        )

      embed_text =
        Transmuter.compile(embed, %{ids_name => zeros({rows, hd(lengths)}, :s64)}, cache)

      encode =
        Transmuter.compile(
          encoder,
          %{
            "inputs_embeds" => zeros({rows, sequence, hidden}, embeds_type),
            "attention_mask" => zeros({rows, sequence}, mask_type)
          },
          cache
        )

      fn inputs, prompts, generate ->
        features = see.(%{"pixel_values" => inputs["pixel_values"]})[features_name]
        text = embed_text.(%{ids_name => inputs["input_ids"]})[embeds_name]
        {rows, count, _hidden} = Nx.shape(features)

        image_mask =
          Nx.broadcast(
            Nx.tensor(1, type: mask_type, backend: Nx.Defn.to_backend(defn_options)),
            {rows, count}
          )

        mask =
          Nx.concatenate([image_mask, Nx.as_type(inputs["attention_mask"], mask_type)], axis: 1)

        embeds = Nx.concatenate([features, Nx.as_type(text, Nx.type(features))], axis: 1)

        %{"last_hidden_state" => hidden} =
          encode.(%{"inputs_embeds" => embeds, "attention_mask" => mask})

        generate.(%{"encoder_hidden_states" => hidden, "encoder_attention_mask" => mask}, prompts)
      end
    end

    preprocess = fn input ->
      task = input.task || default_task
      text = Florence2Processor.prompt!(processor, task, input.text)
      {tensors, _encodings} = Tokenizer.encode(text_tokenizer, [text])

      tensors =
        for {name, tensor} <- Map.take(tensors, ["input_ids", "attention_mask"]),
            into: %{"pixel_values" => pixels(featurizer, input.image, backend)},
            do: {name, tensor[0]}

      seed = input.seed || opts[:seed] || :rand.uniform(2 ** 32)
      length = Nx.size(tensors["input_ids"])
      count = Nx.to_number(Nx.sum(tensors["attention_mask"]))
      {image_height, image_width, _channels} = Nx.shape(input.image)

      [
        %{
          prompt: prompt,
          seed: seed,
          tensors: tensors,
          key: length,
          summary: %{input: count, padding: length - count},
          task: task,
          size: {image_height, image_width}
        }
      ]
    end

    postprocess = fn [integers], [run] ->
      {ids, logprobs} = Serving.tokens(integers, ctx.token_logprobs?)
      text = Tokenizer.decode(tokenizer, ids, skip_special_tokens: false)
      result = Florence2Processor.parse(processor, run.task, text, run.size)
      %{results: [Serving.token_logprobs(result, logprobs, ctx.token_logprobs?)]}
    end

    task = %{
      validate: &validate(&1, true),
      preprocess: preprocess,
      init: init,
      postprocess: postprocess,
      keys: lengths
    }

    {ctx, task, compile}
  end

  defp start_token(repo),
    do: repo.generation_config["decoder_start_token_id"] || repo.config["decoder_start_token_id"]

  defp pixels(featurizer, image, backend) do
    featurizer
    |> Featurizer.process([image], backend: backend)
    |> Nx.backend_transfer(Nx.BinaryBackend)
    |> then(& &1[0])
  end

  defp zeros(shape, type),
    do: Nx.broadcast(Nx.tensor(0, type: type, backend: Nx.BinaryBackend), shape)

  defp validate(input = %{image: image}, tasks?) do
    cond do
      not tasks? and (Map.has_key?(input, :task) or Map.has_key?(input, :text)) ->
        {:error, "the model takes no :task or :text, got: #{inspect(input)}"}

      not (input[:text] == nil or is_binary(input[:text])) ->
        {:error, "expected :text to be a string, got: #{inspect(input[:text])}"}

      true ->
        with {:ok, image} <- image(image, tasks?) do
          {:ok, %{image: image, seed: input[:seed], task: input[:task], text: input[:text]}}
        end
    end
  end

  defp validate(image, tasks?), do: validate(%{image: image}, tasks?)

  defp image(image, tasks?) do
    case Featurizer.tensor(image) do
      {:ok, image} ->
        {:ok, image}

      {:error, _message} ->
        map =
          if tasks?,
            do: "%{image: image, task: task, text: text, seed: integer}",
            else: "%{image: image, seed: integer}"

        {:error,
         "expected an StbImage, a tensor of shape {height, width, channels} or #{map}, got: #{inspect(image)}"}
    end
  end
end
