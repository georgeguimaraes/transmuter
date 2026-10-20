defmodule Transmuter.Vision.ImageTextToText do
  @moduledoc """
  The image-text-to-text task of decoder-only vision language models, such as SmolVLM. See `Transmuter.Vision.image_text_to_text/2`.

  The repository has three models, as the `ImageTextToText` exports of transformers.js: `vision_encoder` turns the tiles of the images into features, `embed_tokens` turns token ids into embeddings, and `decoder_model_merged` generates from the embeddings. `Transmuter.Generation` puts the features in the place of the image tokens of the prompt.

  The client renders the chat template, turns each image into tiles with `Transmuter.Vision.Idefics3Processor`, and replaces the placeholder of each image in the prompt with its image tokens. The tiles of a chat are one row of the batch. Their count, rounded up to a power of two, is the batch key, and tiles of zeros fill the row to that count. The vision encoder removes tiles of zeros, so its features are those of the real tiles of all rows, in order.
  """

  @compile {:no_warn_undefined, [Tokenizers.Tokenizer, Tokenizers.Encoding]}

  alias Transmuter.{ChatTemplate, Featurizer, Generation, Hub, Repo, Serving, Tokenizer}
  alias Transmuter.Vision.Idefics3Processor

  # The batch keys: the number of tiles of a chat, rounded up to a power of two, or 0 for a
  # chat without images.
  @buckets [0, 1, 2, 4, 8, 16, 32, 64]

  @doc false
  def image_text_to_text(source, opts) do
    opts =
      Keyword.validate!(opts, [
        :variant,
        :cache_dir,
        :tokenizer,
        :backend,
        :compute_type,
        :max_new_tokens,
        :strategy,
        :temperature,
        :top_k,
        :top_p,
        :repetition_penalty,
        :seed,
        :template_options,
        stream: false,
        stream_done: false,
        compile: [],
        defn_options: []
      ])

    compile = Keyword.validate!(opts[:compile], batch_size: 1)
    repo = Repo.load!(source, Keyword.take(opts, [:variant, :cache_dir]))
    model_opts = Keyword.take(opts, [:backend, :compute_type])
    vision = Repo.load_component!(repo, "vision_encoder", :q4, model_opts)
    embed = Repo.load_component!(repo, "embed_tokens", :q8, model_opts)
    decoder = Repo.load_component!(repo, "decoder_model_merged", :q4, model_opts)
    tokenizer = opts[:tokenizer] || Tokenizer.load!(repo)
    processor = processor!(repo)
    backend = opts[:backend] || Nx.default_backend()

    image_token = repo.config["image_token_id"]
    ctx = Generation.context(repo, decoder, [embed: embed, image_token: image_token] ++ opts)

    init = fn defn_options ->
      template = Idefics3Processor.template(processor, compile[:batch_size])
      encode = Transmuter.compile(vision, template, [cache_size: 64] ++ defn_options)
      backend = Nx.Defn.to_backend(defn_options)

      fn
        %{"pixel_values" => pixels}, prompts, generate ->
          features = Idefics3Processor.encode(encode, pixels, backend)
          generate.(%{image_features: features}, prompts)

        _no_images, prompts, generate ->
          generate.(%{}, prompts)
      end
    end

    preprocess = fn input ->
      {text, images} = prompt(repo, input, opts)
      %{prompt: text, tensors: tensors, key: key} = expand!(processor, text, images, backend)
      seed = input[:seed] || opts[:seed] || :rand.uniform(2 ** 32)
      ids = token_ids(tokenizer, text)
      summary = %{input: length(ids), padding: 0}
      [%{prompt: ids, seed: seed, tensors: tensors, key: key, summary: summary}]
    end

    postprocess = fn [ids], [run] ->
      summary = Map.put(run.summary, :output, length(ids))
      %{results: [%{text: Tokenizer.decode(tokenizer, ids), token_summary: summary}]}
    end

    task = %{
      validate: &validate/1,
      preprocess: preprocess,
      init: init,
      postprocess: postprocess,
      keys: @buckets
    }

    Serving.generation(ctx, tokenizer, task, Keyword.put(opts, :compile, compile))
  end

  defp processor!(repo) do
    case Hub.json(repo.source, "preprocessor_config.json", repo.opts) do
      %{"image_processor_type" => "Idefics3ImageProcessor"} ->
        Idefics3Processor.load!(repo)

      config ->
        raise ArgumentError,
              "image_text_to_text supports Idefics3ImageProcessor, got: #{inspect(config["image_processor_type"])}"
    end
  end

  # The chat template renders an image part without its image. As in the pipeline of
  # Python transformers, a chat that ends with an assistant message continues that message.
  defp prompt(repo, %{messages: messages}, opts) do
    {messages, images} = Enum.map_reduce(messages, [], &take_images/2)
    final = List.last(messages)

    template_opts = [
      template_options: opts[:template_options],
      continue_final_message: (final[:role] || final["role"]) == "assistant"
    ]

    {ChatTemplate.render(repo, messages, template_opts), Enum.reverse(images)}
  end

  defp take_images(message, images) do
    case content(message) do
      parts when is_list(parts) ->
        {parts, images} =
          Enum.map_reduce(parts, images, fn part, images ->
            case image(part) do
              nil -> {part, images}
              image -> {Map.drop(part, [:image, "image"]), [image | images]}
            end
          end)

        {put_content(message, parts), images}

      _text ->
        {message, images}
    end
  end

  defp content(message), do: message[:content] || message["content"]

  defp put_content(message, parts) when is_map_key(message, :content),
    do: %{message | content: parts}

  defp put_content(message, parts), do: %{message | "content" => parts}

  defp image(part = %{}) do
    if (part[:type] || part["type"]) == "image", do: part[:image] || part["image"]
  end

  defp image(_part), do: nil

  # Each placeholder of the rendered prompt takes the text of its image, and the tiles of all
  # images go into one row, with tiles of zeros up to the batch key.
  defp expand!(processor, text, images, backend) do
    [first | rest] = String.split(text, Idefics3Processor.placeholder())

    if length(rest) != length(images) do
      raise ArgumentError,
            "the prompt has #{length(rest)} image placeholders, and the chat has #{length(images)} images"
    end

    {pixels, texts} =
      images |> Enum.map(&Idefics3Processor.process(processor, &1, backend)) |> Enum.unzip()

    prompt = Enum.join([first | Enum.zip_with(texts, rest, &(&1 <> &2))])
    count = Enum.sum_by(pixels, &Nx.axis_size(&1, 0))

    key =
      Enum.find(@buckets, &(&1 >= count)) ||
        raise ArgumentError,
              "the images of the chat give #{count} tiles, and a chat takes up to #{List.last(@buckets)}"

    tensors =
      if pixels == [] do
        %{}
      else
        pixels = Nx.concatenate(pixels)
        [_count | axes] = List.duplicate({0, 0, 0}, Nx.rank(pixels))
        pixels = Nx.pad(pixels, 0.0, [{0, key - count, 0} | axes])
        %{"pixel_values" => Nx.backend_transfer(pixels, Nx.BinaryBackend)}
      end

    %{prompt: prompt, tensors: tensors, key: key}
  end

  # As Idefics3Processor of Python transformers, the tokenizer adds its special tokens.
  defp token_ids(tokenizer, text) do
    {:ok, encoding} =
      Tokenizers.Tokenizer.encode(tokenizer.native, text, add_special_tokens: true)

    Tokenizers.Encoding.get_ids(encoding)
  end

  defp validate(input = %{messages: [_ | _] = messages}) do
    {_messages, images} = Enum.map_reduce(messages, [], &take_images/2)

    case Enum.find(images, &match?({:error, _}, Featurizer.tensor(&1))) do
      nil ->
        {:ok, input}

      image ->
        {:error,
         "expected an StbImage or a tensor of shape {height, width, channels}, got: #{inspect(image)}"}
    end
  end

  defp validate(input) do
    {:error,
     ~s(expected %{messages: [%{role: role, content: [%{type: "image", image: image}, %{type: "text", text: text}]}]}, got: #{inspect(input)})}
  end
end
