defmodule Transmuter.Vision.ImageTextToText do
  @moduledoc """
  The image-text-to-text task of decoder-only vision language models, such as SmolVLM, Qwen2-VL, Qwen3-VL, Ministral 3 and Gemma 3. See `Transmuter.Vision.image_text_to_text/2`.

  The repository has three models, as the `ImageTextToText` exports of transformers.js: `vision_encoder` turns the pixel values of the images into features, `embed_tokens` turns token ids into embeddings, and `decoder_model_merged` generates from the embeddings. `Transmuter.Generation` puts the features in the place of the image tokens of the prompt.

  The processor of the model comes from `image_processor_type` in `preprocessor_config.json`: `Transmuter.Vision.Idefics3Processor`, `Transmuter.Vision.Qwen2VLProcessor`, `Transmuter.Vision.PixtralProcessor` or `Transmuter.Vision.Gemma3Processor`. The Qwen2-VL processor also gets the input names of the vision encoder. The client renders the chat template, turns each image into tensors with the processor, and replaces the placeholder of each image in the prompt with its image tokens. The tensors of the images of a chat are one row of the batch. The rows of their pixel values (tiles, patches, blocks or images) and of the other tensors are concatenated. The number of rows of the pixel values, rounded up to a power of two, is the batch key, and rows of zeros fill each tensor to that number of rows. In the serving process, the processor runs the vision encoder on the images of the batch and gives the features and the positions of each prompt to `Transmuter.Generation`.

  The tokenizer adds its special tokens to the rendered prompt, unless the prompt starts with the bos token, as `apply_chat_template` of the processors of Python transformers.
  """

  @compile {:no_warn_undefined, [Tokenizers.Tokenizer, Tokenizers.Encoding]}

  alias Transmuter.{ChatTemplate, Featurizer, Generation, Hub, Repo, Serving, Tokenizer}
  alias Transmuter.Vision.{Gemma3Processor, Idefics3Processor, PixtralProcessor, Qwen2VLProcessor}

  # The batch keys: the number of rows of the pixel values of a chat (tiles, patches, blocks or
  # images), rounded up to a power of two, or 0 for a chat without images.
  @buckets [0 | for(exponent <- 0..16, do: 2 ** exponent)]

  @doc false
  def image_text_to_text(source, opts) do
    opts =
      Keyword.validate!(
        opts,
        [
          :variant,
          :tokenizer,
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
          :seed,
          :template_options,
          token_logprobs: false,
          stream: false,
          stream_done: false,
          compile: [],
          defn_options: []
        ] ++ Hub.options()
      )
      |> Serving.generation_options!()

    compile = Keyword.validate!(opts[:compile], batch_size: 1)
    repo = Repo.load!(source, Keyword.take(opts, [:variant | Hub.options()]))
    model_opts = Keyword.take(opts, [:backend, :compute_type])
    vision = Repo.load_component!(repo, "vision_encoder", :q4, model_opts)
    embed = Repo.load_component!(repo, "embed_tokens", :q8, model_opts)
    decoder_opts = [outputs: &Generation.outputs/1] ++ model_opts
    decoder = Repo.load_component!(repo, "decoder_model_merged", :q4, decoder_opts)
    tokenizer = opts[:tokenizer] || Tokenizer.load!(repo)
    {module, processor} = processor!(repo, vision)
    backend = opts[:backend] || Nx.default_backend()

    ctx =
      Generation.context(
        repo,
        decoder,
        [embed: embed, image_token: Repo.image_token_id(repo)] ++ opts
      )

    bos = Repo.special_token(repo, "bos")

    init = fn defn_options ->
      template = module.template(processor, compile[:batch_size])
      encode = Transmuter.compile(vision, template, [cache_size: 64] ++ defn_options)
      backend = Nx.Defn.to_backend(defn_options)

      fn
        tensors, prompts, generate when map_size(tensors) == 0 ->
          generate.(%{}, prompts)

        tensors, prompts, generate ->
          generate.(module.encode(processor, encode, tensors, prompts, backend), prompts)
      end
    end

    preprocess = fn input ->
      {text, images} = prompt(repo, input, opts)

      %{prompt: text, tensors: tensors, key: key} =
        expand!({module, processor}, text, images, backend)

      seed = input[:seed] || opts[:seed] || :rand.uniform(2 ** 32)
      ids = token_ids(tokenizer, text, bos)
      summary = %{input: length(ids), padding: 0}
      [%{prompt: ids, seed: seed, tensors: tensors, key: key, summary: summary}]
    end

    task = %{
      validate: &validate/1,
      preprocess: preprocess,
      init: init,
      postprocess: Serving.text_postprocess(tokenizer, ctx.token_logprobs?),
      keys: @buckets
    }

    Serving.generation(ctx, tokenizer, task, Keyword.put(opts, :compile, compile))
  end

  # The lookup takes exact names, as AutoImageProcessor of Python transformers, and not the
  # name without "Fast". A fast image processor can differ from the slow one: for example,
  # Idefics3ImageProcessorFast resizes with bicubic in place of the lanczos of
  # Idefics3ImageProcessor, because torchvision has no lanczos for tensors.
  defp processor!(repo, vision) do
    qwen2_vl = fn -> {Qwen2VLProcessor, Qwen2VLProcessor.load!(repo, inputs: vision.inputs)} end
    pixtral = fn -> {PixtralProcessor, PixtralProcessor.load!(repo)} end
    gemma3 = fn -> {Gemma3Processor, Gemma3Processor.load!(repo)} end

    processors = %{
      "Idefics3ImageProcessor" => fn -> {Idefics3Processor, Idefics3Processor.load!(repo)} end,
      "Qwen2VLImageProcessor" => qwen2_vl,
      "Qwen2VLImageProcessorFast" => qwen2_vl,
      "PixtralImageProcessor" => pixtral,
      "PixtralImageProcessorFast" => pixtral,
      "Gemma3ImageProcessor" => gemma3,
      "Gemma3ImageProcessorFast" => gemma3
    }

    config = Hub.json(repo.source, "preprocessor_config.json", repo.opts) || %{}

    case processors[config["image_processor_type"]] do
      nil ->
        {last, names} = processors |> Map.keys() |> Enum.sort() |> List.pop_at(-1)

        raise ArgumentError,
              "image_text_to_text supports #{Enum.join(names, ", ")} and #{last}, got: #{inspect(config["image_processor_type"])}"

      load ->
        load.()
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

  # Each placeholder of the rendered prompt takes the text of its image. The tensors of all
  # images go into one row, with rows of zeros up to the batch key.
  defp expand!({module, processor}, text, images, backend) do
    [first | rest] = String.split(text, module.placeholder(processor))

    if length(rest) != length(images) do
      raise ArgumentError,
            "the prompt has #{length(rest)} image placeholders, and the chat has #{length(images)} images"
    end

    {tensors, texts} =
      images |> Enum.map(&module.process(processor, &1, backend)) |> Enum.unzip()

    prompt = Enum.join([first | Enum.zip_with(texts, rest, &(&1 <> &2))])
    count = Enum.sum_by(tensors, &Nx.axis_size(&1["pixel_values"], 0))

    key =
      Enum.find(@buckets, &(&1 >= count)) ||
        raise ArgumentError,
              "the images of the chat give #{count} rows of pixel values, and a chat takes up to #{List.last(@buckets)}"

    tensors =
      for name <- tensors |> Enum.flat_map(&Map.keys/1) |> Enum.uniq(), into: %{} do
        tensor = tensors |> Enum.map(& &1[name]) |> Nx.concatenate()
        [_rows | axes] = List.duplicate({0, 0, 0}, Nx.rank(tensor))
        tensor = Nx.pad(tensor, 0, [{0, key - Nx.axis_size(tensor, 0), 0} | axes])
        {name, Nx.backend_transfer(tensor, Nx.BinaryBackend)}
      end

    %{prompt: prompt, tensors: tensors, key: key}
  end

  # As apply_chat_template of ProcessorMixin of Python transformers: a prompt that starts
  # with the bos token, such as that of Ministral 3, gets no more special tokens.
  defp token_ids(tokenizer, text, bos) do
    special? = not (is_binary(bos) and String.starts_with?(text, bos))

    {:ok, encoding} =
      Tokenizers.Tokenizer.encode(tokenizer.native, text, add_special_tokens: special?)

    Tokenizer.ids(tokenizer, encoding)
  end

  @doc false
  # The images of each row of a batch, before the rows of zeros, as {size, pixels}: a row
  # of `sizes` (such as the grid of the image) and the `count.(size)` rows of `pixels` of the
  # image. A size of zeros is padding.
  def images(pixels, sizes, count) do
    for {sizes, row} <- sizes |> Nx.to_list() |> Enum.with_index() do
      sizes
      |> Enum.filter(fn [first | _] -> first > 0 end)
      |> Enum.map_reduce(0, fn size, start ->
        {{size, Nx.slice_along_axis(pixels[row], start, count.(size), axis: 0)},
         start + count.(size)}
      end)
      |> elem(0)
    end
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
