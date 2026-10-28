defmodule Transmuter.Vision.ImageTextToText do
  @moduledoc """
  The image-text-to-text task of decoder-only vision language models, such as SmolVLM, Qwen2-VL, Qwen2.5-VL, Qwen3-VL, Ministral 3, Gemma 3 and LLaVA. See `Transmuter.Vision.image_text_to_text/2`.

  The repository has three models, as the `ImageTextToText` exports of transformers.js: `vision_encoder` turns the pixel values of the images into features, `embed_tokens` turns token ids into embeddings, and `decoder_model_merged` generates from the embeddings. `Transmuter.Generation` puts the features in the place of the image tokens of the prompt.

  The processor of the model comes from its `processor_class`, as in `AutoProcessor` of Python transformers: `Transmuter.Vision.Idefics3Processor`, `Transmuter.Vision.Qwen2VLProcessor`, `Transmuter.Vision.PixtralProcessor`, `Transmuter.Vision.Gemma3Processor` or `Transmuter.Vision.LlavaProcessor`. The chat template is that of the processor. The Qwen2-VL processor also gets the input names of the vision encoder. The client renders the chat template, turns each image into tensors with the processor, and replaces the placeholder of each image in the prompt with its image tokens. The tensors of the images of a chat are one row of the batch. The rows of their pixel values (tiles, patches, blocks or images) and of the other tensors are concatenated. The number of rows of the pixel values, rounded up to a power of two, is the batch key, and rows of zeros fill each tensor to that number of rows. In the serving process, the processor runs the vision encoder on the images of the batch and gives the features and the positions of each prompt to `Transmuter.Generation`.

  The tokenizer adds its special tokens to the rendered prompt, unless the prompt starts with the bos token, as `apply_chat_template` of the processors of Python transformers.
  """

  @compile {:no_warn_undefined, [Tokenizers.Tokenizer, Tokenizers.Encoding, Transmuter.EMLX]}

  alias Transmuter.{ChatTemplate, Featurizer, Generation, Hub, Repo, Serving, Tokenizer}

  alias Transmuter.Vision.{
    Gemma3Processor,
    Idefics3Processor,
    LlavaProcessor,
    PixtralProcessor,
    Qwen2VLProcessor
  }

  # The batch keys: the number of rows of the pixel values of a chat (tiles, patches, blocks or
  # images), rounded up to a power of two, or 0 for a chat without images.
  @buckets [0 | for(exponent <- 0..16, do: 2 ** exponent)]

  @options [
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
             defn_options: []
           ] ++ Hub.options()

  # The rows of the batch axis of one call of the vision encoder, at most. A call keeps the
  # device until it ends, also for the steps of a server of ContinuousGeneration on the same
  # device, so smaller calls make the steps wait less. See in_chunks/3.
  @encode_rows 1

  @doc false
  # The options of the task without the options of the serving.
  def options, do: @options

  @doc false
  def image_text_to_text(source, opts) do
    opts =
      opts
      |> Keyword.validate!(@options ++ [stream: false, stream_done: false, compile: []])
      |> Serving.generation_options!()

    compile = Keyword.validate!(opts[:compile], batch_size: 1)
    opts = Keyword.put(opts, :compile, compile)
    {ctx, tokenizer, task} = task(source, opts)
    Serving.generation(ctx, tokenizer, task, opts)
  end

  @doc false
  # The context, the tokenizer and the task of `Serving.generation/4`, for image_text_to_text/2
  # and `Transmuter.ContinuousGeneration`. The vision encoder compiles for `:batch_size` of
  # `:compile` rows, 1 without it.
  def task(source, opts) do
    batch_size = get_in(opts, [:compile, :batch_size]) || 1
    repo = Repo.load!(source, Keyword.take(opts, [:variant | Hub.options()]))
    tokenizer = Serving.tokenizer(repo, opts)
    {image_token, image_text} = image_token!(repo, tokenizer)

    chat_template =
      repo.processor_chat_template ||
        raise ArgumentError,
              "#{inspect(repo.source)} has no chat template of the processor in chat_template.json, chat_template.jinja or processor_config.json"

    # The processor reads its configuration before the larger models load, so a bad
    # configuration raises early. The Qwen2-VL processor needs the inputs of the vision encoder.
    model_opts = Keyword.take(opts, [:backend, :compute_type])
    vision = Repo.load_component!(repo, "vision_encoder", :q4, model_opts)
    {module, processor} = processor!(repo, vision, image_text)
    embed = Repo.load_component!(repo, "embed_tokens", :q8, model_opts)
    decoder_opts = [outputs: &Generation.outputs/1] ++ model_opts
    decoder = Repo.load_component!(repo, "decoder_model_merged", :q4, decoder_opts)
    backend = opts[:backend] || Nx.default_backend()
    ctx = Generation.context(repo, decoder, [embed: embed, image_token: image_token] ++ opts)

    bos = Repo.special_token(repo, "bos")

    prepare = fn defn_options ->
      template = module.template(processor, batch_size)
      encode = Transmuter.compile(vision, template, [cache_size: 64] ++ defn_options)
      encode = in_chunks(encode, vision, @encode_rows)
      backend = Nx.Defn.to_backend(defn_options)

      fn
        tensors, _prompts when map_size(tensors) == 0 -> %{}
        tensors, prompts -> module.encode(processor, encode, tensors, prompts, backend)
      end
    end

    preprocess = fn input ->
      {text, images} = prompt(repo, chat_template, input, opts)

      %{prompt: text, tensors: tensors, key: key, texts: texts} =
        expand!({module, processor}, text, images, backend)

      seed = input[:seed] || opts[:seed] || :rand.uniform(2 ** 32)
      ids = token_ids(tokenizer, text, bos)
      check_image_tokens!(ids, texts, image_token, image_text)
      summary = %{input: length(ids), padding: 0}
      [%{prompt: ids, seed: seed, tensors: tensors, key: key, summary: summary}]
    end

    task = %{
      validate: &validate/1,
      preprocess: preprocess,
      prepare: prepare,
      init: Serving.prepare_init(prepare),
      postprocess: Serving.text_postprocess(tokenizer, ctx.token_logprobs?),
      keys: @buckets
    }

    {ctx, tokenizer, task}
  end

  # The id of the image token from config.json and its text from the tokenizer.
  defp image_token!(repo, tokenizer) do
    id =
      Repo.image_token_id(repo) ||
        raise ArgumentError,
              "config.json of #{inspect(repo.source)} has no image_token_id or image_token_index"

    text =
      Tokenizers.Tokenizer.id_to_token(tokenizer.native, id) ||
        raise ArgumentError, "the tokenizer has no token for the image token id #{id}"

    {id, text}
  end

  # The processor classes of Python transformers that the serving supports. SmolVLMProcessor
  # prepares images as Idefics3Processor does, and Qwen2_5_VLProcessor and Qwen3VLProcessor as
  # Qwen2VLProcessor does.
  @processors %{
    "Idefics3Processor" => Idefics3Processor,
    "SmolVLMProcessor" => Idefics3Processor,
    "Qwen2VLProcessor" => Qwen2VLProcessor,
    "Qwen2_5_VLProcessor" => Qwen2VLProcessor,
    "Qwen3VLProcessor" => Qwen2VLProcessor,
    "PixtralProcessor" => PixtralProcessor,
    "Gemma3Processor" => Gemma3Processor,
    "LlavaProcessor" => LlavaProcessor
  }

  # The processor class of each model_type of these models in PROCESSOR_MAPPING_NAMES of
  # Python transformers.
  @model_types %{
    "idefics3" => "Idefics3Processor",
    "smolvlm" => "SmolVLMProcessor",
    "qwen2_vl" => "Qwen2VLProcessor",
    "qwen2_5_vl" => "Qwen2_5_VLProcessor",
    "qwen3_vl" => "Qwen3VLProcessor",
    "mistral3" => "PixtralProcessor",
    "pixtral" => "PixtralProcessor",
    "gemma3" => "Gemma3Processor",
    "llava" => "LlavaProcessor"
  }

  # As AutoProcessor of Python transformers: the processor_class of processor_config.json,
  # then of preprocessor_config.json, then of tokenizer_config.json, then of config.json, then
  # the class of the model_type. The image processor of preprocessor_config.json, such as a
  # fast one, is an input of the processor.
  defp processor!(repo, vision, image_text) do
    case processor_module!(repo) do
      Qwen2VLProcessor -> {Qwen2VLProcessor, Qwen2VLProcessor.load!(repo, inputs: vision.inputs)}
      LlavaProcessor -> {LlavaProcessor, LlavaProcessor.load!(repo, image_token: image_text)}
      module -> {module, module.load!(repo)}
    end
  end

  @doc false
  def processor_module!(repo) do
    class =
      Enum.find_value(["processor_config.json", "preprocessor_config.json"], fn file ->
        (Hub.json(repo.source, file, repo.opts) || %{})["processor_class"]
      end) || repo.tokenizer_config["processor_class"] || repo.config["processor_class"] ||
        @model_types[repo.config["model_type"]]

    @processors[class] ||
      (
        {last, names} = @processors |> Map.keys() |> Enum.sort() |> List.pop_at(-1)

        raise ArgumentError,
              "image_text_to_text supports the processors #{Enum.join(names, ", ")} and #{last}, got: #{inspect(class)}"
      )
  end

  # The chat template of the processor renders an image part without its image. As in the pipeline of
  # Python transformers, a chat that ends with an assistant message continues that message.
  defp prompt(repo, chat_template, %{messages: messages}, opts) do
    {messages, images} = Enum.map_reduce(messages, [], &take_images/2)
    final = List.last(messages)

    template_opts = [
      template: chat_template,
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

    %{prompt: prompt, tensors: tensors, key: key, texts: texts}
  end

  # As get_placeholder_mask of Python transformers: the image tokens of the prompt must be
  # those of the texts of the images, one for each feature. The text of a chat can also have
  # the image token, which would move the features of the images to other tokens, also in a
  # chat without images.
  defp check_image_tokens!(ids, texts, image_token, image_text) do
    count = Enum.count(ids, &(&1 == image_token))
    expected = Enum.sum_by(texts, &(length(String.split(&1, image_text)) - 1))

    if count != expected do
      raise ArgumentError,
            "the prompt has #{count} image tokens #{inspect(image_text)}, and the images give #{expected}. The text of the chat must not have the image token"
    end
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
  # The tensors of an image for a processor that gives all images one `size`, such as Gemma 3
  # and LLaVA: "pixel_values" {1, 3, height, width} from `featurizer` and "image_sizes" {1, 2}.
  def image_tensors(featurizer, {height, width}, image, backend) do
    pixels = Featurizer.process(featurizer, [image], backend: backend)
    sizes = Nx.tensor([[height, width]], type: :s64, backend: Nx.BinaryBackend)
    %{"pixel_values" => pixels, "image_sizes" => sizes}
  end

  @doc false
  # The inputs of the vision encoder for one image of `size`, on Nx.BinaryBackend.
  def image_template({height, width}) do
    zero = Nx.tensor(0.0, backend: Nx.BinaryBackend)
    %{"pixel_values" => Nx.broadcast(zero, {1, 3, height, width})}
  end

  @doc false
  # Runs `encode` on each image of `tensors` alone: the "pixel_values" {1, 3, height, width} of
  # each image of each row, as "image_sizes" shows, without the rows of zeros. So the encoder
  # has one compiled shape and the memory of one image. Each image must give `count` features.
  # Returns the features of all images in order, of shape {images * count, hidden}.
  def encode_each(encode, %{"pixel_values" => pixels, "image_sizes" => sizes}, count) do
    features =
      for row <- images(pixels, sizes, fn _size -> 1 end), {_size, image} <- row do
        %{"image_features" => features} = encode.(%{"pixel_values" => image})
        {1, tokens, hidden} = Nx.shape(features)

        if tokens != count do
          raise ArgumentError,
                "the vision encoder gives #{tokens} features for each image, and the processor gives #{count} image tokens"
        end

        Nx.reshape(features, {tokens, hidden})
      end

    Nx.concatenate(features)
  end

  @doc false
  # Runs `encode` in calls of at most `rows` rows and concatenates the outputs of the calls,
  # when the first axis of each input of the vision encoder is its batch axis ("batch_size"
  # in the model), so the encoder runs each row alone. An encoder whose first axis holds the
  # patches of an image, such as that of Qwen2-VL, runs in one call. Each call evaluates its
  # outputs: EMLX keeps them as expressions until a read, and then the device runs all calls
  # of a request as one piece of work, which the steps of another process wait for.
  def in_chunks(encode, vision, rows) do
    call = &(&1 |> encode.() |> Map.new(fn {name, t} -> {name, evaluate(t)} end))

    if Enum.all?(vision.inputs, &match?({_type, ["batch_size" | _]}, vision.input_types[&1])),
      do: &encode_in_chunks(call, &1, rows),
      else: call
  end

  defp encode_in_chunks(encode, inputs, rows) do
    size = inputs |> Map.values() |> hd() |> Nx.axis_size(0)

    outputs =
      for start <- 0..(size - 1)//rows do
        length = min(rows, size - start)

        inputs
        |> Map.new(fn {name, t} -> {name, Nx.slice_along_axis(t, start, length)} end)
        |> encode.()
      end

    for name <- Map.keys(hd(outputs)), into: %{} do
      {name, outputs |> Enum.map(& &1[name]) |> Nx.concatenate()}
    end
  end

  defp evaluate(tensor = %Nx.Tensor{data: %{__struct__: EMLX.Backend}}),
    do: Transmuter.EMLX.evaluate(tensor)

  defp evaluate(tensor), do: tensor

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
