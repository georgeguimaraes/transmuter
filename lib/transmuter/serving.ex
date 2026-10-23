defmodule Transmuter.Serving do
  @moduledoc """
  Builds the `Nx.Serving` of a task from a model, a tokenizer or a featurizer, and the functions of the task.

  `text/7` and `image/6` build the servings that run the model once. The client side turns the inputs into tensors and gives each batch a key: the sequence length for text, and the image size for images. The serving compiles the model once for each key in its init function. Init runs in the serving process, so the cache of compiled graph segments lives as long as the serving. Each batch is padded to the batch size, so the model only runs the compiled shapes.

  `each/2` builds the servings that run each input as a batch of its own, such as text-to-speech, whose outputs have a shape that depends on the input.

  `generation/4` builds the servings that generate tokens with `Transmuter.Generation`. The rows of a batch generate together in one loop. With `stream: true`, the loop sends each new token to the client through the `:token` hook of `Nx.Serving.streaming/2`, and the client decodes the text.
  """

  @behaviour Nx.Serving

  alias Transmuter.Audio.Featurizer, as: AudioFeaturizer
  alias Transmuter.{Featurizer, Generation, Hub, Repo, Shared, Tokenizer}
  alias Transmuter.Onnx.{AttributeProto, NodeProto}
  alias Transmuter.Ops.NN

  @options [:compile, :variant, :model, :backend, :compute_type | Hub.options()] ++
             [defn_options: []]

  # Nx.Serving runs each batch in a new process, whose heap starts small. A step of
  # generation allocates more than that, so the heap grows through many collections. A
  # heap of 2 MB (in 8-byte words) from the start removes them.
  @generation_heap_size 262_144

  # The masks of tokens of a step are binaries on the host, and each step makes new ones
  # as large as the vocabulary: 195 binaries of 256 KB in one request of 64 tokens with
  # Gemma 3. With the default binary heap, they start about two collections in each step. A
  # binary heap of 256 MB (in 8-byte words) from the start removes these collections.
  @generation_bin_vheap_size 33_554_432

  @doc """
  The options that all text tasks take. See `Transmuter.Text`.
  """
  def options, do: [:tokenizer | @options]

  @doc """
  The options that all image tasks take. See `Transmuter.Vision`.
  """
  def image_options, do: [:featurizer | @options]

  @doc """
  The options that all audio tasks with a `Transmuter.Audio.Featurizer` take. See `Transmuter.Audio`.
  """
  def audio_options, do: [:featurizer | @options]

  @doc """
  Loads the repository of `source`, its model and its tokenizer. See `load_model!/4`.
  """
  def load!(source, opts, variant, components \\ ["model"]) do
    {repo, model} = load_model!(source, opts, variant, components)
    {repo, model, opts[:tokenizer] || Tokenizer.load!(repo)}
  end

  @doc """
  Loads the repository of `source` and its model. The model is the first of `components` that the repository has, in the default variant `variant` of the task. The `:outputs` in `opts` go to `Transmuter.load!/2`, so the model gives only the outputs that the task reads.
  """
  def load_model!(source, opts, variant, components \\ ["model"]) do
    repo = Repo.load!(source, Keyword.take(opts, [:variant | Hub.options()]))
    component = Enum.find(components, hd(components), &(Repo.variants(repo, &1) != []))

    model =
      opts[:model] ||
        Repo.load_component!(
          repo,
          component,
          variant,
          Keyword.take(opts, [:backend, :compute_type, :outputs])
        )

    {repo, model}
  end

  @doc """
  Returns an `Nx.Serving` that runs `model` on text.

  `validate` takes one input and returns `{:ok, sequences}` or `{:error, message}`. The sequences are the strings or `{text, text_pair}` tuples to encode for the input. Each sequence gives one row of the batch, or one row for each window when the tokenizer has a `:stride` (see `Transmuter.Tokenizer.load!/2`).

  `fun` takes the outputs of the model and the inputs of a batch, and returns a map of tensors with the rows on the first axis. `postprocess` takes the rows of one input and returns the result for that input. Its arguments are the list of rows of the map from `fun`, and a list of `{sequence, encoding}` for these rows.

  The `:max_length` option of a task limits the default sequence lengths, as do the limits in `tokenizer_config.json` and `config.json`. The `:encode_options` of a task go to `Transmuter.Tokenizer.encode/3`.
  """
  def text(repo, model, tokenizer, opts, fun, postprocess, validate \\ &string/1) do
    compile =
      Keyword.validate!(opts[:compile] || [],
        batch_size: 1,
        sequence_length: sequence_lengths(repo, opts[:max_length])
      )

    batch_size = compile[:batch_size]
    lengths = List.wrap(compile[:sequence_length])
    tokenizer = Tokenizer.configure(tokenizer, length: lengths)

    templates =
      Map.new(lengths, fn length ->
        {{:sequence_length, length},
         Map.new(model.inputs, &{&1, zeros({batch_size, length}, :s64)})}
      end)

    preprocess = fn sequences ->
      {tensors, encodings} = Tokenizer.encode(tokenizer, sequences, opts[:encode_options] || [])
      key = {:sequence_length, Nx.axis_size(tensors["input_ids"], 1)}

      rows =
        Enum.zip_with(sequences, encodings, fn sequence, windows ->
          Enum.map(windows, &{sequence, &1})
        end)

      {[{tensors, key}], rows}
    end

    steps =
      Map.new(lengths, fn length ->
        {{:sequence_length, length}, Generation.prompt_inputs(model, batch_size, length)}
      end)

    shapes = %{batch_size: batch_size, templates: templates, inputs: %{}, steps: steps}
    new(model, opts, shapes, preprocess, fun, postprocess, validate)
  end

  # The compiled model of a serving whose image size depends on the image keeps this many
  # image sizes.
  @image_sizes 64

  # The serving compiles the size of a 640x480 image when it starts, a common size of photos
  # and of the COCO images. It compiles the other sizes on their first call.
  @nominal_size {480, 640}

  @doc """
  Returns an `Nx.Serving` that runs `model` on images.

  The featurizer turns the images into `"pixel_values"` on the `:backend` of `opts`. When the model has a `"pixel_mask"` input, each image also gets a mask of ones. The mask has the shape that the model declares, such as 64x64 in the DETR exports of Xenova, or else the size of the image. The images of a batch have one size, so no pixel is padding. `inputs` are other inputs of the model that are the same for all images, such as the text of the labels in zero-shot classification. Their first axis is not the batch.

  When the featurizer gives one image size, the serving compiles that shape, and the images go into batches of the `:batch_size` of `:compile`. When the ONNX file declares a fixed size of the batch axis of `pixel_values`, the batch size must be that size. When the size depends on the image, such as with the `shortest_edge` and `longest_edge` of DETR, the batch size must be 1. Each image is then a batch, and the compiled model compiles each new image size on its first call. It keeps up to #{@image_sizes} sizes.

  `fun` and `postprocess` are as in `text/7`. The tensors of `fun` must have the same shape for all image sizes. An image is one row, so `postprocess` takes a list of one row and a list of one image tensor.
  """
  def image(model, featurizer, opts, fun, postprocess, inputs \\ %{}) do
    compile = Keyword.validate!(opts[:compile] || [], batch_size: 1)
    size = Featurizer.size(featurizer)
    batch_size = image_batch_size!(size, compile[:batch_size])
    declared_batch_size!(model, batch_size)
    key = {:image_size, size}
    {height, width} = compile_size(featurizer)
    template = image_inputs(model, zeros({batch_size, 3, height, width}, :f32))
    backend = opts[:backend] || Nx.default_backend()

    preprocess = fn images ->
      groups =
        for group <- if(size, do: [images], else: Enum.map(images, &[&1])) do
          pixels =
            featurizer
            |> Featurizer.process(group, backend: backend)
            |> Nx.backend_transfer(Nx.BinaryBackend)

          {image_inputs(model, pixels), key}
        end

      {groups, Enum.map(images, &[&1])}
    end

    validate = fn input ->
      with {:ok, image} <- Featurizer.tensor(input), do: {:ok, [image]}
    end

    shapes = %{
      batch_size: batch_size,
      templates: %{key => template},
      inputs: inputs,
      cache_size: @image_sizes
    }

    new(model, opts, shapes, preprocess, fun, postprocess, validate)
  end

  @doc """
  Returns the `{height, width}` of the images that a serving of `featurizer` compiles when it starts: the size of the featurizer, or else the size that it gives for a 640x480 image.
  """
  def compile_size(featurizer),
    do: Featurizer.size(featurizer) || Featurizer.image_size(featurizer, @nominal_size)

  defp image_batch_size!(nil, batch_size) when batch_size != 1,
    do:
      raise(
        ArgumentError,
        "the featurizer gives images of different sizes, so each image is a batch of its own. " <>
          "Use batch_size: 1, or set a size with height and width in preprocessor_config.json"
      )

  defp image_batch_size!(_size, batch_size), do: batch_size

  # Some exports, such as onnx-community/ormbg-ONNX, have a fixed batch size in the shapes
  # of their graph. ONNX Runtime rejects other batch sizes too.
  defp declared_batch_size!(model, batch_size) do
    case model.input_types["pixel_values"] do
      {_type, [declared | _]} when is_integer(declared) and declared != batch_size ->
        raise ArgumentError,
              "the ONNX file has a fixed batch size of #{declared} on the first axis of pixel_values. Use batch_size: #{declared}"

      _type ->
        :ok
    end
  end

  @doc """
  Returns the image inputs of `model` for `pixels` of shape `{batch, channels, height, width}`: `"pixel_values"` and, when the model has a `"pixel_mask"` input, a mask of ones. The mask has the shape that the model declares, or else the size of the images.
  """
  def image_inputs(model, pixels) do
    if "pixel_mask" in model.inputs do
      {type, [_batch, mask_height, mask_width]} = model.input_types["pixel_mask"]
      {batch, _channels, height, width} = Nx.shape(pixels)
      one = Nx.tensor(1, type: type, backend: Nx.BinaryBackend)

      shape =
        {batch, Transmuter.dim_size(mask_height, height), Transmuter.dim_size(mask_width, width)}

      mask = Nx.broadcast(one, shape)
      %{"pixel_values" => pixels, "pixel_mask" => mask}
    else
      %{"pixel_values" => pixels}
    end
  end

  # The compiled model of a serving whose features depend on the length of the audio keeps
  # this many lengths. The serving compiles 1 s of audio when it starts.
  @audio_lengths 64

  @doc """
  Returns an `Nx.Serving` that runs `model` on audio.

  The featurizer (`Transmuter.Audio.Featurizer`) turns the audio into the input of `Transmuter.Audio.Featurizer.input_name/1`, such as `"input_values"`, on the `:backend` of `opts`. When the features of all audio have one shape, such as the 1024 frames of AST, the serving compiles that shape, and the audio goes into batches of the `:batch_size` of `:compile`. When the shape depends on the length of the audio, such as the samples of wav2vec2, the batch size must be 1. Each audio is then a batch, and the compiled model compiles each new length on its first call. It keeps up to #{@audio_lengths} lengths.

  The input is an audio of `Transmuter.Audio.Featurizer.audio/2`. The serving joins the chunks of a stream into one tensor.

  `fun` and `postprocess` are as in `text/7`. An audio is one row, so `postprocess` takes a list of one row and a list of one audio tensor. `inputs` are other inputs of the model that are the same for all audio, as in `image/6`.
  """
  def audio(model, featurizer, opts, fun, postprocess, inputs \\ %{}) do
    compile = Keyword.validate!(opts[:compile] || [], batch_size: 1)
    shape = AudioFeaturizer.shape(featurizer)
    batch_size = audio_batch_size!(shape, compile[:batch_size])
    key = {:audio_shape, shape}
    name = AudioFeaturizer.input_name(featurizer)
    template = zeros(Tuple.insert_at(shape || {featurizer.sampling_rate}, 0, batch_size), :f32)
    backend = opts[:backend] || Nx.default_backend()

    preprocess = fn audio ->
      groups =
        for group <- if(shape, do: [audio], else: Enum.map(audio, &[&1])) do
          values =
            featurizer
            |> AudioFeaturizer.process(group, backend: backend)
            |> Nx.backend_transfer(Nx.BinaryBackend)

          {%{name => values}, key}
        end

      {groups, Enum.map(audio, &[&1])}
    end

    # The featurizer takes the whole audio, so the chunks of a stream go into one tensor.
    validate = fn input ->
      with {:ok, chunks} <- AudioFeaturizer.audio(input, featurizer.sampling_rate) do
        samples = chunks |> Enum.map(&Nx.to_binary/1) |> IO.iodata_to_binary()
        {:ok, [Nx.from_binary(samples, :f32, backend: Nx.BinaryBackend)]}
      end
    end

    shapes = %{
      batch_size: batch_size,
      templates: %{key => %{name => template}},
      inputs: inputs,
      cache_size: @audio_lengths
    }

    new(model, opts, shapes, preprocess, fun, postprocess, validate)
  end

  defp audio_batch_size!(nil, batch_size) when batch_size != 1,
    do:
      raise(
        ArgumentError,
        "the shape of the features depends on the length of the audio, so each audio is a batch of its own. Use batch_size: 1"
      )

  defp audio_batch_size!(_shape, batch_size), do: batch_size

  @doc """
  Returns the `{height, width}` of the images of `featurizer`. Raises when the size depends on the image, because images of different sizes cannot go in one batch.
  """
  def image_size!(featurizer) do
    Featurizer.size(featurizer) ||
      raise ArgumentError,
            "the featurizer gives images of different sizes, so they cannot go in one batch. " <>
              "Set a size with height and width, or a crop size, in preprocessor_config.json"
  end

  @doc """
  Returns an `Nx.Serving` that runs `model` on the tensors of a task. `text/7`, `image/6` and `audio/6` build their servings with this function.

  `shapes` is a map:

    * `:batch_size` - the number of rows of a batch.

    * `:templates` - the input tensors of each batch key. The serving compiles the model for them.

    * `:inputs` - inputs of the model that are the same for all batches. Their first axis is not the batch.

    * `:cache_size` - optional, the `:cache_size` option of `Transmuter.compile/3`.

    * `:steps` - optional, the inputs and the static inputs of a step of a decoder for each batch key (see `Transmuter.Generation.prompt_inputs/3`).

    * `:prepare` - optional, a function that takes the inputs of a batch, padded to the batch size, and the number of rows that are not padding. It returns the inputs of the batch for the model, such as inputs whose shapes depend on the values of the batch. The fixed inputs and the inputs of the steps join them after this function. The serving compiles the model for the templates through this function too, as a full batch. The rows of the batch stay first. `fun` gets the outputs of the prepared batch, which can have one more row than the inputs. The serving gives the outputs of the rows that are not padding to the client.

    * `:streaming` - optional, `true` when the outputs of a batch can have other shapes than those of another batch, such as with `:prepare`. The serving then streams the outputs of each batch to the client, so that Nx.Serving does not join them.

  `validate` gives the sequences of each input. `preprocess` takes the sequences of all inputs and gives a list of `{tensors, batch key}`, one for each batch, and for each sequence the list of its rows. The rows of the batches are those of the sequences, in order. `fun` and `postprocess` are as in `text/7`.
  """
  def new(model, opts, shapes, preprocess, fun, postprocess, validate) do
    %{batch_size: batch_size, templates: templates} = shapes
    fixed = Map.take(shapes.inputs, model.inputs)
    compile_options = shapes |> Map.take([:cache_size]) |> Map.to_list()
    steps = Map.get(shapes, :steps, %{})
    step = &Map.get(steps, &1, {%{}, %{}})
    prepare = Map.get(shapes, :prepare, fn inputs, _size -> inputs end)

    given =
      for {_key, {inputs, static}} <- steps,
          name <- Map.keys(inputs) ++ Map.keys(static),
          uniq: true,
          do: name

    # The client does not give the inputs that :prepare adds.
    template = templates |> Map.values() |> hd()
    given = given ++ (Map.keys(prepare.(template, batch_size)) -- Map.keys(template))

    Nx.Serving.new(
      __MODULE__,
      fn defn_options ->
        {keys, defn_options} = Keyword.pop!(defn_options, :batch_keys)
        defn_options = compile_options ++ defn_options

        runs =
          Map.new(keys, fn key ->
            {inputs, static} = step.(key)
            model_inputs = &(&1 |> prepare.(&2) |> Map.merge(fixed) |> Map.merge(inputs))
            template = model_inputs.(templates[key], batch_size)
            run = Transmuter.compile(model, template, [static_inputs: static] ++ defn_options)
            {key, &run.(model_inputs.(&1, &2))}
          end)

        fn batch ->
          run = runs[batch.key]
          size = batch.size
          batch = Nx.Batch.pad(batch, batch_size - size)
          inputs = Nx.Defn.jit_apply(& &1, [batch], compiler: Nx.Defn.Evaluator)

          inputs
          |> run.(size)
          |> fun.(inputs)
          |> Nx.backend_transfer(Nx.BinaryBackend)
        end
      end,
      opts[:defn_options]
    )
    |> Nx.Serving.batch_size(batch_size)
    |> Nx.Serving.process_options(batch_keys: Map.keys(templates))
    |> then(&if(shapes[:streaming], do: Nx.Serving.streaming(&1), else: &1))
    |> Nx.Serving.client_preprocessing(fn input ->
      {inputs, multi?} = validate!(input, validate)
      {groups, rows} = preprocess.(Enum.concat(inputs))

      batches =
        for {tensors, key} <- groups do
          [model_inputs!(model, tensors, fixed, given)]
          |> Nx.Batch.concatenate()
          |> Nx.Batch.key(key)
        end

      # Nx.Serving splits one batch that is larger than the batch size, but not a list.
      batches = with [batch] <- batches, do: batch
      items = rows |> split_like(inputs) |> Enum.map(&Enum.concat/1)
      {batches, {multi?, items}}
    end)
    |> Nx.Serving.client_postprocessing(fn result, {multi?, items} ->
      results =
        result
        |> batch_outputs()
        |> Enum.flat_map(&output_rows/1)
        |> split_like(items)
        |> Enum.zip_with(items, postprocess)

      if multi?, do: results, else: hd(results)
    end)
  end

  # The outputs of the batches: one map of all rows, or a stream of the map of each batch.
  defp batch_outputs({outputs, _metadata}), do: [outputs]

  defp batch_outputs(stream),
    do: Stream.map(stream, fn {:batch, outputs, _metadata} -> outputs end)

  defp output_rows(outputs) do
    rows = outputs |> Map.values() |> hd() |> Nx.axis_size(0)
    for i <- 0..(rows - 1)//1, do: Map.new(outputs, fn {name, t} -> {name, t[i]} end)
  end

  @doc """
  Returns an `Nx.Serving` that runs each input as a batch of one row, for outputs whose shapes depend on the input, such as the audio of text-to-speech. The serving streams the outputs of each batch to the client, so the outputs of the inputs of a list can have different shapes.

  `task` is a map:

    * `:validate` takes one input and returns `{:ok, input}` or `{:error, message}`.

    * `:preprocess` takes a valid input and returns a map of `Nx.BinaryBackend` tensors without a batch axis.

    * `:init` takes the defn options and returns a function. The serving calls `:init` in its process, so it can compile models there. The function takes the tensors of one input, with a batch axis of size 1, and returns a map of tensors with a batch axis of size 1.

    * `:postprocess` takes the map of tensors of one input, without the batch axis, and returns the result.

  The serving reads `:defn_options` of `opts`.
  """
  def each(task, opts) do
    %{validate: validate, preprocess: preprocess, init: init, postprocess: postprocess} = task

    Nx.Serving.new(
      __MODULE__,
      fn defn_options ->
        run = init.(Keyword.delete(defn_options, :batch_keys))

        fn batch ->
          Nx.Defn.jit_apply(& &1, [batch], compiler: Nx.Defn.Evaluator)
          |> run.()
          |> Nx.backend_transfer(Nx.BinaryBackend)
        end
      end,
      opts[:defn_options]
    )
    |> Nx.Serving.batch_size(1)
    |> Nx.Serving.streaming()
    |> Nx.Serving.client_preprocessing(fn input ->
      {inputs, multi?} = validate!(input, validate)
      {Enum.map(inputs, &Nx.Batch.stack([preprocess.(&1)])), multi?}
    end)
    |> Nx.Serving.client_postprocessing(fn stream, multi? ->
      results =
        Enum.map(stream, fn {:batch, outputs, _metadata} ->
          outputs |> Map.new(fn {name, t} -> {name, t[0]} end) |> postprocess.()
        end)

      if multi?, do: results, else: hd(results)
    end)
  end

  @doc """
  Raises unless the `:batch_size` of the `:compile` option is 1, for a task of `each/2`. `reason` says why each input is a batch of its own.
  """
  def batch_size_one!(compile, reason) do
    if Keyword.validate!(compile || [], batch_size: 1)[:batch_size] != 1,
      do:
        raise(ArgumentError, "#{reason}, so each input is a batch of its own. Use batch_size: 1")
  end

  # Nx.Serving runs each batch of a serving process in a new process, which copies the
  # function of the batch and all that it holds, such as the compiled models. So a serving
  # process keeps the functions in a scope of Transmuter.Shared, and each batch reads them
  # without a copy. Nx.Serving.run/2 runs init in the caller, or for a streaming serving in a
  # process that it starts for the call, which keeps the functions only as long as it needs
  # them.
  @doc false
  @impl Nx.Serving
  def init(type, init, partitions) do
    funs = Enum.map(partitions, init)

    case type do
      :inline ->
        {:ok, fn -> funs end}

      :process ->
        scope = Shared.scope()
        Shared.put(scope, :batch, funs)
        {:ok, fn -> Shared.get(scope, :batch) end}
    end
  end

  @doc false
  @impl Nx.Serving
  def handle_batch(batch, partition, funs) do
    {:execute, fn -> {Enum.at(funs.(), partition).(batch), :server_info} end, funs}
  end

  defp zeros(shape, type),
    do: Nx.broadcast(Nx.tensor(0, type: type, backend: Nx.BinaryBackend), shape)

  # Splits list into consecutive parts with the lengths of the lists in groups.
  defp split_like(list, groups) do
    {parts, []} = Enum.map_reduce(groups, list, &Enum.split(&2, length(&1)))
    parts
  end

  @doc """
  Loads the repository of `source`, its encoder (`encoder_model`) and its merged decoder (`decoder_model_merged`). The default variant is fp32. The encoder gives only `last_hidden_state`, and the decoder gives only the outputs that `decoder_outputs` selects, by default those of `Transmuter.Generation.outputs/1`.
  """
  def load_encoder_decoder!(source, opts, decoder_outputs \\ &Generation.outputs/1) do
    repo = Repo.load!(source, Keyword.take(opts, [:variant | Hub.options()]))
    model_opts = Keyword.take(opts, [:backend, :compute_type])
    encoder_opts = [outputs: ["last_hidden_state"]] ++ model_opts
    encoder = Repo.load_component!(repo, "encoder_model", :fp32, encoder_opts)
    decoder_opts = [outputs: decoder_outputs] ++ model_opts
    decoder = Repo.load_component!(repo, "decoder_model_merged", :fp32, decoder_opts)
    {repo, encoder, decoder}
  end

  # The prompts of a batch have one width, so prompts of up to 4096 tokens go into one
  # batch. Longer prompts go into batches with prompts of the same power-of-two width.
  @prompt_widths for exponent <- 12..20, do: 2 ** exponent

  @doc """
  Returns an `Nx.Serving` that generates tokens with a context of `Transmuter.Generation`. The rows of a batch generate together. `tokenizer` decodes the text of a stream.

  `task` is a map:

    * `:validate` takes one input and returns `{:ok, input}` or `{:error, message}`.

    * `:preprocess` takes a valid input and returns an enumerable of runs, such as the windows of long audio. The client makes the rows and the batches of a list of runs, so an error in them raises in the caller. A stream of runs, such as the windows of an audio stream, goes into batches when Nx.Serving reads them, once, in a process of its own. An error in such a stream ends the call with an exit, not an exception that the caller can rescue. A run is a map with the token ids of the prompt of the decoder in `:prompt`, the seed of sampling in `:seed`, a map of `Nx.BinaryBackend` tensors without a batch axis for `:init` in `:tensors`, a batch key in `:key`, and `%{input: count, padding: count}` in `:summary` for the token summary of a stream. Runs with the same key have tensors of the same shapes. Each run is one row of a batch.

    * `:init` takes the defn options and returns a function. The serving calls `:init` in its process, so it can compile models there. The function takes the tensors of a batch, the prompts of its rows and a function `generate`. It calls `generate` with the other inputs of the decoder, such as the outputs of an encoder, and the prompts, one or more times. An empty prompt is a padding row. `generate` returns the tokens of each row, see `Transmuter.Generation.generate/5`. The function returns a list for each row, such as the result of `generate`. Its values are integers, f32 values such as log-probabilities, and lists of them. The serving sends the row as a flat list of integers, with each f32 value as its bits (see `Transmuter.Generation.from_bits/1`).

    * `:postprocess` takes the list of the integers of each run of an input, and the runs from `:preprocess` for this input, and returns the result.

    * `:keys` lists the keys of the runs. The default is `[nil]`.

  `opts` are the options of the task, after `generation_options!/1`. The serving reads `:batch_size` of `:compile`, `:stream`, `:stream_done` and `:defn_options`. The done event of a stream has the log-probabilities when the context has `token_logprobs: true`.
  """
  def generation(ctx, tokenizer, task, opts) do
    batch_size = Keyword.get(opts[:compile] || [], :batch_size, 1)
    keys = for key <- Map.get(task, :keys, [nil]), width <- @prompt_widths, do: {key, width}
    stream? = Keyword.get(opts, :stream, false)

    # The closures of the serving copy what they reference, so each one takes only its
    # function of the task.
    %{validate: validate, preprocess: preprocess, init: init, postprocess: postprocess} = task

    # One function runs the batches of all keys.
    Nx.Serving.new(
      __MODULE__,
      fn defn_options ->
        {hooks, defn_options} =
          defn_options |> Keyword.delete(:batch_keys) |> Keyword.pop(:hooks, %{})

        run = init.(defn_options)
        backend = Nx.Defn.to_backend(defn_options)
        ctx = Generation.compile(ctx, batch_size, defn_options)

        fn batch ->
          Process.flag(:min_heap_size, @generation_heap_size)
          Process.flag(:min_bin_vheap_size, @generation_bin_vheap_size)
          inputs = Nx.Defn.jit_apply(& &1, [batch], compiler: Nx.Defn.Evaluator)
          {rows, tensors} = Map.split(inputs, ["prompt", "length", "seed", "input"])

          # Padding rows have no prompt. Nx.Batch.pad pads on the host, which is slow for
          # large rows such as audio, so the tensors of the task get their padding rows on
          # the device.
          padding = batch_size - batch.size

          prompts =
            Enum.zip_with(Nx.to_list(rows["prompt"]), Nx.to_list(rows["length"]), &Enum.take/2)

          prompts = prompts ++ List.duplicate([], padding)
          seeds = Nx.to_list(rows["seed"]) ++ List.duplicate(0, padding)

          tensors =
            Map.new(tensors, fn {name, tensor} -> {name, pad_rows(tensor, padding, backend)} end)

          generate = &Generation.generate(ctx, &2, seeds, &1, hooks[:token])
          generated = tensors |> run.(prompts, generate) |> Enum.map(&integers/1)
          lengths = Enum.map(generated, &length/1)

          # A trailing zero, because Nx has no tensors of length 0.
          width = Enum.max(lengths) + 1
          ids = for ids <- generated, do: ids ++ List.duplicate(0, width - length(ids))

          %{
            ids: Nx.tensor(ids, type: :s64, backend: Nx.BinaryBackend),
            length: Nx.tensor(lengths, backend: Nx.BinaryBackend),
            input: pad_rows(rows["input"], padding, Nx.BinaryBackend)
          }
        end
      end,
      opts[:defn_options]
    )
    |> Nx.Serving.batch_size(batch_size)
    |> Nx.Serving.process_options(batch_keys: keys)
    |> Nx.Serving.streaming(hooks: if(stream?, do: [:token], else: []))
    |> Nx.Serving.client_preprocessing(fn input ->
      {inputs, multi?} = validate!(input, validate)

      if stream? and multi? do
        raise ArgumentError, "a serving with stream: true takes one input, got a list"
      end

      runs = Enum.map(inputs, preprocess)
      rows = for {runs, input} <- Enum.with_index(runs), do: rows(runs, input)

      # A batch ends when it is full or when the key changes. Lists of runs give their
      # batches here, so an error in a row raises in the caller. A stream of runs gives its
      # batches when Nx.Serving reads them, in a process of its own.
      batches =
        rows
        |> Stream.concat()
        |> Stream.chunk_while([], &batch_rows(&1, &2, batch_size), &last_batch_rows/1)
        |> Stream.map(fn rows = [{key, _row} | _] ->
          rows |> Enum.map(&elem(&1, 1)) |> Nx.Batch.stack() |> Nx.Batch.key(key)
        end)

      batches = if Enum.all?(rows, &is_list/1), do: Enum.to_list(batches), else: batches
      {if(stream?, do: one_batch!(batches), else: batches), {multi?, runs}}
    end)
    |> generation_postprocessing(
      tokenizer,
      postprocess,
      stream?,
      {opts[:stream_done], ctx.token_logprobs?}
    )
  end

  @doc """
  Validates the options of a generation task that `Keyword.validate!/2` does not check, before the task loads its models, and returns them. `:token_logprobs` is `true` or `false`. A stream gives the log-probabilities in its done event, so `token_logprobs: true` with `stream: true` needs `stream_done: true`.
  """
  def generation_options!(opts) do
    logprobs? = Keyword.get(opts, :token_logprobs, false)

    unless is_boolean(logprobs?) do
      raise ArgumentError,
            "expected :token_logprobs to be true or false, got: #{inspect(logprobs?)}"
    end

    if logprobs? and opts[:stream] == true and opts[:stream_done] != true do
      raise ArgumentError,
            "a stream gives the token log-probabilities in its done event, so token_logprobs: true with stream: true needs stream_done: true"
    end

    opts
  end

  @doc """
  Puts `logprobs`, the log-probability of each token of `result`, in `result` under `:token_logprobs` when `logprobs?` is `true`, the `:token_logprobs?` of the context.
  """
  def token_logprobs(result, logprobs, logprobs?),
    do: if(logprobs?, do: Map.put(result, :token_logprobs, logprobs), else: result)

  @doc """
  Returns the token ids and their log-probabilities of the integers of a row whose values are the tokens of `Transmuter.Generation.generate/5`. Without log-probabilities, the integers are the ids, and the list of log-probabilities is empty.
  """
  def tokens(integers, false), do: {integers, []}

  def tokens(integers, true) do
    integers
    |> Enum.chunk_every(2)
    |> Enum.map(fn [id, bits] -> {id, Generation.from_bits(bits)} end)
    |> Enum.unzip()
  end

  @doc """
  Returns the `:postprocess` of a task whose run generates text: `%{results: [%{text: text, token_summary: summary}]}`, with `:token_logprobs` when `logprobs?` is `true`, the `:token_logprobs?` of the context. The token summary of the run gets the number of new tokens.
  """
  def text_postprocess(tokenizer, logprobs?) do
    fn [integers], [run] ->
      {ids, logprobs} = tokens(integers, logprobs?)
      summary = Map.put(run.summary, :output, length(ids))
      result = %{text: Tokenizer.decode(tokenizer, ids), token_summary: summary}
      %{results: [token_logprobs(result, logprobs, logprobs?)]}
    end
  end

  # The flat integers of a row of :init, with each f32 value as its bits.
  defp integers(row), do: row |> List.flatten() |> Enum.map(&integer/1)

  defp integer(value) when is_integer(value), do: value
  defp integer(value), do: Generation.to_bits(value)

  defp generation_postprocessing(serving, tokenizer, postprocess, stream?, flags) do
    if stream? do
      Nx.Serving.client_postprocessing(serving, fn stream, {false, [[run]]} ->
        stream_text(stream, tokenizer, run.summary, flags)
      end)
    else
      Nx.Serving.client_postprocessing(serving, fn stream, {multi?, runs} ->
        rows = rows_by_input(stream)

        results =
          for {runs, input} <- Enum.with_index(runs),
              do: postprocess.(Map.get(rows, input, []), runs)

        if multi?, do: results, else: hd(results)
      end)
    end
  end

  # The integers of the rows of each input, from the index of its input that each row gives
  # back.
  defp rows_by_input(stream) do
    stream
    |> Enum.flat_map(fn {:batch, %{ids: ids, length: length, input: input}, _metadata} ->
      Enum.zip_with([Nx.to_list(input), Nx.to_list(ids), Nx.to_list(length)], fn
        [input, ids, length] -> {input, Enum.take(ids, length)}
      end)
    end)
    |> Enum.group_by(&elem(&1, 0), &elem(&1, 1))
  end

  # The rule of Bumblebee and of TextStreamer in Python transformers. The text of the
  # pending tokens comes out up to its last space, because a byte-level token can hold a
  # part of a character, and a tokenizer can add or remove the space before a word from
  # the tokens around it. A newline or a CJK character ends a chunk at once, and the
  # pending tokens start again after a newline.
  defp stream_text(stream, tokenizer, summary, {done?, logprobs?}) do
    state = %{ids: [], logprobs: [], count: 0, sent: 0, finished?: false}
    logprob = if logprobs?, do: &[Nx.to_number(&1.logprob[0])], else: fn _event -> [] end

    Stream.transform(stream, state, fn
      _event, state = %{finished?: true} ->
        {:halt, state}

      {:batch, _outputs, _metadata}, state ->
        {:halt, state}

      {:token, event = %{token: token, finished?: finished?}}, state ->
        token = Nx.to_number(token[0])
        finished? = Nx.to_number(finished?[0]) == 1

        state =
          if token < 0 do
            state
          else
            logprobs = logprob.(event) ++ state.logprobs
            %{state | ids: state.ids ++ [token], logprobs: logprobs, count: state.count + 1}
          end

        text = tokenizer |> Tokenizer.decode(state.ids) |> binary_slice(state.sent..-1//1)
        {chunk, state} = chunk(text, finished?, state)

        done =
          if finished? and done?, do: [{:done, done(summary, state, logprobs?)}], else: []

        {Enum.reject([chunk], &(&1 == "")) ++ done, %{state | finished?: finished?}}
    end)
  end

  defp done(summary, state, logprobs?) do
    result = %{token_summary: Map.put(summary, :output, state.count)}
    token_logprobs(result, Enum.reverse(state.logprobs), logprobs?)
  end

  defp chunk(text, finished?, state) do
    cond do
      finished? or String.ends_with?(text, "\n") ->
        {text, %{state | ids: [], sent: 0}}

      text != "" and cjk?(text |> String.last() |> String.to_charlist() |> List.last()) ->
        {text, %{state | sent: state.sent + byte_size(text)}}

      true ->
        case :binary.matches(text, " ") do
          [_ | _] = matches ->
            {space, _} = List.last(matches)
            {binary_part(text, 0, space), %{state | sent: state.sent + space}}

          [] ->
            {"", state}
        end
    end
  end

  # The CJK ranges of BERT. CJK text has no spaces, so a chunk ends after each whole CJK
  # character. A part of a character decodes to U+FFFD, which is not in these ranges.
  defp cjk?(codepoint) do
    codepoint in 0x4E00..0x9FFF or codepoint in 0x3400..0x4DBF or
      codepoint in 0x20000..0x2A6DF or codepoint in 0x2A700..0x2B73F or
      codepoint in 0x2B740..0x2B81F or codepoint in 0x2B820..0x2CEAF or
      codepoint in 0xF900..0xFAFF or codepoint in 0x2F800..0x2FA1F
  end

  defp pad_rows(tensor, padding, backend) do
    [_batch | axes] = List.duplicate({0, 0, 0}, Nx.rank(tensor))
    tensor |> Nx.backend_transfer(backend) |> Nx.pad(0, [{0, padding, 0} | axes])
  end

  defp rows(runs, input) when is_list(runs), do: Enum.map(runs, &generation_row(&1, input))
  defp rows(runs, input), do: Stream.map(runs, &generation_row(&1, input))

  # Streaming hooks take one batch, not a stream of batches.
  defp one_batch!([batch]), do: batch

  defp one_batch!(_batches),
    do: raise(ArgumentError, "a serving with stream: true takes an input of one batch")

  defp batch_rows(row = {key, _row}, rows, batch_size) do
    case rows do
      [{^key, _row} | _] when length(rows) < batch_size -> {:cont, [row | rows]}
      [_ | _] -> {:cont, Enum.reverse(rows), [row]}
      [] -> {:cont, [row]}
    end
  end

  defp last_batch_rows([]), do: {:cont, []}
  defp last_batch_rows(rows), do: {:cont, Enum.reverse(rows), []}

  defp generation_row(run, input) do
    width =
      Enum.find(@prompt_widths, &(&1 >= length(run.prompt))) ||
        raise ArgumentError,
              "the prompt has #{length(run.prompt)} tokens, and generation takes up to #{List.last(@prompt_widths)}"

    prompt = run.prompt ++ List.duplicate(0, width - length(run.prompt))

    row =
      Map.merge(run.tensors, %{
        "prompt" => Nx.tensor(prompt, type: :s64, backend: Nx.BinaryBackend),
        "length" => Nx.tensor(length(run.prompt), type: :s64, backend: Nx.BinaryBackend),
        "seed" => Nx.tensor(run.seed, type: :s64, backend: Nx.BinaryBackend),
        "input" => Nx.tensor(input, type: :s64, backend: Nx.BinaryBackend)
      })

    {{Map.get(run, :key), width}, row}
  end

  @doc """
  The default sequence lengths of a text task: powers of 4 from 32, up to the smallest of `max_length` and the limits in `tokenizer_config.json` and `config.json`.
  """
  def sequence_lengths(repo, max_length) do
    max =
      [
        max_length,
        repo.tokenizer_config["model_max_length"],
        repo.config["max_position_embeddings"]
      ]
      |> Enum.filter(&(is_integer(&1) and &1 <= 1_000_000))
      |> Enum.min(fn -> 512 end)

    Enum.uniq(Enum.map([32, 128, 512], &min(&1, max)))
  end

  # The inputs in `given` come from the step of the batch key, and they replace those of the
  # batch, such as the attention mask of a decoder. The task reads the tensors of the batch.
  defp model_inputs!(model, tensors, fixed, given) do
    case model.inputs -- (Map.keys(tensors) ++ Map.keys(fixed) ++ given) do
      [] ->
        Map.take(tensors, model.inputs -- Map.keys(fixed))

      missing ->
        raise ArgumentError,
              "the model needs the inputs #{inspect(missing)}, which the task does not give"
    end
  end

  defp validate!(inputs, validate) when is_list(inputs) do
    inputs =
      for input <- inputs do
        case validate.(input) do
          {:ok, input} -> input
          {:error, message} -> raise ArgumentError, "invalid input in the batch, #{message}"
        end
      end

    {inputs, true}
  end

  defp validate!(input, validate) do
    case validate.(input) do
      {:ok, input} -> {[input], false}
      {:error, message} -> raise ArgumentError, "invalid input, #{message}"
    end
  end

  @doc """
  Accepts a string as one sequence.
  """
  def string(input) when is_binary(input), do: {:ok, [input]}
  def string(input), do: {:error, "expected a string, got: #{inspect(input)}"}

  @doc """
  Returns the label of class `index` from the `id2label` map of `config.json`, or `"LABEL_<index>"`.
  """
  def label(repo, index),
    do: get_in(repo.config, ["id2label", Integer.to_string(index)]) || "LABEL_#{index}"

  @doc """
  Returns `%{predictions: [%{label: label, score: score}]}` from the scores of one input, from the highest score. `label` takes the index of a class and returns its label. `top_k` is the number of predictions, or `nil` for all.
  """
  def predictions(scores, label, top_k) do
    predictions =
      scores
      |> Nx.to_list()
      |> Enum.with_index(&%{label: label.(&2), score: &1})
      |> Enum.sort_by(& &1.score, :desc)

    %{predictions: if(top_k, do: Enum.take(predictions, top_k), else: predictions)}
  end

  @doc """
  Returns the default scores function of a classification model, as the classification pipelines of Python transformers: `:none` for regression, `:sigmoid` for one label or multi-label classification, and else `:softmax`.
  """
  def scores_function(repo) do
    cond do
      repo.config["problem_type"] == "regression" -> :none
      map_size(repo.config["id2label"] || %{}) == 1 -> :sigmoid
      repo.config["problem_type"] == "multi_label_classification" -> :sigmoid
      true -> :softmax
    end
  end

  @doc """
  Applies a scores function (`:softmax`, `:sigmoid` or `:none`) to logits.
  """
  def scores(logits, :softmax), do: softmax(logits)
  def scores(logits, :sigmoid), do: Nx.sigmoid(logits)
  def scores(logits, :none), do: logits

  @doc """
  The softmax over the last axis.
  """
  def softmax(logits) do
    exp = Nx.exp(Nx.subtract(logits, Nx.reduce_max(logits, axes: [-1], keep_axes: true)))
    Nx.divide(exp, Nx.sum(exp, axes: [-1], keep_axes: true))
  end

  @interpolations %{bilinear: "linear", bicubic: "cubic"}

  @doc """
  Resizes the last two axes of `tensor` to `{height, width}` as `torch.nn.functional.interpolate` with `align_corners=False` and without antialias, which the post-processing of Python transformers uses. `mode` is `:bilinear` or `:bicubic`, whose cubic coefficient is -0.75 as in PyTorch.

  The resize is the ONNX `Resize` operator with the `pytorch_half_pixel` coordinates. `Transmuter.Featurizer.resize/3` gives the pixels of Pillow, whose filters widen when the image gets smaller. This filter does not widen.
  """
  def interpolate(tensor, {height, width}, mode) do
    node = %NodeProto{
      op_type: "Resize",
      attribute: [
        %AttributeProto{name: "mode", type: :STRING, s: Map.fetch!(@interpolations, mode)},
        %AttributeProto{
          name: "coordinate_transformation_mode",
          type: :STRING,
          s: "pytorch_half_pixel"
        },
        %AttributeProto{name: "cubic_coeff_a", type: :FLOAT, f: -0.75},
        %AttributeProto{name: "axes", type: :INTS, ints: [-2, -1]}
      ]
    }

    sizes = Nx.tensor([height, width], type: :s64, backend: Nx.BinaryBackend)
    [resized] = NN.apply(node, [tensor, {:empty, {0}}, {:empty, {0}}, sizes])
    resized
  end

  @doc """
  Divides each vector on the last axis by its L2 norm.
  """
  def l2_norm(tensor) do
    norm = tensor |> Nx.pow(2) |> Nx.sum(axes: [-1], keep_axes: true) |> Nx.sqrt()
    Nx.divide(tensor, Nx.max(norm, 1.0e-12))
  end
end
