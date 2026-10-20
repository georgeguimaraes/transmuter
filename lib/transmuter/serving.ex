defmodule Transmuter.Serving do
  @moduledoc """
  Builds the `Nx.Serving` of a task from a model, a tokenizer or a featurizer, and the functions of the task.

  `text/7` and `image/6` build the servings that run the model once. The client side turns the inputs into tensors and gives each batch a key: the sequence length for text, and the image size for images. The serving compiles the model once for each key in its init function. Init runs in the serving process, so the cache of compiled graph segments lives as long as the serving. Each batch is padded to the batch size, so the model only runs the compiled shapes.

  `generation/3` builds the servings that generate tokens with `Transmuter.Generation`. The rows of a batch generate together in one loop.
  """

  alias Transmuter.{Featurizer, Generation, Repo, Tokenizer}

  @options [:compile, :variant, :cache_dir, :model, :backend, :compute_type, defn_options: []]

  # Nx.Serving runs each batch in a new process, whose heap starts small. A step of
  # generation allocates more than that, so the heap grows through many collections. A
  # heap of 2 MB (in 8-byte words) from the start removes them.
  @generation_heap_size 262_144

  # A step reads its compiled segments from the cache table of Transmuter.GraphBreaks, and
  # each read references the binary of the graph again. The references count in the binary
  # heap of the process, but they use no new memory. With a small binary heap, the collector
  # moves these references between the young and the old heap, and it sweeps the whole heap
  # again and again. A binary heap of 256 MB (in 8-byte words) from the start removes these
  # sweeps.
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
  Loads the repository of `source`, its model and its tokenizer. See `load_model!/4`.
  """
  def load!(source, opts, variant, components \\ ["model"]) do
    {repo, model} = load_model!(source, opts, variant, components)
    {repo, model, opts[:tokenizer] || Tokenizer.load!(repo)}
  end

  @doc """
  Loads the repository of `source` and its model. The model is the first of `components` that the repository has, in the default variant `variant` of the task.
  """
  def load_model!(source, opts, variant, components \\ ["model"]) do
    repo = Repo.load!(source, Keyword.take(opts, [:variant, :cache_dir]))
    component = Enum.find(components, hd(components), &(Repo.variants(repo, &1) != []))

    model =
      opts[:model] ||
        Repo.load_component!(
          repo,
          component,
          variant,
          Keyword.take(opts, [:backend, :compute_type])
        )

    {repo, model}
  end

  @doc """
  Returns an `Nx.Serving` that runs `model` on text.

  `validate` takes one input and returns `{:ok, sequences}` or `{:error, message}`. The sequences are the strings or `{text, text_pair}` tuples to encode for the input. Each sequence gives one row of the batch, or one row for each window when the tokenizer has a `:stride` (see `Transmuter.Tokenizer.load!/2`).

  `fun` takes the outputs of the model and the inputs of a batch, and returns a map of tensors with the rows on the first axis. `postprocess` takes the rows of one input and returns the result for that input. Its arguments are the list of rows of the map from `fun`, and a list of `{sequence, encoding}` for these rows.

  The `:max_length` option of a task limits the default sequence lengths, as do the limits in `tokenizer_config.json` and `config.json`.
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
      {tensors, encodings} = Tokenizer.encode(tokenizer, sequences)
      key = {:sequence_length, Nx.axis_size(tensors["input_ids"], 1)}

      rows =
        Enum.zip_with(sequences, encodings, fn sequence, windows ->
          Enum.map(windows, &{sequence, &1})
        end)

      {tensors, key, rows}
    end

    shapes = %{batch_size: batch_size, templates: templates, inputs: %{}}
    new(model, opts, shapes, preprocess, fun, postprocess, validate)
  end

  @doc """
  Returns an `Nx.Serving` that runs `model` on images.

  The featurizer turns the images into `"pixel_values"` on the `:backend` of `opts`. The featurizer must give one image size, so the serving compiles one shape. `inputs` are other inputs of the model that are the same for all images, such as the text of the labels in zero-shot classification. Their first axis is not the batch.

  `fun` and `postprocess` are as in `text/7`. An image is one row, so `postprocess` takes a list of one row and a list of one image tensor.
  """
  def image(model, featurizer, opts, fun, postprocess, inputs \\ %{}) do
    compile = Keyword.validate!(opts[:compile] || [], batch_size: 1)
    batch_size = compile[:batch_size]

    {height, width} =
      Featurizer.size(featurizer) ||
        raise ArgumentError,
              "the featurizer gives images of different sizes, so they cannot go in one batch. " <>
                "Set a size with height and width, or a crop size, in preprocessor_config.json"

    key = {:image_size, {height, width}}
    templates = %{key => %{"pixel_values" => zeros({batch_size, 3, height, width}, :f32)}}
    backend = opts[:backend] || Nx.default_backend()

    preprocess = fn images ->
      pixels =
        featurizer
        |> Featurizer.process(images, backend: backend)
        |> Nx.backend_transfer(Nx.BinaryBackend)

      {%{"pixel_values" => pixels}, key, Enum.map(images, &[&1])}
    end

    validate = fn input ->
      with {:ok, image} <- Featurizer.tensor(input), do: {:ok, [image]}
    end

    shapes = %{batch_size: batch_size, templates: templates, inputs: inputs}
    new(model, opts, shapes, preprocess, fun, postprocess, validate)
  end

  # `shapes` has the batch size, the input templates of each batch key, and the inputs
  # that are the same for all batches. `validate` gives the sequences of each input.
  # `preprocess` takes the sequences of all inputs and gives the tensors, the batch key, and
  # for each sequence the list of its rows.
  defp new(model, opts, shapes, preprocess, fun, postprocess, validate) do
    %{batch_size: batch_size, templates: templates} = shapes
    fixed = Map.take(shapes.inputs, model.inputs)

    Nx.Serving.new(
      fn key, defn_options ->
        run = Transmuter.compile(model, Map.merge(fixed, templates[key]), defn_options)

        fn batch ->
          batch = Nx.Batch.pad(batch, batch_size - batch.size)
          inputs = Nx.Defn.jit_apply(& &1, [batch], compiler: Nx.Defn.Evaluator)

          fixed
          |> Map.merge(inputs)
          |> run.()
          |> fun.(inputs)
          |> Nx.backend_transfer(Nx.BinaryBackend)
        end
      end,
      opts[:defn_options]
    )
    |> Nx.Serving.batch_size(batch_size)
    |> Nx.Serving.process_options(batch_keys: Map.keys(templates))
    |> Nx.Serving.client_preprocessing(fn input ->
      {inputs, multi?} = validate!(input, validate)
      {tensors, key, rows} = preprocess.(Enum.concat(inputs))
      tensors = model_inputs!(model, tensors, fixed)
      batch = [tensors] |> Nx.Batch.concatenate() |> Nx.Batch.key(key)
      items = rows |> split_like(inputs) |> Enum.map(&Enum.concat/1)
      {batch, {multi?, items}}
    end)
    |> Nx.Serving.client_postprocessing(fn {outputs, _metadata}, {multi?, items} ->
      results =
        for(
          i <- 0..(Enum.sum_by(items, &length/1) - 1)//1,
          do: Map.new(outputs, fn {name, t} -> {name, t[i]} end)
        )
        |> split_like(items)
        |> Enum.zip_with(items, postprocess)

      if multi?, do: results, else: hd(results)
    end)
  end

  defp zeros(shape, type),
    do: Nx.broadcast(Nx.tensor(0, type: type, backend: Nx.BinaryBackend), shape)

  # Splits list into consecutive parts with the lengths of the lists in groups.
  defp split_like(list, groups) do
    {parts, []} = Enum.map_reduce(groups, list, &Enum.split(&2, length(&1)))
    parts
  end

  @doc """
  Loads the repository of `source`, its encoder (`encoder_model`) and its merged decoder (`decoder_model_merged`). The default variant is fp32.
  """
  def load_encoder_decoder!(source, opts) do
    repo = Repo.load!(source, Keyword.take(opts, [:variant, :cache_dir]))
    model_opts = Keyword.take(opts, [:backend, :compute_type])
    encoder = Repo.load_component!(repo, "encoder_model", :fp32, model_opts)
    decoder = Repo.load_component!(repo, "decoder_model_merged", :fp32, model_opts)
    {repo, encoder, decoder}
  end

  # The prompts of a batch have one width, so prompts of up to 4096 tokens go into one
  # batch. Longer prompts go into batches with prompts of the same power-of-two width.
  @prompt_widths for exponent <- 12..20, do: 2 ** exponent

  @doc """
  Returns an `Nx.Serving` that generates tokens with a context of `Transmuter.Generation`. The rows of a batch generate together.

  `task` is a map:

    * `:validate` takes one input and returns `{:ok, input}` or `{:error, message}`.

    * `:preprocess` takes a valid input and returns a list of runs, such as the windows of long audio. A run is a map with the token ids of the prompt of the decoder in `:prompt`, the seed of sampling in `:seed`, a map of `Nx.BinaryBackend` tensors without a batch axis for `:init` in `:tensors`, and a batch key in `:key`. Runs with the same key have tensors of the same shapes. Each run is one row of a batch.

    * `:init` takes the defn options and returns a function. The serving calls `:init` in its process, so it can compile models there. The function takes the tensors of a batch, the prompts of its rows and a function `generate`. It calls `generate` with the other inputs of the decoder, such as the outputs of an encoder, and the prompts, one or more times. An empty prompt is a padding row. The function returns the token ids of each row.

    * `:postprocess` takes the list of the token ids of each run and the list of runs from `:preprocess`, and returns the result.

    * `:keys` lists the keys of the runs. The default is `[nil]`.

  `opts` are the options of the task. The serving reads `:batch_size` of `:compile` and `:defn_options`.
  """
  def generation(ctx, task, opts) do
    batch_size = Keyword.get(opts[:compile] || [], :batch_size, 1)
    keys = for key <- Map.get(task, :keys, [nil]), width <- @prompt_widths, do: {key, width}
    # The closures of the serving copy what they reference, so each one takes only its
    # function of the task.
    %{validate: validate, preprocess: preprocess, init: init, postprocess: postprocess} = task

    # One function runs the batches of all keys.
    Nx.Serving.new(
      fn defn_options ->
        defn_options = Keyword.delete(defn_options, :batch_keys)
        run = init.(defn_options)
        backend = Nx.Defn.to_backend(defn_options)
        ctx = Generation.compile(ctx, batch_size, defn_options)

        fn batch ->
          Process.flag(:min_heap_size, @generation_heap_size)
          Process.flag(:min_bin_vheap_size, @generation_bin_vheap_size)
          inputs = Nx.Defn.jit_apply(& &1, [batch], compiler: Nx.Defn.Evaluator)
          {rows, tensors} = Map.split(inputs, ["prompt", "length", "seed"])

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

          generated = run.(tensors, prompts, &Generation.generate(ctx, &2, seeds, &1))
          lengths = Enum.map(generated, &length/1)

          # A trailing zero, because Nx has no tensors of length 0.
          width = Enum.max(lengths) + 1
          ids = for ids <- generated, do: ids ++ List.duplicate(0, width - length(ids))

          %{
            ids: Nx.tensor(ids, type: :s64, backend: Nx.BinaryBackend),
            length: Nx.tensor(lengths, backend: Nx.BinaryBackend)
          }
        end
      end,
      opts[:defn_options]
    )
    |> Nx.Serving.batch_size(batch_size)
    |> Nx.Serving.process_options(batch_keys: keys)
    |> Nx.Serving.streaming()
    |> Nx.Serving.client_preprocessing(fn input ->
      {inputs, multi?} = validate!(input, validate)

      runs = Enum.map(inputs, preprocess)

      batches =
        runs
        |> Enum.concat()
        |> Enum.map(&generation_row/1)
        |> Enum.chunk_by(&elem(&1, 0))
        |> Enum.flat_map(&Enum.chunk_every(&1, batch_size))
        |> Enum.map(fn rows = [{key, _row} | _] ->
          rows |> Enum.map(&elem(&1, 1)) |> Nx.Batch.stack() |> Nx.Batch.key(key)
        end)

      {batches, {multi?, runs}}
    end)
    |> Nx.Serving.client_postprocessing(fn stream, {multi?, runs} ->
      results =
        stream
        |> Enum.flat_map(fn {:batch, %{ids: ids, length: length}, _metadata} ->
          Enum.zip_with(Nx.to_list(ids), Nx.to_list(length), &Enum.take/2)
        end)
        |> split_like(runs)
        |> Enum.zip_with(runs, postprocess)

      if multi?, do: results, else: hd(results)
    end)
  end

  defp pad_rows(tensor, padding, backend) do
    [_batch | axes] = List.duplicate({0, 0, 0}, Nx.rank(tensor))
    tensor |> Nx.backend_transfer(backend) |> Nx.pad(0, [{0, padding, 0} | axes])
  end

  defp generation_row(run) do
    width =
      Enum.find(@prompt_widths, &(&1 >= length(run.prompt))) ||
        raise ArgumentError,
              "the prompt has #{length(run.prompt)} tokens, and generation takes up to #{List.last(@prompt_widths)}"

    prompt = run.prompt ++ List.duplicate(0, width - length(run.prompt))

    row =
      Map.merge(run.tensors, %{
        "prompt" => Nx.tensor(prompt, type: :s64, backend: Nx.BinaryBackend),
        "length" => Nx.tensor(length(run.prompt), type: :s64, backend: Nx.BinaryBackend),
        "seed" => Nx.tensor(run.seed, type: :s64, backend: Nx.BinaryBackend)
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

  defp model_inputs!(model, tensors, fixed) do
    case model.inputs -- Map.keys(Map.merge(tensors, fixed)) do
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

  @doc """
  Divides each vector on the last axis by its L2 norm.
  """
  def l2_norm(tensor) do
    norm = tensor |> Nx.pow(2) |> Nx.sum(axes: [-1], keep_axes: true) |> Nx.sqrt()
    Nx.divide(tensor, Nx.max(norm, 1.0e-12))
  end
end
