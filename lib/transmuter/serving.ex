defmodule Transmuter.Serving do
  @moduledoc """
  Builds the `Nx.Serving` of a text task from a model, a tokenizer and the functions of the task.

  The client side tokenizes the inputs and gives each batch the key of its sequence length. The serving compiles the model once for each key in its init function. Init runs in the serving process, so the cache of compiled graph segments lives as long as the serving. Each batch is padded to the batch size, so the model only runs the compiled shapes.
  """

  alias Transmuter.{Repo, Tokenizer}

  @doc """
  The options that all text tasks take. See `Transmuter.Text`.
  """
  def options,
    do: [
      :compile,
      :variant,
      :cache_dir,
      :model,
      :tokenizer,
      :backend,
      :compute_type,
      defn_options: []
    ]

  @doc """
  Loads the repository of `source`, its model and its tokenizer. The model is the first of `components` that the repository has, in the default variant `variant` of the task.
  """
  def load!(source, opts, variant, components \\ ["model"]) do
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

    {repo, model, opts[:tokenizer] || Tokenizer.load!(repo)}
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
        sequence_length: lengths(repo, opts[:max_length])
      )

    batch_size = compile[:batch_size]
    lengths = List.wrap(compile[:sequence_length])
    tokenizer = Tokenizer.configure(tokenizer, length: lengths)

    Nx.Serving.new(
      fn {:sequence_length, length}, defn_options ->
        zeros =
          Nx.broadcast(Nx.tensor(0, type: :s64, backend: Nx.BinaryBackend), {batch_size, length})

        run = Transmuter.compile(model, Map.new(model.inputs, &{&1, zeros}), defn_options)

        fn batch ->
          batch = Nx.Batch.pad(batch, batch_size - batch.size)
          inputs = Nx.Defn.jit_apply(& &1, [batch], compiler: Nx.Defn.Evaluator)
          inputs |> run.() |> fun.(inputs) |> Nx.backend_transfer(Nx.BinaryBackend)
        end
      end,
      opts[:defn_options]
    )
    |> Nx.Serving.batch_size(batch_size)
    |> Nx.Serving.process_options(batch_keys: Enum.map(lengths, &{:sequence_length, &1}))
    |> Nx.Serving.client_preprocessing(fn input ->
      {inputs, multi?} = validate!(input, validate)
      sequences = Enum.concat(inputs)
      {tensors, encodings} = Tokenizer.encode(tokenizer, sequences)
      key = {:sequence_length, Nx.axis_size(tensors["input_ids"], 1)}
      batch = [model_inputs!(model, tensors)] |> Nx.Batch.concatenate() |> Nx.Batch.key(key)

      items =
        sequences
        |> Enum.zip_with(encodings, fn sequence, windows -> Enum.map(windows, &{sequence, &1}) end)
        |> split_like(inputs)
        |> Enum.map(&Enum.concat/1)

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

  # Splits list into consecutive parts with the lengths of the lists in groups.
  defp split_like(list, groups) do
    {parts, []} = Enum.map_reduce(groups, list, &Enum.split(&2, length(&1)))
    parts
  end

  # Powers of 4 from 32, up to the longest input of the model.
  defp lengths(repo, max_length) do
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

  defp model_inputs!(model, tensors) do
    case model.inputs -- Map.keys(tensors) do
      [] ->
        Map.take(tensors, model.inputs)

      missing ->
        raise ArgumentError, "the tokenizer does not give the model inputs #{inspect(missing)}"
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
  The softmax over the last axis.
  """
  def softmax(logits) do
    exp = Nx.exp(Nx.subtract(logits, Nx.reduce_max(logits, axes: [-1], keep_axes: true)))
    Nx.divide(exp, Nx.sum(exp, axes: [-1], keep_axes: true))
  end
end
