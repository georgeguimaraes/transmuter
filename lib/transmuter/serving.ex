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
  Loads the repository of `source`, its `"model"` component and its tokenizer. `variant` is the default variant of the task.
  """
  def load!(source, opts, variant) do
    repo = Repo.load!(source, Keyword.take(opts, [:variant, :cache_dir]))

    model =
      opts[:model] ||
        Repo.load_component!(
          repo,
          "model",
          variant,
          Keyword.take(opts, [:backend, :compute_type])
        )

    {repo, model, opts[:tokenizer] || Tokenizer.load!(repo)}
  end

  @doc """
  Returns an `Nx.Serving` that runs `model` on text.

  `fun` takes the outputs of the model and the inputs of a batch, and returns a map of tensors with the batch on the first axis. `postprocess` takes the row of one input from that map and `{input, encoding}`, and returns the result for that input. `validate` takes one input and returns `{:ok, input}` or `{:error, message}`.

  The `:max_length` option of a task limits the default sequence lengths, before the limits in `tokenizer_config.json` and `config.json`.
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
      {tensors, encodings} = Tokenizer.encode(tokenizer, inputs)
      key = {:sequence_length, Nx.axis_size(tensors["input_ids"], 1)}
      batch = [model_inputs!(model, tensors)] |> Nx.Batch.concatenate() |> Nx.Batch.key(key)
      {batch, {multi?, Enum.zip(inputs, encodings)}}
    end)
    |> Nx.Serving.client_postprocessing(fn {outputs, _metadata}, {multi?, items} ->
      results =
        for {item, i} <- Enum.with_index(items) do
          postprocess.(Map.new(outputs, fn {name, t} -> {name, t[i]} end), item)
        end

      if multi?, do: results, else: hd(results)
    end)
  end

  # Powers of 4 from 32, up to the longest input of the model.
  defp lengths(repo, max_length) do
    max =
      Enum.find(
        [
          max_length,
          repo.tokenizer_config["model_max_length"],
          repo.config["max_position_embeddings"]
        ],
        512,
        &(is_integer(&1) and &1 <= 1_000_000)
      )

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
  Accepts a string.
  """
  def string(input) when is_binary(input), do: {:ok, input}
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
