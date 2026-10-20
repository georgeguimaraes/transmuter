defmodule Transmuter.Generation do
  @moduledoc """
  The generation loop of a decoder with a KV cache. `Transmuter.Text.generation/2`, `Transmuter.Text.text_to_text/2` and `Transmuter.Audio.speech_to_text/2` use it.

  Generation runs the model in steps. A step takes new tokens and the KV cache, and gives the logits and the next cache. The prompt runs in steps of power-of-two lengths (45 tokens run as 32, 8, 4 and 1), and each new token is a step of length 1, so a small number of compiled shapes covers all prompts.

  The cache works in one of three modes, from the structure of the graph:

    * `:in_place` - the graph has GroupQueryAttention. The cache is a buffer as long as the attention mask, and the operator writes the new keys and values after the valid rows.

    * `:append` - the graph declares `position_ids`. The cache is a buffer whose free rows the attention mask hides. The present cache is the buffer and the new rows after it, and the new rows go into the buffer.

    * `:grow` - the graph has no `position_ids`, so it computes the positions from the length of the cache. The cache has its exact length, so each length is a new shape that compiles on its first step. The first step takes the branch without cache of a merged decoder (`use_cache_branch`), because Nx has no tensors of length 0.

  The decoder of an encoder-decoder model also takes the outputs of the encoder, such as `encoder_hidden_states`, in each step. Its first step computes the cross-attention cache (`past_key_values.N.encoder.key` and `.value`) from these outputs. Later steps give an empty present for this cache, so the cache keeps its value.

  The token selection applies the logits processors of Python transformers that `generation_config.json` sets: `repetition_penalty`, `suppress_tokens`, `begin_suppress_tokens` (on the first new token), the bad words of one token in `bad_words_ids`, `temperature`, `top_k` and `top_p`.

  A task can add one more processor in the `:processor` key of the context, as `{state, process}`. Before each step, `state` takes the new tokens (the last token first) and gives a tensor. `process` takes the logits and that tensor in the compiled selection, after the suppressed tokens and before the temperature, and gives the new logits.
  """

  alias Transmuter.Serving

  # Buffers have a capacity of a power of two, from this size.
  @min_capacity 256

  @doc """
  Returns the context of generation for `model`: the cache mode and the settings from `generation_config.json` and `opts`. `given` are the names of other inputs that the caller gives to `generate/4`, such as the outputs of an encoder.
  """
  def context(repo, model, opts, given \\ []) do
    gen = repo.generation_config

    unless opts[:strategy] in [nil, :greedy, :sampling] do
      raise ArgumentError,
            "expected :strategy to be :greedy or :sampling, got: #{inspect(opts[:strategy])}"
    end

    %{
      model: model,
      mode: mode(model, given),
      cache: Enum.filter(model.inputs, &String.starts_with?(&1, "past_key_values.")),
      max_new_tokens: opts[:max_new_tokens] || gen["max_new_tokens"] || 256,
      sampling?: Keyword.get(opts, :strategy, if(gen["do_sample"], do: :sampling)) == :sampling,
      temperature: opts[:temperature] || gen["temperature"] || 1.0,
      top_k: opts[:top_k] || gen["top_k"] || 50,
      top_p: opts[:top_p] || gen["top_p"] || 1.0,
      repetition_penalty: opts[:repetition_penalty] || gen["repetition_penalty"] || 1.0,
      suppress: suppress_tokens(gen),
      begin_suppress: List.wrap(gen["begin_suppress_tokens"]),
      eos: List.wrap(gen["eos_token_id"] || repo.config["eos_token_id"]),
      bos: gen["bos_token_id"] || repo.config["bos_token_id"],
      processor: {fn _generated -> 0 end, fn logits, _state -> logits end}
    }
  end

  # The tokens that each step suppresses: suppress_tokens and the bad words of one token.
  defp suppress_tokens(gen),
    do: List.wrap(gen["suppress_tokens"]) ++ for([id] <- List.wrap(gen["bad_words_ids"]), do: id)

  @inputs ~w(input_ids attention_mask position_ids num_logits_to_keep use_cache_branch)

  defp mode(model, given) do
    case model.inputs --
           (@inputs ++
              given ++
              Enum.filter(model.inputs, &String.starts_with?(&1, "past_key_values."))) do
      [] ->
        :ok

      other ->
        raise ArgumentError,
              "the decoder has inputs that generation does not give: #{inspect(other)}"
    end

    gqa = Enum.find(model.nodes, &(&1.op_type == "GroupQueryAttention"))

    cond do
      # A buffer as long as the mask works when GroupQueryAttention counts the valid rows
      # from the values of the mask. Gemma 3 counts them from its length.
      gqa && reads_values?(model, Enum.at(gqa.input, 5), "attention_mask") ->
        :in_place

      gqa ->
        :grow

      "position_ids" in model.inputs ->
        :append

      "use_cache_branch" in model.inputs ->
        :grow

      true ->
        raise ArgumentError, "the decoder declares neither position_ids nor use_cache_branch"
    end
  end

  # Whether the values of `name` depend on the values of `target`. The output of Shape and
  # Size depends only on the shape of its input.
  defp reads_values?(model, name, target) do
    producers = for node <- model.nodes, output <- node.output, into: %{}, do: {output, node}
    walk([name], target, producers, MapSet.new())
  end

  defp walk([], _target, _producers, _seen), do: false

  defp walk([name | rest], target, producers, seen) do
    node = producers[name]

    cond do
      name == target ->
        true

      name in seen or node == nil or node.op_type in ["Shape", "Size"] ->
        walk(rest, target, producers, seen)

      true ->
        walk(node.input ++ rest, target, producers, MapSet.put(seen, name))
    end
  end

  @doc """
  Compiles the model and the token selection of `ctx`. Call this function in the serving process, which owns the cache of compiled shapes.
  """
  def compile(ctx, defn_options) do
    ctx = Map.put(ctx, :backend, Nx.Defn.to_backend(defn_options))

    # The first step of :grow takes the branch without cache, so each value of the static
    # inputs has a function. The inputs that the caller gives have their declared dimensions,
    # with 1 for a symbolic one. Other shapes compile on their first step and stay cached,
    # and one request of :grow has a shape for each new token and each prompt chunk.
    state = initial_state(ctx, @min_capacity)
    opts = [cache_size: ctx.max_new_tokens + @min_capacity] ++ defn_options

    runs =
      [0, 1]
      |> Enum.map(&step_inputs(ctx, %{state | length: &1}, [0]))
      |> Enum.uniq_by(fn {_inputs, static} -> static end)
      |> Map.new(fn {inputs, static} ->
        given = templates(ctx.model, ctx.model.inputs -- Map.keys(Map.merge(inputs, static)))

        compiled =
          Transmuter.compile(ctx.model, Map.merge(given, inputs), [static_inputs: static] ++ opts)

        {static, compiled}
      end)

    ctx = Map.put(ctx, :run, fn inputs, static -> runs[static].(inputs) end)

    settings =
      Map.take(ctx, [:repetition_penalty, :sampling?, :temperature, :top_k, :top_p, :processor])

    ctx
    |> Map.put(:select, Nx.Defn.jit(&select_token(&1, &2, &3, &4, &5, settings), defn_options))
    |> Map.put(:append, Nx.Defn.jit(&append_rows/3, defn_options))
  end

  defp templates(model, names) do
    Map.new(names, fn name ->
      {type, dims} = model.input_types[name]
      zero = Nx.tensor(0, type: type, backend: Nx.BinaryBackend)
      {name, Nx.broadcast(zero, List.to_tuple(Enum.map(dims, &(&1 || 1))))}
    end)
  end

  @doc """
  Generates tokens after the prompt `ids` with a context from `compile/2`. `inputs` are the inputs of each step that `context/4` names in `given`. Returns the new token ids, without the eos token.
  """
  def generate(ctx, ids, seed, inputs \\ %{}) do
    capacity = capacity(length(ids) + ctx.max_new_tokens)
    state = %{initial_state(ctx, capacity) | inputs: inputs}

    {logits, state} =
      Enum.reduce(chunks(ids), {nil, state}, fn chunk, {_, state} -> step(ctx, state, chunk) end)

    none = Nx.broadcast(Nx.tensor(0, type: :u8), logits.shape)
    suppressed = mark(none, ctx.suppress)
    ctx = Map.merge(ctx, %{suppressed: suppressed, first: mark(suppressed, ctx.begin_suppress)})

    # The tokens that the repetition penalty applies to: the prompt and the new tokens.
    seen = mark(none, ids)
    decode(ctx, logits, state, Nx.Random.key(seed), seen, [], ctx.max_new_tokens)
  end

  defp decode(_ctx, _logits, _state, _key, _seen, generated, 0), do: Enum.reverse(generated)

  defp decode(ctx, logits, state, key, seen, generated, remaining) do
    keys = Nx.Random.split(key)
    suppressed = if generated == [], do: ctx.first, else: ctx.suppressed
    {processor_state, _process} = ctx.processor

    token =
      ctx.select.(logits, seen, suppressed, processor_state.(generated), keys[0])
      |> Nx.to_number()

    cond do
      token in ctx.eos ->
        Enum.reverse(generated)

      remaining == 1 ->
        Enum.reverse([token | generated])

      true ->
        {logits, state} = step(ctx, state, [token])

        decode(
          ctx,
          logits,
          state,
          keys[1],
          mark(seen, [token]),
          [token | generated],
          remaining - 1
        )
    end
  end

  defp mark(seen, []), do: seen

  defp mark(seen, ids) do
    indices = Nx.tensor(Enum.map(ids, &[&1]))
    Nx.indexed_put(seen, indices, Nx.broadcast(Nx.tensor(1, type: :u8), {length(ids)}))
  end

  # 45 -> [32, 8, 4, 1] tokens.
  defp chunks(ids) do
    sizes =
      for bit <- 30..0//-1,
          Bitwise.band(length(ids), Bitwise.bsl(1, bit)) != 0,
          do: Bitwise.bsl(1, bit)

    {chunks, []} = Enum.map_reduce(sizes, ids, &Enum.split(&2, &1))
    chunks
  end

  defp capacity(length), do: max(@min_capacity, 2 ** ceil(:math.log2(length)))

  defp initial_state(ctx, capacity) do
    # :grow starts with a placeholder row that the branch without cache does not read.
    # The zeros are made on the backend of the compiler, so they do not cross from the host.
    rows = if ctx.mode == :grow, do: 1, else: capacity

    cache =
      Map.new(ctx.cache, fn name ->
        {type, [_batch, heads, _length, head_dim]} = ctx.model.input_types[name]
        zero = Nx.tensor(0, type: type, backend: ctx.backend)
        {name, Nx.broadcast(zero, {1, heads, rows, head_dim})}
      end)

    %{cache: cache, inputs: %{}, length: 0, capacity: capacity}
  end

  defp step(ctx, state, tokens) do
    {inputs, static} = step_inputs(ctx, state, tokens)
    outputs = ctx.run.(inputs, static)

    # Transmuter.run leaves out empty outputs, such as the present of a cache that does not
    # change, so that cache keeps its value.
    present =
      for "past_key_values." <> rest = name <- ctx.cache,
          Map.has_key?(outputs, "present." <> rest),
          into: %{},
          do: {name, outputs["present." <> rest]}

    present =
      if ctx.mode == :append,
        do: ctx.append.(Map.take(state.cache, Map.keys(present)), present, state.length),
        else: present

    logits = outputs["logits"][[0, -1]]
    cache = Map.merge(state.cache, present)
    {logits, %{state | cache: cache, length: state.length + length(tokens)}}
  end

  # The rows of each present cache after the buffer are the new rows, and they go into the
  # buffer after its valid rows. One compiled function writes all of them, because each
  # eager slice is a call to the device.
  defp append_rows(cache, present, offset) do
    Map.new(cache, fn {name, buffer} ->
      capacity = Nx.axis_size(buffer, 2)
      rows = Nx.axis_size(present[name], 2) - capacity
      new = Nx.slice_along_axis(present[name], capacity, rows, axis: 2)
      {name, Nx.put_slice(buffer, [0, 0, offset, 0], new)}
    end)
  end

  defp step_inputs(ctx, state, tokens) do
    n = length(tokens)
    past = state.length

    mask =
      case ctx.mode do
        :in_place ->
          List.duplicate(1, past + n) ++ List.duplicate(0, state.capacity - past - n)

        :append ->
          List.duplicate(1, past) ++
            List.duplicate(0, state.capacity - past) ++ List.duplicate(1, n)

        :grow ->
          List.duplicate(1, past + n)
      end

    values = %{
      "input_ids" => [tokens],
      "attention_mask" => [mask],
      "position_ids" => [Enum.to_list(past..(past + n - 1))],
      "num_logits_to_keep" => 1
    }

    inputs =
      for {name, value} <- values,
          name in ctx.model.inputs,
          into: Map.merge(state.cache, state.inputs) do
        {type, _dims} = ctx.model.input_types[name]
        {name, Nx.tensor(value, type: type, backend: Nx.BinaryBackend)}
      end

    static =
      if "use_cache_branch" in ctx.model.inputs do
        flag = if ctx.mode == :grow and past == 0, do: 0, else: 1
        %{"use_cache_branch" => Nx.tensor([flag], type: :u8, backend: Nx.BinaryBackend)}
      else
        %{}
      end

    {inputs, static}
  end

  # The logits processors of Python transformers, in its order: repetition penalty,
  # suppressed tokens, the processor of the task, temperature, top-k, top-p. The serving
  # compiles this function, because eager random numbers over a large vocabulary are slow.
  # Removed tokens get the smallest finite value, because EMLX 0.5 does not compile infinite
  # constants.
  defp select_token(logits, seen, suppressed, state, key, settings) do
    logits = Nx.as_type(logits, :f32)
    penalty = settings.repetition_penalty
    {_state, process} = settings.processor

    penalized =
      Nx.select(Nx.greater(logits, 0), Nx.divide(logits, penalty), Nx.multiply(logits, penalty))

    logits = Nx.select(seen, penalized, logits)
    logits = Nx.select(suppressed, Nx.Constants.min_finite(:f32), logits)
    logits = process.(logits, state)

    if settings.sampling? do
      logits
      |> Nx.divide(settings.temperature)
      |> top_k(settings.top_k)
      |> top_p(settings.top_p)
      |> sample(key)
    else
      Nx.argmax(logits)
    end
  end

  defp top_k(logits, k) when k > 0 and k < elem(logits.shape, 0) do
    {values, _} = Nx.top_k(logits, k: k)
    Nx.select(Nx.less(logits, values[-1]), Nx.Constants.min_finite(:f32), logits)
  end

  defp top_k(logits, _k), do: logits

  # Keeps the most likely tokens whose probabilities add up to p, and always the first one.
  defp top_p(logits, p) when p < 1.0 do
    sorted = Nx.sort(logits, direction: :desc)
    probabilities = Serving.softmax(sorted)
    before = Nx.subtract(Nx.cumulative_sum(probabilities), probabilities)
    last = before |> Nx.less(p) |> Nx.sum() |> Nx.subtract(1) |> Nx.max(0)
    Nx.select(Nx.less(logits, Nx.take(sorted, last)), Nx.Constants.min_finite(:f32), logits)
  end

  defp top_p(logits, _p), do: logits

  # The Gumbel-max trick samples from the softmax of the logits.
  defp sample(logits, key) do
    {uniform, _} = Nx.Random.uniform(key, 1.0e-10, 1.0, shape: logits.shape)
    gumbel = uniform |> Nx.log() |> Nx.negate() |> Nx.log() |> Nx.negate()
    logits |> Nx.add(gumbel) |> Nx.argmax()
  end
end
