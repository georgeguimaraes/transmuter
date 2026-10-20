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

  A task can add one more processor in the `:processor` key of the context, as `{state, process}`. Before each step, `state` takes the new tokens of a row (the last token first) and gives a tensor. `process` takes the logits and that tensor of one row in the compiled selection, after the suppressed tokens and before the temperature, and gives the new logits.

  ## Batches

  `generate/4` takes a batch of prompts. Each row has its own length, sampling key and stop. A step gives the same number of tokens to each row. A row with fewer tokens gets padding tokens after its tokens, and a finished row gets only padding tokens. The next token of a row comes from the logits of its last real token. An empty prompt is a padding row, which gives no tokens.

  In `:in_place` and `:append`, all rows share one buffer. Each row writes its keys and values after its own valid rows, so the padding tokens of a step go into rows that the next step overwrites. In `:in_place`, the attention mask of a row has ones for its valid rows and the new tokens, and GroupQueryAttention gets the write position of each row from the sum of the mask. In `:append`, the attention mask hides the free rows of each row, and the new rows go into the buffer at the length of each row. Both modes give the positions of each row in `position_ids`.

  In `:grow`, the positions come from the length of the cache, which is the same for all rows. So only prompts of the same length share a cache. The prompts of each length run as one group, and the groups run one after the other. A group runs with all rows of the batch, and the rows of other lengths are padding rows of the group. So each length of the cache has one compiled shape, and a batch with prompts of three lengths runs the steps of three groups. Text-to-text and speech-to-text have prompts of one length, so all their rows run as one group.
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
      processor:
        {fn _generated -> Nx.tensor(0, backend: Nx.BinaryBackend) end,
         fn logits, _state -> logits end}
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
  Compiles the model and the token selection of `ctx` for batches of `batch_size` rows. Call this function in the serving process, which owns the cache of compiled shapes.
  """
  def compile(ctx, batch_size, defn_options) do
    ctx = Map.put(ctx, :backend, Nx.Defn.to_backend(defn_options))

    # The first step of :grow takes the branch without cache, so each value of the static
    # inputs has a function. The inputs that the caller gives have their declared dimensions,
    # with the batch size for the batch and 1 for other symbolic ones. Other shapes compile on
    # their first step and stay cached, and one request of :grow has a shape for each new
    # token and each prompt chunk.
    state = initial_state(ctx, batch_size, @min_capacity)
    opts = [cache_size: ctx.max_new_tokens + @min_capacity] ++ defn_options
    tokens = List.duplicate([0], batch_size)

    runs =
      [0, 1]
      |> Enum.map(
        &step_inputs(ctx, %{state | lengths: List.duplicate(&1, batch_size)}, tokens, 1)
      )
      |> Enum.uniq_by(fn {_inputs, static} -> static end)
      |> Map.new(fn {inputs, static} ->
        names = ctx.model.inputs -- Map.keys(Map.merge(inputs, static))
        given = templates(ctx.model, names, batch_size)

        compiled =
          Transmuter.compile(ctx.model, Map.merge(given, inputs), [static_inputs: static] ++ opts)

        {static, compiled}
      end)

    ctx = Map.put(ctx, :run, fn inputs, static -> runs[static].(inputs) end)

    settings =
      Map.take(ctx, [:repetition_penalty, :sampling?, :temperature, :top_k, :top_p, :processor])

    ctx
    |> Map.put(
      :select,
      Nx.Defn.jit(&select_tokens(&1, &2, &3, &4, &5, &6, &7, settings), defn_options)
    )
    |> Map.put(:append, Nx.Defn.jit(&append_rows/3, defn_options))
  end

  defp templates(model, names, batch_size) do
    Map.new(names, fn name ->
      {type, [batch | dims]} = model.input_types[name]
      zero = Nx.tensor(0, type: type, backend: Nx.BinaryBackend)
      shape = List.to_tuple([batch || batch_size | Enum.map(dims, &(&1 || 1))])
      {name, Nx.broadcast(zero, shape)}
    end)
  end

  @doc """
  Generates tokens after each prompt of `prompts` with a context from `compile/3`. A prompt is a list of token ids, and an empty prompt is a padding row. `seeds` has the seed of sampling of each row. `inputs` are the inputs of each step that `context/4` names in `given`, with one row for each prompt.

  `on_token` is `nil` or a function that receives `%{token: tensor, finished?: tensor}` after each step of a group, with one row for each prompt. `:token` is the new token of the row, or -1 when the row has no new token. `:finished?` is 1 when the row has no more tokens.

  Returns the new token ids of each row, without the eos token.
  """
  def generate(ctx, prompts, seeds, inputs \\ %{}, on_token \\ nil) do
    ids =
      for prompts <- groups(ctx, prompts) do
        notify = &notify(on_token, prompts, &1)
        group = decode(ctx, start(ctx, prompts, seeds, inputs), ctx.max_new_tokens, notify)
        Enum.map(group.ids, &Enum.reverse/1)
      end

    Enum.zip_with([Enum.map(prompts, fn _ -> [] end) | ids], &Enum.concat/1)
  end

  # The prompts of each group of rows that share a cache, with empty prompts for the other
  # rows. A buffer keeps the length of each row, so all prompts are one group.
  defp groups(ctx, prompts) do
    key = if ctx.mode == :grow, do: &length/1, else: fn _prompt -> :buffer end

    for group <- prompts |> Enum.reject(&(&1 == [])) |> Enum.map(key) |> Enum.uniq() do
      Enum.map(prompts, &if(&1 != [] and key.(&1) == group, do: &1, else: []))
    end
  end

  # Runs the prompts in parts of power-of-two lengths. Each prompt fills the parts from the
  # last one, so the last token of each row is in the last part.
  defp start(ctx, prompts, seeds, inputs) do
    longest = prompts |> Enum.map(&length/1) |> Enum.max()
    state = initial_state(ctx, length(prompts), capacity(longest + ctx.max_new_tokens))
    sizes = chunks(longest)

    state =
      prompts
      |> Enum.map(&split_from_end(&1, sizes))
      |> Enum.zip_with(& &1)
      |> Enum.reduce(%{state | inputs: inputs}, &step(ctx, &2, &1))

    {rows, _n, vocab} = Nx.shape(state.logits)
    suppressed = mark(none({vocab}), Enum.map(ctx.suppress, &[&1]))

    # The tokens that the repetition penalty applies to: the prompt and the new tokens.
    prompt_ids = for {prompt, i} <- Enum.with_index(prompts), id <- prompt, do: [i, id]
    seen = mark(none({rows, vocab}), prompt_ids)

    %{
      state: state,
      seen: seen,
      suppressed: suppressed,
      first: mark(suppressed, Enum.map(ctx.begin_suppress, &[&1])),
      keys: seeds |> Enum.map(&Nx.Random.key/1) |> Nx.stack() |> Nx.backend_transfer(ctx.backend),
      ids: Enum.map(prompts, fn _ -> [] end),
      new: [],
      done: Enum.map(prompts, &(&1 == []))
    }
  end

  defp decode(ctx, group, remaining, notify) do
    group = select(ctx, group, remaining)
    notify.(group)

    if Enum.all?(group.done),
      do: group,
      else: decode(ctx, advance(ctx, group), remaining - 1, notify)
  end

  defp select(ctx, group, remaining) do
    suppressed = if remaining == ctx.max_new_tokens, do: group.first, else: group.suppressed
    {processor_state, _process} = ctx.processor
    states = group.ids |> Enum.map(processor_state) |> Nx.stack()

    step = Nx.tensor(ctx.max_new_tokens - remaining, type: :u32, backend: Nx.BinaryBackend)
    %{logits: logits, index: index} = group.state
    tokens = ctx.select.(logits, index, group.seen, suppressed, states, group.keys, step)

    new =
      Enum.zip_with(Nx.to_flat_list(tokens), group.done, fn token, done? ->
        if done? or token in ctx.eos, do: nil, else: token
      end)

    marks = for {token, i} <- Enum.with_index(new), token, do: [i, token]

    %{
      group
      | seen: mark(group.seen, marks),
        new: new,
        done: Enum.map(new, &(&1 == nil or remaining == 1)),
        ids: Enum.zip_with(new, group.ids, &if(&1, do: [&1 | &2], else: &2))
    }
  end

  # The rows of other groups have no token and are not finished.
  defp notify(nil, _prompts, _group), do: :ok

  defp notify(on_token, prompts, group) do
    {tokens, finished} =
      Enum.zip_with([prompts, group.new, group.done], fn
        [[], _token, _done?] -> {-1, 0}
        [_prompt, token, done?] -> {token || -1, if(done?, do: 1, else: 0)}
      end)
      |> Enum.unzip()

    on_token.(%{
      token: Nx.tensor(tokens, type: :s64, backend: Nx.BinaryBackend),
      finished?: Nx.tensor(finished, type: :u8, backend: Nx.BinaryBackend)
    })
  end

  defp advance(ctx, group),
    do: %{group | state: step(ctx, group.state, Enum.map(group.new, &List.wrap/1))}

  # The masks of tokens are on the host, because a mark there is faster than a call to the
  # device. They are made from binaries, because Nx.broadcast on Nx.BinaryBackend goes
  # through a list of the elements, which is slow for a large vocabulary.
  defp none(shape), do: shape |> Tuple.product() |> bytes(0) |> Nx.reshape(shape)

  defp mark(tensor, []), do: tensor

  defp mark(tensor, indices) do
    indices = Nx.tensor(indices, backend: Nx.BinaryBackend)
    Nx.indexed_put(tensor, indices, bytes(Nx.axis_size(indices, 0), 1))
  end

  defp bytes(count, value),
    do: Nx.from_binary(:binary.copy(<<value>>, count), :u8, backend: Nx.BinaryBackend)

  defp split_from_end(prompt, sizes) do
    {parts, []} =
      sizes
      |> Enum.reverse()
      |> Enum.map_reduce(Enum.reverse(prompt), fn size, rest ->
        {part, rest} = Enum.split(rest, size)
        {Enum.reverse(part), rest}
      end)

    Enum.reverse(parts)
  end

  # 45 -> [32, 8, 4, 1] tokens.
  defp chunks(length) do
    for bit <- 30..0//-1,
        Bitwise.band(length, Bitwise.bsl(1, bit)) != 0,
        do: Bitwise.bsl(1, bit)
  end

  defp capacity(length), do: max(@min_capacity, 2 ** ceil(:math.log2(length)))

  defp initial_state(ctx, batch_size, capacity) do
    # :grow starts with a placeholder row that the branch without cache does not read.
    # The zeros are made on the backend of the compiler, so they do not cross from the host.
    rows = if ctx.mode == :grow, do: 1, else: capacity

    cache =
      Map.new(ctx.cache, fn name ->
        {type, [_batch, heads, _length, head_dim]} = ctx.model.input_types[name]
        zero = Nx.tensor(0, type: type, backend: ctx.backend)
        {name, Nx.broadcast(zero, {batch_size, heads, rows, head_dim})}
      end)

    lengths = List.duplicate(0, batch_size)
    %{cache: cache, inputs: %{}, lengths: lengths, capacity: capacity, logits: nil, index: nil}
  end

  # Gives each row its tokens of `parts` and padding tokens after them, so that all rows
  # have the same number of tokens. The logits of the step start at the last real token of
  # the shortest row, and `index` is the place of the last real token of each row in them.
  defp step(ctx, state, parts) do
    counts = Enum.map(parts, &length/1)
    n = Enum.max(counts)
    tokens = Enum.map(parts, &(&1 ++ List.duplicate(0, n - length(&1))))
    keep = n - (counts |> Enum.reject(&(&1 == 0)) |> Enum.min()) + 1
    {inputs, static} = step_inputs(ctx, state, tokens, keep)
    outputs = ctx.run.(inputs, static)

    # Transmuter.run leaves out empty outputs, such as the present of a cache that does not
    # change, so that cache keeps its value.
    present =
      for "past_key_values." <> rest = name <- ctx.cache,
          Map.has_key?(outputs, "present." <> rest),
          into: %{},
          do: {name, outputs["present." <> rest]}

    present =
      if ctx.mode == :append do
        offsets = Nx.tensor(state.lengths, backend: Nx.BinaryBackend)
        ctx.append.(Map.take(state.cache, Map.keys(present)), present, offsets)
      else
        present
      end

    # A buffer keeps the real tokens of each row. The cache of :grow has all tokens.
    lengths =
      if ctx.mode == :grow,
        do: Enum.map(state.lengths, &(&1 + n)),
        else: Enum.zip_with(state.lengths, counts, &+/2)

    logits = outputs["logits"]
    skipped = n - Nx.axis_size(logits, 1)
    index = Nx.tensor(Enum.map(counts, &max(&1 - 1 - skipped, 0)), backend: Nx.BinaryBackend)

    %{
      state
      | cache: Map.merge(state.cache, present),
        lengths: lengths,
        logits: logits,
        index: index
    }
  end

  # The rows of each present cache after the buffer are the new rows, and they go into the
  # buffer after the valid rows of each batch row. One compiled function writes all of
  # them, because each eager slice is a call to the device.
  defp append_rows(cache, present, offsets) do
    Map.new(cache, fn {name, buffer} ->
      {batch_size, _heads, capacity, _dim} = Nx.shape(buffer)
      rows = Nx.axis_size(present[name], 2) - capacity
      new = Nx.slice_along_axis(present[name], capacity, rows, axis: 2)

      buffer =
        Enum.reduce(0..(batch_size - 1), buffer, fn i, buffer ->
          row = Nx.slice_along_axis(new, i, 1, axis: 0)
          Nx.put_slice(buffer, [i, 0, offsets[i], 0], row)
        end)

      {name, buffer}
    end)
  end

  defp step_inputs(ctx, state, tokens, keep) do
    n = length(hd(tokens))

    masks =
      for past <- state.lengths do
        case ctx.mode do
          :in_place ->
            List.duplicate(1, past + n) ++ List.duplicate(0, state.capacity - past - n)

          :append ->
            List.duplicate(1, past) ++
              List.duplicate(0, state.capacity - past) ++ List.duplicate(1, n)

          :grow ->
            List.duplicate(1, past + n)
        end
      end

    values = %{
      "input_ids" => tokens,
      "attention_mask" => masks,
      "position_ids" => for(past <- state.lengths, do: Enum.to_list(past..(past + n - 1))),
      "num_logits_to_keep" => keep
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
        flag = if ctx.mode == :grow and hd(state.lengths) == 0, do: 0, else: 1
        %{"use_cache_branch" => Nx.tensor([flag], type: :u8, backend: Nx.BinaryBackend)}
      else
        %{}
      end

    {inputs, static}
  end

  # Selects the token of each row from its logits at `index`. The key of a step is the key
  # of the row folded with the number of the step. The serving compiles this function,
  # because eager random numbers over a large vocabulary are slow. EMLX evaluates each
  # output of a compiled function separately, so the function gives only the tokens.
  defp select_tokens(logits, index, seen, suppressed, states, keys, step, settings) do
    [logits, index, seen, states, keys] =
      Enum.map([logits, index, seen, states, keys], &Nx.vectorize(&1, :rows))

    logits = Nx.take(logits, index)
    key = Nx.Random.fold_in(keys, step)
    token = select_token(logits, seen, suppressed, states, key, settings)
    Nx.devectorize(token, keep_names: false)
  end

  # The logits processors of Python transformers, in its order: repetition penalty,
  # suppressed tokens, the processor of the task, temperature, top-k, top-p. Removed tokens
  # get the smallest finite value, because EMLX 0.5 does not compile infinite constants.
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
