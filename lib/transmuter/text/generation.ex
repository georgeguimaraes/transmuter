defmodule Transmuter.Text.Generation do
  @moduledoc """
  The text generation task. See `Transmuter.Text.generation/2`.

  Generation runs the model in steps. A step takes new tokens and the KV cache, and gives the logits and the next cache. The prompt runs in steps of power-of-two lengths (45 tokens run as 32, 8, 4 and 1), and each new token is a step of length 1, so a small number of compiled shapes covers all prompts.

  The cache works in one of three modes, from the structure of the graph:

    * `:in_place` - the graph has GroupQueryAttention. The cache is a buffer as long as the attention mask, and the operator writes the new keys and values after the valid rows.

    * `:append` - the graph declares `position_ids`. The cache is a buffer whose free rows the attention mask hides. The present cache is the buffer and the new rows after it, and the new rows go into the buffer.

    * `:grow` - the graph has no `position_ids`, so it computes the positions from the length of the cache. The cache has its exact length, and each step runs eagerly, because each length is a new shape. The first step takes the branch without cache of a merged decoder (`use_cache_branch`), because Nx has no tensors of length 0.
  """

  @compile {:no_warn_undefined, [Tokenizers.Tokenizer, Tokenizers.Encoding]}

  alias Transmuter.{ChatTemplate, Serving, Tokenizer}

  # Buffers have a capacity of a power of two, from this size.
  @min_capacity 256

  @doc false
  def generation(source, opts) do
    opts =
      Keyword.validate!(opts, [
        :variant,
        :cache_dir,
        :model,
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
        defn_options: []
      ])

    {repo, model, tokenizer} = Serving.load!(source, opts, :q4, ["model", "decoder_model_merged"])
    ctx = context(repo, model, opts)

    Nx.Serving.new(
      fn _key, defn_options -> run_fun(ctx, defn_options) end,
      opts[:defn_options]
    )
    |> Nx.Serving.batch_size(1)
    |> Nx.Serving.client_preprocessing(fn input ->
      {inputs, multi?} = validate!(input)

      prompts =
        for input <- inputs do
          {text, special?} = prompt(repo, input, opts)
          seed = input[:seed] || opts[:seed] || :rand.uniform(2 ** 32)
          ids = Nx.tensor(token_ids(tokenizer, text, special?, ctx.bos), type: :s64)
          %{"ids" => ids, "seed" => Nx.tensor(seed, type: :s64)}
        end

      batches = Enum.map(prompts, &Nx.Batch.stack([&1]))
      {batches, {multi?, Enum.map(prompts, &Nx.size(&1["ids"]))}}
    end)
    |> Nx.Serving.client_postprocessing(fn {outputs, _metadata}, {multi?, input_lengths} ->
      results =
        for {length, i} <- Enum.with_index(input_lengths) do
          count = Nx.to_number(outputs.length[i])
          ids = outputs.ids[i] |> Nx.to_flat_list() |> Enum.take(count)
          summary = %{input: length, output: count, padding: 0}
          %{results: [%{text: Tokenizer.decode(tokenizer, ids), token_summary: summary}]}
        end

      if multi?, do: results, else: hd(results)
    end)
  end

  defp context(repo, model, opts) do
    gen = repo.generation_config

    unless opts[:strategy] in [nil, :greedy, :sampling] do
      raise ArgumentError,
            "expected :strategy to be :greedy or :sampling, got: #{inspect(opts[:strategy])}"
    end

    %{
      model: model,
      mode: mode(model),
      cache: Enum.filter(model.inputs, &String.starts_with?(&1, "past_key_values.")),
      max_new_tokens: opts[:max_new_tokens] || gen["max_new_tokens"] || 256,
      sampling?: Keyword.get(opts, :strategy, if(gen["do_sample"], do: :sampling)) == :sampling,
      temperature: opts[:temperature] || gen["temperature"] || 1.0,
      top_k: opts[:top_k] || gen["top_k"] || 50,
      top_p: opts[:top_p] || gen["top_p"] || 1.0,
      repetition_penalty: opts[:repetition_penalty] || gen["repetition_penalty"] || 1.0,
      eos: List.wrap(gen["eos_token_id"] || repo.config["eos_token_id"]),
      bos: gen["bos_token_id"] || repo.config["bos_token_id"]
    }
  end

  @inputs ~w(input_ids attention_mask position_ids num_logits_to_keep use_cache_branch)

  defp mode(model) do
    case model.inputs --
           (@inputs ++ Enum.filter(model.inputs, &String.starts_with?(&1, "past_key_values."))) do
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

  # The function of a batch. Buffer modes compile here, in the serving process, which owns
  # the cache of compiled shapes.
  defp run_fun(ctx, defn_options) do
    ctx =
      if ctx.mode == :grow do
        Map.put(ctx, :run, fn inputs, static ->
          Transmuter.run(ctx.model, ctx.model.params, inputs, static)
        end)
      else
        state = initial_state(ctx, @min_capacity)
        {inputs, static} = step_inputs(ctx, state, [0])
        opts = [static_inputs: static, cache_size: 32] ++ defn_options
        compiled = Transmuter.compile(ctx.model, inputs, opts)
        Map.put(ctx, :run, fn inputs, _static -> compiled.(inputs) end)
      end

    settings = Map.take(ctx, [:repetition_penalty, :sampling?, :temperature, :top_k, :top_p])
    ctx = Map.put(ctx, :select, Nx.Defn.jit(&select_token(&1, &2, &3, settings), defn_options))

    fn batch ->
      %{"ids" => ids, "seed" => seed} =
        Nx.Defn.jit_apply(& &1, [batch], compiler: Nx.Defn.Evaluator)

      ids = Nx.to_flat_list(ids)
      generated = generate(ctx, ids, Nx.to_number(seed[0]))
      padded = generated ++ List.duplicate(0, ctx.max_new_tokens - length(generated))

      %{
        ids: Nx.tensor([padded], type: :s64, backend: Nx.BinaryBackend),
        length: Nx.tensor([length(generated)], backend: Nx.BinaryBackend)
      }
    end
  end

  defp generate(ctx, ids, seed) do
    capacity = capacity(length(ids) + ctx.max_new_tokens)

    {logits, state} =
      Enum.reduce(chunks(ids), {nil, initial_state(ctx, capacity)}, fn chunk, {_, state} ->
        step(ctx, state, chunk)
      end)

    # The tokens that the repetition penalty applies to: the prompt and the new tokens.
    seen = Nx.broadcast(Nx.tensor(0, type: :u8), logits.shape) |> mark(ids)
    decode(ctx, logits, state, Nx.Random.key(seed), seen, [], ctx.max_new_tokens)
  end

  defp decode(_ctx, _logits, _state, _key, _seen, generated, 0), do: Enum.reverse(generated)

  defp decode(ctx, logits, state, key, seen, generated, remaining) do
    keys = Nx.Random.split(key)
    token = ctx.select.(logits, seen, keys[0]) |> Nx.to_number()

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
    rows = if ctx.mode == :grow, do: 1, else: capacity

    cache =
      Map.new(ctx.cache, fn name ->
        {type, [_batch, heads, _length, head_dim]} = ctx.model.input_types[name]

        {name,
         Nx.broadcast(
           Nx.tensor(0, type: type, backend: Nx.BinaryBackend),
           {1, heads, rows, head_dim}
         )}
      end)

    %{cache: cache, length: 0, capacity: capacity}
  end

  defp step(ctx, state, tokens) do
    {inputs, static} = step_inputs(ctx, state, tokens)
    outputs = ctx.run.(inputs, static)
    n = length(tokens)

    cache =
      Map.new(ctx.cache, fn "past_key_values." <> rest = name ->
        present = outputs["present." <> rest]

        case ctx.mode do
          :append ->
            new = Nx.slice_along_axis(present, state.capacity, n, axis: 2)
            {name, Nx.put_slice(state.cache[name], [0, 0, state.length, 0], new)}

          _in_place_or_grow ->
            {name, present}
        end
      end)

    logits = outputs["logits"][[0, -1]]
    {logits, %{state | cache: cache, length: state.length + n}}
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
      for {name, value} <- values, name in ctx.model.inputs, into: state.cache do
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
  # temperature, top-k, top-p. The serving compiles this function, because eager random
  # numbers over a large vocabulary are slow. Removed tokens get the smallest finite value,
  # because EMLX 0.5 does not compile infinite constants.
  defp select_token(logits, seen, key, settings) do
    logits = Nx.as_type(logits, :f32)
    penalty = settings.repetition_penalty

    penalized =
      Nx.select(Nx.greater(logits, 0), Nx.divide(logits, penalty), Nx.multiply(logits, penalty))

    logits = Nx.select(seen, penalized, logits)

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

  # A chat goes through the chat template, which adds the special tokens itself.
  defp prompt(_repo, %{text: text}, _opts), do: {text, true}

  # As the text generation pipeline of transformers, a chat that ends with an assistant
  # message continues that message.
  defp prompt(repo, input = %{messages: messages}, opts) do
    final = List.last(messages)

    template_opts = [
      tools: input[:tools],
      template_options: opts[:template_options],
      continue_final_message: (final[:role] || final["role"]) == "assistant"
    ]

    {ChatTemplate.render(repo, messages, template_opts), false}
  end

  # An empty prompt starts from the bos token, as in transformers.
  defp token_ids(tokenizer, text, special?, bos) do
    {:ok, encoding} =
      Tokenizers.Tokenizer.encode(tokenizer.native, text, add_special_tokens: special?)

    case Tokenizers.Encoding.get_ids(encoding) do
      [] ->
        [bos || raise(ArgumentError, "the prompt is empty, and the model has no bos_token_id")]

      ids ->
        ids
    end
  end

  defp validate!(inputs) when is_list(inputs), do: {Enum.map(inputs, &validate_one!/1), true}
  defp validate!(input), do: {[validate_one!(input)], false}

  defp validate_one!(text) when is_binary(text), do: %{text: text}
  defp validate_one!(input = %{text: text}) when is_binary(text), do: input
  defp validate_one!(input = %{messages: [_ | _]}), do: input

  defp validate_one!(input) do
    raise ArgumentError,
          "expected a string, %{text: string} or %{messages: [%{role: role, content: content}]}, got: #{inspect(input)}"
  end
end
