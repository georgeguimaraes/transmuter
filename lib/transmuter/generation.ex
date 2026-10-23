defmodule Transmuter.Generation do
  @moduledoc """
  The generation loop of a decoder with a KV cache. `Transmuter.Text.generation/2`, `Transmuter.Text.text_to_text/2`, `Transmuter.Vision.image_to_text/2`, `Transmuter.Vision.image_text_to_text/2` and `Transmuter.Audio.speech_to_text/2` use it.

  Generation runs the model in steps. A step takes new tokens and the KV cache, and gives the logits and the next cache. The prompt runs in steps of power-of-two lengths (45 tokens run as 32, 8, 4 and 1), and each new token is a step of length 1, so a small number of compiled shapes covers all prompts.

  The cache works in one of three modes, from the structure of the graph:

    * `:in_place` - the graph has GroupQueryAttention, which counts the valid rows of the cache from the values of the attention mask. The cache is a buffer as long as the attention mask, and the operator writes the new keys and values after the valid rows.

    * `:append` - the graph declares `position_ids`. The cache is a buffer whose free rows the attention mask hides. The present cache is the buffer and the new rows after it, and the new rows go into the buffer.

    * `:grow` - the graph computes the positions from the length of the cache. It has no `position_ids`, or its GroupQueryAttention counts the valid rows from the length of the attention mask, as in Gemma 3. The cache has its exact length, so each length is a new shape that compiles on its first step. The first step takes the branch without cache of a merged decoder (`use_cache_branch`), because Nx has no tensors of length 0.

  The decoder of an encoder-decoder model also takes the outputs of the encoder, such as `encoder_hidden_states`, in each step. Its first step computes the cross-attention cache (`past_key_values.N.encoder.key` and `.value`) from these outputs. Later steps give an empty present for this cache, so the cache keeps its value.

  The token selection applies the logits processors of Python transformers that `generation_config.json` sets, in the order of `_get_logits_processor`: `repetition_penalty`, `no_repeat_ngram_size`, the bad words of one token in `bad_words_ids`, `min_length` or `min_new_tokens`, `forced_bos_token_id`, `forced_eos_token_id`, `suppress_tokens`, `begin_suppress_tokens`, the processors of the task, `renormalize_logits`, `temperature`, `top_k` and `top_p`.

  ## Lengths

  The processors count the tokens of each row as Python transformers counts the columns of `input_ids`: the prompt and the new tokens. The prompt of a decoder-only model is the text, and the prompt of the decoder of an encoder-decoder model starts with `decoder_start_token_id`. Each row has its own length, and the rows of a batch are not padded. Transmuter limits the new tokens with `max_new_tokens`, so the `max_length` of a row in Python is the length of its prompt plus `max_new_tokens`. Thus:

    * `no_repeat_ngram_size` - the n-grams of the prompt and the new tokens can occur only once.
    * `min_length` - the eos tokens are removed while the prompt and the new tokens are fewer than `min_length`.
    * `min_new_tokens` - the eos tokens are removed while the new tokens are fewer than `min_new_tokens`. As in Python, `min_new_tokens` replaces `min_length` when both are set.
    * `forced_bos_token_id` - the token when the row has one token, which is the first new token of an encoder-decoder model. A decoder-only prompt of more than one token gets no forced token, as in Python.
    * `forced_eos_token_id` - the token when the row has `max_length - 1` tokens, which is the last new token.
    * `begin_suppress_tokens` - removed at the first new token, or at the second new token after a forced bos token of a prompt of one token.

  ## Embeddings and image features

  The decoder of a vision language model, such as SmolVLM or Qwen2-VL, declares `inputs_embeds` in place of `input_ids`. Its context has the model of the embeddings in `:embed` (`embed_tokens.onnx`) and the id of the image token in `:image_token`. Each step turns its tokens into embeddings with that model, so the steps are the same as for `input_ids`.

  The task gives the features of the images to `generate/5` in `:image_features`. They are the features of the image tokens of all rows, in order, as `masked_scatter` in Python transformers uses them. A row starts at the features after those of the rows before it. In the steps of the prompt, the features replace the embeddings of the image tokens, and a cursor of each row counts the image tokens that the row used. Parts of the prompt without image tokens and the steps of new tokens only embed their tokens.

  ## Positions

  Each token of a row has the position of its index in the row. The task can give other positions for the prompt of each row in `:position_ids` of `generate/5`, such as the 3-D positions of M-RoPE in Qwen2-VL. There, the image tokens of a prompt have the positions of their rows and columns in the image. A decoder whose `position_ids` has 3 axes takes 3 positions for each token. The tokens after the prompt continue from the largest position of the prompt: the token at index `i` has the position `i + delta` on all axes. `delta` is the largest position of the prompt, plus 1, minus the length of the prompt, as the rope delta of Python transformers.

  ## Processors

  The `:processors` key of the context is the list of processors, which apply in order after the repetition penalty and before the temperature. `context/4` puts the processors of `generation_config.json` in it, and a task can add its own processors at the end. A processor is `{state, process}`. Before each step, `state` takes a map of one row and gives a tensor of the same shape at each step. The map has the token ids of the prompt in `:prompt`, the new token ids in `:generated` (the last token first), and the size of the vocabulary in `:vocab`. `process` takes the logits and that tensor of one row in the compiled selection, and gives the new logits.

  The atoms `:bad_words` and `:suppress_tokens` in the list are the places of two masks that do not change between steps: the bad words of one token and `suppress_tokens`. Generation makes these masks once for each group of rows.

  ## Log-probabilities

  With `token_logprobs: true` in the options of `context/4`, each step also gives the log-probability of the token of each row: the log-softmax of the scores that the step selects from, at the token. This is the rule of `compute_transition_scores` of Python transformers with `normalize_logits=True`, for greedy selection, sampling and beam search (see `generate/5`). The compiled selection subtracts the log-sum-exp of the scores of the row from the score of the token. Without the option, the selection does not compute it.

  The compiled selection gives one s32 output, because EMLX evaluates each output of a compiled function separately. The tokens and the indices of the candidates of beam search are integers in it, and the scores and log-probabilities are the bits of their f32 values (`Nx.bitcast/2`, see `from_bits/1`).

  ## Beam search

  With `num_beams` above 1, generation does the beam search of `_beam_search` in Python transformers 4.57. The beams of a prompt are rows of the batch: prompt `i` has the rows `i * num_beams` to `i * num_beams + num_beams - 1`, and its inputs, image features and positions repeat for each row. `compile/3` compiles the decoder for `batch_size * num_beams` rows. At the start, all beams of a prompt are equal, and only the first beam counts.

  Each step takes the log-softmax of the logits, applies the processors to it, and adds the score of the beam. Of the `num_beams * vocabulary` candidates of a prompt, the compiled selection keeps the best `max(2, 1 + eos tokens) * num_beams`. On the host:

    1. A candidate ends when its token is an eos token or when it is the last new token. The candidates that end among the first `num_beams` become finished hypotheses. The score of a hypothesis is its sum divided by its length to the power of `length_penalty`, and its length counts the eos token. A prompt keeps its best `num_beams` hypotheses.
    2. The best `num_beams` candidates that do not end are the next beams. Each row takes the cache and the tokens of the row of its beam. The cache entries that the step did not write, such as the cross-attention cache, are the same for all beams of a prompt, so they do not move.
    3. A prompt stops when it cannot improve: it has `num_beams` hypotheses, and its best beam divided by the best length to the power of `length_penalty` scores less than the worst hypothesis. The best length is the length now, or the longest length for `early_stopping: :never` with a positive `length_penalty`. With `early_stopping: true`, a prompt stops when it has `num_beams` hypotheses. A stopped prompt has padding rows until all prompts stop.

  The result of a prompt is its best hypothesis. The scores are f32 as in transformers. Removed tokens have the smallest finite value in place of an infinite one, and a score never goes below it.

  ## Batches

  `generate/4` takes a batch of prompts. Each row has its own length, sampling key and stop. A step gives the same number of tokens to each row. A row with fewer tokens gets padding tokens after its tokens, and a finished row gets only padding tokens. The next token of a row comes from the logits of its last real token. An empty prompt is a padding row, which gives no tokens.

  In `:in_place` and `:append`, all rows share one buffer. Each row writes its keys and values after its own valid rows, so the padding tokens of a step go into rows that the next step overwrites. In `:in_place`, the attention mask of a row has ones for its valid rows and the new tokens, and GroupQueryAttention gets the write position of each row from the sum of the mask. In `:append`, the attention mask hides the free rows of each row, and the new rows go into the buffer at the length of each row. Both modes give the positions of each row in `position_ids`.

  In `:grow`, the positions come from the length of the cache, which is the same for all rows. So only prompts of the same length share a cache. The prompts of each length run as one group, and the groups run one after the other. A group runs with all rows of the batch, and the rows of other lengths are padding rows of the group. So each length of the cache has one compiled shape, and a batch with prompts of three lengths runs the steps of three groups. Text-to-text, image-to-text and speech-to-text have prompts of one length, so all their rows run as one group.
  """

  alias Transmuter.{GraphBreaks, Serving}

  # Buffers have a capacity of a power of two, from this size.
  @min_capacity 256

  # max_length of GenerationConfig and max_new_tokens of the pipelines in Python transformers.
  @default_max_length 20
  @default_max_new_tokens 256

  @doc """
  Returns the context of generation for `model`: the cache mode and the settings from `generation_config.json` and `opts`. `given` are the names of other inputs that the caller gives to `generate/5`, such as the outputs of an encoder.

  A decoder that declares `inputs_embeds` needs two more options. `:embed` is the model that turns token ids into embeddings, such as `embed_tokens.onnx`. `:image_token` is the id of the image token, whose embeddings the image features replace. See the module documentation.
  """
  def context(repo, model, opts, given \\ []) do
    gen = repo.generation_config

    unless opts[:strategy] in [nil, :greedy, :sampling] do
      raise ArgumentError,
            "expected :strategy to be :greedy or :sampling, got: #{inspect(opts[:strategy])}"
    end

    max_new_tokens = opts[:max_new_tokens] || gen["max_new_tokens"] || @default_max_new_tokens
    eos = List.wrap(gen["eos_token_id"] || repo.config["eos_token_id"])
    sampling? = Keyword.get(opts, :strategy, if(gen["do_sample"], do: :sampling)) == :sampling

    %{
      model: model,
      mode: mode(model, given, opts[:embed]),
      embed: opts[:embed],
      image_token: opts[:image_token],
      cache: Enum.filter(model.inputs, &String.starts_with?(&1, "past_key_values.")),
      max_new_tokens: max_new_tokens,
      sampling?: sampling?,
      token_logprobs?: Keyword.get(opts, :token_logprobs, false),
      temperature: opts[:temperature] || gen["temperature"] || 1.0,
      top_k: opts[:top_k] || gen["top_k"] || 50,
      top_p: opts[:top_p] || gen["top_p"] || 1.0,
      repetition_penalty: opts[:repetition_penalty] || gen["repetition_penalty"] || 1.0,
      renormalize?: gen["renormalize_logits"] == true,
      masks: masks(gen),
      eos: eos,
      bos: gen["bos_token_id"] || repo.config["bos_token_id"],
      processors: processors(gen, opts, max_new_tokens, eos)
    }
    |> Map.merge(beam_search(gen, opts, sampling?))
  end

  # The settings of beam search. An option replaces the value of generation_config.json.
  defp beam_search(gen, opts, sampling?) do
    %{
      num_beams: num_beams(opts[:num_beams] || gen["num_beams"] || 1, sampling?, opts[:stream]),
      length_penalty: opts[:length_penalty] || gen["length_penalty"] || 1.0,
      early_stopping: early_stopping(Keyword.get(opts, :early_stopping, gen["early_stopping"]))
    }
  end

  @doc """
  Returns the inputs of a decoder, other than `input_ids`, for one step over `batch_size` rows of `length` tokens with an empty cache, and the static inputs of that step. A task that runs a decoder once over padded text, such as an embedding, gives these inputs with the tokens. A model without a cache has none, so the result is two empty maps.

  The cache has length 0 and `use_cache_branch` is 0, as in the first step of transformers.js. Each row has the positions from 0. The attention mask counts all tokens, also the padding tokens after the text, as in generation. The decoder is causal, so the tokens of the text do not see the padding tokens.
  """
  def prompt_inputs(model, batch_size, length) do
    case Enum.filter(model.inputs, &String.starts_with?(&1, "past_key_values.")) do
      [] ->
        {%{}, %{}}

      cache ->
        tensor = fn name, value ->
          {type, _dims} = model.input_types[name]
          Nx.tensor(value, type: type, backend: Nx.BinaryBackend)
        end

        row = Enum.to_list(0..(length - 1))

        values = %{
          "attention_mask" => List.duplicate(List.duplicate(1, length), batch_size),
          "position_ids" => List.duplicate(row, batch_size),
          "num_logits_to_keep" => length
        }

        inputs =
          for {name, value} <- values, name in model.inputs, into: %{} do
            {name, tensor.(name, value)}
          end

        inputs =
          for name <- cache, into: inputs do
            {_type, [_batch, heads, _length, head_dim]} = model.input_types[name]
            {name, {:empty, {batch_size, heads, 0, head_dim}}}
          end

        static =
          for name <- ["use_cache_branch"], name in model.inputs, into: %{} do
            {name, tensor.(name, [0])}
          end

        {inputs, static}
    end
  end

  @doc """
  Returns the outputs of a decoder that generation reads: `logits` and the present cache. `names` are the names of the graph outputs. Give this function to the `:outputs` option of `Transmuter.load!/2`, so the decoder computes no other outputs, such as the attention probabilities of the Whisper exports of Xenova.
  """
  def outputs(names),
    do: Enum.filter(names, &(&1 == "logits" or String.starts_with?(&1, "present.")))

  @doc """
  Returns the `:max_new_tokens` of a decoder whose `max_length` counts its prompt, such as the decoder of an encoder-decoder model. The rule is that of the pipelines in Python transformers 4:

    1. `:max_new_tokens` of `opts`, else `max_new_tokens` of `generation_config.json`.
    2. When `max_length` is not in the configuration or is `20`, the default of `GenerationConfig`, the value is 256.
    3. Else `max_length` minus `prompt_length`, such as 448 minus the prompt for Whisper and 511 for Marian.
  """
  def max_new_tokens(repo, opts, prompt_length) do
    gen = repo.generation_config

    cond do
      max_new_tokens = opts[:max_new_tokens] || gen["max_new_tokens"] -> max_new_tokens
      gen["max_length"] in [nil, @default_max_length] -> @default_max_new_tokens
      true -> gen["max_length"] - prompt_length
    end
  end

  defp num_beams(num_beams, _sampling?, _stream?) when not is_integer(num_beams) or num_beams < 1,
    do:
      raise(
        ArgumentError,
        "expected :num_beams to be a positive integer, got: #{inspect(num_beams)}"
      )

  defp num_beams(num_beams, true, _stream?) when num_beams > 1,
    do: raise(ArgumentError, "beam search takes strategy: :greedy, got strategy: :sampling")

  defp num_beams(num_beams, _sampling?, true) when num_beams > 1,
    do:
      raise(ArgumentError, "beam search changes earlier tokens, so it does not take stream: true")

  defp num_beams(num_beams, _sampling?, _stream?), do: num_beams

  defp early_stopping(value) when value in [nil, false], do: false
  defp early_stopping(true), do: true
  defp early_stopping(value) when value in [:never, "never"], do: :never

  defp early_stopping(value),
    do:
      raise(
        ArgumentError,
        "expected :early_stopping to be true, false or :never, got: #{inspect(value)}"
      )

  # The token ids of the masks of the processors :bad_words and :suppress_tokens.
  defp masks(gen) do
    case Enum.reject(List.wrap(gen["bad_words_ids"]), &match?([_], &1)) do
      [] ->
        :ok

      other ->
        raise ArgumentError, "generation takes bad_words_ids of one token, got: #{inspect(other)}"
    end

    %{
      bad_words: for([id] <- List.wrap(gen["bad_words_ids"]), do: id),
      suppress_tokens: List.wrap(gen["suppress_tokens"])
    }
  end

  # The processors in the order of _get_logits_processor of Python transformers. An option
  # replaces the value of generation_config.json.
  defp processors(gen, opts, max_new_tokens, eos) do
    setting = &(opts[&1] || gen[Atom.to_string(&1)])
    ngram_size = setting.(:no_repeat_ngram_size) || 0
    forced_bos = setting.(:forced_bos_token_id)
    forced_eos = setting.(:forced_eos_token_id)
    begin_suppress = List.wrap(gen["begin_suppress_tokens"])
    min_new_tokens = setting.(:min_new_tokens)
    min_length = setting.(:min_length) || 0

    # _prepare_generated_length of transformers replaces min_length with min_new_tokens plus
    # the length of the prompt.
    {min_length?, min_length} =
      if min_new_tokens,
        do: {min_new_tokens > 0, &(length(&1) + min_new_tokens)},
        else: {min_length > 0, fn _prompt -> min_length end}

    [
      ngram_size > 0 && no_repeat_ngram(ngram_size),
      :bad_words,
      min_length? && eos != [] && min_length(min_length, eos),
      forced_bos != nil && forced(&(length(&1.prompt) + length(&1.generated) == 1), [forced_bos]),
      # The row has max_length - 1 tokens, where max_length is the prompt and max_new_tokens.
      forced_eos != nil &&
        forced(&(length(&1.generated) == max_new_tokens - 1), List.wrap(forced_eos)),
      :suppress_tokens,
      begin_suppress != [] && begin_suppress(begin_suppress, forced_bos != nil)
    ]
    |> Enum.filter(& &1)
  end

  # NoRepeatNGramLogitsProcessor: the last token of each n-gram whose first n - 1 tokens are the
  # last n - 1 tokens of the row.
  defp no_repeat_ngram(size) do
    state = fn %{prompt: prompt, generated: generated, vocab: vocab} ->
      tokens = prompt ++ Enum.reverse(generated)
      last = Enum.take(tokens, 1 - size)

      banned =
        for ngram <- Enum.chunk_every(tokens, size, 1, :discard),
            Enum.take(ngram, size - 1) == last,
            uniq: true,
            do: [List.last(ngram)]

      mark(none({vocab}), banned)
    end

    {state, &remove/2}
  end

  # MinLengthLogitsProcessor, with the minimum length of the row from its prompt.
  defp min_length(min_length, eos) do
    state = fn %{prompt: prompt, generated: generated} ->
      flag(length(prompt) + length(generated) < min_length.(prompt))
    end

    {state, &remove(&1, Nx.logical_and(&2, any_of(&1, eos)))}
  end

  # ForcedBOSTokenLogitsProcessor and ForcedEOSTokenLogitsProcessor: when `at?` is true for the
  # row, the forced tokens get 0 and the other tokens are removed.
  defp forced(at?, ids) do
    process = fn logits, at ->
      forced = Nx.select(any_of(logits, ids), 0.0, Nx.Constants.min_finite(:f32))
      Nx.select(at, forced, logits)
    end

    {&flag(at?.(&1)), process}
  end

  # SuppressTokensAtBeginLogitsProcessor. Its begin_index in transformers is the length of
  # the prompt, plus 1 after a forced bos token of a prompt of one token.
  defp begin_suppress(ids, forced_bos?) do
    state = fn %{prompt: prompt, generated: generated} ->
      flag(length(generated) == if(forced_bos? and length(prompt) == 1, do: 1, else: 0))
    end

    {state, &remove(&1, Nx.logical_and(&2, any_of(&1, ids)))}
  end

  defp flag(value), do: Nx.tensor(if(value, do: 1, else: 0), type: :u8, backend: Nx.BinaryBackend)

  defp any_of(logits, ids) do
    iota = Nx.iota(Nx.shape(logits))
    ids |> Enum.map(&Nx.equal(iota, &1)) |> Enum.reduce(&Nx.logical_or/2)
  end

  defp remove(logits, mask), do: Nx.select(mask, Nx.Constants.min_finite(:f32), logits)

  @inputs ~w(input_ids inputs_embeds attention_mask position_ids num_logits_to_keep use_cache_branch)

  defp mode(model, given, embed) do
    if "inputs_embeds" in model.inputs and embed == nil do
      raise ArgumentError,
            "the decoder takes inputs_embeds, so generation needs the model of the embeddings in :embed"
    end

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

  # Whether the values of `name` depend on the values of `target`.
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

      name in seen or node == nil ->
        walk(rest, target, producers, seen)

      true ->
        walk(
          Transmuter.value_dependencies(node) ++ rest,
          target,
          producers,
          MapSet.put(seen, name)
        )
    end
  end

  @doc """
  Compiles the model and the token selection of `ctx` for batches of `batch_size` prompts. With beam search, the decoder runs `batch_size * num_beams` rows. Call this function in the serving process, which owns the cache of compiled shapes.
  """
  def compile(ctx, batch_size, defn_options) do
    ctx = Map.put(ctx, :backend, Nx.Defn.to_backend(defn_options))
    batch_size = batch_size * ctx.num_beams

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

    # A copy of the context, such as the one that a serving keeps in :persistent_term, copies
    # each reference to a term again. The compiled functions hold the nodes of the model, so
    # the context keeps only its inputs.
    ctx =
      ctx
      |> Map.put(:run, fn inputs, static -> runs[static].(inputs) end)
      |> compile_embed(batch_size, defn_options)
      |> Map.put(:model, Map.take(ctx.model, [:inputs, :input_types]))

    settings =
      ctx
      |> Map.take([
        :repetition_penalty,
        :renormalize?,
        :sampling?,
        :temperature,
        :top_k,
        :top_p,
        :token_logprobs?
      ])
      |> Map.put(:num_beams, ctx.num_beams)
      |> Map.put(:candidates, candidates(ctx))
      |> Map.put(
        :processes,
        Enum.map(ctx.processors, fn
          {_state, process} -> process
          mask -> mask
        end)
      )

    ctx
    |> Map.put(
      :select,
      GraphBreaks.jit(&select_tokens(&1, &2, &3, &4, &5, &6, &7, settings), defn_options)
    )
    |> Map.put(
      :select_beams,
      GraphBreaks.jit(&select_beams(&1, &2, &3, &4, &5, &6, settings), defn_options)
    )
    |> Map.put(:append, GraphBreaks.jit(&append_rows/3, defn_options))
    |> Map.put(:reorder, GraphBreaks.jit(&reorder_rows/2, defn_options))
  end

  # _beam_search of transformers keeps the best candidates of each prompt: num_beams for each
  # eos token and num_beams more, at least 2 * num_beams. So num_beams of the candidates do
  # not end, even when each beam gives each eos token.
  defp candidates(ctx), do: max(2, 1 + length(ctx.eos)) * ctx.num_beams

  # A prompt runs in chunks of each power of two up to 2^30, see chunks/1.
  @chunk_sizes 31

  defp compile_embed(ctx = %{embed: nil}, _batch_size, _defn_options), do: ctx

  # The embedding model compiles for the shape of each chunk size, and its output gets the
  # type of inputs_embeds of the decoder. The merge puts the image features in the place of
  # the image tokens.
  defp compile_embed(ctx, batch_size, defn_options) do
    %{embed: model = %{inputs: [input], outputs: [output]}} = ctx
    {id_type, _dims} = model.input_types[input]
    {type, _dims} = ctx.model.input_types["inputs_embeds"]

    template =
      Nx.broadcast(Nx.tensor(0, type: id_type, backend: Nx.BinaryBackend), {batch_size, 1})

    opts = [cache_size: @chunk_sizes + 1] ++ defn_options
    run = Transmuter.compile(model, %{input => template}, opts)

    embed = fn tokens ->
      ids = Nx.tensor(tokens, type: id_type, backend: Nx.BinaryBackend)
      run.(%{input => ids}) |> Map.fetch!(output) |> Nx.as_type(type)
    end

    %{ctx | embed: embed}
    |> Map.put(:merge, GraphBreaks.jit(&merge_features/4, defn_options))
  end

  defp merge_features(embeds, features, index, mask) do
    picked = features |> Nx.take(index) |> Nx.as_type(Nx.type(embeds))
    mask = Nx.broadcast(mask, Nx.shape(embeds), axes: [0, 1])
    Nx.select(mask, picked, embeds)
  end

  defp templates(model, names, batch_size) do
    Map.new(names, fn name ->
      {type, [batch | dims]} = model.input_types[name]
      zero = Nx.tensor(0, type: type, backend: Nx.BinaryBackend)

      shape =
        List.to_tuple([
          Transmuter.dim_size(batch, batch_size) | Enum.map(dims, &Transmuter.dim_size(&1, 1))
        ])

      {name, Nx.broadcast(zero, shape)}
    end)
  end

  @doc """
  Generates tokens after each prompt of `prompts` with a context from `compile/3`. A prompt is a list of token ids, and an empty prompt is a padding row. `seeds` has the seed of sampling of each row. `inputs` are the inputs of each step that `context/4` names in `given`, with one row for each prompt.

  `inputs` can also have the key `:image_features`, a tensor of shape `{count, hidden}` with the features of the image tokens of all prompts: the features of the first prompt, in the order of its image tokens, then those of the next prompt. It can have more rows than the prompts have image tokens. It is not an input of the decoder. See the module documentation.

  `inputs` can also have the key `:position_ids`, a list with the positions of each prompt. The positions of a prompt are a list with the positions of each axis of the decoder's `position_ids`, one for each token. An empty list gives the default positions. See the module documentation.

  `on_token` is `nil` or a function that receives `%{token: tensor, finished?: tensor}` after each step of a group, with one row for each prompt. `:token` is the new token of the row, or -1 when the row has no new token. `:finished?` is 1 when the row has no more tokens. With `token_logprobs: true` in the options of `context/4`, the map also has `:logprob`, the log-probability of the new token, or NaN when the row has no new token.

  Returns the new tokens of each row, without the eos token. A token is the list of its values: `[id]`, or `[id, logprob]` with `token_logprobs: true` in the options of `context/4`. The log-probability of a token is the log-softmax of the scores that the step selects from, at the token. These scores are the logits after the processors, also after the temperature, top-k and top-p of sampling. For beam search, they are the scores of the step before the sum with the score of the beam. This is `compute_transition_scores` of Python transformers with `normalize_logits=True`.
  """
  def generate(ctx, prompts, seeds, inputs \\ %{}, on_token \\ nil) do
    {features, inputs} = Map.pop(inputs, :image_features)

    {positions, inputs} =
      Map.pop_lazy(inputs, :position_ids, fn -> Enum.map(prompts, fn _ -> [] end) end)

    # The features of each row start after those of the rows before it.
    {cursors, _count} =
      Enum.map_reduce(prompts, 0, &{&2, &2 + Enum.count(&1, fn id -> id == ctx.image_token end)})

    extras = %{
      features: features,
      cursors: cursors,
      positions: Enum.map(positions, &{&1, delta(&1)})
    }

    ids =
      for prompts <- groups(ctx, prompts) do
        if ctx.num_beams == 1 do
          notify = &notify(on_token, prompts, &1, ctx.token_logprobs?)

          group =
            decode(ctx, start(ctx, prompts, seeds, inputs, extras), ctx.max_new_tokens, notify)

          Enum.zip_with(group.ids, group.logprobs, &tokens/2)
        else
          search(ctx, prompts, seeds, inputs, extras)
        end
      end

    Enum.zip_with([Enum.map(prompts, fn _ -> [] end) | ids], &Enum.concat/1)
  end

  # The positions after the prompt continue from its largest position. Without given
  # positions, the delta is 0.
  defp delta(axes),
    do: Enum.max(List.flatten(axes), fn -> -1 end) + 1 - length(List.first(axes, []))

  # The prompts of each group of rows that share a cache, with empty prompts for the other
  # rows. A buffer keeps the length of each row, so all prompts are one group.
  defp groups(ctx, prompts) do
    key = if ctx.mode == :grow, do: &length/1, else: fn _prompt -> :buffer end

    for group <- prompts |> Enum.reject(&(&1 == [])) |> Enum.map(key) |> Enum.uniq() do
      Enum.map(prompts, &if(&1 != [] and key.(&1) == group, do: &1, else: []))
    end
  end

  # Runs the prompts in parts of power-of-two lengths. Each prompt fills the parts from the
  # last one, so the last token of each row is in the last part. Only the prompt takes image
  # features, and the cursors start again for each group.
  defp start(ctx, prompts, seeds, inputs, extras) do
    longest = prompts |> Enum.map(&length/1) |> Enum.max()
    state = initial_state(ctx, length(prompts), capacity(longest + ctx.max_new_tokens))
    sizes = chunks(longest)

    state =
      prompts
      |> Enum.map(&split_from_end(&1, sizes))
      |> Enum.zip_with(& &1)
      |> Enum.reduce(Map.merge(state, Map.put(extras, :inputs, inputs)), &step(ctx, &2, &1))

    state = %{state | features: nil}

    {rows, _n, vocab} = Nx.shape(state.logits)

    masks =
      Map.new(ctx.masks, fn {name, ids} ->
        mask = mark(none({vocab}), Enum.map(ids, &[&1]))
        {name, Nx.backend_transfer(mask, ctx.backend)}
      end)

    # The tokens that the repetition penalty applies to: the prompt and the new tokens.
    prompt_ids = for {prompt, i} <- Enum.with_index(prompts), id <- prompt, do: [i, id]
    seen = mark(none({rows, vocab}), prompt_ids)

    %{
      state: state,
      seen: seen,
      masks: masks,
      keys: seeds |> Enum.map(&Nx.Random.key/1) |> Nx.stack() |> Nx.backend_transfer(ctx.backend),
      prompts: prompts,
      ids: Enum.map(prompts, fn _ -> [] end),
      logprobs: Enum.map(prompts, fn _ -> [] end),
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
    %{logits: logits, index: index} = group.state
    states = processor_states(ctx, group, Nx.axis_size(logits, -1))
    step = Nx.tensor(ctx.max_new_tokens - remaining, type: :u32, backend: Nx.BinaryBackend)
    selected = ctx.select.(logits, index, group.seen, group.masks, states, group.keys, step)

    {new, logprobs} =
      selected
      |> Nx.to_list()
      |> Enum.zip_with(group.done, fn [token | logprob], done? ->
        if done? or token in ctx.eos, do: {nil, nil}, else: {token, from_bits(logprob)}
      end)
      |> Enum.unzip()

    marks = for {token, i} <- Enum.with_index(new), token, do: [i, token]

    %{
      group
      | seen: mark(group.seen, marks),
        new: new,
        done: Enum.map(new, &(&1 == nil or remaining == 1)),
        ids: push(group.ids, new, new),
        logprobs: push(group.logprobs, new, logprobs)
    }
  end

  # Beam search as _beam_search of Python transformers 4.57. The rows of prompt `i` are
  # `i * num_beams` to `i * num_beams + num_beams - 1`, and the inputs, the image features and
  # the positions of the prompt repeat for each of them. Each step selects the best
  # candidates of each prompt on the device, and the host keeps the beams, the finished
  # hypotheses and their scores. Returns the tokens of the best hypothesis of each prompt.
  defp search(ctx, prompts, seeds, inputs, extras) do
    beams = ctx.num_beams
    repeat = &Enum.flat_map(&1, fn value -> List.duplicate(value, beams) end)
    index = Nx.tensor(repeat.(Enum.to_list(0..(length(prompts) - 1))), backend: Nx.BinaryBackend)
    inputs = Map.new(inputs, fn {name, tensor} -> {name, Nx.take(tensor, index)} end)
    extras = %{extras | cursors: repeat.(extras.cursors), positions: repeat.(extras.positions)}
    group = start(ctx, repeat.(prompts), repeat.(seeds), inputs, extras)

    # The first beam of each prompt starts at 0 and the others at -1e9, so the first step
    # selects only from the first beam.
    scores = Enum.flat_map(prompts, fn _ -> [0.0 | List.duplicate(-1.0e9, beams - 1)] end)
    items = Enum.map(prompts, &%{finished: [], open?: true, done?: &1 == []})

    %{items: items} = search_steps(ctx, group, items, scores, 0)

    for item <- items do
      case item.finished do
        [] -> []
        [{_score, ids, logprobs} | _] -> tokens(ids, logprobs)
      end
    end
  end

  defp search_steps(ctx, group, items, scores, step) do
    %{logits: logits, index: index} = group.state
    vocab = Nx.axis_size(logits, -1)
    states = processor_states(ctx, group, vocab)
    scores_tensor = Nx.tensor(scores, type: :f32, backend: Nx.BinaryBackend)

    candidates =
      ctx.select_beams.(logits, index, group.seen, group.masks, states, scores_tensor)
      |> Nx.to_list()

    {items, beams} =
      items
      |> Enum.zip(candidates)
      |> Enum.with_index()
      |> Enum.map(fn {{item, columns}, i} ->
        rows = Enum.to_list((i * ctx.num_beams)..(i * ctx.num_beams + ctx.num_beams - 1))

        if item.done? do
          {item, Enum.map(rows, &padding_beam/1)}
        else
          candidates =
            columns
            |> Enum.zip()
            |> Enum.map(fn candidate ->
              [score, flat | logprob] = Tuple.to_list(candidate)
              row = hd(rows) + div(flat, vocab)
              token = rem(flat, vocab)
              %{score: from_bits(score), row: row, token: token, logprob: from_bits(logprob)}
            end)

          beam_step(ctx, item, rows, candidates, group, step)
        end
      end)
      |> Enum.unzip()

    if Enum.all?(items, & &1.done?) do
      %{items: items}
    else
      beams = Enum.concat(beams)
      group = reorder(ctx, group, Enum.map(beams, & &1.row))
      new = Enum.map(beams, & &1.token)
      marks = for {token, i} <- Enum.with_index(new), token, do: [i, token]

      group = %{
        group
        | seen: mark(group.seen, marks),
          ids: push(group.ids, new, new),
          logprobs: push(group.logprobs, new, Enum.map(beams, & &1.logprob))
      }

      group = %{group | state: step(ctx, group.state, Enum.map(new, &List.wrap/1))}
      search_steps(ctx, group, items, Enum.map(beams, & &1.score), step + 1)
    end
  end

  # One step of a prompt with its candidates, best first. The best candidates that do not end
  # are the next beams. The candidates that end among the first num_beams are finished
  # hypotheses, scored by their sum of log-probabilities over their length (with the eos
  # token) to the power of length_penalty. The prompt keeps its best num_beams hypotheses.
  # A hypothesis keeps the log-probability of each token, without the eos token. Returns the
  # item and, for each row of the prompt, its candidate: its source row, its token, its
  # log-probability and its score.
  defp beam_step(ctx, item, rows, candidates, group, step) do
    %{num_beams: beams, length_penalty: penalty, early_stopping: early_stopping} = ctx
    last? = step + 1 == ctx.max_new_tokens
    candidates = Enum.map(candidates, &Map.put(&1, :end?, last? or &1.token in ctx.eos))

    added =
      if item.open? and not (early_stopping == true and length(item.finished) == beams) do
        for candidate <- Enum.take(candidates, beams), candidate.end? do
          [ids, logprobs] = Enum.map([group.ids, group.logprobs], &Enum.at(&1, candidate.row))

          {ids, logprobs} =
            if candidate.token in ctx.eos,
              do: {ids, logprobs},
              else: {[candidate.token | ids], [candidate.logprob | logprobs]}

          {f32(candidate.score / f32((step + 1) ** penalty)), ids, logprobs}
        end
      else
        []
      end

    finished = (item.finished ++ added) |> Enum.sort_by(&elem(&1, 0), :desc) |> Enum.take(beams)
    running = candidates |> Enum.reject(& &1.end?) |> Enum.take(beams)

    # _check_early_stop_heuristic: the prompt can improve while it has fewer than num_beams
    # finished hypotheses, or while its best beam at the best length scores more than the
    # worst finished hypothesis. The best length is the longest one for "never" with a
    # positive length_penalty, else the length now.
    best_length =
      if early_stopping == :never and penalty > 0, do: ctx.max_new_tokens, else: step + 1

    open? =
      item.open? and not last? and
        (length(finished) < beams or
           f32(hd(running).score / f32(best_length ** penalty)) >
             finished |> Enum.map(&elem(&1, 0)) |> Enum.min())

    done? = not open? or (early_stopping == true and length(finished) == beams)
    item = %{item | finished: finished, open?: open?, done?: done?}

    if done?,
      do: {item, Enum.map(rows, &padding_beam/1)},
      else: {item, running}
  end

  # The beam of a row of a prompt that is done: the row keeps its cache and gets no token.
  defp padding_beam(row), do: %{row: row, token: nil, logprob: [], score: 0.0}

  # The tokens of a row from its ids and their log-probabilities, the last first. The
  # log-probability of a token is a list: empty without token_logprobs.
  defp tokens(ids, logprobs),
    do: Enum.zip_with(Enum.reverse(ids), Enum.reverse(logprobs), &[&1 | &2])

  # Puts the value of each row with a new token before the other values of the row.
  defp push(lists, tokens, values) do
    Enum.zip_with([lists, tokens, values], fn
      [list, nil, _value] -> list
      [list, _token, value] -> [value | list]
    end)
  end

  @doc """
  Returns the f32 value of `bits`, the signed 32-bit integer of its bits, as `Nx.to_list/1` gives an f32: a float, `:nan`, `:infinity` or `:neg_infinity`. A list gives the value of each element. The compiled selection gives its scores and log-probabilities in this form (see the module documentation).
  """
  def from_bits(bits) when is_list(bits), do: Enum.map(bits, &from_bits/1)

  def from_bits(bits) do
    case <<bits::signed-32>> do
      <<value::float-32>> -> value
      <<0::1, 0xFF::8, 0::23>> -> :infinity
      <<1::1, 0xFF::8, 0::23>> -> :neg_infinity
      _nan -> :nan
    end
  end

  @doc """
  Returns the signed 32-bit integer of the bits of `value` in f32, the inverse of `from_bits/1`. NaN is the quiet NaN with the sign bit 0.
  """
  def to_bits(:nan), do: 0x7FC00000
  def to_bits(:infinity), do: 0x7F800000
  def to_bits(:neg_infinity), do: -0x800000

  def to_bits(value) do
    <<bits::signed-32>> = <<value::float-32>>
    bits
  end

  # The state of each processor for the rows of the group.
  defp processor_states(ctx, group, vocab) do
    rows = Enum.zip_with(group.prompts, group.ids, &%{prompt: &1, generated: &2, vocab: vocab})

    states =
      for {state, _process} <- ctx.processors do
        rows |> Enum.map(state) |> Nx.stack()
      end

    List.to_tuple(states)
  end

  # Gives each row the cache, the tokens and the other state of its source row. The cache
  # entries that the last step did not write, such as the cross-attention cache, are the same
  # for all beams of a prompt, so they keep their rows.
  defp reorder(ctx, group, sources) do
    if sources == Enum.to_list(0..(length(sources) - 1)) do
      group
    else
      take = fn list -> Enum.map(sources, &Enum.at(list, &1)) end
      %{state: state} = group
      index = Nx.tensor(sources, type: :s32, backend: Nx.BinaryBackend)
      cache = Map.merge(state.cache, ctx.reorder.(Map.take(state.cache, state.written), index))

      state = %{
        state
        | cache: cache,
          cursors: take.(state.cursors),
          positions: take.(state.positions),
          lengths: take.(state.lengths)
      }

      %{
        group
        | state: state,
          prompts: take.(group.prompts),
          ids: take.(group.ids),
          logprobs: take.(group.logprobs),
          seen: take_rows(group.seen, sources)
      }
    end
  end

  defp reorder_rows(cache, index),
    do: Map.new(cache, fn {name, tensor} -> {name, Nx.take(tensor, index)} end)

  # The rows of a mask of tokens on the host, from its binary.
  defp take_rows(tensor, sources) do
    {_rows, width} = Nx.shape(tensor)
    binary = Nx.to_binary(tensor)

    sources
    |> Enum.map(&binary_part(binary, &1 * width, width))
    |> IO.iodata_to_binary()
    |> Nx.from_binary(:u8, backend: Nx.BinaryBackend)
    |> Nx.reshape(Nx.shape(tensor))
  end

  @min_f32 Nx.to_number(Nx.Constants.min_finite(:f32, backend: Nx.BinaryBackend))

  # The scores of transformers are f32, so each division rounds to f32 as torch does, and a
  # score below the f32 range stays at its lowest value.
  defp f32(value) do
    <<rounded::float-32>> = <<max(value, @min_f32)::float-32>>
    rounded
  end

  # The rows of other groups have no token and are not finished. The log-probability of a new
  # token is the first one of its row.
  defp notify(nil, _prompts, _group, _logprobs?), do: :ok

  defp notify(on_token, prompts, group, logprobs?) do
    {tokens, finished} =
      Enum.zip_with([prompts, group.new, group.done], fn
        [[], _token, _done?] -> {-1, 0}
        [_prompt, token, done?] -> {token || -1, if(done?, do: 1, else: 0)}
      end)
      |> Enum.unzip()

    event = %{
      token: Nx.tensor(tokens, type: :s64, backend: Nx.BinaryBackend),
      finished?: Nx.tensor(finished, type: :u8, backend: Nx.BinaryBackend)
    }

    event =
      if logprobs? do
        logprobs =
          Enum.zip_with(group.new, group.logprobs, fn
            nil, _logprobs -> :nan
            _token, [[logprob] | _] -> logprob
          end)

        Map.put(event, :logprob, Nx.tensor(logprobs, type: :f32, backend: Nx.BinaryBackend))
      else
        event
      end

    on_token.(event)
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

    %{
      cache: cache,
      written: [],
      inputs: %{},
      features: nil,
      cursors: List.duplicate(0, batch_size),
      positions: List.duplicate({[], 0}, batch_size),
      lengths: List.duplicate(0, batch_size),
      capacity: capacity,
      logits: nil,
      index: nil
    }
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
    {inputs, state} = embed(ctx, state, inputs, tokens)
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
        written: Map.keys(present),
        lengths: lengths,
        logits: logits,
        index: index
    }
  end

  # A decoder with inputs_embeds gets the embeddings of the tokens. In a part of the prompt
  # with image tokens, the features of the images replace their embeddings. The cursor of
  # a row is the index of the features of its next image token.
  defp embed(%{embed: nil}, state, inputs, _tokens), do: {inputs, state}

  defp embed(ctx, state, inputs, tokens) do
    embeds = ctx.embed.(tokens)

    {rows, cursors} =
      Enum.zip_with(tokens, state.cursors, fn row, cursor ->
        Enum.map_reduce(row, cursor, fn
          id, cursor when id == ctx.image_token -> {{cursor, 1}, cursor + 1}
          _id, cursor -> {{0, 0}, cursor}
        end)
      end)
      |> Enum.unzip()

    embeds =
      if state.features == nil or cursors == state.cursors do
        embeds
      else
        index = for row <- rows, do: Enum.map(row, &elem(&1, 0))
        mask = for row <- rows, do: Enum.map(row, &elem(&1, 1))
        index = Nx.tensor(index, type: :s32, backend: Nx.BinaryBackend)
        mask = Nx.tensor(mask, type: :u8, backend: Nx.BinaryBackend)
        ctx.merge.(embeds, state.features, index, mask)
      end

    {Map.put(inputs, "inputs_embeds", embeds), %{state | cursors: cursors}}
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
      "position_ids" => positions(ctx, state, n),
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

  # The positions of the tokens at the indices past..past + n - 1 of each row: the given
  # positions of the prompt, and after them the index plus the delta of the row. A decoder
  # without position_ids gets 2-D positions, which step_inputs leaves out.
  defp positions(ctx, state, n) do
    {_type, dims} = Map.get(ctx.model.input_types, "position_ids", {:s64, [nil, nil]})
    count = if length(dims) == 3, do: 3, else: 1

    rows =
      Enum.zip_with(state.lengths, state.positions, fn past, {given, delta} ->
        for axis <- 0..(count - 1) do
          prompt = given |> Enum.at(axis, []) |> Enum.slice(past, n)
          prompt ++ Enum.to_list((past + length(prompt) + delta)..(past + n - 1 + delta)//1)
        end
      end)

    if count == 3, do: Enum.zip_with(rows, & &1), else: Enum.map(rows, &hd/1)
  end

  # Selects the token of each row from its logits at `index`. The key of a step is the key
  # of the row folded with the number of the step. The serving compiles this function,
  # because eager random numbers over a large vocabulary are slow. EMLX evaluates each
  # output of a compiled function separately, so the function gives one s32 output: the token
  # of each row, and with token_logprobs the bits of its log-probability.
  defp select_tokens(logits, index, seen, masks, states, keys, step, settings) do
    [logits, index, seen, keys | states] =
      Enum.map([logits, index, seen, keys | Tuple.to_list(states)], &Nx.vectorize(&1, :rows))

    logits = Nx.take(logits, index)
    key = Transmuter.Random.fold_in(keys, step)
    {scores, token} = select_token(logits, seen, masks, states, key, settings)
    token = Nx.as_type(token, :s32)

    values =
      if settings.token_logprobs?,
        do: [token, bits(Nx.subtract(Nx.take(scores, token), Nx.logsumexp(scores)))],
        else: [token]

    values |> Nx.stack() |> Nx.devectorize(keep_names: false)
  end

  defp bits(tensor), do: Nx.bitcast(tensor, :s32)

  # The logits processors of Python transformers: repetition penalty, the processors of the
  # context with the masks in their places, temperature, top-k, top-p. Removed tokens get the
  # smallest finite value, because EMLX 0.5 does not compile infinite constants. Returns the
  # scores that the token comes from and the token.
  defp select_token(logits, seen, masks, states, key, settings) do
    logits = logits |> Nx.as_type(:f32) |> process(seen, masks, states, settings)

    if settings.sampling? do
      scores =
        logits
        |> Nx.divide(settings.temperature)
        |> top_k(settings.top_k)
        |> top_p(settings.top_p)

      {scores, sample(scores, key)}
    else
      {logits, Nx.argmax(logits)}
    end
  end

  # The processors before the temperature: the repetition penalty, the processors of the
  # context with the masks in their places, and the log-softmax of renormalize_logits, which
  # changes the scores of beam search but not the token of greedy or sampling selection.
  defp process(logits, seen, masks, states, settings) do
    penalty = settings.repetition_penalty

    penalized =
      Nx.select(Nx.greater(logits, 0), Nx.divide(logits, penalty), Nx.multiply(logits, penalty))

    logits = Nx.select(seen, penalized, logits)

    {logits, []} =
      Enum.reduce(settings.processes, {logits, states}, fn
        mask, {logits, states} when is_atom(mask) -> {remove(logits, masks[mask]), states}
        process, {logits, [state | states]} -> {process.(logits, state), states}
      end)

    if settings.renormalize?, do: log_softmax(logits), else: logits
  end

  defp log_softmax(logits), do: Nx.subtract(logits, Nx.logsumexp(logits))

  # The candidates of beam search for each prompt, as s32 {prompts, columns, candidates}: the
  # bits of their scores, their indices in the scores of all beams of the prompt
  # (beam * vocabulary + token) and with token_logprobs the bits of their log-probabilities.
  # The score of a candidate is the log-softmax of the logits after the processors, plus the
  # score of its beam, as in _beam_search of transformers. Removed tokens of beams at -1e9
  # stay at the smallest finite value, so the sum is never infinite.
  defp select_beams(logits, index, seen, masks, states, scores, settings) do
    [logits, index, seen, scores | states] =
      Enum.map([logits, index, seen, scores | Tuple.to_list(states)], &Nx.vectorize(&1, :rows))

    step_scores =
      logits
      |> Nx.take(index)
      |> Nx.as_type(:f32)
      |> log_softmax()
      |> process(seen, masks, states, settings)

    sums = step_scores |> Nx.add(scores) |> Nx.max(Nx.Constants.min_finite(:f32))
    sums = Nx.devectorize(sums, keep_names: false)
    {rows, vocab} = Nx.shape(sums)
    shape = {div(rows, settings.num_beams), settings.num_beams * vocab}
    {values, indices} = Nx.top_k(Nx.reshape(sums, shape), k: settings.candidates)
    indices = Nx.as_type(indices, :s32)

    columns =
      if settings.token_logprobs? do
        # The log-softmax of the step at a candidate is its step score minus the log-sum-exp
        # of the step scores of its beam.
        totals = step_scores |> Nx.logsumexp() |> Nx.devectorize(keep_names: false)
        step_scores = step_scores |> Nx.devectorize(keep_names: false) |> Nx.reshape(shape)
        totals = Nx.reshape(totals, {elem(shape, 0), settings.num_beams})
        beams = Nx.quotient(indices, vocab)

        logprobs =
          Nx.subtract(
            Nx.take_along_axis(step_scores, indices, axis: 1),
            Nx.take_along_axis(totals, beams, axis: 1)
          )

        [bits(values), indices, bits(logprobs)]
      else
        [bits(values), indices]
      end

    Nx.stack(columns, axis: 1)
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
    uniform = Transmuter.Random.uniform(key, 1.0e-10, 1.0, shape: logits.shape, type: :f32)
    gumbel = uniform |> Nx.log() |> Nx.negate() |> Nx.log() |> Nx.negate()
    logits |> Nx.add(gumbel) |> Nx.argmax()
  end
end
