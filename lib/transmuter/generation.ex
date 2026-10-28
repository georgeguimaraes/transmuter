defmodule Transmuter.Generation do
  @moduledoc """
  The generation loop of a decoder with a KV cache. `Transmuter.Text.generation/2`, `Transmuter.Text.text_to_text/2`, `Transmuter.Vision.image_to_text/2`, `Transmuter.Vision.image_text_to_text/2` and `Transmuter.Audio.speech_to_text/2` use it.

  Generation runs the model in steps. A step takes new tokens and the KV cache, and gives the logits and the next cache. The prompt runs in steps of power-of-two lengths (45 tokens run as 32, 8, 4 and 1), and each new token is a step of length 1, so a small number of compiled shapes covers all prompts.

  The cache works in one of three modes, from the structure of the graph:

    * `:in_place` - the graph has GroupQueryAttention, which counts the valid rows of the cache from the values of the attention mask. Generation gives a GroupQueryAttention that counts them from the length of the mask, as in Gemma 3, the sum of the mask in place of that count, when no other node reads the cache or its length, or when `Transmuter.Generation.MergedDecoder` gives those nodes the position offset, as in Apertus (see "Merged decoders"). The cache is a buffer as long as the attention mask, and the operator writes the new keys and values after the valid rows.

    * `:append` - the graph declares `position_ids`, or it is a merged decoder whose positions are reads of a table at the length of the cache plus the index of the token, as in GPT-2, Whisper, Marian and TrOCR (see "Merged decoders"). The cache is a buffer whose free rows the attention mask hides. The present cache is the buffer and the new rows after it, and the new rows go into the buffer.

    * `:grow` - the graph computes the positions from the length of the cache in a way that `Transmuter.Generation.MergedDecoder` does not cover, such as the rotary table of the merged llama2.c, or a node other than GroupQueryAttention reads the cache or its length, as in Gemma 4. The cache has its exact length, so each length is a new shape that compiles on its first step.

  ## Merged decoders

  A merged decoder of optimum has a branch without cache and a branch with cache (`use_cache_branch`). The first step takes the branch without cache, because Nx has no tensors of length 0. The exports trace the branch with cache for one new token, and some of them, such as Whisper and Marian, have no causal mask in it, so a step of several tokens there gives wrong logits (ONNX Runtime gives the same). So the prompt of a merged decoder runs as one step on the branch without cache, and only the new tokens take the branch with cache. The prompts of a batch run as one step for each prompt length, from the longest to the shortest, and each row takes its first token from the step of its prompt.

  A merged decoder without `position_ids` takes the length of its cache from the shape of the past, so with a buffer it sees the capacity of the buffer and not the valid rows. `Transmuter.Generation.MergedDecoder` gives the branch with cache the valid rows where the positions need them, and a mask of the free rows, so such a decoder runs in `:append`. Other merged decoders stay in `:grow`. A decoder without an `If`, such as Apertus, whose rotary positions come from the length of its past, gets the same rewrite on its graph. Its GroupQueryAttention hides its own free rows, so it takes only the offset and runs in `:in_place`, and the rows of a group have prompts of one length. When the valid rows have one value for each row, as in GPT-2, the rows of prompts of different lengths run together, and the steps of the other prompt lengths leave gap rows in each row. A group takes the next prompt length while the sum of its prompt lengths and the new tokens fits the static tables of the decoder, else the length starts a new group. When they have one value for all rows, as in M2M100 and Whisper, the prompts of each length run as one group, as in `:grow` (see "Batches").

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

  The `:processors` key of the context is the list of processors, which apply in order after the repetition penalty and before the temperature. `context/4` puts the processors of `generation_config.json` in it, and a task can add its own processors at the end. A processor is a function of the logits of one row and a map of that row, and gives the new logits. It runs in the compiled selection, and the map has device tensors: the token ids of the row in `:tokens` (the prompt and the new tokens, then free places up to the capacity), the number of these tokens in `:count`, and the length of the prompt in `:prompt`. So the processors read the tokens on the device, and the host builds no state for them at each step.

  The atoms `:bad_words` and `:suppress_tokens` in the list are the places of two masks that do not change between steps: the bad words of one token and `suppress_tokens`. Generation makes these masks once for each group of rows.

  ## Log-probabilities

  With `token_logprobs: true` in the options of `context/4`, each step also gives the log-probability of the token of each row: the log-softmax of the scores that the step selects from, at the token. This is the rule of `compute_transition_scores` of Python transformers with `normalize_logits=True`, for greedy selection, sampling and beam search (see `generate/5`). The compiled selection subtracts the log-sum-exp of the scores of the row from the score of the token. Without the option, the selection does not compute it.

  The selection runs in the compiled program of each step, after the decoder (the `:tail` of `Transmuter.compile/3`), so a step is one call and the logits stay on the device. The vocabulary size comes from the last dimension of the `logits` output of the graph, or from `vocab_size` of the configuration when that dimension is symbolic. Each part of the prompt also gives a selection, and generation reads the one of the last part. The selection gives one s32 output, because EMLX evaluates each output of a compiled function separately. The tokens and the indices of the candidates of beam search are integers in it, and the scores and log-probabilities are the bits of their f32 values (`Nx.bitcast/2`, see `from_bits/1`).

  ## Beam search

  With `num_beams` above 1, generation does the beam search of `_beam_search` in Python transformers 4.57. The beams of a prompt are rows of the batch: prompt `i` has the rows `i * num_beams` to `i * num_beams + num_beams - 1`, and its inputs, image features and positions repeat for each row. `compile/3` compiles the decoder for `batch_size * num_beams` rows. At the start, all beams of a prompt are equal, and only the first beam counts.

  Each step takes the log-softmax of the logits, applies the processors to it, and adds the score of the beam. Of the `num_beams * vocabulary` candidates of a prompt, the compiled selection keeps the best `max(2, 1 + eos tokens) * num_beams`. On the host:

    1. A candidate ends when its token is an eos token or when it is the last new token. The candidates that end among the first `num_beams` become finished hypotheses. The score of a hypothesis is its sum divided by its length to the power of `length_penalty`, and its length counts the eos token. A prompt keeps its best `num_beams` hypotheses.
    2. The best `num_beams` candidates that do not end are the next beams. Each row takes the cache and the tokens of the row of its beam. The cache entries that the step did not write, such as the cross-attention cache, are the same for all beams of a prompt, so they do not move.
    3. A prompt stops when it cannot improve: it has `num_beams` hypotheses, and its best beam divided by the best length to the power of `length_penalty` scores less than the worst hypothesis. The best length is the length now, or the longest length for `early_stopping: :never` with a positive `length_penalty`. With `early_stopping: true`, a prompt stops when it has `num_beams` hypotheses. A stopped prompt has padding rows until all prompts stop.

  The result of a prompt is its best hypothesis. The scores are f32 as in transformers. Removed tokens have the smallest finite value in place of an infinite one, and a score never goes below it.

  ## Batches

  `generate/4` takes a batch of prompts. Each row has its own length, sampling key and stop. A step gives the same number of tokens to each row. A row with fewer tokens gets padding tokens after its tokens, and a finished row gets only padding tokens. The next token of a row comes from the logits of its last real token. An empty prompt is a padding row, which gives no tokens.

  In `:in_place` and `:append`, all rows share one buffer. In `:in_place`, each row writes its keys and values after its own valid rows, so the padding tokens of a step go into rows that the next step overwrites. The attention mask of a row has ones for its valid rows and the new tokens, and GroupQueryAttention gets the write position of each row from the sum of the mask. In `:append`, all rows of a step write their new rows at one offset: the sum of the lengths of the earlier steps. So one update of each buffer writes the step, in place of one update for each row. A row with fewer tokens than the step, such as a shorter part of a prompt or a finished row, leaves gap rows after its tokens. The host keeps the valid rows of each row, and the attention mask has ones only for them and for the new tokens, so it hides the gap rows and the free rows. Both modes give the positions of each row in `position_ids`, from the tokens of the row and not from its rows in the buffer.

  The buffer starts with a capacity that fits the rows of the prompt steps and 16 more rows: the smallest capacity, or above it a multiple of a step. The step is a quarter of the power of two below the rows, at least 64 rows: 64 rows up to 512, 128 rows up to 1024, 256 rows up to 2048. These rows are the longest prompt, or with a merged decoder the sum of the prompt lengths of the group. When a step does not fit, the buffers grow to the next capacity, and the steps compile for it. Each step reads the whole buffers, so the small steps keep them near the rows that the longest row needs: a row of 300 rows runs its steps at 320 rows, not at 512. The step grows with the rows, so a long answer compiles four capacities for each doubling of its rows: the first answer of 1024 tokens compiles 12 shapes in place of 6 with powers of two, about 13 s more on CUDA. A process compiles each capacity one time, so later answers do not pay it. The smallest capacity is 256 rows, so the first request compiles the steps one time for most answers, and an answer of 600 tokens to a short prompt compiles them at 256, 320, 384, 448, 512 and 640 rows. A decoder with the position offset of `Transmuter.Generation.MergedDecoder` starts at 32 rows. In `:grow`, such a decoder compiles a shape for each length of its cache, so a few compiles of capacities cost little against that, and its steps are faster with a smaller buffer, as in Whisper and in Marian with beam search. The capacity of a merged decoder stops at the rows of its static tables. In `:append`, the rows of all steps of a batch count, also after a row ends, so a batch whose longest row ends early can need more rows than its longest row, grow one step earlier, or pass the rows of the static tables.

  In `:grow`, the positions come from the length of the cache, which is the same for all rows. So only prompts of the same length share a cache. The prompts of each length run as one group, and the groups run one after the other. A group runs with all rows of the batch, and the rows of other lengths are padding rows of the group. So each length of the cache has one compiled shape, and a batch with prompts of three lengths runs the steps of three groups. Text-to-text, image-to-text and speech-to-text have prompts of one length, so all their rows run as one group. A merged decoder in `:append` with one position offset for all rows also runs a group for each prompt length (see "Merged decoders").

  ## Logits

  Generation reads the logits of one position of each row, at its last token. The last part of the prompt is no longer than the shortest prompt of the batch, so the last token of each row is at the last position of each step. A merged decoder runs the whole prompt of a row in the step of its length, so the prompt ends at the last position of that step. A decoder that declares `num_logits_to_keep`, such as Gemma 3, gets the static input 1, so the lm_head computes the logits of one position. `context/4` adds this input to a decoder that does not declare it, with the nodes of the Gemma 3 exports: a Slice keeps the last positions of the input of the lm_head, the MatMul or MatMulNBits that gives `logits`. Between the lm_head and `logits`, the graph can have only operators that act on each position alone (Cast, Identity, Tanh, and Add, Sub, Mul or Div with one input that is not static) and `If` nodes, such as the `use_cache_branch` of a merged decoder. Each branch of an `If` gets its own Slice before its lm_head, so the `If` gives the same shape from each branch. Each value on the way to `logits` has no other use. At another operator, such as a Reshape or the dynamic quantization of an lm_head, the graph stays as it is, and the decoder computes the logits of all positions. `Transmuter.load!/2` does not change the graph.

  ## Device memory

  A step makes large device tensors that die at the next step: the selection of the step, the present cache, and in `:append` the old buffers. The garbage collector of the BEAM does not see the size of a device buffer, so it frees them late, and a request of 24 tokens from a prompt with an image kept 13 GB of dead buffers on the EXLA CPU. So each step frees the tensors that it owns with `Nx.backend_deallocate/1`: the selection of the last step after it is read, the present after it is in the cache, and the old buffers of the cache and of the reorder of beam search. A compiler with donation, such as EXLA, writes the present of `:in_place` and `:append` into the buffers of the step before, and the free of a donated buffer does nothing. A present that shares a buffer with a kept tensor stays, as when a merged decoder gives a past back. The embeddings of a step are freed after its call, and when a group ends, generation frees its cache and the rest of its state. Generation never frees the tensors of the caller, such as the outputs of an encoder or the image features.

  A backend that frees a released buffer only when the BEAM collects its tensor, such as EMLX 0.5 (see elixir-nx/emlx#159), still keeps the dead caches of the steps until a collection. A loop with little host garbage collects rarely, so these caches pile up. So the generation process collects its garbage after it releases 256 MB of device buffers. A small cache then rarely costs a collection, and a large one gets one before its dead copies take much memory.
  """

  alias Transmuter.Generation.MergedDecoder
  alias Transmuter.{GraphBreaks, Serving}
  alias Transmuter.Onnx.{AttributeProto, NodeProto}

  require Logger

  # The names of the extra inputs of the selection and of its output. The prefix keeps them
  # apart from the names of the graph.
  @offset "transmuter.select.offset"
  @seen "transmuter.select.seen"
  @mask "transmuter.select.mask."
  @history "transmuter.select.history"
  @count "transmuter.select.count"
  @prompt_count "transmuter.select.prompt_count"
  @keys "transmuter.select.keys"
  @scores "transmuter.select.scores"
  @tokens "transmuter.select.tokens"
  @selected "transmuter.select.selected"
  @next_seen "transmuter.select.next_seen"
  @next_history "transmuter.select.next_history"
  @next_count "transmuter.select.next_count"
  @write_at "transmuter.cache.write_at"

  @no_vocab "generation needs the vocabulary size: the logits output of the graph has no number in its last dimension, and the configuration has no vocab_size"

  # The inputs that MergedDecoder.rewrite/1 adds.
  @position_offset MergedDecoder.position_offset()
  @key_mask MergedDecoder.key_mask()

  # Buffers have a capacity of this size, or of the smaller size for a decoder with the
  # position offset of MergedDecoder, or above it a multiple of the step of @capacity_step
  # (see "Batches"). The capacity starts at
  # the rows of the prompt parts and a margin, and grows when a step does not fit (see grow/3).
  @min_capacity 256
  @merged_min_capacity 32

  # Above the smallest capacity, the buffers grow by a step of a quarter of the power of two
  # below the rows, at least this many rows: 64 rows up to 512, 128 up to 1024, and so on. So
  # a group runs at most a quarter more rows than it needs, and a long answer compiles four
  # capacities for each doubling of its rows.
  @capacity_step 64
  @capacity_margin 16

  # The bytes of released device buffers after which the generation process collects its
  # garbage, and the key of their count in the process dictionary.
  @collect_bytes 256 * 1024 * 1024
  @released {__MODULE__, :released}

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

    model = model |> keep_last_logits() |> MergedDecoder.rewrite() |> count_rows()

    %{
      model: model,
      vocab: vocab(model, repo.config),
      mode: mode(model, given, opts[:embed]),
      embed: opts[:embed],
      image_token: opts[:image_token],
      cache: Enum.filter(model.inputs, &String.starts_with?(&1, "past_key_values.")),
      buffers: buffers(model),
      min_capacity: min_capacity(model),
      max_capacity: model[:max_capacity],
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
          "position_ids" => List.duplicate(row, batch_size)
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
          for {name, value} <- [{"use_cache_branch", [0]}, {"num_logits_to_keep", 0}],
              name in model.inputs,
              into: %{} do
            {name, tensor.(name, value)}
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
        do: {min_new_tokens > 0, &Nx.add(&1, min_new_tokens)},
        else: {min_length > 0, fn _prompt -> min_length end}

    [
      ngram_size > 0 && no_repeat_ngram(ngram_size),
      :bad_words,
      min_length? && eos != [] && min_length(min_length, eos),
      forced_bos != nil && forced(&Nx.equal(&1.count, 1), [forced_bos]),
      # The row has max_length - 1 tokens, where max_length is the prompt and max_new_tokens.
      forced_eos != nil &&
        forced(&Nx.equal(generated(&1), max_new_tokens - 1), List.wrap(forced_eos)),
      :suppress_tokens,
      begin_suppress != [] && begin_suppress(begin_suppress, forced_bos != nil)
    ]
    |> Enum.filter(& &1)
  end

  # NoRepeatNGramLogitsProcessor: the last token of each n-gram whose first n - 1 tokens are the
  # last n - 1 tokens of the row. Window i of the tokens is an n-gram when it ends before the
  # count of the row. Its last token goes into a mask with one more place, where the windows
  # that do not match put their mark.
  defp no_repeat_ngram(size) do
    fn logits, row ->
      vocab = Nx.axis_size(logits, 0)
      windows = Nx.axis_size(row.tokens, 0) - size + 1
      column = &Nx.slice_along_axis(row.tokens, &1, windows, axis: 0)
      last = &Nx.take(row.tokens, Nx.max(Nx.add(Nx.subtract(row.count, size - 1), &1), 0))

      match =
        Enum.reduce(
          0..(size - 2)//1,
          Nx.less_equal(Nx.add(Nx.iota({windows}), size), row.count),
          fn k, match ->
            Nx.logical_and(match, Nx.equal(column.(k), last.(k)))
          end
        )

      index = Nx.select(match, column.(size - 1), vocab)

      banned =
        Nx.broadcast(Nx.tensor(0, type: :u8), {vocab + 1})
        |> Nx.indexed_put(Nx.new_axis(index, 1), Nx.broadcast(Nx.tensor(1, type: :u8), {windows}))
        |> Nx.slice_along_axis(0, vocab, axis: 0)

      remove(logits, banned)
    end
  end

  # MinLengthLogitsProcessor, with the minimum length of the row from the length of its prompt.
  defp min_length(min_length, eos) do
    fn logits, row ->
      short = Nx.less(row.count, min_length.(row.prompt))
      remove(logits, Nx.logical_and(short, any_of(logits, eos)))
    end
  end

  # ForcedBOSTokenLogitsProcessor and ForcedEOSTokenLogitsProcessor: when `at?` is true for the
  # row, the forced tokens get 0 and the other tokens are removed.
  defp forced(at?, ids) do
    fn logits, row ->
      forced = Nx.select(any_of(logits, ids), 0.0, Nx.Constants.min_finite(:f32))
      Nx.select(at?.(row), forced, logits)
    end
  end

  # SuppressTokensAtBeginLogitsProcessor. Its begin_index in transformers is the length of
  # the prompt, plus 1 after a forced bos token of a prompt of one token.
  defp begin_suppress(ids, forced_bos?) do
    fn logits, row ->
      begin = if forced_bos?, do: Nx.equal(row.prompt, 1), else: 0
      at = Nx.equal(generated(row), begin)
      remove(logits, Nx.logical_and(at, any_of(logits, ids)))
    end
  end

  defp generated(row), do: Nx.subtract(row.count, row.prompt)

  defp any_of(logits, ids) do
    iota = Nx.iota(Nx.shape(logits))
    ids |> Enum.map(&Nx.equal(iota, &1)) |> Enum.reduce(&Nx.logical_or/2)
  end

  defp remove(logits, mask), do: Nx.select(mask, Nx.Constants.min_finite(:f32), logits)

  @inputs ~w(input_ids inputs_embeds attention_mask position_ids num_logits_to_keep use_cache_branch) ++
            [@position_offset, @key_mask]

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
      # from the values of the mask (see count_rows/1).
      gqa && reads_values?(model, Enum.at(gqa.input, 5), "attention_mask") ->
        :in_place

      gqa ->
        :grow

      "position_ids" in model.inputs or @position_offset in model.inputs ->
        :append

      "use_cache_branch" in model.inputs ->
        :grow

      true ->
        raise ArgumentError, "the decoder declares neither position_ids nor use_cache_branch"
    end
  end

  # The operators that can be between the lm_head and the logits: each acts on each position
  # alone, with one input that is not static, such as a cast or the softcap of Gemma.
  @per_position ~w(Cast Identity Tanh Add Sub Mul Div)

  # Adds num_logits_to_keep to a decoder that does not declare it, with the nodes of the
  # Gemma 3 exports before the lm_head: the hidden states keep their last num_logits_to_keep
  # positions on axis 1, or all positions for 0. Generation reads the rank-3 logits, and the
  # operators after the lm_head keep the shape, so axis 1 of its input is the sequence. The
  # values from the lm_head to the logits have no other use, so no other value changes. An If
  # gives each branch the shape of its output, so each branch that gives the logits gets its
  # own cut, as in the merged decoders of optimum.
  defp keep_last_logits(model) do
    static = MapSet.new(Map.keys(model.params) ++ Map.keys(model.static))

    with false <- "num_logits_to_keep" in model.inputs,
         {:ok, lm_heads} <- follow(scope(model.nodes, model.outputs, static), "logits", nil) do
      slice_hidden(model, MapSet.new(lm_heads))
    else
      true ->
        model

      {:bail, node} ->
        Logger.debug(
          "generation computes the logits of all positions, because the way from the logits to the lm_head stops at #{where(node)}"
        )

        model
    end
  end

  defp where(nil), do: "a graph input"
  defp where(node), do: "#{node.op_type} #{node.name}"

  # A graph or a branch of an If: the node of each value, the uses of each value (the nodes
  # that read it and the outputs of the graph), and the static values: the constants of the
  # graph and the static values of the outer graphs that the graph does not make again.
  defp scope(nodes, outputs, outer) do
    producers = for node <- nodes, output <- node.output, into: %{}, do: {output, node}

    {constants, made} =
      Enum.split_with(producers, fn {_name, node} -> node.op_type == "Constant" end)

    static =
      outer
      |> MapSet.difference(MapSet.new(made, &elem(&1, 0)))
      |> MapSet.union(MapSet.new(constants, &elem(&1, 0)))

    %{
      producers: producers,
      uses: Enum.frequencies(Enum.flat_map(nodes, &Transmuter.dependencies/1) ++ outputs),
      static: static
    }
  end

  # The MatMul or MatMulNBits nodes that give the logits through operators of @per_position
  # and the branches of If nodes. Input 0 of each is the hidden states, also when tied
  # embeddings give its weight from a Transpose.
  defp lm_heads(scope, name) do
    case scope.producers[name] do
      node = %{op_type: op, input: [hidden | _]} when op in ["MatMul", "MatMulNBits"] ->
        if hidden in scope.static, do: {:bail, node}, else: {:ok, [node]}

      node = %{op_type: "If"} ->
        index = Enum.find_index(node.output, &(&1 == name))

        Enum.reduce_while(node.attribute, {:ok, []}, fn
          %{g: nil}, acc ->
            {:cont, acc}

          %{g: graph}, {:ok, found} ->
            outputs = Enum.map(graph.output, & &1.name)
            static = MapSet.union(scope.static, MapSet.new(graph.initializer, & &1.name))

            case follow(scope(graph.node, outputs, static), Enum.at(outputs, index), node) do
              {:ok, lm_heads} -> {:cont, {:ok, lm_heads ++ found}}
              bail -> {:halt, bail}
            end
        end)

      node = %{op_type: op} when op in @per_position ->
        case Enum.reject(node.input, &(&1 == "" or &1 in scope.static)) do
          [input] -> follow(scope, input, node)
          _inputs -> {:bail, node}
        end

      node ->
        {:bail, node}
    end
  end

  # Continues at a value that the scope makes and uses only on the way to the logits.
  defp follow(scope, name, node) do
    if scope.uses[name] == 1 and Map.has_key?(scope.producers, name),
      do: lm_heads(scope, name),
      else: {:bail, scope.producers[name] || node}
  end

  defp slice_hidden(model, lm_heads) do
    name = &("transmuter.num_logits_to_keep." <> &1)
    constant = &Nx.tensor(&1, type: :s64, backend: Nx.BinaryBackend)

    starts = [
      %NodeProto{
        op_type: "Neg",
        name: name.("neg"),
        input: ["num_logits_to_keep"],
        output: [name.("neg")]
      },
      %NodeProto{
        op_type: "Reshape",
        name: name.("starts"),
        input: [name.("neg"), name.("shape")],
        output: [name.("starts")]
      }
    ]

    constants = %{
      name.("shape") => constant.([1]),
      name.("ends") => constant.([2_147_483_647]),
      name.("axes") => constant.([1])
    }

    %{
      model
      | nodes: starts ++ cut(model.nodes, lm_heads, name),
        static: Map.merge(model.static, constants),
        inputs: model.inputs ++ ["num_logits_to_keep"],
        input_types: Map.put(model.input_types, "num_logits_to_keep", {:s64, []})
    }
  end

  # Puts a Slice of the hidden states before each lm_head, also in the branches of If nodes.
  # A branch reads the starts and the constants of the slice from the outer graph. The nodes
  # are found by value, because a branch can give its value the name of the If output.
  defp cut(nodes, lm_heads, name) do
    Enum.flat_map(nodes, fn node ->
      if MapSet.member?(lm_heads, node) do
        [hidden | rest] = node.input
        kept = name.(hidden)
        slice = [hidden, name.("starts"), name.("ends"), name.("axes")]

        [
          %NodeProto{op_type: "Slice", name: kept, input: slice, output: [kept]},
          %{node | input: [kept | rest]}
        ]
      else
        [%{node | attribute: Enum.map(node.attribute, &cut_branch(&1, lm_heads, name))}]
      end
    end)
  end

  defp cut_branch(attribute = %{g: graph}, lm_heads, name) when graph != nil,
    do: %{attribute | g: %{graph | node: cut(graph.node, lm_heads, name)}}

  defp cut_branch(attribute, _lm_heads, _name), do: attribute

  # GroupQueryAttention takes the valid rows of each sequence less one in seqlens_k. Gemma 3
  # counts them from the length of the attention mask. When no other node reads the cache or
  # its length, the rows come from the sum of the mask, as in Qwen3. The decoder then runs in
  # :in_place, where the mask of a row has ones for its valid rows and the new tokens.
  defp count_rows(model) do
    gqas = Enum.filter(model.nodes, &(&1.op_type == "GroupQueryAttention"))
    by_length = Enum.reject(gqas, &reads_values?(model, Enum.at(&1.input, 5), "attention_mask"))
    # The walk of MergedDecoder gave the other readers of the past the position offset.
    past = Enum.filter(model.inputs, &String.starts_with?(&1, "past_key_values."))
    past = if @position_offset in model.inputs, do: [], else: past
    cache = MapSet.new(past ++ Enum.flat_map(gqas, &Enum.drop(&1.output, 1)))
    reads_cache? = &Enum.any?(Transmuter.dependencies(&1), fn name -> name in cache end)

    with [_ | _] <- by_length,
         nil <- Enum.find(model.nodes -- gqas, reads_cache?) do
      name = &("transmuter.seqlens_k." <> &1)
      constant = &Nx.tensor(&1, type: :s64, backend: Nx.BinaryBackend)

      rows = [
        %NodeProto{
          op_type: "ReduceSum",
          name: name.("rows"),
          input: ["attention_mask", name.("axes")],
          output: [name.("rows")],
          attribute: [%AttributeProto{name: "keepdims", type: :INT, i: 0}]
        },
        %NodeProto{
          op_type: "Sub",
          name: name.("last"),
          input: [name.("rows"), name.("one")],
          output: [name.("last")]
        }
      ]

      nodes =
        Enum.map(model.nodes, fn node ->
          if node in by_length,
            do: %{node | input: List.replace_at(node.input, 5, name.("last"))},
            else: node
        end)

      constants = %{name.("axes") => constant.([1]), name.("one") => constant.(1)}
      %{model | nodes: rows ++ nodes, static: Map.merge(model.static, constants)}
    else
      [] ->
        model

      node ->
        Logger.debug(
          "generation keeps a cache of exact length, because #{where(node)} reads the cache or its length"
        )

        model
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
    unless ctx.vocab, do: raise(ArgumentError, @no_vocab)
    ctx = Map.put(ctx, :backend, Nx.Defn.to_backend(defn_options))

    batch_size = batch_size * ctx.num_beams

    # The first step of :grow takes the branch without cache, so each value of the static
    # inputs has a function. The inputs that the caller gives have their declared dimensions,
    # with the batch size for the batch and 1 for other symbolic ones. Other shapes compile on
    # their first step and stay cached, and one request of :grow has a shape for each new
    # token and each prompt chunk.
    state = initial_state(ctx, batch_size, capacity(ctx, 1))
    tail = tail(ctx)
    # In :in_place and :append, the present of each buffer has the shape of its past, so the
    # step writes it into the buffer of the past.
    donate = if ctx.mode in [:in_place, :append], do: ctx.buffers, else: []

    opts =
      [cache_size: ctx.max_new_tokens + @min_capacity, tail: tail, donate: donate] ++
        defn_options

    tokens = List.duplicate([0], batch_size)
    selection = selection_template(ctx, batch_size, state.capacity)

    runs =
      [0, 1]
      |> Enum.map(
        &step_inputs(
          ctx,
          %{state | lengths: List.duplicate(&1, batch_size)},
          tokens,
          List.duplicate(1, batch_size)
        )
      )
      |> Enum.uniq_by(fn {_inputs, static} -> static end)
      |> Map.new(fn {inputs, static} ->
        names = ctx.model.inputs -- Map.keys(Map.merge(inputs, static))
        given = templates(ctx.model, names, batch_size)

        compiled =
          Transmuter.compile(
            ctx.model,
            given |> Map.merge(inputs) |> Map.merge(selection),
            [static_inputs: static] ++ opts
          )

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

    ctx
    |> Map.put(:grow, GraphBreaks.jit(&grow_rows/2, defn_options))
    |> Map.put(:reorder, GraphBreaks.jit(&reorder_rows/2, defn_options))
    |> Map.put(:admit_rows, GraphBreaks.jit(&write_admitted/3, defn_options))
  end

  # The settings of the selection, and the processes of the processors with the masks in their
  # places.
  defp settings(ctx) do
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
    |> Map.put(:processes, ctx.processors)
  end

  # The selection of a step runs in the compiled program after the decoder. It takes the
  # logits from the outputs and the extra inputs of selection/3, and gives the selection in
  # place of the logits. In :append, the tail also writes the new rows of each present into
  # the buffer of its past, so the present that the step gives is the buffer. The term of
  # the tail names it in the key of the EXLA disk cache.
  defp tail(ctx) do
    settings = settings(ctx)
    beams? = ctx.num_beams > 1
    names = Map.keys(ctx.masks)
    buffers = if ctx.mode == :append, do: ctx.buffers, else: []

    fun = fn outputs, extra ->
      outputs = write_rows(outputs, extra, buffers)
      {logits, outputs} = Map.pop!(outputs, "logits")
      masks = Map.new(names, &{&1, extra[@mask <> Atom.to_string(&1)]})

      # The new tokens of the step go into the tokens that the repetition penalty applies to,
      # and after the tokens of each row, which the processors read. The next step takes both
      # from the device. A row without a new token has -1.
      seen = extra[@seen]
      {rows, vocab} = Nx.shape(seen)
      new = Nx.equal(Nx.iota({rows, vocab}, axis: 1), Nx.new_axis(extra[@tokens], 1))
      seen = seen |> Nx.logical_or(new) |> Nx.as_type(Nx.type(seen))
      row = append_token(extra[@history], extra[@count], extra[@tokens])
      row = Map.put(row, :prompt, extra[@prompt_count])

      # The last real token of each row is offset positions before the last position of the
      # step, and the logits keep the last positions of the step.
      index =
        Nx.axis_size(logits, 1)
        |> Nx.subtract(1)
        |> Nx.subtract(extra[@offset])
        |> Nx.max(0)

      selected =
        if beams?,
          do: select_beams(logits, index, seen, masks, row, extra[@scores], settings),
          else: select_tokens(logits, index, seen, masks, row, extra[@keys], settings)

      outputs
      |> Map.put(@selected, anchor(selected, [seen, row.tokens, row.count]))
      |> Map.put(@next_seen, seen)
      |> Map.put(@next_history, row.tokens)
      |> Map.put(@next_count, row.count)
    end

    {{:select, settings}, fun}
  end

  # The host reads only the selection, and the next step reads the other outputs of the
  # step. On EMLX 0.5, each of these stays an expression until the next step evaluates it, in
  # a wait of its own (elixir-nx/emlx#158 removes this). So the selection gets 0 times the first
  # value of each integer output, and the read of the selection evaluates them in the same
  # wait. 0 times an integer is 0, so the selection does not change. A float output has no
  # part in it, because 0 times NaN or infinity is NaN.
  defp anchor(selected, outputs) do
    for tensor <- outputs, Nx.Type.integer?(Nx.type(tensor)), reduce: selected do
      selected ->
        first = tensor |> Nx.reshape({Nx.size(tensor)}) |> Nx.slice([0], [1])
        Nx.add(selected, Nx.multiply(Nx.as_type(first, Nx.type(selected)), 0))
    end
  end

  # Writes the new token of each row after its tokens, and counts it. A row without a new
  # token (-1) keeps its tokens.
  defp append_token(tokens, count, new) do
    {rows, capacity} = Nx.shape(tokens)
    added = Nx.greater_equal(new, 0)
    at = Nx.equal(Nx.iota({rows, capacity}, axis: 1), Nx.new_axis(count, 1))
    write = Nx.logical_and(at, Nx.new_axis(added, 1))
    tokens = Nx.select(write, Nx.new_axis(new, 1), tokens)
    %{tokens: tokens, count: Nx.add(count, Nx.as_type(added, Nx.type(count)))}
  end

  # The vocabulary size: the last dimension of the logits output of the graph, or else the
  # vocab_size of the configuration when the graph gives a symbolic dimension.
  defp vocab(model, config) do
    with {_type, [_ | _] = dims} <- model[:output_types]["logits"],
         size when is_integer(size) <- List.last(dims) do
      size
    else
      _symbolic -> config["vocab_size"] || get_in(config, ["text_config", "vocab_size"])
    end
  end

  # The extra inputs of the selection for the compile, with zeros.
  defp selection_template(ctx, rows, capacity) do
    zeros = &Nx.broadcast(Nx.tensor(0, type: &1, backend: Nx.BinaryBackend), &2)

    group = %{
      state: %{
        seen: none({rows, ctx.vocab}),
        history: zeros.(:s32, {rows, capacity}),
        count: zeros.(:s32, {rows}),
        prompt_count: zeros.(:s32, {rows})
      },
      masks: token_masks(ctx),
      keys: zeros.(:u32, {rows, 2}),
      prompts: List.duplicate([0], rows),
      ids: List.duplicate([], rows),
      new: List.duplicate(nil, rows),
      scores: List.duplicate(0.0, rows)
    }

    selection = ctx |> selection(group) |> Map.put(@offset, zeros.(:s32, {rows}))

    if ctx.mode == :append,
      do: Map.put(selection, @write_at, zeros.(:s32, {})),
      else: selection
  end

  # The extra inputs of the selection of a step, from the group: the tokens that the
  # repetition penalty applies to and the tokens of each row, which stay on the device, the
  # new tokens of the step, the masks, the state of each processor, the keys of sampling,
  # and the scores of beam search.
  defp selection(ctx, group) do
    masks = for {name, mask} <- group.masks, into: %{}, do: {@mask <> Atom.to_string(name), mask}

    other =
      if ctx.num_beams > 1,
        do: %{@scores => Nx.tensor(group.scores, type: :f32, backend: Nx.BinaryBackend)},
        else: %{}

    tokens = Enum.map(group.new, &(&1 || -1))

    %{
      @seen => group.state.seen,
      @history => group.state.history,
      @count => group.state.count,
      @prompt_count => group.state.prompt_count,
      @keys => group.keys,
      @tokens => Nx.tensor(tokens, type: :s32, backend: Nx.BinaryBackend)
    }
    |> Map.merge(masks)
    |> Map.merge(other)
  end

  # The masks of :bad_words and :suppress_tokens, on the host.
  defp token_masks(ctx),
    do:
      Map.new(ctx.masks, fn {name, ids} ->
        {name, mark(none({ctx.vocab}), Enum.map(ids, &[&1]))}
      end)

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
          lengths = for prompt <- prompts, prompt != [], do: length(prompt)

          group =
            ctx
            |> open(length(prompts), Enum.sum(part_sizes(ctx, lengths)) + @capacity_margin)
            |> then(&admit(ctx, &1, requests(ctx, prompts, seeds, extras), inputs, features))
            |> then(
              &decode(ctx, &1, fn group -> notify(on_token, group, ctx.token_logprobs?) end)
            )

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
  # rows. A group has the prompts of one length when the rows of a step need one length of the
  # cache (see one_length?/1). Otherwise a buffer keeps the length of each row, and a group
  # takes the next prompt length while the rows of its prompt parts and its new tokens fit the
  # largest capacity of the decoder.
  defp groups(ctx, prompts) do
    lengths = prompts |> Enum.reject(&(&1 == [])) |> Enum.map(&length/1) |> Enum.uniq()

    chunk = fn length, group ->
      if group == [] or (not one_length?(ctx) and fits?(ctx, [length | group])),
        do: {:cont, [length | group]},
        else: {:cont, group, [length]}
    end

    for group <- Enum.chunk_while(lengths, [], chunk, &{:cont, &1, []}) do
      Enum.map(prompts, &if(length(&1) in group, do: &1, else: []))
    end
  end

  defp fits?(ctx, lengths) do
    rows = Enum.sum(part_sizes(ctx, lengths)) + ctx.max_new_tokens - 1
    rows <= (ctx.max_capacity || rows)
  end

  # The lengths of the prompt parts of rows with these numbers of tokens: one part for each
  # length with a merged decoder, else parts of power-of-two lengths with a last part no
  # longer than the shortest row.
  defp part_sizes(ctx, lengths) do
    longest = Enum.max(lengths)
    last = min(List.last(chunks(longest)), hd(chunks(Enum.min(lengths))))

    if merged?(ctx),
      do: lengths |> Enum.uniq() |> Enum.sort(:desc),
      else: chunks(longest - last) ++ [last]
  end

  # The requests of the rows of `prompts` that have a prompt: the row, its prompt, its seed,
  # its budget of new tokens, its positions and the cursor of its image features.
  defp requests(ctx, prompts, seeds, extras) do
    for {{prompt, seed}, row} <- Enum.with_index(Enum.zip(prompts, seeds)), prompt != [] do
      %{
        row: row,
        prompt: prompt,
        seed: seed,
        budget: ctx.max_new_tokens,
        position: Enum.at(extras.positions, row),
        cursor: Enum.at(extras.cursors, row)
      }
    end
  end

  @doc false
  # A group of `rows` empty rows, with buffers for `length` rows of cache. An empty row has no
  # request and is done. admit/5 gives requests to rows, and next/2 runs the steps.
  def open(ctx, rows, length \\ @capacity_margin) do
    unless Map.has_key?(ctx, :backend) do
      raise ArgumentError, "a group needs a context from compile/3, got a context of context/4"
    end

    state = initial_state(ctx, rows, capacity(ctx, length))
    zeros = &Nx.broadcast(Nx.tensor(0, type: &1, backend: ctx.backend), &2)

    state = %{
      state
      | seen: Nx.backend_transfer(none({rows, ctx.vocab}), ctx.backend),
        history: zeros.(:s32, {rows, state.capacity}),
        count: zeros.(:s32, {rows}),
        prompt_count: zeros.(:s32, {rows})
    }

    %{
      state: state,
      masks:
        Map.new(token_masks(ctx), fn {name, mask} ->
          {name, Nx.backend_transfer(mask, ctx.backend)}
        end),
      keys: zeros.(:u32, {rows, 2}),
      prompts: List.duplicate([], rows),
      ids: List.duplicate([], rows),
      logprobs: List.duplicate([], rows),
      new: List.duplicate(nil, rows),
      done: List.duplicate(true, rows),
      budgets: List.duplicate(0, rows),
      scores: nil
    }
  end

  @doc false
  # Gives each request of `requests` its row, runs the prompts and reads the first token of
  # each new row (see prefill/5). A request is a map with the row, the prompt, the seed and
  # the budget of new tokens, and can have the positions of the prompt and the cursor of its
  # image features. The other rows that have not ended take their next token in
  # the same steps. `inputs` and `features` are the inputs and the image features of the
  # prompts, which only the first requests of a group can have.
  def admit(ctx, group, requests, inputs \\ %{}, features \\ nil) do
    ctx |> prefill(group, requests, inputs, features) |> then(&select(ctx, &1))
  end

  @doc false
  # Runs the step of the new token of each row that has not ended, and reads its next token.
  def next(ctx, group), do: select(ctx, advance(ctx, group))

  # Runs the prompts of `requests` in parts of power-of-two lengths. Each prompt fills the
  # parts from the last one, so the last token of each prompt is in the last part. A row that
  # has a new token from an earlier step takes it in the last part, and has no tokens in the
  # other parts. The last part is no longer than the tokens of each row in it, so it ends each
  # row at its last position, where the step keeps the logits. A merged decoder runs each
  # prompt as one part, on the branch without cache (see the module documentation): one part
  # for each prompt length. So when the rows of a step need one length of the cache, and with
  # a merged decoder, requests join only a group without other rows. Each row takes its first
  # token from the part where its prompt ends.
  defp prefill(ctx, group, requests, inputs, features) do
    pending = Enum.zip_with(group.new, group.done, &if(&2, do: nil, else: &1))

    if (one_length?(ctx) or merged?(ctx)) and Enum.any?(pending) do
      raise ArgumentError,
            "requests can join a group with other rows only when the rows of a step can have different lengths of the cache, without a merged decoder"
    end

    # A row with a new token has one token in these steps.
    joining = Map.new(requests, &{&1.row, &1})
    lengths = Enum.map(requests, &length(&1.prompt)) ++ for(token <- pending, token, do: 1)
    sizes = part_sizes(ctx, lengths)

    group =
      Enum.reduce(requests, group, fn request, group ->
        %{
          group
          | prompts: List.replace_at(group.prompts, request.row, request.prompt),
            ids: List.replace_at(group.ids, request.row, []),
            logprobs: List.replace_at(group.logprobs, request.row, []),
            new: List.replace_at(group.new, request.row, nil),
            done: List.replace_at(group.done, request.row, false),
            budgets: List.replace_at(group.budgets, request.row, request.budget),
            state: %{
              group.state
              | lengths: List.replace_at(group.state.lengths, request.row, 0),
                valid: List.replace_at(group.state.valid, request.row, []),
                positions:
                  List.replace_at(
                    group.state.positions,
                    request.row,
                    Map.get(request, :position, {[], 0})
                  ),
                cursors:
                  List.replace_at(group.state.cursors, request.row, Map.get(request, :cursor, 0))
            }
        }
      end)

    group = %{group | state: fit(ctx, group.state, Enum.sum(sizes))}
    group = admit_rows(ctx, group, requests)

    # The parts of each row: the parts of its prompt, its new token in the last part, or no
    # tokens. The selection of each part starts from the state after admit_rows/3, and only
    # the last part adds the new tokens of the rows to it.
    parts =
      for {token, row} <- Enum.with_index(pending) do
        case joining do
          %{^row => request} -> prompt_parts(ctx, request.prompt, sizes)
          _other -> List.duplicate([], length(sizes) - 1) ++ [List.wrap(token)]
        end
      end

    before = selection(ctx, %{group | new: List.duplicate(nil, length(pending))})

    selections =
      List.duplicate(before, length(sizes) - 1) ++ [selection(ctx, %{group | new: pending})]

    # The steps release the state of the step before, so they start without the state that the
    # selections read, which is released after the last part.
    own = Map.take(group.state, [:seen, :history, :count])
    state = Map.merge(group.state, %{seen: nil, history: nil, count: nil, features: features})
    state = %{state | inputs: Map.merge(state.inputs, inputs)}

    # The part where the prompt of each row of the selection ends, counted from the last part,
    # whose selection gives its first token. The selection has one row for the num_beams rows
    # of each prompt. The rows that end in an earlier part take its selection.
    ends =
      parts
      |> Enum.take_every(ctx.num_beams)
      |> Enum.map(fn row -> Enum.find_index(Enum.reverse(row), &(&1 != [])) end)

    count = length(sizes)

    {state, picked} =
      parts
      |> Enum.zip_with(& &1)
      |> Enum.zip(selections)
      |> Enum.with_index()
      |> Enum.reduce({state, []}, fn {{parts, selection}, k}, {state, picked} ->
        state = step(ctx, state, parts, selection)
        rows = for {last, row} <- Enum.with_index(ends), last == count - 1 - k, last > 0, do: row
        {state, if(rows == [], do: picked, else: [{rows, Nx.to_list(state.selected)} | picked])}
      end)

    release(Map.values(own), [])
    %{group | state: %{state | features: nil, selected: selected(state.selected, picked)}}
  end

  # The parts of a prompt: the parts of power-of-two lengths that fill from the last one, or for
  # a merged decoder the whole prompt in the part of its length.
  defp prompt_parts(ctx, prompt, sizes) do
    if merged?(ctx),
      do: Enum.map(sizes, &if(&1 == length(prompt), do: prompt, else: [])),
      else: split_from_end(prompt, sizes)
  end

  # The selection of each row from the part where its prompt ends. `picked` has the rows of the
  # selection and the selection of each earlier part where prompts end. The other rows keep the
  # selection of the last part, which stays on the device when no earlier part has rows.
  defp selected(last, []), do: last

  defp selected(last, picked) do
    rows =
      Enum.reduce(picked, Nx.to_list(last), fn {ended, selected}, rows ->
        Enum.reduce(ended, rows, &List.replace_at(&2, &1, Enum.at(selected, &1)))
      end)

    tensor = Nx.tensor(rows, type: Nx.type(last), backend: Nx.BinaryBackend)
    release([last], [])
    tensor
  end

  # Writes the device state of the rows of `requests`: the seen tokens and the tokens of its
  # prompt, its count and prompt count, and the key of its seed.
  defp admit_rows(ctx, group, requests) do
    capacity = group.state.capacity
    prompt_ids = for {request, i} <- Enum.with_index(requests), id <- request.prompt, do: [i, id]
    counts = Enum.map(requests, &length(&1.prompt))
    host = &Nx.tensor(&1, type: :s32, backend: Nx.BinaryBackend)

    values = %{
      seen: mark(none({length(requests), ctx.vocab}), prompt_ids),
      history:
        host.(Enum.map(requests, &(&1.prompt ++ List.duplicate(0, capacity - length(&1.prompt))))),
      count: host.(counts),
      prompt_count: host.(counts),
      keys: requests |> Enum.map(&Nx.Random.key(&1.seed)) |> Nx.stack()
    }

    old =
      group.state
      |> Map.take([:seen, :history, :count, :prompt_count])
      |> Map.put(:keys, group.keys)

    written = ctx.admit_rows.(old, host.(Enum.map(requests, & &1.row)), values)
    release(Map.values(old), [])
    {keys, written} = Map.pop!(written, :keys)
    %{group | keys: keys, state: Map.merge(group.state, written)}
  end

  # The rows in `index` get the rows of `values`.
  defp write_admitted(tensors, index, values) do
    Map.new(tensors, fn {name, tensor} ->
      {name,
       Nx.indexed_put(tensor, Nx.new_axis(index, 1), Nx.as_type(values[name], Nx.type(tensor)))}
    end)
  end

  # The selection of each step, until each row has ended.
  defp decode(ctx, group, notify) do
    notify.(group)
    if Enum.all?(group.done), do: finish(group), else: decode(ctx, next(ctx, group), notify)
  end

  # Frees the device tensors that the group owns when it ends: its cache, its last selection
  # and selection state, its masks and its keys.
  defp finish(group = %{state: state}) do
    own = [state.selected, state.seen, state.history, state.count, state.prompt_count]
    own = [group.keys | own] ++ Map.values(state.cache)
    release(own ++ Map.values(group.masks), [])
    group
  end

  defp select(ctx, group) do
    {new, logprobs} =
      group.state.selected
      |> Nx.to_list()
      |> Enum.zip_with(group.done, fn [token | logprob], done? ->
        if done? or token in ctx.eos, do: {nil, nil}, else: {token, from_bits(logprob)}
      end)
      |> Enum.unzip()

    # A row ends at an eos token or with the new tokens of its budget.
    ids = push(group.ids, new, new)

    done =
      Enum.zip_with([new, ids, group.budgets], fn [token, ids, budget] ->
        token == nil or length(ids) == budget
      end)

    %{
      group
      | new: new,
        done: done,
        ids: ids,
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
    # The first beam of each prompt starts at 0 and the others at -1e9, so the first step
    # selects only from the first beam.
    scores = Enum.flat_map(prompts, fn _ -> [0.0 | List.duplicate(-1.0e9, beams - 1)] end)
    rows = repeat.(prompts)
    lengths = for row <- rows, row != [], do: length(row)

    group =
      ctx
      |> open(length(rows), Enum.sum(part_sizes(ctx, lengths)) + @capacity_margin)
      |> Map.put(:scores, scores)
      |> then(
        &prefill(ctx, &1, requests(ctx, rows, repeat.(seeds), extras), inputs, extras.features)
      )

    items = Enum.map(prompts, &%{finished: [], open?: true, done?: &1 == []})

    %{items: items} = search_steps(ctx, group, items, 0)

    # The rows of each prompt are copies of its inputs that the search owns.
    release(Map.values(inputs), [])

    for item <- items do
      case item.finished do
        [] -> []
        [{_score, ids, logprobs} | _] -> tokens(ids, logprobs)
      end
    end
  end

  defp search_steps(ctx, group, items, step) do
    vocab = ctx.vocab
    candidates = Nx.to_list(group.state.selected)

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
      finish(group)
      %{items: items}
    else
      beams = Enum.concat(beams)
      group = reorder(ctx, group, Enum.map(beams, & &1.row))
      new = Enum.map(beams, & &1.token)

      group = %{
        group
        | new: new,
          ids: push(group.ids, new, new),
          logprobs: push(group.logprobs, new, Enum.map(beams, & &1.logprob)),
          scores: Enum.map(beams, & &1.score)
      }

      group = %{group | state: fit(ctx, group.state, 1)}
      selection = selection(ctx, group)
      group = %{group | state: step(ctx, group.state, Enum.map(new, &List.wrap/1), selection)}
      search_steps(ctx, group, items, step + 1)
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

      rows = %{
        @seen => state.seen,
        @history => state.history,
        @count => state.count,
        @prompt_count => state.prompt_count
      }

      written = state.cache |> Map.take(state.written) |> Map.merge(rows)
      {moved, reordered} = Map.split(ctx.reorder.(written, index), Map.keys(rows))
      release(Map.values(written), Map.values(moved) ++ Map.values(reordered))
      cache = Map.merge(state.cache, reordered)

      state = %{
        state
        | cache: cache,
          seen: moved[@seen],
          history: moved[@history],
          count: moved[@count],
          prompt_count: moved[@prompt_count],
          cursors: take.(state.cursors),
          positions: take.(state.positions),
          lengths: take.(state.lengths),
          valid: take.(state.valid)
      }

      %{
        group
        | state: state,
          prompts: take.(group.prompts),
          ids: take.(group.ids),
          logprobs: take.(group.logprobs)
      }
    end
  end

  defp reorder_rows(cache, index),
    do: Map.new(cache, fn {name, tensor} -> {name, Nx.take(tensor, index)} end)

  @min_f32 Nx.to_number(Nx.Constants.min_finite(:f32, backend: Nx.BinaryBackend))

  # The scores of transformers are f32, so each division rounds to f32 as torch does, and a
  # score below the f32 range stays at its lowest value.
  defp f32(value) do
    <<rounded::float-32>> = <<max(value, @min_f32)::float-32>>
    rounded
  end

  # The rows of other groups have no token and are not finished. The log-probability of a new
  # token is the first one of its row.
  defp notify(nil, _group, _logprobs?), do: :ok

  defp notify(on_token, group, logprobs?) do
    {tokens, finished} =
      Enum.zip_with([group.prompts, group.new, group.done], fn
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

  # The step of the new tokens gives the selection for the remaining tokens.
  defp advance(ctx, group) do
    group = %{group | state: fit(ctx, group.state, 1)}
    selection = selection(ctx, group)
    %{group | state: step(ctx, group.state, Enum.map(group.new, &List.wrap/1), selection)}
  end

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

  defp min_capacity(model),
    do: if(@position_offset in model.inputs, do: @merged_min_capacity, else: @min_capacity)

  # The smallest capacity, or above it a multiple of the step of @capacity_step, at most the
  # largest capacity of the decoder (see MergedDecoder).
  defp capacity(ctx, rows) do
    capacity =
      if rows <= ctx.min_capacity do
        ctx.min_capacity
      else
        below = 2 ** (length(Integer.digits(rows - 1, 2)) - 1)
        step = max(@capacity_step, div(below, 4))
        div(rows + step - 1, step) * step
      end

    min(capacity, ctx.max_capacity || capacity)
  end

  defp initial_state(ctx, batch_size, capacity) do
    # A cache of exact length starts with a placeholder row that the branch without cache does
    # not read. The zeros are made on the backend of the compiler, so they do not cross from
    # the host.
    cache =
      Map.new(ctx.cache, fn name ->
        {type, [_batch, heads, _length, head_dim]} = ctx.model.input_types[name]
        rows = if ctx.mode != :grow and name in ctx.buffers, do: capacity, else: 1
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
      used: 0,
      valid: List.duplicate([], batch_size),
      capacity: capacity,
      selected: nil,
      seen: nil,
      history: nil,
      count: nil,
      prompt_count: nil
    }
  end

  # Gives each row its tokens of `parts` and padding tokens after them, so that all rows
  # have the same number of tokens. The selection of the step reads the logits at the last
  # real token of each row.
  defp step(ctx, state, parts, selection) do
    counts = Enum.map(parts, &length/1)
    n = Enum.max(counts)
    tokens = Enum.map(parts, &(&1 ++ List.duplicate(0, n - length(&1))))
    {inputs, static} = step_inputs(ctx, state, tokens, counts)
    {inputs, state} = embed(ctx, state, inputs, tokens)

    # The selection takes the logits at the last real token of each row, which is this many
    # positions before the last position of the step.
    offsets = Nx.tensor(Enum.map(counts, &(n - &1)), type: :s32, backend: Nx.BinaryBackend)

    # The selection of the last step is read, or is that of an earlier part of the prompt,
    # so nothing reads it now.
    release(List.wrap(state.selected), [])
    selection = Map.put(selection, @offset, offsets)

    # In :append, the new rows of all rows go into the buffer after the rows of the earlier
    # steps.
    selection =
      if ctx.mode == :append,
        do:
          Map.put(
            selection,
            @write_at,
            Nx.tensor(state.used, type: :s32, backend: Nx.BinaryBackend)
          ),
        else: selection

    outputs = ctx.run.(Map.merge(inputs, selection), static)

    # The embeddings of the tokens come from ctx.embed, and nothing reads them after the call.
    release(List.wrap(inputs["inputs_embeds"]), [])

    # The seen tokens, the tokens and the count of each row that the last step gave are the
    # inputs of this step, or those of a part of the prompt that nothing reads. The parts of
    # the prompt take theirs from start/6.
    release([state.seen, state.history, state.count], [])

    # Transmuter.run leaves out empty outputs, such as the present of a cache that does not
    # change, so that cache keeps its value.
    present =
      for "past_key_values." <> rest = name <- ctx.cache,
          Map.has_key?(outputs, "present." <> rest),
          into: %{},
          do: {name, outputs["present." <> rest]}

    # The present replaces the cache entries that the step wrote, and their old buffers are
    # dead. In :append, the tail wrote the new rows into the buffers (see tail/1). The
    # cross-attention cache of a merged decoder is not a buffer, so its present replaces it.
    buffers = Map.take(state.cache, Map.keys(present))
    release(Map.values(buffers), Map.values(present))

    # A buffer keeps the real tokens of each row. The cache of :grow has all tokens.
    lengths =
      if ctx.mode == :grow,
        do: Enum.map(state.lengths, &(&1 + n)),
        else: Enum.zip_with(state.lengths, counts, &+/2)

    %{
      state
      | cache: Map.merge(state.cache, present),
        written: Map.keys(present),
        lengths: lengths,
        used: state.used + n,
        valid: Enum.zip_with(state.valid, counts, &add_rows(&1, state.used, &2)),
        selected: outputs[@selected],
        seen: outputs[@next_seen],
        history: outputs[@next_history],
        count: outputs[@next_count]
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
        merged = ctx.merge.(embeds, state.features, index, mask)
        release([embeds], [merged])
        merged
      end

    {Map.put(inputs, "inputs_embeds", embeds), %{state | cursors: cursors}}
  end

  @doc false
  # Frees the device buffers of `dead`, tensors that generation made and that nothing reads
  # after this point. A buffer that a tensor of `kept` shares stays, because a merged
  # decoder can give a past back as its present. Host tensors need no release.
  def release(dead, kept) do
    shared = for %{data: data} <- kept, do: data

    freed =
      for tensor = %{data: data} <- dead,
          not is_struct(data, Nx.BinaryBackend),
          data not in shared do
        Nx.backend_deallocate(tensor)
        Nx.byte_size(tensor)
      end

    collect(Enum.sum(freed))
  end

  # The process collects its garbage after it releases @collect_bytes of device buffers, see
  # the device memory section of the module documentation.
  defp collect(bytes) do
    released = Process.get(@released, 0) + bytes

    if released >= @collect_bytes do
      :erlang.garbage_collect()
      Process.put(@released, 0)
    else
      Process.put(@released, released)
    end

    :ok
  end

  # Grows the state when the next step of n tokens does not fit, before the selection of the
  # step reads the tokens of the rows from the state.
  defp fit(ctx, state, n) when is_integer(n) do
    rows = if ctx.mode == :append, do: state.used + n, else: Enum.max(state.lengths) + n
    if rows > state.capacity, do: grow(ctx, state, rows), else: state
  end

  # Gives the buffers the capacity of capacity/2 that fits the rows of a step, up to the
  # largest capacity of the decoder. One compiled function pads all buffers with free
  # rows, which the masks hide, and each growth compiles the steps of one more capacity. The
  # tokens of the rows, which the processors read, have the same capacity and grow with the
  # buffers, also in :grow, where the cache has its exact length and only the tokens grow.
  defp grow(ctx, state, rows) do
    capacity = capacity(ctx, rows)

    if rows > capacity do
      raise ArgumentError,
            "generation needs #{rows} rows of cache, more than the #{capacity} rows that the static tables of the decoder fit"
    end

    buffers = if ctx.mode == :grow, do: %{}, else: Map.take(state.cache, ctx.buffers)
    old = Map.put(buffers, @history, state.history)

    {history, grown} =
      Map.pop!(ctx.grow.(old, Nx.iota({capacity}, backend: Nx.BinaryBackend)), @history)

    release(Map.values(old), [history | Map.values(grown)])
    %{state | cache: Map.merge(state.cache, grown), history: history, capacity: capacity}
  end

  # The rows are axis 2 of a buffer, {batch, heads, rows, dim}, and axis 1 of the tokens,
  # {batch, rows}.
  defp grow_rows(tensors, capacity) do
    Map.new(tensors, fn {name, tensor} ->
      axis = if Nx.rank(tensor) == 2, do: 1, else: 2
      free = Nx.axis_size(capacity, 0) - Nx.axis_size(tensor, axis)
      padding = List.replace_at(List.duplicate({0, 0, 0}, Nx.rank(tensor)), axis, {0, free, 0})
      {name, Nx.pad(tensor, 0, padding)}
    end)
  end

  # The last rows of each present, as many as the tokens of the step, are the new rows:
  # after the buffer on the branch with cache, or all rows on the branch without cache of a
  # merged decoder. They go into the buffer at the write offset of the step, for all batch
  # rows in one update. The present that the step gives is then the buffer.
  defp write_rows(outputs, _inputs, []), do: outputs

  defp write_rows(outputs, inputs, buffers) do
    rows = Nx.axis_size(inputs["input_ids"] || inputs["inputs_embeds"], 1)

    for "past_key_values." <> rest = name <- buffers,
        present = outputs["present." <> rest],
        present != nil,
        into: outputs do
      new = Nx.slice_along_axis(present, Nx.axis_size(present, 2) - rows, rows, axis: 2)
      {"present." <> rest, Nx.put_slice(inputs[name], [0, 0, inputs[@write_at], 0], new)}
    end
  end

  # The valid rows of a row in the buffer of :append are ranges {start, count}, the newest
  # first. A step adds the real tokens of the row at its write offset, as a new range or as
  # the end of the newest range.
  defp add_rows(ranges, _at, 0), do: ranges

  defp add_rows([{start, count} | rest], at, n) when start + count == at,
    do: [{start, count + n} | rest]

  defp add_rows(ranges, at, n), do: [{at, n} | ranges]

  # The mask of the buffer of a row: ones at its valid rows, and zeros at its gap rows and
  # the free rows.
  defp valid_mask(ranges, capacity) do
    {mask, first} =
      Enum.reduce(ranges, {[], capacity}, fn {start, count}, {mask, stop} ->
        {List.duplicate(1, count) ++ List.duplicate(0, stop - start - count) ++ mask, start}
      end)

    List.duplicate(0, first) ++ mask
  end

  defp step_inputs(ctx, state, tokens, counts) do
    n = length(hd(tokens))

    # The prompt of a merged decoder takes the branch without cache, which reads no past: a step
    # whose rows with tokens have no rows in the cache.
    cached? =
      not (merged?(ctx) and
             Enum.all?(Enum.zip(state.lengths, counts), fn {past, count} ->
               past == 0 or count == 0
             end))

    masks =
      for {past, valid} <- Enum.zip(state.lengths, state.valid) do
        case ctx.mode do
          :in_place ->
            List.duplicate(1, past + n) ++ List.duplicate(0, state.capacity - past - n)

          :append when cached? ->
            valid_mask(valid, state.capacity) ++ List.duplicate(1, n)

          :append ->
            List.duplicate(1, n)

          :grow ->
            List.duplicate(1, past + n)
        end
      end

    values = %{
      "input_ids" => tokens,
      "attention_mask" => masks,
      "position_ids" => positions(ctx, state, n),
      @position_offset => position_offset(ctx, state),
      @key_mask => masks
    }

    inputs =
      for {name, value} <- values,
          name in ctx.model.inputs,
          into: Map.merge(state.cache, state.inputs) do
        {type, _dims} = ctx.model.input_types[name]
        {name, Nx.tensor(value, type: type, backend: Nx.BinaryBackend)}
      end

    # The last token of each row is at the last position of the last part of the prompt (see
    # start/6), and nothing reads the logits of the other parts, so a step keeps one position.
    flag = if cached?, do: 1, else: 0

    static =
      for {name, value} <- [{"use_cache_branch", [flag]}, {"num_logits_to_keep", 1}],
          name in ctx.model.inputs,
          into: %{} do
        {type, _dims} = ctx.model.input_types[name]
        {name, Nx.tensor(value, type: type, backend: Nx.BinaryBackend)}
      end

    {inputs, static}
  end

  # The cache entries in buffers: all of them, or the self-attention cache of a merged decoder
  # that MergedDecoder.rewrite/1 moved to :append.
  defp buffers(model) do
    if @position_offset in model.inputs,
      do: MergedDecoder.buffers(model),
      else: Enum.filter(model.inputs, &String.starts_with?(&1, "past_key_values."))
  end

  # A merged decoder has a branch without cache and a branch with cache (use_cache_branch).
  # The prompt rule applies to the merged decoders in :grow and to those that
  # MergedDecoder.rewrite/1 moves to :append.
  defp merged?(ctx) do
    "use_cache_branch" in ctx.model.inputs and
      (ctx.mode == :grow or @position_offset in ctx.model.inputs)
  end

  # The offset of the positions of a merged decoder in :append (see MergedDecoder): the valid
  # rows of each row less the capacity, or one value for all rows. In a group of one prompt
  # length, the valid rows of each row are the rows of all steps.
  defp position_offset(ctx, state) do
    if per_row_offset?(ctx),
      do: for(past <- state.lengths, do: [past - state.capacity]),
      else: state.used - state.capacity
  end

  defp per_row_offset?(ctx),
    do: match?({_type, [_ | _]}, ctx.model.input_types[@position_offset])

  # The rows of a step need one length of the cache in :grow, where the positions come from it,
  # and with one position offset for all rows.
  defp one_length?(ctx),
    do: ctx.mode == :grow or match?({_type, []}, ctx.model.input_types[@position_offset])

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
  # of the row folded with the number of new tokens of the row, so the tokens of a row do not
  # depend on the step at which the row starts. The function runs in the compiled
  # program of the step (see tail/1), because eager random numbers over a large vocabulary
  # are slow. EMLX evaluates each output of a compiled function separately, so the function
  # gives one s32 output: the token of each row, and with token_logprobs the bits of its
  # log-probability.
  defp select_tokens(logits, index, seen, masks, row, keys, settings) do
    [logits, index, seen, keys] = Enum.map([logits, index, seen, keys], &Nx.vectorize(&1, :rows))
    row = Map.new(row, fn {name, tensor} -> {name, Nx.vectorize(tensor, :rows)} end)
    logits = Nx.take(logits, index)
    key = Transmuter.Random.fold_in(keys, generated(row))
    {scores, token} = select_token(logits, seen, masks, row, key, settings)
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
  defp select_token(logits, seen, masks, row, key, settings) do
    logits = logits |> Nx.as_type(:f32) |> process(seen, masks, row, settings)

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
  defp process(logits, seen, masks, row, settings) do
    penalty = settings.repetition_penalty

    penalized =
      Nx.select(Nx.greater(logits, 0), Nx.divide(logits, penalty), Nx.multiply(logits, penalty))

    logits = Nx.select(seen, penalized, logits)

    logits =
      Enum.reduce(settings.processes, logits, fn
        mask, logits when is_atom(mask) -> remove(logits, masks[mask])
        process, logits -> process.(logits, row)
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
  defp select_beams(logits, index, seen, masks, row, scores, settings) do
    [logits, index, seen, scores] =
      Enum.map([logits, index, seen, scores], &Nx.vectorize(&1, :rows))

    row = Map.new(row, fn {name, tensor} -> {name, Nx.vectorize(tensor, :rows)} end)

    step_scores =
      logits
      |> Nx.take(index)
      |> Nx.as_type(:f32)
      |> log_softmax()
      |> process(seen, masks, row, settings)

    sums = step_scores |> Nx.add(scores) |> Nx.max(Nx.Constants.min_finite(:f32))
    sums = Nx.devectorize(sums, keep_names: false)
    {rows, vocab} = Nx.shape(sums)
    shape = {div(rows, settings.num_beams), settings.num_beams * vocab}
    {values, indices} = largest(Nx.reshape(sums, shape), settings.candidates)
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

  @doc false
  # Nx.top_k on the last axis in two stages: the top k of each chunk of the row, then the top k
  # of the chunk results. On CUDA, XLA computes the top_k of each row in one thread block or in
  # one sort, so a long row in few rows, such as the candidates of beam search or the vocabulary
  # of a sampled row, uses a small part of the GPU. A chunk has the power of two at or above the square root of the row length, and at
  # least k values, so a row of up to that length is one chunk. The padding of the last chunk
  # has the smallest value of the tensor, and the top_k of each stage gives the lower index
  # first among equal values, so a real value wins each tie with the padding, and the result
  # is that of Nx.top_k.
  def largest(tensor, k) do
    n = Nx.axis_size(tensor, -1)
    power = &(2 ** ceil(:math.log2(&1)))
    size = min(max(power.(:math.sqrt(n)), power.(k)), n)
    chunks = div(n + size - 1, size)
    rest = tensor.shape |> Tuple.delete_at(tuple_size(tensor.shape) - 1) |> Tuple.to_list()
    padding = List.duplicate({0, 0, 0}, length(rest)) ++ [{0, chunks * size - n, 0}]
    padded = Nx.pad(tensor, Nx.reduce_min(Nx.devectorize(tensor)), padding)
    {values, indices} = Nx.top_k(Nx.reshape(padded, List.to_tuple(rest ++ [chunks, size])), k: k)
    indices = Nx.add(indices, Nx.multiply(Nx.iota({chunks, 1}), size))
    flat = List.to_tuple(rest ++ [chunks * k])
    {values, at} = Nx.top_k(Nx.reshape(values, flat), k: k)
    {values, Nx.take_along_axis(Nx.reshape(indices, flat), at, axis: -1)}
  end

  defp top_k(logits, k) when k > 0 and k < elem(logits.shape, 0) do
    {values, _} = largest(logits, k)
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
