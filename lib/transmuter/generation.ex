defmodule Transmuter.Generation do
  @moduledoc """
  The generation loop of a decoder with a KV cache. `Transmuter.Text.generation/2`, `Transmuter.Text.text_to_text/2`, `Transmuter.Vision.image_to_text/2`, `Transmuter.Vision.image_text_to_text/2` and `Transmuter.Audio.speech_to_text/2` use it.

  Generation runs the model in steps. A step takes new tokens and the KV cache, and gives the logits and the next cache. The prompt runs in steps of power-of-two lengths (45 tokens run as 32, 8, 4 and 1), and each new token is a step of length 1, so a small number of compiled shapes covers all prompts.

  The cache works in one of three modes, from the structure of the graph:

    * `:in_place` - the graph has GroupQueryAttention, which counts the valid rows of the cache from the values of the attention mask. Generation gives a GroupQueryAttention that counts them from the length of the mask, as in Gemma 3, the sum of the mask in place of that count, when no other node reads the cache or its length, or when `Transmuter.Generation.MergedDecoder` gives those nodes the position offset, as in Apertus (see "Merged decoders"). The cache is a buffer as long as the attention mask, and the operator writes the new keys and values after the valid rows.

    * `:append` - the graph declares `position_ids`, or it is a merged decoder whose positions are reads of a table at the length of the cache plus the index of the token, as in GPT-2, Whisper, Marian and TrOCR (see "Merged decoders"). The cache is a buffer whose free rows the attention mask hides. The present cache is the buffer and the new rows after it, and the new rows go into the buffer.

    * `:grow` - the graph computes the positions from the length of the cache in a way that `Transmuter.Generation.MergedDecoder` does not cover, such as the rotary table of the merged llama2.c, or a node other than GroupQueryAttention reads the cache or its length, as in Gemma 4. This holds also when the GroupQueryAttention of the decoder counts the valid rows from the values of the mask. The cache has its exact length, so each length is a new shape that compiles on its first step.

  ## Merged decoders

  A merged decoder of optimum has a branch without cache and a branch with cache (`use_cache_branch`). The exports trace the branch with cache for one new token, and some of them, such as Whisper and Marian, have no causal mask in it, so a step of several tokens there gives wrong logits (ONNX Runtime gives the same).

  A step whose rows with tokens have no cache yet runs on the branch without cache, because Nx has no tensors of length 0. This is the first step of a new group, or of a group whose other rows ended. This branch has the causal mask, so the step takes the parts of the prompts as for the other decoders. A prompt of up to 64 tokens runs in one step at the smallest power of two that holds it, so a short prompt, such as the 4 tokens of Whisper, runs at about its own length. A longer prompt takes parts of 64 or 512 tokens, as the other steps. Each row reads the logits at its last token, and the padding after a shorter part is in gap rows of the buffers. The other steps run on the branch with cache.

  When requests can join a group (see "Logits"), the prompt of a request that joins running rows runs on the branch with cache, next to their new tokens. When each attention of the branch with cache that reads a buffer has a causal mask, as in GPT-2 and Florence-2, a step takes up to 64 tokens of the prompt, as for the other decoders. Otherwise, as in Whisper and Marian, a step takes one token of each prompt (see `Transmuter.Generation.MergedDecoder`). So each branch has the widths of its steps.

  Otherwise, as with beam search, the prompts of a batch run on the branch without cache as one step for each prompt length, from the longest to the shortest, and each row takes its first token from the step of its prompt.

  A merged decoder without `position_ids` takes the length of its cache from the shape of the past, so with a buffer it sees the capacity of the buffer and not the valid rows. `Transmuter.Generation.MergedDecoder` gives the branch with cache the valid rows where the positions need them, and a mask of the free rows, so such a decoder runs in `:append`. Other merged decoders stay in `:grow`. A decoder without an `If`, such as Apertus, whose rotary positions come from the length of its past, gets the same rewrite on its graph. Its GroupQueryAttention hides its own free rows, so it takes only the offset and runs in `:in_place`. The valid rows have one value for each row, as in GPT-2, T5 and Apertus, so the rows of prompts of different lengths run together, and the steps of the other prompt lengths leave gap rows in each row. A group takes the next prompt length while the sum of its prompt lengths and the new tokens fits the static tables of the decoder, else the length starts a new group. A decoder whose positions cannot take an offset for each row stays in `:grow`.

  The decoder of an encoder-decoder model also takes the outputs of the encoder, such as `encoder_hidden_states`, in each step. Its branch without cache computes the cross-attention cache (`past_key_values.N.encoder.key` and `.value`) from these outputs. The branch with cache gives an empty present for this cache, so the cache keeps its value.

  When requests can join, each row keeps its own outputs of the encoder and its own cross-attention cache. The cache of a row comes from its first step. When the row joins rows that run, that step is on the branch with cache, so the cross function of `Transmuter.Generation.MergedDecoder.cross/1` computes the cache of the request alone when it joins. Otherwise the first step is on the branch without cache, whose presents give the cache of each row from its outputs of the encoder, with no other call. A decoder whose cross-attention cache reads other values, such as the tokens, has no cross function, and its requests cannot join. The rows of a group have the same number of encoder rows, the encoder capacity. The capacity starts at the encoder rows of the first rows of the group, a later row with more rows grows it to the encoder capacity of its rows, and a row with fewer rows pads to the capacity of the group. So `generate/5` runs at the exact encoder rows of its batch, with no pad. The rows of zeros of a pad need the encoder mask to hide them, so the rows of a group need the same number of encoder rows when the branch with cache does not read the mask, as in Whisper and TrOCR. When it does, as in Marian, M2M100 and Florence-2, the encoder capacity of a row is a power of two of at least 32 rows, up to the longest output of the encoder in the configuration. The source lengths of the servings, 32, 128 and 512 tokens, are such capacities. The pad of the outputs of another length runs on the host, because a device op compiles for each length on EXLA. The capacity does not shrink, so the steps compile for a few shapes.

  The token selection applies the logits processors of Python transformers that `generation_config.json` sets, in the order of `_get_logits_processor`: `repetition_penalty`, `no_repeat_ngram_size`, the bad words of one token in `bad_words_ids`, `min_length` or `min_new_tokens`, `forced_bos_token_id`, `forced_eos_token_id`, `suppress_tokens`, `begin_suppress_tokens`, the processors of the task, `renormalize_logits`, `temperature`, `top_k` and `top_p`.

  ## Lengths

  The processors count the tokens of each row as Python transformers counts the columns of `input_ids`: the prompt and the new tokens. The prompt of a decoder-only model is the text, and the prompt of the decoder of an encoder-decoder model starts with `decoder_start_token_id`. Each row has its own length, and the rows of a batch are not padded. Transmuter limits the new tokens of each row with its budget, `max_new_tokens` or the `:max_new_tokens` of its request in `Transmuter.ContinuousBatching`, so the `max_length` of a row in Python is the length of its prompt plus its budget. Thus:

    * `no_repeat_ngram_size` - the n-grams of the prompt and the new tokens can occur only once.
    * `min_length` - the eos tokens are removed while the prompt and the new tokens are fewer than `min_length`.
    * `min_new_tokens` - the eos tokens are removed while the new tokens are fewer than `min_new_tokens`. As in Python, `min_new_tokens` replaces `min_length` when both are set.
    * `forced_bos_token_id` - the token when the row has one token, which is the first new token of an encoder-decoder model. A decoder-only prompt of more than one token gets no forced token, as in Python.
    * `forced_eos_token_id` - the token when the row has `max_length - 1` tokens, which is the last new token.
    * `begin_suppress_tokens` - removed at the first new token, or at the second new token after a forced bos token of a prompt of one token.

  ## Embeddings and image features

  The decoder of a vision language model, such as SmolVLM or Qwen2-VL, declares `inputs_embeds` in place of `input_ids`. Its context has the model of the embeddings in `:embed` (`embed_tokens.onnx`) and the id of the image token in `:image_token`. Each step turns its tokens into embeddings with that model, so the steps are the same as for `input_ids`.

  The task gives the features of the images to `generate/5` in `:image_features`: for each prompt, the features of its image tokens in order, as `masked_scatter` in Python transformers uses them, or `nil`. In the steps of the prompt, the features replace the embeddings of the image tokens, and a cursor of each row counts the image tokens that the row used. Parts of the prompt without image tokens and the steps of new tokens only embed their tokens. A new token takes no features, also when it is the image token, as in Python transformers.

  Each row of a group has a slot for the features of its request, and a request that joins writes its features at the start of the slot of its row. A slot has the rows of the power of two of the largest capacity (see `context/4`), so for 8 rows and a capacity of 4096 rows the slots take about 75 MB for SmolVLM-256M and about 200 MB for Qwen2-VL-2B in f32, while a row reads features. The features of a prompt have a power of two of rows, with rows of zeros after them. So the writes and the merge of the features compile for few shapes, and a group of a server, where requests join at any time, keeps one shape of features.

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

  In `:in_place` and `:append`, all rows share one buffer. In `:in_place`, each row writes its keys and values after its own valid rows, so the padding tokens of a step go into rows that the next step overwrites. A row that ended has no valid rows, so the padding of its steps goes into its first rows, and it needs no rows of the buffers, as in `:append` after a compaction. The attention mask of a row has ones for its valid rows and the new tokens, and GroupQueryAttention gets the write position of each row from the sum of the mask. In `:append`, all rows of a step write their new rows at one offset: the sum of the lengths of the earlier steps. So one update of each buffer writes the step, in place of one update for each row. A row with fewer tokens than the step, such as a shorter part of a prompt or a finished row, leaves gap rows after its tokens. The host keeps the valid rows of each row, and the attention mask has ones only for them and for the new tokens, so it hides the gap rows and the free rows. Both modes give the positions of each row in `position_ids`, from the tokens of the row and not from its rows in the buffer.

  The buffer starts with a capacity that fits the rows of the prompt steps and 16 more rows: the smallest capacity, or above it a multiple of a step. The step is a quarter of the power of two below the rows, at least 64 rows: 64 rows up to 512, 128 rows up to 1024, 256 rows up to 2048. These rows are the longest prompt, or with a merged decoder the sum of the prompt lengths of the group. When a step does not fit, the buffers grow to the next capacity, and the steps compile for it. Each step reads the whole buffers, so the small steps keep them near the rows that the longest row needs: a row of 300 rows runs its steps at 320 rows, not at 512. The step grows with the rows, so a long answer compiles four capacities for each doubling of its rows: the first answer of 1024 tokens compiles 12 shapes in place of 6 with powers of two, about 13 s more on CUDA. A process compiles each capacity one time, so later answers do not pay it. The smallest capacity is 256 rows, so the first request compiles the steps one time for most answers, and an answer of 600 tokens to a short prompt compiles them at 256, 320, 384, 448, 512 and 640 rows. A decoder with the position offset of `Transmuter.Generation.MergedDecoder` starts at 32 rows. In `:grow`, such a decoder compiles a shape for each length of its cache, so a few compiles of capacities cost little against that, and its steps are faster with a smaller buffer, as in Whisper and in Marian with beam search. The capacity of a merged decoder stops at the rows of its static tables. In `:append`, the rows of all steps of a batch count, also after a row ends, so a batch whose longest row ends early can need more rows than its longest row, grow one step earlier, or pass the rows of the static tables.

  In `:grow`, the positions come from the length of the cache, which is the same for all rows. So only prompts of the same length share a cache. The prompts of each length run as one group, and the groups run one after the other. A group runs with all rows of the batch, and the rows of other lengths are padding rows of the group. So each length of the cache has one compiled shape, and a batch with prompts of three lengths runs the steps of three groups. Text-to-text, image-to-text and speech-to-text have prompts of one length, so all their rows run as one group.

  ## Logits

  Generation reads the logits of one position of each row, at its last token. `context/4` puts a GatherND before the lm_head, the MatMul or MatMulNBits that gives `logits`: it takes the hidden state of one position of each row, from the input `transmuter.logit_index`. So the rows of a step can end their tokens at different positions, and the lm_head computes one position for each row. A decoder that declares `num_logits_to_keep`, such as Gemma 3, gets the static input 0, all positions, and the same gather. Between the lm_head and `logits`, the graph can have only operators that act on each position alone (Cast, Identity, Tanh, and Add, Sub, Mul or Div with one input that is not static) and `If` nodes, such as the `use_cache_branch` of a merged decoder. Each branch of an `If` gets its own gather before its lm_head, so the `If` gives the same shape from each branch. Each value on the way to `logits` has no other use. At another operator, such as a Reshape or the dynamic quantization of an lm_head, the graph stays as it is, the decoder computes the logits of all positions, and the selection reads the position of each row. `Transmuter.load!/2` does not change the graph.

  When requests can join a group with other rows (`:in_place` and `:append` with greedy selection or sampling, and for a merged decoder an offset for each row and, with a cross-attention cache, a cross function), a prompt runs in the steps of the other rows. This is also the path of `generate/5` and of the servings of these decoders. Each step takes a part of each prompt from its front, and the new token of each other row, which takes the logits at its own position. A step has one of five widths: 1 for new tokens, 16, 32, 64 and 512 for prompt parts, and a shorter part pads to the smallest width that holds the longest part of the step. A decoder whose short prompts run in steps of their own rows (see below) has no widths 16 and 32, because its prompts join the steps of the group only when they are long or when all new rows run together. Each step runs all rows of the group at its width, so its time grows with the width: for Qwen3-0.6B fp16 with 48 rows on CUDA (RTX 3060), a step takes 13 ms at width 1, 49 ms at 16, 88 ms at 32 and 165 ms at 64. A part takes the largest width that the longest prompt left fills and that the largest capacity holds after the longest row that has not ended, at least 64, and at most 64 when another row takes a new token. A row can end early, at an eos token or a cancel, and then a step can be wider than the steps that the admission of a request planned. A step of 64 tokens always fits the rows that the admission counted, and a wider step fits the room that is left. So the other rows wait at most one step of 64 tokens for a token, and a group has at most five step shapes for each capacity, whatever the timing of the requests (see `Transmuter.ContinuousBatching`). A prompt of 37 tokens runs in one step, and a prompt of 4000 tokens in an idle group in seven steps of 512 tokens and seven of at most 64. Against parts of each power of two, the time to the first token of one prompt alone is 0.80x for 100 tokens, 1.09x for 874 tokens and 1.18x for 4000 tokens on the GPU of a Mac, and 0.56x, 0.94x and 1.20x on CUDA, for SmolLM2-135M. In `:append`, all rows of a step write its width, so a join step of 64 tokens needs 64 free rows after the write offset, and a group can grow its buffers one capacity step earlier than with parts of each power of two. The other decoders run all prompts in parts that end all prompts in the same step, with no other rows (see "Merged decoders" and beam search).

  A step of a group runs all its rows at its width, also the rows that only take a new token, so a step of a short prompt in a group of many rows costs the width for each row: for Qwen3-0.6B fp16 with 48 rows on CUDA (RTX 3060), 49 ms at width 16 against 13 ms at width 1. So in `:in_place`, unless `alone_prompts: false`, when the new requests of an admission are fewer than the rows of the group in a power of 4, each prompt of at most 511 tokens runs whole in one step of the new rows alone. That step runs in a group of a power of 4 of rows, with the stored prefix of each row as its past (see "Prefixes"), at the smallest width of 16, 64, 256 and 512 tokens that holds the rest of each prompt and its first token. Its first token is read, and the cache, the selection state and the tokens of each row move into its row of the group, which starts at the first row of the buffers. The other rows take their next token in the next step of the group. On a mix of 512 ShareGPT prompts of 20 to 512 tokens with 16 rows of Qwen3-0.6B fp16 on CUDA, the prompt steps took 22 s against 58 s as joins, and the time for each output token was 15.7 ms against 20.6 ms (p50). A longer prompt, the prompts of `:append`, whose rows share one write offset, and the prompts of a merged decoder whose rows keep their own encoder outputs run in the steps of the group as above.

  ## Decode loops

  With `decode_steps: k` above 1 in the options of `context/4`, `compile/3` also compiles decode loops for `:in_place` and `:append` with greedy selection or sampling, without beams. A loop runs the decoder and the selection for 2, 4 ... up to `k` steps in one call, in a while loop with a static trip count (the `:repeat` option of `Transmuter.compile/3`). So the cost of a call on the host and in the launch of the program comes once for these steps. `:grow` has no loops, because its cache grows in each step, and the carry of a while loop keeps its shape. A decoder with `:embed` runs the nodes of the embedding model in the loop, before its own nodes.

  In the loop, each step makes the inputs of the next step on the device: the new token, the attention mask, the positions and the position offset come from the length of each row, and in `:append` the write offset moves by one row. A row stops at an eos token or when it has no new tokens left. A row that stopped still runs in the later steps of the call: it writes no token, and its length does not move. The call gives the selection of each step, with -1 for a row that stopped in an earlier step, and the host reads the steps in order, with the work of one step for each of them. On the EXLA host, the loops give the same tokens and log-probabilities as one step for each call. On CUDA in fp16, the loop is another compiled program, so its log-probabilities can move slightly, as with any change of the program: up to 0.013 at 256 tokens for Qwen3-0.6B. So a near tie can give another token.

  `next/3` runs a loop only in a step where each running row takes its new token, with no part of a prompt left. It takes the largest loop of at most `k` steps, at most the new tokens that each running row has left, and at most the tokens of the newest row. So a row that reaches its budget frees its row at the end of the call, and a new row gets its first tokens in calls of 1, 2, 4 ... steps. A text stream gives a chunk only at the end of a word, so the first chunk of a stream does not wait for a long loop. A row that stops at an eos token runs the rest of the call.

  EMLX 0.5 cannot compile the loop of a model, so `compile/3` raises for `:decode_steps` above 1 with `compiler: EMLX`.

  ## Device memory

  A step makes large device tensors that die at the next step: the selection of the step, the present cache, and in `:append` the old buffers. The garbage collector of the BEAM does not see the size of a device buffer, so it frees them late, and a request of 24 tokens from a prompt with an image kept 13 GB of dead buffers on the EXLA CPU. So each step frees the tensors that it owns with `Nx.backend_deallocate/1`: the selection of the last step after it is read, the present after it is in the cache, and the old buffers of the cache and of the reorder of beam search. A compiler with donation, such as EXLA, writes the present of `:in_place` and `:append` into the buffers of the step before, and the free of a donated buffer does nothing. A present that shares a buffer with a kept tensor stays, as when a merged decoder gives a past back. The embeddings of a step are freed after its call, and when a group ends, generation frees its cache and the rest of its state. Generation never frees the tensors of the caller, such as the outputs of an encoder or the image features.

  A backend that frees a released buffer only when the BEAM collects its tensor, such as EMLX 0.5 (see elixir-nx/emlx#159), still keeps the dead caches of the steps until a collection. A loop with little host garbage collects rarely, so these caches pile up. So the generation process collects its garbage after its releases free 256 MB of device buffers. A small cache then rarely costs a collection, and a large one gets one before its dead copies take much memory. A buffer that a call took by donation is not freed again, so the donated cache of each EXLA step costs no collection.

  ## Prefixes

  A request of `admit/4` can reuse the keys and values of a prefix of its prompt from a store of `prefix_store/3`, such as the system prompt of a chat. The key of a token depends only on the tokens before it and on its position, not on the row or the column of the buffer that holds it, so the rows of a prefix hold the same keys and values in each row. The request gives `%{store: store, slot: slot, size: size}` in `:prefix`: its row starts with the first `size` rows of the slot, from its first row in `:in_place`, and in `:append` at the write offset of the group, which then moves after them as after a prompt part. The steps run only the rest of its prompt. `store_prefix/6` copies the first valid rows of a row into a slot after its prompt ran. A prefix has whole blocks of `prefix_block/0` tokens and leaves at least the last token of the prompt, which gives the logits. `prefix_reason/1` tells why a decoder cannot reuse prefixes: rows that cannot join a group, a merged decoder, inputs of the steps other than the tokens, or a state that is not a cache of keys and values.
  """

  @compile {:no_warn_undefined, Transmuter.EMLX}

  alias Nx.Defn.Composite
  alias Transmuter.Generation.MergedDecoder
  alias Transmuter.{GraphBreaks, Ops, Serving}
  alias Transmuter.Onnx.{AttributeProto, NodeProto}
  alias Transmuter.Ops.Transformer

  require Logger

  # The names of the extra inputs of the selection and of its output. The prefix keeps them
  # apart from the names of the graph.
  @offset "transmuter.select.offset"
  # The position of the last token of each row in a step, see keep_last_logits/1.
  @logit_index "transmuter.logit_index"
  # The most tokens of a prompt in a step where other rows take a new token, see plan/1.
  @join_tokens 64

  # The widths of the steps of a group: 1 for new tokens, the powers of two from 16 to
  # @join_tokens for the shorter parts of the prompts, and 512 for the long parts in an idle
  # group. Each step runs all rows of the group at its width, so its time grows with the
  # width: a short prompt runs in a step of about its own length, not of @join_tokens. See
  # step_width/1.
  # The widths of the step of the prompts of new rows alone, see admit_alone/3: powers of 4 from
  # 16, then 512, so a prompt pads to at most 4 times its tokens and few shapes compile.
  @alone_widths [16, 64, 256, 512]

  @step_widths [1] ++
                 for(exponent <- 4..trunc(:math.log2(@join_tokens)), do: 2 ** exponent) ++ [512]

  # The widths of the steps on the branch without cache of a merged decoder: the powers of two up
  # to @join_tokens, then the wider steps of @step_widths. That branch always has the causal
  # mask, so a prompt of up to @join_tokens tokens runs in one step at about its own length, and
  # a longer prompt runs in the parts of the other steps. A step of more tokens costs more to
  # compile (about 1.7 s for 64 tokens and 2.5 s for 256 on CUDA for Marian), so the widths
  # between @join_tokens and 512 are not steps.
  @first_widths for(exponent <- 0..trunc(:math.log2(@join_tokens)), do: 2 ** exponent) ++
                  Enum.filter(@step_widths, &(&1 > @join_tokens))
  @seen "transmuter.select.seen"
  @mask "transmuter.select.mask."
  @history "transmuter.select.history"
  @count "transmuter.select.count"
  @prompt_count "transmuter.select.prompt_count"
  @budget "transmuter.select.budget"
  @keys "transmuter.select.keys"
  @scores "transmuter.select.scores"
  @tokens "transmuter.select.tokens"
  @selected "transmuter.select.selected"
  @next_seen "transmuter.select.next_seen"
  @next_history "transmuter.select.next_history"
  @next_count "transmuter.select.next_count"
  @write_at "transmuter.cache.write_at"
  # The carry of the decode loop (see compile_loop/4).
  @lengths "transmuter.loop.lengths"
  @active "transmuter.loop.active"
  @remaining "transmuter.loop.remaining"
  @delta "transmuter.loop.delta"
  @out "transmuter.loop.out"

  @no_vocab "generation needs the vocabulary size: the logits output of the graph has no number in its last dimension, and the configuration has no vocab_size"

  # The inputs that MergedDecoder.rewrite/2 adds.
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

    model =
      model
      |> keep_last_logits()
      |> MergedDecoder.rewrite(List.last(@step_widths))
      |> count_rows()

    mode = mode(model, given, opts[:embed])
    model = exact_length(model, mode)

    %{
      model: model,
      vocab: vocab(model, repo.config),
      mode: mode,
      embed: opts[:embed],
      image_token: opts[:image_token],
      cache: Enum.filter(model.inputs, &String.starts_with?(&1, "past_key_values.")),
      query_heads: query_heads(model, repo.config),
      matmul_bytes: matmul_bytes(model, vocab(model, repo.config)),
      buffers: buffers(model),
      min_capacity: min_capacity(model),
      max_capacity: model[:max_capacity],
      widths: %{with_cache: widths(model), without_cache: @first_widths},
      given: given,
      one_length: model[:one_length],
      cross: MergedDecoder.cross(model),
      pad_encoder?: "encoder_attention_mask" in MergedDecoder.cache_reads(model),
      encoder_length: encoder_length(repo.config),
      max_new_tokens: max_new_tokens,
      sampling?: sampling?,
      token_logprobs?: Keyword.get(opts, :token_logprobs, false),
      alone_prompts: Keyword.get(opts, :alone_prompts, true),
      decode_steps: Keyword.get(opts, :decode_steps, 1),
      temperature: opts[:temperature] || gen["temperature"] || 1.0,
      top_k: opts[:top_k] || gen["top_k"] || 50,
      top_p: opts[:top_p] || gen["top_p"] || 1.0,
      repetition_penalty: opts[:repetition_penalty] || gen["repetition_penalty"] || 1.0,
      renormalize?: gen["renormalize_logits"] == true,
      masks: masks(gen),
      eos: eos,
      bos: gen["bos_token_id"] || repo.config["bos_token_id"],
      processors: processors(gen, opts, eos)
    }
    |> Map.merge(beam_search(gen, opts, sampling?))
    |> then(&put_in(&1.widths.with_cache, join_widths(&1, &1.widths.with_cache)))
  end

  # The query heads of the attention: the largest head count of the attention nodes of the
  # graph, else the attention heads of the decoder in the config, else nil.
  defp query_heads(model, config) do
    attributes = %{
      "GroupQueryAttention" => "num_heads",
      "MultiHeadAttention" => "num_heads",
      "Attention" => "q_num_heads"
    }

    keys = ["num_attention_heads", "decoder_attention_heads", "num_heads", "n_head"]
    config = Map.merge(config, config["text_config"] || %{})

    heads =
      for node <- model.nodes,
          name = attributes[node.op_type],
          heads = name && Transmuter.Ops.attr(node, name),
          do: heads

    Enum.max(heads, fn -> Enum.find_value(keys, &config[&1]) end)
  end

  # The bytes of one token of the widest output of a matmul of the decoder, other than the
  # lm_head, which computes one position of each row (see "Logits"). A MatMul weight is
  # {K, N}, and the moved weight of a MatMulNBits is {N, K}.
  defp matmul_bytes(model, vocab) do
    sizes =
      for node = %{op_type: op_type} <- model.nodes,
          op_type in ["MatMul", "MatMulNBits"],
          weight = model.params[Enum.at(node.input, 1, "")],
          match?(%Nx.Tensor{shape: {_, _}}, weight),
          size = elem(weight.shape, if(op_type == "MatMul", do: 1, else: 0)),
          size != vocab,
          do: size * Nx.byte_size(Nx.tensor(0, type: Nx.type(weight)))

    Enum.max(sizes, fn -> 0 end)
  end

  # The length of the past comes from the mask only in :grow (see MergedDecoder.past_length/1),
  # after count_rows/1, which would take a decoder whose past only GroupQueryAttention reads to
  # :in_place.
  defp exact_length(model, :grow), do: MergedDecoder.past_length(model)
  defp exact_length(model, _mode), do: model

  # The widths of the steps on the branch with cache (see plan/3): one token for a merged decoder
  # whose branch with cache has no causal mask (see "Merged decoders").
  defp widths(%{causal: false}), do: [1]
  defp widths(_model), do: @step_widths

  # The widths of the steps of a group whose new prompts run in steps of their own rows (see
  # admit_alone/3): a prompt joins the steps of the group only when it is longer than those
  # steps take, or when all new rows run together, so its parts need no width below
  # @join_tokens. The steps of the group then have fewer shapes to compile.
  defp join_widths(ctx, widths) do
    if runs_alone?(ctx),
      do: Enum.reject(widths, &(&1 > 1 and &1 < @join_tokens)),
      else: widths
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
  defp processors(gen, opts, eos) do
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
      # The row has max_length - 1 tokens, where max_length is the prompt and the budget of
      # new tokens of the row.
      forced_eos != nil &&
        forced(&Nx.equal(generated(&1), Nx.subtract(&1.budget, 1)), List.wrap(forced_eos)),
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
            [@position_offset, @key_mask, @logit_index]

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
      # A decoder that reads its cache or its length where MergedDecoder cannot give the
      # offset of each row keeps a cache of exact length, whatever its GroupQueryAttention
      # reads, so MergedDecoder.past_length/1 gives it the true past or raises.
      model[:one_length] ->
        :grow

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

  # Gives the lm_head the hidden state of one position of each row: a GatherND before it takes
  # the position of each row from @logit_index, {batch, 1, 1}. So each row of a step can end
  # its tokens at another position, and the lm_head still computes one position for each row.
  # Generation reads the rank-3 logits, and the operators after the lm_head keep the shape, so
  # axis 1 of its input is the sequence. The values from the lm_head to the logits have no
  # other use, so no other value changes. An If gives each branch the shape of its output, so
  # each branch that gives the logits gets its own cut, as in the merged decoders of optimum.
  # An export that declares num_logits_to_keep, such as Gemma 3, keeps it and gets 0, all
  # positions, so the gather sees each position.
  defp keep_last_logits(model) do
    static = MapSet.new(Map.keys(model.params) ++ Map.keys(model.static))

    case follow(scope(model.nodes, model.outputs, static), "logits", nil) do
      {:ok, lm_heads} ->
        gather_hidden(model, MapSet.new(lm_heads))

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

  defp gather_hidden(model, lm_heads) do
    %{
      model
      | nodes: cut(model.nodes, lm_heads, &(@logit_index <> "." <> &1)),
        inputs: model.inputs ++ [@logit_index],
        input_types: Map.put(model.input_types, @logit_index, {:s64, ["batch_size", 1, 1]})
    }
  end

  # Puts a GatherND of the hidden states before each lm_head, also in the branches of If
  # nodes, which read @logit_index from the outer graph. The nodes are found by value,
  # because a branch can give its value the name of the If output.
  defp cut(nodes, lm_heads, name) do
    Enum.flat_map(nodes, fn node ->
      if MapSet.member?(lm_heads, node) do
        [hidden | rest] = node.input
        kept = name.(hidden)
        batch = %AttributeProto{name: "batch_dims", type: :INT, i: 1}

        [
          %NodeProto{
            op_type: "GatherND",
            name: kept,
            input: [hidden, @logit_index],
            output: [kept],
            attribute: [batch]
          },
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
        reason =
          "it keeps a cache of exact length, because #{where(node)} reads the cache or its length"

        Logger.debug("generation: #{reason}")
        Map.put(model, :one_length, reason)
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

    if ctx.decode_steps > 1 and GraphBreaks.compiler(defn_options) == EMLX do
      raise ArgumentError,
            ":decode_steps above 1 needs a compiler that compiles a while loop of the model, such as EXLA, and EMLX 0.5 cannot"
    end

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
      |> Map.put(:compile_concurrency, compile_concurrency(defn_options))
      |> Map.put(:run_steps, compile_loop(ctx, state, opts, batch_size))
      |> Map.put(
        :loop_model,
        ctx.model |> loop_model(ctx.embed) |> Map.take([:inputs, :input_types])
      )
      |> compile_embed(batch_size, defn_options)
      |> compile_cross(defn_options)
      |> Map.put(:model, Map.take(ctx.model, [:inputs, :input_types]))

    ctx
    |> Map.put(:grow, state_jit(&grow_rows/2, defn_options))
    |> Map.put(:compact, state_jit(&compact_rows/2, defn_options))
    |> Map.put(:reorder, state_jit(&reorder_rows/2, defn_options))
    |> Map.put(:admit_rows, state_jit(&write_admitted/3, defn_options))
    |> Map.put(:write_row, state_jit(&write_row/3, defn_options))
    |> Map.put(:take_row, state_jit(&take_row/2, defn_options))
    |> Map.put(:widen, state_jit(&widen_rows/2, defn_options))
    |> Map.put(:prefix_in, state_jit(&prefix_in/3, defn_options))
    |> Map.put(:prefix_out, state_jit(&prefix_out/3, defn_options))
  end

  # The decode loops: for each count of loop_steps/1, that many steps of width 1 in one call,
  # with the selection of each step in the graph (see next/3). Only :in_place and :append with
  # greedy selection or sampling, without beams, have them. :grow cannot: its cache grows
  # in each step, and the carry of a loop keeps its shape. The call returns the buffers, the
  # selection state and the selection of each step, [steps, rows, 1 or 2], with the eos token
  # of a row that ends and -1 for a row that ended before the step.
  defp compile_loop(ctx, state, opts, batch_size)
       when ctx.decode_steps > 1 and ctx.mode in [:in_place, :append] and ctx.num_beams == 1 do
    ctx = %{ctx | model: loop_model(ctx.model, ctx.embed)}
    Map.new(loop_steps(ctx), &{&1, compile_loop(ctx, state, opts, batch_size, &1)})
  end

  defp compile_loop(_ctx, _state, _opts, _batch_size), do: nil

  defp compile_loop(ctx, state, opts, batch_size, k) do
    tokens = List.duplicate([0], batch_size)
    counts = List.duplicate(1, batch_size)
    state = %{state | lengths: List.duplicate(1, batch_size)}
    {inputs, static} = step_inputs(ctx, state, tokens, counts)
    names = ctx.model.inputs -- Map.keys(Map.merge(inputs, static))
    selection = selection_template(ctx, batch_size, state.capacity)
    carry = loop_carry(ctx, state, counts, List.duplicate(0, batch_size), k)
    returns = ctx.buffers ++ [@seen, @history, @count, @out]

    run =
      Transmuter.compile(
        ctx.model,
        ctx.model
        |> templates(names, batch_size)
        |> Map.merge(inputs)
        |> Map.merge(selection)
        |> Map.merge(carry),
        [
          static_inputs: static,
          repeat: {{:decode, ctx.eos}, k, loop_next(ctx), returns},
          donate: ctx.buffers
        ] ++ opts
      )

    fn inputs, _static -> run.(inputs) end
  end

  # The decoder of the loops. A decoder that takes inputs_embeds gets the nodes of the
  # embedding model before its nodes, so input_ids goes to inputs_embeds in the graph. The
  # names of the embedding model get a prefix, so they do not meet the names of the decoder.
  # The tokens of a decode step are never image tokens with features (see embed/4), so the
  # loop needs no merge of image features.
  defp loop_model(model, nil), do: model

  defp loop_model(model, embed = %{inputs: [input], outputs: [output]}) do
    prefix = "transmuter.embed."

    rename = fn
      ^input -> "input_ids"
      "" -> ""
      name -> prefix <> name
    end

    nodes =
      for node <- embed.nodes,
          do: %{node | input: Enum.map(node.input, rename), output: Enum.map(node.output, rename)}

    {type, _dims} = model.input_types["inputs_embeds"]
    to = Enum.find([1, 10, 16, 11], &(Ops.dtype(&1) == type))

    cast = %NodeProto{
      op_type: "Cast",
      name: prefix <> "cast",
      input: [rename.(output)],
      output: ["inputs_embeds"],
      attribute: [%AttributeProto{name: "to", type: :INT, i: to}]
    }

    renamed = &Map.new(&1, fn {name, value} -> {rename.(name), value} end)

    %{
      model
      | nodes: nodes ++ [cast | model.nodes],
        params: Map.merge(model.params, renamed.(embed.params)),
        static: Map.merge(model.static, renamed.(embed.static)),
        inputs: ["input_ids" | model.inputs -- ["inputs_embeds"]],
        input_types:
          model.input_types
          |> Map.delete("inputs_embeds")
          |> Map.put("input_ids", embed.input_types[input])
    }
  end

  # The step counts of the loops: the powers of two below decode_steps, and decode_steps.
  defp loop_steps(%{decode_steps: k}),
    do: Enum.uniq(for(e <- 1..trunc(:math.log2(k))//1, 2 ** e < k, do: 2 ** e) ++ [k])

  # The write offset of :append, which the tail reads (see tail/1).
  defp write_at(%{mode: :append}, state),
    do: %{@write_at => Nx.tensor(state.used, type: :s32, backend: Nx.BinaryBackend)}

  defp write_at(_ctx, _state), do: %{}

  # The carry of the loop for a step of `counts` tokens: the lengths, the rows with a token,
  # the new tokens that each row has left, the deltas of the positions and the tokens out.
  defp loop_carry(ctx, state, counts, remaining, k) do
    rows = length(counts)
    width = if ctx.token_logprobs?, do: 2, else: 1
    s32 = &host_tensor(runs(&1), :s32)
    none = Nx.tensor(-1, type: :s32, backend: Nx.BinaryBackend)

    %{
      @lengths => s32.(state.lengths),
      @active => s32.(counts),
      @remaining => s32.(remaining),
      @delta => s32.(Enum.map(state.positions, &elem(&1, 1))),
      @out => Nx.broadcast(none, {k, rows, width})
    }
  end

  # The inputs of the next step of the loop, from the outputs of the tail of step `i` and the
  # inputs of step `i`. A row stops at an eos token or when it has no new tokens left. A row
  # that stopped still runs, without a token: it writes nothing in its tokens, and its length
  # does not move.
  defp loop_next(ctx) do
    eos = ctx.eos
    {mode, buffers} = {ctx.mode, ctx.buffers}

    fn outputs, c, i ->
      selected = outputs[@selected]
      token = selected |> Nx.slice_along_axis(0, 1, axis: 1) |> Nx.squeeze(axes: [1])
      eos? = Enum.reduce(eos, Nx.equal(token, -2), &Nx.logical_or(&2, Nx.equal(token, &1)))
      running = Nx.not_equal(c[@active], 0)
      emitted = Nx.logical_and(running, Nx.logical_not(eos?))
      out = Nx.select(Nx.broadcast(Nx.new_axis(running, 1), Nx.shape(selected)), selected, -1)
      remaining = Nx.subtract(c[@remaining], Nx.as_type(emitted, :s32))
      lengths = Nx.add(c[@lengths], c[@active])
      active = Nx.logical_and(emitted, Nx.greater(remaining, 0))
      new_token = Nx.select(emitted, token, 0)
      capacity = Nx.axis_size(c[@history], 1)
      mask = next_mask(mode, c, lengths, running)
      positions = Nx.add(lengths, c[@delta])

      values = %{
        "input_ids" => Nx.new_axis(new_token, 1),
        "attention_mask" => mask,
        @key_mask => mask,
        "position_ids" => positions,
        @position_offset => Nx.new_axis(Nx.subtract(lengths, capacity), 1),
        @tokens => Nx.select(emitted, token, -1),
        @seen => outputs[@next_seen],
        @history => outputs[@next_history],
        @count => outputs[@next_count],
        @lengths => lengths,
        @active => Nx.as_type(active, :s32),
        @remaining => remaining,
        @out => Nx.put_slice(c[@out], [i, 0, 0], Nx.new_axis(out, 0)),
        @write_at => Nx.add(c[@write_at] || 0, 1)
      }

      present =
        for "past_key_values." <> rest = name <- buffers,
            Map.has_key?(outputs, "present." <> rest),
            into: %{},
            do: {name, outputs["present." <> rest]}

      updates =
        for {name, value} <- Map.merge(values, present), Map.has_key?(c, name), into: %{} do
          template = c[name]
          value = Nx.as_type(value, Nx.type(template))
          {name, Nx.broadcast(value, Nx.shape(template), axes: broadcast_axes(value, template))}
        end

      Map.merge(c, updates)
    end
  end

  # The mask of the next step. In :in_place, each row sees its rows and the new one. In :append,
  # the column of the write offset becomes valid for the rows that had a token, and the new
  # token has the last column.
  defp next_mask(:in_place, c, lengths, _running) do
    {rows, capacity} = Nx.shape(c[@history])
    Nx.less(Nx.iota({rows, capacity}, axis: 1), Nx.new_axis(Nx.add(lengths, 1), 1))
  end

  defp next_mask(:append, c, _lengths, running) do
    mask = c[@key_mask] || c["attention_mask"]
    {rows, columns} = Nx.shape(mask)
    past = Nx.slice_along_axis(mask, 0, columns - 1, axis: 1)
    written = Nx.equal(Nx.iota({rows, columns - 1}, axis: 1), c[@write_at])
    past = Nx.logical_or(Nx.not_equal(past, 0), Nx.logical_and(written, Nx.new_axis(running, 1)))
    Nx.concatenate([past, Nx.broadcast(Nx.tensor(1, type: :u8), {rows, 1})], axis: 1)
  end

  # The axes of `template` that the axes of `value` take: the last ones, so [rows] goes to
  # [rows, 1] at axis 0 and to [3, rows, 1] at axis 1.
  defp broadcast_axes(value, template) do
    rank = Nx.rank(template)

    case Nx.rank(value) do
      ^rank -> Enum.to_list(0..(rank - 1)//1)
      1 when rank == 2 -> [0]
      1 when rank == 3 -> [1]
      2 when rank == 3 -> [1, 2]
    end
  end

  # The jit function of `fun`, whose outputs become the state of a group, such as a row that a
  # request writes. Each compiled function whose outputs become group state evaluates them:
  # the step evaluates its state with its selection (see anchor/2), and this function evaluates
  # the outputs when the call returns. On EMLX 0.5, an output otherwise stays an MLX expression
  # that each later call that reads it computes again, and a later step that read the outputs
  # of write_row/3 this way gave wrong values in some runs, in the joins of Marian.
  defp state_jit(fun, defn_options) do
    jit = GraphBreaks.jit(fun, defn_options)

    case Function.info(fun, :arity) do
      {:arity, 2} -> fn a, b -> evaluated(jit.(a, b)) end
      {:arity, 3} -> fn a, b, c -> evaluated(jit.(a, b, c)) end
    end
  end

  @doc false
  # The tensors of `outputs`, a container, evaluated on EMLX (see state_jit/2).
  def evaluated(outputs) do
    Composite.traverse(outputs, fn
      tensor = %Nx.Tensor{data: %{__struct__: EMLX.Backend}} -> Transmuter.EMLX.evaluate(tensor)
      tensor -> tensor
    end)
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
      row = Map.merge(row, %{prompt: extra[@prompt_count], budget: extra[@budget]})

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
        prompt_count: zeros.(:s32, {rows}),
        budget: zeros.(:s32, {rows})
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
        do: %{@scores => host_tensor(runs(group.scores), :f32)},
        else: %{}

    tokens = Enum.map(group.new, &(&1 || -1))

    %{
      @seen => group.state.seen,
      @history => group.state.history,
      @count => group.state.count,
      @prompt_count => group.state.prompt_count,
      @budget => group.state.budget,
      @keys => group.keys,
      @tokens => host_tensor(runs(tokens), :s32)
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
    |> Map.put(:write_features, state_jit(&write_features/3, defn_options))
  end

  defp compile_cross(ctx = %{cross: nil}, _defn_options), do: ctx

  # The cross function compiles for one request at the smallest encoder capacity, and takes
  # templates of one row and one token for its other inputs, which give only shapes (see
  # MergedDecoder.cross/1).
  defp compile_cross(ctx, defn_options) do
    model = ctx.cross
    others = templates(model, model.inputs -- ctx.given, 1)
    rows = encoder_capacity(ctx, 1)

    given =
      Map.new(templates(model, model.inputs -- Map.keys(others), 1), fn {name, t} ->
        {name, Nx.broadcast(Nx.tensor(0, type: Nx.type(t)), put_elem(Nx.shape(t), 1, rows))}
      end)

    run = Transmuter.compile(model, Map.merge(others, given), [cache_size: 16] ++ defn_options)
    %{ctx | cross: fn inputs -> run.(Map.merge(others, Map.take(inputs, model.inputs))) end}
  end

  defp write_features(slots, features, start) do
    Nx.put_slice(slots, [start, 0], Nx.as_type(features, Nx.type(slots)))
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
  Generates tokens after each prompt of `prompts` with a context from `compile/3`. A prompt is a list of token ids, and an empty prompt is a padding row. `seeds` has the seed of sampling of each row. `inputs` are the inputs of each step that `context/4` names in `given`, with one row for each prompt. When the rows of a merged decoder keep their own encoder outputs (see "Merged decoders"), outputs whose number of rows is an encoder capacity avoid a copy to the host for the pad.

  `inputs` can also have the key `:image_features`, a list with an item for each prompt: `nil`, or a tensor of shape `{count, hidden}` with the features of the image tokens of the prompt in their order. `count` is a power of two, and rows of zeros follow the features. It is not an input of the decoder. See the module documentation.

  `inputs` can also have the key `:position_ids`, a list with the positions of each prompt. The positions of a prompt are a list with the positions of each axis of the decoder's `position_ids`, one for each token. An empty list gives the default positions. See the module documentation.

  `on_token` is `nil` or a function that receives `%{token: tensor, finished?: tensor}` after each step of a group, with one row for each prompt. `:token` is the new token of the row, or -1 when the row has no new token. `:finished?` is 1 when the row has no more tokens. With `token_logprobs: true` in the options of `context/4`, the map also has `:logprob`, the log-probability of the new token, or NaN when the row has no new token.

  Returns the new tokens of each row, without the eos token. A token is the list of its values: `[id]`, or `[id, logprob]` with `token_logprobs: true` in the options of `context/4`. The log-probability of a token is the log-softmax of the scores that the step selects from, at the token. These scores are the logits after the processors, also after the temperature, top-k and top-p of sampling. For beam search, they are the scores of the step before the sum with the score of the beam. This is `compute_transition_scores` of Python transformers with `normalize_logits=True`.
  """
  def generate(ctx, prompts, seeds, inputs \\ %{}, on_token \\ nil) do
    none = Enum.map(prompts, fn _ -> nil end)
    {features, inputs} = Map.pop(inputs, :image_features, none)

    {positions, inputs} =
      Map.pop_lazy(inputs, :position_ids, fn -> Enum.map(prompts, fn _ -> [] end) end)

    extras = %{features: features, positions: positions}

    ids =
      for prompts <- groups(ctx, prompts) do
        if ctx.num_beams == 1 do
          lengths = for prompt <- prompts, prompt != [], do: length(prompt)

          group =
            ctx
            |> open(length(prompts), Enum.sum(part_sizes(ctx, lengths)) + @capacity_margin)
            |> then(&admit(ctx, &1, requests(ctx, prompts, seeds, extras), inputs))
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

  # The positions of the prompt of a request and the delta of the positions after it, which
  # continue from its largest position. Without given positions, the delta is 0.
  defp positions(request) do
    axes = Map.get(request, :position, [])
    {axes, Enum.max(List.flatten(axes), fn -> -1 end) + 1 - length(List.first(axes, []))}
  end

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

  # The rows of the prompts of `lengths` and their new tokens: the rows that the steps of
  # plan/1 write when the prompts can join (see rows_written/2), else the parts of prefill/4.
  defp fits?(ctx, lengths) do
    count = length(lengths)

    rows =
      if joins?(ctx) do
        rows_written(ctx, %{
          left: Enum.map(lengths, &List.duplicate(0, &1)),
          new: List.duplicate(nil, count),
          done: List.duplicate(false, count),
          lengths: List.duplicate(0, count),
          remaining: List.duplicate(ctx.max_new_tokens, count)
        })
      else
        Enum.sum(part_sizes(ctx, lengths)) + ctx.max_new_tokens - 1
      end

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
  # its budget of new tokens, its positions and its image features.
  defp requests(ctx, prompts, seeds, extras) do
    for {{prompt, seed}, row} <- Enum.with_index(Enum.zip(prompts, seeds)), prompt != [] do
      %{
        row: row,
        prompt: prompt,
        seed: seed,
        budget: ctx.max_new_tokens,
        position: Enum.at(extras.positions, row),
        features: Enum.at(extras.features, row)
      }
    end
  end

  @doc false
  # A group of `rows` empty rows, with buffers for `length` rows of cache. An empty row has no
  # request and is done. admit/4 gives requests to rows, and next/2 runs the steps.
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
        prompt_count: zeros.(:s32, {rows}),
        budget: zeros.(:s32, {rows})
    }

    # The rows of the slot of image features of each row: the power of two of the largest
    # prompt that a row can hold. See join_features/3.
    state = Map.put(state, :slot, power_of_two(ctx.max_capacity || length))

    %{
      state: state,
      masks:
        Map.new(token_masks(ctx), fn {name, mask} ->
          {name, Nx.backend_transfer(mask, ctx.backend)}
        end),
      keys: zeros.(:u32, {rows, 2}),
      prompts: List.duplicate([], rows),
      left: List.duplicate([], rows),
      ids: List.duplicate([], rows),
      logprobs: List.duplicate([], rows),
      new: List.duplicate(nil, rows),
      done: List.duplicate(true, rows),
      budgets: List.duplicate(0, rows),
      scores: nil,
      steps: [],
      moves: []
    }
  end

  @doc false
  # Gives each request of `requests` its row. A request is a map with the row, the prompt, the
  # seed and the budget of new tokens. It can have the positions of the prompt (see "Positions"
  # in the module documentation) and the image features of its image tokens.
  # `inputs` are the inputs of the prompts, which only the first requests of a group can have.
  # When the rows keep their own inputs, a request has them in `:inputs`, and inputs whose
  # number of rows is an encoder capacity avoid a copy to the host (see pad_rows/2).
  #
  # When requests can join a group with other rows (joins?/1), the prompts run in the steps
  # of next/2, next to the new tokens of the other rows (see plan/1). In :in_place, unless
  # `alone_prompts: false`, a prompt of fewer than 512 tokens runs here instead, in a step of the new rows
  # alone, when they are fewer than the rows of the group (see admit_alone/3). Otherwise the
  # prompts run here, in parts that end all prompts in the same step, and the first token of each
  # new row is read (see prefill/4): a merged decoder runs a prompt on its branch without cache,
  # which no row with a cache can share, the rows of :grow need one length of the cache, and
  # the beams of all prompts start their search in the same step.
  def admit(ctx, group, requests, inputs \\ %{}) do
    # The group keeps the inputs of the prompts across steps, so they are evaluated as the
    # outputs of state_jit/2.
    inputs = evaluated(inputs)
    pending = Enum.zip_with(group.new, group.done, &if(&2, do: nil, else: &1))

    if not joins?(ctx) and Enum.any?(pending) do
      raise ArgumentError,
            "requests can join a group with other rows only when the rows of a step can have different lengths of the cache, without a merged decoder and without beams"
    end

    if joins?(ctx) do
      {alone, requests} = Enum.split_with(requests, &(inputs == %{} and alone?(ctx, &1)))

      {alone, requests} =
        if alone != [] and power_of_four(length(alone)) < length(group.done) and
             alone_shape(ctx, padded(alone)),
           do: {alone, requests},
           else: {[], alone ++ requests}

      group = ctx |> admit_alone(group, alone) |> then(&admit_joining(ctx, &1, requests, inputs))
      %{group | steps: [admitted_view(group, alone)]}
    else
      ctx |> prefill(group, requests, inputs) |> then(&select(ctx, &1))
    end
  end

  # The prompts of `requests` join the steps of the group (see plan/3).
  defp admit_joining(_ctx, group, [], _inputs), do: group

  defp admit_joining(ctx, group, requests, inputs) do
    {features, cursors} = join_features(ctx, group, requests)

    left =
      Enum.reduce(requests, group.left, fn request, left ->
        List.replace_at(left, request.row, Enum.drop(request.prompt, prefix_size(request)))
      end)

    # The tokens of each prompt go into the rows of its tokens now, and its keys and values
    # in the steps of its parts, after the rows of a stored prefix (see "Prefixes"). The
    # buffers make room for the parts before the writes (see fit_next/2).
    group =
      group
      |> reset_rows(requests)
      |> copy_prefixes(ctx, requests)
      |> then(&fit_next(ctx, %{&1 | left: left}))
      |> then(&admit_rows(ctx, &1, requests))

    # The rows of a merged decoder keep their own inputs: the inputs of the prompts, one row
    # for each row of the group, or else the inputs of each request (see admit_inputs/3).
    {group, inputs} =
      cond do
        not rows_keep_inputs?(ctx) -> {group, Map.merge(group.state.inputs, inputs)}
        inputs == %{} -> {admit_inputs(ctx, group, requests), group.state.inputs}
        true -> {put_in(group.state.encoder_capacity, encoder_rows(%{inputs: inputs})), inputs}
      end

    %{group | state: %{group.state | inputs: inputs, features: features, cursors: cursors}}
  end

  # A request runs its whole prompt in a step of the new rows alone when its prompt and its first
  # token fit the widest of @alone_widths, and its group runs new prompts alone (see
  # runs_alone?/1).
  defp alone?(ctx, request),
    do: length(request.prompt) < List.last(@alone_widths) and runs_alone?(ctx)

  # The groups that run new prompts in steps of their own rows: unless `alone_prompts: false`,
  # rows that join, in :in_place, and that keep no encoder outputs of their own. The layout of the
  # cache decides the mode. In :in_place, each row starts at the first row of the buffers, so a
  # row of the small group moves into its row with no offset. In :append, all rows write at one shared write offset, so
  # a moved row is a block of the capacity of the small group at that offset, and it must fit
  # next to the valid rows of every row. A prompt of a few hundred tokens then needs, for
  # example, 632 rows for a decoder whose static tables hold 512.
  defp runs_alone?(ctx) do
    Map.get(ctx, :alone_prompts, true) and joins?(ctx) and ctx.mode == :in_place and
      not rows_keep_inputs?(ctx)
  end

  # The prompts of `requests` run in one step of a group of their own rows: a power of 4 of
  # rows, with the stored prefix of each row as its past (see "Prefixes"), so the step runs the
  # rest of each prompt. The step costs the tokens of the new rows, and not its width for each
  # row of `group`. The rows after the requests repeat the last request, so the admission writes
  # all rows of the small group. Then the rows of the requests move into their rows of `group`
  # (see move_rows/4), each with the first token of its prompt.
  defp admit_alone(_ctx, group, []), do: group

  defp admit_alone(ctx, group, requests) do
    rows = Enum.with_index(padded(requests), &%{&1 | row: &2})
    count = length(rows)
    {width, capacity} = alone_shape(ctx, rows)
    small = admit_small(ctx, count, capacity, width, rows)
    state = step(ctx, small.state, small.left, selection(ctx, small), width)

    small =
      select(ctx, %{small | left: List.duplicate([], count), state: %{state | features: nil}})

    group = move_rows(ctx, group, small, requests)
    finish(small)
    group
  end

  # The requests of a small group: `requests`, then the last one again up to a power of 4.
  defp padded(requests) do
    count = power_of_four(length(requests))
    requests ++ List.duplicate(List.last(requests), count - length(requests))
  end

  # The width of the step of the prompts of `requests` alone, the smallest of @alone_widths that
  # holds the rest of each prompt after its prefix and its first token, and the capacity of the
  # small group, the smallest of @alone_widths that also holds the longest prefix before that
  # step, up to the largest capacity of the decoder. Nil when no width holds them.
  defp alone_shape(ctx, requests) do
    sizes = Enum.map(requests, &prefix_size/1)
    rest = requests |> Enum.zip_with(sizes, &(length(&1.prompt) - &2)) |> Enum.max()
    widths = Enum.filter(@alone_widths, &(&1 <= (ctx.max_capacity || &1)))

    with width when width != nil <- width_for(rest + 1, widths),
         capacity when capacity != nil <- width_for(Enum.max(sizes) + width, widths),
         do: {width, capacity}
  end

  # A group of `rows` rows whose buffers have `capacity` rows, which admits `requests` for one
  # step of `width` tokens. Its admission makes room for that step and no other (see
  # fit_next/2), so its buffers keep that capacity.
  defp admit_small(ctx, rows, capacity, width, requests) do
    ctx = %{ctx | min_capacity: capacity, max_capacity: capacity}
    ctx = %{ctx | widths: %{with_cache: [width], without_cache: [width]}}
    admit(ctx, open(ctx, rows, capacity), requests)
  end

  # Moves the rows of `small` that ran the prompts of `requests` (row i for request i) into
  # their rows of `group`: the cache, the selection state and the key of each row, and on the
  # host its tokens, length, valid rows and positions. Each row starts at the first row of the
  # buffers, so the buffers of `group` first grow when they do not hold each prompt and its
  # first token.
  defp move_rows(ctx, group, small, requests) do
    group = reset_rows(group, requests)
    room = Enum.max(Enum.take(small.state.lengths, length(requests))) + 1

    group =
      if group.state.capacity < room,
        do: grow(ctx, group, fitting(ctx, room, 0)),
        else: group

    names = [:seen, :history, :count, :prompt_count, :budget]

    own =
      Map.take(group.state, names) |> Map.put(:keys, group.keys) |> Map.merge(group.state.cache)

    values =
      Map.take(small.state, names) |> Map.put(:keys, small.keys) |> Map.merge(small.state.cache)

    # One row in each write, so the writes compile for each capacity of `small` and of `group`,
    # and not for each count of rows.
    written =
      requests
      |> Enum.with_index()
      |> Enum.reduce(own, fn {request, i}, own ->
        row = ctx.take_row.(values, Nx.tensor(i, type: :s32, backend: Nx.BinaryBackend))
        target = Nx.tensor(request.row, backend: Nx.BinaryBackend)
        written = ctx.write_row.(Nx.donatable(own), target, row)
        release(Map.values(own) ++ Map.values(row), Map.values(written))
        written
      end)

    {keys, written} = Map.pop!(written, :keys)
    {cache, written} = Map.split(written, Map.keys(group.state.cache))
    state = %{Map.merge(group.state, written) | cache: cache}

    requests
    |> Enum.with_index()
    |> Enum.reduce(%{group | keys: keys, state: state}, fn {request, i}, group ->
      put_row(group, request.row, small, i)
    end)
  end

  # Row `row` of `group` takes the host state of row `i` of `small`.
  defp put_row(group, row, small, i) do
    take = &Enum.at(&1, i)

    %{
      group
      | ids: List.replace_at(group.ids, row, take.(small.ids)),
        logprobs: List.replace_at(group.logprobs, row, take.(small.logprobs)),
        new: List.replace_at(group.new, row, take.(small.new)),
        done: List.replace_at(group.done, row, take.(small.done)),
        left: List.replace_at(group.left, row, []),
        state: %{
          group.state
          | lengths: List.replace_at(group.state.lengths, row, take.(small.state.lengths)),
            valid: List.replace_at(group.state.valid, row, take.(small.state.valid)),
            positions: List.replace_at(group.state.positions, row, take.(small.state.positions)),
            cursors: List.replace_at(group.state.cursors, row, {0, 0})
        }
    }
  end

  # The view of the steps of an admission: the first tokens of the rows of `requests`, whose
  # prompts ran in admit_alone/3. The other rows have no new token in it.
  defp admitted_view(group, requests) do
    rows = MapSet.new(requests, & &1.row)
    new = for {token, row} <- Enum.with_index(group.new), do: if(row in rows, do: token)
    %{view(group) | new: new}
  end

  @doc false
  # Runs the step of the new token of each row that has not ended, and reads its next token.
  #
  # With `steps` above 1, a group where each row that runs takes its new token runs a decode
  # loop of compile_loop/4: the largest count of loop_steps/1 up to `steps`, up to the new
  # tokens that each running row has left, and up to the tokens of the newest row. So no row
  # reaches its budget before the last step of the call, and its row of the group is free at
  # the end of that step.
  def next(ctx, group, steps \\ 1) do
    case loop_count(ctx, group, steps) do
      nil -> select(ctx, advance(ctx, group))
      k -> next_steps(ctx, group, k)
    end
  end

  defp loop_count(%{run_steps: loops}, group, steps) when steps > 1 and loops != nil do
    running =
      for {ids, budget, false} <- Enum.zip([group.ids, group.budgets, group.done]),
          do: {ids, budget}

    left = for {ids, budget} <- running, do: budget - length(ids)
    # The loop is at most as long as the tokens of the newest row, so the first tokens of a row
    # come in calls of 1, 2, 4 ... steps. A text stream gives a chunk only at the end of a
    # word, so the first chunk does not wait for a long loop.
    newest = for {ids, _budget} <- running, do: length(ids)
    limit = Enum.min([steps | left] ++ newest)

    if decoding?(group),
      do: loops |> Map.keys() |> Enum.filter(&(&1 <= limit)) |> Enum.max(fn -> nil end)
  end

  defp loop_count(_ctx, _group, _steps), do: nil

  # A group where each row that runs takes its new token, with no part of a prompt left.
  defp decoding?(group) do
    Enum.all?(group.left, &(&1 == [])) and
      Enum.any?(Enum.zip(group.new, group.done), &match?({token, false} when token != nil, &1))
  end

  # The decode loop of compile_loop/4 in one call. The host then reads the tokens of each
  # step and does the work of step/5 and select/2 for each step in order.
  defp next_steps(ctx, group, k) do
    group = fit(ctx, drop_ended(group), k)
    state = group.state
    {parts, _left, 1} = next_parts(ctx, group)
    counts = Enum.map(parts, &length/1)
    tokens = Enum.map(parts, &(&1 ++ List.duplicate(0, 1 - length(&1))))
    {inputs, static} = step_inputs(%{ctx | model: ctx.loop_model}, state, tokens, counts)

    remaining =
      Enum.zip_with([group.ids, group.budgets, group.done], fn [ids, budget, done?] ->
        if done?, do: 0, else: budget - length(ids)
      end)

    offsets = host_tensor([{0, length(counts)}], :s32)
    release(List.wrap(state.selected), [])

    inputs =
      inputs
      |> Map.merge(selection(ctx, group))
      |> Map.put(@offset, offsets)
      |> Map.merge(loop_carry(ctx, state, counts, remaining, k))
      |> Map.merge(write_at(ctx, state))

    outputs = ctx.run_steps[k].(inputs, static)
    release([state.seen, state.history, state.count], [])
    present = Map.take(outputs, ctx.buffers)
    release(Map.values(Map.take(state.cache, Map.keys(present))), Map.values(present))

    state = %{
      state
      | cache: Map.merge(state.cache, present),
        written: Map.keys(present),
        selected: outputs[@out],
        seen: outputs[@seen],
        history: outputs[@history],
        count: outputs[@count]
    }

    {views, group} =
      outputs[@out]
      |> Nx.to_list()
      |> Enum.map_reduce(%{group | state: state}, fn selected, group = %{state: state} ->
        counts = Enum.zip_with(group.new, group.done, &if(&1 != nil and not &2, do: 1, else: 0))

        state = %{
          state
          | lengths: Enum.zip_with(state.lengths, counts, &+/2),
            used: state.used + 1,
            valid: Enum.zip_with(state.valid, counts, &add_rows(&1, state.used, &2))
        }

        group = select_rows(ctx, %{group | state: state}, selected)
        {view(group), group}
      end)

    %{group | steps: views}
  end

  @doc false
  # The tokens of a block of a stored prefix: a prompt reuses the keys and values of its first
  # blocks, and the steps of its parts run the rest (see "Prefixes").
  def prefix_block, do: @join_tokens

  @doc false
  # Why the rows of `ctx` cannot reuse the keys and values of a stored prefix, or nil when they
  # can. The cache of a row must start at its first row and hold only the keys and values of
  # the tokens of its prompt, so a copy of them gives the same cache as the steps of the prompt.
  def prefix_reason(ctx) do
    others =
      Enum.reject(ctx.model.inputs, &String.starts_with?(&1, "past_key_values.")) --
        ["input_ids", "attention_mask", "position_ids"]

    state = Enum.filter(others, &String.starts_with?(&1, "past"))

    cond do
      reason = cannot_join(ctx) ->
        reason

      merged?(ctx) ->
        "a merged decoder runs the first part of a prompt on its branch without cache"

      ctx.given != [] ->
        "its steps read inputs other than the tokens: #{inspect(ctx.given)}"

      state != [] ->
        "it keeps a state that is not a cache of keys and values: #{inspect(state)}"

      true ->
        nil
    end
  end

  @doc false
  # The store of prefixes of a server: for each buffer of the cache, `slots` prefixes of
  # `length` rows on the device.
  def prefix_store(ctx, slots, length) do
    zeros = &Nx.broadcast(Nx.tensor(0, type: &1, backend: ctx.backend), &2)

    for name <- ctx.buffers, name in ctx.cache, into: %{} do
      {type, [_batch, heads, _length, dim]} = ctx.model.input_types[name]
      {name, zeros.(type, {slots, heads, length, dim})}
    end
  end

  @doc false
  # Copies the keys and values of the first `size` rows of the cache of `row` into the slot
  # `slot` of the store, and returns the store.
  def store_prefix(ctx, group, store, row, slot, size) do
    buffers = Map.take(group.state.cache, Map.keys(store))

    # The first `size` valid rows of the row, which are not consecutive in :append.
    columns =
      group.state.valid
      |> Enum.at(row)
      |> Enum.flat_map(fn {start, count} -> Enum.to_list(start..(start + count - 1)//1) end)
      |> Enum.take(size)

    written = ctx.prefix_out.(store, buffers, prefix_indices(row, slot, columns))
    release(Map.values(store), Map.values(written))
    written
  end

  @doc false
  # Whether requests can join a group with other rows, see cannot_join/1.
  def joins?(ctx), do: cannot_join(ctx) == nil

  @doc false
  # Why requests cannot join a group with other rows, or nil when they can: the rows of a step
  # can have different lengths of the cache, the rows of a prompt do not move together as
  # beams, and a merged decoder has a cross function or no cross-attention cache, so a row gets
  # its cache before its prompt runs on the branch with cache (see admit_inputs/3).
  def cannot_join(ctx) do
    cond do
      ctx.num_beams > 1 ->
        "it has num_beams #{ctx.num_beams}, and the rows of a prompt move together as beams"

      one_length?(ctx) ->
        "the rows of a step need one length of the cache: " <>
          (ctx.one_length || "it keeps a cache of exact length")

      merged?(ctx) and ctx.cross == nil and ctx.cache -- ctx.buffers != [] ->
        "its cross-attention cache reads other values than the outputs of the encoder"

      true ->
        nil
    end
  end

  @doc false
  # Runs one step of each shape that the steps of a group of `rows` rows can take, so each
  # shape compiles before a request needs it: each capacity of the buffers that capacity/2
  # gives up to the largest, with each width that fits it, and for a merged decoder each first
  # step of a new group (see first_steps/1).
  # It also compiles the writes of the rows of 1 to `rows` requests that join together at each
  # capacity, and a growth, or in :append a compaction, from each capacity to the next. When the
  # rows keep their own inputs, `inputs` are the inputs of one request, which give their types
  # and shapes, and the steps also run at each encoder capacity of encoder_capacities/2, with
  # the cross function, the writes of a row and each move between neighbour encoder capacities.
  # With `prefixes: true`, it also compiles the copies of each size of a stored prefix into the
  # buffers and into the store at each capacity, and the steps of new prompts alone after a
  # prefix (see "Prefixes").
  # A shape that it misses compiles on its first step, so that step is slower and gives the
  # same result. Returns the number of step shapes.
  def compile_steps(ctx, rows, inputs \\ %{}, opts \\ []) do
    capacities = capacities(ctx, ctx.max_capacity)
    encoders = encoder_capacities(ctx, inputs)

    # Each step of a decoder that is not merged, and for a merged decoder the first steps of a
    # new group (first_steps/1) and the steps on the branch with cache, where the row has one row
    # of cache before its part. Such a step fits the part next to that row only below the
    # capacity.
    steps =
      if merged?(ctx) do
        later = for c <- capacities, size <- ctx.widths.with_cache, size < c, do: {c, size, 1}
        for({c, size} <- first_steps(ctx), do: {c, size, 0}) ++ later
      else
        for c <- capacities, size <- ctx.widths.with_cache, size <= c, do: {c, size, 0}
      end

    shapes =
      for {capacity, size, past} <- steps,
          encoder <- encoders,
          do: {capacity, size, past, encoder}

    step_items =
      for {capacity, size, past, encoder} <- shapes do
        fn ctx ->
          request = %{row: 0, prompt: List.duplicate(0, size), seed: 0, budget: 1}
          request = Map.put(request, :inputs, encoder_inputs(ctx, inputs, encoder))
          group = admit_joining(ctx, open(ctx, rows, capacity), [request], %{})
          finish(next(ctx, cached_rows(group, past)))
        end
      end

    run_items(ctx, :run, step_items)

    compile_alone(ctx, rows, capacities, opts[:prefixes])

    # A later request with more rows of the encoder, which grows the encoder capacity to the
    # next one, and a request that joins a row that runs, which runs the cross function and
    # writes its row, at each encoder capacity (see admit_inputs/3).
    request =
      &%{row: &1, prompt: [0], seed: 0, budget: 1, inputs: encoder_inputs(ctx, inputs, &2)}

    for [from, to] <- Enum.chunk_every(encoders, 2, 1, :discard) do
      group = admit(ctx, open(ctx, rows, hd(capacities)), [request.(0, from)])
      finish(admit(ctx, cancel(group, 0), [request.(0, to)]))
    end

    run_items(ctx, :run_steps, loop_items(ctx, rows, capacities, request.(0, hd(encoders))))

    for encoder <- encoders, encoder != nil, rows > 1 do
      group = admit(ctx, open(ctx, rows, hd(capacities)), [request.(0, encoder)])
      finish(admit(ctx, cached_rows(group, 1), [request.(1, encoder)]))
    end

    for capacity <- capacities, count <- 2..rows//1 do
      requests =
        for row <- 0..(count - 1) do
          inputs = encoder_inputs(ctx, inputs, hd(encoders))
          %{row: row, prompt: [0], seed: 0, budget: 1, inputs: inputs}
        end

      finish(admit_joining(ctx, open(ctx, rows, capacity), requests, %{}))
    end

    # A growth to each next capacity and a shrink to each capacity before (see resize_buffers/4),
    # and in :append a compaction at each capacity (see compact/3).
    moves =
      for [capacity, next] <- Enum.chunk_every(capacities, 2, 1, :discard),
          move <- [{capacity, next}, {next, capacity}],
          do: move

    for {from, to} <- moves do
      group = open(ctx, rows, from)
      grown = ctx.grow.(buffer_tensors(ctx, group), Nx.iota({to}, backend: Nx.BinaryBackend))
      release(Map.values(grown), [])
      finish(group)
    end

    for capacity <- capacities, ctx.mode == :append do
      group = open(ctx, rows, capacity)
      index = Nx.broadcast(Nx.tensor(0, type: :s32, backend: Nx.BinaryBackend), {rows, capacity})
      release(Map.values(ctx.compact.(buffer_tensors(ctx, group), index)), [])
      finish(group)
    end

    length(shapes)
  end

  # The prompts that run in a step of their own rows (see admit_alone/3): each count of rows and
  # each width of that step, and their moves into a group at each capacity, and with
  # `prefixes?` the copies and the steps of stored prefixes.
  defp compile_alone(ctx, rows, capacities, prefixes?) do
    items =
      for capacity <- capacities, count <- alone_counts(ctx, rows), size <- alone_sizes() do
        fn ctx ->
          requests =
            for row <- 0..(count - 1),
                do: %{row: row, prompt: List.duplicate(0, size), seed: 0, budget: 2}

          finish(admit(ctx, open(ctx, rows, capacity), requests))
        end
      end

    run_items(ctx, :run, items)

    if prefixes?, do: compile_prefixes(ctx, rows, capacities)
  end

  # The copies of each size of a stored prefix into the buffers and into the store at each
  # capacity, and each step of new prompts alone after prefixes: for each count of rows and
  # each width of that step, each capacity of the small group above the width that a size of
  # prefix gives (see alone_shape/2).
  defp compile_prefixes(ctx, rows, capacities) do
    store = prefix_store(ctx, 1, Enum.max(ctx.widths.with_cache))

    sizes =
      Enum.to_list(@join_tokens..(Enum.max(ctx.widths.with_cache) - @join_tokens)//@join_tokens)

    for capacity <- capacities, size <- sizes, size < capacity do
      group = open(ctx, rows, capacity)
      buffers = Map.take(group.state.cache, Map.keys(store))
      indices = prefix_indices(0, 0, Enum.to_list(0..(size - 1)))
      release(Map.values(ctx.prefix_in.(buffers, store, indices)), [])
      release(Map.values(ctx.prefix_out.(store, buffers, indices)), Map.values(store))
      finish(group)
    end

    shapes =
      for count <- alone_counts(ctx, rows),
          width <- @alone_widths,
          size <- sizes,
          requests = for(row <- 0..(count - 1), do: prefixed_request(row, store, size, width)),
          {^width, capacity} <- [alone_shape(ctx, requests)],
          capacity > width,
          uniq: true,
          do: {count, width, capacity, size}

    items =
      for capacity <- capacities,
          {count, width, _small, size} <- Enum.uniq_by(shapes, &Tuple.delete_at(&1, 3)) do
        fn ctx ->
          requests = for row <- 0..(count - 1), do: prefixed_request(row, store, size, width)
          finish(admit(ctx, open(ctx, rows, capacity), requests))
        end
      end

    run_items(ctx, :run, items)

    release(Map.values(store), [])
  end

  # A request whose prompt has a stored prefix of `size` tokens and the shortest rest of a step
  # of `width` tokens (see alone_sizes/0).
  defp prefixed_request(row, store, size, width) do
    rest = Enum.at(alone_sizes(), Enum.find_index(@alone_widths, &(&1 == width)))
    prompt = List.duplicate(0, size + rest)
    %{row: row, prompt: prompt, seed: 0, budget: 2, prefix: %{store: store, slot: 0, size: size}}
  end

  # The counts of rows of a step of new prompts alone, the powers of 4 below the rows of a group
  # (see admit/4).
  defp alone_counts(ctx, rows) do
    if runs_alone?(ctx),
      do: Stream.iterate(1, &(&1 * 4)) |> Enum.take_while(&(&1 < rows)),
      else: []
  end

  # The shortest prompt of each width of a step of new prompts alone (see alone_shape/2): the
  # width below it, or 1. A move of its row needs the fewest rows of the group, so its writes
  # compile at each capacity that a longer prompt of that width can meet, where the end of the
  # rows of the small group is cut.
  defp alone_sizes,
    do: Enum.zip_with([1 | @alone_widths], @alone_widths, fn below, _width -> below end)

  # Each decode loop of compile_loop/4 at each capacity, with `request` in row 0.
  defp loop_items(%{run_steps: loops}, rows, capacities, request) when loops != nil do
    for capacity <- capacities, k <- Map.keys(loops) do
      fn ctx ->
        group = next(ctx, admit(ctx, open(ctx, rows, capacity), [%{request | budget: k + 1}]))
        finish(next_steps(ctx, group, k))
      end
    end
  end

  defp loop_items(_ctx, _rows, _capacities, _request), do: []

  # Runs `items`, functions of the context that each run a call of ctx.run or of a loop of
  # ctx.run_steps (`field`), and compiles those calls first in parallel tasks. A first pass
  # runs each item up to its call, keeps the templates of the inputs of the call and stops
  # there, so nothing runs on the model. The tasks then call the compiled functions with the
  # templates, which only compiles them (see Transmuter.GraphBreaks), at most
  # ctx.compile_concurrency at a time and without device memory. The items then run, and
  # their calls find the compiled segments.
  defp run_items(ctx, field, items) do
    if ctx.compile_concurrency > 1 and items != [] do
      calls = record_calls(ctx, field, items)

      calls
      |> Task.async_stream(
        fn {key, inputs, static} -> call_of(ctx, field, key).(inputs, static) end,
        max_concurrency: ctx.compile_concurrency,
        ordered: false,
        timeout: :infinity
      )
      |> Stream.run()
    end

    Enum.each(items, & &1.(ctx))
  end

  defp record_calls(ctx, field, items) do
    ref = make_ref()

    record = fn key ->
      fn inputs, static ->
        inputs = Map.new(inputs, fn {name, value} -> {name, template(value)} end)
        send(self(), {ref, {key, inputs, static}})
        throw(ref)
      end
    end

    recording =
      case field do
        :run -> %{ctx | run: record.(:run)}
        :run_steps -> %{ctx | run_steps: Map.new(ctx.run_steps, fn {k, _} -> {k, record.(k)} end)}
      end

    for item <- items do
      try do
        item.(recording)
      catch
        ^ref -> :ok
      end

      # The buffers of the group of the item die at its stop.
      :erlang.garbage_collect()
    end

    receive_calls(ref, [])
    |> Enum.uniq_by(fn {key, inputs, static} ->
      {key, Map.new(inputs, fn {n, t} -> {n, shape_of(t)} end), static}
    end)
  end

  defp receive_calls(ref, calls) do
    receive do
      {^ref, call} -> receive_calls(ref, [call | calls])
    after
      0 -> Enum.reverse(calls)
    end
  end

  defp call_of(ctx, :run, :run), do: ctx.run
  defp call_of(ctx, :run_steps, k), do: ctx.run_steps[k]

  defp template(tensor = %Nx.Tensor{}), do: Nx.to_template(tensor)
  defp template(other), do: other

  defp shape_of(tensor = %Nx.Tensor{}), do: {Nx.shape(tensor), Nx.type(tensor)}
  defp shape_of(other), do: other

  # The compiles of compile_steps/3 that can run at the same time: one for each scheduler, and
  # one with EMLX, whose compile cache belongs to the first process that compiles.
  defp compile_concurrency(defn_options) do
    if GraphBreaks.compiler(defn_options) == EMLX, do: 1, else: System.schedulers_online()
  end

  # The capacity and the width of each first step of a new group of a merged decoder, on its
  # branch without cache: for each prompt length up to the largest capacity, the width of the
  # first part of plan/3 (part_size/4) and the capacity of fit_capacity/4 in a group that open/3
  # just made, whose rows have no rows of cache: the room that fit_next/2 makes at the admission,
  # which the first step does not move.
  # The server opens a new group when no row runs (see Transmuter.ContinuousBatching), so its
  # steps on that branch are such first steps. One row stands for the rows of a group, because
  # the longest prompt decides the width and the capacity.
  defp first_steps(ctx) do
    start = capacity(ctx, @capacity_margin)

    widths = ctx.widths.without_cache
    room = room(ctx, %{lengths: [0], done: [false]})

    for length <- 1..ctx.max_capacity, uniq: true do
      width = width_for(min(length, part_size(false, length, room, widths)), widths)
      rows = %{written: 0, kept: 0, running: length}
      {fit_capacity(ctx, start, rows, width), width}
    end
  end

  # The group with `past` rows of cache in its row 0, which compile_steps/3 gives a step on the
  # branch with cache.
  defp cached_rows(group, 0), do: group

  defp cached_rows(group = %{state: state}, past) do
    state = %{
      state
      | lengths: List.replace_at(state.lengths, 0, past),
        valid: List.replace_at(state.valid, 0, [{0, past}]),
        used: max(state.used, past)
    }

    %{group | state: state}
  end

  @doc false
  # Compiles the image features of a group of `rows` rows for the type and the width of
  # `features`, the features of one prompt: the slots, a write of the features of each power
  # of two of rows up to a slot, and the merge of embed/4 at each width of step_width/1. See
  # join_features/3. The small groups of the prompts that run alone (see admit_alone/3) have
  # their own slots of the width of their step, so their writes and merges compile too.
  def compile_features(ctx, rows, features) do
    widths = ctx.widths.with_cache ++ if(merged?(ctx), do: ctx.widths.without_cache, else: [])
    compile_slots(ctx, rows, power_of_two(ctx.max_capacity), Enum.uniq(widths), features)

    for count <- alone_counts(ctx, rows),
        width <- @alone_widths,
        width <= ctx.max_capacity,
        do: compile_slots(ctx, count, width, [width], features)

    :ok
  end

  # The slots of `rows` rows of `slot` rows, a write of each power of two of rows up to a slot,
  # and the merge at each of `widths`.
  defp compile_slots(ctx, rows, slot, widths, features) do
    slots = empty_slots(ctx, rows, slot, [%{features: features}])
    zero = Nx.tensor(0, type: Nx.type(features), backend: ctx.backend)
    hidden = Nx.axis_size(features, 1)

    for exponent <- 0..(length(Integer.digits(slot, 2)) - 1) do
      part = Nx.broadcast(zero, {2 ** exponent, hidden})
      ctx.write_features.(slots, part, Nx.tensor(0, backend: Nx.BinaryBackend))
    end

    for width <- widths do
      embeds = ctx.embed.(List.duplicate(List.duplicate(0, width), rows))
      index = Nx.broadcast(Nx.tensor(0, type: :s32, backend: Nx.BinaryBackend), {rows, width})
      mask = Nx.broadcast(Nx.tensor(0, type: :u8, backend: Nx.BinaryBackend), {rows, width})
      ctx.merge.(embeds, slots, index, mask)
    end
  end

  @doc false
  # Ends `row` before its last token: the next steps give it no token and no part of its
  # prompt, and admit/4 can give its row to a new request.
  def cancel(group, row) do
    %{
      group
      | done: List.replace_at(group.done, row, true),
        new: List.replace_at(group.new, row, nil),
        left: List.replace_at(group.left, row, [])
    }
  end

  # The image features of a group after `requests` join it, and the cursor of each row in
  # them: the index of the features of its next image token and the end of its features. Each
  # row has a slot of the features of one request, and a request writes its features at the
  # start of the slot of its row. The slot of a group comes from open/3, and the features of a
  # request have a power of two of rows (see generate/5). So the writes and the merge of
  # embed/4 have few shapes, the features of a group have one shape for its life, and a row
  # that ends leaves its slot to the next request.
  defp join_features(ctx, group, requests) do
    rows = length(group.done)
    writes = for request = %{features: %Nx.Tensor{}} <- requests, do: request

    slot = group.state.slot

    features =
      case {writes, group.state.features} do
        {[], features} -> features
        {_writes, nil} -> write_slots(ctx, empty_slots(ctx, rows, slot, writes), slot, writes)
        {_writes, features} -> write_slots(ctx, features, slot, writes)
      end

    cursors =
      Enum.reduce(requests, group.state.cursors, fn request, cursors ->
        start = request.row * slot
        count = Enum.count(request.prompt, &(&1 == ctx.image_token))
        List.replace_at(cursors, request.row, {start, start + count})
      end)

    {features, cursors}
  end

  # The smallest power of two at or above `count`.
  defp power_of_two(count), do: 2 ** length(Integer.digits(max(count, 1) - 1, 2))

  # The smallest power of 4 at or above `count`.
  defp power_of_four(count), do: Enum.find(Stream.iterate(1, &(&1 * 4)), &(&1 >= count))

  defp empty_slots(ctx, rows, slot, [%{features: first} | _writes]) do
    zero = Nx.tensor(0, type: Nx.type(first), backend: ctx.backend)
    Nx.broadcast(zero, {rows * slot, Nx.axis_size(first, 1)})
  end

  defp write_slots(ctx, features, slot, writes) do
    Enum.reduce(writes, features, fn request, features ->
      if Nx.axis_size(request.features, 0) > slot do
        raise ArgumentError,
              "the image features of a prompt have #{Nx.axis_size(request.features, 0)} rows, more than the #{slot} rows of a slot of the group, the power of two of its largest prompt"
      end

      start = Nx.tensor(request.row * slot, backend: Nx.BinaryBackend)
      ctx.write_features.(features, request.features, start)
    end)
  end

  @doc false
  # The rows of cache that `group` needs to admit `requests` and to run each row to the end of
  # its budget: the rows that its steps write, see rows_written/2.
  def rows_needed(ctx, group, requests) do
    left = Enum.reduce(requests, group.left, &List.replace_at(&2, &1.row, &1.prompt))
    group = %{reset_rows(group, requests) | left: left}

    remaining =
      Enum.zip_with([group.done, group.budgets, group.ids], fn [done?, budget, ids] ->
        if done?, do: 0, else: budget - length(ids)
      end)

    rows_written(ctx, %{
      left: group.left,
      new: group.new,
      done: group.done,
      lengths: group.state.lengths,
      remaining: remaining
    })
  end

  # The rows of cache that the steps of `rows` write until each row ends. The steps of the
  # prompt parts are those of plan/2, each at the width of step_width/1, where each row of a
  # step writes the width at its length. Then each row writes its new tokens, one in each step.
  # A row selects a token after the last part of its prompt and after each new token, until
  # its budget ends, and its last token is not written. The rows that ended do not count: a
  # compaction drops them in :append, and in :in_place they write from their first row (see
  # advance/2).
  defp rows_written(ctx, rows) do
    if Enum.all?(rows.left, &(&1 == [])) do
      ends =
        for {length, false, remaining} <- Enum.zip([rows.lengths, rows.done, rows.remaining]),
            do: length + remaining

      Enum.max(ends ++ Enum.map(running_lengths(rows), &(&1 + 1)), fn -> 0 end)
    else
      widths = branch_widths(ctx, rows, rows.lengths)
      {parts, left} = plan(rows, room(ctx, rows), widths)
      written = Enum.max(running_lengths(rows), fn -> 0 end) + step_width(parts, widths)

      [selects, remaining] =
        [parts, left, rows.done, rows.remaining]
        |> Enum.zip_with(fn [part, left, done?, remaining] ->
          select? = part != [] and left == [] and not done?
          [select?, if(select?, do: remaining - 1, else: remaining)]
        end)
        |> Enum.zip_with(& &1)

      rows = %{
        rows
        | left: left,
          lengths: Enum.zip_with(rows.lengths, parts, &(&1 + length(&2))),
          new: Enum.zip_with(selects, rows.new, &if(&1, do: 0, else: &2)),
          done:
            Enum.zip_with([rows.done, selects, remaining], fn [done?, select?, remaining] ->
              done? or (select? and remaining == 0)
            end),
          remaining: remaining
      }

      max(written, rows_written(ctx, rows))
    end
  end

  @doc false
  # The tokens of `row`, as generate/5 gives them.
  def row_tokens(group, row), do: tokens(Enum.at(group.ids, row), Enum.at(group.logprobs, row))

  # Runs the prompts of `requests` in parts of power-of-two lengths. Each prompt fills the
  # parts from the last one, so the last token of each prompt is in the last part. A row that
  # has a new token from an earlier step takes it in the last part, and has no tokens in the
  # other parts. The last part is no longer than the tokens of each row in it, so it ends each
  # row at its last position, where the step keeps the logits. A merged decoder runs each
  # prompt as one part, on the branch without cache (see the module documentation): one part
  # for each prompt length. So when the rows of a step need one length of the cache, and with
  # a merged decoder, requests join only a group without other rows. Each row takes its first
  # token from the part where its prompt ends.
  defp prefill(ctx, group, requests, inputs) do
    pending = Enum.zip_with(group.new, group.done, &if(&2, do: nil, else: &1))
    {features, cursors} = join_features(ctx, group, requests)

    # A row with a new token has one token in these steps.
    joining = Map.new(requests, &{&1.row, &1})
    lengths = Enum.map(requests, &length(&1.prompt)) ++ for(token <- pending, token, do: 1)
    sizes = part_sizes(ctx, lengths)
    group = fit(ctx, reset_rows(group, requests), Enum.sum(sizes))
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
    state = %{group.state | seen: nil, history: nil, count: nil}
    state = %{state | features: features, cursors: cursors}
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

  # The rows of `requests` start again: their prompt, budget and positions, no tokens, and no
  # rows of cache.
  defp reset_rows(group, requests) do
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
              positions: List.replace_at(group.state.positions, request.row, positions(request))
          }
      }
    end)
  end

  defp prefix_size(request), do: get_in(request, [:prefix, :size]) || 0

  # Writes the keys and values of the stored prefix of each request into its row, and the row
  # starts with them: from its first row in :in_place, and in :append at the write offset of the
  # group, which then moves after them, as after a prompt part. The buffers first make room for
  # them (see fit/3), so a growth or a compaction does not move rows that hold no values yet.
  defp copy_prefixes(group, ctx, requests) do
    for request = %{prefix: %{store: store, slot: slot, size: size}} <- requests,
        reduce: group do
      group ->
        group = %{state: state} = fit(ctx, group, size)
        column = if ctx.mode == :append, do: state.used, else: 0
        buffers = Map.take(state.cache, Map.keys(store))
        columns = Enum.to_list(column..(column + size - 1))
        written = ctx.prefix_in.(buffers, store, prefix_indices(request.row, slot, columns))
        release(Map.values(buffers), Map.values(written))

        state = %{
          state
          | cache: Map.merge(state.cache, written),
            lengths: List.replace_at(state.lengths, request.row, size),
            valid: List.replace_at(state.valid, request.row, [{column, size}]),
            used: max(state.used, column + size)
        }

        %{group | state: state}
    end
  end

  # The row, the slot and the columns of the buffer of a copy of a prefix. The number of
  # columns is the size of a tensor, so each size compiles once.
  defp prefix_indices(row, slot, columns) do
    host = &Nx.tensor(&1, type: :s32, backend: Nx.BinaryBackend)
    %{row: host.(row), slot: host.(slot), columns: host.(columns)}
  end

  # The prefix goes into consecutive columns of the row from the first one of `columns`.
  defp prefix_in(buffers, store, %{row: row, slot: slot, columns: columns}) do
    Map.new(buffers, fn {name, buffer} ->
      {_batch, heads, _length, dim} = Nx.shape(buffer)
      part = Nx.slice(store[name], [slot, 0, 0, 0], [1, heads, Nx.axis_size(columns, 0), dim])
      {name, Nx.put_slice(buffer, [row, 0, columns[0], 0], part)}
    end)
  end

  # The values of `columns` of the row, in their order, go into the first rows of the slot.
  defp prefix_out(store, buffers, %{row: row, slot: slot, columns: columns}) do
    Map.new(store, fn {name, slots} ->
      part = buffers[name] |> Nx.slice_along_axis(row, 1, axis: 0) |> Nx.take(columns, axis: 2)
      {name, Nx.put_slice(slots, [slot, 0, 0, 0], part)}
    end)
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
  # prompt, its count and prompt count, its budget of new tokens, and the key of its seed.
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
      budget: host.(Enum.map(requests, & &1.budget)),
      keys: requests |> Enum.map(&Nx.Random.key(&1.seed)) |> Nx.stack()
    }

    old =
      group.state
      |> Map.take([:seen, :history, :count, :prompt_count, :budget])
      |> Map.put(:keys, group.keys)

    written = ctx.admit_rows.(old, host.(Enum.map(requests, & &1.row)), values)
    release(Map.values(old), [])
    {keys, written} = Map.pop!(written, :keys)
    %{group | keys: keys, state: Map.merge(group.state, written)}
  end

  # The row `index` of each tensor, with its axis of rows.
  defp take_row(tensors, index),
    do: Map.new(tensors, fn {name, t} -> {name, Nx.take(t, Nx.reshape(index, {1}), axis: 0)} end)

  # The rows in `index` get the rows of `values`.
  defp write_admitted(tensors, index, values) do
    Map.new(tensors, fn {name, tensor} ->
      {name,
       Nx.indexed_put(tensor, Nx.new_axis(index, 1), Nx.as_type(values[name], Nx.type(tensor)))}
    end)
  end

  # The row `row` of each tensor gets the row of `values`, from the start of the row. A value can
  # be shorter than the row on the axes after the first, and zeros fill the rest of the row. A
  # longer value loses its end, which holds no rows of a request. A request writes its encoder
  # outputs and cross-attention cache (see write_request/4), and a prompt that runs in its own
  # step moves its row into its group (see move_rows/4).
  defp write_row(tensors, row, values) do
    Map.new(tensors, fn {name, tensor} ->
      value = Nx.as_type(values[name], Nx.type(tensor))
      shape = put_elem(Nx.shape(tensor), 0, 1)
      lengths = Enum.zip_with(Tuple.to_list(Nx.shape(value)), Tuple.to_list(shape), &min/2)
      value = Nx.slice(value, List.duplicate(0, Nx.rank(value)), lengths)

      value =
        if Nx.shape(value) == shape,
          do: value,
          else:
            Nx.put_slice(
              Nx.broadcast(Nx.tensor(0, type: Nx.type(tensor)), shape),
              List.duplicate(0, Nx.rank(tensor)),
              value
            )

      {name, Nx.put_slice(tensor, [row | List.duplicate(0, Nx.rank(tensor) - 1)], value)}
    end)
  end

  # The rows of a merged decoder whose requests join keep their own inputs, such as the outputs
  # of the encoder, in the cache of the group, next to their cross-attention cache.
  defp rows_keep_inputs?(ctx), do: ctx.given != [] and merged?(ctx) and joins?(ctx)

  # Writes the inputs of each request into its row. The inputs have the rows of the encoder on
  # axis 1, and the cache on axis 2. All rows of a group have one encoder capacity: it starts at
  # the rows of the encoder of the first rows, a later row with more rows grows it to the
  # encoder_capacity/2 of its rows, and a row with fewer rows pads to it with rows of zeros,
  # which its encoder mask hides. The capacity does not shrink, so the steps compile for few
  # shapes.
  #
  # The cross-attention cache of a row comes from its first step. When the group has rows that
  # run, that step is on the branch with cache, which reads the cache, so the cross function
  # writes it here. Otherwise the first step is on the branch without cache, whose presents
  # give the cache of all rows from their inputs (see step/5).
  defp admit_inputs(ctx, group = %{state: state}, requests) do
    if Enum.any?(ctx.given, &Map.has_key?(state.inputs, &1)) do
      raise ArgumentError,
            "a request with its own encoder outputs cannot join a group whose rows took the encoder outputs of the prompts"
    end

    longest = requests |> Enum.map(&encoder_rows/1) |> Enum.max()

    capacity =
      cond do
        state.encoder_capacity == 0 -> longest
        longest <= state.encoder_capacity -> state.encoder_capacity
        true -> encoder_capacity(ctx, longest)
      end

    cross? = Enum.any?(running_lengths(%{lengths: state.lengths, done: group.done}), &(&1 > 0))
    group = widen(ctx, group, hd(requests), capacity)
    Enum.reduce(requests, group, &write_request(ctx, &2, &1, cross?))
  end

  # The rows of the encoder in the inputs of a request.
  defp encoder_rows(request) do
    case request.inputs |> Map.values() |> Enum.map(&Nx.axis_size(&1, 1)) |> Enum.uniq() do
      [rows] ->
        rows

      sizes ->
        raise ArgumentError,
              "the inputs of a prompt have #{inspect(sizes)} rows of the encoder, expected one number"
    end
  end

  # The smallest encoder capacity: the smallest source length of the servings (see
  # Serving.sequence_lengths/2), so their encoder outputs, of 32, 128 or 512 rows, need no pad.
  @min_encoder_capacity 32

  # The encoder capacity for `rows` rows of the encoder. When the branch with cache reads the
  # encoder mask, the padding rows are hidden, so the capacity is a power of two of at least
  # @min_encoder_capacity rows, at most the longest output of the encoder (see
  # encoder_length/1). Otherwise all rows of a group need the same rows of the encoder.
  defp encoder_capacity(%{pad_encoder?: false}, rows), do: rows

  defp encoder_capacity(ctx, rows) do
    capacity = power_of_two(max(rows, @min_encoder_capacity))
    max(rows, if(ctx.encoder_length, do: min(capacity, ctx.encoder_length), else: capacity))
  end

  # The longest output of the encoder that the configuration gives, or nil.
  defp encoder_length(config) do
    config["max_source_positions"] || config["max_position_embeddings"] ||
      get_in(config, ["text_config", "max_position_embeddings"])
  end

  # Gives the inputs and the cross-attention cache of the rows the encoder capacity
  # `capacity`, one capacity of encoder_capacity/2 at a time, so each move compiles only between
  # neighbour capacities. The first rows of a group make these tensors at their capacity, as
  # zeros on the device, from the types and the shapes of the inputs of a request and of the
  # placeholders of the cache.
  defp widen(ctx, group = %{state: state, done: done}, request, capacity)
       when state.encoder_capacity == 0 do
    shapes =
      Enum.map(request.inputs, fn {name, t} ->
        {name, Nx.type(t), put_elem(Nx.shape(t), 0, length(done))}
      end) ++
        for name <- ctx.cache -- ctx.buffers,
            t = state.cache[name],
            do: {name, Nx.type(t), Nx.shape(t)}

    made =
      Map.new(shapes, fn {name, type, shape} ->
        axis = if String.starts_with?(name, "past_key_values."), do: 2, else: 1
        shape = put_elem(shape, axis, capacity)
        {name, Nx.broadcast(Nx.tensor(0, type: type, backend: ctx.backend), shape)}
      end)

    release(Map.values(Map.take(state.cache, Map.keys(made))), [])
    %{group | state: %{state | cache: Map.merge(state.cache, made), encoder_capacity: capacity}}
  end

  defp widen(ctx, group = %{state: state}, request, capacity) do
    tensors = Map.take(state.cache, (ctx.cache -- ctx.buffers) ++ Map.keys(request.inputs))

    widened =
      ctx
      |> encoder_path(state.encoder_capacity, capacity)
      |> Enum.reduce(tensors, fn rows, tensors ->
        widened = ctx.widen.(tensors, Nx.iota({rows}, backend: Nx.BinaryBackend))
        release(Map.values(tensors), Map.values(widened))
        widened
      end)

    cache = Map.merge(state.cache, widened)
    %{group | state: %{state | cache: cache, encoder_capacity: capacity}}
  end

  # The encoder capacities that the rows of a group can have, from the rows of the encoder in
  # `inputs`, the inputs of its first request: those rows, and when padding rows are hidden each
  # larger one up to the longest output of the encoder. Without inputs, [nil].
  defp encoder_capacities(_ctx, inputs) when inputs == %{}, do: [nil]

  defp encoder_capacities(ctx, inputs) do
    rows = encoder_rows(%{inputs: inputs})
    top = encoder_capacity(ctx, max(rows, ctx.encoder_length || rows))
    [rows | encoder_path(ctx, rows, top)]
  end

  # Inputs of the types of `inputs` with `rows` rows of the encoder, all zeros.
  defp encoder_inputs(_ctx, inputs, nil), do: inputs

  defp encoder_inputs(ctx, inputs, rows) do
    Map.new(inputs, fn {name, t} ->
      zero = Nx.tensor(0, type: Nx.type(t), backend: ctx.backend)
      {name, Nx.broadcast(zero, put_elem(Nx.shape(t), 1, rows))}
    end)
  end

  # The encoder capacities after `from` on the way to `to`.
  defp encoder_path(ctx, from, to) do
    below = Stream.iterate(@min_encoder_capacity, &(&1 * 2)) |> Enum.take_while(&(&1 < to))
    steps = if ctx.pad_encoder?, do: below ++ [to], else: [to]
    Enum.filter(steps, &(&1 > from))
  end

  defp widen_rows(tensors, capacity) do
    Map.new(tensors, fn {name, tensor} ->
      axis = if String.starts_with?(name, "past_key_values."), do: 2, else: 1
      free = Nx.axis_size(capacity, 0) - Nx.axis_size(tensor, axis)
      padding = List.replace_at(List.duplicate({0, 0, 0}, Nx.rank(tensor)), axis, {0, free, 0})
      {name, Nx.pad(tensor, 0, padding)}
    end)
  end

  # Pads the inputs of `request` to the encoder capacity of the group, runs its cross function
  # when `cross?` (see admit_inputs/3), and writes both into its row.
  defp write_request(ctx, group = %{state: state}, request, cross?) do
    capacity = state.encoder_capacity
    rows = encoder_rows(request)

    if rows != capacity and not ctx.pad_encoder? do
      raise ArgumentError,
            "the branch with cache of the decoder reads no encoder mask, so all rows of a group need the same rows of the encoder, got #{rows} and #{capacity}"
    end

    padded = Map.new(request.inputs, fn {name, t} -> {name, pad_rows(t, capacity)} end)

    cross =
      for {"present." <> rest, t} <- if(cross? and ctx.cross, do: ctx.cross.(padded), else: %{}),
          into: %{},
          do: {"past_key_values." <> rest, t}

    values = Map.merge(padded, cross)
    old = Map.take(state.cache, Map.keys(values))
    row = Nx.tensor(request.row, backend: Nx.BinaryBackend)
    # The old tensors die here, so the write can reuse their buffers (EMLX copies them).
    written = ctx.write_row.(Nx.donatable(old), row, values)
    release(Map.values(old) ++ Map.values(values), Map.values(request.inputs))
    %{group | state: %{state | cache: Map.merge(state.cache, written)}}
  end

  # Pads the encoder rows of an input of one request, {1, rows, ...}, to the bucket of the row,
  # `capacity` rows, with zero bytes after its data. The pad runs on the host, because a device
  # op on a tensor of `rows` rows compiles for each number of rows on EXLA, and the rows of a
  # caller have no bound. An input with `capacity` rows gets no bytes and stays on the device.
  defp pad_rows(tensor, capacity) do
    case (capacity - Nx.axis_size(tensor, 1)) * div(Nx.size(tensor), Nx.axis_size(tensor, 1)) do
      0 ->
        tensor

      count ->
        {_kind, bits} = Nx.type(tensor)
        data = Nx.to_binary(tensor) <> :binary.copy(<<0>>, div(count * bits, 8))
        shape = put_elem(Nx.shape(tensor), 1, capacity)
        data |> Nx.from_binary(Nx.type(tensor), backend: Nx.BinaryBackend) |> Nx.reshape(shape)
    end
  end

  # The selection of each step, until each row has ended.
  defp decode(ctx, group, notify) do
    notify.(group)

    if Enum.all?(group.done),
      do: finish(group),
      else: decode(ctx, next(ctx, group, ctx.decode_steps), notify)
  end

  @doc false
  # Frees the device tensors that the group owns when it ends: its cache, its last selection
  # and selection state, its masks and its keys.
  def finish(group = %{state: state}) do
    own = [state.selected, state.seen, state.history, state.count, state.prompt_count]
    own = [group.keys, state.budget | own] ++ Map.values(state.cache)
    release(own ++ Map.values(group.masks), [])
    group
  end

  # A row with a part of its prompt left gets no token, and has not ended.
  defp select(ctx, group), do: select_rows(ctx, group, Nx.to_list(group.state.selected))

  defp select_rows(ctx, group, selected) do
    {new, logprobs} =
      [selected, group.done, group.left]
      |> Enum.zip_with(fn [[token | logprob], done?, left] ->
        if done? or left != [] or token in ctx.eos,
          do: {nil, nil},
          else: {token, from_bits(logprob)}
      end)
      |> Enum.unzip()

    # A row ends at an eos token or with the new tokens of its budget.
    ids = push(group.ids, new, new)

    done =
      Enum.zip_with([new, ids, group.budgets, group.left], fn [token, ids, budget, left] ->
        left == [] and (token == nil or length(ids) == budget)
      end)

    group = %{
      group
      | new: new,
        done: done,
        ids: ids,
        logprobs: push(group.logprobs, new, logprobs)
    }

    %{group | steps: [view(group)]}
  end

  # The new tokens of a step, the rows that ended, and the log-probabilities of each row after
  # it. next/3 gives one for each step, and the stream of each row reads them in order.
  defp view(group), do: Map.take(group, [:new, :done, :logprobs])

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
    extras = %{extras | features: repeat.(extras.features), positions: repeat.(extras.positions)}
    # The first beam of each prompt starts at 0 and the others at -1e9, so the first step
    # selects only from the first beam.
    scores = Enum.flat_map(prompts, fn _ -> [0.0 | List.duplicate(-1.0e9, beams - 1)] end)
    rows = repeat.(prompts)
    lengths = for row <- rows, row != [], do: length(row)

    group =
      ctx
      |> open(length(rows), Enum.sum(part_sizes(ctx, lengths)) + @capacity_margin)
      |> Map.put(:scores, scores)
      |> then(&prefill(ctx, &1, requests(ctx, rows, repeat.(seeds), extras), inputs))

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

      group = fit(ctx, group, 1)
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
        @prompt_count => state.prompt_count,
        @budget => state.budget
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
          budget: moved[@budget],
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
    for view <- group.steps, do: notify_step(on_token, Map.merge(group, view), logprobs?)
    :ok
  end

  defp notify_step(on_token, group, logprobs?) do
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

  # The step of the new tokens gives the selection for the remaining tokens. In :in_place, a
  # row that ended loses its valid rows, so it writes the padding of the step from its first
  # row and needs no rows of the buffers. In :append, a compaction drops it.
  defp advance(ctx, group) do
    group = fit_next(ctx, group)
    {parts, left, width} = next_parts(ctx, group)
    group = %{group | left: left}
    state = step(ctx, group.state, parts, selection(ctx, group), width)

    # The image features stay until the last part of a prompt has run.
    state = if Enum.all?(left, &(&1 == [])), do: %{state | features: nil}, else: state
    %{group | state: state}
  end

  # A step of a merged decoder whose rows with tokens, of `lengths` rows of cache, have no
  # cache yet runs on the branch without cache, which reads no past. This is the first step of a
  # new or idle group, and of each prompt with beam search (see "Merged decoders").
  defp without_cache?(ctx, lengths), do: merged?(ctx) and Enum.all?(lengths, &(&1 == 0))

  # The widths of the next step of `rows`: those of the branch that it runs on (see context/4).
  # Its rows with tokens are the rows with a part of a prompt left or with a new token.
  defp branch_widths(ctx, rows, lengths) do
    taking =
      for {length, left, token, done?} <- Enum.zip([lengths, rows.left, rows.new, rows.done]),
          left != [] or (token != nil and not done?),
          do: length

    if without_cache?(ctx, taking), do: ctx.widths.without_cache, else: ctx.widths.with_cache
  end

  # Makes room in the buffers for the next step of a group (see fit/3), after the rows that
  # ended in :in_place lose their valid rows (see drop_ended/1). admit/4 calls it after the rows
  # of its requests take their prompts, so the room covers each token that admit/4 writes, the
  # tokens of each prompt, and each row that the first step writes. Then the first step after an
  # admission does not move the buffers. advance/2 calls it before each later step.
  defp fit_next(ctx, group) do
    group = if ctx.mode == :in_place, do: drop_ended(group), else: group
    {_parts, _left, width} = next_parts(ctx, group)
    fit(ctx, group, width)
  end

  # The parts of the next step of a group (see plan/3), the prompts that are left after it, and
  # its width, on the branch that the step runs on (see branch_widths/3).
  defp next_parts(ctx, group) do
    widths = branch_widths(ctx, group, group.state.lengths)
    rows = %{lengths: group.state.lengths, done: group.done}
    {parts, left} = plan(group, room(ctx, rows), widths)
    {parts, left, step_width(parts, widths)}
  end

  # The tokens of each row in the next step, and the tokens of the prompts that are left after
  # it. A row with a new token takes it. A row with a prompt takes a part of it from the front:
  # at most @join_tokens tokens when another row takes a new token, so a long prompt delays the
  # other rows by steps of that size, and at most the largest step width otherwise. Each row
  # reads the logits at its own last token (see keep_last_logits/1), so the rows can end their
  # tokens at different positions, and a prompt of 37 tokens runs in one step with a new token
  # of each other row. A part is also at most the `room` that the largest capacity has after
  # the longest row that has not ended, except a part of @join_tokens, which the rows that
  # admitted the request counted (see rows_needed/3).
  defp plan(group, room, widths) do
    running? =
      Enum.any?(Enum.zip(group.new, group.done), &match?({token, false} when token != nil, &1))

    longest = group.left |> Enum.map(&length/1) |> Enum.max()
    size = part_size(running?, longest, room, widths)

    [group.left, group.new, group.done]
    |> Enum.zip_with(fn
      [[], token, false] when token != nil -> {[token], []}
      [[], _token, _done?] -> {[], []}
      [left, _token, _done?] -> Enum.split(left, size)
    end)
    |> Enum.unzip()
  end

  # The most tokens of a prompt in the next step, when the longest prompt has `longest` tokens
  # left, see plan/3.
  defp part_size(running?, longest, room, widths) do
    join = min(@join_tokens, List.last(widths))
    fills = for width <- widths, width <= longest, width <= room, do: width
    if running?, do: join, else: Enum.max([join | fills])
  end

  # The lengths of the rows that have not ended.
  defp running_lengths(rows),
    do: for({length, false} <- Enum.zip(rows.lengths, rows.done), do: length)

  # The rows that a step can write after the longest row that has not ended, up to the largest
  # capacity of the decoder, or :infinity without one.
  defp room(%{max_capacity: nil}, _rows), do: :infinity

  defp room(ctx, rows),
    do: ctx.max_capacity - Enum.max(running_lengths(rows), fn -> 0 end)

  # The rows that ended have no valid rows.
  defp drop_ended(group = %{state: state}) do
    lengths = Enum.zip_with(state.lengths, group.done, &if(&2, do: 0, else: &1))
    valid = Enum.zip_with(state.valid, group.done, &if(&2, do: [], else: &1))
    %{group | state: %{state | lengths: lengths, valid: valid}}
  end

  # The width of a step: the smallest width of `widths` (see context/4) that holds its longest
  # part. The shorter parts pad to it, as the parts of the rows of a step pad to the longest
  # one. So the steps of a group have at most five shapes for each capacity (see
  # compile_steps/3), whatever the timing of the requests.
  defp step_width(parts, widths),
    do: width_for(parts |> Enum.map(&length/1) |> Enum.max(), widths)

  defp width_for(tokens, widths), do: Enum.find(widths, &(&1 >= tokens))

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
      cursors: List.duplicate({0, 0}, batch_size),
      positions: List.duplicate({[], 0}, batch_size),
      lengths: List.duplicate(0, batch_size),
      used: 0,
      valid: List.duplicate([], batch_size),
      capacity: capacity,
      encoder_capacity: 0,
      selected: nil,
      seen: nil,
      history: nil,
      count: nil,
      prompt_count: nil,
      budget: nil
    }
  end

  # Gives each row its tokens of `parts` and padding tokens after them, so that all rows
  # have the same number of tokens. The selection of the step reads the logits at the last
  # real token of each row.
  defp step(ctx, state, parts, selection, width \\ 1) do
    counts = Enum.map(parts, &length/1)
    n = max(width, Enum.max(counts))
    tokens = Enum.map(parts, &(&1 ++ List.duplicate(0, n - length(&1))))
    {inputs, static} = step_inputs(ctx, state, tokens, counts)
    {inputs, state} = embed(ctx, state, inputs, tokens)

    # The selection takes the logits at the last real token of each row, which is this many
    # positions before the last position of the logits. With @logit_index, the logits have
    # that position alone.
    offsets =
      if @logit_index in ctx.model.inputs,
        do: List.duplicate(0, length(counts)),
        else: Enum.map(counts, &(n - &1))

    offsets = host_tensor(runs(offsets), :s32)

    # The selection of the last step is read, or is that of an earlier part of the prompt,
    # so nothing reads it now.
    release(List.wrap(state.selected), [])
    selection = Map.put(selection, @offset, offsets)

    # In :append, the new rows of all rows go into the buffer after the rows of the earlier
    # steps.
    selection = Map.merge(selection, write_at(ctx, state))

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
  # a row is the index of the features of its next image token and the end of its features,
  # so the image tokens after its prompt, which the row generates, keep their embeddings.
  defp embed(%{embed: nil}, state, inputs, _tokens), do: {inputs, state}

  defp embed(ctx, state, inputs, tokens) do
    embeds = ctx.embed.(tokens)

    {rows, cursors} =
      Enum.zip_with(tokens, state.cursors, fn row, cursor ->
        Enum.map_reduce(row, cursor, fn
          id, {next, stop} when id == ctx.image_token and next < stop ->
            {{next, 1}, {next + 1, stop}}

          _id, cursor ->
            {{0, 0}, cursor}
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
  # decoder can give a past back as its present. Host tensors need no release. Only the
  # buffers that the backend frees count for collect/1: a buffer that a call took by
  # donation is already gone, and its backend gives :already_deallocated.
  def release(dead, kept) do
    shared = for %{data: data} <- kept, do: data

    freed =
      for tensor = %{data: data} <- dead,
          not is_struct(data, Nx.BinaryBackend),
          data not in shared,
          Nx.backend_deallocate(tensor) == :ok,
          do: Nx.byte_size(tensor)

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

  # Makes room for the next step of n tokens when it does not fit, before the selection of the
  # step reads the tokens of the rows from the state: a compaction in :append, where the
  # write offset passes the rows of all steps, else a growth. The buffers also hold the tokens
  # of each row that runs with the prompt that it has left (running_rows/1), because the
  # tokens of a prompt go into the state when it joins.
  defp fit(ctx, group = %{state: state}, n) when is_integer(n) do
    rows = buffer_rows(ctx, group)

    if rows_for(rows, rows.written, n) <= state.capacity,
      do: group,
      else: move(ctx, group, fit_capacity(ctx, state.capacity, rows, n))
  end

  # Gives the buffers `capacity`: a compaction in :append, else a growth or a shrink. The group
  # keeps the time of each move and the rows of the capacities that it copies, for the costs of
  # ContinuousBatching, which clears them. A read of one token waits for the copy to end.
  defp move(ctx, group, capacity) do
    from = group.state.capacity
    start = System.monotonic_time()

    group =
      if ctx.mode == :append,
        do: compact(ctx, group, capacity),
        else: grow(ctx, group, capacity)

    Nx.to_binary(group.state.history, limit: 1)
    ms = System.convert_time_unit(System.monotonic_time() - start, :native, :microsecond) / 1000
    rows = max(Enum.sum(capacity_path(ctx, from, capacity)), capacity)
    %{group | moves: [%{from: from, to: capacity, rows: rows, ms: ms} | group.moves]}
  end

  # The rows of a group that size its buffers: `written`, the rows that the buffers hold (the
  # write offset in :append, else the longest row), `kept`, the rows of the longest row that a
  # compaction or a growth keeps (its valid rows in :append, else its length), and `running`,
  # the rows of the longest row that has not ended with the prompt that it has left.
  defp buffer_rows(ctx, group = %{state: state}) do
    longest = Enum.max(state.lengths)

    if ctx.mode == :append do
      kept = group |> valid_rows() |> row_counts() |> Enum.max()
      %{written: state.used, kept: kept, running: running_rows(group)}
    else
      %{written: longest, kept: longest, running: running_rows(group)}
    end
  end

  # The capacity of the buffers after fit/3 for a step of n tokens, from the capacity now and
  # the rows of buffer_rows/2: the capacity now when the step fits, else the capacity of a
  # compaction in :append (see compact/3) or of a growth (see grow/3). compile_steps/3 reads it
  # to find the capacities of the first steps of a group.
  defp fit_capacity(ctx, capacity, rows, n) do
    if rows_for(rows, rows.written, n) <= capacity,
      do: capacity,
      else: moved_capacity(ctx, rows, n)
  end

  # The capacity of a compaction in :append, with @capacity_margin free rows, or of a growth,
  # for a step of n tokens after the rows that it keeps.
  defp moved_capacity(ctx, rows, n) do
    slack = if ctx.mode == :append, do: @capacity_margin, else: 0
    fitting(ctx, rows_for(rows, rows.kept, n), slack)
  end

  # The valid rows of each row that has not ended.
  defp valid_rows(group = %{state: state}),
    do: Enum.zip_with(state.valid, group.done, &if(&2, do: [], else: &1))

  defp row_counts(valid),
    do: Enum.map(valid, fn ranges -> ranges |> Enum.map(&elem(&1, 1)) |> Enum.sum() end)

  # The rows that the buffers need for a step of n tokens after `written` rows, and for the
  # tokens of each row that runs with the prompt that it has left. fit/3, grow/3, compact/3
  # and shrink/2 all size the buffers with it.
  defp rows_for(rows, written, n), do: max(written + n, rows.running)

  # The rows of a step write at one offset in :append, so the rows of all steps count, also
  # the gap rows of the shorter rows and the rows of the rows that ended. The compaction moves
  # the valid rows of each row that has not ended to the front of its buffers, in their order,
  # and the write offset goes back to the longest row. The buffers get `capacity`, which
  # moved_capacity/3 gives for the longest row and the next n tokens, or for the tokens of the
  # longest row with the prompt that it has left, and @capacity_margin rows, so a compaction frees rows for at least that many
  # steps and never cuts the tokens of a prompt. The capacity can be smaller or larger than
  # before. The keys and values keep the positions of their tokens, the masks come from the
  # new ranges, and the position offset of a merged decoder reads the valid rows of each row
  # or the write offset, which is the longest row.
  defp compact(ctx, group = %{state: state}, capacity) do
    valid = valid_rows(group)
    counts = row_counts(valid)
    longest = Enum.max(counts)

    index =
      for ranges <- valid do
        rows =
          for {start, count} <- Enum.reverse(ranges),
              row <- start..(start + count - 1)//1,
              do: row

        rows ++ List.duplicate(0, state.capacity - length(rows))
      end

    old = state.cache |> Map.take(ctx.buffers) |> Map.put(@history, state.history)
    moved = ctx.compact.(old, Nx.tensor(index, type: :s32, backend: Nx.BinaryBackend))
    release(Map.values(old), Map.values(moved))
    {history, moved} = Map.pop!(resize_buffers(ctx, moved, state.capacity, capacity), @history)

    state = %{
      state
      | cache: Map.merge(state.cache, moved),
        history: history,
        capacity: capacity,
        used: longest,
        valid: Enum.map(counts, &if(&1 == 0, do: [], else: [{0, &1}]))
    }

    %{group | state: state}
  end

  @doc false
  # Moves the buffers to the capacity of the longest row that runs, with the prompt that it has
  # left, and @join_tokens rows, when that capacity is smaller and `pays?` says that the move
  # pays, as after a long row ends in a server. So the steps run at the capacity of the rows
  # that run, not of the longest row since the group opened. The @join_tokens rows hold a step
  # of a joining prompt, so the next step does not grow the buffers again. `pays?` gets the
  # capacity, the target, the steps that the group can run at the target (see steps_below/2)
  # and the capacities of the path of the move. In :in_place, the rows that ended lose their
  # valid rows first (see drop_ended/1), so the move keeps only the rows that run.
  def shrink(ctx, group, pays?) do
    group = if ctx.mode == :in_place, do: drop_ended(group), else: group
    from = group.state.capacity
    to = capacity(ctx, running_rows(group) + @join_tokens)

    if to < from and pays?.(from, to, steps_below(group, to), capacity_path(ctx, from, to)),
      do: move(ctx, group, to),
      else: group
  end

  # The steps that the rows that run can take at `capacity` rows before one of them needs more:
  # a row grows by one row in each step until its budget ends. When no row needs more, the steps
  # until the last row ends.
  defp steps_below(group, capacity) do
    rows =
      for {rows, left, done?, budget, ids} <-
            Enum.zip([group.state.lengths, group.left, group.done, group.budgets, group.ids]),
          not done?,
          do: {rows + length(left), budget - length(ids)}

    passing = for {rows, remaining} <- rows, rows + remaining > capacity, do: capacity - rows
    Enum.min(passing, fn -> Enum.max(Enum.map(rows, &elem(&1, 1)), fn -> 0 end) end)
  end

  # The rows of the longest row that has not ended, with the tokens of its prompt that are left
  # (see plan/1), or 0.
  defp running_rows(group) do
    [group.state.lengths, group.left, group.done]
    |> Enum.zip_with(fn [rows, left, done?] -> if done?, do: 0, else: rows + length(left) end)
    |> Enum.max(fn -> 0 end)
  end

  # The capacity for `rows` rows and a slack of free rows, up to the largest capacity of the
  # decoder, which must fit the rows.
  defp fitting(ctx, rows, slack) do
    capacity = capacity(ctx, rows + slack)

    if rows > capacity do
      raise ArgumentError,
            "generation needs #{rows} rows of cache, more than the #{capacity} rows that the static tables of the decoder fit"
    end

    capacity
  end

  # Gives the buffers the capacity of capacity/2 that fits the rows of a step, up to the
  # largest capacity of the decoder. One compiled function pads all buffers with free
  # rows, which the masks hide, and each growth compiles the steps of one more capacity. The
  # tokens of the rows, which the processors read, have the same capacity and grow with the
  # buffers, also in :grow, where the cache has its exact length and only the tokens grow.
  defp grow(ctx, group = %{state: state}, capacity) do
    old = buffer_tensors(ctx, group)
    {history, grown} = Map.pop!(resize_buffers(ctx, old, state.capacity, capacity), @history)
    state = %{state | cache: Map.merge(state.cache, grown), history: history, capacity: capacity}
    %{group | state: state}
  end

  # The rows are axis 2 of a buffer, {batch, heads, rows, dim}, and axis 1 of the tokens,
  # {batch, rows}.
  defp rows_axis(tensor), do: if(Nx.rank(tensor) == 2, do: 1, else: 2)

  # The buffers of the cache of a group and its tokens, which a growth or a compaction moves.
  defp buffer_tensors(ctx, group) do
    buffers = if ctx.mode == :grow, do: %{}, else: Map.take(group.state.cache, ctx.buffers)
    Map.put(buffers, @history, group.state.history)
  end

  # Gives the buffers of `tensors` the capacity `to`, one step of capacity/2 at a time from the
  # capacity `from`, so each growth or shrink compiles only between neighbour capacities (see
  # compile_steps/3). Each step frees the buffers of the step before.
  defp resize_buffers(ctx, tensors, from, to) do
    Enum.reduce(capacity_path(ctx, from, to), tensors, fn capacity, tensors ->
      resized = ctx.grow.(tensors, Nx.iota({capacity}, backend: Nx.BinaryBackend))
      release(Map.values(tensors), Map.values(resized))
      resized
    end)
  end

  # The capacities after `from` on the way to `to`, up or down, each one step of capacity/2.
  defp capacity_path(ctx, from, to) do
    steps = capacities(ctx, max(from, to))

    if to >= from,
      do: Enum.filter(steps, &(&1 > from and &1 <= to)),
      else: steps |> Enum.filter(&(&1 >= to and &1 < from)) |> Enum.reverse()
  end

  @doc false
  # The largest capacity of capacity/2, up to ctx.max_capacity, whose peak of device memory
  # for `rows` rows (see peak/4) stays at most 95% of `limit` bytes, with `live` bytes already
  # in use. Without a limit, the capacity is ctx.max_capacity.
  def largest_capacity(ctx, _rows, _live, nil), do: ctx.max_capacity

  def largest_capacity(ctx, rows, live, limit) do
    [smallest | _] = capacities = capacities(ctx, ctx.max_capacity)

    case Enum.filter(capacities, &(peak(ctx, rows, live, &1) <= 0.95 * limit)) do
      [] -> raise memory_error(ctx, rows, live, limit, smallest)
      fits -> List.last(fits)
    end
  end

  # The peak of device memory of a group of `rows` rows at `capacity`, with `live` bytes in
  # use: the larger of a step at the capacity, with the temporary of the widest step, and a
  # growth into it from the capacity before, which holds the buffers of both capacities.
  defp peak(ctx, rows, live, capacity) do
    # Each row and position of a buffer holds a row of each cache buffer and a token of history.
    {position, cache_heads} =
      ctx.buffers
      |> Enum.filter(&(&1 in ctx.cache))
      |> Enum.map(fn name ->
        {type, [_batch, heads, _length, head_dim]} = ctx.model.input_types[name]
        {_kind, bits} = Nx.Type.normalize!(type)
        {heads * head_dim * div(bits, 8), heads}
      end)
      |> Enum.reduce({4, 0}, fn {bytes, h}, {sum, heads} -> {sum + bytes, max(heads, h)} end)

    # The temporaries of the widest step, from the buffers that XLA gives it, which it reuses
    # for each layer: two outputs of the widest matmul for each token, such as the merged gate
    # and up of the MLP and its activation, and the attention scores of one chunk of query rows
    # in f32, at most the budget of the chunks of the attention core, with the output of its
    # first dot, at most half of that in f16. For Qwen3-0.6B fp16 on EXLA CUDA, XLA gave a temp
    # buffer of 573 MB at 48 rows of 512 tokens, with 302 MB of gate and up and 151 MB of the
    # activation, and of 412 MB at 8 rows and a capacity of 1024, with 268 MB of f32 scores and
    # 134 MB of the f16 output of the dot. This estimate bounds each of 10 measured steps from 8
    # to 48 rows and from 512 to 4096 rows of capacity. A bf16 cache also has the f32 copy of the
    # cache of all layers that the core reads (see Transmuter.Ops.Transformer). Without a count
    # of query heads in the graph or the config, it takes 8 query heads for each head of the
    # cache, as in Llama 3 70B. That is not a bound for multi-query attention, which has one
    # cache head.
    width = Enum.max(ctx.widths.with_cache ++ ctx.widths.without_cache)
    heads = ctx.query_heads || 8 * cache_heads
    chunk = min(rows * heads * width * capacity * 4, Transformer.scores_budget())
    scores = div(3 * chunk, 2)
    widened = if bf16_cache?(ctx), do: rows * capacity * 2 * (position - 4), else: 0
    temporary = 2 * rows * width * ctx.matmul_bytes + scores + widened
    before = capacities(ctx, capacity - 1) |> List.last(0)
    buffers = &(rows * position * &1)
    live + max(buffers.(capacity) + temporary, buffers.(before) + buffers.(capacity))
  end

  defp bf16_cache?(ctx) do
    Enum.any?(ctx.buffers, fn name ->
      {type, _dims} = ctx.model.input_types[name]
      name in ctx.cache and Nx.Type.normalize!(type) == {:bf, 16}
    end)
  end

  @doc false
  # The error of a server whose group of `rows` rows at `capacity` does not fit the device. The
  # estimate and the budget are those of largest_capacity/4: the peak with the `live` bytes in
  # use, and 95% of `limit`.
  def memory_error(ctx, rows, live, limit, capacity) do
    budget =
      if limit,
        do: "#{round(0.95 * limit)} bytes, 95% of the :device_memory of #{limit} bytes",
        else: "not known"

    ArgumentError.exception(
      "the server cannot start: a :capacity of #{capacity} rows with #{rows} :slots needs " <>
        "about #{peak(ctx, rows, live, capacity)} bytes of device memory with the #{live} " <>
        "bytes in use, and the budget is #{budget}. Give a smaller :capacity, fewer :slots " <>
        "or a :device_memory that the device has"
    )
  end

  # The capacities of capacity/2 up to `bound`, from the smallest one.
  defp capacities(ctx, bound), do: capacities(ctx, capacity(ctx, 1), bound)

  defp capacities(ctx, capacity, bound) do
    next = capacity(ctx, capacity + 1)

    cond do
      capacity > bound -> []
      next == capacity -> [capacity]
      true -> [capacity | capacities(ctx, next, bound)]
    end
  end

  defp grow_rows(tensors, capacity) do
    Map.new(tensors, fn {name, tensor} -> {name, resize(tensor, Nx.axis_size(capacity, 0))} end)
  end

  # Pads the rows of a tensor with free rows, or keeps its first rows.
  defp resize(tensor, capacity) do
    axis = rows_axis(tensor)
    free = capacity - Nx.axis_size(tensor, axis)

    if free >= 0 do
      padding = List.replace_at(List.duplicate({0, 0, 0}, Nx.rank(tensor)), axis, {0, free, 0})
      Nx.pad(tensor, 0, padding)
    else
      Nx.slice_along_axis(tensor, 0, capacity, axis: axis)
    end
  end

  # Each row of a buffer takes the rows of `index`, {batch, capacity}, on its rows axis: a take
  # on a view with the axes before the rows axis and the rows axis together. The tokens keep
  # their places, and only their capacity follows the index.
  defp compact_rows(tensors, index) do
    {batch, capacity} = Nx.shape(index)

    Map.new(tensors, fn
      {@history, tokens} ->
        {@history, resize(tokens, capacity)}

      {name, tensor} ->
        axis = rows_axis(tensor)
        dims = Tuple.to_list(Nx.shape(tensor))
        rows = Enum.at(dims, axis)
        inner = dims |> Enum.slice(1, axis - 1) |> Enum.product()
        outer = Nx.iota({batch, inner, 1}, axis: 0) |> Nx.multiply(inner)
        start = Nx.add(outer, Nx.iota({batch, inner, 1}, axis: 1)) |> Nx.multiply(rows)
        flat = start |> Nx.add(Nx.new_axis(index, 1)) |> Nx.flatten()
        view = Nx.reshape(tensor, {:auto, dims |> Enum.drop(axis + 1) |> Enum.product()})
        shape = dims |> List.replace_at(axis, capacity) |> List.to_tuple()
        {name, view |> Nx.take(flat) |> Nx.reshape(shape)}
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

  # The mask of the buffer of a row, as runs (see host_tensor/2): ones at its valid rows, and
  # zeros at its gap rows and the free rows.
  defp valid_runs(ranges, capacity) do
    {runs, first} =
      Enum.reduce(ranges, {[], capacity}, fn {start, count}, {runs, stop} ->
        {[{1, count}, {0, stop - start - count} | runs], start}
      end)

    [{0, first} | runs]
  end

  # A host tensor of `type` from nested lists whose innermost lists are runs {value, count}.
  # Each run is one binary copy, so a mask of the capacity of a row costs two copies and not
  # one conversion for each element, as Nx.tensor/2 of a list does.
  defp host_tensor(nested, type) do
    {shape, data} = encode_runs(nested, Nx.Type.normalize!(type))

    data
    |> IO.iodata_to_binary()
    |> Nx.from_binary(type, backend: Nx.BinaryBackend)
    |> Nx.reshape(shape)
  end

  defp encode_runs(runs = [{_value, _count} | _], type) do
    data = for {value, count} <- runs, do: :binary.copy(scalar_binary(value, type), count)
    {{runs |> Enum.map(&elem(&1, 1)) |> Enum.sum()}, data}
  end

  defp encode_runs(lists, type) do
    parts = Enum.map(lists, &encode_runs(&1, type))
    {inner, _data} = hd(parts)
    {Tuple.insert_at(inner, 0, length(lists)), Enum.map(parts, &elem(&1, 1))}
  end

  defp scalar_binary(value, type),
    do: Nx.to_binary(Nx.tensor(value, type: type, backend: Nx.BinaryBackend))

  # The runs of one value each of nested lists of numbers, for host_tensor/2.
  defp runs(values = [value | _]) when is_number(value), do: Enum.map(values, &{&1, 1})
  defp runs(lists), do: Enum.map(lists, &runs/1)

  defp step_inputs(ctx, state, tokens, counts) do
    n = length(hd(tokens))

    cached? =
      not without_cache?(
        ctx,
        for({past, count} <- Enum.zip(state.lengths, counts), count > 0, do: past)
      )

    masks =
      for {past, valid} <- Enum.zip(state.lengths, state.valid) do
        case ctx.mode do
          :in_place -> [{1, past + n}, {0, state.capacity - past - n}]
          :append when cached? -> valid_runs(valid, state.capacity) ++ [{1, n}]
          :append -> [{1, n}]
          :grow -> [{1, past + n}]
        end
      end

    values = %{
      "input_ids" => runs(tokens),
      "attention_mask" => masks,
      "position_ids" => runs(positions(ctx, state, n)),
      @position_offset => runs(position_offset(ctx, state)),
      @key_mask => masks,
      @logit_index => runs(for count <- counts, do: [[max(count - 1, 0)]])
    }

    inputs =
      for {name, value} <- values,
          name in ctx.model.inputs,
          into: Map.merge(state.cache, state.inputs) do
        {type, _dims} = ctx.model.input_types[name]
        {name, host_tensor(value, type)}
      end

    # An export that declares num_logits_to_keep keeps all positions (0). With @logit_index,
    # the gather then takes the last token of each row before the lm_head. Without it, the
    # logits have all positions, and the selection reads each row at its offset, because the
    # rows of a step can end their tokens at different positions.
    flag = if cached?, do: 1, else: 0

    static =
      for {name, value} <- [{"use_cache_branch", [flag]}, {"num_logits_to_keep", 0}],
          name in ctx.model.inputs,
          into: %{} do
        {type, _dims} = ctx.model.input_types[name]
        {name, Nx.tensor(value, type: type, backend: Nx.BinaryBackend)}
      end

    {inputs, static}
  end

  # The cache entries in buffers: all of them, or the self-attention cache of a merged decoder
  # that MergedDecoder.rewrite/2 moved to :append.
  defp buffers(model) do
    if @position_offset in model.inputs,
      do: MergedDecoder.buffers(model),
      else: Enum.filter(model.inputs, &String.starts_with?(&1, "past_key_values."))
  end

  # A merged decoder has a branch without cache and a branch with cache (use_cache_branch).
  # The prompt rule applies to the merged decoders in :grow and to those that
  # MergedDecoder.rewrite/2 moves to :append.
  defp merged?(ctx) do
    "use_cache_branch" in ctx.model.inputs and
      (ctx.mode == :grow or @position_offset in ctx.model.inputs)
  end

  # The offset of the positions of each row of a decoder with the position offset of
  # MergedDecoder: the valid rows of the row less the capacity.
  defp position_offset(_ctx, state), do: for(past <- state.lengths, do: [past - state.capacity])

  # The rows of a step need one length of the cache in :grow, where the positions come from it.
  defp one_length?(ctx), do: ctx.mode == :grow

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
