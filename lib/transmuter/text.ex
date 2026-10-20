defmodule Transmuter.Text do
  @moduledoc """
  Servings for text tasks. Each function loads a model repository (the ONNX file, the tokenizer and the configuration) and returns an `Nx.Serving` that takes text and gives results in the shapes of Bumblebee.

      serving = Transmuter.Text.text_embedding({:hf, "Xenova/all-MiniLM-L6-v2"})
      Nx.Serving.run(serving, "The cat sat on the mat")
      #=> %{embedding: #Nx.Tensor<f32[384]>}

  The tasks need the optional `tokenizers` dependency: add `{:tokenizers, "~> 0.5"}` to your deps.

  ## Options

  All tasks take these options:

    * `:compile` - `:batch_size` (the default is `1`) and `:sequence_length`, an integer or a list of lengths (the default is `[32, 128, 512]`, limited to the longest input of the model). The serving compiles the model for each length and pads each input to the smallest length that fits. Inputs longer than the largest length are truncated. `generation/2` uses only `:batch_size`.

    * `:defn_options` - options of `Transmuter.compile/3`, such as `compiler: EMLX`.

    * `:backend` and `:compute_type` - options of `Transmuter.load!/2`, such as `backend: EMLX.Backend`.

    * `:variant` - the variant of the ONNX file, such as `:q4`. See `Transmuter.Repo`.

    * `:cache_dir` - see `Transmuter.Hub`.

    * `:model` and `:tokenizer` - a model from `Transmuter.load!/2` and a tokenizer from `Transmuter.Tokenizer.load!/2`, to use in place of the files of the repository.

  Run a serving with `Nx.Serving.run/2`, or start it under your supervision tree and call `Nx.Serving.batched_run/2`:

      children = [
        {Nx.Serving,
         serving: Transmuter.Text.text_embedding({:hf, "Xenova/all-MiniLM-L6-v2"},
           compile: [batch_size: 8],
           backend: EMLX.Backend,
           defn_options: [compiler: EMLX]
         ),
         name: MyApp.Embedding,
         batch_timeout: 50}
      ]

  `Nx.Serving.run/2` compiles the model on each call. A serving under a supervision tree compiles once, when it starts.
  """

  @doc """
  Builds a serving that turns text into an embedding.

  The input is a string or a list of strings. The result is `%{embedding: tensor}`, or a list of them for a list.

  A sentence-transformers model gives its pooling and normalization in `1_Pooling/config.json` and `modules.json`. When the repository does not have these files, they come from the `base_model` in its model card. Without them, the embedding is the mean of the token embeddings.

  ## Options

    * `:output_pool` - `:mean_pooling`, `:cls_token_pooling`, `:last_token_pooling`, or `nil` for the token embeddings.

    * `:embedding_processor` - `:l2_norm` or `nil`.

  See the module documentation for the other options.
  """
  defdelegate text_embedding(source, opts \\ []), to: Transmuter.Text.TextEmbedding

  @doc """
  Builds a serving that classifies text, such as sentiment analysis.

  The input is a string, a `{text, text_pair}` tuple for models that compare two texts (such as cross-encoders for reranking), or a list of them. The result is `%{predictions: [%{label: label, score: score}]}`, from the highest score.

  The labels come from `id2label` in `config.json`. As in Python transformers, the scores are the logits when `problem_type` is `"regression"`, a sigmoid when the model has one label or `problem_type` is `"multi_label_classification"`, and else a softmax.

  ## Options

    * `:top_k` - the number of predictions, or `nil` for all. The default is `5`.

    * `:scores_function` - `:softmax`, `:sigmoid` or `:none` for the logits.

  See the module documentation for the other options.
  """
  defdelegate text_classification(source, opts \\ []), to: Transmuter.Text.TextClassification

  @doc """
  Builds a serving that classifies text with labels that you give, with a natural language inference (NLI) model.

  The model compares the text with one hypothesis for each label, such as "This example is sports.". The score of a label comes from the entailment class of the model. The input is a string or a list of strings. The result is `%{predictions: [%{label: label, score: score}]}`, from the highest score.

  The entailment class is the class in `id2label` of `config.json` that starts with "entail". As in Python transformers, the contradiction class is the last class when the entailment class is the first, else the first class. Each label adds one sequence to the batch, so a text with 5 labels uses 5 rows of `:batch_size`.

  ## Options

    * `:hypothesis_template` - a function that takes a label and returns its hypothesis. The default is `&"This example is \#{&1}."`.

    * `:multi_label` - `false` (the default) gives scores that add up to 1: a softmax of the entailment logits over the labels. `true` scores each label alone: a softmax over the contradiction and entailment logits of the label. A single label always gets the score of `true`, as in Python transformers.

    * `:top_k` - the number of predictions, or `nil` for all. The default is `5`.

  See the module documentation for the other options.
  """
  defdelegate zero_shot_classification(source, labels, opts \\ []),
    to: Transmuter.Text.ZeroShotClassification

  @doc """
  Builds a serving that labels the tokens of a text, such as named entity recognition.

  The input is a string or a list of strings. The result is `%{entities: [%{start: start, end: end, label: label, score: score, phrase: phrase}]}`. `start` and `end` are byte offsets in the text.

  ## Options

    * `:aggregation` - how tokens become entities, as in Python transformers:
      * `nil` (the default) - each token is an entity.
      * `:same` - adjacent tokens with the same label become one entity. A `B-` label begins a new entity.
      * `:word_first`, `:word_average` and `:word_max` - the tokens of a word first get one label: the label of the first token, the label of the average scores, or the label of the token with the highest score. Then adjacent words group as with `:same`.

    * `:ignored_labels` - labels to leave out. The default is `["O"]`.

  See the module documentation for the other options.
  """
  defdelegate token_classification(source, opts \\ []), to: Transmuter.Text.TokenClassification

  @doc """
  Builds a serving that predicts the masked word of a text.

  The input is a string with `[MASK]` once, or a list of them. `[MASK]` becomes the mask token of the tokenizer, such as `<mask>`. The serving raises when the encoded text does not have exactly one mask token, for example when truncation removes it. The result is `%{predictions: [%{score: score, token: token}]}`, from the highest score.

  ## Options

    * `:top_k` - the number of predictions. The default is `5`.

  See the module documentation for the other options.
  """
  defdelegate fill_mask(source, opts \\ []), to: Transmuter.Text.FillMask

  @doc """
  Builds a serving that finds the answer to a question in a context, with an extractive question answering model.

  The input is `%{question: question, context: context}` or a list of them. The result is `%{results: [%{text: text, start: start, end: end, score: score}]}`. The text is a part of the context, and `start` and `end` are byte offsets in the context. The results are empty when no window has an answer in the context.

  The search follows the question answering pipeline of Python transformers. The score of a span of context tokens is the start probability of its first token times the end probability of its last token. A long context becomes windows of the largest sequence length, which is 384 tokens by default. The windows overlap by 128 tokens. Each window is one row of `:batch_size`. Spans with the same text add their scores, also from different windows, and the text with the highest score is the answer. `Transmuter.Text.QuestionAnswering` gives all the steps.

  ## Options

    * `:max_answer_length` - the maximum number of tokens in the answer. The default is `15`.

  See the module documentation for the other options.
  """
  defdelegate question_answering(source, opts \\ []), to: Transmuter.Text.QuestionAnswering

  @doc """
  Builds a serving that generates text with a decoder, such as a chat model.

  The input is a string, `%{text: string}`, `%{messages: messages}` or a list of them. Messages are maps with `:role` and `:content`, and they go through the chat template of the model (see `Transmuter.ChatTemplate`). A map input can also have `:seed`, and a chat can have `:tools`. The result is `%{results: [%{text: text, token_summary: %{input: count, output: count, padding: 0}}]}`. The text has only the new tokens.

  The settings come from `generation_config.json`, and the options below override them. Generation stops at an `eos_token_id` of the configuration or after `:max_new_tokens`.

  The default variant is `:q4` when the repository has it, because the 4-bit weights of a decoder are smaller and faster.

  ## Options

    * `:max_new_tokens` - the maximum number of new tokens. The default is `256`.

    * `:strategy` - `:greedy` or `:sampling`. The default is `:sampling` when `do_sample` in the configuration is `true`, else `:greedy`.

    * `:temperature`, `:top_k`, `:top_p` and `:repetition_penalty` - as in Python transformers. The defaults are `1.0`, `50`, `1.0` and `1.0`.

    * `:seed` - the seed of sampling for inputs without their own seed. The default is a new seed for each input.

    * `:template_options` - other variables of the chat template, such as `enable_thinking: false` for Qwen3.

    * `:compile` - `:batch_size`, the number of inputs that generate together. The default is `1`. The serving does not use `:sequence_length`, because it does not pad or truncate the prompt.

  The other options are those of the module documentation.

  ## Batches

  With `compile: [batch_size: 4]`, the serving runs up to 4 inputs in one loop. The inputs come from a list, or from concurrent calls to `Nx.Serving.batched_run/2`. The prompts can have different lengths. Each row stops at its own eos token. When a batch has fewer inputs, padding rows fill it. A larger batch gives more tokens per second in total, but each input gets its tokens more slowly.

  Each row samples with its own key from its seed, so the random numbers of a row do not depend on the batch. With greedy selection, and with sampling from a seed, an input gives the same text in a batch as alone. A batch can change the rounding of the logits, so in rare cases a token can be different.

  `Transmuter.Generation` tells how each mode of the KV cache keeps the rows apart. A merged decoder without `position_ids`, such as `Xenova/gpt2` or Gemma 3, runs the prompts of each length as a separate group, one group after the other. Each step of a group runs all rows of the batch. So a batch is faster for these models only when prompts have the same length.

  Prompts of up to 4096 tokens go into the same batch. A longer prompt goes into a batch with prompts of the same power-of-two width.
  """
  defdelegate generation(source, opts \\ []), to: Transmuter.Text.Generation

  @doc """
  Builds a serving that generates text from a source text with an encoder-decoder model, such as T5 or Marian for translation.

  The input is a string, `%{text: string}` or a list of them. A map input can also have `:seed`. The result is `%{results: [%{text: text, token_summary: %{input: count, output: count, padding: count}}]}`. `padding` is the number of pad tokens that the source text gets.

  The repository must have `onnx/encoder_model.onnx` and `onnx/decoder_model_merged.onnx`, as the exports of Xenova and onnx-community. The default variant is fp32. The settings come from `generation_config.json`, as in `generation/2`. The serving does not do beam search. It ignores `num_beams`.

      serving = Transmuter.Text.text_to_text({:hf, "Xenova/t5-small"})
      Nx.Serving.run(serving, "translate English to German: The house is wonderful.")
      #=> %{results: [%{text: "Das Haus ist wunderbar.", token_summary: %{input: 11, output: 5, padding: 21}}]}

  ## Options

    * `:compile` - `:sequence_length`, as for the other text tasks. The serving pads the source text to the smallest length that fits. `:batch_size` is the number of inputs that generate together, as in `generation/2`. Inputs of the same sequence length go into one batch.

    * `:max_new_tokens`, `:strategy`, `:temperature`, `:top_k`, `:top_p`, `:repetition_penalty`, and `:seed` - as in `generation/2`.

  The other options are those of the module documentation, but `:model` does not apply.
  """
  defdelegate text_to_text(source, opts \\ []), to: Transmuter.Text.TextToText
end
