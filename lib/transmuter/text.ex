defmodule Transmuter.Text do
  @moduledoc """
  Servings for text tasks. Each function loads a model repository (the ONNX file, the tokenizer and the configuration) and returns an `Nx.Serving` that takes text and gives results in the shapes of Bumblebee.

      serving = Transmuter.Text.text_embedding({:hf, "Xenova/all-MiniLM-L6-v2"})
      Nx.Serving.run(serving, "The cat sat on the mat")
      #=> %{embedding: #Nx.Tensor<f32[384]>}

  The tasks need the optional `tokenizers` dependency: add `{:tokenizers, "~> 0.5"}` to your deps.

  ## Options

  All tasks take these options:

    * `:compile` - `:batch_size` (the default is `1`) and `:sequence_length`, an integer or a list of lengths (the default is `[32, 128, 512]`, limited to the longest input of the model). The serving compiles the model for each length and pads each input to the smallest length that fits. Inputs longer than the largest length are truncated.

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
end
