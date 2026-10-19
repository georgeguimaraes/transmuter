# Pipelines

This guide shows how to run text models with the servings of `Transmuter.Text`. A serving takes text and gives results. It does the tokenization, the batching and the steps after the model, such as pooling and softmax.

## Installation

The servings need the `tokenizers` package:

```elixir
def deps do
  [
    {:transmuter, "~> 0.1"},
    {:emlx, "~> 0.5"},
    {:tokenizers, "~> 0.5"}
  ]
end
```

Chat templates need `minijinja_ex` with its Hugging Face compatibility. This version is not released yet. To use it, do these steps:

1. Add `{:minijinja_ex, github: "georgeguimaraes/minijinja_ex", branch: "feat/chat-template-compat"}` to your deps.
2. Set `MINIJINJA_EX_BUILD=1` when you compile the dependencies. This builds the NIF from source, so you need a Rust compiler.

The examples in this guide do not show the backend options. Without them, the model runs on `Nx.BinaryBackend`, which is very slow. Add `backend: EMLX.Backend, defn_options: [compiler: EMLX]` on Apple silicon, or the EXLA options on other machines. See [Backends and performance](backends_and_performance.md).

## What a serving loads

A serving loads a Hugging Face repository: the ONNX file, `tokenizer.json`, and the configuration files. You do not give a file name:

```elixir
serving = Transmuter.Text.text_classification({:hf, "Xenova/distilbert-base-uncased-finetuned-sst-2-english"})
```

A local directory with the same files also works:

```elixir
serving = Transmuter.Text.text_classification("/models/sst2")
```

ONNX exports keep each file in variants, for example `onnx/model.onnx` (fp32) and `onnx/model_q4.onnx` (4-bit weights). The serving selects a variant in this order:

1. The `:variant` option, such as `variant: :q4`.
2. The `dtype` in the `transformers.js_config` block of `config.json`.
3. The default of the task: fp32 for encoders and `:q4` for generation, when the repository has the file.
4. fp32, then the other variants.

## Tasks

| Function | Input | Result |
|---|---|---|
| `text_embedding/2` | text | `%{embedding: tensor}` |
| `text_classification/2` | text, or a `{text, text_pair}` tuple | `%{predictions: [%{label: label, score: score}]}` |
| `zero_shot_classification/3` | text | `%{predictions: [%{label: label, score: score}]}` |
| `token_classification/2` | text | `%{entities: [%{start: start, end: end, label: label, score: score, phrase: phrase}]}` |
| `fill_mask/2` | text with `[MASK]` | `%{predictions: [%{score: score, token: token}]}` |
| `question_answering/2` | `%{question: question, context: context}` | `%{results: [%{text: text, start: start, end: end, score: score}]}` |
| `generation/2` | text, or `%{messages: messages}` | `%{results: [%{text: text, token_summary: summary}]}` |

Each function also takes a list of inputs and then gives a list of results. The results have the shapes of Bumblebee. Their values match the pipelines of Python transformers and sentence-transformers, and the tests of Transmuter check this for each task.

### Embeddings

```elixir
serving = Transmuter.Text.text_embedding({:hf, "Xenova/bge-small-en-v1.5"})
%{embedding: embedding} = Nx.Serving.run(serving, "The cat sat on the mat.")
```

A sentence-transformers model gives its pooling, its normalization and its maximum length in `1_Pooling/config.json`, `modules.json` and `sentence_bert_config.json`. ONNX exports often do not have these files. The serving then reads them from the `base_model` in the model card. For example, `Xenova/bge-small-en-v1.5` gets CLS pooling and L2 normalization from `BAAI/bge-small-en-v1.5`. Without these files, the embedding is the mean of the token embeddings.

The serving raises an error when `modules.json` has a module that Transmuter does not run, such as a `Dense` layer. Set `:output_pool` and `:embedding_processor` to use the model without these modules.

### Classification and reranking

```elixir
serving = Transmuter.Text.text_classification({:hf, "Xenova/ms-marco-MiniLM-L-6-v2"})

Nx.Serving.run(serving, [
  {"How many people live in Berlin?", "Berlin has a population of 3,520,031 registered inhabitants."},
  {"How many people live in Berlin?", "New York City is famous for the Metropolitan Museum of Art."}
])
```

The labels come from `id2label` in `config.json`. The scores are a softmax, a sigmoid for a model with one label or with multi-label classification, and the logits for regression.

### Zero-shot classification

```elixir
serving =
  Transmuter.Text.zero_shot_classification(
    {:hf, "Xenova/distilbert-base-uncased-mnli"},
    ["urgent", "phone", "tablet"]
  )

Nx.Serving.run(serving, "My iphone is broken and I need it fixed today!")
```

You give the labels when you build the serving. A natural language inference (NLI) model compares the text with the hypothesis of each label, such as "This example is phone.". Set `:hypothesis_template` to change the hypothesis. The scores add up to 1. With `multi_label: true`, the model scores each label alone, so more than one label can have a high score.

Each label is one row of the batch. A text with 5 labels uses 5 rows of `:batch_size`.

### Question answering

```elixir
serving = Transmuter.Text.question_answering({:hf, "Xenova/distilbert-base-cased-distilled-squad"})

Nx.Serving.run(serving, %{
  question: "Where is the Eiffel Tower?",
  context: "The Eiffel Tower is a wrought-iron lattice tower in Paris."
})
#=> %{results: [%{text: "Paris", start: 52, end: 57, score: ...}]}
```

The answer is a part of the context, and `start` and `end` are byte offsets in the context. A long context becomes windows of 384 tokens that overlap by 128 tokens, as in Python transformers. A different `:sequence_length` in `:compile` changes the length of the windows. Each window is one row of the batch. Spans with the same text add their scores, also from different windows, and the text with the highest score is the answer.

### Named entities

```elixir
serving = Transmuter.Text.token_classification({:hf, "Xenova/bert-base-NER"}, aggregation: :same)
Nx.Serving.run(serving, "Hugging Face Inc. is a company based in New York City.")
#=> %{entities: [%{label: "ORG", phrase: "Hugging Face Inc", start: 0, end: 16, ...}, ...]}
```

`start` and `end` are byte offsets in the text, so `binary_part(text, start, end - start)` gives the span.

### Generation

```elixir
serving =
  Transmuter.Text.generation({:hf, "onnx-community/Qwen3-0.6B-ONNX"},
    max_new_tokens: 512,
    template_options: [enable_thinking: false]
  )

Nx.Serving.run(serving, %{messages: [%{role: "user", content: "Write a haiku about Elixir."}]})
```

Messages go through the chat template of the repository (see `Transmuter.ChatTemplate`). A string input does not. The settings of `generation_config.json` apply, such as sampling with `temperature`, `top_k` and `top_p`. The options override them. Give `:seed` in the input or as an option to get the same text again.

The serving works with any decoder export that has a KV cache. It reads the shape of the cache from the graph:

- Exports with GroupQueryAttention, such as the `onnx-community` exports of Qwen3 and SmolLM2, write the cache in place.
- Exports that declare `position_ids` keep the cache in a buffer with masked free rows.
- Merged decoders without `position_ids`, such as `Xenova/gpt2`, grow the cache by one row for each token. They run without compilation, so they are slower.

The prompt runs in parts with a power-of-two length, and each new token is a part of length 1. So a small number of compiled shapes covers all prompt lengths.

The serving runs one input at a time and does not stream tokens.

## Options of all tasks

- `:compile`: `:batch_size` (the default is 1) and `:sequence_length`, an integer or a list of lengths (the default is `[32, 128, 512]`, limited to the maximum length of the model). The serving compiles the model for each length and pads each input to the smallest length that fits. Inputs longer than the largest length are truncated. Generation does not use this option.
- `:backend` and `:compute_type`: options of `Transmuter.load!/2`.
- `:defn_options`: options of `Transmuter.compile/3`, such as `compiler: EMLX`.
- `:variant` and `:cache_dir`: see above and `Transmuter.Hub`.
- `:model` and `:tokenizer`: a model and a tokenizer that you loaded yourself.

## Run a serving in your application

`Nx.Serving.run/2` compiles the model on each call. In an application, start the serving under your supervision tree. It then compiles once, when it starts:

```elixir
children = [
  {Nx.Serving,
   serving:
     Transmuter.Text.text_embedding({:hf, "Xenova/all-MiniLM-L6-v2"},
       compile: [batch_size: 8, sequence_length: [64, 256]],
       backend: EMLX.Backend,
       defn_options: [compiler: EMLX]
     ),
   name: MyApp.Embedding,
   batch_timeout: 50}
]
```

Then call it from any process:

```elixir
Nx.Serving.batched_run(MyApp.Embedding, "The cat sat on the mat.")
```

Concurrent calls go into batches of up to 8 inputs, and each batch runs one compiled shape.
