# Pipelines

This guide shows how to run text models with the servings of `Transmuter.Text`, image models with the servings of `Transmuter.Vision`, and speech models with the servings of `Transmuter.Audio`. A serving takes text, images or audio and gives results. It does the tokenization or the preprocessing, the batching and the steps after the model, such as pooling and softmax.

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
| `text_to_text/2` | text | `%{results: [%{text: text, token_summary: summary}]}` |
| `Transmuter.Audio.speech_to_text/2` | PCM samples at 16 kHz | `%{chunks: [%{text: text, start_timestamp_seconds: start, end_timestamp_seconds: end}]}` |

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

#### Streaming

Set `stream: true` to get the text in chunks as the tokens arrive:

```elixir
serving = Transmuter.Text.generation({:hf, "HuggingFaceTB/SmolLM2-135M-Instruct"}, stream: true)

serving
|> Nx.Serving.run(%{messages: [%{role: "user", content: "Write a haiku about Elixir."}]})
|> Enum.each(&IO.write/1)
```

The serving returns a stream of strings. Consume the stream in the process that called the serving. The chunks join to the text of a serving without `:stream`. The serving decodes the text with the rule of Bumblebee and of `TextStreamer` in Python transformers:

- A chunk ends before the last space of the pending text, so a token that changes the space before it does not change a chunk that is already out.
- A newline or a CJK character ends a chunk at once.
- A character of several byte-level tokens, such as an emoji, comes out only when all of its tokens are there.

Set `stream_done: true` to get `{:done, %{token_summary: summary}}` after the last chunk. With `:stream`, the input must be one input, not a list. `text_to_text/2` takes the same options. `Transmuter.Audio.speech_to_text/2` does not stream, because it merges the windows of long audio only at the end.

#### Batches

Set `compile: [batch_size: n]` to generate for up to `n` inputs in one loop. The inputs come from a list, or from concurrent calls to `Nx.Serving.batched_run/2`. Each step of the loop runs the model once for all rows. On a GPU, a batch gives more tokens per second in total:

| Model | `batch_size: 1` | `batch_size: 4` | `batch_size: 8` |
|---|---|---|---|
| `HuggingFaceTB/SmolLM2-135M-Instruct` (q4) | 152 tokens/s | 214 tokens/s | 318 tokens/s |
| `onnx-community/Qwen3-0.6B-ONNX` (q4) | 89 tokens/s | 112 tokens/s | 122 tokens/s |

The table shows the total tokens per second of 16 concurrent requests to a started serving with `batch_timeout: 10`, with greedy selection and up to 128 new tokens for each request. The values are the median of 3 or more runs with EMLX on an Apple M2 Pro GPU. A batch runs until its longest row stops, so the gain is smaller when the lengths of the answers are very different.

The prompts of a batch can have different lengths. Each row has its own length in the KV cache, its own positions, its own sampling key and its own stop at the eos token. A row that stops gets padding tokens until the batch ends. When fewer inputs than `n` arrive, padding rows fill the batch. With greedy selection, and with sampling from a seed, each input gives the same text as when it runs alone.

The mode of the KV cache decides how the rows share the cache:

- With GroupQueryAttention, the attention mask of each row tells the operator how many rows of the cache are valid. The operator writes the new keys and values of each row after its valid rows.
- With `position_ids`, the attention mask hides the free rows of each row, and the new rows go into the buffer after the valid rows of each row.
- A merged decoder without `position_ids` gets its positions from the length of the cache. That length is the same for all rows, so only prompts of the same length can share a cache. The serving runs a group for each prompt length, one group after the other, and each step of a group runs all rows of the batch. So a batch is faster only for the prompts that have the same length. For 16 prompts with 7 different lengths, `Xenova/gpt2` gave 133 tokens/s with `batch_size: 1` and 148 tokens/s with `batch_size: 8`. The prompts of text-to-text and speech-to-text have one length, so all their rows run as one group. With streaming, the rows of a later group get their first token when the groups before it finish.

Prompts of up to 4096 tokens go into the same batch. A longer prompt goes into a batch with prompts of the same power-of-two width.

### Translation and other text-to-text tasks

```elixir
serving = Transmuter.Text.text_to_text({:hf, "Xenova/opus-mt-en-de"})
Nx.Serving.run(serving, "The house is wonderful.")
```

The serving runs an encoder-decoder model, such as T5 or Marian. The repository must have `onnx/encoder_model.onnx` and `onnx/decoder_model_merged.onnx`. The serving does these steps:

1. It encodes the text and pads it to a length of `:compile`, as the other text tasks.
2. It runs the encoder once.
3. It starts the decoder from `decoder_start_token_id`. The first step of the decoder computes the cross-attention cache from the outputs of the encoder. The later steps use this cache again.

T5 takes the task in the text, for example `"translate English to German: The house is wonderful."`. A Marian model translates one language pair, and its tokenizer encodes the text with `source.spm`, as `MarianTokenizer` of Python transformers.

The settings of `generation_config.json` apply, as for `generation/2`. The serving does not do beam search, so it ignores `num_beams`. With `strategy: :greedy`, the text is the same as the `generate` function of transformers with `num_beams=1`.

### Speech recognition

```elixir
audio = Nx.from_binary(File.read!("speech.f32"), :f32)
serving = Transmuter.Audio.speech_to_text({:hf, "onnx-community/whisper-base"}, language: "en")
Nx.Serving.run(serving, audio)
#=> %{chunks: [%{text: " Mr. Quilter is the apostle of the middle classes, ...", ...}]}
```

`Transmuter.Audio.speech_to_text/2` runs Whisper on PCM samples at 16 kHz. Convert other audio before you run it, for example with `ffmpeg -i speech.mp3 -ac 1 -ar 16000 -f f32le speech.f32`.

The audio can have any length. The serving splits audio longer than 30 s into windows that overlap, as the pipeline of transformers with `chunk_length_s=30`. It transcribes each window and merges the texts at the overlaps. The `:chunk_num_seconds` and `:context_num_seconds` options set the windows.

With `compile: [batch_size: n]`, the decoder transcribes up to `n` windows together. The windows come from one long audio, from a list of audio, or from concurrent calls. For 62 s of audio, `Xenova/whisper-tiny.en` takes 732 ms with `batch_size: 4` and 1040 ms with `batch_size: 1`.

With `timestamps: :segments`, the result has one chunk for each segment of speech, with its start and end in seconds:

```elixir
serving = Transmuter.Audio.speech_to_text({:hf, "Xenova/whisper-tiny.en"}, timestamps: :segments)
Nx.Serving.run(serving, audio)
#=> %{chunks: [%{text: " Mr. Quilter is ...", start_timestamp_seconds: 0.0, end_timestamp_seconds: 50.44}, ...]}
```

See [Speech and audio](speech_and_audio.md) for the details.

## Images

The servings of `Transmuter.Vision` take a `StbImage` struct or a tensor of shape `{height, width, channels}` of type `:u8`. To read image files, add `{:stb_image, "~> 1.0"}` to your deps:

```elixir
image = StbImage.read_file!("cats.jpg")
```

| Function | Input | Result |
|---|---|---|
| `image_classification/2` | image | `%{predictions: [%{label: label, score: score}]}` |
| `image_embedding/2` | image | `%{embedding: tensor}` |
| `zero_shot_image_classification/3` | image | `%{predictions: [%{label: label, score: score}]}` |

Each function also takes a list of images and then gives a list of results. The tests of Transmuter compare each task with the pipeline of Python transformers on the same image. The labels are the same, and the scores agree within 1.0e-4.

### Image preprocessing

`Transmuter.Featurizer` reads `preprocessor_config.json` and prepares the images as the image processors of Python transformers do. It converts the image to RGB, resizes it, crops the center, rescales the values and normalizes them. The result is a tensor of shape `{batch, 3, height, width}` and type `:f32`.

The resize gives the same pixels as Pillow, for all `resample` filters of Pillow. With the hamming filter, a small number of pixels can differ by 1. The scores of a model change when the pixels change, so this match is important. The serving resizes on the `:backend` of the task, because the resize is slow on `Nx.BinaryBackend`.

The serving compiles the model for one image size, which comes from `size` or `crop_size` in `preprocessor_config.json`. The serving raises an error when the size depends on the image, for example with `shortest_edge` and no crop.

### Image classification

```elixir
serving = Transmuter.Vision.image_classification({:hf, "Xenova/vit-base-patch16-224"})
Nx.Serving.run(serving, image)
#=> %{predictions: [%{label: "Egyptian cat", score: 0.94}, ...]}
```

The labels come from `id2label` in `config.json`.

### Image embeddings

```elixir
serving = Transmuter.Vision.image_embedding({:hf, "Xenova/clip-vit-base-patch32"})
%{embedding: embedding} = Nx.Serving.run(serving, image)
```

For CLIP and SigLIP, the serving uses `vision_model.onnx`, and the embedding is the `image_embeds` output. For other models, the embedding is `pooler_output`, or else the first token of `last_hidden_state`. Set `embedding_processor: :l2_norm` to normalize the embedding.

### Zero-shot image classification

```elixir
labels = ["cats", "dogs", "a remote control"]
serving = Transmuter.Vision.zero_shot_image_classification({:hf, "Xenova/clip-vit-base-patch32"}, labels)
Nx.Serving.run(serving, image)
#=> %{predictions: [%{label: "cats", score: 0.98}, ...]}
```

The serving uses `model.onnx`, which has the text tower, the image tower, and the logit scale and bias of the model. `vision_model.onnx` and `text_model.onnx` do not have the logit scale and bias. Each label goes into the template `"This is a photo of {}."`, as in Python transformers. Use `:hypothesis_template` to change it.

The scores are a softmax over the labels for CLIP. For SigLIP, the scores are a sigmoid of each label, so they do not add up to 1. The SigLIP tokenizer also removes the punctuation of the text and makes it lowercase, as in Python transformers.

## Options of all tasks

- `:compile`: `:batch_size` (the default is 1) and, for text, `:sequence_length`, an integer or a list of lengths (the default is `[32, 128, 512]`, limited to the maximum length of the model). The serving compiles the model for each length and pads each input to the smallest length that fits. Inputs longer than the largest length are truncated. Generation and speech-to-text use only `:batch_size`, and they do not truncate the prompt.
- `:backend` and `:compute_type`: options of `Transmuter.load!/2`.
- `:defn_options`: options of `Transmuter.compile/3`, such as `compiler: EMLX`.
- `:variant` and `:cache_dir`: see above and `Transmuter.Hub`.
- `:model`, `:tokenizer` and `:featurizer`: a model, a tokenizer and a featurizer that you loaded yourself.

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
