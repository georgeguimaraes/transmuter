# Pipelines

This guide shows how to run text models with the servings of `Transmuter.Text`, image models with the servings of `Transmuter.Vision`, and speech models with the servings of `Transmuter.Audio`. A serving takes text, images or audio and gives results. It does the tokenization or the preprocessing, the batching and the steps after the model, such as pooling and softmax.

## Installation

The servings need the `tokenizers` package:

```elixir
def deps do
  [
    {:transmuter, "~> 2.0"},
    {:emlx, "~> 0.5"},
    {:tokenizers, "~> 0.5"}
  ]
end
```

Chat templates need `{:otera, "~> 0.1"}`. [Otera](https://hex.pm/packages/otera) renders Jinja templates in pure Elixir, with the same output as `apply_chat_template` of Python transformers.

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
| `text_ranking/2` | `%{query: query, documents: documents}` | `%{results: [%{index: index, score: score}]}` |
| `zero_shot_classification/3` | text | `%{predictions: [%{label: label, score: score}]}` |
| `token_classification/2` | text | `%{entities: [%{start: start, end: end, label: label, score: score, phrase: phrase}]}` |
| `zero_shot_entity_recognition/2` | `%{text: text, labels: labels}` | `%{entities: [%{start: start, end: end, label: label, score: score, phrase: phrase}]}` |
| `fill_mask/2` | text with `[MASK]` | `%{predictions: [%{score: score, token: token}]}` |
| `question_answering/2` | `%{question: question, context: context}` | `%{results: [%{text: text, start: start, end: end, score: score}]}` |
| `generation/2` | text, or `%{messages: messages}` | `%{results: [%{text: text, token_summary: summary}]}` |
| `text_to_text/2` | text | `%{results: [%{text: text, token_summary: summary}]}` |
| `Transmuter.Audio.speech_to_text/2` | PCM samples at 16 kHz, a file or a stream | `%{chunks: [%{text: text, start_timestamp_seconds: start, end_timestamp_seconds: end}]}` |
| `Transmuter.Audio.audio_classification/2` | PCM samples at 16 kHz, a file or a stream | `%{predictions: [%{label: label, score: score}]}` |
| `Transmuter.Audio.zero_shot_audio_classification/3` | PCM samples at 48 kHz, a file or a stream | `%{predictions: [%{label: label, score: score}]}` |
| `Transmuter.Audio.text_to_speech/2` | text, or `%{text: text, seed: seed}` | `%{audio: tensor, sampling_rate: rate}` |

Each function also takes a list of inputs and then gives a list of results. The results have the shapes of Bumblebee. Bumblebee has no text-to-speech, so `text_to_speech/2` gives the shape of the pipeline of Python transformers. Their values match the pipelines of Python transformers and sentence-transformers, and the tests of Transmuter check this for each task. The audio of text-to-speech depends on a random seed, so the tests compare the audio without noise.

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

### Reranking

```elixir
serving = Transmuter.Text.text_ranking({:hf, "Xenova/ms-marco-MiniLM-L-6-v2"})

Nx.Serving.run(serving, %{
  query: "How many people live in Berlin?",
  documents: [
    "New York City is famous for the Metropolitan Museum of Art.",
    "Berlin has a population of 3,520,031 registered inhabitants."
  ]
})
#=> %{results: [%{index: 1, score: 8.76}, %{index: 0, score: -11.25}]}
```

The scores are those of `CrossEncoder.rank` of sentence-transformers. The serving reads the sentence-transformers files of the model, or of the `base_model` in the model card, as for embeddings. A causal language model with a `LogitScore` module, such as `onnx-community/Qwen3-Reranker-0.6B-ONNX`, scores the text of its chat template with the logits of "yes" and "no". These models give the logits of all positions, so use a small `:batch_size` and `:sequence_length`.

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

### Entities of your labels

```elixir
serving = Transmuter.Text.zero_shot_entity_recognition({:hf, "onnx-community/gliner_small-v2.1"})

Nx.Serving.run(serving, %{
  text: "Cristiano Ronaldo plays for Al Nassr.",
  labels: ["person", "organization"]
})
#=> %{entities: [%{label: "person", phrase: "Cristiano Ronaldo", start: 0, end: 17, ...}, ...]}
```

A GLiNER model finds the entities of the labels that you give, and each input can have its own labels. The steps follow gliner in Python. Span models and token models (such as `knowledgator/gliner-pii-small-v1.0`) both run. `:threshold` sets the lowest score, `flat_ner: false` keeps an entity inside another entity, and `multi_label: true` lets a span have more than one label.

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

The logits processors of Python transformers that read the tokens of a row also apply: `no_repeat_ngram_size`, `min_length`, `min_new_tokens`, `forced_bos_token_id` and `forced_eos_token_id`. They count the tokens as Python does. `min_length` and the n-grams include the prompt, and `min_new_tokens` counts only the new tokens. `Transmuter.Generation` gives the rule of each processor.

The serving works with any decoder export that has a KV cache. It reads the shape of the cache from the graph:

- Exports with GroupQueryAttention that counts the valid rows from the values of the attention mask, such as the `onnx-community` exports of Qwen3 and SmolLM2, write the cache in place.
- Exports that declare `position_ids` keep the cache in a buffer with masked free rows.
- Merged decoders without `position_ids`, such as `Xenova/gpt2`, also keep the cache in a buffer. `Transmuter.Generation.MergedDecoder` gives their positions the valid rows of the buffer.
- Other decoders, such as Gemma 4 E2B, where a node reads the cache, grow the cache by one row for each token. Each cache length is a new shape, which compiles on its first step and stays cached. So the first request is slower.

`Transmuter.Generation` describes the three modes of the cache.

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

Set `stream_done: true` to get `{:done, %{token_summary: summary}}` after the last chunk. With `:stream`, the input must be one input, not a list. `text_to_text/2`, `Transmuter.Vision.image_to_text/2` and `Transmuter.Vision.image_text_to_text/2` take the same options. `Transmuter.Audio.speech_to_text/2` does not stream, because it merges the windows of long audio only at the end.

#### Batches

Set `compile: [batch_size: n]` to generate for up to `n` inputs in one loop. The inputs come from a list, or from concurrent calls to `Nx.Serving.batched_run/2`. Each step of the loop runs the model once for all rows. On a GPU, a batch gives more tokens per second in total:

| Model | `batch_size: 1` | `batch_size: 4` | `batch_size: 8` |
|---|---|---|---|
| `HuggingFaceTB/SmolLM2-135M-Instruct` (q4) | 128 tokens/s | 337 tokens/s | 529 tokens/s |
| `onnx-community/Qwen3-0.6B-ONNX` (q4) | 86 tokens/s | 199 tokens/s | 268 tokens/s |
| `Xenova/gpt2` (fp32) | 166 tokens/s | 401 tokens/s | 497 tokens/s |

The table shows the total tokens per second of 16 concurrent requests to a started serving with `batch_timeout: 10`, with greedy selection and up to 128 new tokens for each request. The values are the median of 5 runs with EMLX on an Apple M2 Pro GPU, from `bench/generation_throughput.exs`. The prompts of `Xenova/gpt2` have 7 different lengths. A batch runs until its longest row stops, so the gain is smaller when the lengths of the answers are very different.

The prompts of a batch can have different lengths. Each row has its own length in the KV cache, its own positions, its own sampling key and its own stop at the eos token. A row that stops gets padding tokens until the batch ends. When fewer inputs than `n` arrive, padding rows fill the batch. With greedy selection, and with sampling from a seed, each input gives the same text as when it runs alone.

The mode of the KV cache decides how the rows share the cache:

- When GroupQueryAttention writes the cache in place, the attention mask of each row tells the operator how many rows of the cache are valid. The operator writes the new keys and values of each row after its valid rows.
- With `position_ids`, all rows of a step write their new rows at one place in the buffer, after the rows of the earlier steps. A row with fewer tokens than the step leaves gap rows there. The attention mask of each row hides its gap rows and the free rows.
- A decoder that grows its cache gets its positions from the length of the cache, which is the same for all rows.

A decoder that grows its cache needs one length for all rows. For such a decoder, the serving runs a group for each prompt length, one group after the other, and each step of a group runs all rows of the batch. So a batch is faster only for the prompts that have the same length. A merged decoder of optimum runs the prompt of each length as one step on its branch without cache (see "Merged decoders" in `Transmuter.Generation`). When it reads its positions from a table, as `Xenova/gpt2` does, the rows of all prompt lengths then decode together, while the sum of the prompt lengths and the new tokens fits the static tables of the decoder. Other merged decoders, such as M2M100 and Whisper, run a group for each prompt length. The prompts of text-to-text and speech-to-text have one length, so all their rows run as one group. With streaming, the rows of a later group get their first token when the groups before it finish.

Prompts of up to 4096 tokens go into the same batch. A longer prompt goes into a batch with prompts of the same power-of-two width.

#### Continuous batching

A serving of `Transmuter.Text.generation/2` runs each batch until its longest row ends. A request that arrives during a batch waits for the whole batch, and the rows that end early stay empty. `Transmuter.Text.ContinuousGeneration` is a server that gives a free row to a waiting request at the next step:

```elixir
children = [
  {Transmuter.Text.ContinuousGeneration,
   name: MyGen, source: {:hf, "HuggingFaceTB/SmolLM2-135M-Instruct"}, slots: 8}
]

Supervisor.start_link(children, strategy: :one_for_one)

Transmuter.Text.ContinuousGeneration.run(MyGen, %{messages: [%{role: "user", content: "Hi"}]})
```

`run/3` returns the same result as `Nx.Serving.batched_run/2`, and `stream/3` returns the same chunks as a serving with `stream: true`. A request gives the same text as alone. The server takes decoders that write the cache in place. It raises at the start for a merged decoder, a decoder that grows its cache and beam search. A decoder in `:append` compacts its buffers when the rows of its steps pass them. The module documentation gives the reasons.

Requests that arrive over time, at 0.8 and 1.2 times what 8 rows can give, get their first token much sooner from the server, with about the same tokens per second:

| Model, on CUDA | Requests/s | Serving: tokens/s, first token p50 | Server: tokens/s, first token p50 |
|---|---|---|---|
| `HuggingFaceTB/SmolLM2-135M-Instruct` (q4) | 1.61 | 300.6 tokens/s, 1391 ms | 313.7 tokens/s, 65 ms |
| `onnx-community/Qwen3-0.6B-ONNX` (q4) | 0.68 | 160.0 tokens/s, 3144 ms | 172.4 tokens/s, 176 ms |
| `onnx-community/Qwen3-0.6B-ONNX` (q4) | 1.03 | 242.4 tokens/s, 5188 ms | 238.5 tokens/s, 2300 ms |

The table shows 48 requests with greedy selection and up to 256 new tokens, with 8 rows on an RTX 3060 with EXLA, from `bench/continuous_generation.exs`. A warm round of the same requests at the same times comes first, so the values do not include the first compiles.

#### Beam search

Set `:num_beams` to keep the best `n` sequences at each step and return the best one at the end:

```elixir
serving = Transmuter.Text.generation({:hf, "HuggingFaceTB/SmolLM2-135M-Instruct"}, num_beams: 3)
```

The default is `num_beams` of `generation_config.json`, else 1, as in Python transformers. `:strategy` does not change it, so a model with `num_beams` in its configuration, such as Marian, runs beam search by default. Give `num_beams: 1` for greedy selection. `:length_penalty` and `:early_stopping` work as in Python transformers, and their defaults also come from the configuration. Beam search takes greedy selection, not sampling, and it does not stream, because a later step can change the earlier tokens of the best sequence.

The beams of an input are rows of the batch, so the decoder compiles for `batch_size * num_beams` rows. The encoder of an encoder-decoder model runs once for each input. The text is the same as the `generate` function of Python transformers 4.57 with the same `num_beams`. All tasks that generate text take these options.

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

The settings of `generation_config.json` apply, as for `generation/2`. As in Python transformers, `num_beams` of the configuration gives beam search, so Marian runs 4 beams by default. The text is the same as the `generate` function of transformers with the same `num_beams`.

#### NLLB and M2M100

```elixir
serving =
  Transmuter.Text.text_to_text({:hf, "Xenova/nllb-200-distilled-600M"},
    source_language_token: "eng_Latn",
    target_language_token: "fra_Latn"
  )

Nx.Serving.run(serving, "The house is wonderful.")
```

NLLB and M2M100 translate between many languages, and each language has a token in the tokenizer: `"eng_Latn"` and `"fra_Latn"` for NLLB, `"__en__"` and `"__fr__"` for M2M100. The serving does what the `translation` pipeline of transformers does with `src_lang` and `tgt_lang`:

1. The source language token goes at the start of the text, in the place of the language token in the template of the post-processor of `tokenizer.json`. The text then ends with `</s>`.
2. The id of the target language token is `forced_bos_token_id`, so the first new token of the decoder is the target language token.

Without `:source_language_token`, the text gets the language token of `tokenizer.json`, which is English. Without `:target_language_token`, the decoder does not get a forced token.

### Speech recognition

```elixir
audio = Nx.from_binary(File.read!("speech.f32"), :f32)
serving = Transmuter.Audio.speech_to_text({:hf, "onnx-community/whisper-base"}, language: "en")
Nx.Serving.run(serving, audio)
#=> %{chunks: [%{text: " Mr. Quilter is the apostle of the middle classes, ...", ...}]}
```

`Transmuter.Audio.speech_to_text/2` runs Whisper on PCM samples at 16 kHz. For an audio file, give `{:file, path}`: ffmpeg decodes it and changes its sampling rate to 16 kHz. The `ffmpeg` executable must be in the `PATH`.

```elixir
Nx.Serving.run(serving, {:file, "speech.mp3"})
```

The input can also be a stream of tensors of PCM samples, such as the parts of a long recording. The serving transcribes the windows of the stream when they are complete, and gives the result of the whole audio when the stream ends. See `Transmuter.Audio` for the inputs.

The audio can have any length. The serving splits audio longer than 30 s into windows that overlap, as the pipeline of transformers with `chunk_length_s=30`. It transcribes each window and merges the texts at the overlaps. The `:chunk_num_seconds` and `:context_num_seconds` options set the windows.

With `compile: [batch_size: n]`, the decoder transcribes up to `n` windows together. The windows come from one long audio, from a list of audio, or from concurrent calls. For 62 s of audio, `Xenova/whisper-tiny.en` takes 732 ms with `batch_size: 4` and 1040 ms with `batch_size: 1`.

With `timestamps: :segments`, the result has one chunk for each segment of speech, with its start and end in seconds:

```elixir
serving = Transmuter.Audio.speech_to_text({:hf, "Xenova/whisper-tiny.en"}, timestamps: :segments)
Nx.Serving.run(serving, audio)
#=> %{chunks: [%{text: " Mr. Quilter is ...", start_timestamp_seconds: 0.0, end_timestamp_seconds: 50.44}, ...]}
```

With `timestamps: :words`, the result has one chunk for each word, as `return_timestamps="word"` in transformers. The decoder must give its cross attentions, as the decoder of `Xenova/whisper-tiny.en` and the `_timestamped` exports of onnx-community do:

```elixir
serving = Transmuter.Audio.speech_to_text({:hf, "Xenova/whisper-tiny.en"}, timestamps: :words)
Nx.Serving.run(serving, audio)
#=> %{chunks: [%{text: " Mr.", start_timestamp_seconds: 0.0, end_timestamp_seconds: 0.96}, ...]}
```

See [Speech and audio](speech_and_audio.md) for the details.

### Audio classification

```elixir
audio = Nx.from_binary(File.read!("speech.f32"), :f32)
serving = Transmuter.Audio.audio_classification({:hf, "Xenova/ast-finetuned-audioset-10-10-0.4593"})
Nx.Serving.run(serving, audio)
#=> %{predictions: [%{label: "Speech", score: 0.84}, %{label: "Speech synthesizer", score: 0.025}, ...]}
```

`Transmuter.Audio.audio_classification/2` runs AST and wav2vec2 classifiers on PCM samples at the sampling rate of the model. `Transmuter.Audio.Featurizer` prepares the audio as the feature extractor of Python transformers, which `feature_extractor_type` in `preprocessor_config.json` selects. The scores are a softmax of the logits, as in the `audio-classification` pipeline of Python transformers.

AST takes 1024 frames of log-mel features, about 10 s of audio. The serving cuts longer audio and pads shorter audio, so the audio goes into batches of `compile: [batch_size: n]`. wav2vec2 takes the mean over all frames of the audio, so padding would change the result. The serving runs each audio alone, and the model compiles each new audio length on its first call.

The tests compare the labels and the scores with the pipeline of Python transformers for AST, and with ONNX Runtime for `onnx-community/wav2vec2-base-Speech_Emotion_Recognition-ONNX`, whose PyTorch model is no longer on the Hub. The labels are the same, and the scores agree within 1.0e-4.

### Zero-shot audio classification

```elixir
audio = Nx.from_binary(File.read!("speech_48k.f32"), :f32)
labels = ["a man speaking", "music", "a dog barking", "rain"]
serving = Transmuter.Audio.zero_shot_audio_classification({:hf, "Xenova/clap-htsat-unfused"}, labels)
Nx.Serving.run(serving, audio)
#=> %{predictions: [%{label: "a man speaking", score: 0.95}, %{label: "music", score: 0.047}, ...]}
```

`Transmuter.Audio.zero_shot_audio_classification/3` runs CLAP on PCM samples at 48 kHz, the sampling rate of the model. For an audio file at another sampling rate, give `{:file, path}`, and ffmpeg changes the sampling rate to 48 kHz. Each label goes into the hypothesis template `"This is a sound of {}."`, and the scores are a softmax of `logits_per_audio` over the labels, as in the `zero-shot-audio-classification` pipeline of Python transformers.

`Transmuter.Audio.Featurizer` gives the log-mel features of `ClapFeatureExtractor`. CLAP takes 10 s of audio. Shorter audio repeats and gets zeros at the end, so the audio goes into batches of `compile: [batch_size: n]`. For longer audio, Python transformers takes 10 s at a random position, and the serving takes the first 10 s.

The tests compare the features with `ClapFeatureExtractor` within 1.0e-3 dB. With the fp32 file, the labels are the same as those of the pipeline of Python transformers, and the scores agree within 1.0e-4. The q8 file quantizes the activations when the model runs, and its scores are up to 0.13 away from the fp32 model. Its scores are within 0.007 of ONNX Runtime on the same file on the EXLA CPU, and within 0.047 on CUDA. The quantization makes small differences large: when the features change by 1e-5 of their values, the scores of ONNX Runtime change by up to 0.014.

### Speech synthesis

```elixir
serving = Transmuter.Audio.text_to_speech({:hf, "Xenova/mms-tts-eng"})
Nx.Serving.run(serving, %{text: "Hello, my dog is cute.", seed: 42})
#=> %{audio: #Nx.Tensor<f32[31744]>, sampling_rate: 16000}
```

`Transmuter.Audio.text_to_speech/2` turns text into PCM samples. Write them to a file with `File.write!("speech.f32", Nx.to_binary(audio))`, and convert the file with `ffmpeg -f f32le -ar 16000 -ac 1 -i speech.f32 speech.wav`. The serving runs two model types:

- VITS, such as the MMS-TTS models of Meta (`Xenova/mms-tts-eng`), makes the audio in one pass.
- SpeechT5 (`Xenova/speecht5_tts`) makes a spectrogram step by step and turns it into audio with a vocoder. It needs a speaker embedding of 512 values in the `:speaker_embedding` option or in the input.

```elixir
speaker = Nx.from_binary(File.read!("speaker.f32"), :f32)
serving = Transmuter.Audio.text_to_speech({:hf, "Xenova/speecht5_tts"}, speaker_embedding: speaker)
Nx.Serving.run(serving, %{text: "Hello, my dog is cute.", seed: 42})
#=> %{audio: #Nx.Tensor<f32[25088]>, sampling_rate: 16000}
```

Both models are random, also in Python. The seed of an input decides the audio, and an input without a seed gets a random seed. With `noise: false`, the random operators give fixed values. VITS then gives the audio of Python transformers with `noise_scale` and `noise_scale_duration` at 0, and the decoder of SpeechT5 keeps all values in its dropout. The tests compare this audio with PyTorch: the length is the same, and the samples are within 1e-4.

The serving runs each text alone, because the length of the audio depends on the text. Kokoro reads phonemes, not text, so the serving does not run it. See [Speech and audio](speech_and_audio.md) for the details.

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
| `image_to_text/2` | image, or `%{image: image, seed: seed}`. Florence-2 also takes `:task` and `:text` | `%{results: [%{text: text}]}`. Florence-2 adds the keys of the task, such as `:boxes` and `:labels` |
| `image_text_to_text/2` | `%{messages: messages}` with image parts | `%{results: [%{text: text, token_summary: summary}]}` |
| `object_detection/2` | image | `%{detections: [%{label: label, score: score, box: box}]}` |
| `zero_shot_object_detection/3` | image | `%{detections: [%{label: label, score: score, box: box}]}` |
| `depth_estimation/2` | image | `%{predicted_depth: tensor, depth: tensor}` |
| `image_segmentation/2` | image | `%{segments: [%{label: label, score: nil, mask: tensor}]}` |
| `background_removal/2` | image | `%{mask: tensor}` |
| `mask_generation/2` | `%{image: image, prompts: prompts}` | `%{masks: [[tensor]], iou_scores: [[score]]}` |
| `image_to_image/2` | image | `%{image: tensor}` |

Each function also takes a list of images and then gives a list of results. The tests of Transmuter compare each task with the pipeline of Python transformers on the same image. The labels are the same, and the scores agree within 1.0e-4. For `image_to_text/2`, the text is the same as the text of the `image-to-text` pipeline in Python transformers. For Florence-2, the text is the same as the beam search of Python transformers on the image features of the export, and the coordinates are those of `post_process_generation` before it truncates them to integers. For `image_text_to_text/2`, the text is the same as the text of ONNX Runtime on the same ONNX files. For `object_detection/2`, the labels are the same as those of `post_process_object_detection` in Python transformers, the scores agree within 2.0e-5, and the boxes agree within 0.001 pixels. For `zero_shot_object_detection/3`, the labels are the same as those of the `zero-shot-object-detection` pipeline, and the boxes agree within 0.011 pixels. The scores of OWL-ViT and OWLv2 agree within 1.1e-4. The reference of Grounding DINO is ONNX Runtime with the post-processing of the pipeline, and its scores agree within 6.0e-4. For `depth_estimation/2`, the depth agrees within 1.0e-4 relative, and a value of the 8-bit depth image can be 1 step away. For `image_segmentation/2`, the labels and the masks are the same as those of the `image-segmentation` pipeline. For `background_removal/2`, a value of the mask can be 1 step away from ONNX Runtime with the post-processing of transformers.js and the bilinear resize of Pillow. For `mask_generation/2`, the masks are those of `SamModel` and `Sam2Model` of Python transformers with `post_process_masks`, except for at most a few pixels at the edges, where the logits are near 0, and the scores agree within 1.0e-4. The PyTorch weights of SAM 3 are gated, so the reference of the SAM 3 tracker is ONNX Runtime on the export, with the processor of Python transformers 5.

### Image preprocessing

`Transmuter.Featurizer` reads `preprocessor_config.json` and prepares the images as the image processors of Python transformers do. It converts the image to RGB, resizes it, crops the center, rescales the values and normalizes them. The result is a tensor of shape `{batch, 3, height, width}` and type `:f32`.

The resize gives the same pixels as Pillow, for all `resample` filters of Pillow. With the hamming filter, a small number of pixels can differ by 1. A fast image processor, whose `image_processor_type` ends in `Fast`, resizes with torchvision in Python, so the featurizer uses the weights of torchvision for it. These weights have fewer bits, and a pixel can be one step away from Pillow when the image gets smaller. The scores of a model change when the pixels change, so this match is important. The serving resizes on the `:backend` of the task, because the resize is slow on `Nx.BinaryBackend`.

The serving compiles the model for one image size, which comes from `size` or `crop_size` in `preprocessor_config.json`. When the size depends on the image, for example with `shortest_edge` and no crop, the serving runs each image alone, and the model compiles each new image size on its first call. Then the batch size must be 1. `image_to_text/2` raises an error for these sizes.

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

### Image to text

```elixir
serving = Transmuter.Vision.image_to_text({:hf, "Xenova/vit-gpt2-image-captioning"})
Nx.Serving.run(serving, image)
#=> %{results: [%{text: "a cat laying on a blanket next to a cat laying on a bed "}]}
```

The serving runs encoder-decoder exports of `VisionEncoderDecoderModel`, such as ViT-GPT2 for captions and TrOCR for handwritten text. The encoder reads the image once, and the decoder generates the text. The settings come from `generation_config.json`, so ViT-GPT2 and TrOCR use greedy selection. When the configuration has no `max_length`, the text has at most 256 new tokens, as in the `image-to-text` pipeline of Python transformers. Use `:max_new_tokens` to change this limit. The serving takes the options of generation, such as `:compile`, `:stream` and `:strategy`.

The serving also runs Florence-2, whose encoder reads the image features and the embeddings of a prompt. Give the task in the input:

```elixir
serving = Transmuter.Vision.image_to_text({:hf, "onnx-community/Florence-2-base-ft"})
Nx.Serving.run(serving, %{image: image, task: "<CAPTION_TO_PHRASE_GROUNDING>", text: "two cats"})
#=> %{results: [%{text: "two cats<loc_11>...", boxes: [%{xmin: 3.68, ymin: 11.16, xmax: 319.52, ymax: 235.32}, ...], labels: ["two cats", ...]}]}
```

A task is a task token of `preprocessor_config.json`, such as `"<CAPTION>"`, `"<OD>"` or `"<OCR_WITH_REGION>"`. The default is `"<CAPTION>"`, and the `:task` option changes it. The result has the keys of `post_process_generation` of Python transformers: `:boxes` (as in `object_detection/2`), `:quad_boxes` or `:polygons`, and `:labels`. `<OPEN_VOCABULARY_DETECTION>` gives `:boxes`, `:box_labels`, `:polygons` and `:polygon_labels`. The coordinates are floats in pixels of the input image. The generation config of Florence-2 does beam search with 3 beams. The vision encoder of `onnx-community/Florence-2-base-ft` gives other features than PyTorch, so the text can differ from the PyTorch model (see the README).

### Object detection

```elixir
serving = Transmuter.Vision.object_detection({:hf, "Xenova/detr-resnet-50"})
Nx.Serving.run(serving, image)
#=> %{detections: [%{label: "remote", score: 0.997, box: %{xmin: 19.9, ymin: 35.1, xmax: 87.3, ymax: 59.2}}, ...]}
```

The serving runs DETR, YOLOS, RT-DETR and RF-DETR exports, such as `Xenova/detr-resnet-50`, `Xenova/yolos-tiny` and `onnx-community/rfdetr_nano-ONNX`. The boxes are floats in pixels of the input image. The pipeline of Python transformers truncates them to integers. The serving keeps the detections with a score higher than `threshold`. The default threshold is 0.5, as in Python transformers.

`model_type` in `config.json` selects the scores. DETR and YOLOS take the softmax over the classes, without the last class, which is "no object". Each query gives one detection, in the order of the queries. RT-DETR and RF-DETR take a sigmoid of each pair of a query and a class, and keep the `:top_k` pairs with the highest scores. The default `:top_k` is the number of queries.

DETR and YOLOS resize the short side to `shortest_edge`, so each image has its own size. The serving runs each image alone, and the model compiles each new image size on its first call. Use `batch_size: 1` with these models. RF-DETR resizes each image to 384x384, so its images go into batches.

### Zero-shot object detection

```elixir
labels = ["cat", "remote control", "couch"]
serving = Transmuter.Vision.zero_shot_object_detection({:hf, "Xenova/owlv2-base-patch16-ensemble"}, labels)
Nx.Serving.run(serving, image)
#=> %{detections: [%{label: "remote control", score: 0.747, box: %{xmin: 167.3, ymin: 38.5, xmax: 185.0, ymax: 93.7}}, ...]}
```

The serving runs OWL-ViT, OWLv2 and Grounding DINO exports, such as `Xenova/owlvit-base-patch32`, `Xenova/owlv2-base-patch16-ensemble` and `onnx-community/grounding-dino-tiny-ONNX`. You give the labels when you build the serving. As in the `zero-shot-object-detection` pipeline of Python transformers, the score of a box for a label is the sigmoid of the highest logit of the box. The serving keeps each box with a score higher than `threshold` for each label, so one box can be a detection of more than one label. The default threshold is 0.1. The boxes are floats in pixels of the input image.

OWL-ViT and OWLv2 score each text query alone, so the serving runs all labels as the text queries of one model run, and the images go into batches. The scores are the same as with one run for each label, as in Python. OWLv2 pads the image to a square with gray before it resizes it. Its resize is the one of scikit-image: a gaussian blur when the image gets smaller, then a linear interpolation.

Grounding DINO reads the text and the image together, so the model runs once for each label and image. The ONNX exports of Grounding DINO compute the phrases of the text for one phrase and a period, so the text of a label is the label in lowercase and a period, as `GroundingDinoProcessor` of Python transformers makes it for a list of labels. The pipeline of Python transformers with the PyTorch model gives the label alone, so its detections are different. A label cannot have `"."` or `"?"`, because these characters end a phrase. The scores of Grounding DINO change by up to 5.0e-4 when the pixel values change by 1.0e-7, also in ONNX Runtime, because the model keeps 900 of 13294 boxes with `TopK`.

### Depth estimation

```elixir
serving = Transmuter.Vision.depth_estimation({:hf, "onnx-community/depth-anything-v2-small"})
Nx.Serving.run(serving, image)
#=> %{predicted_depth: #Nx.Tensor<f32[240][320]>, depth: #Nx.Tensor<u8[240][320]>}
```

The serving runs Depth Anything exports, such as `onnx-community/depth-anything-v2-small` and `Xenova/depth-anything-small-hf`. `predicted_depth` is the output of the model, resized to the size of the image with the bicubic interpolation of PyTorch, as in the `depth-estimation` pipeline of Python transformers. `depth` is the same map as an 8-bit gray image: the nearest pixel is 255 and the farthest pixel is 0.

The DPT image processor keeps the aspect ratio and makes each side a multiple of 14, so a 320x240 image gives pixel values of 518x392. The size depends on the image, so the serving runs each image alone, and the model compiles each new image size on its first call.

The ONNX file of `Xenova/depth-anything-small-hf` gives a depth up to 0.9% away from its PyTorch model, also in ONNX Runtime. Transmuter gives the depth of ONNX Runtime.

### Image segmentation

```elixir
serving = Transmuter.Vision.image_segmentation({:hf, "Xenova/segformer-b0-finetuned-ade-512-512"})
Nx.Serving.run(serving, image)
#=> %{segments: [%{label: "wall", score: nil, mask: #Nx.Tensor<u8[240][320]>}, ...]}
```

The serving runs semantic segmentation models, such as the Segformer exports of Xenova. As in the `image-segmentation` pipeline of Python transformers, the logits are resized to the size of the image with the bilinear interpolation of PyTorch, and each pixel gets the label with the highest logit. Each label that is present gives one segment, in the order of the label ids. A mask has 255 for the pixels of its label and 0 for the other pixels. The score is `nil`, because semantic segmentation gives no scores. Panoptic and instance segmentation models, such as DETR and Mask2Former, are not supported yet.

### Background removal

```elixir
serving = Transmuter.Vision.background_removal({:hf, "onnx-community/ormbg-ONNX"})
%{mask: mask} = Nx.Serving.run(serving, image)
```

Python transformers has no pipeline for this task, so the serving follows the `background-removal` pipeline of transformers.js. The mask is the first output of the model, with a sigmoid when a value is outside 0..1, as 8 bits. The bilinear resize of Pillow gives the mask the size of the image. transformers.js resizes with sharp in Node, which does not filter when the image gets smaller, so its mask can be different at the edges of the foreground. The ONNX file of `onnx-community/ormbg-ONNX` has a fixed batch size of 1. See `Transmuter.Vision.background_removal/2` for an example that makes an image with an alpha channel.

### Mask generation

```elixir
serving = Transmuter.Vision.mask_generation({:hf, "onnx-community/sam2.1-hiera-tiny-ONNX"})

prompts = [
  %{points: [[80, 120]]},
  %{points: [[80, 120], [150, 200]], labels: [1, 0]},
  %{box: [40, 60, 200, 230]}
]

%{masks: masks, iou_scores: scores} = Nx.Serving.run(serving, %{image: image, prompts: prompts})
```

The serving runs the SAM, SAM 2 and SAM 3 tracker exports of transformers.js, such as `Xenova/slimsam-77-uniform`, `onnx-community/sam2.1-hiera-tiny-ONNX` and `onnx-community/sam3-tracker-ONNX`, with prompts of points and boxes. A point with the label 1 is on the object, and a point with the label 0 is not on the object. The default label is 1. A box is `[x1, y1, x2, y2]`. Only an export whose decoder has a box input, such as SAM 2.1 and the SAM 3 tracker, takes boxes. Each prompt gives 3 masks of different sizes and the score that the model predicts for each mask. A mask has 255 for the pixels of the object and 0 for the other pixels.

The vision encoder runs once for each image, and the decoder runs once for each prompt. Python transformers pads the points of the prompts of a batch to one length, and the padding changes the masks, so the serving runs each prompt alone, as one call of Python with one prompt. The decoder compiles once for each number of points, with or without a box. The `mask-generation` pipeline of Python transformers segments the whole image with a grid of points. The serving does not do that yet.

SAM 2.1 and SAM 3 have only a fast image processor, which resizes with torchvision. SAM 3 resizes to 1008x1008. torchvision gives the weights of an 8-bit resize fewer bits than Pillow, so the featurizer does the same for each `image_processor_type` that ends in `Fast`. When the image gets smaller, a pixel of Pillow can be one step away.

### Image to image

```elixir
serving = Transmuter.Vision.image_to_image({:hf, "Xenova/swin2SR-classical-sr-x2-64"})
%{image: image} = Nx.Serving.run(serving, image)
StbImage.write_file!(StbImage.from_nx(image), "cats_x2.png")
```

`Transmuter.Vision.image_to_image/2` runs super-resolution models, such as Swin2SR. The result is a `:u8` image of shape `{height, width, 3}`, `upscale` times larger than the input image. The featurizer pads the image to a multiple of `pad_size` as `Swin2SRImageProcessor` of Python transformers, and the serving rounds the output of the model to 8 bits as the `image-to-image` pipeline. The pipeline of Python transformers also gives the upscaled padding at the bottom and the right of the image. The serving removes it. The output size depends on the image, so each image runs alone, and the model compiles each new image size on its first call.

The tests compare the image with the `image-to-image` pipeline of Python transformers, without its padding, for `Xenova/swin2SR-classical-sr-x2-64` and `Xenova/swin2SR-realworld-sr-x4-64-bsrgan-psnr`. A value can be 1 step away.

### Image-text-to-text

```elixir
serving = Transmuter.Vision.image_text_to_text({:hf, "HuggingFaceTB/SmolVLM-256M-Instruct"})

chat = %{
  messages: [
    %{role: "user", content: [%{type: "image", image: image}, %{type: "text", text: "Can you describe this image?"}]}
  ]
}

Nx.Serving.run(serving, chat)
#=> %{results: [%{text: " In the image we can see a cat sleeping on a pink bed. ...", token_summary: %{input: 876, output: 24, padding: 0}}]}
```

The serving runs decoder-only vision language models whose repository has `vision_encoder`, `embed_tokens` and `decoder_model_merged`, as the `ImageTextToText` exports of transformers.js. It supports Idefics3 models, such as SmolVLM, and Qwen2-VL. The messages use the chat format of Python transformers: the content of a message is a list of parts, `%{type: "image", image: image}` and `%{type: "text", text: text}`. A chat can have more than one image.

For SmolVLM, the image processor is the one of Idefics3. The long side of each image becomes 2048 pixels, each side becomes a multiple of 512, and the image splits into tiles of 512x512 pixels. The image resized to one tile comes after the tiles. Each tile gives 64 image tokens in the prompt, so an image of 4x3 tiles gives 832 image tokens. The vision encoder turns the tiles into features, and the features replace the embeddings of the image tokens. Then the decoder generates as `Transmuter.Text.generation/2` does, with the same options.

The default files are the q4 vision encoder, the q8 embeddings and the q4 decoder: 179 MB for SmolVLM-256M. The fp32 files give the text of ONNX Runtime on the fp32 files too. They do not give the text of the PyTorch model, because the vision encoder of the export gives other features than transformers 4.57.

For Qwen2-VL, the image processor is the one of Qwen2-VL. Each side of the image becomes the nearest multiple of 28 pixels, within `min_pixels` and `max_pixels` of `preprocessor_config.json`, and the image splits into patches of 14x14 pixels. Each group of 2x2 patches gives one image token, so the 320x240 image of the tests gives 99 image tokens. The vision encoder of the export takes one image in each call. Its graph has shapes that depend on the size of the image, so it compiles once for each new image size. On a Mac GPU, this compile takes 20 to 40 seconds, and later calls with that size take less than 1 second. The decoder takes 3-D positions (M-RoPE): the image tokens have the positions of their rows and columns in the image, as `get_rope_index` of Python transformers.

```elixir
serving =
  Transmuter.Vision.image_text_to_text({:hf, "onnx-community/Qwen2-VL-2B-Instruct"},
    variant: %{vision_encoder: :fp32}
  )
```

The configuration of the repository picks the 8-bit vision encoder, the fp16 embeddings and the q4 decoder. The 8-bit vision encoder gives other features in Transmuter than in ONNX Runtime, so the generated text can be different. With the fp32 vision encoder (2.7 GB), the text is the same as the text of ONNX Runtime. The q4 decoder is 1 GB and the fp16 embeddings are 0.5 GB.

With `compile: [batch_size: n]`, chats go into one batch when the number of their tiles or patches rounds up to the same power of two. Rows of zeros fill each chat to that number. The vision encoder of SmolVLM removes the tiles of zeros, and the processor of Qwen2-VL gives the vision encoder only the patches of each image.

## Options of all tasks

- `:compile`: `:batch_size` (the default is 1) and, for text, `:sequence_length`, an integer or a list of lengths (the default is `[32, 128, 512]`, limited to the maximum length of the model). The serving compiles the model for each length and pads each input to the smallest length that fits. Inputs longer than the largest length are truncated. Generation and speech-to-text use only `:batch_size`, and they do not truncate the prompt.
- `:backend` and `:compute_type`: options of `Transmuter.load!/2`.
- `:defn_options`: options of `Transmuter.compile/3`, such as `compiler: EMLX`.
- `:variant`, `:cache_dir`, `:revision` and `:offline`: see above and `Transmuter.Hub`.
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
