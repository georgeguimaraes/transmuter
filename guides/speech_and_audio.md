# Speech and Audio

This guide shows how to run text-to-speech and speech recognition models with Transmuter.

## Kokoro text-to-speech

[Kokoro-82M](https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX) is a text-to-speech model. Its ONNX export runs in Transmuter with no static inputs. The predicted duration of each phoneme decides the length of the audio, so the graph has a break there. Compilation caches each audio length.

Kokoro reads phonemes, not text. The Python and JavaScript versions make the phonemes with misaki or espeak-ng. Elixir has no phonemizer yet, so you must provide phonemes from an external tool or prepare them in advance.

To run Kokoro:

1. Convert text to phonemes with misaki (Python) or espeak-ng.
2. Encode the phonemes with the Kokoro tokenizer (`tokenizer.json` in the repository).
3. Load a voice style vector from the repository (`voices/*.bin`).
4. Pass the encoded phonemes, the style vector, and a speed value to the model.

```elixir
repo = "onnx-community/Kokoro-82M-v1.0-ONNX"
backend = {EMLX.Backend, device: :gpu}

# Set a memory limit so that EMLX does not grow its freed-buffer cache
# unboundedly as new audio lengths arrive.
EMLX.set_cache_limit(1_000_000_000)

model =
  Transmuter.load!({:hf, repo, "onnx/model.onnx"}, backend: backend)
  |> fix_atan2()

# Build the inputs (phoneme ids, style vector, speed).
inputs = %{
  "input_ids" => Nx.tensor([ids], type: :s64, backend: backend),
  "style" => style,
  "speed" => Nx.tensor([1.0], backend: backend)
}

# Each 200 ms bucket of audio length compiles its own decoder.
# Keep 64 compiled lengths in the cache.
speak = Transmuter.compile(model, inputs, compiler: EMLX, cache_size: 64)
waveform = speak.(inputs)["waveform"]
```

The `fix_atan2/1` function changes one node of the export to remove a 4.8 kHz tone from the audio (see [The atan2 fix](#the-atan2-fix) below). The `waveform` output is a float tensor at 24,000 Hz. See `bench/kokoro_tts.exs` for a complete example that writes a WAV file and rounds audio lengths to 200 ms buckets to reduce recompilation.

### The atan2 fix

The Kokoro export computes the STFT phase with the `atan2` of `torch.onnx`. Its quadrant test is `im > 0`. For `atan2(0, x)` when `x` is negative, this gives -π, but `torch.atan2` gives π. An exact STFT has `im = 0` at DC and Nyquist, so these bins get the wrong phase in every frame, and the audio has a tone at 4.8 kHz (the frame rate). ONNX Runtime hides the error, because its STFT has small rounding errors at those bins. The fix changes the test to `im >= 0`, as in `torch.atan2`. With the fix, the spectrum matches PyTorch and there is no tone.

The fix changes the node named `/decoder/decoder/generator/Greater` from `Greater` to `GreaterOrEqual`:

```elixir
defp fix_atan2(model) do
  nodes =
    Enum.map(model.nodes, fn
      %{name: "/decoder/decoder/generator/Greater"} = node ->
        %{node | op_type: "GreaterOrEqual"}
      node ->
        node
    end)

  %{model | nodes: nodes}
end
```

### Audio length buckets

Each new audio length compiles a new graph segment. To limit the number of compilations, round the total duration up to a multiple of a fixed step (for example, 8 frames = 200 ms) by adding the difference to the last phoneme duration. Sentences of similar length then share one compiled segment. The `bench/kokoro_tts.exs` example does this.

### Performance on Apple silicon GPU

| Audio length | Without compilation | First call | Later calls |
|---|---|---|---|
| 7.0 s | 2.3 s | 2.3 s | 0.28 s |
| 20.2 s | 13.9 s | 10.6 s | 0.84 s |
| 26.6 s | 25.8 s | 11.4 s | 1.11 s |

## Text-to-speech with VITS and SpeechT5

`Transmuter.Audio.text_to_speech/2` speaks text with two model types. It gives `%{audio: audio, sampling_rate: rate}`, as the `text-to-speech` pipeline of Python transformers. `audio` is an f32 tensor of PCM samples.

```elixir
serving =
  Transmuter.Audio.text_to_speech({:hf, "Xenova/mms-tts-eng"},
    backend: EMLX.Backend,
    defn_options: [compiler: EMLX]
  )

%{audio: audio, sampling_rate: 16000} = Nx.Serving.run(serving, %{text: "Hello, my dog is cute.", seed: 42})
File.write!("speech.f32", Nx.to_binary(audio))
# ffmpeg -f f32le -ar 16000 -ac 1 -i speech.f32 speech.wav
```

### VITS

[MMS-TTS](https://huggingface.co/Xenova/mms-tts-eng) is a family of VITS models from Meta, one for each language. The model makes the audio in one pass. The tokenizer reads characters and puts the pad token between them, as `VitsTokenizer` of Python transformers. It lowercases the text and removes the characters that the vocabulary does not have.

The export adds random noise to the durations of the characters and to the sounds, with the `noise_scale_duration` (0.8) and `noise_scale` (0.667) of the configuration. The seed of a request decides the noise, so the length of the audio also depends on the seed.

### SpeechT5

[SpeechT5](https://huggingface.co/Xenova/speecht5_tts) needs a speaker embedding: an x-vector of 512 values, such as the x-vectors of the CMU ARCTIC speakers that the examples of transformers use. Give it in the `:speaker_embedding` option or in the input:

```elixir
speaker = Nx.from_binary(File.read!("speaker.f32"), :f32)

serving =
  Transmuter.Audio.text_to_speech({:hf, "Xenova/speecht5_tts"},
    speaker_embedding: speaker,
    backend: EMLX.Backend,
    defn_options: [compiler: EMLX]
  )

Nx.Serving.run(serving, %{text: "Hello, my dog is cute.", seed: 42})
#=> %{audio: #Nx.Tensor<f32[25088]>, sampling_rate: 16000}
```

The serving does the steps of `generate_speech` in Python transformers:

1. The encoder runs once on the characters of the text.
2. The decoder makes the spectrogram in steps. Each step gives 2 frames of 80 mel values, and the next step takes the frames so far and the KV cache.
3. The loop stops when the sum of the two stop probabilities of a step is 0.5 or more, or after 10 steps for each token of the text (`maxlenratio` 20 divided by the `reduction_factor` 2).
4. The postnet and the HiFi-GAN vocoder (`decoder_postnet_and_vocoder.onnx`) turn the frames into audio. This file is the same as `onnx/model.onnx` of `Xenova/speecht5_hifigan`.

The prenet of the decoder applies dropout with a ratio of 0.5 in each step, also in eval mode in Python. So the audio of SpeechT5 also depends on the seed. Each step draws from its own key, which comes from the seed and the number of the step.

### Noise and references

With `noise: false`, the random operators give fixed values (see the `:noise` option of `Transmuter.load!/2`). VITS then gives the audio of Python transformers with `noise_scale` and `noise_scale_duration` at 0. In SpeechT5, the dropout keeps all values. The tests compare this audio with PyTorch, and `test/support/pipeline_references.py` makes the references:

| Model | Text | Length | Largest difference |
|---|---|---|---|
| `Xenova/mms-tts-eng` | "Hello, my dog is cute." | 33,024 samples, the same as PyTorch | 8.5e-5 |
| `Xenova/speecht5_tts` | "Hi." | 9,728 samples (19 steps), the same as PyTorch | 8.4e-5 |

ONNX Runtime on the fp32 SpeechT5 export, with the same dropout, gives 19 steps and differs from PyTorch by 1.1e-4. The quantized SpeechT5 files do not give the audio of ONNX Runtime on the same files. With the 8-bit files, the decoder stops after 19 steps in Transmuter and after 21 steps in ONNX Runtime. With the 4-bit files, the samples differ by up to 5e-3. Use the fp32 files, the default.

### Speed

On an Apple M-series GPU, `Xenova/mms-tts-eng` makes 2 s of speech in 0.23 s. `Xenova/speecht5_tts` makes 1.6 s of speech in 3.3 s: each of its 49 decoder steps takes about 68 ms. The decoder computes its positions from the length of its inputs, so each step has a new shape. Each shape compiles on its first run, and the compiled decoder keeps 1024 shapes.

The serving runs each text alone, because the length of the audio depends on the text. A list of texts runs one text after the other.

### Kokoro

`text_to_speech/2` does not run Kokoro, because Kokoro reads phonemes. See [Kokoro text-to-speech](#kokoro-text-to-speech) for Kokoro with tensors.

### MMS-TTS with tensors

The model takes `input_ids` and `attention_mask`. The tokenizer of the repository gives them. Give an `Nx.Random` key in the `Transmuter.random_input/0` input to get the same audio again.

```elixir
model = Transmuter.load!(
  {:hf, "Xenova/mms-tts-eng", "onnx/model.onnx"},
  backend: EMLX.Backend
)

tokenizer = Transmuter.Tokenizer.load!({:hf, "Xenova/mms-tts-eng"})
{tokens, _encodings} = Transmuter.Tokenizer.encode(tokenizer, ["Hello, my dog is cute."])

inputs = %{
  "input_ids" => tokens["input_ids"],
  "attention_mask" => tokens["attention_mask"],
  Transmuter.random_input() => Nx.Random.key(42)
}

run = Transmuter.compile(model, inputs, compiler: EMLX)
waveform = run.(inputs)["waveform"]
```

With the random values of ONNX Runtime, the model gives the same audio length and a waveform within 6e-5 of ONNX Runtime.

## Whisper speech recognition

[Whisper](https://huggingface.co/onnx-community/whisper-base) is a speech recognition model. `Transmuter.Audio.speech_to_text/2` runs it from PCM samples to text:

```elixir
# 16 kHz mono f32 samples, or {:file, "speech.mp3"}, which ffmpeg decodes to 16 kHz.
audio = Nx.from_binary(File.read!("speech.f32"), :f32)

serving =
  Transmuter.Audio.speech_to_text({:hf, "Xenova/whisper-tiny.en"},
    backend: EMLX.Backend,
    defn_options: [compiler: EMLX]
  )

Nx.Serving.run(serving, audio)
#=> %{chunks: [%{text: " Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel.",
#=>              start_timestamp_seconds: nil, end_timestamp_seconds: nil}]}
```

The serving does the steps of Python transformers:

1. The featurizer makes a log-mel spectrogram of 30 s, as `WhisperFeatureExtractor`. It pads the audio with zeros to 30 s, takes an STFT with frames of 400 samples every 160 samples and a Hann window, and applies 80 mel filters (128 for large-v3). The features are within 2e-5 of transformers.
2. The encoder runs once on the spectrogram.
3. The decoder starts from the prompt `<|startoftranscript|>`, the language token, the task token and `<|notimestamps|>`. An English-only model, such as `whisper-tiny.en`, has no language and task tokens. The `:language` option (the default is `"en"`) and the `:task` option (`:transcribe` or `:translate`) select the tokens. The serving does not detect the language.
4. The selection of each token suppresses `suppress_tokens`, and `begin_suppress_tokens` for the first token, from `generation_config.json`.

With `strategy: :greedy`, the text is the same as the `generate` function of transformers for `whisper-tiny.en` and `whisper-base`.

### Long audio

The audio can have any length. The serving does the steps of the `automatic-speech-recognition` pipeline of transformers with `chunk_length_s` and `stride_length_s`:

1. The client splits the audio into windows of `:chunk_num_seconds` (the default is 30 s). Two windows that follow each other overlap by two contexts of `:context_num_seconds` (the default is `:chunk_num_seconds / 6`, 5 s). The first window has no left context, and the last window has no right context. With the defaults, the windows start every 20 s.
2. The serving transcribes each window as one row of a batch. With `compile: [batch_size: n]`, the decoder transcribes up to `n` windows together.
3. The client merges the tokens of the windows. It finds the alignment of two token sequences with the most equal tokens, and cuts the overlap at its middle. Audio of 30 s or less is one window with no context, so the merge does not change its text.

The options map to the pipeline of transformers as follows:

| Transmuter | transformers | Default |
|---|---|---|
| `:chunk_num_seconds` | `chunk_length_s` | 30 s, the length of the features |
| `:context_num_seconds` | `stride_length_s` (the same on the left and the right) | `:chunk_num_seconds / 6` |

`:chunk_num_seconds` must be 30 s or less and more than two times `:context_num_seconds`.

On 62.5 s of speech, with greedy selection, the text is the same as the pipeline of transformers with `chunk_length_s=30` for `whisper-tiny.en` and `whisper-base`.

### Streams of audio

The input can be a stream of tensors of PCM samples, such as the parts of a long recording. The client cuts the stream into the windows of `chunk_iter` when the tensors come. A full window waits for the next sample, which tells that the window is not the last. The windows are the same for each size of the tensors, so the result of a stream is the result of its whole audio:

```elixir
stream =
  File.stream!("speech.f32", 16_000 * 4)
  |> Stream.map(&Nx.from_binary(&1, :f32))

Nx.Serving.run(serving, stream)
```

The serving transcribes the windows before the stream ends, and merges them when it ends. The windows of a stream go into batches of `compile: [batch_size: n]`, so a batch waits until it has `n` windows or the stream ends. Nx.Serving reads the stream in a process of its own.

### Timestamps

With `timestamps: :segments`, the decoder predicts timestamp tokens, and the result has one chunk for each segment of speech:

```elixir
serving =
  Transmuter.Audio.speech_to_text({:hf, "Xenova/whisper-tiny.en"},
    timestamps: :segments,
    backend: EMLX.Backend,
    defn_options: [compiler: EMLX]
  )

Nx.Serving.run(serving, audio)
#=> %{chunks: [%{text: " Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel.",
#=>              start_timestamp_seconds: 0.0, end_timestamp_seconds: 5.44}]}
```

The serving does the steps of `return_timestamps=True` in transformers:

1. The prompt has no `<|notimestamps|>` token.
2. The token selection applies the rules of `WhisperTimeStampLogitsProcessor`. The first token is a timestamp. Timestamps come in pairs, except before the end of the text. Timestamps do not decrease. When all the timestamps together are more probable than each text token, the decoder selects a timestamp.
3. When the tokens of a window end with two timestamps, the last segment is not complete. The serving removes the tokens after these timestamps. It then transcribes the window again from the last timestamp, as the `generate` function of transformers does.
4. The merge ignores the timestamps in the context of each window. A timestamp ends a chunk only outside the contexts.

The timestamps have steps of 0.02 s. On 62.5 s of speech, the chunks and their timestamps are the same as the pipeline of transformers for `whisper-tiny.en` and `whisper-base`. The end of the last chunk can be after the end of the audio, as in transformers.

### Word timestamps

With `timestamps: :words`, the result has one chunk for each word, as `return_timestamps="word"` in transformers:

```elixir
serving =
  Transmuter.Audio.speech_to_text({:hf, "Xenova/whisper-tiny.en"},
    timestamps: :words,
    backend: EMLX.Backend,
    defn_options: [compiler: EMLX]
  )

Nx.Serving.run(serving, audio)
#=> %{chunks: [%{text: " Mr.", start_timestamp_seconds: 0.0, end_timestamp_seconds: 1.02},
#=>            %{text: " Quilter", start_timestamp_seconds: 1.02, end_timestamp_seconds: 1.24}, ...]}
```

The decoder predicts timestamp tokens as with `:segments`. The time of each token then comes from the cross attentions of the decoder, as `_extract_token_timestamps` of transformers computes it:

1. After each pass over a window, the decoder runs again over the prompt and the new tokens, without the cache. Only the heads of `alignment_heads` in `generation_config.json` stay, so the serving keeps a matrix of tokens by encoder positions (0.02 s each) for each head.
2. Each head is normalized over the tokens, and a median filter of 7 values smooths it over the positions. The matrix is the mean of the heads.
3. Dynamic time warping finds the path of the least cost through the matrix. A token ends at the position where the path enters its row.
4. A token starts at the end of the token before it. The tokens join into words at spaces, and punctuation joins the word next to it, as `_combine_tokens_into_words` of the Whisper tokenizer. A word starts at the start of its first token and ends at the end of its last token.

The decoder must give the outputs `cross_attentions.N`. `Xenova/whisper-tiny.en` and the `_timestamped` exports of onnx-community, such as `onnx-community/whisper-tiny.en_timestamped`, give them. `onnx-community/whisper-base` does not, and the serving then raises an error.

In Chinese, Japanese, Thai, Lao, Burmese and Cantonese, each piece of complete characters is a word, as OpenAI whisper and transformers.js do. The `:language` option gives the language. With `task: :translate`, the text is English, so the words split at spaces. The pipeline of Python transformers does not know the language there, so it splits all languages at spaces.

With `whisper-tiny.en` and greedy selection, the words and their times are the same as the pipeline of transformers with `chunk_length_s=30`: 17 words on 5.9 s of speech, and 150 words on 62.5 s of speech. On the CPU with EXLA, the extra run of the decoder and the time warping add 2% to 4% to the time of `:segments`.

### Speed

On an Apple M-series GPU, 5.9 s of speech takes 0.9 s with `whisper-tiny.en` and 1.0 s with `whisper-base`, a real-time factor of 0.15 to 0.17. The featurizer and the encoder take less than 0.12 s, and the decoder gives 25 to 28 tokens per second. The decoder of these exports computes the positions from the length of the cache, so each cache length compiles on its first step and stays cached (see `Transmuter.Generation`).

On 62.5 s of speech (3 windows), the real-time factor is 0.12 with `whisper-tiny.en` and 0.17 with `whisper-base`. With timestamps, it is 0.15 and 0.16. The overlaps of the windows and the passes again over windows add work.

### Whisper with tensors

The ONNX export has four files: an encoder and three decoder variants. The merged decoder (`decoder_model_merged.onnx`) uses `use_cache_branch` to select the prefill or decode path.

```elixir
encoder = Transmuter.load!(
  {:hf, "onnx-community/whisper-base", "onnx/encoder_model.onnx"},
  backend: EMLX.Backend
)

decoder = Transmuter.load!(
  {:hf, "onnx-community/whisper-base", "onnx/decoder_model_merged.onnx"},
  backend: EMLX.Backend
)
```

Pass `use_cache_branch` as a static input to compile separate functions for prefill and decode:

```elixir
prefill_flag = Nx.tensor([0], type: :u8, backend: Nx.BinaryBackend)
decode_flag = Nx.tensor([1], type: :u8, backend: Nx.BinaryBackend)

prefill_run = Transmuter.compile(
  decoder,
  prefill_inputs,
  static_inputs: %{"use_cache_branch" => prefill_flag},
  compiler: EMLX
)

decode_run = Transmuter.compile(
  decoder,
  decode_inputs,
  static_inputs: %{"use_cache_branch" => decode_flag},
  compiler: EMLX
)
```

## Moonshine speech recognition

[Moonshine](https://huggingface.co/onnx-community/moonshine-base-ONNX) is a fast speech recognition model. The tested version gives the same tokens as ONNX Runtime for 2 s, 6 s, and 15 s of speech.

```elixir
encoder = Transmuter.load!(
  {:hf, "onnx-community/moonshine-base-ONNX", "onnx/encoder_model.onnx"},
  backend: EMLX.Backend
)

decoder = Transmuter.load!(
  {:hf, "onnx-community/moonshine-base-ONNX", "onnx/decoder_model_merged.onnx"},
  backend: EMLX.Backend
)
```

The encoder output shape changes with the length of the audio. The decoder accepts variable-length encoder outputs.
