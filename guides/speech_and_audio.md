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

## MMS-TTS

[MMS-TTS](https://huggingface.co/Xenova/mms-tts-eng) is a text-to-speech model from Meta. It generates audio from phoneme input. The tested version gives the same audio length and a waveform within 6e-5 of ONNX Runtime (when given the same random seed).

```elixir
model = Transmuter.load!(
  {:hf, "Xenova/mms-tts-eng", "onnx/model.onnx"},
  backend: EMLX.Backend
)

# Inputs: "input_ids" (encoded phonemes, shape {1, n}) and "attention_mask" (shape {1, n}).
inputs = %{
  "input_ids" => input_ids,
  "attention_mask" => attention_mask
}

run = Transmuter.compile(model, inputs, compiler: EMLX)
output = run.(inputs)
waveform = output["waveform"]
```

## Whisper speech recognition

[Whisper](https://huggingface.co/onnx-community/whisper-base) is a speech recognition model. `Transmuter.Audio.speech_to_text/2` runs it from PCM samples to text:

```elixir
# 16 kHz mono f32 samples, for example from ffmpeg -i speech.mp3 -ac 1 -ar 16000 -f f32le speech.f32
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

With `strategy: :greedy`, the text is the same as the `generate` function of transformers for `whisper-tiny.en` and `whisper-base`. The serving does not give timestamps.

The serving takes audio of up to 30 s and raises an error for longer audio. Split long audio into parts of 30 s or less, for example at silences, and run each part.

On an Apple M-series GPU, 5.9 s of speech takes 0.9 s with `whisper-tiny.en` and 1.0 s with `whisper-base`, a real-time factor of 0.15 to 0.17. The featurizer and the encoder take less than 0.12 s, and the decoder gives 25 to 28 tokens per second. The decoder of these exports computes the positions from the length of the cache, so its steps run without compilation (see `Transmuter.Generation`).

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
