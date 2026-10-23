defmodule Transmuter.Audio do
  @moduledoc """
  Servings for audio tasks. Each function loads a model repository (the ONNX files, the tokenizer or `preprocessor_config.json`, and the configuration) and returns an `Nx.Serving`. `speech_to_text/2` takes audio and gives text in the shape of Bumblebee. `audio_classification/2` takes audio and gives labels in the shape of Bumblebee. `zero_shot_audio_classification/3` takes audio and gives scores for the labels that you give. `text_to_speech/2` takes text and gives audio in the shape of Python transformers.

      serving = Transmuter.Audio.speech_to_text({:hf, "Xenova/whisper-tiny.en"})
      Nx.Serving.run(serving, audio)
      #=> %{chunks: [%{text: " Mr. Quilter is the apostle of the middle classes, ...", ...}]}

      serving = Transmuter.Audio.text_to_speech({:hf, "Xenova/mms-tts-eng"})
      Nx.Serving.run(serving, %{text: "Hello, my dog is cute.", seed: 42})
      #=> %{audio: #Nx.Tensor<f32[31744]>, sampling_rate: 16000}

  The tasks with text need the optional `tokenizers` dependency: add `{:tokenizers, "~> 0.5"}` to your deps.

  The options `:defn_options`, `:backend`, `:compute_type`, `:variant`, `:cache_dir`, `:revision`, `:offline` and `:tokenizer` are the same as in `Transmuter.Text`. Use `backend: EMLX.Backend, defn_options: [compiler: EMLX]` on Apple silicon, because `Nx.BinaryBackend` is very slow.

  ## Audio input

  `speech_to_text/2`, `audio_classification/2` and `zero_shot_audio_classification/3` take one audio or a list of them. An audio is one of these:

    * A tensor of PCM samples `{samples}` at the sampling rate of the model, of any float type.

    * `{:file, path}` - an audio or video file. As `ffmpeg_read` of Python transformers, ffmpeg decodes the file, mixes its channels into one and changes its sampling rate to the sampling rate of the model. The `ffmpeg` executable must be in the `PATH`. The whole file decodes into memory before the serving runs: about 230 MB for one hour at 16 kHz. Use a stream for longer audio.

    * A stream of tensors `{samples}`: a `Stream`, a `File.Stream`, an `IO.Stream` or a stream function. Each tensor is the next part of the audio, at the sampling rate of the model. A list is not a stream, because a list is a batch of inputs.

  An input error raises an `ArgumentError` in the caller, also for one input of a list: a tensor that is not `{samples}`, a path that does not exist or is a directory, a file that ffmpeg cannot decode, audio without samples, and a stream that is empty or whose first chunk is not a tensor `{samples}`. The serving reads the first chunk of a stream to check it.

  ```elixir
  serving = Transmuter.Audio.speech_to_text({:hf, "Xenova/whisper-tiny.en"})
  Nx.Serving.run(serving, {:file, "speech.mp3"})
  ```

  `speech_to_text/2` cuts a stream into windows when the tensors come, and transcribes each window before the stream ends. The windows are the same for each size of the tensors, so the result of a stream is the result of its audio in one tensor. The result comes when the stream ends. Nx.Serving reads the stream after its first chunk in a process of its own, so the stream must not depend on the process that calls the serving, for example to receive messages. A later chunk that is not a tensor `{samples}` ends the call with an exit, which the caller cannot rescue as an exception.

  The classification tasks take the whole audio, so they join the tensors of a stream before they run. A stream that does not end gives no result.

  `Transmuter.Audio.Featurizer.audio/2` reads these inputs. The samples become f32, as in Python transformers.
  """

  @doc """
  Builds a serving that turns speech into text with Whisper.

  The input is an audio at 16 kHz, the sampling rate of Whisper, or a list of them. An audio is a tensor of PCM samples, `{:file, path}` or a stream of tensors (see the module documentation). The result is `%{chunks: [%{text: text, start_timestamp_seconds: start, end_timestamp_seconds: end}]}`, as in Bumblebee. The text starts with a space, as in Python transformers.

  Without timestamps, the result has one chunk with all the text, and its timestamps are `nil`. With `timestamps: :segments`, the result has one chunk for each segment of speech, with its start and end in seconds. The end of the last chunk is `nil` when the model does not predict it. With `timestamps: :words`, the result has one chunk for each word, with its start and end in seconds. The text of a word starts with its space, and has the punctuation next to it.

  The audio can have any length. The serving does the steps of the `automatic-speech-recognition` pipeline of transformers with `chunk_length_s` and `stride_length_s`. It splits the audio into windows that overlap, transcribes each window, and merges the tokens of the windows at the overlaps (`_decode_asr` of the Whisper tokenizer). Audio of 30 s or less is one window.

  The repository must have `onnx/encoder_model.onnx` and `onnx/decoder_model_merged.onnx`, as the Whisper exports of Xenova and onnx-community. The default variant is fp32. The prompt of the decoder is the one of Python transformers: the start token, the language and the task for a multilingual model, and `<|notimestamps|>` without timestamps. The settings of generation come from `generation_config.json`, such as `suppress_tokens` and `begin_suppress_tokens`.

      serving = Transmuter.Audio.speech_to_text({:hf, "onnx-community/whisper-base"}, language: "fr")

  ## Options

    * `:language` - the language code of the speech for a multilingual model, such as `"fr"`. The default is `"en"`. The model does not detect the language.

    * `:task` - `:transcribe` (the default) gives the text in the language of the speech. `:translate` gives English text.

    * `:chunk_num_seconds` - the length of a window in seconds, `chunk_length_s` in transformers. The default is 30, the length of the features. The value must be 30 or less.

    * `:context_num_seconds` - the length of the context at each side of a window in seconds, `stride_length_s` in transformers. Two windows that follow each other overlap by two contexts. The default is `:chunk_num_seconds / 6`. The value must be less than half of `:chunk_num_seconds`.

    * `:timestamps` - `:segments` gives a chunk for each segment of speech, with timestamps, as `return_timestamps=True` in transformers. The decoder then predicts timestamp tokens with the rules of `WhisperTimeStampLogitsProcessor`. `:words` gives a chunk for each word, as `return_timestamps="word"` in transformers. The time of each token comes from the cross attentions of the `alignment_heads` of `generation_config.json`, with dynamic time warping (see `Transmuter.Audio.WordTimestamps`). The decoder must give the outputs `cross_attentions.N`, as `Xenova/whisper-tiny.en` and the `_timestamped` exports of onnx-community do, such as `onnx-community/whisper-tiny.en_timestamped`. `onnx-community/whisper-base` does not give them. In Chinese, Japanese, Thai, Lao, Burmese and Cantonese, a word is each piece of complete characters, as in OpenAI whisper and transformers.js. Other languages and translations split at spaces. The pipeline of transformers splits all languages at spaces. The default is `nil`, with no timestamps.

    * `:token_logprobs` - `true` adds `:token_logprobs` to each chunk: the log-probability of each text token of the chunk, as in `Transmuter.Text.generation/2`. The timestamp tokens, the special tokens and the eos token are not in the text, so they have no value. In the overlap of two windows, a token has the value of the window that the merge keeps it from. With `timestamps: :words`, the values are those of the tokens of the word. The default is `false`.

    * `:compile` - `:batch_size`, the number of windows that the decoder transcribes together. The windows come from one long audio, from a list of audio, or from concurrent calls to `Nx.Serving.batched_run/2`. The windows of a stream wait until they fill a batch or the stream ends. The default is `1`.

    * `:max_new_tokens`, `:strategy`, `:num_beams`, `:length_penalty`, `:early_stopping`, `:temperature`, `:top_k`, `:top_p`, `:repetition_penalty`, `:no_repeat_ngram_size`, `:min_length`, `:min_new_tokens` and `:seed` - as in `Transmuter.Text.generation/2`. The prompt of the decoder counts in `:min_length`. Python transformers makes the prompt of Whisper and its first suppressed tokens itself, so the serving does not take `:forced_bos_token_id` and `:forced_eos_token_id`. `:max_new_tokens` applies to each pass of the decoder over a window. Its default is `max_length` of `generation_config.json` (448 for Whisper) minus the length of the prompt, as in the pipeline of transformers.

  An English-only model, such as `whisper-tiny.en`, raises an error for a `:language` or `:task` other than English transcription.

  The serving does not stream the text, because it merges the windows of long audio only at the end.
  """
  defdelegate speech_to_text(source, opts \\ []), to: Transmuter.Audio.SpeechToText

  @doc """
  Builds a serving that classifies audio, such as AST and wav2vec2 models.

  The input is an audio at the sampling rate of the model (16 kHz for AST and wav2vec2), or a list of them. An audio is a tensor of PCM samples, `{:file, path}` or a stream of tensors (see the module documentation). The result is `%{predictions: [%{label: label, score: score}]}`, from the highest score.

      serving = Transmuter.Audio.audio_classification({:hf, "Xenova/ast-finetuned-audioset-10-10-0.4593"})
      Nx.Serving.run(serving, audio)
      #=> %{predictions: [%{label: "Speech", score: 0.84}, ...]}

  The labels come from `id2label` in `config.json`. As in the `audio-classification` pipeline of Python transformers, the scores are a softmax of the logits for all models, also for models with many labels per input, such as AST on AudioSet.

  `Transmuter.Audio.Featurizer` prepares the audio as the feature extractor of Python transformers, which `feature_extractor_type` of `preprocessor_config.json` selects:

    * AST (`ASTFeatureExtractor`) takes 1024 frames of log-mel features, about 10 s of audio. The serving cuts longer audio and pads shorter audio, so all audio has one shape, and the audio of a list or of concurrent calls goes into batches.

    * wav2vec2 (`Wav2Vec2FeatureExtractor`) takes the samples. The model takes the mean over all frames, so padding would change the result. The serving runs each audio alone, and the model compiles each new audio length on its first call. It keeps 64 lengths.

  The default variant is fp32.

  ## Options

    * `:top_k` - the number of predictions, or `nil` for all. The default is `5`.

    * `:scores_function` - `:softmax`, `:sigmoid` or `:none` for the logits. The default is `:softmax`.

    * `:compile` - `:batch_size`. The default is `1`. With wav2vec2, the batch size must be `1`.

    * `:featurizer` - a featurizer from `Transmuter.Audio.Featurizer.load!/2`, to use in place of the `preprocessor_config.json` of the repository.

    * `:model` - a model from `Transmuter.load!/2`, to use in place of the ONNX file of the repository.

  The serving computes the features on the `:backend`. See the module documentation for the other options.
  """
  defdelegate audio_classification(source, opts \\ []), to: Transmuter.Audio.AudioClassification

  @doc """
  Builds a serving that classifies audio with labels that you give, such as CLAP.

  `labels` is a list of strings. The input is an audio at the sampling rate of the model (48 kHz for CLAP), or a list of them. An audio is a tensor of PCM samples, `{:file, path}` or a stream of tensors (see the module documentation). The result is `%{predictions: [%{label: label, score: score}]}`, from the highest score.

      serving = Transmuter.Audio.zero_shot_audio_classification({:hf, "Xenova/clap-htsat-unfused"}, ["a man speaking", "music", "a dog barking"])
      Nx.Serving.run(serving, audio)
      #=> %{predictions: [%{label: "a man speaking", score: 0.99}, ...]}

  The serving uses `model.onnx`, which has the text tower, the audio tower and the logit scale of the model. Each label goes into the hypothesis template. The text of all labels is padded to the longest text, as in Python transformers, and goes into each batch. As in the `zero-shot-audio-classification` pipeline of Python transformers, the scores are a softmax of `logits_per_audio` over the labels.

  `Transmuter.Audio.Featurizer` prepares the audio as `ClapFeatureExtractor` of Python transformers, see its documentation. CLAP takes 10 s of audio. Shorter audio repeats and gets zeros at the end, so all audio has one shape, and the audio of a list or of concurrent calls goes into batches. For longer audio, Python transformers takes 10 s at a random position. The serving takes the first 10 s.

  The fused CLAP models, such as `laion/clap-htsat-fused`, take four spectrograms of each audio. The featurizer does not support them, and raises an `ArgumentError`.

  The default variant is fp32.

  ## Options

    * `:hypothesis_template` - the text for each label, with `{}` in the place of the label. The default is `"This is a sound of {}."`, as in Python transformers.

    * `:top_k` - the number of predictions, or `nil` for all. The default is `5`.

    * `:compile` - `:batch_size`. The default is `1`.

    * `:featurizer` - a featurizer from `Transmuter.Audio.Featurizer.load!/2`, to use in place of the `preprocessor_config.json` of the repository.

    * `:model` - a model from `Transmuter.load!/2`, to use in place of the ONNX file of the repository.

  The serving computes the features on the `:backend`. See the module documentation for the other options.
  """
  defdelegate zero_shot_audio_classification(source, labels, opts \\ []),
    to: Transmuter.Audio.ZeroShotAudioClassification

  @doc """
  Builds a serving that turns text into speech with VITS (MMS-TTS) or SpeechT5.

  The input is a string, or a map with the text in `:text`, and optionally `:seed` and `:speaker_embedding`. The input can also be a list of them. The result is `%{audio: audio, sampling_rate: rate}`, as the `text-to-speech` pipeline of Python transformers. `audio` is an f32 tensor of PCM samples `{samples}`, and `rate` is the sampling rate in Hz.

  The model type in `config.json` selects the steps:

    * VITS, such as `Xenova/mms-tts-eng`, makes the audio in one pass of `onnx/model.onnx`. The tokenizer reads characters, and puts the pad token between them, as `VitsTokenizer` of transformers.

    * SpeechT5, such as `Xenova/speecht5_tts`, needs a speaker embedding of 512 values, such as an x-vector of the CMU ARCTIC speakers. The encoder (`encoder_model`) runs once. The decoder (`decoder_model_merged`) then makes the spectrogram in steps of two frames. It stops when the sum of the two stop probabilities of a step is 0.5 or more, or after 10 steps for each token, as `generate_speech` of transformers with its defaults. The postnet and the vocoder (`decoder_postnet_and_vocoder`) turn the spectrogram into audio.

  Kokoro reads phonemes, not text, so `text_to_speech/2` does not run it. See the [Speech and audio guide](guides/speech_and_audio.md) for Kokoro with tensors.

  Both models are random. VITS adds noise to the durations and to the sounds, and the decoder of SpeechT5 applies dropout in each step, also in Python. The seed of an input decides this noise, so a seed gives the same audio each time. An input without a seed gets a random seed.

  The default variant is fp32. The serving runs each text alone, because the length of the audio depends on the text. The compiled models keep the shapes of 64 text lengths and audio lengths. The decoder of SpeechT5 compiles a shape for each step on its first run, and keeps 1024 steps.

      serving =
        Transmuter.Audio.text_to_speech({:hf, "Xenova/speecht5_tts"},
          speaker_embedding: Nx.from_binary(File.read!("speaker.f32"), :f32)
        )

      Nx.Serving.run(serving, %{text: "Hello, my dog is cute.", seed: 42})
      #=> %{audio: #Nx.Tensor<f32[25088]>, sampling_rate: 16000}

  ## Options

    * `:seed` - the seed of the inputs without a seed. The default is a random seed for each input.

    * `:speaker_embedding` - the speaker embedding of the inputs without one, a tensor `{512}` for SpeechT5. VITS takes no speaker embedding.

    * `:noise` - `false` removes the noise, so the audio does not depend on the seed. The random operators of the model then give fixed values (see the `:noise` option of `Transmuter.load!/2`). VITS then gives the audio of transformers with `noise_scale` and `noise_scale_duration` at 0. In SpeechT5, the dropout keeps all values. The audio without noise is less natural, and SpeechT5 can then continue to its maximum length. Use it to compare with a reference. The default is `true`.
  """
  defdelegate text_to_speech(source, opts \\ []), to: Transmuter.Audio.TextToSpeech
end
