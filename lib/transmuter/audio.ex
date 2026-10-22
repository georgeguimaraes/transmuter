defmodule Transmuter.Audio do
  @moduledoc """
  Servings for audio tasks. Each function loads a model repository (the ONNX files, the tokenizer or `preprocessor_config.json`, and the configuration) and returns an `Nx.Serving`. `speech_to_text/2` takes audio and gives text in the shape of Bumblebee. `audio_classification/2` takes audio and gives labels in the shape of Bumblebee. `text_to_speech/2` takes text and gives audio in the shape of Python transformers.

      serving = Transmuter.Audio.speech_to_text({:hf, "Xenova/whisper-tiny.en"})
      Nx.Serving.run(serving, audio)
      #=> %{chunks: [%{text: " Mr. Quilter is the apostle of the middle classes, ...", ...}]}

      serving = Transmuter.Audio.text_to_speech({:hf, "Xenova/mms-tts-eng"})
      Nx.Serving.run(serving, %{text: "Hello, my dog is cute.", seed: 42})
      #=> %{audio: #Nx.Tensor<f32[31744]>, sampling_rate: 16000}

  The tasks with text need the optional `tokenizers` dependency: add `{:tokenizers, "~> 0.5"}` to your deps.

  The options `:defn_options`, `:backend`, `:compute_type`, `:variant`, `:cache_dir` and `:tokenizer` are the same as in `Transmuter.Text`. Use `backend: EMLX.Backend, defn_options: [compiler: EMLX]` on Apple silicon, because `Nx.BinaryBackend` is very slow.
  """

  @doc """
  Builds a serving that turns speech into text with Whisper.

  The input is a tensor of PCM samples at the sampling rate of the model (16 kHz for Whisper), of any float type, or a list of them. The result is `%{chunks: [%{text: text, start_timestamp_seconds: start, end_timestamp_seconds: end}]}`, as in Bumblebee. The text starts with a space, as in Python transformers.

  Without timestamps, the result has one chunk with all the text, and its timestamps are `nil`. With `timestamps: :segments`, the result has one chunk for each segment of speech, with its start and end in seconds. The end of the last chunk is `nil` when the model does not predict it.

  The audio can have any length. The serving does the steps of the `automatic-speech-recognition` pipeline of transformers with `chunk_length_s` and `stride_length_s`. It splits the audio into windows that overlap, transcribes each window, and merges the tokens of the windows at the overlaps (`_decode_asr` of the Whisper tokenizer). Audio of 30 s or less is one window.

  The repository must have `onnx/encoder_model.onnx` and `onnx/decoder_model_merged.onnx`, as the Whisper exports of Xenova and onnx-community. The default variant is fp32. The prompt of the decoder is the one of Python transformers: the start token, the language and the task for a multilingual model, and `<|notimestamps|>` without timestamps. The settings of generation come from `generation_config.json`, such as `suppress_tokens` and `begin_suppress_tokens`.

      serving = Transmuter.Audio.speech_to_text({:hf, "onnx-community/whisper-base"}, language: "fr")

  ## Options

    * `:language` - the language code of the speech for a multilingual model, such as `"fr"`. The default is `"en"`. The model does not detect the language.

    * `:task` - `:transcribe` (the default) gives the text in the language of the speech. `:translate` gives English text.

    * `:chunk_num_seconds` - the length of a window in seconds, `chunk_length_s` in transformers. The default is 30, the length of the features. The value must be 30 or less.

    * `:context_num_seconds` - the length of the context at each side of a window in seconds, `stride_length_s` in transformers. Two windows that follow each other overlap by two contexts. The default is `:chunk_num_seconds / 6`. The value must be less than half of `:chunk_num_seconds`.

    * `:timestamps` - `:segments` gives a chunk for each segment of speech, with timestamps, as `return_timestamps=True` in transformers. The decoder then predicts timestamp tokens with the rules of `WhisperTimeStampLogitsProcessor`. The default is `nil`, with no timestamps.

    * `:compile` - `:batch_size`, the number of windows that the decoder transcribes together. The windows come from one long audio, from a list of audio, or from concurrent calls to `Nx.Serving.batched_run/2`. The default is `1`.

    * `:max_new_tokens`, `:strategy`, `:temperature`, `:top_k`, `:top_p`, `:repetition_penalty` and `:seed` - as in `Transmuter.Text.generation/2`. `:max_new_tokens` applies to each pass of the decoder over a window. Its default is `max_length` of `generation_config.json` (448 for Whisper) minus the length of the prompt, as in the pipeline of transformers.

  An English-only model, such as `whisper-tiny.en`, raises an error for a `:language` or `:task` other than English transcription.

  The serving does not stream the text, because it merges the windows of long audio only at the end.
  """
  defdelegate speech_to_text(source, opts \\ []), to: Transmuter.Audio.SpeechToText

  @doc """
  Builds a serving that classifies audio, such as AST and wav2vec2 models.

  The input is a tensor of PCM samples at the sampling rate of the model (16 kHz for AST and wav2vec2), of any float type, or a list of them. The result is `%{predictions: [%{label: label, score: score}]}`, from the highest score.

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
