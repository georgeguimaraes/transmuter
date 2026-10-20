defmodule Transmuter.Audio do
  @moduledoc """
  Servings for audio tasks. Each function loads a model repository (the ONNX files, the tokenizer and the configuration) and returns an `Nx.Serving` that takes audio and gives results in the shapes of Bumblebee.

      serving = Transmuter.Audio.speech_to_text({:hf, "Xenova/whisper-tiny.en"})
      Nx.Serving.run(serving, audio)
      #=> %{chunks: [%{text: " Mr. Quilter is the apostle of the middle classes, ...", ...}]}

  The tasks need the optional `tokenizers` dependency: add `{:tokenizers, "~> 0.5"}` to your deps.

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

    * `:max_new_tokens`, `:strategy`, `:temperature`, `:top_k`, `:top_p`, `:repetition_penalty` and `:seed` - as in `Transmuter.Text.generation/2`. `:max_new_tokens` applies to each pass of the decoder over a window. Its default is `max_length` of `generation_config.json` (448 for Whisper) minus the length of the prompt, as in the pipeline of transformers.

  An English-only model, such as `whisper-tiny.en`, raises an error for a `:language` or `:task` other than English transcription.
  """
  defdelegate speech_to_text(source, opts \\ []), to: Transmuter.Audio.SpeechToText
end
