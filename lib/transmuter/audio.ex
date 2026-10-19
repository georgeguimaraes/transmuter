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

  The input is a tensor of PCM samples at the sampling rate of the model (16 kHz for Whisper), of any float type, or a list of them. The result is `%{chunks: [%{text: text, start_timestamp_seconds: nil, end_timestamp_seconds: nil}]}`, as in Bumblebee. The text starts with a space, as in Python transformers.

  The serving takes audio of up to 30 s and raises an error for longer audio. Split longer audio into parts of 30 s or less, for example at silences, and run each part.

  The repository must have `onnx/encoder_model.onnx` and `onnx/decoder_model_merged.onnx`, as the Whisper exports of Xenova and onnx-community. The default variant is fp32. The prompt of the decoder is the one of Python transformers without timestamps: the start token, the language and the task for a multilingual model, and `<|notimestamps|>`. The settings of generation come from `generation_config.json`, such as `suppress_tokens` and `begin_suppress_tokens`.

      serving = Transmuter.Audio.speech_to_text({:hf, "onnx-community/whisper-base"}, language: "fr")

  ## Options

    * `:language` - the language code of the speech for a multilingual model, such as `"fr"`. The default is `"en"`. The model does not detect the language.

    * `:task` - `:transcribe` (the default) gives the text in the language of the speech. `:translate` gives English text.

    * `:max_new_tokens`, `:strategy`, `:temperature`, `:top_k`, `:top_p`, `:repetition_penalty` and `:seed` - as in `Transmuter.Text.generation/2`.

  An English-only model, such as `whisper-tiny.en`, raises an error for a `:language` or `:task` other than English transcription.
  """
  defdelegate speech_to_text(source, opts \\ []), to: Transmuter.Audio.SpeechToText
end
