defmodule Transmuter.AudioTest do
  use ExUnit.Case, async: true

  # Reference values come from WhisperFeatureExtractor and the generate function of Python
  # transformers, see test/support/pipeline_references.py. The audio is the first sample of
  # hf-internal-testing/librispeech_asr_dummy, 5.86 s at 16 kHz.
  @moduletag timeout: 900_000

  @device (System.get_env("TRANSMUTER_TEST_DEVICE") ||
             if(match?({:unix, :darwin}, :os.type()), do: "gpu", else: "cpu"))
          |> String.to_existing_atom()
  @backend {EMLX.Backend, device: @device}
  @opts [
    backend: @backend,
    defn_options: [compiler: EMLX, device: @device],
    strategy: :greedy
  ]

  @audio Nx.from_binary(File.read!("test/fixtures/librispeech_0.f32"), :f32)
  @text " Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel."

  alias Transmuter.Audio.SpeechToText

  test "the features match WhisperFeatureExtractor" do
    featurizer = SpeechToText.featurizer(Transmuter.Repo.load!({:hf, "Xenova/whisper-tiny.en"}))
    padding = featurizer.n_samples - Nx.size(@audio)
    audio = @audio |> Nx.pad(0.0, [{0, padding, 0}]) |> Nx.new_axis(0)
    on_device = &Nx.backend_transfer(&1, @backend)

    features =
      SpeechToText.features(
        on_device.(audio),
        on_device.(featurizer.filters),
        on_device.(featurizer.window),
        featurizer.hop_length
      )[0]
      |> Nx.backend_transfer(Nx.BinaryBackend)

    assert Nx.shape(features) == {80, 3000}
    assert_all_close(features[[0, 0..4]], [0.11933, -0.09458, -0.10978, -0.01956, 0.02249])
    assert_all_close(features[[40, 100..102]], [-0.16322, -0.16829, -0.01104])
    assert_in_delta Nx.to_number(Nx.mean(features)), -0.65635, 1.0e-4
  end

  test "greedy transcription matches transformers" do
    for repo <- ["Xenova/whisper-tiny.en", "onnx-community/whisper-base"] do
      serving = Transmuter.Audio.speech_to_text({:hf, repo}, @opts)

      assert %{chunks: [%{text: @text, start_timestamp_seconds: nil}]} =
               Nx.Serving.run(serving, @audio)
    end
  end

  test "a forced language changes the prompt of a multilingual model" do
    serving =
      Transmuter.Audio.speech_to_text(
        {:hf, "onnx-community/whisper-base"},
        [language: "fr"] ++ @opts
      )

    assert %{chunks: [%{text: text}]} = Nx.Serving.run(serving, @audio)

    assert text ==
             " Mr. Quilter est le passé de la classe de la ville, et nous sommes glad de bienvenue son gosse-boule."
  end

  test "audio longer than 30 s raises" do
    serving = Transmuter.Audio.speech_to_text({:hf, "Xenova/whisper-tiny.en"}, @opts)
    long = Nx.broadcast(Nx.tensor(0.0), {16_000 * 31})

    assert_raise ArgumentError, ~r/takes up to 30 s/, fn -> Nx.Serving.run(serving, long) end
  end

  defp assert_all_close(tensor, expected) do
    expected = Nx.tensor(expected)

    assert Nx.all_close(tensor, expected, atol: 1.0e-4) == Nx.tensor(1, type: :u8),
           "expected #{inspect(Nx.to_list(expected))}, got #{inspect(Nx.to_list(tensor))}"
  end
end
