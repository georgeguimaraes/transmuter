defmodule Transmuter.AudioTest do
  use ExUnit.Case, async: true

  # Reference values come from WhisperFeatureExtractor, the generate function and the
  # automatic-speech-recognition pipeline of Python transformers, see
  # test/support/pipeline_references.py. The audio is the first sample of
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

  test "short audio with timestamps gives one segment" do
    serving =
      Transmuter.Audio.speech_to_text(
        {:hf, "Xenova/whisper-tiny.en"},
        [timestamps: :segments] ++ @opts
      )

    assert %{chunks: [%{text: @text, start_timestamp_seconds: +0.0, end_timestamp_seconds: 5.44}]} =
             Nx.Serving.run(serving, @audio)
  end

  # The first 5 samples of librispeech_asr_dummy, 62.46 s. The pipeline of transformers with
  # chunk_length_s=30 runs 3 windows. The texts differ between the models.
  @long Nx.from_binary(File.read!("test/fixtures/librispeech_long.f32"), :f32)
  @long_text " Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel. Nor is Mr. Quilter's manner less interesting than his matter. He tells us that at this festive season of the year, with Christmas and roast beef looming before us, "

  test "long audio matches the pipeline of transformers" do
    for {repo, text} <- [
          {"Xenova/whisper-tiny.en",
           "similes drawn from eating and its results occur most readily to the mind. He has grave doubts whether Sir Frederick Layton's work is really Greek after all, and can discover in it but little of rocky Ithaca. Linell's pictures are a sort of up-gards and atom paintings, and Mason's exquisite idles are as national as a jingo poem. Mr. Birkett Foster's landscapes smile at one much in the same way that Mr. Carker used to flash his teeth. And Mr. John Collier gives his sitter a cheerful slap in the back, before he says like a shampooer and a Turkish bath. Next man,"},
          {"onnx-community/whisper-base",
           "similarly he's drawn from eating and its results occur most readily to the mind. He has graved doubts whether Sir Frederick Layton's work is really Greek after all, and can discover in it but little of Rocky Ithaca. Lynelle's pictures are a sort of upgards and atom paintings, and Mason's exquisite ittles are as national as a jingo poem. Mr. Birkitt Foster's landscapes smile at one much in the same way that Mr. Carker used to flash his teeth. And Mr. John Collier gives his sitter a cheerful slap on the back before he says like a shampooer in a Turkish bath. Next man"}
        ] do
      serving = Transmuter.Audio.speech_to_text({:hf, repo}, @opts)
      assert %{chunks: [%{text: result}]} = Nx.Serving.run(serving, @long)
      assert result == @long_text <> text
    end
  end

  # With timestamps, generate in transformers transcribes a window again from its last
  # timestamp when the last segment is not complete: 6 passes for whisper-tiny.en and 5 for
  # whisper-base.
  test "long audio timestamps match the pipeline of transformers" do
    for {repo, expected} <- [
          {"Xenova/whisper-tiny.en",
           [
             {0.0, 50.44,
              "similes drawn from eating and its results occur most readily to the mind. He has grave doubts whether Sir Frederick Layton's work is really Greek after all, and can discover in it but little of rocky Ithaca. Linnell's pictures are a sort of up-gards and atom paintings, and Mason's exquisite idles are as national as a jingo poem. Mr. Birkett Foster's landscapes smile at one much in the same way that Mr. Carker used to"},
             {50.44, 52.92, " flash his teeth."},
             {52.92, 58.68,
              " And Mr. John Collier gives his sitter a cheerful slap in the back, before he says, like"},
             {58.68, 62.0, " a shampooer and a Turkish bath, next man!"}
           ]},
          {"onnx-community/whisper-base",
           [
             {0.0, 47.88,
              "similarly he's drawn from eating and its results occur most readily to the mind. He has graved doubts whether Sir Frederick Layton's work is really Greek after all, and can discover in it but little of Rocky Ithaca. Lynelle's pictures are a sort of upgards and atom paintings, and Mason's exquisite ittles are as national as a jingo poem. Mr. Birk at Foster's landscapes smile at one much"},
             {47.88, 53.96,
              " in the same way that Mr. Carker used to flash his teeth. And Mr. John Collier"},
             {53.96, 59.52,
              " gives his sitter a cheerful slap on the back before he says like a shampoo"},
             {59.52, 63.76, " in a Turkish bath. Next man,"}
           ]}
        ] do
      serving =
        Transmuter.Audio.speech_to_text({:hf, repo}, [timestamps: :segments] ++ @opts)

      %{chunks: chunks} = Nx.Serving.run(serving, @long)

      [{start, stop, text} | rest] = expected

      assert Enum.map(chunks, &{&1.start_timestamp_seconds, &1.end_timestamp_seconds, &1.text}) ==
               [{start, stop, @long_text <> text} | rest]
    end
  end

  # The 3 windows of the long audio and the short audio are 4 rows: 2 batches of 2, or a
  # batch of 3 and a batch with 2 padding rows. With timestamps, the windows of the long
  # audio have different numbers of passes.
  test "a batch of windows gives the result of each audio alone" do
    for {batch_size, timestamps} <- [{2, nil}, {3, :segments}] do
      opts = [timestamps: timestamps] ++ @opts
      single = Transmuter.Audio.speech_to_text({:hf, "Xenova/whisper-tiny.en"}, opts)

      batched =
        Transmuter.Audio.speech_to_text(
          {:hf, "Xenova/whisper-tiny.en"},
          [compile: [batch_size: batch_size]] ++ opts
        )

      alone = for audio <- [@long, @audio], do: Nx.Serving.run(single, audio)
      assert Nx.Serving.run(batched, [@long, @audio]) == alone
    end
  end

  # The tokenizer.json of mms-tts-eng has a model without a type, and the one of speecht5_tts
  # has a Precompiled normalizer without a charsmap.
  test "the tokenizers of the speech exports give the ids of transformers" do
    text = "Hello World! It's 3 o'clock, café? ÀB_x–y"

    for {repo, expected} <- [
          {"Xenova/mms-tts-eng",
           [0, 6, 0, 7, 0, 21, 0, 21, 0, 22, 0, 19, 0, 9, 0, 22, 0, 25, 0, 21, 0, 5, 0, 19, 0, 18] ++
             [0, 33, 0, 1, 0, 8, 0, 19, 0, 11, 0, 19, 0, 22, 0, 1, 0, 12, 0, 21, 0, 22, 0, 12, 0] ++
             [0, 0, 19, 0, 12, 0, 26, 0, 20, 0, 19, 0, 24, 0, 30, 0, 31, 0, 10, 0, 3, 0]},
          {"Xenova/speecht5_tts",
           [4, 35, 5, 15, 15, 8, 4, 38, 8, 13, 15, 14, 44, 4, 30, 6, 31, 12, 4, 3, 4, 8, 31, 17] ++
             [15, 8, 17, 28, 23, 4, 17, 7, 19, 73, 41, 4, 3, 40, 3, 37, 3, 22, 2]}
        ] do
      tokenizer = Transmuter.Tokenizer.load!({:hf, repo})
      {%{"input_ids" => ids}, _encodings} = Transmuter.Tokenizer.encode(tokenizer, [text])
      assert Nx.to_flat_list(ids) == expected, repo
    end

    tokenizer = Transmuter.Tokenizer.load!({:hf, "Xenova/mms-tts-eng"})

    assert_raise ArgumentError, ~r/gives no tokens/, fn ->
      Transmuter.Tokenizer.encode(tokenizer, ["🙂"])
    end
  end

  defp assert_all_close(tensor, expected) do
    expected = Nx.tensor(expected)

    assert Nx.all_close(tensor, expected, atol: 1.0e-4) == Nx.tensor(1, type: :u8),
           "expected #{inspect(Nx.to_list(expected))}, got #{inspect(Nx.to_list(tensor))}"
  end
end
