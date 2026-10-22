defmodule Transmuter.AudioTest do
  use ExUnit.Case, async: true

  # Reference values come from WhisperFeatureExtractor, the generate function and the
  # automatic-speech-recognition pipeline of Python transformers, see
  # test/support/pipeline_references.py. The audio is the first sample of
  # hf-internal-testing/librispeech_asr_dummy, 5.86 s at 16 kHz.
  @moduletag :hub
  @moduletag timeout: 900_000

  @backend Transmuter.TestBackend.backend()
  @opts [
    backend: @backend,
    defn_options: Transmuter.TestBackend.compiler_opts(),
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

  @tag :smoke
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

  # The decoder of Xenova/whisper-tiny.en gives the cross attentions that word timestamps read.
  test "short audio with word timestamps matches transformers" do
    serving =
      Transmuter.Audio.speech_to_text(
        {:hf, "Xenova/whisper-tiny.en"},
        [timestamps: :words] ++ @opts
      )

    assert_words(Nx.Serving.run(serving, @audio), [
      {" Mr.", 0.0, 1.02},
      {" Quilter", 1.02, 1.24},
      {" is", 1.24, 1.48},
      {" the", 1.48, 1.68},
      {" apostle", 1.68, 2.08},
      {" of", 2.08, 2.36},
      {" the", 2.36, 2.5},
      {" middle", 2.5, 2.7},
      {" classes,", 2.7, 3.38},
      {" and", 3.38, 3.54},
      {" we", 3.54, 3.7},
      {" are", 3.7, 3.82},
      {" glad", 3.82, 4.1},
      {" to", 4.1, 4.32},
      {" welcome", 4.32, 4.64},
      {" his", 4.64, 4.94},
      {" gospel.", 4.94, 5.84}
    ])
  end

  # The audio of transformers without noise: VITS with noise_scale and noise_scale_duration at
  # 0, and SpeechT5 with a prenet dropout that keeps all values. See
  # test/support/pipeline_references.py. Transmuter is within 8.5e-5 of PyTorch on the GPU and
  # the CPU of a Mac. ONNX Runtime on the SpeechT5 export is 1.1e-4 from PyTorch, so the test
  # allows 2e-4.
  @speaker Nx.from_binary(File.read!("test/fixtures/speaker_embeddings.f32"), :f32)
  @speech_opts Keyword.delete(@opts, :strategy)
  @vits {"Xenova/mms-tts-eng", "Hello, my dog is cute.", "test/fixtures/mms_tts_eng.f32", []}
  @speecht5 {"Xenova/speecht5_tts", "Hi.", "test/fixtures/speecht5_hi.f32",
             [speaker_embedding: @speaker]}

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

  test "VITS without noise matches transformers, and the seed decides its audio" do
    assert_reference_audio(@vits)
    assert_seeded_audio(@vits)
  end

  # The SpeechT5 files are 642 MB, so CI leaves this test out.
  @tag :large_model
  test "SpeechT5 without noise matches transformers, and the seed decides its audio" do
    assert_reference_audio(@speecht5)
    assert_seeded_audio(@speecht5)
  end

  defp assert_reference_audio({repo, text, path, opts}) do
    serving =
      Transmuter.Audio.text_to_speech({:hf, repo}, [noise: false] ++ opts ++ @speech_opts)

    expected = Nx.from_binary(File.read!(path), :f32)
    assert %{audio: audio, sampling_rate: 16_000} = Nx.Serving.run(serving, text)
    assert Nx.shape(audio) == Nx.shape(expected), repo
    difference = audio |> Nx.subtract(expected) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
    assert difference < 2.0e-4, "#{repo}: the audio differs by #{difference}"
  end

  # The noise of VITS and the dropout of SpeechT5 come from the seed, so there is no
  # reference. The audio of a seed is the same alone and in a list, and its length is near
  # the length without noise.
  defp assert_seeded_audio({repo, text, path, opts}) do
    serving = Transmuter.Audio.text_to_speech({:hf, repo}, opts ++ @speech_opts)
    inputs = [%{text: text, seed: 1}, %{text: text, seed: 2}]
    results = [one, two] = Nx.Serving.run(serving, inputs)

    assert results == Enum.map(inputs, &Nx.Serving.run(serving, &1))
    refute one.audio == two.audio

    length = div(File.stat!(path).size, 4)
    assert Nx.size(one.audio) in div(length, 2)..(length * 2), repo
  end

  # The words are equal, and their times are within one step of 0.02 s.
  @doc false
  def assert_words(%{chunks: chunks}, expected) do
    assert Enum.map(chunks, & &1.text) == Enum.map(expected, &elem(&1, 0))

    for {chunk, {_text, start, stop}} <- Enum.zip(chunks, expected) do
      assert_in_delta chunk.start_timestamp_seconds, start, 0.021
      assert_in_delta chunk.end_timestamp_seconds, stop, 0.021
    end
  end

  defp assert_all_close(tensor, expected) do
    expected = Nx.tensor(expected)

    assert Nx.all_close(tensor, expected, atol: 1.0e-4) == Nx.tensor(1, type: :u8),
           "expected #{inspect(Nx.to_list(expected))}, got #{inspect(Nx.to_list(tensor))}"
  end
end

# The long audio tests and the batch test are each slower than all other audio tests
# together on EXLA in CI, so they are modules of their own, in their own CI partitions.
defmodule Transmuter.AudioLongTest do
  use ExUnit.Case, async: true

  @moduletag :hub
  @moduletag timeout: 900_000
  @moduletag ci_partition: 2

  @opts [
    backend: Transmuter.TestBackend.backend(),
    defn_options: Transmuter.TestBackend.compiler_opts(),
    strategy: :greedy
  ]

  # The first 5 samples of librispeech_asr_dummy, 62.46 s. The pipeline of transformers with
  # chunk_length_s=30 runs 3 windows. The texts differ between the models.
  @long Nx.from_binary(File.read!("test/fixtures/librispeech_long.f32"), :f32)
  @long_text " Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel. Nor is Mr. Quilter's manner less interesting than his matter. He tells us that at this festive season of the year, with Christmas and roast beef looming before us, "

  # The words of whisper-tiny.en in the pipeline of transformers with
  # return_timestamps="word", see test/support/pipeline_references.py. A word starts at the
  # end of the token before it, so a word can start before the end of the word before it or
  # have no length, as in transformers.
  @long_words [
    {" Mr.", 0.0, 0.96},
    {" Quilter", 0.96, 1.24},
    {" is", 1.24, 1.48},
    {" the", 1.48, 1.68},
    {" apostle", 1.68, 2.08},
    {" of", 2.08, 2.34},
    {" the", 2.34, 2.5},
    {" middle", 2.5, 2.7},
    {" classes,", 2.7, 3.4},
    {" and", 3.4, 3.54},
    {" we", 3.54, 3.7},
    {" are", 3.7, 3.82},
    {" glad", 3.82, 4.1},
    {" to", 4.1, 4.3},
    {" welcome", 4.3, 4.64},
    {" his", 4.64, 4.94},
    {" gospel.", 4.94, 6.44},
    {" Nor", 6.56, 6.74},
    {" is", 6.74, 7.0},
    {" Mr.", 7.0, 7.34},
    {" Quilter's", 7.34, 7.88},
    {" manner", 7.88, 8.1},
    {" less", 8.1, 8.64},
    {" interesting", 8.64, 9.04},
    {" than", 9.04, 9.48},
    {" his", 9.48, 9.98},
    {" matter.", 9.98, 11.26},
    {" He", 11.26, 11.46},
    {" tells", 11.46, 11.7},
    {" us", 11.7, 12.12},
    {" that", 12.12, 12.36},
    {" at", 12.36, 12.6},
    {" this", 12.6, 12.86},
    {" festive", 12.86, 13.32},
    {" season", 13.32, 13.74},
    {" of", 13.74, 13.98},
    {" the", 13.98, 14.18},
    {" year,", 14.18, 14.98},
    {" with", 14.98, 15.24},
    {" Christmas", 15.24, 15.62},
    {" and", 15.62, 15.94},
    {" roast", 15.94, 16.28},
    {" beef", 16.28, 16.64},
    {" looming", 16.64, 16.98},
    {" before", 16.98, 17.32},
    {" us,", 17.32, 18.62},
    {" similes", 18.62, 18.98},
    {" drawn", 18.98, 19.32},
    {" from", 19.32, 19.62},
    {" eating", 19.62, 19.92},
    {" and", 19.92, 20.16},
    {" its", 20.16, 20.4},
    {" results", 20.4, 20.84},
    {" occur", 20.84, 21.24},
    {" most", 21.24, 21.64},
    {" readily", 21.64, 21.96},
    {" to", 21.96, 22.24},
    {" the", 22.24, 22.44},
    {" mind.", 22.44, 23.76},
    {" He", 23.76, 23.96},
    {" has", 23.96, 24.22},
    {" grave", 24.22, 24.5},
    {" doubts", 24.5, 24.92},
    {" whether", 24.92, 25.44},
    {" Sir", 25.42, 25.7},
    {" Frederick", 25.7, 26.02},
    {" Layton's", 26.02, 26.56},
    {" work", 26.56, 26.82},
    {" is", 26.82, 27.14},
    {" really", 27.14, 27.64},
    {" Greek", 27.64, 28.02},
    {" after", 28.02, 28.4},
    {" all,", 28.4, 29.22},
    {" and", 29.22, 29.52},
    {" can", 29.52, 29.8},
    {" discover", 29.8, 30.18},
    {" in", 30.18, 30.36},
    {" it", 30.36, 30.58},
    {" but", 30.58, 30.82},
    {" little", 30.82, 31.16},
    {" of", 31.16, 31.66},
    {" rocky", 31.66, 32.06},
    {" Ithaca.", 32.06, 33.68},
    {" Linnell's", 33.8, 34.36},
    {" pictures", 34.36, 34.64},
    {" are", 34.64, 35.2},
    {" a", 35.2, 35.54},
    {" sort", 35.54, 35.54},
    {" of", 35.54, 36.1},
    {" up", 36.1, 36.42},
    {"-gards", 36.42, 36.66},
    {" and", 36.66, 36.94},
    {" atom", 36.94, 37.48},
    {" paintings,", 37.48, 38.48},
    {" and", 38.48, 38.76},
    {" Mason's", 38.76, 39.4},
    {" exquisite", 39.4, 39.84},
    {" idles", 40.08, 40.32},
    {" are", 40.32, 40.66},
    {" as", 40.66, 41.12},
    {" national", 41.12, 41.66},
    {" as", 41.66, 41.94},
    {" a", 41.94, 42.08},
    {" jingo", 42.08, 42.38},
    {" poem.", 42.38, 44.22},
    {" Mr.", 44.64, 44.96},
    {" Birkett", 44.96, 45.28},
    {" Foster's", 45.28, 46.1},
    {" landscapes", 46.1, 46.38},
    {" smile", 46.38, 47.18},
    {" at", 47.18, 47.48},
    {" one", 47.48, 47.8},
    {" much", 47.8, 48.08},
    {" in", 48.08, 48.22},
    {" the", 48.22, 48.4},
    {" same", 48.4, 48.62},
    {" way", 48.62, 48.86},
    {" that", 48.86, 49.1},
    {" Mr.", 49.1, 49.46},
    {" Carker", 49.46, 49.94},
    {" used", 49.94, 50.46},
    {" to", 50.46, 50.64},
    {" flash", 50.92, 50.92},
    {" his", 50.92, 51.28},
    {" teeth.", 51.28, 52.7},
    {" And", 52.94, 53.04},
    {" Mr.", 53.04, 53.4},
    {" John", 53.4, 53.64},
    {" Collier", 53.64, 54.1},
    {" gives", 54.1, 54.7},
    {" his", 54.7, 54.98},
    {" sitter", 54.98, 55.38},
    {" a", 55.38, 55.66},
    {" cheerful", 55.66, 56.04},
    {" slap", 56.04, 56.58},
    {" in", 56.58, 56.78},
    {" the", 56.78, 56.98},
    {" back,", 56.98, 57.62},
    {" before", 57.62, 57.92},
    {" he", 57.92, 58.24},
    {" says,", 58.24, 58.62},
    {" like", 58.62, 58.84},
    {" a", 58.84, 59.16},
    {" shampooer", 59.16, 59.84},
    {" and", 59.84, 60.0},
    {" a", 60.0, 60.16},
    {" Turkish", 60.16, 60.44},
    {" bath,", 60.44, 61.32},
    {" next", 61.32, 61.66},
    {" man!", 61.66, 62.44}
  ]

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

  # The 3 windows are one batch. They have different numbers of frames and of passes, so a
  # window that is done is a padding row in the later passes of the others. The merge
  # compares the times of the tokens in the overlaps, as in transformers.
  test "long audio word timestamps in a batch match the pipeline of transformers" do
    serving =
      Transmuter.Audio.speech_to_text(
        {:hf, "Xenova/whisper-tiny.en"},
        [timestamps: :words, compile: [batch_size: 3]] ++ @opts
      )

    Transmuter.AudioTest.assert_words(Nx.Serving.run(serving, @long), @long_words)
  end
end

defmodule Transmuter.AudioBatchTest do
  use ExUnit.Case, async: true

  @moduletag :hub
  @moduletag timeout: 900_000
  @moduletag ci_partition: 3

  @opts [
    backend: Transmuter.TestBackend.backend(),
    defn_options: Transmuter.TestBackend.compiler_opts(),
    strategy: :greedy
  ]

  @audio Nx.from_binary(File.read!("test/fixtures/librispeech_0.f32"), :f32)
  @long Nx.from_binary(File.read!("test/fixtures/librispeech_long.f32"), :f32)

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
end

defmodule Transmuter.Audio.WordTimestampsTest do
  use ExUnit.Case, async: true

  alias Transmuter.Audio.WordTimestamps

  # _median_filter of transformers reflects the columns of the audio at both of its ends. With
  # 1499 frames, the last columns read the columns before column 1498, not the padding of
  # the 1500 encoder positions.
  test "the median filter reflects the columns at the end of the audio" do
    values = for i <- 0..1499, do: :math.sin(i * 1.7)
    weights = Nx.tensor([values], type: :f32)
    frames = 1499
    result = WordTimestamps.median(weights, Nx.tensor(frames), 7)

    values = weights |> Nx.to_flat_list() |> List.to_tuple()
    reflect = &if(abs(&1) > frames - 1, do: 2 * (frames - 1) - abs(&1), else: abs(&1))

    expected =
      for c <- 0..(frames - 1) do
        (c - 3)..(c + 3) |> Enum.map(&elem(values, reflect.(&1))) |> Enum.sort() |> Enum.at(3)
      end

    assert result[[0, 0..(frames - 1)]] |> Nx.to_flat_list() == expected
  end
end

defmodule Transmuter.AudioClassificationTest do
  use ExUnit.Case, async: true

  # Reference values come from the feature extractors and the audio-classification pipeline of
  # Python transformers, see test/support/pipeline_references.py. The reference of the
  # wav2vec2 export is ONNX Runtime, because its PyTorch model is no longer on the Hub.
  @moduletag :hub
  @moduletag timeout: 900_000
  @moduletag ci_partition: 5

  @backend Transmuter.TestBackend.backend()
  @opts [backend: @backend, defn_options: Transmuter.TestBackend.compiler_opts()]

  alias Transmuter.Audio.Featurizer

  @ast {:hf, "Xenova/ast-finetuned-audioset-10-10-0.4593"}
  @wav2vec2 {:hf, "onnx-community/wav2vec2-base-Speech_Emotion_Recognition-ONNX"}

  # 5.86 s, 62.46 s and the first 2 s of speech. AST takes 1024 frames, about 10.24 s.
  @audio Nx.from_binary(File.read!("test/fixtures/librispeech_0.f32"), :f32)
  @long Nx.from_binary(File.read!("test/fixtures/librispeech_long.f32"), :f32)
  @cut @audio[0..31_999]

  # The AST features of 5.86 s have 584 frames, then frames of zeros, which normalize to
  # 0.46703. The features of 62.46 s keep the first 1024 frames.
  test "the featurizers match the feature extractors of transformers" do
    for {repo, audio, shape, points, expected, mean} <- [
          {@ast, @audio, {1024, 128}, [[0, 0], [100, 37], [583, 127], [584, 0], [1023, 127]],
           [-0.98939, 0.34332, -0.72132, 0.46703, 0.46703], 0.03871},
          {@ast, @long, {1024, 128}, [[0, 0], [500, 64], [1023, 127]],
           [-0.98939, -0.09159, 0.04984], -0.25028},
          {@wav2vec2, @audio, {93_680}, [[0], [1000], [93_679]], [0.03858, 0.00064, 0.0169], 0.0},
          {@wav2vec2, @cut, {32_000}, [[0], [1000], [31_999]], [0.03465, 0.00015, 0.29631], 0.0}
        ] do
      features =
        Featurizer.load!(repo)
        |> Featurizer.process([audio], backend: @backend)
        |> Nx.backend_transfer(Nx.BinaryBackend)

      features = features[0]
      assert Nx.shape(features) == shape
      assert_all_close(Nx.to_list(Nx.gather(features, Nx.tensor(points))), expected)
      assert_in_delta Nx.to_number(Nx.mean(features)), mean, 1.0e-4
    end
  end

  test "audio classification matches the pipeline of transformers" do
    for {repo, inputs, expected} <- [
          {@ast, [@audio, @long],
           [
             [
               {"Speech", 0.841193},
               {"Speech synthesizer", 0.024827},
               {"Narration, monologue", 0.01955},
               {"Male speech, man speaking", 0.01018},
               {"Gasp", 0.006851}
             ],
             [
               {"Speech", 0.807631},
               {"Speech synthesizer", 0.091229},
               {"Narration, monologue", 0.049116},
               {"Male speech, man speaking", 0.008419},
               {"Clicking", 0.007501}
             ]
           ]},
          {@wav2vec2, [@audio, @cut],
           [
             [
               {"ANGRY", 0.681742},
               {"HAPPY", 0.134327},
               {"DISGUST", 0.068611},
               {"NEUTRAL", 0.065566},
               {"FEAR", 0.025946}
             ],
             [
               {"ANGRY", 0.721429},
               {"HAPPY", 0.127242},
               {"DISGUST", 0.055477},
               {"NEUTRAL", 0.05137},
               {"FEAR", 0.024066}
             ]
           ]}
        ] do
      serving = Transmuter.Audio.audio_classification(repo, @opts)

      expected =
        Enum.map(expected, &%{predictions: Enum.map(&1, fn {l, s} -> %{label: l, score: s} end)})

      assert_predictions(Nx.Serving.run(serving, inputs), expected, 1.0e-4)
    end
  end

  # Three audio in batches of 2: a full batch and a batch with a padding row. A batch of
  # another size can round the last bits of the scores in another way on the GPU.
  test "AST gives the same result in a batch as alone" do
    inputs = [@audio, @long, @cut]
    alone = Transmuter.Audio.audio_classification(@ast, @opts)
    batched = Transmuter.Audio.audio_classification(@ast, [compile: [batch_size: 2]] ++ @opts)
    expected = Enum.map(inputs, &Nx.Serving.run(alone, &1))
    assert_predictions(Nx.Serving.run(batched, inputs), expected, 1.0e-6)
  end

  # CLAP takes 10 s at 48 kHz. test/fixtures/librispeech_0_48k.f32 is librispeech_0.f32
  # resampled by librosa. The 5.86 s repeat once and end with zeros. The first 2 s repeat 5
  # times and fill the 10 s. Both have no energy above 8 kHz, so the highest mel bins are
  # -100 dB.
  @clap {:hf, "Xenova/clap-htsat-unfused"}
  @audio_48k Nx.from_binary(File.read!("test/fixtures/librispeech_0_48k.f32"), :f32)
  @clap_labels ["a man speaking", "music", "a dog barking", "rain"]

  test "the CLAP featurizer matches ClapFeatureExtractor" do
    points = [[0, 0], [100, 37], [281, 63], [290, 10], [600, 20], [1000, 63]]

    for {audio, expected, mean} <- [
          {@audio_48k, [-35.2137, -27.546, -100.0, -22.3653, -100.0, -100.0], -66.9249},
          {@audio_48k[0..95_999], [-35.2137, -27.546, -100.0, -28.7763, -31.2363, -66.9281],
           -44.5233}
        ] do
      features =
        Featurizer.load!(@clap)
        |> Featurizer.process([audio], backend: @backend)
        |> Nx.backend_transfer(Nx.BinaryBackend)

      features = features[[0, 0]]
      assert Nx.shape(features) == {1001, 64}

      for {got, want} <- Enum.zip(Nx.to_list(Nx.gather(features, Nx.tensor(points))), expected),
          do: assert_in_delta(got, want, 1.0e-3)

      assert_in_delta Nx.to_number(Nx.mean(features)), mean, 1.0e-3
    end
  end

  # The q8 file quantizes the activations 105 times when the model runs, so its scores
  # depend on float rounding inside the network. On the first 2 s, all operators of EMLX and
  # the EXLA CPU agree within 1e-6, but the score of "a man speaking" is 0.654 on the EXLA
  # CPU without compilation, 0.696 with compilation, 0.858 on EMLX CPU and 0.607 on EMLX GPU.
  # ONNX Runtime gives 0.755. So this test checks only the order of the two top labels, and
  # the fp32 test in ZeroShotAudioClassificationLargeModelTest compares the scores within 1e-4.
  test "zero-shot audio classification with the q8 CLAP file gives the top labels" do
    serving =
      Transmuter.Audio.zero_shot_audio_classification(
        @clap,
        @clap_labels,
        [variant: :q8, top_k: 2, compile: [batch_size: 2]] ++ @opts
      )

    for %{predictions: predictions} <-
          Nx.Serving.run(serving, [@audio_48k, @audio_48k[0..95_999]]) do
      assert Enum.map(predictions, & &1.label) == ["a man speaking", "music"]
    end
  end

  defp assert_predictions(results, expected, delta) do
    for {%{predictions: got}, %{predictions: want}} <- Enum.zip(results, expected) do
      assert Enum.map(got, & &1.label) == Enum.map(want, & &1.label)

      for {g, w} <- Enum.zip(got, want),
          do:
            assert_in_delta(
              g.score,
              w.score,
              delta,
              "expected #{inspect(want)}, got #{inspect(got)}"
            )
    end
  end

  defp assert_all_close(got, expected) do
    for {g, e} <- Enum.zip(got, expected) do
      assert_in_delta g, e, 1.0e-4, "expected #{inspect(expected)}, got #{inspect(got)}"
    end
  end
end

# The fp32 CLAP file is 619 MB, more than the model cache of CI can keep, so this test is
# :large_model. A module with a :ci_partition tag cannot have it, see
# test/transmuter/ci_partition_test.exs.
defmodule Transmuter.ZeroShotAudioClassificationLargeModelTest do
  use ExUnit.Case, async: true

  # The reference is the zero-shot-audio-classification pipeline of Python transformers with
  # the PyTorch model, see test/support/pipeline_references.py.
  @moduletag :hub
  @moduletag timeout: 900_000

  @opts [
    backend: Transmuter.TestBackend.backend(),
    defn_options: Transmuter.TestBackend.compiler_opts()
  ]

  @audio Nx.from_binary(File.read!("test/fixtures/librispeech_0_48k.f32"), :f32)

  @tag :large_model
  test "zero-shot audio classification with CLAP matches the pipeline" do
    labels = ["a man speaking", "music", "a dog barking", "rain"]

    serving =
      Transmuter.Audio.zero_shot_audio_classification(
        {:hf, "Xenova/clap-htsat-unfused"},
        labels,
        [top_k: nil] ++ @opts
      )

    expected = [
      [
        {"a man speaking", 0.950146},
        {"music", 0.046542},
        {"rain", 0.002626},
        {"a dog barking", 0.000685}
      ],
      [
        {"a man speaking", 0.88723},
        {"music", 0.106899},
        {"a dog barking", 0.004245},
        {"rain", 0.001626}
      ]
    ]

    results = Nx.Serving.run(serving, [@audio, @audio[0..95_999]])

    for {%{predictions: got}, want} <- Enum.zip(results, expected) do
      assert Enum.map(got, & &1.label) == Enum.map(want, &elem(&1, 0))

      for {%{score: score}, {_label, expected}} <- Enum.zip(got, want),
          do: assert_in_delta(score, expected, 1.0e-4, inspect(got))
    end
  end
end
