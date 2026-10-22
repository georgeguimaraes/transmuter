defmodule Transmuter.GenerationTest do
  use ExUnit.Case, async: true

  # Beam search. The references come from the generate function of Python transformers with
  # num_beams, see the beam search section of test/support/pipeline_references.py. Each case
  # gives another text than greedy selection, or another text than another row of its table.
  @moduletag :hub
  @moduletag timeout: 900_000
  @moduletag ci_partition: 5

  @opts [
    backend: Transmuter.TestBackend.backend(),
    defn_options: Transmuter.TestBackend.compiler_opts()
  ]

  alias Transmuter.{Audio, Text}

  # The best beam of "What is 2 + 2?" ends with eos at the eighth token while the other
  # beams still run. Greedy selection gives "2 + 2 is a classic arithmetic operation that
  # can be" and " with the printing press. The first printing press was the first".
  test "beam search matches transformers in each mode of the KV cache" do
    chat = %{messages: [%{role: "user", content: "What is 2 + 2?"}]}

    for {repo, input, expected} <- [
          {"HuggingFaceTB/SmolLM2-135M-Instruct", chat, "2 + 2 = 4."},
          {"onnx-community/SmolLM2-135M-Instruct-ONNX", chat, "2 + 2 = 4."},
          {"Xenova/gpt2", "The history of the printing press begins",
           " with the invention of the printing press. The printing press was"}
        ] do
      opts = [variant: :fp32, num_beams: 3, max_new_tokens: 12] ++ @opts
      serving = Text.generation({:hf, repo}, opts)
      assert %{results: [%{text: ^expected}]} = Nx.Serving.run(serving, input)
    end
  end

  test "length_penalty and early_stopping change the text as in transformers" do
    chat = %{messages: [%{role: "user", content: "Say yes or no."}]}

    for {opts, expected} <- [
          {[], "Yes"},
          {[length_penalty: 2.0], "Yes."},
          {[length_penalty: 2.0, early_stopping: :never],
           "I am ready to assist you with any question or request you may have."}
        ] do
      opts = [variant: :fp32, num_beams: 3, max_new_tokens: 16] ++ opts ++ @opts
      serving = Text.generation({:hf, "HuggingFaceTB/SmolLM2-135M-Instruct"}, opts)
      assert %{results: [%{text: ^expected}]} = Nx.Serving.run(serving, chat)
    end

    # The two inputs run as one batch. Greedy selection gives "frog is very small and has
    # bright colors . it lives near streams and eats insects ." and "Le temps est agréable
    # aujourd'hui, et nous sommes à pied jusqu'au parc." As the text2text-generation pipeline,
    # the text keeps the spaces of the tokens (clean_up_tokenization_spaces=False).
    inputs = [
      "summarize: Scientists found a new species of frog in the rainforest. The frog is very small and has bright colors. It lives near streams and eats insects.",
      "translate English to French: The weather is nice today, so we walk to the park."
    ]

    for {opts, expected} <- [
          {[],
           [
             "the frog is very small and has bright colors . it lives near streams and eats insects .",
             "Le temps est agréable aujourd'hui, et nous sommes en train de marcher jusqu'au parc."
           ]},
          {[early_stopping: true],
           [
             "frog is very small and has bright colors . it lives near streams and eats insects .",
             "Le temps est agréable aujourd'hui, donc nous sommes à pied jusqu'au parc."
           ]},
          {[length_penalty: -1.0],
           [
             "the frog lives near streams and eats insects .",
             "Le temps est agréable aujourd'hui, donc nous sommes à pied jusqu'au parc."
           ]}
        ] do
      opts = [num_beams: 3, max_new_tokens: 48, compile: [batch_size: 2]] ++ opts ++ @opts
      serving = Text.text_to_text({:hf, "Xenova/t5-small"}, opts)
      assert Enum.map(Nx.Serving.run(serving, inputs), &hd(&1.results).text) == expected
    end
  end

  # Marian takes 4 beams, bad_words_ids, forced_eos_token_id and renormalize_logits from its
  # generation_config.json. Greedy selection gives "Transmuter führt ONNX-Modelle auf jedem
  # Nx-Backend aus, ohne Code für jede Architektur." With 5 new tokens, the fifth token of
  # each beam is the forced eos token.
  test "Marian takes beam search from its generation_config.json" do
    sentence =
      "Transmuter runs ONNX models on any Nx backend, with no code for each architecture."

    for {opts, expected} <- [
          {[max_new_tokens: 64],
           "Transmuter läuft ONNX-Modelle auf jedem Nx-Backend, ohne Code für jede Architektur."},
          {[max_new_tokens: 5], "Transmuter führt"}
        ] do
      serving = Text.text_to_text({:hf, "Xenova/opus-mt-en-de"}, opts ++ @opts)
      assert %{results: [%{text: ^expected}]} = Nx.Serving.run(serving, sentence)
    end
  end

  # Greedy selection gives " Mr. Quilter est le passé de la classe de la ville, et nous sommes
  # glad de bienvenue son gosse-boule." The timestamps of whisper-tiny.en come from the
  # processor of the timestamps on the rows of the beams.
  test "beam search transcribes as transformers" do
    audio = Nx.from_binary(File.read!("test/fixtures/librispeech_0.f32"), :f32)
    opts = [num_beams: 3] ++ @opts

    serving = Audio.speech_to_text({:hf, "onnx-community/whisper-base"}, [language: "fr"] ++ opts)

    assert %{chunks: [%{text: text}]} = Nx.Serving.run(serving, audio)

    assert text ==
             " Mr. Quilter est le passé de la classe de la ville, et nous sommes glad d'en welcoming ses gosses."

    serving =
      Audio.speech_to_text({:hf, "Xenova/whisper-tiny.en"}, [timestamps: :segments] ++ opts)

    assert %{chunks: [%{start_timestamp_seconds: +0.0, end_timestamp_seconds: 5.44, text: text}]} =
             Nx.Serving.run(serving, audio)

    assert text ==
             " Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel."
  end

  # The fourth input runs in a partial batch with two padding rows, and gpt2 runs a group for
  # each prompt length. The rows of the beams of each prompt follow each other.
  test "a batch with beams gives the text of each prompt alone" do
    questions = [
      "What is the capital of France?",
      "What is 2 + 2?",
      "Say yes or no.",
      "Write a long poem about the sea and the wind, in three verses."
    ]

    chats = for q <- questions, do: %{messages: [%{role: "user", content: q}]}

    for {repo, inputs} <- [
          {"onnx-community/SmolLM2-135M-Instruct-ONNX", chats},
          {"Xenova/gpt2", questions}
        ] do
      opts = [variant: :fp32, num_beams: 3, max_new_tokens: 12] ++ @opts
      single = Text.generation({:hf, repo}, opts)
      batched = Text.generation({:hf, repo}, [compile: [batch_size: 3]] ++ opts)
      alone = for input <- inputs, do: Nx.Serving.run(single, input)
      assert Nx.Serving.run(batched, inputs) == alone
    end
  end

  test "beam search raises with sampling and with a stream" do
    repo = {:hf, "Xenova/gpt2"}

    assert_raise ArgumentError, ~r/strategy: :greedy/, fn ->
      Text.generation(repo, [variant: :fp32, num_beams: 3, strategy: :sampling] ++ @opts)
    end

    assert_raise ArgumentError, ~r/stream: true/, fn ->
      Text.generation(repo, [variant: :fp32, num_beams: 3, stream: true] ++ @opts)
    end
  end
end
