defmodule Transmuter.TextGLiNERTest do
  use ExUnit.Case, async: true

  alias Transmuter.{Repo, Tokenizer}
  alias Transmuter.Text.ZeroShotEntityRecognition, as: GLiNER

  @opts [threshold: 0.5, flat_ner: true, multi_label: false]

  # Synthetic logits of one row: -10 everywhere, and the given logits at their indices. A
  # logit of 0 gives a probability of exactly 0.5, the default threshold.
  defp logits(shape, entries) do
    Enum.reduce(entries, Nx.broadcast(-10.0, shape), fn {index, logit}, logits ->
      Nx.indexed_put(logits, Nx.tensor([index]), Nx.tensor([logit]))
    end)
  end

  test "decoding keeps the entities of the decoders of gliner" do
    # Span logits of {1, words, max_width, labels} at [0, start, width, label], and the
    # expected {first word, last word, label}.
    span_cases = [
      {"a span needs a score more than the threshold", [],
       [{[0, 0, 0, 0], 0.0}, {[0, 1, 0, 0], 2.0}], [{1, 1, 0}]},
      {"a span ends inside the text and has a label of the row", [words: 3, labels: 1],
       [{[0, 2, 1, 0], 2.0}, {[0, 0, 0, 1], 2.0}, {[0, 1, 1, 0], 1.0}], [{1, 2, 0}]},
      {"a span that overlaps a span with a higher score goes", [],
       [{[0, 0, 1, 0], 2.0}, {[0, 1, 1, 1], 1.0}, {[0, 3, 0, 1], 0.5}], [{0, 1, 0}, {3, 3, 1}]},
      {"without flat_ner, a span inside another stays and a partial overlap goes",
       [flat_ner: false], [{[0, 0, 2, 0], 2.0}, {[0, 1, 0, 1], 1.0}, {[0, 2, 1, 1], 1.5}],
       [{0, 2, 0}, {1, 1, 1}]},
      {"with multi_label, a span keeps more labels", [multi_label: true],
       [{[0, 0, 1, 0], 2.0}, {[0, 0, 1, 1], 1.0}, {[0, 1, 0, 1], 1.5}], [{0, 1, 0}, {0, 1, 1}]},
      {"without multi_label, a span keeps its best label", [],
       [{[0, 0, 1, 0], 2.0}, {[0, 0, 1, 1], 1.0}], [{0, 1, 0}]},
      {"equal scores keep the order of the candidates", [],
       [{[0, 2, 0, 1], 1.0}, {[0, 0, 0, 0], 1.0}, {[0, 0, 0, 1], 1.0}], [{0, 0, 0}, {2, 2, 1}]}
    ]

    # Token logits of {1, words, labels, 3} at [0, word, label, kind], with kind 0 for the
    # start, 1 for the end and 2 for the inside.
    token_cases = [
      {"an inside score at the threshold counts, and the score is the lowest score", [],
       [{[0, 0, 0, 0], 2.0}, {[0, 2, 0, 1], 2.0}] ++
         [{[0, 0, 0, 2], 0.0}, {[0, 1, 0, 2], 3.0}, {[0, 2, 0, 2], 1.0}], [{0, 2, 0}]},
      {"a start at the threshold does not count", [],
       [{[0, 1, 1, 0], 0.0}, {[0, 1, 1, 1], 2.0}, {[0, 1, 1, 2], 2.0}], []},
      {"an inside score under the threshold breaks the span", [],
       [{[0, 0, 0, 0], 2.0}, {[0, 2, 0, 1], 2.0}, {[0, 0, 0, 2], 2.0}, {[0, 1, 0, 2], -1.0}] ++
         [{[0, 2, 0, 2], 2.0}, {[0, 3, 0, 0], 2.0}, {[0, 3, 0, 1], 2.0}, {[0, 3, 0, 2], 2.0}],
       [{3, 3, 0}]},
      {"a start pairs with each later end of its label, and the best span stays", [],
       [{[0, 0, 0, 0], 3.0}, {[0, 1, 0, 1], 1.0}, {[0, 2, 0, 1], 2.0}, {[0, 0, 1, 1], 3.0}] ++
         for(word <- 0..2, do: {[0, word, 0, 2], 3.0}), [{0, 2, 0}]},
      {"words past the text and labels past the row do not count", [words: 2, labels: 1],
       [{[0, 2, 0, 0], 2.0}, {[0, 2, 0, 1], 2.0}, {[0, 2, 0, 2], 2.0}] ++
         [{[0, 0, 1, 0], 2.0}, {[0, 0, 1, 1], 2.0}, {[0, 0, 1, 2], 2.0}], []}
    ]

    run = fn scores, kind, settings ->
      opts = Keyword.merge(@opts, Keyword.drop(settings, [:words, :labels]))
      words = Keyword.get(settings, :words, 4)
      labels = Keyword.get(settings, :labels, 2)
      scores = Nx.sigmoid(scores)[0]

      for {first, last, label, _score} <- GLiNER.decode(scores, kind, words, labels, opts),
          do: {first, last, label}
    end

    for {name, settings, entries, expected} <- span_cases do
      result = run.(logits({1, 4, 3, 2}, entries), :span, settings)

      if result != expected,
        do: flunk("#{name}: expected #{inspect(expected)}, got #{inspect(result)}")
    end

    for {name, settings, entries, expected} <- token_cases do
      result = run.(logits({1, 4, 2, 3}, entries), :token, settings)

      if result != expected,
        do: flunk("#{name}: expected #{inspect(expected)}, got #{inspect(result)}")
    end

    # The score of a token span is the lowest of its start, end and inside scores.
    scores = Nx.sigmoid(logits({1, 4, 2, 3}, elem(hd(token_cases), 2)))[0]
    assert [{0, 2, 0, 0.5}] = GLiNER.decode(scores, :token, 4, 2, @opts)
  end

  # The token ids, the words mask, the word count and the byte offsets of the words of
  # gliner 0.2.29, from test/support/gliner_references.py.
  @tag :hub
  test "the inputs of the model match gliner for the Metaspace and the ByteLevel tokenizers" do
    fixtures = "test/fixtures/gliner_inputs.json" |> File.read!() |> JSON.decode!()

    for {source, rows} <- fixtures do
      repo = Repo.load!({:hf, source})
      encoder = GLiNER.encoder(Tokenizer.load!(repo), GLiNER.config!(repo), 512)

      for row <- rows do
        input = GLiNER.encode(encoder, %{text: row["text"], labels: row["labels"]})

        assert {input.ids, input.words_mask, [input.text_length]} ==
                 {row["input_ids"], row["words_mask"], row["text_lengths"]},
               "#{source}: #{row["text"]}"

        assert Enum.map(Tuple.to_list(input.words), &Tuple.to_list/1) == row["words"]
      end
    end
  end
end

# The compile count comes from a global telemetry event, so these tests do not run with
# other tests.
defmodule Transmuter.TextGLiNERModelTest do
  use ExUnit.Case, async: false

  # Reference values come from gliner 0.2.29 with ONNX Runtime, see
  # test/support/gliner_references.py.
  @moduletag :hub
  @moduletag timeout: 900_000
  @moduletag ci_partition: 11

  @opts [
    backend: Transmuter.TestBackend.backend(),
    defn_options: Transmuter.TestBackend.compiler_opts()
  ]

  defp assert_entities(result, expected, delta) do
    assert Enum.map(result.entities, &{&1.start, &1.end, &1.label, &1.phrase}) ==
             Enum.map(expected, &Tuple.delete_at(&1, 4))

    for {entity, {_, _, _, _, score}} <- Enum.zip(result.entities, expected),
        do: assert_in_delta(entity.score, score, delta)
  end

  test "a span model gives the entities of gliner, with the labels of each input" do
    serving =
      Transmuter.Text.zero_shot_entity_recognition(
        {:hf, "onnx-community/gliner_small-v2.1"},
        [variant: :fp16, compile: [batch_size: 2]] ++ @opts
      )

    results =
      Nx.Serving.run(serving, [
        %{
          text:
            "Cristiano Ronaldo dos Santos Aveiro, born 5 February 1985, is a Portuguese footballer who plays for Al Nassr and the Portugal national team.",
          labels: ["person", "date", "organization", "country"]
        },
        %{text: "José Müller visitó São Paulo y Zürich en 2023.", labels: ["person", "city"]},
        %{text: "Cristiano Ronaldo plays for Al Nassr.", labels: ["person", "organization"]}
      ])

    # The export is fp16, so the scores differ from ONNX Runtime by up to about 1e-3.
    expected = [
      [
        {0, 35, "person", "Cristiano Ronaldo dos Santos Aveiro", 0.8181},
        {42, 57, "date", "5 February 1985", 0.959},
        {64, 74, "country", "Portuguese", 0.8608},
        {100, 108, "organization", "Al Nassr", 0.8785},
        {117, 125, "country", "Portugal", 0.9884}
      ],
      [
        {0, 13, "person", "José Müller", 0.9799},
        {22, 32, "city", "São Paulo", 0.9539},
        {35, 42, "city", "Zürich", 0.9525}
      ],
      [
        {0, 17, "person", "Cristiano Ronaldo", 0.9912},
        {28, 36, "organization", "Al Nassr", 0.9689}
      ]
    ]

    for {result, expected} <- Enum.zip(results, expected),
        do: assert_entities(result, expected, 2.0e-3)
  end

  test "a token model gives the entities of gliner" do
    serving =
      Transmuter.Text.zero_shot_entity_recognition(
        {:hf, "knowledgator/gliner-pii-small-v1.0"},
        @opts
      )

    result =
      Nx.Serving.run(serving, %{
        text:
          "John Smith lives at 42 Baker Street in London. His email is john.smith@example.com and his phone number is +44 20 7946 0958.",
        labels: ["person", "address", "email", "phone number"]
      })

    assert_entities(
      result,
      [
        {0, 10, "person", "John Smith", 0.6579},
        {20, 35, "address", "42 Baker Street", 0.7955},
        {60, 82, "email", "john.smith@example.com", 0.753},
        {107, 123, "phone number", "+44 20 7946 0958", 0.9767}
      ],
      1.0e-4
    )
  end

  test "a started serving compiles nothing for a new text of the same sequence length and word bucket" do
    test_pid = self()
    handler = "gliner-compile-#{inspect(test_pid)}"

    :telemetry.attach(
      handler,
      [:transmuter, :segment, :compile, :stop],
      fn _, _, _, pid -> send(pid, :compiled) end,
      test_pid
    )

    on_exit(fn -> :telemetry.detach(handler) end)

    serving =
      Transmuter.Text.zero_shot_entity_recognition(
        {:hf, "knowledgator/gliner-pii-small-v1.0"},
        [compile: [sequence_length: [32, 64]]] ++ @opts
      )

    name = :"gliner_#{System.unique_integer([:positive])}"
    start_supervised!({Nx.Serving, serving: serving, name: name, batch_timeout: 20})
    labels = ["person", "email"]
    run = &Nx.Serving.batched_run(name, %{text: &1, labels: labels})

    # The texts have 6, 16 and 8 words, all in the bucket of 16 words, and fit in 32 tokens.
    assert %{entities: [%{phrase: "Anna Lee"}]} = run.("Anna Lee lives in Boston.")
    assert compiles() > 0

    for text <- [
          "Call Peter Brown at 555 0134 or write to peter@mail.com today.",
          "Maria Garcia works in Madrid and London."
        ] do
      assert %{entities: [%{label: "person"} | _]} = run.(text)
      assert compiles() == 0, "#{text} compiled again"
    end
  end

  defp compiles(count \\ 0) do
    receive do
      :compiled -> compiles(count + 1)
    after
      0 -> count
    end
  end
end
