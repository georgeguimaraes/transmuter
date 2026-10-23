defmodule Transmuter.TextTest do
  use ExUnit.Case, async: true

  # Reference values come from the pipelines of Python transformers and sentence-transformers,
  # see test/support/pipeline_references.py. The first run downloads the models.
  @moduletag :hub
  @moduletag timeout: 900_000
  @moduletag ci_partition: 13

  @opts [
    backend: Transmuter.TestBackend.backend(),
    defn_options: Transmuter.TestBackend.compiler_opts()
  ]

  alias Transmuter.Text.{TextRanking, TextToText}
  alias Transmuter.Tokenizer

  test "the tokenizer pads to the smallest length that fits and truncates to the largest" do
    tokenizer = Tokenizer.load!({:hf, "Xenova/all-MiniLM-L6-v2"}, length: [8, 16])
    long = String.duplicate("word ", 40)

    {inputs, _} = Tokenizer.encode(tokenizer, ["hello world"])
    assert Nx.to_flat_list(inputs["input_ids"]) == [101, 7592, 2088, 102, 0, 0, 0, 0]
    assert Nx.to_flat_list(inputs["attention_mask"]) == [1, 1, 1, 1, 0, 0, 0, 0]

    {inputs, _} =
      Tokenizer.encode(tokenizer, ["hello world", "one two three four five six seven"])

    assert Nx.shape(inputs["input_ids"]) == {2, 16}

    {inputs, _} = Tokenizer.encode(tokenizer, [long])
    assert Nx.shape(inputs["input_ids"]) == {1, 16}
    assert inputs["input_ids"][[0, -1]] |> Nx.to_number() == 102
  end

  @tag :smoke
  test "an embedding takes the pooling and normalization of the base model" do
    # Neither repository has 1_Pooling/config.json: MiniLM uses mean pooling and bge the
    # CLS token, both normalized.
    for {repo, expected} <- [
          {"Xenova/all-MiniLM-L6-v2", [0.1302, -0.0158, -0.0367, 0.058]},
          {"Xenova/bge-small-en-v1.5", [0.0313, -0.0109, 0.0031, 0.0661]}
        ] do
      serving = Transmuter.Text.text_embedding({:hf, repo}, @opts)
      %{embedding: embedding} = Nx.Serving.run(serving, "The cat sat on the mat.")
      assert_all_close(embedding[0..3], expected)
    end
  end

  test "an embedding truncates at max_seq_length of sentence_bert_config.json" do
    # 541 tokens. all-MiniLM-L6-v2 truncates at 256, not at 512 as its tokenizer.
    text = String.duplicate("the quick brown fox jumps over the lazy dog ", 60)
    serving = Transmuter.Text.text_embedding({:hf, "Xenova/all-MiniLM-L6-v2"}, @opts)
    %{embedding: embedding} = Nx.Serving.run(serving, text)
    assert_all_close(embedding[0..3], [0.0529, -0.0371, 0.0714, 0.0049])
  end

  test "a cross-encoder scores text pairs with a sigmoid" do
    serving = Transmuter.Text.text_classification({:hf, "Xenova/ms-marco-MiniLM-L-6-v2"}, @opts)
    query = "How many people live in Berlin?"

    [relevant, other] =
      Nx.Serving.run(serving, [
        {query, "Berlin has a population of 3,520,031 registered inhabitants."},
        {query, "New York City is famous for the Metropolitan Museum of Art."}
      ])

    assert [%{label: "LABEL_0", score: score}] = relevant.predictions
    assert_in_delta score, 0.99984, 1.0e-4
    assert_in_delta hd(other.predictions).score, 1.3065e-5, 1.0e-6
  end

  # The reference is CrossEncoder.rank of sentence-transformers, see
  # test/support/ranking_references.py. config.json has the activation Identity, so the
  # scores are the logits.
  test "text ranking gives the scores of a cross-encoder, from the highest score" do
    serving = Transmuter.Text.text_ranking({:hf, "Xenova/ms-marco-MiniLM-L-6-v2"}, @opts)

    %{results: [first, second]} =
      Nx.Serving.run(serving, %{
        query: "How many people live in Berlin?",
        documents: [
          "New York City is famous for the Metropolitan Museum of Art.",
          "Berlin has a population of 3,520,031 registered inhabitants."
        ]
      })

    assert {first.index, second.index} == {1, 0}
    assert_in_delta first.score, 8.7619, 1.0e-3
    assert_in_delta second.score, -11.2456, 1.0e-3
  end

  # The token ids of the preprocessing of sentence-transformers, see
  # test/support/ranking_references.py. The text has the instruction of the default prompt.
  # Truncation at 128 tokens cuts the long document, and the end of the chat template
  # replaces the last tokens. The tokenizer pads to 128 tokens, and the end of the template
  # does not depend on it. A length that leaves no tokens for the text raises.
  test "text ranking gives the text of a reranker as sentence-transformers" do
    repo = Transmuter.Repo.load!({:hf, "onnx-community/Qwen3-Reranker-0.6B-ONNX"})
    tokenizer = Tokenizer.load!(repo, length: 128)
    ranking = TextRanking.ranking(repo, tokenizer, [])

    # The system message, the instruction, the query and "<Document>:".
    prefix =
      [151_644, 8948, 198, 60_256, 3425, 279, 11_789, 20_027, 279, 8502, 3118, 389, 279] ++
        [11_361, 323, 279, 758, 1235, 3897, 13, 7036, 429, 279, 4226, 646, 1172, 387, 330] ++
        [9693, 1, 476, 330, 2152, 3263, 151_645, 198, 151_644, 872, 198, 27, 641, 1235] ++
        [26_818, 16_246, 264, 3482, 2711, 3239, 11, 17_179, 9760, 46_769, 429, 4226, 279] ++
        [3239, 198, 27, 2859, 26_818, 2585, 1657, 1251, 3887, 304, 19_846, 5267, 75_692] ++
        [26_818]

    suffix = [151_645, 198, 151_644, 77_091, 198, 151_667, 271, 151_668, 271]
    # " the quick brown fox jumps over the lazy dog"
    sentence = [279, 3974, 13_876, 38_835, 34_208, 916, 279, 15_678, 5562]
    long = String.duplicate("the quick brown fox jumps over the lazy dog ", 60)

    for {document, length, expected} <- [
          {"Berlin is the capital of Germany.", nil,
           prefix ++ [19_846, 374, 279, 6722, 315, 9856, 13] ++ suffix},
          {long, 128, prefix ++ Enum.take(Enum.concat(List.duplicate(sentence, 6)), 50) ++ suffix}
        ] do
      text = ranking.sequence.("How many people live in Berlin?", document)
      tokenizer = Tokenizer.configure(tokenizer, length: length)
      {%{"input_ids" => ids}, _} = Tokenizer.encode(tokenizer, [text], ranking.encode_options)
      assert Nx.to_flat_list(ids) == expected
    end

    assert_raise ArgumentError, ~r/the suffix of 9 tokens leaves no tokens/, fn ->
      Tokenizer.encode(
        Tokenizer.configure(tokenizer, length: 9),
        ["text"],
        ranking.encode_options
      )
    end
  end

  test "text ranking raises when the chat template cannot carry a query and a document" do
    assert_raise ArgumentError, ~r/cannot carry a query and a document/, fn ->
      Transmuter.Text.text_ranking({:hf, "onnx-community/Qwen3-0.6B-ONNX"}, @opts)
    end
  end

  # A model without sentence-transformers files takes the pair as chat messages when it has a
  # chat template, as sentence-transformers infers the "message" modality. An empty prompt is
  # no prompt, and a template without the roles raises.
  @tag :tmp_dir
  test "text ranking gives the pair to the chat template of a classifier", %{tmp_dir: dir} do
    for file <- ~w(config.json tokenizer.json tokenizer_config.json) do
      {:hf, "Xenova/ms-marco-MiniLM-L-6-v2"}
      |> Transmuter.Hub.fetch!(file)
      |> File.cp!(Path.join(dir, file))
    end

    ranking = fn template, opts ->
      File.write!(Path.join(dir, "chat_template.jinja"), template)
      repo = Transmuter.Repo.load!(dir)
      TextRanking.ranking(repo, Tokenizer.load!(repo), opts)
    end

    roles = "{% for m in messages %}{{ m.role }}: {{ m.content }}\n{% endfor %}"

    for {prompt, expected} <- [
          {"Rank.", "system: Rank.\nquery: q\ndocument: d\n"},
          {"", "query: q\ndocument: d\n"}
        ] do
      ranking = ranking.(roles, prompt: prompt)
      assert ranking.sequence.("q", "d") == expected
      assert ranking.encode_options[:add_special_tokens] == false
    end

    assert_raise ArgumentError, ~r/the query and document content does not reach/, fn ->
      ranking.(
        "{% for m in messages %}{% if m.role == 'user' %}{{ m.content }}{% endif %}{% endfor %}",
        []
      )
    end
  end

  test "word aggregation gives each word one label before grouping" do
    text =
      "Hugging Face Inc. is a company based in New York City. Its headquarters are in DUMBO, São Paulo."

    run = fn aggregation ->
      {:hf, "Xenova/bert-base-NER"}
      |> Transmuter.Text.token_classification([aggregation: aggregation] ++ @opts)
      |> Nx.Serving.run(text)
      |> Map.fetch!(:entities)
    end

    spans = [
      {0, 16, "ORG", "Hugging Face Inc"},
      {40, 53, "LOC", "New York City"},
      {79, 84, "LOC", "DUMBO"},
      {86, 96, "LOC", "São Paulo"}
    ]

    for {aggregation, scores} <- [
          same: [0.99161, 0.99944, 0.96740, 0.99926],
          word_first: [0.99722, 0.99944, 0.98146, 0.99926]
        ] do
      entities = run.(aggregation)
      assert Enum.map(entities, &{&1.start, &1.end, &1.label, &1.phrase}) == spans

      for {entity, score} <- Enum.zip(entities, scores),
          do: assert_in_delta(entity.score, score, 1.0e-4)
    end
  end

  test "an unknown token keeps its text in the entity" do
    serving =
      Transmuter.Text.token_classification(
        {:hf, "Xenova/bert-base-NER"},
        [aggregation: :same] ++ @opts
      )

    %{entities: [person | _]} =
      Nx.Serving.run(serving, "My friend Angela☃Merkel lives in Berlin.")

    assert {person.label, person.phrase} == {"PER", "Angela☃Merkel"}
  end

  test "zero-shot classification scores the labels together, or each label alone" do
    text = "I have a problem with my iphone that needs to be resolved asap!!"
    labels = ["urgent", "not urgent", "phone", "tablet", "computer"]

    run = fn labels, opts ->
      {:hf, "Xenova/distilbert-base-uncased-mnli"}
      |> Transmuter.Text.zero_shot_classification(labels, opts ++ @opts)
      |> Nx.Serving.run(text)
      |> Map.fetch!(:predictions)
      |> Enum.map(&{&1.label, &1.score})
    end

    # One label gets the score of :multi_label, because a softmax over one label is always 1.
    for {labels, opts, expected} <- [
          {labels, [multi_label: false],
           [
             {"phone", 0.80463},
             {"computer", 0.13484},
             {"not urgent", 0.02647},
             {"urgent", 0.02374},
             {"tablet", 0.01033}
           ]},
          {labels, [multi_label: true],
           [
             {"phone", 0.80205},
             {"urgent", 0.39549},
             {"computer", 0.20127},
             {"not urgent", 0.06874},
             {"tablet", 0.00023}
           ]},
          {["urgent"], [], [{"urgent", 0.39549}]}
        ] do
      predictions = run.(labels, opts)
      assert Enum.map(predictions, &elem(&1, 0)) == Enum.map(expected, &elem(&1, 0))

      for {{_, score}, {_, expected}} <- Enum.zip(predictions, expected),
          do: assert_in_delta(score, expected, 1.0e-4)
    end
  end

  test "question answering finds the answer at the end of the context and in a long context" do
    serving =
      Transmuter.Text.question_answering(
        {:hf, "Xenova/distilbert-base-cased-distilled-squad"},
        @opts
      )

    # The long context has 1281 tokens. The answer is in the third of five windows, and
    # candidates in the subwords of "Zanzibar" add their scores. The byte offsets count the
    # 2 bytes of each "é" before the answer.
    filler =
      String.duplicate(
        "The café on the corner sells bread, coffee and small cakes to the people of the town every morning. ",
        30
      )

    results =
      Nx.Serving.run(serving, [
        %{
          question: "Where is the Eiffel Tower?",
          context: "The Eiffel Tower is a wrought-iron lattice tower in Paris"
        },
        %{
          question: "What is the code word?",
          context: filler <> "The code word for the meeting is Zanzibar. " <> filler
        }
      ])

    for {%{results: [answer]}, {text, start, stop, score}} <-
          Enum.zip(results, [{"Paris", 52, 57, 0.966489}, {"Zanzibar", 3063, 3071, 0.967573}]) do
      assert {answer.text, answer.start, answer.end} == {text, start, stop}
      assert_in_delta answer.score, score, 1.0e-4
    end
  end

  test "fill-mask raises when truncation removes the mask" do
    serving =
      Transmuter.Text.fill_mask(
        {:hf, "Xenova/distilroberta-base"},
        [compile: [sequence_length: 32]] ++ @opts
      )

    assert_raise ArgumentError, ~r/one mask token in the encoded text, got 0/, fn ->
      Nx.Serving.run(serving, String.duplicate("word ", 60) <> "Paris is the [MASK] of France.")
    end
  end

  test "fill-mask uses the mask token of the tokenizer" do
    serving = Transmuter.Text.fill_mask({:hf, "Xenova/distilroberta-base"}, @opts)

    %{predictions: [first, second | _]} =
      Nx.Serving.run(serving, "Paris is the [MASK] of France.")

    assert first.token == " capital"
    assert second.token == " birthplace"
    assert_in_delta first.score, 0.67902, 1.0e-4
  end

  test "the chat template renders like apply_chat_template" do
    repo = Transmuter.Repo.load!({:hf, "onnx-community/Qwen3-0.6B-ONNX"})
    messages = [%{role: "system", content: "Be brief."}, %{role: "user", content: "Hi"}]

    assert Transmuter.ChatTemplate.render(repo, messages,
             template_options: [enable_thinking: false]
           ) ==
             "<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n" <>
               "<|im_start|>assistant\n<think>\n\n</think>\n\n"
  end

  test "the chat template continues a final assistant message and runs generation blocks" do
    repo = Transmuter.Repo.load!({:hf, "HuggingFaceTB/SmolLM2-135M-Instruct"})

    messages = [
      %{role: "user", content: "Write a haiku"},
      %{role: "assistant", content: "Soft rain on the"}
    ]

    assert Transmuter.ChatTemplate.render(repo, messages, continue_final_message: true) =~
             ~r/<\|im_start\|>assistant\nSoft rain on the$/

    # SmolLM3 marks assistant messages with {% generation %}, a tag of transformers.
    repo = Transmuter.Repo.load!({:hf, "HuggingFaceTB/SmolLM3-3B"})
    messages = [%{role: "user", content: "Hi"}, %{role: "assistant", content: "Hello!"}]

    assert String.ends_with?(
             Transmuter.ChatTemplate.render(repo, messages, add_generation_prompt: false),
             "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\nHello!<|im_end|>\n"
           )
  end

  test "Gemma 3 generates with a cache of exact length" do
    # Its GroupQueryAttention counts the cache rows from the length of the attention mask.
    serving =
      Transmuter.Text.generation(
        {:hf, "onnx-community/gemma-3-270m-it-ONNX"},
        [strategy: :greedy, max_new_tokens: 12] ++ @opts
      )

    chat = %{
      messages: [
        %{role: "user", content: "Write a long story about a fox who lives near a river."}
      ]
    }

    assert %{results: [%{text: "**The Whispering of the Willow**\n\nThe biting wind"}]} =
             Nx.Serving.run(serving, chat)
  end

  @tag :smoke
  @tag ci_partition: 14
  test "greedy generation matches transformers in each mode of the KV cache" do
    chat = %{messages: [%{role: "user", content: "What is the capital of France?"}]}

    # GroupQueryAttention, a cache with position_ids, and gpt2 without position_ids.
    for {repo, input, expected} <- [
          {"HuggingFaceTB/SmolLM2-135M-Instruct", chat, "The capital of France is Paris."},
          {"onnx-community/SmolLM2-135M-Instruct-ONNX", chat, "The capital of France is Paris."},
          {"Xenova/gpt2", "The history of the printing press begins",
           " with the printing press. The first printing press was the first"}
        ] do
      serving =
        Transmuter.Text.generation(
          {:hf, repo},
          [variant: :fp32, strategy: :greedy, max_new_tokens: 12] ++ @opts
        )

      assert %{results: [%{text: ^expected}]} = Nx.Serving.run(serving, input)
    end
  end

  # The n-grams include the prompt, and min_new_tokens counts only the new tokens. The empty
  # prompt is the bos token of gpt2, a prompt of one token, so the forced bos token is the
  # first new token.
  @tag ci_partition: 14
  test "the logits processors that read the tokens of a row change the text as in transformers" do
    gpt2 = {:hf, "Xenova/gpt2"}
    chat = %{messages: [%{role: "user", content: "What is the capital of France?"}]}

    for {repo, opts, input, expected} <- [
          {gpt2, [no_repeat_ngram_size: 2, max_new_tokens: 12],
           "The history of the printing press begins",
           " with the invention of printing presses in the early 19th century"},
          {gpt2, [forced_bos_token_id: 464, max_new_tokens: 8], "",
           "The first time I saw the new version"},
          {{:hf, "HuggingFaceTB/SmolLM2-135M-Instruct"}, [min_new_tokens: 12, max_new_tokens: 16],
           chat,
           "The capital of France is Paris. It is a city known for its historical landmarks"}
        ] do
      serving =
        Transmuter.Text.generation(repo, [variant: :fp32, strategy: :greedy] ++ opts ++ @opts)

      assert %{results: [%{text: ^expected}]} = Nx.Serving.run(serving, input)
    end
  end

  @tag ci_partition: 14
  test "a half-precision export gives the f32 text in native f16" do
    # The fp16 export writes RMSNorm as Pow, ReduceMean, Add and Sqrt, which overflow in
    # native f16. Transmuter.Fusion replaces them with SimplifiedLayerNormalization.
    serving =
      Transmuter.Text.generation(
        {:hf, "onnx-community/SmolLM2-135M-Instruct-ONNX"},
        [variant: :fp16, strategy: :greedy, max_new_tokens: 12] ++ @opts
      )

    chat = %{messages: [%{role: "user", content: "What is the capital of France?"}]}

    assert %{results: [%{text: "The capital of France is Paris."}]} =
             Nx.Serving.run(serving, chat)
  end

  @tag ci_partition: 14
  test "sampling with the same seed gives the same text" do
    serving =
      Transmuter.Text.generation(
        {:hf, "HuggingFaceTB/SmolLM2-135M-Instruct"},
        [variant: :fp32, strategy: :sampling, max_new_tokens: 12] ++ @opts
      )

    text = fn seed ->
      %{results: [%{text: text}]} =
        Nx.Serving.run(serving, %{text: "Once upon a time", seed: seed})

      text
    end

    assert text.(1) == text.(1)
    assert text.(1) != text.(2)
  end

  @tag ci_partition: 14
  test "streamed chunks join to the text, and characters of several byte tokens stay whole" do
    # Each of 景 and 谊 is three byte-level tokens of SmolLM2.
    repo = {:hf, "HuggingFaceTB/SmolLM2-135M-Instruct"}
    opts = [variant: :fp32, strategy: :greedy, max_new_tokens: 14] ++ @opts

    chat = %{
      messages: [%{role: "user", content: "Write the word 'hello' in Chinese and Japanese."}]
    }

    %{results: [%{text: text, token_summary: summary}]} =
      Nx.Serving.run(Transmuter.Text.generation(repo, opts), chat)

    events =
      repo
      |> Transmuter.Text.generation([stream: true, stream_done: true] ++ opts)
      |> Nx.Serving.run(chat)
      |> Enum.to_list()

    {chunks, [{:done, %{token_summary: streamed_summary}}]} = Enum.split(events, -1)
    assert text =~ "景谊"
    assert Enum.join(chunks) == text
    refute Enum.any?(chunks, &String.contains?(&1, "�"))
    assert streamed_summary == summary
  end

  @tag ci_partition: 14
  test "concurrent streams of a started serving get the tokens of their own rows" do
    serving =
      Transmuter.Text.generation(
        {:hf, "HuggingFaceTB/SmolLM2-135M-Instruct"},
        [
          variant: :fp32,
          strategy: :greedy,
          max_new_tokens: 12,
          compile: [batch_size: 2],
          stream: true
        ] ++ @opts
      )

    start_supervised!({Nx.Serving, serving: serving, name: __MODULE__.Stream, batch_timeout: 100})

    texts =
      ["What is the capital of France?", "Hi"]
      |> Enum.map(&%{messages: [%{role: "user", content: &1}]})
      |> Task.async_stream(&Enum.join(Nx.Serving.batched_run(__MODULE__.Stream, &1)),
        timeout: 60_000
      )
      |> Enum.map(fn {:ok, text} -> text end)

    assert texts == ["The capital of France is Paris.", "Hello! How can I help you today?"]
  end

  test "greedy text-to-text matches transformers for T5 and Marian" do
    sentence =
      "Transmuter runs ONNX models on any Nx backend, with no code for each architecture."

    # Marian encodes with source.spm. The vocabulary of tokenizer.json gives "▁Trans", "mut",
    # "er" for "Transmuter", and then the translation starts with "Transmuter läuft". The T5
    # inputs run as one batch. Marian has 4 beams in its generation_config.json, so the test
    # gives num_beams: 1.
    for {repo, inputs, expected} <- [
          {"Xenova/t5-small",
           [
             "translate English to German: The house is wonderful.",
             "translate English to German: " <> sentence
           ],
           [
             "Das Haus ist wunderbar.",
             "Transmuter läuft ONNX-Modelle auf jedem Nx-Backend, ohne Code für jede Architektur."
           ]},
          {"Xenova/opus-mt-en-de", [sentence],
           [
             "Transmuter führt ONNX-Modelle auf jedem Nx-Backend aus, ohne Code für jede Architektur."
           ]}
        ] do
      serving =
        Transmuter.Text.text_to_text(
          {:hf, repo},
          [strategy: :greedy, num_beams: 1, max_new_tokens: 64, compile: [batch_size: 2]] ++
            @opts
        )

      results = Nx.Serving.run(serving, inputs)
      assert Enum.map(results, &hd(&1.results).text) == expected
    end
  end

  # The test downloads only the tokenizer and the configuration files. M2M100 has only the
  # slow tokenizer in Python, and its tokenizer.json has merges of pieces that are not in
  # its vocabulary. A bare language code, such as "fr", is an ordinary piece of the
  # vocabulary, so it raises.
  test "the language tokens of NLLB and M2M100 give the ids and the forced bos token of transformers" do
    text = "La vie est belle et le soleil brille sur la ville."

    for {repo, source, target, ids, forced} <- [
          {"Xenova/nllb-200-distilled-600M", "fra_Latn", "deu_Latn",
           [
             256_057,
             1034,
             6651,
             613,
             48_970,
             410,
             96,
             229_372,
             1212,
             3305,
             2562,
             82,
             11_959,
             248_075,
             2
           ], 256_042},
          {"Xenova/m2m100_418M", "__fr__", "__de__",
           [128_028, 343, 7315, 769, 26_181, 118, 86, 102_928, 951, 1479, 954, 14, 8919, 5, 2],
           128_020}
        ] do
      repo = Transmuter.Repo.load!({:hf, repo})
      base = Tokenizer.load!(repo)
      opts = [source_language_token: source, target_language_token: target]
      {tokenizer, opts} = TextToText.languages(repo, base, opts)
      {%{"input_ids" => input_ids}, _encodings} = Tokenizer.encode(tokenizer, [text])
      assert Nx.to_flat_list(input_ids) == ids
      assert opts[:forced_bos_token_id] == forced

      assert_raise ArgumentError, ~r/a special token of the tokenizer/, fn ->
        TextToText.languages(repo, base, target_language_token: "fr")
      end
    end
  end

  defp assert_all_close(tensor, expected) do
    expected = Nx.tensor(expected)

    assert Nx.all_close(Nx.backend_transfer(tensor, Nx.BinaryBackend), expected, atol: 1.0e-4) ==
             Nx.tensor(1, type: :u8),
           "expected #{inspect(Nx.to_list(expected))}, got #{inspect(Nx.to_list(tensor))}"
  end
end

# --only ci_partition:N in CI would also include :large_model tests of TextTest, because an
# included tag wins over an excluded one. So these tests are in a module without a
# :ci_partition tag, and CI leaves them out.
defmodule Transmuter.TextLargeModelTest do
  use ExUnit.Case, async: true

  @moduletag :hub
  @moduletag timeout: 900_000

  @opts [
    backend: Transmuter.TestBackend.backend(),
    defn_options: Transmuter.TestBackend.compiler_opts()
  ]

  # The reference is the greedy translation pipeline of transformers with src_lang and
  # tgt_lang, see test/support/pipeline_references.py. The configuration of m2m100_418M has
  # num_beams: 5, so the test gives num_beams: 1.
  @tag :large_model
  test "NLLB and M2M100 translate as the translation pipeline of transformers" do
    sentence =
      "Transmuter runs ONNX models on any Nx backend, with no code for each architecture."

    french = "La vie est belle et le soleil brille sur la ville."

    for {repo, variant, cases} <- [
          {"Xenova/nllb-200-distilled-600M", :fp32,
           [
             {"eng_Latn", "fra_Latn", sentence,
              "Le transmutateur exécute des modèles ONNX sur n'importe quel backend Nx, sans code pour chaque architecture."},
             {"fra_Latn", "deu_Latn", french,
              "Das Leben ist schön und die Sonne scheint auf der Stadt."}
           ]},
          {"Xenova/m2m100_418M", :fp32,
           [
             {"__en__", "__fr__", sentence,
              "Le transmutateur fonctionne avec des modèles ONNX sur n'importe quel backend Nx, sans code pour chaque architecture."},
             {"__fr__", "__de__", french,
              "Das Leben ist schön und die Sonne schlägt auf die Stadt."}
           ]}
        ],
        {source, target, text, expected} <- cases do
      serving =
        Transmuter.Text.text_to_text(
          {:hf, repo},
          [
            variant: variant,
            strategy: :greedy,
            num_beams: 1,
            source_language_token: source,
            target_language_token: target
          ] ++ @opts
        )

      assert %{results: [%{text: ^expected}]} = Nx.Serving.run(serving, text)
    end
  end

  # The graph gives sentence_embedding, the result of the whole sentence-transformers model,
  # so the serving does not read the gated base model google/embeddinggemma-300m. The fp32
  # files are 1.2 GB. The reference is ONNX Runtime on the export, without a prompt. The
  # export differs from sentence-transformers, see test/support/pipeline_references.py. The
  # 543 tokens need the length 1024.
  @tag :large_model
  test "EmbeddingGemma gives the embedding of sentence-transformers" do
    serving =
      Transmuter.Text.text_embedding(
        {:hf, "onnx-community/embeddinggemma-300m-ONNX"},
        [compile: [sequence_length: [16, 1024]]] ++ @opts
      )

    long = String.duplicate("the quick brown fox jumps over the lazy dog ", 60)

    for {text, expected} <- [
          {"The cat sat on the mat.", [-0.131, 0.0266, 0.0, 0.0388]},
          {long, [-0.0525, 0.0364, 0.0458, -0.0258]}
        ] do
      %{embedding: embedding} = Nx.Serving.run(serving, text)
      assert Nx.shape(embedding) == {768}
      assert_all_close(embedding[0..3], expected)
    end
  end

  # The scores are those of ONNX Runtime on the q4 export with the token ids of
  # sentence-transformers, and the order is that of sentence-transformers on PyTorch, see
  # test/support/ranking_references.py. The texts have at most 128 tokens, so truncation cuts
  # the long document. The model file is 995 MB. On the EXLA CPU, the scores were within
  # 1.1e-4 of ONNX Runtime, the rounding of the reference.
  @tag :large_model
  test "Qwen3-Reranker ranks documents as sentence-transformers" do
    serving =
      Transmuter.Text.text_ranking(
        {:hf, "onnx-community/Qwen3-Reranker-0.6B-ONNX"},
        [variant: :q4, compile: [batch_size: 2, sequence_length: 128]] ++ @opts
      )

    %{results: results} =
      Nx.Serving.run(serving, %{
        query: "How many people live in Berlin?",
        documents: [
          "New York City is famous for the Metropolitan Museum of Art.",
          "Berlin has a population of 3,520,031 registered inhabitants.",
          "Berlin is the capital of Germany.",
          String.duplicate("the quick brown fox jumps over the lazy dog ", 60)
        ]
      })

    assert Enum.map(results, & &1.index) == [1, 2, 0, 3]

    for {result, expected} <- Enum.zip(results, [6.5081, -4.5089, -10.0807, -10.8879]),
        do: assert_in_delta(result.score, expected, 1.0e-3)
  end

  defp assert_all_close(tensor, expected) do
    expected = Nx.tensor(expected)

    assert Nx.all_close(Nx.backend_transfer(tensor, Nx.BinaryBackend), expected, atol: 1.0e-4) ==
             Nx.tensor(1, type: :u8),
           "expected #{inspect(Nx.to_list(expected))}, got #{inspect(Nx.to_list(tensor))}"
  end
end

# The compile count comes from a global telemetry event, so these tests do not run with
# other tests.
defmodule Transmuter.TextServingTest do
  use ExUnit.Case, async: false

  @moduletag :hub
  @moduletag timeout: 900_000
  @moduletag ci_partition: 6

  @opts [
    backend: Transmuter.TestBackend.backend(),
    defn_options: Transmuter.TestBackend.compiler_opts()
  ]

  test "a started serving compiles each length once and batches inputs of any length" do
    test_pid = self()
    handler = "compile-#{inspect(test_pid)}"

    :telemetry.attach(
      handler,
      [:transmuter, :compile, :stop],
      fn _, _, _, pid -> send(pid, :compiled) end,
      test_pid
    )

    on_exit(fn -> :telemetry.detach(handler) end)

    serving =
      Transmuter.Text.text_embedding(
        {:hf, "Xenova/all-MiniLM-L6-v2"},
        [compile: [batch_size: 4, sequence_length: [16, 64]]] ++ @opts
      )

    name = :"embedding_#{System.unique_integer([:positive])}"
    start_supervised!({Nx.Serving, serving: serving, name: name, batch_timeout: 20})

    texts = for i <- 1..8, do: String.duplicate("word #{i} ", i * 4)

    batched =
      texts
      |> Task.async_stream(&Nx.Serving.batched_run(name, &1))
      |> Enum.map(fn {:ok, %{embedding: embedding}} -> embedding end)

    assert compiles() == 2

    for {text, embedding} <- Enum.zip(texts, batched) do
      assert_all_close(embedding, Nx.Serving.run(serving, text).embedding)
    end
  end

  # The client makes the rows of a list of inputs, so a prompt that is too long raises in the
  # caller.
  test "a prompt longer than generation takes raises in the caller" do
    serving =
      Transmuter.Text.generation(
        {:hf, "Xenova/gpt2"},
        [variant: :fp32, max_new_tokens: 1] ++ @opts
      )

    assert_raise ArgumentError, ~r/generation takes up to 1048576/, fn ->
      Nx.Serving.run(serving, String.duplicate(" a", 1_100_000))
    end
  end

  defp compiles(count \\ 0) do
    receive do
      :compiled -> compiles(count + 1)
    after
      0 -> count
    end
  end

  defp assert_all_close(tensor, expected) do
    expected = Nx.tensor(expected)

    assert Nx.all_close(Nx.backend_transfer(tensor, Nx.BinaryBackend), expected, atol: 1.0e-4) ==
             Nx.tensor(1, type: :u8),
           "expected #{inspect(Nx.to_list(expected))}, got #{inspect(Nx.to_list(tensor))}"
  end
end

# The batch test is one of the slowest tests on EXLA in CI, so it is a module of its own, in
# its own CI partition.
defmodule Transmuter.TextBatchTest do
  use ExUnit.Case, async: true

  @moduletag :hub
  @moduletag timeout: 900_000
  @moduletag ci_partition: 8

  @opts [
    backend: Transmuter.TestBackend.backend(),
    defn_options: Transmuter.TestBackend.compiler_opts()
  ]

  test "a batch gives the text of each prompt alone in each mode of the KV cache" do
    questions = [
      "What is the capital of France?",
      "Hi",
      "What is the capital of Spain?",
      "Write a long poem about the sea and the wind, in three verses."
    ]

    chats = for q <- questions, do: %{messages: [%{role: "user", content: q}]}

    # The fourth input runs in a partial batch with two padding rows. gpt2 runs a group for
    # each prompt length, and the questions about France and Spain have the same length.
    for {repo, inputs} <- [
          {"HuggingFaceTB/SmolLM2-135M-Instruct", chats},
          {"onnx-community/SmolLM2-135M-Instruct-ONNX", chats},
          {"Xenova/gpt2", questions}
        ] do
      opts = [variant: :fp32, strategy: :greedy, max_new_tokens: 12] ++ @opts
      single = Transmuter.Text.generation({:hf, repo}, opts)
      batched = Transmuter.Text.generation({:hf, repo}, [compile: [batch_size: 3]] ++ opts)
      alone = for input <- inputs, do: Nx.Serving.run(single, input)
      assert Nx.Serving.run(batched, inputs) == alone
    end
  end
end

# Text tasks on the exports of decoders, whose graphs take a KV cache. The models are
# larger than those of Transmuter.TextTest.
defmodule Transmuter.TextDecoderTest do
  use ExUnit.Case, async: true

  @moduletag :hub
  @moduletag timeout: 900_000
  @moduletag ci_partition: 6

  @opts [
    backend: Transmuter.TestBackend.backend(),
    defn_options: Transmuter.TestBackend.compiler_opts()
  ]

  # The reference is ONNX Runtime on the q4 export with each text alone, see
  # test/support/pipeline_references.py. In the batch, the first text has padding tokens.
  test "a decoder with a cache gives the embedding of each text alone" do
    serving =
      Transmuter.Text.text_embedding(
        {:hf, "onnx-community/Qwen3-Embedding-0.6B-ONNX"},
        [variant: :q4, compile: [batch_size: 2, sequence_length: 32]] ++ @opts
      )

    [short, long] =
      Nx.Serving.run(serving, [
        "The cat sat on the mat.",
        "Transmuter runs ONNX models on any Nx backend, with no code for each architecture."
      ])

    assert_all_close(short.embedding[0..3], [0.0084, -0.0335, -0.002, -0.0375])
    assert_all_close(long.embedding[0..3], [-0.0196, -0.0228, -0.0071, -0.1178])
  end

  defp assert_all_close(tensor, expected) do
    expected = Nx.tensor(expected)

    assert Nx.all_close(Nx.backend_transfer(tensor, Nx.BinaryBackend), expected, atol: 1.0e-4) ==
             Nx.tensor(1, type: :u8),
           "expected #{inspect(Nx.to_list(expected))}, got #{inspect(Nx.to_list(tensor))}"
  end
end
