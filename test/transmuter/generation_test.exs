defmodule Transmuter.GenerationTest do
  use ExUnit.Case, async: true

  # Beam search. The references come from the generate function of Python transformers with
  # num_beams, see the beam search section of test/support/pipeline_references.py. Each case
  # gives another text than greedy selection, or another text than another row of its table.
  @moduletag :hub
  @moduletag timeout: 900_000
  @moduletag ci_partition: 11

  @opts [
    backend: Transmuter.TestBackend.backend(),
    defn_options: Transmuter.TestBackend.compiler_opts()
  ]

  alias Transmuter.{Audio, Generation, Text}

  # The best beam of "What is 2 + 2?" ends with eos at the eighth token while the other
  # beams still run. Greedy selection gives "2 + 2 is a classic arithmetic operation that
  # can be" and " with the printing press. The first printing press was the first".
  @tag ci_partition: 10
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

  @tag ci_partition: 9
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
  @tag ci_partition: 10
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

  # no_repeat_ngram_size reads the tokens of each row on the device, where each step writes
  # its new token after the tokens of the row. In a batch whose prompts have different
  # lengths, each row gets its tokens alone, also with beams, whose tokens move with their
  # rows, and no row repeats a 3-gram of its prompt and new tokens.
  @tag ci_partition: 10
  test "no_repeat_ngram_size gives each row of a batch its tokens alone, also with beams" do
    opts = [outputs: &Generation.outputs/1, variant: :fp32] ++ @opts
    source = {:hf, "onnx-community/SmolLM2-135M-Instruct-ONNX"}
    {repo, model, _tokenizer} = Transmuter.Serving.load!(source, opts, :fp32)
    prompts = [[504, 3575, 282, 4649, 314, 504, 3575], [504, 2365], [42, 43, 44, 45, 46]]

    for num_beams <- [1, 2] do
      ctx =
        Generation.context(repo, model,
          no_repeat_ngram_size: 3,
          max_new_tokens: 24,
          num_beams: num_beams
        )

      batch = Generation.compile(ctx, 3 * num_beams, @opts[:defn_options])
      single = Generation.compile(ctx, num_beams, @opts[:defn_options])
      tokens = Generation.generate(batch, prompts, [0, 0, 0])

      assert tokens == Enum.flat_map(prompts, &Generation.generate(single, [&1], [0]))

      for {prompt, new} <- Enum.zip(prompts, tokens) do
        ngrams = Enum.chunk_every(prompt ++ Enum.map(new, &hd/1), 3, 1, :discard)
        assert Enum.uniq(ngrams) == ngrams
      end
    end

    # A row of more than 256 tokens does not fit the first places of the tokens, so they grow
    # with the buffers of the cache. min_new_tokens keeps the row from ending before that.
    ctx =
      Generation.context(repo, model,
        no_repeat_ngram_size: 3,
        max_new_tokens: 260,
        min_new_tokens: 260
      )

    [new] =
      Generation.generate(Generation.compile(ctx, 1, @opts[:defn_options]), [hd(prompts)], [0])

    ngrams = Enum.chunk_every(hd(prompts) ++ Enum.map(new, &hd/1), 3, 1, :discard)

    assert length(new) == 260
    assert Enum.uniq(ngrams) == ngrams
  end

  # The log-probabilities of the tokens, from the token log-probabilities sections of
  # test/support/pipeline_references.py. ONNX Runtime on the same exports gives each value
  # within 4.4e-5 for gpt2 and 1.2e-4 for whisper-tiny.en. compute_transition_scores of the
  # PyTorch models with normalize_logits=True gives the same tokens, and values up to 2.4e-4
  # (gpt2) and 4.6e-4 (whisper-tiny.en) from those of Transmuter. These differences are
  # already in the first token and are the same for greedy selection and beams, which select
  # in different functions, so they come from the logits. The log-softmax of EXLA is within
  # 3e-6 of f64. Python also gives the value of the eos token, which the text does not have.
  # Sampling takes the tokens of Transmuter on EXLA with seed 7, and Python gives their values
  # after the temperature and top-k. A stream gives the values of all its tokens in the done
  # event, so it needs stream_done: true.
  @tag ci_partition: 10
  test "token_logprobs match transformers for greedy selection, beams and sampling" do
    prompt = "The history of the printing press begins"
    gpt2 = [variant: :fp32, max_new_tokens: 8, token_logprobs: true] ++ @opts

    greedy =
      {[-0.722505, -0.811622, -2.300688, -0.463317, -1.118775, -1.638226, -1.721262, -0.92038],
       [-0.722308, -0.811518, -2.300612, -0.463239, -1.118658, -1.638176, -1.721224, -0.920297]}

    for {opts, text, expected} <- [
          {[], " with the printing press. The first printing", greedy},
          {[num_beams: 2], " with the invention of the printing press in",
           {[-0.722505, -0.811622, -2.490965, -0.074146, -0.631518, -0.865999, -0.023429] ++
              [-1.084662],
            [-0.722321, -0.811528, -2.490889, -0.074013, -0.631451, -0.865934, -0.023328] ++
              [-1.084462]}},
          {[strategy: :sampling, temperature: 0.7, top_k: 5, seed: 7],
           " with the first printing presses.\n\n",
           {[-0.28989, -0.083239, -1.974234, -0.033321, -2.342517, -0.945498, -1.874403] ++
              [-0.000142],
            [-0.289905, -0.083239, -1.974146, -0.033321, -2.342534, -0.945499, -1.874423] ++
              [-0.000142]}}
        ] do
      serving = Text.generation({:hf, "Xenova/gpt2"}, opts ++ gpt2)

      assert %{results: [%{text: ^text, token_logprobs: logprobs}]} =
               Nx.Serving.run(serving, prompt)

      assert_logprobs(logprobs, expected)
    end

    serving = Text.generation({:hf, "Xenova/gpt2"}, [stream: true, stream_done: true] ++ gpt2)
    assert {:done, %{token_logprobs: logprobs}} = serving |> Nx.Serving.run(prompt) |> Enum.at(-1)
    assert_logprobs(logprobs, greedy)

    assert_raise ArgumentError, ~r/needs stream_done: true/, fn ->
      Text.generation({:hf, "Xenova/gpt2"}, [stream: true] ++ gpt2)
    end

    audio = Nx.from_binary(File.read!("test/fixtures/librispeech_0.f32"), :f32)

    expected =
      {[-0.075897, -0.032374, -0.283304, -0.085364, -0.007146, -0.008866, -0.005784] ++
         [-0.221374, -0.002404, -0.003975, -0.262812, -0.069947, -0.599032, -0.00657] ++
         [-0.002136, -0.270061, -0.002165, -0.003492, -0.001052, -0.026898, -0.464047] ++
         [-0.125146],
       [-0.075686, -0.03204, -0.28328, -0.085126, -0.006954, -0.008703, -0.005566] ++
         [-0.221347, -0.002253, -0.003812, -0.262625, -0.069761, -0.598903, -0.006324] ++
         [-0.002064, -0.26986, -0.002081, -0.00336, -0.001017, -0.026686, -0.463964] ++
         [-0.124798]}

    for num_beams <- [1, 2] do
      opts = [num_beams: num_beams, token_logprobs: true] ++ @opts
      serving = Audio.speech_to_text({:hf, "Xenova/whisper-tiny.en"}, opts)
      assert %{chunks: [%{token_logprobs: logprobs}]} = Nx.Serving.run(serving, audio)
      assert_logprobs(logprobs, expected, 2.0e-4)
    end
  end

  # Within `tolerance` of ONNX Runtime and 1e-3 of PyTorch.
  defp assert_logprobs(logprobs, {onnxruntime, pytorch}, tolerance \\ 1.0e-4) do
    for {expected, tolerance} <- [{onnxruntime, tolerance}, {pytorch, 1.0e-3}] do
      assert length(logprobs) == length(expected)
      assert logprobs |> Enum.zip_with(expected, &abs(&1 - &2)) |> Enum.max() < tolerance
    end
  end

  # The branch with cache of the merged Whisper decoder has no causal mask, so a step of several
  # tokens there gives wrong logits, also in ONNX Runtime. The decoder prompts have 7, 10 and 12
  # tokens, so parts of power-of-two lengths would run 2 or 4 of them on that branch. The
  # references are ONNX Runtime with the prompt as one step without cache and each new token as
  # one step with cache, greedy and without logits processors, see the speech-to-text section of
  # test/support/pipeline_references.py. The log-probabilities show the wrong logits, also when
  # the tokens do not change.
  test "a merged decoder runs its prompt as one step on the branch without cache" do
    {repo, encoder, decoder} =
      Transmuter.Serving.load_encoder_decoder!({:hf, "Xenova/whisper-tiny.en"}, @opts)

    featurizer = Audio.SpeechToText.featurizer(repo)
    audio = Nx.from_binary(File.read!("test/fixtures/librispeech_0.f32"), :f32)

    audio =
      audio |> Nx.pad(0.0, [{0, featurizer.n_samples - Nx.size(audio), 0}]) |> Nx.new_axis(0)

    features =
      Audio.SpeechToText.features(
        audio,
        featurizer.filters,
        featurizer.window,
        featurizer.hop_length
      )

    encode = Transmuter.compile(encoder, %{"input_features" => features}, @opts[:defn_options])
    %{"last_hidden_state" => hidden} = encode.(%{"input_features" => features})

    opts = [max_new_tokens: 12, token_logprobs: true]

    ctx =
      Generation.context(%{repo | generation_config: %{}}, decoder, opts, [
        "encoder_hidden_states"
      ])

    ctx = Generation.compile(ctx, 2, @opts[:defn_options])
    hidden = Nx.concatenate([hidden, hidden])
    start = [50_257, 50_362, 1770, 13, 2264, 346, 353]

    cases = [
      {start, [318, 262, 46_329, 286, 262, 3504, 6097, 11, 290, 356, 389, 9675],
       [-0.0092, -0.0068, -0.2222, -0.0025, -0.0041, -0.2631] ++
         [-0.0701, -0.604, -0.0067, -0.0022, -0.2702, -0.0022]},
      {start ++ [318, 262, 46_329],
       [286, 262, 3504, 6097, 11, 290, 356, 389, 9675, 284, 7062, 465],
       [-0.0025, -0.0041, -0.2631, -0.0701, -0.604, -0.0067] ++
         [-0.0022, -0.2702, -0.0022, -0.0036, -0.0011, -0.0271]},
      {start ++ [318, 262, 46_329, 286, 262],
       [3504, 6097, 11, 290, 356, 389, 9675, 284, 7062, 465, 21_443, 13],
       [-0.2631, -0.0701, -0.604, -0.0067, -0.0022, -0.2702] ++
         [-0.0022, -0.0036, -0.0011, -0.0271, -0.4642, -0.1263]}
    ]

    check = fn tokens, ids, logprobs ->
      assert Enum.map(tokens, &hd/1) == ids

      for {[_id, logprob], expected} <- Enum.zip(tokens, logprobs) do
        assert_in_delta logprob, expected, 1.0e-3
      end
    end

    for {prompt, ids, logprobs} <- cases do
      [tokens, []] =
        Generation.generate(ctx, [prompt, []], [0, 0], %{"encoder_hidden_states" => hidden})

      check.(tokens, ids, logprobs)
    end

    # Rows with prompts of 7 and 12 tokens run as two groups, because :grow groups the rows by
    # the length of the cache, so each row gives its tokens alone.
    [{short, short_ids, short_logprobs}, _, {long, long_ids, long_logprobs}] = cases

    [first, second] =
      Generation.generate(ctx, [short, long], [0, 0], %{"encoder_hidden_states" => hidden})

    check.(first, short_ids, short_logprobs)
    check.(second, long_ids, long_logprobs)
  end

  # The merged Whisper decoder takes the length of its cache from the shape of the past. In
  # :append, a row with 5, 9 or 440 valid cache rows in a buffer of 512 rows gets the logits of
  # the cache of exact length: the positions read the table of 448 rows at the valid rows, not
  # at the capacity, and the key mask hides the free rows, also with the heads in the batch.
  test "a merged decoder takes a buffer with its position offset and key mask" do
    model =
      Transmuter.load!({:hf, "Xenova/whisper-tiny.en", "onnx/decoder_model_merged.onnx"},
        backend: @opts[:backend],
        outputs: &Generation.outputs/1
      )

    repo = %{generation_config: %{}, config: %{"eos_token_id" => 50_256}}
    ctx = Generation.context(repo, model, [], ["encoder_hidden_states"])
    assert ctx.mode == :append

    host = &Nx.tensor(&1, type: :s64, backend: Nx.BinaryBackend)
    flag = &%{"use_cache_branch" => Nx.tensor([&1], type: :u8, backend: Nx.BinaryBackend)}
    run = &Transmuter.run(&1, &1.params, Map.take(&2, &1.inputs), &3)

    hidden =
      Nx.iota({1, 1500, 384}, type: :f32, backend: @opts[:backend])
      |> Nx.remainder(17)
      |> Nx.subtract(8)
      |> Nx.divide(8)

    rows =
      for prompt <- [Enum.to_list(100..104), Enum.to_list(200..208), Enum.to_list(300..739)] do
        {inputs, _static} = Generation.prompt_inputs(model, 1, length(prompt))

        inputs =
          Map.merge(inputs, %{"input_ids" => host.([prompt]), "encoder_hidden_states" => hidden})

        first = run.(model, inputs, flag.(0))

        past =
          for {"present." <> name, value} <- first,
              into: %{},
              do: {"past_key_values." <> name, value}

        step = Map.merge(past, %{"input_ids" => host.([[50]]), "encoder_hidden_states" => hidden})
        {past, length(prompt), run.(model, step, flag.(1))["logits"]}
      end

    # Each row runs alone in a buffer of 512 rows, as generation runs the rows of one length.
    static = Map.put(flag.(1), "num_logits_to_keep", host.(1))

    for {past, length, alone} <- rows do
      buffers =
        for name <- ctx.cache, into: %{} do
          free = if name in ctx.buffers, do: 512 - length, else: 0
          {name, Nx.pad(past[name], 0.0, [{0, 0, 0}, {0, 0, 0}, {0, free, 0}, {0, 0, 0}])}
        end

      step =
        Map.merge(buffers, %{
          "input_ids" => host.([[50]]),
          "encoder_hidden_states" => hidden,
          "transmuter.position_offset" => host.(length - 512),
          "transmuter.key_mask" =>
            host.([List.duplicate(1, length) ++ List.duplicate(0, 512 - length) ++ [1]])
        })

      logits = run.(ctx.model, step, static)["logits"]
      diff = Nx.subtract(logits[0], alone[0]) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
      assert diff < 1.0e-3
    end
  end

  # The GPT-2 export slices a causal table of 1024 rows at the length of its cache, so in
  # :append the buffer and the new token must fit it: 900 + 32 tokens take a buffer of 1023
  # rows, not 1024, and give the greedy tokens of the cache of exact length. 900 + 132 tokens
  # raise at the step that does not fit.
  test "a merged decoder keeps its buffer within its causal table" do
    {repo, model} = gpt2()
    prompt = for i <- 1..900, do: rem(i * 37, 50_000)
    ctx = Generation.context(repo, model, max_new_tokens: 32)
    assert {ctx.mode, ctx.max_capacity} == {:append, 1023}

    [tokens] =
      Generation.generate(Generation.compile(ctx, 1, @opts[:defn_options]), [prompt], [0])

    assert Enum.map(tokens, &hd/1) == exact_greedy(model, prompt, 32)

    long = Generation.context(repo, model, max_new_tokens: 132)
    long = Generation.compile(long, 1, @opts[:defn_options])

    assert_raise ArgumentError, ~r/the 1023 rows/, fn ->
      Generation.generate(long, [prompt], [0])
    end
  end

  # A merged decoder that the rewrite moves to :append starts at 32 rows, and other decoders
  # at 256 rows. In GPT-2, a batch of 3 prompts of 30 tokens takes 64 rows (the prompt and a
  # margin of 16 rows), which grow to 128 rows at the step of the 65th row, and each row gives
  # the greedy tokens of the cache of exact length. In SmolLM2, prompts of 240 tokens take 256
  # rows, which grow to 512 rows. The capacities are those of the compiled steps, with the
  # first one from Generation.compile/3.
  test "the buffers start at the capacity of the decoder and grow when a step does not fit" do
    {repo, model} = gpt2()
    prompts = for row <- 1..3, do: for(i <- 1..30, do: rem(i * 37 * row, 50_000))
    {rows, capacities} = capacities(repo, model, prompts)
    assert capacities == [32, 64, 128]

    for {tokens, prompt} <- Enum.zip(rows, prompts) do
      assert Enum.map(tokens, &hd/1) == exact_greedy(model, prompt, 40)
    end

    opts = [outputs: &Generation.outputs/1, variant: :q4] ++ @opts
    source = {:hf, "onnx-community/SmolLM2-135M-Instruct-ONNX"}
    {repo, model, _tokenizer} = Transmuter.Serving.load!(source, opts, :q4)
    prompts = for row <- 1..2, do: for(i <- 1..240, do: rem(i * 37 * row, 49_000))
    assert {_rows, [256, 512]} = capacities(%{repo | generation_config: %{}}, model, prompts)
  end

  # In :in_place, GroupQueryAttention counts the valid rows of each row from the mask, so a
  # growth must keep the rows and the mask of each row. A prompt of 240 tokens takes 256 rows,
  # which grow to 512 rows. In a batch with a prompt of 500 tokens, the buffer starts at 1024
  # rows and does not grow, and the prompt gives the same tokens.
  test "a growth of the buffers in :in_place keeps the tokens of each row" do
    opts = [outputs: &Generation.outputs/1, variant: :fp32] ++ @opts
    source = {:hf, "HuggingFaceTB/SmolLM2-135M-Instruct"}
    {repo, model, _tokenizer} = Transmuter.Serving.load!(source, opts, :fp32)
    repo = %{repo | generation_config: %{}}
    assert Generation.context(repo, model, []).mode == :in_place
    prompt = for i <- 1..240, do: rem(i * 37, 49_000)
    long = for i <- 1..500, do: rem(i * 53, 49_000)

    assert {[grown], [256, 512]} = capacities(repo, model, [prompt])
    assert {[fits, _long], [256, 1024]} = capacities(repo, model, [prompt, long])
    assert Enum.map(grown, &hd/1) == Enum.map(fits, &hd/1)
  end

  # In :append, all rows of a step write at one offset, so the parts of prompts of 229, 31 and
  # 4 tokens leave gap rows in the shorter rows. The longest row ends early, at a token of its
  # own answer that the middle row does not give. The offset then passes the length of each
  # row, and the buffers grow from 256 to 512 rows for the 40 steps of the middle row. Each
  # row gives the tokens that it gives alone, where it has no gap rows.
  @essay "The printing press changed Europe in the fifteenth century. Before it, scribes copied each book by hand, and a single copy of a long text could take months. Johannes Gutenberg combined movable metal type, an oil-based ink and a screw press adapted from wine making. His workshop in Mainz printed a Bible around 1455, and within fifty years presses ran in more than two hundred cities. Books became cheaper, and more people learned to read. Scholars could compare identical copies of a text, so errors were easier to find and correct. Pamphlets spread news and arguments quickly, and the Reformation used them to reach ordinary readers. Printers also standardized spelling and grammar, because they wanted their books to sell in many regions. Over time, the press helped science, as researchers shared tables, maps and drawings that stayed the same in every copy. Newspapers appeared in the seventeenth century, and by the nineteenth century steam-powered presses printed thousands of pages each hour. Today historians still debate how fast these changes came, but most agree that the press made knowledge easier to keep. In the next part of this essay, we look at"
  @fox "A fox lived near a wide river at the edge of an old forest. Every morning she walked along the bank and watched the water. One day she saw"
  @tag ci_partition: 10
  test "rows of :append keep their tokens with gap rows, a row that ends early and a growth" do
    opts = [outputs: &Generation.outputs/1, variant: :fp32] ++ @opts
    source = {:hf, "onnx-community/SmolLM2-135M-Instruct-ONNX"}
    {repo, model, tokenizer} = Transmuter.Serving.load!(source, opts, :fp32)
    repo = %{repo | generation_config: %{}}
    assert Generation.context(repo, model, []).mode == :append

    prompts =
      for text <- [@essay, @fox, "Once upon a time"] do
        {:ok, encoding} = Tokenizers.Tokenizer.encode(tokenizer.native, text)
        Transmuter.Tokenizer.ids(tokenizer, encoding)
      end

    assert Enum.map(prompts, &length/1) == [229, 31, 4]
    [long, middle, _short] = prompts
    {[answer], _capacities} = capacities(repo, model, [long])
    {[other], _capacities} = capacities(repo, model, [middle])
    other = Enum.map(other, &hd/1)
    eos = answer |> Enum.map(&hd/1) |> Enum.drop(4) |> Enum.find(&(&1 not in other))
    repo = %{repo | generation_config: %{"eos_token_id" => eos}}

    {rows, capacities} = capacities(repo, model, prompts)
    alone = for prompt <- prompts, do: hd(elem(capacities(repo, model, [prompt]), 0))

    assert rows == alone
    assert length(hd(rows)) < 40
    assert length(Enum.at(rows, 1)) == 40
    assert capacities == [256, 512]
  end

  # A request joins a running group with admit/5: its prompt runs in the same steps as the
  # next tokens of the other rows. The key of a row folds the number of new tokens of the
  # row, so a request that joins at a later step gives the tokens that it gives alone, with
  # greedy selection and with sampling. A row that ends early is free for the next request,
  # which also gives its tokens alone. Both cache modes of buffers run the same requests.
  @tag ci_partition: 10
  test "requests that join a running group give the tokens that they give alone" do
    sources = [
      {{:hf, "onnx-community/SmolLM2-135M-Instruct-ONNX"}, :append},
      {{:hf, "HuggingFaceTB/SmolLM2-135M-Instruct"}, :in_place}
    ]

    for {source, mode} <- sources,
        opts = [outputs: &Generation.outputs/1, variant: :fp32] ++ @opts,
        {repo, model, tokenizer} = Transmuter.Serving.load!(source, opts, :fp32),
        strategy <- [:greedy, :sampling] do
      repo = %{repo | generation_config: %{}}
      ctx = Generation.context(repo, model, strategy: strategy, max_new_tokens: 10)
      assert ctx.mode == mode

      [fox, story, rainbow] =
        for text <- [@fox, "Once upon a time", "Explain how a rainbow forms."] do
          {:ok, encoding} = Tokenizers.Tokenizer.encode(tokenizer.native, text)
          Transmuter.Tokenizer.ids(tokenizer, encoding)
        end

      single = Generation.compile(ctx, 1, @opts[:defn_options])

      alone = fn prompt, seed ->
        single |> Generation.generate([prompt], [seed]) |> hd() |> Enum.map(&hd/1)
      end

      request = &%{row: &1, prompt: &2, seed: &3, budget: &4}
      ctx = Generation.compile(ctx, 2, @opts[:defn_options])

      # The fox runs alone for 3 steps, then the story joins. The story ends after 4 tokens,
      # and the rainbow takes its row while the fox still runs.
      group =
        ctx |> Generation.open(2) |> then(&Generation.admit(ctx, &1, [request.(0, fox, 7, 10)]))

      group = Enum.reduce(1..3, group, fn _step, group -> Generation.next(ctx, group) end)
      group = Generation.admit(ctx, group, [request.(1, story, 11, 4)])
      group = run_until(ctx, group, &Enum.at(&1.done, 1))
      story_tokens = row_tokens(group, 1)
      group = Generation.admit(ctx, group, [request.(1, rainbow, 13, 10)])
      group = run_until(ctx, group, &Enum.all?(&1.done))

      assert row_tokens(group, 0) == alone.(fox, 7)
      assert story_tokens == Enum.take(alone.(story, 11), 4)
      assert row_tokens(group, 1) == alone.(rainbow, 13)
    end
  end

  defp run_until(ctx, group, done?) do
    if done?.(group), do: group, else: run_until(ctx, Generation.next(ctx, group), done?)
  end

  defp row_tokens(group, row), do: group.ids |> Enum.at(row) |> Enum.reverse()

  # The merged T5 decoder reads the positions of all keys for its relative bias, and the
  # rewrite takes them from the union of the key masks of the rows. Row 0 ends early, at a
  # token of its own answer that row 1 does not give, so its key mask has fewer valid rows
  # for the later steps of row 1. Each row gives the tokens that it gives alone.
  @tag ci_partition: 10
  test "the relative bias of T5 takes the positions of the active rows" do
    {repo, encoder, decoder} =
      Transmuter.Serving.load_encoder_decoder!({:hf, "Xenova/t5-small"}, @opts)

    tokenizer = Transmuter.Serving.tokenizer(repo, [])

    ids =
      for text <- [
            "translate English to German: The house is wonderful.",
            "summarize: Scientists found a new species of frog in the rainforest. It is small."
          ] do
        {:ok, encoding} = Tokenizers.Tokenizer.encode(tokenizer.native, text)
        Transmuter.Tokenizer.ids(tokenizer, encoding)
      end

    width = ids |> Enum.map(&length/1) |> Enum.max()
    pad = &(&1 ++ List.duplicate(&2, width - length(&1)))
    host = &Nx.tensor(&1, type: :s64, backend: Nx.BinaryBackend)
    mask = host.(for row <- ids, do: pad.(List.duplicate(1, length(row)), 0))
    inputs = %{"input_ids" => host.(Enum.map(ids, &pad.(&1, 0))), "attention_mask" => mask}
    %{"last_hidden_state" => states} = Transmuter.run(encoder, encoder.params, inputs)
    given = %{"encoder_hidden_states" => states, "encoder_attention_mask" => mask}
    row = fn i -> Map.new(given, fn {k, t} -> {k, Nx.slice_along_axis(t, i, 1, axis: 0)} end) end

    generate = fn repo, rows ->
      inputs = if rows == [0, 1], do: given, else: row.(hd(rows))
      ctx = Generation.context(repo, decoder, [max_new_tokens: 40], Map.keys(given))
      assert ctx.mode == :append
      ctx = Generation.compile(ctx, length(rows), @opts[:defn_options])
      starts = List.duplicate([0], length(rows))
      Generation.generate(ctx, starts, List.duplicate(0, length(rows)), inputs)
    end

    # Without an eos token, each row gives 40 tokens.
    repo = %{repo | generation_config: %{"eos_token_id" => -1}}
    [answer] = generate.(repo, [0])
    [other] = generate.(repo, [1])
    other = Enum.map(other, &hd/1)
    eos = answer |> Enum.map(&hd/1) |> Enum.drop(4) |> Enum.find(&(&1 not in other))
    repo = %{repo | generation_config: %{"eos_token_id" => eos}}

    rows = generate.(repo, [0, 1])
    alone = for i <- [0, 1], do: hd(generate.(repo, [i]))

    assert rows == alone
    assert length(hd(rows)) < 40
    assert length(Enum.at(rows, 1)) == 40
  end

  # Generates 40 tokens for each prompt, and gives the rows and the capacities of the buffers
  # in the compiled steps.
  defp capacities(repo, model, prompts) do
    id = make_ref()
    pid = self()

    handler = fn _event, _measurements, metadata, _config ->
      for {"past_key_values." <> _, {{_batch, _heads, capacity, _dim}, _type}} <- metadata.shapes,
          self() == pid,
          do: send(pid, {:capacity, capacity})
    end

    :telemetry.attach(id, [:transmuter, :segment, :compile, :stop], handler, nil)
    ctx = Generation.context(repo, model, strategy: :greedy, max_new_tokens: 40)
    ctx = Generation.compile(ctx, length(prompts), @opts[:defn_options])
    rows = Generation.generate(ctx, prompts, List.duplicate(0, length(prompts)))
    :telemetry.detach(id)
    {rows, received_capacities()}
  end

  defp received_capacities do
    receive do
      {:capacity, capacity} -> Enum.uniq(Enum.sort([capacity | received_capacities()]))
    after
      0 -> []
    end
  end

  defp gpt2 do
    opts = [outputs: &Generation.outputs/1, variant: :fp32] ++ @opts
    {repo, model, _tokenizer} = Transmuter.Serving.load!({:hf, "Xenova/gpt2"}, opts, :fp32)
    {%{repo | generation_config: %{}}, model}
  end

  # The greedy tokens of a GPT-2 merged decoder as loaded, with a cache of exact length: the
  # prompt without cache, then one token at a time.
  defp exact_greedy(model, prompt, count) do
    host = &Nx.tensor(&1, type: :s64, backend: Nx.BinaryBackend)
    cache = Enum.filter(model.inputs, &String.starts_with?(&1, "past_key_values."))
    zero = Nx.tensor(0.0, backend: @opts[:backend])
    placeholder = Map.new(cache, &{&1, Nx.broadcast(zero, {1, 12, 1, 64})})

    step = fn tokens, past, length, branch ->
      flag = Nx.tensor([branch], type: :u8, backend: Nx.BinaryBackend)
      mask = host.([List.duplicate(1, length)])
      inputs = Map.merge(past, %{"input_ids" => host.([tokens]), "attention_mask" => mask})
      outputs = Transmuter.run(model, model.params, inputs, %{"use_cache_branch" => flag})
      token = outputs["logits"][[0, -1]] |> Nx.argmax() |> Nx.to_number()

      past =
        for {"present." <> name, value} <- outputs,
            into: %{},
            do: {"past_key_values." <> name, value}

      {token, past}
    end

    {first, past} = step.(prompt, placeholder, length(prompt), 0)

    {rest, _last} =
      Enum.map_reduce(
        (length(prompt) + 1)..(length(prompt) + count - 1)//1,
        {first, past},
        fn length, {token, past} ->
          {next, past} = step.([token], past, length, 1)
          {next, {next, past}}
        end
      )

    [first | rest]
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

  # In :grow, each length of the cache is a shape of the step. The prompt of 2 tokens runs as
  # one step on the branch without cache. Each shape compiles one program, which also selects
  # the token, and a second request compiles nothing. The prompt of 1 token has the shape of
  # the step that the compile of the context gives, and its later steps have the lengths of the
  # first request, so it compiles nothing either.
  @tag ci_partition: 10
  test "each shape of a step compiles one program, with the selection in it" do
    opts = [outputs: &Generation.outputs/1, variant: :fp32] ++ @opts
    {repo, model, _tokenizer} = Transmuter.Serving.load!({:hf, "Xenova/gpt2"}, opts, :fp32)
    id = make_ref()
    pid = self()

    handler = fn _event, _measurements, metadata, _config ->
      if self() == pid and metadata.compiled do
        inputs = for {name, shape} <- metadata.shapes, name in model.inputs, do: {name, shape}
        send(pid, {:compiled, Map.new(inputs)})
      end
    end

    :telemetry.attach(id, [:transmuter, :segment, :compile, :stop], handler, nil)
    on_exit(fn -> :telemetry.detach(id) end)

    ctx = Generation.context(repo, model, strategy: :greedy, max_new_tokens: 4)
    ctx = Generation.compile(ctx, 1, @opts[:defn_options])
    request = fn prompt -> Generation.generate(ctx, [prompt], [0]) end

    first = request.([464, 3290])
    compiled = flush()
    assert request.([464, 3290]) == first
    request.([464])
    assert flush() == []

    assert length(compiled) > 2
    assert compiled == Enum.uniq(compiled)
  end

  defp flush do
    receive do
      {:compiled, shapes} -> [shapes | flush()]
    after
      0 -> []
    end
  end
end

defmodule Transmuter.GenerationBitsTest do
  use ExUnit.Case, async: true

  import ExUnit.CaptureLog

  alias Transmuter.{Generation, GraphBreaks}
  alias Transmuter.Onnx.{AttributeProto, GraphProto, NodeProto, ValueInfoProto}

  # The lm_head of a decoder without num_logits_to_keep computes the logits of one position
  # with the input that generation adds, through the softcap of Gemma. A Reshape after the
  # lm_head, another reader of a value on the way to the logits, or an input that the graph
  # already declares, leaves the graph as it is.
  test "the context keeps the logits of the last positions before the lm_head" do
    node = &%NodeProto{op_type: &1, name: &1, input: &2, output: &3}
    host = &Nx.tensor(&1, backend: Nx.BinaryBackend)

    softcap = [
      node.("Gather", ["embed", "input_ids"], ["hidden"]),
      node.("MatMul", ["hidden", "lm_head"], ["scores"]),
      node.("Div", ["scores", "cap"], ["scaled"]),
      node.("Tanh", ["scaled"], ["capped"]),
      node.("Mul", ["capped", "cap"], ["logits"])
    ]

    model = %{
      nodes: softcap,
      params: %{"embed" => Nx.iota({5, 3}, type: :f32), "lm_head" => Nx.iota({3, 4}, type: :f32)},
      static: %{"cap" => host.(30.0)},
      inputs: ["input_ids", "position_ids"],
      input_types: %{
        "input_ids" => {:s64, [1, "length"]},
        "position_ids" => {:s64, [1, "length"]}
      },
      outputs: ["logits"],
      opset: 17,
      backend: nil,
      compute_type: nil
    }

    repo = %{generation_config: %{}, config: %{"eos_token_id" => 0}}
    inputs = %{"input_ids" => host.([[1, 4, 2]])}
    %{"logits" => all} = Transmuter.run(model, model.params, inputs)

    %{model: kept} = Generation.context(repo, model, [])
    assert "num_logits_to_keep" in kept.inputs

    for {keep, expected} <- [{1, all[[.., 2..2]]}, {2, all[[.., 1..2]]}, {0, all}] do
      static = %{"num_logits_to_keep" => host.(keep)}
      assert Transmuter.run(kept, kept.params, inputs, static)["logits"] == expected
    end

    flat = %{
      model
      | nodes: [node.("Reshape", ["capped", "shape"], ["logits"]) | Enum.drop(softcap, -1)],
        static: Map.put(model.static, "shape", host.([1, -1, 4]))
    }

    declared = %{
      model
      | inputs: model.inputs ++ ["num_logits_to_keep"],
        input_types: Map.put(model.input_types, "num_logits_to_keep", {:s64, []})
    }

    shared = %{model | nodes: softcap ++ [node.("Identity", ["scores"], ["copy"])]}

    # A merged decoder gives the logits from an If, whose branches each get a cut. A branch
    # can give its value the name of the If output, as in the exports of optimum.
    branch = fn nodes ->
      %GraphProto{node: nodes, output: [%ValueInfoProto{name: hd(List.last(nodes).output)}]}
    end

    merged = fn cache_branch ->
      if_node = %{
        node.("If", ["use_cache_branch"], ["logits"])
        | attribute: [
            %AttributeProto{name: "then_branch", type: :GRAPH, g: branch.(cache_branch)},
            %AttributeProto{
              name: "else_branch",
              type: :GRAPH,
              g: branch.([node.("MatMul", ["hidden", "lm_head"], ["logits"])])
            }
          ]
      }

      %{
        model
        | nodes: [hd(softcap), if_node],
          inputs: model.inputs ++ ["use_cache_branch"],
          input_types: Map.put(model.input_types, "use_cache_branch", {:u8, [1]})
      }
    end

    cached = [
      node.("MatMul", ["hidden", "lm_head"], ["cached_scores"]),
      node.("Div", ["cached_scores", "cap"], ["cached"])
    ]

    merged_model = merged.(cached)
    %{model: kept} = Generation.context(repo, merged_model, [])

    for flag <- [0, 1], {keep, positions} <- [{1, 2..2}, {2, 1..2}, {0, 0..2}] do
      static = %{"use_cache_branch" => Nx.tensor([flag], type: :u8, backend: Nx.BinaryBackend)}
      %{"logits" => all} = Transmuter.run(merged_model, model.params, inputs, static)
      static = Map.put(static, "num_logits_to_keep", host.(keep))
      assert Transmuter.run(kept, kept.params, inputs, static)["logits"] == all[[.., positions]]

      # The static inputs select the branch and fold the slice, so the graph does not break.
      compiler = Transmuter.TestBackend.compiler_opts()
      {compiled, segments} = GraphBreaks.run_once(kept, inputs, static, compiler)
      assert [%{start: 0, stop: :end}] = segments
      compiled = Nx.backend_transfer(compiled["logits"], Nx.BinaryBackend)
      assert Nx.to_number(Nx.all_close(compiled, all[[.., positions]])) == 1
    end

    flat_branch = merged.([node.("Reshape", ["hidden", "shape"], ["cached"])])
    flat_branch = %{flat_branch | static: flat.static}

    # A branch value with the name of a static value of the outer graph is not static.
    shadowed = merged.([node.("Identity", ["hidden"], ["cap"]) | cached])

    for {model, op} <- [
          {flat, "Reshape"},
          {shared, "MatMul"},
          {flat_branch, "Reshape"},
          {shadowed, "Div"}
        ] do
      log = capture_log(fn -> assert Generation.context(repo, model, []).model == model end)
      assert log =~ "stops at #{op}"
    end

    assert Generation.context(repo, declared, []).model == declared

    # The vocabulary size is the last dimension of the logits, else the vocab_size of the
    # configuration.
    typed = Map.put(model, :output_types, %{"logits" => {:f32, ["batch", "length", 4]}})
    symbolic = Map.put(model, :output_types, %{"logits" => {:f32, ["batch", "length", "vocab"]}})
    configured = %{repo | config: Map.put(repo.config, "vocab_size", 6)}

    assert Generation.context(configured, typed, []).vocab == 4
    assert Generation.context(configured, symbolic, []).vocab == 6
    assert Generation.context(repo, symbolic, []).vocab == nil
  end

  # The selection gives f32 values as the integers of their bits. NaN and the infinities have
  # no float in Elixir, so they come back as the atoms of Nx.to_list/1.
  test "from_bits/1 and to_bits/1 give every f32 value and its bits" do
    for {bits, value} <- [
          {0x3FC00000, 1.5},
          {-0x40800000, -1.0},
          {0x00000001, 1.401298464324817e-45},
          {0x7F800000, :infinity},
          {-0x800000, :neg_infinity},
          {0x7FC00000, :nan}
        ] do
      assert Generation.from_bits(bits) == value
      assert Generation.to_bits(value) == bits
    end

    assert Generation.from_bits([0x7F800001, -1]) == [:nan, :nan]
  end

  # A step frees the device tensors that it owns, except one whose buffer a kept tensor
  # shares. A host tensor has no device buffer to free.
  test "release/2 frees the dead device tensors that no kept tensor shares" do
    backend = Transmuter.TestBackend.backend()
    [dead, shared] = for value <- [1.0, 2.0], do: Nx.tensor([value], backend: backend)
    host = Nx.tensor([3.0], backend: Nx.BinaryBackend)

    assert Generation.release([dead, shared, host], [shared]) == :ok
    assert freed?(dead)
    refute freed?(shared)
    refute freed?(host)
  end

  # The two stages of largest/2 give the values and indices of Nx.top_k, in the compiled
  # programs of the test backend. Ties take the lower index first in one chunk and across
  # chunks. The last chunk of a row of 5123 values (chunks of 128) has 3 values, fewer than
  # k = 8, and its padding has the smallest value, as do the -inf values of a row with 2 finite
  # values. With k = 300, the chunks have 512 values, and a row of 200 values is one chunk.
  test "largest/2 gives the values and indices of Nx.top_k" do
    opts = Transmuter.TestBackend.compiler_opts()
    {random, _key} = Nx.Random.normal(Nx.Random.key(7), shape: {2, 5123})
    top = Nx.reduce_max(random)
    put = fn tensor, at, value -> Nx.put_slice(tensor, [0, at], Nx.broadcast(value, {2, 1})) end
    min = Nx.Constants.min_finite(:f32)
    masked = Nx.broadcast(min, {2, 5123}) |> put.(5, 1.0) |> put.(5121, 2.0) |> put.(1700, 1.0)

    inf =
      Nx.broadcast(Nx.Constants.neg_infinity(:f32), {2, 5123}) |> put.(7, 1.0) |> put.(5122, 1.0)

    for {tensor, k} <- [
          {random, 8},
          {random |> put.(100, top) |> put.(101, top) |> put.(102, top), 8},
          {random |> put.(100, top) |> put.(2600, top) |> put.(5120, top), 8},
          {masked, 8},
          {inf, 8},
          {random, 300},
          {Nx.slice_along_axis(random, 0, 200, axis: 1), 150}
        ] do
      tensor = Nx.backend_transfer(tensor, Transmuter.TestBackend.backend())
      {values, indices} = Nx.Defn.jit(&Generation.largest(&1, k), opts).(tensor)
      {expected, at} = Nx.Defn.jit(&Nx.top_k(&1, k: k), opts).(tensor)
      assert Nx.to_flat_list(values) == Nx.to_flat_list(expected)
      assert Nx.to_flat_list(indices) == Nx.to_flat_list(at)
    end
  end

  # A read of a freed tensor raises, with the error of the backend.
  defp freed?(tensor) do
    Nx.to_flat_list(tensor)
    false
  rescue
    _error -> true
  end
end

defmodule Transmuter.GenerationMemoryTest do
  # The memory of the EXLA client is shared by all processes, so this test runs alone. It has
  # no :ci_partition tag, so it runs in EXLA partition 1, and an --only of a partition cannot
  # select it on EMLX.
  use ExUnit.Case, async: false

  @moduletag :hub
  @moduletag :exla
  @moduletag timeout: 900_000

  @opts [
    backend: Transmuter.TestBackend.backend(),
    defn_options: Transmuter.TestBackend.compiler_opts()
  ]

  alias Transmuter.Generation

  # Generation frees the device tensors that it makes: the dead cache of each step, and the
  # cache and the rest of the state of a group when it ends. So after a request, the client
  # holds what it held before, within a margin for its own buffers. EMLX 0.5 frees an MLX array
  # only when the BEAM collects its tensor, also after Nx.backend_deallocate/1, so the test
  # runs on EXLA.
  test "a request leaves no device memory, with greedy selection and with beams" do
    opts = [outputs: &Generation.outputs/1, variant: :q4] ++ @opts
    source = {:hf, "onnx-community/SmolLM2-135M-Instruct-ONNX"}
    {repo, model, _tokenizer} = Transmuter.Serving.load!(source, opts, :q4)
    client = EXLA.Client.fetch!(Keyword.fetch!(elem(@opts[:backend], 1), :client))
    allocated = fn -> EXLA.Client.get_memory_statistics(client).allocated end

    for num_beams <- [1, 2] do
      ctx = Generation.context(repo, model, max_new_tokens: 8, num_beams: num_beams)
      ctx = Generation.compile(ctx, 2 * num_beams, @opts[:defn_options])
      request = fn -> Generation.generate(ctx, [[1, 9690, 198], [1, 9690, 198, 2683]], [0, 0]) end

      request.()

      # A garbage collection would free dropped tensors too, so the request runs in a process
      # with a heap large enough that it does not collect.
      test = self()

      Process.spawn(
        fn ->
          before = allocated.()
          request.()
          send(test, {:left, allocated.() - before})
        end,
        min_heap_size: 10_000_000
      )

      assert_receive {:left, left}, 300_000
      assert left < 4_000_000
    end
  end
end
