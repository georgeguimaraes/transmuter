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

  test "beam search raises with sampling and with a stream" do
    repo = {:hf, "Xenova/gpt2"}

    assert_raise ArgumentError, ~r/strategy: :greedy/, fn ->
      Text.generation(repo, [variant: :fp32, num_beams: 3, strategy: :sampling] ++ @opts)
    end

    assert_raise ArgumentError, ~r/stream: true/, fn ->
      Text.generation(repo, [variant: :fp32, num_beams: 3, stream: true] ++ @opts)
    end
  end

  # In :grow, each length of the cache is a shape of the step. The prompt of 3 tokens runs as
  # parts of 2 and 1 tokens. Each shape compiles one program, which also selects the token,
  # and a second request compiles nothing. The prompt of 1 token has the shape of the step
  # that the compile of the context gives, so it compiles nothing either.
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

    first = request.([464, 3290, 286])
    compiled = flush()
    assert request.([464, 3290, 286]) == first
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

  # A read of a freed tensor raises, with the error of the backend.
  defp freed?(tensor) do
    Nx.to_flat_list(tensor)
    false
  rescue
    _error -> true
  end
end
