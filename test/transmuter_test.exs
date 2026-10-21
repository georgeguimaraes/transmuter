defmodule TransmuterTest do
  use ExUnit.Case, async: true

  # Reference values come from ONNX Runtime with graph optimizations disabled,
  # see test/support/model_references.py. The first run downloads the models.
  @moduletag timeout: 900_000

  # Metal on macOS; the CPU device elsewhere, such as Linux CI. The EMLX compiler
  # takes the device too, because it defaults to the GPU.
  # TRANSMUTER_TEST_DEVICE=cpu runs the CI configuration on a Mac.
  @device (System.get_env("TRANSMUTER_TEST_DEVICE") ||
             if(match?({:unix, :darwin}, :os.type()), do: "gpu", else: "cpu"))
          |> String.to_existing_atom()
  @backend {EMLX.Backend, device: @device}
  @compiler [compiler: EMLX, device: @device]

  test "runs a BERT encoder" do
    model = load("Xenova/all-MiniLM-L6-v2", "onnx/model.onnx")
    ids = Nx.tensor([[101, 7592, 2088, 2003, 1037, 3231, 102]], type: :s64, backend: @backend)

    inputs = %{
      "input_ids" => ids,
      "attention_mask" => Nx.broadcast(Nx.tensor(1, type: :s64, backend: @backend), {1, 7}),
      "token_type_ids" => Nx.broadcast(Nx.tensor(0, type: :s64, backend: @backend), {1, 7})
    }

    %{"last_hidden_state" => hidden} = run(model, inputs)

    assert_all_close(hidden[[0, 0..2, 0..2]], [
      [0.2091, -0.0291, 0.0011],
      [-0.4452, 0.2465, 0.5265],
      [0.0783, 0.2770, -0.2609]
    ])
  end

  test "resizes with interpolation matrices" do
    model = load("Xenova/segformer-b0-finetuned-ade-512-512", "onnx/model.onnx")
    %{"logits" => logits} = run(model, %{"pixel_values" => pattern({1, 3, 512, 512})})

    assert_all_close(logits[[0, 0..2, 0, 0..2]], [
      [-5.9847, -6.1833, -6.5003],
      [-7.3063, -7.5704, -7.7874],
      [-9.4081, -9.2398, -9.7014]
    ])
  end

  test "selects an If branch from a static input" do
    model = load("Xenova/whisper-tiny.en", "onnx/decoder_model_merged.onnx")

    past =
      for name <- model.inputs, String.starts_with?(name, "past_key_values"), into: %{} do
        {name, pattern({1, 6, if(name =~ ".encoder.", do: 1500, else: 3), 64})}
      end

    inputs =
      Map.merge(past, %{
        "input_ids" => Nx.tensor([[1000]], type: :s64, backend: @backend),
        "encoder_hidden_states" => pattern({1, 1500, 384})
      })

    flag = Nx.tensor([1], type: :u8, backend: Nx.BinaryBackend)
    %{"logits" => logits} = run(model, inputs, %{"use_cache_branch" => flag})
    assert_all_close(logits[[0, 0, 0..4]], [11.7904, 7.4783, 7.9252, 8.0604, 6.4908])

    # As a traced input, the condition breaks the graph and the If runs on its value.
    %{"logits" => traced} =
      run(model, Map.put(inputs, "use_cache_branch", Nx.backend_copy(flag, @backend)))

    assert_all_close(traced[[0, 0, 0..4]], [11.7904, 7.4783, 7.9252, 8.0604, 6.4908])
  end

  test "generates with a static KV cache like transformers" do
    model = load("HuggingFaceTB/SmolLM2-135M-Instruct", "onnx/model.onnx")

    prompt = [
      1,
      9690,
      198,
      2683,
      359,
      253,
      5356,
      5646,
      11_173,
      3365,
      3511,
      308,
      34_519,
      28,
      7018,
      411,
      407,
      19_712,
      8182,
      2,
      198,
      1,
      4093,
      198,
      36_971,
      281,
      827,
      8545,
      1701,
      260,
      6376,
      314,
      4461,
      30,
      2,
      198,
      1,
      520,
      9531,
      198
    ]

    max_len = 64

    # The KV buffer is as long as the attention mask, so GroupQueryAttention writes
    # new keys in place and every decode step has the same shapes.
    cache =
      for name <- model.inputs, String.starts_with?(name, "past_key_values"), into: %{} do
        {name, Nx.broadcast(Nx.tensor(0.0, backend: @backend), {1, 3, max_len, 64})}
      end

    step_inputs = fn ids, pos, cache ->
      n = pos + length(ids)

      Map.merge(cache, %{
        "input_ids" => Nx.tensor([ids], type: :s64, backend: @backend),
        "attention_mask" =>
          Nx.tensor([Enum.map(0..(max_len - 1), &if(&1 < n, do: 1, else: 0))],
            type: :s64,
            backend: @backend
          ),
        "position_ids" => Nx.tensor([Enum.to_list(pos..(n - 1))], type: :s64, backend: @backend)
      })
    end

    prefill = Transmuter.compile(model, step_inputs.(prompt, 0, cache), @compiler)
    decode = Transmuter.compile(model, step_inputs.([0], length(prompt), cache), @compiler)

    next_cache = fn out ->
      for {"present." <> rest, t} <- out, into: %{}, do: {"past_key_values." <> rest, t}
    end

    next_token = fn out -> out["logits"][[0, -1]] |> Nx.argmax() |> Nx.to_number() end

    out = prefill.(step_inputs.(prompt, 0, cache))

    {tokens, _} =
      Enum.reduce(1..7, {[next_token.(out)], next_cache.(out)}, fn i, {tokens, cache} ->
        out = decode.(step_inputs.([hd(tokens)], length(prompt) + i - 1, cache))
        {[next_token.(out) | tokens], next_cache.(out)}
      end)

    assert Enum.reverse(tokens) == [504, 6376, 4541, 4461, 975, 260, 2591, 506]
  end

  # The BERT pooler gathers the first token on axis 1, a path whose gradient EMLX 0.5
  # scatters to the wrong place. Autodiff must match central finite differences.
  test "computes gradients that match finite differences" do
    model = load("Xenova/ms-marco-MiniLM-L-6-v2", "onnx/model.onnx", trainable: true)
    params = model.params

    # Two sequences: with a batch of 1 the misplaced gradient happens to land correctly.
    ids =
      Nx.tensor(
        [
          [101, 2054, 2003, 1041, 102, 3793, 2003, 1037, 2653, 102],
          [101, 3007, 1997, 2605, 102, 3000, 2003, 1996, 3007, 102]
        ],
        type: :s64,
        backend: @backend
      )

    segments = Nx.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], type: :s64, backend: @backend)

    inputs = %{
      "input_ids" => ids,
      "attention_mask" => Nx.broadcast(Nx.tensor(1, type: :s64, backend: @backend), {2, 10}),
      "token_type_ids" => Nx.broadcast(segments, {2, 10})
    }

    score = fn p, i -> Transmuter.run(model, p, i)["logits"] |> Nx.sum() end
    forward = Nx.Defn.jit(score, @compiler)

    {_, grads} =
      Nx.Defn.jit(&Nx.Defn.value_and_grad(&1, fn p -> score.(p, &2) end), @compiler).(
        params,
        inputs
      )

    name = "bert.embeddings.LayerNorm.weight"

    bump = fn delta ->
      Map.update!(
        params,
        name,
        &Nx.indexed_add(
          &1,
          Nx.tensor([[1]], backend: @backend),
          Nx.tensor([delta], backend: @backend)
        )
      )
    end

    numeric =
      (Nx.to_number(forward.(bump.(1.0e-2), inputs)) -
         Nx.to_number(forward.(bump.(-1.0e-2), inputs))) / 2.0e-2

    autodiff = grads[name] |> Nx.backend_copy(Nx.BinaryBackend) |> Nx.to_flat_list() |> Enum.at(1)

    assert_in_delta autodiff, numeric, 0.02 * abs(numeric) + 1.0e-3
  end

  test "returns a string output only without compiling" do
    model = Transmuter.load!(Path.expand("fixtures/ops/Identity_strings/model.onnx", __DIR__))
    strings = %{"x" => {:strings, {2}, ["a", "b"]}}

    assert Transmuter.run(model, model.params, %{}, strings) == %{
             "y" => {:strings, {2}, ["a", "b"]}
           }

    assert_raise ArgumentError, ~r/output y is a string tensor/, fn ->
      Transmuter.compile(model, %{}, [static_inputs: strings] ++ @compiler).(%{})
    end
  end

  test "draws new random values on each call" do
    model = load_fixture("RandomNormal_mean_scale")
    compiled = Transmuter.compile(model, %{}, @compiler)
    assert different?(compiled.(%{})["y"], compiled.(%{})["y"])

    eager = fn -> Transmuter.run(model, model.params, %{})["y"] end
    assert different?(eager.(), eager.())
  end

  test "draws the same values from the same key" do
    model = load_fixture("RandomNormal_mean_scale")
    compiled = Transmuter.compile(model, %{}, @compiler)
    key = &%{Transmuter.random_input() => Nx.backend_transfer(Nx.Random.key(&1), @backend)}
    first = compiled.(key.(1))["y"]

    refute different?(first, compiled.(key.(1))["y"])
    assert different?(first, compiled.(key.(2))["y"])
    eager = Transmuter.run(model, model.params, key.(1))["y"]
    assert_all_close(first, Nx.backend_copy(eager, Nx.BinaryBackend))
  end

  test "draws the same values on each call from a seed" do
    model = load_fixture("RandomNormal_seed")
    compiled = Transmuter.compile(model, %{}, @compiler)
    first = compiled.(%{})["y"]
    refute different?(first, compiled.(%{})["y"])
    refute different?(first, Transmuter.compile(model, %{}, @compiler).(%{})["y"])
  end

  test "draws different values in two random nodes" do
    model = load_fixture("RandomUniform_two_nodes")
    %{"a" => a, "b" => b} = Transmuter.compile(model, %{}, @compiler).(%{})
    assert different?(a, b)
  end

  test "a model without noise gives fixed values" do
    f32 = &Nx.tensor(&1, type: :f32, backend: @backend)

    for {name, inputs, expected} <- [
          {"RandomNormal_mean_scale", %{}, &Nx.broadcast(2.0, &1)},
          {"RandomNormalLike", %{"x" => f32.([[0.5, 2.0]])}, fn _ -> [[-1.0, -1.0]] end},
          {"RandomUniform_low_high", %{}, &Nx.broadcast(4.0, &1)},
          {"RandomUniformLike_dtype", %{"x" => Nx.tensor([[3, 7]], type: :s32)},
           fn _ -> [[1.0, 1.0]] end},
          {"Bernoulli", %{"p" => f32.([0.0, 0.3, 1.0])}, fn _ -> [0.0, 1.0, 1.0] end},
          {"Multinomial", %{"x" => f32.([[0.1, 2.0, -1.0]])}, &Nx.broadcast(1, &1)}
        ] do
      path = Path.expand("fixtures/ops/#{name}/model.onnx", __DIR__)
      model = Transmuter.load!(path, backend: @backend, noise: false)
      inputs = Map.new(inputs, fn {k, t} -> {k, Nx.backend_transfer(t, @backend)} end)

      for y <- [
            Transmuter.run(model, model.params, inputs)["y"],
            Transmuter.compile(model, inputs, @compiler).(inputs)["y"]
          ] do
        y = Nx.backend_copy(y, Nx.BinaryBackend)
        expected = Nx.tensor(expected.(Nx.shape(y)), type: Nx.type(y))
        assert y == expected, "#{name} gave #{inspect(y)}"
      end
    end
  end

  defp load_fixture(name) do
    Transmuter.load!(Path.expand("fixtures/ops/#{name}/model.onnx", __DIR__), backend: @backend)
  end

  defp different?(a, b), do: Nx.to_number(Nx.all(Nx.equal(a, b))) == 0

  test "compiles past a value that is only known at run time" do
    path = Path.expand("fixtures/ops/Break_Range_long_tail/model.onnx", __DIR__)
    model = Transmuter.load!(path, backend: @backend)
    input = &%{"x" => Nx.tensor(&1, backend: @backend)}
    eager = &(Transmuter.run(model, model.params, input.(&1))["y"] |> Nx.backend_copy())

    # The function compiles in a process that exits, so its cache table is gone.
    compiled =
      Task.async(fn -> Transmuter.compile(model, input.([1.0, 1, 1, 1, 1, 1]), @compiler) end)
      |> Task.await()

    for x <- [[1.0, 1, 1, 1, 1, 1], [2.0, 2, 1, 1, 1, 1], [1.0, 1, 1, 1, 1, 1]] do
      result = compiled.(input.(x))["y"]
      assert Nx.shape(result) == {round(Enum.sum(x))}
      assert_all_close(result, eager.(x))
    end

    # Calls from the process that compiled it, and from another process, share the cache.
    compiled = Transmuter.compile(model, input.([1.0, 1, 1, 1, 1, 1]), @compiler)

    for x <- [[2.0, 2, 1, 1, 1, 1], [1.0, 1, 1, 1, 1, 1], [2.0, 2, 1, 1, 1, 1]] do
      assert_all_close(compiled.(input.(x))["y"], eager.(x))
      assert_all_close(Task.async(fn -> compiled.(input.(x))["y"] end) |> Task.await(), eager.(x))
    end
  end

  defmodule CountingCompiler do
    @moduledoc false
    @behaviour Nx.Defn.Compiler

    defdelegate __jit__(key, vars, fun, args, opts), to: EMLX
    defdelegate __partitions_options__(opts), to: EMLX
    defdelegate __to_backend__(opts), to: EMLX

    def __compile__(key, vars, fun, opts) do
      send(self(), :compiled)
      EMLX.__compile__(key, vars, fun, opts)
    end
  end

  test "a compiled function takes inputs of other shapes" do
    model = load_fixture("Cosh")
    input = &%{"x" => Nx.iota(&1, type: :f32, backend: @backend) |> Nx.divide(10)}

    compiled =
      Transmuter.compile(model, input.({3, 5}), compiler: CountingCompiler, device: @device)

    assert flush_compiles(0) == 1

    for shape <- [{2, 7}, {2, 7}, {3, 5}] do
      expected = Transmuter.run(model, model.params, input.(shape))["y"]
      assert_all_close(compiled.(input.(shape))["y"], expected)
    end

    assert flush_compiles(0) == 1
  end

  test "keeps the most recently used segments after a break" do
    model = load_fixture("Break_Range_long_tail")
    input = &%{"x" => Nx.tensor([&1, 1, 1, 1, 1, 1], type: :f32, backend: @backend)}
    compiled = Transmuter.compile(model, input.(1), compiler: CountingCompiler, device: @device)

    compiles = fn firsts ->
      Enum.each(firsts, &compiled.(input.(&1)))
      flush_compiles(0)
    end

    assert flush_compiles(0) == 1
    assert compiles.(1..10) == 10
    assert compiles.(3..10) == 0
    assert compiles.([1]) == 1
  end

  test "the cache size sets how many shapes stay compiled" do
    model = load_fixture("Break_Range_long_tail")
    input = &%{"x" => Nx.tensor([&1, 1, 1, 1, 1, 1], type: :f32, backend: @backend)}

    compiled =
      Transmuter.compile(model, input.(1),
        compiler: CountingCompiler,
        device: @device,
        cache_size: 2
      )

    compiles = fn firsts ->
      Enum.each(firsts, &compiled.(input.(&1)))
      flush_compiles(0)
    end

    assert flush_compiles(0) == 1
    assert compiles.(1..3) == 3
    assert compiles.(2..3) == 0
    assert compiles.([1]) == 1
  end

  defp flush_compiles(count) do
    receive do
      :compiled -> flush_compiles(count + 1)
    after
      0 -> count
    end
  end

  alias Transmuter.Onnx.{
    AttributeProto,
    GraphProto,
    ModelProto,
    NodeProto,
    OperatorSetIdProto,
    TensorProto,
    TensorShapeProto,
    TypeProto,
    ValueInfoProto
  }

  # ONNX Runtime rejects this graph too: "Subgraph output (y) is an outer scope value".
  # The merged fp16 decoders of some Xenova exports have such If branches.
  test "a subgraph output that no node of the subgraph computes fails at load" do
    branch = %GraphProto{name: "then", node: [node("Neg", ["x"], ["z"])], output: [value("y")]}

    graph = %GraphProto{
      input: [value("x", [4]), value("flag", [1], 9)],
      output: [value("y")],
      node: [
        node("If", ["flag"], ["y"],
          attribute: [
            %AttributeProto{name: "then_branch", type: :GRAPH, g: branch},
            %AttributeProto{name: "else_branch", type: :GRAPH, g: branch}
          ]
        )
      ]
    }

    assert_raise ArgumentError, ~r/then_branch of If if gives the outputs y, which no node/, fn ->
      graph |> write_model() |> Transmuter.load!()
    end
  end

  # Each Add reads its input twice, so the expression of x16 is a graph with 16 levels and
  # 2^16 paths. An error that kept it would copy all paths when it leaves the process.
  test "an error in a traced node keeps no traced values" do
    adds = for i <- 0..15, do: node("Add", ["x#{i}", "x#{i}"], ["x#{i + 1}"])

    graph = %GraphProto{
      input: [value("x0", [4])],
      output: [value("y")],
      node: adds ++ [node("Cast", ["x16", "x16"], ["y"], attribute: [int_attribute("to", 1)])]
    }

    model = Transmuter.load!(write_model(graph), backend: @backend)
    inputs = %{"x0" => Nx.iota({4}, type: :f32, backend: @backend)}

    {error, stacktrace} =
      try do
        Transmuter.compile(model, inputs, @compiler)
      rescue
        e -> {e, __STACKTRACE__}
      end

    assert %RuntimeError{message: "Cast " <> _} = error
    assert :erts_debug.flat_size({error, stacktrace}) < 10_000
  end

  test "compute_type converts the half-precision params on the backend" do
    weight = Nx.iota({4, 8}, type: :f16, backend: Nx.BinaryBackend) |> Nx.divide(8)

    initializer = %TensorProto{
      name: "w",
      data_type: 10,
      dims: [4, 8],
      raw_data: Nx.to_binary(weight)
    }

    graph = %GraphProto{
      input: [value("x", [1, 4], 10)],
      output: [value("y")],
      initializer: [initializer],
      node: [node("MatMul", ["x", "w"], ["y"])]
    }

    model = Transmuter.load!(write_model(graph), backend: @backend, compute_type: :f32)
    assert %Nx.Tensor{type: {:f, 32}, data: %EMLX.Backend{}} = model.params["w"]

    x = Nx.tensor([[1.0, 2.0, 3.0, 4.0]], type: :f16, backend: @backend)
    %{"y" => y} = Transmuter.run(model, model.params, %{"x" => x})
    assert Nx.type(y) == {:f, 32}
    assert_all_close(y, Nx.dot(Nx.tensor([[1.0, 2.0, 3.0, 4.0]]), Nx.as_type(weight, :f32)))
  end

  # A large value that a traced function folds from static values is known, but the device
  # builds it. A long Loop that carries it runs as a while loop and does not unroll.
  test "a long Loop over a large folded value runs as a while loop" do
    int64 = &%TensorProto{name: &1, data_type: 7, dims: &2, raw_data: <<&3::64-signed-native>>}

    body = %GraphProto{
      name: "body",
      input: [value("i", [], 7), value("cond_in", [], 9), value("acc", [8192])],
      output: [value("cond_out", [], 9), value("acc_out", [8192])],
      node: [
        node("Identity", ["cond_in"], ["cond_out"]),
        node("Mul", ["acc", "acc"], ["square"]),
        node("Add", ["square", "acc"], ["acc_out"])
      ]
    }

    graph = %GraphProto{
      input: [value("x", [4])],
      output: [value("y")],
      initializer: [int64.("size", [1], 8192), int64.("trips", [], 64)],
      node: [
        node("ConstantOfShape", ["size"], ["zeros"]),
        node("Loop", ["trips", "", "zeros"], ["acc"],
          attribute: [%AttributeProto{name: "body", type: :GRAPH, g: body}]
        ),
        node("ReduceSum", ["acc"], ["total"], attribute: [int_attribute("keepdims", 0)]),
        node("Add", ["x", "total"], ["y"])
      ]
    }

    model = Transmuter.load!(write_model(graph), backend: @backend)
    x = Nx.iota({4}, type: :f32, backend: @backend)
    expr = Nx.Defn.debug_expr(&Transmuter.run(model, &1, &2)).(model.params, %{"x" => x})
    assert inspect(expr, limit: :infinity) =~ "while"
    assert_all_close(Transmuter.compile(model, %{"x" => x}, @compiler).(%{"x" => x})["y"], x)
  end

  # EyeLike builds its output from Nx.iota and reads only the shape of its input.
  test "an eager run builds the constants of an operator on the backend of the model" do
    graph = %GraphProto{
      input: [value("x", [3, 3])],
      output: [value("y")],
      node: [node("EyeLike", ["x"], ["y"])]
    }

    model = Transmuter.load!(write_model(graph), backend: @backend)
    x = Nx.iota({3, 3}, type: :f32, backend: @backend)
    %{"y" => y} = Transmuter.run(model, model.params, %{"x" => x})
    assert %Nx.Tensor{data: %EMLX.Backend{}} = y
    assert_all_close(y, Nx.eye(3))
  end

  defp node(op_type, input, output, opts \\ []) do
    struct!(
      %NodeProto{op_type: op_type, name: String.downcase(op_type), input: input, output: output},
      opts
    )
  end

  defp int_attribute(name, value), do: %AttributeProto{name: name, type: :INT, i: value}

  defp value(name, dims \\ nil, elem_type \\ 1) do
    shape =
      dims &&
        %TensorShapeProto{
          dim: Enum.map(dims, &%TensorShapeProto.Dimension{value: {:dim_value, &1}})
        }

    type = %TypeProto{
      value: {:tensor_type, %TypeProto.Tensor{elem_type: elem_type, shape: shape}}
    }

    %ValueInfoProto{name: name, type: type}
  end

  defp write_model(graph) do
    model = %ModelProto{
      ir_version: 10,
      opset_import: [%OperatorSetIdProto{domain: "", version: 17}],
      graph: %{graph | name: "g"}
    }

    path = Path.join(System.tmp_dir!(), "transmuter_#{System.unique_integer([:positive])}.onnx")
    {iodata, _size} = Protox.encode!(model)
    File.write!(path, iodata)
    path
  end

  defp load(repo, file, opts \\ []) do
    Transmuter.load!({:hf, repo, file}, [backend: @backend] ++ opts)
  end

  defp run(model, inputs, static_inputs \\ %{}) do
    Transmuter.compile(model, inputs, [static_inputs: static_inputs] ++ @compiler).(inputs)
  end

  # sin(i * 0.01) * 0.5, the same pattern that test/support/model_references.py feeds to ORT.
  defp pattern(shape) do
    {Tuple.product(shape)}
    |> Nx.iota(type: :f32, backend: @backend)
    |> Nx.multiply(0.01)
    |> Nx.sin()
    |> Nx.multiply(0.5)
    |> Nx.reshape(shape)
  end

  defp assert_all_close(tensor, expected) do
    expected = Nx.tensor(expected)
    tensor = Nx.backend_copy(tensor, Nx.BinaryBackend)

    assert Nx.to_number(Nx.all_close(tensor, expected, atol: 1.0e-4, rtol: 1.0e-4)) == 1,
           "expected #{inspect(Nx.to_list(expected))}, got #{inspect(Nx.to_list(tensor))}"
  end
end
