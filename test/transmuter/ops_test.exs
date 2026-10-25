defmodule Transmuter.OpsTest do
  use ExUnit.Case, async: true

  alias Transmuter.Ops.Transformer

  # Each fixture is one small ONNX graph around one operator variant, with reference
  # outputs from ONNX Runtime (graph optimizations off) or from the ONNX reference
  # evaluator. test/support/make_op_fixtures.py writes them, so these tests need no Python.
  @fixtures Path.expand("../fixtures/ops", __DIR__)

  @backend Transmuter.TestBackend.backend()
  @compiler Transmuter.TestBackend.compiler_opts()

  for dir <- @fixtures |> Path.join("*") |> Path.wildcard() |> Enum.sort() do
    @dir dir
    test Path.basename(dir) do
      assert_matches_reference(@dir)
    end
  end

  # A packed MatMulNBits weight holds its scales and zero points. Transmuter.EMLX.move gives
  # the node the packed weight as "B:packed", blanks its scales and zero points, and drops B,
  # the scales (a param) and the zero points (a static value) when no other node reads them.
  # A second move leaves the model as it is. The CPU device dequantizes the weight, so the
  # test runs where EMLX runs.
  if Code.ensure_loaded?(Transmuter.EMLX) do
    test "Transmuter.EMLX.move drops the scales and zero points that only packed weights read" do
      model = Transmuter.load!(Path.join(@fixtures, "MatMulNBits_b4_bs256_packed/model.onnx"))
      [node] = model.nodes
      assert {node.input, Map.has_key?(model.static, "zp")} == {["A", "B", "scales", "zp"], true}

      moved = Transmuter.EMLX.move(model, device: :cpu)
      assert hd(moved.nodes).input == ["A", "B:packed", "", ""]
      assert Map.keys(moved.params) == ["B:packed"]
      refute Map.has_key?(moved.static, "zp")
      assert moved.backend == {EMLX.Backend, device: :cpu}
      assert Nx.shape(moved.params["B:packed"]) == {40, 512}

      again = Transmuter.EMLX.move(moved, device: :cpu)
      assert {again.nodes, Map.keys(again.params)} == {moved.nodes, Map.keys(moved.params)}

      # The same B in an external file, as read!/2 defers it with a backend, packs the same.
      dir =
        Path.join(System.tmp_dir!(), "transmuter-deferred-#{System.unique_integer([:positive])}")

      File.mkdir_p!(dir)
      File.write!(Path.join(dir, "data"), Nx.to_binary(model.params["B"]))

      proto = %Transmuter.Onnx.TensorProto{
        name: "B",
        dims: [40, 2, 128],
        data_type: 2,
        data_location: :EXTERNAL,
        external_data: [%Transmuter.Onnx.StringStringEntryProto{key: "location", value: "data"}]
      }

      deferred = %{
        model
        | params: %{model.params | "B" => Transmuter.DeferredTensor.new(proto, dir)}
      }

      from_file = Transmuter.EMLX.move(deferred, device: :cpu)
      assert hd(from_file.nodes).input == ["A", "B:packed", "", ""]
      assert Nx.to_list(from_file.params["B:packed"]) == Nx.to_list(moved.params["B:packed"])

      # Identity nodes read B and the scales, as GatherBlockQuantized reads those of tied
      # embeddings. They keep the raw values. The device of the tests packs the weight, and
      # the compiled graph gives Y and the raw values.
      readers =
        for name <- ["B", "scales"],
            do: %{
              node
              | op_type: "Identity",
                domain: "",
                input: [name],
                output: [name <> "_copy"]
            }

      shared = %{model | nodes: [node | readers], outputs: ["Y", "B_copy", "scales_copy"]}
      kept = Transmuter.EMLX.move(shared, device: :cpu)
      assert hd(kept.nodes).input == ["A", "B:packed", "", ""]
      assert Enum.sort(Map.keys(kept.params)) == ["B", "B:packed", "scales"]
      assert Nx.shape(kept.params["B"]) == {40, 2, 128}

      if Transmuter.TestBackend.compiler() == EMLX do
        {EMLX.Backend, device_opts} = @backend
        packed = Transmuter.EMLX.move(shared, device_opts)
        a = Nx.iota({1, 512}, type: :f32, backend: @backend) |> Nx.divide(512)
        run = Transmuter.compile(packed, %{"A" => a}, @compiler)
        out = run.(%{"A" => a})
        assert Nx.shape(out["Y"]) == {1, 40}
        assert Nx.to_list(out["B_copy"]) == Nx.to_list(model.params["B"])
        assert Nx.to_list(out["scales_copy"]) == Nx.to_list(model.params["scales"])
      end
    end
  end

  # A move gives a MatMulNBits node its {N, K} weight once, on the device, in place of the
  # unpacking in the graph. The cases: 4-bit with packed zero points (the fixture), 4-bit
  # without zero points (the default zero point), 8-bit with zero points, and f16 scales with
  # compute_type :f32, each from the same fixture in memory. The last case dequantizes in f32,
  # as the graph does after read!/2 upcasts the scales, and not in f16. On EXLA the weight is
  # the dequantize of the graph, so the outputs are equal to the bit. On EMLX the weight is an
  # MLX quantized tensor, or MLX's dequantize `q * scale + bias` with the bias `-zero * scale`
  # on the CPU, so the last bits differ.
  test "a moved MatMulNBits gives the outputs of the dequantize in the graph" do
    model = Transmuter.load!(Path.join(@fixtures, "MatMulNBits_b4_bs256_packed/model.onnx"))
    [node] = model.nodes
    key = Nx.Random.key(3)
    {a, key} = Nx.Random.normal(key, shape: {2, 3, 512})
    {b8, key} = Nx.Random.randint(key, 0, 256, shape: {40, 2, 256})
    {zp8, _key} = Nx.Random.randint(key, 0, 256, shape: {80})
    {b8, zp8} = {Nx.as_type(b8, :u8), Nx.as_type(zp8, :u8)}
    bits8 = Enum.map(node.attribute, fn a -> if a.name == "bits", do: %{a | i: 8}, else: a end)

    cases = [
      {"4-bit with zero points", model},
      {"4-bit without zero points",
       %{
         model
         | nodes: [%{node | input: ["A", "B", "scales", ""]}],
           static: Map.delete(model.static, "zp")
       }},
      {"8-bit with zero points",
       %{
         model
         | nodes: [%{node | attribute: bits8}],
           params: %{model.params | "B" => b8},
           static: %{model.static | "zp" => zp8}
       }},
      {"f16 scales in f32",
       %{
         model
         | params: %{model.params | "scales" => Nx.as_type(model.params["scales"], :f16)},
           compute_type: :f32
       }}
    ]

    inputs = %{"A" => Nx.backend_transfer(a, @backend)}
    {backend, _opts} = @backend

    for {name, model} <- cases do
      params =
        Map.new(model.params, fn {k, t} ->
          {k, t |> Nx.backend_transfer(@backend) |> Transmuter.upcast(model.compute_type)}
        end)

      in_graph = %{model | params: params, backend: @backend}
      moved = Transmuter.move(model, @backend)
      assert hd(moved.nodes).input == ["A", "B:packed", "", ""], name
      assert Map.keys(moved.params) == ["B:packed"], name
      weight = moved.params["B:packed"]
      assert Nx.shape(weight) == {40, 512}, name
      assert %Nx.Tensor{data: %{__struct__: ^backend}} = weight, name

      [expected, got] =
        for m <- [in_graph, moved] do
          Transmuter.compile(m, inputs, @compiler).(inputs)["Y"] |> Nx.backend_transfer()
        end

      if Transmuter.TestBackend.compiler() == EXLA do
        assert Nx.type(weight) == {:f, 32}, name
        assert Nx.to_binary(got) == Nx.to_binary(expected), name
      else
        assert Nx.to_number(Nx.all_close(got, expected, atol: 1.0e-4)) == 1, name
      end
    end
  end

  # ONNX allows external data without a length, which goes to the end of the file. With a
  # backend, the loader defers the tensor and knows its shape and type before it reads it.
  test "an external tensor without a length reads to the end of the file" do
    dir =
      Path.join(System.tmp_dir!(), "transmuter-external-#{System.unique_integer([:positive])}")

    File.mkdir_p!(dir)
    File.write!(Path.join(dir, "data"), <<0::32, 1.5::float-32-little, 2.5::float-32-little>>)

    proto = %Transmuter.Onnx.TensorProto{
      name: "w",
      dims: [2],
      data_type: 1,
      data_location: :EXTERNAL,
      external_data: [
        %Transmuter.Onnx.StringStringEntryProto{key: "location", value: "data"},
        %Transmuter.Onnx.StringStringEntryProto{key: "offset", value: "4"}
      ]
    }

    assert Nx.to_list(Transmuter.Ops.tensor(proto, dir)) == [1.5, 2.5]
    assert Transmuter.Ops.tensor_shape_type(proto) == {{2}, {:f, 32}}
  end

  # A budget of 5 rows of scores gives 3 chunks of 5 rows. The 13 query rows then run as
  # the chunks at rows 0, 5 and 8. The float mask has a row for each query, the hidden keys
  # are the same for all queries, and the causal mask hides the keys after each query. Two
  # query heads share each key head. v is f16, so the chunks and the output buffer must
  # take the type of the probabilities, f32, as the values of all rows at once do.
  test "the attention core in chunks of query rows gives the values of all rows at once" do
    {b, heads, kv_heads, s, t, head_size} = {2, 4, 2, 13, 17, 8}
    key = Nx.Random.key(1)
    {q, key} = Nx.Random.normal(key, shape: {b, heads, s, head_size})
    {k, key} = Nx.Random.normal(key, shape: {b, kv_heads, t, head_size})
    {v, key} = Nx.Random.normal(key, shape: {b, kv_heads, t, head_size}, type: :f16)
    {bias, _key} = Nx.Random.normal(key, shape: {1, 1, s, t})
    lengths = Nx.reshape(Nx.tensor([17, 11]), {b, 1, 1, 1})
    hidden = Nx.greater_equal(Nx.iota({b, 1, 1, t}, axis: 3), lengths)
    opts = %{scale: 0.3, softcap: 0.0, bias: bias, hidden: hidden, causal: true}
    budget = b * heads * t * 4 * 5
    attend = &Transformer.attend_rows(q, k, v, opts, &1)
    assert attend.(budget) == attend.(2 ** 40)

    jit = fn budget ->
      Nx.Defn.jit_apply(
        &Transformer.attend_rows(&1, &2, &3, %{opts | bias: &4, hidden: &5}, budget),
        Enum.map([q, k, v, bias, hidden], &Nx.backend_transfer(&1, @backend)),
        @compiler
      )
    end

    assert Nx.to_number(Nx.all_close(jit.(budget), jit.(2 ** 40), atol: 1.0e-5)) == 1
  end

  # MLX 0.32.0 has fused kernels for some head sizes. A head size of 72 pads to 80 for the
  # "full" kernel, 64 takes it as it is, and 256 with 16 queries has no fused size within
  # 1.25 times, so the Nx core runs in chunks. Each case gives the values of the Nx core.
  test "attention takes the fused MLX kernel, a padded head size, or the Nx core" do
    key = Nx.Random.key(2)
    opts = %{scale: 0.1, softcap: 0.0, bias: nil, hidden: nil, causal: false}

    for {head_size, kind} <- [{72, {:pad, 80}}, {64, :fused}, {256, :plain}] do
      q_shape = {1, 4, 16, head_size}
      k_shape = {1, 2, 24, head_size}
      assert Transmuter.Ops.mlx_attention(q_shape, k_shape, head_size, :none) == kind

      {q, key} = Nx.Random.normal(key, shape: q_shape)
      {k, key} = Nx.Random.normal(key, shape: k_shape)
      {v, _key} = Nx.Random.normal(key, shape: k_shape)
      expected = Transformer.attend_rows(q, k, v, opts, 2 ** 40)
      [q, k, v] = Enum.map([q, k, v], &Nx.backend_transfer(&1, @backend))
      got = q |> Transformer.attend(k, v, opts) |> Nx.backend_transfer(Nx.BinaryBackend)
      assert Nx.shape(got) == Nx.shape(expected)
      assert Nx.to_number(Nx.all_close(got, expected, atol: 1.0e-4)) == 1, inspect(kind)
    end
  end

  # The scores of all rows fit in the budget, also with a zero size, so there are no chunks
  # and no division. Otherwise the rows split into the fewest chunks whose scores fit, with
  # the same number of rows in each chunk, and at least one row.
  test "the chunks of query rows are balanced and fit in the budget" do
    mib = 256 * 1024 * 1024
    assert Transformer.chunk_rows({2, 4, 13, 17}, {:f, 32}, 2 * 4 * 13 * 17 * 4) == nil

    for shape <- [{0, 4, 13, 17}, {2, 0, 13, 17}, {2, 4, 0, 17}, {2, 4, 13, 0}] do
      assert Transformer.chunk_rows(shape, {:f, 32}, 0) == nil
    end

    assert Transformer.chunk_rows({2, 4, 13, 17}, {:f, 32}, 2 * 4 * 17 * 4 * 5) == {3, 5}
    # Ministral 3 at 1540x1540: 346 rows of 16 heads and 12100 keys fit in 256 MiB.
    assert Transformer.chunk_rows({1, 16, 12_100, 12_100}, {:f, 32}, mib) == {35, 346}
    assert Transformer.chunk_rows({1, 16, 12_100, 12_100}, {:f, 64}, mib) == {70, 173}
    assert Transformer.chunk_rows({1, 1, 3, 100}, {:f, 32}, 10) == {3, 1}
  end

  defp assert_matches_reference(dir) do
    meta = dir |> Path.join("meta.json") |> File.read!() |> JSON.decode!()

    # A "half" fixture runs f16 and bf16 in their own type, as a model loaded without
    # :compute_type. The others run them in f32.
    compute_type = if meta["half"], do: nil, else: :f32

    model =
      Transmuter.load!(Path.join(dir, meta["onnx_path"]),
        backend: @backend,
        compute_type: compute_type
      )

    # Strings, sequences and empty optionals cannot be compiled inputs, so they are static.
    {static, inputs} =
      Enum.split_with(
        meta["inputs"],
        &(&1["name"] in meta["static_inputs"] or not Map.has_key?(&1, "file"))
      )

    inputs = Map.new(inputs, &{&1["name"], read(dir, &1, @backend)})
    static_inputs = Map.new(static, &{&1["name"], read(dir, &1, Nx.BinaryBackend)})

    outputs =
      if meta["eager"] do
        Transmuter.run(model, model.params, inputs, static_inputs)
      else
        Transmuter.compile(model, inputs, [static_inputs: static_inputs] ++ @compiler).(inputs)
      end

    for spec <- meta["outputs"], not zero_size?(spec) or Map.has_key?(outputs, spec["name"]) do
      expected = read(dir, spec, Nx.BinaryBackend)
      actual = Map.fetch!(outputs, spec["name"])

      if distribution = meta["distribution"],
        do: assert_distribution(spec["name"], actual, expected, distribution),
        else: assert_same(spec["name"], actual, expected, meta["atol"])
    end
  end

  # Random outputs match the reference only in shape and dtype. Their statistics must be
  # within 0.05 standard deviations (0.025 for rates), about 5 standard errors for
  # 10,000 samples.
  defp assert_distribution(name, actual, expected, distribution) do
    actual = Nx.backend_copy(actual, Nx.BinaryBackend)
    assert {Nx.shape(actual), Nx.type(actual)} == {Nx.shape(expected), Nx.type(expected)}, name
    values = Nx.as_type(actual, :f64)
    stat = &Nx.to_number(&1.(values))

    case distribution do
      %{"frequencies" => rows} ->
        for {row, b} <- Enum.with_index(rows), {frequency, class} <- Enum.with_index(row) do
          actual_frequency = values[b] |> Nx.equal(class) |> Nx.mean() |> Nx.to_number()
          assert_in_delta actual_frequency, frequency, 0.025, "#{name}[#{b}] class #{class}"
        end

      %{"rate" => rate} ->
        assert_in_delta stat.(&Nx.mean/1), rate, 0.025, name

      %{"mean" => mean, "std" => std} ->
        assert_in_delta stat.(&Nx.mean/1), mean, 0.05 * std, "#{name} mean"
        assert_in_delta stat.(&Nx.standard_deviation/1), std, 0.05 * std, "#{name} std"

        if low = distribution["low"] do
          {min, max} = {stat.(&Nx.reduce_min/1), stat.(&Nx.reduce_max/1)}
          assert min >= low and max < distribution["high"], "#{name} range"
        end
    end
  end

  defp zero_size?(spec), do: 0 in Map.get(spec, "shape", [1])

  defp assert_same(name, actual, {:sequence, expected}, atol) do
    assert is_list(actual), "#{name}: expected a sequence, got #{inspect(actual)}"
    assert length(actual) == length(expected), "#{name}: sequence length"

    for {a, e, i} <- Enum.zip([actual, expected, 0..(length(expected) - 1)//1]),
        do: assert_same("#{name}[#{i}]", a, e, atol)
  end

  defp assert_same(name, actual, expected, _atol) when not is_struct(expected, Nx.Tensor),
    do: assert(actual == expected, "#{name}: #{inspect(actual)}, expected #{inspect(expected)}")

  defp assert_same(name, actual, expected, atol) do
    actual = Nx.backend_copy(actual, Nx.BinaryBackend)

    assert Nx.shape(actual) == Nx.shape(expected),
           "#{name}: shape #{inspect(Nx.shape(actual))}, expected #{inspect(Nx.shape(expected))}"

    {max_diff, tolerance} = difference(actual, expected)

    tolerance =
      cond do
        atol -> atol
        Nx.Type.integer?(Nx.type(expected)) -> 0
        true -> tolerance
      end

    assert max_diff <= tolerance, "#{name}: max difference #{max_diff}, tolerance #{tolerance}"
  end

  # Largest difference, with equal infinities and matching NaNs counted as equal, and
  # the tolerance 1e-4 + 1e-3 * max |expected| over the finite expected values. Integer
  # outputs must be equal, unless the fixture sets atol.
  defp difference(actual, expected) do
    actual = Nx.as_type(actual, :f64)
    expected = Nx.as_type(expected, :f64)
    expected_nan = Nx.is_nan(expected)

    same =
      Nx.logical_or(Nx.logical_and(expected_nan, Nx.is_nan(actual)), Nx.equal(actual, expected))

    diff =
      Nx.select(same, 0, Nx.abs(Nx.subtract(actual, expected)))
      |> Nx.reduce_max()
      |> Nx.to_number()

    finite = Nx.logical_not(Nx.logical_or(expected_nan, Nx.is_infinity(expected)))
    scale = Nx.select(finite, Nx.abs(expected), 0) |> Nx.reduce_max() |> Nx.to_number()
    {if(diff == :nan, do: :infinity, else: diff), 1.0e-4 + 1.0e-3 * scale}
  end

  defp read(_dir, %{"none" => true}, _backend), do: :none

  defp read(dir, %{"sequence" => elements}, backend),
    do: {:sequence, Enum.map(elements, &read(dir, &1, backend))}

  defp read(_dir, %{"dtype" => "string", "shape" => shape, "values" => values}, _backend),
    do: {:strings, List.to_tuple(shape), values}

  defp read(dir, %{"file" => file, "dtype" => type, "shape" => shape}, backend) do
    if 0 in shape do
      {:empty, List.to_tuple(shape)}
    else
      dir
      |> Path.join(file)
      |> File.read!()
      |> Nx.from_binary(String.to_atom(type), backend: backend)
      |> Nx.reshape(List.to_tuple(shape))
    end
  end
end
