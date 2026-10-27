defmodule Transmuter.FusionTest do
  use ExUnit.Case, async: true

  alias Transmuter.Onnx.{
    AttributeProto,
    GraphProto,
    ModelProto,
    NodeProto,
    OperatorSetIdProto,
    TensorProto,
    ValueInfoProto
  }

  @moduletag :tmp_dir

  @x Nx.iota({2, 3, 8}, type: :f32) |> Nx.multiply(37.0) |> Nx.remainder(19.0) |> Nx.subtract(9.0)

  defp rms_norm(divide, opts \\ []) do
    [
      node("Pow", ["x", Keyword.get(opts, :exponent, "two")], ["square"]),
      node("ReduceMean", ["square"], ["mean"], axes: Keyword.get(opts, :axes, [-1]), keepdims: 1),
      node("Add", ["mean", "eps"], ["var"]),
      node("Sqrt", ["var"], ["sqrt"])
    ] ++ divide ++ [node("Mul", [Keyword.get(opts, :weight, "w"), "normed"], ["y"])]
  end

  @rms_constants %{
    "two" => Nx.tensor(2.0),
    "three" => Nx.tensor(3.0),
    "one" => Nx.tensor(1.0),
    "eps" => Nx.tensor(1.0e-5),
    "w" => Nx.linspace(0.5, 2.0, n: 8),
    "w2" => Nx.linspace(0.5, 2.0, n: 8) |> Nx.reshape({1, 8})
  }

  defp one_over_sqrt,
    do: [node("Div", ["one", "sqrt"], ["inv"]), node("Mul", ["inv", "x"], ["normed"])]

  test "RMSNorm becomes SimplifiedLayerNormalization", %{tmp_dir: dir} do
    for divide <- [
          one_over_sqrt(),
          [node("Reciprocal", ["sqrt"], ["inv"]), node("Mul", ["x", "inv"], ["normed"])],
          [node("Div", ["x", "sqrt"], ["normed"])]
        ] do
      model = write(dir, rms_norm(divide), @rms_constants, ["x"], ["y"])
      assert ops(model) == ["SimplifiedLayerNormalization"]
      assert_same_outputs(model, %{"x" => @x})
    end
  end

  test "a replacement lists the nodes that it replaces", %{tmp_dir: dir} do
    nodes =
      Enum.map(rms_norm(one_over_sqrt()), fn node ->
        %{node | name: "/norm/#{node.op_type}_#{hd(node.output)}"}
      end)

    model = write(dir, nodes, @rms_constants, ["x"], ["y"])
    [%{metadata_props: [entry]}] = Transmuter.load!(model).nodes

    assert entry.key == "transmuter.fused"

    assert entry.value |> String.split(", ") |> Enum.sort() ==
             Enum.sort(Enum.map(nodes, &"#{&1.op_type} #{&1.name}"))
  end

  test "LayerNorm becomes LayerNormalization", %{tmp_dir: dir} do
    # Opset 18 gives the axes of ReduceMean as an input.
    nodes = [
      node("ReduceMean", ["x", "axes"], ["mu"], keepdims: 1),
      node("Sub", ["x", "mu"], ["d"]),
      node("Pow", ["d", "two"], ["square"]),
      node("ReduceMean", ["square", "axes"], ["var"]),
      node("Add", ["var", "eps"], ["var_eps"]),
      node("Sqrt", ["var_eps"], ["sqrt"]),
      node("Div", ["d", "sqrt"], ["normed"]),
      node("Mul", ["normed", "gamma"], ["scaled"]),
      node("Add", ["beta", "scaled"], ["y"])
    ]

    constants = %{
      "axes" => Nx.tensor([-1], type: :s64),
      "two" => Nx.tensor(2.0),
      "eps" => Nx.tensor(1.0e-12),
      "gamma" => Nx.linspace(0.5, 2.0, n: 8),
      "beta" => Nx.linspace(-1.0, 1.0, n: 8)
    }

    model = write(dir, nodes, constants, ["x"], ["y"], 18)
    assert ops(model) == ["LayerNormalization"]
    assert_same_outputs(model, %{"x" => @x})
  end

  test "the attention core becomes AttentionCore", %{tmp_dir: dir} do
    # With a softmax over a middle axis, AttentionCore keeps the Nx operations.
    for axis <- [3, -1, 2] do
      nodes = [
        node("Transpose", ["k"], ["kt"], perm: [0, 1, 3, 2]),
        node("MatMul", ["q", "kt"], ["qk"]),
        node("Div", ["qk", "sqrt_d"], ["scores"]),
        node("Add", ["mask", "scores"], ["masked"]),
        node("Softmax", ["masked"], ["probs"], axis: axis),
        node("MatMul", ["probs", "v"], ["y"])
      ]

      model = write(dir, nodes, %{"sqrt_d" => Nx.tensor(8.0)}, ["q", "k", "v", "mask"], ["y"], 11)
      assert ops(model) == ["Transpose", "AttentionCore"]

      key = Nx.Random.key(0)
      {q, key} = Nx.Random.normal(key, shape: {2, 4, 3, 64})
      {k, key} = Nx.Random.normal(key, shape: {2, 4, 5, 64})
      {v, _key} = Nx.Random.normal(key, shape: {2, 4, 5, 64})
      mask = Nx.tensor([[0, 0, 0, -1.0e9, -1.0e9], [0, 0, 0, 0, 0]]) |> Nx.reshape({2, 1, 1, 5})

      assert_same_outputs(model, %{"q" => q, "k" => k, "v" => v, "mask" => mask})
    end
  end

  # Whisper and TrOCR exports put the heads in the batch, {B * N, S, d}. EMLX runs these
  # as the heads of a batch of one, with the mask of each head.
  test "an attention core with the heads in the batch becomes AttentionCore", %{tmp_dir: dir} do
    nodes = [
      node("Transpose", ["k"], ["kt"], perm: [0, 2, 1]),
      node("MatMul", ["q", "kt"], ["scores"]),
      node("Add", ["scores", "mask"], ["masked"]),
      node("Softmax", ["masked"], ["probs"], axis: 2),
      node("MatMul", ["probs", "v"], ["y"])
    ]

    model = write(dir, nodes, %{}, ["q", "k", "v", "mask"], ["y"], 13)
    assert ops(model) == ["Transpose", "AttentionCore"]

    key = Nx.Random.key(0)
    {q, key} = Nx.Random.normal(key, shape: {6, 3, 64})
    {k, key} = Nx.Random.normal(key, shape: {6, 5, 64})
    {v, _key} = Nx.Random.normal(key, shape: {6, 5, 64})
    mask = Nx.iota({6, 1, 5}) |> Nx.remainder(4) |> Nx.equal(3) |> Nx.multiply(-1.0e9)

    assert_same_outputs(model, %{"q" => q, "k" => k, "v" => v, "mask" => mask})
  end

  # The Whisper exports of Xenova reshape the probabilities to 4-D heads and back. A pair
  # that gives back the shape of the probabilities runs in the kernel. A pair that gives
  # another shape runs as the operators of the graph.
  test "a Reshape pair after the softmax runs in AttentionCore", %{tmp_dir: dir} do
    nodes = [
      node("Transpose", ["k"], ["kt"], perm: [0, 2, 1]),
      node("MatMul", ["q", "kt"], ["scores"]),
      node("Softmax", ["scores"], ["probs"], axis: 2),
      node("Reshape", ["probs", "split"], ["heads"]),
      node("Reshape", ["heads", "merge"], ["merged"]),
      node("MatMul", ["merged", "v"], ["y"])
    ]

    key = Nx.Random.key(2)
    {q, key} = Nx.Random.normal(key, shape: {6, 3, 64})
    {k, key} = Nx.Random.normal(key, shape: {6, 4, 64})
    {v, _key} = Nx.Random.normal(key, shape: {6, 4, 64})
    s64 = &Nx.tensor(&1, type: :s64)

    for {split, merge, v} <- [
          {[2, 3, 3, 4], [6, -1, 4], v},
          {[6, 12], [6, 4, 3], v[[.., 0..2]]}
        ] do
      constants = %{"split" => s64.(split), "merge" => s64.(merge)}
      model = write(dir, nodes, constants, ["q", "k", "v"], ["y"], 13)
      assert ops(model) == ["Transpose", "AttentionCore"]
      assert_same_outputs(model, %{"q" => q, "k" => k, "v" => v})
    end
  end

  test "a subgraph that computes another function stays", %{tmp_dir: dir} do
    variants = [
      rms_norm(one_over_sqrt(), exponent: "three"),
      rms_norm(one_over_sqrt(), axes: [1]),
      rms_norm(one_over_sqrt(), weight: "w2")
    ]

    for nodes <- variants do
      model = write(dir, nodes, @rms_constants, ["x"], ["y"])
      assert ops(model) == Enum.map(nodes, & &1.op_type)
    end

    # The square root is also a graph output.
    nodes = rms_norm(one_over_sqrt())
    model = write(dir, nodes, @rms_constants, ["x"], ["y", "sqrt"])
    assert ops(model) == Enum.map(nodes, & &1.op_type)

    # A Where whose fill gives the hidden keys some weight stays.
    nodes = [
      node("MatMul", ["q", "k"], ["qk"]),
      node("Where", ["keep", "qk", "fill"], ["scores"]),
      node("Softmax", ["scores"], ["probs"], axis: -1),
      node("MatMul", ["probs", "v"], ["y"])
    ]

    model = write(dir, nodes, %{"fill" => Nx.tensor(-1.0)}, ["q", "k", "v", "keep"], ["y"])
    assert ops(model) == Enum.map(nodes, & &1.op_type)
  end

  # GPT-2 exports compute the divisor of the scores from the head size, cast the scores to
  # f32, hide the later keys with a Where and add the padding mask. A divisor that is a graph
  # input is not known in the compiled function, so AttentionCore divides with Nx there.
  test "the attention core of GPT-2 becomes AttentionCore", %{tmp_dir: dir} do
    divisor = [
      node("Shape", ["q"], ["q_shape"]),
      node("Gather", ["q_shape", "three"], ["head_size"]),
      node("Cast", ["head_size"], ["head_size_f"], to: 1),
      node("Pow", ["head_size_f", "half"], ["sqrt_d"])
    ]

    core = [
      node("Transpose", ["k"], ["kt"], perm: [0, 1, 3, 2]),
      node("MatMul", ["q", "kt"], ["qk"]),
      node("Div", ["qk", "sqrt_d"], ["scaled"]),
      node("Cast", ["scaled"], ["scores"], to: 1),
      node("Where", ["keep", "scores", "fill"], ["kept"]),
      node("Add", ["kept", "mask"], ["masked"]),
      node("Softmax", ["masked"], ["probs"], axis: -1),
      node("Cast", ["probs"], ["probs_f"], to: 1),
      node("MatMul", ["probs_f", "v"], ["y"])
    ]

    constants = %{
      "three" => Nx.tensor(3, type: :s64),
      "half" => Nx.tensor(0.5),
      "fill" => Nx.Constants.min_finite(:f32)
    }

    key = Nx.Random.key(1)
    {q, key} = Nx.Random.normal(key, shape: {2, 4, 3, 64})
    {k, key} = Nx.Random.normal(key, shape: {2, 4, 5, 64})
    {v, _key} = Nx.Random.normal(key, shape: {2, 4, 5, 64})

    keep =
      Nx.less_equal(Nx.iota({1, 1, 3, 5}, axis: 3), Nx.add(Nx.iota({1, 1, 3, 5}, axis: 2), 2))

    mask = Nx.tensor([[0, 0, 0, -1.0e9, -1.0e9], [0, 0, 0, 0, 0]]) |> Nx.reshape({2, 1, 1, 5})
    inputs = %{"q" => q, "k" => k, "v" => v, "keep" => keep, "mask" => mask}

    model = write(dir, divisor ++ core, constants, ["q", "k", "v", "keep", "mask"], ["y"])
    assert Enum.sort(ops(model)) == ~w(AttentionCore Cast Gather Pow Shape Transpose)
    assert_same_outputs(model, inputs)

    model = write(dir, core, constants, ["q", "k", "v", "keep", "mask", "sqrt_d"], ["y"])
    assert ops(model) == ["Transpose", "AttentionCore"]
    assert_same_outputs(model, Map.put(inputs, "sqrt_d", Nx.tensor(8.0)))
  end

  # The padding row of a batch hides all its keys with the smallest f32 value, through an
  # additive mask as in the HF BERT exports or through a Where. The graph gives that row a
  # uniform softmax, and so must the kernel, which scales the mask on the GPU.
  test "a row whose keys are all hidden gets the softmax of the graph", %{tmp_dir: dir} do
    lowest = Nx.Constants.min_finite(:f32)

    additive = [
      node("Transpose", ["k"], ["kt"], perm: [0, 1, 3, 2]),
      node("MatMul", ["q", "kt"], ["qk"]),
      node("Mul", ["qk", "scale"], ["scores"]),
      node("Add", ["scores", "mask"], ["masked"]),
      node("Softmax", ["masked"], ["probs"], axis: -1),
      node("MatMul", ["probs", "v"], ["y"])
    ]

    kept = [
      node("Transpose", ["k"], ["kt"], perm: [0, 1, 3, 2]),
      node("MatMul", ["q", "kt"], ["qk"]),
      node("Where", ["keep", "qk", "fill"], ["kept"]),
      node("Softmax", ["kept"], ["probs"], axis: -1),
      node("MatMul", ["probs", "v"], ["y"])
    ]

    # 16 queries take the full kernel of MLX, which has the scale of the mask.
    key = Nx.Random.key(4)
    {q, key} = Nx.Random.normal(key, shape: {2, 4, 16, 64})
    {k, key} = Nx.Random.normal(key, shape: {2, 4, 16, 64})
    {v, _key} = Nx.Random.normal(key, shape: {2, 4, 16, 64})

    keep =
      Nx.less(Nx.iota({2, 1, 1, 16}, axis: 3), Nx.tensor([12, 0]) |> Nx.reshape({2, 1, 1, 1}))

    mask = Nx.select(keep, 0.0, lowest)
    constants = %{"scale" => Nx.tensor(0.125), "fill" => lowest}

    for {nodes, given} <- [{additive, %{"mask" => mask}}, {kept, %{"keep" => keep}}] do
      names = ["q", "k", "v" | Map.keys(given)]
      model = write(dir, nodes, constants, names, ["y"])
      assert ops(model) == ["Transpose", "AttentionCore"]
      assert_same_outputs(model, Map.merge(%{"q" => q, "k" => k, "v" => v}, given))
    end
  end

  defp ops(path), do: Enum.map(Transmuter.load!(path).nodes, & &1.op_type)

  # The fused model gives the outputs of the graph without the rewrite, on Nx.BinaryBackend
  # and compiled with the test backend. On EMLX, the compiled model runs the fused kernels.
  defp assert_same_outputs(path, inputs) do
    model = Transmuter.load!(path, fuse: false)
    expected = Transmuter.run(model, model.params, inputs)
    fused = Transmuter.load!(path)
    assert_close(Transmuter.run(fused, fused.params, inputs), expected)

    backend = Transmuter.TestBackend.backend()
    fused = Transmuter.load!(path, backend: backend)
    inputs = Map.new(inputs, fn {k, t} -> {k, Nx.backend_transfer(t, backend)} end)
    run = Transmuter.compile(fused, inputs, Transmuter.TestBackend.compiler_opts())
    assert_close(run.(inputs), expected)
  end

  defp assert_close(actual, expected) do
    for {name, t} <- expected do
      assert Nx.shape(actual[name]) == Nx.shape(t), name

      assert Nx.all_close(Nx.backend_copy(actual[name], Nx.BinaryBackend), t,
               atol: 1.0e-4,
               rtol: 1.0e-4
             )
             |> Nx.to_number() == 1,
             name
    end
  end

  defp write(dir, nodes, constants, inputs, outputs, opset \\ 17) do
    initializers =
      for {name, t} <- constants, Enum.any?(nodes, &(name in &1.input)) do
        %TensorProto{
          name: name,
          dims: Tuple.to_list(Nx.shape(t)),
          data_type: if(Nx.type(t) == {:s, 64}, do: 7, else: 1),
          raw_data: Nx.to_binary(t)
        }
      end

    graph = %GraphProto{
      name: "g",
      node: nodes,
      initializer: initializers,
      input: Enum.map(inputs, &%ValueInfoProto{name: &1}),
      output: Enum.map(outputs, &%ValueInfoProto{name: &1})
    }

    model = %ModelProto{
      ir_version: 8,
      graph: graph,
      opset_import: [%OperatorSetIdProto{domain: "", version: opset}]
    }

    path = Path.join(dir, "model_#{System.unique_integer([:positive])}.onnx")
    {iodata, _size} = Protox.encode!(model)
    File.write!(path, iodata)
    path
  end

  defp node(op_type, inputs, outputs, attrs \\ []) do
    attribute =
      for {name, value} <- attrs do
        case value do
          v when is_integer(v) -> %AttributeProto{name: "#{name}", type: :INT, i: v}
          v when is_list(v) -> %AttributeProto{name: "#{name}", type: :INTS, ints: v}
        end
      end

    %NodeProto{op_type: op_type, input: inputs, output: outputs, attribute: attribute}
  end
end
