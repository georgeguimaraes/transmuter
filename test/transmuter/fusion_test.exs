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

  @device (System.get_env("TRANSMUTER_TEST_DEVICE") ||
             if(match?({:unix, :darwin}, :os.type()), do: "gpu", else: "cpu"))
          |> String.to_existing_atom()

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

    # A scale that is not a constant stays outside of AttentionCore.
    nodes = [
      node("MatMul", ["q", "k"], ["qk"]),
      node("Div", ["qk", "scale"], ["scores"]),
      node("Softmax", ["scores"], ["probs"], axis: -1),
      node("MatMul", ["probs", "v"], ["y"])
    ]

    model = write(dir, nodes, %{}, ["q", "k", "v", "scale"], ["y"])
    assert ops(model) == Enum.map(nodes, & &1.op_type)
  end

  defp ops(path), do: Enum.map(Transmuter.load!(path).nodes, & &1.op_type)

  # The fused model gives the outputs of the graph without the rewrite, on Nx.BinaryBackend
  # and compiled with EMLX, which runs the fused kernels.
  defp assert_same_outputs(path, inputs) do
    model = Transmuter.load!(path, fuse: false)
    expected = Transmuter.run(model, model.params, inputs)
    fused = Transmuter.load!(path)
    assert_close(Transmuter.run(fused, fused.params, inputs), expected)

    backend = {EMLX.Backend, device: @device}
    fused = Transmuter.load!(path, backend: backend)
    inputs = Map.new(inputs, fn {k, t} -> {k, Nx.backend_transfer(t, backend)} end)
    run = Transmuter.compile(fused, inputs, compiler: EMLX, device: @device)
    assert_close(run.(inputs), expected)
  end

  defp assert_close(actual, expected) do
    for {name, t} <- expected do
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
