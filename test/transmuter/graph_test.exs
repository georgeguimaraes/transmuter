defmodule Transmuter.GraphTest do
  use ExUnit.Case, async: true

  alias Transmuter.Graph

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

  @backend Transmuter.TestBackend.backend()
  @compiler Transmuter.TestBackend.compiler_opts()

  test "placement gives where run/4 runs a node" do
    for {op_type, static, placement} <- [
          {"If", [true], :control},
          {"Loop", [false, true], :control},
          {"Scan", [true], :control},
          {"SequenceMap", [false], :control},
          {"RandomNormalLike", [true], :random},
          {"Add", [true, true], :host},
          {"Constant", [], :host},
          {"Add", [true, false], :device}
        ] do
      assert Transmuter.placement(%NodeProto{op_type: op_type}, static) == placement,
             "#{op_type} #{inspect(static)}"
    end
  end

  test "without inputs, only the static values of the model are static" do
    model = %{
      nodes: [
        node("Shape", ["x"], ["s"], "/Shape"),
        node("Add", ["s", "one"], ["t"], "/Add"),
        node("Mul", ["one", "one"], ["two"], "/Mul")
      ],
      static: %{"one" => Nx.tensor(1)}
    }

    assert [shape, add, mul] = Graph.overlay(model).nodes
    assert {shape.placement, add.placement, mul.placement} == {:device, :device, :host}
    assert add.literals == ["one"]
  end

  # The body input x hides the outer x, and the body initializer k is static.
  test "a subgraph resolves its inputs and initializers before the outer values" do
    k = %TensorProto{name: "k", dims: [], data_type: 1, raw_data: <<2.0::float-32-little>>}

    body = %GraphProto{
      node: [node("Neg", ["x"], ["y"], "/body/Neg"), node("Mul", ["k", "k"], ["kk"], "/body/Mul")],
      input: Enum.map(["i", "cond", "x"], &%ValueInfoProto{name: &1}),
      output: Enum.map(["cond", "y"], &%ValueInfoProto{name: &1}),
      initializer: [k]
    }

    model = %{
      nodes: [
        node("Relu", ["in"], ["x"], "/Relu"),
        %{
          node("Loop", ["n", "", "x"], ["out"], "/Loop")
          | attribute: [%AttributeProto{name: "body", type: :GRAPH, g: body}]
        }
      ],
      static: %{}
    }

    assert [relu, loop, neg, mul] = Graph.overlay(model).nodes
    assert Enum.map([relu, loop, neg, mul], & &1.id) == [0, 1, 2, 3]
    assert neg.reads == []
    assert loop.reads == [relu.id, neg.id]
    assert mul.placement == :host
  end

  test "the overlay gives the attributes of a node as plain values" do
    small = %TensorProto{
      dims: [2],
      data_type: 7,
      raw_data: <<3::signed-64-little, 4::signed-64-little>>
    }

    large = %TensorProto{dims: [4, 8], data_type: 1, raw_data: :binary.copy(<<0::32>>, 32)}
    scalar = %TensorProto{dims: [], data_type: 1, raw_data: <<1.5::float-32-little>>}

    attributes = [
      %AttributeProto{name: "axis", type: :INT, i: -1},
      %AttributeProto{name: "perm", type: :INTS, ints: [0, 2, 1]},
      %AttributeProto{name: "alpha", type: :FLOAT, f: 0.5},
      %AttributeProto{name: "scales", type: :FLOATS, floats: [1.0, 2.0]},
      %AttributeProto{name: "mode", type: :STRING, s: "linear"},
      %AttributeProto{name: "bytes", type: :STRING, s: <<255>>},
      %AttributeProto{name: "names", type: :STRINGS, strings: ["a", "b"]},
      %AttributeProto{name: "small", type: :TENSOR, t: small},
      %AttributeProto{name: "scalar", type: :TENSOR, t: scalar},
      %AttributeProto{name: "large", type: :TENSOR, t: large},
      %AttributeProto{name: "tensors", type: :TENSORS, tensors: [small, large]},
      %AttributeProto{name: "then_branch", type: :GRAPH, g: %GraphProto{name: "then"}},
      %AttributeProto{name: "else_branch", type: :GRAPH, g: %GraphProto{}},
      %AttributeProto{name: "graphs", type: :GRAPHS, graphs: [%GraphProto{name: "g0"}]}
    ]

    model = %{
      nodes: [%{node("Custom", [], ["y"], "/Custom") | attribute: attributes}],
      static: %{}
    }

    [%{attributes: values}] = Graph.overlay(model).nodes

    assert values == %{
             "axis" => -1,
             "perm" => [0, 2, 1],
             "alpha" => 0.5,
             "scales" => [1.0, 2.0],
             "mode" => "linear",
             "bytes" => "<<255>>",
             "names" => ["a", "b"],
             "small" => [3, 4],
             "scalar" => 1.5,
             "large" => %{shape: [4, 8], type: "f32"},
             "tensors" => [[3, 4], %{shape: [4, 8], type: "f32"}],
             "then_branch" => "then",
             "else_branch" => "else_branch",
             "graphs" => ["g0"]
           }
  end

  test "the tree groups the nodes by name scope" do
    branch = %GraphProto{
      node: [node("Neg", ["m"], ["t"], "/b/Neg")],
      output: [%ValueInfoProto{name: "t"}]
    }

    model = %{
      nodes: [
        node("Constant", [], ["w"], "/a/l0/Constant"),
        node("MatMul", ["in", "w"], ["mm"], "/a/l0/x/q_proj"),
        node("Add", ["mm", "w"], ["add"], "/a/l0/x/Add_1"),
        node("Mul", ["add", "add"], ["m"], "/a/l1/Mul"),
        node("Relu", ["m"], ["top"], "Top"),
        %{
          node("If", ["c"], ["out"], "/cond/If")
          | attribute: [%AttributeProto{name: "then_branch", type: :GRAPH, g: branch}]
        },
        node("Constant", [], ["z"], "/zeros/Constant")
      ],
      static: %{}
    }

    tree = model |> Graph.overlay() |> Graph.tree()

    assert outline(tree) ==
             {"/", 6,
              [
                {"/a", 3,
                 [{"/a/l0", 2, [{"/a/l0/x", 2, ["MatMul", "Add"]}]}, {"/a/l1", 1, ["Mul"]}]},
                "Relu",
                {"/cond", 2,
                 [
                   "If",
                   {"/cond/If", 1,
                    [{"/cond/If/then_branch", 1, [{"/cond/If/then_branch/b", 1, ["Neg"]}]}]}
                 ]},
                {"/zeros", 0, []}
              ]}

    assert tree.counts.constants == 2
    [a | _] = tree.children
    assert hd(a.children).counts.constants == 1
    [mat_mul, add] = a |> find("/a/l0/x") |> Map.fetch!(:children)
    assert {mat_mul.label, add.label} == {"q_proj", nil}
    assert add.reads == [mat_mul.id, "n0"]

    tree = model |> Graph.overlay() |> Graph.tree(constants: true)
    assert tree.counts.constants == 0
    assert {"/zeros", 1, ["Constant"]} = tree |> outline() |> elem(2) |> List.last()
  end

  test "a scope with many direct nodes splits them in groups of equal size" do
    tree = fn count ->
      nodes = for i <- 1..count, do: node("Neg", ["v#{i - 1}"], ["v#{i}"], "Neg_#{i}")
      %{nodes: nodes, static: %{}} |> Graph.overlay() |> Graph.tree()
    end

    assert [{"/ops 1-30", 30, _}, {"/ops 31-60", 30, _}, {"/ops 61-90", 30, _}] =
             tree.(90) |> outline() |> elem(2)

    assert [{"/ops 1-21", 21, _}, {"/ops 22-41", 20, _}] = tree.(41) |> outline() |> elem(2)
    assert tree.(40) |> outline() |> elem(2) |> length() == 40

    assert tree.(90) |> find("/ops 31-60") |> Map.fetch!(:children) |> hd() |> Map.fetch!(:name) ==
             "Neg_31"
  end

  test "a name that the exporter generated from the op type gives no label" do
    for {op_type, name, generated?} <- [
          {"Shape", "Shape", true},
          {"Shape", "Shape_6488", true},
          {"Shape", "q_shape", false},
          {"MatMul", "MatMul_1x", false},
          {"Add", "Mul_2", false}
        ] do
      assert Graph.generated_name?(op_type, name) == generated?, "#{op_type} #{name}"
    end
  end

  test "with inputs, the overlay gives the segments and records each node" do
    model = load_fixture("Break_two_breaks")
    x = Nx.tensor([1.0, 0.5, 0.25, 1.0, 0.75], backend: @backend)
    overlay = Graph.overlay(model, inputs: %{"x" => x}, compile: @compiler)

    # The Range output is static, so the NonZero after it folds on the host and does not break.
    assert [%{start: 0, stop: 2, compiled: true}, %{start: 3, stop: :end, reason: nil}] =
             overlay.segments

    assert [
             {"ReduceSum", :device, 0, nil},
             {"Cast", :device, 0, nil},
             {"Range", _, nil, %{reason: "a value that decides a shape", after: 0}},
             {"Cast", :host, 1, nil},
             {"Greater", :host, 1, nil},
             {"NonZero", :host, 1, nil},
             {"Cast", :host, 1, nil}
           ] = Enum.map(overlay.nodes, &{&1.op_type, &1.placement, &1.segment, &1.break})
  end

  # The output of Shape is static, also for a traced input. ConstantOfShape folds on the host,
  # but its 6400 values go back to the device, so they are not static.
  test "with inputs, the overlay keeps whether outputs are static and their shapes" do
    model = load_fixture("Fold_large_static")
    x = Nx.iota({80, 80}, type: :f32, backend: @backend)
    nodes = Graph.overlay(model, inputs: %{"x" => x}, compile: @compiler).nodes
    by_op = Map.new(nodes, &{&1.op_type, &1})

    assert {by_op["Shape"].placement, by_op["Shape"].static} == {:device, true}
    assert {by_op["ConstantOfShape"].placement, by_op["ConstantOfShape"].static} == {:host, false}
    assert by_op["Shape"].shapes == [%{shape: [2], type: "s64"}]
    assert by_op["ConstantOfShape"].shapes == [%{shape: [6400], type: "f32"}]
  end

  # Both branches compute "out", and only the then branch runs.
  test "with inputs, a node of a branch that did not run has no placement" do
    model = load_fixture("Break_inside_If")
    x = Nx.tensor([1.0, 2.0, 3.0, 4.0], backend: @backend)
    nodes = Graph.overlay(model, inputs: %{"x" => x}, compile: @compiler).nodes
    by_op = Map.new(nodes, &{&1.op_type, &1.placement})

    assert by_op["Range"] != nil
    assert by_op["Constant"] == nil
  end

  test "with inputs, equal nodes in the two branches of an If have their own records", %{
    tmp_dir: dir
  } do
    branch = fn name ->
      %GraphProto{
        name: name,
        node: [node("Neg", ["x"], ["out"], "/Neg")],
        output: [%ValueInfoProto{name: "out"}]
      }
    end

    graph = %GraphProto{
      name: "g",
      node: [
        %{
          node("If", ["c"], ["y"], "/If")
          | attribute: [
              %AttributeProto{name: "then_branch", type: :GRAPH, g: branch.("then")},
              %AttributeProto{name: "else_branch", type: :GRAPH, g: branch.("else")}
            ]
        }
      ],
      input: [%ValueInfoProto{name: "x"}, %ValueInfoProto{name: "c"}],
      output: [%ValueInfoProto{name: "y"}]
    }

    path = write_model(dir, graph)

    inputs = %{
      "x" => Nx.tensor([1.0, 2.0], backend: @backend),
      "c" => Nx.tensor([1], type: :u8, backend: @backend)
    }

    nodes =
      Graph.overlay(Transmuter.load!(path, backend: @backend), inputs: inputs, compile: @compiler).nodes

    assert [{"If", _}, {"Neg", then_placement}, {"Neg", nil}] =
             Enum.map(nodes, &{&1.op_type, &1.placement})

    assert then_placement != nil
  end

  # The Loop breaks, and the nodes of its body run in the break.
  test "with inputs, the nodes in the subgraph of a break node have the break" do
    model = load_fixture("Break_Loop_outer_sequence")

    inputs = %{
      "sizes" => Nx.tensor([2, 3, 1], backend: @backend),
      "x" => Nx.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], backend: @backend)
    }

    nodes = Graph.overlay(model, inputs: inputs, compile: @compiler).nodes
    [loop] = Enum.filter(nodes, &(&1.op_type == "Loop"))
    body = Enum.filter(nodes, &(&1.position == nil))

    assert %{reason: _, after: _} = loop.break
    assert body != []

    for node <- body do
      assert {node.segment, node.break, node.placement != nil} == {nil, loop.break, true},
             node.op_type
    end

    tree = Graph.tree(%{nodes: nodes})
    assert tree.counts.breaks == 1
  end

  # A Range with a delta of 0 raises when the break runs, after the first segment compiled.
  test "a call that raises leaves no compiled segment and no table", %{tmp_dir: dir} do
    zero = %TensorProto{name: "zero", dims: [], data_type: 7, raw_data: <<0::signed-64-little>>}

    graph = %GraphProto{
      name: "g",
      node: [
        %{
          node("ReduceSum", ["x"], ["s"], "/ReduceSum")
          | attribute: [%AttributeProto{name: "keepdims", type: :INT, i: 0}]
        },
        %{
          node("Cast", ["s"], ["n"], "/Cast")
          | attribute: [%AttributeProto{name: "to", type: :INT, i: 7}]
        },
        node("Range", ["zero", "n", "d"], ["y"], "/Range")
      ],
      initializer: [zero],
      input: [%ValueInfoProto{name: "x"}, %ValueInfoProto{name: "d"}],
      output: [%ValueInfoProto{name: "y"}]
    }

    path = write_model(dir, graph)
    model = Transmuter.load!(path, backend: @backend)

    inputs = %{
      "x" => Nx.tensor([1.0, 2.0, 3.0], backend: @backend),
      "d" => Nx.tensor(0, backend: @backend)
    }

    before = leftovers()

    assert_raise RuntimeError, ~r/Range/, fn ->
      Graph.overlay(model, inputs: inputs, compile: @compiler)
    end

    assert leftovers() == before
  end

  test "the options of Transmuter.Graph do not go to the compiler" do
    model = %{nodes: [], static: %{}}

    assert_raise ArgumentError, ~r/unknown keys \[:compiler\]/, fn ->
      Graph.overlay(model, compiler: EXLA)
    end

    assert_raise ArgumentError, fn -> Graph.tree(Graph.overlay(model), depth: 2) end
  end

  test "each script of the view is in priv" do
    for script <- Graph.scripts(), do: assert(File.exists?(Graph.asset(script)), script)
  end

  # The compiled segments of this process in :persistent_term and the tables of GraphBreaks
  # that it owns.
  defp leftovers do
    terms =
      for {{Transmuter.Shared, {owner, _ref}, _key}, _term} <- :persistent_term.get(),
          owner == self(),
          do: 1

    tables =
      for table <- :ets.all(),
          :ets.info(table, :name) == Transmuter.GraphBreaks,
          :ets.info(table, :owner) == self(),
          do: table

    {length(terms), length(tables)}
  end

  defp write_model(dir, graph) do
    path = Path.join(dir, "model_#{System.unique_integer([:positive])}.onnx")

    model = %ModelProto{
      ir_version: 8,
      graph: graph,
      opset_import: [%OperatorSetIdProto{version: 17}]
    }

    {iodata, _size} = Protox.encode!(model)
    File.write!(path, iodata)
    path
  end

  # {scope, node count, children} for a group, and the op type for a node.
  defp outline(group = %{kind: "group"}),
    do: {group.scope, group.counts.nodes, Enum.map(group.children, &outline/1)}

  defp outline(node), do: node.op_type

  defp find(group, scope) do
    if group.scope == scope,
      do: group,
      else: Enum.find_value(group.children, &(&1.kind == "group" and find(&1, scope)))
  end

  defp load_fixture(name) do
    Transmuter.load!(Path.expand("../fixtures/ops/#{name}/model.onnx", __DIR__),
      backend: @backend
    )
  end

  defp node(op_type, inputs, outputs, name) do
    %NodeProto{op_type: op_type, input: inputs, output: outputs, name: name}
  end
end
