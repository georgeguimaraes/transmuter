defmodule Transmuter.Fusion do
  @moduledoc """
  Replaces normalizations and attention cores that an export writes as separate operators with one operator.

  Many exports, for example from Optimum, write RMSNorm, LayerNorm and the attention core with primitive operators. `Transmuter.load!/2` replaces these subgraphs:

    * `Pow(x, 2)`, `ReduceMean`, `Add(eps)`, `Sqrt`, then `x / sqrt`, `x * (1 / sqrt)` or `x * Reciprocal(sqrt)`, then `Mul(weight)` becomes `SimplifiedLayerNormalization`.
    * `ReduceMean`, `Sub`, the RMSNorm of the difference, then `Mul(gamma)` and `Add(beta)` becomes `LayerNormalization`.
    * `MatMul(q, k)`, an optional `Mul` or `Div` by a scale, an optional `Cast` to a float type, an optional `Where(keep, scores, fill)`, an optional `Add(mask)`, `Softmax`, an optional `Cast` to a float type, an optional pair of `Reshape` and `MatMul(probabilities, v)` becomes `AttentionCore` (see `Transmuter.Ops.Transformer`). GPT-2 exports write the attention core in this way.

  The operators then use the fused kernels of `EMLX.Fast` on EMLX (see `Transmuter.Ops.emlx?/1`), and every other backend runs the same function as before.

  A subgraph matches only when the replacement computes the same function:

    * The exponent is 2, the numerator of `1 / sqrt` is 1, and eps and a `Mul` scale are constants with one value. A `Div` by a value that is not a constant becomes the divisor input of `AttentionCore`, which reads it when the graph runs.
    * The fill of the `Where` is a constant of at most -1.0e4, so the hidden keys get no weight.
    * Eps is an f32, f16 or bf16 value. It has the type of x, so x is not f64.
    * `ReduceMean` reduces the axis -1 and keeps it.
    * The weight, gamma and beta are constants of at most one axis, so they broadcast only along the last axis.
    * No value inside the subgraph has a consumer outside it, also not a graph output or a subgraph.

  Without shapes at load time, `AttentionCore` checks the shapes when it runs and uses the fused kernel only for 4-D heads, or 3-D heads in the batch as in Whisper and TrOCR exports, with a softmax over the last axis. A `Reshape` pair of the probabilities, as in the Whisper exports of Xenova, must give back the shape of the probabilities. Else the core runs as the operators of the graph.
  """

  alias Transmuter.Onnx.{AttributeProto, NodeProto}
  alias Transmuter.Ops

  # The ONNX types FLOAT, FLOAT16 and BFLOAT16.
  @float_types [1, 10, 16]

  @doc """
  Rewrites the matching subgraphs of `graph` and of its subgraphs.

  `opset` is the ONNX opset of the model. `outer` holds the constant tensors of the enclosing graphs, by name.
  """
  def graph(graph, opset, outer \\ %{}) do
    constants = Map.merge(outer, constants(graph))
    nodes = Enum.map(graph.node, &subgraphs(&1, opset, constants))

    ctx = %{
      producers: Map.new(for node <- nodes, name <- node.output, do: {name, node}),
      uses:
        Enum.frequencies(
          Enum.flat_map(nodes, &Transmuter.dependencies/1) ++ Enum.map(graph.output, & &1.name)
        ),
      constants: constants,
      opset: opset
    }

    # Later anchors first, so a LayerNorm claims its Mul before an RMSNorm of the centered values.
    {fused, removed} =
      nodes
      |> Enum.reverse()
      |> Enum.reduce({%{}, MapSet.new()}, fn node, {fused, removed} ->
        with false <- MapSet.member?(removed, node.output),
             {replacement, matched} <- match(ctx, node),
             true <- closed?(ctx, matched),
             true <- Enum.all?(matched, &(not MapSet.member?(removed, &1.output))) do
          {Map.put(fused, node.output, replacement),
           Enum.reduce(matched, removed, &MapSet.put(&2, &1.output))}
        else
          _ -> {fused, removed}
        end
      end)

    nodes =
      Enum.flat_map(nodes, fn node ->
        cond do
          replacement = fused[node.output] -> [replacement]
          MapSet.member?(removed, node.output) -> []
          true -> [node]
        end
      end)

    %{graph | node: nodes}
  end

  defp subgraphs(node, opset, constants) do
    attribute =
      Enum.map(node.attribute, fn a ->
        %{
          a
          | g: a.g && graph(a.g, opset, constants),
            graphs: Enum.map(a.graphs, &graph(&1, opset, constants))
        }
      end)

    %{node | attribute: attribute}
  end

  defp constants(graph) do
    values =
      for node = %{op_type: "Constant", domain: domain} <- graph.node,
          domain in [nil, ""],
          t = Ops.attr(node, "value"),
          do: {hd(node.output), t}

    Map.merge(Map.new(graph.initializer, &{&1.name, &1}), Map.new(values))
  end

  defp match(ctx, node) do
    case node do
      %{op_type: "Mul"} -> rms_norm(ctx, node)
      %{op_type: "Add"} -> layer_norm(ctx, node)
      %{op_type: "MatMul"} -> attention(ctx, node)
      _ -> nil
    end
  end

  # Every value that a matched node computes, except the output of the anchor, is read
  # only by matched nodes.
  defp closed?(ctx, [anchor | inner]) do
    reads = Enum.frequencies(Enum.flat_map([anchor | inner], & &1.input))

    for node <- inner, name <- node.output do
      Map.get(ctx.uses, name, 0) == Map.get(reads, name, 0)
    end
    |> Enum.all?()
  end

  defp rms_norm(ctx, mul) do
    Enum.find_value(orders(mul.input), fn [y, weight] ->
      with true <- vector?(ctx, weight),
           {x, sqrt, scaled} <- quotient(ctx, y),
           {^x, eps, square} <- mean_square(ctx, sqrt) do
        {fused(mul, "SimplifiedLayerNormalization", [x, weight], eps), [mul | scaled ++ square]}
      else
        _ -> nil
      end
    end)
  end

  defp layer_norm(ctx, add) do
    Enum.find_value(orders(add.input), fn [scaled, beta] ->
      with true <- vector?(ctx, beta),
           mul = %{} <- op(ctx, scaled, "Mul"),
           [centered, gamma] <- Enum.find(orders(mul.input), &vector?(ctx, Enum.at(&1, 1))),
           {diff, sqrt, quotient} <- quotient(ctx, centered),
           {^diff, eps, square} <- mean_square(ctx, sqrt),
           sub = %{input: [x, mean]} <- op(ctx, diff, "Sub"),
           reduce = %{input: [^x | _]} <- op(ctx, mean, "ReduceMean"),
           true <- last_axis?(ctx, reduce) do
        {fused(add, "LayerNormalization", [x, gamma, beta], eps),
         [add, mul | quotient ++ square ++ [sub, reduce]]}
      else
        _ -> nil
      end
    end)
  end

  # y = x / s, x * (1 / s) or x * Reciprocal(s). Returns {x, s, nodes}.
  defp quotient(ctx, y) do
    cond do
      div = op(ctx, y, "Div") ->
        [x, s] = div.input
        {x, s, [div]}

      mul = op(ctx, y, "Mul") ->
        Enum.find_value(orders(mul.input), fn [x, r] ->
          cond do
            inv = op(ctx, r, "Reciprocal") ->
              {x, hd(inv.input), [mul, inv]}

            (inv = op(ctx, r, "Div")) && scalar(ctx, hd(inv.input)) == 1 ->
              {x, Enum.at(inv.input, 1), [mul, inv]}

            true ->
              nil
          end
        end)

      true ->
        nil
    end
  end

  # s = Sqrt(ReduceMean(Pow(x, 2)) + eps) over the last axis. Returns {x, eps, nodes}.
  defp mean_square(ctx, s) do
    with sqrt = %{} <- op(ctx, s, "Sqrt"),
         add = %{} <- op(ctx, hd(sqrt.input), "Add"),
         [mean, eps] <- Enum.find(orders(add.input), &epsilon(ctx, Enum.at(&1, 1))),
         reduce = %{} <- op(ctx, mean, "ReduceMean"),
         true <- last_axis?(ctx, reduce),
         pow = %{input: [x, exponent]} <- op(ctx, hd(reduce.input), "Pow"),
         true <- scalar(ctx, exponent) == 2 do
      {x, epsilon(ctx, eps), [sqrt, add, reduce, pow]}
    else
      _ -> nil
    end
  end

  # The probabilities are Softmax(scores + mask), optionally through a Cast to a float type,
  # and the mask is optional. The inputs of AttentionCore are q, k, v, the mask, the keys to
  # keep, the divisor of the scores and the shapes of a Reshape pair of the probabilities,
  # with "" for a missing one.
  defp attention(ctx, matmul = %{input: [probs, v]}) do
    {probs, split, merge, reshaped} = reshape_pair(ctx, probs)

    with {softmax, cast} <- through_cast(ctx, probs, "Softmax"),
         {logits, mask, masked} <- masked(ctx, hd(softmax.input)),
         {qk = %{input: [q, k]}, scores} <- scores(ctx, logits) do
      default_axis = if ctx.opset >= 13, do: -1, else: 1

      node = %NodeProto{
        op_type: "AttentionCore",
        domain: "transmuter",
        name: matmul.name,
        input: [q, k, v, mask, scores.keep, scores.divisor, split, merge],
        output: matmul.output,
        attribute: [
          %AttributeProto{name: "scale", type: :FLOAT, f: scores.scale},
          %AttributeProto{name: "axis", type: :INT, i: Ops.attr(softmax, "axis", default_axis)},
          %AttributeProto{name: "fill", type: :FLOAT, f: scores.fill}
        ]
      }

      {node, [matmul, softmax | reshaped ++ cast ++ masked ++ scores.nodes ++ [qk]]}
    else
      _ -> nil
    end
  end

  # Some exports, such as the Whisper exports of Xenova, reshape the probabilities to 4-D
  # heads and back to the 3-D layout before the second MatMul. Without shapes, the pair is
  # not known to give back the same shape, so AttentionCore checks it when the graph runs.
  # Returns the input of the pair, its two shapes and its nodes.
  defp reshape_pair(ctx, name) do
    with merge = %{input: [split_out, merge_shape]} <- op(ctx, name, "Reshape"),
         split = %{input: [x, split_shape]} <- op(ctx, split_out, "Reshape"),
         true <- Enum.all?([merge, split], &(Ops.attr(&1, "allowzero", 0) == 0)) do
      {x, split_shape, merge_shape, [merge, split]}
    else
      _ -> {name, "", "", []}
    end
  end

  defp masked(ctx, z) do
    with add = %{} <- op(ctx, z, "Add"),
         [logits, mask] <- Enum.find(orders(add.input), &scores(ctx, hd(&1))) do
      {logits, mask, [add]}
    else
      _ -> {z, "", []}
    end
  end

  # The scores are MatMul(q, k), an optional scale, an optional Cast to a float type, and an
  # optional Where that keeps the scores of some keys and gives the others a fill of at most
  # -1.0e4, which gives them no weight. Returns the MatMul and the parts of the scores.
  defp scores(ctx, name) do
    {name, keep, fill, kept} = kept(ctx, name)
    {name, cast} = through_cast(ctx, name)
    {product, scale, divisor, scaled} = scaled(ctx, name)

    if qk = op(ctx, product, "MatMul") do
      {qk,
       %{
         scale: scale,
         divisor: divisor,
         keep: keep,
         fill: fill,
         nodes: kept ++ cast ++ scaled
       }}
    end
  end

  defp kept(ctx, name) do
    with where = %{input: [keep, scores, fill]} <- op(ctx, name, "Where"),
         value when is_float(value) and value <= -1.0e4 <- float_scalar(ctx, fill) do
      {scores, keep, value, [where]}
    else
      _ -> {name, "", 0.0, []}
    end
  end

  # The node of op_type that computes name, directly or through a Cast to a float type.
  defp through_cast(ctx, name, op_type) do
    {inner, cast} = through_cast(ctx, name)
    if node = op(ctx, inner, op_type), do: {node, cast}
  end

  defp through_cast(ctx, name) do
    case op(ctx, name, "Cast") do
      cast = %{input: [inner]} ->
        if Ops.attr(cast, "to") in @float_types, do: {inner, [cast]}, else: {name, []}

      nil ->
        {name, []}
    end
  end

  # A Div by a value that is not a constant keeps the value as the divisor.
  defp scaled(ctx, scores) do
    with nil <- op(ctx, scores, "Div"),
         nil <- op(ctx, scores, "Mul") do
      {scores, 1.0, "", []}
    else
      node = %{op_type: "Div", input: [product, divisor]} ->
        case float_scalar(ctx, divisor) do
          nil -> {product, 1.0, divisor, [node]}
          c when c != 0 -> {product, 1 / c, "", [node]}
          _zero -> {scores, 1.0, "", []}
        end

      node = %{op_type: "Mul"} ->
        Enum.find_value(orders(node.input), {scores, 1.0, "", []}, fn [product, factor] ->
          if c = float_scalar(ctx, factor), do: {product, c, "", [node]}
        end)
    end
  end

  defp fused(anchor, op_type, inputs, eps) do
    %NodeProto{
      op_type: op_type,
      name: anchor.name,
      input: inputs,
      output: anchor.output,
      attribute: [
        %AttributeProto{name: "epsilon", type: :FLOAT, f: eps},
        %AttributeProto{name: "axis", type: :INT, i: -1}
      ]
    }
  end

  defp op(ctx, name, op_type) do
    case ctx.producers[name] do
      node = %{op_type: ^op_type, domain: domain} when domain in [nil, ""] -> node
      _ -> nil
    end
  end

  defp orders([a, b]), do: [[a, b], [b, a]]
  defp orders(_inputs), do: []

  defp last_axis?(ctx, reduce) do
    axes =
      case reduce.input do
        [_x, axes] -> ints(ctx, axes)
        [_x] -> Ops.attr(reduce, "axes")
      end

    axes == [-1] and Ops.attr(reduce, "keepdims", 1) == 1
  end

  defp vector?(ctx, name) do
    case ctx.constants[name] do
      %{dims: dims} -> length(dims) <= 1
      nil -> false
    end
  end

  # The value of an f32, f16 or bf16 constant with one element.
  defp epsilon(ctx, name) do
    case ctx.constants[name] do
      %{data_type: type} when type in @float_types -> scalar(ctx, name)
      _ -> nil
    end
  end

  defp float_scalar(ctx, name) do
    case scalar(ctx, name) do
      value when is_float(value) -> value
      _ -> nil
    end
  end

  defp scalar(ctx, name) do
    case value(ctx, name) do
      t = %Nx.Tensor{shape: shape} when tuple_size(shape) <= 1 ->
        if Nx.size(t) == 1, do: t |> Nx.reshape({}) |> Nx.to_number()

      _ ->
        nil
    end
  end

  defp ints(ctx, name) do
    case value(ctx, name) do
      t = %Nx.Tensor{} -> if Nx.Type.integer?(Nx.type(t)), do: Nx.to_flat_list(t)
      _ -> nil
    end
  end

  # External data of the main graph is not read yet, and these constants are small.
  defp value(ctx, name) do
    case ctx.constants[name] do
      t = %{data_location: location, dims: dims} when location != :EXTERNAL ->
        if Enum.product(dims) <= 16, do: Ops.tensor(t)

      _ ->
        nil
    end
  end
end
