defmodule Transmuter.Ops do
  @moduledoc false

  # ONNX operators as Nx calls. Values known at trace time live on Nx.BinaryBackend
  # ("static"), so shape arithmetic folds to plain integers while tracing. Nx has no
  # zero-size tensors, so an empty static value is the atom :empty.

  alias Transmuter.Ops.{Quantized, Signal, Transformer}

  # Sums, norms and softmax lose too much precision in f16 and bf16, so these ops
  # run in f32 and cast their outputs back.
  @f32_ops ~w(ReduceSum ReduceMean ReduceSumSquare ReduceL2 ReduceProd Softmax LpNormalization
              LayerNormalization InstanceNormalization GroupNormalization BatchNormalization
              RMSNormalization SimplifiedLayerNormalization SkipLayerNormalization
              SkipSimplifiedLayerNormalization EmbedLayerNormalization Attention
              GroupQueryAttention MultiHeadAttention)

  @doc false
  def apply(node, inputs, opset) do
    half =
      Enum.find_value(
        inputs,
        &(is_struct(&1, Nx.Tensor) and Nx.type(&1) in [{:f, 16}, {:bf, 16}] and Nx.type(&1))
      )

    if half && node.op_type in @f32_ops do
      inputs
      |> Enum.map(
        &if(is_struct(&1, Nx.Tensor) and Nx.type(&1) == half, do: Nx.as_type(&1, :f32), else: &1)
      )
      |> then(&dispatch(node, &1, opset))
      |> Enum.map(
        &if(is_struct(&1, Nx.Tensor) and Nx.type(&1) == {:f, 32},
          do: Nx.as_type(&1, half),
          else: &1
        )
      )
    else
      dispatch(node, inputs, opset)
    end
  end

  defp dispatch(node, inputs, opset) do
    cond do
      node.op_type == "Softmax" -> [softmax(node, hd(inputs), opset)]
      node.op_type in Transformer.op_types() -> Transformer.apply(node, inputs)
      node.op_type in Quantized.op_types() -> Quantized.apply(node, inputs)
      node.op_type in Signal.op_types() -> Signal.apply(node, inputs)
      true -> node |> op(inputs) |> List.wrap()
    end
  end

  @doc false
  def static?(:empty), do: true
  def static?(%Nx.Tensor{data: %Nx.BinaryBackend{}}), do: true
  def static?(_), do: false

  @doc false
  def ints!(:empty), do: []

  def ints!(t) do
    unless known?(t) do
      raise ArgumentError,
            "this value decides a shape but is only known when the model runs. " <>
              "Pass the graph inputs it depends on as static_inputs, or call Transmuter.run/4 " <>
              "without Nx.Defn.jit so values are read as the model runs"
    end

    Nx.to_flat_list(t)
  end

  # Concrete values can be read on the host: static ones, and every tensor when the
  # model runs eagerly. Only values inside a traced function are unknown.
  @doc false
  def known?(:empty), do: true
  def known?(%Nx.Tensor{data: %Nx.Defn.Expr{}}), do: false
  def known?(%Nx.Tensor{}), do: true

  @doc false
  def static_tensor(list, type), do: Nx.tensor(list, type: type, backend: Nx.BinaryBackend)

  defp op(%{op_type: "Add"}, [a, b]), do: Nx.add(a, b)
  defp op(%{op_type: "Sub"}, [a, b]), do: Nx.subtract(a, b)
  defp op(%{op_type: "Mul"}, [a, b]), do: Nx.multiply(a, b)

  defp op(%{op_type: "Div"}, [a, b]) do
    if Nx.Type.integer?(Nx.type(a)) and Nx.Type.integer?(Nx.type(b)),
      do: Nx.quotient(a, b),
      else: Nx.divide(a, b)
  end

  defp op(%{op_type: "Pow"}, [a, b]), do: Nx.pow(a, b)
  defp op(%{op_type: "Sqrt"}, [a]), do: Nx.sqrt(a)
  defp op(%{op_type: "Erf"}, [a]), do: Nx.erf(a)
  defp op(%{op_type: "Tanh"}, [a]), do: Nx.tanh(a)
  defp op(%{op_type: "Cast"}, [:empty]), do: :empty
  defp op(node = %{op_type: "Cast"}, [a]), do: Nx.as_type(a, dtype(attr(node, "to")))

  defp op(node = %{op_type: "Constant"}, []) do
    case node.attribute do
      [%{name: "value", t: t}] -> tensor(t)
      [%{name: "value_int", i: i}] -> static_tensor(i, :s64)
      [%{name: "value_ints", ints: ints}] -> static_tensor(ints, :s64)
      [%{name: "value_float", f: f}] -> static_tensor(f, :f32)
      [%{name: "value_floats", floats: fs}] -> static_tensor(fs, :f32)
    end
  end

  defp op(%{op_type: "Abs"}, [a]), do: Nx.abs(a)
  defp op(%{op_type: "Exp"}, [a]), do: Nx.exp(a)
  defp op(%{op_type: "Sigmoid"}, [a]), do: Nx.sigmoid(a)

  defp op(node = %{op_type: "Mod"}, [a, b]) do
    r = Nx.remainder(a, b)

    if attr(node, "fmod", 0) == 1,
      do: r,
      else:
        Nx.select(
          Nx.logical_and(Nx.not_equal(r, 0), Nx.not_equal(Nx.sign(r), Nx.sign(b))),
          Nx.add(r, b),
          r
        )
  end

  defp op(node = %{op_type: "ConstantOfShape"}, [shape]) do
    value =
      node |> attr("value") |> then(&if(&1, do: tensor(&1), else: static_tensor([0.0], :f32)))

    dims = ints!(shape)
    if 0 in dims, do: :empty, else: value |> Nx.reshape({}) |> Nx.broadcast(List.to_tuple(dims))
  end

  defp op(%{op_type: "Expand"}, [a, shape]) do
    target = ints!(shape)
    rank = max(Nx.rank(a), length(target))
    pad = fn list -> List.duplicate(1, rank - length(list)) ++ list end
    out = Enum.zip_with(pad.(Tuple.to_list(Nx.shape(a))), pad.(target), &max/2)

    a
    |> Nx.reshape(List.to_tuple(pad.(Tuple.to_list(Nx.shape(a)))))
    |> Nx.broadcast(List.to_tuple(out))
  end

  defp op(%{op_type: "Tile"}, [a, repeats]), do: Nx.tile(a, ints!(repeats))

  defp op(node = %{op_type: "Squeeze"}, [a | rest]) do
    axes = if rest == [] or rest == [nil], do: attr(node, "axes"), else: ints!(hd(rest))
    if axes, do: Nx.squeeze(a, axes: axes), else: Nx.squeeze(a)
  end

  defp op(node = %{op_type: "ReduceSum"}, [a | rest]) do
    axes = if rest == [] or rest == [nil], do: attr(node, "axes"), else: ints!(hd(rest))
    keep = attr(node, "keepdims", 1) == 1

    cond do
      axes in [nil, []] and attr(node, "noop_with_empty_axes", 0) == 1 -> a
      axes in [nil, []] -> Nx.sum(a, keep_axes: keep)
      true -> Nx.sum(a, axes: axes, keep_axes: keep)
    end
  end

  defp op(node = %{op_type: "Gemm"}, [a, b | c]) do
    a = if attr(node, "transA", 0) == 1, do: Nx.transpose(a), else: a
    b = if attr(node, "transB", 0) == 1, do: Nx.transpose(b), else: b
    out = Nx.multiply(Nx.dot(a, b), attr(node, "alpha", 1.0))

    case c do
      [c] when c != nil -> Nx.add(out, Nx.multiply(c, attr(node, "beta", 1.0)))
      _ -> out
    end
  end

  # Integer convolution, accumulated in f32 like MatMulInteger.
  defp op(node = %{op_type: "ConvInteger"}, [x, w | zero_points]) do
    [x_zero, w_zero] = Enum.take(zero_points ++ [nil, nil], 2)
    x = Nx.as_type(x, :f32)
    x = if x_zero, do: Nx.subtract(x, Nx.reshape(Nx.as_type(x_zero, :f32), {})), else: x
    w = Nx.as_type(w, :f32)

    w =
      cond do
        w_zero == nil ->
          w

        Nx.size(w_zero) == 1 ->
          Nx.subtract(w, Nx.reshape(Nx.as_type(w_zero, :f32), {}))

        true ->
          Nx.subtract(
            w,
            Nx.reshape(
              Nx.as_type(w_zero, :f32),
              List.to_tuple([:auto | List.duplicate(1, Nx.rank(w) - 1)])
            )
          )
      end

    %{node | op_type: "Conv"} |> op([x, w]) |> Nx.round() |> Nx.as_type(:s32)
  end

  defp op(node = %{op_type: "Conv"}, [x, w | b]) do
    spatial = Nx.rank(x) - 2
    pads = attr(node, "pads")

    padding =
      case {attr(node, "auto_pad", "NOTSET"), pads} do
        {"VALID", _} -> :valid
        {"NOTSET", nil} -> :valid
        {"NOTSET", pads} -> Enum.zip(Enum.take(pads, spatial), Enum.drop(pads, spatial))
      end

    out =
      Nx.conv(x, w,
        strides: attr(node, "strides", List.duplicate(1, spatial)),
        padding: padding,
        kernel_dilation: attr(node, "dilations", List.duplicate(1, spatial)),
        feature_group_size: attr(node, "group", 1)
      )

    case b do
      [b] when b != nil ->
        Nx.add(out, Nx.reshape(b, List.to_tuple([1, :auto] ++ List.duplicate(1, spatial))))

      _ ->
        out
    end
  end

  defp op(%{op_type: "Size"}, [a]), do: static_tensor(Nx.size(a), :s64)

  defp op(%{op_type: "Shape"}, [a]),
    do: a |> Nx.shape() |> Tuple.to_list() |> static_tensor(:s64)

  defp op(node = %{op_type: "Gather"}, [data, indices]) do
    axis = attr(node, "axis", 0)
    axis = if axis < 0, do: axis + Nx.rank(data), else: axis
    dim = Nx.axis_size(data, axis)

    indices = Nx.select(Nx.less(indices, 0), Nx.add(indices, dim), indices)

    # EMLX 0.5 scatters gradients to the wrong place when the gathered axis is not 0,
    # so other axes move to the front, gather there and move back.
    if axis == 0 do
      Nx.take(data, indices, axis: 0)
    else
      rank = Nx.rank(data)
      front = [axis | Enum.to_list(0..(rank - 1)) -- [axis]]
      gathered = data |> Nx.transpose(axes: front) |> Nx.take(indices, axis: 0)
      index_rank = Nx.rank(indices)

      back =
        Enum.to_list(index_rank..(index_rank + axis - 1)//1) ++
          Enum.to_list(0..(index_rank - 1)//1) ++
          Enum.to_list((index_rank + axis)..(Nx.rank(gathered) - 1)//1)

      Nx.transpose(gathered, axes: back)
    end
  end

  # Nx has no zero-size tensors, so an empty static 1-D slice is :empty and Concat drops it.
  defp op(node = %{op_type: "Concat"}, inputs),
    do: inputs |> Enum.reject(&(&1 == :empty)) |> Nx.concatenate(axis: attr(node, "axis"))

  defp op(node = %{op_type: "Unsqueeze"}, [a | rest]) do
    axes = if rest == [], do: attr(node, "axes"), else: ints!(hd(rest))
    out_rank = Nx.rank(a) + length(axes)

    axes
    |> Enum.map(&if(&1 < 0, do: &1 + out_rank, else: &1))
    |> Enum.sort()
    |> Enum.reduce(a, fn axis, acc ->
      acc |> Nx.shape() |> Tuple.insert_at(axis, 1) |> then(&Nx.reshape(acc, &1))
    end)
  end

  defp op(%{op_type: "Reshape"}, [a, shape]) do
    dims =
      shape
      |> ints!()
      |> Enum.with_index()
      |> Enum.map(fn
        {0, i} -> elem(Nx.shape(a), i)
        {-1, _} -> :auto
        {d, _} -> d
      end)

    Nx.reshape(a, List.to_tuple(dims))
  end

  defp op(%{op_type: "Slice"}, [data, starts, ends | rest]) do
    starts = ints!(starts)
    ends = ints!(ends)

    axes = optional_ints(Enum.at(rest, 0), Enum.to_list(0..(length(starts) - 1)))
    steps = optional_ints(Enum.at(rest, 1), List.duplicate(1, length(starts)))

    [starts, ends, axes, steps]
    |> Enum.zip()
    |> Enum.reduce(data, fn {s, e, ax, step}, acc ->
      ax = if ax < 0, do: ax + Nx.rank(acc), else: ax
      dim = Nx.axis_size(acc, ax)
      s = if s < 0, do: s + dim, else: s
      e = if e < 0, do: e + dim, else: e

      indices =
        if step > 0,
          do: Enum.to_list(min(max(s, 0), dim)..(min(max(e, 0), dim) - 1)//step),
          else: Enum.to_list(min(max(s, 0), dim - 1)..(min(max(e, -1), dim - 1) + 1)//step)

      cond do
        indices == [] -> :empty
        step == 1 -> Nx.slice_along_axis(acc, hd(indices), length(indices), axis: ax)
        true -> Nx.take(acc, static_tensor(indices, :s64), axis: ax)
      end
    end)
  end

  defp op(node = %{op_type: "Transpose"}, [a]) do
    perm = attr(node, "perm", Enum.reverse(0..(Nx.rank(a) - 1)//1))
    Nx.transpose(a, axes: perm)
  end

  defp op(node = %{op_type: "ReduceMean"}, [a | rest]) do
    axes = if rest == [], do: attr(node, "axes"), else: ints!(hd(rest))
    Nx.mean(a, axes: axes, keep_axes: attr(node, "keepdims", 1) == 1)
  end

  defp op(%{op_type: "MatMul"}, [a, b]), do: matmul(a, b)

  defp op(%{op_type: "Identity"}, [a]), do: a
  defp op(%{op_type: "Equal"}, [a, b]), do: Nx.equal(a, b)
  defp op(%{op_type: "Less"}, [a, b]), do: Nx.less(a, b)
  defp op(%{op_type: "LessOrEqual"}, [a, b]), do: Nx.less_equal(a, b)
  defp op(%{op_type: "Greater"}, [a, b]), do: Nx.greater(a, b)
  defp op(%{op_type: "GreaterOrEqual"}, [a, b]), do: Nx.greater_equal(a, b)
  defp op(%{op_type: "Not"}, [a]), do: Nx.logical_not(a)
  defp op(%{op_type: "And"}, [a, b]), do: Nx.logical_and(a, b)
  defp op(%{op_type: "Or"}, [a, b]), do: Nx.logical_or(a, b)
  defp op(%{op_type: "Xor"}, [a, b]), do: Nx.logical_xor(a, b)
  defp op(%{op_type: "Neg"}, [a]), do: Nx.negate(a)
  defp op(%{op_type: "Sin"}, [a]), do: Nx.sin(a)
  defp op(%{op_type: "Cos"}, [a]), do: Nx.cos(a)
  defp op(%{op_type: "Atan"}, [a]), do: Nx.atan(a)
  defp op(%{op_type: "Log"}, [a]), do: Nx.log(a)
  defp op(%{op_type: "Floor"}, [a]), do: Nx.floor(a)
  defp op(%{op_type: "Ceil"}, [a]), do: Nx.ceil(a)
  defp op(%{op_type: "Round"}, [a]), do: Nx.round(a)
  defp op(%{op_type: "Sign"}, [a]), do: Nx.sign(a)
  defp op(%{op_type: "Reciprocal"}, [a]), do: Nx.divide(1, a)
  defp op(%{op_type: "IsNaN"}, [a]), do: Nx.is_nan(a)
  defp op(%{op_type: "IsInf"}, [a]), do: Nx.is_infinity(a)
  defp op(%{op_type: "Relu"}, [a]), do: Nx.max(a, 0)

  defp op(node = %{op_type: "LeakyRelu"}, [a]),
    do: Nx.select(Nx.less(a, 0), Nx.multiply(a, attr(node, "alpha", 0.01)), a)

  defp op(%{op_type: "Softplus"}, [a]), do: Nx.log1p(Nx.exp(a))

  defp op(node = %{op_type: "HardSigmoid"}, [a]),
    do:
      a
      |> Nx.multiply(attr(node, "alpha", 0.2))
      |> Nx.add(attr(node, "beta", 0.5))
      |> Nx.clip(0, 1)

  defp op(%{op_type: "HardSwish"}, [a]),
    do: Nx.multiply(a, a |> Nx.divide(6) |> Nx.add(0.5) |> Nx.clip(0, 1))

  defp op(node = %{op_type: "Gelu"}, [a]),
    do: if(attr(node, "approximate", "none") == "tanh", do: gelu_tanh(a), else: gelu(a))

  defp op(%{op_type: "Max"}, inputs), do: Enum.reduce(inputs, &Nx.max(&2, &1))
  defp op(%{op_type: "Min"}, inputs), do: Enum.reduce(inputs, &Nx.min(&2, &1))
  defp op(%{op_type: "CastLike"}, [a, like]), do: Nx.as_type(a, Nx.type(like))

  defp op(node = %{op_type: "Clip"}, [a | bounds]) do
    {lo, hi} =
      case bounds do
        [] -> {attr(node, "min"), attr(node, "max")}
        [lo] -> {lo, nil}
        [lo, hi] -> {lo, hi}
      end

    a = if lo != nil, do: Nx.max(a, Nx.as_type(lo, Nx.type(a))), else: a
    if hi != nil, do: Nx.min(a, Nx.as_type(hi, Nx.type(a))), else: a
  end

  defp op(%{op_type: "Where"}, [c, x, y]) do
    shape = broadcast_shape([Nx.shape(c), Nx.shape(x), Nx.shape(y)])
    Nx.select(Nx.broadcast(c, shape), Nx.broadcast(x, shape), Nx.broadcast(y, shape))
  end

  defp op(%{op_type: "Range"}, [start_t, limit, delta]) do
    [start, limit, delta] = Enum.map([start_t, limit, delta], &(&1 |> ints!() |> hd()))
    n = max(ceil((limit - start) / delta), 0)
    static_tensor(Enum.map(0..(n - 1)//1, &(start + &1 * delta)), Nx.type(start_t))
  end

  defp op(node = %{op_type: "Split"}, [a | rest]) do
    axis = attr(node, "axis", 0)
    axis = if axis < 0, do: axis + Nx.rank(a), else: axis
    dim = Nx.axis_size(a, axis)
    n_out = length(node.output)

    sizes =
      cond do
        rest != [] and hd(rest) != nil ->
          ints!(hd(rest))

        attr(node, "split") ->
          attr(node, "split")

        true ->
          chunk = ceil(dim / n_out)
          Enum.map(0..(n_out - 1), &min(chunk, dim - &1 * chunk))
      end

    sizes
    |> Enum.scan({0, 0}, fn size, {start, prev} -> {start + prev, size} end)
    |> Enum.map(fn {start, size} -> Nx.slice_along_axis(a, start, size, axis: axis) end)
  end

  defp op(node = %{op_type: "Pad"}, [a | rest]) do
    mode = attr(node, "mode", "constant")

    {pads, value, axes} =
      case rest do
        [] ->
          {attr(node, "pads"), attr(node, "value", 0.0), nil}

        [pads | more] ->
          {ints!(pads), (Enum.at(more, 0) && Nx.to_number(Enum.at(more, 0))) || 0,
           Enum.at(more, 1) && ints!(Enum.at(more, 1))}
      end

    rank = Nx.rank(a)
    axes = Enum.map(axes || Enum.to_list(0..(rank - 1)), &if(&1 < 0, do: &1 + rank, else: &1))
    {lo, hi} = Enum.split(pads, length(axes))
    by_axis = Map.new(Enum.zip([axes, lo, hi]), fn {ax, l, h} -> {ax, {l, h, 0}} end)

    if mode == "constant" do
      Nx.pad(
        a,
        Nx.tensor(value, type: Nx.type(a)),
        Enum.map(0..(rank - 1), &Map.get(by_axis, &1, {0, 0, 0}))
      )
    else
      Enum.reduce(by_axis, a, fn {ax, {l, h, 0}}, acc ->
        dim = Nx.axis_size(acc, ax)

        indices = Enum.map(-l..(dim - 1 + h)//1, &pad_source(mode, &1, dim))
        Nx.take(acc, static_tensor(indices, :s64), axis: ax)
      end)
    end
  end

  defp op(node = %{op_type: "DepthToSpace"}, [a]) do
    b = attr(node, "blocksize")
    {n, c, h, w} = Nx.shape(a)

    case attr(node, "mode", "DCR") do
      "DCR" ->
        a |> Nx.reshape({n, b, b, div(c, b * b), h, w}) |> Nx.transpose(axes: [0, 3, 4, 1, 5, 2])

      "CRD" ->
        a |> Nx.reshape({n, div(c, b * b), b, b, h, w}) |> Nx.transpose(axes: [0, 1, 4, 2, 5, 3])
    end
    |> Nx.reshape({n, div(c, b * b), h * b, w * b})
  end

  defp op(node = %{op_type: "InstanceNormalization"}, [x, scale, bias]) do
    axes = Enum.to_list(2..(Nx.rank(x) - 1))
    mean = Nx.mean(x, axes: axes, keep_axes: true)
    var = Nx.variance(x, axes: axes, keep_axes: true)
    shape = List.to_tuple([1, :auto | List.duplicate(1, length(axes))])

    x
    |> Nx.subtract(mean)
    |> Nx.multiply(Nx.rsqrt(Nx.add(var, attr(node, "epsilon", 1.0e-5))))
    |> Nx.multiply(Nx.reshape(scale, shape))
    |> Nx.add(Nx.reshape(bias, shape))
  end

  defp op(node = %{op_type: "BatchNormalization"}, [x, scale, bias, mean, var]) do
    shape = List.to_tuple([1, :auto | List.duplicate(1, Nx.rank(x) - 2)])
    r = &Nx.reshape(&1, shape)

    x
    |> Nx.subtract(r.(mean))
    |> Nx.multiply(r.(Nx.rsqrt(Nx.add(var, attr(node, "epsilon", 1.0e-5)))))
    |> Nx.multiply(r.(scale))
    |> Nx.add(r.(bias))
  end

  defp op(node = %{op_type: "LayerNormalization"}, [x, scale | bias]) do
    axis = attr(node, "axis", -1)
    axis = if axis < 0, do: axis + Nx.rank(x), else: axis
    axes = Enum.to_list(axis..(Nx.rank(x) - 1))
    mean = Nx.mean(x, axes: axes, keep_axes: true)
    var = Nx.variance(x, axes: axes, keep_axes: true)

    out =
      x
      |> Nx.subtract(mean)
      |> Nx.multiply(Nx.rsqrt(Nx.add(var, attr(node, "epsilon", 1.0e-5))))
      |> Nx.multiply(scale)

    case bias do
      [b | _] when b != nil -> Nx.add(out, b)
      _ -> out
    end
  end

  defp op(%{op_type: "GlobalAveragePool"}, [x]),
    do: Nx.mean(x, axes: Enum.to_list(2..(Nx.rank(x) - 1)), keep_axes: true)

  defp op(%{op_type: "GlobalMaxPool"}, [x]),
    do: Nx.reduce_max(x, axes: Enum.to_list(2..(Nx.rank(x) - 1)), keep_axes: true)

  defp op(node = %{op_type: pool}, [x]) when pool in ["MaxPool", "AveragePool"] do
    kernel = attr(node, "kernel_shape")
    spatial = length(kernel)
    strides = attr(node, "strides", List.duplicate(1, spatial))
    pads = attr(node, "pads", List.duplicate(0, 2 * spatial))
    sizes = x |> Nx.shape() |> Tuple.to_list() |> Enum.drop(2)

    extra =
      if attr(node, "ceil_mode", 0) == 1 do
        Enum.zip_with(
          [sizes, kernel, strides, Enum.take(pads, spatial), Enum.drop(pads, spatial)],
          fn [n, k, st, lo, hi] ->
            out = ceil((n + lo + hi - k) / st) + 1
            out = if (out - 1) * st >= n + lo, do: out - 1, else: out
            max((out - 1) * st + k - (n + lo + hi), 0)
          end
        )
      else
        List.duplicate(0, spatial)
      end

    padding = [
      {0, 0},
      {0, 0}
      | Enum.zip_with([Enum.take(pads, spatial), Enum.drop(pads, spatial), extra], fn [lo, hi, e] ->
          {lo, hi + e}
        end)
    ]

    window = List.to_tuple([1, 1 | kernel])
    opts = [strides: [1, 1 | strides], padding: padding]

    if pool == "MaxPool" do
      Nx.window_max(x, window, opts)
    else
      sum = Nx.window_sum(x, window, opts)

      if attr(node, "count_include_pad", 0) == 1 or Enum.all?(pads, &(&1 == 0)),
        do: Nx.divide(sum, Enum.product(kernel)),
        else:
          Nx.divide(
            sum,
            Nx.window_sum(
              Nx.broadcast(Nx.tensor(1.0, type: Nx.type(x)), Nx.shape(x)),
              window,
              opts
            )
          )
    end
  end

  defp op(node = %{op_type: "ConvTranspose"}, [x, w | b]) do
    groups = attr(node, "group", 1)

    if groups > 1 do
      group_in = div(Nx.axis_size(x, 1), groups)

      out =
        for g <- 0..(groups - 1) do
          op(%{node | attribute: Enum.reject(node.attribute, &(&1.name == "group"))}, [
            Nx.slice_along_axis(x, g * group_in, group_in, axis: 1),
            Nx.slice_along_axis(w, g * group_in, group_in, axis: 0)
          ])
        end
        |> Nx.concatenate(axis: 1)

      add_channel_bias(out, List.first(b))
    else
      conv_transpose(node, x, w, List.first(b))
    end
  end

  defp op(%{op_type: "FastGelu"}, [x | bias]), do: gelu_tanh(add_optional(x, List.first(bias)))
  defp op(%{op_type: "BiasGelu"}, [x, bias]), do: gelu(Nx.add(x, bias))

  defp op(node = %{op_type: "QuickGelu"}, [x]),
    do: Nx.multiply(x, Nx.sigmoid(Nx.multiply(x, attr(node, "alpha", 1.702))))

  defp op(%{op_type: "PRelu"}, [x, slope]),
    do:
      Nx.select(Nx.less(x, 0), Nx.multiply(x, slope), x)
      |> Nx.broadcast(broadcast_shape([Nx.shape(x), Nx.shape(slope)]))

  defp op(%{op_type: "Mish"}, [x]), do: Nx.multiply(x, Nx.tanh(Nx.log1p(Nx.exp(x))))
  defp op(%{op_type: "Softsign"}, [x]), do: Nx.divide(x, Nx.add(1, Nx.abs(x)))

  defp op(node = %{op_type: "Selu"}, [x]) do
    alpha = attr(node, "alpha", 1.67326319217681884765625)
    gamma = attr(node, "gamma", 1.05070102214813232421875)

    Nx.multiply(
      gamma,
      Nx.select(Nx.greater(x, 0), x, Nx.multiply(alpha, Nx.subtract(Nx.exp(x), 1)))
    )
  end

  defp op(node = %{op_type: "ThresholdedRelu"}, [x]),
    do: Nx.select(Nx.greater(x, attr(node, "alpha", 1.0)), x, Nx.tensor(0, type: Nx.type(x)))

  defp op(node = %{op_type: "LpNormalization"}, [x]) do
    axis = attr(node, "axis", -1)

    norm =
      case attr(node, "p", 2) do
        1 -> Nx.sum(Nx.abs(x), axes: [axis], keep_axes: true)
        2 -> x |> Nx.pow(2) |> Nx.sum(axes: [axis], keep_axes: true) |> Nx.sqrt()
      end

    Nx.divide(x, norm)
  end

  defp op(node = %{op_type: "SpaceToDepth"}, [x]) do
    b = attr(node, "blocksize")
    {n, c, h, w} = Nx.shape(x)

    x
    |> Nx.reshape({n, c, div(h, b), b, div(w, b), b})
    |> Nx.transpose(axes: [0, 3, 5, 1, 2, 4])
    |> Nx.reshape({n, c * b * b, div(h, b), div(w, b)})
  end

  # Opset 21 scales each channel, opset 18 scales each group.
  defp op(node = %{op_type: "GroupNormalization"}, [x, scale, bias]) do
    groups = attr(node, "num_groups")
    shape = Nx.shape(x)
    {n, c} = {elem(shape, 0), elem(shape, 1)}
    grouped = Nx.reshape(x, {n, groups, :auto})
    mean = Nx.mean(grouped, axes: [2], keep_axes: true)
    var = Nx.variance(grouped, axes: [2], keep_axes: true)

    normed =
      grouped
      |> Nx.subtract(mean)
      |> Nx.multiply(Nx.rsqrt(Nx.add(var, attr(node, "epsilon", 1.0e-5))))

    {scale, bias} =
      if Nx.size(scale) == c,
        do: {scale, bias},
        else: {repeat_groups(scale, c), repeat_groups(bias, c)}

    channel = List.to_tuple([1, c | List.duplicate(1, tuple_size(shape) - 2)])

    normed
    |> Nx.reshape(shape)
    |> Nx.multiply(Nx.reshape(scale, channel))
    |> Nx.add(Nx.reshape(bias, channel))
  end

  # Each update goes to its own coordinates with the index replacing the axis coordinate.
  defp op(node = %{op_type: "ScatterElements"}, [data, indices, updates]) do
    rank = Nx.rank(data)
    axis = normalize_axis(attr(node, "axis", 0), rank)
    dim = Nx.axis_size(data, axis)
    indices = Nx.select(Nx.less(indices, 0), Nx.add(indices, dim), indices)
    shape = Nx.shape(indices)

    coords =
      for a <- 0..(rank - 1) do
        if a == axis, do: Nx.as_type(indices, :s64), else: Nx.iota(shape, axis: a, type: :s64)
      end
      |> Nx.stack(axis: -1)
      |> Nx.reshape({:auto, rank})

    updates = Nx.reshape(updates, {:auto})

    case attr(node, "reduction", "none") do
      "none" -> Nx.indexed_put(data, coords, updates)
      "add" -> Nx.indexed_add(data, coords, updates)
    end
  end

  defp op(node = %{op_type: "GatherND"}, [data, indices]) do
    batch_dims = attr(node, "batch_dims", 0)
    k = elem(Nx.shape(indices), Nx.rank(indices) - 1)

    dims =
      data |> Nx.shape() |> Tuple.to_list() |> Enum.slice(batch_dims, k) |> static_tensor(:s64)

    indices = Nx.as_type(indices, :s64)
    indices = Nx.select(Nx.less(indices, 0), Nx.add(indices, dims), indices)
    outer = indices |> Nx.shape() |> Tuple.delete_at(Nx.rank(indices) - 1)

    batch_coords =
      for a <- 0..(batch_dims - 1)//1,
          do: outer |> Nx.iota(axis: a, type: :s64) |> Nx.new_axis(-1)

    full = Nx.concatenate(batch_coords ++ [indices], axis: -1)
    Nx.gather(data, full, axes: Enum.to_list(0..(batch_dims + k - 1)))
  end

  defp op(node = %{op_type: "ScatterND"}, [data, indices, updates]) do
    "none" = attr(node, "reduction", "none")
    k = elem(Nx.shape(indices), Nx.rank(indices) - 1)
    n = div(Nx.size(indices), k)
    slice = data |> Nx.shape() |> Tuple.to_list() |> Enum.drop(k)

    Nx.indexed_put(
      data,
      Nx.reshape(indices, {n, k}),
      Nx.reshape(updates, List.to_tuple([n | slice])),
      axes: Enum.to_list(0..(k - 1))
    )
  end

  # Resize one axis at a time. Each output element reads a few input elements, so the
  # taps become static index and weight tensors, and the result is a weighted sum of gathers.
  defp op(node = %{op_type: "Resize"}, [x | rest]) do
    {scales, sizes} =
      case rest do
        [scales] -> {scales, nil}
        [_roi, scales] -> {scales, nil}
        [_roi, scales, sizes] -> {scales, sizes}
      end

    in_shape = Tuple.to_list(Nx.shape(x))

    scales =
      if scales != nil and scales != :empty and Nx.size(scales) > 0,
        do: Nx.to_flat_list(scales),
        else: nil

    sizes =
      if sizes != nil and sizes != :empty,
        do: ints!(sizes),
        else: Enum.zip_with(in_shape, scales, &floor(&1 * &2))

    scales = scales || Enum.zip_with(sizes, in_shape, &(&1 / &2))

    [in_shape, sizes, scales]
    |> Enum.zip()
    |> Enum.with_index()
    |> Enum.reduce(x, fn
      {{same, same, _}, _}, acc ->
        acc

      {{in_size, out_size, scale}, axis}, acc ->
        resize_axis(acc, node, axis, in_size, out_size, scale)
    end)
  end

  defp op(node = %{op_type: "Flatten"}, [a]) do
    axis = attr(node, "axis", 1)
    axis = if axis < 0, do: axis + Nx.rank(a), else: axis
    {outer, inner} = a |> Nx.shape() |> Tuple.to_list() |> Enum.split(axis)
    Nx.reshape(a, {Enum.product(outer), Enum.product(inner)})
  end

  defp op(node = %{op_type: reduce}, [a | rest])
       when reduce in ["ReduceMax", "ReduceMin", "ReduceProd", "ReduceL2", "ReduceSumSquare"] do
    axes = if rest == [] or rest == [nil], do: attr(node, "axes"), else: ints!(hd(rest))

    opts =
      [keep_axes: attr(node, "keepdims", 1) == 1] ++
        if(axes in [nil, []], do: [], else: [axes: axes])

    case reduce do
      "ReduceMax" -> Nx.reduce_max(a, opts)
      "ReduceMin" -> Nx.reduce_min(a, opts)
      "ReduceProd" -> Nx.product(a, opts)
      "ReduceL2" -> a |> Nx.pow(2) |> Nx.sum(opts) |> Nx.sqrt()
      "ReduceSumSquare" -> a |> Nx.pow(2) |> Nx.sum(opts)
    end
  end

  defp op(node = %{op_type: arg}, [a]) when arg in ["ArgMax", "ArgMin"] do
    0 = attr(node, "select_last_index", 0)
    opts = [axis: attr(node, "axis", 0), keep_axis: attr(node, "keepdims", 1) == 1, type: :s64]
    if arg == "ArgMax", do: Nx.argmax(a, opts), else: Nx.argmin(a, opts)
  end

  defp op(node = %{op_type: "CumSum"}, [x, axis]) do
    axis = axis |> Nx.to_number()
    axis = if axis < 0, do: axis + Nx.rank(x), else: axis
    sum = Nx.cumulative_sum(x, axis: axis, reverse: attr(node, "reverse", 0) == 1)
    if attr(node, "exclusive", 0) == 1, do: Nx.subtract(sum, x), else: sum
  end

  defp op(node = %{op_type: "Trilu"}, [x | k]) do
    k =
      case k do
        [k] when k != nil -> Nx.to_number(k)
        _ -> 0
      end

    rank = Nx.rank(x)
    shape = Nx.shape(x)
    diff = Nx.subtract(Nx.iota(shape, axis: rank - 1), Nx.iota(shape, axis: rank - 2))

    keep =
      if attr(node, "upper", 1) == 1, do: Nx.greater_equal(diff, k), else: Nx.less_equal(diff, k)

    Nx.select(keep, x, Nx.tensor(0, type: Nx.type(x)))
  end

  defp op(node = %{op_type: "OneHot"}, [indices, depth, values]) do
    depth = Nx.to_number(depth)
    axis = attr(node, "axis", -1)
    axis = if axis < 0, do: axis + Nx.rank(indices) + 1, else: axis
    indices = Nx.select(Nx.less(indices, 0), Nx.add(indices, depth), indices)
    expanded = Nx.new_axis(indices, axis)
    shape = expanded |> Nx.shape() |> put_elem(axis, depth)
    hot = Nx.equal(Nx.broadcast(expanded, shape), Nx.iota(shape, axis: axis))
    Nx.select(hot, values[1], values[0])
  end

  defp op(node = %{op_type: "GatherElements"}, [data, indices]) do
    axis = attr(node, "axis", 0)
    axis = if axis < 0, do: axis + Nx.rank(data), else: axis
    dim = Nx.axis_size(data, axis)

    Nx.take_along_axis(data, Nx.select(Nx.less(indices, 0), Nx.add(indices, dim), indices),
      axis: axis
    )
  end

  defp op(node = %{op_type: "TopK"}, [x, k]) do
    k = k |> Nx.to_flat_list() |> hd()
    rank = Nx.rank(x)
    axis = attr(node, "axis", -1)
    axis = if axis < 0, do: axis + rank, else: axis
    direction = if attr(node, "largest", 1) == 1, do: :desc, else: :asc
    # A stable sort keeps the lower index first among equal values, as ORT does.
    idx =
      x
      |> Nx.argsort(axis: axis, direction: direction, stable: true)
      |> Nx.slice_along_axis(0, k, axis: axis)

    [Nx.take_along_axis(x, idx, axis: axis), Nx.as_type(idx, :s64)]
  end

  defp op(node = %{op_type: "Einsum"}, operands) do
    [inputs, output] = node |> attr("equation") |> String.replace(" ", "") |> String.split("->")
    inputs = String.split(inputs, ",")
    letters = Enum.map(?A..?Z, &<<&1>>)

    expand = fn term, rank ->
      if String.contains?(term, "...") do
        n = rank - (String.length(term) - 3)
        String.replace(term, "...", Enum.join(Enum.take(letters, n)))
      else
        term
      end
    end

    terms =
      Enum.zip_with(inputs, operands, fn term, t ->
        expand.(term, Nx.rank(t)) |> String.graphemes()
      end)

    ell = terms |> Enum.map(&Enum.count(&1, fn c -> c in letters end)) |> Enum.max()

    output =
      output |> String.replace("...", Enum.join(Enum.take(letters, ell))) |> String.graphemes()

    einsum(terms, operands, output)
  end

  # 2-D GridSample: bilinear or nearest, zeros or border padding.
  defp op(node = %{op_type: "GridSample"}, [x, grid]) do
    {n, c, h, w} = Nx.shape(x)
    {_, ho, wo, 2} = Nx.shape(grid)
    align = attr(node, "align_corners", 0) == 1
    mode = attr(node, "mode", "linear")
    padding = attr(node, "padding_mode", "zeros")

    unnormalize = fn g, size ->
      if align,
        do: Nx.multiply(Nx.add(g, 1), (size - 1) / 2),
        else: Nx.divide(Nx.subtract(Nx.multiply(Nx.add(g, 1), size), 1), 2)
    end

    gx = unnormalize.(grid[[.., .., .., 0]], w)
    gy = unnormalize.(grid[[.., .., .., 1]], h)

    {gx, gy} =
      if padding == "border", do: {Nx.clip(gx, 0, w - 1), Nx.clip(gy, 0, h - 1)}, else: {gx, gy}

    flat = Nx.reshape(x, {n, c, h * w})

    sample = fn xi, yi ->
      valid =
        Nx.logical_and(
          Nx.logical_and(Nx.greater_equal(xi, 0), Nx.less(xi, w)),
          Nx.logical_and(Nx.greater_equal(yi, 0), Nx.less(yi, h))
        )

      idx =
        Nx.add(Nx.multiply(Nx.clip(yi, 0, h - 1), w), Nx.clip(xi, 0, w - 1))
        |> Nx.as_type(:s32)
        |> Nx.reshape({n, 1, ho * wo})
        |> Nx.broadcast({n, c, ho * wo})

      vals = Nx.take_along_axis(flat, idx, axis: 2) |> Nx.reshape({n, c, ho, wo})
      Nx.multiply(vals, valid |> Nx.new_axis(1) |> Nx.as_type(Nx.type(x)))
    end

    case mode do
      m when m in ["linear", "bilinear"] ->
        x0 = Nx.floor(gx)
        y0 = Nx.floor(gy)
        wx = Nx.subtract(gx, x0) |> Nx.new_axis(1)
        wy = Nx.subtract(gy, y0) |> Nx.new_axis(1)
        x1 = Nx.add(x0, 1)
        y1 = Nx.add(y0, 1)

        top =
          Nx.add(
            Nx.multiply(sample.(x0, y0), Nx.subtract(1, wx)),
            Nx.multiply(sample.(x1, y0), wx)
          )

        bottom =
          Nx.add(
            Nx.multiply(sample.(x0, y1), Nx.subtract(1, wx)),
            Nx.multiply(sample.(x1, y1), wx)
          )

        Nx.add(Nx.multiply(top, Nx.subtract(1, wy)), Nx.multiply(bottom, wy))

      "nearest" ->
        sample.(Nx.round(gx), Nx.round(gy))
    end
  end

  defp op(node = %{op_type: "Elu"}, [a]),
    do:
      Nx.select(
        Nx.less(a, 0),
        Nx.multiply(attr(node, "alpha", 1.0), Nx.subtract(Nx.exp(a), 1)),
        a
      )

  defp op(node = %{op_type: "EyeLike"}, [a]) do
    {r, c} = Nx.shape(a)
    type = if t = attr(node, "dtype"), do: dtype(t), else: Nx.type(a)
    Nx.equal(Nx.add(Nx.iota({r, 1}), attr(node, "k", 0)), Nx.iota({1, c})) |> Nx.as_type(type)
  end

  # The output shape depends on the data, so NonZero only works on values known at trace time.
  defp op(%{op_type: "NonZero"}, [a]) do
    unless known?(a) do
      raise ArgumentError,
            "NonZero needs a static input, pass the graph inputs it depends on as static_inputs"
    end

    rank = Nx.rank(a)
    a = if rank == 0, do: Nx.reshape(a, {1}), else: a

    coords =
      a
      |> Nx.to_flat_list()
      |> Enum.with_index()
      |> Enum.reject(fn {v, _} -> v == 0 end)
      |> Enum.map(&unravel(elem(&1, 1), Nx.shape(a)))

    if coords == [],
      do: :empty,
      else: coords |> Enum.zip_with(& &1) |> static_tensor(:s64)
  end

  defp op(%{op_type: type}, _), do: raise(ArgumentError, "unsupported ONNX op #{type}")

  defp conv_transpose(node, x, w, b) do
    spatial = Nx.rank(x) - 2
    kernel = w |> Nx.shape() |> Tuple.to_list() |> Enum.drop(2)
    strides = attr(node, "strides", List.duplicate(1, spatial))
    dilations = attr(node, "dilations", List.duplicate(1, spatial))
    pads = attr(node, "pads", List.duplicate(0, 2 * spatial))
    output_padding = attr(node, "output_padding", List.duplicate(0, spatial))

    padding =
      Enum.zip([
        kernel,
        dilations,
        Enum.take(pads, spatial),
        Enum.drop(pads, spatial),
        output_padding
      ])
      |> Enum.map(fn {k, d, lo, hi, op} -> {d * (k - 1) - lo, d * (k - 1) - hi + op} end)

    w =
      w
      |> Nx.transpose(axes: [1, 0 | Enum.to_list(2..(spatial + 1))])
      |> Nx.reverse(axes: Enum.to_list(2..(spatial + 1)))

    x
    |> Nx.conv(w,
      strides: 1,
      padding: padding,
      input_dilation: strides,
      kernel_dilation: dilations
    )
    |> add_channel_bias(b)
  end

  defp add_channel_bias(out, nil), do: out

  defp add_channel_bias(out, b),
    do:
      Nx.add(out, Nx.reshape(b, List.to_tuple([1, :auto] ++ List.duplicate(1, Nx.rank(out) - 2))))

  defp add_optional(t, nil), do: t
  defp add_optional(t, other), do: Nx.add(t, other)

  defp repeat_groups(t, channels) do
    groups = Nx.size(t)
    t |> Nx.new_axis(1) |> Nx.broadcast({groups, div(channels, groups)}) |> Nx.reshape({channels})
  end

  defp normalize_axis(axis, rank) when axis < 0, do: axis + rank
  defp normalize_axis(axis, _rank), do: axis

  defp gelu(x),
    do: Nx.multiply(Nx.multiply(x, 0.5), Nx.add(1, Nx.erf(Nx.divide(x, :math.sqrt(2)))))

  defp gelu_tanh(x) do
    inner =
      Nx.multiply(:math.sqrt(2 / :math.pi()), Nx.add(x, Nx.multiply(0.044715, Nx.pow(x, 3))))

    Nx.multiply(Nx.multiply(x, 0.5), Nx.add(1, Nx.tanh(inner)))
  end

  @doc false
  def matmul(a, b) do
    case {Nx.rank(a), Nx.rank(b)} do
      {_, 1} ->
        Nx.dot(a, [Nx.rank(a) - 1], b, [0])

      {1, rb} ->
        Nx.dot(a, [0], b, [rb - 2])

      {ra, 2} ->
        Nx.dot(a, [ra - 1], b, [0])

      {ra, rb} ->
        rank = max(ra, rb)
        batch = Enum.take(Tuple.to_list(broadcast_shape([Nx.shape(a), Nx.shape(b)])), rank - 2)
        a = Nx.broadcast(a, List.to_tuple(batch ++ Enum.take(Tuple.to_list(Nx.shape(a)), -2)))
        b = Nx.broadcast(b, List.to_tuple(batch ++ Enum.take(Tuple.to_list(Nx.shape(b)), -2)))
        axes = Enum.to_list(0..(rank - 3))
        Nx.dot(a, [rank - 1], axes, b, [rank - 2], axes)
    end
  end

  defp softmax(node, a, opset) do
    rank = Nx.rank(a)
    axis = attr(node, "axis", -1)
    axis = if axis < 0, do: axis + rank, else: axis

    cond do
      axis == rank - 1 ->
        softmax_last(a)

      opset >= 13 ->
        perm = (Enum.to_list(0..(rank - 1)) -- [axis]) ++ [axis]
        inverse = perm |> Enum.with_index() |> Enum.sort() |> Enum.map(&elem(&1, 1))
        a |> Nx.transpose(axes: perm) |> softmax_last() |> Nx.transpose(axes: inverse)

      true ->
        shape = Nx.shape(a)
        outer = shape |> Tuple.to_list() |> Enum.take(axis) |> Enum.product()
        a |> Nx.reshape({outer, :auto}) |> softmax_last() |> Nx.reshape(shape)
    end
  end

  defp einsum([term], [a], output) do
    sum_axes = for {c, i} <- Enum.with_index(term), c not in output, do: i
    kept = Enum.reject(term, &(&1 not in output))
    a = if sum_axes == [], do: a, else: Nx.sum(a, axes: sum_axes)
    Nx.transpose(a, axes: Enum.map(output, &Enum.find_index(kept, fn c -> c == &1 end)))
  end

  defp einsum([ta, tb], [a, b], output) do
    reduce_only = fn term, other, t ->
      axes = for {c, i} <- Enum.with_index(term), c not in output and c not in other, do: i

      if axes == [],
        do: {term, t},
        else: {Enum.reject(term, &(&1 not in output and &1 not in other)), Nx.sum(t, axes: axes)}
    end

    {ta, a} = reduce_only.(ta, tb, a)
    {tb, b} = reduce_only.(tb, ta, b)
    index = fn term, c -> Enum.find_index(term, &(&1 == c)) end
    batch = for c <- ta, c in tb and c in output, do: c
    contract = for c <- ta, c in tb and c not in output, do: c

    batch_first = fn term, t ->
      order = batch ++ (term -- batch)
      {order, Nx.transpose(t, axes: Enum.map(order, &index.(term, &1)))}
    end

    {ta, a} = batch_first.(ta, a)
    {tb, b} = batch_first.(tb, b)

    out =
      Nx.dot(
        a,
        Enum.map(contract, &index.(ta, &1)),
        Enum.map(batch, &index.(ta, &1)),
        b,
        Enum.map(contract, &index.(tb, &1)),
        Enum.map(batch, &index.(tb, &1))
      )

    got = batch ++ Enum.reject(ta, &(&1 in tb)) ++ Enum.reject(tb, &(&1 in ta))
    Nx.transpose(out, axes: Enum.map(output, &index.(got, &1)))
  end

  defp einsum([ta, tb | rest], [a, b | more], output) do
    keep = Enum.uniq(output ++ List.flatten(rest))
    mid = Enum.uniq(Enum.filter(ta ++ tb, &(&1 in keep)))
    einsum([mid | rest], [einsum([ta, tb], [a, b], mid) | more], output)
  end

  @doc false
  def broadcast_shape(shapes) do
    rank = shapes |> Enum.map(&tuple_size/1) |> Enum.max()

    shapes
    |> Enum.map(fn s -> List.duplicate(1, rank - tuple_size(s)) ++ Tuple.to_list(s) end)
    |> Enum.zip_with(&Enum.max/1)
    |> List.to_tuple()
  end

  defp optional_ints(nil, default), do: default
  defp optional_ints(t, _default), do: ints!(t)

  defp pad_source("edge", i, dim), do: min(max(i, 0), dim - 1)
  defp pad_source("reflect", i, dim) when abs(i) > dim - 1, do: 2 * (dim - 1) - abs(i)
  defp pad_source("reflect", i, _dim), do: abs(i)

  defp resize_axis(x, node, axis, in_size, out_size, scale) do
    ctm = attr(node, "coordinate_transformation_mode", "half_pixel")

    taps =
      for i <- 0..(out_size - 1),
          do: resize_taps(node, resize_source(ctm, i, in_size, out_size, scale), in_size, scale)

    width = taps |> Enum.map(&length/1) |> Enum.max()
    taps = Enum.map(taps, &(&1 ++ List.duplicate({0, 0.0}, width - length(&1))))
    indices = taps |> Enum.map(fn row -> Enum.map(row, &elem(&1, 0)) end) |> static_tensor(:s64)
    weights = taps |> Enum.map(fn row -> Enum.map(row, &elem(&1, 1)) end) |> static_tensor(:f32)

    weight_shape =
      x
      |> Nx.shape()
      |> Tuple.to_list()
      |> Enum.map(fn _ -> 1 end)
      |> List.replace_at(axis, out_size)
      |> List.to_tuple()

    for k <- 0..(width - 1) do
      w =
        weights
        |> Nx.slice_along_axis(k, 1, axis: 1)
        |> Nx.reshape(weight_shape)
        |> Nx.as_type(Nx.type(x))

      x
      |> Nx.take(indices |> Nx.slice_along_axis(k, 1, axis: 1) |> Nx.reshape({out_size}),
        axis: axis
      )
      |> Nx.multiply(w)
    end
    |> Enum.reduce(&Nx.add/2)
  end

  defp resize_source("half_pixel", i, _in, _out, scale), do: (i + 0.5) / scale - 0.5
  defp resize_source("pytorch_half_pixel", _i, _in, 1, _scale), do: 0.0
  defp resize_source("pytorch_half_pixel", i, _in, _out, scale), do: (i + 0.5) / scale - 0.5
  defp resize_source("align_corners", _i, _in, 1, _scale), do: 0.0

  defp resize_source("align_corners", i, in_size, out_size, _scale),
    do: i * (in_size - 1) / (out_size - 1)

  defp resize_source("asymmetric", i, _in, _out, scale), do: i / scale
  defp resize_source("tf_half_pixel_for_nn", i, _in, _out, scale), do: (i + 0.5) / scale

  # {input index, weight} pairs that make one output element.
  defp resize_taps(node, xo, in_size, scale) do
    mode = attr(node, "mode", "nearest")

    if mode in ["linear", "cubic"] and attr(node, "antialias", 0) == 1,
      do: antialias_taps(node, mode, xo, in_size, scale),
      else: interpolation_taps(node, mode, xo, in_size)
  end

  # Antialiasing widens the kernel by 1 / scale when downscaling, like Pillow. As in
  # ONNX Runtime, taps outside the input move to the edge unless exclude_outside is set,
  # and the weights are normalized.
  defp antialias_taps(node, mode, xo, in_size, scale) do
    a = attr(node, "cubic_coeff_a", -0.75)
    stretch = min(scale, 1.0)
    support = if(mode == "linear", do: 1.0, else: 2.0) / stretch
    window = 2 * ceil(support) + 1
    center = xo + 0.5
    low = floor(center - support + 0.5)
    high = low + min(max(floor(center + support + 0.5) - low, 0), window)

    taps =
      for j <- low..(high - 1)//1 do
        t = (j - xo) * stretch
        {j, if(mode == "linear", do: max(0.0, 1 - abs(t)), else: cubic_weight(a, t))}
      end

    if attr(node, "exclude_outside", 0) == 1,
      do: taps |> Enum.filter(fn {j, _} -> j >= 0 and j < in_size end) |> renormalize(),
      else: taps |> Enum.map(fn {j, w} -> {min(max(j, 0), in_size - 1), w} end) |> renormalize()
  end

  defp interpolation_taps(node, mode, xo, in_size) do
    clamp = &min(max(&1, 0), in_size - 1)

    case mode do
      "nearest" ->
        [{clamp.(nearest_index(attr(node, "nearest_mode", "round_prefer_floor"), xo)), 1.0}]

      "linear" ->
        xo = min(max(xo, 0.0), in_size - 1.0)
        x0 = floor(xo)
        d = xo - x0
        [{x0, 1 - d}, {clamp.(x0 + 1), d}]

      "cubic" ->
        a = attr(node, "cubic_coeff_a", -0.75)
        x0 = floor(xo)
        d = xo - x0

        taps =
          for {off, s} <- [{-1, d + 1}, {0, d}, {1, 1 - d}, {2, 2 - d}],
              do: {x0 + off, cubic_weight(a, s)}

        if attr(node, "exclude_outside", 0) == 1,
          do: renormalize(Enum.filter(taps, fn {j, _} -> j >= 0 and j < in_size end)),
          else: Enum.map(taps, fn {j, w} -> {clamp.(j), w} end)
    end
  end

  defp nearest_index(mode, xo) do
    half? = xo - Float.floor(xo) == 0.5

    case mode do
      "round_prefer_floor" -> if half?, do: floor(xo), else: round(xo)
      "round_prefer_ceil" -> if half?, do: ceil(xo), else: round(xo)
      "floor" -> floor(xo)
      "ceil" -> ceil(xo)
    end
  end

  defp cubic_weight(a, s) do
    s = abs(s)

    cond do
      s <= 1 -> (a + 2) * s ** 3 - (a + 3) * s ** 2 + 1
      s < 2 -> a * s ** 3 - 5 * a * s ** 2 + 8 * a * s - 4 * a
      true -> 0.0
    end
  end

  defp renormalize(taps) do
    total = taps |> Enum.map(&elem(&1, 1)) |> Enum.sum()
    Enum.map(taps, fn {j, w} -> {j, w / total} end)
  end

  @doc false
  def softmax_last(a) do
    max = Nx.reduce_max(a, axes: [-1], keep_axes: true)
    exp = Nx.exp(Nx.subtract(a, max))
    Nx.divide(exp, Nx.sum(exp, axes: [-1], keep_axes: true))
  end

  defp unravel(index, shape) do
    shape
    |> Tuple.to_list()
    |> Enum.reverse()
    |> Enum.map_reduce(index, fn dim, rest -> {rem(rest, dim), div(rest, dim)} end)
    |> elem(0)
    |> Enum.reverse()
  end

  @doc false
  def attr(node, name, default \\ nil) do
    case Enum.find(node.attribute, &(&1.name == name)) do
      nil -> default
      %{type: :INT, i: i} -> i
      %{type: :INTS, ints: ints} -> ints
      %{type: :FLOAT, f: f} -> f
      %{type: :FLOATS, floats: fs} -> fs
      %{type: :TENSOR, t: t} -> t
      %{type: :STRING, s: s} -> s
      %{type: :GRAPH, g: g} -> g
    end
  end

  @doc false
  def tensor(t, dir \\ nil)

  def tensor(t = %{data_location: :EXTERNAL, external_data: entries}, dir) do
    entries = Map.new(entries, &{&1.key, &1.value})
    path = Path.join(dir, entries["location"])
    offset = String.to_integer(entries["offset"] || "0")
    {:ok, file} = :file.open(path, [:read, :raw, :binary])
    {:ok, raw} = :file.pread(file, offset, String.to_integer(entries["length"]))
    :ok = :file.close(file)
    tensor(%{t | data_location: :DEFAULT, raw_data: raw}, dir)
  end

  def tensor(%{dims: dims}, _dir) when dims != [] and hd(dims) == 0, do: :empty

  # 4-bit tensors stay packed two values per byte: signed int4 as s8, uint4 as u8.
  def tensor(%{raw_data: raw, data_type: dt, dims: dims}, _dir) when dt in [21, 22] do
    {last, outer} = List.pop_at(dims, -1)

    raw
    |> Nx.from_binary(if(dt == 22, do: :s8, else: :u8), backend: Nx.BinaryBackend)
    |> Nx.reshape(List.to_tuple(outer ++ [div(last + 1, 2)]))
  end

  def tensor(t = %{raw_data: raw, data_type: dt, dims: dims}, _dir) do
    type = dtype(dt)

    flat =
      cond do
        raw not in [nil, ""] -> Nx.from_binary(raw, type, backend: Nx.BinaryBackend)
        t.float_data != [] -> static_tensor(t.float_data, type)
        t.int64_data != [] -> static_tensor(t.int64_data, type)
        t.int32_data != [] -> static_tensor(t.int32_data, type)
        t.double_data != [] -> static_tensor(t.double_data, type)
      end

    Nx.reshape(flat, List.to_tuple(dims))
  end

  @doc false
  def dtype(4), do: :u16
  def dtype(5), do: :s16
  def dtype(12), do: :u32
  def dtype(13), do: :u64
  def dtype(1), do: :f32
  def dtype(2), do: :u8
  def dtype(3), do: :s8
  def dtype(6), do: :s32
  def dtype(7), do: :s64
  def dtype(9), do: :u8
  def dtype(10), do: :f16
  def dtype(11), do: :f64
  def dtype(16), do: :bf16
end
