defmodule Transmuter.Ops do
  @moduledoc false

  # ONNX operators as Nx calls. Values known at trace time live on Nx.BinaryBackend
  # ("static"), so shape arithmetic folds to plain integers while tracing. Nx has no
  # zero-size tensors, so a zero-size value is {:empty, shape}.

  alias Transmuter.Ops.{
    Elementwise,
    LinAlg,
    NN,
    Quantized,
    Reduction,
    Sequence,
    Shape,
    Signal,
    Text,
    Transformer,
    Vision
  }

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

  # Positions of the inputs whose zero size makes the output zero size, so the operator
  # only computes the output shape. Detection models run per-class branches with no boxes.
  @empty_through %{
    "Unsqueeze" => [0],
    "Squeeze" => [0],
    "Transpose" => [0],
    "Reshape" => [0],
    "Expand" => [0],
    "Gather" => [0, 1],
    "NonMaxSuppression" => [0, 1],
    "RoiAlign" => [0, 1, 2]
  }

  defp dispatch(node, inputs, opset) do
    empty_through = Map.get(@empty_through, node.op_type, [])

    cond do
      Enum.any?(empty_through, &empty?(Enum.at(inputs, &1))) ->
        [{:empty, node |> empty_shape(inputs) |> List.to_tuple()}]

      node.op_type in ["Softmax", "LogSoftmax", "Hardmax"] ->
        [along_axis(node, hd(inputs), opset)]

      node.op_type in Transformer.op_types() ->
        Transformer.apply(node, inputs)

      node.op_type in Quantized.op_types() ->
        Quantized.apply(node, inputs)

      node.op_type in Signal.op_types() ->
        Signal.apply(node, inputs)

      node.op_type in Vision.op_types() ->
        Vision.apply(node, inputs, opset)

      node.op_type in Text.op_types() ->
        Text.apply(node, inputs)

      node.op_type in Elementwise.op_types() ->
        Elementwise.apply(node, inputs)

      node.op_type in Reduction.op_types() ->
        Reduction.apply(node, inputs)

      node.op_type in LinAlg.op_types() ->
        LinAlg.apply(node, inputs)

      node.op_type in Shape.op_types() ->
        Shape.apply(node, inputs)

      node.op_type in NN.op_types() ->
        NN.apply(node, inputs)

      node.op_type in Sequence.op_types() ->
        Sequence.apply(node, inputs)

      true ->
        raise ArgumentError, "unsupported ONNX op #{node.op_type}"
    end
  end

  defp empty_shape(node = %{op_type: "Unsqueeze"}, [x | rest]) do
    axes = if rest == [], do: attr(node, "axes"), else: ints!(hd(rest))
    rank = length(shape_list(x)) + length(axes)

    axes
    |> Enum.map(&normalize_axis(&1, rank))
    |> Enum.sort()
    |> Enum.reduce(shape_list(x), &List.insert_at(&2, &1, 1))
  end

  defp empty_shape(node = %{op_type: "Squeeze"}, [x | rest]) do
    dims = shape_list(x)
    axes = if rest in [[], [nil]], do: attr(node, "axes"), else: ints!(hd(rest))

    if axes do
      axes = Enum.map(axes, &normalize_axis(&1, length(dims)))
      for {d, i} <- Enum.with_index(dims), i not in axes, do: d
    else
      Enum.reject(dims, &(&1 == 1))
    end
  end

  defp empty_shape(node = %{op_type: "Transpose"}, [x]) do
    dims = shape_list(x)
    perm = attr(node, "perm", Enum.reverse(0..(length(dims) - 1)//1))
    Enum.map(perm, &Enum.at(dims, &1))
  end

  defp empty_shape(node = %{op_type: "Reshape"}, [x, shape]) do
    dims = shape_list(x)
    copy? = attr(node, "allowzero", 0) == 0

    target =
      shape
      |> ints!()
      |> Enum.with_index()
      |> Enum.map(fn
        {0, i} when copy? -> Enum.at(dims, i)
        {d, _} -> d
      end)

    known = target |> Enum.reject(&(&1 == -1)) |> Enum.product()
    Enum.map(target, &if(&1 == -1, do: div(Enum.product(dims), max(known, 1)), else: &1))
  end

  # A dimension of 1 takes the target size, any other dimension stays, also 0.
  defp empty_shape(%{op_type: "Expand"}, [x, shape]) do
    {dims, target} = {shape_list(x), ints!(shape)}
    rank = max(length(dims), length(target))
    pad = &(List.duplicate(1, rank - length(&1)) ++ &1)
    Enum.zip_with(pad.(dims), pad.(target), fn d, t -> if d == 1, do: t, else: d end)
  end

  defp empty_shape(node = %{op_type: "Gather"}, [data, indices]) do
    dims = shape_list(data)

    {before, [_ | after_axis]} =
      Enum.split(dims, normalize_axis(attr(node, "axis", 0), length(dims)))

    before ++ shape_list(indices) ++ after_axis
  end

  defp empty_shape(%{op_type: "NonMaxSuppression"}, _inputs), do: [0, 3]

  defp empty_shape(node = %{op_type: "RoiAlign"}, [x, rois | _]) do
    [
      hd(shape_list(rois)),
      Enum.at(shape_list(x), 1),
      attr(node, "output_height", 1),
      attr(node, "output_width", 1)
    ]
  end

  defp shape_list(t), do: t |> shape_of() |> Tuple.to_list()

  @doc false
  def empty?({:empty, _shape}), do: true
  def empty?(_), do: false

  # The shape of a tensor or of a zero-size value.
  @doc false
  def shape_of({:empty, shape}), do: shape
  def shape_of(t), do: Nx.shape(t)

  # Values that Nx cannot hold live in the graph environment as host terms: a string
  # tensor is {:strings, shape, values}, a sequence is {:sequence, elements}, an empty
  # optional is :none and a zero-size tensor is {:empty, shape}.
  @doc false
  def static?({:empty, _shape}), do: true
  def static?(:none), do: true
  def static?({:strings, _shape, _values}), do: true
  def static?({:sequence, elements}), do: Enum.all?(elements, &static?/1)
  def static?(%Nx.Tensor{data: %Nx.BinaryBackend{}}), do: true
  def static?(_), do: false

  @doc false
  def ints!({:empty, _shape}), do: []

  def ints!(t), do: t |> known!("a value that decides a shape") |> Nx.to_flat_list()

  # Every read of a tensor value on the host goes through here, so a traced value
  # raises the one error that Transmuter.compile/3 breaks the graph on.
  @doc false
  def known!(value, description) do
    unless known?(value), do: raise(Transmuter.UnknownValueError, value: description)
    value
  end

  # Concrete values can be read on the host: static ones, and every tensor when the
  # model runs eagerly. Only values inside a traced function are unknown.
  @doc false
  def known?({:empty, _shape}), do: true
  def known?(:none), do: true
  def known?({:strings, _shape, _values}), do: true
  def known?({:sequence, elements}), do: Enum.all?(elements, &known?/1)
  def known?(%Nx.Tensor{data: %Nx.Defn.Expr{}}), do: false
  def known?(%Nx.Tensor{}), do: true

  @doc false
  def static_tensor(list, type), do: Nx.tensor(list, type: type, backend: Nx.BinaryBackend)

  @doc false
  def normalize_axis(axis, rank) when axis < 0, do: axis + rank
  def normalize_axis(axis, _rank), do: axis

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

  # Softmax, LogSoftmax and Hardmax work on one axis from opset 13. Before opset 13
  # they flatten the input to 2-D at the axis and work on the second dimension.
  defp along_axis(node, a, opset) do
    rank = Nx.rank(a)
    axis = attr(node, "axis", if(opset >= 13, do: -1, else: 1))
    axis = if axis < 0, do: axis + rank, else: axis

    last =
      case node.op_type do
        "Softmax" -> &softmax_last/1
        "LogSoftmax" -> &log_softmax_last/1
        "Hardmax" -> &hardmax_last/1
      end

    cond do
      axis == rank - 1 ->
        last.(a)

      opset >= 13 ->
        perm = (Enum.to_list(0..(rank - 1)) -- [axis]) ++ [axis]
        inverse = perm |> Enum.with_index() |> Enum.sort() |> Enum.map(&elem(&1, 1))
        a |> Nx.transpose(axes: perm) |> last.() |> Nx.transpose(axes: inverse)

      true ->
        shape = Nx.shape(a)
        outer = shape |> Tuple.to_list() |> Enum.take(axis) |> Enum.product()
        a |> Nx.reshape({outer, :auto}) |> last.() |> Nx.reshape(shape)
    end
  end

  defp log_softmax_last(a) do
    shifted = Nx.subtract(a, Nx.reduce_max(a, axes: [-1], keep_axes: true))
    Nx.subtract(shifted, Nx.log(Nx.sum(Nx.exp(shifted), axes: [-1], keep_axes: true)))
  end

  defp hardmax_last(a) do
    last = Nx.rank(a) - 1
    first = Nx.argmax(a, axis: last, keep_axis: true, tie_break: :low)
    a |> Nx.shape() |> Nx.iota(axis: last) |> Nx.equal(first) |> Nx.as_type(Nx.type(a))
  end

  @doc false
  def broadcast_shape(shapes) do
    rank = shapes |> Enum.map(&tuple_size/1) |> Enum.max()

    shapes
    |> Enum.map(fn s -> List.duplicate(1, rank - tuple_size(s)) ++ Tuple.to_list(s) end)
    |> Enum.zip_with(&Enum.max/1)
    |> List.to_tuple()
  end

  @doc false
  def softmax_last(a) do
    max = Nx.reduce_max(a, axes: [-1], keep_axes: true)
    exp = Nx.exp(Nx.subtract(a, max))
    Nx.divide(exp, Nx.sum(exp, axes: [-1], keep_axes: true))
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
      %{type: :STRINGS, strings: strings} -> strings
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

  def tensor(%{dims: dims} = t, dir) when dims != [] do
    if 0 in dims, do: {:empty, List.to_tuple(dims)}, else: tensor_data(t, dir)
  end

  def tensor(t, dir), do: tensor_data(t, dir)

  defp tensor_data(%{data_type: 8, string_data: values, dims: dims}, _dir),
    do: {:strings, List.to_tuple(dims), values}

  # 4-bit tensors stay packed two values per byte: signed int4 as s8, uint4 as u8.
  defp tensor_data(%{raw_data: raw, data_type: dt, dims: dims}, _dir) when dt in [21, 22] do
    {last, outer} = List.pop_at(dims, -1)

    raw
    |> Nx.from_binary(if(dt == 22, do: :s8, else: :u8), backend: Nx.BinaryBackend)
    |> Nx.reshape(List.to_tuple(outer ++ [div(last + 1, 2)]))
  end

  defp tensor_data(t = %{raw_data: raw, data_type: dt, dims: dims}, _dir) do
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
