defmodule Transmuter.Ops do
  @moduledoc """
  ONNX operators as Nx calls. Values known at trace time live on Nx.BinaryBackend ("static"), so shape arithmetic folds to plain integers while tracing. Nx has no zero-size tensors, so a zero-size value is {:empty, shape}.
  """

  alias Nx.Defn.Compiler

  alias Transmuter.Ops.{
    Elementwise,
    LinAlg,
    MoE,
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

  # Sums and softmax lose too much precision in f16 and bf16, so these ops run in f32 and
  # cast their outputs back (see widen/2). The normalizations and the attention ops are not
  # here: their EMLX kernels accumulate in f32, and their Nx operations widen themselves.
  @f32_ops ~w(ReduceSum ReduceMean ReduceSumSquare ReduceL1 ReduceL2 ReduceLogSum ReduceLogSumExp
              ReduceProd CumSum CumProd Softmax LogSoftmax LpNormalization
              InstanceNormalization GroupNormalization BatchNormalization MoE QMoE)

  @doc false
  def apply(node, inputs, opset) do
    if node.op_type in @f32_ops,
      do: widen(inputs, &dispatch(node, &1, opset)),
      else: dispatch(node, inputs, opset)
  end

  @doc """
  Calls `fun` with `values`, where f16 and bf16 tensors become f32, and casts the f32 tensors that `fun` returns back to that type. `fun` takes and returns a list.

  Sums, means and softmax in f16 or bf16 lose too much precision, so the Nx operations of such ops run in f32.
  """
  def widen(values, fun) do
    half =
      Enum.find_value(
        values,
        &(is_struct(&1, Nx.Tensor) and Nx.type(&1) in [{:f, 16}, {:bf, 16}] and Nx.type(&1))
      )

    if half do
      values
      |> Enum.map(
        &if(is_struct(&1, Nx.Tensor) and Nx.type(&1) == half, do: Nx.as_type(&1, :f32), else: &1)
      )
      |> fun.()
      |> Enum.map(
        &if(is_struct(&1, Nx.Tensor) and Nx.type(&1) == {:f, 32},
          do: Nx.as_type(&1, half),
          else: &1
        )
      )
    else
      fun.(values)
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
    "GatherND" => [1],
    "GatherElements" => [1],
    "OneHot" => [0],
    "MatMul" => [0, 1],
    "NonMaxSuppression" => [0, 1],
    "RoiAlign" => [0, 1, 2]
  }

  # The f16 value of each FP8 byte, by ONNX type. FLOAT8E5M2 (19) is the high byte of an
  # f16. Nx decodes FLOAT8E4M3FN (17) correctly, but not the subnormals of E5M2.
  @fp8_to_f16 %{
    17 =>
      Nx.iota({256}, type: :u8, backend: Nx.BinaryBackend)
      |> Nx.bitcast(:f8_e4m3fn)
      |> Nx.as_type(:f16)
      |> Nx.to_binary()
      |> then(&for(<<value::binary-2 <- &1>>, do: value))
      |> List.to_tuple(),
    19 => List.to_tuple(for b <- 0..255, do: <<b * 256::16-native>>)
  }

  defp dispatch(node, inputs, opset) do
    empty_outputs(node, inputs) || dispatch_op(node, inputs, opset)
  end

  # The outputs of an operator with a zero-size input, or nil when the operator computes them.
  defp empty_outputs(node, inputs) do
    empty_through = Map.get(@empty_through, node.op_type, [])

    cond do
      Enum.any?(empty_through, &empty?(Enum.at(inputs, &1))) ->
        shape = node |> empty_shape(inputs) |> List.to_tuple()

        # A product over an empty axis is zero, so its output is not empty.
        if 0 in Tuple.to_list(shape),
          do: [{:empty, shape}],
          else: [Nx.broadcast(Nx.tensor(0, type: zero_type(inputs)), shape)]

      node.op_type in Elementwise.op_types() and node.op_type != "SwiGLU" and
          Enum.any?(inputs, &empty?/1) ->
        [{:empty, broadcast_shape(for input <- inputs, input != nil, do: shape_of(input))}]

      true ->
        nil
    end
  end

  defp dispatch_op(node, inputs, opset) do
    cond do
      node.op_type in ["Softmax", "LogSoftmax", "Hardmax"] ->
        [along_axis(node, hd(inputs), opset)]

      node.op_type in Transformer.op_types() ->
        Transformer.apply(node, inputs, opset)

      node.op_type in MoE.op_types() ->
        MoE.apply(node, inputs)

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
    axes = values_or_attr(node, List.first(rest), "axes")
    rank = length(shape_list(x)) + length(axes)

    axes
    |> Enum.map(&normalize_axis(&1, rank))
    |> Enum.sort()
    |> Enum.reduce(shape_list(x), &List.insert_at(&2, &1, 1))
  end

  defp empty_shape(node = %{op_type: "Squeeze"}, [x | rest]) do
    dims = shape_list(x)
    axes = values_or_attr(node, List.first(rest), "axes")

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
      |> values!()
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
    {dims, target} = {shape_list(x), values!(shape)}
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

  defp empty_shape(node = %{op_type: "GatherND"}, [data, indices]) do
    {outer, [k]} = indices |> shape_list() |> Enum.split(-1)
    outer ++ Enum.drop(shape_list(data), attr(node, "batch_dims", 0) + k)
  end

  defp empty_shape(%{op_type: "GatherElements"}, [_data, indices]), do: shape_list(indices)

  defp empty_shape(node = %{op_type: "OneHot"}, [indices, depth, _values]) do
    dims = shape_list(indices)
    List.insert_at(dims, normalize_axis(attr(node, "axis", -1), length(dims) + 1), scalar!(depth))
  end

  defp empty_shape(%{op_type: "MatMul"}, [a, b]) do
    {a_dims, b_dims} = {shape_list(a), shape_list(b)}
    {a_batch, [m, _k]} = Enum.split(if(length(a_dims) == 1, do: [1 | a_dims], else: a_dims), -2)
    {b_batch, [_k, n]} = Enum.split(if(length(b_dims) == 1, do: b_dims ++ [1], else: b_dims), -2)
    batch = broadcast_shape([List.to_tuple(a_batch), List.to_tuple(b_batch)])

    Tuple.to_list(batch) ++
      if(length(a_dims) == 1, do: [], else: [m]) ++ if(length(b_dims) == 1, do: [], else: [n])
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

  defp zero_type(inputs) do
    case Enum.find(inputs, &is_struct(&1, Nx.Tensor)) do
      nil -> :f32
      t -> Nx.type(t)
    end
  end

  @doc false
  def empty?({:empty, _shape}), do: true
  def empty?(_), do: false

  @doc """
  Returns the shape of a tensor or of a zero-size value.
  """
  def shape_of({:empty, shape}), do: shape
  def shape_of(t), do: Nx.shape(t)

  @doc """
  Returns whether a value is known before the graph runs.

  Values that Nx cannot hold live in the graph environment as host terms: a string tensor is `{:strings, shape, values}`, a sequence is `{:sequence, elements}`, an empty optional is `:none` and a zero-size tensor is `{:empty, shape}`.
  """
  def static?({:empty, _shape}), do: true
  def static?(:none), do: true
  def static?({:strings, _shape, _values}), do: true
  def static?({:sequence, elements}), do: Enum.all?(elements, &static?/1)
  def static?(%Nx.Tensor{data: %Nx.BinaryBackend{}}), do: true
  def static?(_), do: false

  @doc false
  def values!({:empty, _shape}), do: []
  def values!(t), do: t |> known!("a value that decides a shape") |> Nx.to_flat_list()

  @doc false
  def scalar!(t), do: t |> values!() |> hd()

  @doc false
  def values_or_attr(node, input, name, default \\ nil)
  def values_or_attr(node, nil, name, default), do: attr(node, name, default)
  def values_or_attr(_node, input, _name, _default), do: values!(input)

  @doc """
  Returns the host value of `value`, or raises `Transmuter.UnknownValueError` when the value is not known on the host. A literal of a traced function and a large value that it folds from static values give their host tensor, and any other known value gives itself, so callers read the value that this function returns.

  Every read of a tensor value on the host goes through here, so a traced value raises the one error that `Transmuter.compile/3` breaks the graph on.
  """
  def known!(%Nx.Tensor{data: %Nx.Defn.Expr{op: :tensor, args: [value]}}, _description),
    do: value

  def known!(
        %Nx.Tensor{data: %Nx.Defn.Expr{op: :metadata, args: [_, %{transmuter_value: value}]}},
        _description
      ),
      do: value

  def known!(value, description) do
    unless known?(value), do: raise(Transmuter.UnknownValueError, value: description)
    value
  end

  @doc """
  Returns whether the host can read a value.

  Concrete values can be read on the host: static ones, literals of a traced function, large values that it folds from static values, and every tensor when the model runs eagerly. Other values inside a traced function are unknown.
  """
  def known?({:empty, _shape}), do: true
  def known?(:none), do: true
  def known?({:strings, _shape, _values}), do: true
  def known?({:sequence, elements}), do: Enum.all?(elements, &known?/1)

  def known?(%Nx.Tensor{data: %Nx.Defn.Expr{op: :metadata, args: [_, %{transmuter_value: _}]}}),
    do: true

  def known?(%Nx.Tensor{data: %Nx.Defn.Expr{op: op}}), do: op == :tensor
  def known?(%Nx.Tensor{}), do: true

  @doc false
  def digest(t = %Nx.Tensor{}), do: {Nx.shape(t), Nx.type(t), :erlang.md5(Nx.to_binary(t))}
  def digest({:sequence, elements}), do: Enum.map(elements, &digest/1)
  def digest(other), do: :erlang.md5(:erlang.term_to_binary(other))

  @doc """
  Returns whether EMLX computes `tensor`: in a function that EMLX compiles, or eagerly on `EMLX.Backend`.

  Operators then use the fused kernels of `EMLX.Fast`. Other compilers, such as `Nx.Defn.Evaluator`, keep the plain Nx operations, also on EMLX tensors.
  """
  def emlx?(tensor) do
    case Compiler.current() do
      nil -> match?(%Nx.Tensor{data: %{__struct__: EMLX.Backend}}, tensor)
      compiler -> compiler == EMLX
    end
  end

  @doc """
  Normalizes `x` over `axes` with `fun`, a kernel of `EMLX.Fast` that normalizes the last axis.

  The normalized axes become one axis, and each weight becomes a vector of this axis. `fun` gets this view of x and the weights.
  """
  def emlx_norm(fun, x, axes, weights) do
    shape = Nx.shape(x)
    {outer, inner} = shape |> Tuple.to_list() |> Enum.split(hd(axes))
    size = Enum.product(inner)

    flat =
      &(&1 |> Nx.as_type(Nx.type(x)) |> Nx.broadcast(List.to_tuple(inner)) |> Nx.reshape({size}))

    view = Nx.reshape(x, List.to_tuple(outer ++ [size]))
    fun |> apply([view | Enum.map(weights, flat)]) |> Nx.reshape(shape)
  end

  @doc false
  def static_tensor([], _type), do: {:empty, {0}}
  def static_tensor(list, type), do: Nx.tensor(list, type: type, backend: Nx.BinaryBackend)

  @doc false
  def normalize_axis(axis, rank) when axis < 0, do: axis + rank
  def normalize_axis(axis, _rank), do: axis

  @doc false
  def optional(inputs, count), do: Enum.take(inputs ++ List.duplicate(nil, count), count)

  @doc false
  def add_optional(t, nil), do: t
  def add_optional(t, other), do: Nx.add(t, other)

  @doc """
  Splits `t` along `axis` into parts of `sizes`. A part of size 0 is a zero-size value.
  """
  def split_along(t, sizes, axis) do
    sizes
    |> Enum.map_reduce(0, fn
      0, start -> {{:empty, t |> Nx.shape() |> put_elem(axis, 0)}, start}
      size, start -> {Nx.slice_along_axis(t, start, size, axis: axis), start + size}
    end)
    |> elem(0)
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

      # A plain matrix applies to every batch of b without a broadcast. The dot puts its
      # rows first, and the transpose moves them before the last axis.
      {2, rb} ->
        a
        |> Nx.dot([1], b, [rb - 2])
        |> Nx.transpose(axes: Enum.to_list(1..(rb - 2)) ++ [0, rb - 1])

      {ra, rb} ->
        rank = max(ra, rb)
        batch = Enum.take(Tuple.to_list(broadcast_shape([Nx.shape(a), Nx.shape(b)])), rank - 2)
        a = Nx.broadcast(a, List.to_tuple(batch ++ Enum.take(Tuple.to_list(Nx.shape(a)), -2)))
        b = Nx.broadcast(b, List.to_tuple(batch ++ Enum.take(Tuple.to_list(Nx.shape(b)), -2)))
        axes = Enum.to_list(0..(rank - 3))
        Nx.dot(a, [rank - 1], axes, b, [rank - 2], axes)
    end
  end

  @doc """
  Computes the `Softmax`, `LogSoftmax` or `Hardmax` of `node` along its axis.

  These operators work on one axis from opset 13. Before opset 13, they flatten the input to 2-D at the axis and work on the second dimension.
  """
  def along_axis(node, a, opset) do
    rank = Nx.rank(a)
    axis = normalize_axis(attr(node, "axis", if(opset >= 13, do: -1, else: 1)), rank)

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

  @doc """
  Returns the shape that ONNX broadcasting gives for `shapes`.

  A dimension of 0 takes the place of 1 like any other size.
  """
  def broadcast_shape(shapes) do
    rank = shapes |> Enum.map(&tuple_size/1) |> Enum.max()

    shapes
    |> Enum.map(fn s -> List.duplicate(1, rank - tuple_size(s)) ++ Tuple.to_list(s) end)
    |> Enum.zip_with(fn dims -> Enum.find(dims, 1, &(&1 != 1)) end)
    |> List.to_tuple()
  end

  @doc """
  Computes `(x - mean) / sqrt(variance + eps)` over `axes`.

  It centers x once and reuses it, where `Nx.variance` would compute the mean and center x again.
  """
  def normalize(x, axes, eps) do
    centered = Nx.subtract(x, Nx.mean(x, axes: axes, keep_axes: true))
    var = centered |> Nx.multiply(centered) |> Nx.mean(axes: axes, keep_axes: true)
    Nx.multiply(centered, Nx.rsqrt(Nx.add(var, eps)))
  end

  @doc """
  Rounds half to even, as ONNX does, on every backend.

  Backends round ties differently: MLX to even, `Nx.BinaryBackend` away from zero. A tie becomes twice its rounded half, which is never a tie, so the rounding mode of the backend does not matter. Infinities and NaN pass through, because `Nx.BinaryBackend` cannot round them.
  """
  def round_half_even(t) do
    finite = t |> Nx.is_infinity() |> Nx.logical_or(Nx.is_nan(t)) |> Nx.logical_not()
    safe = Nx.select(finite, t, 0)
    tie = Nx.equal(Nx.subtract(safe, Nx.floor(safe)), 0.5)
    rounded = Nx.select(tie, Nx.multiply(Nx.round(Nx.divide(safe, 2)), 2), Nx.round(safe))
    Nx.select(finite, rounded, t)
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

  def tensor(t = %{data_location: :EXTERNAL}, dir), do: t |> read_external(dir) |> tensor(dir)

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

  # FP8 data loads as f16, which holds every FLOAT8E4M3FN and FLOAT8E5M2 value exactly.
  # A table maps each byte to its f16 value. Without raw data, int32_data holds one byte
  # in each element.
  defp tensor_data(t = %{data_type: dt, dims: dims}, _dir) when is_map_key(@fp8_to_f16, dt) do
    table = @fp8_to_f16[dt]

    bytes =
      if t.raw_data in [nil, ""],
        do: for(b <- t.int32_data, into: <<>>, do: <<b>>),
        else: t.raw_data

    for(<<b <- bytes>>, into: <<>>, do: elem(table, b))
    |> Nx.from_binary(:f16, backend: Nx.BinaryBackend)
    |> Nx.reshape(List.to_tuple(dims))
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

  @doc """
  Reads the data of a tensor that keeps it in a file next to the model.
  """
  def read_external(t = %{data_location: :EXTERNAL, external_data: entries}, dir) do
    entries = Map.new(entries, &{&1.key, &1.value})
    path = Path.join(dir, entries["location"])
    offset = String.to_integer(entries["offset"] || "0")
    {:ok, file} = :file.open(path, [:read, :raw, :binary])
    {:ok, raw} = :file.pread(file, offset, String.to_integer(entries["length"]))
    :ok = :file.close(file)
    %{t | data_location: :DEFAULT, external_data: [], raw_data: raw}
  end

  def read_external(t, _dir), do: t

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

  def dtype(dt) when dt in 17..20 do
    raise ArgumentError,
          "Transmuter loads FP8 tensors as f16, but it cannot compute FP8 values (ONNX type #{dt})"
  end
end
