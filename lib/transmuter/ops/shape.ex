defmodule Transmuter.Ops.Shape do
  @moduledoc false

  # Operators that build, cast, reshape, index, scatter or reorder tensors. Values
  # that decide output shapes must be static.

  import Transmuter.Ops,
    only: [
      attr: 2,
      attr: 3,
      dtype: 1,
      ints!: 1,
      known?: 1,
      normalize_axis: 2,
      static_tensor: 2,
      tensor: 1
    ]

  @op_types ~w(Cast Constant ConstantOfShape Expand Tile Squeeze Size Shape Gather Concat
               Unsqueeze Reshape Slice Transpose Identity CastLike Range Split Pad DepthToSpace
               SpaceToDepth ScatterElements GatherND ScatterND Flatten Trilu OneHot
               GatherElements EyeLike NonZero ReverseSequence)

  @doc false
  def op_types, do: @op_types

  @doc false
  def apply(node, inputs), do: node |> op(inputs) |> List.wrap()

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

  defp op(%{op_type: "Identity"}, [a]), do: a

  defp op(%{op_type: "CastLike"}, [a, like]), do: Nx.as_type(a, Nx.type(like))

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

  defp op(node = %{op_type: "SpaceToDepth"}, [x]) do
    b = attr(node, "blocksize")
    {n, c, h, w} = Nx.shape(x)

    x
    |> Nx.reshape({n, c, div(h, b), b, div(w, b), b})
    |> Nx.transpose(axes: [0, 3, 5, 1, 2, 4])
    |> Nx.reshape({n, c * b * b, div(h, b), div(w, b)})
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

  defp op(node = %{op_type: "Flatten"}, [a]) do
    axis = attr(node, "axis", 1)
    axis = if axis < 0, do: axis + Nx.rank(a), else: axis
    {outer, inner} = a |> Nx.shape() |> Tuple.to_list() |> Enum.split(axis)
    Nx.reshape(a, {Enum.product(outer), Enum.product(inner)})
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

  # Reverses the first sequence_lens[b] steps along the time axis of each batch item.
  defp op(node = %{op_type: "ReverseSequence"}, [x, lens]) do
    {batch_axis, time_axis} = {attr(node, "batch_axis", 1), attr(node, "time_axis", 0)}
    shape = Nx.shape(x)
    t = Nx.iota(shape, axis: time_axis, type: :s64)

    lens_shape =
      shape
      |> Tuple.to_list()
      |> Enum.map(fn _ -> 1 end)
      |> List.replace_at(batch_axis, :auto)
      |> List.to_tuple()

    lens = lens |> Nx.as_type(:s64) |> Nx.reshape(lens_shape)
    index = Nx.select(Nx.less(t, lens), Nx.subtract(Nx.subtract(lens, 1), t), t)
    Nx.take_along_axis(x, index, axis: time_axis)
  end

  defp optional_ints(nil, default), do: default
  defp optional_ints(t, _default), do: ints!(t)

  defp pad_source("edge", i, dim), do: min(max(i, 0), dim - 1)
  defp pad_source("reflect", i, dim) when abs(i) > dim - 1, do: 2 * (dim - 1) - abs(i)
  defp pad_source("reflect", i, _dim), do: abs(i)

  defp unravel(index, shape) do
    shape
    |> Tuple.to_list()
    |> Enum.reverse()
    |> Enum.map_reduce(index, fn dim, rest -> {rem(rest, dim), div(rest, dim)} end)
    |> elem(0)
    |> Enum.reverse()
  end
end
