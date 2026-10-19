defmodule Transmuter.Ops.Reduction do
  @moduledoc false

  # Reductions along axes, arg reductions, cumulative sums and products, and TopK.

  import Transmuter.Ops, only: [attr: 2, attr: 3, ints!: 1, normalize_axis: 2]

  @op_types ~w(ReduceSum ReduceMean CumSum TopK CumProd ReduceMax ReduceMin ReduceProd ReduceL2
               ReduceSumSquare ReduceL1 ReduceLogSum ReduceLogSumExp ArgMax ArgMin)

  @doc false
  def op_types, do: @op_types

  @doc false
  def apply(node, inputs), do: node |> op(inputs) |> List.wrap()

  defp op(node = %{op_type: "ReduceSum"}, [a | rest]) do
    axes = if rest == [] or rest == [nil], do: attr(node, "axes"), else: ints!(hd(rest))
    keep = attr(node, "keepdims", 1) == 1

    cond do
      axes in [nil, []] and attr(node, "noop_with_empty_axes", 0) == 1 -> a
      axes in [nil, []] -> Nx.sum(a, keep_axes: keep)
      true -> Nx.sum(a, axes: axes, keep_axes: keep)
    end
  end

  defp op(node = %{op_type: "ReduceMean"}, [a | rest]) do
    axes = if rest == [], do: attr(node, "axes"), else: ints!(hd(rest))
    Nx.mean(a, axes: axes, keep_axes: attr(node, "keepdims", 1) == 1)
  end

  defp op(node = %{op_type: reduce}, [a | rest])
       when reduce in [
              "ReduceMax",
              "ReduceMin",
              "ReduceProd",
              "ReduceL2",
              "ReduceSumSquare",
              "ReduceL1",
              "ReduceLogSum",
              "ReduceLogSumExp"
            ] do
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
      "ReduceL1" -> a |> Nx.abs() |> Nx.sum(opts)
      "ReduceLogSum" -> a |> Nx.sum(opts) |> Nx.log()
      "ReduceLogSumExp" -> log_sum_exp(a, opts)
    end
  end

  defp op(node = %{op_type: arg}, [a]) when arg in ["ArgMax", "ArgMin"] do
    0 = attr(node, "select_last_index", 0)
    opts = [axis: attr(node, "axis", 0), keep_axis: attr(node, "keepdims", 1) == 1, type: :s64]
    if arg == "ArgMax", do: Nx.argmax(a, opts), else: Nx.argmin(a, opts)
  end

  defp op(node = %{op_type: "CumSum"}, [x, axis]) do
    [axis] = ints!(axis)
    axis = if axis < 0, do: axis + Nx.rank(x), else: axis
    sum = Nx.cumulative_sum(x, axis: axis, reverse: attr(node, "reverse", 0) == 1)
    if attr(node, "exclusive", 0) == 1, do: Nx.subtract(sum, x), else: sum
  end

  defp op(node = %{op_type: "TopK"}, [x, k]) do
    [k] = ints!(k)
    rank = Nx.rank(x)
    axis = attr(node, "axis", -1)
    axis = if axis < 0, do: axis + rank, else: axis
    direction = if attr(node, "largest", 1) == 1, do: :desc, else: :asc

    if k == 0 do
      empty = {:empty, x |> Nx.shape() |> put_elem(axis, 0)}
      [empty, empty]
    else
      # A stable sort keeps the lower index first among equal values, as ORT does.
      idx =
        x
        |> Nx.argsort(axis: axis, direction: direction, stable: true)
        |> Nx.slice_along_axis(0, k, axis: axis)

      [Nx.take_along_axis(x, idx, axis: axis), Nx.as_type(idx, :s64)]
    end
  end

  defp op(node = %{op_type: "CumProd"}, [x, axis]) do
    axis = normalize_axis(axis |> ints!() |> hd(), Nx.rank(x))
    reverse = attr(node, "reverse", 0) == 1
    x = if reverse, do: Nx.reverse(x, axes: [axis]), else: x
    product = Nx.cumulative_product(x, axis: axis)

    product =
      if attr(node, "exclusive", 0) == 1 do
        ones = x |> Nx.slice_along_axis(0, 1, axis: axis) |> Nx.multiply(0) |> Nx.add(1)

        Nx.concatenate(
          [ones, Nx.slice_along_axis(product, 0, Nx.axis_size(x, axis) - 1, axis: axis)],
          axis: axis
        )
      else
        product
      end

    if reverse, do: Nx.reverse(product, axes: [axis]), else: product
  end

  defp log_sum_exp(a, opts) do
    keep = Keyword.put(opts, :keep_axes, true)
    max = Nx.reduce_max(a, keep)
    sum = a |> Nx.subtract(max) |> Nx.exp() |> Nx.sum(keep) |> Nx.log() |> Nx.add(max)
    if opts[:keep_axes], do: sum, else: Nx.squeeze(sum, axes: opts[:axes] || Nx.axes(a))
  end
end
