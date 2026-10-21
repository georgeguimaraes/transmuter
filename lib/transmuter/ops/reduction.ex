defmodule Transmuter.Ops.Reduction do
  @moduledoc """
  Reductions along axes, arg reductions, cumulative sums and products, and TopK.
  """

  import Transmuter.Ops, only: [attr: 3, normalize_axis: 2, scalar!: 1, values_or_attr: 4]

  @op_types ~w(ReduceSum ReduceMean CumSum TopK CumProd ReduceMax ReduceMin ReduceProd ReduceL2
               ReduceSumSquare ReduceL1 ReduceLogSum ReduceLogSumExp ArgMax ArgMin)

  @doc false
  def op_types, do: @op_types

  @doc false
  def apply(node, inputs), do: node |> op(inputs) |> List.wrap()

  # The axes come from the second input (opset 18) or the axes attribute. No axes reduce
  # every axis, or none with noop_with_empty_axes.
  defp op(node = %{op_type: "Reduce" <> _}, [a | rest]) do
    axes = values_or_attr(node, List.first(rest), "axes", [])
    opts = [keep_axes: attr(node, "keepdims", 1) == 1]

    cond do
      axes == [] and attr(node, "noop_with_empty_axes", 0) == 1 -> a
      axes == [] -> reduce(node.op_type, a, opts)
      true -> reduce(node.op_type, a, [axes: axes] ++ opts)
    end
  end

  # The compiled EMLX argmax ignores tie_break, so the last index of a tie is the first
  # index in the reversed axis, counted from the end.
  defp op(node = %{op_type: arg}, [a]) when arg in ["ArgMax", "ArgMin"] do
    axis = normalize_axis(attr(node, "axis", 0), Nx.rank(a))
    opts = [axis: axis, keep_axis: attr(node, "keepdims", 1) == 1, type: :s64]
    fun = if arg == "ArgMax", do: &Nx.argmax/2, else: &Nx.argmin/2

    if attr(node, "select_last_index", 0) == 1,
      do: Nx.subtract(Nx.axis_size(a, axis) - 1, fun.(Nx.reverse(a, axes: [axis]), opts)),
      else: fun.(a, opts)
  end

  # The exclusive forms shift the inclusive result by one along the axis, with the
  # identity element in the first place.
  defp op(node = %{op_type: cumulative}, [x, axis]) when cumulative in ["CumSum", "CumProd"] do
    axis = normalize_axis(scalar!(axis), Nx.rank(x))
    reverse = attr(node, "reverse", 0) == 1

    {out, identity} =
      if cumulative == "CumSum",
        do: {Nx.cumulative_sum(x, axis: axis, reverse: reverse), 0},
        else: {Nx.cumulative_product(x, axis: axis, reverse: reverse), 1}

    if attr(node, "exclusive", 0) == 1 do
      shift = if reverse, do: {-1, 1, 0}, else: {1, -1, 0}
      config = for a <- Nx.axes(x), do: if(a == axis, do: shift, else: {0, 0, 0})
      Nx.pad(out, Nx.tensor(identity, type: Nx.type(out)), config)
    else
      out
    end
  end

  defp op(node = %{op_type: "TopK"}, [x, k]) do
    k = scalar!(k)
    axis = normalize_axis(attr(node, "axis", -1), Nx.rank(x))
    largest? = attr(node, "largest", 1) == 1

    cond do
      k == 0 ->
        empty = {:empty, x |> Nx.shape() |> put_elem(axis, 0)}
        [empty, empty]

      # One value is an arg reduction, which keeps the lower index among equal values.
      k == 1 ->
        reduce = if largest?, do: &Nx.argmax/2, else: &Nx.argmin/2
        idx = reduce.(x, axis: axis, keep_axis: true, tie_break: :low)
        [Nx.take_along_axis(x, idx, axis: axis), Nx.as_type(idx, :s64)]

      true ->
        # EMLX sorts in descending order by negating, which wraps integers. A bitwise not
        # reverses the order of every integer type without wrapping.
        {keys, direction} =
          cond do
            not largest? -> {x, :asc}
            Nx.Type.integer?(Nx.type(x)) -> {Nx.bitwise_not(x), :asc}
            true -> {x, :desc}
          end

        # A stable sort keeps the lower index first among equal values, as ORT does.
        idx =
          keys
          |> Nx.argsort(axis: axis, direction: direction, stable: true)
          |> Nx.slice_along_axis(0, k, axis: axis)

        [Nx.take_along_axis(x, idx, axis: axis), Nx.as_type(idx, :s64)]
    end
  end

  defp reduce("ReduceSum", a, opts), do: Nx.sum(a, opts)
  defp reduce("ReduceMean", a, opts), do: Nx.mean(a, opts)
  defp reduce("ReduceMax", a, opts), do: Nx.reduce_max(a, opts)
  defp reduce("ReduceMin", a, opts), do: Nx.reduce_min(a, opts)
  defp reduce("ReduceProd", a, opts), do: Nx.product(a, opts)
  defp reduce("ReduceL1", a, opts), do: a |> Nx.abs() |> Nx.sum(opts)
  defp reduce("ReduceL2", a, opts), do: a |> Nx.pow(2) |> Nx.sum(opts) |> Nx.sqrt()
  defp reduce("ReduceSumSquare", a, opts), do: a |> Nx.pow(2) |> Nx.sum(opts)
  defp reduce("ReduceLogSum", a, opts), do: a |> Nx.sum(opts) |> Nx.log()
  defp reduce("ReduceLogSumExp", a, opts), do: log_sum_exp(a, opts)

  defp log_sum_exp(a, opts) do
    keep = Keyword.put(opts, :keep_axes, true)
    max = Nx.reduce_max(a, keep)
    sum = a |> Nx.subtract(max) |> Nx.exp() |> Nx.sum(keep) |> Nx.log() |> Nx.add(max)
    if opts[:keep_axes], do: sum, else: Nx.squeeze(sum, axes: opts[:axes] || Nx.axes(a))
  end
end
