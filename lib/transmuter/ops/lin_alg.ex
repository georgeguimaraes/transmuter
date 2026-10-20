defmodule Transmuter.Ops.LinAlg do
  @moduledoc """
  Matrix products, Einsum and determinants.
  """

  import Transmuter.Ops, only: [attr: 2, attr: 3, matmul: 2]

  @op_types ~w(Gemm MatMul Einsum Det)

  @doc false
  def op_types, do: @op_types

  @doc false
  def apply(node, inputs), do: node |> op(inputs) |> List.wrap()

  defp op(node = %{op_type: "Gemm"}, [a, b | c]) do
    a = if attr(node, "transA", 0) == 1, do: Nx.transpose(a), else: a
    b = if attr(node, "transB", 0) == 1, do: Nx.transpose(b), else: b
    out = Nx.multiply(Nx.dot(a, b), attr(node, "alpha", 1.0))

    case c do
      [c] when c != nil -> Nx.add(out, Nx.multiply(c, attr(node, "beta", 1.0)))
      _ -> out
    end
  end

  defp op(%{op_type: "MatMul"}, [a, b]), do: matmul(a, b)

  # Terms become lists of axis labels: letters, and {:ellipsis, i} for the axes of an
  # ellipsis, numbered from the right so that they line up across operands. Without an
  # output term, the output has the ellipsis axes and the letters that appear once, sorted.
  defp op(node = %{op_type: "Einsum"}, operands) do
    {inputs, output} =
      case node |> attr("equation") |> String.replace(" ", "") |> String.split("->") do
        [inputs, output] -> {String.split(inputs, ","), output}
        [inputs] -> {String.split(inputs, ","), nil}
      end

    terms = Enum.zip_with(inputs, operands, &labels(&1, Nx.rank(&2)))

    width =
      terms |> Enum.map(&Enum.count(&1, fn l -> match?({:ellipsis, _}, l) end)) |> Enum.max()

    output =
      if output do
        labels(output, String.length(String.replace(output, "...", "")) + width)
      else
        letters = terms |> List.flatten() |> Enum.frequencies()
        once = for {l, 1} <- letters, is_binary(l), do: l
        Enum.map(width..1//-1, &{:ellipsis, &1}) ++ Enum.sort(once)
      end

    {terms, operands} = terms |> Enum.zip(operands) |> Enum.map(&diagonal/1) |> Enum.unzip()
    einsum(terms, broadcast_labels(terms, operands), output)
  end

  defp op(%{op_type: "Det"}, [x]) do
    case Nx.shape(x) do
      {_, _} ->
        Nx.LinAlg.determinant(x)

      shape ->
        {batch, [n, n]} = shape |> Tuple.to_list() |> Enum.split(-2)
        flat = Nx.reshape(x, {:auto, n, n})
        dets = for i <- 0..(Nx.axis_size(flat, 0) - 1), do: Nx.LinAlg.determinant(flat[i])
        dets |> Nx.stack() |> Nx.reshape(List.to_tuple(batch))
    end
  end

  defp labels(term, rank) do
    case String.split(term, "...") do
      [letters] ->
        String.graphemes(letters)

      [before, rest] ->
        count = rank - String.length(before) - String.length(rest)

        String.graphemes(before) ++
          Enum.map(count..1//-1, &{:ellipsis, &1}) ++ String.graphemes(rest)
    end
  end

  # A label that repeats in one term takes the diagonal of its axes.
  defp diagonal({term, t}) do
    case Enum.find(term, &(Enum.count(term, fn l -> l == &1 end) > 1)) do
      nil ->
        {term, t}

      label ->
        [first, second | _] = for {^label, i} <- Enum.with_index(term), do: i
        shape = Nx.shape(t)
        on = Nx.equal(Nx.iota(shape, axis: first), Nx.iota(shape, axis: second))

        t =
          on
          |> Nx.select(t, Nx.tensor(0, type: Nx.type(t)))
          |> Nx.sum(axes: [second])
          |> Nx.as_type(Nx.type(t))

        diagonal({List.delete_at(term, second), t})
    end
  end

  # An axis of size 1 broadcasts to the size of its label in the other operands.
  defp broadcast_labels(terms, operands) do
    sizes =
      Enum.zip_reduce(terms, operands, %{}, fn term, t, sizes ->
        term
        |> Enum.zip(Tuple.to_list(Nx.shape(t)))
        |> Enum.reduce(sizes, fn {l, size}, sizes ->
          Map.update(sizes, l, size, &max(&1, size))
        end)
      end)

    Enum.zip_with(terms, operands, fn term, t ->
      Nx.broadcast(t, term |> Enum.map(&sizes[&1]) |> List.to_tuple())
    end)
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
end
