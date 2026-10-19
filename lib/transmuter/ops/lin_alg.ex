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
