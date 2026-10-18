defmodule Transmuter.Ops.Sequence do
  @moduledoc false

  # Sequence and optional operators. A sequence is {:sequence, elements} and an empty
  # optional is :none. The list work happens while tracing, so positions and split
  # sizes must be static, but the elements can be traced tensors.

  import Transmuter.Ops, only: [attr: 2, attr: 3, ints!: 1, normalize_axis: 2, static_tensor: 2]

  @op_types ~w(SequenceEmpty SequenceConstruct SequenceAt SequenceInsert SequenceErase
               SequenceLength SplitToSequence ConcatFromSequence Optional OptionalHasElement
               OptionalGetElement)

  @doc false
  def op_types, do: @op_types

  @doc false
  def apply(node, inputs), do: node |> op(inputs) |> List.wrap()

  defp op(%{op_type: "SequenceEmpty"}, []), do: {:sequence, []}

  defp op(%{op_type: "SequenceConstruct"}, inputs), do: {:sequence, inputs}

  defp op(%{op_type: "SequenceAt"}, [{:sequence, elements}, position]),
    do: Enum.at(elements, position!(position, length(elements), length(elements) - 1))

  defp op(%{op_type: "SequenceInsert"}, [{:sequence, elements}, value | rest]) do
    count = length(elements)
    at = if rest in [[], [nil]], do: count, else: position!(hd(rest), count, count)
    {:sequence, List.insert_at(elements, at, value)}
  end

  defp op(%{op_type: "SequenceErase"}, [{:sequence, elements} | rest]) do
    count = length(elements)
    at = if rest in [[], [nil]], do: count - 1, else: position!(hd(rest), count, count - 1)
    {:sequence, List.delete_at(elements, at)}
  end

  defp op(%{op_type: "SequenceLength"}, [{:sequence, elements}]),
    do: static_tensor(length(elements), :s64)

  defp op(node = %{op_type: "SplitToSequence"}, [a | rest]) do
    axis = normalize_axis(attr(node, "axis", 0), Nx.rank(a))
    dim = Nx.axis_size(a, axis)

    sizes =
      case rest do
        [split] when split != nil ->
          if Nx.rank(split) == 0 do
            chunk = hd(ints!(split))
            Enum.map(0..(dim - 1)//chunk, &min(chunk, dim - &1))
          else
            ints!(split)
          end

        _ ->
          List.duplicate(1, dim)
      end

    parts =
      sizes
      |> Enum.scan({0, 0}, fn size, {start, prev} -> {start + prev, size} end)
      |> Enum.map(fn {start, size} -> Nx.slice_along_axis(a, start, size, axis: axis) end)

    squeeze? = rest in [[], [nil]] and attr(node, "keepdims", 1) == 0
    {:sequence, if(squeeze?, do: Enum.map(parts, &Nx.squeeze(&1, axes: [axis])), else: parts)}
  end

  defp op(node = %{op_type: "ConcatFromSequence"}, [{:sequence, elements}]) do
    elements = Enum.reject(elements, &(&1 == :empty))
    rank = Nx.rank(hd(elements))

    if attr(node, "new_axis", 0) == 1,
      do: Nx.stack(elements, axis: normalize_axis(attr(node, "axis"), rank + 1)),
      else: Nx.concatenate(elements, axis: normalize_axis(attr(node, "axis"), rank))
  end

  defp op(%{op_type: "Optional"}, inputs) when inputs in [[], [nil]], do: :none
  defp op(%{op_type: "Optional"}, [value]), do: value

  defp op(%{op_type: "OptionalHasElement"}, inputs),
    do: static_tensor(if(inputs in [[], [nil], [:none]], do: 0, else: 1), :u8)

  defp op(%{op_type: "OptionalGetElement"}, [:none]),
    do: raise(ArgumentError, "the optional input is empty")

  defp op(%{op_type: "OptionalGetElement"}, [value]), do: value

  # A negative position counts from the end of the sequence.
  defp position!(position, count, last) do
    [given] = ints!(position)
    at = if given < 0, do: given + count, else: given

    unless at in 0..last//1 do
      raise ArgumentError, "position #{given} is out of range for a sequence of #{count}"
    end

    at
  end
end
