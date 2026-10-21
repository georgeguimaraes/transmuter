defmodule Transmuter.Ops.Packed do
  @moduledoc """
  Unpacks sub-byte integers and dequantizes block-wise weights, for the quantized operators and `Transmuter.EMLX`. Packed values hold the lowest bits first, so element 2i of a 4-bit tensor is the low nibble of byte i.
  """

  @doc """
  Unpacks the `bits`-wide unsigned values of the bytes on the last axis of `packed`. 8-bit values stay as they are.
  """
  def unpack(packed, 8), do: packed

  def unpack(packed, bits) do
    per = div(8, bits)
    shifts = Nx.multiply(Nx.iota({per}, type: :s16), bits)
    {last, shape} = packed |> Nx.shape() |> Tuple.to_list() |> List.pop_at(-1)

    packed
    |> Nx.as_type(:s16)
    |> Nx.bitwise_and(255)
    |> Nx.new_axis(-1)
    |> Nx.right_shift(shifts)
    |> Nx.bitwise_and(2 ** bits - 1)
    |> Nx.reshape(List.to_tuple(shape ++ [last * per]))
  end

  @doc """
  Computes `(q - zero) * scales` for each block of the last axis of `q`. `scales` has one value for each block, and `zero` has one value for each block or is a scalar.
  """
  def dequantize(q, zero, scales) do
    blocks = Tuple.insert_at(Nx.shape(scales), Nx.rank(scales), :auto)

    q
    |> Nx.reshape(blocks)
    |> Nx.subtract(Nx.new_axis(zero, -1))
    |> Nx.multiply(Nx.new_axis(scales, -1))
    |> Nx.reshape(Nx.shape(q))
  end
end
