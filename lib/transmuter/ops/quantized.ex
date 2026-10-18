defmodule Transmuter.Ops.Quantized do
  @moduledoc false

  # Quantized weights are dequantized to floats inside the graph. Packed sub-byte
  # values hold the lowest bits first, so element 2i of a 4-bit tensor is the low
  # nibble of byte i.

  import Transmuter.Ops, only: [attr: 2, attr: 3, matmul: 2]

  @op_types ~w(MatMulNBits GatherBlockQuantized DequantizeLinear QuantizeLinear
               DynamicQuantizeLinear MatMulInteger QLinearMatMul QLinearConv QLinearAdd QLinearMul)

  @doc false
  def op_types, do: @op_types

  @doc false
  def apply(node, inputs), do: node |> op(inputs) |> List.wrap()

  # com.microsoft MatMulNBits: B holds N rows of K weights in blocks of block_size.
  defp op(node = %{op_type: "MatMulNBits"}, [a, b, scales | rest]) do
    [zero_points, nil, bias] = Enum.take(rest ++ [nil, nil, nil], 3)

    {k, n, bits, block_size} =
      {attr(node, "K"), attr(node, "N"), attr(node, "bits", 4), attr(node, "block_size")}

    blocks = div(k + block_size - 1, block_size)
    type = Nx.type(a)

    # B given as an {N, K} weight (for example an EMLX quantized tensor) needs no unpacking.
    if Nx.shape(b) == {n, k} do
      out = Nx.dot(a, [-1], b, [1])
      if bias, do: Nx.add(out, bias), else: out
    else
      dequantized_matmul(a, b, scales, zero_points, bias, {k, n, bits, block_size, blocks, type})
    end
  end

  # com.microsoft GatherBlockQuantized: gathers rows of a table quantized along its last axis.
  defp op(node = %{op_type: "GatherBlockQuantized"}, [data, indices, scales | rest]) do
    zero_points = List.first(rest)
    0 = attr(node, "gather_axis", 0)
    last = Nx.rank(data) - 1
    ^last = normalize_axis(attr(node, "quantize_axis", 1), Nx.rank(data))
    bits = attr(node, "bits", 4)
    block_size = attr(node, "block_size", 128)
    type = Nx.type(scales)

    q = data |> Nx.take(indices) |> unpack(bits) |> Nx.as_type(type)
    blocks = Nx.axis_size(scales, -1)
    shape = Nx.shape(q) |> put_elem(Nx.rank(q) - 1, blocks)
    scales = Nx.take(scales, indices)

    zero =
      block_zero_points(
        zero_points && Nx.take(zero_points, indices),
        shape,
        bits,
        type,
        signed?(data)
      )

    q
    |> Nx.reshape(Tuple.insert_at(shape, tuple_size(shape), block_size))
    |> Nx.subtract(Nx.new_axis(zero, -1))
    |> Nx.multiply(Nx.new_axis(scales, -1))
    |> Nx.reshape(Nx.shape(q))
  end

  defp op(node = %{op_type: "DequantizeLinear"}, [x, scale | rest]) do
    zero = List.first(rest)
    type = Nx.type(scale)
    scale = expand_quant_param(scale, x, node)
    zero = if zero, do: zero |> Nx.as_type(type) |> expand_quant_param(x, node), else: 0
    x |> Nx.as_type(type) |> Nx.subtract(zero) |> Nx.multiply(scale)
  end

  defp op(node = %{op_type: "QuantizeLinear"}, [x, scale | rest]) do
    zero = List.first(rest)
    out_type = if zero, do: Nx.type(zero), else: {:u, 8}
    scale = expand_quant_param(scale, x, node)
    zero = if zero, do: zero |> Nx.as_type(Nx.type(x)) |> expand_quant_param(x, node), else: 0
    x |> Nx.divide(scale) |> round_half_even() |> Nx.add(zero) |> saturate(out_type)
  end

  defp op(%{op_type: "DynamicQuantizeLinear"}, [x]) do
    low = Nx.min(Nx.reduce_min(x), 0)
    high = Nx.max(Nx.reduce_max(x), 0)
    scale = Nx.divide(Nx.subtract(high, low), 255)
    safe = Nx.select(Nx.equal(scale, 0), 1, scale)
    zero = low |> Nx.negate() |> Nx.divide(safe) |> round_half_even() |> saturate({:u, 8})
    y = x |> Nx.divide(safe) |> round_half_even() |> Nx.add(zero) |> saturate({:u, 8})
    [y, scale, zero]
  end

  # Accumulates in f32, which is exact while partial sums stay under 2^24.
  defp op(%{op_type: "MatMulInteger"}, [a, b | rest]) do
    [a_zero, b_zero] = Enum.take(rest ++ [nil, nil], 2)
    a = a |> Nx.as_type(:f32) |> subtract_zero(a_zero, Nx.rank(a) - 2)
    b = b |> Nx.as_type(:f32) |> subtract_zero(b_zero, Nx.rank(b) - 1)
    a |> matmul(b) |> Nx.round() |> Nx.as_type(:s32)
  end

  # Integer matmul on zero-shifted values, then requantized: y = round(acc * a_scale * b_scale / y_scale) + y_zero.
  defp op(%{op_type: "QLinearMatMul"}, [a, a_scale, a_zero, b, b_scale, b_zero, y_scale, y_zero]) do
    a = a |> Nx.as_type(:f32) |> subtract_zero(a_zero, Nx.rank(a) - 2)
    b = b |> Nx.as_type(:f32) |> subtract_zero(b_zero, Nx.rank(b) - 1)
    acc = matmul(a, b)

    multiplier =
      Nx.divide(
        Nx.multiply(
          per_axis(a_scale, Nx.rank(acc), Nx.rank(acc) - 2),
          per_axis(b_scale, Nx.rank(acc), Nx.rank(acc) - 1)
        ),
        y_scale
      )

    acc
    |> Nx.multiply(multiplier)
    |> round_half_even()
    |> Nx.add(Nx.as_type(y_zero, :f32))
    |> saturate(Nx.type(y_zero))
  end

  # com.microsoft QLinearAdd: c = (a - a_zero) * a_scale / c_scale + (b - b_zero) * b_scale / c_scale + c_zero.
  # ONNX Runtime on ARM adds in fixed point when both scale ratios fit a 22-bit multiplier,
  # and so does this clause. Integer math gives the same result on every device. The
  # x86 kernel of ONNX Runtime adds in f32, and can round a tie to the other integer.
  defp op(%{op_type: "QLinearAdd"}, [a, a_scale, a_zero, b, b_scale, b_zero, c_scale | rest]) do
    c_zero = List.first(rest)
    type = Nx.type(a)

    case add_multipliers(a_scale, b_scale, c_scale) do
      {shift, a_multiplier, b_multiplier} ->
        shifted = &(&1 |> Nx.as_type(:s32) |> Nx.subtract(int_zero(&2)))

        acc =
          Nx.add(
            Nx.multiply(shifted.(a, a_zero), a_multiplier),
            Nx.multiply(shifted.(b, b_zero), b_multiplier)
          )

        # Subtract 1 from a negative sum, then shift right with rounding, so a tie rounds away from zero.
        acc
        |> Nx.add(Nx.right_shift(acc, 31))
        |> Nx.as_type(:s64)
        |> Nx.add(2 ** (shift - 1))
        |> Nx.right_shift(shift)
        |> Nx.clip(-32_768, 32_767)
        |> Nx.add(int_zero(c_zero))
        |> saturate(type)

      nil ->
        dequantize = &(&1 |> Nx.as_type(:f32) |> Nx.subtract(zero_or(&3)) |> Nx.multiply(&2))

        dequantize.(a, a_scale, a_zero)
        |> Nx.add(dequantize.(b, b_scale, b_zero))
        |> Nx.divide(c_scale)
        |> Nx.add(zero_or(c_zero))
        |> round_half_even()
        |> saturate(type)
    end
  end

  # com.microsoft QLinearMul: c = (a - a_zero) * (b - b_zero) * a_scale * b_scale / c_scale + c_zero,
  # with the zero point added before the rounding, as in ONNX Runtime.
  defp op(%{op_type: "QLinearMul"}, [a, a_scale, a_zero, b, b_scale, b_zero, c_scale | rest]) do
    a
    |> Nx.as_type(:f32)
    |> Nx.subtract(zero_or(a_zero))
    |> Nx.multiply(Nx.subtract(Nx.as_type(b, :f32), zero_or(b_zero)))
    |> Nx.multiply(Nx.divide(Nx.multiply(a_scale, b_scale), c_scale))
    |> Nx.add(zero_or(List.first(rest)))
    |> round_half_even()
    |> saturate(Nx.type(a))
  end

  # Integer convolution with per-tensor or per-output-channel weight scales, then requantized.
  defp op(node = %{op_type: "QLinearConv"}, [
         x,
         x_scale,
         x_zero,
         w,
         w_scale,
         w_zero,
         y_scale,
         y_zero | bias
       ]) do
    x = Nx.subtract(Nx.as_type(x, :f32), Nx.as_type(Nx.reshape(x_zero, {}), :f32))
    w = Nx.subtract(Nx.as_type(w, :f32), per_axis(w_zero, Nx.rank(w), 0))
    # The sum of integer products is an integer, but a large f32 convolution can be off
    # by a fraction (seen on the MLX GPU in vgg16-12-int8), so the sum is rounded back.
    acc = Transmuter.Ops.apply(%{node | op_type: "Conv"}, [x, w], 13) |> hd() |> Nx.round()
    channel = &per_axis(&1, Nx.rank(acc), 1)
    bias = List.first(bias)
    acc = if bias, do: Nx.add(acc, channel.(bias)), else: acc

    multiplier =
      Nx.divide(
        Nx.multiply(Nx.as_type(x_scale, :f32), channel.(w_scale)),
        Nx.as_type(y_scale, :f32)
      )

    acc
    |> Nx.multiply(multiplier)
    |> round_half_even()
    |> Nx.add(Nx.as_type(y_zero, :f32))
    |> saturate(Nx.type(y_zero))
  end

  defp dequantized_matmul(a, b, scales, zero_points, bias, {k, n, bits, block_size, blocks, type}) do
    q = b |> unpack(bits) |> Nx.reshape({n, blocks, block_size}) |> Nx.as_type(type)
    scales = scales |> Nx.reshape({n, blocks, 1}) |> Nx.as_type(type)
    zero = block_zero_points(zero_points, {n, blocks}, bits, type)
    zero = if Nx.rank(zero) == 2, do: Nx.new_axis(zero, 2), else: zero

    w =
      q
      |> Nx.subtract(zero)
      |> Nx.multiply(scales)
      |> Nx.reshape({n, blocks * block_size})
      |> Nx.slice_along_axis(0, k, axis: 1)

    out = Nx.dot(a, [-1], w, [1])
    if bias, do: Nx.add(out, bias), else: out
  end

  defp zero_or(nil), do: 0
  defp zero_or(zero), do: Nx.as_type(zero, :f32)

  defp int_zero(nil), do: 0
  defp int_zero(zero), do: Nx.as_type(zero, :s32)

  # The fixed-point parameters of MlasCalcQLinearAddParameters in ONNX Runtime, or nil
  # when the scales are not known or the ratios are out of its range.
  defp add_multipliers(a_scale, b_scale, c_scale) do
    if Enum.all?([a_scale, b_scale, c_scale], &Transmuter.Ops.known?/1) do
      [a_ratio, b_ratio] =
        for scale <- [a_scale, b_scale] do
          Nx.divide(f32_host(scale), f32_host(c_scale)) |> Nx.to_number()
        end

      <<_sign::1, exponent::8, _fraction::23>> = <<max(a_ratio, b_ratio)::float-32>>
      shift = 21 - (exponent - 127)
      multiplier = &(&1 |> Kernel.*(2 ** shift) |> round_even())
      {a_multiplier, b_multiplier} = {multiplier.(a_ratio), multiplier.(b_ratio)}

      if Enum.all?([a_ratio, b_ratio], &(&1 >= 2 ** -14 and &1 < 256)) and shift in 13..31 and
           max(a_multiplier, b_multiplier) in 0x200000..0x3FFFFF,
         do: {shift, a_multiplier, b_multiplier}
    end
  end

  defp f32_host(t),
    do: t |> Nx.backend_copy(Nx.BinaryBackend) |> Nx.as_type(:f32) |> Nx.reshape({})

  defp round_even(x) do
    rounded = round(x)
    if abs(x - trunc(x)) == 0.5 and rem(rounded, 2) != 0, do: rounded - sign(x), else: rounded
  end

  defp sign(x) when x < 0, do: -1
  defp sign(_x), do: 1

  # A scalar, or a 1-D tensor laid along one axis of a rank-r result, as f32.
  defp per_axis(t, rank, axis) do
    t = Nx.as_type(t, :f32)

    if Nx.size(t) == 1,
      do: Nx.reshape(t, {}),
      else:
        Nx.reshape(
          t,
          1 |> List.duplicate(rank) |> List.replace_at(axis, :auto) |> List.to_tuple()
        )
  end

  # Unpacks bits-wide unsigned values from the last axis of a u8 tensor.
  defp unpack(packed, 8), do: packed

  # Signed packed values (int4 tensors load as s8 bytes) are sign extended.
  defp unpack(packed, bits) do
    per = div(8, bits)
    shifts = Nx.multiply(Nx.iota({per}, type: :s16), bits)
    {last, shape} = packed |> Nx.shape() |> Tuple.to_list() |> List.pop_at(-1)

    values =
      packed
      |> Nx.as_type(:s16)
      |> Nx.bitwise_and(255)
      |> Nx.new_axis(-1)
      |> Nx.right_shift(shifts)
      |> Nx.bitwise_and(2 ** bits - 1)
      |> Nx.reshape(List.to_tuple(shape ++ [last * per]))

    if signed?(packed),
      do: Nx.subtract(values, Nx.multiply(Nx.greater_equal(values, 2 ** (bits - 1)), 2 ** bits)),
      else: values
  end

  defp signed?(t), do: Nx.type(t) == {:s, 8}

  # Zero points per block: none (0 for signed data, else the middle of the range),
  # packed bits, or floats.
  defp block_zero_points(zero, shape, bits, type, signed \\ false)
  defp block_zero_points(nil, _shape, _bits, type, true), do: Nx.tensor(0, type: type)

  defp block_zero_points(nil, _shape, bits, type, false),
    do: Nx.tensor(2 ** (bits - 1), type: type)

  defp block_zero_points(zero, shape, bits, type, _signed) do
    if Nx.Type.float?(Nx.type(zero)) do
      zero |> Nx.reshape(shape) |> Nx.as_type(type)
    else
      rows = put_elem(shape, tuple_size(shape) - 1, :auto)
      blocks = elem(shape, tuple_size(shape) - 1)

      zero
      |> Nx.reshape(rows)
      |> unpack(bits)
      |> Nx.slice_along_axis(0, blocks, axis: tuple_size(shape) - 1)
      |> Nx.as_type(type)
    end
  end

  # Scales and zero points are per tensor, per axis (1-D) or per block along the axis.
  defp expand_quant_param(param, x, node) do
    rank = Nx.rank(x)
    axis = normalize_axis(attr(node, "axis", 1), rank)
    block_size = attr(node, "block_size", 0)

    cond do
      Nx.size(param) == 1 ->
        Nx.reshape(param, {})

      block_size > 0 ->
        dim = Nx.axis_size(x, axis)

        param
        |> Nx.new_axis(axis + 1)
        |> Nx.broadcast(param |> Nx.shape() |> Tuple.insert_at(axis + 1, block_size))
        |> Nx.reshape(
          param
          |> Nx.shape()
          |> put_elem(axis, Nx.axis_size(param, axis) * block_size)
        )
        |> Nx.slice_along_axis(0, dim, axis: axis)

      true ->
        Nx.reshape(
          param,
          1 |> List.duplicate(rank) |> List.replace_at(axis, :auto) |> List.to_tuple()
        )
    end
  end

  defp subtract_zero(t, nil, _axis), do: t

  defp subtract_zero(t, zero, axis) do
    zero = Nx.as_type(zero, :f32)

    if Nx.size(zero) == 1,
      do: Nx.subtract(t, Nx.reshape(zero, {})),
      else:
        Nx.subtract(
          t,
          Nx.reshape(
            zero,
            1 |> List.duplicate(Nx.rank(t)) |> List.replace_at(axis, :auto) |> List.to_tuple()
          )
        )
  end

  defp round_half_even(t) do
    rounded = Nx.round(t)
    tie = Nx.equal(Nx.abs(Nx.subtract(t, Nx.floor(t))), 0.5)
    Nx.select(tie, Nx.multiply(Nx.round(Nx.divide(t, 2)), 2), rounded)
  end

  defp saturate(t, type) do
    {low, high} =
      case type do
        {:u, 8} -> {0, 255}
        {:s, 8} -> {-128, 127}
        {:u, 16} -> {0, 65_535}
        {:s, 16} -> {-32_768, 32_767}
        {:s, 32} -> {-2_147_483_648, 2_147_483_647}
      end

    t |> Nx.clip(low, high) |> Nx.as_type(type)
  end

  defp normalize_axis(axis, rank) when axis < 0, do: axis + rank
  defp normalize_axis(axis, _rank), do: axis
end
