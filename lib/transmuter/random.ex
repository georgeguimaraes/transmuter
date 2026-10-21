defmodule Transmuter.Random do
  @moduledoc """
  The functions of `Nx.Random` that Transmuter uses, with the same values.

  `Nx.Random` runs the 20 rounds of threefry2x32 in `while` loops. EMLX evaluates each iteration of a `while` loop separately, so one draw takes about 3 ms on the GPU of a Mac, also in a compiled function. Here the rounds are unrolled, so a draw is a part of the compiled graph. The functions work in `Nx.Defn` and eagerly, and with vectorized keys.

  The functions return only the values, not the next key of `Nx.Random`.
  """

  @rotations [[13, 15, 26, 6], [17, 29, 16, 24]]

  @doc """
  Folds `data` into `key`, as `Nx.Random.fold_in/2`.
  """
  def fold_in(key, data) do
    data = Nx.as_type(data, :u64)

    {x1, x2} =
      [Nx.right_shift(data, 32), Nx.bitwise_and(data, 0xFFFFFFFF)]
      |> Nx.stack()
      |> Nx.reshape({2, :auto})
      |> Nx.as_type(:u32)
      |> threefry2x32_20(key)

    shape = Nx.shape(data)
    Nx.stack([x1, x2], axis: -1) |> Nx.reshape(Tuple.insert_at(shape, tuple_size(shape), 2))
  end

  @doc """
  Uniform values in `[low, high)`, as `Nx.Random.uniform/4` with the options `:shape` and `:type`.
  """
  def uniform(key, low, high, opts), do: key |> second_key() |> uniform_split(low, high, opts)

  @doc """
  Normal values, as `Nx.Random.normal/4` with the options `:shape` and `:type`.
  """
  def normal(key, mean, scale, opts) do
    type = {_kind, bits} = Nx.Type.normalize!(Keyword.fetch!(opts, :type))

    # The float after -1 towards 0, as in Nx.Random.
    <<minus_one::size(^bits)-big>> = <<-1::float-size(bits)-big>>
    <<low::float-size(^bits)-big>> = <<minus_one - 1::integer-size(bits)-big>>

    key
    |> second_key()
    |> uniform_split(low, 1, shape: opts[:shape], type: type)
    |> Nx.erf_inv()
    |> Nx.multiply(Nx.sqrt(Nx.tensor(2, type: type)))
    |> Nx.multiply(Nx.tensor(scale, type: type))
    |> Nx.add(Nx.tensor(mean, type: type))
  end

  @doc """
  Gumbel values, as `Nx.Random.gumbel/2` with the options `:shape` and `:type`.
  """
  def gumbel(key, opts) do
    type = Keyword.fetch!(opts, :type)
    low = Nx.Constants.smallest_positive_normal(type, backend: Nx.BinaryBackend)

    key
    |> second_key()
    |> uniform_split(Nx.to_number(low), 1, opts)
    |> Nx.log()
    |> Nx.negate()
    |> Nx.log()
    |> Nx.negate()
  end

  # Nx.Random splits the key in two and draws from the second key.
  defp second_key(key), do: threefry2x32(key, {2, 2})[1]

  defp uniform_split(key, low, high, opts) do
    shape = Keyword.fetch!(opts, :shape)
    type = {_kind, bits} = Nx.Type.normalize!(Keyword.fetch!(opts, :type))
    low = Nx.tensor(low, type: type)
    high = Nx.tensor(high, type: type)
    one = Nx.tensor(1.0, type: type)

    key
    |> random_bits(shape, bits)
    |> Nx.right_shift(bits - mantissa(type))
    |> Nx.bitwise_or(Nx.bitcast(one, {:u, bits}))
    |> Nx.bitcast(type)
    |> Nx.subtract(one)
    |> Nx.multiply(Nx.subtract(high, low))
    |> Nx.add(low)
    |> Nx.max(low)
  end

  defp random_bits(key, shape, 64) do
    bits = key |> threefry2x32({2, Nx.size(shape)}) |> Nx.as_type(:u64)
    bits[0] |> Nx.left_shift(32) |> Nx.bitwise_or(bits[1]) |> Nx.reshape(shape)
  end

  defp random_bits(key, shape, 32), do: threefry2x32(key, shape)
  defp random_bits(key, shape, bits), do: key |> threefry2x32(shape) |> Nx.as_type({:u, bits})

  defp mantissa({:f, 8}), do: 2
  defp mantissa({:f8_e4m3fn, 8}), do: 3
  defp mantissa({:bf, 16}), do: 7
  defp mantissa({:f, 16}), do: 10
  defp mantissa({:f, 32}), do: 23
  defp mantissa({:f, 64}), do: 52

  # The counters 0..size - 1, and a 0 after them for an odd size, as two halves.
  defp threefry2x32(key, shape) do
    size = Nx.size(shape)
    half = div(size + 1, 2)
    counters = Nx.iota({size}, type: :u32)
    counters = if rem(size, 2) == 1, do: Nx.pad(counters, 0, [{0, 1, 0}]), else: counters
    {x1, x2} = counters |> Nx.reshape({2, half}) |> threefry2x32_20(key)
    Nx.concatenate([x1, x2]) |> Nx.slice([0], [size]) |> Nx.reshape(shape)
  end

  # Five steps of four rounds each. A step adds two of the three key words and its number.
  defp threefry2x32_20(xs, key) do
    {k1, k2} = {key[0], key[1]}
    keys = [k2, k1 |> Nx.bitwise_xor(k2) |> Nx.bitwise_xor(0x1BD11BDA), k1]

    Enum.reduce(1..5, {Nx.add(xs[0], k1), Nx.add(xs[1], k2)}, fn step, xs ->
      {x1, x2} =
        Enum.reduce(Enum.at(@rotations, rem(step - 1, 2)), xs, fn rotation, {x1, x2} ->
          y1 = Nx.add(x1, x2)
          {y1, x2 |> rotate_left(rotation) |> Nx.bitwise_xor(y1)}
        end)

      {Nx.add(x1, Enum.at(keys, rem(step - 1, 3))),
       x2 |> Nx.add(Enum.at(keys, rem(step, 3))) |> Nx.add(step)}
    end)
  end

  defp rotate_left(x, rotation),
    do: x |> Nx.left_shift(rotation) |> Nx.bitwise_or(Nx.right_shift(x, 32 - rotation))
end
