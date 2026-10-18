defmodule Transmuter.Ops.Signal do
  @moduledoc false

  # Recurrent layers and Fourier transforms. Recurrences unroll over the sequence,
  # which has a static length. Fourier transforms multiply by static DFT matrices,
  # so they need no complex numbers on the backend.

  import Transmuter.Ops, only: [attr: 2, attr: 3, ints!: 1, static_tensor: 2]

  @op_types ~w(LSTM GRU STFT DFT)

  @doc false
  def op_types, do: @op_types

  @doc false
  def apply(node, inputs), do: node |> op(inputs) |> List.wrap()

  # ONNX LSTM, gates in i, o, f, c order. Outputs Y {S, D, B, H}, Y_h and Y_c {D, B, H}.
  defp op(node = %{op_type: "LSTM"}, [x, w, r | rest]) do
    [bias, lengths, h0, c0, peepholes] = Enum.take(rest ++ List.duplicate(nil, 5), 5)
    nil = peepholes
    0 = attr(node, "layout", 0)
    [f, g, h] = activations(node, ["Sigmoid", "Tanh", "Tanh"])

    step = fn x_gates, r, rb, {h_prev, c_prev} ->
      gates = x_gates |> Nx.add(Nx.dot(h_prev, [1], r, [1])) |> Nx.add(rb)
      [i, o, fg, c] = chunks(gates, 4)
      c_new = Nx.add(Nx.multiply(f.(fg), c_prev), Nx.multiply(f.(i), g.(c)))
      {Nx.multiply(f.(o), h.(c_new)), c_new}
    end

    {y, finals} = recur(node, x, {w, r, bias}, lengths, [h0, c0], step)
    [y | finals]
  end

  # ONNX GRU, gates in z, r, h order.
  defp op(node = %{op_type: "GRU"}, [x, w, r | rest]) do
    [bias, lengths, h0] = Enum.take(rest ++ List.duplicate(nil, 3), 3)
    0 = attr(node, "layout", 0)
    [f, g] = activations(node, ["Sigmoid", "Tanh"])
    before_reset? = attr(node, "linear_before_reset", 0) == 1

    step = fn x_gates, r, rb, {h_prev} ->
      [xz, xr, xh] = chunks(x_gates, 3)
      [rz, rr, rh] = chunks(r, 3, 0)
      [bz, br, bh] = chunks(rb, 3)
      z = f.(xz |> Nx.add(Nx.dot(h_prev, [1], rz, [1])) |> Nx.add(bz))
      reset = f.(xr |> Nx.add(Nx.dot(h_prev, [1], rr, [1])) |> Nx.add(br))

      candidate =
        if before_reset?,
          do: g.(Nx.add(xh, Nx.multiply(reset, Nx.add(Nx.dot(h_prev, [1], rh, [1]), bh)))),
          else: g.(xh |> Nx.add(Nx.dot(Nx.multiply(reset, h_prev), [1], rh, [1])) |> Nx.add(bh))

      {Nx.add(Nx.multiply(Nx.subtract(1, z), candidate), Nx.multiply(z, h_prev))}
    end

    {y, finals} = recur(node, x, {w, r, bias}, lengths, [h0], step)
    [y | finals]
  end

  # STFT of signal {B, L, 1 or 2}. Output {B, frames, bins, 2}.
  defp op(node = %{op_type: "STFT"}, [signal, frame_step | rest]) do
    [window, frame_length] = Enum.take(rest ++ [nil, nil], 2)
    [step] = ints!(frame_step)
    size = if window, do: Nx.axis_size(window, 0), else: hd(ints!(frame_length))
    # The spec signal is {B, L, 1 or 2}. ONNX Runtime also takes a real {B, L} signal.
    signal = if Nx.rank(signal) == 2, do: Nx.new_axis(signal, 2), else: signal
    {_b, l, _} = Nx.shape(signal)
    frames = div(l - size, step) + 1
    index = static_tensor(for(f <- 0..(frames - 1), n <- 0..(size - 1), do: f * step + n), :s64)

    framed =
      signal
      |> Nx.take(index, axis: 1)
      |> Nx.reshape({Nx.axis_size(signal, 0), frames, size, Nx.axis_size(signal, 2)})

    framed = if window, do: Nx.multiply(framed, Nx.reshape(window, {1, 1, size, 1})), else: framed
    bins = if attr(node, "onesided", 1) == 1, do: div(size, 2) + 1, else: size
    dft(framed, 2, size, bins, false)
  end

  # DFT along one axis of {..., n, 1 or 2}. Output keeps the layout with 2 components.
  defp op(node = %{op_type: "DFT"}, [x | rest]) do
    [length, axis] = Enum.take(rest ++ [nil, nil], 2)
    rank = Nx.rank(x)
    axis = if axis, do: hd(ints!(axis)), else: attr(node, "axis", -2)
    axis = if axis < 0, do: axis + rank, else: axis
    n = if length, do: hd(ints!(length)), else: Nx.axis_size(x, axis)
    x = fit_length(x, axis, n)
    inverse? = attr(node, "inverse", 0) == 1
    bins = if attr(node, "onesided", 0) == 1, do: div(n, 2) + 1, else: n
    dft(x, axis, n, bins, inverse?)
  end

  # Runs one or two directions and stacks them. step.(x_gates, r, r_bias, state) -> state.
  defp recur(node, x, {w, r, bias}, lengths, initial, step) do
    {seq, batch, _} = Nx.shape(x)
    hidden = attr(node, "hidden_size")
    gates = div(Nx.axis_size(w, 1), hidden)
    directions = Nx.axis_size(w, 0)
    reverse = attr(node, "direction", "forward") == "reverse"

    results =
      for d <- 0..(directions - 1) do
        wd = w[d]
        rd = r[d]
        {wb, rb} = split_bias(bias, d, gates * hidden, Nx.type(x))
        x_gates = x |> Nx.dot([2], wd, [1]) |> Nx.add(wb)

        state =
          initial |> Enum.map(&initial_state(&1, d, batch, hidden, Nx.type(x))) |> List.to_tuple()

        order = if reverse or d == 1, do: (seq - 1)..0//-1, else: 0..(seq - 1)

        {outputs, state} =
          Enum.map_reduce(order, state, fn t, state ->
            next = step.(x_gates[t], rd, rb, state)
            next = keep_finished(next, state, lengths, t)
            {{t, mask_output(elem(next, 0), lengths, t)}, next}
          end)

        y = outputs |> Enum.sort_by(&elem(&1, 0)) |> Enum.map(&elem(&1, 1)) |> Nx.stack()
        {y, Tuple.to_list(state)}
      end

    y = results |> Enum.map(&elem(&1, 0)) |> Nx.stack(axis: 1)
    finals = results |> Enum.map(&elem(&1, 1)) |> Enum.zip_with(&Nx.stack/1)
    {y, finals}
  end

  # Sequences shorter than sequence_lens keep their state and output zeros.
  defp keep_finished(next, _state, nil, _t), do: next

  defp keep_finished(next, state, lengths, t) do
    active = lengths |> Nx.greater(t) |> Nx.new_axis(1)

    next
    |> Tuple.to_list()
    |> Enum.zip_with(Tuple.to_list(state), &Nx.select(Nx.broadcast(active, Nx.shape(&1)), &1, &2))
    |> List.to_tuple()
  end

  defp mask_output(h, nil, _t), do: h

  defp mask_output(h, lengths, t),
    do: Nx.multiply(h, lengths |> Nx.greater(t) |> Nx.new_axis(1) |> Nx.as_type(Nx.type(h)))

  defp split_bias(nil, _d, size, type),
    do: {Nx.tensor(0, type: type), Nx.broadcast(Nx.tensor(0, type: type), {size})}

  defp split_bias(bias, d, size, _type),
    do:
      {Nx.slice_along_axis(bias[d], 0, size, axis: 0),
       Nx.slice_along_axis(bias[d], size, size, axis: 0)}

  defp initial_state(nil, _d, batch, hidden, type),
    do: Nx.broadcast(Nx.tensor(0, type: type), {batch, hidden})

  defp initial_state(state, d, _batch, _hidden, _type), do: state[d]

  defp chunks(t, n, axis \\ -1) do
    size = div(Nx.axis_size(t, axis), n)
    for i <- 0..(n - 1), do: Nx.slice_along_axis(t, i * size, size, axis: axis)
  end

  defp activations(node, defaults) do
    names = node |> attr("activations", defaults) |> Enum.take(length(defaults))

    Enum.map(names, fn
      "Sigmoid" -> &Nx.sigmoid/1
      "Tanh" -> &Nx.tanh/1
      "Relu" -> &Nx.max(&1, 0)
    end)
  end

  # Real and imaginary parts {..., n} along axis times n x bins DFT matrices.
  defp dft(x, axis, n, bins, inverse?) do
    rank = Nx.rank(x)
    {re, im} = components(x)
    perm = (Enum.to_list(0..(rank - 2)) -- [axis]) ++ [axis]
    inverse_perm = perm |> Enum.with_index() |> Enum.sort() |> Enum.map(&elem(&1, 1))
    re = Nx.transpose(re, axes: perm)
    im = im && Nx.transpose(im, axes: perm)
    sign = if inverse?, do: 1, else: -1

    angles = for k <- 0..(n - 1), j <- 0..(bins - 1), do: 2 * :math.pi() * k * j / n

    cos =
      angles
      |> Enum.map(&:math.cos/1)
      |> static_tensor(:f32)
      |> Nx.reshape({n, bins})
      |> Nx.as_type(Nx.type(re))

    sin =
      angles
      |> Enum.map(&(sign * :math.sin(&1)))
      |> static_tensor(:f32)
      |> Nx.reshape({n, bins})
      |> Nx.as_type(Nx.type(re))

    mm = fn a, m -> Nx.dot(a, [rank - 2], m, [0]) end

    {out_re, out_im} =
      if im,
        do: {Nx.subtract(mm.(re, cos), mm.(im, sin)), Nx.add(mm.(re, sin), mm.(im, cos))},
        else: {mm.(re, cos), mm.(re, sin)}

    {out_re, out_im} =
      if inverse?, do: {Nx.divide(out_re, n), Nx.divide(out_im, n)}, else: {out_re, out_im}

    Nx.stack([Nx.transpose(out_re, axes: inverse_perm), Nx.transpose(out_im, axes: inverse_perm)],
      axis: -1
    )
  end

  defp components(x) do
    last = Nx.rank(x) - 1
    re = x |> Nx.slice_along_axis(0, 1, axis: last) |> Nx.squeeze(axes: [last])

    if Nx.axis_size(x, last) == 2,
      do: {re, x |> Nx.slice_along_axis(1, 1, axis: last) |> Nx.squeeze(axes: [last])},
      else: {re, nil}
  end

  defp fit_length(x, axis, n) do
    size = Nx.axis_size(x, axis)

    cond do
      size == n ->
        x

      size > n ->
        Nx.slice_along_axis(x, 0, n, axis: axis)

      true ->
        Nx.pad(
          x,
          Nx.tensor(0, type: Nx.type(x)),
          Enum.map(0..(Nx.rank(x) - 1), &if(&1 == axis, do: {0, n - size, 0}, else: {0, 0, 0}))
        )
    end
  end
end
