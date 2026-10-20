defmodule Transmuter.Ops.Signal do
  @moduledoc """
  Recurrent layers and Fourier transforms. Short recurrences unroll over the sequence, which has a static length. Long traced recurrences run as an Nx while loop, because an unrolled graph gets too deep to compile. Fourier transforms multiply by static DFT matrices, so they need no complex numbers on the backend.
  """

  import Transmuter.Ops, only: [attr: 2, attr: 3, ints!: 1, static_tensor: 2]

  alias Nx.Defn.Expr

  @op_types ~w(LSTM GRU RNN STFT DFT HannWindow HammingWindow BlackmanWindow MelWeightMatrix)

  @doc false
  def op_types, do: @op_types

  @doc false
  def apply(node, inputs), do: node |> op(inputs) |> List.wrap()

  # ONNX LSTM, gates in i, o, f, c order. Outputs Y {S, D, B, H}, Y_h and Y_c {D, B, H}.
  # The peepholes add P * c to the i, o and f gates, zeros without the P input. With
  # input_forget, the forget gate is 1 - i.
  defp op(node = %{op_type: "LSTM"}, [x, w, r | rest]) do
    [bias, lengths, h0, c0, peepholes] = Enum.take(rest ++ List.duplicate(nil, 5), 5)
    hidden = attr(node, "hidden_size")
    coupled? = attr(node, "input_forget", 0) == 1

    peepholes =
      peepholes || Nx.broadcast(Nx.tensor(0, type: Nx.type(x)), {Nx.axis_size(w, 0), 3 * hidden})

    step = fn [f, g, h] ->
      fn x_gates, {r, rb, p}, {h_prev, c_prev} ->
        gates = x_gates |> Nx.add(Nx.dot(h_prev, [1], r, [1])) |> Nx.add(rb)
        [i, o, fg, c] = chunks(gates, 4)
        [pi, po, pf] = chunks(p, 3)
        i = f.(Nx.add(i, Nx.multiply(pi, c_prev)))
        fg = if coupled?, do: Nx.subtract(1, i), else: f.(Nx.add(fg, Nx.multiply(pf, c_prev)))
        c_new = Nx.add(Nx.multiply(fg, c_prev), Nx.multiply(i, g.(c)))
        o = f.(Nx.add(o, Nx.multiply(po, c_new)))
        {Nx.multiply(o, h.(c_new)), c_new}
      end
    end

    recur(
      node,
      x,
      {w, r, bias, peepholes},
      lengths,
      [h0, c0],
      {["Sigmoid", "Tanh", "Tanh"], step}
    )
  end

  # ONNX GRU, gates in z, r, h order.
  defp op(node = %{op_type: "GRU"}, [x, w, r | rest]) do
    [bias, lengths, h0] = Enum.take(rest ++ List.duplicate(nil, 3), 3)
    before_reset? = attr(node, "linear_before_reset", 0) == 1

    step = fn [f, g] ->
      fn x_gates, {r, rb, _}, {h_prev} ->
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
    end

    recur(node, x, {w, r, bias, nil}, lengths, [h0], {["Sigmoid", "Tanh"], step})
  end

  # ONNX RNN: one gate, h = f(x W + h R + Wb + Rb).
  defp op(node = %{op_type: "RNN"}, [x, w, r | rest]) do
    [bias, lengths, h0] = Enum.take(rest ++ List.duplicate(nil, 3), 3)

    step = fn [f] ->
      fn x_gates, {r, rb, _}, {h_prev} ->
        {f.(x_gates |> Nx.add(Nx.dot(h_prev, [1], r, [1])) |> Nx.add(rb))}
      end
    end

    recur(node, x, {w, r, bias, nil}, lengths, [h0], {["Tanh"], step})
  end

  defp op(node = %{op_type: window}, [size])
       when window in ["HannWindow", "HammingWindow", "BlackmanWindow"] do
    [n] = ints!(size)
    denominator = if attr(node, "periodic", 1) == 1, do: n, else: n - 1

    values =
      for i <- 0..(n - 1) do
        angle = 2 * :math.pi() * i / denominator

        case window do
          "HannWindow" -> :math.pow(:math.sin(:math.pi() * i / denominator), 2)
          "HammingWindow" -> 25 / 46 - (1 - 25 / 46) * :math.cos(angle)
          "BlackmanWindow" -> 0.42 - 0.5 * :math.cos(angle) + 0.08 * :math.cos(2 * angle)
        end
      end

    values
    |> static_tensor(:f64)
    |> Nx.as_type(Transmuter.Ops.dtype(attr(node, "output_datatype", 1)))
  end

  # Triangular mel filters, {dft_length / 2 + 1, num_mel_bins}, as the ONNX reference builds them.
  defp op(node = %{op_type: "MelWeightMatrix"}, [bins, dft_length, sample_rate, low, high]) do
    [bins, dft_length, sample_rate] =
      Enum.map([bins, dft_length, sample_rate], &(&1 |> ints!() |> hd()))

    [low, high] = Enum.map([low, high], &(&1 |> ints!() |> hd()))
    to_mel = &(2595 * :math.log10(1 + &1 / 700))
    {low_mel, high_mel} = {to_mel.(low), to_mel.(high)}
    step = (high_mel - low_mel) / (bins + 2)

    points =
      for i <- 0..(bins + 1) do
        hz = 700 * (:math.pow(10, (i * step + low_mel) / 2595) - 1)
        trunc(Float.floor((dft_length + 1) * hz / sample_rate))
      end

    columns =
      for i <- 0..(bins - 1) do
        [left, center, right] = Enum.slice(points, i, 3)

        rising =
          if center == left,
            do: %{center => 1.0},
            else: Map.new(left..center, &{&1, (&1 - left) / (center - left)})

        falling =
          if right > center,
            do: Map.new(center..(right - 1)//1, &{&1, (right - &1) / (right - center)}),
            else: %{}

        Map.merge(rising, falling)
      end

    rows = for j <- 0..div(dft_length, 2), do: Enum.map(columns, &Map.get(&1, j, 0.0))

    rows
    |> static_tensor(:f64)
    |> Nx.as_type(Transmuter.Ops.dtype(attr(node, "output_datatype", 1)))
  end

  # STFT of signal {B, L, 1 or 2}. Output {B, frames, bins, 2}.
  defp op(node = %{op_type: "STFT"}, [signal, frame_step | rest]) do
    [window, frame_length] = Enum.take(rest ++ [nil, nil], 2)
    [step] = ints!(frame_step)
    size = if window, do: Nx.axis_size(window, 0), else: hd(ints!(frame_length))
    # The spec signal is {B, L, 1 or 2}. ONNX Runtime also takes a real {B, L} signal.
    signal = if Nx.rank(signal) == 2, do: Nx.new_axis(signal, 2), else: signal
    stft(signal, window, size, step, attr(node, "onesided", 1) == 1)
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

  @doc """
  The short-time Fourier transform of `signal` {B, L, 1 or 2}, in frames of `size` samples every `step` samples. `window` is a tensor {size} or `nil`. Returns {B, frames, bins, 2}, with the real and the imaginary parts on the last axis. `bins` is `size / 2 + 1` when `onesided?` is `true`, else `size`.
  """
  def stft(signal, window, size, step, onesided?) do
    {_b, l, _} = Nx.shape(signal)
    frames = div(l - size, step) + 1
    framed = frame(signal, size, step, frames)
    framed = if window, do: Nx.multiply(framed, Nx.reshape(window, {1, 1, size, 1})), else: framed
    bins = if onesided?, do: div(size, 2) + 1, else: size
    dft(framed, 2, size, bins, false)
  end

  # Runs one or two directions and stacks them. step.(activations) gives the step of a
  # direction, step.(x_gates, {r, r_bias, extra}, state) -> state, where extra is the
  # direction's row of the extra weights (the LSTM peepholes). With layout 1, the batch
  # axis comes first in x, the initial states and the outputs.
  defp recur(node, x, {w, r, bias, extra}, lengths, initial, {defaults, step}) do
    batch_first? = attr(node, "layout", 0) == 1
    x = if batch_first?, do: Nx.transpose(x, axes: [1, 0, 2]), else: x
    initial = if batch_first?, do: Enum.map(initial, &swap_leading/1), else: initial
    {_seq, batch, _} = Nx.shape(x)
    hidden = attr(node, "hidden_size")
    gates = div(Nx.axis_size(w, 1), hidden)
    directions = Nx.axis_size(w, 0)
    reverse = attr(node, "direction", "forward") == "reverse"
    activations = activations(node, defaults, directions)

    results =
      for d <- 0..(directions - 1) do
        wd = w[d]
        rd = r[d]
        {wb, rb} = split_bias(bias, d, gates * hidden, Nx.type(x))
        extra_d = if extra, do: extra[d], else: Nx.tensor(0, type: Nx.type(x))
        x_gates = x |> Nx.dot([2], wd, [1]) |> Nx.add(wb)

        state =
          initial |> Enum.map(&initial_state(&1, d, batch, hidden, Nx.type(x))) |> List.to_tuple()

        backward? = reverse or d == 1
        step_d = step.(Enum.at(activations, d))
        {y, state} = steps(x_gates, {rd, rb, extra_d}, state, lengths, backward?, step_d)
        {y, Tuple.to_list(state)}
      end

    y = results |> Enum.map(&elem(&1, 0)) |> Nx.stack(axis: 1)
    finals = results |> Enum.map(&elem(&1, 1)) |> Enum.zip_with(&Nx.stack/1)

    if batch_first?,
      do: [Nx.transpose(y, axes: [2, 0, 1, 3]) | Enum.map(finals, &swap_leading/1)],
      else: [y | finals]
  end

  defp swap_leading(nil), do: nil
  defp swap_leading(t), do: Nx.transpose(t, axes: [1, 0, 2])

  # Sequences shorter than sequence_lens keep their state and output zeros.
  # A traced sequence longer than this runs as an Nx while loop. An unrolled loop makes
  # a graph that grows with the sequence, which compiles slowly and, on EMLX, can
  # overflow the stack of the compiler.
  @max_unrolled_steps 16

  defp steps(x_gates, weights, state, lengths, backward?, step) do
    seq = Nx.axis_size(x_gates, 0)

    if traced?(x_gates) and seq > @max_unrolled_steps,
      do: while_steps(x_gates, weights, state, lengths, backward?, step),
      else: unrolled_steps(x_gates, weights, state, lengths, backward?, step)
  end

  defp unrolled_steps(x_gates, weights, state, lengths, backward?, step) do
    seq = Nx.axis_size(x_gates, 0)
    order = if backward?, do: (seq - 1)..0//-1, else: 0..(seq - 1)

    {outputs, state} =
      Enum.map_reduce(order, state, fn t, state ->
        next = step.(x_gates[t], weights, state)
        next = keep_finished(next, state, lengths, t)
        {{t, mask_output(elem(next, 0), lengths, t)}, next}
      end)

    y = outputs |> Enum.sort_by(&elem(&1, 0)) |> Enum.map(&elem(&1, 1)) |> Nx.stack()
    {y, state}
  end

  # The loop state carries every tensor that the steps read, because a while body
  # cannot read tensors from outside the loop. Each iteration unrolls a block of
  # steps, so the loop runs fewer times and the graph stays small.
  defp while_steps(x_gates, weights, state, lengths, backward?, step) do
    seq = Nx.axis_size(x_gates, 0)
    h_shape = state |> elem(0) |> Nx.shape()
    y = Nx.broadcast(Nx.tensor(0, type: Nx.type(x_gates)), Tuple.insert_at(h_shape, 0, seq))

    initial = %{state: state, y: y, x: x_gates, weights: weights}
    initial = if lengths, do: Map.put(initial, :lengths, lengths), else: initial

    body = fn :body, {i, acc = %{state: state}} ->
      t = if backward?, do: Nx.subtract(seq - 1, i), else: i
      lengths = acc[:lengths]
      next = step.(Nx.take(acc.x, t), acc.weights, state)
      next = keep_finished(next, state, lengths, t)
      h = mask_output(elem(next, 0), lengths, t)
      %{acc | state: next, y: Nx.put_slice(acc.y, [t, 0, 0], Nx.new_axis(h, 0))}
    end

    range = {:while, 0..(seq - 1)}
    opts = [unroll: @max_unrolled_steps]
    result = Expr.defn_while(__ENV__.file, __ENV__.line, initial, range, body, opts)
    {result.y, result.state}
  end

  defp traced?(%Nx.Tensor{data: %Nx.Defn.Expr{}}), do: true
  defp traced?(_tensor), do: false

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

  # The activations of each direction. The activations that have an alpha or a beta take
  # the next value of activation_alpha or activation_beta, in order, as in ONNX Runtime.
  # clip bounds the input of every activation.
  defp activations(node, defaults, directions) do
    names = attr(node, "activations", List.flatten(List.duplicate(defaults, directions)))
    clip = attr(node, "clip")

    {funs, _rest} =
      Enum.map_reduce(
        names,
        {attr(node, "activation_alpha", []), attr(node, "activation_beta", [])},
        &activation/2
      )

    funs = if clip, do: Enum.map(funs, fn f -> &f.(Nx.clip(&1, -clip, clip)) end), else: funs
    Enum.chunk_every(funs, length(defaults))
  end

  @activations %{
    "Relu" => {nil, nil},
    "Tanh" => {nil, nil},
    "Sigmoid" => {nil, nil},
    "Softsign" => {nil, nil},
    "Softplus" => {nil, nil},
    "Affine" => {1.0, 0.0},
    "LeakyRelu" => {0.01, nil},
    "ThresholdedRelu" => {1.0, nil},
    "ScaledTanh" => {1.0, 1.0},
    "HardSigmoid" => {0.2, 0.5},
    "Elu" => {1.0, nil}
  }

  defp activation(name, {alphas, betas}) do
    {alpha_default, beta_default} =
      Map.get_lazy(@activations, name, fn ->
        raise ArgumentError, "recurrent activation #{inspect(name)} is not supported"
      end)

    {alpha, alphas} = take_param(alpha_default, alphas)
    {beta, betas} = take_param(beta_default, betas)
    {activation(name, alpha, beta), {alphas, betas}}
  end

  defp take_param(nil, values), do: {nil, values}
  defp take_param(default, []), do: {default, []}
  defp take_param(_default, [value | rest]), do: {value, rest}

  defp activation("Relu", _, _), do: &Nx.max(&1, 0)
  defp activation("Tanh", _, _), do: &Nx.tanh/1
  defp activation("Sigmoid", _, _), do: &Nx.sigmoid/1
  defp activation("Softsign", _, _), do: &Nx.divide(&1, Nx.add(1, Nx.abs(&1)))

  defp activation("Softplus", _, _),
    do: &Nx.add(Nx.max(&1, 0), Nx.log1p(Nx.exp(Nx.negate(Nx.abs(&1)))))

  defp activation("Affine", alpha, beta), do: &Nx.add(Nx.multiply(&1, alpha), beta)

  defp activation("LeakyRelu", alpha, _),
    do: &Nx.select(Nx.less(&1, 0), Nx.multiply(&1, alpha), &1)

  defp activation("ThresholdedRelu", alpha, _),
    do: &Nx.select(Nx.greater(&1, alpha), &1, Nx.tensor(0, type: Nx.type(&1)))

  defp activation("ScaledTanh", alpha, beta),
    do: &Nx.multiply(alpha, Nx.tanh(Nx.multiply(&1, beta)))

  defp activation("HardSigmoid", alpha, beta),
    do: &(&1 |> Nx.multiply(alpha) |> Nx.add(beta) |> Nx.clip(0, 1))

  defp activation("Elu", alpha, _),
    do: &Nx.select(Nx.less(&1, 0), Nx.multiply(alpha, Nx.subtract(Nx.exp(&1), 1)), &1)

  # Real and imaginary parts {..., n} along axis times n x bins DFT matrices.
  # Frame f holds samples f * step .. f * step + size - 1 of a {B, L, C} signal. The
  # signal splits into chunks of gcd(size, step) samples, and a frame is size / gcd
  # consecutive chunks, so strided slices of the chunks build all frames.
  defp frame(signal, size, step, frames) do
    {b, _l, c} = Nx.shape(signal)
    chunk = Integer.gcd(size, step)
    stride = div(step, chunk)
    count = (frames - 1) * stride + div(size, chunk)

    chunks =
      signal
      |> Nx.slice_along_axis(0, count * chunk, axis: 1)
      |> Nx.reshape({b, count, chunk, c})

    for j <- 0..(div(size, chunk) - 1)//1 do
      Nx.slice(chunks, [0, j, 0, 0], [b, (frames - 1) * stride + 1, chunk, c],
        strides: [1, stride, 1, 1]
      )
    end
    |> Nx.concatenate(axis: 2)
  end

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
