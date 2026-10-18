defmodule Transmuter.Ops.NN do
  @moduledoc false

  # Neural network layers: convolutions, pooling, normalizations, Resize, GridSample,
  # LRN and Dropout.

  import Transmuter.Ops, only: [attr: 2, attr: 3, ints!: 1, static_tensor: 2]

  @op_types ~w(ConvInteger Conv InstanceNormalization BatchNormalization LayerNormalization
               GlobalAveragePool GlobalMaxPool ConvTranspose LpNormalization GroupNormalization
               Resize GridSample Dropout LRN GlobalLpPool MeanVarianceNormalization MaxPool
               AveragePool LpPool)

  @doc false
  def op_types, do: @op_types

  @doc false
  def apply(node, inputs), do: node |> op(inputs) |> List.wrap()

  # Integer convolution, accumulated in f32 like MatMulInteger.
  defp op(node = %{op_type: "ConvInteger"}, [x, w | zero_points]) do
    [x_zero, w_zero] = Enum.take(zero_points ++ [nil, nil], 2)
    x = Nx.as_type(x, :f32)
    x = if x_zero, do: Nx.subtract(x, Nx.reshape(Nx.as_type(x_zero, :f32), {})), else: x
    w = Nx.as_type(w, :f32)

    w =
      cond do
        w_zero == nil ->
          w

        Nx.size(w_zero) == 1 ->
          Nx.subtract(w, Nx.reshape(Nx.as_type(w_zero, :f32), {}))

        true ->
          Nx.subtract(
            w,
            Nx.reshape(
              Nx.as_type(w_zero, :f32),
              List.to_tuple([:auto | List.duplicate(1, Nx.rank(w) - 1)])
            )
          )
      end

    %{node | op_type: "Conv"} |> op([x, w]) |> Nx.round() |> Nx.as_type(:s32)
  end

  defp op(node = %{op_type: "Conv"}, [x, w | b]) do
    spatial = Nx.rank(x) - 2
    pads = attr(node, "pads")

    padding =
      case {attr(node, "auto_pad", "NOTSET"), pads} do
        {"VALID", _} -> :valid
        {"NOTSET", nil} -> :valid
        {"NOTSET", pads} -> Enum.zip(Enum.take(pads, spatial), Enum.drop(pads, spatial))
      end

    out =
      Nx.conv(x, w,
        strides: attr(node, "strides", List.duplicate(1, spatial)),
        padding: padding,
        kernel_dilation: attr(node, "dilations", List.duplicate(1, spatial)),
        feature_group_size: attr(node, "group", 1)
      )

    case b do
      [b] when b != nil ->
        Nx.add(out, Nx.reshape(b, List.to_tuple([1, :auto] ++ List.duplicate(1, spatial))))

      _ ->
        out
    end
  end

  defp op(node = %{op_type: "InstanceNormalization"}, [x, scale, bias]) do
    axes = Enum.to_list(2..(Nx.rank(x) - 1))
    mean = Nx.mean(x, axes: axes, keep_axes: true)
    var = Nx.variance(x, axes: axes, keep_axes: true)
    shape = List.to_tuple([1, :auto | List.duplicate(1, length(axes))])

    x
    |> Nx.subtract(mean)
    |> Nx.multiply(Nx.rsqrt(Nx.add(var, attr(node, "epsilon", 1.0e-5))))
    |> Nx.multiply(Nx.reshape(scale, shape))
    |> Nx.add(Nx.reshape(bias, shape))
  end

  defp op(node = %{op_type: "BatchNormalization"}, [x, scale, bias, mean, var]) do
    shape = List.to_tuple([1, :auto | List.duplicate(1, Nx.rank(x) - 2)])
    r = &Nx.reshape(&1, shape)

    x
    |> Nx.subtract(r.(mean))
    |> Nx.multiply(r.(Nx.rsqrt(Nx.add(var, attr(node, "epsilon", 1.0e-5)))))
    |> Nx.multiply(r.(scale))
    |> Nx.add(r.(bias))
  end

  defp op(node = %{op_type: "LayerNormalization"}, [x, scale | bias]) do
    axis = attr(node, "axis", -1)
    axis = if axis < 0, do: axis + Nx.rank(x), else: axis
    axes = Enum.to_list(axis..(Nx.rank(x) - 1))
    mean = Nx.mean(x, axes: axes, keep_axes: true)
    var = Nx.variance(x, axes: axes, keep_axes: true)

    out =
      x
      |> Nx.subtract(mean)
      |> Nx.multiply(Nx.rsqrt(Nx.add(var, attr(node, "epsilon", 1.0e-5))))
      |> Nx.multiply(scale)

    case bias do
      [b | _] when b != nil -> Nx.add(out, b)
      _ -> out
    end
  end

  defp op(%{op_type: "GlobalAveragePool"}, [x]),
    do: Nx.mean(x, axes: Enum.to_list(2..(Nx.rank(x) - 1)), keep_axes: true)

  defp op(%{op_type: "GlobalMaxPool"}, [x]),
    do: Nx.reduce_max(x, axes: Enum.to_list(2..(Nx.rank(x) - 1)), keep_axes: true)

  defp op(node = %{op_type: pool}, [x]) when pool in ["MaxPool", "AveragePool", "LpPool"] do
    kernel = attr(node, "kernel_shape")
    spatial = length(kernel)
    strides = attr(node, "strides", List.duplicate(1, spatial))
    pads = attr(node, "pads", List.duplicate(0, 2 * spatial))
    sizes = x |> Nx.shape() |> Tuple.to_list() |> Enum.drop(2)

    extra =
      if attr(node, "ceil_mode", 0) == 1 do
        Enum.zip_with(
          [sizes, kernel, strides, Enum.take(pads, spatial), Enum.drop(pads, spatial)],
          fn [n, k, st, lo, hi] ->
            out = ceil((n + lo + hi - k) / st) + 1
            out = if (out - 1) * st >= n + lo, do: out - 1, else: out
            max((out - 1) * st + k - (n + lo + hi), 0)
          end
        )
      else
        List.duplicate(0, spatial)
      end

    padding = [
      {0, 0},
      {0, 0}
      | Enum.zip_with([Enum.take(pads, spatial), Enum.drop(pads, spatial), extra], fn [lo, hi, e] ->
          {lo, hi + e}
        end)
    ]

    window = List.to_tuple([1, 1 | kernel])
    opts = [strides: [1, 1 | strides], padding: padding]

    cond do
      pool == "MaxPool" ->
        Nx.window_max(x, window, opts)

      pool == "LpPool" ->
        p = attr(node, "p", 2)
        x |> Nx.abs() |> Nx.pow(p) |> Nx.window_sum(window, opts) |> Nx.pow(1 / p)

      true ->
        sum = Nx.window_sum(x, window, opts)

        if attr(node, "count_include_pad", 0) == 1 or Enum.all?(pads, &(&1 == 0)),
          do: Nx.divide(sum, Enum.product(kernel)),
          else:
            Nx.divide(
              sum,
              Nx.window_sum(
                Nx.broadcast(Nx.tensor(1.0, type: Nx.type(x)), Nx.shape(x)),
                window,
                opts
              )
            )
    end
  end

  defp op(node = %{op_type: "ConvTranspose"}, [x, w | b]) do
    groups = attr(node, "group", 1)

    if groups > 1 do
      group_in = div(Nx.axis_size(x, 1), groups)

      out =
        for g <- 0..(groups - 1) do
          op(%{node | attribute: Enum.reject(node.attribute, &(&1.name == "group"))}, [
            Nx.slice_along_axis(x, g * group_in, group_in, axis: 1),
            Nx.slice_along_axis(w, g * group_in, group_in, axis: 0)
          ])
        end
        |> Nx.concatenate(axis: 1)

      add_channel_bias(out, List.first(b))
    else
      conv_transpose(node, x, w, List.first(b))
    end
  end

  defp op(node = %{op_type: "LpNormalization"}, [x]) do
    axis = attr(node, "axis", -1)

    norm =
      case attr(node, "p", 2) do
        1 -> Nx.sum(Nx.abs(x), axes: [axis], keep_axes: true)
        2 -> x |> Nx.pow(2) |> Nx.sum(axes: [axis], keep_axes: true) |> Nx.sqrt()
      end

    Nx.divide(x, norm)
  end

  # Opset 21 scales each channel, opset 18 scales each group.
  defp op(node = %{op_type: "GroupNormalization"}, [x, scale, bias]) do
    groups = attr(node, "num_groups")
    shape = Nx.shape(x)
    {n, c} = {elem(shape, 0), elem(shape, 1)}
    grouped = Nx.reshape(x, {n, groups, :auto})
    mean = Nx.mean(grouped, axes: [2], keep_axes: true)
    var = Nx.variance(grouped, axes: [2], keep_axes: true)

    normed =
      grouped
      |> Nx.subtract(mean)
      |> Nx.multiply(Nx.rsqrt(Nx.add(var, attr(node, "epsilon", 1.0e-5))))

    {scale, bias} =
      if Nx.size(scale) == c,
        do: {scale, bias},
        else: {repeat_groups(scale, c), repeat_groups(bias, c)}

    channel = List.to_tuple([1, c | List.duplicate(1, tuple_size(shape) - 2)])

    normed
    |> Nx.reshape(shape)
    |> Nx.multiply(Nx.reshape(scale, channel))
    |> Nx.add(Nx.reshape(bias, channel))
  end

  # Resize one axis at a time. Each output element reads a few input elements, so the
  # taps become static index and weight tensors, and the result is a weighted sum of gathers.
  defp op(node = %{op_type: "Resize"}, [x | rest]) do
    {scales, sizes} =
      case rest do
        [scales] -> {scales, nil}
        [_roi, scales] -> {scales, nil}
        [_roi, scales, sizes] -> {scales, sizes}
      end

    in_shape = Tuple.to_list(Nx.shape(x))

    scales =
      if scales != nil and scales != :empty and Nx.size(scales) > 0,
        do: Nx.to_flat_list(scales),
        else: nil

    sizes =
      if sizes != nil and sizes != :empty,
        do: ints!(sizes),
        else: Enum.zip_with(in_shape, scales, &floor(&1 * &2))

    scales = scales || Enum.zip_with(sizes, in_shape, &(&1 / &2))

    [in_shape, sizes, scales]
    |> Enum.zip()
    |> Enum.with_index()
    |> Enum.reduce(x, fn
      {{same, same, _}, _}, acc ->
        acc

      {{in_size, out_size, scale}, axis}, acc ->
        resize_axis(acc, node, axis, in_size, out_size, scale)
    end)
  end

  # 2-D GridSample: bilinear or nearest, zeros or border padding.
  defp op(node = %{op_type: "GridSample"}, [x, grid]) do
    {n, c, h, w} = Nx.shape(x)
    {_, ho, wo, 2} = Nx.shape(grid)
    align = attr(node, "align_corners", 0) == 1
    mode = attr(node, "mode", "linear")
    padding = attr(node, "padding_mode", "zeros")

    unnormalize = fn g, size ->
      if align,
        do: Nx.multiply(Nx.add(g, 1), (size - 1) / 2),
        else: Nx.divide(Nx.subtract(Nx.multiply(Nx.add(g, 1), size), 1), 2)
    end

    gx = unnormalize.(grid[[.., .., .., 0]], w)
    gy = unnormalize.(grid[[.., .., .., 1]], h)

    {gx, gy} =
      if padding == "border", do: {Nx.clip(gx, 0, w - 1), Nx.clip(gy, 0, h - 1)}, else: {gx, gy}

    flat = Nx.reshape(x, {n, c, h * w})

    sample = fn xi, yi ->
      valid =
        Nx.logical_and(
          Nx.logical_and(Nx.greater_equal(xi, 0), Nx.less(xi, w)),
          Nx.logical_and(Nx.greater_equal(yi, 0), Nx.less(yi, h))
        )

      idx =
        Nx.add(Nx.multiply(Nx.clip(yi, 0, h - 1), w), Nx.clip(xi, 0, w - 1))
        |> Nx.as_type(:s32)
        |> Nx.reshape({n, 1, ho * wo})
        |> Nx.broadcast({n, c, ho * wo})

      vals = Nx.take_along_axis(flat, idx, axis: 2) |> Nx.reshape({n, c, ho, wo})
      Nx.multiply(vals, valid |> Nx.new_axis(1) |> Nx.as_type(Nx.type(x)))
    end

    case mode do
      m when m in ["linear", "bilinear"] ->
        x0 = Nx.floor(gx)
        y0 = Nx.floor(gy)
        wx = Nx.subtract(gx, x0) |> Nx.new_axis(1)
        wy = Nx.subtract(gy, y0) |> Nx.new_axis(1)
        x1 = Nx.add(x0, 1)
        y1 = Nx.add(y0, 1)

        top =
          Nx.add(
            Nx.multiply(sample.(x0, y0), Nx.subtract(1, wx)),
            Nx.multiply(sample.(x1, y0), wx)
          )

        bottom =
          Nx.add(
            Nx.multiply(sample.(x0, y1), Nx.subtract(1, wx)),
            Nx.multiply(sample.(x1, y1), wx)
          )

        Nx.add(Nx.multiply(top, Nx.subtract(1, wy)), Nx.multiply(bottom, wy))

      "nearest" ->
        sample.(Nx.round(gx), Nx.round(gy))
    end
  end

  # Inference only: the output is the input and the mask is all true.
  defp op(%{op_type: "Dropout"}, [x | _]),
    do: [x, Nx.broadcast(Nx.tensor(1, type: :u8), Nx.shape(x))]

  # Local response normalization over a window of channels.
  defp op(node = %{op_type: "LRN"}, [x]) do
    size = attr(node, "size")

    {alpha, beta, bias} =
      {attr(node, "alpha", 1.0e-4), attr(node, "beta", 0.75), attr(node, "bias", 1.0)}

    rank = Nx.rank(x)
    window = List.to_tuple([1, size | List.duplicate(1, rank - 2)])
    padding = [{0, 0}, {div(size - 1, 2), div(size, 2)} | List.duplicate({0, 0}, rank - 2)]
    square_sum = Nx.window_sum(Nx.pow(x, 2), window, padding: padding)
    Nx.divide(x, Nx.pow(Nx.add(bias, Nx.multiply(alpha / size, square_sum)), beta))
  end

  defp op(node = %{op_type: "GlobalLpPool"}, [x]) do
    p = attr(node, "p", 2)
    axes = Enum.to_list(2..(Nx.rank(x) - 1))
    x |> Nx.abs() |> Nx.pow(p) |> Nx.sum(axes: axes, keep_axes: true) |> Nx.pow(1 / p)
  end

  defp op(node = %{op_type: "MeanVarianceNormalization"}, [x]) do
    axes = attr(node, "axes", [0, 2, 3])
    mean = Nx.mean(x, axes: axes, keep_axes: true)

    variance =
      x |> Nx.pow(2) |> Nx.mean(axes: axes, keep_axes: true) |> Nx.subtract(Nx.pow(mean, 2))

    Nx.divide(Nx.subtract(x, mean), Nx.add(Nx.sqrt(variance), 1.0e-9))
  end

  defp conv_transpose(node, x, w, b) do
    spatial = Nx.rank(x) - 2
    kernel = w |> Nx.shape() |> Tuple.to_list() |> Enum.drop(2)
    strides = attr(node, "strides", List.duplicate(1, spatial))
    dilations = attr(node, "dilations", List.duplicate(1, spatial))
    pads = attr(node, "pads", List.duplicate(0, 2 * spatial))
    output_padding = attr(node, "output_padding", List.duplicate(0, spatial))

    padding =
      Enum.zip([
        kernel,
        dilations,
        Enum.take(pads, spatial),
        Enum.drop(pads, spatial),
        output_padding
      ])
      |> Enum.map(fn {k, d, lo, hi, op} -> {d * (k - 1) - lo, d * (k - 1) - hi + op} end)

    w =
      w
      |> Nx.transpose(axes: [1, 0 | Enum.to_list(2..(spatial + 1))])
      |> Nx.reverse(axes: Enum.to_list(2..(spatial + 1)))

    x
    |> Nx.conv(w,
      strides: 1,
      padding: padding,
      input_dilation: strides,
      kernel_dilation: dilations
    )
    |> add_channel_bias(b)
  end

  defp add_channel_bias(out, nil), do: out

  defp add_channel_bias(out, b),
    do:
      Nx.add(out, Nx.reshape(b, List.to_tuple([1, :auto] ++ List.duplicate(1, Nx.rank(out) - 2))))

  defp repeat_groups(t, channels) do
    groups = Nx.size(t)
    t |> Nx.new_axis(1) |> Nx.broadcast({groups, div(channels, groups)}) |> Nx.reshape({channels})
  end

  defp resize_axis(x, node, axis, in_size, out_size, scale) do
    ctm = attr(node, "coordinate_transformation_mode", "half_pixel")

    taps =
      for i <- 0..(out_size - 1),
          do: resize_taps(node, resize_source(ctm, i, in_size, out_size, scale), in_size, scale)

    width = taps |> Enum.map(&length/1) |> Enum.max()
    taps = Enum.map(taps, &(&1 ++ List.duplicate({0, 0.0}, width - length(&1))))
    indices = taps |> Enum.map(fn row -> Enum.map(row, &elem(&1, 0)) end) |> static_tensor(:s64)
    weights = taps |> Enum.map(fn row -> Enum.map(row, &elem(&1, 1)) end) |> static_tensor(:f32)

    weight_shape =
      x
      |> Nx.shape()
      |> Tuple.to_list()
      |> Enum.map(fn _ -> 1 end)
      |> List.replace_at(axis, out_size)
      |> List.to_tuple()

    for k <- 0..(width - 1) do
      w =
        weights
        |> Nx.slice_along_axis(k, 1, axis: 1)
        |> Nx.reshape(weight_shape)
        |> Nx.as_type(Nx.type(x))

      x
      |> Nx.take(indices |> Nx.slice_along_axis(k, 1, axis: 1) |> Nx.reshape({out_size}),
        axis: axis
      )
      |> Nx.multiply(w)
    end
    |> Enum.reduce(&Nx.add/2)
  end

  defp resize_source("half_pixel", i, _in, _out, scale), do: (i + 0.5) / scale - 0.5
  defp resize_source("pytorch_half_pixel", _i, _in, 1, _scale), do: 0.0
  defp resize_source("pytorch_half_pixel", i, _in, _out, scale), do: (i + 0.5) / scale - 0.5
  defp resize_source("align_corners", _i, _in, 1, _scale), do: 0.0

  defp resize_source("align_corners", i, in_size, out_size, _scale),
    do: i * (in_size - 1) / (out_size - 1)

  defp resize_source("asymmetric", i, _in, _out, scale), do: i / scale
  defp resize_source("tf_half_pixel_for_nn", i, _in, _out, scale), do: (i + 0.5) / scale

  # {input index, weight} pairs that make one output element.
  defp resize_taps(node, xo, in_size, scale) do
    mode = attr(node, "mode", "nearest")

    if mode in ["linear", "cubic"] and attr(node, "antialias", 0) == 1,
      do: antialias_taps(node, mode, xo, in_size, scale),
      else: interpolation_taps(node, mode, xo, in_size)
  end

  # Antialiasing widens the kernel by 1 / scale when downscaling, like Pillow. As in
  # ONNX Runtime, taps outside the input move to the edge unless exclude_outside is set,
  # and the weights are normalized.
  defp antialias_taps(node, mode, xo, in_size, scale) do
    a = attr(node, "cubic_coeff_a", -0.75)
    stretch = min(scale, 1.0)
    support = if(mode == "linear", do: 1.0, else: 2.0) / stretch
    window = 2 * ceil(support) + 1
    center = xo + 0.5
    low = floor(center - support + 0.5)
    high = low + min(max(floor(center + support + 0.5) - low, 0), window)

    taps =
      for j <- low..(high - 1)//1 do
        t = (j - xo) * stretch
        {j, if(mode == "linear", do: max(0.0, 1 - abs(t)), else: cubic_weight(a, t))}
      end

    if attr(node, "exclude_outside", 0) == 1,
      do: taps |> Enum.filter(fn {j, _} -> j >= 0 and j < in_size end) |> renormalize(),
      else: taps |> Enum.map(fn {j, w} -> {min(max(j, 0), in_size - 1), w} end) |> renormalize()
  end

  defp interpolation_taps(node, mode, xo, in_size) do
    clamp = &min(max(&1, 0), in_size - 1)

    case mode do
      "nearest" ->
        [{clamp.(nearest_index(attr(node, "nearest_mode", "round_prefer_floor"), xo)), 1.0}]

      "linear" ->
        xo = min(max(xo, 0.0), in_size - 1.0)
        x0 = floor(xo)
        d = xo - x0
        [{x0, 1 - d}, {clamp.(x0 + 1), d}]

      "cubic" ->
        a = attr(node, "cubic_coeff_a", -0.75)
        x0 = floor(xo)
        d = xo - x0

        taps =
          for {off, s} <- [{-1, d + 1}, {0, d}, {1, 1 - d}, {2, 2 - d}],
              do: {x0 + off, cubic_weight(a, s)}

        if attr(node, "exclude_outside", 0) == 1,
          do: renormalize(Enum.filter(taps, fn {j, _} -> j >= 0 and j < in_size end)),
          else: Enum.map(taps, fn {j, w} -> {clamp.(j), w} end)
    end
  end

  defp nearest_index(mode, xo) do
    half? = xo - Float.floor(xo) == 0.5

    case mode do
      "round_prefer_floor" -> if half?, do: floor(xo), else: round(xo)
      "round_prefer_ceil" -> if half?, do: ceil(xo), else: round(xo)
      "floor" -> floor(xo)
      "ceil" -> ceil(xo)
    end
  end

  defp cubic_weight(a, s) do
    s = abs(s)

    cond do
      s <= 1 -> (a + 2) * s ** 3 - (a + 3) * s ** 2 + 1
      s < 2 -> a * s ** 3 - 5 * a * s ** 2 + 8 * a * s - 4 * a
      true -> 0.0
    end
  end

  defp renormalize(taps) do
    total = taps |> Enum.map(&elem(&1, 1)) |> Enum.sum()
    Enum.map(taps, fn {j, w} -> {j, w / total} end)
  end
end
