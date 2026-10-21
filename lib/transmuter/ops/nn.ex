defmodule Transmuter.Ops.NN do
  @moduledoc """
  Neural network layers: convolutions, pooling, unpooling, normalizations, Resize, Upsample, GridSample, LRN, Dropout and the classification losses.
  """

  import Transmuter.Ops,
    only: [
      attr: 2,
      attr: 3,
      emlx?: 1,
      emlx_norm: 4,
      empty?: 1,
      normalize: 3,
      normalize_axis: 2,
      on_axis: 3,
      optional: 2,
      round_half_even: 1,
      static_tensor: 2,
      values!: 1,
      widen: 2
    ]

  alias Transmuter.Onnx.AttributeProto

  @op_types ~w(Conv InstanceNormalization BatchNormalization LayerNormalization
               GlobalAveragePool GlobalMaxPool ConvTranspose LpNormalization GroupNormalization
               Resize GridSample Dropout LRN GlobalLpPool MeanVarianceNormalization MaxPool
               AveragePool LpPool Upsample MaxUnpool CausalConvWithState NegativeLogLikelihoodLoss
               SoftmaxCrossEntropyLoss)

  @doc false
  def op_types, do: @op_types

  @doc false
  def apply(node, inputs), do: node |> op(inputs) |> List.wrap()

  defp op(node = %{op_type: "Conv"}, [x, w | b]) do
    out =
      if patches?(node, x, w) do
        patch_conv(x, w, &Nx.dot(&1, [2], &2, [1]))
      else
        {strides, dilations, pads} = window(node, spatial_sizes(x), spatial_sizes(w))

        Nx.conv(x, w,
          strides: strides,
          padding: pads,
          kernel_dilation: dilations,
          feature_group_size: attr(node, "group", 1)
        )
      end

    add_channel_bias(out, List.first(b))
  end

  defp op(node = %{op_type: "InstanceNormalization"}, [x, scale, bias]) do
    channel = &on_axis(&1, Nx.rank(x), 1)

    x
    |> normalize_trailing(Enum.to_list(2..(Nx.rank(x) - 1)), attr(node, "epsilon", 1.0e-5))
    |> Nx.multiply(channel.(scale))
    |> Nx.add(channel.(bias))
  end

  defp op(node = %{op_type: "BatchNormalization"}, [x, scale, bias, mean, var]) do
    r = &on_axis(&1, Nx.rank(x), 1)

    x
    |> Nx.subtract(r.(mean))
    |> Nx.multiply(r.(Nx.rsqrt(Nx.add(var, attr(node, "epsilon", 1.0e-5)))))
    |> Nx.multiply(r.(scale))
    |> Nx.add(r.(bias))
  end

  defp op(node = %{op_type: "LayerNormalization"}, [x, scale | bias]) do
    axis = normalize_axis(attr(node, "axis", -1), Nx.rank(x))
    axes = Enum.to_list(axis..(Nx.rank(x) - 1))
    eps = attr(node, "epsilon", 1.0e-5)

    bias = List.first(bias) || Nx.tensor(0, type: Nx.type(x))

    if emlx?(x) do
      emlx_norm(&EMLX.Fast.layer_norm(&1, &2, &3, eps), x, axes, [scale, bias])
    else
      [x, scale, bias]
      |> widen(fn [x, scale, bias] ->
        [x |> normalize(axes, eps) |> Nx.multiply(scale) |> Nx.add(bias)]
      end)
      |> hd()
    end
  end

  defp op(%{op_type: "GlobalAveragePool"}, [x]),
    do: Nx.mean(x, axes: Enum.to_list(2..(Nx.rank(x) - 1)), keep_axes: true)

  defp op(%{op_type: "GlobalMaxPool"}, [x]),
    do: Nx.reduce_max(x, axes: Enum.to_list(2..(Nx.rank(x) - 1)), keep_axes: true)

  # ceil_mode adds the padding that a last, partial window needs. This padding never
  # counts in the divisor of AveragePool, the explicit padding counts with count_include_pad.
  defp op(node = %{op_type: pool}, [x]) when pool in ["MaxPool", "AveragePool", "LpPool"] do
    kernel = attr(node, "kernel_shape")
    sizes = spatial_sizes(x)
    {strides, dilations, pads} = window(node, sizes, kernel)

    extra =
      if attr(node, "ceil_mode", 0) == 1 do
        Enum.zip_with([sizes, kernel, strides, dilations, pads], fn [n, k, st, d, {lo, hi}] ->
          span = (k - 1) * d + 1
          out = ceil((n + lo + hi - span) / st) + 1
          out = if (out - 1) * st >= n + lo, do: out - 1, else: out
          max((out - 1) * st + span - (n + lo + hi), 0)
        end)
      else
        List.duplicate(0, length(kernel))
      end

    window = List.to_tuple([1, 1 | kernel])
    window_opts = [strides: [1, 1 | strides], window_dilations: [1, 1 | dilations]]
    padding = [{0, 0}, {0, 0} | Enum.zip_with(pads, extra, fn {lo, hi}, e -> {lo, hi + e} end)]
    opts = [padding: padding] ++ window_opts

    case pool do
      "MaxPool" ->
        values = Nx.window_max(x, window, opts)

        if Enum.at(node.output, 1, "") == "",
          do: values,
          else: [values, max_pool_indices(node, x, padding, strides, dilations)]

      "LpPool" ->
        p = attr(node, "p", 2)
        x |> Nx.abs() |> Nx.pow(p) |> Nx.window_sum(window, opts) |> Nx.pow(1 / p)

      "AveragePool" ->
        type = Nx.type(x)
        include = attr(node, "count_include_pad", 0)

        counts =
          Nx.tensor(1, type: type)
          |> Nx.broadcast(List.to_tuple([1, 1 | sizes]))
          |> Nx.pad(
            Nx.tensor(include, type: type),
            [{0, 0, 0}, {0, 0, 0} | for({lo, hi} <- pads, do: {lo, hi, 0})]
          )
          |> Nx.window_sum(
            window,
            [padding: [{0, 0}, {0, 0} | for(e <- extra, do: {0, e})]] ++ window_opts
          )

        Nx.divide(Nx.window_sum(x, window, opts), counts)
    end
  end

  defp op(node = %{op_type: "ConvTranspose"}, [x, w | b]),
    do: conv_transpose(node, x, w, List.first(b))

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
    normed = normalize_trailing(grouped, [2], attr(node, "epsilon", 1.0e-5))

    {scale, bias} =
      if Nx.size(scale) == c,
        do: {scale, bias},
        else: {repeat_groups(scale, c), repeat_groups(bias, c)}

    normed
    |> Nx.reshape(shape)
    |> Nx.multiply(on_axis(scale, tuple_size(shape), 1))
    |> Nx.add(on_axis(bias, tuple_size(shape), 1))
  end

  # Resize one axis at a time. Each output element reads a few input elements, so the
  # taps become static index and weight tensors, and the result is a weighted sum of gathers.
  # Scales, sizes and the roi list the axes of the axes attribute (opset 18), all by
  # default. Integers resize in f32 and truncate back, as in ONNX Runtime, except with
  # nearest, which copies values and stays exact in the integer type.
  defp op(node = %{op_type: "Resize"}, [x | rest]) do
    {roi, scales, sizes} =
      case rest do
        [scales] -> {nil, scales, nil}
        [roi, scales] -> {roi, scales, nil}
        [roi, scales, sizes] -> {roi, scales, sizes}
      end

    in_shape = Tuple.to_list(Nx.shape(x))
    rank = length(in_shape)

    axes =
      node |> attr("axes", Enum.to_list(0..(rank - 1))) |> Enum.map(&normalize_axis(&1, rank))

    in_sizes = Enum.map(axes, &Enum.at(in_shape, &1))
    given = &(&1 != nil and not empty?(&1) and Nx.size(&1) > 0)

    {scales, sizes} =
      if given.(sizes) do
        sizes = values!(sizes)

        case attr(node, "keep_aspect_ratio_policy", "stretch") do
          "stretch" ->
            {Enum.zip_with(sizes, in_sizes, &(&1 / &2)), sizes}

          policy ->
            ratios = Enum.zip_with(sizes, in_sizes, &(&1 / &2))
            scale = if policy == "not_larger", do: Enum.min(ratios), else: Enum.max(ratios)
            {Enum.map(in_sizes, fn _ -> scale end), Enum.map(in_sizes, &round(&1 * scale))}
        end
      else
        scales = values!(scales)
        {scales, Enum.zip_with(in_sizes, scales, &floor(&1 * &2))}
      end

    count = length(axes)

    # Only tf_crop_and_resize reads the roi, so a traced roi does not break the graph.
    crop? = attr(node, "coordinate_transformation_mode") == "tf_crop_and_resize"

    roi =
      if crop? and given.(roi),
        do: values!(roi),
        else: List.duplicate(0, count) ++ List.duplicate(1, count)

    regions = Enum.zip(Enum.take(roi, count), Enum.drop(roi, count))
    type = Nx.type(x)
    widen? = Nx.Type.integer?(type) and attr(node, "mode", "nearest") != "nearest"
    x = if widen?, do: Nx.as_type(x, :f32), else: x

    out =
      [axes, in_sizes, sizes, scales, regions]
      |> Enum.zip()
      |> Enum.reduce(x, fn {axis, in_size, out_size, scale, region}, acc ->
        resize_axis(acc, node, axis, {in_size, out_size, scale, region})
      end)

    if widen?,
      do:
        out
        |> Nx.clip(Nx.Constants.min_finite(type), Nx.Constants.max_finite(type))
        |> Nx.as_type(type),
      else: out
  end

  # Upsample maps coordinates like Resize before opset 11: asymmetric, and nearest rounds
  # down. Opset 7 has the scales as an attribute, opset 9 as an input.
  defp op(node = %{op_type: "Upsample"}, [x | scales]) do
    scales = List.first(scales) || static_tensor(attr(node, "scales"), :f32)
    mode = if attr(node, "mode", "nearest") == "nearest", do: "nearest", else: "linear"

    resize = %{
      node
      | op_type: "Resize",
        attribute: [
          %AttributeProto{name: "mode", type: :STRING, s: mode},
          %AttributeProto{name: "coordinate_transformation_mode", type: :STRING, s: "asymmetric"},
          %AttributeProto{name: "nearest_mode", type: :STRING, s: "floor"}
        ]
    }

    op(resize, [x, scales])
  end

  # The indices are flat over the whole output, as MaxPool gives them.
  defp op(node = %{op_type: "MaxUnpool"}, [x, indices | output_shape]) do
    kernel = attr(node, "kernel_shape")
    [n, c | sizes] = Tuple.to_list(Nx.shape(x))
    {strides, _dilations, pads} = window(node, sizes, kernel)

    shape =
      if s = List.first(output_shape),
        do: values!(s),
        else: [
          n,
          c
          | Enum.zip_with([sizes, kernel, strides, pads], fn [size, k, st, {lo, hi}] ->
              (size - 1) * st - lo - hi + k
            end)
        ]

    Nx.tensor(0, type: Nx.type(x))
    |> Nx.broadcast({Enum.product(shape)})
    |> Nx.indexed_put(Nx.reshape(indices, {:auto, 1}), Nx.reshape(x, {:auto}))
    |> Nx.reshape(List.to_tuple(shape))
  end

  # Depthwise causal Conv1d over the past state and the input. The present state is the
  # last k - 1 positions of that sequence.
  defp op(node = %{op_type: "CausalConvWithState"}, [x, w | rest]) do
    [bias, past] = optional(rest, 2)
    {b, c, _} = Nx.shape(x)
    k = Nx.axis_size(w, 2)
    past = past || Nx.broadcast(Nx.tensor(0, type: Nx.type(x)), {b, c, k - 1})
    padded = Nx.concatenate([past, x], axis: 2)
    out = padded |> Nx.conv(w, feature_group_size: c) |> add_channel_bias(bias)

    out =
      if attr(node, "activation", "none") in ["silu", "swish"],
        do: Nx.multiply(out, Nx.sigmoid(out)),
        else: out

    [out, Nx.slice_along_axis(padded, Nx.axis_size(padded, 2) - (k - 1), k - 1, axis: 2)]
  end

  # GridSample over any number of spatial axes, linear or nearest, as ONNX Runtime does it.
  # The last axis of the grid lists the coordinates from the last spatial axis (x) to the
  # first. A coordinate out of bounds moves to the border or reflects into the bounds, and
  # each corner of the linear interpolation reads zeros, the border or its reflection.
  defp op(node = %{op_type: "GridSample"}, [x, grid]) do
    padding = attr(node, "padding_mode", "zeros")
    axes = grid_coordinates(node, grid, x |> Nx.shape() |> Tuple.to_list() |> Enum.drop(2))
    out = grid |> Nx.shape() |> Tuple.to_list() |> Enum.slice(1, length(axes))

    case attr(node, "mode", "linear") do
      "nearest" ->
        grid_pixels(x, axes, padding, out)

      mode when mode in ["linear", "bilinear"] ->
        corners =
          Enum.reduce(Enum.reverse(axes), [{[], 1}], fn {coord, size, bounds}, corners ->
            low = Nx.floor(coord)
            weight = Nx.subtract(coord, low)

            for {rest, w} <- corners,
                {i, wi} <- [{low, Nx.subtract(1, weight)}, {Nx.add(low, 1), weight}],
                do: {[{i, size, bounds} | rest], Nx.multiply(w, wi)}
          end)

        corners
        |> Enum.map(fn {corner, w} ->
          x |> grid_pixels(corner, padding, out) |> Nx.multiply(Nx.new_axis(w, 1))
        end)
        |> Enum.reduce(&Nx.add/2)

      mode ->
        raise ArgumentError, "GridSample mode #{inspect(mode)} is not supported"
    end
  end

  # Inference only: the output is the input and the mask is all true.
  defp op(%{op_type: "Dropout"}, [x | _]),
    do: [x, Nx.broadcast(Nx.tensor(1, type: :u8), Nx.shape(x))]

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

  # The variance is the mean of the squares of the centered values, which does not cancel
  # out as E[x^2] - E[x]^2 does for a large mean.
  defp op(node = %{op_type: "MeanVarianceNormalization"}, [x]) do
    axes = attr(node, "axes", [0, 2, 3])
    centered = Nx.subtract(x, Nx.mean(x, axes: axes, keep_axes: true))
    variance = centered |> Nx.multiply(centered) |> Nx.mean(axes: axes, keep_axes: true)
    Nx.divide(centered, Nx.add(Nx.sqrt(variance), 1.0e-9))
  end

  defp op(node = %{op_type: "NegativeLogLikelihoodLoss"}, [x, target | weight]),
    do: nll_loss(node, x, target, List.first(weight))

  defp op(node = %{op_type: "SoftmaxCrossEntropyLoss"}, [scores, labels | weight]) do
    shifted = Nx.subtract(scores, Nx.reduce_max(scores, axes: [1], keep_axes: true))
    log_prob = Nx.subtract(shifted, Nx.log(Nx.sum(Nx.exp(shifted), axes: [1], keep_axes: true)))
    [nll_loss(node, log_prob, labels, List.first(weight)), log_prob]
  end

  # x is {N, C, d...} and target {N, d...}. Ignored targets get weight 0, and the mean
  # divides by the sum of the weights that the targets pick.
  defp nll_loss(node, x, target, weight) do
    type = Nx.type(x)
    index = Nx.clip(target, 0, Nx.axis_size(x, 1) - 1)
    picked = x |> Nx.take_along_axis(Nx.new_axis(index, 1), axis: 1) |> Nx.squeeze(axes: [1])

    weights =
      if weight,
        do: Nx.take(weight, index),
        else: Nx.broadcast(Nx.tensor(1, type: type), Nx.shape(target))

    weights =
      case attr(node, "ignore_index") do
        nil -> weights
        ignore -> Nx.select(Nx.equal(target, ignore), Nx.tensor(0, type: type), weights)
      end

    loss = Nx.multiply(Nx.negate(picked), weights)

    case attr(node, "reduction", "mean") do
      "none" -> loss
      "sum" -> Nx.sum(loss)
      "mean" -> Nx.divide(Nx.sum(loss), Nx.sum(weights))
    end
  end

  # Normalizes x over its trailing axes, without a weight. EMLX does it in one kernel.
  defp normalize_trailing(x, axes, eps) do
    if emlx?(x) do
      [one, zero] = Enum.map([1, 0], &Nx.tensor(&1, type: Nx.type(x)))
      emlx_norm(&EMLX.Fast.layer_norm(&1, &2, &3, eps), x, axes, [one, zero])
    else
      normalize(x, axes, eps)
    end
  end

  @doc """
  Computes a convolution of zero-shifted integers, with x and w as f32, to s32.

  It is one exact integer product (see `integer_product/5`) per kernel tap, or one for all patches when the windows do not overlap. A backend convolution can be inexact: MLX uses Winograd on the GPU for 3x3 kernels, and is off by up to 15 on sums of about 1.6e6.
  """
  def integer_conv(node, x, w) do
    if patches?(node, x, w),
      do: patch_conv(x, w, &integer_product(&1, 2, &2, 1, fn a, b -> Nx.dot(a, [2], b, [1]) end)),
      else: tap_conv(node, x, w)
  end

  # A convolution whose windows do not overlap and cover the input without padding, as
  # the patch embedding of a vision transformer, multiplies each patch by the kernel.
  defp patches?(node, x, w) do
    kernel = spatial_sizes(w)
    {_strides, dilations, pads} = window(node, spatial_sizes(x), kernel)

    attr(node, "strides") == kernel and attr(node, "group", 1) == 1 and
      Enum.all?(dilations, &(&1 == 1)) and Enum.all?(pads, &(&1 == {0, 0}))
  end

  # The patches of x become rows {n, patches, c * kernel}, and product.(rows, w) multiplies
  # them by w {o, c * kernel} to {n, patches, o}. Each axis drops the part of the input
  # that a whole patch does not cover.
  defp patch_conv(x, w, product) do
    [n, c | sizes] = Tuple.to_list(Nx.shape(x))
    [o, _c | kernel] = Tuple.to_list(Nx.shape(w))
    spatial = length(kernel)
    out = Enum.zip_with(sizes, kernel, &div/2)
    split = Enum.flat_map(Enum.zip(out, kernel), &Tuple.to_list/1)

    rows =
      x
      |> Nx.slice(List.duplicate(0, spatial + 2), [n, c | Enum.zip_with(out, kernel, &(&1 * &2))])
      |> Nx.reshape(List.to_tuple([n, c | split]))
      |> Nx.transpose(
        axes:
          [0] ++
            Enum.map(0..(spatial - 1), &(2 + 2 * &1)) ++
            [1] ++ Enum.map(0..(spatial - 1), &(3 + 2 * &1))
      )
      |> Nx.reshape({n, Enum.product(out), c * Enum.product(kernel)})

    rows
    |> product.(Nx.reshape(w, {o, :auto}))
    |> Nx.transpose(axes: [0, 2, 1])
    |> Nx.reshape(List.to_tuple([n, o | out]))
  end

  defp tap_conv(node, x, w) do
    [n, c | _] = Tuple.to_list(Nx.shape(x))
    [o, group_c | kernel] = Tuple.to_list(Nx.shape(w))
    spatial = length(kernel)
    {strides, dilations, pads} = window(node, spatial_sizes(x), kernel)
    groups = div(c, group_c)
    x = Nx.pad(x, 0.0, [{0, 0, 0}, {0, 0, 0} | for({lo, hi} <- pads, do: {lo, hi, 0})])

    padded = x |> Nx.shape() |> Tuple.to_list() |> Enum.drop(2)

    out =
      Enum.zip_with([padded, kernel, strides, dilations], fn [size, k, s, d] ->
        div(size - (k - 1) * d - 1, s) + 1
      end)

    # Groups lead, because Nx batches dot products on leading axes.
    x =
      x
      |> Nx.reshape(List.to_tuple([n, groups, group_c | padded]))
      |> Nx.transpose(axes: [1, 0, 2 | Enum.to_list(3..(spatial + 2))])

    w = Nx.reshape(w, List.to_tuple([groups, div(o, groups), group_c | kernel]))

    sum =
      kernel
      |> Enum.map(&Enum.to_list(0..(&1 - 1)))
      |> Enum.reduce([[]], fn range, taps -> for tap <- taps, i <- range, do: tap ++ [i] end)
      |> Enum.map(fn tap ->
        starts = Enum.zip_with(tap, dilations, &(&1 * &2))
        lengths = Enum.zip_with(out, strides, &((&1 - 1) * &2 + 1))

        window =
          Nx.slice(x, [0, 0, 0 | starts], [groups, n, group_c | lengths],
            strides: [1, 1, 1 | strides]
          )

        tap_w =
          w
          |> Nx.slice([0, 0, 0 | tap], [
            groups,
            div(o, groups),
            group_c | List.duplicate(1, spatial)
          ])
          |> Nx.reshape({groups, div(o, groups), group_c})

        integer_product(tap_w, 2, window, 2, &Nx.dot(&1, [2], [0], &2, [2], [0]))
      end)
      |> Enum.reduce(&Nx.add/2)

    # {groups, o / groups, n, out...} -> {n, o, out...}
    sum
    |> Nx.transpose(axes: [2, 0, 1 | Enum.to_list(3..(spatial + 2))])
    |> Nx.reshape(List.to_tuple([n, o | out]))
  end

  @doc """
  Computes `product.(a, b)`, a sum of products along axis `a_axis` of a and `b_axis` of b, exactly, as s32.

  a and b are f32 tensors of integers of magnitude up to 255, such as zero-shifted 8-bit values. An f32 sum of up to 256 such products is exact, because 256 * 255^2 < 2^24. A longer sum splits into chunks of 256 along the axes, and the s32 sums of the chunks are exact.
  """
  def integer_product(a, a_axis, b, b_axis, product) do
    size = Nx.axis_size(a, a_axis)

    for start <- 0..(size - 1)//256 do
      length = min(256, size - start)

      product.(
        Nx.slice_along_axis(a, start, length, axis: a_axis),
        Nx.slice_along_axis(b, start, length, axis: b_axis)
      )
      |> Nx.as_type(:s32)
    end
    |> Enum.reduce(&Nx.add/2)
  end

  # The strides, dilations and {low, high} padding per spatial axis of a window of size
  # kernel over sizes, for convolutions and pooling. SAME_UPPER and SAME_LOWER pad so the
  # output size is ceil(input / stride), with the odd padding at the end or at the start.
  defp window(node, sizes, kernel) do
    spatial = length(sizes)
    strides = attr(node, "strides", List.duplicate(1, spatial))
    dilations = attr(node, "dilations", List.duplicate(1, spatial))

    pads =
      case {attr(node, "auto_pad", "NOTSET"), attr(node, "pads")} do
        {"NOTSET", nil} ->
          List.duplicate({0, 0}, spatial)

        {"NOTSET", pads} ->
          Enum.zip(Enum.take(pads, spatial), Enum.drop(pads, spatial))

        {"VALID", _} ->
          List.duplicate({0, 0}, spatial)

        {same, _} ->
          Enum.zip_with([sizes, kernel, strides, dilations], fn [size, k, s, d] ->
            total = max((div(size + s - 1, s) - 1) * s + (k - 1) * d + 1 - size, 0)
            small = div(total, 2)
            if same == "SAME_UPPER", do: {small, total - small}, else: {total - small, small}
          end)
      end

    {strides, dilations, pads}
  end

  defp spatial_sizes(t), do: t |> Nx.shape() |> Tuple.to_list() |> Enum.drop(2)

  # The flat input position of the first maximum of each window, over the whole input as
  # MaxPool gives it: row-major, or column-major in each channel with storage_order 1.
  # Each kernel tap is one strided slice of the padded input.
  defp max_pool_indices(node, x, padding, strides, dilations) do
    [n, c | sizes] = Tuple.to_list(Nx.shape(x))
    kernel = attr(node, "kernel_shape")
    low = padding |> Enum.drop(2) |> Enum.map(&elem(&1, 0))

    padded =
      Nx.pad(x, Nx.Constants.min_finite(Nx.type(x)), for({lo, hi} <- padding, do: {lo, hi, 0}))

    padded_sizes = spatial_sizes(padded)

    out =
      Enum.zip_with([padded_sizes, kernel, strides, dilations], fn [size, k, s, d] ->
        div(size - (k - 1) * d - 1, s) + 1
      end)

    taps =
      kernel
      |> Enum.map(&Enum.to_list(0..(&1 - 1)))
      |> Enum.reduce([[]], fn range, taps -> for tap <- taps, i <- range, do: tap ++ [i] end)

    best =
      taps
      |> Enum.map(fn tap ->
        starts = Enum.zip_with(tap, dilations, &(&1 * &2))
        lengths = Enum.zip_with(out, strides, &((&1 - 1) * &2 + 1))
        Nx.slice(padded, [0, 0 | starts], [n, c | lengths], strides: [1, 1 | strides])
      end)
      |> Nx.stack()
      |> Nx.argmax(axis: 0)

    shape = List.to_tuple([n, c | out])

    coords =
      for {{s, d, lo}, axis} <- Enum.with_index(Enum.zip([strides, dilations, low])) do
        offsets = static_tensor(Enum.map(taps, &Enum.at(&1, axis)), :s64)

        Nx.iota(shape, axis: axis + 2, type: :s64)
        |> Nx.multiply(s)
        |> Nx.subtract(lo)
        |> Nx.add(Nx.multiply(Nx.take(offsets, best), d))
      end

    # The fastest axis is the last one, or the first one with storage_order 1.
    axes = Enum.to_list(0..(length(sizes) - 1))
    fastest_first = if attr(node, "storage_order", 0) == 1, do: axes, else: Enum.reverse(axes)

    {within, _} =
      Enum.reduce(fastest_first, {0, 1}, fn axis, {flat, place} ->
        {Nx.add(flat, Nx.multiply(Enum.at(coords, axis), place)), place * Enum.at(sizes, axis)}
      end)

    Nx.iota(shape, axis: 0, type: :s64)
    |> Nx.multiply(c)
    |> Nx.add(Nx.iota(shape, axis: 1, type: :s64))
    |> Nx.multiply(Enum.product(sizes))
    |> Nx.add(within)
  end

  # The full output of a transposed convolution has (in - 1) * stride + output_padding +
  # (kernel - 1) * dilation + 1 positions per axis. The pads remove positions from it,
  # explicitly or to reach output_shape, or in * stride with SAME_UPPER and SAME_LOWER.
  defp conv_transpose(node, x, w, b) do
    spatial = Nx.rank(x) - 2
    kernel = spatial_sizes(w)
    strides = attr(node, "strides", List.duplicate(1, spatial))
    dilations = attr(node, "dilations", List.duplicate(1, spatial))
    output_padding = attr(node, "output_padding", List.duplicate(0, spatial))
    sizes = spatial_sizes(x)
    auto_pad = attr(node, "auto_pad", "NOTSET")

    full =
      Enum.zip_with([sizes, strides, output_padding, kernel, dilations], fn [n, s, adj, k, d] ->
        (n - 1) * s + adj + (k - 1) * d + 1
      end)

    targets =
      cond do
        shape = attr(node, "output_shape") -> Enum.take(shape, -spatial)
        auto_pad in ["SAME_UPPER", "SAME_LOWER"] -> Enum.zip_with(sizes, strides, &(&1 * &2))
        true -> nil
      end

    pads =
      cond do
        targets ->
          Enum.zip_with(full, targets, fn size, target ->
            total = max(size - target, 0)
            small = div(total, 2)
            if auto_pad == "SAME_UPPER", do: {small, total - small}, else: {total - small, small}
          end)

        auto_pad == "VALID" ->
          List.duplicate({0, 0}, spatial)

        true ->
          pads = attr(node, "pads", List.duplicate(0, 2 * spatial))
          Enum.zip(Enum.take(pads, spatial), Enum.drop(pads, spatial))
      end

    padding =
      Enum.zip_with([kernel, dilations, pads, output_padding], fn [k, d, {lo, hi}, adj] ->
        {d * (k - 1) - lo, d * (k - 1) - hi + adj}
      end)

    # The weight is {in, out / groups, kernel...}. Each group becomes the
    # {out / groups, in / groups, kernel...} kernel of a grouped conv.
    groups = attr(node, "group", 1)
    [c_in, group_out | _] = Tuple.to_list(Nx.shape(w))
    spatial_axes = Enum.to_list(3..(spatial + 2)//1)

    w =
      w
      |> Nx.reshape(List.to_tuple([groups, div(c_in, groups), group_out | kernel]))
      |> Nx.transpose(axes: [0, 2, 1 | spatial_axes])
      |> Nx.reshape(List.to_tuple([groups * group_out, div(c_in, groups) | kernel]))
      |> Nx.reverse(axes: Enum.to_list(2..(spatial + 1)))

    x
    |> Nx.conv(w,
      strides: 1,
      padding: padding,
      input_dilation: strides,
      kernel_dilation: dilations,
      feature_group_size: groups
    )
    |> add_channel_bias(b)
  end

  defp add_channel_bias(out, nil), do: out

  defp add_channel_bias(out, b), do: Nx.add(out, on_axis(b, Nx.rank(out), 1))

  # {coordinate, size, bounds} for each spatial axis, in input pixels. Nearest rounds the
  # coordinates. A coordinate out of bounds moves to the border or reflects into the bounds.
  defp grid_coordinates(node, grid, sizes) do
    spatial = length(sizes)
    align = attr(node, "align_corners", 0) == 1

    for {size, i} <- Enum.with_index(sizes) do
      g = grid |> Nx.slice_along_axis(spatial - 1 - i, 1, axis: -1) |> Nx.squeeze(axes: [-1])

      coord =
        if align,
          do: Nx.multiply(Nx.add(g, 1), (size - 1) / 2),
          else: Nx.divide(Nx.subtract(Nx.multiply(Nx.add(g, 1), size), 1), 2)

      coord =
        if attr(node, "mode", "linear") == "nearest", do: round_half_even(coord), else: coord

      bounds = {low, high} = if align, do: {0, size - 1}, else: {-0.5, size - 0.5}
      outside = Nx.logical_or(Nx.less(coord, low), Nx.greater(coord, high))

      coord =
        case attr(node, "padding_mode", "zeros") do
          "zeros" -> coord
          "border" -> Nx.select(outside, Nx.clip(coord, 0, size - 1), coord)
          "reflection" -> Nx.select(outside, grid_reflect(coord, bounds), coord)
        end

      {coord, size, bounds}
    end
  end

  # The pixels of x at integer coordinates, one per spatial axis. A pixel outside x is
  # zero, the border pixel, or the reflected pixel.
  defp grid_pixels(x, corner, padding, out) do
    [n, c | sizes] = Tuple.to_list(Nx.shape(x))

    {index, valid} =
      Enum.reduce(corner, {0, 1}, fn {i, size, bounds}, {index, valid} ->
        i =
          case padding do
            "zeros" -> i
            "border" -> Nx.clip(i, 0, size - 1)
            "reflection" -> i |> grid_reflect(bounds) |> Nx.as_type(:s32) |> Nx.clip(0, size - 1)
          end

        inside = Nx.logical_and(Nx.greater_equal(i, 0), Nx.less(i, size))
        i = i |> Nx.clip(0, size - 1) |> Nx.as_type(:s32)
        {Nx.add(Nx.multiply(index, size), i), Nx.logical_and(valid, inside)}
      end)

    index = index |> Nx.reshape({n, 1, :auto}) |> Nx.broadcast({n, c, Enum.product(out)})

    x
    |> Nx.reshape({n, c, Enum.product(sizes)})
    |> Nx.take_along_axis(index, axis: 2)
    |> Nx.reshape(List.to_tuple([n, c | out]))
    |> Nx.multiply(valid |> Nx.new_axis(1) |> Nx.as_type(Nx.type(x)))
  end

  # Reflects a coordinate into [low, high], back and forth, as GsReflect of ONNX Runtime.
  defp grid_reflect(coord, {low, high}) do
    range = high - low
    below = Nx.subtract(low, coord)
    above = Nx.subtract(coord, high)
    distance = Nx.select(Nx.less(coord, low), below, above)
    turns = distance |> Nx.divide(range) |> Nx.floor()
    rest = Nx.subtract(distance, Nx.multiply(turns, range))
    even = turns |> Nx.remainder(2) |> Nx.equal(0)
    from_low = Nx.add(low, rest)
    from_high = Nx.subtract(high, rest)

    reflected =
      Nx.select(
        Nx.less(coord, low),
        Nx.select(even, from_low, from_high),
        Nx.select(even, from_high, from_low)
      )

    Nx.select(Nx.logical_or(Nx.less(coord, low), Nx.greater(coord, high)), reflected, coord)
  end

  defp repeat_groups(t, channels) do
    groups = Nx.size(t)
    t |> Nx.new_axis(1) |> Nx.broadcast({groups, div(channels, groups)}) |> Nx.reshape({channels})
  end

  # An output position outside the input with tf_crop_and_resize has no taps and gets the
  # extrapolation value. As in ONNX Runtime, an axis with scale 1 stays as it is, unless
  # its roi crops it.
  defp resize_axis(x, node, axis, sizes = {in_size, out_size, scale, _region}) do
    ctm = attr(node, "coordinate_transformation_mode", "half_pixel")

    cond do
      scale == 1 and ctm != "tf_crop_and_resize" ->
        x

      periodic?(node, ctm, sizes) ->
        periodic_taps(x, node, axis, ctm, sizes)

      true ->
        taps =
          for i <- 0..(out_size - 1) do
            xo = resize_source(ctm, i, sizes)

            if ctm == "tf_crop_and_resize" and (xo < 0 or xo > in_size - 1),
              do: [],
              else: resize_taps(node, xo, in_size, scale)
          end

        weighted_taps(x, axis, taps, attr(node, "extrapolation_value", 0.0))
    end
  end

  # An upscale by an integer s with a source coordinate of i / s plus a constant reads the
  # same taps, shifted by j, for every output i = j * s + p of phase p. A tap outside the
  # input reads the border, as the edge-padded input gives it. Only the antialias and
  # exclude_outside modes weight the border taps in another way.
  defp periodic?(node, ctm, {in_size, out_size, scale, _region}) do
    ctm in ~w(asymmetric half_pixel pytorch_half_pixel tf_half_pixel_for_nn half_pixel_symmetric) and
      scale > 1 and scale == round(scale) and out_size == in_size * round(scale) and
      attr(node, "exclude_outside", 0) == 0 and attr(node, "antialias", 0) == 0
  end

  # The taps of each phase, as offsets from j, come from an output far enough from the
  # borders that no tap moves. The edge-padded input shifted by each offset, times the
  # weights of the phases, gives the s outputs of each input position.
  defp periodic_taps(x, node, axis, ctm, sizes = {in_size, _out_size, scale, _region}) do
    s = round(scale)
    j0 = 4

    phases =
      for p <- 0..(s - 1) do
        xo = resize_source(ctm, j0 * s + p, sizes)
        for {j, w} <- resize_taps(node, xo, 2 * j0 + 8, scale), do: {j - j0, w}
      end

    offsets = phases |> List.flatten() |> Enum.map(&elem(&1, 0))
    {lo, hi} = Enum.min_max(offsets)
    {before, after_} = {max(-lo, 0), max(hi, 0)}
    shape = Nx.shape(x)

    edge = fn count, index ->
      if count > 0,
        do: [
          x
          |> Nx.slice_along_axis(index, 1, axis: axis)
          |> Nx.broadcast(put_elem(shape, axis, count))
        ],
        else: []
    end

    padded = Nx.concatenate(edge.(before, 0) ++ [x] ++ edge.(after_, in_size - 1), axis: axis)

    for offset <- lo..hi do
      weights =
        for taps <- phases,
            do:
              taps
              |> Enum.filter(&(elem(&1, 0) == offset))
              |> Enum.map(&elem(&1, 1))
              |> Enum.sum()

      padded
      |> Nx.slice_along_axis(offset + before, in_size, axis: axis)
      |> Nx.new_axis(axis + 1)
      |> Nx.multiply(
        weights
        |> static_tensor(:f32)
        |> on_axis(Nx.rank(x) + 1, axis + 1)
        |> Nx.as_type(Nx.type(x))
      )
    end
    |> Enum.reduce(&Nx.add/2)
    |> Nx.reshape(put_elem(shape, axis, in_size * s))
  end

  defp weighted_taps(x, axis, taps, outside_value) do
    out_size = length(taps)
    width = taps |> Enum.map(&length/1) |> Enum.max()
    outside = Enum.map(taps, &if(&1 == [], do: 1, else: 0))
    taps = Enum.map(taps, &(&1 ++ List.duplicate({0, 0.0}, width - length(&1))))
    indices = taps |> Enum.map(fn row -> Enum.map(row, &elem(&1, 0)) end) |> static_tensor(:s64)
    weights = taps |> Enum.map(fn row -> Enum.map(row, &elem(&1, 1)) end) |> static_tensor(:f32)
    per_output = &(&1 |> on_axis(Nx.rank(x), axis) |> Nx.as_type(Nx.type(x)))

    sum =
      for k <- 0..(width - 1) do
        x
        |> Nx.take(indices |> Nx.slice_along_axis(k, 1, axis: 1) |> Nx.reshape({out_size}),
          axis: axis
        )
        |> Nx.multiply(weights |> Nx.slice_along_axis(k, 1, axis: 1) |> per_output.())
      end
      |> Enum.reduce(&Nx.add/2)

    if Enum.any?(outside, &(&1 == 1)),
      do: Nx.add(sum, Nx.multiply(per_output.(static_tensor(outside, :f32)), outside_value)),
      else: sum
  end

  defp resize_source("half_pixel", i, {_in, _out, scale, _region}), do: (i + 0.5) / scale - 0.5
  defp resize_source("pytorch_half_pixel", _i, {_in, 1, _scale, _region}), do: 0.0

  defp resize_source("pytorch_half_pixel", i, {_in, _out, scale, _region}),
    do: (i + 0.5) / scale - 0.5

  defp resize_source("align_corners", _i, {_in, 1, _scale, _region}), do: 0.0

  defp resize_source("align_corners", i, {in_size, out_size, _scale, _region}),
    do: i * (in_size - 1) / (out_size - 1)

  defp resize_source("asymmetric", i, {_in, _out, scale, _region}), do: i / scale

  defp resize_source("tf_half_pixel_for_nn", i, {_in, _out, scale, _region}),
    do: (i + 0.5) / scale

  defp resize_source("half_pixel_symmetric", i, {in_size, out_size, scale, _region}) do
    adjustment = out_size / (in_size * scale)
    in_size / 2 * (1 - adjustment) + (i + 0.5) / scale - 0.5
  end

  defp resize_source("tf_crop_and_resize", _i, {in_size, 1, _scale, {start, stop}}),
    do: 0.5 * (start + stop) * (in_size - 1)

  defp resize_source("tf_crop_and_resize", i, {in_size, out_size, _scale, {start, stop}}),
    do: start * (in_size - 1) + i * (stop - start) * (in_size - 1) / (out_size - 1)

  defp resize_source(mode, _i, _sizes),
    do: raise(ArgumentError, "Resize coordinate_transformation_mode #{inspect(mode)} is unknown")

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
      mode -> raise ArgumentError, "Resize nearest_mode #{inspect(mode)} is unknown"
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
