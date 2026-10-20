defmodule Transmuter.Ops.Vision do
  @moduledoc """
  Spatial operators: sampling grids, crops, patch folding, region pooling, deformable convolution and image decoding. Sizes that decide output shapes must be known: static, or read while the model runs eagerly, as at a graph break of `Transmuter.compile/3`.
  """

  import Transmuter.Ops, only: [attr: 2, attr: 3, ints!: 1, known!: 2, static_tensor: 2]

  @compile {:no_warn_undefined, StbImage}

  @op_types ~w(AffineGrid CenterCropPad Col2Im RoiAlign MaxRoiPool DeformConv ImageDecoder
               NonMaxSuppression)

  @doc false
  def op_types, do: @op_types

  @doc false
  def apply(node, inputs, opset), do: node |> version_defaults(opset) |> op(inputs) |> List.wrap()

  # RoiAlign before opset 16 has no coordinate_transformation_mode and works as output_half_pixel.
  defp version_defaults(node = %{op_type: "RoiAlign"}, opset) when opset < 16 do
    if attr(node, "coordinate_transformation_mode") do
      node
    else
      mode = %Transmuter.Onnx.AttributeProto{
        name: "coordinate_transformation_mode",
        type: :STRING,
        s: "output_half_pixel"
      }

      %{node | attribute: [mode | node.attribute]}
    end
  end

  defp version_defaults(node, _opset), do: node

  # Sampling grid {N, H, W, 2} (or {N, D, H, W, 3}) of theta times the normalized
  # coordinates (x, y[, z], 1) of every output pixel.
  defp op(node = %{op_type: "AffineGrid"}, [theta, size]) do
    [n, _c | spatial] = ints!(size)
    align = attr(node, "align_corners", 0) == 1

    axes =
      Enum.map(spatial, fn dim -> for i <- 0..(dim - 1), do: grid_coordinate(i, dim, align) end)

    # Coordinates are listed fastest axis first: x for width, then y, then z.
    points =
      axes
      |> Enum.reduce([[]], fn values, acc -> for prefix <- acc, v <- values, do: prefix ++ [v] end)
      |> Enum.map(&(Enum.reverse(&1) ++ [1.0]))

    base = points |> static_tensor(:f32) |> Nx.as_type(Nx.type(theta))
    grid = Nx.dot(theta, [2], base, [1]) |> Nx.transpose(axes: [0, 2, 1])
    Nx.reshape(grid, List.to_tuple([n | spatial] ++ [length(spatial)]))
  end

  defp op(node = %{op_type: "CenterCropPad"}, [x, shape]) do
    targets = ints!(shape)
    rank = Nx.rank(x)

    axes =
      node
      |> attr("axes", Enum.to_list(0..(rank - 1)))
      |> Enum.map(&if(&1 < 0, do: &1 + rank, else: &1))

    Enum.zip(axes, targets)
    |> Enum.reduce(x, fn {axis, target}, acc ->
      dim = Nx.axis_size(acc, axis)
      start = div(abs(dim - target), 2)

      cond do
        target == dim ->
          acc

        target < dim ->
          Nx.slice_along_axis(acc, start, target, axis: axis)

        true ->
          config =
            List.replace_at(
              List.duplicate({0, 0, 0}, rank),
              axis,
              {start, target - dim - start, 0}
            )

          Nx.pad(acc, Nx.tensor(0, type: Nx.type(acc)), config)
      end
    end)
  end

  # Sums sliding blocks {N, C * prod(block), L} back into images {N, C, image...}.
  defp op(node = %{op_type: "Col2Im"}, [x, image_shape, block_shape]) do
    image = ints!(image_shape)
    block = ints!(block_shape)
    spatial = length(image)
    dilations = attr(node, "dilations", List.duplicate(1, spatial))
    strides = attr(node, "strides", List.duplicate(1, spatial))
    pads = attr(node, "pads", List.duplicate(0, 2 * spatial))
    {lo, hi} = Enum.split(pads, spatial)
    padded = Enum.zip_with([image, lo, hi], fn [d, l, h] -> d + l + h end)

    counts =
      Enum.zip_with([padded, block, dilations, strides], fn [p, k, d, s] ->
        div(p - d * (k - 1) - 1, s) + 1
      end)

    {n, cols, l} = Nx.shape(x)
    c = div(cols, Enum.product(block))
    ^l = Enum.product(counts)

    # Target pixel (in the padded image) of every (kernel offset, block position) pair.
    targets =
      for k <- product_indices(block), b <- product_indices(counts) do
        [k, b, strides, dilations, padded]
        |> Enum.zip_with(fn [ki, bi, s, d, _] -> bi * s + ki * d end)
        |> flat_index(padded)
      end

    values = x |> Nx.reshape({n * c, :auto}) |> Nx.transpose()
    sums = Nx.broadcast(Nx.tensor(0, type: Nx.type(x)), {Enum.product(padded), n * c})
    sums = Nx.indexed_add(sums, targets |> static_tensor(:s64) |> Nx.new_axis(1), values)

    sums
    |> Nx.transpose()
    |> Nx.reshape(List.to_tuple([n, c | padded]))
    |> crop(lo, image)
  end

  # Bilinear region-of-interest pooling, as ONNX Runtime and torchvision do it.
  defp op(node = %{op_type: "RoiAlign"}, [x, rois, batch_indices]) do
    {out_h, out_w} = {attr(node, "output_height", 1), attr(node, "output_width", 1)}
    ratio = attr(node, "sampling_ratio", 0)
    scale = attr(node, "spatial_scale", 1.0)
    half_pixel? = attr(node, "coordinate_transformation_mode", "half_pixel") == "half_pixel"
    max? = attr(node, "mode", "avg") == "max"

    # With an adaptive sampling ratio the grid size depends on each ROI, so the ROIs must be known.
    if ratio == 0, do: known!(rois, "the RoiAlign ROIs with sampling_ratio 0")

    count = Nx.axis_size(rois, 0)
    known = if ratio == 0, do: Nx.to_list(rois)

    for r <- 0..(count - 1) do
      roi = Nx.slice_along_axis(rois, r, 1, axis: 0) |> Nx.reshape({4})

      image =
        Nx.take(x, Nx.slice_along_axis(batch_indices, r, 1, axis: 0) |> Nx.reshape({}), axis: 0)

      grid = sampling_grid(known && Enum.at(known, r), {out_h, out_w}, ratio, scale, half_pixel?)
      roi_align_one(image, roi, {out_h, out_w}, grid, scale, half_pixel?, max?)
    end
    |> Nx.stack()
  end

  # Max pooling over bins of each ROI; rois are {batch, x1, y1, x2, y2} in input pixels.
  defp op(node = %{op_type: "MaxRoiPool"}, [x, rois]) do
    [ph, pw] = attr(node, "pooled_shape")
    scale = attr(node, "spatial_scale", 1.0)
    {_n, _c, h, w} = Nx.shape(x)

    coords =
      rois
      |> Nx.slice_along_axis(1, 4, axis: 1)
      |> Nx.multiply(scale)
      |> Nx.round()
      |> Nx.as_type(:s32)

    image =
      Nx.take(
        x,
        rois |> Nx.slice_along_axis(0, 1, axis: 1) |> Nx.reshape({:auto}) |> Nx.as_type(:s32),
        axis: 0
      )

    col = &Nx.slice_along_axis(coords, &1, 1, axis: 1)
    {x1, y1, x2, y2} = {col.(0), col.(1), col.(2), col.(3)}

    bins = fn start, stop, bins_count, size ->
      length = Nx.max(Nx.add(Nx.subtract(stop, start), 1), 1) |> Nx.as_type(:f32)
      i = Nx.iota({1, bins_count}, type: :f32)

      bin_start =
        i
        |> Nx.multiply(Nx.divide(length, bins_count))
        |> Nx.floor()
        |> Nx.add(start)
        |> Nx.clip(0, size)

      bin_end =
        i
        |> Nx.add(1)
        |> Nx.multiply(Nx.divide(length, bins_count))
        |> Nx.ceil()
        |> Nx.add(start)
        |> Nx.clip(0, size)

      {bin_start, bin_end}
    end

    {hs, he} = bins.(y1, y2, ph, h)
    {ws, we} = bins.(x1, x2, pw, w)
    rows = Nx.iota({1, 1, h}, type: :f32)
    cols = Nx.iota({1, 1, w}, type: :f32)

    in_rows =
      Nx.logical_and(
        Nx.greater_equal(rows, Nx.new_axis(hs, 2)),
        Nx.less(rows, Nx.new_axis(he, 2))
      )

    in_cols =
      Nx.logical_and(
        Nx.greater_equal(cols, Nx.new_axis(ws, 2)),
        Nx.less(cols, Nx.new_axis(we, 2))
      )

    # {R, ph, pw, H, W}, broadcast against the image {R, C, 1, 1, H, W}.
    mask =
      Nx.logical_and(
        Nx.reshape(in_rows, {:auto, ph, 1, h, 1}),
        Nx.reshape(in_cols, {:auto, 1, pw, 1, w})
      )

    neg = Nx.Constants.min_finite(Nx.type(x))
    {r, c, _, _} = Nx.shape(image)
    full = {r, c, ph, pw, h, w}
    on = Nx.broadcast(Nx.new_axis(mask, 1), full)
    values = Nx.select(on, Nx.broadcast(image |> Nx.new_axis(2) |> Nx.new_axis(2), full), neg)
    pooled = Nx.reduce_max(values, axes: [4, 5])
    empty = mask |> Nx.any(axes: [3, 4]) |> Nx.logical_not() |> Nx.new_axis(1)
    Nx.select(Nx.broadcast(empty, {r, c, ph, pw}), Nx.tensor(0, type: Nx.type(x)), pooled)
  end

  defp op(node = %{op_type: "DeformConv"}, [x, w, offset | rest]) do
    [bias, mask] = Enum.take(rest ++ [nil, nil], 2)
    {n, c, h, wd} = Nx.shape(x)
    {m, c_per_group, kh, kw} = Nx.shape(w)
    groups = attr(node, "group", 1)
    offset_groups = attr(node, "offset_group", 1)
    [sh, sw] = attr(node, "strides", [1, 1])
    [dh, dw] = attr(node, "dilations", [1, 1])
    [pt, pl, _pb, _pr] = attr(node, "pads", [0, 0, 0, 0])
    {_, _, oh, ow} = Nx.shape(offset)
    taps = kh * kw

    # Base sampling position of every tap at every output pixel, {taps, oh, ow}.
    ky = Nx.iota({kh, kw, 1, 1}, axis: 0) |> Nx.multiply(dh) |> Nx.reshape({taps, 1, 1})
    kx = Nx.iota({kh, kw, 1, 1}, axis: 1) |> Nx.multiply(dw) |> Nx.reshape({taps, 1, 1})
    base_y = Nx.add(ky, Nx.iota({1, oh, 1}, axis: 1) |> Nx.multiply(sh) |> Nx.subtract(pt))
    base_x = Nx.add(kx, Nx.iota({1, 1, ow}, axis: 2) |> Nx.multiply(sw) |> Nx.subtract(pl))

    # offset is {N, offset_groups * taps * 2, oh, ow} with (dy, dx) pairs per tap.
    pairs = Nx.reshape(offset, {n, offset_groups, taps, 2, oh, ow})
    y = Nx.add(base_y, Nx.squeeze(Nx.slice_along_axis(pairs, 0, 1, axis: 3), axes: [3]))
    xs = Nx.add(base_x, Nx.squeeze(Nx.slice_along_axis(pairs, 1, 1, axis: 3), axes: [3]))

    # Sample each offset group's channels at that group's positions: {N, C, taps, oh, ow}.
    channels_per_offset_group = div(c, offset_groups)
    image = Nx.reshape(x, {n, offset_groups, channels_per_offset_group, h * wd})

    columns =
      bilinear(
        image,
        h,
        wd,
        Nx.reshape(y, {n, offset_groups, 1, taps * oh * ow}),
        Nx.reshape(xs, {n, offset_groups, 1, taps * oh * ow})
      )
      |> Nx.reshape({n, offset_groups, channels_per_offset_group, taps, oh, ow})

    columns =
      if mask,
        do: Nx.multiply(columns, Nx.reshape(mask, {n, offset_groups, 1, taps, oh, ow})),
        else: columns

    # Groups lead, because Nx batches dot products on leading axes.
    columns =
      columns
      |> Nx.reshape({n, groups, c_per_group, taps, oh, ow})
      |> Nx.transpose(axes: [1, 0, 2, 3, 4, 5])

    kernel = Nx.reshape(w, {groups, div(m, groups), c_per_group, taps})
    # {groups, N, oh, ow, M / groups} -> {N, M, oh, ow}
    out = Nx.dot(columns, [2, 3], [0], kernel, [2, 3], [0])
    out = out |> Nx.transpose(axes: [1, 0, 4, 2, 3]) |> Nx.reshape({n, m, oh, ow})
    if bias, do: Nx.add(out, Nx.reshape(bias, {1, m, 1, 1})), else: out
  end

  # Decodes to {H, W, C} u8. An image that cannot be decoded gives an empty tensor, as the spec says.
  defp op(node = %{op_type: "ImageDecoder"}, [encoded]) do
    unless Code.ensure_loaded?(StbImage) do
      raise ArgumentError,
            ~s(ImageDecoder needs the optional stb_image dependency, add {:stb_image, "~> 1.0"} to your deps)
    end

    known!(encoded, "the ImageDecoder input")

    # stb_image 1.0 returns the wrong shape when asked to convert channels, so the
    # source channels (gray, gray and alpha, RGB or RGBA) become RGB here.
    case StbImage.read_binary(Nx.to_binary(encoded)) do
      {:ok, %{data: data, shape: shape}} ->
        rgb =
          data
          |> Nx.from_binary(:u8, backend: Nx.BinaryBackend)
          |> Nx.reshape(shape)
          |> Transmuter.Featurizer.rgb()

        case attr(node, "pixel_format", "RGB") do
          "RGB" -> rgb
          "BGR" -> Nx.reverse(rgb, axes: [2])
          "Grayscale" -> luma(rgb)
        end

      {:error, _reason} ->
        {:empty, {0}}
    end
  end

  # Greedy per batch and class, as the ONNX Runtime CPU kernel does it. The number of
  # selected boxes depends on the values, so the boxes and scores must be known.
  defp op(node = %{op_type: "NonMaxSuppression"}, [boxes, scores | rest]) do
    [max_boxes, iou, score_threshold] = Enum.take(rest ++ [nil, nil, nil], 3)

    for value <- [boxes, scores | rest],
        value != nil,
        do: known!(value, "the NonMaxSuppression input")

    limit = if max_boxes, do: max(hd(ints!(max_boxes)), 0), else: 0
    iou = if iou, do: hd(Nx.to_flat_list(iou)), else: 0.0
    score_threshold = score_threshold && hd(Nx.to_flat_list(score_threshold))
    center? = attr(node, "center_point_box", 0) == 1
    {_batches, classes, count} = Nx.shape(scores)

    corners =
      boxes
      |> Nx.to_flat_list()
      |> Enum.chunk_every(4)
      |> Enum.map(&box_corners(&1, center?))
      |> Enum.chunk_every(count)
      |> Enum.map(&List.to_tuple/1)

    selected =
      scores
      |> Nx.to_flat_list()
      |> Enum.chunk_every(count)
      |> Enum.with_index()
      |> Enum.flat_map(fn {class_scores, row} ->
        batch = div(row, classes)
        batch_boxes = Enum.at(corners, batch)

        class_scores
        |> Enum.with_index()
        |> Enum.filter(fn {s, _} -> score_threshold == nil or s > score_threshold end)
        |> Enum.sort_by(fn {s, i} -> {-s, i} end)
        |> Enum.reduce_while([], fn {_, i}, kept ->
          cond do
            length(kept) == limit ->
              {:halt, kept}

            Enum.any?(kept, &suppress?(elem(batch_boxes, i), elem(batch_boxes, &1), iou)) ->
              {:cont, kept}

            true ->
              {:cont, [i | kept]}
          end
        end)
        |> Enum.reverse()
        |> Enum.map(&[batch, rem(row, classes), &1])
      end)

    if selected == [], do: {:empty, {0, 3}}, else: static_tensor(selected, :s64)
  end

  # Corners and area in f32 arithmetic, so IoU ties fall the same way as in ONNX Runtime.
  defp box_corners([y1, x1, y2, x2], false),
    do: corners_area(min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))

  defp box_corners([xc, yc, w, h], true) do
    {half_w, half_h} = {f32(w / 2), f32(h / 2)}
    corners_area(f32(xc - half_w), f32(yc - half_h), f32(xc + half_w), f32(yc + half_h))
  end

  defp corners_area(x_min, y_min, x_max, y_max),
    do: {x_min, y_min, x_max, y_max, f32(f32(x_max - x_min) * f32(y_max - y_min))}

  defp suppress?({ax0, ay0, ax1, ay1, area_a}, {bx0, by0, bx1, by1, area_b}, threshold) do
    {x0, x1} = {max(ax0, bx0), min(ax1, bx1)}
    {y0, y1} = {max(ay0, by0), min(ay1, by1)}

    if x1 <= x0 or y1 <= y0 or area_a <= 0 or area_b <= 0 do
      false
    else
      inter = f32(f32(x1 - x0) * f32(y1 - y0))
      union = f32(f32(area_a + area_b) - inter)
      union > 0 and f32(inter / union) > threshold
    end
  end

  defp f32(x) do
    <<y::float-32>> = <<x::float-32>>
    y
  end

  # Samples per bin: the fixed sampling ratio, or ceil(bin size) from the known ROI.
  defp sampling_grid(nil, _out, ratio, _scale, _half_pixel?), do: {ratio, ratio}

  defp sampling_grid([x1, y1, x2, y2], {out_h, out_w}, _ratio, scale, half_pixel?) do
    min_size = if half_pixel?, do: 0.0, else: 1.0
    roi_h = max((y2 - y1) * scale, min_size)
    roi_w = max((x2 - x1) * scale, min_size)
    {max(ceil(roi_h / out_h), 1), max(ceil(roi_w / out_w), 1)}
  end

  defp roi_align_one(image, roi, {out_h, out_w}, {grid_h, grid_w}, scale, half_pixel?, max?) do
    {_c, h, w} = Nx.shape(image)
    offset = if half_pixel?, do: 0.5, else: 0.0
    roi = roi |> Nx.multiply(scale) |> Nx.subtract(offset)

    [x1, y1, x2, y2] =
      for i <- 0..3, do: roi |> Nx.slice_along_axis(i, 1, axis: 0) |> Nx.reshape({})

    min_size = if half_pixel?, do: 0.0, else: 1.0
    roi_w = Nx.max(Nx.subtract(x2, x1), min_size)
    roi_h = Nx.max(Nx.subtract(y2, y1), min_size)
    bin_h = Nx.divide(roi_h, out_h)
    bin_w = Nx.divide(roi_w, out_w)

    # Sample positions {out_h, grid_h} and {out_w, grid_w}.
    ys =
      Nx.iota({out_h, 1})
      |> Nx.multiply(bin_h)
      |> Nx.add(y1)
      |> Nx.add(Nx.multiply(Nx.add(Nx.iota({1, grid_h}), 0.5), Nx.divide(bin_h, grid_h)))

    xs =
      Nx.iota({out_w, 1})
      |> Nx.multiply(bin_w)
      |> Nx.add(x1)
      |> Nx.add(Nx.multiply(Nx.add(Nx.iota({1, grid_w}), 0.5), Nx.divide(bin_w, grid_w)))

    {y_lo, y_hi, wy, y_ok} = roi_axis_taps(ys, h)
    {x_lo, x_hi, wx, x_ok} = roi_axis_taps(xs, w)

    flat = Nx.reshape(image, {:auto, h * w})

    at = fn yi, xi ->
      Nx.take(
        flat,
        Nx.add(
          Nx.multiply(Nx.reshape(yi, {out_h, grid_h, 1, 1}), w),
          Nx.reshape(xi, {1, 1, out_w, grid_w})
        ),
        axis: 1
      )
    end

    wy = Nx.reshape(wy, {1, out_h, grid_h, 1, 1})
    wx = Nx.reshape(wx, {1, 1, 1, out_w, grid_w})

    {channels, _, _} = Nx.shape(image)

    ok =
      Nx.logical_and(
        Nx.reshape(y_ok, {1, out_h, grid_h, 1, 1}),
        Nx.reshape(x_ok, {1, 1, 1, out_w, grid_w})
      )
      |> Nx.broadcast({channels, out_h, grid_h, out_w, grid_w})

    corners = [
      Nx.multiply(at.(y_lo, x_lo), Nx.multiply(Nx.subtract(1, wy), Nx.subtract(1, wx))),
      Nx.multiply(at.(y_lo, x_hi), Nx.multiply(Nx.subtract(1, wy), wx)),
      Nx.multiply(at.(y_hi, x_lo), Nx.multiply(wy, Nx.subtract(1, wx))),
      Nx.multiply(at.(y_hi, x_hi), Nx.multiply(wy, wx))
    ]

    zero = Nx.tensor(0, type: Nx.type(image))

    if max? do
      # ONNX Runtime takes the largest weighted corner, which is the bilinear value for one sample per bin.
      corners
      |> Enum.map(&Nx.select(ok, &1, zero))
      |> Enum.reduce(&Nx.max/2)
      |> Nx.reduce_max(axes: [2, 4])
    else
      corners
      |> Enum.reduce(&Nx.add/2)
      |> then(&Nx.select(ok, &1, zero))
      |> Nx.sum(axes: [2, 4])
      |> Nx.divide(grid_h * grid_w)
    end
  end

  # Low and high neighbors, the high weight, and validity of each sample position along one axis.
  defp roi_axis_taps(pos, size) do
    ok = Nx.logical_and(Nx.greater_equal(pos, -1.0), Nx.less_equal(pos, size))
    pos = Nx.max(pos, 0.0)
    low = pos |> Nx.floor() |> Nx.min(size - 1)
    at_edge = Nx.greater_equal(Nx.floor(pos), size - 1)
    high = Nx.select(at_edge, low, Nx.add(low, 1))
    pos = Nx.select(at_edge, low, pos)
    {Nx.as_type(low, :s32), Nx.as_type(high, :s32), Nx.subtract(pos, low), ok}
  end

  # Samples image {N, G, C, H * W} at positions y, x {N, G, 1, P}; outside corners are zero.
  defp bilinear(image, h, w, y, x) do
    y0 = Nx.floor(y)
    x0 = Nx.floor(x)
    wy = Nx.subtract(y, y0)
    wx = Nx.subtract(x, x0)

    inside =
      Nx.logical_and(
        Nx.logical_and(Nx.greater(y, -1), Nx.less(y, h)),
        Nx.logical_and(Nx.greater(x, -1), Nx.less(x, w))
      )

    {n, g, c, _} = Nx.shape(image)
    {_, _, _, p} = Nx.shape(y)

    sample = fn yi, xi, weight ->
      valid =
        Nx.logical_and(
          Nx.logical_and(Nx.greater_equal(yi, 0), Nx.less(yi, h)),
          Nx.logical_and(Nx.greater_equal(xi, 0), Nx.less(xi, w))
        )

      index =
        Nx.add(Nx.multiply(Nx.clip(yi, 0, h - 1), w), Nx.clip(xi, 0, w - 1))
        |> Nx.as_type(:s32)
        |> Nx.broadcast({n, g, c, p})

      image
      |> Nx.take_along_axis(index, axis: 3)
      |> Nx.multiply(Nx.multiply(weight, Nx.as_type(valid, Nx.type(image))))
    end

    [
      sample.(y0, x0, Nx.multiply(Nx.subtract(1, wy), Nx.subtract(1, wx))),
      sample.(y0, Nx.add(x0, 1), Nx.multiply(Nx.subtract(1, wy), wx)),
      sample.(Nx.add(y0, 1), x0, Nx.multiply(wy, Nx.subtract(1, wx))),
      sample.(Nx.add(y0, 1), Nx.add(x0, 1), Nx.multiply(wy, wx))
    ]
    |> Enum.reduce(&Nx.add/2)
    |> Nx.multiply(Nx.as_type(inside, Nx.type(image)))
  end

  # ITU-R 601-2 luma in 16-bit fixed point, rounded, as Pillow converts RGB to L.
  defp luma(rgb) do
    rgb
    |> Nx.as_type(:u32)
    |> Nx.dot(static_tensor([19_595, 38_470, 7_471], :u32))
    |> Nx.add(32_768)
    |> Nx.right_shift(16)
    |> Nx.as_type(:u8)
    |> Nx.new_axis(-1)
  end

  # Normalized coordinate of pixel i of dim pixels, in [-1, 1].
  defp grid_coordinate(_i, 1, true), do: -1.0
  defp grid_coordinate(i, dim, true), do: -1 + 2 * i / (dim - 1)
  defp grid_coordinate(i, dim, false), do: -1 + (2 * i + 1) / dim

  defp crop(t, starts, sizes) do
    starts
    |> Enum.zip(sizes)
    |> Enum.with_index(2)
    |> Enum.reduce(t, fn {{start, size}, axis}, acc ->
      Nx.slice_along_axis(acc, start, size, axis: axis)
    end)
  end

  defp product_indices(dims) do
    Enum.reduce(Enum.reverse(dims), [[]], fn d, acc ->
      for i <- 0..(d - 1), rest <- acc, do: [i | rest]
    end)
  end

  defp flat_index(coords, dims) do
    coords |> Enum.zip(dims) |> Enum.reduce(0, fn {i, d}, acc -> acc * d + i end)
  end
end
