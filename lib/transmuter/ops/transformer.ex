defmodule Transmuter.Ops.Transformer do
  @moduledoc """
  Attention, linear attention, KV cache writes, rotary embedding and normalization. The same op type can be the ONNX standard op (opset 23) or the com.microsoft op from ORT exporters, so the domain decides the input layout.

  On EMLX (see `Transmuter.Ops.emlx?/1`), the normalizations and the attention core run as fused kernels of `EMLX.Fast`. The kernels take f16 and bf16 values as they are and accumulate in f32. Attention with a softcap keeps the Nx operations, because the kernel has no softcap. The Nx operations compute f16 and bf16 values in f32 (see `Transmuter.Ops.widen/2`).

  All attention ops share one attention core with the scale, the softcap, the masks and the grouping of key and value heads. When the causal mask is the only mask, it runs as the causal kernel, which builds no mask. Other masks become one additive mask for the kernel.

  `AttentionCore` (domain `transmuter`) is the attention core that `Transmuter.Fusion` finds in a graph: `softmax(where(keep, q · k * scale / divisor, fill) + mask) · v`, with the `MatMul` broadcasting of ONNX and the softmax along its `axis` attribute. Its inputs are q, k, v and the optional mask, keep, divisor and the two shapes of a `Reshape` pair of the probabilities. Its `k` is the right operand of the first `MatMul`. On EMLX, it runs as one kernel when q, k and v are 4-D heads of the same batch, or 3-D with the heads in the batch, the softmax is along the last axis, the masks broadcast to the scores, the host can read the divisor and the `Reshape` pair gives back the shape of the probabilities.
  """

  import Transmuter.Ops,
    only: [
      add_optional: 2,
      along_axis: 3,
      attr: 2,
      attr: 3,
      broadcast_shape: 1,
      emlx?: 1,
      emlx_norm: 4,
      empty?: 1,
      known!: 2,
      known?: 1,
      matmul: 2,
      normalize: 3,
      normalize_axis: 2,
      on_axis: 3,
      optional: 2,
      scalar!: 1,
      shape_of: 1,
      softmax_last: 1,
      split_along: 3,
      widen: 2
    ]

  @op_types ~w(Attention AttentionCore GroupQueryAttention MultiHeadAttention RotaryEmbedding RMSNormalization
               SimplifiedLayerNormalization SkipSimplifiedLayerNormalization SkipLayerNormalization
               EmbedLayerNormalization TensorScatter LinearAttention)

  @doc false
  def op_types, do: @op_types

  @doc false
  def apply(node = %{op_type: "AttentionCore"}, inputs, opset),
    do: [attention_core(node, inputs, opset)]

  def apply(node, inputs, _opset), do: node |> op(inputs) |> List.wrap()

  defp op(node = %{op_type: "Attention", domain: "com.microsoft"}, inputs),
    do: packed_attention(node, inputs)

  defp op(node = %{op_type: "Attention"}, [q, k, v | optional]) do
    [mask, past_k, past_v] = optional(optional, 3)
    rank = Nx.rank(q)
    q = to_heads(q, attr(node, "q_num_heads"))
    k = to_heads(k, attr(node, "kv_num_heads"))
    v = to_heads(v, attr(node, "kv_num_heads"))
    k = if past_k, do: Nx.concatenate([past_k, k], axis: 2), else: k
    v = if past_v, do: Nx.concatenate([past_v, v], axis: 2), else: v

    opts = %{
      scale: scale(node, q),
      softcap: attr(node, "softcap", 0.0),
      bias: mask,
      hidden: nil,
      causal: attr(node, "is_causal", 0) == 1
    }

    out = attend(q, k, v, opts)
    out = if rank == 3, do: from_heads(out), else: out

    # The fourth output is one stage of the scores, by qk_matmul_output_mode.
    qk =
      if Enum.at(node.output, 3, "") != "" do
        [q, k, mask]
        |> widen(fn [q, k, mask] ->
          [scores(q, k, %{opts | bias: mask}, attr(node, "qk_matmul_output_mode", 0))]
        end)
        |> hd()
      end

    [out, k, v, qk]
  end

  # GroupQueryAttention with a KV buffer per sequence. seqlens_k holds each sequence's
  # total length minus one, so new keys are written at seqlens_k + 1 - seq_len. When the
  # buffer is as long as total_sequence_length it is a static cache, and every decode
  # step keeps the same shapes.
  defp op(node = %{op_type: "GroupQueryAttention"}, [
         q,
         k,
         v,
         past_k,
         past_v,
         seqlens_k,
         total_len | optional
       ]) do
    [cos_cache, sin_cache, position_ids, attention_bias] = optional(optional, 4)
    heads = attr(node, "num_heads")
    kv_heads = attr(node, "kv_num_heads")
    {q, k, v} = split_packed_qkv(q, k, v, heads, kv_heads)
    {b, s, _} = Nx.shape(q)
    total_len = scalar!(total_len)

    q = to_heads(q, heads)
    k = to_heads(k, kv_heads)
    v = to_heads(v, kv_heads)
    offsets = seqlens_k |> Nx.as_type(:s32) |> Nx.reshape({b}) |> Nx.add(1 - s)

    {q, k} =
      if attr(node, "do_rotary", 0) == 1 do
        positions =
          if position_ids,
            do: Nx.reshape(position_ids, {b, s}),
            else: Nx.add(Nx.iota({b, s}, axis: 1), Nx.new_axis(offsets, 1))

        {cos, sin} = {Nx.take(cos_cache, positions), Nx.take(sin_cache, positions)}
        interleaved = attr(node, "rotary_interleaved", 0)
        {rotate(q, cos, sin, interleaved), rotate(k, cos, sin, interleaved)}
      else
        {q, k}
      end

    capacity = max(elem(shape_of(past_k), 2), total_len)
    present_k = write_cache(past_k, k, offsets, capacity)
    present_v = write_cache(past_v, v, offsets, capacity)

    # The query at row i of a sequence has the position offset + i.
    offsets = Nx.reshape(offsets, {b, 1, 1, 1})
    window = attr(node, "local_window_size", -1)

    opts = %{
      scale: scale(node, q),
      softcap: attr(node, "softcap", 0.0),
      bias: attention_bias,
      hidden:
        any_hidden([
          causal_hidden(s, capacity, offsets),
          window > 0 &&
            Nx.less_equal(key_positions(capacity), Nx.add(query_positions(s, offsets), -window))
        ]),
      causal: false
    }

    out = q |> attend(present_k, present_v, opts) |> from_heads()
    [out, present_k, present_v]
  end

  # com.microsoft MultiHeadAttention: query {B, S, N*H}, key/value {B, L, N*H} or {B, N, L, H}.
  defp op(node = %{op_type: "MultiHeadAttention"}, [q, k, v | optional]) do
    [bias, key_padding_mask, attention_bias, past_k, past_v] = optional(optional, 5)
    heads = attr(node, "num_heads")
    {q, k, v} = add_qkv_bias(q, k, v, bias)
    q = to_heads(q, heads)
    k = if Nx.rank(k) == 4, do: k, else: to_heads(k, heads)
    v = if Nx.rank(v) == 4, do: v, else: to_heads(v, heads)
    k = if past_k, do: Nx.concatenate([past_k, k], axis: 2), else: k
    v = if past_v, do: Nx.concatenate([past_v, v], axis: 2), else: v

    opts = %{
      scale: scale(node, q),
      softcap: 0.0,
      bias: attention_bias,
      hidden: padding_mask(key_padding_mask, Nx.axis_size(q, 0), Nx.axis_size(k, 2)),
      causal: attr(node, "unidirectional", 0) == 1
    }

    out = q |> attend(k, v, opts) |> from_heads()
    [out, k, v]
  end

  # Writes update into the cache along the sequence axis from each batch's write index.
  # Circular mode wraps the positions around the cache length.
  defp op(node = %{op_type: "TensorScatter"}, [cache, update | write_indices]) do
    rank = Nx.rank(cache)
    axis = normalize_axis(attr(node, "axis", -2), rank)
    shape = Nx.shape(update)
    batch = elem(shape, 0)

    start =
      (List.first(write_indices) || Nx.broadcast(Nx.tensor(0, type: :s64), {batch}))
      |> Nx.as_type(:s64)
      |> on_axis(rank, 0)

    positions = Nx.add(Nx.iota(shape, axis: axis, type: :s64), start)

    positions =
      if attr(node, "mode", "linear") == "circular",
        do: Nx.remainder(positions, Nx.axis_size(cache, axis)),
        else: positions

    coords =
      for a <- 0..(rank - 1) do
        if a == axis, do: positions, else: Nx.iota(shape, axis: a, type: :s64)
      end
      |> Nx.stack(axis: -1)
      |> Nx.reshape({:auto, rank})

    Nx.indexed_put(cache, coords, Nx.reshape(update, {:auto}))
  end

  # The recurrence runs token by token in f32 on a {B, H_kv, d_k, d_v} state. Gated rules
  # decay the state by exp(decay), delta rules add beta * (v - S^T k) in place of v.
  defp op(node = %{op_type: "LinearAttention"}, [query, key, value | optional]) do
    [past, decay, beta] = optional(optional, 3)
    heads = attr(node, "q_num_heads")
    kv_heads = attr(node, "kv_num_heads")
    rule = attr(node, "update_rule", "gated_delta")
    q = f32_heads(query, heads)
    k = f32_heads(key, kv_heads)
    v = f32_heads(value, kv_heads)
    {b, _, t, d_k} = Nx.shape(q)
    scale = scale(node, q)
    decay = rule in ["gated", "gated_delta"] && f32_heads(decay, kv_heads)
    beta = rule in ["delta", "gated_delta"] && f32_heads(beta, Nx.axis_size(beta, 2))

    state =
      if past,
        do: Nx.as_type(past, :f32),
        else: Nx.broadcast(Nx.tensor(0.0, type: :f32), {b, kv_heads, d_k, Nx.axis_size(v, 3)})

    at = fn x, i -> x |> Nx.slice_along_axis(i, 1, axis: 2) |> Nx.squeeze(axes: [2]) end

    {outputs, state} =
      Enum.map_reduce(0..(t - 1), state, fn i, state ->
        state =
          if decay,
            do: Nx.multiply(state, decay |> at.(i) |> Nx.exp() |> Nx.new_axis(-1)),
            else: state

        k_i = at.(k, i)
        v_i = at.(v, i)

        v_i =
          if beta do
            retrieved = Nx.dot(state, [2], [0, 1], k_i, [2], [0, 1])
            Nx.multiply(at.(beta, i), Nx.subtract(v_i, retrieved))
          else
            v_i
          end

        state = Nx.add(state, Nx.multiply(Nx.new_axis(k_i, -1), Nx.new_axis(v_i, -2)))
        out = q |> at.(i) |> Nx.dot([2], [0, 1], repeat_kv(state, heads), [2], [0, 1])
        {Nx.multiply(out, scale), state}
      end)

    out = outputs |> Nx.stack(axis: 2) |> from_heads() |> Nx.as_type(Nx.type(query))
    [out, Nx.as_type(state, Nx.type(past || query))]
  end

  defp op(node = %{op_type: "RotaryEmbedding", domain: "com.microsoft"}, [
         x,
         position_ids,
         cos_cache,
         sin_cache
       ]),
       do: rotary_embedding(node, x, cos_cache, sin_cache, position_ids)

  defp op(node = %{op_type: "RotaryEmbedding"}, [x, cos_cache, sin_cache | position_ids]),
    do: rotary_embedding(node, x, cos_cache, sin_cache, List.first(position_ids))

  defp op(node = %{op_type: norm}, [x, w])
       when norm in ["RMSNormalization", "SimplifiedLayerNormalization"],
       do: rms_norm(x, w, attr(node, "epsilon", 1.0e-5), attr(node, "axis", -1))

  defp op(node = %{op_type: "SkipSimplifiedLayerNormalization"}, [x, skip, w | bias]) do
    sum = add_optional(Nx.add(x, skip), List.first(bias))
    [rms_norm(sum, w, attr(node, "epsilon", 1.0e-5), -1), nil, nil, sum]
  end

  defp op(node = %{op_type: "SkipLayerNormalization"}, [x, skip, gamma | rest]) do
    [beta, bias] = optional(rest, 2)
    sum = add_optional(Nx.add(x, skip), bias)
    [layer_norm(sum, gamma, beta, attr(node, "epsilon", 1.0e-12)), nil, nil, sum]
  end

  defp op(node = %{op_type: "EmbedLayerNormalization"}, [
         ids,
         segments,
         words,
         positions,
         segment_table,
         gamma,
         beta | rest
       ]) do
    [mask, position_ids] = optional(rest, 2)
    {b, s} = Nx.shape(ids)
    position_ids = position_ids || Nx.iota({b, s}, axis: 1)

    sum =
      words
      |> Nx.take(ids)
      |> Nx.add(Nx.take(positions, position_ids))
      |> add_optional(segments && Nx.take(segment_table, segments))

    mask_index =
      if mask,
        do: mask |> Nx.sum(axes: [1]) |> Nx.as_type(:s32),
        else: Nx.broadcast(Nx.tensor(0, type: :s32), {b})

    [layer_norm(sum, gamma, beta, attr(node, "epsilon", 1.0e-12)), mask_index, sum]
  end

  # k has the head size before the keys. 4-D heads with the softmax over the keys take the
  # shared attention core with the transpose of k. Exports such as Whisper and TrOCR put the
  # heads in the batch, {B * N, S, d}, which are the heads of a batch of one. Other shapes
  # keep the MatMul broadcasting.
  # A divisor that the host can read becomes part of the scale. The keys where keep is 0 get
  # the fill, at most -1.0e4, which the shared core replaces with the smallest finite value.
  defp attention_core(node, [q, k, v | optional], opset) do
    [mask, keep, divisor, split, merge] = optional(optional, 5)
    scale = attr(node, "scale")

    {scale, divisor} =
      case divisor && known?(divisor) && Nx.size(divisor) == 1 &&
             divisor |> known!("the divisor") |> Nx.reshape({}) |> Nx.to_number() do
        d when is_number(d) and d != 0 -> {scale / d, nil}
        _other -> {scale, divisor}
      end

    batched? = Enum.all?([q, k, v], &(Nx.rank(&1) == 3))
    [q4, k4, v4] = if batched?, do: Enum.map([q, k, v], &Nx.new_axis(&1, 0)), else: [q, k, v]
    last? = attr(node, "axis") in [-1, Nx.rank(q) - 1]

    if divisor == nil and last? and fused_core?(q4, k4, v4, [mask, keep]) and
         same_shape?(scores_shape(q, k), [split, merge]) do
      hidden = keep && Nx.equal(keep, 0)
      opts = %{scale: scale, softcap: 0.0, bias: mask, hidden: hidden, causal: false}
      out = attend(q4, Nx.transpose(k4, axes: [0, 1, 3, 2]), v4, opts)
      if batched?, do: Nx.squeeze(out, axes: [0]), else: out
    else
      plain_core(node, [q, k, v, mask, keep, divisor, split, merge], scale, opset)
    end
  end

  # The scores q · k of a core whose q and k have the same batch axes.
  defp scores_shape(q, k), do: put_elem(Nx.shape(q), Nx.rank(q) - 1, Nx.axis_size(k, -1))

  # Whether the reshapes with these ONNX shapes give back a value of this shape.
  defp same_shape?(shape, shapes) do
    shapes = Enum.reject(shapes, &is_nil/1)

    with true <- Enum.all?(shapes, &known?/1) do
      steps = Enum.scan(shapes, shape, &reshaped(&2, &1))

      Enum.all?(steps, &(Tuple.product(&1) == Tuple.product(shape))) and
        List.last(steps, shape) == shape
    end
  end

  # The shape of an ONNX Reshape of a value of `shape`: 0 copies the axis and -1 takes the
  # rest of the size.
  defp reshaped(shape, target) do
    dims =
      target
      |> Transmuter.Ops.values!()
      |> Enum.with_index(fn
        0, i -> elem(shape, i)
        d, _i -> d
      end)

    rest = div(Tuple.product(shape), dims |> Enum.reject(&(&1 == -1)) |> Enum.product())
    dims |> Enum.map(&if(&1 == -1, do: rest, else: &1)) |> List.to_tuple()
  end

  defp plain_core(node, [q, k, v, mask, keep, divisor, split, merge], scale, opset) do
    [q, k, v, mask, divisor]
    |> widen(fn [q, k, v, mask, divisor] ->
      scores = matmul(q, k)

      scores =
        cond do
          divisor != nil -> Nx.divide(scores, divisor)
          scale == 1 -> scores
          true -> Nx.multiply(scores, scale)
        end

      scores
      |> where_kept(keep, attr(node, "fill"))
      |> add_optional(mask)
      |> then(&along_axis(%{node | op_type: "Softmax"}, &1, opset))
      |> reshape_optional(split)
      |> reshape_optional(merge)
      |> matmul(v)
      |> List.wrap()
    end)
    |> hd()
  end

  defp reshape_optional(x, nil), do: x
  defp reshape_optional(x, shape), do: Nx.reshape(x, reshaped(Nx.shape(x), shape))

  defp where_kept(scores, nil, _fill), do: scores

  defp where_kept(scores, keep, fill) do
    shape = broadcast_shape([Nx.shape(scores), Nx.shape(keep)])
    Nx.select(Nx.broadcast(keep, shape), Nx.broadcast(scores, shape), fill)
  end

  defp fused_core?(q, k, v, masks) do
    with true <- Enum.all?([q, k, v], &(Nx.rank(&1) == 4 and Nx.type(&1) == Nx.type(q))),
         {b, heads, s, _head_size} <- Nx.shape(q),
         true <-
           Enum.all?([k, v], &(&1 |> Nx.shape() |> Tuple.to_list() |> Enum.take(2) == [b, heads])) do
      scores = {b, heads, s, Nx.axis_size(k, 3)}
      Enum.all?(masks, &(&1 == nil or broadcast_shape([scores, Nx.shape(&1)]) == scores))
    end
  end

  # com.microsoft Attention from BERT exports: one input projected to Q, K and V.
  defp packed_attention(node, [x, weights, bias | rest]) do
    [mask, past, attention_bias] = optional(rest, 3)

    if past do
      raise ArgumentError, "Attention (com.microsoft) with the past input is not supported"
    end

    heads = attr(node, "num_heads")
    {b, s, _} = Nx.shape(x)
    qkv = x |> Nx.dot([2], weights, [0]) |> add_optional(bias)
    sizes = attr(node, "qkv_hidden_sizes") || List.duplicate(div(Nx.axis_size(weights, 1), 3), 3)
    [q, k, v] = split_along(qkv, sizes, 2)
    [q, k, v] = Enum.map([q, k, v], &to_heads(&1, heads))
    # ONNX Runtime adds mask_filter_value to the scores of the hidden keys.
    hidden =
      any_hidden([
        attr(node, "unidirectional", 0) == 1 && causal_hidden(s, s, 0),
        padding_mask(mask, b, s)
      ])

    filter =
      hidden &&
        Nx.multiply(Nx.as_type(hidden, Nx.type(q)), attr(node, "mask_filter_value", -10_000.0))

    opts = %{
      scale: scale(node, q),
      softcap: 0.0,
      bias: any_bias([filter, attention_bias]),
      hidden: nil,
      causal: false
    }

    out = q |> attend(k, v, opts) |> from_heads()
    [out]
  end

  # Hidden keys from a mask of key lengths {B}, of keys {B, T}, or of queries and keys
  # {B, S, T}. A zero in the mask hides a key.
  defp padding_mask(nil, _b, _t), do: nil

  defp padding_mask(mask, b, s) do
    case Nx.rank(mask) do
      1 -> Nx.greater_equal(Nx.iota({1, 1, 1, s}, axis: 3), Nx.reshape(mask, {b, 1, 1, 1}))
      2 -> mask |> Nx.equal(0) |> Nx.reshape({b, 1, 1, :auto})
      3 -> mask |> Nx.equal(0) |> Nx.new_axis(1)
    end
  end

  defp rotary_embedding(node, x, cos_cache, sin_cache, position_ids) do
    half = Nx.axis_size(cos_cache, -1)
    {x4, restore} = rotary_heads(x, attr(node, "num_heads", 0), 2 * half)
    {b, _, s, _} = Nx.shape(x4)

    {cos, sin} =
      cond do
        position_ids == nil ->
          {cos_cache, sin_cache}

        Nx.size(position_ids) == 1 and b * s > 1 ->
          positions = Nx.add(Nx.iota({b, s}, axis: 1), Nx.reshape(position_ids, {}))
          {Nx.take(cos_cache, positions), Nx.take(sin_cache, positions)}

        true ->
          positions = Nx.broadcast(position_ids, {b, s})
          {Nx.take(cos_cache, positions), Nx.take(sin_cache, positions)}
      end

    x4 |> rotate(cos, sin, attr(node, "interleaved", 0)) |> restore.()
  end

  # Rotary ops take {B, N, S, H} or {B, S, N * H}. Returns the 4-D view and the way back.
  defp rotary_heads(x, num_heads, rotary_dim) do
    case Nx.shape(x) do
      {_, _, _, _} ->
        {x, & &1}

      {b, s, hidden} ->
        heads = if num_heads > 0, do: num_heads, else: div(hidden, rotary_dim)

        {to_heads(x, heads),
         &(&1 |> Nx.transpose(axes: [0, 2, 1, 3]) |> Nx.reshape({b, s, hidden}))}
    end
  end

  # Rotates the first 2 * half features of x {B, N, S, H} with cos and sin {B, S, half}.
  defp rotate(x, cos, sin, interleaved) do
    {b, n, s, head_size} = Nx.shape(x)
    half = Nx.axis_size(cos, -1)
    rot = 2 * half
    cos = Nx.new_axis(cos, 1)
    sin = Nx.new_axis(sin, 1)
    xr = Nx.slice_along_axis(x, 0, rot, axis: 3)

    rotated =
      if interleaved == 1 do
        pairs = Nx.reshape(xr, {b, n, s, half, 2})
        x1 = Nx.squeeze(Nx.slice_along_axis(pairs, 0, 1, axis: 4), axes: [4])
        x2 = Nx.squeeze(Nx.slice_along_axis(pairs, 1, 1, axis: 4), axes: [4])

        [
          Nx.subtract(Nx.multiply(x1, cos), Nx.multiply(x2, sin)),
          Nx.add(Nx.multiply(x2, cos), Nx.multiply(x1, sin))
        ]
        |> Nx.stack(axis: 4)
        |> Nx.reshape({b, n, s, rot})
      else
        x1 = Nx.slice_along_axis(xr, 0, half, axis: 3)
        x2 = Nx.slice_along_axis(xr, half, half, axis: 3)

        Nx.concatenate(
          [
            Nx.subtract(Nx.multiply(x1, cos), Nx.multiply(x2, sin)),
            Nx.add(Nx.multiply(x2, cos), Nx.multiply(x1, sin))
          ],
          axis: 3
        )
      end

    if rot == head_size,
      do: rotated,
      else:
        Nx.concatenate([rotated, Nx.slice_along_axis(x, rot, head_size - rot, axis: 3)], axis: 3)
  end

  # Writes new {B, N, S, H} into a {B, N, capacity, H} buffer at the offset of each
  # sequence. Row j of a sequence keeps the past before the offset, takes the new row
  # j - offset, and is zero after the new rows, as in ONNX Runtime. One select writes all
  # sequences. A negative offset writes at 0, as the first prompt of ONNX Runtime does.
  # Generation gives buffers with room for the new rows, so none falls after the buffer.
  defp write_cache(past, new, offsets, capacity) do
    {b, heads, past_len, head_size} = shape_of(past)
    s = Nx.axis_size(new, 2)
    shape = {b, heads, capacity, head_size}
    zero = Nx.tensor(0, type: Nx.type(new))

    position =
      Nx.subtract(
        Nx.iota({1, 1, capacity, 1}, axis: 2),
        Nx.reshape(Nx.max(offsets, 0), {b, 1, 1, 1})
      )

    past =
      cond do
        empty?(past) ->
          Nx.broadcast(zero, shape)

        past_len < capacity ->
          Nx.pad(past, zero, [{0, 0, 0}, {0, 0, 0}, {0, capacity - past_len, 0}, {0, 0, 0}])

        true ->
          past
      end

    # A step of one token broadcasts its row, a longer step gathers row j - offset.
    rows =
      if s == 1,
        do: Nx.broadcast(new, shape),
        else: Nx.take_along_axis(new, Nx.broadcast(Nx.clip(position, 0, s - 1), shape), axis: 2)

    written = Nx.select(Nx.broadcast(Nx.less(position, s), shape), rows, zero)
    Nx.select(Nx.broadcast(Nx.less(position, 0), shape), past, written)
  end

  defp split_packed_qkv(qkv, nil, nil, heads, kv_heads) do
    head_size = div(Nx.axis_size(qkv, 2), heads + 2 * kv_heads)

    List.to_tuple(
      split_along(qkv, [heads * head_size, kv_heads * head_size, kv_heads * head_size], 2)
    )
  end

  defp split_packed_qkv(q, k, v, _heads, _kv_heads), do: {q, k, v}

  # The bias of MultiHeadAttention holds the biases of q, k and v, in this order. v can
  # have another hidden size than q and k.
  defp add_qkv_bias(q, k, v, nil), do: {q, k, v}

  defp add_qkv_bias(q, k, v, bias) do
    unless Nx.rank(k) == 3 and Nx.rank(v) == 3 do
      raise ArgumentError, "MultiHeadAttention with a bias needs key and value of rank 3"
    end

    hidden = Nx.axis_size(q, -1)
    [qb, kb, vb] = split_along(bias, [hidden, hidden, Nx.size(bias) - 2 * hidden], 0)
    {Nx.add(q, qb), Nx.add(k, kb), Nx.add(v, vb)}
  end

  # The attention core of the attention ops: softmax(q · k * scale + masks) · v, for q
  # {B, N, S, H} and k and v {B, N_kv, T, H}. The options are:
  #
  #   * :scale and :softcap (0 for none). As in ONNX Runtime, the softcap comes before the
  #     masks.
  #   * :bias - nil, a float mask to add, or a u8 mask with 1 for the keys to keep.
  #   * :hidden - nil, or a u8 mask with 1 for the hidden keys.
  #   * :causal - whether query i sees only the keys up to T - S + i. ONNX Runtime aligns the
  #     causal mask to the last query, so past keys stay visible.
  #
  # The masks broadcast to {B, N, S, T}. On EMLX without softcap, one kernel computes it: the
  # causal kernel when the causal mask is the only mask, or else the kernel with one additive
  # mask in the type of q.
  defp attend(q, k, v, opts) do
    if emlx?(q) and opts.softcap == 0 do
      type = Nx.type(q)
      fused_attention(q, Nx.as_type(k, type), Nx.as_type(v, type), opts)
    else
      [q, k, v, opts.bias]
      |> widen(fn [q, k, v, bias] ->
        [q |> scores(k, %{opts | bias: bias}, 3) |> grouped_values(v)]
      end)
      |> hd()
    end
  end

  defp fused_attention(q, k, v, opts = %{bias: nil, hidden: nil}) do
    if opts.causal,
      do: EMLX.Fast.scaled_dot_product_attention_causal(q, k, v, opts.scale),
      else: EMLX.Fast.scaled_dot_product_attention(q, k, v, opts.scale)
  end

  # The kernel takes one additive mask of rank 4 in the type of q. It is built from the masks
  # without a tensor of zeros, which an eager run would allocate on the default backend.
  defp fused_attention(q, k, v, opts) do
    {s, t} = {Nx.axis_size(q, 2), Nx.axis_size(k, 2)}
    type = Nx.type(q)
    {bias, hidden} = masks(opts, s, t)
    bias = bias && Nx.as_type(bias, type)
    shape = broadcast_shape([{1, 1, 1, t} | for(m <- [bias, hidden], m, do: Nx.shape(m))])

    mask =
      if hidden,
        do: Nx.select(Nx.broadcast(hidden, shape), Nx.Constants.min_finite(type), bias || 0),
        else: Nx.broadcast(bias, shape)

    EMLX.Fast.scaled_dot_product_attention(q, k, v, opts.scale, mask)
  end

  # The scores after each stage, in the order of ONNX Runtime: 0 after the matmul, 1 after
  # the softcap, 2 after the masks and 3 after the softmax.
  defp scores(q, k, opts, stage) do
    {s, t} = {Nx.axis_size(q, 2), Nx.axis_size(k, 2)}

    [&softcap(&1, opts.softcap), &mask_scores(&1, opts, s, t), &softmax_last/1]
    |> Enum.take(stage)
    |> Enum.reduce(grouped_scores(q, k, opts.scale), & &1.(&2))
  end

  # A float bias adds to the scores, and the hidden keys get the smallest finite value.
  defp mask_scores(scores, opts, s, t) do
    {bias, hidden} = masks(opts, s, t)

    scores
    |> add_optional(bias && Nx.as_type(bias, Nx.type(scores)))
    |> mask_where(hidden)
  end

  # The float bias and the u8 mask of hidden keys that opts give, or nil.
  defp masks(opts, s, t) do
    keep? = opts.bias != nil and Nx.type(opts.bias) == {:u, 8}

    hidden =
      any_hidden([
        opts.hidden,
        opts.causal && causal_hidden(s, t, t - s),
        keep? && Nx.equal(opts.bias, 0)
      ])

    {if(not keep?, do: opts.bias), hidden}
  end

  # True where key j is after the query at row i, which has the position offset + i.
  defp causal_hidden(s, t, offset),
    do: Nx.greater(key_positions(t), query_positions(s, offset))

  defp key_positions(t), do: Nx.iota({1, 1, 1, t}, axis: 3)
  defp query_positions(s, offset), do: Nx.add(Nx.iota({1, 1, s, 1}, axis: 2), offset)

  defp any_hidden(masks) do
    case Enum.filter(masks, & &1) do
      [] -> nil
      masks -> Enum.reduce(masks, &Nx.logical_or/2)
    end
  end

  defp any_bias(masks) do
    case Enum.filter(masks, & &1) do
      [] -> nil
      masks -> Enum.reduce(masks, &Nx.add/2)
    end
  end

  # Query head h uses key and value head div(h, group). Folding each group of query heads
  # into the rows of one batched dot avoids copying the keys and values for every head.
  defp grouped_scores(q, k, scale) do
    {b, heads, s, head_size} = Nx.shape(q)
    {_b, kv_heads, t, _head_size} = Nx.shape(k)

    q
    |> Nx.reshape({b, kv_heads, div(heads, kv_heads) * s, head_size})
    |> Nx.dot([3], [0, 1], k, [3], [0, 1])
    |> Nx.multiply(scale)
    |> Nx.reshape({b, heads, s, t})
  end

  defp grouped_values(probs, v) do
    {b, heads, s, t} = Nx.shape(probs)
    {_b, kv_heads, _t, head_size} = Nx.shape(v)

    probs
    |> Nx.reshape({b, kv_heads, div(heads, kv_heads) * s, t})
    |> Nx.dot([3], [0, 1], v, [2], [0, 1])
    |> Nx.reshape({b, heads, s, head_size})
  end

  defp softcap(scores, cap) when cap > 0,
    do: scores |> Nx.divide(cap) |> Nx.tanh() |> Nx.multiply(cap)

  defp softcap(scores, _cap), do: scores

  defp mask_where(scores, nil), do: scores

  defp mask_where(scores, hidden),
    do:
      Nx.select(
        Nx.broadcast(hidden, Nx.shape(scores)),
        Nx.Constants.min_finite(Nx.type(scores)),
        scores
      )

  defp to_heads(t, nil), do: t

  defp to_heads(t, heads) do
    case Nx.shape(t) do
      {_, _, _, _} -> t
      {b, s, _} -> t |> Nx.reshape({b, s, heads, :auto}) |> Nx.transpose(axes: [0, 2, 1, 3])
    end
  end

  defp f32_heads(t, heads), do: t |> to_heads(heads) |> Nx.as_type(:f32)

  defp from_heads(t) do
    {b, heads, s, head_size} = Nx.shape(t)
    t |> Nx.transpose(axes: [0, 2, 1, 3]) |> Nx.reshape({b, s, heads * head_size})
  end

  defp repeat_kv(kv, heads) do
    {b, kv_heads, t, head_size} = Nx.shape(kv)

    if kv_heads == heads,
      do: kv,
      else:
        kv
        |> Nx.new_axis(2)
        |> Nx.broadcast({b, kv_heads, div(heads, kv_heads), t, head_size})
        |> Nx.reshape({b, heads, t, head_size})
  end

  defp rms_norm(x, w, eps, axis) do
    axes = norm_axes(x, axis)

    if emlx?(x) do
      emlx_norm(&EMLX.Fast.rms_norm(&1, &2, eps), x, axes, [w])
    else
      [x, w]
      |> widen(fn [x, w] ->
        inv = x |> Nx.pow(2) |> Nx.mean(axes: axes, keep_axes: true) |> Nx.add(eps) |> Nx.rsqrt()
        [x |> Nx.multiply(inv) |> Nx.multiply(w)]
      end)
      |> hd()
    end
  end

  defp layer_norm(x, gamma, beta, eps) do
    if emlx?(x) do
      beta = beta || Nx.tensor(0, type: Nx.type(x))
      emlx_norm(&EMLX.Fast.layer_norm(&1, &2, &3, eps), x, [Nx.rank(x) - 1], [gamma, beta])
    else
      [x, gamma, beta]
      |> widen(fn [x, gamma, beta] ->
        [x |> normalize([-1], eps) |> Nx.multiply(gamma) |> add_optional(beta)]
      end)
      |> hd()
    end
  end

  defp norm_axes(x, axis) do
    Enum.to_list(normalize_axis(axis, Nx.rank(x))..(Nx.rank(x) - 1))
  end

  # The scale attribute, where 0 or no attribute means 1 / sqrt(head size) of q {B, N, S, H}.
  defp scale(node, q) do
    case attr(node, "scale", 0.0) do
      scale when scale == 0 -> 1 / :math.sqrt(Nx.axis_size(q, 3))
      scale -> scale
    end
  end
end
