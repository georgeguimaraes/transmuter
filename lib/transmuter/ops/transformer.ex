defmodule Transmuter.Ops.Transformer do
  @moduledoc """
  Attention, linear attention, KV cache writes, rotary embedding and normalization. The same op type can be the ONNX standard op (opset 23) or the com.microsoft op from ORT exporters, so the domain decides the input layout.

  On EMLX (see `Transmuter.Ops.emlx?/1`), the normalizations and the attention core run as fused kernels of `EMLX.Fast`. Attention with a softcap keeps the Nx operations, because the kernel has no softcap.

  `AttentionCore` (domain `transmuter`) is the attention core that `Transmuter.Fusion` finds in a graph: `softmax(q · k * scale + mask) · v`, with the `MatMul` broadcasting of ONNX and the softmax along its `axis` attribute. Its `k` is the right operand of the first `MatMul`. On EMLX, it runs as one kernel when q, k and v are 4-D heads of the same batch, the softmax is along the last axis and the mask broadcasts to the scores.
  """

  import Transmuter.Ops,
    only: [
      along_axis: 3,
      attr: 2,
      attr: 3,
      broadcast_shape: 1,
      emlx?: 1,
      emlx_norm: 4,
      empty?: 1,
      ints!: 1,
      matmul: 2,
      normalize: 3,
      normalize_axis: 2,
      shape_of: 1,
      softmax_last: 1
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
    [mask, past_k, past_v] = pad(optional, 3)
    rank = Nx.rank(q)
    q = to_heads(q, attr(node, "q_num_heads"))
    k = to_heads(k, attr(node, "kv_num_heads"))
    v = to_heads(v, attr(node, "kv_num_heads"))
    k = if past_k, do: Nx.concatenate([past_k, k], axis: 2), else: k
    v = if past_v, do: Nx.concatenate([past_v, v], axis: 2), else: v
    {_b, _heads, s, head_size} = Nx.shape(q)
    t = Nx.axis_size(k, 2)

    scale = attr(node, "scale") || 1 / :math.sqrt(head_size)
    cap = attr(node, "softcap", 0.0)

    # ONNX Runtime aligns the causal mask to the last query, so past keys stay visible.
    hidden =
      if attr(node, "is_causal", 0) == 1,
        do:
          Nx.greater(
            Nx.iota({1, 1, 1, t}, axis: 3),
            Nx.add(Nx.iota({1, 1, s, 1}, axis: 2), t - s)
          )

    # As in ONNX Runtime, the softcap comes before the masks. The fourth output is the
    # scores after the matmul, the softcap, the masks or the softmax, by
    # qk_matmul_output_mode, in the stage order of ONNX Runtime.
    masks = &(&1 |> add_mask(mask) |> mask_where(hidden))
    raw = grouped_scores(q, k, scale)
    capped = softcap(raw, cap)
    scores = masks.(capped)
    probs = softmax_last(scores)

    out =
      if emlx?(q) and cap == 0,
        do: emlx_attention(q, k, v, scale, masks, [hidden, mask]),
        else: grouped_values(probs, v)

    out = if rank == 3, do: from_heads(out), else: out
    [out, k, v, Enum.at([raw, capped, scores, probs], attr(node, "qk_matmul_output_mode", 0))]
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
    [cos_cache, sin_cache, position_ids, attention_bias] = pad(optional, 4)
    heads = attr(node, "num_heads")
    kv_heads = attr(node, "kv_num_heads")
    {q, k, v} = split_packed_qkv(q, k, v, heads, kv_heads)
    {b, s, _} = Nx.shape(q)
    [total_len] = ints!(total_len)

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

    key_pos = Nx.iota({1, 1, 1, capacity}, axis: 3)
    query_pos = Nx.add(Nx.iota({1, 1, s, 1}, axis: 2), Nx.reshape(offsets, {b, 1, 1, 1}))

    hidden =
      Nx.logical_or(
        Nx.greater(key_pos, query_pos),
        outside_window(key_pos, query_pos, attr(node, "local_window_size", -1))
      )

    head_size = Nx.axis_size(q, 3)
    scale = nonzero(attr(node, "scale", 0.0)) || 1 / :math.sqrt(head_size)
    cap = attr(node, "softcap", 0.0)

    masks = &(&1 |> add_mask(attention_bias) |> mask_where(hidden))

    out =
      if emlx?(q) and cap == 0 do
        emlx_attention(q, present_k, present_v, scale, masks, [hidden, attention_bias])
      else
        q
        |> grouped_scores(present_k, scale)
        |> softcap(cap)
        |> masks.()
        |> softmax_last()
        |> grouped_values(present_v)
      end
      |> from_heads()

    [out, present_k, present_v]
  end

  # com.microsoft MultiHeadAttention: query {B, S, N*H}, key/value {B, L, N*H} or {B, N, L, H}.
  defp op(node = %{op_type: "MultiHeadAttention"}, [q, k, v | optional]) do
    [nil, key_padding_mask, attention_bias, past_k, past_v] = pad(optional, 5)
    heads = attr(node, "num_heads")
    q = to_heads(q, heads)
    k = if Nx.rank(k) == 4, do: k, else: to_heads(k, heads)
    v = if Nx.rank(v) == 4, do: v, else: to_heads(v, heads)
    k = if past_k, do: Nx.concatenate([past_k, k], axis: 2), else: k
    v = if past_v, do: Nx.concatenate([past_v, v], axis: 2), else: v
    {b, _, s, head_size} = Nx.shape(q)
    t = Nx.axis_size(k, 2)

    padding = padding_mask(key_padding_mask, b, t)

    causal =
      if attr(node, "unidirectional", 0) == 1,
        do:
          Nx.greater(
            Nx.iota({1, 1, 1, t}, axis: 3),
            Nx.add(Nx.iota({1, 1, s, 1}, axis: 2), t - s)
          ),
        else: Nx.tensor(0, type: :u8)

    scale = nonzero(attr(node, "scale", 0.0)) || 1 / :math.sqrt(head_size)
    hidden = Nx.logical_or(padding, causal)
    masks = &(&1 |> add_mask(attention_bias) |> mask_where(hidden))

    out =
      if emlx?(q) do
        emlx_attention(q, k, v, scale, masks, [hidden, attention_bias])
      else
        q
        |> scores(k, scale)
        |> masks.()
        |> softmax_last()
        |> Nx.dot([3], [0, 1], v, [2], [0, 1])
      end
      |> from_heads()

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
      |> Nx.reshape(List.to_tuple([batch | List.duplicate(1, rank - 1)]))

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
    [past, decay, beta] = pad(optional, 3)
    heads = attr(node, "q_num_heads")
    kv_heads = attr(node, "kv_num_heads")
    rule = attr(node, "update_rule", "gated_delta")
    q = f32_heads(query, heads)
    k = f32_heads(key, kv_heads)
    v = f32_heads(value, kv_heads)
    {b, _, t, d_k} = Nx.shape(q)
    scale = nonzero(attr(node, "scale", 0.0)) || 1 / :math.sqrt(d_k)
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
    [beta, bias] = pad(rest, 2)
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
    [mask, position_ids] = pad(rest, 2)
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

  # k has the head size before the keys, so the kernel gets its transpose.
  defp attention_core(node, [q, k, v | mask], opset) do
    mask = List.first(mask)
    scale = attr(node, "scale")

    if emlx?(q) and fused_core?(q, k, v, mask, attr(node, "axis")) do
      k = Nx.transpose(k, axes: [0, 1, 3, 2])
      masks = if mask, do: &Nx.add(&1, mask), else: & &1
      emlx_attention(q, k, v, scale, masks, [mask])
    else
      scores = matmul(q, k)
      scores = if scale == 1, do: scores, else: Nx.multiply(scores, scale)

      scores
      |> add_optional(mask)
      |> then(&along_axis(%{node | op_type: "Softmax"}, &1, opset))
      |> matmul(v)
    end
  end

  defp fused_core?(q, k, v, mask, axis) do
    with true <- Enum.all?([q, k, v], &(Nx.rank(&1) == 4 and Nx.type(&1) == Nx.type(q))),
         {b, heads, s, _head_size} <- Nx.shape(q),
         true <-
           Enum.all?([k, v], &(&1 |> Nx.shape() |> Tuple.to_list() |> Enum.take(2) == [b, heads])) do
      scores = {b, heads, s, Nx.axis_size(k, 3)}
      axis in [-1, 3] and (mask == nil or broadcast_shape([scores, Nx.shape(mask)]) == scores)
    end
  end

  # com.microsoft Attention from BERT exports: one input projected to Q, K and V.
  defp packed_attention(node, [x, weights, bias | rest]) do
    [mask, nil, attention_bias] = pad(rest, 3)
    heads = attr(node, "num_heads")
    {b, s, _} = Nx.shape(x)
    qkv = x |> Nx.dot([2], weights, [0]) |> add_optional(bias)
    sizes = attr(node, "qkv_hidden_sizes") || List.duplicate(div(Nx.axis_size(weights, 1), 3), 3)
    [q, k, v] = split(qkv, sizes, 2)
    [q, k, v] = Enum.map([q, k, v], &to_heads(&1, heads))
    head_size = Nx.axis_size(q, 3)
    filter = attr(node, "mask_filter_value", -10_000.0)

    causal =
      if attr(node, "unidirectional", 0) == 1,
        do: Nx.greater(Nx.iota({1, 1, 1, s}, axis: 3), Nx.iota({1, 1, s, 1}, axis: 2)),
        else: Nx.tensor(0, type: :u8)

    hidden = Nx.logical_or(causal, padding_mask(mask, b, s))
    scale = nonzero(attr(node, "scale", 0.0)) || 1 / :math.sqrt(head_size)

    masks =
      &(&1
        |> add_mask(attention_bias)
        |> Nx.add(Nx.multiply(Nx.as_type(hidden, Nx.type(&1)), filter)))

    out =
      if emlx?(q) do
        emlx_attention(q, k, v, scale, masks, [hidden, attention_bias])
      else
        q
        |> scores(k, scale)
        |> masks.()
        |> softmax_last()
        |> Nx.dot([3], [0, 1], v, [2], [0, 1])
      end
      |> from_heads()

    [out]
  end

  # true where a key is hidden. A 1-D mask holds key lengths, 2-D and 3-D masks hold 1 to keep.
  # Hidden keys from a mask of key lengths {B}, of keys {B, T}, or of queries and keys
  # {B, S, T}. A zero in the mask hides a key.
  defp padding_mask(nil, _b, _t), do: Nx.tensor(0, type: :u8)

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

  # Writes new {B, N, S, H} into a {B, N, capacity, H} buffer at a per-sequence offset.
  defp write_cache(past, new, offsets, capacity) do
    {b, heads, past_len, head_size} = shape_of(past)

    j = Nx.iota({1, 1, capacity, 1}, axis: 2)
    offsets = Nx.reshape(offsets, {b, 1, 1, 1})
    zero = Nx.tensor(0, type: Nx.type(new))

    # A separate present buffer gets each sequence's valid past and zeros after it.
    # A shared buffer (past as long as the capacity) is updated in place.
    base =
      if past_len < capacity do
        zeros = Nx.broadcast(zero, {b, heads, capacity - past_len, head_size})
        padded = [past, zeros] |> Enum.reject(&empty?/1) |> Nx.concatenate(axis: 2)
        Nx.select(Nx.broadcast(Nx.less(j, offsets), Nx.shape(padded)), padded, zero)
      else
        past
      end

    # Each sequence writes its new rows at its own offset.
    Enum.reduce(0..(b - 1)//1, base, fn i, acc ->
      row = Nx.slice_along_axis(new, i, 1, axis: 0)
      Nx.put_slice(acc, [i, 0, Nx.reshape(offsets[i], {}), 0], row)
    end)
  end

  defp outside_window(_key_pos, _query_pos, window) when window <= 0, do: Nx.tensor(0, type: :u8)

  defp outside_window(key_pos, query_pos, window),
    do: Nx.less_equal(key_pos, Nx.subtract(query_pos, window))

  defp split_packed_qkv(qkv, nil, nil, heads, kv_heads) do
    head_size = div(Nx.axis_size(qkv, 2), heads + 2 * kv_heads)
    List.to_tuple(split(qkv, [heads * head_size, kv_heads * head_size, kv_heads * head_size], 2))
  end

  defp split_packed_qkv(q, k, v, _heads, _kv_heads), do: {q, k, v}

  defp split(t, sizes, axis) do
    sizes
    |> Enum.map_reduce(0, fn size, start ->
      {Nx.slice_along_axis(t, start, size, axis: axis), start + size}
    end)
    |> elem(0)
  end

  defp scores(q, k, scale), do: q |> Nx.dot([3], [0, 1], k, [3], [0, 1]) |> Nx.multiply(scale)

  # Query head h uses key and value head div(h, group). Folding each group of query heads
  # into the rows of one batched dot avoids copying the keys and values for every head.
  defp grouped_scores(q, k, scale) do
    {b, heads, s, head_size} = Nx.shape(q)
    {_b, kv_heads, t, _head_size} = Nx.shape(k)

    q
    |> Nx.reshape({b, kv_heads, div(heads, kv_heads) * s, head_size})
    |> scores(k, scale)
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

  # A boolean mask keeps the true positions, a float mask is added to the scores.
  defp add_mask(scores, nil), do: scores

  defp add_mask(scores, mask) do
    if Nx.type(mask) == {:u, 8},
      do: mask_where(scores, Nx.equal(mask, 0)),
      else: Nx.add(scores, Nx.as_type(mask, Nx.type(scores)))
  end

  # The fused kernel adds the masks as one additive mask: masks applied to zeros of the
  # shape that the mask tensors and the key axis broadcast to.
  defp emlx_attention(q, k, v, scale, masks, mask_tensors) do
    shape =
      broadcast_shape([
        {1, 1, 1, Nx.axis_size(k, 2)} | for(m <- mask_tensors, m, do: Nx.shape(m))
      ])

    mask = Nx.tensor(0, type: Nx.type(q)) |> Nx.broadcast(shape) |> masks.()
    EMLX.Fast.scaled_dot_product_attention(q, k, v, scale, mask)
  end

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
      x32 = Nx.as_type(x, :f32)
      inv = x32 |> Nx.pow(2) |> Nx.mean(axes: axes, keep_axes: true) |> Nx.add(eps) |> Nx.rsqrt()
      x32 |> Nx.multiply(inv) |> Nx.as_type(Nx.type(x)) |> Nx.multiply(w)
    end
  end

  defp layer_norm(x, gamma, beta, eps) do
    if emlx?(x) do
      beta = beta || Nx.tensor(0, type: Nx.type(x))
      emlx_norm(&EMLX.Fast.layer_norm(&1, &2, &3, eps), x, [Nx.rank(x) - 1], [gamma, beta])
    else
      x
      |> normalize([-1], eps)
      |> Nx.multiply(gamma)
      |> add_optional(beta)
    end
  end

  defp norm_axes(x, axis) do
    axis = if axis < 0, do: axis + Nx.rank(x), else: axis
    Enum.to_list(axis..(Nx.rank(x) - 1))
  end

  defp add_optional(t, nil), do: t
  defp add_optional(t, other), do: Nx.add(t, other)

  defp nonzero(value) when value == 0, do: nil
  defp nonzero(value), do: value

  defp pad(list, n), do: Enum.take(list ++ List.duplicate(nil, n), n)
end
