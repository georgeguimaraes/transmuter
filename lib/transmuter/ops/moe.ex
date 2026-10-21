defmodule Transmuter.Ops.MoE do
  @moduledoc """
  The ONNX Runtime mixture-of-experts operators `MoE` and `QMoE` (domain com.microsoft). Each token selects its top-k experts, runs the MLP of each expert, and adds the outputs with the routing weights.

  `MoE` takes a softmax over all experts, then uses the top-k probabilities as weights. `QMoE` takes the top-k router values, then a softmax over those k values. `QMoE` dequantizes 4-bit and 8-bit block-wise weights as (q - zero point) * scale.
  """

  import Transmuter.Ops, only: [attr: 2, attr: 3, optional: 2, softmax_last: 1]

  @op_types ~w(MoE QMoE)

  @doc false
  def op_types, do: @op_types

  @doc false
  def apply(node, inputs), do: node |> op(inputs) |> List.wrap()

  # com.microsoft MoE: float expert weights, full-softmax routing.
  defp op(node = %{op_type: "MoE"}, [input, router_probs | rest]) do
    [fc1_w, fc1_bias, fc2_w, fc2_bias, fc3_w, fc3_bias] = optional(rest, 6)
    {k, normalize, _act, _fusion, _alpha, _beta, _limit} = moe_attrs(node)

    # Full softmax over all experts, then top-k selection.
    probs = router_probs |> Nx.as_type(:f32) |> softmax_last()
    {top_vals, top_idx} = topk(probs, k)
    weights = if normalize == 1, do: local_normalize(top_vals), else: top_vals
    route = route(weights, top_idx, Nx.axis_size(fc1_w, 0))
    w = &(&1 && select(&1, route))

    experts(node, input, route, [
      {w.(fc1_w), fc1_bias},
      {w.(fc2_w), fc2_bias},
      {w.(fc3_w), fc3_bias}
    ])
  end

  # com.microsoft QMoE: block-wise quantized expert weights, local-softmax routing.
  defp op(node = %{op_type: "QMoE"}, [input, router_probs | rest]) do
    [fc1_w, fc1_scales, fc1_bias, fc2_w, fc2_scales, fc2_bias | opt] = optional(rest, 12)
    [fc3_w, fc3_scales, fc3_bias, fc1_zp, fc2_zp, fc3_zp] = opt

    {k, _normalize, _act, _fusion, _alpha, _beta, _limit} = moe_attrs(node)
    bits = attr(node, "expert_weight_bits", 4)
    block_size = attr(node, "block_size", 0)
    type = Nx.type(input)

    # Local softmax: top-k selection then softmax over those k values, so the weights of a
    # token already add up to 1 and normalize_routing_weights changes nothing.
    {top_vals, top_idx} = topk(Nx.as_type(router_probs, :f32), k)
    max_v = Nx.reduce_max(top_vals, axes: [1], keep_axes: true)
    exp_v = Nx.exp(Nx.subtract(top_vals, max_v))
    weights = Nx.divide(exp_v, Nx.sum(exp_v, axes: [1], keep_axes: true))
    route = route(weights, top_idx, Nx.axis_size(fc1_w, 0))
    pick = &(&1 && select(&1, route))

    w = fn w, s, z ->
      w && dequant(pick.(w), pick.(s), pick.(z), bits, block_size, type)
    end

    experts(node, input, route, [
      {w.(fc1_w, fc1_scales, fc1_zp), fc1_bias},
      {w.(fc2_w, fc2_scales, fc2_zp), fc2_bias},
      {w.(fc3_w, fc3_scales, fc3_zp), fc3_bias}
    ])
  end

  # The expert MLPs of the routed rows, added with the routing weights of each token.
  defp experts(node, input, route, [{fc1_w, fc1_bias}, {fc2_w, fc2_bias}, {fc3_w, fc3_bias}]) do
    {_k, _normalize, act, fusion, alpha, beta, limit} = moe_attrs(node)
    type = Nx.type(input)
    {t, input_2d} = flatten_input(input)
    h = Nx.axis_size(input_2d, 1)
    rows = input_2d |> Nx.as_type(type) |> Nx.broadcast(route.rows.(t, h), axes: route.axes)
    rows = Nx.reshape(rows, {:auto, h})
    h1 = rows |> expert_dot(fc1_w, route) |> add_bias(fc1_bias, route.experts)

    activated =
      if act == "swiglu" and fusion == 0 and fc3_w do
        h3 = rows |> expert_dot(fc3_w, route) |> add_bias(fc3_bias, route.experts)
        swiglu_gate(h1, h3, alpha, beta, limit)
      else
        activate(h1, act, fusion, alpha, beta, limit)
      end

    out = activated |> expert_dot(fc2_w, route) |> add_bias(fc2_bias, route.experts)
    weights = route.weights |> Nx.reshape({:auto, 1}) |> Nx.as_type(type)

    # A row with weight 0 adds nothing, also when its expert gives an infinity.
    result =
      weights
      |> Nx.equal(0)
      |> Nx.broadcast(Nx.shape(out))
      |> Nx.select(0, Nx.multiply(Nx.as_type(out, type), weights))
      |> Nx.reshape(route.rows.(t, h))
      |> Nx.sum(axes: [route.slot_axis])

    Nx.reshape(result, Nx.shape(input))
  end

  # Each token runs its k experts, so the rows are T * k, token by token, and the weights of
  # their experts are gathered. When T * k is at least the number of experts, every expert
  # runs on every token instead, with weight 0 for the tokens that do not select it. The
  # rows are then E * T, expert by expert, and a batched product reads each weight once.
  defp route(weights, top_idx, experts) do
    {t, k} = Nx.shape(top_idx)

    if t * k >= experts do
      dense =
        top_idx
        |> Nx.new_axis(1)
        |> Nx.equal(Nx.iota({1, experts, 1}))
        |> Nx.multiply(Nx.new_axis(weights, 1))
        |> Nx.sum(axes: [2])
        |> Nx.transpose()

      %{
        dense?: true,
        rows: &{experts, &1, &2},
        axes: [1, 2],
        slot_axis: 0,
        experts: Nx.iota({experts, t}, axis: 0) |> Nx.reshape({:auto}),
        weights: dense
      }
    else
      %{
        dense?: false,
        rows: &{&1, k, &2},
        axes: [0, 2],
        slot_axis: 1,
        experts: Nx.reshape(top_idx, {:auto}),
        weights: weights
      }
    end
  end

  defp select(w, %{dense?: true}), do: w
  defp select(w, %{experts: ids}), do: Nx.take(w, ids)

  # rows {R, K} times the weight {N, K} of the expert of each row: w is {E, N, K} for all
  # experts with E * T rows, or {R, N, K} with the weight of each row.
  defp expert_dot(rows, w, %{dense?: true}) do
    {r, size} = Nx.shape(rows)
    experts = Nx.axis_size(w, 0)

    rows
    |> Nx.reshape({experts, div(r, experts), size})
    |> Nx.dot([2], [0], w, [2], [0])
    |> Nx.reshape({r, :auto})
  end

  defp expert_dot(rows, w, _route), do: Nx.dot(rows, [1], [0], w, [2], [0])

  # --- helpers ---

  defp moe_attrs(node) do
    k = attr(node, "k")
    normalize = attr(node, "normalize_routing_weights", 0)
    act = attr(node, "activation_type", "relu")
    fusion = attr(node, "swiglu_fusion", 0)
    # FLOAT attributes (type 1 in proto); default to 1.0 / 0.0.
    alpha = attr(node, "activation_alpha", 1.0)
    beta = attr(node, "activation_beta", 0.0)
    limit = nonzero_float(attr(node, "swiglu_limit", 0.0))
    {k, normalize, act, fusion, alpha, beta, limit}
  end

  # Returns {T, input_2d}: token count and a 2D view.
  defp flatten_input(input) do
    case Nx.rank(input) do
      2 ->
        {t, _h} = Nx.shape(input)
        {t, input}

      3 ->
        {b, s, h} = Nx.shape(input)
        {b * s, Nx.reshape(input, {b * s, h})}
    end
  end

  # Top-k along axis 1, stable sort (lower index wins on ties, as ORT does).
  defp topk(probs, k) do
    idx =
      probs
      |> Nx.argsort(axis: 1, direction: :desc, stable: true)
      |> Nx.slice_along_axis(0, k, axis: 1)

    {Nx.take_along_axis(probs, idx, axis: 1), idx}
  end

  # Normalizes each row by its sum (used when normalize_routing_weights=1 for MoE).
  defp local_normalize(weights) do
    sum = Nx.sum(weights, axes: [1], keep_axes: true)
    safe = Nx.select(Nx.equal(sum, 0), 1, sum)
    Nx.divide(weights, safe)
  end

  defp add_bias(h, nil, _flat_idx), do: h
  defp add_bias(h, bias, flat_idx), do: Nx.add(h, Nx.take(bias, flat_idx))

  defp activate(h, "relu", _, _alpha, _beta, _limit), do: Nx.max(h, 0)
  defp activate(h, "gelu", _, _alpha, _beta, _limit), do: gelu(h)
  defp activate(h, "silu", _, _alpha, _beta, _limit), do: Nx.multiply(h, Nx.sigmoid(h))
  defp activate(h, "identity", _, _alpha, _beta, _limit), do: h
  defp activate(h, "swiglu", 1, alpha, beta, limit), do: swiglu_interleaved(h, alpha, beta, limit)
  defp activate(h, "swiglu", 2, alpha, beta, limit), do: swiglu_concat(h, alpha, beta, limit)
  # fusion=0 without a separate fc3: treat fc1 as producing interleaved g and l.
  defp activate(h, "swiglu", _, alpha, beta, limit), do: swiglu_interleaved(h, alpha, beta, limit)

  # SwiGLU from two separate GEMMs (fusion=0): h1 is the gate, h3 is the linear branch.
  defp swiglu_gate(g, l, alpha, beta, limit) do
    g = if limit, do: Nx.min(g, limit), else: g
    l = if limit, do: Nx.clip(l, Nx.negate(limit), limit), else: l
    Nx.multiply(Nx.multiply(g, Nx.sigmoid(Nx.multiply(alpha, g))), Nx.add(l, beta))
  end

  # SwiGLU interleaved layout (swiglu_fusion=1): [g0, l0, g1, l1, ...].
  defp swiglu_interleaved(h, alpha, beta, limit) do
    {t_k, two_i} = Nx.shape(h)
    inter = div(two_i, 2)
    pairs = Nx.reshape(h, {t_k, inter, 2})
    g = pairs |> Nx.slice_along_axis(0, 1, axis: 2) |> Nx.squeeze(axes: [2])
    l = pairs |> Nx.slice_along_axis(1, 1, axis: 2) |> Nx.squeeze(axes: [2])
    swiglu_gate(g, l, alpha, beta, limit)
  end

  # SwiGLU concatenated layout (swiglu_fusion=2): first half g, second half l.
  defp swiglu_concat(h, alpha, beta, limit) do
    inter = div(Nx.axis_size(h, 1), 2)
    g = Nx.slice_along_axis(h, 0, inter, axis: 1)
    l = Nx.slice_along_axis(h, inter, inter, axis: 1)
    swiglu_gate(g, l, alpha, beta, limit)
  end

  # Exact GELU: x * 0.5 * (1 + erf(x / sqrt(2))).
  defp gelu(h) do
    inv_sqrt2 = 1.0 / :math.sqrt(2)
    Nx.multiply(h, Nx.multiply(0.5, Nx.add(1, Nx.erf(Nx.multiply(h, inv_sqrt2)))))
  end

  # Dequantizes packed integer weights: (q - zero_point) * scale.
  # packed: {M, N, K_packed}
  # scales: {M, N, blocks} for block_size > 0, or {M, N} for per-row, or nil
  # Returns {M, N, K} as type.
  defp dequant(packed, scales, zp, bits, block_size, type) do
    q = packed |> unpack(bits) |> Nx.as_type(type)
    {m, n, k} = Nx.shape(q)
    default_zero = 2 ** (bits - 1)

    if block_size > 0 and scales != nil do
      blocks = Nx.axis_size(scales, 2)
      padded = blocks * block_size

      # Pad if k is not aligned to block_size (keeps the same K after dequant).
      q_for_blocks =
        if padded > k,
          do: Nx.pad(Nx.as_type(q, :f32), 0.0, [{0, 0, 0}, {0, 0, 0}, {0, padded - k, 0}]),
          else: Nx.as_type(q, :f32)

      zero_4d =
        if zp do
          # zp is packed; unpack and slice to match blocks.
          zp
          |> unpack(bits)
          |> Nx.slice_along_axis(0, blocks, axis: 2)
          |> Nx.as_type(:f32)
          |> Nx.reshape({m, n, blocks, 1})
        else
          Nx.tensor(default_zero, type: :f32)
        end

      scales_4d = scales |> Nx.as_type(:f32) |> Nx.reshape({m, n, blocks, 1})

      q_for_blocks
      |> Nx.reshape({m, n, blocks, block_size})
      |> Nx.subtract(zero_4d)
      |> Nx.multiply(scales_4d)
      |> Nx.reshape({m, n, padded})
      |> Nx.slice_along_axis(0, k, axis: 2)
      |> Nx.as_type(type)
    else
      # Per-row scales {M, N} or {M, N, 1}, and per-row zero points {M, N}, packed along N.
      zero =
        if zp,
          do: zp |> unpack(bits) |> Nx.as_type(type) |> Nx.reshape({m, n, 1}),
          else: default_zero

      q_sub = Nx.subtract(q, zero)

      if scales do
        s = scales |> Nx.as_type(type) |> Nx.reshape({m, n, 1})
        Nx.multiply(q_sub, s)
      else
        q_sub
      end
    end
  end

  # Unpacks bits-wide unsigned values from the last axis of a u8 tensor.
  defp unpack(packed, 8), do: packed

  defp unpack(packed, bits) do
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

  defp nonzero_float(v) when v == 0 or v == 0.0, do: nil
  defp nonzero_float(v), do: v
end
