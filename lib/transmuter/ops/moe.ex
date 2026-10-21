defmodule Transmuter.Ops.MoE do
  @moduledoc """
  The ONNX Runtime mixture-of-experts operators `MoE` and `QMoE` (domain com.microsoft). Each token selects its top-k experts, runs the MLP of each expert, and adds the outputs with the routing weights.

  `MoE` takes a softmax over all experts, then uses the top-k probabilities as weights. `QMoE` takes the top-k router values, then a softmax over those k values. `QMoE` dequantizes 4-bit and 8-bit block-wise weights as (q - zero point) * scale.
  """

  import Transmuter.Ops, only: [attr: 2, attr: 3, optional: 2, softmax_last: 1]

  alias Transmuter.Ops.Packed

  @op_types ~w(MoE QMoE)

  @doc false
  def op_types, do: @op_types

  @doc false
  def apply(node, inputs), do: node |> op(inputs) |> List.wrap()

  defp op(node = %{op_type: "MoE"}, [input, router_probs | rest]) do
    [fc1_w, fc1_bias, fc2_w, fc2_bias, fc3_w, fc3_bias] = optional(rest, 6)
    probs = router_probs |> Nx.as_type(:f32) |> softmax_last()
    {top_vals, top_idx} = topk(probs, attr(node, "k"))

    weights =
      if attr(node, "normalize_routing_weights", 0) == 1,
        do: local_normalize(top_vals),
        else: top_vals

    route = route(weights, top_idx, Nx.axis_size(fc1_w, 0))
    w = &(&1 && select(&1, route))

    experts(node, input, route, [
      {w.(fc1_w), fc1_bias},
      {w.(fc2_w), fc2_bias},
      {w.(fc3_w), fc3_bias}
    ])
  end

  # The weights of a token add up to 1 after the softmax over its k values, so
  # normalize_routing_weights changes nothing.
  defp op(node = %{op_type: "QMoE"}, [input, router_probs | rest]) do
    [fc1_w, fc1_scales, fc1_bias, fc2_w, fc2_scales, fc2_bias | opt] = optional(rest, 12)
    [fc3_w, fc3_scales, fc3_bias, fc1_zp, fc2_zp, fc3_zp] = opt
    bits = attr(node, "expert_weight_bits", 4)
    block_size = attr(node, "block_size", 0)
    type = Nx.type(input)
    {top_vals, top_idx} = topk(Nx.as_type(router_probs, :f32), attr(node, "k"))
    route = route(softmax_last(top_vals), top_idx, Nx.axis_size(fc1_w, 0))
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
    activation = %{
      type: attr(node, "activation_type", "relu"),
      fusion: attr(node, "swiglu_fusion", 0),
      alpha: attr(node, "activation_alpha", 1.0),
      beta: attr(node, "activation_beta", 0.0),
      limit: attr(node, "swiglu_limit", 0.0)
    }

    type = Nx.type(input)
    h = Nx.axis_size(input, -1)
    input_2d = Nx.reshape(input, {:auto, h})
    t = Nx.axis_size(input_2d, 0)
    rows = input_2d |> Nx.broadcast(route.rows.(t, h), axes: route.axes) |> Nx.reshape({:auto, h})
    h1 = rows |> expert_dot(fc1_w, route) |> add_bias(fc1_bias, route.experts)

    # With swiglu_fusion 0, fc1 gives the gate and fc3 the linear values.
    activated =
      if activation.type == "swiglu" and activation.fusion == 0 and fc3_w do
        h3 = rows |> expert_dot(fc3_w, route) |> add_bias(fc3_bias, route.experts)
        swiglu_gate(h1, h3, activation)
      else
        activate(h1, activation)
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

  # A stable sort keeps the lower index first among equal values, as ONNX Runtime does.
  defp topk(probs, k) do
    idx =
      probs
      |> Nx.argsort(axis: 1, direction: :desc, stable: true)
      |> Nx.slice_along_axis(0, k, axis: 1)

    {Nx.take_along_axis(probs, idx, axis: 1), idx}
  end

  defp local_normalize(weights) do
    sum = Nx.sum(weights, axes: [1], keep_axes: true)
    safe = Nx.select(Nx.equal(sum, 0), 1, sum)
    Nx.divide(weights, safe)
  end

  defp add_bias(h, nil, _flat_idx), do: h
  defp add_bias(h, bias, flat_idx), do: Nx.add(h, Nx.take(bias, flat_idx))

  defp activate(h, %{type: "relu"}), do: Nx.max(h, 0)
  defp activate(h, %{type: "gelu"}), do: gelu(h)
  defp activate(h, %{type: "silu"}), do: Nx.multiply(h, Nx.sigmoid(h))
  defp activate(h, %{type: "identity"}), do: h

  # swiglu_fusion 2 puts the gates in the first half and the linear values in the second.
  defp activate(h, activation = %{type: "swiglu", fusion: 2}) do
    inter = div(Nx.axis_size(h, 1), 2)
    g = Nx.slice_along_axis(h, 0, inter, axis: 1)
    l = Nx.slice_along_axis(h, inter, inter, axis: 1)
    swiglu_gate(g, l, activation)
  end

  # swiglu_fusion 1, and 0 without fc3, interleave the gates and the linear values.
  defp activate(h, activation = %{type: "swiglu"}) do
    {rows, size} = Nx.shape(h)
    pairs = Nx.reshape(h, {rows, div(size, 2), 2})
    g = pairs |> Nx.slice_along_axis(0, 1, axis: 2) |> Nx.squeeze(axes: [2])
    l = pairs |> Nx.slice_along_axis(1, 1, axis: 2) |> Nx.squeeze(axes: [2])
    swiglu_gate(g, l, activation)
  end

  # A swiglu_limit of 0 is no limit.
  defp swiglu_gate(g, l, %{alpha: alpha, beta: beta, limit: limit}) do
    {g, l} =
      if limit == 0,
        do: {g, l},
        else: {Nx.min(g, limit), Nx.clip(l, Nx.negate(limit), limit)}

    Nx.multiply(Nx.multiply(g, Nx.sigmoid(Nx.multiply(alpha, g))), Nx.add(l, beta))
  end

  defp gelu(h) do
    inv_sqrt2 = 1.0 / :math.sqrt(2)
    Nx.multiply(h, Nx.multiply(0.5, Nx.add(1, Nx.erf(Nx.multiply(h, inv_sqrt2)))))
  end

  # Dequantizes the packed weights {M, N, K / per byte} to {M, N, K} of type. Block-wise
  # scales {M, N, blocks} work in f32, and K pads to whole blocks. Per-row scales are {M, N}
  # or {M, N, 1}, with zero points {M, N} packed along N.
  defp dequant(packed, scales, zp, bits, block_size, type) do
    q = packed |> Packed.unpack(bits) |> Nx.as_type(type)
    {m, n, k} = Nx.shape(q)
    default_zero = 2 ** (bits - 1)

    if block_size > 0 and scales != nil do
      blocks = Nx.axis_size(scales, 2)
      padded = blocks * block_size
      q = Nx.as_type(q, :f32)
      q = if padded > k, do: Nx.pad(q, 0.0, [{0, 0, 0}, {0, 0, 0}, {0, padded - k, 0}]), else: q

      zero =
        if zp do
          zp |> Packed.unpack(bits) |> Nx.slice_along_axis(0, blocks, axis: 2) |> Nx.as_type(:f32)
        else
          Nx.tensor(default_zero, type: :f32)
        end

      q
      |> Packed.dequantize(zero, Nx.as_type(scales, :f32))
      |> Nx.slice_along_axis(0, k, axis: 2)
      |> Nx.as_type(type)
    else
      zero =
        if zp,
          do: zp |> Packed.unpack(bits) |> Nx.as_type(type) |> Nx.reshape({m, n, 1}),
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
end
