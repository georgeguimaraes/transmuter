if Code.ensure_loaded?(EMLX.Quantization) do
  defmodule Transmuter.EMLX do
    @moduledoc """
    Moves a model's params to EMLX and keeps 4-bit and 8-bit weights packed on the GPU.

    On the GPU, `MatMulNBits` weights become MLX quantized tensors. `Nx.dot`
    then runs `mx::quantized_matmul` on the packed weights, so the float weights
    never exist in memory. ONNX packs the lowest bits first, which is also the
    MLX order, and an ONNX zero point becomes the MLX bias `-zero_point * scale`.

    On the CPU (`device: :cpu`), MLX runs `mx::quantized_matmul` on one thread.
    So the weights are dequantized once, when the model loads, and the matmul
    runs in the BLAS of the CPU on all cores. The weights then take the memory
    of float weights. See the guide "Backends and Performance".

        model = Transmuter.load!({:hf, "onnx-community/Qwen3-0.6B-ONNX", "onnx/model_q4.onnx"}, backend: EMLX.Backend)

    `Transmuter.load!/2` with `backend: EMLX.Backend` calls `move/2` for you.

    Weights that MLX cannot hold (other bit widths, block sizes other than 32,
    64 or a multiple of 128, or a K that is not a whole number of blocks) stay
    packed bytes and are dequantized in the graph. A block of more than 128
    weights is MLX groups of 128 with the scale of the block.
    """

    alias Transmuter.Ops.Packed

    @doc """
    Moves the params of `model` to EMLX, with compatible `MatMulNBits` weights packed on the GPU, and returns the model.

    On the GPU, compatible `MatMulNBits` weights become MLX quantized tensors, so
    `mx::quantized_matmul` reads the packed weights directly. This avoids
    materializing the float weights in memory on each forward pass. On the CPU,
    these weights become float tensors of shape `{N, K}`, in the type of the
    scales (or `:compute_type`), so the matmul uses BLAS. Other params are
    transferred to EMLX with `Nx.backend_transfer/2`.

    A weight is compatible when:

    - The bit width is 4 or 8.
    - The block size is 32, 64, or a multiple of 128.
    - The K dimension is a whole number of blocks.

    Weights that do not meet these criteria stay as packed bytes and are
    dequantized in the graph.

    The packed weight holds its scales and zero points. A scales or zero points
    initializer that only packed `MatMulNBits` nodes read does not move to EMLX:
    it leaves the params, and these nodes get an empty name in its place. A node
    that still reads it raises, because the value is missing. The scales are
    computed on EMLX, from the params moved to EMLX.

    `Transmuter.load!/2` with `backend: EMLX.Backend` calls this function for
    you. Call it directly only when you need to move the params separately.

    ## Options

      * `:device` (`t:atom/0`) - the EMLX device. The default is `:gpu`.

    ## Examples

        model = Transmuter.load!({:hf, "onnx-community/Qwen3-0.6B-ONNX", "onnx/model_q4.onnx"})
        model = Transmuter.EMLX.move(model, device: :gpu)
    """
    def move(model, opts \\ []) do
      backend = {EMLX.Backend, device: Keyword.get(opts, :device, :gpu)}
      packed = quantizable_nodes(model)
      quantized = Map.new(packed, &quantize(&1, model, backend))
      unused = unused_inputs(model, packed)

      params =
        for {name, t} <- model.params, not MapSet.member?(unused, name), into: %{} do
          {name, Map.get_lazy(quantized, name, fn -> Nx.backend_transfer(t, backend) end)}
        end

      packed = MapSet.new(packed)

      nodes =
        Enum.map(model.nodes, fn node ->
          if MapSet.member?(packed, node),
            do: %{
              node
              | input: Enum.map(node.input, &if(MapSet.member?(unused, &1), do: "", else: &1))
            },
            else: node
        end)

      %{model | nodes: nodes, params: params}
    end

    # The scales and zero points (inputs 2 and 3) of packed nodes that no other reader needs:
    # no other node, no other input of a packed node, no subgraph and no graph output.
    defp unused_inputs(model, packed) do
      packed_set = MapSet.new(packed)

      quantization =
        for node <- packed,
            name <- Enum.slice(node.input, 2, 2),
            name != "",
            into: MapSet.new(),
            do: name

      used =
        for node <- model.nodes,
            name <-
              if(MapSet.member?(packed_set, node),
                do: List.delete_at(List.delete_at(node.input, 3), 2),
                else: Transmuter.dependencies(node)
              ),
            into: MapSet.new(model.outputs),
            do: name

      MapSet.difference(quantization, used)
    end

    defp quantizable_nodes(model) do
      for node = %{op_type: "MatMulNBits", input: [_, b, _scales | rest]} <- model.nodes,
          Map.has_key?(model.params, b),
          Enum.at(rest, 1, "") == "",
          Transmuter.Ops.attr(node, "bits", 4) in [4, 8],
          block_size?(Transmuter.Ops.attr(node, "block_size")),
          rem(Transmuter.Ops.attr(node, "K"), Transmuter.Ops.attr(node, "block_size")) == 0,
          uniq: true,
          do: node
    end

    defp block_size?(size), do: size in [32, 64] or rem(size, 128) == 0

    defp quantize(node = %{input: [_, b_name, scales_name | rest]}, model, backend) do
      {EMLX.Backend, device: device} = backend

      [k, n, bits, block] =
        Enum.map(["K", "N", "bits", "block_size"], &Transmuter.Ops.attr(node, &1))

      blocks = div(k, block)
      group = min(block, 128)
      value = &(Map.get(model.params, &1) || Map.fetch!(model.static, &1))
      # The scales and the zero points move first, so their math runs on EMLX and not in the
      # lists of Nx.BinaryBackend.
      on_device = &(&1 |> then(value) |> Nx.backend_transfer(backend))

      per_group =
        &(&1
          |> Nx.new_axis(-1)
          |> Nx.broadcast({n, blocks, div(block, group)})
          |> Nx.reshape({n, :auto}))

      packed =
        b_name
        |> then(value)
        |> Nx.to_binary()
        |> Nx.from_binary(:u32, backend: Nx.BinaryBackend)
        |> Nx.reshape({n, div(k * bits, 32)})
        |> Nx.backend_transfer(backend)

      scales =
        scales_name
        |> on_device.()
        |> Transmuter.upcast(model[:compute_type])
        |> Nx.reshape({n, blocks})

      biases =
        scales
        |> Nx.multiply(zero_points(List.first(rest), on_device, n, blocks, bits))
        |> Nx.negate()

      {scales, biases} = {per_group.(scales), per_group.(biases)}

      tensor =
        EMLX.Quantization.quantized_tensor(
          EMLX.Backend.from_nx(packed),
          EMLX.Backend.from_nx(scales),
          EMLX.Backend.from_nx(biases),
          {n, k},
          type: {:s, bits},
          group_size: group
        )

      {b_name, if(device == :cpu, do: EMLX.dequantize(tensor), else: tensor)}
    end

    defp zero_points(name, _value, _n, _groups, bits) when name in [nil, ""], do: 2 ** (bits - 1)

    defp zero_points(name, value, n, groups, bits) do
      zero = value.(name)

      if Nx.Type.float?(Nx.type(zero)) do
        Nx.reshape(zero, {n, groups})
      else
        zero
        |> Nx.reshape({n, :auto})
        |> Packed.unpack(bits)
        |> Nx.slice_along_axis(0, groups, axis: 1)
      end
    end
  end
end
