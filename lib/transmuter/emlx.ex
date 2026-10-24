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

    The packed weight holds its scales and zero points. A packed node reads A, the
    packed weight under the name of B with `:packed` after it, and its bias, and its
    scales and zero points inputs become empty names. The B, scales and zero points
    that no other node, subgraph or graph output reads then leave the params and the
    static values. A node that also reads them, such as `GatherBlockQuantized` of tied
    embeddings, gets the raw values, in an array of its own, because MLX does not compile
    one array as an input and as a part of a quantized tensor. Each name moves to EMLX
    once, and the scales are computed on EMLX. The returned model has the EMLX backend.
    A second move leaves the packed nodes as they are.

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
      packed_outputs = MapSet.new(packed, &hd(&1.output))

      # A packed node reads A, its packed weight and its bias. The packed weight has a name of
      # its own, so a node that reads the raw B, such as GatherBlockQuantized of tied
      # embeddings, still reads the raw bytes.
      nodes =
        Enum.map(model.nodes, fn node ->
          if MapSet.member?(packed_outputs, List.first(node.output)),
            do: %{node | input: packed_inputs(node.input, model)},
            else: node
        end)

      used =
        for node <- nodes,
            name <- Transmuter.dependencies(node),
            into: MapSet.new(model.outputs),
            do: name

      # Packing frees the B, scales and zero points of the packed nodes that no other node
      # and no graph output reads. Other names stay as they are.
      freed =
        for node <- packed,
            name <- Enum.slice(node.input, 1, 3),
            name != "",
            not MapSet.member?(used, name),
            into: MapSet.new(),
            do: name

      quantization =
        for node <- packed, name <- Enum.slice(node.input, 2, 2), name != "", uniq: true, do: name

      kept = model.params |> Map.keys() |> Enum.reject(&MapSet.member?(freed, &1))

      # Each name moves once, also when several packed nodes or other nodes read it.
      device = names_on_device(model, quantization ++ kept, backend)

      quantized =
        packed
        |> Enum.uniq_by(&Enum.at(&1.input, 1))
        |> Map.new(&quantize(&1, model, device, backend))

      # A scales or zero points name that another node also reads, such as those of tied
      # embeddings, gives that node a copy. MLX does not compile one array as an input and
      # as a part of a quantized tensor.
      shared = MapSet.new(quantization)

      params =
        for name <- kept, into: quantized do
          if MapSet.member?(shared, name),
            do: {name, Nx.backend_copy(device[name], backend)},
            else: {name, device[name]}
        end

      static = Map.drop(model.static, MapSet.to_list(freed))
      %{model | nodes: nodes, params: params, static: static, backend: backend}
    end

    # The inputs of a packed node: the packed weight in place of B, and no scales or zero
    # points.
    defp packed_inputs([a, b | rest], model) do
      name = packed_name(b)

      if Map.has_key?(model.params, name) or Map.has_key?(model.static, name) do
        raise ArgumentError, "the model already has a value named #{name}"
      end

      [a, name | rest] |> List.replace_at(2, "") |> List.replace_at(3, "")
    end

    defp packed_name(b), do: b <> ":packed"

    # The names on the device, each moved once. A packed weight B moves in quantize/4.
    defp names_on_device(model, names, backend) do
      for name <- Enum.uniq(names), into: %{} do
        value = Map.get(model.params, name) || Map.fetch!(model.static, name)
        {name, Nx.backend_transfer(value, backend)}
      end
    end

    # A MatMulNBits node whose B holds the packed bytes of its N x K weights and whose scales
    # have a name. After a move, the node reads the packed weight and no scales, so a second
    # move leaves it as it is.
    defp quantizable_nodes(model) do
      Enum.filter(model.nodes, &quantizable?(&1, model))
    end

    defp quantizable?(node = %{op_type: "MatMulNBits", input: [_, b, scales | rest]}, model) do
      [k, n, bits, block] = sizes(node)

      scales != "" and Enum.at(rest, 1, "") == "" and bits in [4, 8] and block_size?(block) and
        rem(k, block) == 0 and packed_bytes?(Map.get(model.params, b), n * k * bits)
    end

    defp quantizable?(_node, _model), do: false

    # K, N, bits and block_size of a MatMulNBits node. bits is 4 by default.
    defp sizes(node) do
      Enum.map([{"K", nil}, {"N", nil}, {"bits", 4}, {"block_size", nil}], fn {name, default} ->
        Transmuter.Ops.attr(node, name, default)
      end)
    end

    # The bytes of packed weights: u8 values, as many as the bits of the weights need.
    defp packed_bytes?(b = %Nx.Tensor{}, bits),
      do: Nx.type(b) == {:u, 8} and Nx.size(b) * 8 == bits

    defp packed_bytes?(_b, _bits), do: false

    defp block_size?(size), do: size in [32, 64] or rem(size, 128) == 0

    defp quantize(node = %{input: [_, b_name, scales_name | rest]}, model, device_values, backend) do
      {EMLX.Backend, device: device} = backend

      [k, n, bits, block] = sizes(node)

      blocks = div(k, block)
      group = min(block, 128)
      # The scales and the zero points are on the device, so their math runs on EMLX and not
      # in the lists of Nx.BinaryBackend.
      on_device = &Map.fetch!(device_values, &1)

      per_group =
        &(&1
          |> Nx.new_axis(-1)
          |> Nx.broadcast({n, blocks, div(block, group)})
          |> Nx.reshape({n, :auto}))

      packed =
        model.params
        |> Map.fetch!(b_name)
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

      {packed_name(b_name), if(device == :cpu, do: EMLX.dequantize(tensor), else: tensor)}
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
