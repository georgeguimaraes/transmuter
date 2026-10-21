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

    `Transmuter.load!/2` with `backend: EMLX.Backend` calls `params/2` for you.

    Weights that MLX cannot hold (other bit widths, block sizes other than 32,
    64 or a multiple of 128, or a K that is not a whole number of blocks) stay
    packed bytes and are dequantized in the graph. A block of more than 128
    weights is MLX groups of 128 with the scale of the block.
    """

    alias Transmuter.Ops.Packed

    @doc """
    Returns `model.params` on EMLX, with compatible `MatMulNBits` weights packed on the GPU.

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

    `Transmuter.load!/2` with `backend: EMLX.Backend` calls this function for
    you. Call it directly only when you need to transfer params separately.

    ## Options

      * `:device` (`t:atom/0`) - the EMLX device. The default is `:gpu`.

    ## Examples

        model = Transmuter.load!({:hf, "onnx-community/Qwen3-0.6B-ONNX", "onnx/model_q4.onnx"})
        params = Transmuter.EMLX.params(model, device: :gpu)
    """
    def params(model, opts \\ []) do
      backend = {EMLX.Backend, device: Keyword.get(opts, :device, :gpu)}
      quantized = model |> quantizable_nodes() |> Map.new(&quantize(&1, model, backend))

      Map.new(model.params, fn {name, t} ->
        {name, Map.get_lazy(quantized, name, fn -> Nx.backend_transfer(t, backend) end)}
      end)
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
        |> then(value)
        |> Transmuter.upcast(model[:compute_type])
        |> Nx.reshape({n, blocks})

      biases =
        scales
        |> Nx.multiply(zero_points(List.first(rest), value, n, blocks, bits))
        |> Nx.negate()

      {scales, biases} = {per_group.(scales), per_group.(biases)}

      tensor =
        EMLX.Quantization.quantized_tensor(
          EMLX.Backend.from_nx(packed),
          EMLX.Backend.from_nx(Nx.backend_transfer(scales, backend)),
          EMLX.Backend.from_nx(Nx.backend_transfer(biases, backend)),
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
