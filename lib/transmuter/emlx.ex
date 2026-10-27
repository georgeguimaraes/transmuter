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

    alias Transmuter.Ops.{Packed, Quantized}

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

      * `:device` (`t:atom/0`) - the EMLX device. The default follows `EMLX.default_device/0`, which is `:gpu` unless the `:emlx` app env sets `:default_device`.

    ## Examples

        model = Transmuter.load!({:hf, "onnx-community/Qwen3-0.6B-ONNX", "onnx/model_q4.onnx"})
        model = Transmuter.EMLX.move(model, device: :gpu)
    """
    def move(model, opts \\ []) do
      Transmuter.move(model, {EMLX.Backend, opts})
    end

    @doc false
    # The EMLX backend with a device. Without a :device, the params go to the device of the
    # EMLX compiler, EMLX.default_device/0. EMLX.Backend puts them on the CPU, and the
    # compiler then copies each param to the GPU on each call.
    def backend(EMLX.Backend), do: backend({EMLX.Backend, []})

    def backend({EMLX.Backend, opts}),
      do: {EMLX.Backend, Keyword.put_new_lazy(opts, :device, &EMLX.default_device/0)}

    @doc false
    # Evaluates a tensor that EMLX ops made, and the scales and biases of a quantized tensor.
    # Until then, MLX keeps it as an expression, and a compiled call that reads it computes
    # the expression again each time.
    def evaluate(tensor = %Nx.Tensor{data: %EMLX.Backend{ref: ref, quantization_config: config}}) do
      parts = if config, do: Enum.reject([config.scales, config.biases], &is_nil/1), else: []
      for part <- [ref | Enum.map(parts, &EMLX.Backend.from_nx/1)], do: EMLX.eval(part)
      tensor
    end

    @doc false
    # A MatMulNBits weight that MLX holds as a quantized tensor.
    def compatible?(node) do
      {k, _n, bits, block} = Quantized.nbits_sizes(node)
      bits in [4, 8] and block_size?(block) and rem(k, block) == 0
    end

    defp block_size?(size), do: size in [32, 64] or rem(size, 128) == 0

    @doc false
    # The MLX quantized tensor of a MatMulNBits node, from the packed B and the scales and the
    # zero points on the device, or its float weights on the CPU.
    def weight(node, b, scales, zero_points, model, backend) do
      {EMLX.Backend, opts} = backend
      device = Keyword.fetch!(opts, :device)
      {k, n, bits, block} = Quantized.nbits_sizes(node)

      blocks = div(k, block)
      group = min(block, 128)

      per_group =
        &(&1
          |> Nx.new_axis(-1)
          |> Nx.broadcast({n, blocks, div(block, group)})
          |> Nx.reshape({n, :auto}))

      packed =
        b
        |> Nx.to_binary()
        |> Nx.from_binary(:u32, backend: Nx.BinaryBackend)
        |> Nx.reshape({n, div(k * bits, 32)})
        |> Nx.backend_transfer(backend)

      scales =
        scales
        |> Transmuter.upcast(model[:compute_type])
        |> Nx.reshape({n, blocks})

      biases =
        scales
        |> Nx.multiply(zero_points(zero_points, n, blocks, bits))
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

      if device == :cpu, do: EMLX.dequantize(tensor), else: tensor
    end

    defp zero_points(nil, _n, _groups, bits), do: 2 ** (bits - 1)

    defp zero_points(zero, n, groups, bits) do
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
