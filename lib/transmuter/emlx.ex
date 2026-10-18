if Code.ensure_loaded?(EMLX.Quantization) do
  defmodule Transmuter.EMLX do
    @moduledoc """
    Moves a model's params to EMLX and keeps 4-bit and 8-bit weights packed.

    `MatMulNBits` weights become MLX quantized tensors. `Nx.dot` then runs
    `mx::quantized_matmul` on the packed weights, so the float weights never
    exist in memory. ONNX packs the lowest bits first, which is also the MLX
    order, and an ONNX zero point becomes the MLX bias `-zero_point * scale`.

        model = Transmuter.load!({:hf, "onnx-community/Qwen3-0.6B-ONNX", "onnx/model_q4.onnx"})
        params = Transmuter.EMLX.params(model)
        run = Nx.Defn.jit(&Transmuter.run(model, &1, &2), compiler: EMLX)

    Weights that MLX cannot hold (other bit widths, group sizes other than 32,
    64 or 128, or a K that is not a whole number of groups) stay packed bytes
    and are dequantized in the graph.
    """

    @group_sizes [32, 64, 128]

    @doc """
    Returns `model.params` on EMLX, with compatible `MatMulNBits` weights quantized.

    ## Options

      * `:device` - the EMLX device, `:gpu` (default) or `:cpu`
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
          Transmuter.Ops.attr(node, "block_size") in @group_sizes,
          rem(Transmuter.Ops.attr(node, "K"), Transmuter.Ops.attr(node, "block_size")) == 0,
          uniq: true,
          do: node
    end

    defp quantize(node = %{input: [_, b_name, scales_name | rest]}, model, backend) do
      [k, n, bits, group] =
        Enum.map(["K", "N", "bits", "block_size"], &Transmuter.Ops.attr(node, &1))

      groups = div(k, group)
      value = &(Map.get(model.params, &1) || Map.fetch!(model.static, &1))

      packed =
        b_name
        |> then(value)
        |> Nx.to_binary()
        |> Nx.from_binary(:u32, backend: Nx.BinaryBackend)
        |> Nx.reshape({n, div(k * bits, 32)})
        |> Nx.backend_transfer(backend)

      scales = scales_name |> then(value) |> Nx.reshape({n, groups})

      biases =
        scales
        |> Nx.multiply(zero_points(List.first(rest), value, n, groups, bits))
        |> Nx.negate()

      tensor =
        EMLX.Quantization.quantized_tensor(
          EMLX.Backend.from_nx(packed),
          EMLX.Backend.from_nx(Nx.backend_transfer(scales, backend)),
          EMLX.Backend.from_nx(Nx.backend_transfer(biases, backend)),
          {n, k},
          type: {:s, bits},
          group_size: group
        )

      {b_name, tensor}
    end

    defp zero_points(name, _value, _n, _groups, bits) when name in [nil, ""], do: 2 ** (bits - 1)

    defp zero_points(name, value, n, groups, bits) do
      zero = value.(name)

      if Nx.Type.float?(Nx.type(zero)) do
        Nx.reshape(zero, {n, groups})
      else
        per = div(8, bits)
        bytes = zero |> Nx.reshape({n, :auto}) |> Nx.as_type(:u16)
        shifts = Nx.multiply(Nx.iota({per}, type: :u16), bits)

        bytes
        |> Nx.new_axis(-1)
        |> Nx.right_shift(shifts)
        |> Nx.bitwise_and(2 ** bits - 1)
        |> Nx.reshape({n, :auto})
        |> Nx.slice_along_axis(0, groups, axis: 1)
      end
    end
  end
end
