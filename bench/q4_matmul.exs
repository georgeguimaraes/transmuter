# Times the 210 MatMulNBits weights of SmolLM2-135M q4 (4 bits, blocks of 32) on EMLX, eagerly:
# packed weights with mx::quantized_matmul, and the dequantized f32 weights with a BLAS matmul.
# Eager ops run on the device that is given. On macOS, EMLX 0.5 runs compiled graphs on the GPU
# also with device: :cpu, so this script does not compile.
# Usage: mix run bench/q4_matmul.exs [device] [rows]

[device, rows] =
  case System.argv() do
    [] -> ["cpu", "1"]
    [device] -> [device, "1"]
    [device, rows] -> [device, rows]
  end

backend = {EMLX.Backend, device: String.to_atom(device)}
rows = String.to_integer(rows)

# {K, N, count} of the MatMulNBits nodes of onnx-community/SmolLM2-135M-Instruct-ONNX model_q4.onnx.
shapes = [{576, 192, 60}, {576, 576, 60}, {576, 1536, 60}, {1536, 576, 30}]
Nx.default_backend(backend)

weights =
  for {k, n, count} <- shapes, i <- 1..count do
    {w, _key} = Nx.Random.normal(Nx.Random.key(i * n + k), 0.0, 0.02, shape: {n, k})
    {k, EMLX.quantize(w, type: {:s, 4}, group_size: 32)}
  end

inputs =
  Map.new(shapes, fn {k, _n, _count} -> {k, Nx.iota({rows, k}, type: :f32) |> Nx.divide(k)} end)

time = fn weights ->
  run = fn -> for {k, w} <- weights, do: Nx.dot(inputs[k], [-1], w, [1]) |> Nx.to_binary() end
  run.()
  {t, _} = :timer.tc(fn -> for _ <- 1..5, do: run.() end)
  t / 5000
end

packed = time.(weights)
dense = time.(Enum.map(weights, fn {k, w} -> {k, EMLX.dequantize(w)} end))

IO.puts(
  "#{length(weights)} MatMulNBits of SmolLM2-135M, #{rows} rows, #{device}: packed #{Float.round(packed, 1)} ms, dense #{Float.round(dense, 1)} ms"
)
