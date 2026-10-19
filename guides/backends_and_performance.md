# Backends and Performance

This guide explains the available Nx backends, how to choose a compute type, and how to manage memory.

## Choosing a backend

Pass the `:backend` option to `Transmuter.load!/2` to move the model weights to the backend on load:

```elixir
model = Transmuter.load!(source, backend: EMLX.Backend)
```

Pass `compiler:` to `Transmuter.compile/3` to compile with that backend.

### EMLX

EMLX runs on Apple silicon (M-series). It uses the Metal GPU and the ANE (Apple Neural Engine) where possible. Add it to your dependencies:

```elixir
{:emlx, "~> 0.5"}
```

Use EMLX for:
- Apple silicon Macs and iPhones
- Models with `MatMulNBits` (4-bit or 8-bit quantized weights), which EMLX keeps packed for fast matmul

Set the device explicitly when needed:

```elixir
model = Transmuter.load!(source, backend: {EMLX.Backend, device: :gpu})
run = Transmuter.compile(model, inputs, compiler: EMLX, device: :gpu)
```

### EXLA

EXLA uses XLA and runs on NVIDIA GPUs (CUDA) or on the CPU. Add it to your dependencies:

```elixir
{:exla, "~> 1.0"}
```

Use EXLA for:
- NVIDIA GPU servers
- CPU-only deployments

```elixir
# CUDA
model = Transmuter.load!(source, backend: {EXLA.Backend, client: :cuda})
run = Transmuter.compile(model, inputs, compiler: EXLA, client: :cuda)

# CPU
model = Transmuter.load!(source, backend: {EXLA.Backend, client: :host})
run = Transmuter.compile(model, inputs, compiler: EXLA, client: :host)
```

### Nx.BinaryBackend (CPU, no compilation)

`Nx.BinaryBackend` runs computations in Elixir on the CPU. It has no compilation step. Use it for testing and for models that are too small to benefit from a GPU.

```elixir
model = Transmuter.load!(source)  # params stay on Nx.BinaryBackend by default
outputs = Transmuter.run(model, model.params, inputs)
```

## Compute types

By default, Transmuter runs half-precision exports (`model_fp16.onnx` or `model_bf16.onnx`) in f32. It converts the f16 and bf16 weights to f32 when the model loads. ONNX Runtime on the CPU also computes half-precision graphs in f32, so the results match.

The f32 weights use as much memory as an fp32 export. With `compute_type: nil`, the model runs in native f16 or bf16 with half the memory:

```elixir
model = Transmuter.load!(source, compute_type: nil)
```

In native f16, precision-sensitive operators (sums, normalizations, softmax, attention scores) run in f32 and cast their output back to f16 or bf16. This does not protect a normalization that the export writes as separate operators. Many fp16 decoder exports compute RMSNorm as `Pow`, `ReduceMean`, `Add`, `Sqrt`, `Div` and `Mul`. `Pow` overflows to infinity in f16 for values above 256, and the normalized values become 0. The fp16 export of SmolLM2 135M then gives logits of 0.

Native f16 can also drift in a deep graph, because each intermediate value is stored in f16. The fp16 vision encoder of Qwen2-VL 2B has 32 blocks. In native f16, its output differs from the f32 result by 60%. In f32, it matches ONNX Runtime within 2e-5.

Use native f16 only after you compare its output with the default f32 run. On an Apple silicon GPU, the fp16 export of SmolLM2 135M generates 60.5 tokens per second in f32 and 66.2 in native f16.

## Memory management with EMLX

EMLX maintains two caches of its own that grow with new shapes.

### Freed-buffer cache

MLX keeps freed buffers in a reuse cache. Without a limit, this cache can grow to several gigabytes when models have many audio lengths or image sizes. Set a limit with `EMLX.set_cache_limit/1`:

```elixir
EMLX.set_cache_limit(1_000_000_000)  # 1 GB
```

Call this before loading or running the model. Without a limit, a Kokoro TTS service with many different audio lengths grew to 11.7 GB on a 16 GB Mac and slowed from 0.87 s to 22.5 s per call.

### Compiled function cache

`EMLX.CompileCache` keeps each compiled function. Configure `max_items` in the `:emlx` application config:

```elixir
# config/config.exs
config :emlx, max_items: 128
```

EMLX also keeps each compiled MLX program without a configurable limit.

### Transmuter compile cache

Transmuter's own cache (managed by `Transmuter.compile/3`) is per-process. The ETS table that holds the cache belongs to the process that called `Transmuter.compile/3`, such as an `Nx.Serving` process. When that process exits, the cache is lost and the model recompiles on the next call.

Use `:cache_size` to balance memory and recompilation cost:

```elixir
run = Transmuter.compile(model, inputs, compiler: EMLX, cache_size: 32)
```

## Benchmarks

These results are from an Apple silicon GPU:

| Model | Without compilation | First call | Later calls |
|---|---|---|---|
| Kokoro, 7.0 s of audio | 2.3 s | 2.3 s | 0.28 s |
| Kokoro, 20.2 s of audio | 13.9 s | 10.6 s | 0.84 s |
| Kokoro, 26.6 s of audio | 25.8 s | 11.4 s | 1.11 s |
| Faster R-CNN 12, the same image | 33.8 s | 6.5 s | 0.35 s |

Compilation pays off after a small number of calls. For services that call the same shapes repeatedly, the compiled path is 10x to 100x faster than eager execution.
