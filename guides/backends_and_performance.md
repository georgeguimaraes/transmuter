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

By default, Transmuter keeps the types in the ONNX file. A half-precision export (`model_fp16.onnx` or `model_bf16.onnx`) runs in f16 or bf16.

Precision-sensitive operators (sums, normalizations, softmax, attention scores) automatically run in f32 and cast their output back to f16 or bf16. This matches the behavior of standard transformer implementations.

ONNX Runtime on the CPU computes all operations in f32, even for half-precision exports. To match ONNX Runtime results, load with `compute_type: :f32`:

```elixir
model = Transmuter.load!(source, compute_type: :f32)
```

`compute_type: :f32` upcasts every f16 and bf16 tensor before each operator. This gives the same numerical result as ONNX Runtime on the CPU but uses more memory and is slower on a GPU.

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
