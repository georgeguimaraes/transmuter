# Backends and Performance

This guide explains the available Nx backends, how to choose a compute type, and how to manage memory.

## Choosing a backend

Pass the `:backend` option to `Transmuter.load!/2` to move the model weights to the backend on load:

```elixir
model = Transmuter.load!(source, backend: EMLX.Backend)
```

Pass `compiler:` to `Transmuter.compile/3` to compile with that backend.

### EMLX

EMLX runs on Apple silicon (M-series). It uses the Metal GPU, or the CPU with `device: :cpu`. Add it to your dependencies:

```elixir
{:emlx, "~> 0.5"}
```

Use EMLX for:
- Apple silicon Macs
- Models with `MatMulNBits` (4-bit or 8-bit quantized weights), which EMLX keeps packed on the GPU for fast matmul

Set the device explicitly when needed:

```elixir
model = Transmuter.load!(source, backend: {EMLX.Backend, device: :gpu})
run = Transmuter.compile(model, inputs, compiler: EMLX, device: :gpu)
```

On EMLX, the normalization and attention operators run as the fused kernels of `EMLX.Fast`. This applies when EMLX compiles the model and when the model runs eagerly on `EMLX.Backend`:

- `RMSNormalization`, `SimplifiedLayerNormalization`, `SkipSimplifiedLayerNormalization`, `LayerNormalization`, `SkipLayerNormalization` and `EmbedLayerNormalization` use one normalization kernel.
- `Attention`, `GroupQueryAttention`, `MultiHeadAttention` and the `com.microsoft` `Attention` use one attention kernel. Attention with a softcap keeps the Nx operations, because the kernel has no softcap.

The fused kernels change the last bits of the results. On an Apple silicon GPU, the fused normalization makes a decode step of Qwen3-0.6B q4 about 0.6 ms faster, and the fused attention makes it 1.3 ms faster again (12.7 ms to 11.4 ms). The vision encoder of Ministral 3 3B takes 298 ms instead of 364 ms for a 448x448 image.

Many exports write RMSNorm, LayerNorm and the attention core as separate operators, such as `Pow`, `ReduceMean`, `Sqrt`, `Softmax` and `MatMul`. When the model loads, `Transmuter.Fusion` replaces these subgraphs with `SimplifiedLayerNormalization`, `LayerNormalization` and an attention operator. These operators then also use the fused kernels. A subgraph changes only when the replacement computes the same function. The load option `fuse: false` keeps the graph as it is. On an Apple silicon GPU:

- The decode step of SmolLM2 135M q4 (onnx-community export) takes 7.8 ms instead of 8.6 ms.
- all-MiniLM-L6-v2 embeds 128 tokens in 2.34 ms instead of 2.57 ms, and a batch of 8 in 9.15 ms instead of 10.0 ms.
- bert-base-uncased embeds 128 tokens in 10.5 ms instead of 11.5 ms, and a batch of 8 in 58.4 ms instead of 64.2 ms.

### 4-bit and 8-bit weights on the EMLX CPU

On the GPU, `Transmuter.EMLX` keeps the `MatMulNBits` weights packed, and `mx::quantized_matmul` reads them. On the CPU (`device: :cpu`), MLX runs `mx::quantized_matmul` on one thread. So on the CPU, `Transmuter.EMLX` dequantizes these weights once, when the model loads, and each matmul runs in the BLAS of the CPU (OpenBLAS on Linux, Accelerate on macOS) on all cores.

The float weights take the memory of an fp32 export: 8 times the memory of the 4-bit weights. The 106M `MatMulNBits` weights of SmolLM2 135M take about 425 MB instead of about 66 MB. The 85M weights of the SmolVLM 256M vision encoder take about 340 MB instead of about 53 MB. On the CPU, a q4 export thus saves only download size. When memory is the limit on the CPU, use the q8 export (`model_quantized.onnx`). Its `MatMulInteger` weights stay 8-bit integers.

On a Linux runner with 4 vCPUs and OpenBLAS (packed weights against float weights):

| Work | Packed | Float |
| --- | --- | --- |
| SmolLM2 135M q4, one decode step | 109.7 ms | 75.2 ms |
| SmolLM2 135M q4, prompt and first token | 3435 ms | 600 ms |
| SmolVLM 256M q4 vision encoder, one 512x512 tile | 57.5 s | 3.8 s |
| SmolVLM 256M q4 vision encoder, 13 tiles | not measured | 40.9 s |
| Peak memory, SmolLM2 135M q4 (two servings) | 881 MB | 1279 MB |
| Peak memory, SmolVLM 256M q4 vision encoder | 423 MB | 716 MB |

`bench/q4_matmul.exs` times the 210 `MatMulNBits` weights of SmolLM2 135M alone, eagerly. On an Apple M2 Pro CPU with Accelerate, packed weights take 33.3 ms for 1 row, 998 ms for 64 rows and 16647 ms for 1024 rows. Float weights take 26.3 ms, 46.1 ms and 382 ms. The quantized matmul uses one core on macOS too, so the rule does not depend on the operating system. With f16 weights (the scales of a q4f16 export), packed weights take 42.7 ms for 1 row and 1532 ms for 64 rows, and float weights take 56.4 ms and 85.0 ms.

On macOS, EMLX 0.5 runs compiled graphs on the GPU also with `device: :cpu`. The outputs say `:cpu`, but the GPU does the work. Eager operations run on the CPU. Thus the numbers of compiled servings with `device: :cpu` on a Mac are GPU numbers.

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

On NVIDIA GPUs of the Ampere generation and later, XLA computes the f32 matrix products in TF32 by default. TF32 keeps 10 bits of the mantissa, so the results differ from ONNX Runtime by about 1e-3. For f32 precision, set these environment variables before the BEAM starts:

```bash
NVIDIA_TF32_OVERRIDE=0 XLA_FLAGS=--xla_gpu_enable_triton_gemm=false
```

On an RTX 3060, the test suite passes with both variables. Without them, 15 tests that compare with reference values fail.

### Nx.BinaryBackend (CPU, no compilation)

`Nx.BinaryBackend` runs computations in Elixir on the CPU. It has no compilation step. Use it for testing and for models that are too small to benefit from a GPU.

```elixir
model = Transmuter.load!(source)  # params stay on Nx.BinaryBackend by default
outputs = Transmuter.run(model, model.params, inputs)
```

## Compute types

By default, Transmuter runs half-precision exports (`model_fp16.onnx` or `model_bf16.onnx`) in native f16 or bf16. The weights use half the memory of an fp32 export.

With `compute_type: :f32`, Transmuter converts the f16 and bf16 weights to f32 when the model loads. ONNX Runtime on the CPU also computes half-precision graphs in f32, so the results match. The f32 weights use as much memory as an fp32 export:

```elixir
model = Transmuter.load!(source, compute_type: :f32)
```

In native f16, precision-sensitive operators (sums, normalizations, softmax, attention scores) compute in f32. On EMLX, the fused kernels of the normalizations and of attention take the f16 or bf16 values and accumulate in f32. Other operators cast their inputs to f32 and their outputs back to f16 or bf16. Many fp16 decoder exports compute RMSNorm as `Pow`, `ReduceMean`, `Add`, `Sqrt`, `Div` and `Mul`. `Pow` overflows to infinity in f16 for values above 256, and the normalized values become 0. `Transmuter.Fusion` replaces this subgraph with `SimplifiedLayerNormalization`, which runs in f32. A normalization that it does not recognize still overflows. With `fuse: false`, the fp16 export of SmolLM2 135M gives logits of 0.

Native f16 can also drift in a deep graph, because each intermediate value is stored in f16. The fp16 vision encoder of Qwen2-VL 2B has 32 blocks. Against ONNX Runtime on the CPU, the relative L2 error of its output is 0.027 in native f16 and 0.006 with `compute_type: :f32`. With `fuse: false`, its LayerNorm overflows in native f16 and the relative error is 0.76.

For a deep half-precision graph, compare the native output with a `compute_type: :f32` run. On an Apple silicon GPU, the fp16 export of SmolLM2 135M gives the same 128 greedy tokens in f32 and in native f16. A decode step takes 10.3 ms in f32 and 9.1 ms in native f16, and the peak memory is 692 MB in f32 and 463 MB in native f16.

## Dynamic 8-bit quantization

The q8 exports (`model_quantized.onnx`) quantize the activations when the model runs, with `DynamicQuantizeLinear`. This operator takes one scale and one zero point from the minimum and the maximum of the whole tensor. In a batch, the tensor holds every row, so the padding rows and the other inputs change the scale of each input. An input in a batch can then give results that are different from its results alone.

Transmuter computes these operators as ONNX Runtime does, so a batch gives the results of ONNX Runtime on the same batch. When the results of an input must not depend on the batch, use a batch size of 1, or a q4 or fp32 variant.

The q8 results are also sensitive to very small float differences. Each `DynamicQuantizeLinear` puts its input values in 256 buckets. A difference of one float rounding step can move a value to the next bucket, and the next `MatMulInteger` multiplies that difference by the weights. Each block adds new differences, and activations with large outliers make the buckets wider. The output of a deep q8 graph thus changes when a sum is computed in a different order.

ONNX Runtime shows the same effect against itself. These numbers compare the outputs of the q8 vision encoder of SmolVLM 256M on one image (13 tiles):

| Comparison | Mean abs difference | Max abs difference |
| --- | --- | --- |
| ONNX Runtime `ORT_ENABLE_ALL` against `ORT_DISABLE_ALL` | 0.512 | 11.06 |
| Transmuter on EMLX against `ORT_DISABLE_ALL` | 0.504 | 8.60 |
| Transmuter on EMLX against `ORT_ENABLE_ALL` | 0.506 | 8.90 |

The mean absolute value of the output is 3.72. With the exact inputs that ONNX Runtime gives each node, the `DynamicQuantizeLinear`, `MatMulInteger` and `ConvInteger` nodes of Transmuter give the same bits as ONNX Runtime. The other nodes differ by less than 0.000003 relative. The q8 encoder of vit-gpt2 shows the same pattern: 0.0151 mean abs difference between the two ONNX Runtime levels, and 0.0175 for Transmuter, on a mean abs value of 0.198.

Thus a q8 model does not give the same output in two runtimes, or in one runtime with two optimization levels. The generated text can change after some tokens. For outputs that agree with ONNX Runtime to float precision, use the fp32 variant, or a q4 variant (`MatMulNBits` quantizes only the weights). On the same image, the max abs difference between Transmuter and ONNX Runtime is 0.0002 for the q4 vision encoder of SmolVLM 256M.

## Memory management with EMLX

EMLX maintains two caches of its own that grow with new shapes.

### Freed-buffer cache

MLX keeps freed buffers in a reuse cache. Without a limit, this cache can grow to several gigabytes when models have many audio lengths or image sizes. Set a limit with `EMLX.set_cache_limit/1`:

```elixir
EMLX.set_cache_limit(1_000_000_000)  # 1 GB
```

Call this before loading or running the model. Without a limit, a Kokoro TTS service with many different audio lengths grew to 11.7 GB on a 16 GB Mac and slowed from 0.87 s to 22.5 s per call.

### Compiled function cache

`EMLX.CompileCache` keeps each compiled function. Configure `max_items` for it in the `:emlx` application config:

```elixir
# config/config.exs
config :emlx, EMLX.CompileCache, max_items: 128
```

EMLX also keeps each compiled MLX program without a configurable limit.

### Transmuter compile cache

Transmuter's own cache (managed by `Transmuter.compile/3`) is per-process. The compiled segments are in `:persistent_term`, so a read does not copy them. An ETS table keeps only the keys and the time of use of the segments. Both belong to the process that called `Transmuter.compile/3`, such as an `Nx.Serving` process. When that process exits, Transmuter erases the segments, and the model compiles its segments again on each call.

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
