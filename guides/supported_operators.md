# Supported Operators and Data Types

This guide lists the operators and data types that Transmuter supports, and describes known limits.

## Standard domain operators

Transmuter supports all 203 operators in the ONNX standard domain (`ai.onnx`). It also supports the ONNX Runtime contrib operators that Hugging Face exports use, including:

- `MatMulNBits`: 4-bit and 8-bit quantized matrix multiply
- `GatherBlockQuantized`: block-quantized gather
- `DynamicQuantizeLinear`: dynamic int8 quantization
- `MatMulInteger`: integer matrix multiply
- `ConvInteger`: integer convolution
- `Attention`: multi-head attention with fused QKV
- `GroupQueryAttention`: grouped-query attention
- `MultiHeadAttention`: multi-head attention with separate QKV
- `BiasGelu`: fused bias and GELU activation
- `SkipLayerNormalization`, `SkipSimplifiedLayerNormalization`: fused residual add and layer norm
- `EmbedLayerNormalization`: fused embedding and layer norm
- `RMSNormalization`, `SimplifiedLayerNormalization`: RMS normalization variants
- `MoE`: mixture-of-experts with SiLU and SwiGLU activation, top-k routing, optional normalize
- `QMoE`: quantized mixture-of-experts with 4-bit and 8-bit block-wise weights and top-k routing

The [coverage section of the README](../README.md) lists the tested models. `test/fixtures/ops` has one test graph for each operator variant.

## Data types

Transmuter supports these numeric Nx types: `f32`, `f64`, `f16`, `bf16`, `s8`, `s16`, `s32`, `s64`, `u8`, `u16`, `u32`, `u64`. It does not load the complex types of ONNX.

Nx has no zero-size tensors. A tensor with a dimension of 0 is represented as `{:empty, shape}`. Operators that pass a zero-size tensor through (such as `Reshape`, `Transpose`, `MatMul`) propagate the empty shape without computing.

Non-tensor types have the following representations:

| ONNX type | Elixir representation |
|---|---|
| String tensor | `{:strings, shape, values}` with `values` as a flat list of binaries |
| Sequence | `{:sequence, tensors}`. An eager run of `Transmuter.run/4` returns it as a list of tensors |
| Empty optional | `:none` |

Compiled functions can only return `Nx.Tensor` values. To read a string or sequence output, call `Transmuter.run/4` without compilation.

## Precision notes

In native f16 and bf16 (the default), sums, normalizations, softmax, and attention scores run in f32 when the input is f16 or bf16. The output is cast back to f16 or bf16. This matches the precision behavior of standard transformer libraries. Use `compute_type: :f32` when loading to match ONNX Runtime on the CPU.

## Known numeric differences

These models do not match ONNX Runtime value by value. The first three make small rounding differences large. For the int8 falcon-rw-1b and slovak-gpt-j-405M files, the int8 kernel of ONNX Runtime on x86 is not exact, and slovak-gpt-j-405M also has a rounding tie. These differences are not bugs in Transmuter.

**Grounding DINO**: `TopK` selects 900 of 13,294 queries. Some scores differ by only 9.5e-7, which can change the selected set. All operators before `TopK` match exactly. On a real image with a text prompt, Transmuter gives the same detections as ONNX Runtime, with scores within 0.0001.

**Gemma 4 E2B decoder**: The decoder has activations up to about 1300 and final norm weights up to about 700. When the test changes the float inputs of ONNX Runtime by 1e-7 of their values, the logits of ONNX Runtime change by up to 5.5. The logits of Transmuter differ from ONNX Runtime by 5.3 on the CPU and by 13.6 on the GPU, which is the same noise. The two give the same token for 7 of 8 positions of the random test input. That input has random embeddings with sigma 0.5, without the scale of the real embeddings. With a real prompt of 20 tokens through `embed_tokens_q4` and `decoder_model_merged_q4`, Transmuter on the GPU gives the 32 greedy tokens of ONNX Runtime on the CPU, with logits within about 1e-4 at each step, also at a step where the top two logits of ONNX Runtime are only 0.055 apart.

**Kokoro TTS**: The model sums a phase over all audio samples and takes its STFT. ONNX Runtime and MLX round the phase sum differently, and phases near zero amplitude can change by up to π. When given the ONNX Runtime outputs for the `Sin` and `STFT` nodes, all other Kokoro operators match. On real speech, Transmuter is as close to ONNX Runtime as PyTorch is.

**int8 falcon-rw-1b and slovak-gpt-j-405M on x86**: The ONNX Runtime reference ran on x86 with AVX2 and no VNNI. There, the `MatMulInteger` kernel of ONNX Runtime adds each pair of u8 × s8 products in int16, and a pair above 32767 saturates. Transmuter gives the exact int32 sum. In `Xenova/falcon-rw-1b` (int8), one value of the layer 3 `query_key_value` product differs by 912, and the logits then differ by up to 1.46. A numpy product with the int16 saturation gives the output of ONNX Runtime exactly. ONNX Runtime on arm64 (macOS) gives the result of Transmuter, with logits within 1.9e-5. In the int8 `Xenova/slovak-gpt-j-405M`, 9 of the 145 `MatMulInteger` nodes in the `If` branch of the merged decoder saturate on x86, from the layer 0 `k_proj` on, and the numpy emulation gives the output of ONNX Runtime exactly for each of them. On arm64, all 145 products are exact, and one `DynamicQuantizeLinear` value of layer 8 rounds to the other integer: its x / scale is -1.4999986, 1.4e-6 from a .5 tie. The float differences of about 1e-6 between two machines decide such a tie. The logits then differ by up to 0.74 on arm64 and 0.83 on x86.

## Coverage

The following table shows the result of running Transmuter against ONNX Runtime for each tested corpus:

| Corpus | Result |
|---|---|
| Top exports for each Hugging Face task | 75 of 77 repositories, 112 of 114 files match |
| q4, int8, and fp16 variants of 10 repositories | all 12 q4 files match. int8 and fp16 are within the noise of ONNX Runtime |
| Models requested in Bumblebee issues, with Apertus, a tiny Qwen3-MoE for the routing of Mixtral, GPT-J and Falcon | 15 of 18 files match. The Stable Diffusion VAE encoder samples noise. The int8 Falcon and GPT-J files differ from ONNX Runtime on x86, whose int8 kernel is not exact. The Falcon file matches ONNX Runtime on arm64, and the GPT-J file also has one rounding tie |
| Image classifiers from the ONNX Model Zoo, with int8 VGG16, AlexNet, and ZFNet | 7 of 7 match |
| Faster R-CNN 10 and 12 and Mask R-CNN 12 from the ONNX Model Zoo | 3 of 3 match |
| One graph for each operator variant | 255 of 255 match |

"Within noise" means the output differs from ONNX Runtime by its own rounding noise.

## Running the coverage matrix

1. Run `bench/coverage/select_models.py` to select the models.
2. Run `bench/coverage/record_references.py` to download them and record ONNX Runtime outputs.
3. Run `bench/coverage/run_cases.py` to run each case through Transmuter.
4. Run `bench/coverage/print_matrix.py` to show the matrix.
5. Run `bench/coverage/list_models.py` to make the table of tested models.

When a case does not match, run `bench/coverage/trace_ort.py` and `bench/coverage/find_divergence.exs`. They show the first node where Transmuter and ONNX Runtime give different values.

## Missing operators

If you load a model and get an error about an unimplemented operator, open an issue on [GitHub](https://github.com/georgeguimaraes/transmuter) with the operator name and a link to the ONNX file.
