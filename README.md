# Transmuter

Transmuter runs ONNX models on Nx. Load an ONNX export from the Hugging Face Hub and run it on any Nx backend: EMLX on Apple silicon, EXLA on CUDA or CPU.

Transmuter does not port model architectures. The ONNX file contains the forward pass that the exporter traced from the original model. Transmuter turns each ONNX operator into Nx operations when the function is traced. Thus a new architecture needs no new Elixir code, only the operators that its export uses.

## Example

```elixir
model = Transmuter.load!({:hf, "Xenova/all-MiniLM-L6-v2", "onnx/model.onnx"})
params = Map.new(model.params, fn {name, t} -> {name, Nx.backend_transfer(t, EMLX.Backend)} end)

inputs = %{
  "input_ids" => Nx.tensor([[101, 7592, 2088, 102]]),
  "attention_mask" => Nx.tensor([[1, 1, 1, 1]]),
  "token_type_ids" => Nx.tensor([[0, 0, 0, 0]])
}

run = Nx.Defn.compile(&Transmuter.run(model, &1, &2), [params, inputs], compiler: EMLX)
%{"last_hidden_state" => hidden} = run.(params, inputs)
```

Shapes are resolved when the function is traced. Each new input shape compiles again, so keep one compiled function per shape. `Nx.Defn.jit/2` with a closure over the model compiles on every call.

## Static inputs

Some graphs read input values to decide a shape or a branch. Give these inputs as `Nx.BinaryBackend` tensors in the fourth argument of `Transmuter.run/4`:

- `use_cache_branch` in merged decoder exports, which selects an `If` branch.
- `num_logits_to_keep` in some decoder exports, which sets a slice length.
- `input_ids` in Grounding DINO, which feed `NonZero`.

Each new value of a static input compiles again.

## Precision and quantization

Half-precision graphs (`model_fp16.onnx`) run in f16 or bf16. Sums, norms, softmax and attention scores accumulate in f32 and cast back. ONNX Runtime on the CPU computes half-precision graphs in f32, so `Transmuter.load!(source, compute_type: :f32)` does the same when you need to match it.

Quantized exports run with their weights dequantized inside the graph:

- `model_q4.onnx` and `model_q4f16.onnx`: 4-bit weights with `MatMulNBits` and `GatherBlockQuantized`, including int4 tables.
- `model_quantized.onnx` and `model_int8.onnx`: dynamic int8 with `DynamicQuantizeLinear`, `MatMulInteger` and `ConvInteger`.

On EMLX, use `Transmuter.EMLX.params(model)` instead of moving `model.params` yourself. It keeps `MatMulNBits` weights packed as MLX quantized tensors, so `mx::quantized_matmul` reads the 4-bit or 8-bit weights directly. Without it, the weights are dequantized to floats in every forward pass. Measured on an Apple silicon GPU with one 8-token forward pass:

| Model | Weights | Peak memory | Time |
|---|---|---|---|
| Qwen3-0.6B fp32 | 2,294 MB | 2,321 MB | 34 ms |
| Qwen3-0.6B q4, dequantized in the graph | 876 MB | 1,612 MB | 64 ms |
| Qwen3-0.6B q4, `Transmuter.EMLX.params` | 981 MB | 1,005 MB | 28 ms |

Dynamic int8 requantizes every activation. A change of one f32 rounding step can move a value to the next int8 step, and later layers amplify that change. Compare these models with their own noise: scaling all weights of SmolLM2 int8 by 1 + 1e-7 changes its logits by 22.

## Coverage

`bench/coverage` runs the most downloaded ONNX exports for each Hub task and compares every output with ONNX Runtime. The reference is ONNX Runtime with graph optimizations disabled, because its default optimizer gives wrong results for some exports. For `MatMulNBits`, the reference uses `accuracy_level` 0. Level 4 lets ONNX Runtime quantize activations to int8, which is an approximation.

In the last run, 75 of 77 repositories and 112 of 114 files matched ONNX Runtime. The two other repositories are:

- Grounding DINO: it runs, but the outputs do not match by position. With random inputs, its `TopK` proposal selection changes when the input has noise of 1e-6. ONNX Runtime shows the same difference (0.85 on boxes, 2.7 on logits) when it compares against itself with that noise.
- Kokoro text-to-speech: it does not run. Its output length comes from predicted phoneme durations, and a `Range` on that value has a shape that is only known at run time.

The quantized and half-precision variants of 10 repositories (33 files) are in `bench/coverage/corpus_variants.json`. All 12 q4 files match. The int8 and fp16 files differ from ONNX Runtime by the same amount that ONNX Runtime differs from itself (int8), or are closer to the f32 model than ONNX Runtime is (fp16).

`bench/coverage/op_cases.py` builds one small graph for each operator variant and records the ONNX Runtime output. All 60 cases match.

To run the matrix:

1. Run `bench/coverage/select_corpus.py` to select the models.
2. Run `bench/coverage/prepare.py` to download them and record the ONNX Runtime outputs.
3. Run `bench/coverage/run_all.py` to run each case through Transmuter.
4. Run `bench/coverage/report.py` to print the matrix.

If a case does not match, run `bench/coverage/debug_ort.py` and `bench/coverage/debug_case.exs`. They show the first node where Transmuter and ONNX Runtime give different values.

## Tests

`mix test` downloads four models and compares them with ONNX Runtime. `test/support/reference.py` prints the reference values.
