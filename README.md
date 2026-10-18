# ⚗️ Transmuter

Transmuter runs ONNX models on Nx. Load a model from Hugging Face and run it on any Nx backend: EMLX on Apple silicon, or EXLA on CUDA or on the CPU.

Transmuter does not port model architectures. An ONNX file contains the forward pass that the exporter traced from the original model. Transmuter changes each ONNX operator into Nx operations. A new architecture needs no new Elixir code. It needs only the operators that its export uses.

## The gap

Elixir has three ways to run Hugging Face models today. Each way has a limit:

- **Bumblebee** runs models with good accuracy on all Nx backends. But a person must port each architecture to Axon by hand, and a port takes weeks of work and review. Hugging Face gets new architectures faster than people can port them.
- **Ortex** runs ONNX models with ONNX Runtime. But it runs outside of Nx: you cannot put it in a `defn` function, it has no Apple silicon GPU path, and it is in maintenance mode.
- **axon_onnx** changed ONNX graphs into Axon models. It stopped in 2024. It set all shapes when it loaded a model, and many ONNX graphs need shapes that change.

Transmuter fills this gap. It runs the exported graph directly as Nx operations, so:

- Models that Bumblebee does not have run with no new code. Examples are SAM, Depth Anything, Grounding DINO, OWLv2, Florence-2, RF-DETR, Kokoro and Stable Diffusion.
- The result is plain Nx. It runs on EMLX and EXLA, it works with `Nx.Serving`, and `Nx.Defn.grad` can differentiate it.
- Shapes are set when the function compiles, not when the model loads. Thus decoders with a KV cache and graphs with `If` branches work.

When a model has no ONNX file on Hugging Face, export it one time with [Optimum](https://huggingface.co/docs/optimum/exporters/onnx/usage_guides/export_a_model). In the 300 most downloaded transformers models, 62% of downloads already have an ONNX file on Hugging Face.

## Example

```elixir
model = Transmuter.load!({:hf, "Xenova/all-MiniLM-L6-v2", "onnx/model.onnx"}, backend: EMLX.Backend)

inputs = %{
  "input_ids" => Nx.tensor([[101, 7592, 2088, 102]]),
  "attention_mask" => Nx.tensor([[1, 1, 1, 1]]),
  "token_type_ids" => Nx.tensor([[0, 0, 0, 0]])
}

run = Transmuter.compile(model, inputs, compiler: EMLX)
%{"last_hidden_state" => hidden} = run.(inputs)
```

`Transmuter.compile/3` compiles the model for the shapes of the inputs that you give. For other shapes, compile again.

`Transmuter.run/4` is the function that `Transmuter.compile/3` compiles. It reads the ONNX graph and calls one Nx function for each operator. You can use it in two ways:

- Call it directly. Each operator runs immediately, one after the other. This is eager execution. It is slower, but the model can read values while it runs.
- Give it to `Nx.Defn.jit/2` or `Nx.Defn.value_and_grad/2`. Nx then records the operators as one graph, and the backend compiles that graph.

## Installation

```elixir
def deps do
  [
    {:transmuter, "~> 0.1"},
    {:emlx, "~> 0.5"}
  ]
end
```

Use EXLA instead of EMLX for CUDA.

## Static inputs and eager execution

Some graphs use the value of an input to set a shape or to select a branch. Give these inputs as `Nx.BinaryBackend` tensors in the `:static_inputs` option of `Transmuter.compile/3`:

- `use_cache_branch` in merged decoder exports selects an `If` branch.
- `num_logits_to_keep` in some decoder exports sets the length of a slice.
- `input_ids` in Grounding DINO go into `NonZero`.

Each new value of a static input compiles the model again.

Some graphs calculate a shape inside the graph. For these graphs, call `Transmuter.run/4` without compilation. Then Transmuter reads the values while the model runs. Kokoro text-to-speech needs this, because the length of its audio comes from the predicted duration of each phoneme. On an Apple silicon GPU, Kokoro makes 4.45 seconds of audio in 1.4 seconds.

## Precision and quantization

Half-precision exports (`model_fp16.onnx`) run in f16 or bf16. Sums, normalizations, softmax and attention scores use f32 inside the operation. ONNX Runtime on the CPU calculates half-precision graphs in f32. To get the same result, use `Transmuter.load!(source, compute_type: :f32)`.

Quantized exports run with their weights dequantized in the graph:

- `model_q4.onnx` and `model_q4f16.onnx` use 4-bit weights with `MatMulNBits` and `GatherBlockQuantized`.
- `model_quantized.onnx` and `model_int8.onnx` use dynamic int8 with `DynamicQuantizeLinear`, `MatMulInteger` and `ConvInteger`.

With `backend: EMLX.Backend`, `Transmuter.load!/2` keeps `MatMulNBits` weights packed as MLX quantized tensors. Then `mx::quantized_matmul` reads the 4-bit or 8-bit weights directly. On other backends, each forward pass dequantizes the weights to floats. These results are from an Apple silicon GPU with one forward pass of 8 tokens:

| Model | Weights | Peak memory | Time |
|---|---|---|---|
| Qwen3-0.6B fp32 | 2,294 MB | 2,321 MB | 34 ms |
| Qwen3-0.6B q4, dequantized in the graph | 876 MB | 1,612 MB | 64 ms |
| Qwen3-0.6B q4, packed on EMLX | 981 MB | 1,005 MB | 28 ms |

Dynamic int8 quantizes each activation again. A change of one f32 rounding step can move a value to the next int8 step, and the next layers make that change larger. For example, when all weights of SmolLM2 int8 change by a factor of 1 + 1e-7, its logits change by 22. Thus, compare an int8 model with its own noise.

## Fine-tuning

`Transmuter.run/4` is plain Nx. Thus `Nx.Defn.value_and_grad/2` calculates gradients through the graph, and an optimizer such as Polaris updates the params. Load the model with `trainable: true`. Then small weights, such as a bias with 1 element, are params and not constants.

`bench/finetune.exs` fine-tunes a MiniLM cross-encoder. Its gradients and its loss curve for 30 AdamW steps are the same as in PyTorch.

Fine-tune an fp32 export. Inference exports have no dropout, and quantized weights have no gradients.

## Coverage

`bench/coverage` runs the most downloaded ONNX exports for each Hugging Face task. It compares each output with ONNX Runtime. The reference is ONNX Runtime with graph optimizations off, because its default optimizer gives wrong results for some exports. For `MatMulNBits`, the reference uses `accuracy_level` 0. At level 4, ONNX Runtime quantizes the activations to int8, and that result is an approximation.

| Corpus | Result |
|---|---|
| Top exports for each Hugging Face task | 75 of 77 repositories, 112 of 114 files match |
| q4, int8 and fp16 variants of 10 repositories | all 12 q4 files match; int8 and fp16 are within the noise of ONNX Runtime |
| Models requested in Bumblebee issues | 11 of 12 files match |
| One graph for each operator variant | 60 of 60 match |

These models do not match by position, or do not run with compilation:

- **Grounding DINO** runs. But with random inputs, its `TopK` selection changes when the input has noise of 1e-6. ONNX Runtime has the same difference when it compares with itself.
- **Kokoro** runs only in eager mode, because its audio length is calculated in the graph.

To run the matrix:

1. Run `bench/coverage/select_corpus.py` to select the models.
2. Run `bench/coverage/prepare.py` to download them and record the ONNX Runtime outputs.
3. Run `bench/coverage/run_all.py` to run each case through Transmuter.
4. Run `bench/coverage/report.py` to show the matrix.

If a case does not match, run `bench/coverage/debug_ort.py` and `bench/coverage/debug_case.exs`. They show the first node where Transmuter and ONNX Runtime give different values.

## Tests

`mix test` downloads five models and compares them with ONNX Runtime. `test/support/reference.py` shows the reference values. On a Mac, `TRANSMUTER_TEST_DEVICE=cpu mix test` runs the tests on the CPU, as the Linux CI does.

## License

Apache-2.0
