# ⚗️ Transmuter

Transmuter runs ONNX models on Nx. Load a model from Hugging Face and run it on any Nx backend: EMLX on Apple silicon, or EXLA on CUDA or on the CPU.

The goal of Transmuter is to implement the ONNX operators in Nx. Then ONNX models run without model code.

An ONNX file contains a graph of operators. An operator is one small, standard computation, such as `MatMul`, `Conv`, `Softmax`, `Add` or `Reshape`. Each operator has inputs, outputs and attributes, for example the kernel size of a `Conv`. The ONNX standard defines about 200 operators, and a model connects a few hundred to a few thousand of them. When a model is exported, the exporter traces its forward pass and writes these operators to the file.

Transmuter implements each operator one time, as Nx operations. It then runs a model by running its graph, operator by operator. So Transmuter does not port model architectures. A new architecture needs no new Elixir code. It needs only operators that Transmuter already has.

## The gap

Elixir has three ways to run Hugging Face models today. Each way has a limit:

- **Bumblebee** runs models with great accuracy on all Nx backends. But a person must port each architecture to Axon by hand, and a port takes weeks of work and review. Hugging Face gets new architectures faster than people can port them.
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

# Sentence embedding: mean of the token vectors, then L2 normalization.
mask = Nx.new_axis(inputs["attention_mask"], -1)
embedding = Nx.sum(Nx.multiply(hidden, mask), axes: [1]) |> Nx.divide(Nx.sum(mask, axes: [1]))
embedding = Nx.divide(embedding, Nx.LinAlg.norm(embedding, axes: [1], keep_axes: true))
```

The export sets the names of the inputs and outputs. `model.inputs` and `model.outputs` show them. For this export, `model.outputs` is `["last_hidden_state"]`.

Transmuter runs the graph that is in the file, and nothing more. The graph of all-MiniLM-L6-v2 stops at the token vectors, because sentence-transformers adds the pooling outside of the model. Thus the example above does the pooling. Bumblebee servings, such as `Bumblebee.Text.text_embedding/3`, also do the tokenization, pooling and batching for you. Transmuter has no servings yet. For tokenization, use [tokenizers](https://github.com/elixir-nx/tokenizers).

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

## Values that decide shapes

Some graphs use a value to set a shape or to select a branch. Examples are the length of the audio in Kokoro text-to-speech, which comes from the predicted duration of each phoneme, and the boxes that `NonMaxSuppression` keeps in Faster R-CNN. A compiled graph must know all its shapes before it runs, so `Transmuter.compile/3` breaks the graph at these operators:

1. It compiles the operators before the break and runs them.
2. It runs the break operator on the values.
3. It compiles the operators after the break for the shapes that the values decide, and continues.

The compiled parts are cached, so a later call with the same shapes runs only compiled code. The cache keeps the 8 most recently used compiled parts for each break. A new shape after that replaces the part that was used least recently. The cache belongs to the process that calls `Transmuter.compile/3`, for example the `Nx.Serving` process.

These results are from an Apple silicon GPU:

| Model | Without compilation | First call | Later calls |
|---|---|---|---|
| Kokoro, 7.0 seconds of audio | 2.3 s | 2.1 s | 0.34 s |
| Kokoro, 20.2 seconds of audio | 13.2 s | 8.8 s | 0.87 s |
| Faster R-CNN 12, the same image | 33.8 s | 6.5 s | 0.35 s |

A new audio length in Kokoro compiles the operators after the break again, and that first call takes about as long as a call without compilation. `bench/kokoro_tts.exs` rounds the total duration up to steps of 200 ms by making the final pause longer, so sentences of a similar length share one compiled decoder.

EMLX has caches of its own, and each new shape adds to them:

- MLX keeps freed buffers for reuse. Without a limit, this cache grew to 11.7 GB on a 16 GB Mac, and later calls for 20.2 seconds of audio took 22.5 s instead of 0.87 s. Set a limit with `EMLX.set_cache_limit(1_000_000_000)`.
- `EMLX.CompileCache` keeps each compiled function. Set `max_items` for it in the `:emlx` config. EMLX also keeps each MLX program, with no limit.

If you know such a value before the model runs, give it as an `Nx.BinaryBackend` tensor in the `:static_inputs` option of `Transmuter.compile/3`. Then the graph has no break there. Each new value of a static input compiles the model again. Examples:

- `use_cache_branch` in merged decoder exports selects an `If` branch.
- `num_logits_to_keep` in some decoder exports sets the length of a slice.
- `input_ids` in Grounding DINO go into `NonZero`.

`Transmuter.run/4` without compilation reads all values while the model runs.

## Strings, sequences and random values

Nx tensors hold only numbers. Transmuter keeps other ONNX values in the graph as Elixir terms:

- A string tensor is `{:strings, shape, values}`, with the values as a flat list of binaries.
- A sequence is a list of tensors.
- An empty optional is `:none`.
- A tensor with a dimension of 0 is `{:empty, shape}`, because Nx has no zero-size tensors.

Give string and sequence inputs as `:static_inputs`. A compiled function returns only tensors, so call `Transmuter.run/4` without compilation to read string or sequence outputs.

Random operators draw from `Nx.Random`. Each call of a compiled model draws new values. A node with a `seed` attribute draws the same values on each call. The values cannot be the same as in ONNX Runtime, because the random number generators are different.

`ImageDecoder` needs the optional `stb_image` dependency.

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
| Image classifiers from the ONNX Model Zoo, with the int8 VGG16, AlexNet and ZFNet | 8 of 8 match |
| Faster R-CNN 10 and 12 and Mask R-CNN 12 from the ONNX Model Zoo | 3 of 3 match |
| One graph for each operator variant | 222 of 222 match |

Transmuter supports all 203 operators of the ONNX standard domain, and the ONNX Runtime contrib operators that Hugging Face exports use.

Two models do not match value by value, because they make small rounding differences large:

- **Grounding DINO** selects 900 of 13,294 queries with `TopK`, and some scores differ by only 9.5e-7. All operators before `TopK` match. On a real image with a text prompt, Transmuter gives the same detections as ONNX Runtime, with scores within 0.0001.
- **Kokoro** sums a phase over all audio samples and takes its STFT. ONNX Runtime and MLX round that sum differently, and phases near zero amplitude change by up to π. When the test gives the `Sin` and `STFT` outputs of ONNX Runtime to Transmuter, all other operators match. On real speech, Transmuter is as close to ONNX Runtime as PyTorch is.

To run the matrix:

1. Run `bench/coverage/select_models.py` to select the models.
2. Run `bench/coverage/record_references.py` to download them and record the ONNX Runtime outputs.
3. Run `bench/coverage/run_cases.py` to run each case through Transmuter.
4. Run `bench/coverage/print_matrix.py` to show the matrix.

If a case does not match, run `bench/coverage/trace_ort.py` and `bench/coverage/find_divergence.exs`. They show the first node where Transmuter and ONNX Runtime give different values.

## Tests

`mix test` downloads five models and compares them with ONNX Runtime. `test/support/model_references.py` shows the reference values.

`mix test` also runs one small graph for each operator variant in `test/fixtures/ops`, with no download. `test/support/make_op_fixtures.py` makes these graphs and records their outputs from ONNX Runtime, or from the ONNX reference evaluator when ONNX Runtime does not have the operator. For random operators, the tests check the shape, the type and the distribution of the values.

On a Mac, `TRANSMUTER_TEST_DEVICE=cpu mix test` uses the EMLX CPU device. But EMLX 0.5 runs compiled functions on the GPU when Metal is available, also with `device: :cpu`. Thus only the Linux CI tests the CPU fully.

## License

Apache-2.0
