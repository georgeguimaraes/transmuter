# Language Models with a KV Cache

This guide explains how to run a decoder language model with a static key-value (KV) cache, and how to run quantized ONNX exports.

To generate text without the details of this guide, use `Transmuter.Text.generation/2`. It does the tokenization, the chat template, the KV cache and the sampling. See the [Pipelines guide](pipelines.md). This guide is for the work that the serving does not do, or to control each step.

## How decoder exports work

Hugging Face exporters produce one or more ONNX files for a decoder model:

| File | What it does |
|---|---|
| `model.onnx` | One file, static cache pattern |
| `decoder_model.onnx` | Decoder without KV cache inputs |
| `decoder_with_past_model.onnx` | Decoder that reads the KV cache |
| `decoder_model_merged.onnx` | Both of the above combined with an `If` branch |

The merged file is the most common choice. It uses `use_cache_branch` to select the prefill path (no cache) or the decode path (with cache).

## Static cache pattern

A static KV cache keeps all positions in a fixed-size buffer. Every decode step has the same input shapes, so the compiled function is reused every step without recompilation.

The following example runs SmolLM2-135M-Instruct with a static cache. The cache buffer holds `max_len` positions. Each step writes the new key-value pair into the next position.

```elixir
repo = "HuggingFaceTB/SmolLM2-135M-Instruct"
backend = {EMLX.Backend, device: :gpu}
max_len = 128

model = Transmuter.load!({:hf, repo, "onnx/model.onnx"}, backend: backend)

# Build an empty cache buffer for each past_key_values input.
empty_cache =
  for name <- model.inputs, String.starts_with?(name, "past_key_values"), into: %{} do
    {name, Nx.broadcast(Nx.tensor(0.0, backend: backend), {1, 3, max_len, 64})}
  end

# Build the full input map for one step.
step_inputs = fn ids, pos, cache ->
  n = pos + length(ids)

  Map.merge(cache, %{
    "input_ids" => Nx.tensor([ids], type: :s64, backend: backend),
    "attention_mask" =>
      Nx.tensor(
        [Enum.map(0..(max_len - 1), &if(&1 < n, do: 1, else: 0))],
        type: :s64,
        backend: backend
      ),
    "position_ids" =>
      Nx.tensor([Enum.to_list(pos..(n - 1))], type: :s64, backend: backend)
  })
end

# Compile two functions: one for the prefill, one for the decode step.
# Nx.Defn.compile takes params and inputs as separate arguments so that
# the same params work with both shapes.
fun = &Transmuter.run(model, &1, &2)
prefill = Nx.Defn.compile(fun, [model.params, step_inputs.(prompt_ids, 0, empty_cache)], compiler: EMLX)
decode = Nx.Defn.compile(fun, [model.params, step_inputs.([0], length(prompt_ids), empty_cache)], compiler: EMLX)
```

After compilation, run the prefill step, then run the decode step once per new token:

```elixir
# Run the prefill step with the prompt.
out = prefill.(model.params, step_inputs.(prompt_ids, 0, empty_cache))

# Extract the KV cache from the output.
next_cache = fn out ->
  for {"present." <> rest, t} <- out, into: %{}, do: {"past_key_values." <> rest, t}
end

# Extract the next token.
next_token = fn out -> out["logits"][[0, -1]] |> Nx.argmax() |> Nx.to_number() end

# Run decode steps.
Enum.reduce(1..new_tokens, {[next_token.(out)], next_cache.(out)}, fn i, {tokens, cache} ->
  out = decode.(model.params, step_inputs.([hd(tokens)], length(prompt_ids) + i - 1, cache))
  {[next_token.(out) | tokens], next_cache.(out)}
end)
```

The decode step uses `Nx.Defn.compile` on `Transmuter.run/4` directly. This pattern compiles the model without a graph break because the cache size does not change between steps.

## Using Transmuter.compile/3 with a static cache

You can also use `Transmuter.compile/3` if you do not need to separate params from inputs:

```elixir
flag = Nx.tensor([1], type: :u8, backend: Nx.BinaryBackend)

run = Transmuter.compile(
  model,
  step_inputs.(prompt_ids, 0, empty_cache),
  static_inputs: %{"use_cache_branch" => flag},
  compiler: EMLX
)
```

Set `use_cache_branch` to `[1]` for the decode path and `[0]` for the prefill path. Each value compiles a separate version, cached by the value.

## Quantized exports

Hugging Face provides quantized exports for most decoder models. Use the filename to select the format:

| File | Format | Notes |
|---|---|---|
| `model_q4.onnx` | 4-bit weights with `MatMulNBits` | Smallest file, fast on EMLX |
| `model_q4f16.onnx` | 4-bit weights, activations in f16 | |
| `model_quantized.onnx` | Dynamic int8 with `MatMulInteger` | Close to f32 quality |
| `model_int8.onnx` | Same as `model_quantized.onnx` | |
| `model_fp16.onnx` | Half-precision weights | Runs in native f16, see `:compute_type` |

Load a quantized model the same way as a full-precision model:

```elixir
model = Transmuter.load!(
  {:hf, "onnx-community/Qwen3-0.6B-ONNX", "onnx/model_q4.onnx"},
  backend: EMLX.Backend
)
```

With `backend: EMLX.Backend`, Transmuter keeps the 4-bit weights packed as MLX quantized tensors. MLX calls `mx::quantized_matmul` directly on the packed weights, so the float weights never exist in memory. On every other backend, such as EXLA, the weights become `{N, K}` float tensors when the model loads, so each forward pass runs one matmul for each weight and does not unpack it. These float weights take about 4 bytes per weight: Qwen3-1.7B q4 takes about 6.9 GB, and a 4B q4 model needs about 16 GB. A model loaded without a backend dequantizes the weights in the graph on each forward pass. See [Backends and performance](backends_and_performance.md).

Performance comparison on Apple silicon GPU, one forward pass of 8 tokens:

| Model | Weights | Peak memory | Time |
|---|---|---|---|
| Qwen3-0.6B fp32 | 2,294 MB | 2,321 MB | 34 ms |
| Qwen3-0.6B q4, dequantized in the graph | 876 MB | 1,612 MB | 64 ms |
| Qwen3-0.6B q4, packed on EMLX | 981 MB | 1,005 MB | 28 ms |

## Half-precision exports

Half-precision exports (`model_fp16.onnx`) run in native f16 by default. Many fp16 decoder exports write RMSNorm as separate operators (`Pow`, `ReduceMean`, `Add`, `Sqrt`). In native f16, `Pow` overflows for values above 256. When the model loads, `Transmuter.Fusion` replaces the subgraphs that it recognizes with one normalization operator, which runs in f32. Use `compute_type: :f32` to run the model in f32, as ONNX Runtime on the CPU does:

```elixir
model = Transmuter.load!({:hf, repo, "onnx/model_fp16.onnx"}, compute_type: :f32)
```

See [Backends and Performance](backends_and_performance.md) for more on compute types.

## Dynamic int8 precision

Dynamic int8 exports quantize each activation at runtime. A change of one f32 rounding step can move a value to the next int8 step, and the next layers make that change larger. Compare an int8 model with its own noise, not with a float model.

## Fine-tuning

`Transmuter.run/4` is plain Nx. Pass `trainable: true` to `Transmuter.load!/2` to make small weights (such as a 1-element bias) trainable params instead of constants. Then `Nx.Defn.value_and_grad/2` calculates gradients through the graph. Fine-tune on fp32 exports: inference exports have no dropout, and quantized weights have no gradients.

See `bench/finetune.exs` for an example that fine-tunes a MiniLM cross-encoder with AdamW.
