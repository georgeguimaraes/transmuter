# Getting Started

This guide shows how to install Transmuter, load an ONNX model from the Hugging Face Hub, and run it with EMLX or EXLA.

For text, image and speech models, the servings of `Transmuter.Text`, `Transmuter.Vision` and `Transmuter.Audio` take text, images and audio, and give results. Start with the [Pipelines guide](pipelines.md) for these tasks. This guide shows the lower level: tensors in, tensors out.

## Installation

Add Transmuter and a backend to your dependencies in `mix.exs`.

For Apple silicon, use EMLX:

```elixir
def deps do
  [
    {:transmuter, "~> 1.0"},
    {:emlx, "~> 0.5"}
  ]
end
```

For NVIDIA GPUs, use EXLA:

```elixir
def deps do
  [
    {:transmuter, "~> 1.0"},
    {:exla, "~> 1.0"}
  ]
end
```

The servings need the optional `tokenizers` package for text, and `stb_image` to read image files:

```elixir
{:tokenizers, "~> 0.5"},
{:stb_image, "~> 1.0"}
```

Run `mix deps.get` to install.

## Load a model

Load a model from the Hugging Face Hub with the `{:hf, repo, file}` tuple:

```elixir
model = Transmuter.load!(
  {:hf, "Xenova/all-MiniLM-L6-v2", "onnx/model.onnx"},
  backend: EMLX.Backend
)
```

Transmuter downloads the ONNX file to `~/.cache/transmuter` and keeps it for later calls. Set `:cache_dir` to change the download location. Set `offline: true`, or the `HF_HUB_OFFLINE=1` environment variable, to use only the files in the cache.

Load a model from a local path:

```elixir
model = Transmuter.load!("/path/to/model.onnx", backend: EMLX.Backend)
```

Some large models split their weights into external data files. The files must be in the same directory as the ONNX file, with names such as `model.onnx_data` or `model.onnx_data_1`, `model.onnx_data_2`. When you load from the Hugging Face Hub, Transmuter downloads these files automatically alongside the ONNX file.

## Inspect the model

After loading, check the input and output names of the model:

```elixir
model.inputs
# => ["input_ids", "attention_mask", "token_type_ids"]

model.outputs
# => ["last_hidden_state"]
```

Use these names to build the input map and to read the output map. The exporter of the ONNX file sets these names.

## Run the model eagerly

Call `Transmuter.run/4` to run the model without compilation. Each operator runs immediately, one after the other.

```elixir
inputs = %{
  "input_ids" => Nx.tensor([[101, 7592, 2088, 102]], type: :s64, backend: EMLX.Backend),
  "attention_mask" => Nx.tensor([[1, 1, 1, 1]], type: :s64, backend: EMLX.Backend),
  "token_type_ids" => Nx.tensor([[0, 0, 0, 0]], type: :s64, backend: EMLX.Backend)
}

outputs = Transmuter.run(model, model.params, inputs)
hidden = outputs["last_hidden_state"]
```

Eager execution is slower than compilation, but the model can read intermediate values while it runs. This is useful when a model has graph breaks (see [Compiling and Graph Breaks](graph_breaks.md)) and you want to skip compilation.

## Compile the model

Call `Transmuter.compile/3` to compile the model for the shapes of the given inputs. Pass the `compiler: EMLX` option to compile with EMLX:

```elixir
run = Transmuter.compile(model, inputs, compiler: EMLX)
outputs = run.(inputs)
hidden = outputs["last_hidden_state"]
```

The returned function caches compiled shapes. When you call it with inputs of a different shape, it compiles those shapes on the first call and reuses them on later calls. Use `:cache_size` to set how many compiled shapes to keep per graph segment. The default is 8.

To use EXLA:

```elixir
model = Transmuter.load!({:hf, repo, file}, backend: {EXLA.Backend, client: :cuda})
run = Transmuter.compile(model, inputs, compiler: EXLA, client: :cuda)
```

## What Transmuter does and does not do

Transmuter runs the graph that is in the ONNX file, and nothing more. If the graph stops at the token vectors (as in the MiniLM export above), the pooling step is not in the file and you must do it yourself.

For text models, the servings of `Transmuter.Text` do the tokenization, the pooling and the batching for you. See the [Pipelines guide](pipelines.md).

If a model has no ONNX file on Hugging Face, export it once with [Optimum](https://huggingface.co/docs/optimum/exporters/onnx/usage_guides/export_a_model):

```
optimum-cli export onnx --model <repo> <output_dir>
```
