# Compiling and Graph Breaks

This guide explains how `Transmuter.compile/3` works, what graph breaks are, and how to control them with static inputs.

## How compilation works

`Transmuter.compile/3` compiles the model graph for the shapes and types of the inputs you give:

```elixir
run = Transmuter.compile(model, inputs, compiler: EMLX)
```

It returns a function. The function takes the same inputs and returns the outputs:

```elixir
outputs = run.(inputs)
```

Compilation turns the Nx operations in the model into a single program that the backend (EMLX or EXLA) runs. Later calls with the same input shapes run the compiled program directly. Calls with different input shapes compile those shapes on their first call.

The `:cache_size` option sets how many compiled shapes to keep per graph segment. The default is 8. When you reach the limit, the least recently used shape is removed. Set `:cache_size` to a larger value when the inputs have many shapes that come back often, such as audio lengths in a speech service:

```elixir
run = Transmuter.compile(model, inputs, compiler: EMLX, cache_size: 64)
```

## What a graph break is

A compiled graph must know all its shapes before it runs. Some operators use a tensor value to decide a shape or to select a branch. Examples:

- `Range` uses the upper limit to set the length of its output.
- `NonZero` returns the positions of all non-zero elements. The count is not known until the values arrive.
- `If` selects one of two subgraphs based on a condition tensor.
- `NonMaxSuppression` returns the boxes that pass the threshold. The count depends on the input scores.

When the compiler reaches one of these operators and the value is not yet known (because it depends on the model inputs), it raises `Transmuter.UnknownValueError`. Transmuter catches this error and creates a graph break at that operator.

Transmuter handles a graph break in three steps:

1. It compiles and runs the operators before the break.
2. It runs the break operator on the real output values of step 1.
3. It compiles the operators after the break for the shapes that the values decide, and continues.

Compiled segments are cached, so a later call with the same shapes does not compile again. The break operators still run eagerly on each call. A segment of fewer than 20 nodes after a break is not compiled, and it also runs eagerly on each call.

## What goes into the key of a segment

The cache key of a segment has these parts:

- The position of the segment in the graph.
- The shapes and types of its arguments.
- The values of the static values that it reads.

The small values that a break node reads, such as the positions that a `NonZero` returns, become host values. Values that the graph computes from them on the host are host values too. A host value goes into the next segment as an argument when no node of that segment reads it on the host. Then its value is not in the key. For example, a `Gather` that reads the word positions of a text model compiles once for each number of words, and not once for each text.

A host value stays static in a segment when a node of that segment needs it on the host, for example as the shape of a `Reshape` or the start of a `Slice`. Then its value is in the key of that segment, and each new value compiles that segment again. The keys of the other segments do not change. Transmuter finds these values when it traces a segment: the node breaks on the argument, and Transmuter traces the segment again with the value static. A small value that an earlier segment computes from host values, such as the first of the positions, is copied to the host when the segment starts. A value that is too large to copy, or that comes from a random operator, stays an argument, and the node breaks there as a graph break. While Transmuter learns these values, the first calls with new values can compile a segment once more. After that, a new value compiles a segment again only when it is in the key. A segment that runs eagerly reads host values without a compilation, so its values stay static. The `:host_values` of the `[:transmuter, :segment, :compile, :stop]` event give the names that a trace found (see `Transmuter.Telemetry`).

Static inputs always stay static, so each new value of a static input is in the key.

## Reading the error

When a graph break occurs, Transmuter re-raises the error if it cannot handle it:

```
Transmuter.UnknownValueError: NonZero NonZero_0: input_ids is only known when the model runs.
Pass the graph inputs it depends on as static_inputs, or call Transmuter.run/4 without compiling.
```

The error names the operator type, the node name, and the tensor name. To fix it, either pass the tensor as a static input (see below) or use `Transmuter.run/4` without compilation.

## Static inputs

A static input is a tensor whose value is known when the model compiles. Pass static inputs in the `:static_inputs` option:

```elixir
run = Transmuter.compile(
  model,
  inputs,
  static_inputs: %{"use_cache_branch" => Nx.tensor([1], type: :u8, backend: Nx.BinaryBackend)},
  compiler: EMLX
)
```

Give static inputs as `Nx.BinaryBackend` tensors. Each new value of a static input compiles the model again.

Common uses for static inputs:

- `use_cache_branch` in merged decoder exports. It selects an `If` branch for the prefill or the decode step.
- `num_logits_to_keep` in some decoder exports. It sets the length of a slice.
- `input_ids` in Grounding DINO. They feed `NonZero`, which counts the non-zero elements.

When a static input changes, Transmuter compiles a new version of the model. With `:cache_size` segments cached, a value that comes back later reuses its compiled version.

## See the segments

`Transmuter.Graph` shows where the graph breaks and why, which nodes fold on the host, and which subgraphs `Transmuter.Fusion` replaced. With `:inputs`, it compiles the model and calls it once:

```elixir
overlay = Transmuter.Graph.overlay(model, inputs: inputs, compile: [compiler: EXLA])
overlay.segments
# => [%{start: 0, stop: 2, compiled: true, reason: "a value that decides a shape"},
#     %{start: 3, stop: :end, compiled: false, reason: nil}]
```

In Livebook, `Transmuter.Graph.as_graph/2` shows the same facts in an interactive view, with one fill color for each segment. Click a group to open it, and click a node to see its details, such as the shapes of its outputs. It needs the optional `kino` dependency. Outside Livebook, `Transmuter.Graph.to_html/3` writes the view to one HTML file:

```elixir
Transmuter.Graph.as_graph(model, inputs: inputs, compile: [compiler: EXLA])
Transmuter.Graph.to_html(model, "graph.html", inputs: inputs, compile: [compiler: EXLA])
```

The view works without a network connection. Without `:inputs`, the view is static only and shows no segments.

The `[:transmuter, :segment, :compile, :stop]` event also gives the reason of each break (see `Transmuter.Telemetry`). For the graph as the exporter wrote it, load the model with `fuse: false` and call `Transmuter.Graph.to_html/3` without `:inputs`, or run `mix transmuter.graph REPO --plain`.

## Limit the number of compiled segments

If the total number of compiled segments grows large (for example, with many unique audio lengths in Kokoro), set a `:cache_size` that matches the number of expected shapes:

```elixir
run = Transmuter.compile(model, inputs, compiler: EMLX, cache_size: 64)
```

When the cache is full, Transmuter removes the least recently used segment. A new audio length triggers one recompilation. Later calls with the same length reuse the compiled segment.

## Eager execution

`Transmuter.run/4` runs the model without compilation. Call it directly when you do not need compiled speed, when you need to read intermediate values, or when your graph has non-tensor outputs such as strings or sequences:

```elixir
outputs = Transmuter.run(model, model.params, inputs)
```

Give `Transmuter.run/4` to `Nx.Defn.jit/2` to compile it yourself:

```elixir
jit_run = Nx.Defn.jit(&Transmuter.run(model, &1, &2))
outputs = jit_run.(model.params, inputs)
```

## Strings, sequences, and empty optionals

Nx tensors hold only numbers. Transmuter keeps other ONNX values as Elixir terms:

| ONNX type | Elixir representation |
|---|---|
| String tensor | `{:strings, shape, values}` where `values` is a flat list of binaries |
| Sequence of tensors | `{:sequence, tensors}`. An eager run of `Transmuter.run/4` returns it as a list of tensors |
| Empty optional | `:none` |
| Tensor with a zero dimension | `{:empty, shape}` |

A compiled function can only return `Nx.Tensor` values. To read a string or sequence output, use `Transmuter.run/4` without compilation.

Give string and sequence inputs as static inputs with `:static_inputs`.

## Random operators

Random operators draw from `Nx.Random`. Each call of a compiled model draws new values. A node with a `seed` attribute draws the same values on each call.
