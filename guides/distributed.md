# Distributed Nodes

This guide shows how to run the models on the nodes that have a device, such as a GPU, and call them from any node of an Erlang cluster. A call from another node uses the same functions as a call on the node of the model.

## The nodes

A model node runs the models. It has EMLX or EXLA in its dependencies and a device backend in its configuration:

```elixir
# mix.exs
{:transmuter, "~> 2.0"},
{:emlx, "~> 0.5"},
{:tokenizers, "~> 0.5"}

# config/config.exs
config :nx, default_backend: EMLX.Backend
```

A caller node calls the models. It has Transmuter without the optional dependencies, so it compiles and runs without EMLX, EXLA, tokenizers or stb_image. Its default backend is `Nx.BinaryBackend`:

```elixir
# mix.exs
{:transmuter, "~> 2.0"}
```

All nodes must have the same versions of Nx and Transmuter.

## Connect the nodes

Start each node with a name and the same cookie, and connect them with `Node.connect/1`:

```elixir
Node.connect(:"gpu@gpu-host")
```

In production, a library such as [libcluster](https://github.com/bitwalker/libcluster) connects the nodes. Give it the same topology on each node:

```elixir
topologies = [
  transmuter: [
    strategy: Cluster.Strategy.Epmd,
    config: [hosts: [:"gpu@gpu-host", :"web@web-host"]]
  ]
]

children = [{Cluster.Supervisor, [topologies, [name: MyApp.ClusterSupervisor]]}]
```

## Servings

Start the serving with a name on the model node:

```elixir
children = [
  {Nx.Serving,
   serving: Transmuter.Text.text_embedding({:hf, "Xenova/all-MiniLM-L6-v2"}),
   name: MyEmbeddings}
]
```

On a caller node, call it with `Nx.Serving.batched_run/3`:

```elixir
%{embedding: embedding} =
  Nx.Serving.batched_run(MyEmbeddings, "The cat sat on the mat", &Transmuter.to_host/1)
```

The serving runs the tokenizer, the featurizer, the model and the postprocessing on the model node. The result tensors are on `Nx.BinaryBackend`. `Transmuter.to_host/1` copies the tensors of the input to `Nx.BinaryBackend`, because a tensor on a device backend cannot go to another node. A caller node without a device backend does not need it.

When several model nodes start a serving with the same name, `Nx.Serving` sends each call to one of them at random.

### Pipelines that do not work across nodes yet

The pipelines of `Transmuter.Serving.each/2`, generation without `stream: true`, and zero-shot entity recognition do not give the result of a local call to a caller on another node. The caller gets a stream of the parts of the result: a map comes as a list of `{key, value}` tuples. These are the pipelines:

- `Transmuter.Vision.depth_estimation/2`, `Transmuter.Vision.image_to_image/2`, `Transmuter.Vision.mask_generation/2`, and `Transmuter.Vision.zero_shot_object_detection/3` with Grounding DINO.
- `Transmuter.Audio.text_to_speech/2`.
- `Transmuter.Text.zero_shot_entity_recognition/2`.
- `Transmuter.Text.generation/2`, `Transmuter.Text.text_to_text/2`, `Transmuter.Vision.image_to_text/2`, `Transmuter.Vision.image_text_to_text/2` and `Transmuter.Audio.speech_to_text/2` without `stream: true`.

A change in Transmuter will make these pipelines give the result of a local call across nodes. Until then, call them on the model node. Generation with `stream: true` gives its stream of text to a caller on another node.

An input error raises an `ArgumentError` in a local caller. A caller on another node gets an exit until [elixir-nx/nx#1903](https://github.com/elixir-nx/nx/pull/1903).

## ContinuousBatching

Start the server with a name on the model node:

```elixir
children = [
  {Transmuter.ContinuousBatching,
   name: MyChat, source: {:hf, "HuggingFaceTB/SmolLM2-135M-Instruct"}, slots: 8}
]
```

On a caller node, call `Transmuter.ContinuousBatching.run/3` and `Transmuter.ContinuousBatching.stream/3` with the name:

```elixir
%{results: [%{text: text}]} =
  Transmuter.ContinuousBatching.run(MyChat, %{messages: [%{role: "user", content: "Hi"}]})

MyChat
|> Transmuter.ContinuousBatching.stream("Once upon a time")
|> Enum.each(&IO.write/1)
```

A process on the model node runs the request, and the caller gets the result, the exceptions and the exits of a local call. A stream gets each chunk of text as a message. Consume the stream in the process that called `stream/3`. When the caller exits, or stops the stream before its end, the server frees the row of the request.

Several model nodes can start a server with the same name. A call tries the server on its own node first, and then the other servers in a random order. It goes to the first server whose queue takes it, and it raises `Transmuter.ContinuousBatching.QueueFullError` when no queue takes it.

The call copies the tensors of the input to `Nx.BinaryBackend` with `Transmuter.to_host/1`.

## Inputs

An input goes to the model node, so it must be data: strings, numbers, maps, lists, `StbImage` structs and tensors. These inputs do not work across nodes:

- `{:file, path}` reads the file on the model node. Decode the file on the caller node first. For example, `Transmuter.Audio.Featurizer.audio/2` gives the samples of an audio file, and needs `ffmpeg` on the caller node:

  ```elixir
  {:ok, [samples]} = Transmuter.Audio.Featurizer.audio({:file, "speech.mp3"}, 16_000)
  Nx.Serving.batched_run(MySpeech, samples)
  ```

- A stream of audio chunks holds functions of the caller node, so it does not go to another node.

A large input goes through Erlang distribution as one message. For example, one hour of audio at 16 kHz is about 230 MB.
