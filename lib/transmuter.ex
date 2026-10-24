defmodule Transmuter do
  @moduledoc """
  Runs Hugging Face ONNX models on any Nx backend, with no per-architecture code.

  Transmuter reads an ONNX file, such as an export from the Hugging Face Hub, and
  turns its graph into Nx operations. It does not port model architectures. The
  graph that the exporter traced is the model.

      model = Transmuter.load!({:hf, "Xenova/all-MiniLM-L6-v2", "onnx/model.onnx"}, backend: EMLX.Backend)
      run = Transmuter.compile(model, inputs, compiler: EMLX)
      %{"last_hidden_state" => hidden} = run.(inputs)

  Shapes are resolved when the function is compiled. `compile/3` compiles for
  the shapes of the inputs it was given, and the function compiles inputs of
  other shapes on their first call. `run/4` is the uncompiled function, for
  `Nx.Defn.jit/2`, gradients or eager execution.
  """

  alias Nx.Defn.{Compiler, Composite, Expr, Tree}
  alias Transmuter.Onnx.ModelProto
  alias Transmuter.Ops
  alias Transmuter.Ops.Random

  @max_loop_iterations 100_000
  @max_unrolled_steps 16

  # Small initializers and integer initializers are shape data, so they stay static.
  # Large integer initializers are packed quantized weights, so they are params.
  @static_max_size 16
  @static_int_max_size 4096

  # A value that a node computes from static values stays static up to the size of the
  # largest static initializer. A graph break also makes values static only up to this size.
  @fold_max_size @static_int_max_size

  @random_key "transmuter.random_key"
  @fold_memo {__MODULE__, :fold_memo}
  @record {__MODULE__, :record}
  @control_ops ["If", "Loop", "Scan", "SequenceMap"]

  @doc """
  Loads an ONNX model.

  `source` is a local file path or `{:hf, repo, file}`. Hub files are downloaded
  once into `~/.cache/transmuter`, together with any external data files.

  ## Options

    * `:cache_dir`, `:revision` and `:offline` - the options of `Transmuter.Hub` for Hub
      files.

    * `:backend` (`t:module/0` or `{t:module/0, keyword()}`) - moves the params
      to this backend after loading. For example, `EMLX.Backend` or
      `{EXLA.Backend, client: :cuda}`. On the EMLX GPU, 4-bit and 8-bit
      `MatMulNBits` weights stay packed as MLX quantized tensors. On the EMLX
      CPU, they become float weights when the model loads (see
      `Transmuter.EMLX`). The default is `Nx.BinaryBackend`.

    * `:compute_type` (`t:atom/0`) - runs f16 and bf16 tensors in this type
      through every operator. The default is `nil`, which keeps the types of the
      graph. In native f16 and bf16, sums, normalizations, softmax and attention
      scores run in f32. `:f32` converts half-precision weights to f32 when the
      model loads, as ONNX Runtime on the CPU does, and uses as much memory as
      f32 weights. Use `:f32` when a deep half-precision graph drifts from the
      f32 result. See [Backends and performance](guides/backends_and_performance.md).

    * `:fuse` (`t:boolean/0`) - replaces normalizations and attention cores that
      the graph writes as separate operators with one operator, which runs as a
      fused kernel on EMLX (see `Transmuter.Fusion`). The default is `true`.

    * `:trainable` (`t:boolean/0`) - when `true`, every float initializer is a
      param, so small weights such as a 1-element bias get gradients. The
      default is `false`, which folds float initializers with up to 16 elements
      as constants.

    * `:noise` (`t:boolean/0`) - when `false`, the random operators give fixed
      values, so each run gives the same outputs. RandomNormal and
      RandomNormalLike give their mean. RandomUniform and RandomUniformLike give
      their upper bound, so a dropout that keeps the values where the draw is
      greater than the ratio keeps all values. Bernoulli gives 1 where the
      probability is more than 0. Multinomial gives the class with the largest
      logit. Use it to compare a model with a reference that has no noise. The
      default is `true`.

    * `:outputs` (a list of `t:String.t/0`, or a function) - the graph outputs to keep. The model then gives only these outputs, and the loader removes the nodes and the initializers that no kept output reads. An `If` node keeps only the outputs that a later node or the graph reads, and its two branches keep the same outputs. `Loop` and `Scan` keep all outputs of their bodies. A function gets the names of the graph outputs and returns the names to keep. A name that is not a graph output raises an `ArgumentError`. The servings that generate text, classify or embed give the outputs that they read. The default is `nil`, which keeps all outputs.

  Returns a map with:

    * `:params` - the weights and the other large tensors of the graph, on the `:backend` when it is given. These are the large initializers and the values of large `Constant` nodes, also those of subgraphs, such as the branches of an `If`
    * `:backend` - the `:backend` option. An eager `run/4` computes large values that do not depend on the inputs, such as attention masks from shapes, on this backend
    * `:inputs` - the names of the graph inputs
    * `:outputs` - the names of the graph outputs
    * `:source` - the source, which `Transmuter.Telemetry` events give as metadata
    * `:noise` - the `:noise` option. Put `false` in a loaded model to give fixed values in place of random draws

  ## Examples

  Load from the Hugging Face Hub:

      model = Transmuter.load!(
        {:hf, "Xenova/all-MiniLM-L6-v2", "onnx/model.onnx"},
        backend: EMLX.Backend
      )
      model.inputs
      # => ["input_ids", "attention_mask", "token_type_ids"]
      model.outputs
      # => ["last_hidden_state"]

  Load a quantized model:

      model = Transmuter.load!(
        {:hf, "onnx-community/Qwen3-0.6B-ONNX", "onnx/model_q4.onnx"},
        backend: EMLX.Backend
      )

  Load a half-precision model and run it in f32, as ONNX Runtime on the CPU:

      model = Transmuter.load!(
        {:hf, "Xenova/bert-base-uncased", "onnx/model_fp16.onnx"},
        backend: EMLX.Backend,
        compute_type: :f32
      )

  Keep only the outputs that a generation loop reads. The Whisper exports of Xenova also give the attention probabilities of each layer:

      model = Transmuter.load!(
        {:hf, "Xenova/whisper-tiny.en", "onnx/decoder_model_merged.onnx"},
        outputs: &Enum.filter(&1, fn name -> name == "logits" or name =~ ~r/^present\./ end)
      )

  Load a local file:

      model = Transmuter.load!("/path/to/model.onnx")
  """
  def load!(source, opts \\ []) do
    :telemetry.span([:transmuter, :load], %{source: source}, fn ->
      model = source |> path!(opts) |> read!(opts) |> Map.put(:source, source)
      {model, %{source: source, node_count: length(model.nodes), opset: model.opset}}
    end)
  end

  defp path!({:hf, repo, file}, opts), do: Transmuter.Hub.fetch!({:hf, repo}, file, opts)
  defp path!(path, _opts) when is_binary(path), do: path

  defp read!(path, opts) do
    dir = Path.dirname(path)
    model = ModelProto.decode!(File.read!(path))

    opset =
      Enum.find_value(model.opset_import, &(&1.domain in [nil, "", "ai.onnx"] && &1.version))

    graph = inline_external(model.graph, dir)
    check_outputs!(graph, "the graph")

    known = Enum.map(graph.input, & &1.name) ++ Enum.map(graph.initializer, & &1.name)
    graph = %{graph | node: toposort(graph.node, known)}
    graph = if outputs = opts[:outputs], do: prune(graph, outputs), else: graph

    graph =
      if Keyword.get(opts, :fuse, true), do: Transmuter.Fusion.graph(graph, opset), else: graph

    trainable = Keyword.get(opts, :trainable, false)
    {graph, constants} = hoist_constants(graph, false)

    # A merged decoder, such as that of NLLB, can have the same positional table, with the
    # same name, in each branch of If. Equal values are one param.
    constants =
      for {name, [tensor | others]} <- Enum.group_by(constants, &elem(&1, 0), &elem(&1, 1)) do
        unless Enum.all?(others, &(&1 == tensor)) do
          raise ArgumentError,
                "the graph defines the large constant #{name} more than once, with different values"
        end

        {name, tensor}
      end

    {static, params} =
      graph.initializer
      |> Enum.map(&{&1.name, Ops.tensor(&1, dir)})
      |> Enum.split_with(fn {_, t} -> static_initializer?(t, trainable) end)

    params = params ++ constants

    initializers = Enum.map(graph.initializer, & &1.name)
    graph_inputs = for i <- graph.input, i.name not in initializers, do: i
    inputs = Enum.map(graph_inputs, & &1.name)
    compute_type = Keyword.get(opts, :compute_type)

    model = %{
      nodes: graph.node,
      static: Map.new(static, fn {k, t} -> {k, upcast(t, compute_type)} end),
      params: Map.new(params),
      backend: opts[:backend],
      compute_type: compute_type,
      noise: Keyword.get(opts, :noise, true),
      inputs: inputs,
      input_types: Map.new(graph_inputs, &{&1.name, value_type(&1)}),
      outputs: Enum.map(graph.output, & &1.name),
      opset: opset
    }

    # The params move before they get compute_type, so the device converts them. On
    # Nx.BinaryBackend, the 153M f16 weights of vit-gpt2 take 14 s to convert.
    model = if backend = opts[:backend], do: move(model, backend), else: model
    %{model | params: Map.new(model.params, fn {k, t} -> {k, upcast(t, compute_type)} end)}
  end

  # Tensors in node attributes, such as the value of a Constant, and the initializers of
  # subgraphs can also keep their data in the external file. Only the loader knows the
  # directory of that file, so it reads them here, and every operator gets plain tensors.
  # The initializers of the main graph stay external, because the loader reads them once.
  defp inline_external(graph, dir) do
    node =
      Enum.map(graph.node, fn node ->
        attribute =
          Enum.map(node.attribute, fn a ->
            %{
              a
              | t: a.t && Ops.read_external(a.t, dir),
                tensors: Enum.map(a.tensors, &Ops.read_external(&1, dir)),
                g: a.g && inline_subgraph(a.g, dir, "the #{a.name} of #{where(node)}"),
                graphs:
                  Enum.map(a.graphs, &inline_subgraph(&1, dir, "the #{a.name} of #{where(node)}"))
            }
          end)

        %{node | attribute: attribute}
      end)

    %{graph | node: node}
  end

  defp inline_subgraph(graph, dir, where) do
    graph = inline_external(graph, dir)
    check_outputs!(graph, where)
    %{graph | initializer: Enum.map(graph.initializer, &Ops.read_external(&1, dir))}
  end

  # Each output of a graph is a value of the graph: an input, an initializer or the output
  # of one of its nodes. ONNX Runtime also rejects a subgraph whose output is an outer
  # value, such as the merged fp16 decoders whose If branches name the graph outputs.
  defp check_outputs!(graph, where) do
    local =
      Enum.map(graph.input, & &1.name) ++
        Enum.map(graph.initializer, & &1.name) ++ Enum.flat_map(graph.node, & &1.output)

    case Enum.map(graph.output, & &1.name) -- local do
      [] ->
        :ok

      missing ->
        raise ArgumentError,
              "#{where} gives the outputs #{Enum.join(missing, ", ")}, which no node of it computes. " <>
                "ONNX Runtime also rejects this model"
    end
  end

  defp where(node), do: "#{node.op_type} #{node.name}"

  # Keeps the graph outputs that `outputs` names. The initializers that the graph then does
  # not read are also graph inputs in exports before IR version 4, so they leave both lists.
  defp prune(graph, outputs) do
    names = Enum.map(graph.output, & &1.name)
    kept = if is_function(outputs, 1), do: outputs.(names), else: outputs

    case kept -- names do
      [] ->
        :ok

      unknown ->
        raise ArgumentError,
              "the graph has no outputs #{Enum.join(unknown, ", ")}, outputs: #{inspect(names)}"
    end

    pruned = live(%{graph | output: Enum.filter(graph.output, &(&1.name in kept))})

    removed =
      MapSet.difference(
        MapSet.new(graph.initializer, & &1.name),
        MapSet.new(pruned.initializer, & &1.name)
      )

    %{pruned | input: Enum.reject(graph.input, &MapSet.member?(removed, &1.name))}
  end

  # Removes the nodes and the initializers that no output of the graph reads, from the last
  # node to the first. Nodes are in the order of their dependencies. An If node keeps only its
  # outputs that a later node or the graph reads, and both branches keep the same outputs.
  # Loop and Scan keep all outputs of their bodies, because each step reads its state.
  defp live(graph) do
    {nodes, reads} =
      graph.node
      |> Enum.reverse()
      |> Enum.reduce({[], MapSet.new(graph.output, & &1.name)}, fn node, {nodes, reads} ->
        if Enum.any?(node.output, &read?(reads, &1)) do
          node = live_subgraphs(node, reads)
          {[node | nodes], Enum.into(dependencies(node), reads)}
        else
          {nodes, reads}
        end
      end)

    %{graph | node: nodes, initializer: Enum.filter(graph.initializer, &read?(reads, &1.name))}
  end

  defp read?(reads, name), do: name != "" and MapSet.member?(reads, name)

  defp live_subgraphs(node = %{op_type: "If"}, reads) do
    kept = for {name, i} <- Enum.with_index(node.output), read?(reads, name), do: i
    take = fn list -> Enum.map(kept, &Enum.at(list, &1)) end

    attribute =
      Enum.map(node.attribute, fn
        a = %{g: g} when g != nil -> %{a | g: live(%{g | output: take.(g.output)})}
        a -> a
      end)

    %{node | output: take.(node.output), attribute: attribute}
  end

  defp live_subgraphs(node, _reads) do
    attribute =
      Enum.map(node.attribute, fn a ->
        %{a | g: a.g && live(a.g), graphs: Enum.map(a.graphs, &live/1)}
      end)

    %{node | attribute: attribute}
  end

  @doc """
  Compiles the model for the shapes and types of `inputs`.

  Returns a function that takes inputs and returns outputs. The model params are
  captured from `model.params`, so load the model with `:backend` first. The
  returned function compiles inputs of other shapes on their first call and
  keeps the most recently used shapes compiled.

  When the graph has operators that require a value to decide a shape (such as
  `Range`, `NonZero`, or `If`), `compile/3` creates graph breaks. Each segment
  between breaks is compiled separately. See [Compiling and Graph Breaks](guides/graph_breaks.md).

  ## Options

    * `:static_inputs` (`t:map/0`) - inputs whose values are known at compile
      time. Give them as `Nx.BinaryBackend` tensors. Each new value compiles a
      new version of the model. Use this for flags that select a branch (such as
      `use_cache_branch`) and for inputs that decide an output shape (such as
      the `input_ids` that feed `NonZero`). The default is `%{}`.

    * `:cache_size` (`t:pos_integer/0`) - how many compiled shapes to keep for
      each graph segment. The default is `8`. Use a larger value when the inputs
      have many shapes that come back often, such as audio lengths in a speech
      service. Each cached shape uses memory.

  All other options go to `Nx.Defn.compile/3`, for example `compiler: EMLX` or
  `compiler: EXLA`.

  With EXLA, a `:cache` path keeps the compiled segments on disk, so a later compile
  of the same model, also in a new OS process, loads them. Each segment gets its own
  file, `<path>-<hash>`, where the hash comes from the graph, the shapes of the
  segment arguments and the values of the static inputs. So many models and tasks
  can share one path. The weights are arguments of the segments, so new weights
  use the same files.

  Each call of a model with random operators draws new values. A call that gives an
  `Nx.Random` key in the `Transmuter.random_input/0` input draws from that key, so the
  same key gives the same values.

  ## Examples

  Compile with EMLX on Apple silicon:

      run = Transmuter.compile(model, inputs, compiler: EMLX)
      outputs = run.(inputs)

  Compile with a static input to select an `If` branch:

      flag = Nx.tensor([1], type: :u8, backend: Nx.BinaryBackend)

      run = Transmuter.compile(
        model,
        inputs,
        static_inputs: %{"use_cache_branch" => flag},
        compiler: EMLX
      )

  Compile with a larger cache for speech models with many audio lengths:

      run = Transmuter.compile(model, inputs, compiler: EMLX, cache_size: 64)
  """
  def compile(model, inputs, opts \\ []) do
    {static_inputs, opts} = Keyword.pop(opts, :static_inputs, %{})
    Transmuter.GraphBreaks.compile(model, inputs, static_inputs, opts)
  end

  @doc """
  Runs the model graph operator by operator.

  Call this function inside `Nx.Defn.jit/2` or `Nx.Defn.compile/3` to compile it
  with a backend. Call it directly for eager execution, to read intermediate values,
  or to read non-tensor outputs such as strings or sequences.

  `params` is the weight map from `model.params` (or a map of tensors you have
  moved to a device). `inputs` is a map of input name to tensor.

  `static_inputs` are graph inputs whose values are known when the function is
  traced. Give them as `Nx.BinaryBackend` tensors. Use them for flags that select
  a branch (such as `use_cache_branch`) and for inputs that decide an output shape
  (such as the `input_ids` that feed `NonZero` in Grounding DINO). Each new value
  of a static input compiles a new version of the graph. The default is `%{}`.

  Outputs with a zero-size dimension are left out of the result.

  Random operators draw from the `Nx.Random` key in the `"#{@random_key}"` input.
  Without it, each call makes a new key. A function that you compile yourself makes
  the key when it is traced, so it gives the same values on every call unless you
  pass the key. A model with `noise: false` (see `load!/2`) gives fixed values.

  ## Examples

  Eager execution (no compilation):

      outputs = Transmuter.run(model, model.params, inputs)

  Compile with `Nx.Defn.jit/2`:

      jit_run = Nx.Defn.jit(&Transmuter.run(model, &1, &2))
      outputs = jit_run.(model.params, inputs)

  Compile two shapes (prefill and decode) for a decoder with a static KV cache:

      fun = &Transmuter.run(model, &1, &2)
      prefill = Nx.Defn.compile(fun, [params, prefill_inputs], compiler: EMLX)
      decode = Nx.Defn.compile(fun, [params, decode_inputs], compiler: EMLX)

  Run with a static branch selector:

      flag = Nx.tensor([1], type: :u8, backend: Nx.BinaryBackend)
      outputs = Transmuter.run(model, model.params, inputs, %{"use_cache_branch" => flag})
  """
  def run(model, params, inputs, static_inputs \\ %{}) do
    metadata = %{source: model[:source]}

    :telemetry.span([:transmuter, :run], metadata, fn ->
      {run_graph(model, params, inputs, static_inputs), metadata}
    end)
  end

  defp run_graph(model, params, inputs, static_inputs) do
    # Without noise, the random operators get no key and give fixed values.
    {key, inputs} = Map.pop(inputs, @random_key)
    key = if Map.get(model, :noise, true), do: key || random_key()

    ctx = %{
      opset: model.opset,
      compute_type: model[:compute_type],
      key: key,
      backend: model[:backend] || Nx.default_backend()
    }

    cast = &Map.new(&1, fn {k, t} -> {k, upcast(t, ctx.compute_type)} end)

    env =
      model.static
      |> Map.merge(params)
      |> Map.merge(cast.(inputs))
      |> Map.merge(cast.(static_inputs))

    env = with_fold_memo(fn -> run_nodes(model.nodes, env, ctx) end)

    model.outputs
    |> Map.new(&{&1, Map.fetch!(env, &1)})
    |> Map.reject(fn {_name, t} -> Ops.empty?(t) end)
    |> host_outputs(Compiler.current())
    |> tag_literals(Compiler.current())
  end

  @doc group: "Internals"
  @doc """
  Prepares the outputs of a graph to leave the function that runs it.

  Strings, sequences and empty optionals cannot leave a compiled function. Without a compiler, a sequence comes back as a list.
  """
  def host_outputs(outputs, nil) do
    Map.new(outputs, fn
      {name, {:sequence, elements}} -> {name, elements}
      other -> other
    end)
  end

  def host_outputs(outputs, _compiler) do
    for {name, value} <- outputs, not is_struct(value, Nx.Tensor) do
      raise ArgumentError,
            "output #{name} is #{describe(value)}, which cannot leave a compiled function. " <>
              "Call Transmuter.run/4 without compiling to read it"
    end

    outputs
  end

  defp describe({:strings, _shape, _values}), do: "a string tensor"
  defp describe({:sequence, _elements}), do: "a sequence"
  defp describe(:none), do: "an empty optional"

  @doc group: "Internals"
  @doc """
  Puts the values of literal tensors in the compile cache key of EMLX.

  EMLX 0.5 keys its compile cache by the structure of the graph, and a literal tensor adds only its shape and type to the key. Two graphs that differ only in literal values then share one compiled program. A metadata node with a hash of the value puts the value in the key. EMLX compiles a metadata node as its inner node.

  EMLX compiles a kernel of `EMLX.Fast` from the operands in its metadata, so these operands get the tags too.
  """
  def tag_literals(outputs, EMLX) do
    {outputs, _seen} = Composite.traverse(outputs, %{}, &tag_literal/2)
    outputs
  end

  def tag_literals(outputs, _compiler), do: outputs

  defp tag_literal(t = %Nx.Tensor{data: %Expr{id: id, op: op, args: args}}, seen) do
    case seen do
      %{^id => tagged} ->
        {tagged, seen}

      %{} when op == :tensor ->
        tagged = Expr.metadata(t, %{literal: :erlang.md5(Nx.to_binary(hd(args)))})
        {tagged, Map.put(seen, id, tagged)}

      %{} ->
        {args, seen} = Tree.apply_args(t, :all, seen, &tag_literal/2)
        {args, seen} = tag_operands(args, seen)
        tagged = put_in(t.data.args, args)
        {tagged, Map.put(seen, id, tagged)}
    end
  end

  defp tag_literal(t = %Nx.Tensor{shape: {}}, seen), do: {t, seen}
  defp tag_literal(t = %Nx.Tensor{}, seen), do: tag_literal(Expr.tensor(t), seen)

  defp tag_operands([expr, meta = %{__EMLX__: kernel}], seen) do
    {operands, seen} = Enum.map_reduce(kernel.operands, seen, &tag_literal/2)
    {[expr, %{meta | __EMLX__: %{kernel | operands: operands}}], seen}
  end

  defp tag_operands(args, seen), do: {args, seen}

  @doc false
  def run_nodes(nodes, env, ctx) do
    Enum.reduce(nodes, env, fn node, env ->
      inputs = Enum.map(node.input, &input!(env, &1, node))
      placement = placement(node, inputs, &static_value?/1)

      outputs =
        try do
          case placement do
            :control -> run_control(node, inputs, env, ctx)
            :random -> Random.apply(node, inputs, ctx.key)
            :host -> node |> fold(inputs, ctx) |> unfold(node, inputs, ctx)
            :device -> apply_on_backend(node, inputs, ctx)
          end
          |> Enum.map(&upcast(&1, ctx.compute_type))
        rescue
          e in Transmuter.UnknownValueError ->
            reraise %{e | node: e.node || node.name, op_type: e.op_type || node.op_type},
                    without_arguments(__STACKTRACE__)

          e ->
            shapes =
              Enum.map(inputs, fn
                %Nx.Tensor{} = t -> Nx.shape(t)
                {:strings, shape, _values} -> {:strings, shape}
                {:sequence, elements} -> {:sequence, length(elements)}
                other -> other
              end)

            message =
              "#{node.op_type} #{node.name} with inputs #{inspect(shapes)}: #{Exception.message(e)}"

            reraise RuntimeError, message, without_arguments(__STACKTRACE__)
        end

      record(node, placement, inputs, outputs, ctx)
      Map.merge(env, Map.new(Enum.zip(node.output, outputs)))
    end)
  end

  @doc group: "Internals"
  @doc """
  Returns where `run/4` runs `node`.

  `inputs` has one entry for each input of the node, and `static?` tells whether the value of an entry is static. By default, the entries are booleans. `run/4` gives the input values with a function of `Transmuter.Ops.static?/1`, in which a missing optional input counts as static, so it checks the inputs only up to the first value that is not static.

    * `:control` - `If`, `Loop`, `Scan` and `SequenceMap` run their subgraphs.
    * `:random` - a random operator draws from the random key.
    * `:host` - all inputs are static, so the node folds on the host. In a traced function, an output of more than #{@fold_max_size} elements is computed again on the device from the static inputs.
    * `:device` - the node runs on the backend. In a traced function, its static inputs are literals of the trace.
  """
  def placement(node, inputs, static? \\ & &1) do
    cond do
      node.op_type in @control_ops -> :control
      node.op_type in Random.op_types() -> :random
      Enum.all?(inputs, static?) -> :host
      true -> :device
    end
  end

  defp static_value?(value), do: value == nil or Ops.static?(value)

  defp run_control(node = %{op_type: "If"}, inputs, env, ctx), do: run_if(node, inputs, env, ctx)

  defp run_control(node = %{op_type: "Loop"}, inputs, env, ctx),
    do: run_loop(node, inputs, env, in_subgraph(ctx, node, "body"))

  defp run_control(node = %{op_type: "Scan"}, inputs, env, ctx),
    do: run_scan(node, inputs, env, in_subgraph(ctx, node, "body"))

  defp run_control(node = %{op_type: "SequenceMap"}, inputs, env, ctx),
    do: run_sequence_map(node, inputs, env, in_subgraph(ctx, node, "body"))

  # The path of a subgraph is the list of the nodes and attributes that hold it, from the
  # main graph. The outputs of a node name it in its graph.
  defp in_subgraph(ctx, node, attribute),
    do: Map.put(ctx, :path, Map.get(ctx, :path, []) ++ [{node.output, attribute}])

  defp shape_type(t = %Nx.Tensor{}),
    do: %{shape: Tuple.to_list(Nx.shape(t)), type: Nx.Type.to_string(Nx.type(t))}

  defp shape_type({:empty, shape}), do: %{shape: Tuple.to_list(shape), type: nil}
  defp shape_type({:strings, shape, _values}), do: %{shape: Tuple.to_list(shape), type: "string"}
  defp shape_type(_value), do: nil

  @doc false
  # The key of the record of a node: the path of its subgraph and its outputs. The branches of
  # an If can hold equal nodes, and the path tells them apart.
  def record_key(path, node), do: {path, node.output}

  # In with_record/1, each node that runs records its placement, the static flags of its
  # inputs, whether it runs in a traced function, whether its outputs are static and the
  # shapes and types of its outputs.
  defp record(node, placement, inputs, outputs, ctx) do
    if records = Process.get(@record) do
      record = {
        placement,
        Enum.map(inputs, &static_value?/1),
        Compiler.current() != nil,
        Enum.all?(outputs, &Ops.static?/1),
        Enum.map(outputs, &shape_type/1)
      }

      key = record_key(Map.get(ctx, :path, []), node)
      Process.put(@record, Map.put(records, key, record))
    end
  end

  @doc false
  # Runs fun and returns its result with the records of the nodes that ran in this process.
  def with_record(fun) do
    Process.put(@record, %{})

    try do
      result = fun.()
      {result, Process.get(@record)}
    after
      Process.delete(@record)
    end
  end

  # Operators build constants, such as the positions of a causal mask, on the default
  # backend. A traced function makes them part of the graph. An eager run makes them on the
  # backend of the model, as its other values, and not on Nx.BinaryBackend in Elixir.
  #
  # In a traced function, the static inputs of the node are literals of the trace. So an
  # operator that expands a static input, such as the Nx.broadcast of a scalar in Where,
  # builds the large value on the device, and the program holds only the small literal.
  # EXLA writes each literal into the program as text: the 36 window masks of Swin2SR
  # (63x64x64 at 72x56 pixels) took 11.5 s of the 14 s compile on the EXLA CPU.
  defp apply_on_backend(node, inputs, ctx) do
    if Compiler.current(),
      do: Ops.apply(node, Enum.map(inputs, &literal/1), ctx.opset),
      else: Nx.with_default_backend(ctx.backend, fn -> Ops.apply(node, inputs, ctx.opset) end)
  end

  # Nx makes a scalar literal a :constant with the number, which is an atom for an infinity
  # or NaN, and EMLX cannot compile such a constant. So a scalar is a literal of one
  # element, reshaped in the trace. Its metadata keeps the host value for Ops.known!/2.
  defp literal(value = %Nx.Tensor{data: %Nx.BinaryBackend{}, shape: {}}) do
    value
    |> Nx.reshape({1})
    |> Expr.tensor()
    |> Nx.reshape({})
    |> Expr.metadata(%{transmuter_value: value})
  end

  defp literal(value = %Nx.Tensor{data: %Nx.BinaryBackend{}}), do: Expr.tensor(value)
  defp literal(value), do: value

  defp input!(_env, "", _node), do: nil

  defp input!(env, name, node) do
    case env do
      %{^name => value} -> value
      %{} -> raise ArgumentError, "#{where(node)} reads #{name}, which no node computes before it"
    end
  end

  # The top frame of some errors, such as a function clause error, keeps the arguments of
  # the call. In a traced function they are expressions, which share their arguments. An
  # error that leaves a process copies its reason without that sharing, which can take
  # minutes for a large graph, so the frames keep only the arity.
  defp without_arguments(stacktrace) do
    Enum.map(stacktrace, fn
      {module, fun, args, location} when is_list(args) -> {module, fun, length(args), location}
      entry -> entry
    end)
  end

  # A node with static inputs runs on the host. Nodes with the same operator, attributes and
  # input values give the same outputs, so each runs once in a call. For example, each layer
  # of a vision encoder builds the same attention mask from the same shapes. A tensor
  # attribute counts by its value, without the name that each node gives it.
  defp fold(node, [], ctx), do: fold_on_host(node, [], ctx)

  defp fold(node, inputs, ctx) do
    memo = Process.get(@fold_memo)
    attributes = Enum.map(node.attribute, &if(&1.t, do: %{&1 | t: %{&1.t | name: ""}}, else: &1))
    key = {ctx.opset, node.op_type, node.domain, attributes, Enum.map(inputs, &Ops.digest/1)}

    case memo do
      %{^key => outputs} ->
        outputs

      _ ->
        outputs = fold_on_host(node, inputs, ctx)
        if memo, do: Process.put(@fold_memo, Map.put(memo, key, outputs))
        outputs
    end
  end

  defp fold_on_host(node, inputs, ctx) do
    Nx.with_default_backend(Nx.BinaryBackend, fn -> Ops.apply(node, inputs, ctx.opset) end)
  end

  # A large folded value is data, so the nodes that read it run on the device. An eager run
  # moves it to the backend of the model. A traced function computes it again from the
  # small static inputs, so the device builds it and the program holds no large literal.
  # The traced value keeps the host value, so a node in the same trace can read it. When
  # the operator cannot trace on literals, the value becomes a literal.
  defp unfold(outputs, node, inputs, ctx) do
    cond do
      not Enum.any?(outputs, &large?/1) ->
        outputs

      Compiler.current() ->
        traced = trace_folded(node, inputs, ctx) || outputs

        Enum.zip_with(outputs, traced, fn
          host, t = %Nx.Tensor{data: %Expr{}} when is_struct(host, Nx.Tensor) ->
            if large?(host) and Nx.to_template(t) == Nx.to_template(host),
              do: Expr.metadata(t, %{transmuter_value: host}),
              else: unfold_literal(host)

          host, _traced ->
            unfold_literal(host)
        end)

      true ->
        Enum.map(outputs, &if(large?(&1), do: Nx.backend_transfer(&1, ctx.backend), else: &1))
    end
  end

  defp large?(value), do: is_struct(value, Nx.Tensor) and Nx.size(value) > @fold_max_size

  defp unfold_literal(value), do: if(large?(value), do: Expr.tensor(value), else: value)

  defp trace_folded(node, inputs, ctx) do
    literals =
      Enum.map(
        inputs,
        &if(is_struct(&1, Nx.Tensor) and Nx.rank(&1) > 0, do: Expr.tensor(&1), else: &1)
      )

    Ops.apply(node, literals, ctx.opset)
  rescue
    _error -> nil
  end

  @doc false
  def with_fold_memo(fun) do
    if Process.get(@fold_memo) do
      fun.()
    else
      Process.put(@fold_memo, %{})

      try do
        fun.()
      after
        Process.delete(@fold_memo)
      end
    end
  end

  defp run_if(node, [condition], env, ctx) do
    name =
      if Nx.to_number(Nx.any(Ops.known!(condition, "the If condition"))) == 1,
        do: "then_branch",
        else: "else_branch"

    branch = Ops.attr(node, name)
    ctx = in_subgraph(ctx, node, name)
    env = put_initializers(env, branch, ctx)
    env = run_nodes(branch.node, env, ctx)
    Enum.map(branch.output, &Map.fetch!(env, &1.name))
  end

  # The initializers of a subgraph get compute_type, as those of the main graph do.
  defp put_initializers(env, graph, ctx) do
    Enum.reduce(graph.initializer, env, fn t, env ->
      Map.put(env, t.name, t |> Ops.tensor() |> upcast(ctx.compute_type))
    end)
  end

  # A long traced loop with a condition that stays true runs as an Nx while loop. Other
  # loops unroll while tracing, so the trip count and the condition must be static.
  defp run_loop(node, [trip_count, condition | initial], env, ctx) do
    body = Ops.attr(node, "body")
    [counter, condition_name | carried_names] = Enum.map(body.input, & &1.name)

    trips =
      if trip_count,
        do: trip_count |> Ops.known!("the Loop trip count") |> Nx.reshape({}) |> Nx.to_number(),
        else: @max_loop_iterations

    env = put_initializers(env, body, ctx)

    step = fn i, carried, env ->
      step_env =
        env
        |> Map.merge(Map.new(Enum.zip(carried_names, carried)))
        |> Map.merge(%{counter => i, condition_name => condition || Ops.static_tensor(1, :u8)})

      [next_condition | outputs] =
        body.node
        |> run_nodes(step_env, ctx)
        |> then(&Enum.map(body.output, fn o -> Map.fetch!(&1, o.name) end))

      # Without a condition input, the loop runs trip_count times and the body's
      # condition output is ignored.
      next_condition = if condition == nil, do: nil, else: next_condition
      {next_condition, Enum.split(outputs, length(carried))}
    end

    while_step = fn i, carried, env ->
      {next_condition, outputs} = step.(i, carried, env)
      unless always_true?(next_condition), do: throw(:unroll)
      outputs
    end

    looped =
      if trip_count != nil and always_true?(condition),
        do: while_loop(body, trips, initial, env, while_step)

    case looped do
      {carried, stacked} -> carried ++ stacked
      nil -> unroll_loop(trips, initial, condition, &step.(&1, &2, env))
    end
  end

  defp unroll_loop(trips, initial, condition, step) do
    {carried, scans, _condition} =
      Enum.reduce_while(0..(trips - 1)//1, {initial, [], condition}, fn i,
                                                                        {carried, scans,
                                                                         condition} ->
        if condition != nil and
             Nx.to_number(Nx.any(Ops.known!(condition, "the Loop condition"))) == 0 do
          {:halt, {carried, scans, condition}}
        else
          {next_condition, {next_carried, scan_values}} =
            step.(Ops.static_tensor(i, :s64), carried)

          {:cont, {next_carried, [scan_values | scans], next_condition}}
        end
      end)

    stacked =
      case scans do
        [] -> []
        _ -> scans |> Enum.reverse() |> Enum.zip_with(&Nx.stack/1)
      end

    carried ++ stacked
  end

  defp always_true?(nil), do: true

  defp always_true?(condition) do
    Ops.known?(condition) and
      Nx.to_number(Nx.all(Ops.known!(condition, "the Loop condition"))) == 1
  end

  # Runs step.(i, carried, env) for i in 0..(count - 1) as an Nx while loop, because an
  # unrolled graph can get too deep to compile. The step returns the next carried
  # values and its outputs, and the outputs of all steps are stacked on a new first
  # axis. Returns nil when the loop is short, is not traced, or needs a static i, for
  # example to decide a shape. A traced value that the host knows, such as a large folded
  # value, is still part of the traced function. Then the caller unrolls. A while body cannot read outer
  # values, so the traced values that the step reads from env go in the loop state:
  # the outer reads of the body and the names in inputs.
  defp while_loop(body, count, carried, env, step, inputs \\ []) do
    captured =
      env
      |> Map.take(outer_reads(body) ++ inputs)
      |> Map.reject(fn {_k, v} -> Ops.static?(v) end)

    values = carried ++ Map.values(captured)

    if count > @max_unrolled_steps and Enum.all?(values, &is_struct(&1, Nx.Tensor)) and
         Enum.any?(values, &match?(%Nx.Tensor{data: %Expr{}}, &1)) and not random?(body.node) do
      {probe, outputs} = step.(Ops.static_tensor(0, :s64), carried, env)

      if Enum.map(probe, &Nx.to_template/1) != Enum.map(carried, &Nx.to_template/1),
        do: throw(:unroll)

      buffers =
        Enum.map(outputs, fn o ->
          Nx.broadcast(Nx.tensor(0, type: Nx.type(o)), Tuple.insert_at(Nx.shape(o), 0, count))
        end)

      initial = {List.to_tuple(carried), List.to_tuple(buffers), captured}

      loop = fn :body, {i, {carried, buffers, captured}} ->
        i = Nx.as_type(i, :s64)
        {next, outputs} = step.(i, Tuple.to_list(carried), Map.merge(env, captured))

        buffers =
          Enum.zip_with(Tuple.to_list(buffers), outputs, fn buffer, o ->
            Nx.put_slice(buffer, [i | List.duplicate(0, Nx.rank(o))], Nx.new_axis(o, 0))
          end)

        {List.to_tuple(next), List.to_tuple(buffers), captured}
      end

      {carried, buffers, _captured} =
        Expr.defn_while(__ENV__.file, __ENV__.line, initial, {:while, 0..(count - 1)}, loop,
          unroll: @max_unrolled_steps
        )

      {Tuple.to_list(carried), Tuple.to_list(buffers)}
    end
  rescue
    _error in [Transmuter.UnknownValueError, ArgumentError, RuntimeError] -> nil
  catch
    :unroll -> nil
  end

  @doc group: "Internals"
  @doc """
  The name of the graph input that holds the `Nx.Random` key of random operators.

  A key is two u32 words. It is on the default backend, so eager runs draw there.
  """
  def random_input, do: @random_key

  @doc false
  def random_key do
    Nx.tensor([:rand.uniform(2 ** 32) - 1, :rand.uniform(2 ** 32) - 1], type: :u32)
  end

  @doc false
  def random?(nodes) do
    Enum.any?(nodes, fn node ->
      node.op_type in Random.op_types() or
        Enum.any?(node.attribute, fn a ->
          Enum.any?(List.wrap(a.g) ++ a.graphs, &random?(&1.node))
        end)
    end)
  end

  # On EMLX, Transmuter.EMLX keeps quantized weights packed.
  defp move(model, backend) do
    {module, backend_opts} = if is_atom(backend), do: {backend, []}, else: backend

    if module == EMLX.Backend and Code.ensure_loaded?(Transmuter.EMLX),
      do: Transmuter.EMLX.move(model, backend_opts),
      else: %{
        model
        | params: Map.new(model.params, fn {k, t} -> {k, Nx.backend_transfer(t, backend)} end)
      }
  end

  @doc false
  # The size of a dimension of model.input_types, or default for a symbolic dimension.
  def dim_size(size, _default) when is_integer(size), do: size
  def dim_size(_symbolic, default), do: default

  # The type and the dimensions of a tensor input. A symbolic dimension is its name, and a
  # dimension with neither a size nor a name is nil.
  defp value_type(%{type: %{value: {:tensor_type, %{elem_type: elem_type, shape: shape}}}}) do
    dims =
      for d <- (shape && shape.dim) || [] do
        case d.value do
          {:dim_value, size} -> size
          {:dim_param, name} -> name
          nil -> nil
        end
      end

    {if(elem_type == 8, do: :string, else: Ops.dtype(elem_type)), dims}
  end

  defp value_type(_input), do: nil

  # Large Constant nodes and the large initializers of subgraphs become params, as the large
  # initializers of the main graph do. A static value is a literal of each compiled function,
  # and EXLA writes each literal into the program as text. The 1876x768 positional table in
  # each If branch of the SpeechT5 decoder took 3.5 s of each compile of a step.
  defp hoist_constants(graph, subgraph?) do
    {initializers, large} =
      if subgraph?,
        do: Enum.split_with(graph.initializer, &static_initializer?(Ops.tensor(&1), false)),
        else: {graph.initializer, []}

    hoisted = Enum.map(large, &{&1.name, Ops.tensor(&1)})

    {nodes, hoisted} =
      Enum.flat_map_reduce(graph.node, hoisted, fn node, hoisted ->
        case large_constant(node) do
          nil ->
            {attribute, hoisted} = Enum.map_reduce(node.attribute, hoisted, &hoist_attribute/2)
            {[%{node | attribute: attribute}], hoisted}

          constant ->
            {[], [constant | hoisted]}
        end
      end)

    {%{graph | node: nodes, initializer: initializers}, hoisted}
  end

  defp hoist_attribute(a, hoisted) do
    {g, hoisted} = if a.g, do: hoist_subgraph(a.g, hoisted), else: {nil, hoisted}
    {graphs, hoisted} = Enum.map_reduce(a.graphs, hoisted, &hoist_subgraph/2)
    {%{a | g: g, graphs: graphs}, hoisted}
  end

  defp hoist_subgraph(graph, hoisted) do
    {graph, inner} = hoist_constants(graph, true)
    {graph, inner ++ hoisted}
  end

  defp large_constant(%{op_type: "Constant", domain: domain, output: [name], attribute: [a]})
       when domain in [nil, ""] and a.name == "value" do
    tensor = Ops.tensor(a.t)
    if not static_initializer?(tensor, false), do: {name, tensor}
  end

  defp large_constant(_node), do: nil

  defp static_initializer?({:empty, _shape}, _trainable), do: true
  defp static_initializer?({:strings, _shape, _values}, _trainable), do: true

  defp static_initializer?(t, trainable) do
    if Nx.Type.integer?(Nx.type(t)),
      do: Nx.size(t) <= @static_int_max_size,
      else: not trainable and Nx.size(t) <= @static_max_size
  end

  @doc false
  def upcast(t = %Nx.Tensor{}, type) when type != nil do
    if Nx.type(t) in [{:f, 16}, {:bf, 16}], do: Nx.as_type(t, type), else: t
  end

  def upcast(t, _type), do: t

  # Runs Scan over the static length of its scan inputs, as a while loop when it is long
  # and traced, or else unrolled. Each step takes one slice of every scan input, and
  # the per-step outputs are stacked along their output axes.
  defp run_scan(node, inputs, env, ctx) do
    body = Ops.attr(node, "body")
    scan_count = Ops.attr(node, "num_scan_inputs")
    {states, scans} = Enum.split(inputs, length(inputs) - scan_count)
    {state_names, scan_names} = body.input |> Enum.map(& &1.name) |> Enum.split(length(states))

    in_axes =
      node
      |> Ops.attr("scan_input_axes", List.duplicate(0, scan_count))
      |> Enum.zip_with(scans, &normalize_axis(&1, Nx.rank(&2)))

    in_reversed =
      node
      |> Ops.attr("scan_input_directions", List.duplicate(0, scan_count))
      |> Enum.map(&(&1 == 1))

    length = Nx.axis_size(hd(scans), hd(in_axes))
    scan_inputs = scans |> Enum.with_index(&{{:scan, &2}, &1}) |> Map.new()

    env = env |> put_initializers(body, ctx) |> Map.merge(scan_inputs)

    step = fn t, states, env ->
      slices =
        for {j, axis, reversed} <- Enum.zip([0..(scan_count - 1), in_axes, in_reversed]),
            do:
              scan_slice(
                env[{:scan, j}],
                if(reversed, do: Nx.subtract(length - 1, t), else: t),
                axis
              )

      step_env =
        env
        |> Map.merge(Map.new(Enum.zip(state_names, states)))
        |> Map.merge(Map.new(Enum.zip(scan_names, slices)))

      result = run_nodes(body.node, step_env, ctx)
      body.output |> Enum.map(&Map.fetch!(result, &1.name)) |> Enum.split(length(states))
    end

    {final_states, stacked} =
      while_loop(body, length, states, env, step, Map.keys(scan_inputs)) ||
        Enum.map_reduce(0..(length - 1), states, fn t, states ->
          {new_states, outputs} = step.(Ops.static_tensor(t, :s64), states, env)
          {outputs, new_states}
        end)
        |> then(fn {steps, final_states} -> {final_states, Enum.zip_with(steps, &Nx.stack/1)} end)

    output_count = length(body.output) - length(states)
    out_axes = Ops.attr(node, "scan_output_axes", List.duplicate(0, output_count))

    out_reversed =
      node
      |> Ops.attr("scan_output_directions", List.duplicate(0, output_count))
      |> Enum.map(&(&1 == 1))

    outputs =
      for {values, axis, reversed} <- Enum.zip([stacked, out_axes, out_reversed]) do
        values = if reversed, do: Nx.reverse(values, axes: [0]), else: values
        rank = Nx.rank(values)
        axes = Enum.to_list(1..(rank - 1)//1) |> List.insert_at(normalize_axis(axis, rank), 0)
        Nx.transpose(values, axes: axes)
      end

    final_states ++ outputs
  end

  # A static index slices, so the unrolled graph has no gather.
  defp scan_slice(x, t = %Nx.Tensor{data: %Nx.BinaryBackend{}}, axis) do
    x |> Nx.slice_along_axis(Nx.to_number(t), 1, axis: axis) |> Nx.squeeze(axes: [axis])
  end

  defp scan_slice(x, t, axis), do: Nx.take(x, t, axis: axis)

  # Runs the body once per element. Sequence inputs give one element to each run, and
  # tensor inputs go to every run. Each body output becomes a sequence.
  defp run_sequence_map(node, inputs = [{:sequence, first} | _], env, ctx) do
    body = Ops.attr(node, "body")
    names = Enum.map(body.input, & &1.name)
    env = put_initializers(env, body, ctx)

    steps =
      for i <- 0..(length(first) - 1)//1 do
        values =
          Enum.map(inputs, fn
            {:sequence, elements} -> Enum.at(elements, i)
            tensor -> tensor
          end)

        result = run_nodes(body.node, Map.merge(env, Map.new(Enum.zip(names, values))), ctx)
        Enum.map(body.output, &Map.fetch!(result, &1.name))
      end

    for j <- 0..(length(body.output) - 1), do: {:sequence, Enum.map(steps, &Enum.at(&1, j))}
  end

  defp normalize_axis(axis, rank) when axis < 0, do: axis + rank
  defp normalize_axis(axis, _rank), do: axis

  # Some exports list nodes out of order, so sort them by data dependencies.
  defp toposort(nodes, known) do
    {sorted, _, []} =
      {[], MapSet.new(known), nodes}
      |> Stream.iterate(fn {done, known, pending} ->
        {ready, waiting} =
          Enum.split_with(pending, fn n ->
            Enum.all?(dependencies(n), &(&1 == "" or &1 in known))
          end)

        if ready == [], do: raise(ArgumentError, "the graph has a cycle or an undefined input")

        {done ++ ready, Enum.reduce(ready, known, &MapSet.union(&2, MapSet.new(&1.output))),
         waiting}
      end)
      |> Enum.find(fn {_, _, pending} -> pending == [] end)

    sorted
  end

  @doc group: "Internals"
  @doc """
  Returns the names of the values that `node` reads.

  A node with subgraphs (If branches) also depends on the outer values its branches read.
  """
  def dependencies(node) do
    outer =
      for attribute <- node.attribute,
          graph <- List.wrap(attribute.g) ++ attribute.graphs,
          name <- outer_reads(graph),
          do: name

    node.input ++ outer
  end

  @doc false
  # The names whose values decide the outputs of `node`. The outputs of Shape and Size
  # depend only on the shape of their input.
  def value_dependencies(%{op_type: op_type}) when op_type in ["Shape", "Size"], do: []
  def value_dependencies(node), do: dependencies(node)

  defp outer_reads(graph) do
    local = Enum.map(graph.initializer, & &1.name) ++ Enum.map(graph.input, & &1.name)
    produced = Enum.flat_map(graph.node, & &1.output)
    Enum.uniq(Enum.flat_map(graph.node, &dependencies/1)) -- (local ++ produced)
  end
end
