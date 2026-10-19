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

  @random_key "transmuter.random_key"

  @doc """
  Loads an ONNX model.

  `source` is a local file path or `{:hf, repo, file}`. Hub files are downloaded
  once into `~/.cache/transmuter`, together with any external data files.

  ## Options

    * `:cache_dir` (`t:String.t/0`) - the directory for Hub downloads.
      The default is `~/.cache/transmuter`.

    * `:backend` (`t:module/0` or `{t:module/0, keyword()}`) - moves the params
      to this backend after loading. For example, `EMLX.Backend` or
      `{EXLA.Backend, client: :cuda}`. On EMLX, 4-bit and 8-bit `MatMulNBits`
      weights stay packed as MLX quantized tensors (see `Transmuter.EMLX`). The
      default is `Nx.BinaryBackend`.

    * `:compute_type` (`t:atom/0`) - runs f16 and bf16 tensors in this type
      through every operator. The default is `:f32`, as ONNX Runtime on the CPU,
      which calculates half-precision graphs in f32. Half-precision weights are
      converted when the model loads, so they use as much memory as f32 weights.
      `nil` keeps the types of the graph. In native f16, an export that writes a
      normalization as separate operators can overflow and give zeros. See
      [Backends and performance](guides/backends_and_performance.md).

    * `:trainable` (`t:boolean/0`) - when `true`, every float initializer is a
      param, so small weights such as a 1-element bias get gradients. The
      default is `false`, which folds float initializers with up to 16 elements
      as constants.

  Returns a map with:

    * `:params` - the weights, on the `:backend` when it is given
    * `:inputs` - the names of the graph inputs
    * `:outputs` - the names of the graph outputs
    * `:source` - the source, which `Transmuter.Telemetry` events give as metadata

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

  Load a half-precision model and run it in native f16, with half the memory:

      model = Transmuter.load!(
        {:hf, "Xenova/bert-base-uncased", "onnx/model_fp16.onnx"},
        backend: EMLX.Backend,
        compute_type: nil
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
    graph = inline_external(model.graph, dir)

    trainable = Keyword.get(opts, :trainable, false)

    {static, params} =
      graph.initializer
      |> Enum.map(&{&1.name, Ops.tensor(&1, dir)})
      |> Enum.split_with(fn {_, t} -> static_initializer?(t, trainable) end)

    initializers = Enum.map(graph.initializer, & &1.name)
    graph_inputs = for i <- graph.input, i.name not in initializers, do: i
    inputs = Enum.map(graph_inputs, & &1.name)
    compute_type = Keyword.get(opts, :compute_type, :f32)

    model = %{
      nodes: toposort(graph.node, inputs ++ initializers),
      static: Map.new(static, fn {k, t} -> {k, upcast(t, compute_type)} end),
      params: Map.new(params, fn {k, t} -> {k, upcast(t, compute_type)} end),
      compute_type: compute_type,
      inputs: inputs,
      input_types: Map.new(graph_inputs, &{&1.name, value_type(&1)}),
      outputs: Enum.map(graph.output, & &1.name),
      opset:
        Enum.find_value(model.opset_import, &(&1.domain in [nil, "", "ai.onnx"] && &1.version))
    }

    if backend = opts[:backend], do: %{model | params: move_params(model, backend)}, else: model
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
                g: a.g && inline_subgraph(a.g, dir),
                graphs: Enum.map(a.graphs, &inline_subgraph(&1, dir))
            }
          end)

        %{node | attribute: attribute}
      end)

    %{graph | node: node}
  end

  defp inline_subgraph(graph, dir) do
    graph = inline_external(graph, dir)
    %{graph | initializer: Enum.map(graph.initializer, &Ops.read_external(&1, dir))}
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

  Each call of a model with random operators draws new values.

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
  pass the key.

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
    {key, inputs} = Map.pop_lazy(inputs, @random_key, &random_key/0)
    ctx = %{opset: model.opset, compute_type: model[:compute_type], key: key}
    cast = &Map.new(&1, fn {k, t} -> {k, upcast(t, ctx.compute_type)} end)

    env =
      model.static
      |> Map.merge(params)
      |> Map.merge(cast.(inputs))
      |> Map.merge(cast.(static_inputs))

    env = run_nodes(model.nodes, env, ctx)

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
      inputs =
        Enum.map(node.input, fn
          "" -> nil
          name -> Map.fetch!(env, name)
        end)

      outputs =
        try do
          cond do
            node.op_type == "If" ->
              run_if(node, inputs, env, ctx)

            node.op_type == "Loop" ->
              run_loop(node, inputs, env, ctx)

            node.op_type == "Scan" ->
              run_scan(node, inputs, env, ctx)

            node.op_type == "SequenceMap" ->
              run_sequence_map(node, inputs, env, ctx)

            node.op_type in Random.op_types() ->
              Random.apply(node, inputs, ctx.key)

            Enum.all?(inputs, &(&1 == nil or Ops.static?(&1))) ->
              Nx.with_default_backend(Nx.BinaryBackend, fn ->
                Ops.apply(node, inputs, ctx.opset)
              end)

            true ->
              Ops.apply(node, inputs, ctx.opset)
          end
          |> Enum.map(&upcast(&1, ctx.compute_type))
        rescue
          e in Transmuter.UnknownValueError ->
            reraise %{e | node: e.node || node.name, op_type: e.op_type || node.op_type},
                    __STACKTRACE__

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

            reraise RuntimeError, message, __STACKTRACE__
        end

      Map.merge(env, Map.new(Enum.zip(node.output, outputs)))
    end)
  end

  defp run_if(node, [condition], env, ctx) do
    branch =
      if Nx.to_number(Nx.any(Ops.known!(condition, "the If condition"))) == 1,
        do: Ops.attr(node, "then_branch"),
        else: Ops.attr(node, "else_branch")

    env = Enum.reduce(branch.initializer, env, &Map.put(&2, &1.name, Ops.tensor(&1)))
    env = run_nodes(branch.node, env, ctx)
    Enum.map(branch.output, &Map.fetch!(env, &1.name))
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

    env = Enum.reduce(body.initializer, env, &Map.put(&2, &1.name, Ops.tensor(&1)))

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
  defp always_true?(condition), do: Ops.known?(condition) and Nx.to_number(Nx.all(condition)) == 1

  # Runs step.(i, carried, env) for i in 0..(count - 1) as an Nx while loop, because an
  # unrolled graph can get too deep to compile. The step returns the next carried
  # values and its outputs, and the outputs of all steps are stacked on a new first
  # axis. Returns nil when the loop is short, is not traced, or needs a static i, for
  # example to decide a shape. Then the caller unrolls. A while body cannot read outer
  # values, so the traced values that the step reads from env go in the loop state:
  # the outer reads of the body and the names in inputs.
  defp while_loop(body, count, carried, env, step, inputs \\ []) do
    captured =
      env
      |> Map.take(outer_reads(body) ++ inputs)
      |> Map.reject(fn {_k, v} -> Ops.static?(v) end)

    values = carried ++ Map.values(captured)

    if count > @max_unrolled_steps and Enum.all?(values, &is_struct(&1, Nx.Tensor)) and
         not Enum.all?(values, &Ops.known?/1) and not random?(body.node) do
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
        Enum.any?(node.attribute, &(&1.g != nil and random?(&1.g.node)))
    end)
  end

  # On EMLX, Transmuter.EMLX keeps quantized weights packed.
  defp move_params(model, backend) do
    {module, backend_opts} = if is_atom(backend), do: {backend, []}, else: backend

    if module == EMLX.Backend and Code.ensure_loaded?(Transmuter.EMLX),
      do: Transmuter.EMLX.params(model, backend_opts),
      else: Map.new(model.params, fn {name, t} -> {name, Nx.backend_transfer(t, backend)} end)
  end

  # The type and the dimensions of a tensor input. A symbolic dimension is nil.
  defp value_type(%{type: %{value: {:tensor_type, %{elem_type: elem_type, shape: shape}}}}) do
    dims =
      for d <- (shape && shape.dim) || [] do
        case d.value do
          {:dim_value, size} -> size
          _symbolic -> nil
        end
      end

    {if(elem_type == 8, do: :string, else: Ops.dtype(elem_type)), dims}
  end

  defp value_type(_input), do: nil

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

    env =
      body.initializer
      |> Enum.reduce(env, &Map.put(&2, &1.name, Ops.tensor(&1)))
      |> Map.merge(scan_inputs)

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
    env = Enum.reduce(body.initializer, env, &Map.put(&2, &1.name, Ops.tensor(&1)))

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
      for %{g: graph} when graph != nil <- node.attribute,
          name <- outer_reads(graph),
          do: name

    node.input ++ outer
  end

  defp outer_reads(graph) do
    local = Enum.map(graph.initializer, & &1.name) ++ Enum.map(graph.input, & &1.name)
    produced = Enum.flat_map(graph.node, & &1.output)
    Enum.uniq(Enum.flat_map(graph.node, &dependencies/1)) -- (local ++ produced)
  end
end
