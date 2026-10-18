defmodule Transmuter do
  @moduledoc """
  Runs ONNX models on any Nx backend.

  Transmuter reads an ONNX file, such as an export from the Hugging Face Hub, and
  turns its graph into Nx operations. It does not port model architectures. The
  graph that the exporter traced is the model.

      model = Transmuter.load!({:hf, "Xenova/all-MiniLM-L6-v2", "onnx/model.onnx"}, backend: EMLX.Backend)
      run = Transmuter.compile(model, inputs, compiler: EMLX)
      %{"last_hidden_state" => hidden} = run.(inputs)

  Shapes are resolved when the function is compiled, so `compile/3` returns a
  function for the shapes of the inputs it was given. Compile again for other
  shapes. `run/4` is the uncompiled function, for `Nx.Defn.jit/2`, gradients or
  eager execution.
  """

  alias Nx.Defn.{Compiler, Composite, Expr, Tree}
  alias Transmuter.Onnx.ModelProto
  alias Transmuter.Ops
  alias Transmuter.Ops.Random

  @max_loop_iterations 100_000

  # Small initializers and integer initializers are shape data, so they stay static.
  # Large integer initializers are packed quantized weights, so they are params.
  @static_max_size 16
  @static_int_max_size 4096

  @random_key "transmuter.random_key"

  @doc """
  Loads an ONNX model.

  The source is a local path or `{:hf, repo, file}`. Hub files are downloaded
  once into `~/.cache/transmuter`, together with their external data files.

  ## Options

    * `:cache_dir` - the directory for Hub downloads

    * `:backend` - moves the params to this backend, for example `EMLX.Backend`
      or `{EXLA.Backend, client: :cuda}`. On EMLX, 4-bit and 8-bit `MatMulNBits`
      weights stay packed, see `Transmuter.EMLX`. By default the params stay on
      `Nx.BinaryBackend`.

    * `:compute_type` - runs f16 and bf16 graphs in this type, for example `:f32`.
      ONNX Runtime on the CPU computes half-precision graphs in f32, so use `:f32`
      to match it. By default the graph keeps its own types.

    * `:trainable` - when `true`, every float initializer is a param, so small
      weights such as a 1-element bias get gradients. By default float
      initializers with up to 16 elements are folded as constants.

  Returns a map with:

    * `:params` - the weights, on the `:backend` when it is given
    * `:inputs` and `:outputs` - the graph input and output names
  """
  def load!(source, opts \\ [])

  def load!({:hf, repo, file}, opts), do: repo |> download!(file, opts) |> load!(opts)

  def load!(path, opts) when is_binary(path) do
    dir = Path.dirname(path)
    model = ModelProto.decode!(File.read!(path))
    graph = model.graph

    trainable = Keyword.get(opts, :trainable, false)

    {static, params} =
      graph.initializer
      |> Enum.map(&{&1.name, Ops.tensor(&1, dir)})
      |> Enum.split_with(fn {_, t} -> static_initializer?(t, trainable) end)

    initializers = Enum.map(graph.initializer, & &1.name)
    inputs = for i <- graph.input, i.name not in initializers, do: i.name
    compute_type = opts[:compute_type]

    model = %{
      nodes: toposort(graph.node, inputs ++ initializers),
      static: Map.new(static, fn {k, t} -> {k, upcast(t, compute_type)} end),
      params: Map.new(params, fn {k, t} -> {k, upcast(t, compute_type)} end),
      compute_type: compute_type,
      inputs: inputs,
      outputs: Enum.map(graph.output, & &1.name),
      opset:
        Enum.find_value(model.opset_import, &(&1.domain in [nil, "", "ai.onnx"] && &1.version))
    }

    if backend = opts[:backend], do: %{model | params: move_params(model, backend)}, else: model
  end

  @doc """
  Compiles the model for the shapes and types of `inputs`.

  Returns a function that takes the inputs and returns the outputs. The params
  are the model's own, so load it with `:backend` first.

  ## Options

    * `:static_inputs` - inputs known when the model compiles, see `run/4`

  Other options go to `Nx.Defn.compile/3`, for example `compiler: EMLX`.

  Each call of a model with random operators draws new values.
  """
  def compile(model, inputs, opts \\ []) do
    {static_inputs, opts} = Keyword.pop(opts, :static_inputs, %{})

    with_key =
      if random?(model.nodes), do: &Map.put(&1, @random_key, random_key()), else: & &1

    fun =
      Nx.Defn.compile(&run(model, &1, &2, static_inputs), [model.params, with_key.(inputs)], opts)

    fn inputs -> fun.(model.params, with_key.(inputs)) end
  end

  @doc """
  Runs the model graph. Call it inside `Nx.Defn.jit/2` or `Nx.Defn.compile/3`.

  `static_inputs` are graph inputs whose values are known when the function is
  traced. Give them as `Nx.BinaryBackend` tensors. Use them for flags that select
  a branch, such as `use_cache_branch`, and for inputs that decide an output
  shape, such as the `input_ids` that feed `NonZero` in Grounding DINO. Each new
  value compiles again.

  Outputs that are empty are left out of the result.

  Random operators draw from the `Nx.Random` key in the `"#{@random_key}"` input.
  Without it, each call makes a new key. A function that you compile yourself makes
  the key when it is traced, so it gives the same values on every call unless you
  pass the key.
  """
  def run(model, params, inputs, static_inputs \\ %{}) do
    {key, inputs} = Map.pop_lazy(inputs, @random_key, &random_key/0)
    ctx = %{opset: model.opset, compute_type: model[:compute_type], key: key}
    cast = &Map.new(&1, fn {k, t} -> {k, upcast(t, ctx.compute_type)} end)

    env =
      model.static
      |> Map.merge(params)
      |> Map.merge(cast.(inputs))
      |> Map.merge(cast.(static_inputs))

    env = run_nodes(model.nodes, env, ctx)

    for(name <- model.outputs, (t = Map.fetch!(env, name)) != :empty, into: %{}, do: {name, t})
    |> host_outputs(Compiler.current())
    |> tag_literals(Compiler.current())
  end

  # Strings, sequences and empty optionals cannot leave a compiled function. Without a
  # compiler, a sequence comes back as a list.
  defp host_outputs(outputs, nil) do
    Map.new(outputs, fn
      {name, {:sequence, elements}} -> {name, elements}
      other -> other
    end)
  end

  defp host_outputs(outputs, _compiler) do
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

  # EMLX 0.5 keys its compile cache by the structure of the graph, and a literal tensor
  # adds only its shape and type to the key. Two graphs that differ only in literal
  # values then share one compiled program. A metadata node with a hash of the value
  # puts the value in the key. EMLX compiles a metadata node as its inner node.
  defp tag_literals(outputs, EMLX) do
    {outputs, _seen} = Composite.traverse(outputs, %{}, &tag_literal/2)
    outputs
  end

  defp tag_literals(outputs, _compiler), do: outputs

  defp tag_literal(t = %Nx.Tensor{data: %Expr{id: id, op: op, args: args}}, seen) do
    case seen do
      %{^id => tagged} ->
        {tagged, seen}

      %{} when op == :tensor ->
        tagged = Expr.metadata(t, %{literal: :erlang.md5(Nx.to_binary(hd(args)))})
        {tagged, Map.put(seen, id, tagged)}

      %{} ->
        {args, seen} = Tree.apply_args(t, :all, seen, &tag_literal/2)
        tagged = put_in(t.data.args, args)
        {tagged, Map.put(seen, id, tagged)}
    end
  end

  defp tag_literal(t = %Nx.Tensor{shape: {}}, seen), do: {t, seen}
  defp tag_literal(t = %Nx.Tensor{}, seen), do: tag_literal(Expr.tensor(t), seen)

  defp run_nodes(nodes, env, ctx) do
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
    unless Ops.known?(condition) do
      raise ArgumentError,
            "If needs a static condition, pass the graph inputs it depends on as static_inputs"
    end

    branch =
      if Nx.to_number(Nx.any(condition)) == 1,
        do: Ops.attr(node, "then_branch"),
        else: Ops.attr(node, "else_branch")

    env = Enum.reduce(branch.initializer, env, &Map.put(&2, &1.name, Ops.tensor(&1)))
    env = run_nodes(branch.node, env, ctx)
    Enum.map(branch.output, &Map.fetch!(env, &1.name))
  end

  # Unrolls the loop while tracing, so the trip count and the condition must be static.
  defp run_loop(node, [trip_count, condition | initial], env, ctx) do
    body = Ops.attr(node, "body")
    [counter, condition_name | carried_names] = Enum.map(body.input, & &1.name)
    trips = if trip_count, do: Nx.to_number(trip_count), else: @max_loop_iterations
    env = Enum.reduce(body.initializer, env, &Map.put(&2, &1.name, Ops.tensor(&1)))

    {carried, scans} =
      Enum.reduce_while(0..(trips - 1)//1, {initial, [], condition}, fn i,
                                                                        {carried, scans,
                                                                         condition} ->
        if condition != nil and Nx.to_number(Nx.any(static!(condition, "Loop"))) == 0 do
          {:halt, {carried, scans, condition}}
        else
          step =
            env
            |> Map.merge(Map.new(Enum.zip(carried_names, carried)))
            |> Map.merge(%{counter => Ops.static_tensor(i, :s64), condition_name => condition})

          [next_condition | outputs] =
            body.node
            |> run_nodes(step, ctx)
            |> then(&Enum.map(body.output, fn o -> Map.fetch!(&1, o.name) end))

          {next_carried, scan_values} = Enum.split(outputs, length(carried))
          {:cont, {next_carried, [scan_values | scans], next_condition}}
        end
      end)
      |> then(fn {carried, scans, _} -> {carried, scans} end)

    stacked =
      case scans do
        [] -> []
        _ -> scans |> Enum.reverse() |> Enum.zip_with(&Nx.stack/1)
      end

    carried ++ stacked
  end

  # A key is two u32 words. It is on the default backend, so eager runs draw there.
  defp random_key do
    Nx.tensor([:rand.uniform(2 ** 32) - 1, :rand.uniform(2 ** 32) - 1], type: :u32)
  end

  defp random?(nodes) do
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

  defp static_initializer?(:empty, _trainable), do: true
  defp static_initializer?({:strings, _shape, _values}, _trainable), do: true

  defp static_initializer?(t, trainable) do
    if Nx.Type.integer?(Nx.type(t)),
      do: Nx.size(t) <= @static_int_max_size,
      else: not trainable and Nx.size(t) <= @static_max_size
  end

  defp upcast(t = %Nx.Tensor{}, type) when type != nil do
    if Nx.type(t) in [{:f, 16}, {:bf, 16}], do: Nx.as_type(t, type), else: t
  end

  defp upcast(t, _type), do: t

  # Unrolls Scan over the static length of its scan inputs. Each step takes one slice of
  # every scan input; per-step outputs are stacked along their output axes.
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
    env = Enum.reduce(body.initializer, env, &Map.put(&2, &1.name, Ops.tensor(&1)))

    {steps, final_states} =
      Enum.map_reduce(0..(length - 1), states, fn t, states ->
        slices =
          for {x, axis, reversed} <- Enum.zip([scans, in_axes, in_reversed]) do
            x
            |> Nx.slice_along_axis(if(reversed, do: length - 1 - t, else: t), 1, axis: axis)
            |> Nx.squeeze(axes: [axis])
          end

        step =
          env
          |> Map.merge(Map.new(Enum.zip(state_names, states)))
          |> Map.merge(Map.new(Enum.zip(scan_names, slices)))

        result = run_nodes(body.node, step, ctx)

        {new_states, outputs} =
          body.output |> Enum.map(&Map.fetch!(result, &1.name)) |> Enum.split(length(states))

        {outputs, new_states}
      end)

    output_count = length(body.output) - length(states)
    out_axes = Ops.attr(node, "scan_output_axes", List.duplicate(0, output_count))

    out_reversed =
      node
      |> Ops.attr("scan_output_directions", List.duplicate(0, output_count))
      |> Enum.map(&(&1 == 1))

    stacked =
      for {j, axis, reversed} <- Enum.zip([0..(output_count - 1)//1, out_axes, out_reversed]) do
        values = Enum.map(steps, &Enum.at(&1, j))
        values = if reversed, do: Enum.reverse(values), else: values
        Nx.stack(values, axis: normalize_axis(axis, Nx.rank(hd(values)) + 1))
      end

    final_states ++ stacked
  end

  defp normalize_axis(axis, rank) when axis < 0, do: axis + rank
  defp normalize_axis(axis, _rank), do: axis

  defp static!(t, op) do
    unless Ops.known?(t) do
      raise ArgumentError,
            "#{op} needs a static condition, pass the graph inputs it depends on as static_inputs"
    end

    t
  end

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

  # A node with subgraphs (If branches) also depends on the outer values its branches read.
  defp dependencies(node) do
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

  defp download!(repo, file, opts) do
    dir =
      Path.join(
        opts[:cache_dir] || Path.join([System.user_home!(), ".cache", "transmuter"]),
        repo
      )

    path = Path.join(dir, file)

    unless File.exists?(path) do
      %{"siblings" => siblings} = hub_get!("https://huggingface.co/api/models/#{repo}").body

      data_files =
        for %{"rfilename" => f} <- siblings,
            String.starts_with?(f, file <> "_data") or f == file <> ".data",
            do: f

      for f <- data_files ++ [file] do
        target = Path.join(dir, f)
        File.mkdir_p!(Path.dirname(target))

        hub_get!("https://huggingface.co/#{repo}/resolve/main/#{f}",
          into: File.stream!(target <> ".part")
        )

        File.rename!(target <> ".part", target)
      end
    end

    path
  end

  defp hub_get!(url, opts \\ []) do
    headers =
      if token = System.get_env("HF_TOKEN"), do: [authorization: "Bearer #{token}"], else: []

    Req.get!(url, [headers: headers, receive_timeout: 600_000, redirect_log_level: false] ++ opts)
  end
end
