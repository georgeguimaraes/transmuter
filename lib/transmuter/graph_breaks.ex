defmodule Transmuter.GraphBreaks do
  @moduledoc """
  Compiles a model whose graph needs values that are only known when it runs, such as the limit of a Range or the boxes of a NonMaxSuppression.

  The graph runs as segments. A segment traces top-level nodes from a position until a node raises Transmuter.UnknownValueError. That node runs eagerly on the values of the segment, and the next segment starts after it. Small inputs of the break node (integer shapes, float scales) become static, so later segments fold the shapes they decide. Compiled segments are cached by the shapes of their arguments and the values of their static inputs, with least recently used eviction.

  Static values are those of the model, the static inputs, promoted values and values folded from them. Everything else enters a segment as an argument.
  """

  alias Nx.Defn.Compiler
  alias Transmuter.{Ops, UnknownValueError}

  # A segment of fewer nodes runs eagerly. The cache keeps :cache_size segments for each
  # position, and a new key at a full position evicts the least recently used one.
  @min_compiled_nodes 20

  @doc false
  def compile(model, inputs, static_inputs, opts) do
    {cache_size, opts} = Keyword.pop(opts, :cache_size, 8)
    compiler = opts[:compiler] || Nx.Defn.default_options()[:compiler] || Nx.Defn.Evaluator
    metadata = %{source: model[:source], compiler: compiler}

    :telemetry.span([:transmuter, :compile], metadata, fn ->
      plan = %{
        cache_size: cache_size,
        model: model,
        source: model[:source],
        nodes: List.to_tuple(model.nodes),
        last_use: last_use(model),
        opts: opts,
        backend: Nx.Defn.to_backend(opts),
        table: :ets.new(__MODULE__, [:public, read_concurrency: true]),
        random?: Transmuter.random?(model.nodes)
      }

      {call, graph_break} = call(plan, inputs, static_inputs)
      {instrument(call, plan.source), Map.put(metadata, :graph_break, graph_break)}
    end)
  end

  defp instrument(call, source) do
    metadata = %{source: source}

    fn inputs ->
      :telemetry.span([:transmuter, :call], metadata, fn -> {call.(inputs), metadata} end)
    end
  end

  defp call(plan, inputs, static_inputs) do
    general = fn inputs ->
      plan |> run(initial(plan, inputs, static_inputs), 0) |> outputs(plan)
    end

    # A graph without breaks keeps its compiled function here for the shapes of these
    # inputs, because a read from the cache table copies the function. Inputs of other
    # shapes compile on their first call and are cached.
    case segment(plan, initial(plan, inputs, static_inputs), 0) do
      {:end, entry} ->
        shapes = input_shapes(inputs)

        fast = fn inputs ->
          if input_shapes(inputs) == shapes do
            state = initial(plan, inputs, static_inputs)
            state |> merge(run_compiled(entry, arguments(plan, state, 0))) |> outputs(plan)
          else
            general.(inputs)
          end
        end

        {fast, false}

      _break ->
        {general, true}
    end
  end

  defp input_shapes(inputs) do
    Map.new(inputs, fn
      {name, t = %Nx.Tensor{}} -> {name, {Nx.shape(t), Nx.type(t)}}
      other -> other
    end)
  end

  # The values of one call: names, the set of static names and the random key.
  defp initial(plan, inputs, static_inputs) do
    model = plan.model
    cast = &Map.new(&1, fn {k, t} -> {k, Transmuter.upcast(t, model[:compute_type])} end)

    inputs =
      if plan.random?,
        do: Map.put(inputs, Transmuter.random_input(), Transmuter.random_key()),
        else: inputs

    env =
      model.static
      |> Map.merge(model.params)
      |> Map.merge(cast.(inputs))
      |> Map.merge(cast.(static_inputs))

    %{env: env, static: MapSet.new(Map.keys(model.static) ++ Map.keys(static_inputs))}
  end

  defp run(plan, state, pos) when pos >= tuple_size(plan.nodes), do: state

  defp run(plan, state, pos) do
    case segment(plan, state, pos) do
      {:end, entry} ->
        merge(state, run_compiled(entry, arguments(plan, state, pos)))

      {:break, entry} ->
        state = merge(state, run_compiled(entry, arguments(plan, state, pos)))
        run(plan, run_break(plan, state, entry.stop), entry.stop + 1)

      {:eager, stop} ->
        state = run_eager(plan, state, pos, stop)
        if stop == :end, do: state, else: run(plan, run_break(plan, state, stop), stop + 1)
    end
  end

  # Finds or builds the compiled segment that starts at pos for these values.
  defp segment(plan, state, pos) do
    args = arguments(plan, state, pos)
    statics = live(plan, state, pos, &(not argument?(state, &1, &2)))
    shapes = templates(args)
    key = {pos, shapes, static_key(plan, statics)}
    metadata = %{source: plan.source, position: pos, shapes: shapes}

    case lookup(plan.table, key) do
      {:ok, entry} ->
        :telemetry.execute([:transmuter, :cache, :hit], %{count: 1}, metadata)
        classify(entry)

      :miss ->
        :telemetry.execute([:transmuter, :cache, :miss], %{count: 1}, metadata)
        build(plan, key, statics, args, metadata)
    end
  end

  defp build(plan, key = {pos, _shapes, _static_key}, statics, args, metadata) do
    entry =
      :telemetry.span([:transmuter, :segment, :compile], metadata, fn ->
        ref = make_ref()
        fun = fn args -> trace(plan, statics, args, pos, ref) end
        entry = compile_segment(plan, fun, args, pos, ref)
        Process.delete(ref)
        {entry, Map.merge(metadata, %{stop: entry.stop, compiled: entry.compiled != nil})}
      end)

    store(plan, key, entry)
    classify(entry)
  end

  # The first segment compiles at once, as a model without breaks always did. A later
  # segment is traced first, and a short one runs eagerly without compiling.
  defp compile_segment(plan, fun, args, 0, ref) do
    compiled = Nx.Defn.compile(fun, [args], plan.opts)
    {stop, out_names, produced} = Process.get(ref)
    %{compiled: compiled, stop: stop, out_names: out_names, produced: produced}
  end

  defp compile_segment(plan, fun, args, pos, ref) do
    Nx.Defn.debug_expr(fun).(args)
    {stop, out_names, produced} = Process.get(ref)
    stop_index = if stop == :end, do: tuple_size(plan.nodes), else: stop

    compiled =
      if stop_index - pos >= @min_compiled_nodes, do: Nx.Defn.compile(fun, [args], plan.opts)

    %{compiled: compiled, stop: stop, out_names: out_names, produced: produced}
  end

  defp classify(%{compiled: nil, stop: stop}), do: {:eager, stop}
  defp classify(entry = %{stop: :end}), do: {:end, entry}
  defp classify(entry), do: {:break, entry}

  # The traced body of a segment. It records where it stopped, which traced values it
  # returns and which static values it folded, so the caller can resume after it.
  defp trace(plan, statics, args, pos, ref) do
    env = Map.merge(statics, decode(args))
    ctx = context(plan, env)
    {env, stop} = run_until_break(plan, env, ctx, pos)
    stop_index = if stop == :end, do: tuple_size(plan.nodes), else: stop

    live =
      for i <- pos..(stop_index - 1)//1,
          name <- elem(plan.nodes, i).output,
          name != "" and Map.get(plan.last_use, name, -1) >= stop_index,
          into: %{},
          do: {name, Map.fetch!(env, name)}

    {traced, produced} = Map.split_with(live, fn {_name, value} -> traced?(value) end)
    Process.put(ref, {stop, Map.keys(traced), produced})
    traced |> encode() |> Transmuter.tag_literals(Compiler.current())
  end

  defp run_until_break(plan, env, ctx, pos) do
    Enum.reduce_while(pos..(tuple_size(plan.nodes) - 1)//1, {env, :end}, fn i, {env, _} ->
      try do
        {:cont, {Transmuter.run_nodes([elem(plan.nodes, i)], env, ctx), :end}}
      rescue
        UnknownValueError -> {:halt, {env, i}}
      end
    end)
  end

  defp run_compiled(%{compiled: compiled, out_names: names, produced: produced}, args) do
    outputs = compiled.(args) |> decode()
    %{values: Map.merge(produced, Map.take(outputs, names)), static: Map.keys(produced)}
  end

  # Runs nodes from pos up to stop (or to the end) one by one on concrete values.
  defp run_eager(plan, state, pos, stop) do
    stop_index = if stop == :end, do: tuple_size(plan.nodes), else: stop
    ctx = context(plan, state.env)

    Enum.reduce(pos..(stop_index - 1)//1, state, fn i, state ->
      run_node(plan, state, elem(plan.nodes, i), ctx)
    end)
  end

  # The break node runs eagerly. Its small inputs become static first, so later
  # segments can fold the shapes they decide.
  defp run_break(plan, state, index) do
    node = elem(plan.nodes, index)
    metadata = %{source: plan.source, position: index, node: node.name, op_type: node.op_type}

    :telemetry.span([:transmuter, :break], metadata, fn ->
      {run_break_node(plan, state, node), metadata}
    end)
  end

  defp run_break_node(plan, state, node) do
    state =
      Enum.reduce(node.input, state, fn name, state ->
        value = Map.get(state.env, name)

        if promote?(value) and not MapSet.member?(state.static, name),
          do: %{
            env: Map.put(state.env, name, Nx.backend_copy(value, Nx.BinaryBackend)),
            static: MapSet.put(state.static, name)
          },
          else: state
      end)

    run_node(plan, state, node, context(plan, state.env))
  end

  defp run_node(plan, state, node, ctx) do
    env = Transmuter.run_nodes([node], state.env, ctx)
    outputs = Map.take(env, node.output)

    static? =
      Enum.all?(Transmuter.dependencies(node), &(&1 == "" or MapSet.member?(state.static, &1)))

    outputs =
      if static?,
        do: outputs,
        else: Map.new(outputs, fn {k, v} -> {k, to_backend(v, plan.backend)} end)

    merge(state, %{values: outputs, static: if(static?, do: Map.keys(outputs), else: [])})
  end

  defp merge(state, %{values: values, static: static}) do
    %{env: Map.merge(state.env, values), static: MapSet.union(state.static, MapSet.new(static))}
  end

  defp promote?(t = %Nx.Tensor{data: %data{}}) when data != Nx.BinaryBackend do
    if Nx.Type.integer?(Nx.type(t)), do: Nx.size(t) <= 4096, else: Nx.size(t) <= 16
  end

  defp promote?(_value), do: false

  defp outputs(state, plan) do
    plan.model.outputs
    |> Map.new(&{&1, Map.fetch!(state.env, &1)})
    |> Map.reject(fn {_name, value} -> Ops.empty?(value) end)
    |> Map.new(fn {name, value} -> {name, to_backend(value, plan.backend)} end)
    |> Transmuter.host_outputs(Compiler)
  end

  # The non-static values that nodes at pos or later read.
  defp arguments(plan, state, pos) do
    plan |> live(state, pos, &argument?(state, &1, &2)) |> encode()
  end

  # Zero-size values, strings and empty optionals hold no device data, so they are
  # captured like static values whatever their origin.
  defp argument?(state, name, value),
    do: not MapSet.member?(state.static, name) and device_value?(value)

  defp device_value?(%Nx.Tensor{}), do: true
  defp device_value?({:sequence, elements}), do: Enum.all?(elements, &device_value?/1)
  defp device_value?(_value), do: false

  defp live(plan, state, pos, keep?) do
    for {name, value} <- state.env,
        keep?.(name, value),
        Map.get(plan.last_use, name, -1) >= pos,
        into: %{},
        do: {name, value}
  end

  defp context(plan, env) do
    %{
      opset: plan.model.opset,
      compute_type: plan.model[:compute_type],
      key: Map.get(env, Transmuter.random_input())
    }
  end

  # The index of the last node that reads each name. Graph outputs and the random key
  # stay live to the end.
  defp last_use(model) do
    reads =
      for {node, i} <- Enum.with_index(model.nodes),
          name <- Transmuter.dependencies(node),
          name != "",
          reduce: %{},
          do: (acc -> Map.put(acc, name, i))

    Enum.reduce([Transmuter.random_input() | model.outputs], reads, &Map.put(&2, &1, :infinity))
  end

  # A sequence of tensors crosses a segment boundary as a tuple.
  defp encode(values) do
    Map.new(values, fn
      {name, {:sequence, elements}} -> {{:sequence, name}, List.to_tuple(elements)}
      other -> other
    end)
  end

  defp decode(values) do
    Map.new(values, fn
      {{:sequence, name}, elements} -> {name, {:sequence, Tuple.to_list(elements)}}
      other -> other
    end)
  end

  defp traced?(%Nx.Tensor{data: %Nx.Defn.Expr{}}), do: true
  defp traced?({:sequence, elements}), do: Enum.any?(elements, &traced?/1)
  defp traced?(_value), do: false

  defp templates(args) do
    args
    |> Enum.map(fn {name, value} -> {name, template(value)} end)
    |> Enum.sort()
  end

  defp template(t = %Nx.Tensor{}), do: {Nx.shape(t), Nx.type(t)}
  defp template(tuple) when is_tuple(tuple), do: tuple |> Tuple.to_list() |> Enum.map(&template/1)

  # Values of the static inputs that are not part of the model, which every call shares.
  defp static_key(plan, statics) do
    statics
    |> Map.drop(Map.keys(plan.model.static))
    |> Enum.sort()
    |> Enum.map(fn {name, value} -> {name, digest(value)} end)
  end

  defp digest(t = %Nx.Tensor{}), do: {Nx.shape(t), Nx.type(t), :erlang.md5(Nx.to_binary(t))}
  defp digest({:sequence, elements}), do: Enum.map(elements, &digest/1)
  defp digest(other), do: :erlang.md5(:erlang.term_to_binary(other))

  defp to_backend(t = %Nx.Tensor{}, backend), do: Nx.backend_transfer(t, backend)

  defp to_backend({:sequence, elements}, backend),
    do: {:sequence, Enum.map(elements, &to_backend(&1, backend))}

  defp to_backend(other, _backend), do: other

  # The cache table belongs to the process that called Transmuter.compile/3. When that
  # process is gone, segments compile again on each call.
  defp lookup(table, key) do
    case :ets.lookup(table, key) do
      [{^key, entry, _used}] ->
        :ets.update_element(table, key, {3, :erlang.unique_integer([:monotonic])})
        {:ok, entry}

      [] ->
        :miss
    end
  rescue
    ArgumentError -> :miss
  end

  defp store(plan, key = {pos, _templates, _static_key}, entry) do
    entries = :ets.match_object(plan.table, {{pos, :_, :_}, :_, :_})

    if length(entries) >= plan.cache_size do
      {oldest = {_pos, shapes, _static_key}, _entry, _used} = Enum.min_by(entries, &elem(&1, 2))
      :ets.delete(plan.table, oldest)

      :telemetry.execute([:transmuter, :cache, :evict], %{count: 1}, %{
        source: plan.source,
        position: pos,
        shapes: shapes,
        cache_size: plan.cache_size
      })
    end

    :ets.insert(plan.table, {key, entry, :erlang.unique_integer([:monotonic])})
  rescue
    ArgumentError -> :ok
  end
end
