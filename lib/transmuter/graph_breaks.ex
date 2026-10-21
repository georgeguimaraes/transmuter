defmodule Transmuter.GraphBreaks do
  @moduledoc """
  Compiles a model whose graph needs values that are only known when it runs, such as the limit of a Range or the boxes of a NonMaxSuppression.

  The graph runs as segments. A segment traces top-level nodes from a position until a node raises Transmuter.UnknownValueError. That node runs eagerly on the values of the segment, and the next segment starts after it. The small values that the break node reads (integer shapes, float scales) become static, also the outer values that its subgraphs read, so later segments fold the shapes they decide. Compiled segments are cached by the shapes of their arguments and the values of their static inputs, with least recently used eviction.

  Static values are those of the model, the static inputs, promoted values and values folded from them. Everything else enters a segment as an argument.
  """

  alias Nx.Defn.{Compiler, Composite}
  alias Transmuter.{Ops, Shared, UnknownValueError}

  # A segment of fewer nodes runs eagerly. The cache keeps :cache_size segments for each
  # position, and a new key at a full position evicts the least recently used one.
  @min_compiled_nodes 20

  @doc false
  def compile(model, inputs, static_inputs, opts) do
    {cache_size, opts} = Keyword.pop(opts, :cache_size, 8)
    compiler = opts[:compiler] || Nx.Defn.default_options()[:compiler] || Nx.Defn.Evaluator
    metadata = %{source: model[:source], compiler: compiler}

    :telemetry.span([:transmuter, :compile], metadata, fn ->
      nodes = List.to_tuple(model.nodes)
      last_use = last_use(model)

      # The plan keeps the nodes once, as a tuple. A copy of a closure, such as in a new
      # process or in :persistent_term, copies each reference to a term again. So the
      # closure of a call holds the plan once.
      plan = %{
        cache_size: cache_size,
        model: Map.delete(model, :nodes),
        source: model[:source],
        nodes: nodes,
        last_use: last_use,
        defined: defined(model),
        constant: constant(model),
        key: make_ref(),
        opts: opts,
        backend: Nx.Defn.to_backend(opts),
        table: :ets.new(__MODULE__, [:public, read_concurrency: true]),
        scope: Shared.scope(),
        random?: Map.get(model, :noise, true) and Transmuter.random?(model.nodes)
      }

      {call, graph_break} =
        with_graph(plan, fn ->
          Transmuter.with_fold_memo(fn -> call(plan, inputs, static_inputs) end)
        end)

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
    # A graph without breaks keeps its compiled function here for the shapes of these
    # inputs, so a call with these shapes does not search the cache. Inputs of other shapes
    # compile on their first call and are cached.
    case segment(plan, initial(plan, inputs, static_inputs), 0) do
      {{:end, entry}, _args} ->
        shapes = input_shapes(inputs)

        fast = fn inputs ->
          if input_shapes(inputs) == shapes,
            do: whole(plan, entry, inputs, static_inputs),
            else: general(plan, inputs, static_inputs)
        end

        {fast, false}

      _break ->
        {&general(plan, &1, static_inputs), true}
    end
  end

  # EMLX traces again in a call whose quantized params it has not compiled.
  defp whole(plan, entry, inputs, static_inputs) do
    state = initial(plan, inputs, static_inputs)
    {args, _static_key, _statics} = live(plan, state, 0)
    with_graph(plan, fn -> state |> merge(run_compiled(entry, encode(args))) |> outputs(plan) end)
  end

  defp general(plan, inputs, static_inputs) do
    with_graph(plan, fn ->
      Transmuter.with_fold_memo(fn ->
        plan |> run(initial(plan, inputs, static_inputs), 0) |> outputs(plan)
      end)
    end)
  end

  # The traces of a call read the plan of the call. The compiled function of a segment keeps
  # its trace function, and each cached segment has a copy of it, so the trace function
  # holds only the key of the plan.
  defp with_graph(plan, fun) do
    key = {__MODULE__, plan.key}

    if Process.get(key) do
      fun.()
    else
      Process.put(key, plan)

      try do
        fun.()
      after
        Process.delete(key)
      end
    end
  end

  # The random key is not a part of the shapes, so a call with a key takes the fast path.
  defp input_shapes(inputs) do
    inputs
    |> Map.delete(Transmuter.random_input())
    |> Map.new(fn
      {name, t = %Nx.Tensor{}} -> {name, {Nx.shape(t), Nx.type(t)}}
      other -> other
    end)
  end

  # The values of one call: names, and the digest of each static value that is not a
  # value of the model. A value gets its digest once, when it becomes static.
  defp initial(plan, inputs, static_inputs) do
    model = plan.model
    cast = &Map.new(&1, fn {k, t} -> {k, Transmuter.upcast(t, model[:compute_type])} end)

    # A call can give its own key, so that it draws the same values again.
    inputs =
      if plan.random?,
        do: Map.put_new_lazy(inputs, Transmuter.random_input(), &Transmuter.random_key/0),
        else: Map.delete(inputs, Transmuter.random_input())

    env =
      model.static
      |> Map.merge(model.params)
      |> Map.merge(cast.(inputs))
      |> Map.merge(cast.(static_inputs))

    %{env: env, static: digests(cast.(static_inputs))}
  end

  defp digests(values), do: Map.new(values, fn {name, value} -> {name, Ops.digest(value)} end)

  defp static?(plan, state, name),
    do: Map.has_key?(state.static, name) or Map.has_key?(plan.model.static, name)

  defp run(plan, state, pos) when pos >= tuple_size(plan.nodes), do: state

  defp run(plan, state, pos) do
    case segment(plan, state, pos) do
      {{:end, entry}, args} ->
        merge(state, run_compiled(entry, args))

      {{:break, entry}, args} ->
        state = merge(state, run_compiled(entry, args))
        run(plan, run_break(plan, state, entry.stop), entry.stop + 1)

      {{:eager, stop}, _args} ->
        state = run_eager(plan, state, pos, stop)
        if stop == :end, do: state, else: run(plan, run_break(plan, state, stop), stop + 1)
    end
  end

  # Finds or builds the compiled segment that starts at pos for these values, and gives
  # its arguments.
  defp segment(plan, state, pos) do
    {args, static_key, statics} = live(plan, state, pos)
    shapes = Enum.map(args, fn {name, value} -> {name, template(value)} end)
    key = {pos, shapes, static_key}
    metadata = %{source: plan.source, position: pos, shapes: shapes}
    args = encode(args)

    case lookup(plan, key) do
      {:ok, entry} ->
        :telemetry.execute([:transmuter, :cache, :hit], %{count: 1}, metadata)
        {classify(entry), args}

      :miss ->
        :telemetry.execute([:transmuter, :cache, :miss], %{count: 1}, metadata)
        {build(plan, key, statics.(), args, metadata), args}
    end
  end

  defp build(plan, key = {pos, _shapes, _static_key}, statics, args, metadata) do
    entry =
      :telemetry.span([:transmuter, :segment, :compile], metadata, fn ->
        ref = make_ref()
        key = plan.key
        fun = fn args -> trace(key, statics, args, pos, ref) end
        entry = compile_segment(plan, fun, args, pos, ref)
        Process.delete(ref)
        {entry, Map.merge(metadata, %{stop: entry.stop, compiled: entry.compiled != nil})}
      end)

    store(plan, key, entry)
    classify(entry)
  end

  # The compiler traces the segment before it builds a program. A short segment after a
  # break runs eagerly, so its trace stops the compiler. The first segment always compiles,
  # as a model without breaks always did. The compiled function keeps the trace function,
  # so it holds the number of nodes and not the plan.
  defp compile_segment(plan, fun, args, pos, ref) do
    count = tuple_size(plan.nodes)

    bounded = fn args ->
      outputs = fun.(args)
      {stop, _out_names, _produced} = Process.get(ref)
      stop_index = if stop == :end, do: count, else: stop
      if pos > 0 and stop_index - pos < @min_compiled_nodes, do: throw({ref, :eager})
      outputs
    end

    compiled =
      try do
        compile(plan, bounded, args)
      catch
        {^ref, :eager} -> nil
      end

    {stop, out_names, produced} = Process.get(ref)

    %{
      compiled: compiled,
      stop: stop,
      out_names: out_names,
      produced: produced,
      digests: digests(produced)
    }
  end

  # The compiled function keeps its example arguments, so it gets templates. With the
  # values, each cached segment would keep the device tensors of its first call.
  defp compile(plan, fun, args),
    do: Nx.Defn.compile(fun, [Composite.traverse(args, &Nx.to_template/1)], plan.opts)

  defp classify(%{compiled: nil, stop: stop}), do: {:eager, stop}
  defp classify(entry = %{stop: :end}), do: {:end, entry}
  defp classify(entry), do: {:break, entry}

  # The traced body of a segment. It records where it stopped, which traced values it
  # returns and which static values it folded, so the caller can resume after it.
  defp trace(key, statics, args, pos, ref) do
    %{} = plan = Process.get({__MODULE__, key})
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

  defp run_compiled(entry = %{compiled: compiled, out_names: names, produced: produced}, args) do
    outputs = compiled.(args) |> decode()
    %{values: Map.merge(produced, Map.take(outputs, names)), static: entry.digests}
  end

  # Runs nodes from pos up to stop (or to the end) one by one on concrete values.
  defp run_eager(plan, state, pos, stop) do
    stop_index = if stop == :end, do: tuple_size(plan.nodes), else: stop
    ctx = context(plan, state.env)

    Enum.reduce(pos..(stop_index - 1)//1, state, fn i, state ->
      run_node(plan, state, elem(plan.nodes, i), ctx)
    end)
  end

  # The break node runs eagerly. The small values that it reads become static first, so
  # later segments can fold the shapes they decide. A Loop body, for example, reads the
  # sizes of the images of a Qwen2-VL encoder as an outer value.
  defp run_break(plan, state, index) do
    node = elem(plan.nodes, index)
    metadata = %{source: plan.source, position: index, node: node.name, op_type: node.op_type}

    :telemetry.span([:transmuter, :break], metadata, fn ->
      {run_break_node(plan, state, node), metadata}
    end)
  end

  defp run_break_node(plan, state, node) do
    state =
      Enum.reduce(Transmuter.dependencies(node), state, fn name, state ->
        value = Map.get(state.env, name)

        if promote?(value) and not static?(plan, state, name) do
          value = to_host(value)
          merge(state, %{values: %{name => value}, static: %{name => Ops.digest(value)}})
        else
          state
        end
      end)

    run_node(plan, state, node, context(plan, state.env))
  end

  defp run_node(plan, state, node, ctx) do
    env = Transmuter.run_nodes([node], state.env, ctx)
    outputs = Map.take(env, node.output)

    static? =
      Enum.all?(
        Transmuter.value_dependencies(node),
        &(&1 == "" or static?(plan, state, &1))
      )

    # A node with static inputs leaves a large output on the backend, which is not static.
    {static, other} = Map.split_with(outputs, fn {_k, v} -> static? and Ops.static?(v) end)
    other = Map.new(other, fn {k, v} -> {k, to_backend(v, plan.backend)} end)
    merge(state, %{values: Map.merge(static, other), static: digests(static)})
  end

  defp merge(state, %{values: values, static: static}) do
    %{env: Map.merge(state.env, values), static: Map.merge(state.static, static)}
  end

  defp promote?(t = %Nx.Tensor{data: %data{}}) when data != Nx.BinaryBackend do
    if Nx.Type.integer?(Nx.type(t)), do: Nx.size(t) <= 4096, else: Nx.size(t) <= 16
  end

  defp promote?({:sequence, elements}),
    do: Enum.all?(elements, &(promote?(&1) or Ops.static?(&1)))

  defp promote?(_value), do: false

  defp to_host({:sequence, elements}), do: {:sequence, Enum.map(elements, &to_host/1)}
  defp to_host(t = %Nx.Tensor{}), do: Nx.backend_copy(t, Nx.BinaryBackend)
  defp to_host(value), do: value

  defp outputs(state, plan) do
    plan.model.outputs
    |> Map.new(&{&1, Map.fetch!(state.env, &1)})
    |> Map.reject(fn {_name, value} -> Ops.empty?(value) end)
    |> Map.new(fn {name, value} -> {name, to_backend(value, plan.backend)} end)
    |> Transmuter.host_outputs(Compiler)
  end

  defp device_value?(%Nx.Tensor{}), do: true
  defp device_value?({:sequence, elements}), do: Enum.all?(elements, &device_value?/1)
  defp device_value?(_value), do: false

  # The values that nodes at pos or later read: the arguments of a segment, the key of the
  # static values that can change between calls, and a function that gives the static
  # values that the segment captures. Zero-size values, strings and empty optionals hold no
  # device data, so they are captured like static values whatever their origin.
  defp live(plan, state, pos) do
    {names, constants} = live_names(plan, state, pos)

    {args, statics, key} =
      Enum.reduce(names, {[], [], []}, fn name, acc = {args, statics, key} ->
        case state.env do
          %{^name => value} ->
            cond do
              digest = state.static[name] ->
                {args, [{name, value} | statics], [{name, digest} | key]}

              device_value?(value) ->
                {[{name, value} | args], statics, key}

              true ->
                {args, [{name, value} | statics], [{name, Ops.digest(value)} | key]}
            end

          %{} ->
            acc
        end
      end)

    {args, key, fn -> Map.merge(Map.take(state.env, constants), Map.new(statics)) end}
  end

  # The names of the values that a call has before pos and that nodes at pos or later read,
  # in an order that is the same for each call. The static values with a name of
  # plan.constant are the same in every call, so a call reads them only to build a
  # segment. Each position computes the names once.
  defp live_names(plan, state, pos) do
    key = {:live, pos}

    with nil <- Shared.get(plan.scope, key) do
      names =
        Enum.split_with(
          for(
            {name, index} <- plan.defined,
            index < pos and Map.get(plan.last_use, name, -1) >= pos,
            do: name
          ),
          &(not (Map.has_key?(plan.constant, &1) and static?(plan, state, &1)))
        )

      Shared.put(plan.scope, key, names)
      names
    end
  end

  defp context(plan, env) do
    %{
      opset: plan.model.opset,
      compute_type: plan.model[:compute_type],
      key: Map.get(env, Transmuter.random_input()),
      backend: plan[:backend]
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

  # The names whose values are the same in every call: the static values of the model and
  # the outputs of nodes that read only such values and draw no random values.
  defp constant(model) do
    Enum.reduce(model.nodes, Map.new(model.static, fn {name, _} -> {name, true} end), fn node,
                                                                                         constant ->
      if not Transmuter.random?([node]) and
           Enum.all?(Transmuter.dependencies(node), &(&1 == "" or Map.has_key?(constant, &1))),
         do: Enum.reduce(node.output, constant, &Map.put(&2, &1, true)),
         else: constant
    end)
  end

  # Each name with the index of the node that computes it, or -1 for a value that a call
  # starts with.
  defp defined(model) do
    initial = Map.keys(model.static) ++ Map.keys(model.params) ++ model.inputs

    Enum.uniq_by(
      Enum.map([Transmuter.random_input() | initial], &{&1, -1}) ++
        for(
          {node, i} <- Enum.with_index(model.nodes),
          name <- node.output,
          name != "",
          do: {name, i}
        ),
      &elem(&1, 0)
    )
  end

  # A sequence crosses a segment boundary as a tuple of its tensors. Its other elements,
  # such as zero-size values, hold no data, so the key keeps them with their positions.
  defp encode(values) do
    Map.new(values, fn
      {name, {:sequence, elements}} ->
        {tensors, others} =
          elements |> Enum.with_index() |> Enum.split_with(&is_struct(elem(&1, 0), Nx.Tensor))

        {{:sequence, name, others}, tensors |> Enum.map(&elem(&1, 0)) |> List.to_tuple()}

      other ->
        other
    end)
  end

  defp decode(values) do
    Map.new(values, fn
      {{:sequence, name, others}, tensors} ->
        elements =
          Enum.reduce(others, Tuple.to_list(tensors), fn {value, i}, elements ->
            List.insert_at(elements, i, value)
          end)

        {name, {:sequence, elements}}

      other ->
        other
    end)
  end

  defp traced?(%Nx.Tensor{data: %Nx.Defn.Expr{}}), do: true
  defp traced?({:sequence, elements}), do: Enum.any?(elements, &traced?/1)
  defp traced?(_value), do: false

  defp template(t = %Nx.Tensor{}), do: {Nx.shape(t), Nx.type(t)}
  defp template({:sequence, elements}), do: Enum.map(elements, &template/1)

  defp to_backend(t = %Nx.Tensor{}, backend), do: Nx.backend_transfer(t, backend)

  defp to_backend({:sequence, elements}, backend),
    do: {:sequence, Enum.map(elements, &to_backend(&1, backend))}

  defp to_backend(other, _backend), do: other

  # The entries are in the scope of Transmuter.Shared, because each read from ETS copies
  # the compiled function. The cache table has the time of use of each key. Both belong to
  # the process that called Transmuter.compile/3. When that process is gone, segments
  # compile again on each call.
  defp lookup(plan, key) do
    case Shared.get(plan.scope, key) do
      nil ->
        :miss

      entry ->
        :ets.update_element(plan.table, key, {2, :erlang.unique_integer([:monotonic])})
        {:ok, entry}
    end
  rescue
    ArgumentError -> :miss
  end

  defp store(plan, key = {pos, _templates, _static_key}, entry) do
    used =
      :ets.select(plan.table, [
        {{:"$1", :"$2"}, [{:==, {:element, 1, :"$1"}, pos}], [{{:"$1", :"$2"}}]}
      ])

    if length(used) >= plan.cache_size do
      {oldest = {_pos, shapes, _static_key}, _used} = Enum.min_by(used, &elem(&1, 1))
      :ets.delete(plan.table, oldest)
      Shared.erase(plan.scope, oldest)

      :telemetry.execute([:transmuter, :cache, :evict], %{count: 1}, %{
        source: plan.source,
        position: pos,
        shapes: shapes,
        cache_size: plan.cache_size
      })
    end

    :ets.insert(plan.table, {key, :erlang.unique_integer([:monotonic])})
    Shared.put(plan.scope, key, entry)
  rescue
    ArgumentError -> :ok
  end
end
