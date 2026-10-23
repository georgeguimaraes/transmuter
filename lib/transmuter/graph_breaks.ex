defmodule Transmuter.GraphBreaks do
  @moduledoc """
  Compiles a model whose graph needs values that are only known when it runs, such as the limit of a Range or the boxes of a NonMaxSuppression.

  The graph runs as segments. A segment traces top-level nodes from a position until a node raises Transmuter.UnknownValueError. That node runs eagerly on the values of the segment, and the next segment starts after it. The small values that the break node reads (integer shapes, float scales) become static, also the outer values that its subgraphs read, so later segments can fold the shapes they decide. The cache key of a compiled segment holds the shapes of its arguments, the values of the static inputs and the host values that the segment reads. A new value of such a host value compiles that segment again. The cache evicts the least recently used segments.

  Static values are those of the model, the static inputs, promoted values and values folded from them. A promoted value, or a value folded from it, enters a segment as an argument, so its value is not in the key. It stays static when a node of that segment needs its host value. The plan learns these names for each position from its traces. A node can need the host value of an argument that follows only from static values of the call. Then the break is induced, and the names where the way enters the segment stay static at that position. When a small value of an earlier segment is among them, the host gets a copy of it when the segment starts. Values that follow only from shapes and static inputs stay static. Everything else enters a segment as an argument.
  """

  alias Nx.Defn.{Compiler, Composite}
  alias Transmuter.{Ops, Shared, UnknownValueError}

  # A segment of fewer nodes runs eagerly. The cache keeps :cache_size segments for each
  # position, and a new key at a full position evicts the least recently used one.
  @min_compiled_nodes 20

  @doc false
  def compile(model, inputs, static_inputs, opts) do
    {opts, metadata} = options(model, opts)

    :telemetry.span([:transmuter, :compile], metadata, fn ->
      plan = plan(model, opts, static_inputs)
      {call, graph_break} = start(plan, inputs, static_inputs)
      {instrument(call, plan.source), Map.put(metadata, :graph_break, graph_break)}
    end)
  end

  @doc false
  # Compiles the model for inputs, calls it once and returns the outputs with the segments
  # that the call ran: %{start, stop, compiled, reason}, in the order of the graph. The plan
  # leaves no segment and no table behind, also when the call raises.
  def run_once(model, inputs, static_inputs, opts) do
    {opts, _metadata} = options(model, opts)
    plan = plan(model, opts, static_inputs)

    try do
      {call, _graph_break} = start(plan, inputs, static_inputs)
      outputs = call.(inputs)

      segments =
        for {key = {start, _shapes, _static_key}, _used} <- :ets.tab2list(plan.table),
            entry = Shared.get(plan.scope, key),
            uniq: true do
          %{start: start, stop: entry.stop, compiled: entry.compiled != nil, reason: entry.reason}
        end

      {outputs, Enum.sort_by(segments, & &1.start)}
    after
      Shared.erase_scope(plan.scope)
      :ets.delete(plan.table)
      :ets.delete(plan.learned)
    end
  end

  # Compiles the first segment for inputs and returns the function of a call, and whether
  # the graph breaks.
  defp start(plan, inputs, static_inputs) do
    with_graph(plan, fn ->
      Transmuter.with_fold_memo(fn -> call(plan, inputs, static_inputs) end)
    end)
  end

  # EXLA keeps each compiled function in a global cache that never evicts. The segments are
  # already cached here, with eviction and erasure when the owner exits, so EXLA must not keep
  # a second copy. Other compilers, such as EMLX, reject the :cache option.
  defp options(model, opts) do
    compiler = opts[:compiler] || Nx.Defn.default_options()[:compiler] || Nx.Defn.Evaluator

    opts =
      if compiler == EXLA,
        do: opts |> Keyword.put(:disk_cache, opts[:cache]) |> Keyword.put(:cache, false),
        else: opts

    {opts, %{source: model[:source], compiler: compiler}}
  end

  # The plan keeps the nodes once, as a tuple. A copy of a closure, such as in a new process
  # or in :persistent_term, copies each reference to a term again. So the closure of a call
  # holds the plan once.
  defp plan(model, opts, static_inputs) do
    {cache_size, opts} = Keyword.pop(opts, :cache_size, 8)
    {disk_cache, opts} = Keyword.pop(opts, :disk_cache)
    constant = constant(model)

    %{
      cache_size: cache_size,
      model: Map.delete(model, :nodes),
      source: model[:source],
      nodes: List.to_tuple(model.nodes),
      last_use: last_use(model),
      defined: defined(model),
      constant: constant,
      shaped: shaped(model, constant, static_inputs),
      key: make_ref(),
      opts: opts,
      backend: Nx.Defn.to_backend(opts),
      table: :ets.new(__MODULE__, [:public, read_concurrency: true]),
      learned: :ets.new(__MODULE__, [:public, :bag, read_concurrency: true]),
      scope: Shared.scope(),
      disk_cache: disk_cache(disk_cache, model),
      random?: Map.get(model, :noise, true) and Transmuter.random?(model.nodes)
    }
  end

  @doc """
  Returns `Nx.Defn.jit/2` of `fun` with `defn_options`, without an EXLA `:cache` path.

  A jit function compiles for the shapes of each call, and EXLA loads the program from a `:cache` path again on each call. So the small jit functions of the tasks keep their programs only in memory.
  """
  def jit(fun, defn_options) do
    defn_options =
      if is_binary(defn_options[:cache]),
        do: Keyword.delete(defn_options, :cache),
        else: defn_options

    Nx.Defn.jit(fun, defn_options)
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
      {{:end, entry}, _args, _state} ->
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
    {args, _static_key, _statics} = live(plan, state, 0, MapSet.new())
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

    %{env: env, static: digests(cast.(static_inputs)), breaks: MapSet.new(), uploads: %{}}
  end

  defp digests(values), do: Map.new(values, fn {name, value} -> {name, Ops.digest(value)} end)

  defp static?(plan, state, name),
    do: Map.has_key?(state.static, name) or Map.has_key?(plan.model.static, name)

  defp run(plan, state, pos) when pos >= tuple_size(plan.nodes), do: state

  defp run(plan, state, pos) do
    case segment(plan, state, pos) do
      {{:end, entry}, args, state} ->
        {args, state} = upload(plan, state, args)
        merge(state, run_compiled(entry, args))

      {{:break, entry}, args, state} ->
        {args, state} = upload(plan, state, args)
        state = merge(state, run_compiled(entry, args))
        run(plan, run_break(plan, state, entry.stop), entry.stop + 1)

      {{:eager, stop}, _args, state} ->
        state = run_eager(plan, state, pos, stop)
        if stop == :end, do: state, else: run(plan, run_break(plan, state, stop), stop + 1)
    end
  end

  # Finds or builds the compiled segment that starts at pos for these values, and gives
  # its arguments and the values of the call when it starts.
  defp segment(plan, state, pos),
    do: attempt(plan, state, pos, host_names(plan, pos), %{added: %{}, real: MapSet.new()}, :call)

  # A trace stops at a node that reads the host value of an argument. When that value
  # follows only from static values of the call, the break is induced: the names where the
  # way enters the segment become static, and the segment is traced again. When the node
  # still breaks, the break is real, and the names become arguments again. The plan keeps
  # the real breaks of each position and shapes, so a later miss does not try them again.
  defp attempt(plan, state, pos, known, tried, mode) do
    host = tried.added |> Map.values() |> Enum.reduce(known, &MapSet.union/2)
    state = enter(plan, state, host)
    {key = {^pos, shapes, _static_key}, args, statics} = key(plan, state, pos, host)
    metadata = %{source: plan.source, position: pos, shapes: shapes}

    case {lookup(plan, key), mode} do
      {{:ok, entry}, :call} ->
        :telemetry.execute([:transmuter, :cache, :hit], %{count: 1}, metadata)
        {classify(entry), args, state}

      {{:ok, entry}, :retry} ->
        {entry, args, state, learn(plan, pos, known, host), false, shapes}

      {:miss, :call} ->
        :telemetry.execute([:transmuter, :cache, :miss], %{count: 1}, metadata)
        tried = Map.merge(tried, %{real: real_breaks(plan, pos, shapes), shapes: shapes})

        :telemetry.span([:transmuter, :segment, :compile], metadata, fn ->
          {entry, args, state, added, built?, shapes} =
            build(plan, state, pos, {known, host}, tried, {key, args, statics})

          {{classify(entry), args, state},
           Map.merge(metadata, %{
             shapes: shapes,
             stop: entry.stop,
             compiled: built? and entry.compiled != nil,
             reason: entry.reason,
             host_values: Enum.sort(added)
           })}
        end)

      {:miss, :retry} ->
        build(plan, state, pos, {known, host}, tried, {key, args, statics})
    end
  end

  defp build(plan, state, pos, {known, host}, tried, {key = {pos, shapes, _}, args, statics}) do
    ref = make_ref()
    plan_key = plan.key
    statics = statics.()
    fun = fn args -> trace(plan_key, statics, args, pos, ref) end
    Process.put({ref, :context}, learn_context(state, pos, host, tried))

    try do
      compile_segment(plan, key, fun, args, ref)
    after
      Process.delete(ref)
      Process.delete({ref, :context})
    end
    |> case do
      {:induced, node, names} ->
        added = Map.update(tried.added, node, names, &MapSet.union(&1, names))
        attempt(plan, state, pos, known, %{tried | added: added}, :retry)

      {:real, node} ->
        keep(plan, {:real, pos, tried.shapes}, node)

        tried = %{
          tried
          | added: Map.delete(tried.added, node),
            real: MapSet.put(tried.real, node)
        }

        attempt(plan, state, pos, known, tried, :retry)

      entry ->
        store(plan, key, entry)
        {entry, args, state, learn(plan, pos, known, host), true, shapes}
    end
  end

  defp key(plan, state, pos, host) do
    {args, static_key, statics} = live(plan, state, pos, host)
    shapes = Enum.map(args, fn {name, value} -> {name, template(value)} end)
    {{pos, shapes, static_key}, encode(args), statics}
  end

  # The names whose host value a node of the segment at pos needs. They enter the segment
  # as static values. The plan learns them from the induced breaks of its traces, so the
  # set only grows.
  defp host_names(plan, pos), do: plan |> kept(pos) |> MapSet.new()

  defp real_breaks(plan, pos, shapes), do: plan |> kept({:real, pos, shapes}) |> MapSet.new()

  defp kept(plan, key) do
    plan.learned |> :ets.lookup(key) |> Enum.map(&elem(&1, 1))
  rescue
    ArgumentError -> []
  end

  defp keep(plan, key, values) do
    :ets.insert(plan.learned, Enum.map(List.wrap(values), &{key, &1}))
  rescue
    ArgumentError -> true
  end

  # A small value of an earlier segment whose host value a node of this segment needs
  # enters the segment as a static value: the host gets a copy, and its digest goes into
  # the key of the segment. It must follow only from static values in this call. A value
  # that follows from device data in this call stays an argument, and the node breaks.
  defp enter(plan, state, host) do
    context = %{static: state.static, breaks: state.breaks}

    {state, _memo} =
      Enum.reduce(host, {state, %{}}, fn name, {state, memo} ->
        if promote?(Map.get(state.env, name)) and not Map.has_key?(state.static, name) do
          case upstream(plan, context, name, memo) do
            {:static, memo} -> {promote(state, name), memo}
            {:device, memo} -> {state, memo}
          end
        else
          {state, memo}
        end
      end)

    state
  end

  # The device value stays as the upload of the name, so an argument does not copy the
  # host value back.
  defp promote(state, name) do
    value = Map.get(state.env, name)

    if promote?(value) and not Map.has_key?(state.static, name) do
      host = to_host(value)
      state = %{state | uploads: Map.put(state.uploads, name, value)}
      merge(state, %{values: %{name => host}, static: %{name => Ops.digest(host)}})
    else
      state
    end
  end

  # The arguments of a compiled segment on the device. A static value goes to the device
  # once in a call.
  defp upload(plan, state, args) do
    {args, state} =
      Enum.map_reduce(args, state, fn {name, value}, state ->
        if host_value?(value) do
          upload = Map.get_lazy(state.uploads, name, fn -> on_device(value, plan.backend) end)
          {{name, upload}, %{state | uploads: Map.put(state.uploads, name, upload)}}
        else
          {{name, value}, state}
        end
      end)

    {Map.new(args), state}
  end

  # An argument is a tensor, or the tuple of the tensors of a sequence.
  defp host_value?(%Nx.Tensor{data: %Nx.BinaryBackend{}}), do: true

  defp host_value?(tensors) when is_tuple(tensors),
    do: tensors |> Tuple.to_list() |> Enum.any?(&host_value?/1)

  defp host_value?(_value), do: false

  defp on_device(tensors, backend) when is_tuple(tensors),
    do: tensors |> Tuple.to_list() |> Enum.map(&on_device(&1, backend)) |> List.to_tuple()

  defp on_device(t = %Nx.Tensor{data: %Nx.BinaryBackend{}}, backend),
    do: Nx.backend_copy(t, backend)

  defp on_device(t, _backend), do: t

  defp learn_context(state, pos, host, tried) do
    %{
      pos: pos,
      static: state.static,
      env: state.env,
      breaks: state.breaks,
      host: host,
      real: tried.real,
      tried: Map.keys(tried.added)
    }
  end

  # Keeps the names that a build added for its position, and gives them.
  defp learn(plan, pos, known, host) do
    added = MapSet.difference(host, known)
    keep(plan, pos, Enum.to_list(added))
    Enum.to_list(added)
  end

  # The compiler traces the segment before it builds a program. A short segment after a
  # break runs eagerly, so its trace stops the compiler. The first segment always compiles,
  # as a model without breaks always did. The compiled function keeps the trace function,
  # so it holds the number of nodes and not the plan.
  defp compile_segment(plan, key = {pos, _shapes, _static_key}, fun, args, ref) do
    count = tuple_size(plan.nodes)

    bounded = fn args ->
      outputs = fun.(args)
      {stop, _reason, _out_names, _produced} = Process.get(ref)
      stop_index = if stop == :end, do: count, else: stop
      if pos > 0 and stop_index - pos < @min_compiled_nodes, do: throw({ref, :eager})
      outputs
    end

    try do
      compiled =
        try do
          compile_program(plan, key, bounded, args)
        catch
          {^ref, :eager} -> nil
        end

      # A compiler that loads the program from its disk cache does not trace the segment.
      # Then a trace gives where the segment stops and what it returns.
      unless Process.get(ref), do: Nx.Defn.debug_expr(bounded).(args)

      {stop, reason, out_names, produced} = Process.get(ref)

      %{
        compiled: compiled,
        stop: stop,
        reason: reason,
        out_names: out_names,
        produced: produced,
        digests: digests(produced)
      }
    catch
      {^ref, :induced, node, names} -> {:induced, node, names}
      {^ref, :real, node} -> {:real, node}
    end
  end

  # The compiled function keeps its example arguments, so it gets templates. With the
  # values, each cached segment would keep the device tensors of its first call.
  defp compile_program(plan, key, fun, args) do
    opts =
      case plan.disk_cache do
        nil -> plan.opts
        {path, model} -> Keyword.put(plan.opts, :cache, path <> "-" <> hash({model, key}))
      end

    Nx.Defn.compile(fun, [Composite.traverse(args, &Nx.to_template/1)], opts)
  end

  # EXLA keeps one program in each :cache path and checks only the shapes of its arguments.
  # So each segment gets its own path, from the graph and the key of the segment, and a
  # function with other shapes or another graph does not overwrite or load it.
  defp disk_cache(path, model) when is_binary(path),
    do: {path, hash(Map.delete(model, :params))}

  defp disk_cache(_cache, _model), do: nil

  defp hash(term) do
    hash = :crypto.hash(:sha256, :erlang.term_to_binary(term, [:deterministic]))
    Base.encode16(hash, case: :lower)
  end

  defp classify(%{compiled: nil, stop: stop}), do: {:eager, stop}
  defp classify(entry = %{stop: :end}), do: {:end, entry}
  defp classify(entry), do: {:break, entry}

  # The traced body of a segment. It records where it stopped and why, which traced values it
  # returns and which static values it folded, so the caller can resume after it.
  defp trace(key, statics, args, pos, ref) do
    %{} = plan = Process.get({__MODULE__, key})
    env = Map.merge(statics, decode(args))
    ctx = context(plan, env)
    {env, stop, reason} = run_until_break(plan, env, ctx, pos, ref)
    stop_index = if stop == :end, do: tuple_size(plan.nodes), else: stop

    live =
      for i <- pos..(stop_index - 1)//1,
          name <- elem(plan.nodes, i).output,
          name != "" and Map.get(plan.last_use, name, -1) >= stop_index,
          into: %{},
          do: {name, Map.fetch!(env, name)}

    {traced, produced} = Map.split_with(live, fn {_name, value} -> traced?(value) end)
    Process.put(ref, {stop, reason, Map.keys(traced), produced})
    traced |> encode() |> Transmuter.tag_literals(Compiler.current())
  end

  # The reason of a break is the description of the value that the break node needs.
  defp run_until_break(plan, env, ctx, pos, ref) do
    Enum.reduce_while(pos..(tuple_size(plan.nodes) - 1)//1, {env, :end, nil}, fn i, {env, _, _} ->
      try do
        {:cont, {Transmuter.run_nodes([elem(plan.nodes, i)], env, ctx), :end, nil}}
      rescue
        e in UnknownValueError ->
          unknown(plan, env, i, e, ref)
          {:halt, {env, i, e.value}}
      end
    end)
  end

  # Node i of a trace needs a value that is not known. When that value follows only from
  # static values of the call, the plan could have known it, so the break is induced: the
  # trace throws the names where the way enters the segment, and the segment is traced
  # again with them static.
  #
  # A value that follows from an input, a parameter, a random node or a device value of a
  # break node makes the break real. So does a value of an earlier segment that is too large
  # to copy to the host. When the node read a value that it computed from its inputs, the
  # trace cannot tell which input it came from. Then the inputs that follow from static
  # values become static for one more trace, and the break is real when the node still
  # breaks. A trace without a context, such as a trace of EMLX in a call, breaks.
  defp unknown(plan, env, i, e, ref) do
    context = Process.get({ref, :context})

    if context && not MapSet.member?(context.real, i) do
      {found?, inputs} = needed(elem(plan.nodes, i), env, e)
      {origins, _memo} = Enum.map_reduce(inputs, %{}, &origin(plan, context, env, &1, &2))
      device? = :device in origins

      names =
        origins
        |> Enum.reject(&(&1 == :device))
        |> Enum.reduce(MapSet.new(), &MapSet.union/2)
        |> MapSet.difference(context.host)

      new? = MapSet.size(names) > 0

      cond do
        i in context.tried and (device? or not new?) -> throw({ref, :real, i})
        new? and not (device? and found?) -> throw({ref, :induced, i, names})
        true -> :real
      end
    end
  end

  # The traced inputs of the node that the error names, or all of its traced inputs when
  # the node read a value that it computed from them.
  defp needed(node, env, e) do
    traced =
      for name <- Enum.uniq(Transmuter.value_dependencies(node)),
          traced?(Map.get(env, name)),
          do: name

    case Enum.filter(traced, &match?(%Nx.Tensor{data: %{id: id}} when id == e.expr, env[&1])) do
      [] -> {false, traced}
      read -> {true, read}
    end
  end

  # The names where the way from static values of the call to the value of name enters the
  # segment, or :device. A name that enters it is a static value of the call, or a small
  # value of an earlier segment that follows only from static values.
  defp origin(plan, context, env, name, memo) do
    index = Map.get(plan.defined, name, -1)

    cond do
      Map.has_key?(memo, name) ->
        {memo[name], memo}

      Map.has_key?(plan.shaped, name) ->
        {MapSet.new(), memo}

      index >= context.pos ->
        {found, memo} =
          walk(elem(plan.nodes, index), memo, &origin(plan, context, env, &1, &2))

        origin =
          if found == :device,
            do: :device,
            else: Enum.reduce(found, MapSet.new(), &MapSet.union/2)

        {origin, Map.put(memo, name, origin)}

      Map.has_key?(context.static, name) ->
        {MapSet.new([name]), memo}

      promote?(Map.get(context.env, name)) ->
        {found, memo} = upstream(plan, context, name, memo)
        origin = if found == :device, do: :device, else: MapSet.new([name])
        {origin, Map.put(memo, name, origin)}

      true ->
        {:device, memo}
    end
  end

  # Whether the value of a name of an earlier segment follows only from static values of
  # the call. The way stops at a break node of the call, because the break gives its node
  # host values.
  defp upstream(plan, context, name, memo) do
    index = Map.get(plan.defined, name, -1)

    cond do
      Map.has_key?(memo, {:upstream, name}) ->
        {memo[{:upstream, name}], memo}

      Map.has_key?(plan.shaped, name) or Map.has_key?(context.static, name) ->
        {:static, memo}

      index < 0 or MapSet.member?(context.breaks, index) ->
        {:device, memo}

      true ->
        {found, memo} = walk(elem(plan.nodes, index), memo, &upstream(plan, context, &1, &2))
        found = if found == :device, do: :device, else: :static
        {found, Map.put(memo, {:upstream, name}, found)}
    end
  end

  # The results of fun for the value inputs of node, or :device at the first input that
  # gives :device. The outputs of a random node are not static, so it gives :device.
  defp walk(node, memo, fun) do
    if Transmuter.random?([node]) do
      {:device, memo}
    else
      node
      |> Transmuter.value_dependencies()
      |> Enum.reject(&(&1 == ""))
      |> Enum.reduce_while({[], memo}, fn input, {found, memo} ->
        case fun.(input, memo) do
          {:device, memo} -> {:halt, {:device, memo}}
          {result, memo} -> {:cont, {[result | found], memo}}
        end
      end)
    end
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
    state = %{state | breaks: MapSet.put(state.breaks, index)}
    metadata = %{source: plan.source, position: index, node: node.name, op_type: node.op_type}

    :telemetry.span([:transmuter, :break], metadata, fn ->
      {run_break_node(plan, state, node), metadata}
    end)
  end

  defp run_break_node(plan, state, node) do
    state =
      Enum.reduce(Transmuter.dependencies(node), state, fn name, state ->
        if Map.has_key?(plan.model.static, name), do: state, else: promote(state, name)
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
    %{state | env: Map.merge(state.env, values), static: Map.merge(state.static, static)}
  end

  # A value that is not static yet and small enough to copy to the host.
  defp promote?(t = %Nx.Tensor{data: %data{}}) when data != Nx.BinaryBackend, do: small?(t)
  defp promote?(value = {:sequence, _elements}), do: small?(value)
  defp promote?(_value), do: false

  defp small?(t = %Nx.Tensor{}),
    do: if(Nx.Type.integer?(Nx.type(t)), do: Nx.size(t) <= 4096, else: Nx.size(t) <= 16)

  defp small?({:sequence, elements}), do: Enum.all?(elements, &(small?(&1) or Ops.static?(&1)))
  defp small?(_value), do: false

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
  defp live(plan, state, pos, host) do
    {names, constants} = live_names(plan, state, pos)

    {args, statics, key} =
      Enum.reduce(names, {[], [], []}, fn name, acc ->
        case state.env do
          %{^name => value} -> live_value(plan, state, host, name, value, acc)
          %{} -> acc
        end
      end)

    {args, key, fn -> Map.merge(Map.take(state.env, constants), Map.new(statics)) end}
  end

  defp live_value(plan, state, host, name, value, {args, statics, key}) do
    cond do
      argument?(plan, state, host, name, value) ->
        {[{name, value} | args], statics, key}

      digest = state.static[name] ->
        {args, [{name, value} | statics], [{name, digest} | key]}

      device_value?(value) ->
        {[{name, value} | args], statics, key}

      true ->
        {args, [{name, value} | statics], [{name, Ops.digest(value)} | key]}
    end
  end

  # A value that the plan made static, such as a value that a break node read, goes into a
  # segment as an argument, so that its digest is not in the key. A value that follows only
  # from shapes and static inputs and a value whose host value a node needs stay static.
  defp argument?(plan, state, host, name, value) do
    Map.has_key?(state.static, name) and device_value?(value) and
      not MapSet.member?(host, name) and not Map.has_key?(plan.shaped, name)
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
    known = Map.new(model.static, fn {name, _} -> {name, true} end)
    follow(model.nodes, known, &Transmuter.dependencies/1)
  end

  # The names whose values follow only from the static values of the model, the static
  # inputs and the shapes of other values, such as the targets of Reshape nodes. They change
  # only with shapes and static inputs, so a call keeps them static.
  defp shaped(model, constant, static_inputs) do
    known = Enum.reduce(Map.keys(static_inputs), constant, &Map.put(&2, &1, true))
    follow(model.nodes, known, &Transmuter.value_dependencies/1)
  end

  # Adds to known the outputs of each node that draws no random values and whose
  # dependencies are all known.
  defp follow(nodes, known, dependencies) do
    Enum.reduce(nodes, known, fn node, known ->
      if not Transmuter.random?([node]) and
           Enum.all?(dependencies.(node), &(&1 == "" or Map.has_key?(known, &1))),
         do: Enum.reduce(node.output, known, &Map.put(&2, &1, true)),
         else: known
    end)
  end

  # Each name with the index of the node that computes it, or -1 for a value that a call
  # starts with.
  defp defined(model) do
    initial = Map.keys(model.static) ++ Map.keys(model.params) ++ model.inputs

    Enum.map([Transmuter.random_input() | initial], &{&1, -1})
    |> Kernel.++(
      for {node, i} <- Enum.with_index(model.nodes),
          name <- node.output,
          name != "",
          do: {name, i}
    )
    |> Enum.uniq_by(&elem(&1, 0))
    |> Map.new()
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
