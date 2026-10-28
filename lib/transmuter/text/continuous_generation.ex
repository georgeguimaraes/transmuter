defmodule Transmuter.Text.ContinuousGeneration do
  @moduledoc """
  A server of text generation with continuous batching: a request joins the running batch at the next step, and a row that ends gives its place to the next request.

  `Transmuter.Text.generation/2` runs each batch until its longest row ends, so a request that arrives during a batch waits for all of it, and the rows that end early stay empty. This server keeps `:slots` rows, and between two steps it gives each free row to a waiting request. The text of a request does not depend on the other rows: with greedy selection, and with sampling from a seed, a request gives the same text as alone.

      children = [
        {Transmuter.Text.ContinuousGeneration,
         name: MyGen, source: {:hf, "HuggingFaceTB/SmolLM2-135M-Instruct"}, slots: 8}
      ]

      Supervisor.start_link(children, strategy: :one_for_one)

      %{results: [%{text: text}]} =
        Transmuter.Text.ContinuousGeneration.run(MyGen, %{messages: [%{role: "user", content: "Hi"}]})

  ## Options

    * `:name` - the name of the server.

    * `:source` - the model, as in `Transmuter.Text.generation/2`.

    * `:slots` - the number of rows that generate together. The default is `8`.

    * `:max_queue` - the number of requests that can wait for a row. The default is 4 times `:slots`. When the queue is full, `run/3` and `stream/3` raise `Transmuter.Text.ContinuousGeneration.QueueFullError`, so the caller can rescue this case and reject the request.

    * `:capacity` - the largest number of rows of the KV cache of a row, at most the rows of the static tables of the decoder. A request whose prompt and `:max_new_tokens` need more rows raises an `ArgumentError` in the caller. The default is `4096`.

  The other options are those of `Transmuter.Text.generation/2`, without `:compile`, `:stream` and `:stream_done`. The server loads the model and compiles the steps when it starts.

  ## Scope

  The server takes a decoder whose rows can join a running batch, and it raises an `ArgumentError` at the start for the others:

    * A merged decoder, such as `Xenova/gpt2`, runs each prompt on its branch without cache, but the rows that already run need the branch with cache in the same step.
    * A decoder that grows its cache, or that takes one position offset for all rows, such as Apertus, needs one length of the cache for all rows of a step.
    * With `num_beams` above 1, the rows of a prompt move together as beams, so a request cannot join between them.
    * A decoder in `:append` writes the rows of all steps at one offset, so a group that never ends would need a cache that grows without a limit. The server needs a compaction of these rows first.

  ## The KV cache

  The server runs decoders in `:in_place`, see `Transmuter.Generation`. Each row writes its cache after its own rows, and a new request starts again at the first row. The buffers have one capacity for all rows, and they grow by 64 rows when the longest row does not fit them. They do not shrink while rows run, so a group runs at the capacity of its longest row. In a batch of `Transmuter.Text.generation/2`, all rows start together, so the buffers grow only near the end of the batch. In the server, the rows start at different steps, so after the first row passes a capacity, most steps run at the larger one. Each step reads the whole buffers, so these steps are a little slower: for Qwen3-0.6B with 8 rows on CUDA, 26.2 ms at 320 rows against 24.9 ms at 256. A shrink when the longest row ends would copy the buffers again each time the next row passes the capacity.

  A request joins with the steps of its prompt parts (see `Transmuter.Generation`), and the running rows get their next token only in the last of these steps. So each join delays the tokens of the running rows: about 125 ms for a chat prompt of Qwen3-0.6B on CUDA, against 26 ms for a step without a join.

  On EMLX, MLX keeps the freed buffers of each size in a cache, which has no limit by default. Set a limit, for example `EMLX.set_cache_limit(1_000_000_000)`, when the application starts.

  ## Telemetry

  Each step emits `[:transmuter, :continuous_generation, :step, :start]` and `:stop`, with the running rows, the requests that join and the waiting requests (see `Transmuter.Telemetry`). The duration of a step is the time between two tokens of a running row.

  ## Failures

  The callers of `run/3` and `stream/3` monitor the server. When the server stops, for example when a step raises an out-of-memory error of the device, each waiting caller exits with the reason of the server. When a caller exits, its row ends at the next step and goes to the next request.
  """

  use GenServer

  alias Transmuter.{Generation, Serving}
  alias Transmuter.Text.Generation, as: TextGeneration

  defmodule QueueFullError do
    @moduledoc """
    The queue of a `Transmuter.Text.ContinuousGeneration` server is full.
    """
    defexception [:message]
  end

  @doc false
  def child_spec(opts) do
    %{id: Keyword.get(opts, :name, __MODULE__), start: {__MODULE__, :start_link, [opts]}}
  end

  @doc """
  Loads the model, checks that requests can join a running batch, and starts the server. See the options in the module documentation.
  """
  def start_link(opts) do
    opts =
      opts
      |> Keyword.validate!(
        [:name, :source, slots: 8, max_queue: nil, capacity: 4096] ++ TextGeneration.options()
      )
      |> Serving.generation_options!()

    {ctx, tokenizer, task} = TextGeneration.task(opts[:source], opts)

    cond do
      not Generation.joins?(ctx) ->
        raise ArgumentError,
              "requests cannot join a running batch of this model: it is a merged decoder, it needs one length of the cache for all rows, or it has num_beams above 1"

      ctx.mode == :append ->
        raise ArgumentError,
              "a decoder in :append needs a compaction of its cache for continuous batching, which the server does not have yet"

      true ->
        :ok
    end

    ctx = %{ctx | max_capacity: min(ctx.max_capacity || opts[:capacity], opts[:capacity])}

    state = %{
      ctx: ctx,
      tokenizer: tokenizer,
      task: task,
      slots: opts[:slots],
      max_queue: opts[:max_queue] || 4 * opts[:slots],
      defn_options: opts[:defn_options]
    }

    GenServer.start_link(__MODULE__, state,
      name: opts[:name],
      spawn_opt: Serving.generation_flags()
    )
  end

  @doc """
  Generates the text of `input`, a string, a map or a list of them, as `Nx.Serving.batched_run/2` with a serving of `Transmuter.Text.generation/2`, and returns the same result.

  Raises `QueueFullError` when the queue cannot take the inputs. After `timeout` milliseconds, the rows of the inputs end and the caller exits.
  """
  def run(server, input, timeout \\ :infinity) do
    deadline = if timeout == :infinity, do: :infinity, else: now() + timeout
    call = {__MODULE__, :run, [server, input, timeout]}
    info = info!(server, call)
    {inputs, multi?} = Serving.validate!(input, info.task.validate)
    runs = Enum.map(inputs, info.task.preprocess)
    refs = for [run] <- runs, do: {make_ref(), run}
    monitor = enqueue!(info, refs, false, server, deadline, call)

    results =
      for {ref, run} <- refs do
        tokens = await(info, monitor, ref, deadline, refs, call)

        info.task.postprocess.([Serving.integers(tokens)], [run])
      end

    Process.demonitor(monitor, [:flush])
    if multi?, do: results, else: hd(results)
  end

  @doc """
  Returns a stream of the text of `input`, as a serving of `Transmuter.Text.generation/2` with `stream: true`. Consume the stream in the process that called this function.

  `stream_done: true` adds `{:done, %{token_summary: summary}}` at the end of the stream. Raises `QueueFullError` when the queue is full.
  """
  def stream(server, input, opts \\ []) do
    opts = Keyword.validate!(opts, stream_done: false)
    reason = {__MODULE__, :stream, [server, input, opts]}
    info = info!(server, reason)

    input =
      case Serving.validate!(input, info.task.validate) do
        {[input], false} -> input
        {_inputs, true} -> raise ArgumentError, "stream/3 takes one input, got a list"
      end

    [run] = info.task.preprocess.(input)
    ref = make_ref()
    monitor = enqueue!(info, [{ref, run}], true, server, :infinity, reason)

    events =
      Stream.resource(
        fn -> false end,
        fn
          true ->
            {:halt, true}

          false ->
            receive do
              {^ref, {:token, token, logprob, done?}} ->
                event = %{
                  token: Nx.tensor([token]),
                  finished?: Nx.tensor([if(done?, do: 1, else: 0)]),
                  logprob: Nx.tensor([logprob || :nan])
                }

                {[{:token, event}], done?}

              {^ref, {:error, message}} ->
                Process.demonitor(monitor, [:flush])
                raise ArgumentError, message

              {:DOWN, ^monitor, :process, _pid, down} ->
                exit({down, reason})
            end
        end,
        fn done? ->
          unless done?, do: GenServer.cast(info.pid, {:cancel, [ref]})
          Process.demonitor(monitor, [:flush])
        end
      )

    Serving.stream_text(events, info.tokenizer, run.summary, {opts[:stream_done], info.logprobs?})
  end

  # The task, the tokenizer and the flags of the server, which it puts in :persistent_term when
  # it starts, so a caller does not wait for a step of the server to read them.
  defp info!(server, call) do
    with pid when is_pid(pid) <- GenServer.whereis(server),
         %{} = info <- :persistent_term.get({__MODULE__, pid}, nil) do
      info
    else
      _missing -> exit({:noproc, call})
    end
  end

  # The server reads its mailbox between two steps, so the reply waits for the step that runs,
  # up to the deadline of the caller.
  defp enqueue!(info, refs, stream?, server, deadline, call) do
    monitor = Process.monitor(info.pid)

    requests =
      for {ref, run} <- refs,
          do: %{ref: ref, prompt: run.prompt, seed: run.seed, stream?: stream?}

    reply =
      try do
        GenServer.call(info.pid, {:enqueue, requests}, timeout(deadline))
      catch
        :exit, {reason, _call} ->
          GenServer.cast(info.pid, {:cancel, Enum.map(refs, &elem(&1, 0))})
          Process.demonitor(monitor, [:flush])
          exit({reason, call})
      end

    case reply do
      :ok ->
        monitor

      {:error, :queue_full} ->
        Process.demonitor(monitor, [:flush])

        raise QueueFullError,
              "the queue of #{inspect(server)} is full: it holds #{info.max_queue} requests"
    end
  end

  defp await(info, monitor, ref, deadline, refs, call) do
    receive do
      {^ref, {:done, tokens}} ->
        tokens

      {^ref, {:error, message}} ->
        GenServer.cast(info.pid, {:cancel, Enum.map(refs, &elem(&1, 0))})
        Process.demonitor(monitor, [:flush])
        raise ArgumentError, message

      {:DOWN, ^monitor, :process, _pid, reason} ->
        exit({reason, call})
    after
      timeout(deadline) ->
        GenServer.cast(info.pid, {:cancel, Enum.map(refs, &elem(&1, 0))})
        Process.demonitor(monitor, [:flush])
        exit({:timeout, call})
    end
  end

  defp now, do: System.monotonic_time(:millisecond)

  defp timeout(:infinity), do: :infinity
  defp timeout(deadline), do: max(deadline - now(), 0)

  @impl true
  def init(state) do
    ctx = Generation.compile(state.ctx, state.slots, state.defn_options)
    Process.flag(:trap_exit, true)

    info = %{
      pid: self(),
      task: state.task,
      tokenizer: state.tokenizer,
      logprobs?: ctx.token_logprobs?,
      max_queue: state.max_queue
    }

    :persistent_term.put({__MODULE__, self()}, info)

    {:ok,
     Map.merge(state, %{
       ctx: ctx,
       group: nil,
       rows: %{},
       queue: :queue.new(),
       callers: %{},
       stepping?: false
     })}
  end

  @impl true
  def handle_call({:enqueue, requests}, {pid, _tag}, state) do
    if :queue.len(state.queue) + length(requests) > state.max_queue do
      {:reply, {:error, :queue_full}, state}
    else
      requests = Enum.map(requests, &Map.put(&1, :pid, pid))
      queue = Enum.reduce(requests, state.queue, &:queue.in/2)
      callers = Map.put_new_lazy(state.callers, pid, fn -> Process.monitor(pid) end)
      {:reply, :ok, kick(%{state | queue: queue, callers: callers})}
    end
  end

  @impl true
  def handle_cast({:cancel, refs}, state), do: {:noreply, drop(state, &(&1.ref in refs))}

  @impl true
  def handle_info(:step, state) do
    state = %{state | stepping?: false}
    state = if active?(state) or :queue.is_empty(state.queue), do: state, else: open(state)
    state = if state.group, do: forget_callers(step(state)), else: state

    if active?(state) or not :queue.is_empty(state.queue),
      do: {:noreply, kick(state)},
      else: {:noreply, close(state)}
  end

  def handle_info({:DOWN, _monitor, :process, pid, _reason}, state) do
    {:noreply, drop(%{state | callers: Map.delete(state.callers, pid)}, &(&1.pid == pid))}
  end

  @impl true
  def terminate(_reason, _state), do: :persistent_term.erase({__MODULE__, self()})

  defp kick(state = %{stepping?: true}), do: state

  defp kick(state) do
    send(self(), :step)
    %{state | stepping?: true}
  end

  # A group without running rows starts again: the next requests write the cache from its
  # first row, and the buffers have the capacity of the first step again.
  defp open(state), do: %{close(state) | group: Generation.open(state.ctx, state.slots)}

  defp close(state = %{group: nil}), do: state

  defp close(state) do
    Generation.finish(state.group)
    %{state | group: nil}
  end

  # Stops the monitors of the callers without rows or waiting requests.
  defp forget_callers(state) do
    pids = MapSet.new(Map.values(state.rows) ++ :queue.to_list(state.queue), & &1.pid)

    {kept, gone} = Map.split_with(state.callers, fn {pid, _monitor} -> pid in pids end)
    Enum.each(gone, fn {_pid, monitor} -> Process.demonitor(monitor, [:flush]) end)
    %{state | callers: kept}
  end

  defp active?(%{group: nil}), do: false
  defp active?(%{group: group}), do: not Enum.all?(group.done)

  # Gives the free rows to the waiting requests in their order, while the cache fits them,
  # and runs the step. A request that does not fit a group without running rows never fits,
  # so its caller gets an error.
  defp step(state) do
    {admitted, state} = admissions(state)

    if admitted == [] and not active?(state) do
      state
    else
      metadata = %{
        running: map_size(state.rows),
        admitted: length(admitted),
        waiting: :queue.len(state.queue),
        slots: state.slots,
        capacity: state.group.state.capacity
      }

      :telemetry.span([:transmuter, :continuous_generation, :step], metadata, fn ->
        {step(state, admitted), metadata}
      end)
    end
  end

  defp step(state, admitted) do
    requests = for {row, request} <- admitted, do: request(state, row, request)

    group =
      if requests == [],
        do: Generation.next(state.ctx, state.group),
        else: Generation.admit(state.ctx, state.group, requests)

    emit(%{state | rows: Map.merge(state.rows, Map.new(admitted))}, group)
  end

  defp admissions(state) do
    free = for {true, row} <- Enum.with_index(state.group.done), do: row
    admissions(state, free, [])
  end

  defp admissions(state, [], admitted), do: {Enum.reverse(admitted), state}

  defp admissions(state, [row | free], admitted) do
    case :queue.out(state.queue) do
      {:empty, _queue} ->
        {Enum.reverse(admitted), state}

      {{:value, waiting}, queue} ->
        candidates = [{row, waiting} | admitted]
        requests = for {row, waiting} <- candidates, do: request(state, row, waiting)

        cond do
          Generation.rows_needed(state.ctx, state.group, requests) <= state.ctx.max_capacity ->
            admissions(%{state | queue: queue}, free, candidates)

          admitted == [] and not active?(state) ->
            send(waiting.pid, {waiting.ref, {:error, too_long(state, waiting)}})
            admissions(%{state | queue: queue}, [row | free], admitted)

          true ->
            {Enum.reverse(admitted), state}
        end
    end
  end

  defp request(state, row, request) do
    %{row: row, prompt: request.prompt, seed: request.seed, budget: state.ctx.max_new_tokens}
  end

  defp too_long(state, request) do
    "a prompt of #{length(request.prompt)} tokens and max_new_tokens of #{state.ctx.max_new_tokens} need more than the #{state.ctx.max_capacity} rows of the cache of the server"
  end

  # Sends the new token of each row to its stream, and the tokens of each row that ends to
  # its caller. A row that ends is free for the next step.
  defp emit(state, group) do
    rows =
      Enum.reduce(state.rows, state.rows, fn {row, request}, rows ->
        token = Enum.at(group.new, row)
        done? = Enum.at(group.done, row)

        cond do
          request.stream? ->
            logprob = if token, do: List.first(hd(Enum.at(group.logprobs, row)))
            send(request.pid, {request.ref, {:token, token || -1, logprob, done?}})

          done? ->
            send(request.pid, {request.ref, {:done, Generation.row_tokens(group, row)}})

          true ->
            :ok
        end

        if done?, do: Map.delete(rows, row), else: rows
      end)

    %{state | group: group, rows: rows}
  end

  # Ends the rows and removes the waiting requests that `drop?` selects.
  defp drop(state, drop?) do
    queue = :queue.filter(&(not drop?.(&1)), state.queue)
    {dropped, rows} = Enum.split_with(state.rows, fn {_row, request} -> drop?.(request) end)

    group =
      Enum.reduce(dropped, state.group, fn {row, _request}, group ->
        Generation.cancel(group, row)
      end)

    %{state | queue: queue, rows: Map.new(rows), group: group}
  end
end
