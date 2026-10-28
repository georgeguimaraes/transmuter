defmodule Transmuter.ContinuousGeneration do
  @moduledoc """
  A server of generation with continuous batching: a request joins the running batch at the next step, and a row that ends gives its place to the next request.

  A serving of `Transmuter.Text.generation/2` or `Transmuter.Vision.image_text_to_text/2` runs each batch until its longest row ends, so a request that arrives during a batch waits for all of it, and the rows that end early stay empty. This server keeps `:slots` rows, and between two steps it gives each free row to a waiting request. The text of a request does not depend on the other rows: with greedy selection, and with sampling from a seed, a request gives the same text as alone.

      children = [
        {Transmuter.ContinuousGeneration,
         name: MyGen, source: {:hf, "HuggingFaceTB/SmolLM2-135M-Instruct"}, slots: 8}
      ]

      Supervisor.start_link(children, strategy: :one_for_one)

      %{results: [%{text: text}]} =
        Transmuter.ContinuousGeneration.run(MyGen, %{messages: [%{role: "user", content: "Hi"}]})

  A server of a vision language model takes the chats of `Transmuter.Vision.image_text_to_text/2`:

      {Transmuter.ContinuousGeneration,
       name: MyChat,
       task: Transmuter.Vision.ImageTextToText,
       source: {:hf, "HuggingFaceTB/SmolVLM-256M-Instruct"},
       slots: 8}

  ## Options

    * `:name` - the name of the server.

    * `:task` - `Transmuter.Text.Generation`, the task of `Transmuter.Text.generation/2`, or `Transmuter.Vision.ImageTextToText`, the task of `Transmuter.Vision.image_text_to_text/2`. The default is `Transmuter.Text.Generation`.

    * `:source` - the model, as in the function of the task.

    * `:slots` - the number of rows that generate together. The default is `8`.

    * `:max_queue` - the number of requests that can wait for a row. The default is 4 times `:slots`. When the queue is full, `run/3` and `stream/3` raise `Transmuter.ContinuousGeneration.QueueFullError`, so the caller can rescue this case and reject the request.

    * `:capacity` - the largest number of rows of the KV cache of a row, at most the rows of the static tables of the decoder. A request whose prompt and `:max_new_tokens` need more rows raises an `ArgumentError` in the caller. The default is `4096`.

    * `:compile_steps` - when `true`, the server compiles the shape of each step that its rows can take before it starts (see "Compiles"). The default is `false`.

  The other options are those of the function of the task, without `:compile`, `:stream` and `:stream_done`. The server loads the models and compiles the steps when it starts.

  ## Scope

  The server takes a decoder whose rows can join a running batch, and it raises an `ArgumentError` at the start for the others:

    * A merged decoder, such as `Xenova/gpt2`, runs each prompt on its branch without cache, but the rows that already run need the branch with cache in the same step.
    * A decoder that grows its cache, or that takes one position offset for all rows, such as Apertus, needs one length of the cache for all rows of a step.
    * With `num_beams` above 1, the rows of a prompt move together as beams, so a request cannot join between them.

  ## The server and the serving

  These numbers come from `bench/continuous_generation.exs` on CUDA (RTX 3060), with the server and a serving of batch size 8 at the same request rates. The gap is the time between two chunks of text of a stream in the caller.

  For text, the server gives more tokens per second and earlier first tokens. With SmolLM2-135M, the limit of the server is about 3.9 requests per second. At 1.94, 3.10 and 4.65 requests per second, the server gives 7%, 7% and 13% more tokens per second than the serving. Below the limit, the first token comes about 20 times earlier: 61 ms against 1284 ms at 1.94 requests per second, and 64 ms against 1715 ms at 3.10. The gap at p99 is about 11 ms longer: 44 ms against 33 ms.

  For a vision language model, the serving is as good as the server or better. With SmolVLM-256M, the limit of the server is about 0.84 requests per second. At 0.42 and 0.68 requests per second, both give the same tokens per second, and at 1.01 the serving gives 17.7 against 13.5. At 0.68, the median first token is 2.1 s against 3.3 s, and the gap at p99 is 36 ms against 112 ms. The worker encodes the images of a request on the same GPU while the rows decode, so each decode step and each encoder call takes longer. The serving runs the encoder and the decoder one after the other.

  ## The KV cache

  The server runs decoders in `:in_place` and in `:append`, see `Transmuter.Generation`. In `:in_place`, each row writes its cache after its own rows, and a new request starts again at the first row. The buffers have one capacity for all rows, and they grow by 64 rows when the longest row does not fit them. They do not shrink while rows run, so a group runs at the capacity of its longest row. In a batch of `Transmuter.Text.generation/2`, all rows start together, so the buffers grow only near the end of the batch. In the server, the rows start at different steps, so after the first row passes a capacity, most steps run at the larger one. Each step reads the whole buffers, so these steps are a little slower: for Qwen3-0.6B with 8 rows on CUDA, 26.2 ms at 320 rows against 24.9 ms at 256. A shrink when the longest row ends would copy the buffers again each time the next row passes the capacity.

  In `:append`, all rows of a step write at one offset, so the offset passes the rows of all steps, also the rows of the requests that ended. When a step does not fit, the buffers compact: the valid rows of each running row move to the front, and the offset goes back to the longest running row. After a step, the buffers also compact when the longest running row and 16 rows fit in half of them, so the rows after a long request do not keep its capacity.

  A request joins with the steps of its prompt parts (see "Logits" in `Transmuter.Generation`), and each running row gets its next token in each of these steps. A step with running rows takes at most 64 tokens of a prompt, so a join delays the tokens of the running rows by at most one such step: for Qwen3-0.6B on CUDA, the time between two tokens is at most about 53 ms during a join, against 26 ms for a step without a join.

  On EMLX, MLX keeps the freed buffers of each size in a cache, which has no limit by default. Set a limit, for example `EMLX.set_cache_limit(1_000_000_000)`, when the application starts.

  ## Images

  A request with images needs the features of its images before it joins, and for Qwen2-VL, Qwen2.5-VL and Qwen3-VL the 3-D positions of its prompt. A worker process of the server prepares one request at a time, in the order of the requests, and then puts the request in the queue. It runs the processor on the images and the tokenizer on the chat, and then the vision encoder on the images. So the steps of the running rows do not wait for the encoder, the caller does no work on the device, and the device memory of this work is that of one request. The processor and the encoder compile one time, in the worker. A request waits in the worker for the requests before it, also a request without images. The queue limit counts the requests in the worker. An error of the processor or of the encoder goes to the caller of the request (see "Failures").

  The encoder and the steps run on the same device. On CUDA, EXLA runs the calls of the two processes side by side: for SmolVLM-256M on an RTX 3060, a decode step takes about 7 ms alone and at most about 16 ms while the worker encodes. On EMLX, both processes use the one GPU stream of EMLX, so a step waits for the encoder call that runs. So the worker runs the encoder in calls of one tile or one image when the encoder takes them on its batch axis, and each call ends before the next starts. While a request with images is prepared, the time between two tokens of a running row can reach one encoder call. For SmolVLM-256M on the GPU of a Mac, a decode step takes about 7 ms alone and about 70 ms while the worker encodes in calls of one tile, against 863 ms with one call for the 13 tiles of an image, and the encoder takes about 1 s for each image. An encoder that takes the patches of an image in one call, such as that of Qwen2-VL, keeps the device for one image.

  The group keeps the features of the image tokens that its rows have not run, and the features of a request join them. Then the prompt of the request runs in parts of at most 64 tokens, next to the new tokens of the running rows (see "The KV cache").

  ## Compiles

  A step compiles the decoder for its shape the first time that the server meets it: the width of the step and the capacity of the buffers. A step has one of three widths, 1, 64 and 512 tokens (see "Batches" in `Transmuter.Generation`), so a group has at most three shapes for each capacity, 47 for a capacity of 4096 rows. Without `compile_steps: true`, a step that meets a new shape compiles it while rows decode, for example about 3 s on CUDA for SmolLM2-135M.

  With `compile_steps: true`, the server runs one step of each shape before it starts, with the writes of the rows of 1 to `:slots` requests that join together and a growth of the buffers from each capacity to the next, and the worker prepares one input of the task, such as a chat with a small image, so that the processor and the vision encoder compile too. With 8 rows and a capacity of 4096 rows, the start takes 12 s for SmolLM2-135M and 14 s for SmolVLM-256M on the GPU of a Mac, and 146 s and 153 s on CUDA (RTX 3060), against under 1 s and about 6 s without it. The image features of a group have a power of two of rows (see "Embeddings and image features" in `Transmuter.Generation`), so their merge compiles for few shapes. The processor and an encoder that takes images of each size in one call, such as that of Qwen2-VL, still compile for each new size of image.

  ## Telemetry

  Each step emits `[:transmuter, :continuous_generation, :step, :start]` and `:stop`, with the running rows, the requests that join and the waiting requests. The duration of a step is the time between two tokens of a running row. The worker emits `[:transmuter, :continuous_generation, :prepare, :start]` and `:stop` for each request, with the time that the request waited for the worker. See `Transmuter.Telemetry`.

  ## Failures

  The callers of `run/3` and `stream/3` monitor the server. When the server stops, for example when a step raises an out-of-memory error of the device, each waiting caller exits with the reason of the server. When a caller exits, its row ends at the next step and goes to the next request.

  When the worker raises on a request, for example with an out-of-memory error of the device in the vision encoder, the caller of that request gets the exception, and the server and the other requests continue. When the worker stops for another reason, the server stops with the same reason.
  """

  use GenServer

  alias Transmuter.{Generation, Serving}

  defmodule QueueFullError do
    @moduledoc """
    The queue of a `Transmuter.ContinuousGeneration` server is full.
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
    module = Keyword.get(opts, :task, Transmuter.Text.Generation)
    server_opts = [:name, :task, :source, slots: 8, max_queue: nil, capacity: 4096]
    server_opts = server_opts ++ [compile_steps: false]

    opts =
      opts |> Keyword.validate!(server_opts ++ module.options()) |> Serving.generation_options!()

    {ctx, tokenizer, task} = module.task(opts[:source], opts)

    unless Generation.joins?(ctx) do
      raise ArgumentError,
            "requests cannot join a running batch of this model: it is a merged decoder, it needs one length of the cache for all rows, or it has num_beams above 1"
    end

    ctx = %{ctx | max_capacity: min(ctx.max_capacity || opts[:capacity], opts[:capacity])}

    state = %{
      ctx: ctx,
      tokenizer: tokenizer,
      task: task,
      slots: opts[:slots],
      max_queue: opts[:max_queue] || 4 * opts[:slots],
      defn_options: opts[:defn_options],
      compile_steps: opts[:compile_steps]
    }

    GenServer.start_link(__MODULE__, state,
      name: opts[:name],
      spawn_opt: Serving.generation_flags()
    )
  end

  @doc """
  Generates the text of `input`, an input of the task or a list of them, as `Nx.Serving.batched_run/2` with a serving of the task, and returns the same result.

  Raises `QueueFullError` when the queue cannot take the inputs. After `timeout` milliseconds, the rows of the inputs end and the caller exits.
  """
  def run(server, input, timeout \\ :infinity) do
    deadline = if timeout == :infinity, do: :infinity, else: now() + timeout
    call = {__MODULE__, :run, [server, input, timeout]}
    info = info!(server, call)
    {inputs, multi?} = Serving.validate!(input, info.task.validate)
    refs = for input <- inputs, do: {make_ref(), input}
    monitor = enqueue!(info, refs, false, server, deadline, call)

    results =
      for {ref, _input} <- refs do
        run = await(info, monitor, {ref, :run}, deadline, refs, call)
        tokens = await(info, monitor, {ref, :done}, deadline, refs, call)
        info.task.postprocess.([Serving.integers(tokens)], [run])
      end

    Process.demonitor(monitor, [:flush])
    if multi?, do: results, else: hd(results)
  end

  @doc """
  Returns a stream of the text of `input`, as a serving of the task with `stream: true`. Consume the stream in the process that called this function.

  The function returns when the worker has prepared the request (see "Images"), and it raises the exception of the worker on the request. `stream_done: true` adds `{:done, %{token_summary: summary}}` at the end of the stream. Raises `QueueFullError` when the queue is full.
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

    ref = make_ref()
    monitor = enqueue!(info, [{ref, input}], true, server, :infinity, reason)
    run = await(info, monitor, {ref, :run}, :infinity, [{ref, input}], reason)

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

              {^ref, {:error, exception}} ->
                Process.demonitor(monitor, [:flush])
                raise exception

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

    requests = for {ref, input} <- refs, do: %{ref: ref, input: input, stream?: stream?}

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

  # The value of the message `tag` of the request `ref`: :run when the worker has run
  # :preprocess on its input, and :done with its tokens when its row ends.
  defp await(info, monitor, {ref, tag}, deadline, refs, call) do
    receive do
      {^ref, {^tag, value}} ->
        value

      {^ref, {:error, exception}} ->
        GenServer.cast(info.pid, {:cancel, Enum.map(refs, &elem(&1, 0))})
        Process.demonitor(monitor, [:flush])
        raise exception

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
    Process.flag(:trap_exit, true)
    server = self()
    %{preprocess: preprocess, prepare: prepare, example: example} = state.task
    compile_steps? = state.compile_steps
    defn_options = state.defn_options

    # The worker loads the models of :prepare, such as the vision encoder, while the server
    # compiles the steps. The server starts when both are ready, so an error of either fails
    # the start.
    worker =
      spawn_link(fn ->
        task = {preprocess, prepare.(defn_options)}
        backend = Nx.Defn.to_backend(defn_options)

        # The features of the example show the shape of the features of a request.
        features =
          if compile_steps? do
            {:ok, %{features: features}, _run} = prepare_request(task, backend, %{input: example})
            features
          end

        send(server, {self(), :ready, features})
        :erlang.garbage_collect()
        prepare_requests(task, backend, server)
      end)

    ctx = Generation.compile(state.ctx, state.slots, state.defn_options)
    if state.compile_steps, do: Generation.compile_steps(ctx, state.slots)

    receive do
      {^worker, :ready, features} ->
        if features, do: Generation.compile_features(ctx, state.slots, features)
        {:ok, start(state, ctx, worker)}

      {:EXIT, ^worker, reason} ->
        {:stop, reason}
    end
  end

  defp start(state, ctx, worker) do
    info = %{
      pid: self(),
      task: Map.take(state.task, [:validate, :postprocess]),
      tokenizer: state.tokenizer,
      logprobs?: ctx.token_logprobs?,
      max_queue: state.max_queue
    }

    :persistent_term.put({__MODULE__, self()}, info)

    Map.merge(state, %{
      ctx: ctx,
      worker: worker,
      group: nil,
      rows: %{},
      preparing: %{},
      queue: :queue.new(),
      callers: %{},
      stepping?: false
    })
  end

  # The requests go to the worker, which sends each one back with its inputs.
  @impl true
  def handle_call({:enqueue, requests}, {pid, _tag}, state) do
    if :queue.len(state.queue) + map_size(state.preparing) + length(requests) > state.max_queue do
      {:reply, {:error, :queue_full}, state}
    else
      now = System.monotonic_time()
      requests = Enum.map(requests, &Map.merge(&1, %{pid: pid, enqueued: now}))
      Enum.each(requests, &send(state.worker, {:prepare, &1}))
      preparing = Map.merge(state.preparing, Map.new(requests, &{&1.ref, &1}))
      callers = Map.put_new_lazy(state.callers, pid, fn -> Process.monitor(pid) end)
      {:reply, :ok, %{state | preparing: preparing, callers: callers}}
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

  # A request that the caller cancelled in the worker has no entry.
  def handle_info({:prepared, ref, reply}, state) do
    case {Map.pop(state.preparing, ref), reply} do
      {{nil, _preparing}, _reply} ->
        {:noreply, state}

      {{request, preparing}, {:ok, fields, run}} ->
        send(request.pid, {ref, {:run, run}})
        request = request |> Map.delete(:input) |> Map.merge(fields)
        {:noreply, kick(%{state | preparing: preparing, queue: :queue.in(request, state.queue)})}

      {{request, preparing}, {:error, exception}} ->
        send(request.pid, {ref, {:error, exception}})
        {:noreply, forget_callers(%{state | preparing: preparing})}
    end
  end

  def handle_info({:DOWN, _monitor, :process, pid, _reason}, state) do
    {:noreply, drop(%{state | callers: Map.delete(state.callers, pid)}, &(&1.pid == pid))}
  end

  def handle_info({:EXIT, worker, reason}, state = %{worker: worker}),
    do: {:stop, reason, state}

  @impl true
  # The worker holds the compiled encoder. It does not trap exits, so a stop with :normal
  # does not end it.
  def terminate(_reason, state) do
    Process.exit(state.worker, :kill)
    :persistent_term.erase({__MODULE__, self()})
  end

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
    requests =
      Map.values(state.rows) ++ Map.values(state.preparing) ++ :queue.to_list(state.queue)

    pids = MapSet.new(requests, & &1.pid)

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
        prompts: Enum.count(state.group.left, &(&1 != [])) + length(admitted),
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
        do: state.group,
        else: Generation.admit(state.ctx, state.group, requests)

    group = Generation.next(state.ctx, group)
    group = Generation.shrink(state.ctx, group)
    # The group keeps the features, so the rows do not keep them for their callers.
    admitted = Map.new(admitted, fn {row, request} -> {row, Map.drop(request, [:features])} end)
    emit(%{state | rows: Map.merge(state.rows, admitted)}, group)
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
            exception = ArgumentError.exception(too_long(state, waiting))
            send(waiting.pid, {waiting.ref, {:error, exception}})
            admissions(%{state | queue: queue}, [row | free], admitted)

          true ->
            {Enum.reverse(admitted), state}
        end
    end
  end

  defp request(state, row, request) do
    request
    |> Map.take([:prompt, :seed, :features, :position])
    |> Map.merge(%{row: row, budget: state.ctx.max_new_tokens})
  end

  defp too_long(state, request) do
    "a prompt of #{length(request.prompt)} tokens and max_new_tokens of #{state.ctx.max_new_tokens} need more than the #{state.ctx.max_capacity} rows of the cache of the server"
  end

  # Sends the new token of each row to its stream, and the tokens of each row that ends to
  # its caller. A row with a part of its prompt left has no token yet. A row that ends is free
  # for the next step.
  defp emit(state, group) do
    rows =
      Enum.reduce(state.rows, state.rows, fn {row, request}, rows ->
        token = Enum.at(group.new, row)
        done? = Enum.at(group.done, row)

        cond do
          token == nil and not done? ->
            :ok

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
    preparing = Map.reject(state.preparing, fn {_ref, request} -> drop?.(request) end)
    {dropped, rows} = Enum.split_with(state.rows, fn {_row, request} -> drop?.(request) end)

    group =
      Enum.reduce(dropped, state.group, fn {row, _request}, group ->
        Generation.cancel(group, row)
      end)

    %{state | queue: queue, preparing: preparing, rows: Map.new(rows), group: group}
  end

  # The worker runs :preprocess of the task on the input of each request alone, and :prepare
  # on its tensors and its prompt. It sends the server the fields of the request for
  # Generation and the run without its tensors, or the exception.
  defp prepare_requests(task, backend, server) do
    receive do
      {:prepare, request} ->
        metadata = %{wait: System.monotonic_time() - request.enqueued}

        reply =
          :telemetry.span([:transmuter, :continuous_generation, :prepare], metadata, fn ->
            {prepare_request(task, backend, request), metadata}
          end)

        send(server, {:prepared, request.ref, reply})
        # The device tensors of the request, such as its pixel values on the device, die here,
        # and only a garbage collection frees them on EMLX. The worker makes little other
        # garbage, so without it they stay for many requests.
        :erlang.garbage_collect()
        prepare_requests(task, backend, server)
    end
  end

  defp prepare_request({preprocess, prepare}, backend, request) do
    [run] = preprocess.(request.input)

    tensors =
      Map.new(run.tensors, fn {name, tensor} ->
        {name, tensor |> Nx.new_axis(0) |> Nx.backend_transfer(backend)}
      end)

    inputs = prepare.(tensors, [run.prompt])
    [position] = Map.get(inputs, :position_ids, [[]])
    [features] = Map.get(inputs, :image_features, [nil])
    fields = %{prompt: run.prompt, seed: run.seed, features: features}
    {:ok, Map.put(fields, :position, position), Map.delete(run, :tensors)}
  rescue
    exception -> {:error, exception}
  end
end
