defmodule Transmuter.ContinuousBatching do
  @moduledoc """
  A server of generation with continuous batching, also called iteration-level scheduling or in-flight batching: a request joins the running batch at the next step, and a row that ends gives its place to the next request.

  A serving of `Transmuter.Text.generation/2` or `Transmuter.Vision.image_text_to_text/2` runs each batch until its longest row ends, so a request that arrives during a batch waits for all of it, and the rows that end early stay empty. This server keeps `:slots` rows, and between two steps it gives each free row to a waiting request. The text of a request does not depend on the other rows: with greedy selection, and with sampling from a seed, a request gives the same text as alone.

      children = [
        {Transmuter.ContinuousBatching,
         name: MyGen, source: {:hf, "HuggingFaceTB/SmolLM2-135M-Instruct"}, slots: 8}
      ]

      Supervisor.start_link(children, strategy: :one_for_one)

      %{results: [%{text: text}]} =
        Transmuter.ContinuousBatching.run(MyGen, %{messages: [%{role: "user", content: "Hi"}]})

  A server of a vision language model takes the chats of `Transmuter.Vision.image_text_to_text/2`:

      {Transmuter.ContinuousBatching,
       name: MyChat,
       task: Transmuter.Vision.ImageTextToText,
       source: {:hf, "HuggingFaceTB/SmolVLM-256M-Instruct"},
       slots: 8}

  ## Options

    * `:name` - the name of the server. The server also joins the group of its name in the `:pg` scope `Transmuter.PG`, so callers on other nodes find it (see "Distribution").

    * `:task` - the module of the task: `Transmuter.Text.Generation`, the task of `Transmuter.Text.generation/2`, `Transmuter.Text.TextToText`, the task of `Transmuter.Text.text_to_text/2`, `Transmuter.Vision.ImageTextToText`, the task of `Transmuter.Vision.image_text_to_text/2`, `Transmuter.Vision.ImageToText`, the task of `Transmuter.Vision.image_to_text/2`, or `Transmuter.Audio.SpeechToText`, the task of `Transmuter.Audio.speech_to_text/2`. The default is `Transmuter.Text.Generation`. Each input of `Transmuter.Vision.ImageToText` gives its task of Florence-2, such as `%{image: image, task: "<OD>"}`.

    * `:source` - the model, as in the function of the task.

    * `:slots` - the number of rows that generate together. The default is `8`.

    * `:max_queue` - the number of requests that can wait for their rows. A request counts one time, also when its input gives more than one row. The default is 4 times `:slots`. When the queue is full, `run/3` and `stream/3` raise `Transmuter.ContinuousBatching.QueueFullError`, so the caller can rescue this case and reject the request.

    * `:capacity` - the largest number of rows of the KV cache of a row, at most the rows of the static tables of the decoder. A request whose prompt and `:max_new_tokens` need more rows raises an `ArgumentError` in the caller. Without it, the server takes the largest capacity up to 4096 whose buffers fit the device memory (see "Device memory"), or 4096 when the device memory is not known.

    * `:device_memory` - the bytes of device memory that the server can use. Without it, the server takes the limit that the backend reports, when it reports one (see "Device memory").

    * `:compile_steps` - when `true`, the server compiles the shape of each step that its rows can take before it starts (see "Compiles"). The default is `false`.

    * `:prefix_cache` - the rows of a store of prompt prefixes on the device (see "Prefix cache"), in slots of the rows of the widest step (512 rows). The default is `0`, no store.

  The other options are those of the function of the task, without `:compile`, `:stream` and `:stream_done`. The server loads the models and compiles the steps when it starts.

  ## Device memory

  All rows of a group have the capacity of its buffers, so one long request grows the KV cache of all `:slots` rows. Without `:capacity`, the server takes the largest capacity whose peak of device memory stays at most 95% of `:device_memory`, with the bytes that the backend has in use after the load, such as the parameters. The peak is the larger of a step at the capacity, with the temporary buffers of its widest step, and a growth into it from the capacity before, which holds the buffers of both capacities. The temporary buffers are two outputs of the widest matmul for each token of the step, one chunk of the attention scores, and with a bf16 cache the f32 copy of the cache that the attention reads. The bytes of each row and position come from the shapes and types of the cache inputs of the decoder, so a model in f16 takes half of the bytes of the same model in f32. The server logs the capacity that it takes.

  EXLA gives the limit of its allocator when `EXLA.Client.get_memory_statistics/1` has a `:limit` key, and EXLA 1.0 has none. EMLX gives no limit. Without a limit, the capacity is 4096. With `compile_steps: true`, a capacity that does not fit fails at the start with an `ArgumentError` that gives the estimate and the budget. Without it, a step fails when the group grows to it.

  For example, Qwen3-0.6B q4 has f32 caches of 229,376 bytes for each row and position. With 8 slots on an RTX 3060 (12 GB, an allocator limit of about 10.6 GiB) and about 2.2 GiB of parameters, `device_memory: trunc(10.6 * 2 ** 30)` gives a capacity of 2048 rows: a growth to 2560 rows holds about 10.1 GiB.

  ## Scope

  The server takes a decoder whose rows can join a running batch, and it raises an `ArgumentError` at the start for the others:

    * A decoder that grows its cache, such as the merged decoder of llama2.c, needs one length of the cache for all rows of a step.
    * With `num_beams` above 1, the rows of a prompt move together as beams, so a request cannot join between them.
    * A merged decoder whose cross-attention cache reads other values than the outputs of the encoder cannot compute that cache for one request (see `Transmuter.Generation.MergedDecoder.cross/1`).

  A merged decoder with an offset for each row, such as `Xenova/gpt2`, runs the prompt of a request on its branch with cache, next to the new tokens of the other rows (see "Merged decoders" in `Transmuter.Generation`).

  The input of a request gives the runs of `:preprocess` of its task, as in a serving of the task, and each run is one row. The rows of a request take the free rows in their order, and the request stays first in the queue until each of its rows has a row. `run/3` gives the result when all rows of the request end, and `stream/3` takes an input of one row.

  ## The server and the serving

  These numbers come from `bench/continuous_batching.exs` on CUDA (RTX 3060), with the server and a serving of batch size 8 at the same request rates. The gap is the time between two chunks of text of a stream in the caller.

  For text, the server gives more tokens per second and earlier first tokens. With SmolLM2-135M, the limit of the server is about 3.9 requests per second. At 1.94, 3.10 and 4.65 requests per second, the server gives 7%, 7% and 13% more tokens per second than the serving. Below the limit, the first token comes about 20 times earlier: 61 ms against 1284 ms at 1.94 requests per second, and 64 ms against 1715 ms at 3.10. The gap at p99 is about 11 ms longer: 44 ms against 33 ms.

  For a vision language model, the serving is as good as the server or better. With SmolVLM-256M, the limit of the server is about 0.84 requests per second. At 0.42 and 0.68 requests per second, both give the same tokens per second, and at 1.01 the serving gives 17.7 against 13.5. At 0.68, the median first token is 2.1 s against 3.3 s, and the gap at p99 is 36 ms against 112 ms. The worker encodes the images of a request on the same GPU while the rows decode, so each decode step and each encoder call takes longer. The serving runs the encoder and the decoder one after the other.

  ## The KV cache

  The server runs decoders in `:in_place` and in `:append`, see `Transmuter.Generation`. In `:in_place`, each row writes its cache after its own rows, and a new request starts again at the first row. The buffers have one capacity for all rows, and they grow by 64 rows when the longest row does not fit them. After each step, the buffers shrink to the capacity of the longest row that runs, with its prompt left and 64 more rows, when that capacity is smaller, in `:in_place` and in `:append`. So a group runs at the capacity of the rows that run, not of the longest row since the group started. Without the key lengths of cuDNN, each step reads the whole buffers: for Qwen3-0.6B fp16 with 16 rows of 64 to 150 tokens, a decode step takes 10.2 ms at 896 rows against 7.9 ms at 512 on CUDA (RTX 3060) with hex EXLA, and 57.9 ms against 45.9 ms on the GPU of a Mac (EMLX). With the cuDNN key lengths, the steps read only the live rows, and a decode step takes 6.0 ms against 5.8 ms. Each move of the buffers by one capacity step copies them: about 22 ms at 896 rows on that CUDA GPU and 13 ms on the Mac.

  In `:append`, all rows of a step write at one offset, so the offset passes the rows of all steps, also the rows of the requests that ended. When a step does not fit, the buffers compact: the valid rows of each running row move to the front, and the offset goes back to the longest running row. After a step, the buffers also compact when the longest running row and 16 rows fit in half of them, so the rows after a long request do not keep its capacity.

  A request joins with the steps of its prompt parts (see "Logits" in `Transmuter.Generation`), and each running row gets its next token in each of these steps. A step with running rows takes at most 64 tokens of a prompt, so a join delays the tokens of the running rows by at most one such step: for Qwen3-0.6B on CUDA, the time between two tokens is at most about 53 ms during a join, against 26 ms for a step without a join.

  On EMLX, MLX keeps the freed buffers of each size in a cache, which has no limit by default. Set a limit, for example `EMLX.set_cache_limit(1_000_000_000)`, when the application starts.

  ## Encoders and images

  A request with images needs the features of its images before it joins, and for Qwen2-VL, Qwen2.5-VL and Qwen3-VL the 3-D positions of its prompt. A worker process of the server prepares one request at a time, in the order of the requests, and then puts the request in the queue. It runs the processor on the images and the tokenizer on the chat, and then the vision encoder on the images. So the steps of the running rows do not wait for the encoder, the caller does no work on the device, and the device memory of this work is that of one request. The processor and the encoder compile one time, in the worker. A request waits in the worker for the requests before it, also a request without images. The queue limit counts the requests in the worker. An error of the processor or of the encoder goes to the caller of the request (see "Failures").

  The encoder and the steps run on the same device. On CUDA, EXLA runs the calls of the two processes side by side: for SmolVLM-256M on an RTX 3060, a decode step takes about 7 ms alone and at most about 16 ms while the worker encodes. On EMLX, both processes use the one GPU stream of EMLX, so a step waits for the encoder call that runs. So the worker runs the encoder in calls of one tile or one image, and on EMLX each call ends before the next starts. While a request with images is prepared, the time between two tokens of a running row can reach one encoder call. For SmolVLM-256M on the GPU of a Mac, a decode step takes about 7 ms alone and about 70 ms while the worker encodes in calls of one tile, against 863 ms with one call for the 13 tiles of an image, and the encoder takes about 1 s for each image. An encoder that takes the patches of an image in one call, such as that of Qwen2-VL, keeps the device for one image.

  A request of an encoder-decoder model, such as Marian with `Transmuter.Text.TextToText` or Florence-2 with `Transmuter.Vision.ImageToText`, needs the outputs of its encoder in the same way. The worker runs the encoder on the input of the request alone, and the request joins with these outputs. The group keeps the outputs of each row, and a request with more rows of the encoder than the group grows the outputs of all rows (see "Merged decoders" in `Transmuter.Generation`). When no row runs, the next request starts a new group, so it takes only the rows of its own outputs.

  A request of `Transmuter.Audio.SpeechToText` gives one row for each window of its audio, as `Transmuter.Audio.speech_to_text/2` cuts long audio, and its result is the result of that serving. The server only transcribes, because `:task` is the module of the task, and it raises an `ArgumentError` at the start for `:timestamps`, whose passes of a window run one after the other. The server runs one pass of each window, so `run/3` raises an `ArgumentError` when the tokens of a window have two timestamps together, which start a second pass in the serving. `stream/3` gives the tokens of the one pass and does not do this check. The worker encodes all windows of a request before it prepares the next request, and the encoder outputs of all windows stay on the device until their rows join: for `Xenova/whisper-tiny.en`, 2.3 MB for each window of 30 s, about 410 MB for one hour of audio. A stream of audio is read in the caller before the request goes to the server.

  The group keeps the features of the image tokens that its rows have not run, and the features of a request join them. Then the prompt of the request runs in parts of at most 64 tokens, next to the new tokens of the running rows (see "The KV cache").

  ## Decode loops

  With `decode_steps: k` above 1, a step runs a decode loop of up to `k` steps in one call (see "Decode loops" in `Transmuter.Generation`) when no request can join in that call: no row is free, or no request waits or prepares. Otherwise it runs one step, so a waiting request joins as soon as a row is free. A loop ends at the latest when a running row reaches its budget, so the row is free after the call. With `compile_steps: true`, the server also compiles each loop at each capacity. On CUDA, Qwen3-0.6B fp16 at 20 requests/s gives 767 tokens/s with 8 slots and 1782 tokens/s with 48 slots at `1`, and 1000 and 2076 tokens/s at `8`.

  ## Compiles

  A step compiles the decoder for its shape the first time that the server meets it: the width of the step and the capacity of the buffers. A step has one of five widths, 1, 16, 32, 64 and 512 tokens (see "Batches" in `Transmuter.Generation`), so a group has at most five shapes for each capacity, 81 for a capacity of 4096 rows. Without `compile_steps: true`, a step that meets a new shape compiles it while rows decode, for example about 3 s on CUDA for SmolLM2-135M.

  With `compile_steps: true`, the server runs one step of each shape before it starts, with the writes of the rows of 1 to `:slots` requests that join together and a growth of the buffers from each capacity to the next, and the worker prepares one input of the task, such as a chat with a small image, so that the processor and the encoder compile too. The steps also compile for the encoder outputs of this input and for each larger encoder capacity. With 8 rows and a capacity of 4096 rows, the start takes 12 s for SmolLM2-135M and 14 s for SmolVLM-256M on the GPU of a Mac, and 146 s and 153 s on CUDA (RTX 3060), against under 1 s and about 6 s without it. These times are from widths 1, 64 and 512. The widths 16 and 32 add two shapes for each capacity: for Qwen3-0.6B fp16 with 48 rows and a capacity of 512 rows on CUDA, the start takes 189 s against 112 s with the three widths. The image features of a group have a power of two of rows (see "Embeddings and image features" in `Transmuter.Generation`), so their merge compiles for few shapes. The processor and an encoder that takes images of each size in one call, such as that of Qwen2-VL, still compile for each new size of image.

  ## Prefix cache

  With `:prefix_cache`, a request reuses the keys and values of the start of its prompt when an earlier request had the same start, such as a system prompt. The server copies them from the store into the row of the request, and the steps run only the rest of the prompt, so the first token comes earlier. The rows of a prefix are whole blocks of 64 tokens, and a prefix leaves at least the last token of the prompt. After the prompt of a request ran, the server copies its whole blocks into the slot that was used the longest time ago, unless the store has them.

  The key of each block is a hash of the hash of the block before it and the tokens of the block, so a block matches only after the same tokens. The server also keeps the tokens of each stored block and compares them on a hit, so a collision of hashes cannot give the keys and values of other tokens. A request with image features or encoder outputs does not use the store, because its tokens do not show them. The server raises at the start for a decoder whose rows cannot reuse a prefix (see "Prefixes" in `Transmuter.Generation`). The store is on the device before the server chooses its capacity, so the capacity counts its bytes. With greedy selection, a request gives the text that it gives without the store, as for any other order of the steps.

  ## Telemetry

  Each step emits `[:transmuter, :continuous_batching, :step, :start]` and `:stop`, with the running rows, the requests that join and the waiting requests. The `:stop` event also has `:steps`, the decode steps of the call, and `:row_steps`, the steps of the rows that had not ended before each step. The duration of a step is the time between two tokens of a running row. The worker emits `[:transmuter, :continuous_batching, :prepare, :start]` and `:stop` for each request, with the time that the request waited for the worker. See `Transmuter.Telemetry`.

  ## Distribution

  The server runs on a node with a device, and `run/3` and `stream/3` take its name on any node of the cluster. A call tries the server of that name on its own node first, and then the servers of that name on the other nodes in a random order. It goes to the first server whose queue takes it, and it raises `QueueFullError` when no queue takes it. The caller node needs Transmuter, but not EMLX, EXLA, the tokenizers or the models: a proxy process on the node of the server validates the input, runs the request and postprocesses the result, and the caller gets the result, the exceptions and the exits of a local call. A stream gets each chunk of text as a message. When the caller exits, or stops a stream before its end, the proxy exits, and the server frees the rows of the request.

  An input goes to the other node, so it must be data, such as strings, maps, lists, `StbImage` structs and tensors. A call copies the tensors of the input to `Nx.BinaryBackend` first, with `Transmuter.to_host/1`, because a tensor on a device backend refers to memory of the node of the caller.

  ## Failures

  The callers of `run/3` and `stream/3` monitor the server. When the server stops, for example when a step raises an out-of-memory error of the device, each waiting caller exits with the reason of the server. When a caller exits, its row ends at the next step and goes to the next request.

  When the worker raises on a request, for example with an out-of-memory error of the device in the vision encoder, the caller of that request gets the exception, and the server and the other requests continue. When the worker stops for another reason, the server stops with the same reason.
  """

  use GenServer

  require Logger

  alias Transmuter.{Generation, Serving}

  @compile {:no_warn_undefined, [EXLA.Client, EMLX]}
  @default_capacity 4096

  defmodule QueueFullError do
    @moduledoc """
    The queue of a `Transmuter.ContinuousBatching` server is full.
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
    server_opts = [:name, :task, :source, slots: 8, max_queue: nil, capacity: nil]
    server_opts = server_opts ++ [device_memory: nil, compile_steps: false, prefix_cache: 0]

    opts =
      opts |> Keyword.validate!(server_opts ++ module.options()) |> Serving.generation_options!()

    # :task is the module here, not an option of the task, such as the task of Florence-2.
    {ctx, tokenizer, task} = module.task(opts[:source], Keyword.delete(opts, :task))

    if reason = Generation.cannot_join(ctx) do
      raise ArgumentError, "requests cannot join a running batch of this model: #{reason}"
    end

    prefix_slots = prefix_slots!(ctx, opts[:prefix_cache])

    # Without :capacity, init/1 takes the capacity up to this one that fits the device memory.
    capacity = opts[:capacity] || @default_capacity
    ctx = %{ctx | max_capacity: min(ctx.max_capacity || capacity, capacity)}

    state = %{
      name: opts[:name],
      ctx: ctx,
      tokenizer: tokenizer,
      task: task,
      slots: opts[:slots],
      max_queue: opts[:max_queue] || 4 * opts[:slots],
      defn_options: opts[:defn_options],
      compile_steps: opts[:compile_steps],
      fixed_capacity?: opts[:capacity] != nil,
      device_memory: opts[:device_memory],
      prefix_slots: prefix_slots
    }

    GenServer.start_link(__MODULE__, state,
      name: opts[:name],
      spawn_opt: Serving.generation_flags()
    )
  end

  # The slots of the store of prefixes: :prefix_cache rows in slots of the rows of the widest
  # step. A decoder whose rows cannot reuse a prefix raises.
  defp prefix_slots!(_ctx, 0), do: 0

  defp prefix_slots!(ctx, rows) do
    if reason = Generation.prefix_reason(ctx) do
      raise ArgumentError, "the rows of this model cannot reuse a stored prefix: #{reason}"
    end

    length = prefix_length(ctx)

    if rows < length do
      raise ArgumentError,
            "a :prefix_cache holds slots of #{length} rows, so it needs at least #{length} rows, got: #{rows}"
    end

    div(rows, length)
  end

  # The rows of a slot of the store: the widest step, the most tokens of a prompt in one step.
  defp prefix_length(ctx), do: Enum.max(ctx.widths.with_cache)

  @doc """
  Generates the text of `input`, an input of the task or a list of them, as `Nx.Serving.batched_run/2` with a serving of the task, and returns the same result.

  Raises `QueueFullError` when no queue of a server of the name can take the inputs (see "Distribution").

  ## Options

    * `:timeout` - after this number of milliseconds, the rows of the inputs end and the caller exits. The default is `:infinity`.

    * `:max_new_tokens` - the most new tokens of each row of the inputs, from 1 to the `:max_new_tokens` of the server. The default is the `:max_new_tokens` of the server. A row ends at its own budget, and its processors, such as `forced_eos_token_id`, use it as `max_new_tokens`.
  """
  def run(server, input, opts \\ []) do
    opts = Keyword.validate!(opts, timeout: :infinity, max_new_tokens: nil)
    call = {__MODULE__, :run, [server, input, opts]}
    server |> servers!(call) |> with_room(&run_on(&1, server, input, opts, call))
  end

  defp run_on(pid, server, input, opts, call) when node(pid) == node(),
    do: run(info!(pid, call), server, input, opts, call)

  defp run_on(pid, _server, input, opts, call), do: remote_run(pid, input, opts, call)

  defp run(info, server, input, opts, call) do
    timeout = opts[:timeout]
    deadline = if timeout == :infinity, do: :infinity, else: now() + timeout
    budget = budget!(info, opts)
    {inputs, multi?} = Serving.validate!(input, info.task.validate)
    refs = for input <- inputs, do: {make_ref(), input}
    monitor = enqueue!(info, refs, {false, budget}, server, deadline, call)

    # The rows of an input end in any order, and :postprocess takes them in the order of
    # their runs.
    results =
      for {ref, _input} <- refs do
        runs = await(info, monitor, {ref, :run}, deadline, refs, call)
        rows = for _run <- runs, do: await(info, monitor, {ref, :done}, deadline, refs, call)
        rows = rows |> List.keysort(0) |> Enum.map(&Serving.integers(elem(&1, 1)))
        info.task.postprocess.(rows, runs)
      end

    Process.demonitor(monitor, [:flush])
    if multi?, do: results, else: hd(results)
  end

  @doc """
  Returns a stream of the text of `input`, as a serving of the task with `stream: true`. Consume the stream in the process that called this function.

  The function returns when the worker has prepared the request (see "Encoders and images"), and it raises the exception of the worker on the request. An input that gives more than one row raises an `ArgumentError`. `stream_done: true` adds `{:done, %{token_summary: summary}}` at the end of the stream, and `:max_new_tokens` is as in `run/3`. Raises `QueueFullError` when no queue of a server of the name can take the input.
  """
  def stream(server, input, opts \\ []) do
    opts = Keyword.validate!(opts, stream_done: false, max_new_tokens: nil)
    reason = {__MODULE__, :stream, [server, input, opts]}
    server |> servers!(reason) |> with_room(&stream_on(&1, server, input, opts, reason))
  end

  defp stream_on(pid, server, input, opts, reason) when node(pid) == node() do
    info = info!(pid, reason)
    budget = budget!(info, opts)

    input =
      case Serving.validate!(input, info.task.validate) do
        {[input], false} -> input
        {_inputs, true} -> raise ArgumentError, "stream/3 takes one input, got a list"
      end

    ref = make_ref()
    monitor = enqueue!(info, [{ref, input}], {true, budget}, server, :infinity, reason)

    run =
      case await(info, monitor, {ref, :run}, :infinity, [{ref, input}], reason) do
        [run] ->
          run

        runs ->
          GenServer.cast(info.pid, {:cancel, [ref]})
          Process.demonitor(monitor, [:flush])

          raise ArgumentError,
                "stream/3 takes an input of one row, and this input gives #{length(runs)} rows"
      end

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

  defp stream_on(pid, _server, input, opts, reason), do: remote_stream(pid, input, opts, reason)

  # The servers of a call, in the order that the call tries them. The server on this node goes
  # first while it has room. Else the servers of the group of the name go from the most room
  # to the least, with a random order among equal rooms, and the servers whose room is not
  # known go last in a random order (see rooms/1).
  defp servers!(server, call) do
    local = GenServer.whereis(server)
    members = if is_atom(server), do: :pg.get_members(Transmuter.PG, server), else: []
    pids = for pid <- [local | members], is_pid(pid), uniq: true, do: pid

    case pids do
      [] ->
        exit({:noproc, call})

      [^local | others] ->
        if has_room?(local), do: [local | Enum.shuffle(others)], else: by_room(pids)

      pids ->
        by_room(pids)
    end
  end

  defp has_room?(pid), do: (room = room(pid)) != nil and room > 0

  # The servers of this node get their room from their info, and the servers of each other
  # node from one call of rooms/1 there. A node that does not answer in 50 ms gives no rooms.
  defp by_room(pids) do
    {here, there} = Enum.split_with(pids, &(node(&1) == node()))
    nodes = there |> Enum.map(&node/1) |> Enum.uniq()

    known =
      for {:ok, rooms} <- :erpc.multicall(nodes, __MODULE__, :rooms, [there], 50),
          pair <- rooms,
          into: Map.new(rooms(here)),
          do: pair

    {rooms, unknown} = Enum.split_with(pids, &Map.has_key?(known, &1))
    sorted = rooms |> Enum.shuffle() |> Enum.sort_by(&known[&1], :desc)
    sorted ++ Enum.shuffle(unknown)
  end

  @doc false
  # The room of each server of `pids` on this node: its free rows less its waiting requests.
  def rooms(pids), do: for(pid <- pids, node(pid) == node(), room = room(pid), do: {pid, room})

  defp room(pid) do
    with %{load: load} <- :persistent_term.get({__MODULE__, pid}, nil), do: :atomics.get(load, 1)
  end

  # Calls `fun` on each server in turn until the queue of one takes the request.
  defp with_room([pid], fun), do: fun.(pid)

  defp with_room([pid | pids], fun) do
    fun.(pid)
  rescue
    QueueFullError -> with_room(pids, fun)
  end

  # The task, the tokenizer and the flags of a server on this node, which it puts in
  # :persistent_term when it starts, so a caller does not wait for a step of the server to
  # read them.
  defp info!(pid, call), do: :persistent_term.get({__MODULE__, pid}, nil) || exit({:noproc, call})

  # The validation, the postprocessing and the tokenizer of a request stay on the node of the
  # server: a proxy process there runs run/3 or stream/3, and exits with the result, as the
  # process of a distributed call of Nx.Serving does. The caller only monitors the proxy, so
  # it gets each end of the proxy as an exit of a local call. The timeout goes as a duration,
  # because monotonic time is local to each node.
  defp remote_run(pid, input, opts, call) do
    {_proxy, ref, monitor} = proxy(pid, :run, [input, opts])

    receive do
      {:DOWN, ^monitor, :process, _proxy, {^ref, {:ok, result}}} -> result
      {:DOWN, ^monitor, :process, _proxy, reason} -> remote_exit!(ref, reason, call)
    end
  end

  # The proxy sends each chunk of the stream as a message, and the caller stops the proxy when
  # it stops the stream before its end.
  defp remote_stream(pid, input, opts, call) do
    {proxy, ref, monitor} = proxy(pid, :stream, [input, opts])

    receive do
      {^ref, :streaming} -> :ok
      {:DOWN, ^monitor, :process, _proxy, reason} -> remote_exit!(ref, reason, call)
    end

    Stream.resource(
      fn -> false end,
      fn false ->
        receive do
          {^ref, {:chunk, chunk}} -> {[chunk], false}
          {:DOWN, ^monitor, :process, _proxy, {^ref, :done}} -> {:halt, true}
          {:DOWN, ^monitor, :process, _proxy, reason} -> remote_exit!(ref, reason, call)
        end
      end,
      fn done? ->
        unless done?, do: Process.exit(proxy, :kill)
        Process.demonitor(monitor, [:flush])
      end
    )
  end

  # The arguments go to the other node as data, so their device tensors become host tensors.
  defp proxy(pid, fun, args) do
    ref = make_ref()
    args = [self(), ref, fun, [pid | Transmuter.to_host(args)]]
    {proxy, monitor} = Node.spawn_monitor(node(pid), __MODULE__, :__proxy__, args)
    {proxy, ref, monitor}
  end

  defp remote_exit!(ref, {ref, {:raise, exception, stacktrace}}, _call),
    do: reraise(exception, stacktrace)

  defp remote_exit!(_ref, reason, call), do: exit({reason, call})

  # The proxy runs the call in a linked runner, which the server knows as the caller of the
  # request, and sends the messages of the runner to the caller. The messages and the end of
  # the proxy then reach the caller in order. The proxy monitors the caller and kills the
  # runner when the caller exits, so the server frees the rows of the request. When the proxy
  # ends, the link ends the runner too.
  @doc false
  def __proxy__(caller, ref, fun, args) do
    Process.flag(:trap_exit, true)
    proxy = self()
    monitor = Process.monitor(caller)
    runner = spawn_link(fn -> exit(reply(proxy, ref, fun, args)) end)
    relay(caller, ref, runner, monitor)
  end

  defp relay(caller, ref, runner, monitor) do
    receive do
      {^ref, _event} = message ->
        send(caller, message)
        relay(caller, ref, runner, monitor)

      {:EXIT, ^runner, reason} ->
        exit(reason)

      {:DOWN, ^monitor, :process, _caller, _reason} ->
        Process.exit(runner, :kill)
        exit(:normal)
    end
  end

  # An exit of run/3 or stream/3 gives its reason without the call on this node, and the
  # caller adds its own call.
  defp reply(proxy, ref, fun, args) do
    {ref, local_call(proxy, ref, fun, args)}
  rescue
    exception -> {ref, {:raise, exception, __STACKTRACE__}}
  catch
    :exit, {reason, {__MODULE__, _fun, _args}} -> reason
  end

  defp local_call(_proxy, _ref, :run, [pid, input, opts]), do: {:ok, run(pid, input, opts)}

  defp local_call(proxy, ref, :stream, [pid, input, opts]) do
    chunks = stream(pid, input, opts)
    send(proxy, {ref, :streaming})
    Enum.each(chunks, &send(proxy, {ref, {:chunk, &1}}))
    :done
  end

  # The server reads its mailbox between two steps, so the reply waits for the step that runs,
  # up to the deadline of the caller.
  defp enqueue!(info, refs, {stream?, budget}, server, deadline, call) do
    monitor = Process.monitor(info.pid)

    requests =
      for {ref, input} <- refs,
          do: %{ref: ref, input: input, stream?: stream?, max_new_tokens: budget}

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

  # The value of the message `tag` of the request `ref`: :run with its runs when the worker
  # has run :preprocess on its input, and :done with the index and the tokens of a row when
  # that row ends.
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

  # The budget of new tokens of each row of a call, at most that of the server, which sizes
  # its cache and its steps.
  defp budget!(info, opts) do
    case opts[:max_new_tokens] || info.max_new_tokens do
      budget when is_integer(budget) and budget in 1..info.max_new_tokens//1 ->
        budget

      budget ->
        raise ArgumentError,
              "expected :max_new_tokens to be an integer from 1 to #{info.max_new_tokens}, the :max_new_tokens of the server, got: #{inspect(budget)}"
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
    # compiles the decoder. The server starts when both are ready, so an error of either fails
    # the start.
    worker =
      spawn_link(fn ->
        task = {preprocess, prepare.(defn_options)}

        # The features and the encoder outputs of the example show the shapes of those of a
        # request, so the steps compile for them.
        example =
          if compile_steps? do
            {:ok, [fields | _rows], _runs} = prepare_request(task, %{input: example})
            fields
          end

        send(server, {self(), :ready, example})
        :erlang.garbage_collect()
        prepare_requests(task, server)
      end)

    ctx = Generation.compile(state.ctx, state.slots, state.defn_options)

    # The store is on the device before the choice of the capacity, so its bytes are in use.
    prefix = prefix_cache(ctx, state.prefix_slots)
    {live, limit} = device_memory(ctx.backend)
    limit = state.device_memory || limit

    capacity =
      if state.fixed_capacity?,
        do: ctx.max_capacity,
        else: Generation.largest_capacity(ctx, state.slots, live, limit)

    ctx = %{ctx | max_capacity: capacity}

    Logger.info(
      "Transmuter.ContinuousBatching: a :capacity of #{capacity} rows for #{state.slots} :slots, " <>
        "with #{live} bytes of device memory in use and a limit of #{inspect(limit)} bytes"
    )

    receive do
      {^worker, :ready, example} ->
        if example do
          compile_steps(ctx, state.slots, example.inputs, {live, limit})
          if example.features, do: Generation.compile_features(ctx, state.slots, example.features)
        end

        {:ok, start(Map.put(state, :prefix, prefix), ctx, worker)}

      {:EXIT, ^worker, reason} ->
        {:stop, reason}
    end
  end

  # The store of prefixes and its index on the host (see "Prefix cache"): for each chained hash
  # of the blocks of a stored prompt, its slot and the tokens up to its block, and for each
  # slot the time of its last use and its hashes.
  defp prefix_cache(_ctx, 0), do: nil

  defp prefix_cache(ctx, slots) do
    %{
      store: Generation.prefix_store(ctx, slots, prefix_length(ctx)),
      length: prefix_length(ctx),
      entries: %{},
      slots: Map.new(0..(slots - 1), &{&1, %{used: 0, hashes: []}}),
      clock: 0
    }
  end

  # The chained hash of each whole block of `prompt` that fits a slot, with the tokens up to
  # it: the hash of a block reads the hash of the block before it, so a block matches only
  # after the same tokens.
  defp prefix_blocks(prompt, length) do
    block = Generation.prefix_block()

    prompt
    |> Enum.take(length)
    |> Enum.chunk_every(block, block, :discard)
    |> Enum.scan({"", []}, fn ids, {hash, tokens} ->
      {:crypto.hash(:sha256, [hash, :erlang.term_to_binary(ids)]), tokens ++ ids}
    end)
  end

  # The stored prefix of a request: its longest stored blocks that leave at least the last
  # token of its prompt. A hit compares the tokens, so a collision of hashes gives no prefix.
  # A request with image features or encoder outputs has no prefix, because its tokens do not
  # show them.
  defp with_prefix(request, prefix)
       when prefix == nil or request.features != nil or
              request.inputs != %{},
       do: {request, prefix}

  defp with_prefix(request, prefix) do
    hit =
      request.prompt
      |> prefix_blocks(prefix.length)
      |> Enum.filter(fn {_hash, tokens} -> length(tokens) < length(request.prompt) end)
      |> Enum.reverse()
      |> Enum.find_value(fn {hash, tokens} ->
        with %{slot: slot, tokens: ^tokens} <- prefix.entries[hash], do: {slot, length(tokens)}
      end)

    case hit do
      {slot, size} ->
        stored = %{store: prefix.store, slot: slot, size: size}
        {Map.put(request, :prefix, stored), touch(prefix, slot)}

      nil ->
        {request, prefix}
    end
  end

  defp touch(prefix, slot) do
    prefix = %{prefix | clock: prefix.clock + 1}
    put_in(prefix.slots[slot].used, prefix.clock)
  end

  # Stores the whole blocks of the prompt of each row whose prompt ran in the last step, in the
  # slot used the longest time ago, unless the store has them.
  defp store_prompts(prefix, _ctx, _group, _rows) when prefix == nil, do: prefix

  defp store_prompts(prefix, ctx, group, rows) do
    for {row, request = %{store?: true}} <- rows,
        Enum.at(group.left, row) == [],
        reduce: prefix do
      prefix ->
        blocks = prefix_blocks(request.prompt, prefix.length)
        store_blocks(prefix, ctx, group, row, blocks)
    end
  end

  defp store_blocks(prefix, _ctx, _group, _row, []), do: prefix

  defp store_blocks(prefix, ctx, group, row, blocks) do
    {last, tokens} = List.last(blocks)

    case prefix.entries[last] do
      %{slot: slot, tokens: ^tokens} ->
        touch(prefix, slot)

      _missing ->
        {slot, %{hashes: old}} = Enum.min_by(prefix.slots, fn {_slot, %{used: used}} -> used end)

        entries =
          Map.reject(prefix.entries, fn {hash, entry} -> hash in old and entry.slot == slot end)

        store = Generation.store_prefix(ctx, group, prefix.store, row, slot, length(tokens))
        hashes = Enum.map(blocks, &elem(&1, 0))
        added = Map.new(blocks, fn {hash, tokens} -> {hash, %{slot: slot, tokens: tokens}} end)
        prefix = %{prefix | store: store, entries: Map.merge(entries, added)}
        touch(put_in(prefix.slots[slot].hashes, hashes), slot)
    end
  end

  # A step that does not fit the device fails in EXLA with a RuntimeError whose message has
  # RESOURCE_EXHAUSTED or "Out of memory". No other field tells this error apart, so the match
  # reads the text, and other errors pass.
  defp compile_steps(ctx, slots, inputs, {live, limit}) do
    Generation.compile_steps(ctx, slots, inputs)
  rescue
    error in RuntimeError ->
      if error.message =~ ~r/RESOURCE_EXHAUSTED|Out of memory/ do
        error = Generation.memory_error(ctx, slots, live, limit, ctx.max_capacity)

        reraise ArgumentError,
                "the device ran out of memory in the steps of the server, so the estimate is " <>
                  "too low: " <> error.message,
                __STACKTRACE__
      else
        reraise error, __STACKTRACE__
      end
  end

  # The bytes in use on the device of `backend` and the limit of its allocator, when the backend
  # reports one: EXLA in a :limit key of its memory statistics, for each device. EMLX reports no
  # limit.
  defp device_memory({EXLA.Backend, opts}) do
    client = EXLA.Client.fetch!(opts[:client] || EXLA.Client.default_name())
    device = opts[:device_id] || client.default_device_id
    stats = EXLA.Client.get_memory_statistics(client)
    {Map.get(stats.per_device, device, stats.allocated), stats[:limit][device]}
  end

  defp device_memory({EMLX.Backend, _opts}), do: {EMLX.memory_info().active_memory, nil}
  defp device_memory(_backend), do: {0, nil}

  defp start(state, ctx, worker) do
    info = %{
      pid: self(),
      task: Map.take(state.task, [:validate, :postprocess]),
      tokenizer: state.tokenizer,
      logprobs?: ctx.token_logprobs?,
      max_new_tokens: ctx.max_new_tokens,
      max_queue: state.max_queue,
      load: :atomics.new(1, signed: true)
    }

    :persistent_term.put({__MODULE__, self()}, info)
    if state.name, do: :pg.join(Transmuter.PG, state.name, self())

    state
    |> Map.merge(%{
      ctx: ctx,
      worker: worker,
      group: nil,
      rows: %{},
      preparing: %{},
      queue: :queue.new(),
      callers: %{},
      stepping?: false,
      load: info.load
    })
    |> publish()
  end

  # Each callback ends here, so the room of the server always follows its state: its free
  # rows less the rows of its waiting requests and its requests in the worker.
  defp publish(state) do
    waiting = state.queue |> :queue.to_list() |> Enum.map(&length(&1.rows)) |> Enum.sum()
    room = state.slots - map_size(state.rows) - waiting - map_size(state.preparing)
    :atomics.put(state.load, 1, room)
    state
  end

  defp published({:reply, reply, state}), do: {:reply, reply, publish(state)}
  defp published({:noreply, state}), do: {:noreply, publish(state)}
  defp published(stop = {:stop, _reason, _state}), do: stop

  @impl true
  def handle_call(message, from, state), do: message |> on_call(from, state) |> published()

  @impl true
  def handle_cast(message, state), do: message |> on_cast(state) |> published()

  @impl true
  def handle_info(message, state), do: message |> on_info(state) |> published()

  # The requests go to the worker, which sends each one back with its inputs.
  defp on_call({:enqueue, requests}, {pid, _tag}, state) do
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

  defp on_cast({:cancel, refs}, state), do: {:noreply, drop(state, &(&1.ref in refs))}

  defp on_info(:step, state) do
    state = %{state | stepping?: false}
    state = if active?(state) or :queue.is_empty(state.queue), do: state, else: open(state)
    state = if state.group, do: forget_callers(step(state)), else: state

    if active?(state) or not :queue.is_empty(state.queue),
      do: {:noreply, kick(state)},
      else: {:noreply, close(state)}
  end

  # A request that the caller cancelled in the worker has no entry.
  defp on_info({:prepared, ref, reply}, state) do
    case {Map.pop(state.preparing, ref), reply} do
      {{nil, _preparing}, _reply} ->
        {:noreply, state}

      {{request, preparing}, {:ok, rows, runs}} ->
        send(request.pid, {ref, {:run, runs}})
        request = request |> Map.delete(:input) |> Map.put(:rows, Enum.with_index(rows))
        {:noreply, kick(%{state | preparing: preparing, queue: :queue.in(request, state.queue)})}

      {{request, preparing}, {:error, exception}} ->
        send(request.pid, {ref, {:error, exception}})
        {:noreply, forget_callers(%{state | preparing: preparing})}
    end
  end

  defp on_info({:DOWN, _monitor, :process, pid, _reason}, state) do
    {:noreply, drop(%{state | callers: Map.delete(state.callers, pid)}, &(&1.pid == pid))}
  end

  defp on_info({:EXIT, worker, reason}, state = %{worker: worker}),
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

  # Gives the free rows to the rows of the waiting requests in their order, while the cache
  # fits them, and runs the step. A request stays first in the queue until all its rows have
  # a row of the group. A row that does not fit a group without running rows never fits, so
  # the caller of its request gets an error.
  defp step(state) do
    {admitted, state} = admissions(state)

    {admitted, prefix} =
      Enum.map_reduce(admitted, state.prefix, fn {row, request}, prefix ->
        {request, prefix} = with_prefix(request, prefix)
        store? = prefix != nil and request.features == nil and request.inputs == %{}
        {{row, Map.put(request, :store?, store?)}, prefix}
      end)

    state = %{state | prefix: prefix}

    if admitted == [] and not active?(state) do
      state
    else
      metadata = %{
        running: map_size(state.rows),
        admitted: length(admitted),
        prefix_tokens:
          Enum.sum(for {_row, request} <- admitted, do: get_in(request, [:prefix, :size]) || 0),
        prompts: Enum.count(state.group.left, &(&1 != [])) + length(admitted),
        waiting: :queue.len(state.queue),
        slots: state.slots,
        capacity: state.group.state.capacity,
        encoder_capacity: state.group.state.encoder_capacity
      }

      :telemetry.span([:transmuter, :continuous_batching, :step], metadata, fn ->
        state = step(state, admitted)
        {state, Map.merge(metadata, loop_metadata(state.group.steps, metadata.running))}
      end)
    end
  end

  defp step(state, admitted) do
    requests = for {row, request} <- admitted, do: request(row, request)

    group =
      if requests == [],
        do: state.group,
        else: Generation.admit(state.ctx, state.group, requests)

    # A loop of steps when no request can join in this call: no row is free, or no request
    # waits or prepares. Generation caps the loop at the tokens that the running rows have
    # left, so a row that reaches its budget frees its row at the end of the call.
    free? = Enum.any?(group.done)
    waiting? = not :queue.is_empty(state.queue) or state.preparing != %{}
    steps = if free? and waiting?, do: 1, else: state.ctx.decode_steps

    group = Generation.next(state.ctx, group, steps)

    # The group keeps the features, the encoder outputs and the prefixes, so the rows do not
    # keep them for their callers.
    admitted =
      Map.new(admitted, fn {row, request} ->
        {row, Map.drop(request, [:features, :inputs, :prefix])}
      end)

    rows = Map.merge(state.rows, admitted)
    prefix = store_prompts(state.prefix, state.ctx, group, rows)
    rows = Map.new(rows, fn {row, request} -> {row, stored(request, group, row)} end)
    group = Generation.shrink(state.ctx, group)
    emit(%{state | rows: rows, prefix: prefix}, group)
  end

  # The steps of the call, and the steps of the rows that had not ended before each step. A
  # row that ends in a step of the decode loop runs the later steps of the call without a token.
  defp loop_metadata(views, running) do
    before = Enum.map(Enum.drop(views, -1), fn view -> Enum.count(view.done, &(not &1)) end)
    %{steps: length(views), row_steps: running + Enum.sum(before)}
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

      {{:value, waiting}, rest} ->
        [{fields, index} | rows] = waiting.rows
        next = waiting |> Map.delete(:rows) |> Map.merge(fields) |> Map.put(:index, index)
        queue = if rows == [], do: rest, else: :queue.in_r(%{waiting | rows: rows}, rest)
        candidates = [{row, next} | admitted]
        requests = for {row, waiting} <- candidates, do: request(row, waiting)

        cond do
          Generation.rows_needed(state.ctx, state.group, requests) <= state.ctx.max_capacity ->
            admissions(%{state | queue: queue}, free, candidates)

          admitted == [] and not active?(state) ->
            exception = ArgumentError.exception(too_long(state, next))
            send(waiting.pid, {waiting.ref, {:error, exception}})
            admissions(%{state | queue: rest}, [row | free], admitted)

          true ->
            {Enum.reverse(admitted), state}
        end
    end
  end

  # A row stores the blocks of its prompt once, after its prompt ran.
  defp stored(request = %{store?: true}, group, row),
    do: %{request | store?: Enum.at(group.left, row) != []}

  defp stored(request, _group, _row), do: request

  defp request(row, request) do
    request
    |> Map.take([:prompt, :seed, :features, :position, :inputs, :prefix])
    |> Map.merge(%{row: row, budget: request.max_new_tokens})
  end

  defp too_long(state, request) do
    "a prompt of #{length(request.prompt)} tokens and max_new_tokens of #{request.max_new_tokens} need more than the #{state.ctx.max_capacity} rows of the cache of the server"
  end

  # Sends the new token of each row to its stream, and the tokens of each row that ends to
  # its caller. A row with a part of its prompt left has no token yet. A row that ends is free
  # for the next step.
  defp emit(state, group) do
    rows = Enum.reduce(group.steps, state.rows, &emit_step(&2, Map.merge(group, &1)))
    %{state | group: group, rows: rows}
  end

  defp emit_step(rows, group) do
    Enum.reduce(rows, rows, fn {row, request}, rows ->
      token = Enum.at(group.new, row)
      done? = Enum.at(group.done, row)

      cond do
        token == nil and not done? ->
          :ok

        request.stream? ->
          logprob = if token, do: List.first(hd(Enum.at(group.logprobs, row)))
          send(request.pid, {request.ref, {:token, token || -1, logprob, done?}})

        done? ->
          tokens = Generation.row_tokens(group, row)
          send(request.pid, {request.ref, {:done, {request.index, tokens}}})

        true ->
          :ok
      end

      if done?, do: Map.delete(rows, row), else: rows
    end)
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
  # on the tensors on the host and the prompt of each run. It sends the server the fields of
  # each row for Generation and the runs without their tensors, or the exception.
  defp prepare_requests(task, server) do
    receive do
      {:prepare, request} ->
        metadata = %{wait: System.monotonic_time() - request.enqueued}

        reply =
          :telemetry.span([:transmuter, :continuous_batching, :prepare], metadata, fn ->
            {prepare_request(task, request), metadata}
          end)

        send(server, {:prepared, request.ref, reply})
        # The device tensors of the request, such as the tiles of its images, die here,
        # and only a garbage collection frees them on EMLX. The worker makes little other
        # garbage, so without it they stay for many requests.
        :erlang.garbage_collect()
        prepare_requests(task, server)
    end
  end

  defp prepare_request({preprocess, prepare}, request) do
    runs = Enum.to_list(preprocess.(request.input))
    rows = Enum.map(runs, &prepare_row(prepare, &1))
    {:ok, rows, Enum.map(runs, &Map.delete(&1, :tensors))}
  rescue
    exception -> {:error, exception}
  end

  defp prepare_row(prepare, run) do
    tensors = Map.new(run.tensors, fn {name, tensor} -> {name, Nx.new_axis(tensor, 0)} end)

    # The other inputs of the prompt, such as the outputs of an encoder, have a batch axis of 1.
    # The worker evaluates them, so the encoder runs here and not in a step of the server.
    {[position], inputs} = Map.pop(prepare.(tensors, [run.prompt]), :position_ids, [[]])
    {[features], inputs} = Map.pop(inputs, :image_features, [nil])
    features = features && Generation.evaluated(features)
    fields = %{prompt: run.prompt, seed: run.seed, features: features, position: position}
    Map.put(fields, :inputs, Generation.evaluated(inputs))
  end
end
