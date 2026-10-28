# Requests that arrive over time, to a server of Transmuter.ContinuousGeneration with 8
# slots or to a started serving of Transmuter.Text.generation/2 with batch_size: 8 and
# batch_timeout: 10, on the GPU of EMLX, or of EXLA with CUDA when BENCH_BACKEND is exla.
# EXLA allocates device memory when it needs it, not in one arena at the start, so the memory
# of the device shows the demand of the run. With smolvlm, the requests are chats of
# Transmuter.Vision.image_text_to_text/2: 3 of 4 have one of 3 images, and 1 of 4 has no
# image. The requests arrive at `rate` requests per second, with exponential times between them from a
# fixed seed, and stream their text with greedy selection and up to 256 new tokens. EMLX gets
# a limit of 1 GB on its MLX cache. Warm rounds of the same requests at the same times come
# first, so that the measured requests find the shapes of their steps compiled. The batches of
# the serving depend on the times of its calls, so a round with compiles can make other
# batches than a round without them. So the warm rounds repeat until one has no compile, at
# most 3 times. The server
# starts with compile_steps: true, so it compiles each step shape before the warm round.
#
# Prints, for the measured requests:
#   - tokens/s: all new tokens, divided by the time from the first arrival to the last end;
#   - first token: the time from the arrival to the first chunk of text (p50 and p95);
#   - gaps (in RESULT and GAPS): the time between two chunks of text of one stream, in the
#     caller, for both modes, which is what a client sees;
#   - step time (STEPTIME): for the server, each step for each row with a request, also a row
#     in its prompt; for the serving, the time between two compiled calls of one batch. It is
#     not what a client sees;
#   - gaps: the time between two chunks of text of one stream, in the caller, for both modes
#     (p50, p95, p99 and the largest). A chunk can hold more than one token, but this time
#     shows each wait of a row that runs, also for the encoder or for the prompt of a join;
#   - tokens per call: all new tokens, divided by the compiled calls;
#   - for the server, the time of :prepare in the worker for each request, such as the vision
#     encoder, and the time that the request waited for the worker before it (p50, p95 and the
#     largest);
#   - for the server, the time between tokens in the decode steps (no prompt part), with and without a
#     run of :prepare in the worker at the same time (p50, p99 and the largest).
#
# Usage: mix run bench/continuous_generation.exs smollm2|smollm2-onnx|qwen3|smolvlm server|serving <rate> [requests]
Logger.configure(level: :warning)

# The MLX cache of freed buffers has no limit by default, see the moduledoc of
# Transmuter.ContinuousGeneration. Both modes run with the same limit.
device =
  if System.get_env("BENCH_BACKEND", "emlx") == "emlx" do
    EMLX.set_cache_limit(1_000_000_000)
    [backend: {EMLX.Backend, device: :gpu}, defn_options: [compiler: EMLX]]
  else
    Application.put_env(:exla, :clients, cuda: [platform: :cuda, preallocate: false])
    [backend: {EXLA.Backend, client: :cuda}, defn_options: [compiler: EXLA, client: :cuda]]
  end

[name, mode, rate | rest] = System.argv()
rate = String.to_float(rate)
count = String.to_integer(List.first(rest) || "48")

questions = [
  "What is the capital of France?",
  "Explain how a rainbow forms.",
  "Write a short poem about the sea.",
  "Why is the sky blue?",
  "Give me three tips for learning a new language.",
  "What does a compiler do?",
  "Describe the water cycle.",
  "How do bees make honey?",
  "Summarize the plot of a heist movie.",
  "What is the difference between weather and climate?",
  "Write a haiku about autumn.",
  "How does a bicycle stay upright?",
  "List five fruits that are rich in vitamin C.",
  "What causes the seasons on Earth?",
  "Explain recursion to a child.",
  "Tell me a fun fact about octopuses."
]

{repo, task, function} =
  case name do
    "smollm2" ->
      {"HuggingFaceTB/SmolLM2-135M-Instruct", Transmuter.Text.Generation, :text}

    "smollm2-onnx" ->
      {"onnx-community/SmolLM2-135M-Instruct-ONNX", Transmuter.Text.Generation, :text}

    "qwen3" ->
      {"onnx-community/Qwen3-0.6B-ONNX", Transmuter.Text.Generation, :text}

    "smolvlm" ->
      {"HuggingFaceTB/SmolVLM-256M-Instruct", Transmuter.Vision.ImageTextToText, :image}
  end

variant = if function == :text, do: [variant: :q4], else: []
opts = variant ++ [strategy: :greedy, max_new_tokens: 256] ++ device

image = StbImage.read_file!("test/fixtures/cats.png")
pixels = StbImage.to_nx(image)

images = [
  image,
  StbImage.from_nx(Nx.transpose(pixels, axes: [1, 0, 2])),
  StbImage.from_nx(pixels[[0..159, 0..213]])
]

image_questions = [
  "Describe this image.",
  "What animals are in the image?",
  "What colors do you see in the image?"
]

inputs =
  for i <- 0..(count - 1) do
    content =
      if function == :image and rem(i, 4) != 3 do
        [
          %{type: "image", image: Enum.at(images, rem(i, 3))},
          %{type: "text", text: Enum.at(image_questions, rem(div(i, 4), 3))}
        ]
      else
        Enum.at(questions, rem(i, 16))
      end

    %{messages: [%{role: "user", content: content}]}
  end

stream =
  case mode do
    "server" ->
      {:ok, _pid} =
        Transmuter.ContinuousGeneration.start_link(
          [name: Bench, task: task, source: {:hf, repo}, slots: 8, max_queue: count] ++
            [compile_steps: true] ++ opts
        )

      &Transmuter.ContinuousGeneration.stream(Bench, &1, stream_done: true)

    "serving" ->
      serving_opts = [compile: [batch_size: 8], stream: true, stream_done: true] ++ opts

      serving =
        if function == :text,
          do: Transmuter.Text.generation({:hf, repo}, serving_opts),
          else: Transmuter.Vision.image_text_to_text({:hf, repo}, serving_opts)

      {:ok, _pid} = Nx.Serving.start_link(serving: serving, name: Bench, batch_timeout: 10)
      &Nx.Serving.batched_run(Bench, &1)
  end

# The arrival, the first chunk, the end and the new tokens of a request, in milliseconds.
request = fn input, start ->
  arrival = System.monotonic_time(:millisecond) - start

  {chunks, tokens} =
    input
    |> stream.()
    |> Enum.reduce({[], 0}, fn
      {:done, %{token_summary: %{output: output}}}, {chunks, _tokens} -> {chunks, output}
      _chunk, {chunks, tokens} -> {[System.monotonic_time(:microsecond) | chunks], tokens}
    end)

  chunks = Enum.reverse(chunks)
  gaps = chunks |> Enum.chunk_every(2, 1, :discard) |> Enum.map(fn [a, b] -> (b - a) / 1000 end)

  %{
    arrival: arrival,
    first: div(hd(chunks), 1000) - start,
    end: System.monotonic_time(:millisecond) - start,
    tokens: tokens,
    gaps: gaps
  }
end

# The arrival times, from a fixed seed, and a round of the requests at these times.
:rand.seed(:exsss, 42)
gaps = for _ <- 1..count, do: round(-:math.log(:rand.uniform()) / rate * 1000)

run_round = fn schedule, start ->
  for {at, input} <- schedule do
    Task.async(fn ->
      Process.sleep(max(at - (System.monotonic_time(:millisecond) - start), 0))
      request.(input, start)
    end)
  end
end

compiles = :counters.new(6, [])
phase = :atomics.new(1, [])
compiled = :ets.new(:compiled, [:public, :bag])

# The shapes of the compiles in the measured requests: the arguments of the segment whose
# names show the step (tokens and cache rows) or the encoder call.
:telemetry.attach(
  :compiles,
  [:transmuter, :segment, :compile, :stop],
  fn _event, measurements, metadata, _config ->
    measured = :atomics.get(phase, 1)
    :counters.add(compiles, measured + 1, 1)

    if measured == 1 do
      :counters.add(compiles, 3, measurements.duration)

      shapes =
        for {name, {shape, _type}} <- metadata.shapes,
            name in ["inputs_embeds", "input_ids", "attention_mask", "pixel_values"],
            do: {name, shape}

      :ets.insert(compiled, {metadata.position, shapes})
    end
  end,
  nil
)

# Each compile of EXLA in the measured requests, also of the small jit functions of
# generation and of the eager operations of EXLA.Backend, which give no segment events.
:telemetry.attach(
  :exla_compiles,
  [:exla, :compilation],
  fn _event, measurements, metadata, _config ->
    if :atomics.get(phase, 1) == 0, do: :counters.add(compiles, 6, 1)

    if :atomics.get(phase, 1) == 1 do
      :counters.add(compiles, 4, 1)
      :counters.add(compiles, 5, measurements.total_time)

      # The frames of Transmuter that called the compile.
      {:current_stacktrace, stack} = Process.info(self(), :current_stacktrace)

      site =
        for {module, function, arity, _location} <- stack,
            String.starts_with?(inspect(module), "Transmuter"),
            do: "#{inspect(module)}.#{function}/#{arity}"

      key = inspect(metadata.key, limit: 12, printable_limit: 200)

      :ets.insert(
        compiled,
        {:exla, self(),
         String.slice(key, 0, 80) <> " from " <> Enum.join(Enum.take(site, 3), " < ")}
      )
    end
  end,
  nil
)

schedule = gaps |> Enum.scan(&(&1 + &2)) |> Enum.zip(inputs)

warm = fn warm, round ->
  before = {:counters.get(compiles, 1), :counters.get(compiles, 6)}
  schedule |> run_round.(System.monotonic_time(:millisecond)) |> Task.await_many(:infinity)
  after_round = {:counters.get(compiles, 1), :counters.get(compiles, 6)}
  if after_round != before and round < 3, do: warm.(warm, round + 1), else: round
end

warm_rounds = warm.(warm, 1)

# The step events of the server and the calls of each process, as {pid, time, duration}.
events = :ets.new(:events, [:public, :bag])

:telemetry.attach_many(
  :bench,
  [
    [:transmuter, :call, :stop],
    [:transmuter, :continuous_generation, :step, :stop],
    [:transmuter, :continuous_generation, :prepare, :stop]
  ],
  fn event, measurements, metadata, _config ->
    :ets.insert(
      events,
      {event, self(), measurements.monotonic_time, measurements.duration, metadata}
    )
  end,
  nil
)

:atomics.put(phase, 1, 1)
start = System.monotonic_time(:millisecond)
results = schedule |> run_round.(start) |> Task.await_many(:infinity)
gaps = Enum.flat_map(results, & &1.gaps)

:telemetry.detach(:bench)
ms = &(System.convert_time_unit(&1, :native, :microsecond) / 1000)
percentile = fn values, p -> values |> Enum.sort() |> Enum.at(round(p * (length(values) - 1))) end
tokens = results |> Enum.map(& &1.tokens) |> Enum.sum()
span = Enum.max(Enum.map(results, & &1.end)) - Enum.min(Enum.map(results, & &1.arrival))
first = Enum.map(results, &(&1.first - &1.arrival))
calls = :ets.lookup(events, [:transmuter, :call, :stop])
steps = :ets.lookup(events, [:transmuter, :continuous_generation, :step, :stop])
prepares = :ets.lookup(events, [:transmuter, :continuous_generation, :prepare, :stop])

between =
  case mode do
    "server" ->
      for {_event, _pid, _time, duration, %{running: running}} <- steps,
          _row <- List.duplicate(nil, running),
          do: ms.(duration)

    "serving" ->
      calls
      |> Enum.group_by(&elem(&1, 1), &elem(&1, 2))
      |> Enum.flat_map(fn {_pid, times} ->
        times
        |> Enum.sort()
        |> Enum.chunk_every(2, 1, :discard)
        |> Enum.map(fn [a, b] -> ms.(b - a) end)
      end)
  end

# The server admits the requests in the order of their arrival, so the request of rank k
# joins in the step where the count of admitted requests reaches k. Each split is the time
# from the arrival to the start of that step, and from there to the first chunk.
splits =
  steps
  |> Enum.map(fn {_event, _pid, time, duration, metadata} ->
    {System.convert_time_unit(time - duration, :native, :millisecond) - start, metadata.admitted}
  end)
  |> Enum.sort()
  |> Enum.flat_map(fn {at, admitted} -> List.duplicate(at, admitted) end)
  |> Enum.zip(Enum.sort_by(results, & &1.arrival))
  |> Enum.map(fn {admit, result} -> {admit - result.arrival, result.first - admit} end)

f = &:erlang.float_to_binary(&1 / 1, decimals: 1)

IO.puts(
  "RESULT #{name} #{mode} rate #{rate}: #{f.(tokens / (span / 1000))} tokens/s, first token p50 #{percentile.(first, 0.5)} ms p95 #{percentile.(first, 0.95)} ms, gaps between chunks p50 #{f.(percentile.(gaps, 0.5))} ms p95 #{f.(percentile.(gaps, 0.95))} ms p99 #{f.(percentile.(gaps, 0.99))} ms max #{f.(Enum.max(gaps))} ms, #{f.(tokens / length(calls))} tokens per call, #{tokens} tokens"
)

# The step time for each running row (the server) or between two calls (the serving). It is
# not what a client sees: a step of a row in its prompt counts too, and a call of the serving
# can be an embedding. See GAPS for the client.
IO.puts(
  "STEPTIME #{name} #{mode} rate #{rate}: p50 #{f.(percentile.(between, 0.5))} ms p95 #{f.(percentile.(between, 0.95))} ms p99 #{f.(percentile.(between, 0.99))} ms max #{f.(Enum.max(between))} ms"
)

# The steps without and with requests that join, the compiled calls in each, and the mean
# rows of the decode steps, which give one token to each running row and run no prompt part.
step_calls = fn {_event, pid, time, duration, _metadata} ->
  Enum.count(calls, fn {_call, call_pid, at, _duration, _metadata} ->
    call_pid == pid and at > time - duration and at <= time
  end)
end

{prompted, decodes} =
  Enum.split_with(steps, fn {_e, _p, _t, _d, metadata} -> metadata.prompts > 0 end)

if steps != [] do
  mean = fn values -> Enum.sum(values) / max(length(values), 1) end

  IO.puts(
    "COUNTS #{name} #{mode} rate #{rate}: #{length(decodes)} decode steps with #{Enum.sum(Enum.map(decodes, step_calls))} calls and #{f.(mean.(Enum.map(decodes, &elem(&1, 4).running)))} mean rows, #{length(prompted)} prompt steps with #{Enum.sum(Enum.map(prompted, step_calls))} calls, #{f.(mean.(Enum.map(prompted, &elem(&1, 4).prompts)))} mean prompt rows and #{f.(mean.(Enum.map(prompted, &elem(&1, 4).running)))} mean running rows, #{length(calls)} calls in all"
  )
end

server_pid = Process.whereis(Bench)

exla_in = fn pid ->
  keys = for {:exla, ^pid, key} <- :ets.tab2list(compiled), do: key
  "#{length(keys)} #{inspect(keys)}"
end

# The slowest steps of the server: duration, and the rows, joins, prompt rows and capacity
# before each.
if steps != [] do
  slowest =
    steps
    |> Enum.sort_by(&(-elem(&1, 3)))
    |> Enum.take(3)
    |> Enum.map(fn {_event, _pid, _time, duration, metadata} ->
      {f.(ms.(duration)), Map.take(metadata, [:running, :admitted, :prompts, :capacity])}
    end)

  IO.puts("SLOWEST #{name} #{mode} rate #{rate}: #{inspect(slowest)}")
end

# The functions of the EXLA compiles in the measured requests, with their count, the most
# frequent first.
exla_keys =
  for({:exla, _pid, key} <- :ets.tab2list(compiled), do: key)
  |> Enum.frequencies()
  |> Enum.sort_by(&(-elem(&1, 1)))
  |> Enum.take(20)

IO.puts("EXLA #{name} #{mode} rate #{rate}: #{inspect(exla_keys)}")

IO.puts(
  "COMPILES #{name} #{mode} rate #{rate}: #{:counters.get(compiles, 1)} segment and #{:counters.get(compiles, 6)} EXLA compiles in #{warm_rounds} warm rounds, #{:counters.get(compiles, 2)} in the measured requests: #{inspect(for entry = {position, _shapes} <- :ets.tab2list(compiled), position != :exla, do: entry)}, and #{:counters.get(compiles, 4)} EXLA compiles of #{f.(:counters.get(compiles, 5) / 1000)} ms in all in the measured requests, #{exla_in.(server_pid)} of them in the server process"
)

# The wall time of the measured requests, split into the decode steps and the steps with a
# prompt part, the
# compiles in them, and the time between steps. A compile runs inside a step.
step_time = fn steps -> steps |> Enum.map(&ms.(elem(&1, 3))) |> Enum.sum() end
compile_time = ms.(:counters.get(compiles, 3))

if steps != [] do
  IO.puts(
    "TIME #{name} #{mode} rate #{rate}: #{span} ms in all, #{f.(step_time.(decodes))} ms in decode steps, #{f.(step_time.(prompted))} ms in prompt steps, #{f.(compile_time)} ms of compiles, #{f.(span - step_time.(decodes) - step_time.(prompted))} ms between steps"
  )
end

# The time of the decode steps, for each capacity of the buffers.
for {capacity, durations} <-
      Enum.group_by(decodes, &elem(&1, 4).capacity, &ms.(elem(&1, 3))) do
  IO.puts(
    "CAPACITY #{name} #{mode} rate #{rate}: #{capacity} rows, #{length(durations)} decode steps, p50 #{f.(percentile.(durations, 0.5))} ms"
  )
end

if splits != [] do
  {waits, firsts} = Enum.unzip(splits)
  slowest = splits |> Enum.sort_by(fn {wait, first} -> -(wait + first) end) |> Enum.take(3)

  IO.puts(
    "SPLIT #{name} #{mode} rate #{rate}: arrival to admit p50 #{percentile.(waits, 0.5)} ms p95 #{percentile.(waits, 0.95)} ms, admit to first chunk p50 #{percentile.(firsts, 0.5)} ms p95 #{percentile.(firsts, 0.95)} ms, slowest #{inspect(slowest)}"
  )
end

if prepares != [] do
  durations = Enum.map(prepares, &ms.(elem(&1, 3)))
  waits = Enum.map(prepares, &ms.(elem(&1, 4).wait))

  IO.puts(
    "PREPARE #{name} #{mode} rate #{rate}: #{length(prepares)} requests, prepare p50 #{f.(percentile.(durations, 0.5))} ms p95 #{f.(percentile.(durations, 0.95))} ms max #{f.(Enum.max(durations))} ms, wait for the worker p50 #{f.(percentile.(waits, 0.5))} ms p95 #{f.(percentile.(waits, 0.95))} ms max #{f.(Enum.max(waits))} ms"
  )

  # The decode steps, with and without a run of :prepare at the same time.
  spans = for {_event, _pid, time, duration, _metadata} <- prepares, do: {time - duration, time}

  {shared, alone} =
    Enum.split_with(decodes, fn {_event, _pid, time, duration, _metadata} ->
      Enum.any?(spans, fn {from, to} -> from < time and to > time - duration end)
    end)

  for {label, steps} <- [{"with prepare", shared}, {"without prepare", alone}], steps != [] do
    durations = Enum.map(steps, &ms.(elem(&1, 3)))

    IO.puts(
      "OVERLAP #{name} #{mode} rate #{rate}: #{length(steps)} decode steps #{label}, between tokens p50 #{f.(percentile.(durations, 0.5))} ms p99 #{f.(percentile.(durations, 0.99))} ms max #{f.(Enum.max(durations))} ms"
    )
  end
end

IO.puts(
  "GAPS #{name} #{mode} rate #{rate}: #{length(gaps)} gaps between chunks, p50 #{f.(percentile.(gaps, 0.5))} ms p95 #{f.(percentile.(gaps, 0.95))} ms p99 #{f.(percentile.(gaps, 0.99))} ms max #{f.(Enum.max(gaps))} ms"
)
