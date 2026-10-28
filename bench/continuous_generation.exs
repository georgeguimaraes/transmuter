# Requests that arrive over time, to a server of Transmuter.Text.ContinuousGeneration with 8
# slots or to a started serving of Transmuter.Text.generation/2 with batch_size: 8 and
# batch_timeout: 10, on the GPU of EMLX, or of EXLA with CUDA when BENCH_BACKEND is exla. The
# requests arrive at `rate` requests per second, with exponential times between them from a
# fixed seed, and stream their text with greedy selection and up to 256 new tokens. EMLX gets
# a limit of 1 GB on its MLX cache. A warm round of the same requests at the same times comes
# first, so that the measured requests find the shapes of their steps compiled.
#
# Prints, for the measured requests:
#   - tokens/s: all new tokens, divided by the time from the first arrival to the last end;
#   - first token: the time from the arrival to the first chunk of text (p50 and p95);
#   - between tokens: the time of a step for each running row (p50 and p95). For the server,
#     from the step events, where a step with prompt parts that join counts for the rows
#     that already ran. For the serving, the time between two compiled calls of one batch;
#   - tokens per call: all new tokens, divided by the compiled calls.
#
# Usage: mix run bench/continuous_generation.exs smollm2|qwen3 server|serving <rate> [requests]
Logger.configure(level: :warning)

# The MLX cache of freed buffers has no limit by default, see the moduledoc of
# Transmuter.Text.ContinuousGeneration. Both modes run with the same limit.
device =
  if System.get_env("BENCH_BACKEND", "emlx") == "emlx" do
    EMLX.set_cache_limit(1_000_000_000)
    [backend: {EMLX.Backend, device: :gpu}, defn_options: [compiler: EMLX]]
  else
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

repo =
  case name do
    "smollm2" -> "HuggingFaceTB/SmolLM2-135M-Instruct"
    "qwen3" -> "onnx-community/Qwen3-0.6B-ONNX"
  end

opts = [variant: :q4, strategy: :greedy, max_new_tokens: 256] ++ device

inputs =
  for i <- 0..(count - 1),
      do: %{messages: [%{role: "user", content: Enum.at(questions, rem(i, 16))}]}

stream =
  case mode do
    "server" ->
      {:ok, _pid} =
        Transmuter.Text.ContinuousGeneration.start_link(
          [name: Bench, source: {:hf, repo}, slots: 8, max_queue: count] ++ opts
        )

      &Transmuter.Text.ContinuousGeneration.stream(Bench, &1, stream_done: true)

    "serving" ->
      serving =
        Transmuter.Text.generation(
          {:hf, repo},
          [compile: [batch_size: 8], stream: true, stream_done: true] ++ opts
        )

      {:ok, _pid} = Nx.Serving.start_link(serving: serving, name: Bench, batch_timeout: 10)
      &Nx.Serving.batched_run(Bench, &1)
  end

# The arrival, the first chunk, the end and the new tokens of a request, in milliseconds.
request = fn input, start ->
  arrival = System.monotonic_time(:millisecond) - start

  {first, tokens} =
    input
    |> stream.()
    |> Enum.reduce({nil, 0}, fn
      {:done, %{token_summary: %{output: output}}}, {first, _tokens} -> {first, output}
      _chunk, {nil, tokens} -> {System.monotonic_time(:millisecond) - start, tokens}
      _chunk, acc -> acc
    end)

  %{
    arrival: arrival,
    first: first,
    end: System.monotonic_time(:millisecond) - start,
    tokens: tokens
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

compiles = :counters.new(3, [])
phase = :atomics.new(1, [])

:telemetry.attach(
  :compiles,
  [:transmuter, :segment, :compile, :stop],
  fn _event, measurements, _metadata, _config ->
    measured = :atomics.get(phase, 1)
    :counters.add(compiles, measured + 1, 1)
    if measured == 1, do: :counters.add(compiles, 3, measurements.duration)
  end,
  nil
)

schedule = gaps |> Enum.scan(&(&1 + &2)) |> Enum.zip(inputs)
schedule |> run_round.(System.monotonic_time(:millisecond)) |> Task.await_many(:infinity)

# The step events of the server and the calls of each process, as {pid, time, duration}.
events = :ets.new(:events, [:public, :bag])

:telemetry.attach_many(
  :bench,
  [[:transmuter, :call, :stop], [:transmuter, :continuous_generation, :step, :stop]],
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

:telemetry.detach(:bench)
ms = &(System.convert_time_unit(&1, :native, :microsecond) / 1000)
percentile = fn values, p -> values |> Enum.sort() |> Enum.at(round(p * (length(values) - 1))) end
tokens = results |> Enum.map(& &1.tokens) |> Enum.sum()
span = Enum.max(Enum.map(results, & &1.end)) - Enum.min(Enum.map(results, & &1.arrival))
first = Enum.map(results, &(&1.first - &1.arrival))
calls = :ets.lookup(events, [:transmuter, :call, :stop])
steps = :ets.lookup(events, [:transmuter, :continuous_generation, :step, :stop])

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
  "RESULT #{name} #{mode} rate #{rate}: #{f.(tokens / (span / 1000))} tokens/s, first token p50 #{percentile.(first, 0.5)} ms p95 #{percentile.(first, 0.95)} ms, between tokens p50 #{f.(percentile.(between, 0.5))} ms p95 #{f.(percentile.(between, 0.95))} ms, #{f.(tokens / length(calls))} tokens per call, #{tokens} tokens"
)

# The steps without and with requests that join, the compiled calls in each, and the mean
# rows of the steps without joins, which give one token to each running row.
step_calls = fn {_event, pid, time, duration, _metadata} ->
  Enum.count(calls, fn {_call, call_pid, at, _duration, _metadata} ->
    call_pid == pid and at > time - duration and at <= time
  end)
end

{admits, decodes} =
  Enum.split_with(steps, fn {_e, _p, _t, _d, metadata} -> metadata.admitted > 0 end)

if steps != [] do
  mean = fn values -> Enum.sum(values) / max(length(values), 1) end

  IO.puts(
    "COUNTS #{name} #{mode} rate #{rate}: #{length(decodes)} decode steps with #{Enum.sum(Enum.map(decodes, step_calls))} calls and #{f.(mean.(Enum.map(decodes, &elem(&1, 4).running)))} mean rows, #{length(admits)} admit steps with #{Enum.sum(Enum.map(admits, step_calls))} calls, #{f.(mean.(Enum.map(admits, &elem(&1, 4).admitted)))} mean admitted and #{f.(mean.(Enum.map(admits, &elem(&1, 4).running)))} mean running rows, #{length(calls)} calls in all"
  )
end

IO.puts(
  "COMPILES #{name} #{mode} rate #{rate}: #{:counters.get(compiles, 1)} in the warm round, #{:counters.get(compiles, 2)} in the measured requests"
)

# The wall time of the measured requests, split into the steps without and with joins, the
# compiles in them, and the time between steps. A compile runs inside a step.
step_time = fn steps -> steps |> Enum.map(&ms.(elem(&1, 3))) |> Enum.sum() end
compile_time = ms.(:counters.get(compiles, 3))

if steps != [] do
  IO.puts(
    "TIME #{name} #{mode} rate #{rate}: #{span} ms in all, #{f.(step_time.(decodes))} ms in decode steps, #{f.(step_time.(admits))} ms in admit steps, #{f.(compile_time)} ms of compiles, #{f.(span - step_time.(decodes) - step_time.(admits))} ms between steps"
  )
end

# The time of the steps without joins, for each capacity of the buffers.
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
