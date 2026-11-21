# A serving benchmark with a ShareGPT mix of prompts, at saturation, with one client for two
# servers: a Transmuter.ContinuousBatching server of Qwen3-0.6B in this process, or an
# OpenAI-compatible server such as vLLM over HTTP. Both get the same prompt strings, rendered
# by the chat template of onnx-community/Qwen3-0.6B-ONNX with enable_thinking: false, and the
# same settings: greedy selection, up to MAX_NEW new tokens (default 512), stop at eos.
#
#   prepare: takes `count` conversations of a ShareGPT JSON file (default 512, seed 0) whose
#   first turn is a user message, rendered to 20 to `max_prompt` tokens (default 512), and 64
#   more for the warm round, and writes the prompts and their token counts to a JSON file.
#
#   transmuter: starts the server (VARIANT, default fp16, `slots`, default 48, compile_steps,
#   DEVICE_MEMORY as :device_memory and DECODE_STEPS as :decode_steps when they are set) on the
#   GPU of EMLX, or of EXLA with CUDA when BENCH_BACKEND is exla, and runs the requests. It also
#   prints the calls of the server in the measured round (CALLS): the calls with a prompt part
#   against the decode calls, their time, rows and capacities.
#
#   openai: sends each request to `url`/v1/completions with the rendered prompt and streams the
#   answer. It also samples the running and waiting requests of `url`/metrics (vLLM) every
#   500 ms in the measured round (CONCURRENCY).
#
# The client sends a warm round of the last 64 requests, then all other requests at the same
# time, so that the prefix cache of vLLM has none of the measured prompts.
# It prints the output tokens per second (all new tokens over the time from the first send to
# the last end), the requests per second, the time to the first chunk (p50 and p99), the time
# between chunks of a stream (p50 and p99), the time per output token of each request after its
# first chunk (p50 and p99), the new tokens in all, the answers with no text and the failed
# requests. A chunk of the server can hold more than one token, and a chunk of vLLM holds
# about one. An answer with no text, such as one eos token, counts as done and has no chunks.
#
# Usage:
#   mix run bench/serving_mix.exs prepare <sharegpt.json> <requests.json> [count] [max_prompt]
#   mix run bench/serving_mix.exs transmuter <requests.json> [slots]
#   mix run bench/serving_mix.exs openai <requests.json> <url> <model>
#
# For example, on CUDA, with ShareGPT_V3_unfiltered_cleaned_split.json of
# anon8231489123/ShareGPT_Vicuna_unfiltered on the Hugging Face Hub:
#   mix run bench/serving_mix.exs prepare ShareGPT_V3_unfiltered_cleaned_split.json requests.json
#   MAX_NEW=256 BENCH_BACKEND=exla DEVICE_MEMORY=11291066368 DECODE_STEPS=8 \
#     mix run bench/serving_mix.exs transmuter requests.json 16
#   vllm serve Qwen/Qwen3-0.6B --port 8011 --max-num-seqs 16 &
#   MAX_NEW=256 mix run bench/serving_mix.exs openai requests.json http://127.0.0.1:8011 Qwen/Qwen3-0.6B
Logger.configure(level: :warning)

source = {:hf, "onnx-community/Qwen3-0.6B-ONNX"}
max_new = String.to_integer(System.get_env("MAX_NEW", "512"))
template_options = [enable_thinking: false]
warm = 64

percentile = fn values, p -> values |> Enum.sort() |> Enum.at(round(p * (length(values) - 1))) end
f = &:erlang.float_to_binary(&1 / 1, decimals: 1)

# The arrival, the times of the chunks, the end and the new tokens of each request, in
# microseconds from the start of the round, then the metrics of the round.
report = fn label, results, span ->
  {ok, failed} = Enum.split_with(results, &match?({:ok, _}, &1))
  all = Enum.map(ok, &elem(&1, 1))
  {done, empty} = Enum.split_with(all, &(&1.chunks != []))
  tokens = all |> Enum.map(& &1.tokens) |> Enum.sum()
  first = Enum.map(done, &((hd(&1.chunks) - &1.arrival) / 1000))

  gaps =
    Enum.flat_map(done, fn r ->
      r.chunks |> Enum.chunk_every(2, 1, :discard) |> Enum.map(fn [a, b] -> (b - a) / 1000 end)
    end)

  per_token =
    for r <- done, r.tokens > 1, do: (List.last(r.chunks) - hd(r.chunks)) / 1000 / (r.tokens - 1)

  IO.puts(
    "RESULT #{label}: #{f.(tokens / (span / 1_000_000))} output tokens/s, " <>
      "#{f.(length(all) / (span / 1_000_000))} requests/s, " <>
      "first chunk p50 #{f.(percentile.(first, 0.5))} ms p99 #{f.(percentile.(first, 0.99))} ms, " <>
      "between chunks p50 #{f.(percentile.(gaps, 0.5))} ms p99 #{f.(percentile.(gaps, 0.99))} ms, " <>
      "per output token p50 #{f.(percentile.(per_token, 0.5))} ms p99 #{f.(percentile.(per_token, 0.99))} ms, " <>
      "#{tokens} new tokens, #{length(all)} requests, #{length(empty)} with no text, #{length(failed)} failed, span #{f.(span / 1000)} ms"
  )

  for failure <- Enum.take(failed, 3),
      do: IO.puts("FAILED #{label}: #{inspect(failure, limit: 5)}")
end

# Runs `requests` at the same time with `send`, a function of a request and the start time that
# gives {:ok, result} or {:error, reason}.
round = fn requests, send ->
  start = System.monotonic_time(:microsecond)

  results =
    requests
    |> Task.async_stream(&send.(&1, start), max_concurrency: length(requests), timeout: :infinity)
    |> Enum.map(fn {:ok, result} -> result end)

  ends = for {:ok, r} <- results, do: r.end
  {results, Enum.max(ends, fn -> 0 end)}
end

now = fn start -> System.monotonic_time(:microsecond) - start end

case System.argv() do
  ["prepare", sharegpt, out | rest] ->
    count = String.to_integer(Enum.at(rest, 0, "512"))
    max_prompt = String.to_integer(Enum.at(rest, 1, "512"))

    repo = Transmuter.Repo.load!(source, variant: :fp16)
    {:ok, path} = Transmuter.Hub.fetch(source, "tokenizer.json", [])
    {:ok, tokenizer} = Tokenizers.Tokenizer.from_file(path)
    :rand.seed(:exsss, 0)

    requests =
      sharegpt
      |> File.read!()
      |> :json.decode()
      |> Enum.flat_map(fn
        %{"conversations" => [%{"from" => "human", "value" => text} | _]} -> [text]
        _other -> []
      end)
      |> Enum.shuffle()
      |> Stream.map(fn text ->
        prompt =
          Transmuter.ChatTemplate.render(repo, [%{role: "user", content: text}],
            template_options: template_options
          )

        {:ok, encoding} =
          Tokenizers.Tokenizer.encode(tokenizer, prompt, add_special_tokens: false)

        %{
          text: text,
          prompt: prompt,
          prompt_tokens: length(Tokenizers.Encoding.get_ids(encoding))
        }
      end)
      |> Stream.filter(&(&1.prompt_tokens >= 20 and &1.prompt_tokens <= max_prompt))
      |> Enum.take(count + warm)

    File.write!(out, :json.encode(requests))
    lengths = Enum.map(requests, & &1.prompt_tokens)

    IO.puts(
      "PREPARED #{length(requests)} requests, prompt tokens p50 #{percentile.(lengths, 0.5)} p99 #{percentile.(lengths, 0.99)} max #{Enum.max(lengths)}"
    )

  ["transmuter", file | rest] ->
    slots = String.to_integer(List.first(rest) || "48")
    requests = file |> File.read!() |> :json.decode()

    device =
      if System.get_env("BENCH_BACKEND", "emlx") == "emlx" do
        EMLX.set_cache_limit(1_000_000_000)
        [backend: {EMLX.Backend, device: :gpu}, defn_options: [compiler: EMLX]]
      else
        Application.put_env(:exla, :clients, cuda: [platform: :cuda, preallocate: false])
        [backend: {EXLA.Backend, client: :cuda}, defn_options: [compiler: EXLA, client: :cuda]]
      end

    env = fn name, key ->
      if value = System.get_env(name), do: [{key, String.to_integer(value)}], else: []
    end

    memory = env.("DEVICE_MEMORY", :device_memory) ++ env.("DECODE_STEPS", :decode_steps)

    {:ok, _pid} =
      Transmuter.ContinuousBatching.start_link(
        [
          name: Bench,
          source: source,
          slots: slots,
          max_queue: length(requests),
          compile_steps: true
        ] ++
          [variant: String.to_atom(System.get_env("VARIANT", "fp16")), strategy: :greedy] ++
          [max_new_tokens: max_new, template_options: template_options] ++ memory ++ device
      )

    IO.puts("CAPACITY #{:sys.get_state(Bench).ctx.max_capacity} rows for #{slots} slots")

    send = fn request, start ->
      arrival = now.(start)

      try do
        input = %{messages: [%{role: "user", content: request["text"]}]}

        {chunks, tokens} =
          Bench
          |> Transmuter.ContinuousBatching.stream(input, stream_done: true)
          |> Enum.reduce({[], 0}, fn
            {:done, %{token_summary: %{output: output}}}, {chunks, _} -> {chunks, output}
            _chunk, {chunks, tokens} -> {[now.(start) | chunks], tokens}
          end)

        {:ok, %{arrival: arrival, chunks: Enum.reverse(chunks), end: now.(start), tokens: tokens}}
      rescue
        error -> {:error, Exception.message(error)}
      end
    end

    {measured, warm_requests} = Enum.split(requests, -warm)
    round.(warm_requests, send)
    calls = :ets.new(:calls, [:public, :bag])

    :telemetry.attach(
      :calls,
      [:transmuter, :continuous_batching, :step, :stop],
      fn _event, %{duration: duration}, meta, _ -> :ets.insert(calls, {duration, meta}) end,
      nil
    )

    {results, span} = round.(measured, send)
    :telemetry.detach(:calls)
    label = "transmuter #{System.get_env("VARIANT", "fp16")} #{slots} slots"
    report.(label, results, span)

    # The calls with a prompt part (joins, or the prompts of a new group) against the decode
    # calls, which give each running row up to :decode_steps tokens. A row that only decodes in
    # a prompt call waits for the prompt parts of that call.
    ms = &(System.convert_time_unit(&1, :native, :microsecond) / 1000)
    {prompted, decodes} = :ets.tab2list(calls) |> Enum.split_with(fn {_d, m} -> m.prompts > 0 end)
    total = fn list -> list |> Enum.map(&ms.(elem(&1, 0))) |> Enum.sum() end

    mean = fn list, fun ->
      if list == [], do: 0, else: Enum.sum(Enum.map(list, fun)) / length(list)
    end

    durations = Enum.map(prompted, &ms.(elem(&1, 0)))

    row_steps =
      decodes |> Enum.map(&elem(&1, 1)[:row_steps]) |> Enum.reject(&is_nil/1) |> Enum.sum()

    steps = decodes |> Enum.map(&(elem(&1, 1)[:steps] || 1)) |> Enum.sum()

    bucket = fn
      d when d < 20 -> "<20ms"
      d when d < 50 -> "20-50ms"
      d when d < 100 -> "50-100ms"
      _d -> ">=100ms"
    end

    buckets = Enum.frequencies_by(durations, bucket)

    IO.puts(
      "CALLS #{label}: span #{f.(span / 1000)} ms, " <>
        "#{length(prompted)} prompt calls #{f.(total.(prompted))} ms (p50 #{f.(percentile.(durations, 0.5))} ms, p90 #{f.(percentile.(durations, 0.9))} ms, max #{f.(Enum.max(durations, fn -> 0 end))} ms, #{inspect(buckets)}), " <>
        "mean rows with a prompt part #{f.(mean.(prompted, &elem(&1, 1).prompts))}, admitted #{f.(mean.(prompted, &elem(&1, 1).admitted))}, running #{f.(mean.(prompted, &elem(&1, 1).running))}; " <>
        "#{length(decodes)} decode calls #{f.(total.(decodes))} ms, #{steps} steps (#{f.(steps / max(length(decodes), 1))} per call), mean running rows #{f.(if steps > 0, do: row_steps / steps, else: 0)}"
    )

    capacities = Enum.group_by(:ets.tab2list(calls), &elem(&1, 1).capacity, &ms.(elem(&1, 0)))

    for {capacity, list} <- Enum.sort(capacities) do
      IO.puts(
        "CALLS #{label}: capacity #{capacity} rows: #{length(list)} calls, #{f.(Enum.sum(list))} ms"
      )
    end

  ["openai", file, url, model] ->
    requests = file |> File.read!() |> :json.decode()
    {:ok, _} = Finch.start_link(name: BenchFinch, pools: %{default: [size: length(requests) + 8]})

    # Reads the server-sent events of a stream: a chunk with text is a chunk of the answer, and
    # the last event has the usage.
    events = fn data, {buffer, chunks, tokens}, start ->
      lines = String.split(buffer <> data, "\n")
      {complete, [rest]} = Enum.split(lines, -1)

      Enum.reduce(complete, {rest, chunks, tokens}, fn
        "data: [DONE]", acc ->
          acc

        "data: " <> json, {buffer, chunks, tokens} ->
          event = :json.decode(json)
          tokens = get_in(event, ["usage", "completion_tokens"]) || tokens

          chunks =
            if Enum.any?(event["choices"] || [], &(&1["text"] not in [nil, ""])),
              do: [now.(start) | chunks],
              else: chunks

          {buffer, chunks, tokens}

        _line, acc ->
          acc
      end)
    end

    send = fn request, start ->
      arrival = now.(start)

      body = %{
        model: model,
        prompt: request["prompt"],
        max_tokens: max_new,
        temperature: 0,
        stream: true,
        stream_options: %{include_usage: true}
      }

      into = fn {:data, data}, {req, resp} ->
        {:cont,
         {req, Req.Response.update_private(resp, :events, {"", [], 0}, &events.(data, &1, start))}}
      end

      case Req.post(url <> "/v1/completions",
             json: body,
             finch: BenchFinch,
             receive_timeout: :infinity,
             retry: false,
             into: into
           ) do
        {:ok, %{status: 200} = resp} ->
          {_buffer, chunks, tokens} = Req.Response.get_private(resp, :events)

          {:ok,
           %{arrival: arrival, chunks: Enum.reverse(chunks), end: now.(start), tokens: tokens}}

        {:ok, resp} ->
          {:error, "status #{resp.status}"}

        {:error, error} ->
          {:error, Exception.message(error)}
      end
    end

    {measured, warm_requests} = Enum.split(requests, -warm)
    round.(warm_requests, send)

    # The running and waiting requests of the server, from its Prometheus metrics.
    sampler =
      Task.async(fn ->
        Stream.repeatedly(fn ->
          Process.sleep(500)

          case Req.get(url <> "/metrics", finch: BenchFinch, retry: false) do
            {:ok, %{status: 200, body: body}} ->
              read = fn name ->
                case Regex.run(~r/^vllm:#{name}\{[^}]*\} ([0-9.e+]+)/m, body) do
                  [_, value] -> value |> Float.parse() |> elem(0)
                  nil -> nil
                end
              end

              {read.("num_requests_running"), read.("num_requests_waiting")}

            _other ->
              {nil, nil}
          end
        end)
        |> Enum.reduce_while([], fn sample, samples ->
          receive do
            :stop -> {:halt, [sample | samples]}
          after
            0 -> {:cont, [sample | samples]}
          end
        end)
      end)

    {results, span} = round.(measured, send)
    send(sampler.pid, :stop)

    samples =
      for {running, waiting} <- Task.await(sampler, :infinity),
          running != nil,
          do: {running, waiting}

    report.("openai #{model}", results, span)

    if samples != [] do
      running = Enum.map(samples, &elem(&1, 0))
      waiting = Enum.map(samples, &elem(&1, 1))

      IO.puts(
        "CONCURRENCY openai #{model}: #{length(samples)} samples, running mean #{f.(Enum.sum(running) / length(running))} max #{f.(Enum.max(running))}, waiting mean #{f.(Enum.sum(waiting) / length(waiting))} max #{f.(Enum.max(waiting))}"
      )
    end
end
