# Telemetry in production

Transmuter emits `:telemetry` events when it loads, compiles and runs a model. This guide shows the events that an application needs to run a model in production, and how to turn them into metrics. `Transmuter.Telemetry` lists each event with all its measurements and metadata.

## The events for an application

| Question | Event | Values |
|---|---|---|
| How long until the first token? | `[:transmuter, :continuous_batching, :request, :first_token]` and `:stop` | `:time_to_first_token` |
| How fast does a row give tokens? | `[:transmuter, :continuous_batching, :request, :stop]` | `:time_per_output_token`, `:output_tokens` |
| How many tokens does the server give? | `[:transmuter, :continuous_batching, :step, :stop]` | `:tokens` |
| How long do requests wait? | `[:transmuter, :continuous_batching, :request, :admit]` | `:queue_time`, `:prepare_time` |
| How full is the server? | `[:transmuter, :continuous_batching, :step, :stop]` | `:running`, `:waiting`, `:preparing` |
| How full is the KV cache? | `[:transmuter, :continuous_batching, :step, :stop]` | `:kv_used`, `:kv_capacity`, `:kv_bytes` |
| Does the prefix cache work? | `[:transmuter, :continuous_batching, :request, :admit]` | `:prefix_tokens`, `:prompt_tokens` |
| Which requests fail? | `[:transmuter, :continuous_batching, :refuse]` and `[:transmuter, :continuous_batching, :request, :exception]` | `:reason` |
| How long does the server take to start? | `[:transmuter, :continuous_batching, :init, :stop]` and `[:transmuter, :continuous_batching, :compile_steps, :stop]` | `:duration` |
| How full are the batches of a serving? | `[:transmuter, :serving, :batch, :stop]` | `:batch_size`, `:padded_size` |
| How long does a call of a serving take? | `[:transmuter, :serving, :request, :stop]` | `:duration` |
| Does a model compile again? | `[:transmuter, :cache, :miss]` and `[:transmuter, :segment, :compile, :stop]` | `:position`, `:duration` |

All times are in `:native` time units. Telemetry.Metrics converts them with the `:unit` option, for example `unit: {:native, :millisecond}`.

## Rules for handlers

A handler runs in the process that emits the event. The events of `Transmuter.ContinuousBatching` come from the process of the server, so a slow handler makes each step slower, and all rows wait for it.

  * Keep handlers short. Update a counter or an ETS table, or send a message to another process.
  * Do not call the device or the model in a handler.
  * Do not log the metadata of the `[:nx, :serving, :preprocessing, :*]` events of `Nx.Serving`. It holds the input of the call, such as the text of a user. The events of Transmuter never hold the input or the tokens.

The events themselves cost about 0.3 microseconds each without a handler. Their values are on the host, so they add no read from the device and no wait for it.

## Metrics

`telemetry_metrics` defines metrics from events, and a reporter sends them to a service. Add `{:telemetry_metrics, "~> 1.0"}` to the dependencies of your application, and define the metrics in one module:

```elixir
defmodule MyApp.Telemetry do
  import Telemetry.Metrics

  def metrics do
    server = by_server([:server])
    done = [keep: &(&1.result == :done)]
    ms = [unit: {:native, :millisecond}]

    [
      # Each request of a ContinuousBatching server
      distribution("transmuter.continuous_batching.request.stop.time_to_first_token",
        server ++ done ++ ms ++ [reporter_options: [buckets: [50, 100, 250, 500, 1000, 2500]]]
      ),
      distribution("transmuter.continuous_batching.request.stop.time_per_output_token",
        server ++ done ++ ms ++ [reporter_options: [buckets: [5, 10, 20, 40, 80, 160]]]
      ),
      distribution("transmuter.continuous_batching.request.admit.queue_time",
        server ++ ms ++ [reporter_options: [buckets: [10, 50, 250, 1000, 5000]]]
      ),
      sum("transmuter.continuous_batching.request.stop.output_tokens", server ++ done),
      sum("transmuter.continuous_batching.request.admit.prompt_tokens", server),
      sum("transmuter.continuous_batching.request.admit.prefix_tokens", server),
      counter("transmuter.continuous_batching.request.stop.duration",
        by_server([:server, :result])
      ),
      counter("transmuter.continuous_batching.request.exception.duration", server),
      sum("transmuter.continuous_batching.refuse.count", by_server([:server, :reason])),

      # Each step of a ContinuousBatching server
      sum("transmuter.continuous_batching.step.stop.tokens", server),
      last_value("transmuter.continuous_batching.step.stop.running", server),
      last_value("transmuter.continuous_batching.step.stop.waiting", server),
      last_value("transmuter.continuous_batching.step.stop.kv_used", server),
      last_value("transmuter.continuous_batching.step.stop.kv_capacity", server),
      last_value("transmuter.continuous_batching.step.stop.kv_bytes",
        server ++ [unit: {:byte, :megabyte}]
      ),

      # The start of a ContinuousBatching server
      last_value("transmuter.continuous_batching.init.stop.duration", server ++ ms),
      last_value("transmuter.continuous_batching.compile_steps.stop.duration", server ++ ms),

      # The servings of the tasks
      distribution("transmuter.serving.request.stop.duration",
        ms ++ [tags: [:source], tag_values: &%{&1 | source: inspect(&1.source)},
               reporter_options: [buckets: [5, 10, 25, 50, 100, 250, 1000]]]
      ),
      summary("transmuter.serving.batch.stop.batch_size",
        tags: [:source], tag_values: &%{&1 | source: inspect(&1.source)}
      ),

      # Compiles at run time
      counter("transmuter.cache.miss.count"),
      summary("transmuter.segment.compile.stop.duration", ms)
    ]
  end

  # A server without a name has a pid, so the tag is its text.
  defp by_server(tags), do: [tags: tags, tag_values: &%{&1 | server: inspect(&1.server)}]
end
```

`keep: &(&1.result == :done)` leaves out the rows that the caller cancelled, because their times stop early. The tag values change the `:server` and the `:source` to text, because a server without a name has a pid and a source is a tuple.

The output rate of a server is the rate of the sum of `:tokens`. The use of the KV cache is `:kv_used` divided by `:slots` times `:kv_capacity`. The prefix cache saves the fraction `:prefix_tokens` divided by `:prompt_tokens` of the prompt tokens.

## Prometheus

`telemetry_metrics_prometheus` gives the metrics at an HTTP endpoint for Prometheus. Transmuter does not depend on it. Add it to the dependencies of your application:

```elixir
{:telemetry_metrics_prometheus, "~> 1.1"}
```

Start the reporter in the supervision tree of your application, before the servers of Transmuter:

```elixir
children = [
  {TelemetryMetricsPrometheus, metrics: MyApp.Telemetry.metrics()},
  {Transmuter.ContinuousBatching,
   name: MyApp.Chat, source: {:hf, "onnx-community/Qwen3-0.6B-ONNX"}, variant: :q4, slots: 8}
]

Supervisor.start_link(children, strategy: :one_for_one)
```

The reporter serves the metrics at `http://localhost:9568/metrics`. It changes the dots of a metric name to underscores. These queries show the main values of a server:

```
# Output tokens per second
rate(transmuter_continuous_batching_step_stop_tokens[1m])

# The time to the first token of 95% of the requests, in milliseconds
histogram_quantile(0.95, rate(transmuter_continuous_batching_request_stop_time_to_first_token_bucket[5m]))

# The use of the KV cache, in keys and values
transmuter_continuous_batching_step_stop_kv_used

# Requests that wait for a row
transmuter_continuous_batching_step_stop_waiting
```

## Device memory

On EMLX, MLX keeps freed buffers in a cache. EMLX emits `[:emlx, :memory, :stats]` with `:active_memory`, `:peak_memory` and `:cache_memory` in bytes each time that you call `EMLX.Telemetry.memory_stats/0`. Call it from `:telemetry_poller`, for example each 10 seconds:

```elixir
{:telemetry_poller,
 measurements: [{EMLX.Telemetry, :memory_stats, []}], period: :timer.seconds(10)}
```

Then add metrics for the event:

```elixir
last_value("emlx.memory.stats.active_memory", unit: {:byte, :megabyte}),
last_value("emlx.memory.stats.cache_memory", unit: {:byte, :megabyte})
```

`EMLX.set_cache_limit/1` sets the largest size of the cache. See "Set the MLX cache limit" in the README.

## Compile work

A compile at run time makes a request slow. `Transmuter.Telemetry.Logger` logs each compile, with the shapes and the reason:

```elixir
Transmuter.Telemetry.Logger.attach()
```

With `compile_steps: true`, a `Transmuter.ContinuousBatching` server compiles each step shape before it starts, so it does not compile while it serves requests. The `[:transmuter, :continuous_batching, :compile_steps, :stop]` event gives the time of that compile.
