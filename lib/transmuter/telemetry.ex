defmodule Transmuter.Telemetry do
  @moduledoc """
  Telemetry events that Transmuter emits.

  Transmuter uses the `:telemetry` library. Attach handlers to these events for logs, metrics or traces. The events show where a model spends time, why a compiled function compiles again, and which files come from the Hugging Face Hub.

  ## Cache events

  Transmuter keeps two caches. These events show how each cache works:

    * The cache of compiled segments of a function from `Transmuter.compile/3`. It is in memory and belongs to the process that compiles. The servings and the tasks keep their compiled shapes in this cache too, for example the image sizes and the prompt lengths. See [Graph break events](#module-graph-break-events).
      * `[:transmuter, :cache, :hit]`, `[:transmuter, :cache, :miss]` and `[:transmuter, :cache, :evict]` - a lookup of a compiled segment, and its eviction.
      * `:disk_cache` in the metadata of `[:transmuter, :segment, :compile, :stop]` - whether EXLA loaded the program of the segment from the `:cache` path of `Transmuter.compile/3`.

    * The Hub cache, the files of Hub repositories in the cache directory of `Transmuter.Hub`. See [Hub events](#module-hub-events).
      * `[:transmuter, :hub, :hit]` - a file comes from the cache.
      * `[:transmuter, :hub, :download, :*]` - a file downloads from the Hub.
      * `[:transmuter, :hub, :info, :*]` - the metadata of a repository comes from the Hub, or from the cache when the Hub gives an error.

  The compilers and their backends keep their own caches, such as the compile cache and the buffer cache of EMLX. Transmuter does not emit events for them. See [EMLX memory](#module-emlx-memory).

  ## Events

  Most events are spans from `:telemetry.span/3`. A span emits a `:start` event, and then a `:stop` event or an `:exception` event.

    * A `:start` event has the measurements `%{system_time: integer, monotonic_time: integer}`.
    * A `:stop` event has the measurements `%{duration: integer, monotonic_time: integer}`. The duration is in `:native` time units.
    * An `:exception` event has the measurements `%{duration: integer, monotonic_time: integer}`. Its metadata is the start metadata with `:kind`, `:reason` and `:stacktrace`.

  The `:source` in the metadata is the source that you gave to `Transmuter.load!/2`, for example `{:hf, "Xenova/all-MiniLM-L6-v2", "onnx/model.onnx"}` or a path. It is `nil` for a model that `Transmuter.load!/2` did not make.

  ### Load events

    * `[:transmuter, :load, :start]` - `Transmuter.load!/2` starts.
      * Metadata: `%{source: term}`

    * `[:transmuter, :load, :stop]` - `Transmuter.load!/2` returns. The duration includes the download from the Hugging Face Hub.
      * Metadata: `%{source: term, node_count: integer, opset: integer}`

  `:node_count` is the number of top-level nodes in the graph. `:opset` is the version of the ONNX standard domain.

  ### Compile events

    * `[:transmuter, :compile, :start]` - `Transmuter.compile/3` starts.
      * Metadata: `%{source: term, compiler: module}`

    * `[:transmuter, :compile, :stop]` - `Transmuter.compile/3` returns.
      * Metadata: `%{source: term, compiler: module, graph_break: boolean}`

  `:graph_break` is `true` when the graph breaks at a value that is only known when it runs. Then each call runs the graph in segments. The first call shows where the breaks are, with the `[:transmuter, :break, :*]` events.

  ### Call events

    * `[:transmuter, :call, :start]` - a call of the function from `Transmuter.compile/3` starts.
      * Metadata: `%{source: term}`

    * `[:transmuter, :call, :stop]` - the call returns.
      * Metadata: `%{source: term}`

  On EMLX, a compiled function can return its outputs before the device calculates them. Then the duration does not include the time on the device. The device time goes to the code that first reads the outputs.

  ### Run events

    * `[:transmuter, :run, :start]` - `Transmuter.run/4` starts.
      * Metadata: `%{source: term}`

    * `[:transmuter, :run, :stop]` - `Transmuter.run/4` returns.
      * Metadata: `%{source: term}`

  Inside `Nx.Defn.jit/2`, `Transmuter.run/4` traces the graph. Then the duration is the time of the trace.

  ### Continuous batching events

  A server of `Transmuter.ContinuousBatching` emits these events. All of them have `:server` in the metadata: the `:name` of the server, or its pid when it has no name. All times are in `:native` time units.

  #### The server

    * `[:transmuter, :continuous_batching, :init, :start]` - the server starts: it compiles the decoder and takes its capacity, while the worker loads the models of the task.
      * Metadata: `%{server: term, slots: integer, prefix_slots: integer, decode_steps: integer}`

    * `[:transmuter, :continuous_batching, :init, :stop]` - the server is ready, or the worker failed.
      * Metadata: the start metadata with `%{capacity: integer, live_bytes: integer, limit_bytes: integer | nil}`

  `:capacity` is the largest number of rows of the KV cache of a row (see "Device memory" in `Transmuter.ContinuousBatching`). `:live_bytes` is the device memory in use before the server took the capacity, and `:limit_bytes` the limit of the allocator, `nil` when the backend gives none.

    * `[:transmuter, :continuous_batching, :compile_steps, :start]` and `:stop` - with `compile_steps: true`, the server compiles each step shape before it starts (see "Compiles" in `Transmuter.ContinuousBatching`). The duration is the cost of this compile at the start.
      * Metadata: `%{server: term, slots: integer, capacity: integer}`

  #### Steps

    * `[:transmuter, :continuous_batching, :step, :start]` - the server starts a step: the prompt parts of the requests that join, with the next token of the running rows, or the next token of the running rows.
      * Metadata: `%{server: term, running: integer, admitted: integer, prefix_tokens: integer, prompts: integer, waiting: integer, slots: integer, capacity: integer, encoder_capacity: integer}`

    * `[:transmuter, :continuous_batching, :step, :stop]` - the step ends, and the server has sent its tokens.
      * Measurements: `%{duration: integer, monotonic_time: integer, running: integer, admitted: integer, prefix_tokens: integer, prompts: integer, waiting: integer, preparing: integer, tokens: integer, kv_capacity: integer, kv_max_capacity: integer, kv_used: integer, kv_bytes: integer}`
      * Metadata: the start metadata with `%{steps: integer, row_steps: integer}`

  `:running` is the number of rows with a request before the step, `:admitted` the number of requests that join in it, `:prefix_tokens` the tokens of their prompts whose keys and values come from the prefix cache (see "Prefix cache" in `Transmuter.ContinuousBatching`), `:prompts` the number of rows that run a part of their prompt in it, `:waiting` the number of requests in the queue after it, and `:preparing` the number of requests in the worker. In the metadata, `:capacity` is the rows of the buffers before the step, and `:encoder_capacity` the rows of the encoder outputs of the rows before it, 0 for a model without an encoder. `:steps` is the number of decode steps of the call, and `:row_steps` the steps of the rows that had not ended before each step.

  `:tokens` is the number of tokens that the step sent to the callers, so the sum of `:tokens` over time is the output rate of the server. The `:kv_*` measurements show the KV cache after the step. `:kv_capacity` is the rows of the buffers of each row, and `:kv_max_capacity` the capacity of the server. `:kv_used` is the sum of the lengths of the rows that run: the keys and values in use. The server keeps these lengths on the host, so `:kv_used` reads nothing from the device. `:kv_used` is at most `:slots` times `:kv_capacity`. `:kv_bytes` is the size of the buffers: `:slots` times `:kv_capacity` times the bytes of one row and position, a row of each cache buffer and a token of history.

  The duration of a step is the time between two tokens of a running row. When `:waiting` is above 0 and no request joins a free row, the requests wait for the running rows to end (see "The KV cache" in `Transmuter.ContinuousBatching`).

  #### Requests

  Each input of `Transmuter.ContinuousBatching.run/3` and `Transmuter.ContinuousBatching.stream/3` is a request. The input of a request can give more than one row, for example the windows of long audio. Then the request emits one `:admit`, `:first_token` and `:stop` event for each row, with the `:index` of the row. The events never carry the input or the tokens.

    * `[:transmuter, :continuous_batching, :refuse]` - the queue of the server is full, so a call fails with `Transmuter.ContinuousBatching.QueueFullError`.
      * Measurement: `%{count: integer}`, the number of requests of the call
      * Metadata: `%{server: term, reason: :queue_full, max_queue: integer}`

    * `[:transmuter, :continuous_batching, :request, :start]` - the server accepts a request, and the request goes to the worker.
      * Measurements: `%{system_time: integer, monotonic_time: integer}`
      * Metadata: `%{server: term, request: reference, stream: boolean, index: nil}`

    * `[:transmuter, :continuous_batching, :request, :admit]` - a row of the request gets a row of the server.
      * Measurements: `%{queue_time: integer, prepare_time: integer, prompt_tokens: integer, prefix_tokens: integer}`
      * Metadata: `%{server: term, request: reference, stream: boolean, index: integer}`

    * `[:transmuter, :continuous_batching, :request, :first_token]` - the server sends the first token of the row.
      * Measurement: `%{time_to_first_token: integer}`
      * Metadata: the same as `:admit`

    * `[:transmuter, :continuous_batching, :request, :stop]` - the row ends, or the request ends before it runs.
      * Measurements: `%{duration: integer, monotonic_time: integer, prompt_tokens: integer, output_tokens: integer, prefix_tokens: integer, time_to_first_token: integer, time_per_output_token: integer}`
      * Metadata: the same as `:admit`, with `%{result: :done | :cancelled}`

    * `[:transmuter, :continuous_batching, :request, :exception]` - the request fails before it runs: the worker raised on its input, or the request never fits (see "Device memory" in `Transmuter.ContinuousBatching`). The caller gets the same exception.
      * Measurements: `%{duration: integer, monotonic_time: integer}`
      * Metadata: `%{server: term, request: reference, stream: boolean, index: nil, kind: :error, reason: Exception.t(), stacktrace: []}`

  `:prepare_time` is the time from the start of the request to the end of its preparation in the worker, with the wait for the requests before it. `:queue_time` is the time from then until the row gets a row of the server. `:time_to_first_token` is the time from the start of the request to its first token. `:time_per_output_token` is the time between the first and the last token, divided by the tokens after the first one, and 0 for a row of one token. `:duration` is the time from the start of the request to its last token.

  `:result` is `:cancelled` when the caller exits, cancels or reaches its timeout. Then the measurements have only the values that exist, for example no `:time_to_first_token` for a row without a token, and no `:prompt_tokens` for a request that had not left the worker. Use `keep: &(&1.result == :done)` in a metric of the times.

  With `decode_steps` above 1, the tokens of a decode loop leave the server together at the end of the call. So the times show when the caller gets each token.

  All values of these events are on the host. The server reads the tokens of each step to the host before it sends them, so an event adds no read from the device and no wait for it. Each event costs about 0.3 microseconds without a handler. A handler runs in the process of the server, so a slow handler makes each step slower.

    * `[:transmuter, :continuous_batching, :prepare, :start]` - the worker of the server starts to prepare a request: it runs the processor and the tokenizer on its input, and the encoder, such as the vision encoder on its images (see "Encoders and images" in `Transmuter.ContinuousBatching`).
      * Metadata: `%{server: term, wait: integer}`

    * `[:transmuter, :continuous_batching, :prepare, :stop]` - the request is ready, and it goes to the queue of the server.
      * Measurements: `%{duration: integer, monotonic_time: integer, queue_wait: integer}`
      * Metadata: the same as the start

  `:wait` and `:queue_wait` are the time from the start of the request to the start of its preparation, so the time that the request waited for the requests before it in the worker.

  ### Serving events

  The `Nx.Serving` of a task, such as `Transmuter.Text.text_embedding/2`, emits these events. `:source` in the metadata is the source of the model of the task.

    * `[:transmuter, :serving, :batch, :start]` - a batch of the serving starts to run.
      * Metadata: `%{source: term, batch_key: term}`

    * `[:transmuter, :serving, :batch, :stop]` - the batch is done, and its outputs are on the host.
      * Measurements: `%{duration: integer, monotonic_time: integer, batch_size: integer, padded_size: integer}`
      * Metadata: `%{source: term, batch_key: term}`

    * `[:transmuter, :serving, :request, :stop]` - a call of `Nx.Serving.run/2` or `Nx.Serving.batched_run/2` returns its result. For a serving with `stream: true`, the event comes when the stream ends or halts.
      * Measurements: `%{duration: integer, monotonic_time: integer, inputs: integer, output_tokens: integer, time_to_first_token: integer}`
      * Metadata: `%{source: term, stream: boolean}`

  `:batch_size` is the number of rows of the batch, and `:padded_size` the batch size of the compiled model, so `:batch_size` divided by `:padded_size` shows how full the batches are. `:batch_key` is the key of the batch, such as the sequence length. The batch function copies its outputs to the host, so the duration of a batch includes the time on the device.

  The duration of a request is the time from the start of the client preprocessing to the result, with the wait for a batch. `:inputs` is the number of inputs of the call. A generation serving adds `:output_tokens`, the tokens of all rows of the call. A stream has `:output_tokens` and `:time_to_first_token`, and no `:inputs`.

  `Nx.Serving` emits its own events too: `[:nx, :serving, :execute, :*]`, `[:nx, :serving, :preprocessing, :*]` and `[:nx, :serving, :postprocessing, :*]`. Their `:module` is `Transmuter.Serving` for all tasks, and the metadata of the preprocessing events holds the input of the call, such as the text of a user. Do not log that metadata.

  ### Graph break events

  A compiled function keeps a cache of compiled segments. The cache key is the position where the segment starts, the shapes and types of the segment arguments, and the values of the static values that the segment reads, such as static inputs. Each position keeps the `:cache_size` most recently used segments.

  The `:position` in the metadata is the index of a node in the sorted graph. `:shapes` is a sorted list of `{name, {shape, type}}` for the segment arguments. A sequence argument has a list of `{shape, type}`.

    * `[:transmuter, :cache, :hit]` - a call finds a compiled segment in the cache.
      * Measurement: `%{count: 1}`
      * Metadata: `%{source: term, position: integer, shapes: list}`

    * `[:transmuter, :cache, :miss]` - a call does not find a compiled segment in the cache. A `[:transmuter, :segment, :compile, :*]` span follows.
      * Measurement: `%{count: 1}`
      * Metadata: `%{source: term, position: integer, shapes: list}`

    * `[:transmuter, :cache, :evict]` - a new segment replaces the least recently used segment at a full position.
      * Measurement: `%{count: 1}`
      * Metadata: `%{source: term, position: integer, shapes: list, cache_size: integer}`. `:shapes` are the shapes of the segment that the cache removed.

    * `[:transmuter, :segment, :compile, :start]` - the tracing and the compilation of a segment start.
      * Metadata: `%{source: term, position: integer, shapes: list}`

    * `[:transmuter, :segment, :compile, :stop]` - the segment is ready.
      * Metadata: `%{source: term, position: integer, shapes: list, stop: integer | :end, compiled: boolean, disk_cache: :hit | :miss | nil, reason: String.t() | nil, host_values: [String.t()]}`

    * `[:transmuter, :break, :start]` - a break node starts to run eagerly.
      * Metadata: `%{source: term, position: integer, node: String.t(), op_type: String.t()}`

    * `[:transmuter, :break, :stop]` - the break node returns.
      * Metadata: `%{source: term, position: integer, node: String.t(), op_type: String.t()}`

  `:stop` is the position of the break node where the segment stops, or `:end`. `:reason` describes the value that the break node needs, such as `"the If condition"`. It is `nil` when `:stop` is `:end`. A segment of fewer than 20 nodes after a break is not compiled, and `:compiled` is `false`. Then its nodes run eagerly on each call.

  `:host_values` are the names of new values that a node of the segment reads on the host. An example is the shape of a `Reshape` that comes from the positions of a `NonZero`. From then on, these values enter the segment as static values, so their values are in the cache key of this segment. The keys of other segments do not change. The list is empty when the trace found no new name. The `:shapes` of the stop event are those of the arguments of the compiled segment. They can have fewer values than the `:shapes` of the start event. `:compiled` is `false` when a trace found a compiled segment in the cache.

  A graph without breaks keeps its compiled function for the shapes of the inputs to `Transmuter.compile/3`. A call with these shapes does not use the cache, so it emits no cache events.

  The cache belongs to the process that calls `Transmuter.compile/3`. When that process exits, each call emits a `[:transmuter, :cache, :miss]` event and compiles its segments again.

  With EXLA and a `:cache` path, each compiled segment has its own file. `:disk_cache` is `:hit` when the file of the segment exists before the compile, so EXLA loads the program from it, and `:miss` when the file does not exist, so EXLA compiles the program and writes the file. It is `nil` without a `:cache` path and when `:compiled` is `false`. EXLA does not use a file of another EXLA version or of another configuration. It logs a warning and compiles the program again, but the event shows `:hit`.

  ### Hub events

  These events come from `Transmuter.Hub` for a Hub source. A local directory has no cache, so it emits no Hub events. `Transmuter.load!/2` and the tasks fetch each file of a model, also the configuration files and the external data files, so each load emits a hit or a download for each file.

  The `:source` in the metadata is `{:hf, repo, file}` for a file and `{:hf, repo}` for the metadata of a repository. `:revision` is the `:revision` option, `"main"` by default.

    * `[:transmuter, :hub, :hit]` - the file is in the cache, and it is current.
      * Measurements: `%{count: 1, bytes: integer}`
      * Metadata: `%{source: {:hf, repo, file}, revision: String.t(), check: :etag | :hash | :none | :offline}`

    * `[:transmuter, :hub, :download, :start]` - the file is not in the cache, or its ETag changed, and it starts to download.
      * Metadata: `%{source: {:hf, repo, file}, revision: String.t()}`

    * `[:transmuter, :hub, :download, :stop]` - the download is complete, and the file is in the cache.
      * Measurements: `%{duration: integer, monotonic_time: integer, bytes: integer}`
      * Metadata: `%{source: {:hf, repo, file}, revision: String.t()}`

    * `[:transmuter, :hub, :info, :start]` - a request for the metadata of a repository goes to the Hub API.
      * Metadata: `%{source: {:hf, repo}, revision: String.t()}`

    * `[:transmuter, :hub, :info, :stop]` - the metadata is ready.
      * Metadata: `%{source: {:hf, repo}, revision: String.t(), result: :fetched | :saved}`

  `:bytes` is the size of the file in bytes. `:check` tells how the cache found that the file is current:

    * `:etag` - the saved ETag of the file is the ETag in the metadata of the repository.
    * `:hash` - the file has no saved ETag, for example a file of an earlier version of Transmuter. Transmuter read the whole file to compute its ETag, and saved it.
    * `:none` - the metadata of the repository gives no ETag for the file, so the file is used without a check.
    * `:offline` - the Hub is offline (see `Transmuter.Hub`), so the file is used without a check.

  The metadata of a repository is in the cache too. A request goes to the Hub when the metadata is not in the cache. `Transmuter.Repo.load!/2` reads the list of files again, so each load sends a request, except when the `:revision` is a commit hash of 40 hexadecimal characters. The files of a commit do not change, so its saved metadata is used. A read of the saved metadata without a request emits no event. `:result` is `:fetched` when the metadata comes from the Hub, and `:saved` when the Hub gives an error or cannot be reached and the saved metadata is used. Then the files can be older than the files on the Hub. When there is no saved metadata, the error gives an `:exception` event.

  ## Overhead

  When no handler is attached, one event costs about 0.3 microseconds. A call of a graph without breaks emits 2 events. A call with a break emits 2 more events for each break and 1 event for each segment. Transmuter emits no event for each node. The Hub events come only when a model loads, 1 or 2 for each file.

  ## EMLX memory

  MLX keeps freed buffers in a cache, which can grow to many gigabytes when the shapes change from call to call. EMLX owns this cache, and Transmuter emits no event for it. EMLX emits `[:emlx, :memory, :stats]` with `:active_memory`, `:peak_memory` and `:cache_memory` in bytes each time that you call `EMLX.Telemetry.memory_stats/0`. Call it from a periodic task, or from a handler of `[:transmuter, :call, :stop]`:

      defmodule MyApp.EMLXMemory do
        def attach do
          event = [:transmuter, :call, :stop]
          :telemetry.attach("my-app-emlx-memory", event, &__MODULE__.handle_event/4, nil)
        end

        def handle_event(_event, _measurements, _metadata, _config),
          do: EMLX.Telemetry.memory_stats()
      end

  On EMLX, a call can return before the device calculates its outputs, so the memory at the `:stop` event is not always the peak of that call. `EMLX.set_cache_limit/1` sets the largest size of the cache. `EMLX.reset_peak_memory/0` resets the peak for all processes, so do not call it in a service where other code reads the peak.

  ## Logger

  `Transmuter.Telemetry.Logger` logs the events that show compile work:

      Transmuter.Telemetry.Logger.attach()

  ## Custom handler

      defmodule MyApp.TransmuterMetrics do
        require Logger

        def setup do
          events = [
            [:transmuter, :call, :stop],
            [:transmuter, :cache, :miss]
          ]

          :telemetry.attach_many("my-app-transmuter", events, &__MODULE__.handle_event/4, nil)
        end

        def handle_event([:transmuter, :call, :stop], %{duration: duration}, metadata, _config) do
          ms = System.convert_time_unit(duration, :native, :millisecond)
          Logger.debug("\#{inspect(metadata.source)} call in \#{ms} ms")
        end

        def handle_event([:transmuter, :cache, :miss], _measurements, metadata, _config) do
          Logger.info("\#{inspect(metadata.source)} compiles at node \#{metadata.position}")
        end
      end

  ## Metrics

  The events work with `telemetry_metrics`:

      import Telemetry.Metrics

      [
        summary("transmuter.call.stop.duration", unit: {:native, :millisecond}),
        counter("transmuter.cache.miss.count", tags: [:position]),
        counter("transmuter.cache.evict.count", tags: [:position]),
        counter("transmuter.hub.hit.count"),
        sum("transmuter.hub.download.stop.bytes", unit: {:byte, :megabyte}),
        summary("transmuter.break.stop.duration", unit: {:native, :millisecond}, tags: [:op_type])
      ]
  """
end
