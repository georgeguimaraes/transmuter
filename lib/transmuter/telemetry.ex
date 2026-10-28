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

  ### Continuous generation events

    * `[:transmuter, :continuous_generation, :step, :start]` - a server of `Transmuter.ContinuousGeneration` starts a step: the prompt parts of the requests that join, with the next token of the running rows, or the next token of the running rows.
      * Metadata: `%{running: integer, admitted: integer, waiting: integer, slots: integer, capacity: integer}`

    * `[:transmuter, :continuous_generation, :step, :stop]` - the step ends, and the server has sent its tokens.
      * Metadata: the same as the start

  `:running` is the number of rows with a request before the step, `:admitted` the number of requests that join in it, `:waiting` the number of requests in the queue after it, and `:capacity` the rows of the buffers before it. The duration of a step is the time between two tokens of a running row. When `:waiting` is above 0 and no request joins a free row, the requests wait for the running rows to end (see "The KV cache" in `Transmuter.ContinuousGeneration`).

    * `[:transmuter, :continuous_generation, :prepare, :start]` - the worker of a server of `Transmuter.ContinuousGeneration` starts to prepare a request: it runs the processor and the tokenizer on its input, and the vision encoder on its images (see "Images" in `Transmuter.ContinuousGeneration`).
      * Metadata: `%{wait: integer}`

    * `[:transmuter, :continuous_generation, :prepare, :stop]` - the request is ready, and it goes to the queue of the server.
      * Metadata: the same as the start

  `:wait` is the time from the call of the request to the start, in `:native` time units, so the time that the request waited for the requests before it in the worker.

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
