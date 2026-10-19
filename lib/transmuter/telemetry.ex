defmodule Transmuter.Telemetry do
  @moduledoc """
  Telemetry events that Transmuter emits.

  Transmuter uses the `:telemetry` library. Attach handlers to these events for logs, metrics or traces. The events show where a model spends time, and why a compiled function compiles again.

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

  ### Graph break events

  A compiled function keeps a cache of compiled segments. The cache key is the position where the segment starts, the shapes and types of the segment arguments, and the values of the static inputs. Each position keeps the `:cache_size` most recently used segments.

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
      * Metadata: `%{source: term, position: integer, shapes: list, stop: integer | :end, compiled: boolean}`

    * `[:transmuter, :break, :start]` - a break node starts to run eagerly.
      * Metadata: `%{source: term, position: integer, node: String.t(), op_type: String.t()}`

    * `[:transmuter, :break, :stop]` - the break node returns.
      * Metadata: `%{source: term, position: integer, node: String.t(), op_type: String.t()}`

  `:stop` is the position of the break node where the segment stops, or `:end`. A segment of fewer than 20 nodes after a break is not compiled, and `:compiled` is `false`. Then its nodes run eagerly on each call.

  A graph without breaks keeps its compiled function for the shapes of the inputs to `Transmuter.compile/3`. A call with these shapes does not use the cache, so it emits no cache events.

  The cache belongs to the process that calls `Transmuter.compile/3`. When that process exits, each call emits a `[:transmuter, :cache, :miss]` event and compiles its segments again.

  ## Overhead

  When no handler is attached, one event costs about 0.3 microseconds. A call of a graph without breaks emits 2 events. A call with a break emits 2 more events for each break and 1 event for each segment. Transmuter emits no event for each node.

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
        summary("transmuter.break.stop.duration", unit: {:native, :millisecond}, tags: [:op_type])
      ]
  """
end
