defmodule Transmuter.Telemetry.Logger do
  @moduledoc """
  A Logger handler for the Transmuter events that show compile work.

  It logs model loads, compiles, segment compiles and cache evictions. It does not log the events of each call, because a service makes many calls. See `Transmuter.Telemetry` for all events.

  ## Usage

  Attach the handler in the `start/2` function of your application:

      def start(_type, _args) do
        Transmuter.Telemetry.Logger.attach()

        Supervisor.start_link(children, strategy: :one_for_one)
      end

  ## Example output

      [info] [Transmuter] load completed in 412ms (Xenova/all-MiniLM-L6-v2 onnx/model.onnx, 343 nodes, opset 11)
      [info] [Transmuter] compile completed in 1.2s (Xenova/all-MiniLM-L6-v2 onnx/model.onnx, EMLX, no graph break)
      [info] [Transmuter] segment.compile completed in 85ms (kokoro.onnx, position 0, stop 209 (a value that decides a shape), compiled, shapes [...])
      [info] [Transmuter] segment.compile completed in 640ms (kokoro.onnx, position 210, stop end, compiled, shapes [...])
      [info] [Transmuter] cache.evict (kokoro.onnx, position 210, cache size 8, shapes [...])

  ## Options

      Transmuter.Telemetry.Logger.attach(
        level: :debug,
        handler_id: "my-logger"
      )

    * `:level` - the log level. The default is `:info`.
    * `:handler_id` - the handler ID. The default is `"transmuter-telemetry-logger"`.

  ## Detaching

      Transmuter.Telemetry.Logger.detach()
  """

  require Logger

  @default_handler_id "transmuter-telemetry-logger"

  @events [
    [:transmuter, :load, :stop],
    [:transmuter, :compile, :stop],
    [:transmuter, :segment, :compile, :stop],
    [:transmuter, :cache, :evict]
  ]

  @doc """
  Attaches the handler. See the module documentation for the options.
  """
  def attach(opts \\ []) do
    handler_id = Keyword.get(opts, :handler_id, @default_handler_id)
    level = Keyword.get(opts, :level, :info)
    :telemetry.attach_many(handler_id, @events, &__MODULE__.handle_event/4, %{level: level})
  end

  @doc """
  Detaches the handler. The `:handler_id` option is the same as in `attach/1`.
  """
  def detach(opts \\ []) do
    :telemetry.detach(Keyword.get(opts, :handler_id, @default_handler_id))
  end

  @doc false
  def handle_event([:transmuter, :cache, :evict], _measurements, metadata, config) do
    Logger.log(config.level, "[Transmuter] cache.evict (#{details(:evict, metadata)})")
  end

  def handle_event([:transmuter | name], %{duration: duration}, metadata, config) do
    name = name |> List.delete_at(-1) |> Enum.map_join(".", &Atom.to_string/1)

    Logger.log(
      config.level,
      "[Transmuter] #{name} completed in #{format_duration(duration)} (#{details(name, metadata)})"
    )
  end

  defp details("load", meta) do
    "#{format_source(meta.source)}, #{meta.node_count} nodes, opset #{meta.opset}"
  end

  defp details("compile", meta) do
    break = if meta.graph_break, do: "graph break", else: "no graph break"
    "#{format_source(meta.source)}, #{inspect(meta.compiler)}, #{break}"
  end

  defp details("segment.compile", meta) do
    compiled = if meta.compiled, do: "compiled", else: "eager"

    reason = if meta.reason, do: " (#{meta.reason})", else: ""

    "#{format_source(meta.source)}, position #{meta.position}, stop #{meta.stop}#{reason}, " <>
      "#{compiled}, shapes #{inspect(meta.shapes)}"
  end

  defp details(:evict, meta) do
    "#{format_source(meta.source)}, position #{meta.position}, " <>
      "cache size #{meta.cache_size}, shapes #{inspect(meta.shapes)}"
  end

  defp format_source({:hf, repo, file}), do: "#{repo} #{file}"
  defp format_source(path) when is_binary(path), do: path
  defp format_source(source), do: inspect(source)

  defp format_duration(duration) do
    ms = System.convert_time_unit(duration, :native, :millisecond)

    cond do
      ms >= 1000 -> "#{Float.round(ms / 1000, 2)}s"
      ms >= 1 -> "#{ms}ms"
      true -> "<1ms"
    end
  end
end
