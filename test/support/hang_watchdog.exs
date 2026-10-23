defmodule Transmuter.HangWatchdog do
  @moduledoc """
  Records the state of an EMLX call that does not return, for the intermittent hang of the
  EMLX worker.

  test_helper.exs starts the watchdog when the environment variable HANG_DIR is set, and the
  EMLX job of CI sets it. The watchdog looks at all processes every 10 s. A process that stays
  in `EMLX.await_worker/4` with no new work for 45 s is stuck. For the first stuck process, the
  watchdog writes two files to HANG_DIR, and then stops:

    * report.txt: the stack and the command queue of each stuck process, the result of a
      small eval on the worker of each stuck process, and the stack of each process in EMLX
      or `Transmuter.Shared`. A reply from the worker shows a lost reply. No reply shows a
      worker thread that is blocked.
    * sample.txt: the native stacks of all threads of beam.smp, from the macOS `sample` tool.
      The stack of the blocked worker thread shows the cause.

  The watchdog does not stop the tests. The stuck test fails with its own timeout.
  """

  @tick 10_000
  @threshold 45_000

  def start(dir) do
    File.mkdir_p!(dir)
    spawn(fn -> watch(dir, %{}) end)
  end

  defp watch(dir, seen) do
    Process.sleep(@tick)
    now = System.monotonic_time(:millisecond)

    # A process in a receive keeps its reductions, except about 1 for each Process.info call.
    seen =
      for pid <- Process.list(),
          info = Process.info(pid, [:current_stacktrace, :reductions]),
          info != nil,
          Enum.any?(info[:current_stacktrace], &match?({EMLX, :await_worker, _, _}, &1)),
          into: %{} do
        reductions = info[:reductions]

        case seen do
          %{^pid => {since, last}} when reductions - last < 20 -> {pid, {since, reductions}}
          _ -> {pid, {now, reductions}}
        end
      end

    case for {pid, {since, _}} <- seen, now - since > @threshold, do: pid do
      [] -> watch(dir, seen)
      stuck -> capture(dir, stuck)
    end
  end

  defp capture(dir, stuck) do
    Task.start(fn ->
      System.cmd("sample", ["#{System.pid()}", "5", "-file", Path.join(dir, "sample.txt")])
    end)

    queues = for pid <- stuck, do: {pid, queue(pid)}

    report = [
      "Stuck at #{DateTime.utc_now()}\n",
      for {pid, queue} <- queues do
        {:message_queue_len, messages} = Process.info(pid, :message_queue_len)
        ["\n== #{inspect(pid)} queue=#{inspect(queue)} messages=#{messages}\n", stack(pid)]
      end,
      "\n== A small eval on the worker of each stuck process, with a limit of 10 s\n",
      for queue <- Enum.uniq([nil | Enum.map(queues, &elem(&1, 1))]) do
        "#{inspect(queue)}: #{probe(queue)}\n"
      end,
      "\n== Processes in EMLX or Transmuter.Shared\n",
      for pid <- Process.list(), in_emlx?(pid) do
        {:status, status} = Process.info(pid, :status)
        ["\n-- #{inspect(pid)} #{status} queue=#{inspect(queue(pid))}\n", stack(pid)]
      end
    ]

    File.write!(Path.join(dir, "report.txt"), report)
  end

  # The command queue that EMLX.CommandQueue.with_queue/2 binds. nil is the default worker.
  defp queue(pid) do
    {:dictionary, dictionary} = Process.info(pid, :dictionary)
    dictionary[:emlx_command_queue]
  end

  defp stack(pid) do
    {:current_stacktrace, stack} = Process.info(pid, :current_stacktrace)
    Exception.format_stacktrace(stack)
  end

  defp in_emlx?(pid) do
    case Process.info(pid, :current_stacktrace) do
      {:current_stacktrace, stack} ->
        Enum.any?(stack, fn {module, _, _, _} ->
          String.starts_with?(Atom.to_string(module), ["Elixir.EMLX", "Elixir.Transmuter.Shared"])
        end)

      nil ->
        false
    end
  end

  defp probe(queue) do
    {device, run} =
      case queue do
        nil ->
          {:gpu, fn fun -> fun.() end}

        {ref, device} ->
          queue = struct!(EMLX.CommandQueue, ref: ref, device: device)
          {device, &EMLX.CommandQueue.with_queue(queue, &1)}
      end

    task =
      Task.async(fn ->
        run.(fn ->
          Nx.tensor([1.0, 2.0], backend: {EMLX.Backend, device: device})
          |> Nx.add(1)
          |> Nx.to_flat_list()
        end)
      end)

    case Task.yield(task, 10_000) || Task.shutdown(task, :brutal_kill) do
      {:ok, value} -> "reply #{inspect(value)}"
      {:exit, reason} -> "exit #{inspect(reason)}"
      nil -> "no reply"
    end
  end
end
