defmodule Transmuter.Text.ContinuousGenerationTest do
  use ExUnit.Case, async: true

  @moduletag :hub
  @moduletag timeout: 900_000
  @moduletag ci_partition: 5

  @opts [
    backend: Transmuter.TestBackend.backend(),
    defn_options: Transmuter.TestBackend.compiler_opts(),
    variant: :fp32
  ]

  alias Transmuter.Text
  alias Transmuter.Text.ContinuousGeneration

  @fox "A fox lived near a wide river at the edge of an old forest. Every morning she walked along the bank and watched the water. One day she saw"

  # A server of 2 rows gets a long request, and after its first chunk two more requests: the
  # second joins the running batch, and the third takes the row of the first request that
  # ends, while the other row still runs. So two steps admit a request next to a running row.
  # Each request gives the result that it gives alone, with greedy selection and with sampling
  # from a seed.
  test "requests that arrive while a batch runs give the text that they give alone" do
    source = {:hf, "HuggingFaceTB/SmolLM2-135M-Instruct"}

    for strategy <- [:greedy, :sampling] do
      opts = [strategy: strategy, max_new_tokens: 24] ++ @opts

      [fox, story, rainbow] =
        for {text, seed} <- [
              {@fox, 7},
              {"Once upon a time", 11},
              {"Explain how a rainbow forms.", 13}
            ],
            do: %{text: text, seed: seed}

      single = Text.generation(source, opts)
      alone = for input <- [fox, story, rainbow], do: Nx.Serving.run(single, input)

      server = start_supervised!({ContinuousGeneration, [source: source, slots: 2] ++ opts})

      {{fox_text, others}, steps} =
        step_events(server, fn ->
          parent = self()
          ref = make_ref()

          streamer =
            Task.async(fn ->
              server
              |> ContinuousGeneration.stream(fox)
              |> Enum.map_join(fn chunk ->
                send(parent, {ref, :chunk})
                chunk
              end)
            end)

          assert_receive {^ref, :chunk}, 60_000
          others = ContinuousGeneration.run(server, [story, rainbow])
          {Task.await(streamer, :infinity), others}
        end)

      [%{results: [%{text: text}]} | rest] = alone
      assert {fox_text, others} == {text, rest}, inspect(strategy)
      assert Enum.count(steps, &(&1.admitted == 1 and &1.running == 1)) == 2
      stop_supervised!(ContinuousGeneration)
    end
  end

  # A server of 1 row with a queue of 1 request. A stream that must give 200 tokens runs, a
  # second request waits, and a third raises QueueFullError. The caller of the stream exits
  # after its first chunk, at step k. The server sees the exit at most 2 steps later, so the
  # waiting request runs in its row and gives its result alone, at most k + 2 + its own steps
  # in total. A row that stays with the stream until its 200 tokens gives more. When the
  # server stops, a waiting caller exits with the reason of the server.
  test "a server of one row frees the row of a caller that exits, and its queue has a limit" do
    source = {:hf, "HuggingFaceTB/SmolLM2-135M-Instruct"}
    opts = [strategy: :greedy, max_new_tokens: 200, min_new_tokens: 200] ++ @opts
    alone = Nx.Serving.run(Text.generation(source, opts), "Once upon a time")

    spec = {ContinuousGeneration, [source: source, slots: 1, max_queue: 1] ++ opts}
    server = start_supervised!(Supervisor.child_spec(spec, restart: :temporary))

    {_result, own_calls} =
      count_calls(server, fn _calls -> ContinuousGeneration.run(server, "Once upon a time") end)

    parent = self()

    {{result, killed_at}, calls} =
      count_calls(server, fn calls ->
        streamer =
          spawn(fn ->
            server
            |> ContinuousGeneration.stream(@fox)
            |> Enum.each(fn _chunk -> send(parent, :chunk) end)
          end)

        assert_receive :chunk, 60_000
        waiting = Task.async(fn -> ContinuousGeneration.run(server, "Once upon a time") end)
        wait_for_queue(server, 1)

        assert_raise ContinuousGeneration.QueueFullError, ~r/holds 1 requests/, fn ->
          ContinuousGeneration.run(server, "Explain how a rainbow forms.")
        end

        Process.exit(streamer, :kill)
        killed_at = calls.()
        {Task.await(waiting, :infinity), killed_at}
      end)

    assert result == alone
    assert calls <= killed_at + 2 + own_calls

    {caller, monitor} = spawn_monitor(fn -> ContinuousGeneration.run(server, @fox) end)
    wait_for_rows(server, 1)
    Process.exit(server, :kill)

    assert_receive {:DOWN, ^monitor, :process, ^caller,
                    {:killed, {ContinuousGeneration, :run, _args}}},
                   60_000
  end

  # The server reads its mailbox only between two steps. A step that takes longer than the
  # 5 s of a GenServer call, as a compile of a new shape can, does not end a request that
  # arrives during it: the caller reads the task without the server and waits for its
  # reply. A telemetry handler in the server process makes the first step take 6 s.
  test "a request that arrives during a long step gets its result" do
    source = {:hf, "HuggingFaceTB/SmolLM2-135M-Instruct"}
    opts = [strategy: :greedy, max_new_tokens: 8] ++ @opts
    alone = Nx.Serving.run(Text.generation(source, opts), "Once upon a time")
    server = start_supervised!({ContinuousGeneration, [source: source, slots: 2] ++ opts})
    parent = self()
    id = make_ref()

    :telemetry.attach(
      id,
      [:transmuter, :continuous_generation, :step, :start],
      fn _event, _measurements, _metadata, _config ->
        if self() == server and Process.get(id) == nil do
          Process.put(id, :slept)
          send(parent, {id, :sleeping})
          Process.sleep(6_000)
        end
      end,
      nil
    )

    first = Task.async(fn -> ContinuousGeneration.run(server, @fox) end)
    assert_receive {^id, :sleeping}, 60_000
    second = Task.async(fn -> ContinuousGeneration.run(server, "Once upon a time") end)
    assert Task.await(second, :infinity) == alone
    Task.await(first, :infinity)
    :telemetry.detach(id)
  end

  # Requests cannot join a running batch of a merged decoder, which runs each prompt on its
  # branch without cache, or of beam search, whose rows of a prompt move together. A decoder
  # in :append needs a compaction of its cache first.
  test "a model whose rows cannot join a running batch raises at the start" do
    for {source, opts, message} <- [
          {{:hf, "Xenova/gpt2"}, [], ~r/requests cannot join a running batch/},
          {{:hf, "HuggingFaceTB/SmolLM2-135M-Instruct"}, [num_beams: 2],
           ~r/requests cannot join a running batch/},
          {{:hf, "onnx-community/SmolLM2-135M-Instruct-ONNX"}, [], ~r/needs a compaction/}
        ] do
      assert_raise ArgumentError, message, fn ->
        ContinuousGeneration.start_link([source: source] ++ opts ++ @opts)
      end
    end
  end

  # The result of `fun` and the metadata of the steps that `server` started while it ran.
  defp step_events(server, fun) do
    pid = GenServer.whereis(server)
    parent = self()
    id = make_ref()

    :telemetry.attach(
      id,
      [:transmuter, :continuous_generation, :step, :start],
      fn _event, _measurements, metadata, _config ->
        if self() == pid, do: send(parent, {id, metadata})
      end,
      nil
    )

    result = fun.()
    :telemetry.detach(id)
    {result, receive_steps(id)}
  end

  defp receive_steps(id) do
    receive do
      {^id, metadata} -> [metadata | receive_steps(id)]
    after
      0 -> []
    end
  end

  # The result of `fun` and the compiled calls that `server` made while it ran. `fun` gets a
  # function that reads the calls so far.
  defp count_calls(server, fun) do
    pid = GenServer.whereis(server)
    counter = :counters.new(1, [])
    id = make_ref()

    :telemetry.attach(
      id,
      [:transmuter, :call, :stop],
      fn _event, _measurements, _metadata, _config ->
        if self() == pid, do: :counters.add(counter, 1, 1)
      end,
      nil
    )

    result = fun.(fn -> :counters.get(counter, 1) end)
    :telemetry.detach(id)
    {result, :counters.get(counter, 1)}
  end

  defp wait_for_queue(server, length) do
    unless :queue.len(:sys.get_state(server).queue) == length do
      Process.sleep(10)
      wait_for_queue(server, length)
    end
  end

  defp wait_for_rows(server, count) do
    unless map_size(:sys.get_state(server).rows) == count do
      Process.sleep(10)
      wait_for_rows(server, count)
    end
  end
end
