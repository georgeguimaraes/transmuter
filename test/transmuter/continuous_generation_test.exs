defmodule Transmuter.ContinuousGenerationTest.FailingTask do
  @moduledoc false
  # The text generation task, whose :prepare raises on the prompt "boom", as an out-of-memory
  # error of the device in the vision encoder would.

  alias Transmuter.Text.Generation
  alias Transmuter.Tokenizer

  def options, do: Generation.options()

  def task(source, opts) do
    {ctx, tokenizer, task} = Generation.task(source, opts)

    prepare = fn defn_options ->
      prepare = task.prepare.(defn_options)

      fn tensors, [prompt] ->
        if Tokenizer.decode(tokenizer, prompt) =~ "boom",
          do: raise("out of memory"),
          else: prepare.(tensors, [prompt])
      end
    end

    {ctx, tokenizer, %{task | prepare: prepare}}
  end
end

defmodule Transmuter.ContinuousGenerationTest.BrokenTask do
  @moduledoc false
  # The text generation task, whose :prepare raises when it loads its models.

  alias Transmuter.Text.Generation

  def options, do: Generation.options()

  def task(source, opts) do
    {ctx, tokenizer, task} = Generation.task(source, opts)
    {ctx, tokenizer, %{task | prepare: fn _defn_options -> raise "no encoder" end}}
  end
end

defmodule Transmuter.ContinuousGenerationTest do
  use ExUnit.Case, async: true

  @moduletag :hub
  @moduletag timeout: 900_000
  @moduletag ci_partition: 5

  @opts [
    backend: Transmuter.TestBackend.backend(),
    defn_options: Transmuter.TestBackend.compiler_opts(),
    variant: :fp32
  ]

  alias Transmuter.{ContinuousGeneration, Text, Vision}
  alias Transmuter.ContinuousGenerationTest.{BrokenTask, FailingTask}

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
  # second request waits, and a third raises QueueFullError, while the second is in the worker
  # and when it is in the queue. The caller of the stream exits
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
        %{worker: worker} = :sys.get_state(server)
        id = make_ref()

        :telemetry.attach(
          id,
          [:transmuter, :continuous_generation, :prepare, :start],
          fn _event, _measurements, _metadata, _config ->
            if self() == worker do
              send(parent, {id, :preparing})
              receive do: ({^id, :go} -> :ok)
            end
          end,
          nil
        )

        waiting = Task.async(fn -> ContinuousGeneration.run(server, "Once upon a time") end)
        assert_receive {^id, :preparing}, 60_000

        full = fn ->
          assert_raise ContinuousGeneration.QueueFullError, ~r/holds 1 requests/, fn ->
            ContinuousGeneration.run(server, "Explain how a rainbow forms.", 60_000)
          end
        end

        full.()
        send(worker, {id, :go})
        :telemetry.detach(id)
        wait_for(fn -> :queue.len(:sys.get_state(server).queue) == 1 end)
        full.()

        Process.exit(streamer, :kill)
        killed_at = calls.()
        {Task.await(waiting, :infinity), killed_at}
      end)

    assert result == alone
    assert calls <= killed_at + 2 + own_calls

    {caller, monitor} = spawn_monitor(fn -> ContinuousGeneration.run(server, @fox) end)
    wait_for(fn -> map_size(:sys.get_state(server).rows) == 1 end)
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
  # branch without cache, or of beam search, whose rows of a prompt move together.
  test "a model whose rows cannot join a running batch raises at the start" do
    for {source, opts, message} <- [
          {{:hf, "Xenova/gpt2"}, [], ~r/requests cannot join a running batch/},
          {{:hf, "HuggingFaceTB/SmolLM2-135M-Instruct"}, [num_beams: 2],
           ~r/requests cannot join a running batch/}
        ] do
      assert_raise ArgumentError, message, fn ->
        ContinuousGeneration.start_link([source: source] ++ opts ++ @opts)
      end
    end
  end

  # SmolLM2-ONNX runs in :append, where the rows of all steps write at one offset. A server of
  # 2 rows with a cache of at most 256 rows gets a first request, and after its first chunk 11
  # more, about 40 rows each and about 500 rows in all. The rows then end at different steps,
  # so each request joins a running row. The compactions keep the buffers at 256 rows, and each
  # request gives the text that it gives alone.
  test "a server in :append compacts its buffers and gives each request its text alone" do
    source = {:hf, "onnx-community/SmolLM2-135M-Instruct-ONNX"}
    opts = [strategy: :greedy, max_new_tokens: 30] ++ @opts
    texts = [@fox, "Once upon a time", "Explain how a rainbow forms.", "Why is the sky blue?"]
    inputs = for i <- 0..11, do: Enum.at(texts, rem(i, 4))
    single = Text.generation(source, opts)
    alone = Map.new(texts, &{&1, Nx.Serving.run(single, &1)})

    spec =
      {ContinuousGeneration, [source: source, slots: 2, max_queue: 12, capacity: 256] ++ opts}

    server = start_supervised!(spec)

    [first | rest] = inputs

    {results, steps} =
      step_events(server, fn ->
        parent = self()
        ref = make_ref()

        streamer =
          Task.async(fn ->
            server
            |> ContinuousGeneration.stream(first)
            |> Enum.map_join(fn chunk ->
              send(parent, {ref, :chunk})
              chunk
            end)
          end)

        assert_receive {^ref, :chunk}, 60_000
        others = ContinuousGeneration.run(server, rest)
        {Task.await(streamer, :infinity), others}
      end)

    %{results: [%{text: text}]} = alone[first]
    assert results == {text, Enum.map(rest, &alone[&1])}
    assert steps |> Enum.map(& &1.capacity) |> Enum.max() == 256
    assert Enum.count(steps, &(&1.admitted == 1 and &1.running == 1)) == 11
  end

  # A long request in :append, the fox 16 times, makes the buffers grow past 512 rows. Two
  # short requests arrive after its first chunk and run in the other row. When the long
  # request ends, the longest running row and 16 rows fit in half of the buffers, so the step
  # compacts them into 256 rows, and the steps of the short requests after it run at 256 rows.
  # Each request gives the text that it gives alone.
  test "a server in :append shrinks its buffers after a long request ends" do
    source = {:hf, "onnx-community/SmolLM2-135M-Instruct-ONNX"}
    opts = [strategy: :greedy, max_new_tokens: 30] ++ @opts
    long = String.duplicate(@fox <> " ", 16)
    shorts = ["Once upon a time", "Explain how a rainbow forms."]
    single = Text.generation(source, opts)
    alone = for input <- [long | shorts], do: Nx.Serving.run(single, input)
    server = start_supervised!({ContinuousGeneration, [source: source, slots: 2] ++ opts})

    {{long_text, others}, steps} =
      step_events(server, fn ->
        parent = self()
        ref = make_ref()

        streamer =
          Task.async(fn ->
            server
            |> ContinuousGeneration.stream(long)
            |> Enum.map_join(fn chunk ->
              send(parent, {ref, :chunk})
              chunk
            end)
          end)

        assert_receive {^ref, :chunk}, 60_000
        others = ContinuousGeneration.run(server, shorts)
        {Task.await(streamer, :infinity), others}
      end)

    [%{results: [%{text: text}]} | rest] = alone
    assert {long_text, others} == {text, rest}
    capacities = Enum.map(steps, & &1.capacity)
    assert Enum.max(capacities) > 512
    assert List.last(capacities) == 256
  end

  # A server of 3 rows runs a chat without images. A chat with text before its image joins, and
  # while the image tokens of that chat are left, a chat with another image joins (see
  # join_while_reading/2). So the features of both chats are in the group, and each chat must
  # read its own. Each chat gives
  # the result of image_text_to_text/2 alone, with greedy selection and with sampling from a
  # seed. The random model gives the same text for all chats, so its log-probabilities show
  # which features a chat read. Before them, a chat without images whose text has the image
  # token raises in its caller, as its prompt has image tokens and no features.
  test "chats with images that join a running batch give the result that they give alone" do
    source = {:hf, "onnx-internal-testing/tiny-random-Gemma3ForConditionalGeneration"}
    variant = %{vision_encoder: :fp32, embed_tokens: :fp32, decoder_model_merged: :fp32}
    image = StbImage.read_file!(Path.expand("../fixtures/cats.png", __DIR__))
    tall = Nx.transpose(StbImage.to_nx(image), axes: [1, 0, 2])

    chats = [
      chat([%{type: "text", text: "What is the capital of France?"}]),
      chat([%{type: "text", text: @fox <> " " <> @fox}, %{type: "image", image: image}]),
      chat([%{type: "image", image: tall}, %{type: "text", text: "What animals are here?"}])
    ]

    literal = chat([%{type: "text", text: "What is <image_soft_token> here?"}])

    for strategy <- [:greedy, :sampling] do
      opts = [variant: variant, strategy: strategy, max_new_tokens: 24, token_logprobs: true]
      opts = Keyword.merge(@opts, opts)

      single = Vision.image_text_to_text(source, opts)
      alone = Enum.map(chats, &Nx.Serving.run(single, Map.put(&1, :seed, 7)))
      spec = [task: Vision.ImageTextToText, source: source, slots: 3] ++ opts
      server = start_supervised!({ContinuousGeneration, spec})

      assert_raise ArgumentError, ~r/has 1 image tokens .*, and the images give 0/, fn ->
        ContinuousGeneration.run(server, literal)
      end

      results = join_while_reading(server, Enum.map(chats, &Map.put(&1, :seed, 7)))

      for {%{results: [result]}, %{results: [expected]}} <- Enum.zip(results, alone) do
        assert result.text == expected.text
        assert_all_close(result.token_logprobs, expected.token_logprobs)
      end

      stop_supervised!(ContinuousGeneration)
    end
  end

  # The worker raises on the request "boom" while a stream runs, as an out-of-memory error of
  # the vision encoder would. The caller of that request gets the exception, and the stream
  # and a later request give the text that they give alone.
  test "an exception of the worker goes to the caller of its request, and the server continues" do
    source = {:hf, "HuggingFaceTB/SmolLM2-135M-Instruct"}
    opts = [strategy: :greedy, max_new_tokens: 24] ++ @opts
    single = Text.generation(source, opts)
    alone = for text <- [@fox, "Once upon a time"], do: Nx.Serving.run(single, text)
    spec = [task: FailingTask, source: source, slots: 2] ++ opts
    server = start_supervised!({ContinuousGeneration, spec})
    parent = self()

    streamer =
      Task.async(fn ->
        server
        |> ContinuousGeneration.stream(@fox)
        |> Enum.map_join(fn chunk ->
          send(parent, :chunk)
          chunk
        end)
      end)

    assert_receive :chunk, 60_000

    assert_raise RuntimeError, "out of memory", fn ->
      ContinuousGeneration.run(server, "boom")
    end

    assert ContinuousGeneration.run(server, "Once upon a time") == Enum.at(alone, 1)
    [%{results: [%{text: text}]} | _rest] = alone
    assert Task.await(streamer, :infinity) == text
  end

  # The worker holds the request of a caller, and the caller exits. The server forgets the
  # request, so it never takes a row when the worker sends it back, and a later request gives
  # its text alone in the only step that admits a request.
  test "a request whose caller exits while it is in the worker never runs" do
    source = {:hf, "HuggingFaceTB/SmolLM2-135M-Instruct"}
    opts = [strategy: :greedy, max_new_tokens: 8] ++ @opts
    alone = Nx.Serving.run(Text.generation(source, opts), "Once upon a time")
    server = start_supervised!({ContinuousGeneration, [source: source, slots: 2] ++ opts})
    %{worker: worker} = :sys.get_state(server)
    parent = self()
    id = make_ref()

    :telemetry.attach(
      id,
      [:transmuter, :continuous_generation, :prepare, :start],
      fn _event, _measurements, _metadata, _config ->
        if self() == worker and Process.put(id, :held) == nil do
          send(parent, {id, :preparing})
          receive do: ({^id, :go} -> :ok)
        end
      end,
      nil
    )

    {caller, monitor} = spawn_monitor(fn -> ContinuousGeneration.run(server, @fox) end)
    assert_receive {^id, :preparing}, 60_000
    Process.exit(caller, :kill)
    assert_receive {:DOWN, ^monitor, :process, ^caller, :killed}
    wait_for(fn -> :sys.get_state(server).preparing == %{} end)

    {result, steps} =
      step_events(server, fn ->
        send(worker, {id, :go})
        ContinuousGeneration.run(server, "Once upon a time")
      end)

    :telemetry.detach(id)
    assert result == alone
    assert steps |> Enum.map(& &1.admitted) |> Enum.sum() == 1
  end

  # The worker loads the models of :prepare before the server starts, so its error fails the
  # start.
  test "a server whose worker cannot load its models does not start" do
    spec = [task: BrokenTask, source: {:hf, "HuggingFaceTB/SmolLM2-135M-Instruct"}] ++ @opts
    assert {:error, reason} = start_supervised({ContinuousGeneration, spec})
    assert inspect(reason) =~ "no encoder"
  end

  @doc false
  def chat(content), do: %{messages: [%{role: "user", content: content}]}

  @doc false
  # Runs `chats` in one call and returns their results. The worker prepares the chats in their
  # order. The first chat has no images and runs first. Telemetry handlers hold the server
  # until the second chat joins next to the first, and then until the worker has prepared the
  # third chat, which the worker holds until then. So the third chat joins at the next step,
  # while the prompt of the second chat runs in parts of 64 tokens next to the first chat.
  def join_while_reading(server, chats) do
    %{worker: worker} = :sys.get_state(server)
    pid = GenServer.whereis(server)
    parent = self()
    id = make_ref()

    events = [
      [:transmuter, :continuous_generation, :prepare, :start],
      [:transmuter, :continuous_generation, :prepare, :stop],
      [:transmuter, :continuous_generation, :step, :start]
    ]

    :telemetry.attach_many(id, events, &__MODULE__.hold/4, {id, parent, worker, pid})
    task = Task.async(fn -> ContinuousGeneration.run(server, chats) end)

    # The server holds a step of the first chat when the second chat is not ready yet.
    receive do
      {^id, :running} ->
        assert_receive {^id, :second_prepared}, 60_000
        send(pid, {id, :go})
        assert_receive {^id, :joins}, 60_000

      {^id, :joins} ->
        :ok
    after
      60_000 -> flunk("the second chat did not join")
    end

    assert_receive {^id, :third}, 60_000
    send(worker, {id, :go})
    assert_receive {^id, :third_prepared}, 60_000
    send(pid, {id, :go})
    results = Task.await(task, :infinity)
    :telemetry.detach(id)
    results
  end

  @doc false
  def hold([_, _, :prepare, :start], _measurements, _metadata, {id, parent, worker, _pid})
      when self() == worker do
    Process.put(id, Process.get(id, 0) + 1)

    if Process.get(id) == 3 do
      send(parent, {id, :third})
      receive do: ({^id, :go} -> :ok)
    end
  end

  def hold([_, _, :prepare, :stop], _measurements, _metadata, {id, parent, worker, _pid})
      when self() == worker do
    case Process.get(id) do
      2 -> send(parent, {id, :second_prepared})
      3 -> send(parent, {id, :third_prepared})
      _first -> :ok
    end
  end

  def hold([_, _, :step, :start], _measurements, metadata, {id, parent, _worker, pid})
      when self() == pid and metadata.running == 1 do
    case {metadata.admitted, Process.get(id)} do
      {0, nil} -> wait(id, parent, :running)
      {1, held} when held != :joins -> wait(id, parent, :joins)
      _other -> :ok
    end
  end

  def hold(_event, _measurements, _metadata, _config), do: :ok

  defp wait(id, parent, event) do
    Process.put(id, event)
    send(parent, {id, event})
    receive do: ({^id, :go} -> :ok)
  end

  defp assert_all_close(got, expected) do
    assert length(got) == length(expected)

    for {g, e} <- Enum.zip(got, expected) do
      assert_in_delta g, e, 1.0e-4, "expected #{inspect(expected)}, got #{inspect(got)}"
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

  defp wait_for(done?) do
    unless done?.() do
      Process.sleep(10)
      wait_for(done?)
    end
  end
end

# --only ci_partition:N in CI would also include :large_model tests, because an included tag
# wins over an excluded one. So these tests are in a module without a :ci_partition tag, and
# CI leaves them out.
defmodule Transmuter.ContinuousGenerationLargeModelTest do
  use ExUnit.Case, async: true

  @moduletag :hub
  @moduletag timeout: 900_000

  @opts [
    backend: Transmuter.TestBackend.backend(),
    defn_options: Transmuter.TestBackend.compiler_opts()
  ]

  alias Transmuter.{ContinuousGeneration, ContinuousGenerationTest, Vision}

  # SmolVLM gives 64 image tokens for each tile of an image. A chat with an image joins a
  # running chat without images, and a chat with another image joins while the image tokens
  # of the first are left. Each chat gives the text of image_text_to_text/2 alone.
  @tag :large_model
  test "chats of SmolVLM that join a running batch give the text that they give alone" do
    {server, alone} = server_and_alone({:hf, "HuggingFaceTB/SmolVLM-256M-Instruct"})
    assert ContinuousGenerationTest.join_while_reading(server, chats()) == alone
  end

  # The same joins with Qwen2-VL, whose image tokens have the 3-D positions of M-RoPE.
  @tag :large_model
  test "chats of Qwen2-VL that join a running batch give the text that they give alone" do
    {server, alone} = server_and_alone({:hf, "onnx-community/Qwen2-VL-2B-Instruct"})
    assert ContinuousGenerationTest.join_while_reading(server, chats()) == alone
  end

  defp chats do
    image = StbImage.read_file!(Path.expand("../fixtures/cats.png", __DIR__))
    tall = Nx.transpose(StbImage.to_nx(image), axes: [1, 0, 2])

    Enum.map(
      [
        [%{type: "text", text: "What is the capital of France?"}],
        [%{type: "image", image: image}, %{type: "text", text: "Can you describe this image?"}],
        [%{type: "image", image: tall}, %{type: "text", text: "What animals are in the image?"}]
      ],
      &ContinuousGenerationTest.chat/1
    )
  end

  # The results of a serving alone, and then a server of 3 rows. The serving is out of scope
  # and collected before the server loads its models, so only one copy of the models stays.
  defp server_and_alone(source) do
    opts = [strategy: :greedy, max_new_tokens: 24] ++ @opts
    alone = alone(source, opts)
    :erlang.garbage_collect()
    spec = [task: Vision.ImageTextToText, source: source, slots: 3] ++ opts
    {start_supervised!({ContinuousGeneration, spec}), alone}
  end

  defp alone(source, opts) do
    single = Vision.image_text_to_text(source, opts)
    Enum.map(chats(), &Nx.Serving.run(single, &1))
  end
end
