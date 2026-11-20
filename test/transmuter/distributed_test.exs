defmodule Transmuter.DistributedTest do
  # The test node runs the models, and a peer node calls them. The peer has only the
  # applications that Transmuter requires, without EMLX, EXLA, tokenizers or stb_image, and
  # its default backend is Nx.BinaryBackend, as a node without a GPU.
  use ExUnit.Case, async: false

  @moduletag :hub
  @moduletag timeout: 900_000

  @opts [
    backend: Transmuter.TestBackend.backend(),
    defn_options: Transmuter.TestBackend.compiler_opts()
  ]

  alias Transmuter.ContinuousBatching

  setup_all do
    unless Node.alive?() do
      {_output, 0} = System.cmd("epmd", ["-daemon"])
      name = :"transmuter_test_#{System.unique_integer([:positive])}"
      {:ok, _pid} = :net_kernel.start(name, %{name_domain: :shortnames})
      on_exit(&Node.stop/0)
    end

    paths = for app <- required_apps(:transmuter, []), do: ~c"#{:code.lib_dir(app)}/ebin"
    args = Enum.flat_map(paths, &[~c"-pa", &1])
    {:ok, peer, node} = :peer.start(%{name: :peer.random_name(), args: args})
    on_exit(fn -> :peer.stop(peer) end)
    {:ok, _apps} = :erpc.call(node, Application, :ensure_all_started, [:transmuter])
    %{peer: node}
  end

  test "an embedding serving gives a caller on another node the result of a local call", %{
    peer: peer
  } do
    serving = Transmuter.Text.text_embedding({:hf, "Xenova/all-MiniLM-L6-v2"}, @opts)
    start_supervised!({Nx.Serving, serving: serving, name: __MODULE__.Embedding})
    wait_for(fn -> member?(peer, Nx.Serving.PG, __MODULE__.Embedding) end)

    local = Nx.Serving.batched_run(__MODULE__.Embedding, "The cat sat on the mat")

    remote =
      :erpc.call(peer, Nx.Serving, :batched_run, [__MODULE__.Embedding, "The cat sat on the mat"])

    assert remote == local
    assert %Nx.BinaryBackend{} = remote.embedding.data
  end

  # The servings of each/2, of generation without a stream and of zero-shot entity recognition
  # give Nx.Serving the outputs of their batches, so a remote call returns the map of a local
  # call on Nx 1.0. The 3 prompts with a batch size of 2 give two batches, whose outputs
  # Nx.Serving joins. A list that does not fit in one batch of depth estimation raises an
  # ArgumentError in a local caller, and a remote caller gets it as an exit.
  test "servings that run each input alone or generate give a remote caller the result of a local call",
       %{peer: peer} do
    image = Nx.iota({48, 64, 3}, type: :u8, backend: Nx.BinaryBackend)
    prompts = ["Once upon a time", "Explain how a rainbow forms.", "The capital of France is"]

    entities = %{
      text: "Cristiano Ronaldo plays for Al Nassr.",
      labels: ["person", "organization"]
    }

    servings = [
      {__MODULE__.Depth,
       Transmuter.Vision.depth_estimation({:hf, "onnx-community/depth-anything-v2-small"}, @opts),
       image},
      {__MODULE__.Generation,
       Transmuter.Text.generation(
         {:hf, "HuggingFaceTB/SmolLM2-135M-Instruct"},
         [strategy: :greedy, max_new_tokens: 8, compile: [batch_size: 2]] ++ @opts
       ), prompts},
      {__MODULE__.Entities,
       Transmuter.Text.zero_shot_entity_recognition(
         {:hf, "onnx-community/gliner_small-v2.1"},
         [variant: :fp16] ++ @opts
       ), entities}
    ]

    for {name, serving, input} <- servings do
      start_supervised!(
        Supervisor.child_spec({Nx.Serving, serving: serving, name: name}, id: name)
      )

      wait_for(fn -> member?(peer, Nx.Serving.PG, name) end)
      local = Nx.Serving.batched_run(name, input)
      assert :erpc.call(peer, Nx.Serving, :batched_run, [name, input]) == local, inspect(name)
    end

    assert %{depth: %Nx.Tensor{}, predicted_depth: %Nx.Tensor{}} =
             Nx.Serving.batched_run(__MODULE__.Depth, image)

    assert_raise ArgumentError, ~r/batch size of 1, got: 2/, fn ->
      Nx.Serving.batched_run(__MODULE__.Depth, [image, image])
    end

    assert {:exception, {{%ArgumentError{message: message}, _stacktrace}, _call}} =
             catch_exit(
               :erpc.call(peer, Nx.Serving, :batched_run, [__MODULE__.Depth, [image, image]])
             )

    assert message =~ "batch size of 1, got: 2"
  end

  test "a caller on another node gets the text of a local call and its errors", %{peer: peer} do
    source = {:hf, "HuggingFaceTB/SmolLM2-135M-Instruct"}
    opts = [strategy: :greedy, max_new_tokens: 24] ++ @opts
    spec = [name: __MODULE__.Server, source: source, slots: 2] ++ opts
    start_supervised!({ContinuousBatching, spec})
    wait_for(fn -> member?(peer, Transmuter.PG, __MODULE__.Server) end)

    inputs = ["Once upon a time", "Explain how a rainbow forms."]
    local = ContinuousBatching.run(__MODULE__.Server, inputs)
    assert :erpc.call(peer, ContinuousBatching, :run, [__MODULE__.Server, inputs]) == local

    assert {:exception, %ArgumentError{message: "invalid input, " <> _}, _stacktrace} =
             catch_error(:erpc.call(peer, ContinuousBatching, :run, [__MODULE__.Server, 42]))
  end

  # The request must give 2000 tokens, which takes much longer than the 10 s that the test
  # waits for the row to end.
  test "the row of a caller on another node ends when the caller exits", %{peer: peer} do
    source = {:hf, "HuggingFaceTB/SmolLM2-135M-Instruct"}
    opts = [strategy: :greedy, max_new_tokens: 2000, min_new_tokens: 2000] ++ @opts
    spec = [name: __MODULE__.OneRow, source: source, slots: 1] ++ opts
    server = start_supervised!({ContinuousBatching, spec})
    wait_for(fn -> member?(peer, Transmuter.PG, __MODULE__.OneRow) end)

    caller = Node.spawn(peer, ContinuousBatching, :run, [__MODULE__.OneRow, "Once upon a time"])
    wait_for(fn -> map_size(:sys.get_state(server).rows) == 1 end)
    Process.exit(caller, :kill)

    wait_for(fn ->
      state = :sys.get_state(server)
      state.rows == %{} and state.callers == %{}
    end)
  end

  # A local caller gets an exit that it can catch when the server stops, and a remote caller
  # gets one when the proxy on the node of the server stops. The request ends too.
  test "a caller on another node catches the exit of a proxy that stops", %{peer: peer} do
    source = {:hf, "HuggingFaceTB/SmolLM2-135M-Instruct"}
    opts = [strategy: :greedy, max_new_tokens: 2000, min_new_tokens: 2000] ++ @opts
    spec = [name: __MODULE__.Proxied, source: source, slots: 1] ++ opts
    server = start_supervised!({ContinuousBatching, spec})
    wait_for(fn -> member?(peer, Transmuter.PG, __MODULE__.Proxied) end)

    code = """
    try do
      Transmuter.ContinuousBatching.run(name, "Once upon a time")
    catch
      :exit, reason -> {:caught, reason}
    end
    """

    binding = [name: __MODULE__.Proxied]
    caller = Task.async(fn -> :erpc.call(peer, Code, :eval_string, [code, binding]) end)
    wait_for(fn -> map_size(:sys.get_state(server).rows) == 1 end)

    proxy =
      Enum.find(Process.list(), fn pid ->
        match?(
          {:initial_call, {ContinuousBatching, :__proxy__, _arity}},
          Process.info(pid, :initial_call)
        )
      end)

    Process.exit(proxy, :kill)

    assert {{:caught, {:killed, {ContinuousBatching, :run, _args}}}, _binding} =
             Task.await(caller, 60_000)

    wait_for(fn ->
      state = :sys.get_state(server)
      state.rows == %{} and state.callers == %{}
    end)
  end

  # The requests must give 2000 tokens, so a request that the caller stops ends long before its
  # last token.
  test "a stream on another node gives the chunks of a local stream and ends when the caller stops it",
       %{peer: peer} do
    source = {:hf, "HuggingFaceTB/SmolLM2-135M-Instruct"}
    opts = [strategy: :greedy, max_new_tokens: 2000, min_new_tokens: 2000] ++ @opts
    spec = [name: __MODULE__.Streams, source: source, slots: 2] ++ opts
    server = start_supervised!({ContinuousBatching, spec})
    wait_for(fn -> member?(peer, Transmuter.PG, __MODULE__.Streams) end)

    free? = fn ->
      state = :sys.get_state(server)
      state.rows == %{} and state.callers == %{}
    end

    local = __MODULE__.Streams |> ContinuousBatching.stream("Once upon a time") |> Enum.take(5)
    wait_for(free?)

    code = ~s|name \|> Transmuter.ContinuousBatching.stream("Once upon a time") \|> Enum.take(5)|

    assert {^local, _binding} =
             :erpc.call(peer, Code, :eval_string, [code, [name: __MODULE__.Streams]])

    wait_for(free?)

    assert {:exception, %ArgumentError{message: "invalid input, " <> _}, _stacktrace} =
             catch_error(:erpc.call(peer, ContinuousBatching, :stream, [__MODULE__.Streams, 42]))
  end

  # The server Busy has 1 row, which runs a long request, and 1 waiting request, so its room
  # is -1. The server Spare joins the group of the name Busy and has a room of 1. A call of
  # Busy from each node goes to Spare: the test node reads the rooms of both servers, and the
  # peer gets them from rooms/1 on the test node.
  test "a request goes to the server of the name with the most room", %{peer: peer} do
    source = {:hf, "HuggingFaceTB/SmolLM2-135M-Instruct"}
    long = [strategy: :greedy, max_new_tokens: 2000, min_new_tokens: 2000] ++ @opts
    spec = [name: __MODULE__.Busy, source: source, slots: 1] ++ long
    busy = start_supervised!(Supervisor.child_spec({ContinuousBatching, spec}, id: :busy))

    spawn(ContinuousBatching, :run, [__MODULE__.Busy, "Once upon a time"])
    wait_for(fn -> map_size(:sys.get_state(busy).rows) == 1 end)
    spawn(ContinuousBatching, :run, [__MODULE__.Busy, "Explain how a rainbow forms."])
    wait_for(fn -> :queue.len(:sys.get_state(busy).queue) == 1 end)

    spare = start_spare(peer, __MODULE__.Busy, source)
    assert ContinuousBatching.rooms([busy, spare]) == [{busy, -1}, {spare, 1}]

    alone = ContinuousBatching.run(spare, "Hello")
    assert ContinuousBatching.run(__MODULE__.Busy, "Hello") == alone

    for _call <- 1..3 do
      assert :erpc.call(peer, ContinuousBatching, :run, [__MODULE__.Busy, "Hello"]) == alone
    end
  end

  # The server Full has 2 free rows and a queue of 1 request. Its worker holds one request,
  # so its room is 1, but its queue is full. A call on the test node tries Full first, because
  # it is the server of this node and has room, gets a full queue and goes to Spare.
  test "a request goes to the next server of the name when a queue is full", %{peer: peer} do
    source = {:hf, "HuggingFaceTB/SmolLM2-135M-Instruct"}
    opts = [strategy: :greedy, max_new_tokens: 8] ++ @opts
    spec = [name: __MODULE__.Full, source: source, slots: 2, max_queue: 1] ++ opts
    full = start_supervised!(Supervisor.child_spec({ContinuousBatching, spec}, id: :full))
    %{worker: worker} = :sys.get_state(full)
    parent = self()
    id = make_ref()

    :telemetry.attach(
      id,
      [:transmuter, :continuous_batching, :prepare, :start],
      fn _event, _measurements, _metadata, _config ->
        if self() == worker do
          send(parent, {id, :preparing})
          receive do: ({^id, :go} -> :ok)
        end
      end,
      nil
    )

    spawn(ContinuousBatching, :run, [__MODULE__.Full, "Once upon a time"])
    assert_receive {^id, :preparing}, 60_000
    spare = start_spare(peer, __MODULE__.Full, source)
    assert ContinuousBatching.rooms([full]) == [{full, 1}]

    alone = ContinuousBatching.run(spare, "Hello")
    assert ContinuousBatching.run(__MODULE__.Full, "Hello") == alone

    send(worker, {id, :go})
    :telemetry.detach(id)
  end

  # A server of 1 row with 8 new tokens, which joins the group of `name`.
  defp start_spare(peer, name, source) do
    spec = [name: __MODULE__.Spare, source: source, slots: 1, max_new_tokens: 8] ++ @opts

    spare =
      start_supervised!(
        Supervisor.child_spec({ContinuousBatching, [strategy: :greedy] ++ spec}, id: :spare)
      )

    :ok = :pg.join(Transmuter.PG, name, spare)
    wait_for(fn -> length(:erpc.call(peer, :pg, :get_members, [Transmuter.PG, name])) == 2 end)
    spare
  end

  # librispeech_long is 62.46 s, three windows of Whisper, so its request has three rows. A
  # remote run joins the rows as a local run does, and a remote stream raises the ArgumentError
  # of an input of more than one row.
  test "a remote request of speech-to-text with three rows gives the result of a local call", %{
    peer: peer
  } do
    path = "test/fixtures/librispeech_long.f32"
    audio = Nx.from_binary(File.read!(path), :f32, backend: Nx.BinaryBackend)
    spec = [name: __MODULE__.Speech, source: {:hf, "Xenova/whisper-tiny.en"}, slots: 2]

    start_supervised!(
      {ContinuousBatching, [task: Transmuter.Audio.SpeechToText] ++ spec ++ @opts}
    )

    wait_for(fn -> member?(peer, Transmuter.PG, __MODULE__.Speech) end)

    local = ContinuousBatching.run(__MODULE__.Speech, audio, timeout: 120_000)

    remote =
      :erpc.call(peer, ContinuousBatching, :run, [__MODULE__.Speech, audio, [timeout: 120_000]])

    assert remote == local

    assert {:exception, %ArgumentError{message: message}, _stacktrace} =
             catch_error(
               :erpc.call(peer, ContinuousBatching, :stream, [__MODULE__.Speech, audio])
             )

    assert message =~ "one row, and this input gives 3 rows"
  end

  # The applications that `app` requires, and their required applications, without the
  # optional ones.
  defp required_apps(app, seen) do
    if app in seen do
      seen
    else
      spec = Application.spec(app)
      required = spec[:applications] -- Keyword.get(spec, :optional_applications, [])
      Enum.reduce(required, [app | seen], &required_apps/2)
    end
  end

  defp member?(node, scope, name), do: :erpc.call(node, :pg, :get_members, [scope, name]) != []

  defp wait_for(done?, timeout \\ 10_000) do
    cond do
      done?.() ->
        :ok

      timeout <= 0 ->
        flunk("the condition is false after 10 s")

      true ->
        Process.sleep(10)
        wait_for(done?, timeout - 10)
    end
  end
end
