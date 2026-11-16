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
        Process.info(pid, :initial_call) == {:initial_call, {ContinuousBatching, :__proxy__, 5}}
      end)

    Process.exit(proxy, :kill)

    assert {{:caught, {:killed, {ContinuousBatching, :run, _args}}}, _binding} =
             Task.await(caller, 60_000)

    wait_for(fn ->
      state = :sys.get_state(server)
      state.rows == %{} and state.callers == %{}
    end)
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
