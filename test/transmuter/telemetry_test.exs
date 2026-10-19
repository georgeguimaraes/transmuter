defmodule Transmuter.TelemetryTest do
  use ExUnit.Case, async: true

  @device (System.get_env("TRANSMUTER_TEST_DEVICE") ||
             if(match?({:unix, :darwin}, :os.type()), do: "gpu", else: "cpu"))
          |> String.to_existing_atom()
  @backend {EMLX.Backend, device: @device}
  @compiler [compiler: EMLX, device: @device]

  @events [
    [:transmuter, :compile, :stop],
    [:transmuter, :cache, :hit],
    [:transmuter, :cache, :miss],
    [:transmuter, :cache, :evict],
    [:transmuter, :segment, :compile, :stop],
    [:transmuter, :break, :stop]
  ]

  # Handlers are global and other tests run at the same time, so the handler forwards
  # only the events of the test process.
  setup do
    id = make_ref()
    :telemetry.attach_many(id, @events, &__MODULE__.forward/4, self())
    on_exit(fn -> :telemetry.detach(id) end)
  end

  def forward(event, _measurements, metadata, pid) do
    if self() == pid, do: send(pid, {event, metadata})
  end

  test "a graph break emits the break and the cache events of each segment" do
    model = load_fixture("Break_Range_long_tail")
    input = %{"x" => Nx.tensor([1.0, 1, 1, 1, 1, 1], backend: @backend)}
    compiled = Transmuter.compile(model, input, @compiler)

    assert_received {[:transmuter, :cache, :miss], %{position: 0}}
    assert_received {[:transmuter, :segment, :compile, :stop], %{position: 0, stop: break}}
    assert_received {[:transmuter, :compile, :stop], %{graph_break: true, compiler: EMLX}}
    after_break = break + 1

    compiled.(input)
    assert_received {[:transmuter, :cache, :hit], %{position: 0}}
    assert_received {[:transmuter, :break, :stop], %{position: ^break, op_type: "Range"}}
    assert_received {[:transmuter, :cache, :miss], %{position: ^after_break}}

    assert_received {[:transmuter, :segment, :compile, :stop],
                     %{position: ^after_break, stop: :end, compiled: true}}

    compiled.(input)
    assert_received {[:transmuter, :cache, :hit], %{position: ^after_break}}
    refute_received {[:transmuter, :cache, :miss], _metadata}
  end

  test "a new shape at a full position evicts the least recently used segment" do
    model = load_fixture("Break_Range_long_tail")
    input = &%{"x" => Nx.tensor([&1, 1, 1, 1, 1, 1], type: :f32, backend: @backend)}
    compiled = Transmuter.compile(model, input.(1), [cache_size: 2] ++ @compiler)

    compiled.(input.(1))
    assert_received {[:transmuter, :break, :stop], %{position: break}}
    after_break = break + 1
    assert_received {[:transmuter, :cache, :miss], %{position: ^after_break, shapes: oldest}}

    compiled.(input.(2))
    refute_received {[:transmuter, :cache, :evict], _metadata}

    compiled.(input.(3))

    assert_received {[:transmuter, :cache, :evict],
                     %{position: ^after_break, shapes: ^oldest, cache_size: 2}}
  end

  test "a graph without breaks uses the cache only for other shapes" do
    model = load_fixture("Cosh")
    input = &%{"x" => Nx.iota(&1, type: :f32, backend: @backend)}
    compiled = Transmuter.compile(model, input.({3, 5}), @compiler)
    assert_received {[:transmuter, :compile, :stop], %{graph_break: false}}
    assert_received {[:transmuter, :cache, :miss], %{position: 0}}

    compiled.(input.({3, 5}))
    refute_received {[:transmuter, :cache, _result], _metadata}

    compiled.(input.({2, 7}))

    assert_received {[:transmuter, :cache, :miss],
                     %{position: 0, shapes: [{"x", {{2, 7}, _type}}]}}
  end

  defp load_fixture(name) do
    Transmuter.load!(Path.expand("../fixtures/ops/#{name}/model.onnx", __DIR__),
      backend: @backend
    )
  end
end
