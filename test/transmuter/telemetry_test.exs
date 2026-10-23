defmodule Transmuter.TelemetryTest do
  use ExUnit.Case, async: true

  @backend Transmuter.TestBackend.backend()
  @compiler Transmuter.TestBackend.compiler_opts()

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

    assert_received {[:transmuter, :segment, :compile, :stop],
                     %{position: 0, stop: break, reason: "a value that decides a shape"}}

    compiler = @compiler[:compiler]
    assert_received {[:transmuter, :compile, :stop], %{graph_break: true, compiler: ^compiler}}
    after_break = break + 1

    compiled.(input)
    assert_received {[:transmuter, :cache, :hit], %{position: 0}}
    assert_received {[:transmuter, :break, :stop], %{position: ^break, op_type: "Range"}}
    assert_received {[:transmuter, :cache, :miss], %{position: ^after_break}}

    assert_received {[:transmuter, :segment, :compile, :stop],
                     %{position: ^after_break, stop: :end, compiled: true, reason: nil}}

    compiled.(input)
    assert_received {[:transmuter, :cache, :hit], %{position: ^after_break}}
    refute_received {[:transmuter, :cache, :miss], _metadata}
  end

  # The segment after the Range has 2 nodes, fewer than the 20 that compile.
  test "a short segment after a break runs eagerly" do
    model = load_fixture("Break_Range_computed_limit")
    input = %{"x" => Nx.tensor([1.0, 1, 1, 0.5, 0.25, 0.25], backend: @backend)}
    compiled = Transmuter.compile(model, input, @compiler)
    assert_received {[:transmuter, :segment, :compile, :stop], %{position: 0, stop: break}}
    after_break = break + 1

    for _call <- 1..2 do
      %{"y" => y} = compiled.(input)
      assert Nx.shape(y) == {4}
    end

    assert_received {[:transmuter, :segment, :compile, :stop],
                     %{position: ^after_break, stop: :end, compiled: false}}

    assert_received {[:transmuter, :cache, :hit], %{position: ^after_break}}
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

  # The Loop body reads the sizes as an outer sequence. They become static at the break of
  # the Loop, so the bounds of both Slices fold, and the Slices do not break.
  test "a break makes the small outer values of its subgraphs static" do
    model = load_fixture("Break_Loop_outer_sequence")
    sizes = Nx.tensor([2, 3, 1], backend: @backend)
    x = Nx.tensor([1.0, 2, 3, 4, 5, 6], backend: @backend)

    %{"y" => y} =
      Transmuter.compile(model, %{"sizes" => sizes, "x" => x}, @compiler).(%{
        "sizes" => sizes,
        "x" => x
      })

    assert Nx.to_flat_list(y) == [3.0, 4.0, 5.0, 6.0]
    assert_received {[:transmuter, :break, :stop], %{op_type: "Loop"}}
    refute_received {[:transmuter, :break, :stop], _metadata}
  end

  # The fixture folds two values of 6400 elements from the shape of x, and a NonZero reads
  # each of them. They leave the host, but the graph has no break, and an eager run gives the
  # same values on the backend of the model. The compiled function computes the
  # ConstantOfShape on the device, and the NonZero reads it on the host.
  test "a large static value leaves the host without a graph break" do
    model = load_fixture("Fold_large_static")
    input = %{"x" => Nx.iota({80, 80}, type: :f32, backend: @backend)}
    compiled = Transmuter.compile(model, input, @compiler).(input)
    assert_received {[:transmuter, :compile, :stop], %{graph_break: false}}
    refute_received {[:transmuter, :break, :stop], _metadata}

    eager = Transmuter.run(model, model.params, input)
    assert Nx.to_flat_list(compiled["y"]) == Nx.to_flat_list(eager["y"])
    assert Nx.to_flat_list(compiled["nz"]) == Enum.to_list(1..6399)
    assert Nx.to_flat_list(eager["nz"]) == Enum.to_list(1..6399)
    assert Nx.to_flat_list(compiled["nz_fill"]) == Enum.to_list(0..6399)
    assert Nx.to_flat_list(eager["nz_fill"]) == Enum.to_list(0..6399)
  end

  # NonZero gives the positions of the positive x. Gather reads them on the device, and the
  # Range of Break_NonZero_Range breaks on its limit, which comes from the device. So the
  # positions go into the next segment as arguments, and other positions of the same count
  # compile nothing new.
  test "positions that no node reads on the host are not in the key of a segment" do
    for name <- ["Break_NonZero_gather", "Break_NonZero_Range"] do
      model = load_fixture(name)
      compiled = Transmuter.compile(model, positions(model, [1, 1, 1, 0, 0, 0, 0, 0]), @compiler)
      calls(compiled, model, [[1, 1, 1, 0, 0, 0, 0, 0]])

      assert calls(compiled, model, [[0, 1, 0, 1, 0, 1, 0, 0], [1, 0, 0, 0, 0, 1, 1, 0]]) == [
               %{compiled: 0, breaks: ["NonZero" | tl(model_breaks(name))], host_values: []},
               %{compiled: 0, breaks: ["NonZero" | tl(model_breaks(name))], host_values: []}
             ],
             name
    end
  end

  # The segment after the NonZero computes the first position, and the segment after the
  # Range reads it as a Slice start. The first position enters that segment as a static
  # value, so only that segment compiles for a new first position, and the segment of the
  # positions does not.
  test "a value that a later segment reads on the host is static in that segment only" do
    model = load_fixture("Break_NonZero_cross_segment")
    compiled = Transmuter.compile(model, positions(model, [1, 1, 1, 0, 0, 0, 0, 0]), @compiler)

    assert [%{breaks: ["NonZero", "Range"], host_values: ["first"]}] =
             calls(compiled, model, [[1, 1, 1, 0, 0, 0, 0, 0]])

    assert [
             %{compiled: 0, breaks: ["NonZero", "Range"], host_values: []},
             %{compiled: 1, breaks: ["NonZero", "Range"], host_values: []}
           ] = calls(compiled, model, [[1, 0, 1, 0, 1, 0, 0, 0], [0, 1, 1, 1, 0, 0, 0, 0]])
  end

  # A short segment reads the first position on the host and runs eagerly. It gets the
  # positions after a NonZero, or the first position from a compiled segment after a Range.
  # The compiled segments read the positions on the device, so they stay their arguments.
  # New positions with the same first position add no break and compile nothing.
  test "a segment that runs eagerly does not make the values of other segments static" do
    for name <- ["Break_NonZero_eager_read", "Break_NonZero_eager_cross"] do
      model = load_fixture(name)
      compiled = Transmuter.compile(model, positions(model, [1, 1, 1, 0, 0, 0, 0, 0]), @compiler)
      calls(compiled, model, [[1, 1, 1, 0, 0, 0, 0, 0]])

      assert [
               %{compiled: 0, breaks: ["NonZero", "Range"]},
               %{compiled: 0, breaks: ["NonZero", "Range"]}
             ] = calls(compiled, model, [[1, 0, 1, 0, 1, 0, 0, 0], [1, 0, 0, 1, 1, 0, 0, 0]]),
             name
    end
  end

  # A Slice reads a value that comes from the positions through a random node, or from a
  # value of 6000 elements, more than the host takes, in a later segment or in the same
  # segment. In the same segment, a trace with static positions still breaks, so the break
  # is real. All break as before, and the segment of the positions does not compile for
  # new positions.
  test "a break on a random or large value keeps the positions as arguments" do
    for {name, breaks} <- [
          {"Break_NonZero_random_read", ["NonZero", "Slice"]},
          {"Break_NonZero_large_read", ["NonZero", "Range", "Slice"]},
          {"Break_NonZero_large_local", ["NonZero", "Slice"]}
        ] do
      model = load_fixture(name)
      compiled = Transmuter.compile(model, positions(model, [1, 0, 1, 1, 0, 0, 0, 0]), @compiler)

      assert [
               %{breaks: ^breaks, host_values: []},
               %{compiled: 0, breaks: ^breaks, host_values: []}
             ] = calls(compiled, model, [[1, 0, 1, 1, 0, 0, 0, 0], [0, 1, 1, 1, 0, 0, 0, 0]]),
             name
    end
  end

  # The Loop body expands to a position that it reads as an outer value, so the value that
  # it reads on the host is not an input of the Loop. Its state comes from the device. The
  # positions become static for one more trace, and the Loop does not break.
  test "a node that reads a value that it computed from positions does not break" do
    model = load_fixture("Break_NonZero_Loop_outer")
    compiled = Transmuter.compile(model, positions(model, [1, 1, 1, 0, 0, 0, 0, 0]), @compiler)

    assert [%{breaks: ["NonZero"]}, %{breaks: ["NonZero"]}] =
             calls(compiled, model, [[1, 1, 1, 0, 0, 0, 0, 0], [0, 1, 0, 1, 0, 1, 0, 0]])
  end

  # The ScatterElements also computes its coordinates from the positions. With static
  # positions it still breaks, because its coordinates leave the host, as before the
  # positions were arguments. So the trace with static positions gives the break back, and
  # the positions stay arguments.
  test "a node that still breaks with static positions keeps them arguments" do
    model = load_fixture("Break_NonZero_scatter_max")
    compiled = Transmuter.compile(model, positions(model, [1, 1, 1, 0, 0, 0, 0, 0]), @compiler)

    assert [
             %{breaks: ["NonZero", "ScatterElements"], host_values: []},
             %{breaks: ["NonZero", "ScatterElements"], host_values: []}
           ] = calls(compiled, model, [[1, 1, 1, 0, 0, 0, 0, 0], [0, 1, 0, 1, 0, 1, 0, 0]])
  end

  defp model_breaks("Break_NonZero_gather"), do: ["NonZero"]
  defp model_breaks("Break_NonZero_Range"), do: ["NonZero", "Range"]

  defp positions(model, x) do
    Map.take(
      %{
        "x" => Nx.tensor(x, type: :f32, backend: @backend),
        "data" => Nx.iota({8, 4}, type: :f32, backend: @backend) |> Nx.divide(8),
        "n" => Nx.tensor(5, backend: @backend),
        "v0" => Nx.tensor([0.5, -0.5], backend: @backend),
        "w" => Nx.iota({8}, type: :f32, backend: @backend)
      },
      model.inputs
    )
  end

  # Calls compiled for each x, checks the outputs against an eager run and gives the
  # compiled segments, the breaks and the names that joined the host values of each call.
  defp calls(compiled, model, xs) do
    for x <- xs do
      inputs = positions(model, x)
      outputs = compiled.(inputs)

      for {name, expected} <- Transmuter.run(model, model.params, inputs),
          do: assert(Nx.to_number(Nx.all_close(Nx.backend_copy(outputs[name]), expected)) == 1)

      drain(%{compiled: 0, breaks: [], host_values: []})
    end
  end

  defp drain(acc) do
    receive do
      {[:transmuter, :segment, :compile, :stop], meta} ->
        drain(%{
          acc
          | compiled: acc.compiled + if(meta.compiled, do: 1, else: 0),
            host_values: acc.host_values ++ meta.host_values
        })

      {[:transmuter, :break, :stop], meta} ->
        drain(%{acc | breaks: acc.breaks ++ [meta.op_type]})

      {_event, _meta} ->
        drain(acc)
    after
      0 -> acc
    end
  end

  defp load_fixture(name) do
    Transmuter.load!(Path.expand("../fixtures/ops/#{name}/model.onnx", __DIR__),
      backend: @backend
    )
  end
end
