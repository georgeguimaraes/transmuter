defmodule Transmuter.ContinuousBatching.CostsTest do
  use ExUnit.Case, async: true

  alias Transmuter.ContinuousBatching.Costs

  @path [768, 640, 512]

  # Decode calls of 1, 2, 4 and 8 steps at each capacity, with live keys that do not follow
  # the capacity, a fixed cost of 2 ms for each call, and a small noise. A move costs 0.02 ms
  # for each row that it copies. The first call of each shape compiles and does not count.
  defp measured(slope) do
    calls =
      for capacity <- [256, 384, 512, 640, 768, 896],
          {steps, i} <- Enum.with_index(Stream.cycle([1, 2, 4, 8]) |> Enum.take(24)),
          do: {capacity, steps, 500 + rem(i * 389, 2500), rem(i * 7, 5) * 0.04 - 0.08}

    calls
    |> Enum.reduce(Costs.new(nil), fn {capacity, steps, live, noise}, costs ->
      ms = 2.0 + steps * (3.0 + slope * capacity + 0.0004 * live) + noise
      Costs.sample(costs, capacity, steps, live, ms)
    end)
    |> Costs.moved(1920, 38.4)
  end

  test "a shrink pays only for a slope above its noise and enough steps, and never on unknown costs" do
    assert Costs.pays?(measured(0.006), 896, 512, 256, @path)
    refute Costs.pays?(measured(0.006), 896, 512, 16, @path)
    refute Costs.pays?(measured(0.0), 896, 512, 256, @path)
    refute Costs.pays?(Costs.moved(Costs.new(nil), 1920, 38.4), 896, 512, 256, @path)
    refute Costs.pays?(measured(0.006) |> Map.put(:move_rows, 0), 896, 512, 256, @path)
    assert Costs.pays?(Costs.new(%{slope: 1.0, per_row: 0.0}), 896, 512, 1, @path)
  end
end
