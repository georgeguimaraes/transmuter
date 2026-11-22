defmodule Transmuter.ContinuousBatching.Costs do
  @moduledoc """
  The costs that a `Transmuter.ContinuousBatching` server measures, and the rule that decides when a shrink of its buffers pays.

  A decode call takes about `a + steps * (b + slope * capacity + c * live_keys)`: a fixed cost for the call, and for each step a cost for the rows of the buffers and a cost for the live keys, the rows of the running rows. Without the cuDNN key lengths, a step reads the whole buffers, so the slope is large. With them, it reads only the live keys, so the slope is about 0. The server fits the four terms by least squares over its decode calls: the calls without a prompt part and without a move of the buffers, and not the first call of each shape, which compiles. The live keys are a term of their own, because the buffers are large when a long row runs, so a mean step time for each capacity would also count the longer rows.

  A move of the buffers copies them one capacity step at a time, and its time grows with the capacity. So a move costs `per_row` milliseconds for each row of each capacity on its path.

  A shrink from `from` to `to` rows pays when the time that the `steps` steps save at `to`, with the slope less twice its standard error, is more than twice the time of the move, for the move and the probable move back up. So a flat step time never moves, and a slope from few calls, early in a server, moves only when it is clear. The server keeps one fit for 16 calls, and when a cost is not known yet, the shrink does not pay.
  """

  defstruct n: 0,
            xtx: List.duplicate(List.duplicate(0.0, 4), 4),
            xty: List.duplicate(0.0, 4),
            yty: 0.0,
            seen: MapSet.new(),
            move_ms: 0.0,
            move_rows: 0,
            fit: nil,
            fixed: nil

  # The fewest decode calls for a fit with a standard error, and the calls between two fits.
  @min_calls 16
  @refit 16

  @doc """
  The costs of a server. `nil` measures them. A map with `:slope` and `:per_row`, in milliseconds for each row of the buffers, fixes them, with no error.
  """
  def new(nil), do: %__MODULE__{}
  def new(%{slope: slope, per_row: per_row}), do: %__MODULE__{fixed: {slope, per_row}}

  @doc """
  Adds a decode call of `steps` steps at `capacity` rows, with `live_keys` rows in the running rows at its start, that took `ms` milliseconds. The first call of each capacity and step count compiles, so it does not count.
  """
  def sample(costs, capacity, steps, live_keys, ms) do
    if MapSet.member?(costs.seen, {capacity, steps}) do
      x = [1.0, steps * 1.0, steps * capacity * 1.0, steps * live_keys * 1.0]

      costs = %{
        costs
        | n: costs.n + 1,
          xtx:
            Enum.zip_with(costs.xtx, x, fn row, xi -> Enum.zip_with(row, x, &(&1 + xi * &2)) end),
          xty: Enum.zip_with(costs.xty, x, &(&1 + &2 * ms)),
          yty: costs.yty + ms * ms
      }

      if rem(costs.n, @refit) == 0, do: %{costs | fit: fit(costs)}, else: costs
    else
      %{costs | seen: MapSet.put(costs.seen, {capacity, steps})}
    end
  end

  @doc """
  Adds a move of the buffers through capacities that sum to `rows` rows, which took `ms` milliseconds.
  """
  def moved(costs, rows, ms),
    do: %{costs | move_ms: costs.move_ms + ms, move_rows: costs.move_rows + rows}

  @doc """
  Whether a shrink from `from` rows to `to` rows pays, before `steps` steps that the group runs at `to` rows, with a move through the capacities of `path`.
  """
  def pays?(costs, from, to, steps, path) do
    case estimate(costs) do
      {slope, error, per_row} ->
        (slope - 2 * error) * (from - to) * steps > 2 * per_row * Enum.sum(path)

      nil ->
        false
    end
  end

  @doc """
  The slope, its standard error and the move cost for each row, or `nil` when they are not known: before #{@min_calls} decode calls at two capacities or more, and before the first move. The fit is that of the last multiple of #{@refit} calls.
  """
  def estimate(%__MODULE__{fixed: {slope, per_row}}), do: {slope, 0.0, per_row}

  def estimate(costs = %__MODULE__{fit: {slope, error}, move_rows: rows}) when rows > 0,
    do: {slope, error, costs.move_ms / rows}

  def estimate(_costs), do: nil

  # A server whose calls all have one step count, such as EMLX without decode loops, cannot
  # tell the fixed cost of a call from the base cost of a step. A small weight on the fixed cost
  # keeps it at about 0 then, and costs nothing when the step counts differ. The slope does not
  # depend on it.
  defp fit(costs) do
    capacities = costs.seen |> Enum.map(&elem(&1, 0)) |> Enum.uniq() |> length()
    if costs.n >= @min_calls and capacities >= 2, do: solve(costs)
  end

  defp solve(costs) do
    prior = List.update_at(costs.xtx, 0, &List.update_at(&1, 0, fn n -> n + 1.0 end))

    # Calls that do not tell the terms apart, as early in a server, give no inverse.
    with inverse when inverse != nil <- invert(prior) do
      beta = Enum.map(inverse, &dot(&1, costs.xty))
      variance = max(costs.yty - dot(beta, costs.xty), 0.0) / (costs.n - 4)
      {Enum.at(beta, 2), :math.sqrt(max(variance * Enum.at(Enum.at(inverse, 2), 2), 0.0))}
    end
  end

  defp dot(a, b), do: a |> Enum.zip_with(b, &(&1 * &2)) |> Enum.sum()

  # Gauss-Jordan elimination with partial pivoting, or nil for a singular matrix.
  defp invert(matrix) do
    size = length(matrix)
    tolerance = 1.0e-12 * (matrix |> List.flatten() |> Enum.map(&abs/1) |> Enum.max())
    identity = for i <- 1..size, do: for(j <- 1..size, do: if(i == j, do: 1.0, else: 0.0))

    0..(size - 1)
    |> Enum.reduce_while(Enum.zip_with(matrix, identity, &(&1 ++ &2)), fn k, rows ->
      {pivot, i} =
        rows |> Enum.with_index() |> Enum.drop(k) |> Enum.max_by(&abs(Enum.at(elem(&1, 0), k)))

      if abs(Enum.at(pivot, k)) <= tolerance do
        {:halt, nil}
      else
        pivot = Enum.map(pivot, &(&1 / Enum.at(pivot, k)))
        rows = rows |> List.replace_at(i, Enum.at(rows, k)) |> List.replace_at(k, pivot)
        {:cont, Enum.with_index(rows, &if(&2 == k, do: &1, else: eliminate(&1, pivot, k)))}
      end
    end)
    |> then(&(&1 && Enum.map(&1, fn row -> Enum.drop(row, size) end)))
  end

  defp eliminate(row, pivot, k) do
    factor = Enum.at(row, k)
    Enum.zip_with(row, pivot, &(&1 - factor * &2))
  end
end
