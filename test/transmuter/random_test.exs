defmodule Transmuter.RandomTest do
  use ExUnit.Case, async: true

  # An odd size pads the counters, and 16-bit types cast the bits. A vectorized key is the
  # key of each row in generation.
  test "the draws are the draws of Nx.Random" do
    key = Nx.Random.key(42)
    keys = Nx.stack([key, Nx.Random.key(7)]) |> Nx.vectorize(:rows)

    for {name, ours, theirs} <- [
          {"fold_in", Transmuter.Random.fold_in(key, 99), Nx.Random.fold_in(key, 99)},
          {"fold_in of a tensor",
           Transmuter.Random.fold_in(key, Nx.tensor([[1, 2 ** 40], [3, 4]])),
           Nx.Random.fold_in(key, Nx.tensor([[1, 2 ** 40], [3, 4]]))},
          {"fold_in of vectorized keys", Transmuter.Random.fold_in(keys, 5),
           Nx.Random.fold_in(keys, 5)},
          {"uniform, even size",
           Transmuter.Random.uniform(key, 0.0, 1.0, shape: {4, 6}, type: :f32),
           elem(Nx.Random.uniform(key, 0.0, 1.0, shape: {4, 6}, type: :f32), 0)},
          {"uniform, odd size",
           Transmuter.Random.uniform(key, -2.0, 3.0, shape: {3, 5}, type: :f32),
           elem(Nx.Random.uniform(key, -2.0, 3.0, shape: {3, 5}, type: :f32), 0)},
          {"uniform f16", Transmuter.Random.uniform(key, 0.0, 1.0, shape: {7}, type: :f16),
           elem(Nx.Random.uniform(key, 0.0, 1.0, shape: {7}, type: :f16), 0)},
          {"uniform of vectorized keys",
           Transmuter.Random.uniform(keys, 1.0e-10, 1.0, shape: {5}, type: :f32),
           elem(Nx.Random.uniform(keys, 1.0e-10, 1.0, shape: {5}), 0)},
          {"normal", Transmuter.Random.normal(key, 0.5, 2.0, shape: {3, 3}, type: :f32),
           elem(Nx.Random.normal(key, 0.5, 2.0, shape: {3, 3}, type: :f32), 0)},
          {"normal f16", Transmuter.Random.normal(key, 0.0, 1.0, shape: {5}, type: :f16),
           elem(Nx.Random.normal(key, 0.0, 1.0, shape: {5}, type: :f16), 0)},
          {"gumbel", Transmuter.Random.gumbel(key, shape: {2, 3}, type: :f32),
           elem(Nx.Random.gumbel(key, shape: {2, 3}, type: {:f, 32}), 0)}
        ] do
      assert ours == theirs, name
    end
  end
end
