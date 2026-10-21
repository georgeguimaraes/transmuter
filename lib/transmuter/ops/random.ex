defmodule Transmuter.Ops.Random do
  @moduledoc """
  Random operators on `Transmuter.Random`, which gives the values of `Nx.Random`. Each run has one key, and each node folds a hash of its first output name into it, so nodes draw independent streams. A node with a seed draws the same values on every run. Values cannot match ONNX Runtime, whose generator differs. A model without noise has no key, and each node gives a fixed value (see the `:noise` option of `Transmuter.load!/2`).
  """

  import Transmuter.Ops, only: [attr: 2, attr: 3, dtype: 1]

  @op_types ~w(RandomNormal RandomNormalLike RandomUniform RandomUniformLike Bernoulli Multinomial)

  @doc false
  def op_types, do: @op_types

  @doc false
  def apply(node, inputs, nil), do: [fixed(node, inputs)]

  def apply(node, inputs, key) do
    key =
      case attr(node, "seed") do
        nil -> Transmuter.Random.fold_in(key, :erlang.phash2(hd(node.output)))
        seed -> Nx.Random.key(trunc(seed))
      end

    [op(node, inputs, key)]
  end

  defp op(node = %{op_type: "RandomNormal"}, [], key),
    do: normal(node, key, shape(node), dtype(attr(node, "dtype", 1)))

  defp op(node = %{op_type: "RandomNormalLike"}, [x], key),
    do: normal(node, key, Nx.shape(x), like_type(node, x))

  defp op(node = %{op_type: "RandomUniform"}, [], key),
    do: uniform(node, key, shape(node), dtype(attr(node, "dtype", 1)))

  defp op(node = %{op_type: "RandomUniformLike"}, [x], key),
    do: uniform(node, key, Nx.shape(x), like_type(node, x))

  # The uniform draw is at least f32, because f16 values below p are too coarse for a
  # small p.
  defp op(node = %{op_type: "Bernoulli"}, [p], key) do
    type = Nx.Type.merge(Nx.type(p), {:f, 32})
    u = Transmuter.Random.uniform(key, 0.0, 1.0, shape: Nx.shape(p), type: type)
    u |> Nx.less(Nx.as_type(p, type)) |> Nx.as_type(like_type(node, p))
  end

  # Gumbel-max: the argmax of the log-probabilities plus Gumbel noise is a sample of
  # their softmax, so the logits need no normalization.
  defp op(node = %{op_type: "Multinomial"}, [x], key) do
    {batch, classes} = Nx.shape(x)
    samples = attr(node, "sample_size", 1)
    noise = Transmuter.Random.gumbel(key, shape: {batch, samples, classes}, type: Nx.type(x))

    x
    |> Nx.new_axis(1)
    |> Nx.add(noise)
    |> Nx.argmax(axis: 2)
    |> Nx.as_type(dtype(attr(node, "dtype", 6)))
  end

  # The values of a model without noise, see the :noise option of Transmuter.load!/2.
  defp fixed(node = %{op_type: "RandomNormal"}, []),
    do: fill(attr(node, "mean", 0.0), shape(node), dtype(attr(node, "dtype", 1)))

  defp fixed(node = %{op_type: "RandomNormalLike"}, [x]),
    do: fill(attr(node, "mean", 0.0), Nx.shape(x), like_type(node, x))

  defp fixed(node = %{op_type: "RandomUniform"}, []),
    do: fill(attr(node, "high", 1.0), shape(node), dtype(attr(node, "dtype", 1)))

  defp fixed(node = %{op_type: "RandomUniformLike"}, [x]),
    do: fill(attr(node, "high", 1.0), Nx.shape(x), like_type(node, x))

  defp fixed(node = %{op_type: "Bernoulli"}, [p]),
    do: p |> Nx.greater(0) |> Nx.as_type(like_type(node, p))

  defp fixed(node = %{op_type: "Multinomial"}, [x]) do
    {batch, _classes} = Nx.shape(x)

    x
    |> Nx.argmax(axis: 1, keep_axis: true)
    |> Nx.broadcast({batch, attr(node, "sample_size", 1)})
    |> Nx.as_type(dtype(attr(node, "dtype", 6)))
  end

  defp shape(node), do: List.to_tuple(attr(node, "shape"))
  defp fill(value, shape, type), do: Nx.broadcast(Nx.tensor(value, type: type), shape)

  defp normal(node, key, shape, type) do
    mean = attr(node, "mean", 0.0)
    scale = attr(node, "scale", 1.0)
    Transmuter.Random.normal(key, mean, scale, shape: shape, type: type)
  end

  defp uniform(node, key, shape, type) do
    low = attr(node, "low", 0.0)
    high = attr(node, "high", 1.0)
    Transmuter.Random.uniform(key, low, high, shape: shape, type: type)
  end

  defp like_type(node, x) do
    case attr(node, "dtype") do
      nil -> Nx.type(x)
      dt -> dtype(dt)
    end
  end
end
