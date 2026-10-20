defmodule Transmuter.Ops.Random do
  @moduledoc """
  Random operators on Nx.Random. Each run has one key, and each node folds a hash of its first output name into it, so nodes draw independent streams. A node with a seed draws the same values on every run. Values cannot match ONNX Runtime, whose generator differs.
  """

  import Transmuter.Ops, only: [attr: 2, attr: 3, dtype: 1]

  @op_types ~w(RandomNormal RandomNormalLike RandomUniform RandomUniformLike Bernoulli Multinomial)

  @doc false
  def op_types, do: @op_types

  @doc false
  def apply(node, inputs, key) do
    key =
      case attr(node, "seed") do
        nil -> Nx.Random.fold_in(key, :erlang.phash2(hd(node.output)))
        seed -> Nx.Random.key(trunc(seed))
      end

    [op(node, inputs, key)]
  end

  defp op(node = %{op_type: "RandomNormal"}, [], key),
    do: normal(node, key, List.to_tuple(attr(node, "shape")), dtype(attr(node, "dtype", 1)))

  defp op(node = %{op_type: "RandomNormalLike"}, [x], key),
    do: normal(node, key, Nx.shape(x), like_type(node, x))

  defp op(node = %{op_type: "RandomUniform"}, [], key),
    do: uniform(node, key, List.to_tuple(attr(node, "shape")), dtype(attr(node, "dtype", 1)))

  defp op(node = %{op_type: "RandomUniformLike"}, [x], key),
    do: uniform(node, key, Nx.shape(x), like_type(node, x))

  # The uniform draw is at least f32, because f16 values below p are too coarse for a
  # small p.
  defp op(node = %{op_type: "Bernoulli"}, [p], key) do
    type = Nx.Type.merge(Nx.type(p), {:f, 32})
    {u, _key} = Nx.Random.uniform(key, shape: Nx.shape(p), type: type)
    u |> Nx.less(Nx.as_type(p, type)) |> Nx.as_type(like_type(node, p))
  end

  # Gumbel-max: the argmax of the log-probabilities plus Gumbel noise is a sample of
  # their softmax, so the logits need no normalization.
  defp op(node = %{op_type: "Multinomial"}, [x], key) do
    {batch, classes} = Nx.shape(x)
    samples = attr(node, "sample_size", 1)
    {noise, _key} = Nx.Random.gumbel(key, shape: {batch, samples, classes}, type: Nx.type(x))

    x
    |> Nx.new_axis(1)
    |> Nx.add(noise)
    |> Nx.argmax(axis: 2)
    |> Nx.as_type(dtype(attr(node, "dtype", 6)))
  end

  defp normal(node, key, shape, type) do
    mean = attr(node, "mean", 0.0)
    scale = attr(node, "scale", 1.0)
    key |> Nx.Random.normal(mean, scale, shape: shape, type: type) |> elem(0)
  end

  defp uniform(node, key, shape, type) do
    low = attr(node, "low", 0.0)
    high = attr(node, "high", 1.0)
    key |> Nx.Random.uniform(low, high, shape: shape, type: type) |> elem(0)
  end

  defp like_type(node, x) do
    case attr(node, "dtype") do
      nil -> Nx.type(x)
      dt -> dtype(dt)
    end
  end
end
