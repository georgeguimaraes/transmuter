defmodule Transmuter.Vision.BackgroundRemoval do
  @moduledoc """
  The background removal task. See `Transmuter.Vision.background_removal/2`.

  The featurizer gives one image size, so the images go into batches. The serving turns the first output of the model into a mask of 8 bits at the size of the model. The mask has the size of the input image, so the client resizes the mask of each image on the `:backend` of the options.
  """

  alias Transmuter.{Featurizer, Serving}

  # transformers.js applies the sigmoid when a value is farther than this from 0..1.
  @epsilon 1.0e-5

  @doc false
  def background_removal(source, opts) do
    opts = Keyword.validate!(opts, Serving.image_options())
    {repo, model} = Serving.load_model!(source, [outputs: &Enum.take(&1, 1)] ++ opts, :fp32)
    output = hd(model.outputs)
    backend = opts[:backend] || Nx.default_backend()

    Serving.image(
      model,
      opts[:featurizer] || Featurizer.load!(repo),
      opts,
      fn outputs, _inputs -> %{mask: mask(outputs[output])} end,
      fn [%{mask: mask}], [image] ->
        {height, width, _channels} = Nx.shape(image)

        mask =
          mask
          |> Nx.backend_transfer(backend)
          |> Nx.transpose(axes: [1, 2, 0])
          |> Featurizer.resize({height, width}, :bilinear)
          |> Nx.squeeze(axes: [2])

        %{mask: Nx.backend_transfer(mask, Nx.BinaryBackend)}
      end
    )
  end

  # As the BackgroundRemovalPipeline of transformers.js: a sigmoid for each image whose
  # output has a value outside 0..1, then 255 times the value, truncated to 8 bits.
  defp mask(alphas) do
    axes = Enum.to_list(1..(Nx.rank(alphas) - 1))
    outside = Nx.logical_or(Nx.less(alphas, -@epsilon), Nx.greater(alphas, 1 + @epsilon))
    sigmoid? = Nx.any(outside, axes: axes, keep_axes: true)

    sigmoid?
    |> Nx.broadcast(alphas)
    |> Nx.select(Nx.sigmoid(alphas), alphas)
    |> Nx.multiply(255)
    |> Nx.as_type(:u8)
  end
end
