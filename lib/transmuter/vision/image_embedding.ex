defmodule Transmuter.Vision.ImageEmbedding do
  @moduledoc """
  The image embedding task. See `Transmuter.Vision.image_embedding/2`.
  """

  alias Transmuter.{Featurizer, Serving}

  @outputs ~w(image_embeds pooler_output last_hidden_state)

  @doc false
  def image_embedding(source, opts) do
    opts = Keyword.validate!(opts, Serving.image_options() ++ [:embedding_processor])
    {repo, model} = Serving.load_model!(source, opts, :fp32, ["vision_model", "model"])

    output =
      Enum.find(@outputs, &(&1 in model.outputs)) ||
        raise ArgumentError,
              "the model has none of the outputs #{inspect(@outputs)}, outputs: #{inspect(model.outputs)}"

    Serving.image(
      model,
      opts[:featurizer] || Featurizer.load!(repo),
      opts,
      fn outputs, _inputs ->
        %{embedding: outputs[output] |> pool(output) |> process(opts[:embedding_processor])}
      end,
      fn [%{embedding: embedding}], _images -> %{embedding: embedding} end
    )
  end

  # Exports without pooler_output, such as DINOv2, give the pooled output as the first
  # token of last_hidden_state.
  defp pool(hidden, "last_hidden_state"),
    do: hidden |> Nx.slice_along_axis(0, 1, axis: 1) |> Nx.squeeze(axes: [1])

  defp pool(embedding, _output), do: embedding

  defp process(embedding, nil), do: embedding
  defp process(embedding, :l2_norm), do: Serving.l2_norm(embedding)
end
