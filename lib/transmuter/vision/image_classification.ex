defmodule Transmuter.Vision.ImageClassification do
  @moduledoc """
  The image classification task. See `Transmuter.Vision.image_classification/2`.
  """

  alias Transmuter.{Featurizer, Serving}

  @doc false
  def image_classification(source, opts) do
    opts = Keyword.validate!(opts, Serving.image_options() ++ [:scores_function, top_k: 5])
    {repo, model} = Serving.load_model!(source, opts, :fp32)

    scores_function =
      Keyword.get_lazy(opts, :scores_function, fn -> Serving.scores_function(repo) end)

    Serving.image(
      model,
      opts[:featurizer] || Featurizer.load!(repo),
      opts,
      fn outputs, _inputs -> %{scores: Serving.scores(outputs["logits"], scores_function)} end,
      fn [%{scores: scores}], _images ->
        Serving.predictions(scores, &Serving.label(repo, &1), opts[:top_k])
      end
    )
  end
end
