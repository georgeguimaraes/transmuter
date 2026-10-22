defmodule Transmuter.Audio.AudioClassification do
  @moduledoc """
  The audio classification task. See `Transmuter.Audio.audio_classification/2`.
  """

  alias Transmuter.Audio.Featurizer
  alias Transmuter.Serving

  @doc false
  def audio_classification(source, opts) do
    opts =
      Keyword.validate!(opts, Serving.audio_options() ++ [scores_function: :softmax, top_k: 5])

    {repo, model} = Serving.load_model!(source, [outputs: ["logits"]] ++ opts, :fp32)
    {scores_function, top_k} = {opts[:scores_function], opts[:top_k]}

    Serving.audio(
      model,
      opts[:featurizer] || Featurizer.load!(repo),
      opts,
      fn outputs, _inputs -> %{scores: Serving.scores(outputs["logits"], scores_function)} end,
      fn [%{scores: scores}], _audio ->
        Serving.predictions(scores, &Serving.label(repo, &1), top_k)
      end
    )
  end
end
