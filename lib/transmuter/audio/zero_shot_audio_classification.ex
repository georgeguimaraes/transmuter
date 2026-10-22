defmodule Transmuter.Audio.ZeroShotAudioClassification do
  @moduledoc """
  The zero-shot audio classification task. See `Transmuter.Audio.zero_shot_audio_classification/3`.
  """

  alias Transmuter.Audio.Featurizer
  alias Transmuter.{Serving, Tokenizer}

  @doc false
  def zero_shot_audio_classification(source, labels, opts) do
    opts =
      Keyword.validate!(
        opts,
        Serving.audio_options() ++
          [:tokenizer, top_k: 5, hypothesis_template: "This is a sound of {}."]
      )

    {repo, model} = Serving.load_model!(source, [outputs: ["logits_per_audio"]] ++ opts, :fp32)

    # As the zero-shot-audio-classification pipeline of Python transformers, the text of all
    # labels is padded to the longest text, and it goes into each batch.
    tokenizer = opts[:tokenizer] || Tokenizer.load!(repo)
    texts = Enum.map(labels, &String.replace(opts[:hypothesis_template], "{}", &1))
    {text_inputs, _encodings} = Tokenizer.encode(tokenizer, texts)
    labels = List.to_tuple(labels)

    Serving.audio(
      model,
      opts[:featurizer] || Featurizer.load!(repo),
      opts,
      fn outputs, _inputs -> %{scores: Serving.softmax(outputs["logits_per_audio"])} end,
      fn [%{scores: scores}], _audio ->
        Serving.predictions(scores, &elem(labels, &1), opts[:top_k])
      end,
      text_inputs
    )
  end
end
