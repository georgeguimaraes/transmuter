defmodule Transmuter.Vision.ZeroShotImageClassification do
  @moduledoc """
  The zero-shot image classification task. See `Transmuter.Vision.zero_shot_image_classification/3`.
  """

  alias Transmuter.{Featurizer, Serving, Tokenizer}

  @doc false
  def zero_shot_image_classification(source, labels, opts) do
    opts =
      Keyword.validate!(
        opts,
        Serving.image_options() ++
          [:tokenizer, top_k: 5, hypothesis_template: "This is a photo of {}."]
      )

    {repo, model} = Serving.load_model!(source, opts, :fp32)

    unless "logits_per_image" in model.outputs do
      raise ArgumentError,
            "zero-shot image classification needs the model with the text and the image towers, " <>
              "which gives logits_per_image, got the outputs #{inspect(model.outputs)}"
    end

    # The text of all labels goes into each batch, padded to the longest input of the text
    # tower. SigLIP was trained with this padding. CLIP pools the end token under a causal
    # mask, so the padding does not change its result.
    length = text_length!(repo)
    tokenizer = Tokenizer.configure(Serving.tokenizer(repo, opts), length: length)
    texts = Enum.map(labels, &String.replace(opts[:hypothesis_template], "{}", &1))
    {text_inputs, _encodings} = Tokenizer.encode(tokenizer, texts)
    labels = List.to_tuple(labels)

    # As the zero-shot image classification pipeline of Python transformers.
    scores_function =
      if String.contains?(repo.config["model_type"] || "", "siglip"),
        do: :sigmoid,
        else: :softmax

    Serving.image(
      model,
      opts[:featurizer] || Featurizer.load!(repo),
      opts,
      fn outputs, _inputs ->
        %{scores: Serving.scores(outputs["logits_per_image"], scores_function)}
      end,
      fn [%{scores: scores}], _images ->
        Serving.predictions(scores, &elem(labels, &1), opts[:top_k])
      end,
      text_inputs
    )
  end

  defp text_length!(repo) do
    Enum.find(
      [
        repo.tokenizer_config["model_max_length"],
        get_in(repo.config, ["text_config", "max_position_embeddings"])
      ],
      &(is_integer(&1) and &1 <= 1_000_000)
    ) || raise ArgumentError, "#{inspect(repo.source)} does not give the longest text input"
  end
end
