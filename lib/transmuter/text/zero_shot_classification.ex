defmodule Transmuter.Text.ZeroShotClassification do
  @moduledoc """
  The zero-shot classification task. See `Transmuter.Text.zero_shot_classification/3`.

  The scores follow the zero-shot classification pipeline of Python transformers. The text and the hypothesis of each label are one pair for a natural language inference (NLI) model.
  """

  alias Transmuter.{Serving, Tokenizer}

  @doc false
  def zero_shot_classification(source, labels, opts) do
    opts =
      Keyword.validate!(
        opts,
        Serving.options() ++
          [hypothesis_template: &"This example is #{&1}.", multi_label: false, top_k: 5]
      )

    unless labels != [] and Enum.all?(labels, &is_binary/1) do
      raise ArgumentError, "expected a list of label strings, got: #{inspect(labels)}"
    end

    {repo, model, tokenizer} = Serving.load!(source, opts, :fp32)
    hypotheses = Enum.map(labels, opts[:hypothesis_template])
    entailment = entailment!(repo)

    # As in Python transformers, one label gets the score of :multi_label, because a softmax
    # over one label is always 1. The contradiction class is the last class, or the first when
    # the entailment class is not the first. This also covers models with the two classes
    # entailment and not_entailment.
    scores =
      if opts[:multi_label] or length(labels) == 1 do
        contradiction = if entailment == 0, do: map_size(repo.config["id2label"]) - 1, else: 0

        fn logits ->
          Nx.stack([logits[[.., contradiction]], logits[[.., entailment]]], axis: -1)
          |> Serving.softmax()
          |> then(& &1[[.., 1]])
        end
      else
        &Serving.softmax(&1[[.., entailment]])
      end

    Serving.text(
      repo,
      model,
      Tokenizer.configure(tokenizer, truncation: :only_first),
      opts,
      fn outputs, _inputs -> %{logits: outputs["logits"]} end,
      fn rows, _items ->
        predictions =
          rows
          |> Enum.map(& &1.logits)
          |> Nx.stack()
          |> scores.()
          |> Nx.to_list()
          |> Enum.zip_with(labels, &%{label: &2, score: &1})
          |> Enum.sort_by(& &1.score, :desc)

        %{
          predictions:
            if(opts[:top_k], do: Enum.take(predictions, opts[:top_k]), else: predictions)
        }
      end,
      fn
        text when is_binary(text) -> {:ok, Enum.map(hypotheses, &{text, &1})}
        input -> Serving.string(input)
      end
    )
  end

  defp entailment!(repo) do
    Enum.find_value(repo.config["id2label"] || %{}, fn {index, label} ->
      String.starts_with?(String.downcase(label), "entail") && String.to_integer(index)
    end) ||
      raise ArgumentError,
            "expected a class that starts with \"entail\" in id2label of config.json, " <>
              "got: #{inspect(repo.config["id2label"])}"
  end
end
