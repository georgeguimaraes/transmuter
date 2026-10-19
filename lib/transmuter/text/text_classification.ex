defmodule Transmuter.Text.TextClassification do
  @moduledoc """
  The text classification task. See `Transmuter.Text.text_classification/2`.
  """

  alias Transmuter.Serving

  @doc false
  def text_classification(source, opts) do
    opts = Keyword.validate!(opts, Serving.options() ++ [:scores_function, top_k: 5])
    {repo, model, tokenizer} = Serving.load!(source, opts, :fp32)
    scores_function = Keyword.get_lazy(opts, :scores_function, fn -> scores_function(repo) end)

    Serving.text(
      repo,
      model,
      tokenizer,
      opts,
      fn outputs, _inputs -> %{scores: scores(outputs["logits"], scores_function)} end,
      fn [%{scores: scores}], _items ->
        predictions =
          scores
          |> Nx.to_list()
          |> Enum.with_index(&%{label: Serving.label(repo, &2), score: &1})
          |> Enum.sort_by(& &1.score, :desc)

        %{
          predictions:
            if(opts[:top_k], do: Enum.take(predictions, opts[:top_k]), else: predictions)
        }
      end,
      &text_or_pair/1
    )
  end

  # As in the text classification pipeline of Python transformers.
  defp scores_function(repo) do
    cond do
      repo.config["problem_type"] == "regression" -> :none
      map_size(repo.config["id2label"] || %{}) == 1 -> :sigmoid
      repo.config["problem_type"] == "multi_label_classification" -> :sigmoid
      true -> :softmax
    end
  end

  defp scores(logits, :softmax), do: Serving.softmax(logits)
  defp scores(logits, :sigmoid), do: Nx.sigmoid(logits)
  defp scores(logits, :none), do: logits

  defp text_or_pair(text) when is_binary(text), do: {:ok, [text]}

  defp text_or_pair(input = {text, pair}) when is_binary(text) and is_binary(pair),
    do: {:ok, [input]}

  defp text_or_pair(input),
    do: {:error, "expected a string or a pair of strings, got: #{inspect(input)}"}
end
