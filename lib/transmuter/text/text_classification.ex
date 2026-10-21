defmodule Transmuter.Text.TextClassification do
  @moduledoc """
  The text classification task. See `Transmuter.Text.text_classification/2`.
  """

  alias Transmuter.Serving

  @doc false
  def text_classification(source, opts) do
    opts = Keyword.validate!(opts, Serving.options() ++ [:scores_function, top_k: 5])
    {repo, model, tokenizer} = Serving.load!(source, [outputs: ["logits"]] ++ opts, :fp32)

    scores_function =
      Keyword.get_lazy(opts, :scores_function, fn -> Serving.scores_function(repo) end)

    Serving.text(
      repo,
      model,
      tokenizer,
      opts,
      fn outputs, _inputs -> %{scores: Serving.scores(outputs["logits"], scores_function)} end,
      fn [%{scores: scores}], _items ->
        Serving.predictions(scores, &Serving.label(repo, &1), opts[:top_k])
      end,
      &text_or_pair/1
    )
  end

  defp text_or_pair(text) when is_binary(text), do: {:ok, [text]}

  defp text_or_pair(input = {text, pair}) when is_binary(text) and is_binary(pair),
    do: {:ok, [input]}

  defp text_or_pair(input),
    do: {:error, "expected a string or a pair of strings, got: #{inspect(input)}"}
end
