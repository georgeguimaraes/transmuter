defmodule Transmuter.Text.QuestionAnswering do
  @moduledoc """
  The question answering task. See `Transmuter.Text.question_answering/2`.

  The search for the answer follows the question answering pipeline of Python transformers:

    1. A long context becomes windows of the largest sequence length, which is 384 tokens by default. Each window repeats the last 128 tokens of the window before it, or half of the largest length when that is less.
    2. In each window, the start and end logits of the tokens out of the context get -10000. A softmax gives the start and end scores. The first token, the CLS token of most models, gets the score 0.
    3. The score of a span is the start score of its first token times the end score of its last token. A span has at most `:max_answer_length` tokens.
    4. The 12 spans with the highest scores in each window are candidates. Each candidate grows to the start of its first word and the end of its last word.
    5. Candidates with the same text, without case, become one answer with the sum of their scores. The answer with the highest score is the result.
  """

  @compile {:no_warn_undefined, Tokenizers.Encoding}

  alias Tokenizers.Encoding
  alias Transmuter.{Serving, Tokenizer}

  # The pipeline of Python transformers keeps 2 * top_k + 10 spans of each window, with top_k 1.
  @candidates 12

  @doc false
  def question_answering(source, opts) do
    opts = Keyword.validate!(opts, Serving.options() ++ [max_answer_length: 15])

    {repo, model, tokenizer} =
      Serving.load!(source, [outputs: ~w(start_logits end_logits)] ++ opts, :fp32)

    cls = tokenizer.special_tokens[:cls]
    cls_id = cls && Tokenizer.token_to_id(tokenizer, cls)

    Serving.text(
      repo,
      model,
      Tokenizer.configure(tokenizer, truncation: :only_second, stride: 128),
      Keyword.put(opts, :max_length, 384),
      fn outputs, _inputs -> %{start: outputs["start_logits"], end: outputs["end_logits"]} end,
      fn rows, items ->
        answer =
          rows
          |> Enum.zip(items)
          |> Enum.flat_map(fn {row, {{_question, context}, encoding}} ->
            candidates(
              row,
              context,
              encoding,
              Tokenizer.ids(tokenizer, encoding),
              cls_id,
              opts[:max_answer_length]
            )
          end)
          |> Enum.group_by(&String.downcase(&1.text))
          |> Enum.map(fn {_text, [first | _] = same} ->
            %{first | score: same |> Enum.map(& &1.score) |> Enum.sum()}
          end)
          |> Enum.max_by(& &1.score, fn -> nil end)

        %{results: List.wrap(answer)}
      end,
      fn
        %{question: question, context: context} when is_binary(question) and is_binary(context) ->
          {:ok, [{question, context}]}

        input ->
          {:error, "expected a map with :question and :context strings, got: #{inspect(input)}"}
      end
    )
  end

  defp candidates(row, context, encoding, ids, cls_id, max_answer_length) do
    # The answer is in the context. The CLS token also stays, as in Python transformers,
    # because some models point to it when the context has no answer.
    context? =
      Enum.zip_with(
        Encoding.get_sequence_ids(encoding),
        ids,
        &(&1 == 1 or &2 == cls_id)
      )

    mask = Nx.tensor(Enum.map(context?, &if(&1, do: 1, else: 0)), type: :u8)
    starts = probabilities(row.start, mask)
    ends = probabilities(row.end, mask)
    context? = List.to_tuple(context?)
    last = tuple_size(starts) - 1

    for(
      i <- 0..last,
      j <- i..min(i + max_answer_length - 1, last),
      do: {elem(starts, i) * elem(ends, j), i, j}
    )
    |> Enum.sort_by(&elem(&1, 0), :desc)
    |> Enum.take(@candidates)
    |> Enum.filter(fn {_score, i, j} -> elem(context?, i) and elem(context?, j) end)
    |> Enum.map(fn {score, i, j} ->
      {start, stop} = span(encoding, i, j)
      %{text: binary_part(context, start, stop - start), start: start, end: stop, score: score}
    end)
  end

  defp probabilities(logits, mask) do
    mask
    |> Nx.select(logits, -10_000.0)
    |> Serving.softmax()
    |> Nx.to_list()
    |> List.replace_at(0, 0.0)
    |> List.to_tuple()
  end

  # The byte offsets from the start of the first word to the end of the last word, or of the
  # tokens when they are not in a word.
  defp span(encoding, first_token, last_token) do
    with {sequence, first_word} <- Encoding.token_to_word(encoding, first_token),
         {^sequence, last_word} <- Encoding.token_to_word(encoding, last_token),
         {start, _} <- Encoding.word_to_chars(encoding, first_word, sequence),
         {_, stop} <- Encoding.word_to_chars(encoding, last_word, sequence) do
      {start, stop}
    else
      _ ->
        offsets = Encoding.get_offsets(encoding)
        {elem(Enum.at(offsets, first_token), 0), elem(Enum.at(offsets, last_token), 1)}
    end
  end
end
