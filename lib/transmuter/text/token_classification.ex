defmodule Transmuter.Text.TokenClassification do
  @moduledoc """
  The token classification task. See `Transmuter.Text.token_classification/2`.

  The aggregation follows the token classification pipeline of Python transformers. The `:same` strategy is its "simple" strategy.
  """

  @compile {:no_warn_undefined,
            [Tokenizers.Tokenizer, Tokenizers.Encoding, Tokenizers.Model, Tokenizers.Decoder]}

  alias Tokenizers.Encoding
  alias Transmuter.{Serving, Tokenizer}

  @aggregations [nil, :same, :word_first, :word_average, :word_max]

  @doc false
  def token_classification(source, opts) do
    opts = Keyword.validate!(opts, Serving.options() ++ [:aggregation, ignored_labels: ["O"]])
    {repo, model, tokenizer} = Serving.load!(source, [outputs: ["logits"]] ++ opts, :fp32)
    aggregation = opts[:aggregation]

    unless aggregation in @aggregations do
      raise ArgumentError,
            "expected :aggregation to be one of #{inspect(@aggregations)}, got: #{inspect(aggregation)}"
    end

    # A tokenizer with a prefix for word continuations (## in WordPiece) marks the subwords
    # exactly. With other tokenizers, a token that does not follow a space is a subword.
    prefix? =
      tokenizer.native
      |> Tokenizers.Tokenizer.get_model()
      |> Tokenizers.Model.info()
      |> Map.get("continuing_subword_prefix")
      |> is_binary()

    unk = tokenizer.special_tokens[:unk]
    unk_id = unk && Tokenizer.token_to_id(tokenizer, unk)

    # Joins tokens into text like convert_tokens_to_string in Python transformers.
    decode =
      case Tokenizers.Tokenizer.get_decoder(tokenizer.native) do
        nil -> &Enum.join(&1, " ")
        decoder -> &elem(Tokenizers.Decoder.decode(decoder, &1), 1)
      end

    Serving.text(
      repo,
      model,
      tokenizer,
      opts,
      fn outputs, _inputs -> %{scores: Serving.softmax(outputs["logits"])} end,
      fn [%{scores: scores}], [{text, encoding}] ->
        tokens =
          for {scores, id, token, {start, stop}, 0} <-
                Enum.zip([
                  Nx.to_list(scores),
                  Tokenizer.ids(tokenizer, encoding),
                  Encoding.get_tokens(encoding),
                  Encoding.get_offsets(encoding),
                  Encoding.get_special_tokens_mask(encoding)
                ]) do
            # An unknown token stands for its text, as in Python transformers.
            unk? = id == unk_id
            token = if unk?, do: binary_part(text, start, stop - start), else: token
            subword? = not unk? and subword?(text, token, start, stop, prefix?)
            %{scores: scores, tokens: [token], start: start, end: stop, subword?: subword?}
          end

        entities =
          tokens
          |> aggregate(aggregation, repo, decode)
          |> Enum.reject(&(&1.label in opts[:ignored_labels]))

        %{entities: entities}
      end
    )
  end

  defp subword?(text, token, start, stop, true),
    do: String.length(token) != String.length(binary_part(text, start, stop - start))

  defp subword?(text, _token, start, stop, false) do
    start > 0 and
      not String.contains?(
        String.last(binary_part(text, 0, start)) <>
          (String.first(binary_part(text, start, stop - start)) || ""),
        " "
      )
  end

  defp aggregate(tokens, nil, repo, _decode) do
    for token <- tokens do
      {[phrase], entity} = token |> entity(token.scores, repo) |> Map.pop!(:tokens)
      Map.put(entity, :phrase, phrase)
    end
  end

  defp aggregate(tokens, :same, repo, decode),
    do: tokens |> Enum.map(&entity(&1, &1.scores, repo)) |> group(decode)

  defp aggregate(tokens, strategy, repo, decode) do
    tokens
    |> chunk(fn _last, token -> token.subword? end)
    |> Enum.map(fn [first | _] = word ->
      word_token = %{first | tokens: Enum.flat_map(word, & &1.tokens), end: List.last(word).end}
      entity(word_token, word_scores(word, strategy), repo)
    end)
    |> group(decode)
  end

  defp word_scores([first | _], :word_first), do: first.scores
  defp word_scores(word, :word_max), do: Enum.max_by(word, &Enum.max(&1.scores)).scores

  defp word_scores(word, :word_average),
    do: word |> Enum.map(& &1.scores) |> Enum.zip_with(&(Enum.sum(&1) / length(&1)))

  defp entity(token, scores, repo) do
    {score, index} = scores |> Enum.with_index() |> Enum.max_by(&elem(&1, 0))
    label = Serving.label(repo, index)

    %{
      start: token.start,
      end: token.end,
      label: label,
      score: score,
      tokens: token.tokens
    }
  end

  # An entity joins the group before it when it has the same tag and does not begin a new
  # entity (B-).
  defp group(entities, decode) do
    entities
    |> chunk(fn last, entity ->
      {prefix, tag} = tag(entity.label)
      prefix == :i and tag == elem(tag(last.label), 1)
    end)
    |> Enum.map(fn [first | _] = group ->
      %{
        start: first.start,
        end: List.last(group).end,
        label: elem(tag(first.label), 1),
        score: Enum.sum(Enum.map(group, & &1.score)) / length(group),
        phrase: decode.(Enum.flat_map(group, & &1.tokens))
      }
    end)
  end

  # Splits the list before each element that does not join the element before it.
  defp chunk(list, joins?) do
    Enum.chunk_while(
      list,
      [],
      fn
        x, [] ->
          {:cont, [x]}

        x, [last | _] = acc ->
          if joins?.(last, x), do: {:cont, [x | acc]}, else: {:cont, Enum.reverse(acc), [x]}
      end,
      fn
        [] -> {:cont, []}
        acc -> {:cont, Enum.reverse(acc), []}
      end
    )
  end

  defp tag("B-" <> tag), do: {:b, tag}
  defp tag("I-" <> tag), do: {:i, tag}
  defp tag(tag), do: {:i, tag}
end
