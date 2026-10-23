defmodule Transmuter.Text.ZeroShotEntityRecognition do
  @moduledoc """
  The zero-shot entity recognition task with GLiNER models. See `Transmuter.Text.zero_shot_entity_recognition/2`.

  The steps follow gliner 0.2.29. The text splits into words. The input of the model is a prompt with the labels, `<<ENT>>` and the label for each label and then `<<SEP>>`, followed by the words of the text. Each word is encoded alone, as the tokenizer of Python encodes a list of words. The model scores the words against the labels. `gliner_config.json` gives the variant of the model:

    * A span model, such as `onnx-community/gliner_small-v2.1`, scores each span of up to `max_width` words for each label. A span is an entity when its score is more than the threshold.

    * A token model (`span_mode` is `"token_level"`), such as `knowledgator/gliner-pii-small-v1.0`, scores each word as the start, the end and the inside of an entity of each label. A start and an end of one label with scores more than the threshold give an entity when all words from the start to the end have an inside score of at least the threshold. The score of the entity is the lowest of these scores.

  Then the entities go from the highest score to the lowest. An entity stays when it does not overlap an entity that stayed before it.

  The inputs of the model are those of gliner, with one change for the compiler. The word axis of the model is the largest word count of the batch, and the model finds the first token of each word with NonZero on the words mask. So the shapes after these nodes depend on the word counts, and each new word count would compile segments again. The serving gives each batch a word axis from its bucket instead: a power of two from 16 up to `max_len` of `gliner_config.json`, and one spare word. One more row has this word count. Each token that is not the first token of a word, such as the prompt, the special tokens and the padding, points to the spare word, so NonZero always finds all tokens. The spare word comes after the words of each row, and the model reads only the words of a row, so the results of the rows do not change. Then the shapes of a batch depend only on its sequence length, its bucket and its labels, and a new text with the same shapes compiles nothing.

  """

  @compile {:no_warn_undefined, [Tokenizers.Tokenizer, Tokenizers.Encoding]}

  alias Tokenizers.Encoding
  alias Transmuter.{Hub, Repo, Serving, Tokenizer}

  # The whitespace splitter of gliner, \w+(?:[-_]\w+)*|\S in Python: a word of letters,
  # digits and underscores, with single hyphens or underscores inside, or any other character
  # that is not a space. The classes are those of \w and \s in Python.
  @words ~r/[\p{L}\p{N}_]+(?:[-_][\p{L}\p{N}_]+)*|[^\t\n\x{0b}\f\r\x{1c}-\x{1f} \x{85}\x{a0}\x{1680}\x{2000}-\x{200a}\x{2028}\x{2029}\x{202f}\x{205f}\x{3000}]/u

  @doc false
  def zero_shot_entity_recognition(source, opts) do
    opts =
      Keyword.validate!(
        opts,
        Serving.options() ++ [threshold: 0.5, flat_ner: true, multi_label: false]
      )

    repo = Repo.load!(source, Keyword.take(opts, [:variant | Hub.options()]))
    config = config!(repo)
    tokenizer = opts[:tokenizer] || Tokenizer.load!(repo)

    model =
      opts[:model] ||
        Repo.load_component!(
          repo,
          "model",
          :fp32,
          [outputs: ["logits"]] ++ Keyword.take(opts, [:backend, :compute_type])
        )

    compile =
      Keyword.validate!(opts[:compile] || [],
        batch_size: 1,
        sequence_length: Serving.sequence_lengths(repo, nil)
      )

    lengths = Enum.sort(List.wrap(compile[:sequence_length]))
    encoder = encoder(tokenizer, config, List.last(lengths))
    kind = if config["span_mode"] == "token_level", do: :token, else: :span
    buckets = buckets(config["max_len"])
    types = Map.new(model.input_types, fn {name, {type, _dims}} -> {name, type} end)

    templates =
      Map.new(lengths, fn length ->
        tensors =
          tensors(
            types,
            List.duplicate(%{ids: [], words_mask: [], text_length: 0}, compile[:batch_size]),
            length,
            0
          )

        {{:sequence_length, length}, tensors}
      end)

    preprocess = fn inputs ->
      rows = Enum.map(inputs, &encode(encoder, &1))
      longest = rows |> Enum.map(&length(&1.ids)) |> Enum.max()
      length = Enum.find(lengths, &(&1 >= longest))
      tensors = tensors(types, rows, length, encoder.pad_id)
      {[{tensors, {:sequence_length, length}}], Enum.map(rows, &[&1])}
    end

    shapes = %{
      batch_size: compile[:batch_size],
      templates: templates,
      inputs: %{},
      prepare: &prepare(&1, kind, types, buckets, config["max_width"])
    }

    Serving.new(
      model,
      opts,
      shapes,
      preprocess,
      fn outputs, inputs ->
        rows = Nx.axis_size(inputs["input_ids"], 0)
        scores = outputs["logits"] |> Nx.as_type(:f32) |> Nx.sigmoid()
        %{scores: Nx.slice_along_axis(scores, 0, rows, axis: 0)}
      end,
      fn [%{scores: scores}], [row] ->
        spans = decode(scores, kind, row.text_length, length(row.labels), opts)
        %{entities: Enum.map(spans, &entity(&1, row))}
      end,
      &validate/1
    )
  end

  # The settings of gliner_config.json, with the defaults of gliner.
  @defaults %{
    "max_len" => 384,
    "max_width" => 12,
    "span_mode" => "markerV0",
    "subtoken_pooling" => "first",
    "words_splitter_type" => "whitespace",
    "ent_token" => "<<ENT>>",
    "sep_token" => "<<SEP>>"
  }

  @doc false
  def config!(repo) do
    config =
      Hub.json(repo.source, "gliner_config.json", repo.opts) ||
        raise ArgumentError,
              "#{inspect(repo.source)} has no gliner_config.json, so it is not a GLiNER model"

    config = Map.merge(@defaults, config, fn _key, default, value -> value || default end)
    parts = for key <- ~w(labels_encoder labels_decoder relations_layer), config[key], do: key

    cond do
      config["model_type"] == "gliner_streaming_span" ->
        raise ArgumentError, "Transmuter does not run streaming GLiNER models"

      parts != [] ->
        raise ArgumentError,
              "Transmuter does not run GLiNER models with #{Enum.join(parts, " or ")}"

      config["words_splitter_type"] != "whitespace" ->
        raise ArgumentError,
              "Transmuter splits words only as the whitespace splitter of gliner, " <>
                "got words_splitter_type #{inspect(config["words_splitter_type"])}"

      config["subtoken_pooling"] != "first" ->
        raise ArgumentError,
              "Transmuter pools only the first token of a word, " <>
                "got subtoken_pooling #{inspect(config["subtoken_pooling"])}"

      true ->
        config
    end
  end

  defp buckets(max_len),
    do: Enum.take_while(Stream.iterate(16, &(&1 * 2)), &(&1 < max_len)) ++ [max_len]

  @doc false
  # The settings of encode/2: the tokenizer, the tokens of the prompt, the special tokens
  # around the input, the pad id and the limits of the input.
  def encoder(tokenizer, config, max_tokens) do
    {:ok, probe} = Tokenizers.Tokenizer.encode(tokenizer.native, "a", add_special_tokens: true)
    special = Enum.zip(Tokenizer.ids(tokenizer, probe), Encoding.get_special_tokens_mask(probe))
    {prefix, rest} = Enum.split_while(special, &(elem(&1, 1) == 1))

    %{
      tokenizer: tokenizer,
      ent_token: config["ent_token"],
      sep_token: config["sep_token"],
      prefix: Enum.map(prefix, &elem(&1, 0)),
      suffix: rest |> Enum.drop_while(&(elem(&1, 1) == 0)) |> Enum.map(&elem(&1, 0)),
      pad_id: Tokenizer.token_to_id(tokenizer, Tokenizer.pad_token(tokenizer)),
      max_words: config["max_len"],
      max_tokens: max_tokens
    }
  end

  @doc false
  # The input of the model for one text: the token ids, the words mask (the number of each
  # word, from 1, at its first token), the word count, the byte offsets of the words and the
  # labels without duplicates, as gliner makes them. gliner keeps the first max_len words.
  # Transmuter also keeps only the words that fit in max_tokens.
  def encode(encoder, %{text: text, labels: labels}) do
    labels = Enum.uniq(labels)
    prompt = Enum.flat_map(labels, &[encoder.ent_token, &1]) ++ [encoder.sep_token]

    words =
      @words
      |> Regex.scan(text, return: :index)
      |> Enum.map(fn [{start, length}] -> {start, start + length} end)
      |> Enum.take(encoder.max_words)

    {:ok, encodings} =
      Tokenizers.Tokenizer.encode_batch(
        encoder.tokenizer.native,
        prompt ++ Enum.map(words, fn {start, stop} -> binary_part(text, start, stop - start) end),
        add_special_tokens: false
      )

    {prompt_ids, word_ids} =
      encodings |> Enum.map(&Tokenizer.ids(encoder.tokenizer, &1)) |> Enum.split(length(prompt))

    prompt_ids = List.flatten(prompt_ids)
    fixed = length(encoder.prefix) + length(prompt_ids) + length(encoder.suffix)

    if fixed > encoder.max_tokens do
      raise ArgumentError,
            "the labels and the special tokens take #{fixed} tokens, " <>
              "more than the largest sequence length, #{encoder.max_tokens}"
    end

    word_ids =
      word_ids
      |> Enum.scan({0, nil}, fn ids, {count, _ids} -> {count + length(ids), ids} end)
      |> Enum.take_while(fn {count, _ids} -> fixed + count <= encoder.max_tokens end)
      |> Enum.map(&elem(&1, 1))

    # A word without tokens gets no number, so the words after it get lower numbers, as in
    # prepare_word_mask of gliner.
    {words_mask, _count} =
      Enum.flat_map_reduce(word_ids, 0, fn
        [], count -> {[], count}
        [_ | rest], count -> {[count + 1 | Enum.map(rest, fn _ -> 0 end)], count + 1}
      end)

    zeros = &List.duplicate(0, length(&1))

    %{
      text: text,
      labels: labels,
      words: words |> Enum.take(length(word_ids)) |> List.to_tuple(),
      text_length: length(word_ids),
      ids: encoder.prefix ++ prompt_ids ++ List.flatten(word_ids) ++ encoder.suffix,
      words_mask: zeros.(encoder.prefix ++ prompt_ids) ++ words_mask ++ zeros.(encoder.suffix)
    }
  end

  defp tensors(types, rows, length, pad_id) do
    pad = fn list, value -> list ++ List.duplicate(value, length - length(list)) end
    tensor = &Nx.tensor(&2, type: types[&1], backend: Nx.BinaryBackend)

    %{
      "input_ids" => tensor.("input_ids", Enum.map(rows, &pad.(&1.ids, pad_id))),
      "attention_mask" =>
        tensor.("attention_mask", Enum.map(rows, &pad.(Enum.map(&1.ids, fn _ -> 1 end), 0))),
      "words_mask" => tensor.("words_mask", Enum.map(rows, &pad.(&1.words_mask, 0))),
      "text_lengths" => tensor.("text_lengths", Enum.map(rows, &[&1.text_length]))
    }
  end

  # The inputs that depend on the word count of the batch: the bucket row and the spare word
  # of the module documentation. Each row has at least one word, as in gliner.
  defp prepare(inputs, kind, types, buckets, max_width) do
    words = Enum.find(buckets, &(&1 >= Nx.to_number(Nx.reduce_max(inputs["text_lengths"])))) + 1

    inputs =
      Map.new(inputs, fn {name, tensor} ->
        [_batch | axes] = List.duplicate({0, 0, 0}, Nx.rank(tensor))
        {name, Nx.pad(tensor, 0, [{0, 1, 0} | axes])}
      end)

    rows = Nx.axis_size(inputs["text_lengths"], 0)
    last = Nx.tensor([[words]], type: types["text_lengths"])
    text_lengths = inputs["text_lengths"] |> Nx.put_slice([rows - 1, 0], last) |> Nx.max(1)
    words_mask = Nx.select(inputs["words_mask"], inputs["words_mask"], words)
    inputs = %{inputs | "text_lengths" => text_lengths, "words_mask" => words_mask}
    spans(inputs, kind, types, words, max_width)
  end

  # A span model also takes each span of up to max_width words from each word, and a mask of
  # the spans that end inside the text of the row.
  defp spans(inputs, :span, types, words, max_width) do
    spans = Nx.iota({words * max_width})
    starts = Nx.quotient(spans, max_width)
    ends = Nx.add(starts, Nx.remainder(spans, max_width))
    mask = Nx.less(Nx.new_axis(ends, 0), inputs["text_lengths"])

    span_idx =
      [starts, ends]
      |> Nx.stack(axis: -1)
      |> Nx.new_axis(0)
      |> Nx.multiply(Nx.new_axis(mask, -1))

    Map.merge(inputs, %{
      "span_idx" => Nx.as_type(span_idx, types["span_idx"]),
      "span_mask" => Nx.as_type(mask, types["span_mask"])
    })
  end

  defp spans(inputs, :token, _types, _words, _max_width), do: inputs

  @doc false
  # The entities of one row as {first word, last word, label index, score}, from the first
  # word, as the decoders of gliner give them. `scores` are the probabilities of the row, of
  # shape {words, max_width, labels} for a span model and {words, labels, 3} for a token
  # model. Only the first `words` words and the first `labels` labels count, because the batch
  # can have more.
  def decode(scores, kind, words, labels, opts) do
    scores
    |> candidates(kind, words, labels, opts[:threshold])
    |> Enum.sort_by(&elem(&1, 3), :desc)
    |> Enum.reduce([], fn span, kept ->
      if Enum.any?(kept, &overlap?(span, &1, opts)), do: kept, else: [span | kept]
    end)
    |> Enum.reverse()
    |> Enum.sort_by(&elem(&1, 0))
  end

  defp candidates(scores, :span, words, labels, threshold) do
    {_words, width, classes} = Nx.shape(scores)

    for {score, index} <- scores |> Nx.to_flat_list() |> Enum.with_index(),
        score > threshold,
        start = div(index, width * classes),
        stop = start + rem(div(index, classes), width),
        label = rem(index, classes),
        stop < words and label < labels,
        do: {start, stop, label, score}
  end

  defp candidates(scores, :token, words, labels, threshold) do
    scores =
      for {row, word} <- scores |> Nx.to_list() |> Enum.take(words) |> Enum.with_index(),
          {[start, stop, inside], label} <- row |> Enum.take(labels) |> Enum.with_index(),
          do: {word, label, start, stop, inside}

    inside =
      Map.new(scores, fn {word, label, _start, _stop, inside} -> {{word, label}, inside} end)

    starts = for {word, label, start, _, _} <- scores, start > threshold, do: {word, label, start}
    stops = for {word, label, _, stop, _} <- scores, stop > threshold, do: {word, label, stop}

    for {first, label, start} <- starts,
        {last, ^label, stop} <- stops,
        last >= first,
        insides = Enum.map(first..last, &inside[{&1, label}]),
        Enum.all?(insides, &(&1 >= threshold)),
        do: {first, last, label, Enum.min([start, stop | insides])}
  end

  # The overlap rule of greedy_search in gliner. Two labels of the same span overlap without
  # :multi_label. Other spans overlap when they share a word, but with flat_ner: false a span
  # inside the other does not count.
  defp overlap?({first, last, _, _}, {first, last, _, _}, opts), do: not opts[:multi_label]

  defp overlap?({first, last, _, _}, {other_first, other_last, _, _}, _opts)
       when first > other_last or other_first > last,
       do: false

  defp overlap?({first, last, _, _}, {other_first, other_last, _, _}, opts) do
    opts[:flat_ner] or
      not ((first <= other_first and last >= other_last) or
             (other_first <= first and other_last >= last))
  end

  defp entity({first, last, label, score}, row) do
    {start, _} = elem(row.words, first)
    {_, stop} = elem(row.words, last)

    %{
      start: start,
      end: stop,
      label: Enum.at(row.labels, label),
      score: score,
      phrase: binary_part(row.text, start, stop - start)
    }
  end

  defp validate(input = %{text: text, labels: [_ | _] = labels}) when is_binary(text) do
    if Enum.all?(labels, &is_binary/1), do: {:ok, [input]}, else: invalid(input)
  end

  defp validate(input), do: invalid(input)

  defp invalid(input),
    do: {:error, "expected %{text: string, labels: [string, ...]}, got: #{inspect(input)}"}
end
