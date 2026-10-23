defmodule Transmuter.Audio.WordTimestamps do
  @moduledoc """
  The word timestamps of Whisper, as `return_timestamps="word"` in the `automatic-speech-recognition` pipeline of Python transformers 4. See `Transmuter.Audio.speech_to_text/2`.

  The time of each token comes from the cross attentions of the decoder, as `_extract_token_timestamps` of `WhisperForConditionalGeneration` computes it:

    1. After each pass of generation, the decoder runs again over the prompt and the new tokens, without the cache. This run gives the cross attentions that Python collects in the steps of generation. Only the heads of `alignment_heads` in `generation_config.json` stay.
    2. The rows of the matrix are the new tokens that were an input of a step: all new tokens when the pass ends with eos, else all tokens but the last. The columns are the encoder positions of the audio after the seek of the pass. A position is two feature frames.
    3. Each head is normalized over the rows to a mean of 0 and a standard deviation of 1, and filtered along the columns with the median of `median_filter_width` values (7). The matrix is the mean of the heads.
    4. Dynamic time warping finds the path of the least cost through the negated matrix. The time of a row is the column where the path enters the row, at 0.02 s for each column. A token without a row gets the time of the last row.

  The words come from `_combine_tokens_into_words` of the Whisper tokenizer. The tokens split where their text is complete characters. A piece that starts with a space or that is punctuation starts a new word, and other pieces join the word before them. In Chinese, Japanese, Thai, Lao, Burmese and Cantonese, each piece is a word, as OpenAI whisper and transformers.js do. The pipeline of Python transformers does not know the language there, so it splits all languages at spaces. Then opening punctuation joins the next word, and closing punctuation joins the word before it.
  """

  alias Transmuter.{GraphBreaks, Tokenizer}

  @doc """
  Returns the names of the outputs of a decoder that `aligner/3` can read: the cross attentions of all layers. Give these names with those of `Transmuter.Generation.outputs/1` to the `:outputs` option of `Transmuter.load!/2`.
  """
  def outputs(names), do: Enum.filter(names, &String.starts_with?(&1, "cross_attentions."))

  @doc """
  Returns a function that takes the defn options, compiles the alignment of `decoder`, and returns a function that gives the time of each new token. Raises an `ArgumentError` when `generation_config.json` has no `alignment_heads` or the decoder does not give their cross attentions.

  The compiled function takes the outputs of the encoder, the prompts, the new tokens of each row (without eos), the seek of each row and the number of feature frames of the audio of each row. It returns the end time of each token in feature frames from the start of the window. A row without a prompt is a padding row.

  `settings` has `:prompt_length`, `:max_new_tokens`, `:frames_per_timestamp` and `:positions`, the largest number of tokens of the decoder.
  """
  def aligner(repo, decoder, settings) do
    heads =
      repo.generation_config["alignment_heads"] ||
        raise ArgumentError, "timestamps: :words needs alignment_heads in generation_config.json"

    names = heads |> Enum.map(&hd/1) |> Enum.uniq() |> Enum.map(&"cross_attentions.#{&1}")

    unless Enum.all?(names, &(&1 in decoder.outputs)) do
      raise ArgumentError,
            "timestamps: :words needs the outputs #{Enum.join(names, ", ")} of the decoder, " <>
              "which this export does not give. Use an export with cross attentions, such as " <>
              "Xenova/whisper-tiny.en or onnx-community/whisper-tiny.en_timestamped"
    end

    settings =
      Map.merge(settings, %{heads: heads, width: repo.config["median_filter_width"] || 7})

    # The params are an argument, so the compiled function does not hold them as constants.
    model = %{decoder | outputs: names, params: %{}}
    {type, _dims} = decoder.input_types["input_ids"]
    params = decoder.params
    &compile(model, params, type, settings, &1)
  end

  defp compile(model, params, type, settings, defn_options) do
    flag = Nx.tensor([0], type: :u8, backend: Nx.BinaryBackend)

    run =
      GraphBreaks.jit(
        fn params, inputs, counts, frames ->
          inputs = Map.merge(placeholders(model, inputs["input_ids"]), inputs)
          outputs = Transmuter.run(model, params, inputs, %{"use_cache_branch" => flag})
          matrix(outputs, counts, frames, settings)
        end,
        defn_options
      )

    fn %{"encoder_hidden_states" => hidden}, prompts, tokens, seeks, frames ->
      counts = Enum.zip_with(prompts, tokens, &count(&1, &2, settings.max_new_tokens))
      frames = Enum.zip_with(frames, seeks, &max(div(&1 - &2, settings.frames_per_timestamp), 1))
      rows = Enum.zip_with(prompts, tokens, &++/2)
      longest = rows |> Enum.map(&length/1) |> Enum.max()
      width = min(2 ** ceil(:math.log2(max(longest, 1))), settings.positions)
      ids = for row <- rows, do: row ++ List.duplicate(0, width - length(row))

      matrices =
        run.(
          params,
          %{
            "input_ids" => Nx.tensor(ids, type: type, backend: Nx.BinaryBackend),
            "encoder_hidden_states" => hidden
          },
          Nx.tensor(counts, type: :s32, backend: Nx.BinaryBackend),
          Nx.tensor(frames, type: :s32, backend: Nx.BinaryBackend)
        )

      {_rows, columns, size} = Nx.shape(matrices)
      binary = Nx.to_binary(matrices)

      for {{count, columns_of_audio, seek, tokens}, row} <-
            Enum.with_index(Enum.zip([counts, frames, seeks, tokens])) do
        matrix = binary_part(binary, row * columns * size * 4, columns * size * 4)

        times =
          for jump <- jumps(matrix, size, count, columns_of_audio),
              do: jump * settings.frames_per_timestamp + seek

        times ++ List.duplicate(List.last(times, seek), length(tokens) - length(times))
      end
    end
  end

  # The rows of the matrix: the new tokens that were an input of a step. The last token was
  # one when eos followed it, so when the pass has fewer than max_new_tokens tokens.
  defp count([], _tokens, _max_new_tokens), do: 0
  defp count(_prompt, tokens, max_new_tokens), do: min(length(tokens), max_new_tokens - 1)

  # The branch without cache does not read the cache, so it gets one row of zeros.
  defp placeholders(model, ids) do
    batch = Nx.axis_size(ids, 0)

    for name <- model.inputs, String.starts_with?(name, "past_key_values."), into: %{} do
      {type, [_batch, heads, _length, dim]} = model.input_types[name]
      {name, Nx.broadcast(Nx.tensor(0, type: type), {batch, heads, 1, dim})}
    end
  end

  # The negated matrix {batch, columns, rows} of each row of the batch. The columns after
  # `frames` and the rows after `counts` are not part of it.
  defp matrix(outputs, counts, frames, settings) do
    %{heads: heads, prompt_length: prompt_length, width: width} = settings

    weights =
      heads
      |> Enum.map(fn [layer, head] -> outputs["cross_attentions.#{layer}"][[.., head]] end)
      |> Nx.stack(axis: 1)
      |> Nx.as_type(:f32)

    rows = Nx.axis_size(weights, 2) - prompt_length
    weights = Nx.slice_along_axis(weights, prompt_length, rows, axis: 2)
    [weights, counts, frames] = Enum.map([weights, counts, frames], &Nx.vectorize(&1, :rows))

    valid = Nx.less(Nx.iota({1, rows, 1}, axis: 1), counts)
    count = Nx.max(counts, 1)
    sum = &(&1 |> Nx.multiply(valid) |> Nx.sum(axes: [1], keep_axes: true))
    centered = Nx.subtract(weights, Nx.divide(sum.(weights), count))
    deviation = centered |> Nx.pow(2) |> sum.() |> Nx.divide(count) |> Nx.sqrt()

    # A column with one value in all rows has no deviation, and Python normalizes it to NaN.
    # NaN sorts after all numbers in the median filter, and so does the largest float here.
    # A cell that stays undefined after the median filter, in any head, gets the largest cost.
    undefined = Nx.equal(deviation, 0)
    largest = Nx.Constants.max_finite(:f32)
    normalized = Nx.divide(centered, Nx.select(undefined, 1.0, deviation))
    normalized = Nx.select(Nx.broadcast(undefined, Nx.shape(normalized)), largest, normalized)

    # The median filter needs more columns than half its width.
    filter? = Nx.broadcast(Nx.greater(frames, div(width, 2)), Nx.shape(normalized))
    smoothed = Nx.select(filter?, median(normalized, frames, width), normalized)
    undefined = Nx.equal(smoothed, largest)

    smoothed
    |> then(&Nx.select(undefined, 0.0, &1))
    |> Nx.mean(axes: [0])
    |> Nx.negate()
    |> then(&Nx.select(Nx.any(undefined, axes: [0]), largest, &1))
    |> Nx.transpose()
    |> Nx.devectorize(keep_names: false)
  end

  @doc false
  # The median of `width` values along the last axis, as _median_filter. The columns before
  # the first column and after column `frames - 1` reflect the columns next to them. The
  # columns after `frames` are not part of the result.
  def median(weights, frames, width) do
    half = div(width, 2)
    axis = Nx.rank(weights) - 1
    columns = Nx.axis_size(weights, axis)
    last = Nx.subtract(frames, 1)
    positions = Nx.subtract(Nx.iota({columns + 2 * half}), half)

    reflected =
      positions
      |> Nx.abs()
      |> then(&Nx.select(Nx.greater(&1, last), Nx.subtract(Nx.multiply(2, last), &1), &1))
      |> Nx.clip(0, columns - 1)

    padded = Nx.take(weights, reflected, axis: axis)

    0..(width - 1)
    |> Enum.map(&Nx.slice_along_axis(padded, &1, columns, axis: axis))
    |> Nx.stack(axis: axis + 1)
    |> Nx.sort(axis: axis + 1)
    |> Nx.slice_along_axis(half, 1, axis: axis + 1)
    |> Nx.squeeze(axes: [axis + 1])
  end

  # The column where the path of _dynamic_time_warping enters each of the `count` rows, for
  # the first `frames` columns of `matrix`, a binary of f32 values with `length` values for
  # each column.
  defp jumps(_matrix, _length, 0, _frames), do: []

  defp jumps(matrix, length, count, frames) do
    initial = {[0.0 | List.duplicate(:infinity, count)], []}

    {_costs, traces} =
      Enum.reduce(0..(frames - 1), initial, fn j, {previous, traces} ->
        values = binary_part(matrix, j * length * 4, count * 4)
        {costs, trace} = column(values, previous, :infinity, [], [])
        {[:infinity | costs], [trace | traces]}
      end)

    count
    |> path(frames, traces |> Enum.reverse() |> List.to_tuple(), [])
    |> Enum.dedup_by(&elem(&1, 0))
    |> Enum.map(&elem(&1, 1))
  end

  # One column of the costs, from the costs of the column before. Python keeps the costs in
  # float32, so each cost is rounded to float32. :infinity is larger than all numbers, as an
  # infinite float. A value that is not a finite float, and a cost that is too large for
  # float32, are :infinity.
  defp column(<<bits::binary-size(4), rest::binary>>, [diagonal | previous], up, costs, trace) do
    [left | _] = previous

    {cost, step} =
      cond do
        diagonal < up and diagonal < left -> {diagonal, 0}
        up < diagonal and up < left -> {up, 1}
        true -> {left, 2}
      end

    cost = add(bits, cost)
    column(rest, previous, cost, [cost | costs], [step | trace])
  end

  defp column(<<>>, _previous, _up, costs, trace),
    do: {Enum.reverse(costs), trace |> Enum.reverse() |> :erlang.list_to_binary()}

  @largest Nx.to_number(Nx.Constants.max_finite(:f32, backend: Nx.BinaryBackend))

  defp add(<<value::float-32-native>>, cost) when is_float(cost), do: float32(value + cost)
  defp add(_bits, _cost), do: :infinity

  defp float32(cost) when cost > @largest, do: :infinity
  defp float32(cost) when cost < -@largest, do: -@largest

  defp float32(cost) do
    <<rounded::float-32>> = <<cost::float-32>>
    rounded
  end

  # The cells of the path from the last cell back to the first row, as {row, column}.
  defp path(0, _j, _traces, cells), do: cells

  defp path(i, j, traces, cells) do
    cells = [{i - 1, j - 1} | cells]

    case if(j == 0, do: 1, else: :binary.at(elem(traces, j - 1), i - 1)) do
      0 -> path(i - 1, j - 1, traces, cells)
      1 -> path(i - 1, j, traces, cells)
      2 -> path(i, j - 1, traces, cells)
    end
  end

  @replacement "�"
  @punctuation "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
  @opening "\"'“¡¿([{-"
  @closing "\"'.。,，!！?？:：”)]}、"
  @unicode_languages ~w(zh ja th lo my yue)

  @doc """
  Returns the words of `elements`, a list of tuples whose first value is a token, as `{text, elements}` with the elements of each word. `language` is the language code of the text, which is `"en"` for a translation.
  """
  def words(tokenizer, elements, language) do
    tokens = Enum.map(elements, &elem(&1, 0))
    elements = List.to_tuple(elements)
    pieces = split_on_unicode(tokenizer, tokens)
    pieces = if language in @unicode_languages, do: pieces, else: join_on_spaces(pieces)

    for {text, indices} <- merge_punctuation(pieces),
        do: {text, Enum.map(indices, &elem(elements, &1))}
  end

  # _split_tokens_on_unicode: a piece ends at a token after which the text has no
  # replacement character, or has one that the text of all tokens also has there.
  defp split_on_unicode(tokenizer, tokens) do
    full = tokenizer |> Tokenizer.decode(tokens) |> String.codepoints() |> List.to_tuple()

    {pieces, _current, _offset} =
      tokens
      |> Enum.with_index()
      |> Enum.reduce({[], [], 0}, fn {token, i}, {pieces, current, offset} ->
        current = [{token, i} | current]
        {ids, indices} = current |> Enum.reverse() |> Enum.unzip()
        points = tokenizer |> Tokenizer.decode(ids) |> String.codepoints()

        case Enum.find_index(points, &(&1 == @replacement)) do
          index when index == nil or elem(full, offset + index) == @replacement ->
            {[{Enum.join(points), indices} | pieces], [], offset + length(points)}

          _index ->
            {pieces, current, offset}
        end
      end)

    Enum.reverse(pieces)
  end

  # _split_tokens_on_spaces. Python tests the punctuation as a substring, so a piece of only
  # spaces is punctuation too.
  defp join_on_spaces(pieces) do
    pieces
    |> Enum.reduce([], fn
      {text, indices}, [{word, word_indices} | words] ->
        if String.starts_with?(text, " ") or String.contains?(@punctuation, String.trim(text)),
          do: [{text, indices}, {word, word_indices} | words],
          else: [{word <> text, word_indices ++ indices} | words]

      piece, [] ->
        [piece]
    end)
    |> Enum.reverse()
  end

  # _merge_punctuations: opening punctuation with a space before it joins the next word, from
  # the end. Then closing punctuation joins the word before it, unless that word ends with a
  # space. A joined word becomes empty, and stays in place until the end.
  defp merge_punctuation(words) do
    words = List.to_tuple(words)
    last = tuple_size(words) - 1
    words = words |> prepend(last - 1, last) |> append(0, 1)
    for {text, indices} <- Tuple.to_list(words), text != "", do: {text, indices}
  end

  defp prepend(words, i, _j) when i < 0, do: words

  defp prepend(words, i, j) do
    {text, indices} = elem(words, i)

    if String.starts_with?(text, " ") and String.contains?(@opening, String.trim(text)) do
      {next, next_indices} = elem(words, j)

      words
      |> put_elem(j, {text <> next, indices ++ next_indices})
      |> put_elem(i, {"", []})
      |> prepend(i - 1, j)
    else
      prepend(words, i - 1, i)
    end
  end

  defp append(words, _i, j) when j >= tuple_size(words), do: words

  defp append(words, i, j) do
    {text, indices} = elem(words, i)
    {next, next_indices} = elem(words, j)

    if not String.ends_with?(text, " ") and String.contains?(@closing, next) do
      words
      |> put_elem(i, {text <> next, indices ++ next_indices})
      |> put_elem(j, {"", []})
      |> append(i, j + 1)
    else
      append(words, j, j + 1)
    end
  end
end
