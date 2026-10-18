defmodule Transmuter.Ops.Text do
  @moduledoc false

  # String operators and TfIdfVectorizer. They run on the host, so a numeric
  # TfIdfVectorizer input must be known: static, or read while the model runs eagerly.

  import Transmuter.Ops, only: [attr: 2, attr: 3, broadcast_shape: 1, known?: 1, static_tensor: 2]

  @op_types ~w(StringNormalizer StringConcat StringSplit RegexFullMatch TfIdfVectorizer)

  @doc false
  def op_types, do: @op_types

  @doc false
  def apply(node, inputs), do: node |> op(inputs) |> List.wrap()

  # Drops the elements that are stop words, then changes case, as the spec and ONNX Runtime
  # do. The ONNX reference evaluator also drops stop words inside an element, drops empty
  # strings and strips accents.
  defp op(node = %{op_type: "StringNormalizer"}, [{:strings, shape, values}]) do
    unless tuple_size(shape) == 1 or match?({1, _}, shape) do
      raise ArgumentError, "StringNormalizer takes a [C] or [1, C] tensor"
    end

    check_locale!(attr(node, "locale", ""))
    key = if attr(node, "is_case_sensitive", 0) == 1, do: & &1, else: &change_case(&1, "LOWER")
    stopwords = node |> attr("stopwords", []) |> MapSet.new(key)
    action = attr(node, "case_change_action", "NONE")

    kept =
      values
      |> Enum.reject(&MapSet.member?(stopwords, key.(&1)))
      |> Enum.map(&change_case(&1, action))
      |> then(&if(&1 == [], do: [""], else: &1))

    {:strings, put_elem(shape, tuple_size(shape) - 1, length(kept)), kept}
  end

  defp op(%{op_type: "StringConcat"}, [{:strings, a_shape, a}, {:strings, b_shape, b}]) do
    shape = broadcast_shape([a_shape, b_shape])

    {:strings, shape,
     Enum.zip_with(broadcast(a, a_shape, shape), broadcast(b, b_shape, shape), &<>/2)}
  end

  defp op(node = %{op_type: "StringSplit"}, [{:strings, shape, values}]) do
    delimiter = attr(node, "delimiter", "")
    parts = if max = attr(node, "maxsplit"), do: max + 1, else: :infinity
    substrings = Enum.map(values, &split(&1, delimiter, parts))
    counts = Enum.map(substrings, &length/1)
    width = Enum.max(counts, fn -> 0 end)

    padded =
      if width == 0 do
        {:empty, List.to_tuple(Tuple.to_list(shape) ++ [0])}
      else
        values = Enum.flat_map(substrings, &(&1 ++ List.duplicate("", width - length(&1))))
        {:strings, List.to_tuple(Tuple.to_list(shape) ++ [width]), values}
      end

    [padded, counts |> static_tensor(:s64) |> Nx.reshape(shape)]
  end

  # ONNX uses RE2 syntax, which PCRE mostly extends: PCRE also accepts backreferences and
  # lookarounds, which RE2 rejects. Without :ucp, \d, \w and \s match ASCII only, as in RE2.
  defp op(node = %{op_type: "RegexFullMatch"}, [{:strings, shape, values}]) do
    regex = Regex.compile!("\\A(?:#{attr(node, "pattern")})\\z", [:unicode])

    values
    |> Enum.map(&if(Regex.match?(regex, &1), do: 1, else: 0))
    |> static_tensor(:u8)
    |> Nx.reshape(shape)
  end

  defp op(node = %{op_type: "TfIdfVectorizer"}, [x]) do
    {shape, values} = host_values(x)
    {min_n, max_n} = {attr(node, "min_gram_length"), attr(node, "max_gram_length")}
    max_skip = attr(node, "max_skip_count")
    indexes = node |> attr("ngram_indexes") |> List.to_tuple()
    size = (indexes |> Tuple.to_list() |> Enum.max()) + 1
    weights = node |> attr("weights", []) |> List.to_tuple()
    mode = attr(node, "mode")

    # Pool n-gram k counts into column ngram_indexes[k] with weights[k], as in the spec and
    # ONNX Runtime. The ONNX reference evaluator takes the weight of the column instead.
    grams =
      (attr(node, "pool_int64s") || attr(node, "pool_strings"))
      |> pool_grams(attr(node, "ngram_counts"))
      |> Enum.with_index()
      |> Enum.filter(fn {gram, _k} -> length(gram) in min_n..max_n//1 end)
      |> Map.new()

    rows = if tuple_size(shape) == 2, do: Enum.chunk_every(values, elem(shape, 1)), else: [values]

    rows
    |> Enum.flat_map(fn row ->
      scores =
        row
        |> List.to_tuple()
        |> count_grams(grams, min_n, max_n, max_skip)
        |> Map.new(fn {k, tf} ->
          weight = if k < tuple_size(weights), do: elem(weights, k), else: 1.0

          case mode do
            "TF" -> {elem(indexes, k), tf}
            "IDF" -> {elem(indexes, k), weight}
            "TFIDF" -> {elem(indexes, k), tf * weight}
          end
        end)

      for column <- 0..(size - 1), do: Map.get(scores, column, 0)
    end)
    |> static_tensor(:f32)
    |> Nx.reshape(if tuple_size(shape) == 2, do: {elem(shape, 0), size}, else: {size})
  end

  # Case changes map one code point to one code point, as ONNX Runtime does: "ß" stays
  # "ß" in UPPER instead of becoming "SS".
  defp change_case(s, "NONE"), do: s

  defp change_case(s, action) do
    fun = if action == "LOWER", do: &String.downcase/1, else: &String.upcase/1

    for <<c::utf8 <- s>>, into: "" do
      mapped = fun.(<<c::utf8>>)
      if match?(<<_::utf8>>, mapped), do: mapped, else: <<c::utf8>>
    end
  end

  # Only Turkish, Azeri and Lithuanian change case differently from the default.
  defp check_locale!(locale) do
    language = locale |> String.split(["_", "-", "."]) |> hd() |> String.downcase()

    if language in ~w(tr az lt) do
      raise ArgumentError,
            "StringNormalizer does not support locale #{locale}, its case mapping differs from the default"
    end
  end

  defp broadcast(values, shape, shape), do: values

  defp broadcast(values, from, to) do
    values = List.to_tuple(values)

    from
    |> Nx.iota(backend: Nx.BinaryBackend)
    |> Nx.broadcast(to)
    |> Nx.to_flat_list()
    |> Enum.map(&elem(values, &1))
  end

  # ONNX Runtime returns no substrings for an empty string, also with a delimiter. With
  # no delimiter, whitespace runs separate substrings and the ends are trimmed.
  defp split("", _delimiter, _parts), do: []

  defp split(s, "", parts) do
    case String.trim(s) do
      "" -> []
      trimmed -> Regex.split(~r/\s+/u, trimmed, parts: parts)
    end
  end

  defp split(s, delimiter, parts), do: String.split(s, delimiter, parts: parts)

  defp host_values({:strings, shape, values}), do: {shape, values}

  defp host_values(x) do
    unless known?(x) do
      raise ArgumentError,
            "TfIdfVectorizer reads its input on the host. Pass the graph inputs it depends on " <>
              "as static_inputs, or call Transmuter.run/4 without Nx.Defn.jit"
    end

    {Nx.shape(x), Nx.to_flat_list(x)}
  end

  # The pool lists the 1-grams, then the 2-grams and so on. ngram_counts gives the start
  # of each size.
  defp pool_grams(pool, starts) do
    ends = Enum.drop(starts, 1) ++ [length(pool)]

    [starts, ends, 1..length(starts)]
    |> Enum.zip_with(fn [first, last, n] ->
      pool |> Enum.slice(first, last - first) |> Enum.chunk_every(n, n, :discard)
    end)
    |> Enum.concat()
  end

  # An n-gram takes every (skip + 1)-th item. 1-grams count once, not once per skip.
  defp count_grams(row, grams, min_n, max_n, max_skip) do
    for skip <- 0..max_skip,
        n <- max(min_n, if(skip > 0, do: 2, else: 1))..max_n//1,
        start <- 0..(tuple_size(row) - 1 - (n - 1) * (skip + 1))//1,
        k = Map.get(grams, for(i <- 0..(n - 1), do: elem(row, start + i * (skip + 1)))),
        k != nil,
        reduce: %{} do
      counts -> Map.update(counts, k, 1, &(&1 + 1))
    end
  end
end
