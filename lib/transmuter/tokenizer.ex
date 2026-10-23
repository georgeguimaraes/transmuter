defmodule Transmuter.Tokenizer do
  @moduledoc """
  Turns text into the input tensors of a model, with the `tokenizer.json` of a model repository.

  Needs the optional `tokenizers` dependency: add `{:tokenizers, "~> 0.5"}` to your deps.

      tokenizer = Transmuter.Tokenizer.load!({:hf, "Xenova/all-MiniLM-L6-v2"})
      {inputs, _encodings} = Transmuter.Tokenizer.encode(tokenizer, ["Hello world"])
      inputs["input_ids"]
  """

  @compile {:no_warn_undefined, [Tokenizers.Tokenizer, Tokenizers.Encoding]}

  alias Tokenizers.{Encoding, Tokenizer}
  alias Transmuter.{Hub, Repo}

  defstruct [
    :native,
    :special_tokens,
    :canonicalize,
    :unknown,
    length: nil,
    pad_direction: :right,
    truncation: :longest_first,
    stride: nil
  ]

  @special_tokens ~w(pad eos bos unk mask cls sep)

  @doc """
  Loads the tokenizer of a `Transmuter.Repo` or of a source (`{:hf, repo}` or a local directory).

  ## Options

    * `:length` - the length of the encoded inputs. `nil` (the default) pads to the longest input. An integer pads to that length. A list of lengths pads to the smallest length that fits the longest input. Inputs longer than the largest length are truncated.

    * `:pad_direction` - `:right` (the default) or `:left`.

    * `:truncation` - the text of a pair that truncation shortens: `:longest_first` (the default), `:only_first` or `:only_second`.

    * `:stride` - `nil` (the default) removes the tokens after the largest length. An integer keeps them in more windows of the largest length. Each window repeats the last `stride` tokens of the window before it. The stride is at most half of the largest length, as in the question answering pipeline of Python transformers.

  Other options go to `Transmuter.Repo.load!/2` when the first argument is a source.
  """
  def load!(repo_or_source, opts \\ [])

  def load!(repo = %Repo{}, opts) do
    unless Code.ensure_loaded?(Tokenizer) do
      raise ArgumentError,
            ~s(Transmuter.Tokenizer needs the optional tokenizers dependency, add {:tokenizers, "~> 0.5"} to your deps)
    end

    path = Hub.fetch!(repo.source, "tokenizer.json", repo.opts)

    {{:ok, native}, unknown} =
      case Hub.fetch(repo.source, "source.spm", repo.opts) do
        {:ok, spm} ->
          native =
            path
            |> File.read!()
            |> JSON.decode!()
            |> source_model(File.read!(spm))
            |> JSON.encode!()
            |> Tokenizer.from_buffer()

          {native, nil}

        :error ->
          json = File.read!(path)

          # The tokenizers library panics on a null charsmap, so these files do not go to it.
          with false <- Regex.match?(~r/"precompiled_charsmap"\s*:\s*null/, json),
               {:ok, native} <- Tokenizer.from_buffer(json) do
            {{:ok, native}, nil}
          else
            _unreadable ->
              {json, unknown} = json |> JSON.decode!() |> readable(repo)
              {json |> JSON.encode!() |> Tokenizer.from_buffer(), unknown}
          end
      end

    # Exports often save a fixed padding in tokenizer.json, and encode/3 pads on its own.
    native = Tokenizer.disable_padding(native)

    tokenizer = %__MODULE__{
      native: native,
      special_tokens: special_tokens(repo),
      canonicalize: canonicalize(repo.tokenizer_config),
      unknown: unknown
    }

    configure(tokenizer, opts)
  end

  def load!(source, opts), do: load!(Repo.load!(source, opts), opts)

  @doc """
  Sets the `:length`, `:pad_direction`, `:truncation` and `:stride` options of `load!/2`.
  """
  def configure(tokenizer, opts) do
    tokenizer =
      struct!(tokenizer, Keyword.take(opts, [:length, :pad_direction, :truncation, :stride]))

    native =
      if tokenizer.length do
        max_length = Enum.max(List.wrap(tokenizer.length))

        Tokenizer.set_truncation(tokenizer.native,
          max_length: max_length,
          strategy: tokenizer.truncation,
          stride: min(tokenizer.stride || 0, div(max_length, 2))
        )
      else
        Tokenizer.disable_truncation(tokenizer.native)
      end

    %{tokenizer | native: native}
  end

  # SiglipTokenizer of Python transformers removes the punctuation of the text, and
  # lowercases it with do_lower_case, before it tokenizes. tokenizer.json does not have
  # these steps.
  defp canonicalize(config = %{"tokenizer_class" => "SiglipTokenizer"}),
    do: [lowercase: Map.get(config, "do_lower_case", true)]

  defp canonicalize(_config), do: nil

  # Marian encodes the source text with the SentencePiece model in source.spm and looks up the
  # pieces in the shared vocabulary, as MarianTokenizer of transformers. tokenizer.json has
  # the shared vocabulary, so its pieces get the scores and the normalizer of source.spm. A
  # piece that source.spm does not have gets a score that the search does not pick.
  defp source_model(json, spm) do
    fields = protobuf(spm)

    scores =
      for {1, piece} <- fields, into: %{} do
        %{1 => text, 2 => <<score::float-little-32>>} = Map.new(protobuf(piece))
        {text, score}
      end

    {3, normalizer} = List.keyfind(fields, 3, 0)
    {2, charsmap} = List.keyfind(protobuf(normalizer), 2, 0)

    vocab =
      for [piece, _score] <- json["model"]["vocab"], do: [piece, Map.get(scores, piece, -1.0e6)]

    json
    |> put_in(["model", "vocab"], vocab)
    |> Map.put("normalizer", %{
      "type" => "Precompiled",
      "precompiled_charsmap" => Base.encode64(charsmap)
    })
  end

  # Some tokenizer.json files of transformers.js exports are not readable by the tokenizers
  # library. A model without a type, such as the characters of VITS, is a vocabulary of whole
  # tokens with the unknown token of the tokenizer in transformers.js, and the library needs
  # the type and the unknown token. A Precompiled normalizer without a charsmap, such as in
  # SpeechT5, changes no character in Python, and the library panics on it.
  defp readable(json, repo) do
    {model, unknown} =
      json["model"] |> typed_model(repo) |> merged_pieces(json["added_tokens"] || [])

    json =
      json
      |> Map.put("model", model)
      |> Map.update("normalizer", nil, &charsmap/1)

    {json, unknown}
  end

  # The BPE model of M2M100 merges pieces that its vocabulary does not have, because its
  # merges come from a SentencePiece model with more pieces. In Python transformers and
  # transformers.js, such a piece is a step of the merges, and it is the unknown token when
  # no merge follows. The library needs each piece of a merge in the vocabulary, so these
  # pieces get the ids after all other ids, and ids/2 gives them the unknown token. Returns
  # the model and {the first id of these pieces, the id of the unknown token}.
  #
  # The library gives an added token that the vocabulary does not have the next free id, not
  # its id in tokenizer.json. So the added tokens go into the vocabulary with their ids
  # first, and the pieces do not take their ids.
  defp merged_pieces(model = %{"type" => "BPE", "merges" => merges, "vocab" => vocab}, added) do
    vocab = Map.merge(Map.new(added, &{&1["content"], &1["id"]}), vocab)

    pieces =
      for merge <- merges,
          [left, right] =
            if(is_binary(merge), do: String.split(merge, " ", parts: 2), else: merge),
          piece <- [left, right, left <> right],
          not Map.has_key?(vocab, piece),
          uniq: true,
          do: piece

    case pieces do
      [] ->
        {model, nil}

      pieces ->
        first = Enum.max(Map.values(vocab)) + 1

        unknown =
          vocab[model["unk_token"]] ||
            raise ArgumentError,
                  "the BPE model of tokenizer.json merges pieces that its vocabulary does not have, and it has no unk_token for them"

        vocab = Map.merge(vocab, Map.new(Enum.with_index(pieces, first)))
        {%{model | "vocab" => vocab}, {first, unknown}}
    end
  end

  defp merged_pieces(model, _added), do: {model, nil}

  defp typed_model(model = %{"vocab" => vocab}, repo)
       when is_map(vocab) and not is_map_key(model, "type"),
       do:
         Map.merge(
           %{"type" => "WordLevel", "unk_token" => Repo.special_token(repo, "unk")},
           model
         )

  defp typed_model(model, _repo), do: model

  defp charsmap(%{"type" => "Precompiled", "precompiled_charsmap" => nil}), do: nil
  defp charsmap(normalizer), do: normalizer

  # The fields of a protobuf message as {number, value}. A value is an integer (varint) or the
  # bytes of the field.
  defp protobuf(<<>>), do: []

  defp protobuf(binary) do
    {key, rest} = varint(binary)

    {value, rest} =
      case Bitwise.band(key, 7) do
        0 ->
          varint(rest)

        1 ->
          <<value::binary-8, rest::binary>> = rest
          {value, rest}

        2 ->
          {length, rest} = varint(rest)
          <<value::binary-size(^length), rest::binary>> = rest
          {value, rest}

        5 ->
          <<value::binary-4, rest::binary>> = rest
          {value, rest}
      end

    [{Bitwise.bsr(key, 3), value} | protobuf(rest)]
  end

  defp varint(<<0::1, n::7, rest::binary>>), do: {n, rest}

  defp varint(<<1::1, n::7, rest::binary>>) do
    {high, rest} = varint(rest)
    {n + Bitwise.bsl(high, 7), rest}
  end

  defp special_tokens(repo) do
    for name <- @special_tokens,
        content = Repo.special_token(repo, name),
        into: %{},
        do: {String.to_atom(name), content}
  end

  @doc """
  Encodes `inputs`, a list of strings or `{text, text_pair}` tuples.

  Returns the input tensors and, for each input, the list of its `Tokenizers.Encoding`, one for each row of the tensors. An input has one row, or one row for each window with `:stride`. An encoding has the offsets, the word ids and the special tokens mask. The tensors are `"input_ids"`, `"attention_mask"` and `"token_type_ids"`, of type `:s64` on `Nx.BinaryBackend`. `"input_ids"` has the ids of `ids/2`. Read the ids of an encoding with `ids/2`, not with `Tokenizers.Encoding.get_ids/1`.

  ## Options

    * `:add_special_tokens` - adds the special tokens of the tokenizer, such as `[CLS]` and `[SEP]`. The default is `true`. Use `false` for text that already has them, such as the output of a chat template.
  """
  def encode(tokenizer, inputs, opts \\ []) do
    inputs =
      if tokenizer.canonicalize,
        do: Enum.map(inputs, &canonical(&1, tokenizer.canonicalize)),
        else: inputs

    encodings =
      case Tokenizer.encode_batch(tokenizer.native, inputs,
             add_special_tokens: Keyword.get(opts, :add_special_tokens, true)
           ) do
        {:ok, encodings} ->
          encodings

        # For example, a first text that leaves no tokens for the second text with :only_second.
        {:error, reason} ->
          raise ArgumentError, "the tokenizer cannot encode the inputs: #{reason}"
      end

    windows =
      if tokenizer.stride,
        do: Enum.map(encodings, &[&1 | Encoding.get_overflowing(&1)]),
        else: Enum.map(encodings, &[&1])

    rows = Enum.concat(windows)
    longest = rows |> Enum.map(&Encoding.n_tokens/1) |> Enum.max()
    length = pad_length(tokenizer.length, longest)

    # Nx has no tensors of length 0.
    if length == 0 do
      raise ArgumentError,
            "the tokenizer gives no tokens for #{inspect(inputs)}, such as for a text without characters that the vocabulary has"
    end

    rows =
      if Enum.all?(rows, &(Encoding.n_tokens(&1) == length)),
        do: rows,
        else: pad(tokenizer, rows, length)

    tensor = &Nx.tensor(Enum.map(rows, &1), type: :s64, backend: Nx.BinaryBackend)

    tensors = %{
      "input_ids" => tensor.(&ids(tokenizer, &1)),
      "attention_mask" => tensor.(&Encoding.get_attention_mask/1),
      "token_type_ids" => tensor.(&Encoding.get_type_ids/1)
    }

    {encodings, []} =
      Enum.map_reduce(windows, rows, &Enum.split(&2, length(&1)))

    {tensors, encodings}
  end

  @doc """
  Returns the token ids of `encoding`, a `Tokenizers.Encoding` of `tokenizer`. The BPE model of M2M100 merges pieces that its vocabulary does not have, and such a piece is the unknown token, as in Python transformers. The other ids are those of the encoding.
  """
  def ids(%{unknown: nil}, encoding), do: Encoding.get_ids(encoding)

  def ids(%{unknown: {first, unknown}}, encoding),
    do: Enum.map(Encoding.get_ids(encoding), &if(&1 < first, do: &1, else: unknown))

  defp canonical({text, pair}, opts), do: {canonical(text, opts), canonical(pair, opts)}

  # The punctuation is the ASCII punctuation of Python, string.punctuation.
  defp canonical(text, opts) do
    text = if opts[:lowercase], do: String.downcase(text), else: text

    text
    |> String.replace(~r/[!-\/:-@\[-`{-~]/, "")
    |> String.replace(~r/\s+/u, " ")
    |> String.trim()
  end

  defp pad_length(nil, longest), do: longest
  defp pad_length(length, _longest) when is_integer(length), do: length

  defp pad_length(lengths, longest) when is_list(lengths),
    do: Enum.find(Enum.sort(lengths), Enum.max(lengths), &(&1 >= longest))

  # Tokenizers without a pad token pad with the eos token, as Python transformers does.
  defp pad(tokenizer, encodings, length) do
    token =
      tokenizer.special_tokens[:pad] || tokenizer.special_tokens[:eos] ||
        raise ArgumentError, "the tokenizer has no pad token and no eos token to pad inputs"

    opts = [
      pad_id: Tokenizer.token_to_id(tokenizer.native, token),
      pad_token: token,
      direction: tokenizer.pad_direction
    ]

    Enum.map(encodings, &Encoding.pad(&1, length, opts))
  end

  @doc """
  Returns the id of `token`, or `nil` when the vocabulary does not have it.
  """
  def token_to_id(tokenizer, token), do: Tokenizer.token_to_id(tokenizer.native, token)

  @doc """
  Decodes token ids into text, without the special tokens.
  """
  def decode(tokenizer, ids) do
    {:ok, text} = Tokenizer.decode(tokenizer.native, ids, skip_special_tokens: true)
    text
  end
end
