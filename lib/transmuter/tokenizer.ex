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

    {:ok, native} = Tokenizer.from_file(Hub.fetch!(repo.source, "tokenizer.json", repo.opts))
    # Exports often save a fixed padding in tokenizer.json, and encode/3 pads on its own.
    native = Tokenizer.disable_padding(native)

    tokenizer = %__MODULE__{
      native: native,
      special_tokens: special_tokens(repo),
      canonicalize: canonicalize(repo.tokenizer_config)
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

  defp special_tokens(repo) do
    for name <- @special_tokens,
        content = Repo.special_token(repo, name),
        into: %{},
        do: {String.to_atom(name), content}
  end

  @doc """
  Encodes `inputs`, a list of strings or `{text, text_pair}` tuples.

  Returns the input tensors and, for each input, the list of its `Tokenizers.Encoding`, one for each row of the tensors. An input has one row, or one row for each window with `:stride`. An encoding has the offsets, the word ids and the special tokens mask. The tensors are `"input_ids"`, `"attention_mask"` and `"token_type_ids"`, of type `:s64` on `Nx.BinaryBackend`.

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

    rows =
      if Enum.all?(rows, &(Encoding.n_tokens(&1) == length)),
        do: rows,
        else: pad(tokenizer, rows, length)

    tensor = &Nx.tensor(Enum.map(rows, &1), type: :s64, backend: Nx.BinaryBackend)

    tensors = %{
      "input_ids" => tensor.(&Encoding.get_ids/1),
      "attention_mask" => tensor.(&Encoding.get_attention_mask/1),
      "token_type_ids" => tensor.(&Encoding.get_type_ids/1)
    }

    {encodings, []} =
      Enum.map_reduce(windows, rows, &Enum.split(&2, length(&1)))

    {tensors, encodings}
  end

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
