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

  defstruct [:native, :special_tokens, length: nil, pad_direction: :right]

  @special_tokens ~w(pad eos bos unk mask cls sep)

  @doc """
  Loads the tokenizer of a `Transmuter.Repo` or of a source (`{:hf, repo}` or a local directory).

  ## Options

    * `:length` - the length of the encoded inputs. `nil` (the default) pads to the longest input. An integer pads to that length. A list of lengths pads to the smallest length that fits the longest input. Inputs longer than the largest length are truncated.

    * `:pad_direction` - `:right` (the default) or `:left`.

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

    configure(%__MODULE__{native: native, special_tokens: special_tokens(repo)}, opts)
  end

  def load!(source, opts), do: load!(Repo.load!(source, opts), opts)

  @doc """
  Sets the `:length` and `:pad_direction` options of `load!/2`.
  """
  def configure(tokenizer, opts) do
    tokenizer = struct!(tokenizer, Keyword.take(opts, [:length, :pad_direction]))

    native =
      if tokenizer.length,
        do:
          Tokenizer.set_truncation(tokenizer.native,
            max_length: Enum.max(List.wrap(tokenizer.length))
          ),
        else: Tokenizer.disable_truncation(tokenizer.native)

    %{tokenizer | native: native}
  end

  # special_tokens_map.json overrides tokenizer_config.json, as in Python transformers. A token
  # is a string or a map with "content".
  defp special_tokens(repo) do
    for name <- @special_tokens,
        token = repo.special_tokens[name <> "_token"] || repo.tokenizer_config[name <> "_token"],
        content = if(is_map(token), do: token["content"], else: token),
        is_binary(content),
        into: %{},
        do: {String.to_atom(name), content}
  end

  @doc """
  Encodes `inputs`, a list of strings or `{text, text_pair}` tuples.

  Returns the input tensors and the `Tokenizers.Encoding` of each input, which has the offsets, the word ids and the special tokens mask. The tensors are `"input_ids"`, `"attention_mask"` and `"token_type_ids"`, of type `:s64` on `Nx.BinaryBackend`.

  ## Options

    * `:add_special_tokens` - adds the special tokens of the tokenizer, such as `[CLS]` and `[SEP]`. The default is `true`. Use `false` for text that already has them, such as the output of a chat template.
  """
  def encode(tokenizer, inputs, opts \\ []) do
    {:ok, encodings} =
      Tokenizer.encode_batch(tokenizer.native, inputs,
        add_special_tokens: Keyword.get(opts, :add_special_tokens, true)
      )

    longest = encodings |> Enum.map(&Encoding.n_tokens/1) |> Enum.max()
    length = pad_length(tokenizer.length, longest)

    encodings =
      if Enum.all?(encodings, &(Encoding.n_tokens(&1) == length)),
        do: encodings,
        else: pad(tokenizer, encodings, length)

    tensor = &Nx.tensor(Enum.map(encodings, &1), type: :s64, backend: Nx.BinaryBackend)

    inputs = %{
      "input_ids" => tensor.(&Encoding.get_ids/1),
      "attention_mask" => tensor.(&Encoding.get_attention_mask/1),
      "token_type_ids" => tensor.(&Encoding.get_type_ids/1)
    }

    {inputs, encodings}
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
