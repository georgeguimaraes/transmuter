defmodule Transmuter.TokenizerTest do
  use ExUnit.Case, async: true

  alias Transmuter.Tokenizer

  @moduletag :tmp_dir

  # As in M2M100, the merge "ab c" goes through the piece "ab", which the vocabulary does not
  # have. The added token <extra> has id 10, outside the vocabulary of the model.
  test "a piece of the merges outside the vocabulary is the unknown token, and added tokens keep their ids",
       %{tmp_dir: dir} do
    json = %{
      "version" => "1.0",
      "added_tokens" => [
        %{
          "id" => 10,
          "content" => "<extra>",
          "single_word" => false,
          "lstrip" => false,
          "rstrip" => false,
          "normalized" => false,
          "special" => true
        }
      ],
      "normalizer" => nil,
      "pre_tokenizer" => %{"type" => "WhitespaceSplit"},
      "post_processor" => nil,
      "decoder" => nil,
      "model" => %{
        "type" => "BPE",
        "unk_token" => "<unk>",
        "fuse_unk" => false,
        "byte_fallback" => false,
        "vocab" => %{"<unk>" => 0, "a" => 1, "b" => 2, "c" => 3, "abc" => 4},
        "merges" => ["a b", "ab c"]
      }
    }

    File.write!(Path.join(dir, "tokenizer.json"), JSON.encode!(json))
    tokenizer = Tokenizer.load!(dir)

    {%{"input_ids" => ids}, [[encoding]]} = Tokenizer.encode(tokenizer, ["abc ab <extra>"])
    assert Nx.to_flat_list(ids) == [4, 0, 10]
    assert Tokenizer.ids(tokenizer, encoding) == [4, 0, 10]
  end

  # A WordLevel model without a decoder decodes the tokens with a space between them, as the
  # spaces before punctuation that clean_up_tokenization of Python transformers removes.
  @cleaned [
    {"the end .", "the end."},
    {"why ? now !", "why? now!"},
    {"yes , no", "yes, no"},
    {"rock ' n roll", "rock'n roll"},
    {"do n't", "don't"},
    {"I 'm", "I'm"},
    {"it 's", "it's"},
    {"we 've", "we've"},
    {"they 're", "they're"},
    {"a. b", "a. b"}
  ]

  test "clean_up_tokenization_spaces removes the spaces of clean_up_tokenization", %{
    tmp_dir: dir
  } do
    tokenizer = tokenizer(dir, %{"clean_up_tokenization_spaces" => true})

    for {text, expected} <- @cleaned do
      assert decode(tokenizer, text) == expected

      assert decode(Tokenizer.configure(tokenizer, clean_up_tokenization_spaces: false), text) ==
               text
    end
  end

  for {name, config, expected} <- [
        {"true", %{"clean_up_tokenization_spaces" => true}, "it's"},
        {"false", %{"clean_up_tokenization_spaces" => false}, "it 's"},
        {"null", %{"clean_up_tokenization_spaces" => nil}, "it 's"},
        {"not in the file", %{}, "it 's"}
      ] do
    test "clean_up_tokenization_spaces of tokenizer_config.json is #{name}", %{tmp_dir: dir} do
      assert decode(tokenizer(dir, unquote(Macro.escape(config))), "it 's") == unquote(expected)
    end
  end

  # A task sets the value of its pipeline in Python transformers, such as true for text
  # generation, over the value of tokenizer_config.json.
  test "the :clean_up_tokenization_spaces option replaces the value of tokenizer_config.json",
       %{tmp_dir: dir} do
    tokenizer = tokenizer(dir, %{"clean_up_tokenization_spaces" => false})

    assert decode(Tokenizer.configure(tokenizer, clean_up_tokenization_spaces: true), "it 's") ==
             "it's"
  end

  defp tokenizer(dir, config) do
    words = @cleaned |> Enum.flat_map(&String.split(elem(&1, 0))) |> Enum.uniq()

    json = %{
      "version" => "1.0",
      "added_tokens" => [],
      "normalizer" => nil,
      "pre_tokenizer" => %{"type" => "WhitespaceSplit"},
      "post_processor" => nil,
      "decoder" => nil,
      "model" => %{
        "type" => "WordLevel",
        "unk_token" => "<unk>",
        "vocab" => Map.new(Enum.with_index(["<unk>" | words]))
      }
    }

    File.write!(Path.join(dir, "tokenizer.json"), JSON.encode!(json))
    File.write!(Path.join(dir, "tokenizer_config.json"), JSON.encode!(config))
    Tokenizer.load!(dir)
  end

  defp decode(tokenizer, text) do
    {%{"input_ids" => ids}, _encodings} = Tokenizer.encode(tokenizer, [text])
    Tokenizer.decode(tokenizer, Nx.to_flat_list(ids))
  end
end
