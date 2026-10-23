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
end
