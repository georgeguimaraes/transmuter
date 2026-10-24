defmodule Transmuter.RepoTest do
  use ExUnit.Case, async: true

  alias Transmuter.Repo

  @moduletag :hub
  @moduletag :tmp_dir

  defp repo(dir, files, opts \\ []) do
    for {name, content} <- files do
      path = Path.join(dir, name)
      File.mkdir_p!(Path.dirname(path))
      File.write!(path, if(is_binary(content), do: content, else: JSON.encode!(content)))
    end

    Repo.load!(dir, opts)
  end

  test "the variant comes from the option, then transformers.js_config, then the default", %{
    tmp_dir: dir
  } do
    files = %{
      "onnx/model.onnx" => "",
      "onnx/model_q4.onnx" => "",
      "onnx/embed_tokens_fp16.onnx" => "",
      "onnx/embed_tokens.onnx" => "",
      "config.json" => %{"transformers.js_config" => %{"dtype" => %{"embed_tokens" => "fp16"}}}
    }

    repo = repo(dir, files)
    assert Repo.component_file(repo, "model") == "onnx/model.onnx"
    assert Repo.component_file(repo, "model", :q4) == "onnx/model_q4.onnx"
    assert Repo.component_file(repo, "embed_tokens", :q4) == "onnx/embed_tokens_fp16.onnx"

    repo = repo(dir, %{}, variant: %{"embed_tokens" => :fp32, "model" => :q4})
    assert Repo.component_file(repo, "embed_tokens") == "onnx/embed_tokens.onnx"
    assert Repo.component_file(repo, "model") == "onnx/model_q4.onnx"
  end

  test "a default that the repository does not have falls back to fp32, then other variants", %{
    tmp_dir: dir
  } do
    repo = repo(dir, %{"model.onnx" => "", "onnx/decoder_model_merged_quantized.onnx" => ""})

    assert Repo.component_file(repo, "model", :q4) == "model.onnx"

    assert Repo.component_file(repo, "decoder_model_merged", :q4) ==
             "onnx/decoder_model_merged_quantized.onnx"
  end

  test "a requested variant that the repository does not have raises with the variants", %{
    tmp_dir: dir
  } do
    repo = repo(dir, %{"onnx/model.onnx" => "", "onnx/model_q4.onnx" => ""}, variant: :fp16)

    assert_raise ArgumentError, ~r/no fp16 file for model, variants: \[:fp32, :q4\]/, fn ->
      Repo.component_file(repo, "model")
    end
  end

  test "the chat template comes from chat_template.jinja, then tokenizer_config.json", %{
    tmp_dir: dir
  } do
    named = [
      %{"name" => "tool_use", "template" => "T"},
      %{"name" => "default", "template" => "D"}
    ]

    assert repo(dir, %{"tokenizer_config.json" => %{"chat_template" => named}}).chat_template ==
             "D"

    assert repo(dir, %{"chat_template.jinja" => "J"}).chat_template == "J"
  end

  # As ProcessorMixin of transformers 4.57: chat_template.json comes before chat_template.jinja,
  # and processor_config.json comes last. The tokenizer reads chat_template.jinja first.
  test "the template of the processor comes from chat_template.json, .jinja, then processor_config.json",
       %{tmp_dir: dir} do
    all = %{
      "chat_template.json" => %{"chat_template" => "P"},
      "chat_template.jinja" => "J",
      "tokenizer_config.json" => %{"chat_template" => "T"},
      "processor_config.json" => %{"chat_template" => "C"}
    }

    for {files, tokenizer, processor} <- [
          {all, "J", "P"},
          {Map.delete(all, "chat_template.json"), "J", "J"},
          {Map.take(all, ["tokenizer_config.json", "processor_config.json"]), "T", "C"},
          {Map.take(all, ["processor_config.json"]), "C", "C"}
        ] do
      repo = repo(Path.join(dir, Enum.join(Enum.sort(Map.keys(files)), "-")), files)
      assert {repo.chat_template, repo.processor_chat_template} == {tokenizer, processor}
    end
  end

  # GenerationConfig.from_model_config of transformers: config.json without its null keys, and
  # the decoder for the keys that config.json leaves at their default.
  test "without generation_config.json, the generation settings come from config.json", %{
    tmp_dir: dir
  } do
    config = %{
      "no_repeat_ngram_size" => 3,
      "min_length" => 0,
      "forced_bos_token_id" => nil,
      "decoder" => %{"no_repeat_ngram_size" => 2, "min_length" => 5, "forced_bos_token_id" => 7}
    }

    generation = repo(dir, %{"config.json" => config}).generation_config
    assert generation["no_repeat_ngram_size"] == 3
    assert generation["min_length"] == 5
    assert generation["forced_bos_token_id"] == 7

    File.write!(Path.join(dir, "generation_config.json"), JSON.encode!(%{"min_length" => 1}))
    assert repo(dir, %{}).generation_config == %{"min_length" => 1}
  end
end
