defmodule Transmuter.HubTest do
  use ExUnit.Case, async: true

  @moduletag :hub
  @moduletag :tmp_dir

  test "an error from the Hub raises and is not saved", %{tmp_dir: dir} do
    assert_raise ArgumentError, ~r/gave status 401/, fn ->
      Transmuter.Hub.files({:hf, "Xenova/no-such-repository"}, cache_dir: dir)
    end

    refute File.exists?(Path.join(dir, "Xenova/no-such-repository/.info.json"))
  end

  test "offline mode reads only the cache", %{tmp_dir: dir} do
    source = {:hf, "Xenova/all-MiniLM-L6-v2"}
    online = Transmuter.Repo.load!(source, cache_dir: dir)
    offline = Transmuter.Repo.load!(source, cache_dir: dir, offline: true)
    assert offline.config == online.config and offline.files == online.files

    assert_raise ArgumentError,
                 ~r/onnx\/model.onnx of .* not in the cache, and the Hub is offline/,
                 fn ->
                   Transmuter.Hub.fetch!(source, "onnx/model.onnx", cache_dir: dir, offline: true)
                 end

    assert_raise ArgumentError, ~r/Xenova\/bert-base-uncased is not in the cache/, fn ->
      Transmuter.Hub.files({:hf, "Xenova/bert-base-uncased"}, cache_dir: dir, offline: true)
    end
  end

  test "concurrent downloads of one file all succeed", %{tmp_dir: dir} do
    paths =
      1..8
      |> Task.async_stream(
        fn _ ->
          Transmuter.Hub.fetch!({:hf, "Xenova/all-MiniLM-L6-v2"}, "config.json", cache_dir: dir)
        end,
        timeout: 60_000
      )
      |> Enum.map(fn {:ok, path} -> path end)

    assert [path] = Enum.uniq(paths)
    assert %{"model_type" => "bert"} = path |> File.read!() |> JSON.decode!()
    assert Path.wildcard(Path.join(dir, "**/*.part")) == []
  end
end
