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

  test "a revision gives the files of that commit, branch or tag", %{tmp_dir: dir} do
    source = {:hf, "Xenova/all-MiniLM-L6-v2"}
    files = &Transmuter.Hub.files(source, cache_dir: dir, revision: &1)

    # The commit 57cbdab uploaded model_int8.onnx to the root, and cb3d680 moved it to onnx/.
    assert "model_int8.onnx" in files.("57cbdab1181b7eba8170805260bd39e733b0e25d")
    refute "onnx/model_int8.onnx" in files.("57cbdab1181b7eba8170805260bd39e733b0e25d")
    assert "onnx/model_int8.onnx" in files.(nil)
    assert "onnx/model_fp16.onnx" in files.("refs/pr/1")

    readme = &Transmuter.Hub.fetch!(source, "README.md", cache_dir: dir, revision: &1)
    old = readme.("a162ec1733c9eead894ebbf7cf40d15ecf3f3f65")

    assert Path.dirname(old) ==
             Path.join(dir, "Xenova/all-MiniLM-L6-v2@a162ec1733c9eead894ebbf7cf40d15ecf3f3f65")

    assert File.read!(old) != File.read!(readme.(nil))

    assert_raise ArgumentError, ~r/has no revision no-such-revision/, fn ->
      files.("no-such-revision")
    end
  end

  # config.json is a Git blob, and onnx/model.onnx is an LFS file, so each case checks both
  # kinds of ETag.
  test "a file downloads again when its ETag changes", %{tmp_dir: dir} do
    source = {:hf, "hf-internal-testing/tiny-random-bert"}
    old = {{2000, 1, 1}, {0, 0, 0}}

    for file <- ["config.json", "onnx/model.onnx"] do
      path = Transmuter.Hub.fetch!(source, file, cache_dir: dir)
      etag_path = Path.join(Path.dirname(path), ".#{Path.basename(path)}.etag")
      contents = File.read!(path)
      etag = File.read!(etag_path)

      cases = [
        # A file of an earlier version of Transmuter has no saved ETag. Its contents match.
        {"no saved ETag", fn -> File.rm!(etag_path) end, :kept},
        {"no saved ETag, other contents",
         fn -> File.rm!(etag_path) == :ok and File.write!(path, "x") end, :downloaded},
        {"a new ETag on the Hub",
         fn -> File.write!(etag_path, String.duplicate("0", byte_size(etag))) end, :downloaded}
      ]

      for {name, change, result} <- cases do
        File.touch!(path, old)
        change.()
        assert Transmuter.Hub.fetch!(source, file, cache_dir: dir) == path

        assert {File.read!(path) == contents, File.read!(etag_path),
                File.stat!(path).mtime == old} ==
                 {true, etag, result == :kept},
               "#{file}: #{name}"
      end
    end

    info = Path.join(dir, "hf-internal-testing/tiny-random-bert/.info.json")

    File.write!(
      info,
      String.replace(File.read!(info), etag_of(info, "config.json"), String.duplicate("0", 40))
    )

    File.rm!(Path.join(dir, "hf-internal-testing/tiny-random-bert/config.json"))

    assert_raise ArgumentError, ~r/the download of config.json .* does not match its ETag/, fn ->
      Transmuter.Hub.fetch!(source, "config.json", cache_dir: dir)
    end

    refute File.exists?(Path.join(dir, "hf-internal-testing/tiny-random-bert/config.json"))
  end

  defp etag_of(info, file) do
    info
    |> File.read!()
    |> JSON.decode!()
    |> Map.fetch!("siblings")
    |> Enum.find(&(&1["rfilename"] == file))
    |> Map.fetch!("blobId")
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
