defmodule Transmuter.HubTest do
  use ExUnit.Case, async: true

  @moduletag :tmp_dir

  test "an error from the Hub raises and is not saved", %{tmp_dir: dir} do
    assert_raise ArgumentError, ~r/gave status 401/, fn ->
      Transmuter.Hub.files({:hf, "Xenova/no-such-repository"}, cache_dir: dir)
    end

    refute File.exists?(Path.join(dir, "Xenova/no-such-repository/.info.json"))
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
