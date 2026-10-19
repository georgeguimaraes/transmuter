defmodule Transmuter.HubTest do
  use ExUnit.Case, async: true

  @moduletag :tmp_dir

  test "an error from the Hub raises and is not saved", %{tmp_dir: dir} do
    assert_raise ArgumentError, ~r/gave status 401/, fn ->
      Transmuter.Hub.files({:hf, "Xenova/no-such-repository"}, cache_dir: dir)
    end

    refute File.exists?(Path.join(dir, "Xenova/no-such-repository/.info.json"))
  end
end
