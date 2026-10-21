defmodule Transmuter.CIPartitionTest do
  # The async modules start while other test files still compile, but a sync module runs
  # after all test files are loaded.
  use ExUnit.Case, async: false

  # CI selects partitions 2 to 4 with --only ci_partition:N. An included tag wins over an
  # excluded one, so such a run would also run the :large_model tests of the partition.
  test "no test has both a :ci_partition and a :large_model tag" do
    both =
      for {module, _} <- :code.all_loaded(),
          function_exported?(module, :__ex_unit__, 0),
          test_module = module.__ex_unit__(),
          test <- test_module.tests,
          tags = Map.merge(test_module.tags, test.tags),
          tags[:ci_partition] && tags[:large_model],
          do: "#{inspect(module)}: #{test.name}"

    assert both == []
  end
end
