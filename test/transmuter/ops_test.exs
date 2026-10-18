defmodule Transmuter.OpsTest do
  use ExUnit.Case, async: true

  # Each fixture is one small ONNX graph around one operator variant, with reference
  # outputs from ONNX Runtime (graph optimizations off) or from the ONNX reference
  # evaluator. test/support/op_cases.py writes them, so these tests need no Python.
  @fixtures Path.expand("../fixtures/ops", __DIR__)

  @device (System.get_env("TRANSMUTER_TEST_DEVICE") ||
             if(match?({:unix, :darwin}, :os.type()), do: "gpu", else: "cpu"))
          |> String.to_existing_atom()
  @backend {EMLX.Backend, device: @device}
  @compiler [compiler: EMLX, device: @device]

  for dir <- @fixtures |> Path.join("*") |> Path.wildcard() |> Enum.sort() do
    @dir dir
    test Path.basename(dir) do
      assert_matches_reference(@dir)
    end
  end

  defp assert_matches_reference(dir) do
    meta = dir |> Path.join("meta.json") |> File.read!() |> JSON.decode!()

    model =
      Transmuter.load!(Path.join(dir, meta["onnx_path"]), backend: @backend, compute_type: :f32)

    {static, inputs} = Enum.split_with(meta["inputs"], &(&1["name"] in meta["static_inputs"]))
    inputs = Map.new(inputs, &{&1["name"], read(dir, &1, @backend)})
    static_inputs = Map.new(static, &{&1["name"], read(dir, &1, Nx.BinaryBackend)})

    outputs =
      Transmuter.compile(model, inputs, [static_inputs: static_inputs] ++ @compiler).(inputs)

    for spec <- meta["outputs"] do
      expected = read(dir, spec, Nx.BinaryBackend)
      actual = outputs |> Map.fetch!(spec["name"]) |> Nx.backend_copy(Nx.BinaryBackend)

      assert Nx.shape(actual) == Nx.shape(expected),
             "#{spec["name"]}: shape #{inspect(Nx.shape(actual))}, expected #{inspect(Nx.shape(expected))}"

      {max_diff, tolerance} = difference(actual, expected)

      assert max_diff <= tolerance,
             "#{spec["name"]}: max difference #{max_diff}, tolerance #{tolerance}"
    end
  end

  # Largest difference, with equal infinities and matching NaNs counted as equal, and
  # the tolerance 1e-4 + 1e-3 * max |expected| over the finite expected values.
  defp difference(actual, expected) do
    actual = Nx.as_type(actual, :f64)
    expected = Nx.as_type(expected, :f64)
    expected_nan = Nx.is_nan(expected)

    same =
      Nx.logical_or(Nx.logical_and(expected_nan, Nx.is_nan(actual)), Nx.equal(actual, expected))

    diff =
      Nx.select(same, 0, Nx.abs(Nx.subtract(actual, expected)))
      |> Nx.reduce_max()
      |> Nx.to_number()

    finite = Nx.logical_not(Nx.logical_or(expected_nan, Nx.is_infinity(expected)))
    scale = Nx.select(finite, Nx.abs(expected), 0) |> Nx.reduce_max() |> Nx.to_number()
    {if(diff == :nan, do: :infinity, else: diff), 1.0e-4 + 1.0e-3 * scale}
  end

  defp read(dir, %{"file" => file, "dtype" => type, "shape" => shape}, backend) do
    dir
    |> Path.join(file)
    |> File.read!()
    |> Nx.from_binary(String.to_atom(type), backend: backend)
    |> Nx.reshape(List.to_tuple(shape))
  end
end
