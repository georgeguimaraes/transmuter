defmodule Transmuter.OpsTest do
  use ExUnit.Case, async: true

  # Each fixture is one small ONNX graph around one operator variant, with reference
  # outputs from ONNX Runtime (graph optimizations off) or from the ONNX reference
  # evaluator. test/support/make_op_fixtures.py writes them, so these tests need no Python.
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

    # A "half" fixture runs f16 and bf16 in their own type, as a model loaded without
    # :compute_type. The others run them in f32.
    compute_type = if meta["half"], do: nil, else: :f32

    model =
      Transmuter.load!(Path.join(dir, meta["onnx_path"]),
        backend: @backend,
        compute_type: compute_type
      )

    # Strings, sequences and empty optionals cannot be compiled inputs, so they are static.
    {static, inputs} =
      Enum.split_with(
        meta["inputs"],
        &(&1["name"] in meta["static_inputs"] or not Map.has_key?(&1, "file"))
      )

    inputs = Map.new(inputs, &{&1["name"], read(dir, &1, @backend)})
    static_inputs = Map.new(static, &{&1["name"], read(dir, &1, Nx.BinaryBackend)})

    outputs =
      if meta["eager"] do
        Transmuter.run(model, model.params, inputs, static_inputs)
      else
        Transmuter.compile(model, inputs, [static_inputs: static_inputs] ++ @compiler).(inputs)
      end

    for spec <- meta["outputs"], not zero_size?(spec) or Map.has_key?(outputs, spec["name"]) do
      expected = read(dir, spec, Nx.BinaryBackend)
      actual = Map.fetch!(outputs, spec["name"])

      if distribution = meta["distribution"],
        do: assert_distribution(spec["name"], actual, expected, distribution),
        else: assert_same(spec["name"], actual, expected, meta["atol"])
    end
  end

  # Random outputs match the reference only in shape and dtype. Their statistics must be
  # within 0.05 standard deviations (0.025 for rates), about 5 standard errors for
  # 10,000 samples.
  defp assert_distribution(name, actual, expected, distribution) do
    actual = Nx.backend_copy(actual, Nx.BinaryBackend)
    assert {Nx.shape(actual), Nx.type(actual)} == {Nx.shape(expected), Nx.type(expected)}, name
    values = Nx.as_type(actual, :f64)
    stat = &Nx.to_number(&1.(values))

    case distribution do
      %{"frequencies" => rows} ->
        for {row, b} <- Enum.with_index(rows), {frequency, class} <- Enum.with_index(row) do
          actual_frequency = values[b] |> Nx.equal(class) |> Nx.mean() |> Nx.to_number()
          assert_in_delta actual_frequency, frequency, 0.025, "#{name}[#{b}] class #{class}"
        end

      %{"rate" => rate} ->
        assert_in_delta stat.(&Nx.mean/1), rate, 0.025, name

      %{"mean" => mean, "std" => std} ->
        assert_in_delta stat.(&Nx.mean/1), mean, 0.05 * std, "#{name} mean"
        assert_in_delta stat.(&Nx.standard_deviation/1), std, 0.05 * std, "#{name} std"

        if low = distribution["low"] do
          {min, max} = {stat.(&Nx.reduce_min/1), stat.(&Nx.reduce_max/1)}
          assert min >= low and max < distribution["high"], "#{name} range"
        end
    end
  end

  defp zero_size?(spec), do: 0 in Map.get(spec, "shape", [1])

  defp assert_same(name, actual, {:sequence, expected}, atol) do
    assert is_list(actual), "#{name}: expected a sequence, got #{inspect(actual)}"
    assert length(actual) == length(expected), "#{name}: sequence length"

    for {a, e, i} <- Enum.zip([actual, expected, 0..(length(expected) - 1)//1]),
        do: assert_same("#{name}[#{i}]", a, e, atol)
  end

  defp assert_same(name, actual, expected, _atol) when not is_struct(expected, Nx.Tensor),
    do: assert(actual == expected, "#{name}: #{inspect(actual)}, expected #{inspect(expected)}")

  defp assert_same(name, actual, expected, atol) do
    actual = Nx.backend_copy(actual, Nx.BinaryBackend)

    assert Nx.shape(actual) == Nx.shape(expected),
           "#{name}: shape #{inspect(Nx.shape(actual))}, expected #{inspect(Nx.shape(expected))}"

    {max_diff, tolerance} = difference(actual, expected)

    tolerance =
      cond do
        atol -> atol
        Nx.Type.integer?(Nx.type(expected)) -> 0
        true -> tolerance
      end

    assert max_diff <= tolerance, "#{name}: max difference #{max_diff}, tolerance #{tolerance}"
  end

  # Largest difference, with equal infinities and matching NaNs counted as equal, and
  # the tolerance 1e-4 + 1e-3 * max |expected| over the finite expected values. Integer
  # outputs must be equal, unless the fixture sets atol.
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

  defp read(_dir, %{"none" => true}, _backend), do: :none

  defp read(dir, %{"sequence" => elements}, backend),
    do: {:sequence, Enum.map(elements, &read(dir, &1, backend))}

  defp read(_dir, %{"dtype" => "string", "shape" => shape, "values" => values}, _backend),
    do: {:strings, List.to_tuple(shape), values}

  defp read(dir, %{"file" => file, "dtype" => type, "shape" => shape}, backend) do
    if 0 in shape do
      {:empty, List.to_tuple(shape)}
    else
      dir
      |> Path.join(file)
      |> File.read!()
      |> Nx.from_binary(String.to_atom(type), backend: backend)
      |> Nx.reshape(List.to_tuple(shape))
    end
  end
end
