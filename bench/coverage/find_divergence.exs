# Runs one case eagerly and prints the first node whose output differs from ORT.
# Run trace_ort.py on the case first to write ort_trace.json.
# Usage: mix run bench/coverage/find_divergence.exs CASE_DIR

[case_dir] = System.argv()
meta = case_dir |> Path.join("meta.json") |> File.read!() |> JSON.decode!()
ref = case_dir |> Path.join("ort_trace.json") |> File.read!() |> JSON.decode!()
# The backend of the tests: TRANSMUTER_TEST_BACKEND and TRANSMUTER_TEST_DEVICE.
Code.require_file("../../test/support/test_backend.exs", __DIR__)
backend = Transmuter.TestBackend.backend()

read = fn %{"file" => file, "dtype" => type, "shape" => shape}, backend ->
  case_dir
  |> Path.join(file)
  |> File.read!()
  |> Nx.from_binary(String.to_atom(type), backend: backend)
  |> Nx.reshape(List.to_tuple(shape))
end

static? = fn spec ->
  spec["name"] in ["use_cache_branch", "num_logits_to_keep"] or
    (Map.has_key?(meta["ops"], "NonZero") and spec["dtype"] in ["s64", "s32"])
end

# COMPUTE_TYPE=none keeps the types of the graph, to test half-precision graphs as they are.
compute_type = System.get_env("COMPUTE_TYPE", "f32")
compute_type = if compute_type == "none", do: nil, else: String.to_atom(compute_type)
model = Transmuter.load!(Path.expand(meta["onnx_path"], case_dir), compute_type: compute_type)
params = Map.new(model.params, fn {k, v} -> {k, Nx.backend_transfer(v, backend)} end)

inputs =
  Map.new(meta["inputs"], fn spec ->
    {spec["name"], read.(spec, if(static?.(spec), do: Nx.BinaryBackend, else: backend))}
  end)

# The sample can miss a few changed values, so integer outputs must also match the
# whole-tensor sum of absolute values exactly.
# THRESHOLD sets the relative difference that counts as a divergence. A small value finds
# where a deterministic difference starts, before the graph makes it large.
threshold = "THRESHOLD" |> System.get_env("1.0e-3") |> String.to_float()

diverged? = fn name, t ->
  r = ref[name]
  integer? = Nx.Type.integer?(Nx.type(t))
  # The integer sum is exact, as in trace_ort.py: s64 for the narrow types, and Elixir
  # integers for the 64-bit types. EMLX has no exact f64.
  abssum =
    if integer? do
      t = Nx.backend_copy(t, Nx.BinaryBackend)

      case Nx.type(t) do
        {_, 64} -> t |> Nx.to_flat_list() |> Enum.reduce(0, &(abs(&1) + &2))
        _ -> t |> Nx.as_type(:s64) |> Nx.abs() |> Nx.sum() |> Nx.to_number()
      end
    end

  t = Nx.as_type(t, :f64)
  t = t |> then(&Nx.select(Nx.is_nan(&1), 0, &1)) |> Nx.clip(-3.0e38, 3.0e38)
  flat = Nx.reshape(t, {:auto})
  idx = for k <- 0..1023, do: rem(k * 2_654_435_761, Nx.size(t))
  sample = flat |> Nx.take(Nx.tensor(idx)) |> Nx.to_flat_list()

  diff =
    Enum.zip_with(sample, r["sample"], fn a, b -> if a == b, do: 0, else: abs(a - b) end)
    |> Enum.max()

  shape = Tuple.to_list(Nx.shape(t))
  sum_diff? = integer? and r["abssum"] != nil and abssum != r["abssum"]
  diff = if sum_diff?, do: max(diff, abs(abssum - r["abssum"])), else: diff

  {shape != r["shape"] or sum_diff? or diff > threshold * max(r["absmax"], 1.0), shape, sample,
   diff}
end

env = model.static |> Map.merge(params) |> Map.merge(inputs)
key = Nx.Random.key(System.unique_integer([:positive]))
ctx = %{opset: model.opset, compute_type: compute_type, key: key, backend: backend}

model.nodes
|> Enum.reduce_while(env, fn node, env ->
  ins =
    Enum.map(node.input, fn
      "" -> nil
      name -> Map.fetch!(env, name)
    end)

  # One step of the interpreter, so that If, Loop and Scan nodes run as in Transmuter.run/4.
  result = Transmuter.run_nodes([node], env, ctx)
  outs = Enum.map(node.output, &Map.get(result, &1))

  bad =
    for {name, t} <- Enum.zip(node.output, outs),
        is_struct(t),
        Map.has_key?(ref, name),
        {true, shape, sample, diff} <- [diverged?.(name, t)],
        do: {name, shape, sample, diff}

  # OVERRIDE=name,name replaces our values with ORT's full outputs (dumped by trace_ort.py DUMP=...),
  # to look past a divergence that is only precision.
  overrides = System.get_env("OVERRIDE", "") |> String.split(",", trim: true)

  outs =
    Enum.zip_with(node.output, outs, fn name, t ->
      if name in overrides do
        file = Path.join(case_dir, "full_" <> String.replace(name, "/", "_") <> ".bin")

        file
        |> File.read!()
        |> Nx.from_binary(:f32, backend: backend)
        |> Nx.reshape(Nx.shape(t))
        |> Nx.as_type(Nx.type(t))
      else
        t
      end
    end)

  bad = Enum.reject(bad, fn {name, _, _, _} -> name in overrides end)

  case bad do
    [] ->
      {:cont, Map.merge(env, Map.new(Enum.zip(node.output, outs)))}

    [{name, shape, sample, diff} | _] ->
      IO.puts("FIRST DIVERGENCE at #{node.op_type} #{node.name} -> #{name}")
      IO.puts("  ours: shape=#{inspect(shape)} sample=#{inspect(Enum.take(sample, 8))}")

      IO.puts(
        "  ort:  shape=#{inspect(ref[name]["shape"])} sample=#{inspect(Enum.take(ref[name]["sample"], 8))} max sample diff=#{diff}"
      )

      IO.puts(
        "  attrs: #{inspect(Enum.map(node.attribute, &{&1.name, &1.i, &1.ints, &1.f, &1.s}))}"
      )

      IO.puts(
        "  inputs: #{inspect(Enum.map(ins, &(&1 && ((is_struct(&1) && Nx.shape(&1)) || &1))))}"
      )

      {:halt, :diverged}
  end
end)
|> then(&if(&1 != :diverged, do: IO.puts("No divergence in #{length(model.nodes)} nodes")))
