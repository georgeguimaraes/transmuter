# Runs one case eagerly and prints the first node whose output differs from ORT.
# Run debug_ort.py on the case first to write ort_trace.json.
# Usage: mix run bench/coverage/debug_case.exs CASE_DIR

[case_dir] = System.argv()
meta = case_dir |> Path.join("meta.json") |> File.read!() |> JSON.decode!()
ref = case_dir |> Path.join("ort_trace.json") |> File.read!() |> JSON.decode!()
backend = {EMLX.Backend, device: String.to_atom(System.get_env("DEVICE", "gpu"))}

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

model = Transmuter.load!(meta["onnx_path"], compute_type: :f32)
params = Map.new(model.params, fn {k, v} -> {k, Nx.backend_transfer(v, backend)} end)

inputs =
  Map.new(meta["inputs"], fn spec ->
    {spec["name"], read.(spec, if(static?.(spec), do: Nx.BinaryBackend, else: backend))}
  end)

diverged? = fn name, t ->
  r = ref[name]
  t = Nx.as_type(t, :f64)
  t = t |> then(&Nx.select(Nx.is_nan(&1), 0, &1)) |> Nx.clip(-3.0e38, 3.0e38)
  flat = Nx.reshape(t, {:auto})
  idx = for k <- 0..1023, do: rem(k * 2_654_435_761, Nx.size(t))
  sample = flat |> Nx.take(Nx.tensor(idx)) |> Nx.to_flat_list()
  diff = Enum.zip_with(sample, r["sample"], fn a, b -> if a == b, do: 0, else: abs(a - b) end) |> Enum.max()
  shape = Tuple.to_list(Nx.shape(t))
  {shape != r["shape"] or diff > 1.0e-3 * max(r["absmax"], 1.0), shape, sample, diff}
end

env = model.static |> Map.merge(params) |> Map.merge(inputs)

Enum.reduce_while(model.nodes, env, fn node, env ->
  ins = Enum.map(node.input, fn "" -> nil; name -> Map.fetch!(env, name) end)

  outs =
    if Enum.all?(ins, &(&1 == nil or Transmuter.Ops.static?(&1))),
      do: Nx.with_default_backend(Nx.BinaryBackend, fn -> Transmuter.Ops.apply(node, ins, model.opset) end),
      else: Transmuter.Ops.apply(node, ins, model.opset)

  bad =
    for {name, t} <- Enum.zip(node.output, outs), is_struct(t), Map.has_key?(ref, name),
        {true, shape, sample, diff} <- [diverged?.(name, t)],
        do: {name, shape, sample, diff}

  # OVERRIDE=name,name replaces our values with ORT's full outputs (dumped by debug_ort.py DUMP=...),
  # to look past a divergence that is only precision.
  overrides = System.get_env("OVERRIDE", "") |> String.split(",", trim: true)

  outs =
    Enum.zip_with(node.output, outs, fn name, t ->
      if name in overrides do
        file = Path.join(case_dir, "full_" <> String.replace(name, "/", "_") <> ".bin")
        file |> File.read!() |> Nx.from_binary(:f32, backend: backend) |> Nx.reshape(Nx.shape(t))
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
      IO.puts("  ort:  shape=#{inspect(ref[name]["shape"])} sample=#{inspect(Enum.take(ref[name]["sample"], 8))} max sample diff=#{diff}")
      IO.puts("  attrs: #{inspect(Enum.map(node.attribute, &{&1.name, &1.i, &1.ints, &1.f, &1.s}))}")
      IO.puts("  inputs: #{inspect(Enum.map(ins, &(&1 && (is_struct(&1) && Nx.shape(&1) || &1))))}")
      {:halt, env}
  end
end)
