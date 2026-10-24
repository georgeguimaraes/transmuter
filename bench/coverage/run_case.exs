# Runs one prepared case through Transmuter and writes result.json next to it.
# Usage: mix run bench/coverage/run_case.exs CASE_DIR

[case_dir] = System.argv()
meta = case_dir |> Path.join("meta.json") |> File.read!() |> JSON.decode!()
# The backend of the tests: TRANSMUTER_TEST_BACKEND and TRANSMUTER_TEST_DEVICE.
Code.require_file("../../test/support/test_backend.exs", __DIR__)
backend = Transmuter.TestBackend.backend()
compiler_opts = Transmuter.TestBackend.compiler_opts()

result_path = Path.join(case_dir, "result.json")

read = fn %{"file" => file, "dtype" => type, "shape" => shape}, backend ->
  case_dir
  |> Path.join(file)
  |> File.read!()
  |> Nx.from_binary(String.to_atom(type), backend: backend)
  |> Nx.reshape(List.to_tuple(shape))
end

write = fn result -> File.write!(result_path, JSON.encode!(result)) end
timed = fn fun -> :timer.tc(fun) |> then(fn {t, v} -> {div(t, 1000), v} end) end

# Flags that pick a branch or a slice must be static. When the graph has NonZero,
# the integer inputs are static too, because NonZero output shapes depend on them.
static? = fn spec ->
  spec["name"] in ["use_cache_branch", "num_logits_to_keep", "sr" | meta["static_inputs"] || []] or
    (Map.has_key?(meta["ops"], "NonZero") and spec["dtype"] in ["s64", "s32"])
end

try do
  {load_ms, model} = timed.(fn -> Transmuter.load!(Path.expand(meta["onnx_path"], case_dir), compute_type: :f32) end)
  # With EMLX, EMLX_QUANTIZED=1 keeps MatMulNBits weights packed as MLX quantized tensors.
  model =
    if System.get_env("EMLX_QUANTIZED") == "1" and match?({EMLX.Backend, _}, backend),
      do: Transmuter.EMLX.move(model, elem(backend, 1)),
      else: %{model | params: Map.new(model.params, fn {k, v} -> {k, Nx.backend_transfer(v, backend)} end)}
  params = model.params
  {static, inputs} = Enum.split_with(meta["inputs"], static?)
  inputs = Map.new(inputs, &{&1["name"], read.(&1, backend)})
  static = Map.new(static, &{&1["name"], read.(&1, Nx.BinaryBackend)})
  # EAGER=1 runs without compiling. Otherwise Transmuter.compile/3 compiles the model and
  # breaks the graph at values that are only known when it runs.
  fun =
    if System.get_env("EAGER") == "1",
      do: fn i -> Transmuter.run(model, params, i, static) end,
      else:
        Transmuter.compile(%{model | params: params}, inputs, [static_inputs: static] ++ compiler_opts)
  {run_ms, out} = timed.(fn -> fun.(inputs) end)

  diffs =
    for spec <- meta["outputs"], not (0 in spec["shape"] and not Map.has_key?(out, spec["name"])) do
      ref = read.(spec, Nx.BinaryBackend) |> Nx.as_type(:f64)
      got = out |> Map.fetch!(spec["name"]) |> Nx.backend_copy(Nx.BinaryBackend)

      if Nx.shape(got) == Nx.shape(ref) do
        got = Nx.as_type(got, :f64)
        ref_nan = Nx.is_nan(ref)
        same = Nx.logical_or(Nx.logical_and(ref_nan, Nx.is_nan(got)), Nx.equal(got, ref))
        max_abs = Nx.select(same, 0, Nx.abs(Nx.subtract(got, ref))) |> Nx.reduce_max() |> Nx.to_number()
        finite = Nx.logical_not(Nx.logical_or(ref_nan, Nx.is_infinity(ref)))
        max_ref = Nx.select(finite, Nx.abs(ref), 0) |> Nx.reduce_max() |> Nx.to_number()
        nan = max_abs == :nan or Nx.to_number(Nx.any(Nx.logical_xor(ref_nan, Nx.is_nan(got)))) == 1
        max_abs = if max_abs == :nan, do: -1, else: max_abs
        ok = not nan and max_abs <= 1.0e-4 + 1.0e-3 * max_ref
        %{name: spec["name"], ok: ok, max_abs: max_abs, max_ref: max_ref, nan: nan}
      else
        %{name: spec["name"], ok: false, shape: inspect(Nx.shape(got)), expected: inspect(Nx.shape(ref))}
      end
    end

  status = if Enum.all?(diffs, & &1.ok), do: "pass", else: "mismatch"
  write.(%{status: status, load_ms: load_ms, run_ms: run_ms, nodes: length(model.nodes), diffs: diffs})
  IO.puts("#{status} #{meta["repo"]} #{meta["file"]}")
rescue
  e ->
    message = e |> Exception.message() |> String.slice(0, 600)
    write.(%{status: "error", error: message})
    IO.puts("error #{meta["repo"]} #{meta["file"]}: #{message}")
end
