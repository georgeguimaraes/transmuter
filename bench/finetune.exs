# Fine-tunes an ONNX cross-encoder with transmuter and Polaris, and compares with PyTorch.
# Usage: mix run bench/finetune.exs MODEL_ONNX DATA_DIR
# DATA_DIR holds the inputs, labels and torch_ref.json written by the PyTorch reference.

[onnx_path, dir] = System.argv()
backend = {EMLX.Backend, device: :gpu}
ref = dir |> Path.join("torch_ref.json") |> File.read!() |> JSON.decode!()
read = fn name, type, shape -> dir |> Path.join(name) |> File.read!() |> Nx.from_binary(type, backend: backend) |> Nx.reshape(shape) end

inputs = Map.new(["input_ids", "attention_mask", "token_type_ids"], &{&1, read.("#{&1}.bin", :s64, {4, 24})})
labels = read.("labels.bin", :f32, {4})

model = Transmuter.load!(onnx_path, trainable: true)
params = Map.new(model.params, fn {k, v} -> {k, Nx.backend_transfer(v, backend)} end)

# Binary cross-entropy with logits, averaged over the batch, as torch.nn.BCEWithLogitsLoss.
loss_fn = fn params, inputs, labels ->
  logits = Transmuter.run(model, params, inputs)["logits"] |> Nx.squeeze(axes: [1])
  logits |> Nx.max(0) |> Nx.subtract(Nx.multiply(logits, labels)) |> Nx.add(Nx.log1p(Nx.exp(Nx.negate(Nx.abs(logits))))) |> Nx.mean()
end

grad_fn = Nx.Defn.jit(fn params, inputs, labels -> Nx.Defn.value_and_grad(params, &loss_fn.(&1, inputs, labels)) end, compiler: EMLX)
{loss, grads} = grad_fn.(params, inputs, labels)
grads = Map.new(grads, fn {k, v} -> {k, Nx.backend_copy(v, Nx.BinaryBackend)} end)
norm = fn t -> t |> Nx.pow(2) |> Nx.sum() |> Nx.sqrt() |> Nx.to_number() end

IO.puts("loss: transmuter #{Float.round(Nx.to_number(loss), 6)}, PyTorch #{Float.round(ref["loss"], 6)}")

large = for {k, g} <- grads, Nx.size(g) > 16, do: g |> Nx.pow(2) |> Nx.sum() |> Nx.to_number()
IO.puts("grad norm over params > 16 elements: transmuter #{Float.round(:math.sqrt(Enum.sum(large)), 6)}, PyTorch #{Float.round(ref["grad_norm_over_params_larger_than_16"], 6)}")

for {name, r} <- ref["grads"] do
  g = grads[name]
  first = g |> Nx.flatten() |> Nx.slice([0], [4]) |> Nx.to_flat_list() |> Enum.map(&Float.round(&1, 6))
  IO.puts("#{name}: norm #{Float.round(norm.(g), 6)} vs #{Float.round(r["norm"], 6)}, first #{inspect(first)} vs #{inspect(Enum.map(r["first"], &Float.round(&1, 6)))}")
end

# 30 AdamW steps, as the PyTorch reference.
{init_fn, update_fn} = Polaris.Optimizers.adamw(learning_rate: 1.0e-4, decay: 0.0)

step =
  Nx.Defn.jit(
    fn params, state, inputs, labels ->
      {loss, grads} = Nx.Defn.value_and_grad(params, &loss_fn.(&1, inputs, labels))
      {updates, state} = update_fn.(grads, state, params)
      {Polaris.Updates.apply_updates(params, updates), state, loss}
    end,
    compiler: EMLX
  )

{t, {_params, _state, curve}} =
  :timer.tc(fn ->
    Enum.reduce(1..30, {params, init_fn.(params), []}, fn _, {params, state, curve} ->
      {params, state, loss} = step.(params, state, inputs, labels)
      {params, state, [Nx.to_number(loss) | curve]}
    end)
  end)

curve = Enum.reverse(curve)
fmt = fn list -> Enum.map(list, &Float.round(&1, 4)) end
IO.puts("transmuter curve: #{inspect(fmt.(Enum.take(curve, 5)))} ... #{inspect(fmt.(Enum.take(curve, -3)))} (#{div(t, 30_000)} ms per step)")
IO.puts("PyTorch curve:    #{inspect(fmt.(Enum.take(ref["curve"], 5)))} ... #{inspect(fmt.(Enum.take(ref["curve"], -3)))}")
worst = Enum.zip_with(curve, ref["curve"], &abs(&1 - &2)) |> Enum.max()
IO.puts("largest per-step loss difference: #{Float.round(worst, 5)}")
