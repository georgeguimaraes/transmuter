# Greedy generation with SmolLM2-135M-Instruct on EMLX, with a static KV cache.
# Usage: mix run bench/smollm_generate.exs

repo = "HuggingFaceTB/SmolLM2-135M-Instruct"
backend = {EMLX.Backend, device: :gpu}
max_len = 128
new_tokens = 30

{:ok, tokenizer} = Tokenizers.Tokenizer.from_pretrained(repo)
prompt = "<|im_start|>system\nYou are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>\n<|im_start|>user\nExplain in two sentences why the sky is blue.<|im_end|>\n<|im_start|>assistant\n"
{:ok, encoding} = Tokenizers.Tokenizer.encode(tokenizer, prompt, add_special_tokens: false)
prompt_ids = Tokenizers.Encoding.get_ids(encoding)

model = Transmuter.load!({:hf, repo, "onnx/model.onnx"})
params = Map.new(model.params, fn {k, v} -> {k, Nx.backend_transfer(v, backend)} end)

# The KV buffer is as long as the attention mask, so every decode step has the same shapes.
empty_cache =
  for name <- model.inputs, String.starts_with?(name, "past_key_values"), into: %{} do
    {name, Nx.broadcast(Nx.tensor(0.0, backend: backend), {1, 3, max_len, 64})}
  end

step_inputs = fn ids, pos, cache ->
  n = pos + length(ids)

  Map.merge(cache, %{
    "input_ids" => Nx.tensor([ids], type: :s64, backend: backend),
    "attention_mask" => Nx.tensor([Enum.map(0..(max_len - 1), &if(&1 < n, do: 1, else: 0))], type: :s64, backend: backend),
    "position_ids" => Nx.tensor([Enum.to_list(pos..(n - 1))], type: :s64, backend: backend)
  })
end

fun = &Transmuter.run(model, &1, &2)
{t, prefill} = :timer.tc(fn -> Nx.Defn.compile(fun, [params, step_inputs.(prompt_ids, 0, empty_cache)], compiler: EMLX) end)
IO.puts("compile prefill: #{div(t, 1000)}ms")
{t, decode} = :timer.tc(fn -> Nx.Defn.compile(fun, [params, step_inputs.([0], length(prompt_ids), empty_cache)], compiler: EMLX) end)
IO.puts("compile decode: #{div(t, 1000)}ms")

next_cache = fn out -> for {"present." <> rest, t} <- out, into: %{}, do: {"past_key_values." <> rest, t} end
next_token = fn out -> out["logits"][[0, -1]] |> Nx.argmax() |> Nx.to_number() end

generate = fn ->
  out = prefill.(params, step_inputs.(prompt_ids, 0, empty_cache))

  Enum.reduce(1..(new_tokens - 1)//1, {[next_token.(out)], next_cache.(out)}, fn i, {tokens, cache} ->
    out = decode.(params, step_inputs.([hd(tokens)], length(prompt_ids) + i - 1, cache))
    {[next_token.(out) | tokens], next_cache.(out)}
  end)
  |> elem(0)
  |> Enum.reverse()
end

generate.()
{t, tokens} = :timer.tc(generate)
{:ok, text} = Tokenizers.Tokenizer.decode(tokenizer, tokens)
IO.puts(text)
IO.puts("#{new_tokens} tokens in #{div(t, 1000)}ms, #{Float.round(new_tokens / (t / 1_000_000), 1)} tok/s")
