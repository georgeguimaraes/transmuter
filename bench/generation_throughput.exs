# The batch throughput table of guides/pipelines.md: the total tokens per second of 16 concurrent
# requests to a started serving with batch_timeout: 10, greedy selection and up to 128 new tokens
# each, on EMLX. One warm round, then `runs` measured rounds (default 5). Each round prints a
# fingerprint of the 16 texts, so the rounds of one case can be compared.
# Usage: mix run bench/generation_throughput.exs smollm2|qwen3|gpt2 <batch_size> [runs]
Logger.configure(level: :warning)
[name, batch | rest] = System.argv()
batch = String.to_integer(batch)
runs = String.to_integer(List.first(rest) || "5")

opts = [
  backend: {EMLX.Backend, device: :gpu},
  defn_options: [compiler: EMLX],
  strategy: :greedy,
  max_new_tokens: 128,
  compile: [batch_size: batch]
]

questions = [
  "What is the capital of France?",
  "Explain how a rainbow forms.",
  "Write a short poem about the sea.",
  "Why is the sky blue?",
  "Give me three tips for learning a new language.",
  "What does a compiler do?",
  "Describe the water cycle.",
  "How do bees make honey?",
  "Summarize the plot of a heist movie.",
  "What is the difference between weather and climate?",
  "Write a haiku about autumn.",
  "How does a bicycle stay upright?",
  "List five fruits that are rich in vitamin C.",
  "What causes the seasons on Earth?",
  "Explain recursion to a child.",
  "Tell me a fun fact about octopuses."
]

{repo, variant, inputs} =
  case name do
    "smollm2" ->
      {"HuggingFaceTB/SmolLM2-135M-Instruct", :q4,
       Enum.map(questions, &%{messages: [%{role: "user", content: &1}]})}

    "qwen3" ->
      {"onnx-community/Qwen3-0.6B-ONNX", :q4,
       Enum.map(questions, &%{messages: [%{role: "user", content: &1}]})}

    # 16 prompts of 7 lengths: the sentences of a story, cut after 1 to 7 sentences.
    "gpt2" ->
      sentences = [
        "The old lighthouse stood at the end of the pier.",
        "Every night its lamp turned slowly over the dark water.",
        "Fishermen trusted it more than their maps.",
        "One winter the keeper did not come back from the town.",
        "The lamp went dark for the first time in a hundred years.",
        "A girl from the village climbed the stairs with a lantern.",
        "She found the keeper's notebook open on the table."
      ]

      texts = for i <- 0..15, do: sentences |> Enum.take(rem(i, 7) + 1) |> Enum.join(" ")
      {"Xenova/gpt2", nil, texts}
  end

serving =
  Transmuter.Text.generation({:hf, repo}, if(variant, do: [variant: variant], else: []) ++ opts)

{:ok, _} = Nx.Serving.start_link(serving: serving, name: Bench, batch_timeout: 10)

round = fn ->
  {us, results} =
    :timer.tc(fn ->
      inputs
      |> Task.async_stream(&Nx.Serving.batched_run(Bench, &1),
        max_concurrency: 16,
        timeout: :infinity
      )
      |> Enum.map(fn {:ok, %{results: [result]}} -> result end)
    end)

  tokens = results |> Enum.map(& &1.token_summary.output) |> Enum.sum()

  digest =
    results
    |> Enum.map(& &1.text)
    |> :erlang.term_to_binary()
    |> :erlang.md5()
    |> Base.encode16(case: :lower)
    |> binary_part(0, 8)

  {tokens / (us / 1_000_000), tokens, digest}
end

round.()

rates =
  for i <- 1..runs do
    {rate, tokens, digest} = round.()

    IO.puts(
      "ROUND #{name} batch #{batch} #{i}: #{Float.round(rate, 1)} tokens/s, #{tokens} tokens, text #{digest}"
    )

    rate
  end

IO.puts(
  "THROUGHPUT #{name} batch #{batch}: median #{rates |> Enum.sort() |> Enum.at(div(runs, 2)) |> Float.round(1)} tokens/s of #{runs} rounds"
)
