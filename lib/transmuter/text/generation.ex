defmodule Transmuter.Text.Generation do
  @moduledoc """
  The text generation task. See `Transmuter.Text.generation/2`. The generation loop and the KV cache are in `Transmuter.Generation`.
  """

  @compile {:no_warn_undefined, [Tokenizers.Tokenizer, Tokenizers.Encoding]}

  alias Transmuter.{ChatTemplate, Generation, Serving, Tokenizer}

  @doc false
  def generation(source, opts) do
    opts =
      Keyword.validate!(opts, [
        :variant,
        :cache_dir,
        :model,
        :tokenizer,
        :backend,
        :compute_type,
        :max_new_tokens,
        :strategy,
        :temperature,
        :top_k,
        :top_p,
        :repetition_penalty,
        :seed,
        :template_options,
        defn_options: []
      ])

    {repo, model, tokenizer} = Serving.load!(source, opts, :q4, ["model", "decoder_model_merged"])
    ctx = Generation.context(repo, model, opts)

    Nx.Serving.new(
      fn _key, defn_options -> run_fun(ctx, defn_options) end,
      opts[:defn_options]
    )
    |> Nx.Serving.batch_size(1)
    |> Nx.Serving.client_preprocessing(fn input ->
      {inputs, multi?} = validate!(input)

      prompts =
        for input <- inputs do
          {text, special?} = prompt(repo, input, opts)
          seed = input[:seed] || opts[:seed] || :rand.uniform(2 ** 32)
          ids = Nx.tensor(token_ids(tokenizer, text, special?, ctx.bos), type: :s64)
          %{"ids" => ids, "seed" => Nx.tensor(seed, type: :s64)}
        end

      batches = Enum.map(prompts, &Nx.Batch.stack([&1]))
      {batches, {multi?, Enum.map(prompts, &Nx.size(&1["ids"]))}}
    end)
    |> Nx.Serving.client_postprocessing(fn {outputs, _metadata}, {multi?, input_lengths} ->
      results =
        for {length, i} <- Enum.with_index(input_lengths) do
          count = Nx.to_number(outputs.length[i])
          ids = outputs.ids[i] |> Nx.to_flat_list() |> Enum.take(count)
          summary = %{input: length, output: count, padding: 0}
          %{results: [%{text: Tokenizer.decode(tokenizer, ids), token_summary: summary}]}
        end

      if multi?, do: results, else: hd(results)
    end)
  end

  # The function of a batch.
  defp run_fun(ctx, defn_options) do
    ctx = Generation.compile(ctx, defn_options)

    fn batch ->
      %{"ids" => ids, "seed" => seed} =
        Nx.Defn.jit_apply(& &1, [batch], compiler: Nx.Defn.Evaluator)

      ids = Nx.to_flat_list(ids)
      generated = Generation.generate(ctx, ids, Nx.to_number(seed[0]))
      padded = generated ++ List.duplicate(0, ctx.max_new_tokens - length(generated))

      %{
        ids: Nx.tensor([padded], type: :s64, backend: Nx.BinaryBackend),
        length: Nx.tensor([length(generated)], backend: Nx.BinaryBackend)
      }
    end
  end

  # A chat goes through the chat template, which adds the special tokens itself.
  defp prompt(_repo, %{text: text}, _opts), do: {text, true}

  # As the text generation pipeline of transformers, a chat that ends with an assistant
  # message continues that message.
  defp prompt(repo, input = %{messages: messages}, opts) do
    final = List.last(messages)

    template_opts = [
      tools: input[:tools],
      template_options: opts[:template_options],
      continue_final_message: (final[:role] || final["role"]) == "assistant"
    ]

    {ChatTemplate.render(repo, messages, template_opts), false}
  end

  # An empty prompt starts from the bos token, as in transformers.
  defp token_ids(tokenizer, text, special?, bos) do
    {:ok, encoding} =
      Tokenizers.Tokenizer.encode(tokenizer.native, text, add_special_tokens: special?)

    case Tokenizers.Encoding.get_ids(encoding) do
      [] ->
        [bos || raise(ArgumentError, "the prompt is empty, and the model has no bos_token_id")]

      ids ->
        ids
    end
  end

  defp validate!(inputs) when is_list(inputs), do: {Enum.map(inputs, &validate_one!/1), true}
  defp validate!(input), do: {[validate_one!(input)], false}

  defp validate_one!(text) when is_binary(text), do: %{text: text}
  defp validate_one!(input = %{text: text}) when is_binary(text), do: input
  defp validate_one!(input = %{messages: [_ | _]}), do: input

  defp validate_one!(input) do
    raise ArgumentError,
          "expected a string, %{text: string} or %{messages: [%{role: role, content: content}]}, got: #{inspect(input)}"
  end
end
