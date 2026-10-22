defmodule Transmuter.Text.Generation do
  @moduledoc """
  The text generation task. See `Transmuter.Text.generation/2`. The generation loop and the KV cache are in `Transmuter.Generation`.
  """

  @compile {:no_warn_undefined, [Tokenizers.Tokenizer, Tokenizers.Encoding]}

  alias Transmuter.{ChatTemplate, Generation, Hub, Serving, Tokenizer}

  @doc false
  def generation(source, opts) do
    opts =
      Keyword.validate!(
        opts,
        [
          :variant,
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
          :num_beams,
          :length_penalty,
          :early_stopping,
          :no_repeat_ngram_size,
          :min_length,
          :min_new_tokens,
          :forced_bos_token_id,
          :forced_eos_token_id,
          :seed,
          :template_options,
          stream: false,
          stream_done: false,
          compile: [],
          defn_options: []
        ] ++ Hub.options()
      )

    Keyword.validate!(opts[:compile], [:batch_size])

    load_opts = [outputs: &Generation.outputs/1] ++ opts
    components = ["model", "decoder_model_merged"]
    {repo, model, tokenizer} = Serving.load!(source, load_opts, :q4, components)

    ctx = Generation.context(repo, model, opts)
    bos = ctx.bos

    preprocess = fn input ->
      {text, special?} = prompt(repo, input, opts)
      seed = input[:seed] || opts[:seed] || :rand.uniform(2 ** 32)
      ids = token_ids(tokenizer, text, special?, bos)
      summary = %{input: length(ids), padding: 0}
      [%{prompt: ids, seed: seed, tensors: %{}, summary: summary}]
    end

    postprocess = fn [ids], [run] ->
      summary = Map.put(run.summary, :output, length(ids))
      %{results: [%{text: Tokenizer.decode(tokenizer, ids), token_summary: summary}]}
    end

    task = %{
      validate: &validate/1,
      preprocess: preprocess,
      init: fn _defn_options -> fn _tensors, prompts, generate -> generate.(%{}, prompts) end end,
      postprocess: postprocess
    }

    Serving.generation(ctx, tokenizer, task, opts)
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

  defp validate(text) when is_binary(text), do: {:ok, %{text: text}}
  defp validate(input = %{text: text}) when is_binary(text), do: {:ok, input}
  defp validate(input = %{messages: [_ | _]}), do: {:ok, input}

  defp validate(input) do
    {:error,
     "expected a string, %{text: string} or %{messages: [%{role: role, content: content}]}, got: #{inspect(input)}"}
  end
end
