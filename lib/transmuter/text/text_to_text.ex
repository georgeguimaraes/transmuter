defmodule Transmuter.Text.TextToText do
  @moduledoc """
  The text-to-text task of encoder-decoder models, such as T5 and Marian. See `Transmuter.Text.text_to_text/2`.

  The tokenizer pads the text to a sequence length of the `:compile` option, so the encoder runs a small number of compiled shapes. The decoder reads the attention mask of the text (`encoder_attention_mask`), so the padding has no effect on the result. Generation starts from `decoder_start_token_id`.
  """

  alias Transmuter.{Generation, Serving, Tokenizer}

  @doc false
  def text_to_text(source, opts) do
    opts =
      Keyword.validate!(opts, [
        :compile,
        :variant,
        :cache_dir,
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
        stream: false,
        stream_done: false,
        defn_options: []
      ])

    {repo, encoder, decoder} = Serving.load_encoder_decoder!(source, opts)
    given = ~w(encoder_hidden_states encoder_attention_mask)
    opts = Keyword.put(opts, :max_new_tokens, Generation.max_new_tokens(repo, opts, 1))
    ctx = Generation.context(repo, decoder, opts, given)

    compile =
      Keyword.validate!(opts[:compile] || [],
        batch_size: 1,
        sequence_length: Serving.sequence_lengths(repo, nil)
      )

    lengths = List.wrap(compile[:sequence_length])
    tokenizer = Tokenizer.configure(opts[:tokenizer] || Tokenizer.load!(repo), length: lengths)

    start =
      repo.generation_config["decoder_start_token_id"] || repo.config["decoder_start_token_id"]

    init = fn defn_options ->
      shape = {compile[:batch_size], hd(lengths)}
      template = Nx.broadcast(Nx.tensor(0, type: :s64, backend: Nx.BinaryBackend), shape)
      opts = [cache_size: length(lengths)] ++ defn_options
      encode = Transmuter.compile(encoder, Map.new(encoder.inputs, &{&1, template}), opts)

      fn inputs, prompts, generate ->
        %{"last_hidden_state" => hidden} = encode.(Map.take(inputs, encoder.inputs))

        given = %{
          "encoder_hidden_states" => hidden,
          "encoder_attention_mask" => inputs["attention_mask"]
        }

        generate.(given, prompts)
      end
    end

    preprocess = fn input ->
      {tensors, _encodings} = Tokenizer.encode(tokenizer, [input.text])
      tensors = Map.new(tensors, fn {name, tensor} -> {name, tensor[0]} end)
      seed = input[:seed] || opts[:seed] || :rand.uniform(2 ** 32)
      length = Nx.size(tensors["input_ids"])
      count = Nx.to_number(Nx.sum(tensors["attention_mask"]))
      summary = %{input: count, padding: length - count}
      [%{prompt: [start], seed: seed, tensors: tensors, key: length, summary: summary}]
    end

    postprocess = fn [ids], [run] ->
      summary = Map.put(run.summary, :output, length(ids))
      %{results: [%{text: Tokenizer.decode(tokenizer, ids), token_summary: summary}]}
    end

    task = %{
      validate: &validate/1,
      preprocess: preprocess,
      init: init,
      postprocess: postprocess,
      keys: lengths
    }

    Serving.generation(ctx, tokenizer, task, Keyword.put(opts, :compile, compile))
  end

  defp validate(text) when is_binary(text), do: {:ok, %{text: text}}
  defp validate(input = %{text: text}) when is_binary(text), do: {:ok, input}

  defp validate(input),
    do: {:error, "expected a string or %{text: string}, got: #{inspect(input)}"}
end
