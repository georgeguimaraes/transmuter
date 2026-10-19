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
        defn_options: []
      ])

    {repo, encoder, decoder} = Serving.load_encoder_decoder!(source, opts)
    given = ~w(encoder_hidden_states encoder_attention_mask)
    ctx = Generation.context(repo, decoder, opts, given)

    compile =
      Keyword.validate!(opts[:compile] || [],
        sequence_length: Serving.sequence_lengths(repo, nil)
      )

    lengths = List.wrap(compile[:sequence_length])
    tokenizer = Tokenizer.configure(opts[:tokenizer] || Tokenizer.load!(repo), length: lengths)

    start =
      repo.generation_config["decoder_start_token_id"] || repo.config["decoder_start_token_id"]

    init = fn defn_options ->
      template =
        Nx.broadcast(Nx.tensor(0, type: :s64, backend: Nx.BinaryBackend), {1, hd(lengths)})

      opts = [cache_size: length(lengths)] ++ defn_options
      encode = Transmuter.compile(encoder, Map.new(encoder.inputs, &{&1, template}), opts)

      fn inputs ->
        %{"last_hidden_state" => hidden} = encode.(Map.take(inputs, encoder.inputs))
        %{"encoder_hidden_states" => hidden, "encoder_attention_mask" => inputs["attention_mask"]}
      end
    end

    preprocess = fn input ->
      {tensors, _encodings} = Tokenizer.encode(tokenizer, [input.text])
      seed = input[:seed] || opts[:seed] || :rand.uniform(2 ** 32)

      tensors
      |> Map.new(fn {name, tensor} -> {name, tensor[0]} end)
      |> Map.merge(%{
        "prompt" => Nx.tensor([start], type: :s64),
        "seed" => Nx.tensor(seed, type: :s64)
      })
    end

    postprocess = fn ids, %{"attention_mask" => mask} ->
      count = Nx.to_number(Nx.sum(mask))
      summary = %{input: count, output: length(ids), padding: Nx.size(mask) - count}
      %{results: [%{text: Tokenizer.decode(tokenizer, ids), token_summary: summary}]}
    end

    Serving.generation(ctx, init, &validate/1, preprocess, postprocess, opts[:defn_options])
  end

  defp validate(text) when is_binary(text), do: {:ok, %{text: text}}
  defp validate(input = %{text: text}) when is_binary(text), do: {:ok, input}

  defp validate(input),
    do: {:error, "expected a string or %{text: string}, got: #{inspect(input)}"}
end
