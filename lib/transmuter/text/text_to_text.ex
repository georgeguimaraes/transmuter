defmodule Transmuter.Text.TextToText do
  @moduledoc """
  The text-to-text task of encoder-decoder models, such as T5, Marian, NLLB and M2M100. See `Transmuter.Text.text_to_text/2`.

  The tokenizer pads the text to a sequence length of the `:compile` option, so the encoder runs a small number of compiled shapes. The decoder reads the attention mask of the text (`encoder_attention_mask`), so the padding has no effect on the result. Generation starts from `decoder_start_token_id`.
  """

  @compile {:no_warn_undefined, [Tokenizers.PostProcessor, Tokenizers.Tokenizer]}

  alias Tokenizers.PostProcessor
  alias Transmuter.{Generation, Hub, Repo, Serving, Tokenizer}

  @doc false
  def text_to_text(source, opts) do
    opts =
      Keyword.validate!(
        opts,
        [
          :compile,
          :variant,
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
          :source_language_token,
          :target_language_token,
          stream: false,
          stream_done: false,
          defn_options: []
        ] ++ Hub.options()
      )

    {repo, encoder, decoder} = Serving.load_encoder_decoder!(source, opts)

    compile =
      Keyword.validate!(opts[:compile] || [],
        batch_size: 1,
        sequence_length: Serving.sequence_lengths(repo, nil)
      )

    lengths = List.wrap(compile[:sequence_length])
    tokenizer = Tokenizer.configure(opts[:tokenizer] || Tokenizer.load!(repo), length: lengths)
    {tokenizer, opts} = languages(repo, tokenizer, opts)

    given = ~w(encoder_hidden_states encoder_attention_mask)
    opts = Keyword.put(opts, :max_new_tokens, Generation.max_new_tokens(repo, opts, 1))
    ctx = Generation.context(repo, decoder, opts, given)

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

  @doc false
  # The tokenizer with the source language token, and the options with the target language
  # token in :forced_bos_token_id, as src_lang and tgt_lang of the translation pipeline of
  # transformers for NLLB and M2M100.
  def languages(repo, tokenizer, opts) do
    source = opts[:source_language_token]
    target = opts[:target_language_token]

    if source == nil and target == nil do
      {tokenizer, opts}
    else
      json =
        repo.source |> Hub.fetch!("tokenizer.json", repo.opts) |> File.read!() |> JSON.decode!()

      {single, pair, language} = template!(json, repo)
      special = for %{"special" => true, "content" => token} <- json["added_tokens"], do: token
      id! = &language_id!(tokenizer, special, language, &1)

      tokenizer =
        if source,
          do: source_language(tokenizer, single, pair, language, source, id!),
          else: tokenizer

      opts = if target, do: Keyword.put(opts, :forced_bos_token_id, id!.(target)), else: opts
      {tokenizer, opts}
    end
  end

  # The template of the post-processor in tokenizer.json, and its language token: the
  # special token that is not eos, such as "eng_Latn" in "eng_Latn $A </s>".
  defp template!(json, repo) do
    {single, pair} =
      case json["post_processor"] do
        %{"type" => "TemplateProcessing", "single" => single, "pair" => pair} -> {single, pair}
        _other -> {[], []}
      end

    eos = Repo.special_token(repo, "eos")

    case for(%{"SpecialToken" => %{"id" => id}} <- single, id != eos, uniq: true, do: id) do
      [language] ->
        {single, pair, language}

      other ->
        raise ArgumentError,
              "expected one language token in the template of the post-processor of tokenizer.json, got: #{inspect(other)}"
    end
  end

  # The source language token takes the place of the language token of the template.
  defp source_language(tokenizer, single, pair, language, token, id!) do
    replace = &if(&1 == language, do: token, else: &1)

    template = fn pieces ->
      Enum.map_join(pieces, " ", fn
        %{"SpecialToken" => %{"id" => id, "type_id" => type}} -> "#{replace.(id)}:#{type}"
        %{"Sequence" => %{"id" => id, "type_id" => type}} -> "$#{id}:#{type}"
      end)
    end

    special_tokens =
      for %{"SpecialToken" => %{"id" => id}} <- single ++ pair,
          uniq: true,
          do: {replace.(id), id!.(replace.(id))}

    processor =
      PostProcessor.template(
        single: template.(single),
        pair: template.(pair),
        special_tokens: special_tokens
      )

    %{tokenizer | native: Tokenizers.Tokenizer.set_post_processor(tokenizer.native, processor)}
  end

  # A language token is a special token of tokenizer.json. A bare code, such as "fr" for
  # M2M100, is an ordinary piece of the vocabulary, so it raises.
  defp language_id!(tokenizer, special, language, token) do
    unless token in special do
      raise ArgumentError,
            "expected a language token, a special token of the tokenizer such as #{inspect(language)}, got: #{inspect(token)}"
    end

    Tokenizer.token_to_id(tokenizer, token)
  end

  defp validate(text) when is_binary(text), do: {:ok, %{text: text}}
  defp validate(input = %{text: text}) when is_binary(text), do: {:ok, input}

  defp validate(input),
    do: {:error, "expected a string or %{text: string}, got: #{inspect(input)}"}
end
