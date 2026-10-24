defmodule Transmuter.Text.TextRanking do
  @moduledoc """
  The text ranking task. See `Transmuter.Text.text_ranking/2`.

  The scores follow `CrossEncoder.rank` of sentence-transformers. The query and the document are a pair of texts, or messages of the chat template. A sequence classification model scores the text with its one label. A causal language model, such as Qwen3-Reranker, scores it with the `LogitScore` module: the logit of a "yes" token minus the logit of a "no" token, at the last token of the text.
  """

  alias Transmuter.{ChatTemplate, Hub, Repo, Serving, Tokenizer}

  @modules ~w(Transformer LogitScore)

  # The pairs that sentence-transformers renders to check that a chat template carries the
  # query and the document.
  @probes [{"alpha", "bravo"}, {"charlie", "bravo"}, {"alpha", "foxtrot"}]

  @doc false
  def text_ranking(source, opts) do
    opts = Keyword.validate!(opts, Serving.options() ++ [:prompt, :scores_function, :top_k])
    repo = Repo.load!(source, Keyword.take(opts, [:variant | Hub.options()]))
    # Each document is one row, so the tokenizer does not split a text into windows.
    tokenizer = Tokenizer.configure(Serving.tokenizer(repo, opts), stride: nil)
    ranking = ranking(repo, tokenizer, opts)

    model =
      opts[:model] ||
        Repo.load_component!(
          repo,
          "model",
          :fp32,
          [outputs: ["logits"]] ++ Keyword.take(opts, [:backend, :compute_type])
        )

    opts =
      [max_length: ranking.max_length, encode_options: ranking.encode_options]
      |> Keyword.merge(opts)

    Serving.text(
      repo,
      model,
      tokenizer,
      opts,
      fn outputs, inputs ->
        scores = ranking.score.(outputs["logits"], inputs["attention_mask"])
        %{score: Serving.scores(scores, ranking.scores_function)}
      end,
      fn rows, _items ->
        results =
          rows
          |> Enum.with_index(&%{index: &2, score: Nx.to_number(&1.score)})
          |> Enum.sort_by(& &1.score, :desc)

        %{results: if(opts[:top_k], do: Enum.take(results, opts[:top_k]), else: results)}
      end,
      &validate(&1, ranking.sequence)
    )
  end

  @doc false
  # The steps of the scores of `repo`: `:sequence` takes a query and a document and gives the
  # sequence to encode with `:encode_options`, and `:score` takes the logits and the attention
  # mask of a batch and gives the score of each row, before `:scores_function`.
  def ranking(repo, tokenizer, opts) do
    config = Repo.sentence_transformers(repo, &cross_encoder(&1, repo.opts))
    prompt = Keyword.get(opts, :prompt, config[:prompt])

    # As sentence-transformers, a CrossEncoder takes messages when its modality_config has
    # them, and a model without its files when the tokenizer has a chat template.
    {chat?, logit_score} =
      cond do
        config ->
          {config.chat?, config.logit_score}

        causal?(repo) ->
          {true,
           %{"true_token_id" => id!(tokenizer, "yes"), "false_token_id" => id!(tokenizer, "no")}}

        true ->
          {repo.chat_template != nil, nil}
      end

    input =
      if chat?,
        do:
          chat(
            %{repo | chat_template: config[:template] || repo.chat_template},
            tokenizer,
            prompt
          ),
        else: pair(prompt)

    # The score of a row has no classes, so a softmax does not apply.
    scores_function =
      Keyword.get_lazy(opts, :scores_function, fn -> scores_function(repo, config) end)

    unless scores_function in [:sigmoid, :none] do
      raise ArgumentError,
            "expected :scores_function to be :sigmoid or :none, got: #{inspect(scores_function)}"
    end

    Map.merge(input, %{
      score: if(logit_score, do: logit_score(logit_score), else: classifier(repo)),
      scores_function: scores_function,
      max_length: config[:max_length]
    })
  end

  # The configuration of a sentence-transformers CrossEncoder in `source`.
  defp cross_encoder(source, opts) do
    case Hub.json(source, "config_sentence_transformers.json", opts) do
      %{"model_type" => "CrossEncoder"} = config ->
        modules = Hub.json(source, "modules.json", opts) || []
        types = Enum.map(modules, &module_type/1)

        unless types -- @modules == [] do
          raise ArgumentError,
                "#{inspect(source)} has sentence-transformers modules that Transmuter does not run: " <>
                  "#{inspect(types -- @modules)}"
        end

        logit_score = Enum.find(modules, &(module_type(&1) == "LogitScore"))
        transformer = Hub.json(source, "sentence_bert_config.json", opts) || %{}

        template =
          case Hub.fetch(source, "chat_template.jinja", opts) do
            {:ok, path} -> File.read!(path)
            :error -> nil
          end

        %{
          chat?: Map.has_key?(transformer["modality_config"] || %{}, "message"),
          logit_score:
            logit_score && Hub.json(source, Path.join(logit_score["path"], "config.json"), opts),
          template: template,
          prompt: Map.get(config["prompts"] || %{}, config["default_prompt_name"]),
          activation: config["activation_fn"],
          max_length: transformer["max_seq_length"]
        }

      _other ->
        nil
    end
  end

  defp module_type(module), do: module["type"] |> String.split(".") |> List.last()

  defp causal?(repo),
    do: String.ends_with?(List.first(repo.config["architectures"] || []) || "", "ForCausalLM")

  defp id!(tokenizer, token) do
    Tokenizer.token_to_id(tokenizer, token) ||
      raise ArgumentError,
            "the vocabulary has no token #{inspect(token)} to score a causal language model"
  end

  # As sentence-transformers, a prompt goes before the query.
  defp pair(prompt), do: %{sequence: &{"#{prompt}#{&1}", &2}, encode_options: []}

  # As sentence-transformers, the query and the document are messages with the roles "query"
  # and "document", after the prompt as a system message. The chat template adds the special
  # tokens. When truncation cuts a text, the end of the template replaces its last tokens.
  defp chat(repo, tokenizer, prompt) do
    render = fn prompt, query, document ->
      # As in Python, an empty prompt is no prompt.
      system = if prompt in [nil, ""], do: [], else: [%{role: "system", content: prompt}]

      messages =
        system ++ [%{role: "query", content: query}, %{role: "document", content: document}]

      ChatTemplate.render(repo, messages, add_generation_prompt: false)
    end

    [base | renders] =
      Enum.map(@probes, fn {query, document} -> render.(nil, query, document) end)

    dropped =
      for {role, render} <- Enum.zip(["query", "document"], renders), render == base, do: role

    unless dropped == [] do
      raise ArgumentError,
            "the chat template of #{inspect(repo.source)} cannot carry a query and a document: " <>
              "the #{Enum.join(dropped, " and ")} content does not reach the text"
    end

    %{
      sequence: &render.(prompt, &1, &2),
      encode_options: [add_special_tokens: false, suffix: suffix(tokenizer, render, prompt)]
    }
  end

  defp classifier(repo) do
    # transformers gives 2 labels to a configuration without id2label.
    labels = map_size(repo.config["id2label"] || %{"0" => "LABEL_0", "1" => "LABEL_1"})

    unless labels == 1 do
      raise ArgumentError,
            "text ranking needs a model with one label, as CrossEncoder.rank of sentence-transformers, " <>
              "got #{labels} labels in id2label of config.json"
    end

    fn logits, _mask -> logits[[.., 0]] end
  end

  # The LogitScore module: the logit of the true token minus the logit of the false token, at
  # the last token of each row.
  defp logit_score(config = %{"true_token_id" => yes}) do
    no = config["false_token_id"]

    fn logits, mask ->
      scores =
        if no,
          do: Nx.subtract(logits[[.., .., yes]], logits[[.., .., no]]),
          else: logits[[.., .., yes]]

      last =
        Nx.subtract(Nx.axis_size(mask, 1) - 1, Nx.argmax(Nx.reverse(mask, axes: [1]), axis: 1))

      scores |> Nx.take_along_axis(Nx.new_axis(last, 1), axis: 1) |> Nx.squeeze(axes: [1])
    end
  end

  # The end of the template: the tokens that two texts with different fillers have in common
  # at their end, as sentence-transformers finds it, without padding or truncation.
  defp suffix(tokenizer, render, prompt) do
    tokenizer = Tokenizer.configure(tokenizer, length: nil, stride: nil)

    [first, second] =
      for filler <- ["0", "1 2 3 4"] do
        {_tensors, [[encoding]]} =
          Tokenizer.encode(tokenizer, [render.(prompt, filler, filler)],
            add_special_tokens: false
          )

        Tokenizer.ids(tokenizer, encoding)
      end

    count =
      Enum.zip(Enum.reverse(first), Enum.reverse(second))
      |> Enum.take_while(fn {a, b} -> a == b end)
      |> length()

    if count < min(length(first), length(second)), do: Enum.take(first, -count), else: []
  end

  # As sentence-transformers: the activation_fn of the CrossEncoder, else the activation in
  # config.json, from version 4 or from older versions, else a sigmoid for the one label.
  defp scores_function(repo, config) do
    case config[:activation] ||
           get_in(repo.config, ["sentence_transformers", "activation_fn"]) ||
           repo.config["sbert_ce_default_activation_function"] do
      nil ->
        :sigmoid

      name ->
        case name |> String.split(".") |> List.last() do
          "Identity" ->
            :none

          "Sigmoid" ->
            :sigmoid

          _other ->
            raise ArgumentError, "unsupported activation #{inspect(name)}, set :scores_function"
        end
    end
  end

  defp validate(input = %{query: query, documents: [_ | _] = documents}, sequence)
       when is_binary(query) do
    if Enum.all?(documents, &is_binary/1),
      do: {:ok, Enum.map(documents, &sequence.(query, &1))},
      else: invalid(input)
  end

  defp validate(input, _sequence), do: invalid(input)

  defp invalid(input),
    do: {:error, "expected %{query: string, documents: [string, ...]}, got: #{inspect(input)}"}
end
