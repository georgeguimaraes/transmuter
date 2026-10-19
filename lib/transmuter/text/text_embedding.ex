defmodule Transmuter.Text.TextEmbedding do
  @moduledoc """
  The text embedding task. See `Transmuter.Text.text_embedding/2`.
  """

  alias Transmuter.{Hub, Serving}

  @pools %{
    "pooling_mode_cls_token" => :cls_token_pooling,
    "pooling_mode_mean_tokens" => :mean_pooling,
    "pooling_mode_lasttoken" => :last_token_pooling
  }

  @doc false
  def text_embedding(source, opts) do
    opts = Keyword.validate!(opts, Serving.options() ++ [:output_pool, :embedding_processor])
    {repo, model, tokenizer} = Serving.load!(source, opts, :fp32)
    opts = Keyword.merge(sentence_transformers(repo, opts), opts)
    pool = opts[:output_pool]
    processor = opts[:embedding_processor]

    Serving.text(
      repo,
      model,
      tokenizer,
      opts,
      fn outputs, inputs ->
        hidden = outputs["last_hidden_state"]
        %{embedding: hidden |> pool(pool, inputs["attention_mask"]) |> process(processor)}
      end,
      fn [%{embedding: embedding}], _items -> %{embedding: embedding} end
    )
  end

  @modules ~w(Transformer Pooling Normalize)

  # The pooling, the normalization and the longest input of a sentence-transformers model,
  # from its 1_Pooling/config.json, modules.json and sentence_bert_config.json, else from
  # those of the base model in its card. ONNX exports often leave these files out. Without
  # them, the embedding is the mean of the token embeddings.
  defp sentence_transformers(repo, opts) do
    if Keyword.has_key?(opts, :output_pool) and Keyword.has_key?(opts, :embedding_processor),
      do: [],
      else:
        Enum.find_value(
          [repo.source | base_models(repo)],
          [output_pool: :mean_pooling],
          &sentence_transformers_config(&1, repo.opts)
        )
  end

  defp base_models(repo) do
    case get_in(Hub.info(repo.source, repo.opts) || %{}, ["cardData", "base_model"]) do
      [name | _] -> [{:hf, name}]
      name when is_binary(name) -> [{:hf, name}]
      nil -> []
    end
  end

  defp sentence_transformers_config(source, opts) do
    if pooling = Hub.json(source, "1_Pooling/config.json", opts) do
      types = Enum.map(Hub.json(source, "modules.json", opts) || [], &module_type/1)

      unless types -- @modules == [] do
        raise ArgumentError,
              "#{inspect(source)} has sentence-transformers modules that Transmuter does not run: " <>
                "#{inspect(types -- @modules)}. Set :output_pool and :embedding_processor to use the model without them"
      end

      [
        output_pool: pool(pooling),
        embedding_processor: if("Normalize" in types, do: :l2_norm),
        max_length: (Hub.json(source, "sentence_bert_config.json", opts) || %{})["max_seq_length"]
      ]
    end
  end

  defp module_type(module), do: module["type"] |> String.split(".") |> List.last()

  defp pool(pooling) do
    Enum.find_value(@pools, fn {key, pool} -> pooling[key] && pool end) ||
      raise ArgumentError, "unsupported sentence-transformers pooling #{inspect(pooling)}"
  end

  defp pool(hidden, nil, _mask), do: hidden
  # The first and the last token are the first and the last positions of the attention mask,
  # so that the pooling works with padding on either side.
  defp pool(hidden, :cls_token_pooling, mask), do: take(hidden, Nx.argmax(mask, axis: 1))

  defp pool(hidden, :mean_pooling, mask) do
    mask = mask |> Nx.as_type(Nx.type(hidden)) |> Nx.new_axis(-1)
    Nx.divide(Nx.sum(Nx.multiply(hidden, mask), axes: [1]), Nx.sum(mask, axes: [1]))
  end

  defp pool(hidden, :last_token_pooling, mask) do
    last = Nx.subtract(Nx.axis_size(mask, 1) - 1, Nx.argmax(Nx.reverse(mask, axes: [1]), axis: 1))
    take(hidden, last)
  end

  defp take(hidden, position) do
    index =
      Nx.broadcast(
        Nx.reshape(position, {:auto, 1, 1}),
        {Nx.axis_size(hidden, 0), 1, Nx.axis_size(hidden, 2)}
      )

    hidden |> Nx.take_along_axis(index, axis: 1) |> Nx.squeeze(axes: [1])
  end

  defp process(embedding, nil), do: embedding

  defp process(embedding, :l2_norm) do
    norm = embedding |> Nx.pow(2) |> Nx.sum(axes: [-1], keep_axes: true) |> Nx.sqrt()
    Nx.divide(embedding, Nx.max(norm, 1.0e-12))
  end
end
