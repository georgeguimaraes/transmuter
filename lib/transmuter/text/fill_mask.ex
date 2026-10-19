defmodule Transmuter.Text.FillMask do
  @moduledoc """
  The fill-mask task. See `Transmuter.Text.fill_mask/2`.
  """

  @compile {:no_warn_undefined, Tokenizers.Encoding}

  alias Transmuter.{Serving, Tokenizer}

  @doc false
  def fill_mask(source, opts) do
    opts = Keyword.validate!(opts, Serving.options() ++ [top_k: 5])
    {repo, model, tokenizer} = Serving.load!(source, opts, :fp32)

    mask =
      tokenizer.special_tokens[:mask] || raise ArgumentError, "the tokenizer has no mask token"

    mask_id = Tokenizer.token_to_id(tokenizer, mask)

    Serving.text(
      repo,
      model,
      tokenizer,
      opts,
      fn outputs, inputs ->
        logits = outputs["logits"]
        {batch, _length, vocabulary} = Nx.shape(logits)
        position = inputs["input_ids"] |> Nx.equal(mask_id) |> Nx.argmax(axis: 1)
        index = position |> Nx.reshape({batch, 1, 1}) |> Nx.broadcast({batch, 1, vocabulary})
        logits = logits |> Nx.take_along_axis(index, axis: 1) |> Nx.squeeze(axes: [1])
        {scores, ids} = logits |> Serving.softmax() |> Nx.top_k(k: opts[:top_k])
        %{scores: scores, ids: ids}
      end,
      fn [%{scores: scores, ids: ids}], [{text, encoding}] ->
        masks = Enum.count(Tokenizers.Encoding.get_ids(encoding), &(&1 == mask_id))

        unless masks == 1 do
          raise ArgumentError,
                "expected one mask token in the encoded text, got #{masks}. " <>
                  "Truncation removes a mask past the longest sequence length: #{inspect(text)}"
        end

        predictions =
          Enum.zip_with(Nx.to_list(scores), Nx.to_list(ids), fn score, id ->
            %{score: score, token: Tokenizer.decode(tokenizer, [id])}
          end)

        %{predictions: predictions}
      end,
      &masked(&1, mask)
    )
  end

  defp masked(text, mask) when is_binary(text), do: {:ok, [String.replace(text, "[MASK]", mask)]}
  defp masked(input, _mask), do: Serving.string(input)
end
