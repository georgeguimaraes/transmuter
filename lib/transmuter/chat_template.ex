defmodule Transmuter.ChatTemplate do
  @moduledoc """
  Renders the chat template of a model repository, like `apply_chat_template` in Python transformers.

  Needs the optional `minijinja_ex` dependency, with its Hugging Face compatibility.

      repo = Transmuter.Repo.load!({:hf, "HuggingFaceTB/SmolLM2-135M-Instruct"})
      Transmuter.ChatTemplate.render(repo, [%{role: "user", content: "Hello"}])
      #=> "<|im_start|>system\\nYou are a helpful AI assistant named SmolLM...<|im_start|>assistant\\n"

  Map keys can be atoms or strings. JSON values in the prompt, such as tool definitions, have their keys in alphabetical order, because Elixir maps do not keep the insertion order.
  """

  @compile {:no_warn_undefined, MinijinjaEx}

  alias Transmuter.Repo

  # transformers marks the end of the final message with this tag to find it in the prompt.
  @continue_tag "CONTINUE_FINAL_MESSAGE_TAG "

  @doc """
  Renders `messages`, a list of maps with `:role` and `:content`, with the chat template of `repo`.

  ## Options

    * `:continue_final_message` - ends the prompt inside the final message, so that the model continues it. The default is `false`.

    * `:add_generation_prompt` - ends the prompt with the start of an assistant message. The default is `true`, or `false` with `:continue_final_message`.

    * `:tools` - a list of tool definitions in JSON schema form. A repository with a template named `"tool_use"` renders tools with it.

    * `:template_options` - other variables of the template, such as `enable_thinking: false` for Qwen3.
  """
  def render(repo, messages, opts \\ []) do
    unless Code.ensure_loaded?(MinijinjaEx) do
      raise ArgumentError,
            ~s(Transmuter.ChatTemplate needs the optional minijinja_ex dependency, add {:minijinja_ex, github: "georgeguimaraes/minijinja_ex", branch: "feat/chat-template-compat"} to your deps)
    end

    continue? = Keyword.get(opts, :continue_final_message, false)
    messages = if continue?, do: tag_final_message(messages), else: messages

    special =
      for {name, token} <- [bos_token: "bos", eos_token: "eos"],
          content = Repo.special_token(repo, token),
          into: %{},
          do: {name, content}

    context =
      %{
        messages: messages,
        add_generation_prompt: Keyword.get(opts, :add_generation_prompt, not continue?)
      }
      |> Map.merge(special)
      |> Map.merge(if tools = opts[:tools], do: %{tools: tools}, else: %{})
      |> Map.merge(Map.new(opts[:template_options] || []))
      |> stringify()

    env = MinijinjaEx.new_env(huggingface_compat: true)

    case MinijinjaEx.render_string(env, template(repo, opts[:tools]), context) do
      {:ok, prompt} when continue? ->
        cut_at_tag(prompt)

      {:ok, prompt} ->
        prompt

      {:error, error} ->
        raise ArgumentError, "the chat template failed: #{Exception.message(error)}"
    end
  end

  # transformers renders {% generation %} blocks as their content. minijinja has no custom
  # statements, so the block becomes an if with the same whitespace control.
  defp template(repo, tools) do
    named = repo.tokenizer_config["chat_template"]

    tool_use =
      tools && is_list(named) &&
        Enum.find_value(named, &(&1["name"] == "tool_use" && &1["template"]))

    template =
      tool_use || repo.chat_template ||
        raise ArgumentError, "#{inspect(repo.source)} has no chat template"

    Regex.replace(~r/\{%(-?)\s*(end)?generation\s*(-?)%\}/, template, fn _,
                                                                         open,
                                                                         end_tag,
                                                                         close ->
      "{%#{open} #{if end_tag == "", do: "if true", else: "endif"} #{close}%}"
    end)
  end

  defp tag_final_message(messages) do
    {final, rest} = List.pop_at(messages, -1)
    content = final[:content] || final["content"]

    unless is_binary(content) do
      raise ArgumentError,
            "expected the final message to have text content to continue, got: #{inspect(final)}"
    end

    key = if Map.has_key?(final, :content), do: :content, else: "content"
    rest ++ [Map.put(final, key, content <> @continue_tag)]
  end

  # As transformers: cut the prompt at the tag, and remove the spaces before it when the
  # template trimmed the space of the tag.
  defp cut_at_tag(prompt) do
    tag = String.trim(@continue_tag)

    {position, _} =
      prompt |> :binary.matches(tag) |> List.last() ||
        raise ArgumentError, "the chat template removed the final message"

    before = binary_part(prompt, 0, position)

    if binary_part(prompt, position, min(byte_size(@continue_tag), byte_size(prompt) - position)) ==
         @continue_tag,
       do: before,
       else: String.trim_trailing(before)
  end

  defp stringify(map) when is_map(map),
    do: Map.new(map, fn {key, value} -> {to_string(key), stringify(value)} end)

  defp stringify(list) when is_list(list), do: Enum.map(list, &stringify/1)
  defp stringify(value), do: value
end
