defmodule Transmuter.ChatTemplate do
  @moduledoc """
  Renders the chat template of a model repository, like `apply_chat_template` in Python transformers.

  Needs the optional `otera` dependency, which renders Jinja templates in pure Elixir.

      repo = Transmuter.Repo.load!({:hf, "HuggingFaceTB/SmolLM2-135M-Instruct"})
      Transmuter.ChatTemplate.render(repo, [%{role: "user", content: "Hello"}])
      #=> "<|im_start|>system\\nYou are a helpful AI assistant named SmolLM...<|im_start|>assistant\\n"

  Map keys can be atoms or strings. A plain Elixir map gives its keys to `tojson` in alphabetical order. To get the key order of transformers for tool definitions or messages that come from JSON, decode the JSON with `Otera.JSON.decode!/1`.
  """

  @compile {:no_warn_undefined, Otera.ChatTemplate}

  alias Transmuter.Repo

  @doc """
  Renders `messages`, a list of maps with `:role` and `:content`, with the chat template of `repo`.

  ## Options

    * `:continue_final_message` - ends the prompt inside the final message, so that the model continues it. The default is `false`.

    * `:add_generation_prompt` - ends the prompt with the start of an assistant message. The default is `true`, or `false` with `:continue_final_message`.

    * `:tools` - a list of tool definitions in JSON schema form. A repository with a template named `"tool_use"` renders tools with it.

    * `:template_options` - other variables of the template, such as `enable_thinking: false` for Qwen3.

    * `:template` - the template to render in place of that of the tokenizer, such as `Transmuter.Repo.processor_chat_template/1`.
  """
  def render(repo, messages, opts \\ []) do
    unless Code.ensure_loaded?(Otera.ChatTemplate) do
      raise ArgumentError,
            ~s(Transmuter.ChatTemplate needs the optional otera dependency, add {:otera, "~> 0.1"} to your deps)
    end

    continue? = Keyword.get(opts, :continue_final_message, false)

    special =
      for {name, token} <- [bos_token: "bos", eos_token: "eos"],
          content = Repo.special_token(repo, token),
          into: %{},
          do: {name, content}

    render_opts = [
      add_generation_prompt: Keyword.get(opts, :add_generation_prompt, not continue?),
      continue_final_message: continue?,
      tools: opts[:tools],
      variables: Map.merge(special, Map.new(opts[:template_options] || []))
    ]

    template = opts[:template] || template(repo, opts[:tools])

    case Otera.ChatTemplate.render(template, messages, render_opts) do
      {:ok, prompt} ->
        prompt

      {:error, error} ->
        raise ArgumentError, "the chat template failed: #{Exception.message(error)}"
    end
  end

  defp template(repo, tools) do
    named = repo.tokenizer_config["chat_template"]

    tool_use =
      tools && is_list(named) &&
        Enum.find_value(named, &(&1["name"] == "tool_use" && &1["template"]))

    tool_use || repo.chat_template ||
      raise ArgumentError, "#{inspect(repo.source)} has no chat template"
  end
end
