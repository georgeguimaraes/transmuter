defmodule Transmuter.Repo do
  @moduledoc """
  A model repository: its configuration files and its ONNX files in each variant.

  Hugging Face ONNX exports keep each component (`model`, `encoder_model`, `decoder_model_merged`, ...) in `onnx/`, in variants that differ by a suffix: `model.onnx` is fp32, `model_q4.onnx` has 4-bit weights, and so on. `load!/2` reads the configuration of the repository, and `component_file/3` picks the file of a component.

  The variant of a component comes from, in this order:

    1. the `:variant` option, an atom such as `:q4` or a map from component names to atoms
    2. the `dtype` of the `transformers.js_config` block in `config.json`, a string or a map
    3. the default that the task gives, when the repository has that file
    4. fp32, then the other variants in the order of `@variants`

  A repository has two chat templates. `:chat_template` is that of the tokenizer: `chat_template.jinja`, else the template of `tokenizer_config.json`, else that of the processor. `:processor_chat_template` is that of the processor of a multimodal model, as `ProcessorMixin` of Python transformers 4.57 loads it: the template of `chat_template.json`, else `chat_template.jinja`, else the `"chat_template"` of `processor_config.json`.
  """

  alias Transmuter.{Hub, HubError}

  require Logger

  defstruct [
    :source,
    :files,
    :config,
    :generation_config,
    :tokenizer_config,
    :special_tokens,
    :chat_template,
    :processor_chat_template,
    opts: []
  ]

  # File suffix of each variant, as in transformers.js (utils/dtypes.js). fp16 and q4f16
  # come last, because native f16 can drift from the f32 result in a deep graph.
  @variants [
    fp32: "",
    q4: "_q4",
    q8: "_quantized",
    int8: "_int8",
    uint8: "_uint8",
    bnb4: "_bnb4",
    fp16: "_fp16",
    q4f16: "_q4f16"
  ]

  @doc """
  Reads the configuration files of `source` (`{:hf, repo}` or a local directory).

  A Hub source reads its file list from the Hub again, or from the cache when the Hub cannot be reached.

  Options: `:variant` and the options of `Transmuter.Hub`.
  """
  def load!(source, opts \\ []) do
    # One read of the file list per load, so files added to the repository are found.
    Hub.info(source, [refresh: true] ++ opts)
    json = &Hub.json(source, &1, opts)
    tokenizer_config = json.("tokenizer_config.json") || %{}
    config = json.("config.json") || %{}

    jinja =
      case Hub.fetch(source, "chat_template.jinja", opts) do
        {:ok, path} -> File.read!(path)
        :error -> nil
      end

    processor_template = processor_chat_template(json, jinja)

    %__MODULE__{
      source: source,
      files: Hub.files(source, opts),
      config: config,
      generation_config: json.("generation_config.json") || from_model_config(config),
      tokenizer_config: tokenizer_config,
      special_tokens: json.("special_tokens_map.json") || %{},
      chat_template: jinja || tokenizer_chat_template(tokenizer_config) || processor_template,
      processor_chat_template: processor_template,
      opts: opts
    }
  end

  @doc """
  Returns the path in the repository of the ONNX file of `component` (for example `"model"` or `"decoder_model_merged"`), in the variant that the rules of this module pick. `default` is the variant of the task.
  """
  def component_file(repo, component, default \\ :fp32) do
    case requested(repo, component) || configured(repo, component) do
      nil ->
        Enum.find_value(
          Enum.uniq([default, :fp32 | Keyword.keys(@variants)]),
          &file(repo, component, &1)
        ) ||
          raise ArgumentError, "#{inspect(repo.source)} has no ONNX file for #{component}"

      variant ->
        file(repo, component, variant) ||
          raise ArgumentError,
                "#{inspect(repo.source)} has no #{variant} file for #{component}, " <>
                  "variants: #{inspect(variants(repo, component))}"
    end
  end

  @doc """
  Loads the ONNX file of `component` with `Transmuter.load!/2`. `opts` go to `Transmuter.load!/2`.
  """
  def load_component!(repo, component, default \\ :fp32, opts \\ []) do
    file = component_file(repo, component, default)

    case repo.source do
      {:hf, name} -> Transmuter.load!({:hf, name, file}, Keyword.merge(repo.opts, opts))
      dir -> Transmuter.load!(Path.join(dir, file), opts)
    end
  end

  @doc """
  Returns the special token `name` (such as `"pad"` or `"eos"`) as a string, or `nil`.

  The token comes from `special_tokens_map.json`, then `tokenizer_config.json`, then the `"extra_special_tokens"` map of `tokenizer_config.json`, as in Python transformers. A token in these files is a string or a map with `"content"`.
  """
  def special_token(repo, name) do
    key = name <> "_token"
    extra = repo.tokenizer_config["extra_special_tokens"]
    extra = if is_map(extra), do: extra[key]

    case repo.special_tokens[key] || repo.tokenizer_config[key] || extra do
      %{"content" => content} when is_binary(content) -> content
      content when is_binary(content) -> content
      _none -> nil
    end
  end

  @doc """
  Returns the id of the image token of a vision language model from `config.json`, or `nil`.

  The id is `image_token_id`, or else `image_token_index`, its other name in the `attribute_map` of `Mistral3Config` and `LlavaConfig` of Python transformers.
  """
  def image_token_id(repo), do: repo.config["image_token_id"] || repo.config["image_token_index"]

  @doc """
  Returns the first result of `read` that is not `nil`, on the source of `repo` and then on the base model in its model card. `read` takes a source and reads the sentence-transformers files of that source, such as `modules.json`. ONNX exports often leave these files out, and their base model has them.

  When the Hub denies the base model, such as a gated model without a token or a model that does not exist, this function logs a warning and does not read it.
  """
  def sentence_transformers(repo, read) do
    Enum.find_value([repo.source | base_models(repo)], fn source ->
      try do
        read.(source)
      rescue
        error in HubError ->
          if error.status in [401, 403, 404] do
            Logger.warning(
              "#{Exception.message(error)}. Transmuter cannot read the sentence-transformers " <>
                "files of #{inspect(source)}, such as its modules, so the result can differ from that model"
            )

            nil
          else
            reraise error, __STACKTRACE__
          end
      end
    end)
  end

  defp base_models(repo) do
    case get_in(Hub.info(repo.source, repo.opts) || %{}, ["cardData", "base_model"]) do
      [name | _] -> [{:hf, name}]
      name when is_binary(name) -> [{:hf, name}]
      nil -> []
    end
  end

  @doc """
  Returns the variants that the repository has for `component`.
  """
  def variants(repo, component) do
    for {variant, _} <- @variants, file(repo, component, variant), do: variant
  end

  # The file of a variant, in onnx/ or, for some repositories, at the root.
  defp file(repo, component, variant) do
    name = component <> Keyword.fetch!(@variants, variant) <> ".onnx"
    Enum.find(["onnx/" <> name, name], &(&1 in repo.files))
  end

  defp requested(repo, component) do
    variant =
      case repo.opts[:variant] do
        %{} = map -> Map.get(map, component) || Map.get(map, String.to_atom(component))
        variant -> variant
      end

    if variant != nil and not Keyword.has_key?(@variants, variant) do
      raise ArgumentError,
            "unknown variant #{inspect(variant)}, variants: #{inspect(Keyword.keys(@variants))}"
    end

    variant
  end

  defp configured(repo, component) do
    dtype =
      case get_in(repo.config, ["transformers.js_config", "dtype"]) do
        %{} = map -> map[component]
        dtype -> dtype
      end

    dtype && dtype_variant(dtype)
  end

  defp dtype_variant(dtype) do
    Enum.find(Keyword.keys(@variants), &(Atom.to_string(&1) == dtype)) ||
      raise ArgumentError, "unknown transformers.js dtype #{inspect(dtype)}"
  end

  # The defaults of GenerationConfig in Python transformers that are not None, from
  # _get_global_generation_defaults.
  @generation_defaults %{
    "max_length" => 20,
    "min_length" => 0,
    "do_sample" => false,
    "early_stopping" => false,
    "num_beams" => 1,
    "temperature" => 1.0,
    "top_k" => 50,
    "top_p" => 1.0,
    "typical_p" => 1.0,
    "repetition_penalty" => 1.0,
    "length_penalty" => 1.0,
    "no_repeat_ngram_size" => 0,
    "encoder_no_repeat_ngram_size" => 0,
    "num_return_sequences" => 1,
    "output_scores" => false,
    "return_dict_in_generate" => false,
    "remove_invalid_values" => false,
    "num_beam_groups" => 1,
    "diversity_penalty" => 0.0
  }

  # GenerationConfig.from_model_config of Python transformers, for a repository without
  # generation_config.json: the keys of config.json that are not null. The configuration of
  # the decoder gives the keys that config.json does not set or sets to their default.
  defp from_model_config(config) do
    outer = for {key, value} <- config, value != nil, into: %{}, do: {key, value}
    decoder = Enum.find_value(~w(decoder generator text_config), %{}, &config[&1])

    Enum.reduce(decoder, outer, fn {key, value}, generation ->
      default = @generation_defaults[key]

      if Map.get(generation, key, default) == default,
        do: Map.put(generation, key, value),
        else: generation
    end)
  end

  defp processor_chat_template(json, jinja) do
    get_in(json.("chat_template.json") || %{}, ["chat_template"]) || jinja ||
      get_in(json.("processor_config.json") || %{}, ["chat_template"])
  end

  # The template of tokenizer_config.json: a string, or a list of named templates.
  defp tokenizer_chat_template(tokenizer_config) do
    case tokenizer_config["chat_template"] do
      [_ | _] = named -> Enum.find_value(named, &(&1["name"] == "default" && &1["template"]))
      template -> template
    end
  end
end
