defmodule Transmuter.Repo do
  @moduledoc """
  A model repository: its configuration files and its ONNX files in each variant.

  Hugging Face ONNX exports keep each component (`model`, `encoder_model`, `decoder_model_merged`, ...) in `onnx/`, in variants that differ by a suffix: `model.onnx` is fp32, `model_q4.onnx` has 4-bit weights, and so on. `load!/2` reads the configuration of the repository, and `component_file/3` picks the file of a component.

  The variant of a component comes from, in this order:

    1. the `:variant` option, an atom such as `:q4` or a map from component names to atoms
    2. the `dtype` of the `transformers.js_config` block in `config.json`, a string or a map
    3. the default that the task gives, when the repository has that file
    4. fp32, then the other variants in the order of `@variants`
  """

  alias Transmuter.Hub

  defstruct [
    :source,
    :files,
    :config,
    :generation_config,
    :tokenizer_config,
    :special_tokens,
    :chat_template,
    opts: []
  ]

  # File suffix of each variant, as in transformers.js (utils/dtypes.js). Variants that
  # Transmuter runs in f32 come first, so that a fallback avoids native f16.
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

  Options: `:cache_dir` (see `Transmuter.Hub`) and `:variant`.
  """
  def load!(source, opts \\ []) do
    # One read of the file list per load, so files added to the repository are found.
    Hub.info(source, [refresh: true] ++ opts)
    json = &Hub.json(source, &1, opts)
    tokenizer_config = json.("tokenizer_config.json") || %{}

    %__MODULE__{
      source: source,
      files: Hub.files(source, opts),
      config: json.("config.json") || %{},
      generation_config: json.("generation_config.json") || %{},
      tokenizer_config: tokenizer_config,
      special_tokens: json.("special_tokens_map.json") || %{},
      chat_template: chat_template(source, tokenizer_config, opts),
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

  # chat_template.jinja, then tokenizer_config.json (a string, or a list of named
  # templates), then chat_template.json, which multimodal processors use.
  defp chat_template(source, tokenizer_config, opts) do
    case Hub.fetch(source, "chat_template.jinja", opts) do
      {:ok, path} ->
        File.read!(path)

      :error ->
        case tokenizer_config["chat_template"] do
          template when is_binary(template) -> template
          [_ | _] = named -> Enum.find_value(named, &(&1["name"] == "default" && &1["template"]))
          nil -> get_in(Hub.json(source, "chat_template.json", opts) || %{}, ["chat_template"])
        end
    end
  end
end
