defmodule Transmuter.Hub do
  @moduledoc """
  Reads the files of a model repository on the Hugging Face Hub or in a local directory.

  A source is `{:hf, repo}` or the path of a local directory. Hub files download once into the cache directory and come from there later. The cache directory is `~/.cache/transmuter`, or the `:cache_dir` option. Set the `HF_TOKEN` environment variable for private repositories.
  """

  @doc """
  Returns the local path of `file` in `source`. Raises if the repository does not have the file.

  A Hub file downloads on the first call. An ONNX file also gets its external data files.
  """
  def fetch!(source, file, opts \\ []) do
    case fetch(source, file, opts) do
      {:ok, path} -> path
      :error -> raise ArgumentError, "#{inspect(source)} has no file #{file}"
    end
  end

  @doc """
  Returns `{:ok, path}` for `file` in `source`, or `:error` when the repository does not have the file.
  """
  def fetch(source, file, opts \\ [])

  def fetch(source = {:hf, repo}, file, opts) do
    dir = cache_dir(repo, opts)
    path = Path.join(dir, file)
    files = if File.exists?(path), do: [file], else: files(source, opts)

    if file in files do
      for f <- data_files(file, files) ++ [file], do: download!(repo, f, Path.join(dir, f))
      {:ok, path}
    else
      :error
    end
  end

  def fetch(dir, file, _opts) when is_binary(dir) do
    path = Path.join(dir, file)
    if File.exists?(path), do: {:ok, path}, else: :error
  end

  @doc """
  Reads and decodes the JSON `file` in `source`, or returns `nil` when the repository does not have the file.
  """
  def json(source, file, opts \\ []) do
    case fetch(source, file, opts) do
      {:ok, path} -> path |> File.read!() |> JSON.decode!()
      :error -> nil
    end
  end

  @doc """
  Returns the names of all files in `source`, with paths relative to the repository root.
  """
  def files(source, opts \\ [])

  def files(source = {:hf, _repo}, opts),
    do: Enum.map(info(source, opts)["siblings"], & &1["rfilename"])

  def files(dir, _opts) when is_binary(dir) do
    dir
    |> Path.join("**")
    |> Path.wildcard(match_dot: false)
    |> Enum.filter(&File.regular?/1)
    |> Enum.map(&Path.relative_to(&1, dir))
  end

  @doc """
  Returns the metadata of a Hub repository from the Hub API, such as `"siblings"` (the files), `"pipeline_tag"` and `"cardData"`, or `nil` for a local directory.

  The metadata is saved in the cache directory and comes from there later. The `refresh: true` option reads it from the Hub again, and uses the saved metadata when the Hub cannot be reached, so a repository that was downloaded once also works offline.
  """
  def info(source, opts \\ [])

  def info({:hf, repo}, opts) do
    path = Path.join(cache_dir(repo, opts), ".info.json")
    saved? = File.exists?(path)

    if saved? and not Keyword.get(opts, :refresh, false) do
      path |> File.read!() |> JSON.decode!()
    else
      case get("https://huggingface.co/api/models/#{repo}") do
        {:ok, %{status: 200, body: info}} ->
          File.mkdir_p!(Path.dirname(path))
          File.write!(path, JSON.encode!(info))
          info

        _error when saved? ->
          path |> File.read!() |> JSON.decode!()

        {:ok, %{status: status, body: body}} ->
          raise ArgumentError,
                "the Hugging Face Hub gave status #{status} for #{repo}: #{inspect(body)}. " <>
                  "Set HF_TOKEN for a private or gated repository"

        {:error, error} ->
          raise ArgumentError,
                "could not reach the Hugging Face Hub for #{repo}: #{Exception.message(error)}"
      end
    end
  end

  def info(dir, _opts) when is_binary(dir), do: nil

  # External data files of an ONNX file are named <file>_data, <file>_data_1, ... or <file>.data.
  defp data_files(file, files) do
    if String.ends_with?(file, ".onnx"),
      do:
        Enum.filter(files, &(String.starts_with?(&1, file <> "_data") or &1 == file <> ".data")),
      else: []
  end

  defp download!(repo, file, target) do
    unless File.exists?(target) do
      File.mkdir_p!(Path.dirname(target))

      Req.get!(
        "https://huggingface.co/#{repo}/resolve/main/#{file}",
        req_options(into: File.stream!(target <> ".part"), http_errors: :raise)
      )

      File.rename!(target <> ".part", target)
    end
  end

  defp cache_dir(repo, opts) do
    Path.join(opts[:cache_dir] || Path.join([System.user_home!(), ".cache", "transmuter"]), repo)
  end

  defp get(url), do: Req.get(url, req_options(retry: false))

  defp req_options(opts) do
    headers =
      if token = System.get_env("HF_TOKEN"), do: [authorization: "Bearer #{token}"], else: []

    [headers: headers, receive_timeout: 600_000, redirect_log_level: false] ++ opts
  end
end
