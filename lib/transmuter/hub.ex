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

  def fetch({:hf, repo}, file, opts) do
    path = Path.join(cache_dir(repo, opts), file)

    with false <- File.exists?(path),
         {:ok, files} <- listed({:hf, repo}, file, opts) do
      for f <- data_files(file, files) ++ [file] do
        download!(repo, f, Path.join(cache_dir(repo, opts), f))
      end

      {:ok, path}
    else
      true -> {:ok, path}
      :error -> :error
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

  The list of a Hub repository is saved in the cache directory, so a repository that was downloaded once also works offline. The `refresh: true` option reads the list from the Hub again.
  """
  def files(source, opts \\ [])

  def files({:hf, repo}, opts) do
    listing = Path.join(cache_dir(repo, opts), ".files.json")

    if File.exists?(listing) and not Keyword.get(opts, :refresh, false) do
      listing |> File.read!() |> JSON.decode!()
    else
      %{"siblings" => siblings} = get!("https://huggingface.co/api/models/#{repo}").body
      files = Enum.map(siblings, & &1["rfilename"])
      File.mkdir_p!(Path.dirname(listing))
      File.write!(listing, JSON.encode!(files))
      files
    end
  end

  def files(dir, _opts) when is_binary(dir) do
    dir
    |> Path.join("**")
    |> Path.wildcard(match_dot: false)
    |> Enum.filter(&File.regular?/1)
    |> Enum.map(&Path.relative_to(&1, dir))
  end

  @doc """
  Returns the metadata of a Hub repository from the Hub API, such as `"pipeline_tag"` and `"cardData"`, or `nil` for a local directory.
  """
  def info({:hf, repo}), do: get!("https://huggingface.co/api/models/#{repo}").body
  def info(dir) when is_binary(dir), do: nil

  # A saved list can be older than the repository, so a missing file reads the list again.
  defp listed(source, file, opts) do
    Enum.find_value([opts, [refresh: true] ++ opts], :error, fn opts ->
      files = files(source, opts)
      if file in files, do: {:ok, files}
    end)
  end

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

      get!("https://huggingface.co/#{repo}/resolve/main/#{file}",
        into: File.stream!(target <> ".part")
      )

      File.rename!(target <> ".part", target)
    end
  end

  defp cache_dir(repo, opts) do
    Path.join(opts[:cache_dir] || Path.join([System.user_home!(), ".cache", "transmuter"]), repo)
  end

  defp get!(url, opts \\ []) do
    headers =
      if token = System.get_env("HF_TOKEN"), do: [authorization: "Bearer #{token}"], else: []

    Req.get!(url, [headers: headers, receive_timeout: 600_000, redirect_log_level: false] ++ opts)
  end
end
