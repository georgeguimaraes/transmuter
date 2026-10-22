defmodule Transmuter.Hub do
  @moduledoc """
  Reads the files of a model repository on the Hugging Face Hub or in a local directory.

  A source is `{:hf, repo}` or the path of a local directory. Hub files download into the cache directory and come from there later. When the metadata of the repository gives a new ETag for a file, the file downloads again. Each download must match its ETag. The cache directory is `~/.cache/transmuter`, or the `:cache_dir` option. Set the `HF_TOKEN` environment variable for private repositories.

  ## Options

  * `:cache_dir` - the directory for Hub downloads. The default is `~/.cache/transmuter`.

  * `:revision` - a branch, a tag or a commit hash of the repository. The default is `"main"`. The files of another revision go in a directory of their own in the cache, `<repo>@<revision>`. The files of a commit hash do not change, so its metadata comes from the Hub only once.

  * `:offline` - when `true`, no request goes to the Hub. Files and metadata come only from the cache, and a file that is not in the cache raises. A file in the cache is used also when its ETag differs from the saved metadata. The default is `true` when the `HF_HUB_OFFLINE` or `TRANSFORMERS_OFFLINE` environment variable is `1`, `ON`, `YES` or `TRUE`, as in Python, and `false` otherwise.

  The tasks of `Transmuter.Text`, `Transmuter.Vision` and `Transmuter.Audio`, `Transmuter.Repo.load!/2` and `Transmuter.load!/2` take these options too.

  A large download shows a progress bar on the standard output. To disable it, set `config :transmuter, progress_bar_enabled: false`.
  """

  @options [:cache_dir, :revision, :offline]

  @doc """
  Returns the names of the options of this module, which the tasks take and give to it.
  """
  def options, do: @options

  @doc """
  Returns the local path of `file` in `source`. Raises if the repository does not have the file.

  A Hub file downloads when it is not in the cache or when its ETag in the metadata (see `info/2`) changed. An ONNX file also gets its external data files.
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
    info = info(source, opts)
    siblings = Map.new(info["siblings"], &{&1["rfilename"], &1})

    if Map.has_key?(siblings, file) do
      for f <- data_files(file, Map.keys(siblings)) ++ [file],
          do: download!(repo, info["sha"], siblings[f], Path.join(dir, f), opts)

      {:ok, Path.join(dir, file)}
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
  Returns the metadata of a Hub repository from the Hub API, such as `"sha"` (the commit), `"siblings"` (the files with their ETags), `"pipeline_tag"` and `"cardData"`, or `nil` for a local directory.

  The metadata is saved in the cache directory and comes from there later. The `refresh: true` option reads it from the Hub again, and uses the saved metadata when the Hub cannot be reached, so a repository that was downloaded once also works without a network. With `offline: true`, the metadata comes only from the cache.
  """
  def info(source, opts \\ [])

  def info({:hf, repo}, opts) do
    path = Path.join(cache_dir(repo, opts), ".info.json")
    saved? = File.exists?(path)

    refresh? = Keyword.get(opts, :refresh, false) and not commit?(opts[:revision])

    cond do
      saved? and (offline?(opts) or not refresh?) ->
        path |> File.read!() |> JSON.decode!()

      offline?(opts) ->
        raise ArgumentError, "#{repo} is not in the cache, and the Hub is offline"

      true ->
        case get(
               "https://huggingface.co/api/models/#{repo}/revision/#{revision(opts)}?blobs=true"
             ) do
          {:ok, %{status: 200, body: info}} ->
            File.mkdir_p!(Path.dirname(path))
            write_file!(path, JSON.encode!(info))
            info

          _error when saved? ->
            path |> File.read!() |> JSON.decode!()

          {:ok, %{headers: %{"x-error-code" => ["RevisionNotFound"]}}} ->
            raise ArgumentError, "#{repo} has no revision #{opts[:revision]}"

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

  # Each download writes its own part file, so processes that download the same file at the
  # same time do not write into one file. The rename puts a complete file in place at once.
  # The file comes from the commit of the metadata, so it matches the file list even when the
  # branch moves on.
  defp download!(repo, commit, sibling, target, opts) do
    file = sibling["rfilename"]
    etag = sibling["lfs"]["sha256"] || sibling["blobId"]
    hash = etag && hash_init(etag, sibling["size"])

    cond do
      File.exists?(target) and (offline?(opts) or current?(target, etag)) ->
        :ok

      offline?(opts) ->
        raise ArgumentError, "#{file} of #{repo} is not in the cache, and the Hub is offline"

      true ->
        File.mkdir_p!(Path.dirname(target))
        part = "#{target}.#{System.pid()}-#{System.unique_integer([:positive])}.part"

        try do
          response =
            File.open!(part, [:write, :binary], fn device ->
              Req.get!(
                "https://huggingface.co/#{repo}/resolve/#{commit}/#{file}",
                req_options(into: &write(device, file, hash, &1, &2), http_errors: :raise)
              )
            end)

          if etag && digest(Req.Response.get_private(response, :hash, hash)) != etag,
            do: raise(ArgumentError, "the download of #{file} of #{repo} does not match its ETag")

          File.rename!(part, target)
          if etag, do: save_etag(target, etag)
        after
          File.rm(part)
        end
    end
  end

  # A file without a saved ETag, such as a file of an earlier version of Transmuter, gets the
  # ETag of its contents.
  defp current?(_target, nil), do: true

  defp current?(target, etag) do
    case File.read(etag_path(target)) do
      {:ok, saved} ->
        saved == etag

      {:error, _reason} ->
        hash =
          target
          |> File.stream!(1_048_576)
          |> Enum.reduce(hash_init(etag, File.stat!(target).size), &:crypto.hash_update(&2, &1))

        if digest(hash) == etag, do: save_etag(target, etag) == :ok
    end
  end

  # The ETag of an LFS file is the SHA-256 of its contents, and the ETag of another file is its
  # Git blob id, the SHA-1 of a header and the contents.
  defp hash_init(etag, _size) when byte_size(etag) == 64, do: :crypto.hash_init(:sha256)

  defp hash_init(_etag, size),
    do: :crypto.hash_update(:crypto.hash_init(:sha), "blob #{size}\0")

  defp digest(hash), do: Base.encode16(:crypto.hash_final(hash), case: :lower)

  defp etag_path(target), do: Path.join(Path.dirname(target), ".#{Path.basename(target)}.etag")

  defp save_etag(target, etag), do: write_file!(etag_path(target), etag)

  # Other processes read the file at the same time, so it changes at once.
  defp write_file!(path, contents) do
    part = "#{path}.#{System.pid()}-#{System.unique_integer([:positive])}.part"
    File.write!(part, contents)
    File.rename!(part, path)
  end

  # A retry gives a new response, so the first chunk of a response truncates the part file.
  defp write(device, file, hash, chunk, {request, response})
       when not is_map_key(response.private, :progress) do
    {:ok, 0} = :file.position(device, 0)
    :ok = :file.truncate(device)

    response =
      response
      |> Req.Response.put_private(:progress, {0, content_length(response)})
      |> Req.Response.put_private(:hash, hash)

    write(device, file, hash, chunk, {request, response})
  end

  defp write(device, file, _hash, {:data, data}, {request, response}) do
    {size, total} = response.private.progress
    IO.binwrite(device, data)
    progress(file, size, size + byte_size(data), total)

    response =
      response
      |> Req.Response.put_private(:progress, {size + byte_size(data), total})
      |> Req.Response.update_private(:hash, nil, &(&1 && :crypto.hash_update(&1, data)))

    {:cont, {request, response}}
  end

  defp content_length(response) do
    case Req.Response.get_header(response, "content-length") do
      [length] when response.status == 200 -> String.to_integer(length)
      _other -> nil
    end
  end

  # Draws the bar when the percent changes. A file in one chunk gets no bar, as in Bumblebee.
  defp progress(file, before, size, total) do
    if Application.get_env(:transmuter, :progress_bar_enabled, true) and
         total not in [nil, size - before] and div(100 * size, total) != div(100 * before, total) do
      done = div(30 * size, total)
      bar = String.duplicate("=", done) <> String.duplicate(" ", 30 - done)
      megabytes = :io_lib.format("~.1f/~.1f MB", [size / 1.0e6, total / 1.0e6])
      IO.write(["\r#{file} |#{bar}| #{div(100 * size, total)}% (", megabytes, ")"])
      if size == total, do: IO.write("\n")
    end
  end

  defp offline?(opts) do
    Keyword.get_lazy(opts, :offline, fn ->
      Enum.any?(["HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"], fn name ->
        String.upcase(System.get_env(name, "")) in ["1", "ON", "YES", "TRUE"]
      end)
    end)
  end

  defp cache_dir(repo, opts) do
    root = opts[:cache_dir] || Path.join([System.user_home!(), ".cache", "transmuter"])

    if opts[:revision],
      do: Path.join(root, "#{repo}@#{revision(opts)}"),
      else: Path.join(root, repo)
  end

  # A branch name can have a slash, such as refs/pr/1.
  defp revision(opts), do: URI.encode(opts[:revision] || "main", &URI.char_unreserved?/1)

  defp commit?(revision), do: is_binary(revision) and revision =~ ~r/\A[0-9a-f]{40}\z/

  defp get(url), do: Req.get(url, req_options(retry: false))

  defp req_options(opts) do
    headers =
      if token = System.get_env("HF_TOKEN"), do: [authorization: "Bearer #{token}"], else: []

    [headers: headers, receive_timeout: 600_000, redirect_log_level: false] ++ opts
  end
end
