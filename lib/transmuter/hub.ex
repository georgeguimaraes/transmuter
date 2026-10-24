defmodule Transmuter.Hub do
  @moduledoc """
  Reads the files of a model repository on the Hugging Face Hub or in a local directory.

  A source is `{:hf, repo}` or the path of a local directory. Hub files download into the cache directory and come from there later. When the metadata of the repository gives a new ETag for a file, the file downloads again. Each download must match its ETag. The cache directory is `~/.cache/transmuter`, or the `:cache_dir` option. An error status of the Hub, such as 401 for a gated repository without a token, raises `Transmuter.HubError`.

  ## Environment

  This module reads these environment variables, as Python `huggingface_hub` does:

  * `HF_TOKEN` (or the older `HUGGING_FACE_HUB_TOKEN`) - the token for private and gated repositories. Without it, the token comes from the file that `hf auth login` writes: `HF_TOKEN_PATH`, else `token` in `HF_HOME`, else `huggingface/token` in `XDG_CACHE_HOME`, else `~/.cache/huggingface/token`. A token from the browser login of `hf auth login` expires. Python `huggingface_hub` refreshes it in the token file when less than one day remains, on each command that reads the token, such as `hf auth whoami`. Transmuter only reads the file, so when the Hub gives status 401 for such a token, run `hf auth whoami`.

  * `HTTPS_PROXY` (and `HTTP_PROXY` for an `http` URL), else `ALL_PROXY` - the HTTP or HTTPS proxy, such as `http://user:password@proxy:3128`. A lowercase name wins over the uppercase name.

  * `NO_PROXY` - hosts that connect without the proxy, separated by commas. A host that ends with an entry connects directly, and `*` matches all hosts. The Hub redirects downloads to other hosts, such as `cas-bridge.xethub.hf.co`, and each host gets its own rule.

  * `HF_HUB_OFFLINE` and `TRANSFORMERS_OFFLINE` - see the `:offline` option.

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
        metadata = %{source: {:hf, repo}, revision: revision(opts)}

        :telemetry.span([:transmuter, :hub, :info], metadata, fn ->
          {info, result} = request_info(repo, path, saved?, opts)
          {info, Map.put(metadata, :result, result)}
        end)
    end
  end

  def info(dir, _opts) when is_binary(dir), do: nil

  # With saved metadata, an error gives the saved metadata at once. Without it, Req retries
  # a transient error.
  defp request_info(repo, path, saved?, opts) do
    case Req.get(
           request(
             url:
               "https://huggingface.co/api/models/#{repo}/revision/#{url_revision(opts)}?blobs=true",
             retry: if(saved?, do: false, else: :safe_transient),
             receive_timeout: 30_000
           )
         ) do
      {:ok, %{status: 200, body: info}} ->
        File.mkdir_p!(Path.dirname(path))
        write_file!(path, JSON.encode!(info))
        {info, :fetched}

      _error when saved? ->
        {path |> File.read!() |> JSON.decode!(), :saved}

      {:ok, %{headers: %{"x-error-code" => ["RevisionNotFound"]}}} ->
        raise ArgumentError, "#{repo} has no revision #{opts[:revision]}"

      {:ok, %{status: status, body: body}} ->
        raise Transmuter.HubError, repo: repo, status: status, details: inspect(body)

      {:error, error} ->
        raise ArgumentError,
              "could not reach the Hugging Face Hub for #{repo}: #{Exception.message(error)}"
    end
  end

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
    metadata = %{source: {:hf, repo, file}, revision: revision(opts)}

    cond do
      check = File.exists?(target) && check(target, etag, opts) ->
        bytes = sibling["size"] || File.stat!(target).size
        hit = Map.put(metadata, :check, check)
        :telemetry.execute([:transmuter, :hub, :hit], %{count: 1, bytes: bytes}, hit)

      offline?(opts) ->
        raise ArgumentError, "#{file} of #{repo} is not in the cache, and the Hub is offline"

      true ->
        :telemetry.span([:transmuter, :hub, :download], metadata, fn ->
          download_file!(repo, commit, sibling, target, etag)
          {:ok, %{bytes: File.stat!(target).size}, metadata}
        end)
    end
  end

  defp download_file!(repo, commit, sibling, target, etag) do
    file = sibling["rfilename"]
    hash = etag && hash_init(etag, sibling["size"])
    File.mkdir_p!(Path.dirname(target))
    part = "#{target}.#{System.pid()}-#{System.unique_integer([:positive])}.part"

    try do
      response =
        File.open!(part, [:write, :binary], fn device ->
          Req.get!(
            request(
              url: "https://huggingface.co/#{repo}/resolve/#{commit}/#{file}",
              into: &write(device, file, hash, &1, &2)
            )
          )
        end)

      if response.status != 200,
        do: raise(Transmuter.HubError, repo: repo, status: response.status, details: file)

      if etag && digest(Req.Response.get_private(response, :hash, hash)) != etag,
        do: raise(ArgumentError, "the download of #{file} of #{repo} does not match its ETag")

      File.rename!(part, target)
      if etag, do: save_etag(target, etag)
    after
      File.rm(part)
    end
  end

  # How a file in the cache is current, or nil when it must download again. A file without
  # a saved ETag, such as a file of an earlier version of Transmuter, gets the ETag of its
  # contents. A file whose metadata has no ETag is used without a check.
  defp check(target, etag, opts) do
    cond do
      offline?(opts) -> :offline
      etag == nil -> :none
      true -> check_etag(target, etag)
    end
  end

  defp check_etag(target, etag) do
    case File.read(etag_path(target)) do
      {:ok, ^etag} ->
        :etag

      {:ok, _other} ->
        nil

      {:error, _reason} ->
        hash =
          target
          |> File.stream!(1_048_576)
          |> Enum.reduce(hash_init(etag, File.stat!(target).size), &:crypto.hash_update(&2, &1))

        if digest(hash) == etag and save_etag(target, etag) == :ok, do: :hash
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
      do: Path.join(root, "#{repo}@#{url_revision(opts)}"),
      else: Path.join(root, repo)
  end

  # A branch name can have a slash, such as refs/pr/1.
  defp revision(opts), do: opts[:revision] || "main"

  defp url_revision(opts), do: URI.encode(revision(opts), &URI.char_unreserved?/1)

  defp commit?(revision), do: is_binary(revision) and revision =~ ~r/\A[0-9a-f]{40}\z/

  defp request(opts) do
    headers = if token = token(), do: [authorization: "Bearer #{token}"], else: []

    [headers: headers, receive_timeout: 600_000, redirect_log_level: false]
    |> Keyword.merge(opts)
    |> Req.new()
    |> Req.Request.append_request_steps(proxy: &proxy/1)
  end

  # As in Python huggingface_hub: HF_TOKEN or HUGGING_FACE_HUB_TOKEN, else the file that
  # `hf auth login` writes.
  defp token do
    env = Enum.find_value(["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"], &non_empty(System.get_env(&1)))

    clean_token(env) ||
      case File.read(token_path()) do
        {:ok, token} -> clean_token(token)
        {:error, _reason} -> nil
      end
  end

  defp token_path do
    home =
      System.get_env("HF_HOME") ||
        Path.join(
          System.get_env("XDG_CACHE_HOME", Path.join(System.user_home!(), ".cache")),
          "huggingface"
        )

    Path.expand(System.get_env("HF_TOKEN_PATH") || Path.join(home, "token"))
  end

  defp clean_token(nil), do: nil

  defp clean_token(token),
    do: token |> String.replace(["\r", "\n"], "") |> String.trim() |> non_empty()

  # As in Python requests, the proxy of a URL comes from the <scheme>_proxy variable, else
  # from all_proxy. A host that ends with an entry of no_proxy connects directly, and the
  # entry * matches all hosts. The step runs again for each redirect, because a redirect can
  # go to another host.
  defp proxy(request) do
    request = Req.Request.delete_option(request, :connect_options)
    %URI{scheme: scheme, host: host} = request.url
    proxy = proxy_env("#{scheme}_proxy") || proxy_env("all_proxy")

    if proxy && not bypass?(host) do
      uri = URI.parse(if proxy =~ "://", do: proxy, else: "http://" <> proxy)

      scheme =
        case uri.scheme do
          "http" ->
            :http

          "https" ->
            :https

          other ->
            raise ArgumentError, "Transmuter supports only HTTP and HTTPS proxies, got #{other}"
        end

      headers =
        if uri.userinfo,
          do: [{"proxy-authorization", "Basic " <> Base.encode64(URI.decode(uri.userinfo))}],
          else: []

      Req.Request.put_option(request, :connect_options,
        proxy: {scheme, uri.host, uri.port, []},
        proxy_headers: headers
      )
    else
      request
    end
  end

  defp bypass?(host) do
    (proxy_env("no_proxy") || "")
    |> String.replace(" ", "")
    |> String.split(",", trim: true)
    |> Enum.any?(&(&1 == "*" or String.ends_with?(host, String.trim_leading(&1, "."))))
  end

  # The lowercase variable wins over the uppercase one, also when it is empty.
  defp proxy_env(name) do
    case System.get_env(name) do
      nil -> non_empty(System.get_env(String.upcase(name)))
      value -> non_empty(value)
    end
  end

  defp non_empty(value) when value in [nil, ""], do: nil
  defp non_empty(value), do: value
end
