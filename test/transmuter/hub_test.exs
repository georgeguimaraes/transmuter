defmodule Transmuter.HubTest do
  use ExUnit.Case, async: true

  @moduletag :hub
  @moduletag :tmp_dir

  test "an error from the Hub raises and is not saved", %{tmp_dir: dir} do
    # The Hub gives 404 with a token and 401 without one.
    assert_raise Transmuter.HubError, ~r/gave status 40[14]/, fn ->
      Transmuter.Hub.files({:hf, "Xenova/no-such-repository"}, cache_dir: dir)
    end

    refute File.exists?(Path.join(dir, "Xenova/no-such-repository/.info.json"))
  end

  test "offline mode reads only the cache", %{tmp_dir: dir} do
    source = {:hf, "Xenova/all-MiniLM-L6-v2"}
    online = Transmuter.Repo.load!(source, cache_dir: dir)
    offline = Transmuter.Repo.load!(source, cache_dir: dir, offline: true)
    assert offline.config == online.config and offline.files == online.files

    assert_raise ArgumentError,
                 ~r/onnx\/model.onnx of .* not in the cache, and the Hub is offline/,
                 fn ->
                   Transmuter.Hub.fetch!(source, "onnx/model.onnx", cache_dir: dir, offline: true)
                 end

    assert_raise ArgumentError, ~r/Xenova\/bert-base-uncased is not in the cache/, fn ->
      Transmuter.Hub.files({:hf, "Xenova/bert-base-uncased"}, cache_dir: dir, offline: true)
    end
  end

  test "a revision gives the files of that commit, branch or tag", %{tmp_dir: dir} do
    source = {:hf, "Xenova/all-MiniLM-L6-v2"}
    files = &Transmuter.Hub.files(source, cache_dir: dir, revision: &1)

    # The commit 57cbdab uploaded model_int8.onnx to the root, and cb3d680 moved it to onnx/.
    assert "model_int8.onnx" in files.("57cbdab1181b7eba8170805260bd39e733b0e25d")
    refute "onnx/model_int8.onnx" in files.("57cbdab1181b7eba8170805260bd39e733b0e25d")
    assert "onnx/model_int8.onnx" in files.(nil)
    assert "onnx/model_fp16.onnx" in files.("refs/pr/1")

    readme = &Transmuter.Hub.fetch!(source, "README.md", cache_dir: dir, revision: &1)
    old = readme.("a162ec1733c9eead894ebbf7cf40d15ecf3f3f65")

    assert Path.dirname(old) ==
             Path.join(dir, "Xenova/all-MiniLM-L6-v2@a162ec1733c9eead894ebbf7cf40d15ecf3f3f65")

    assert File.read!(old) != File.read!(readme.(nil))

    assert_raise ArgumentError, ~r/has no revision no-such-revision/, fn ->
      files.("no-such-revision")
    end
  end

  # config.json is a Git blob, and onnx/model.onnx is an LFS file, so each case checks both
  # kinds of ETag.
  test "a file downloads again when its ETag changes", %{tmp_dir: dir} do
    source = {:hf, "hf-internal-testing/tiny-random-bert"}
    old = {{2000, 1, 1}, {0, 0, 0}}

    for file <- ["config.json", "onnx/model.onnx"] do
      path = Transmuter.Hub.fetch!(source, file, cache_dir: dir)
      etag_path = Path.join(Path.dirname(path), ".#{Path.basename(path)}.etag")
      contents = File.read!(path)
      etag = File.read!(etag_path)

      cases = [
        # A file of an earlier version of Transmuter has no saved ETag. Its contents match.
        {"no saved ETag", fn -> File.rm!(etag_path) end, :kept},
        {"no saved ETag, other contents",
         fn -> File.rm!(etag_path) == :ok and File.write!(path, "x") end, :downloaded},
        {"a new ETag on the Hub",
         fn -> File.write!(etag_path, String.duplicate("0", byte_size(etag))) end, :downloaded}
      ]

      for {name, change, result} <- cases do
        File.touch!(path, old)
        change.()
        assert Transmuter.Hub.fetch!(source, file, cache_dir: dir) == path

        assert {File.read!(path) == contents, File.read!(etag_path),
                File.stat!(path).mtime == old} ==
                 {true, etag, result == :kept},
               "#{file}: #{name}"
      end
    end

    info = Path.join(dir, "hf-internal-testing/tiny-random-bert/.info.json")

    File.write!(
      info,
      String.replace(File.read!(info), etag_of(info, "config.json"), String.duplicate("0", 40))
    )

    File.rm!(Path.join(dir, "hf-internal-testing/tiny-random-bert/config.json"))

    assert_raise ArgumentError, ~r/the download of config.json .* does not match its ETag/, fn ->
      Transmuter.Hub.fetch!(source, "config.json", cache_dir: dir)
    end

    refute File.exists?(Path.join(dir, "hf-internal-testing/tiny-random-bert/config.json"))
  end

  defp etag_of(info, file) do
    info
    |> File.read!()
    |> JSON.decode!()
    |> Map.fetch!("siblings")
    |> Enum.find(&(&1["rfilename"] == file))
    |> Map.fetch!("blobId")
  end

  test "concurrent downloads of one file all succeed", %{tmp_dir: dir} do
    paths =
      1..8
      |> Task.async_stream(
        fn _ ->
          Transmuter.Hub.fetch!({:hf, "Xenova/all-MiniLM-L6-v2"}, "config.json", cache_dir: dir)
        end,
        timeout: 60_000
      )
      |> Enum.map(fn {:ok, path} -> path end)

    assert [path] = Enum.uniq(paths)
    assert %{"model_type" => "bert"} = path |> File.read!() |> JSON.decode!()
    assert Path.wildcard(Path.join(dir, "**/*.part")) == []
  end
end

defmodule Transmuter.HubEnvTest do
  # The tests set environment variables and the default options of Req, which all processes
  # read, so they run when no other test runs.
  use ExUnit.Case, async: false

  @moduletag :tmp_dir

  @vars ~w(HF_TOKEN HUGGING_FACE_HUB_TOKEN HF_TOKEN_PATH HF_HOME XDG_CACHE_HOME
           https_proxy HTTPS_PROXY all_proxy ALL_PROXY no_proxy NO_PROXY)

  setup %{tmp_dir: dir} do
    env = Map.new(@vars, &{&1, System.get_env(&1)})
    defaults = Req.default_options()

    on_exit(fn ->
      Req.default_options(defaults)

      Enum.each(env, fn
        {name, nil} -> System.delete_env(name)
        {name, value} -> System.put_env(name, value)
      end)
    end)

    Enum.each(@vars, &System.delete_env/1)
    File.mkdir_p!(Path.join(dir, "hf"))
    System.put_env("HF_HOME", Path.join(dir, "hf"))
    Process.register(self(), __MODULE__)
    Req.default_options(adapter: __MODULE__.Adapter)
    :ok
  end

  defmodule Adapter do
    # Gives the metadata and the file that the test puts in its process, else the metadata
    # of an empty repository, with the status that the test puts in its process. So no
    # request leaves the test. A download writes the file through the into function.
    def run(request) do
      send(Transmuter.HubEnvTest, {:request, request})
      status = Process.get(:hub_status, 200)
      info = Process.get(:hub_info, %{"sha" => "0", "siblings" => []})

      if is_function(request.into, 2) do
        response = Req.Response.new(status: status, body: "")
        {:cont, result} = request.into.({:data, Process.get(:hub_file)}, {request, response})
        result
      else
        {request, Req.Response.new(status: status, body: info)}
      end
    end
  end

  # Handlers are global, so the handler forwards only the events of the test process.
  defp attach(events) do
    id = make_ref()
    :telemetry.attach_many(id, events, &__MODULE__.forward/4, self())
    on_exit(fn -> :telemetry.detach(id) end)
  end

  def forward(event, measurements, metadata, pid) do
    if self() == pid, do: send(pid, {event, measurements, metadata})
  end

  defp request(env, dir) do
    System.put_env(env)
    Transmuter.Hub.info({:hf, "a/b"}, cache_dir: dir, refresh: true)
    Enum.each(env, fn {name, _value} -> System.delete_env(name) end)
    assert_received {:request, request}
    request
  end

  test "the token comes from HF_TOKEN, else from the token file of hf auth login", %{
    tmp_dir: dir
  } do
    File.write!(Path.join(dir, "hf/token"), " from-home\n")
    File.write!(Path.join(dir, "other"), "from-path")

    for {env, expected} <- [
          {%{}, ["Bearer from-home"]},
          {%{"HF_TOKEN_PATH" => Path.join(dir, "other")}, ["Bearer from-path"]},
          {%{"HF_TOKEN" => "from-env"}, ["Bearer from-env"]},
          {%{"HF_HOME" => Path.join(dir, "none")}, []}
        ] do
      assert Req.Request.get_header(request(env, dir), "authorization") == expected,
             inspect(env)
    end
  end

  test "the proxy comes from HTTPS_PROXY, except for the hosts of NO_PROXY", %{tmp_dir: dir} do
    proxy = %{"HTTPS_PROXY" => "http://user:secret@proxy.test:3128"}

    auth = [
      proxy: {:http, "proxy.test", 3128, []},
      proxy_headers: [{"proxy-authorization", "Basic " <> Base.encode64("user:secret")}]
    ]

    for {env, expected} <- [
          {proxy, auth},
          {Map.put(proxy, "NO_PROXY", "example.com, .huggingface.co"), nil},
          {Map.put(proxy, "NO_PROXY", "example.com"), auth},
          {%{"all_proxy" => "proxy.test:8080"},
           [proxy: {:http, "proxy.test", 8080, []}, proxy_headers: []]},
          # The lowercase variable wins, also when it is empty.
          {Map.put(proxy, "https_proxy", ""), nil}
        ] do
      assert request(env, dir).options[:connect_options] == expected, inspect(env)
    end
  end

  # A file downloads once, after a request for the metadata. Later fetches read the saved
  # metadata and find the file in the cache, and each event tells how the cache checked it.
  # A new ETag on the Hub downloads it again.
  test "a Hub file emits a download, then a hit with its check", %{tmp_dir: dir} do
    attach([
      [:transmuter, :hub, :hit],
      [:transmuter, :hub, :download, :stop],
      [:transmuter, :hub, :info, :stop]
    ])

    body = ~s({"model_type": "bert"})
    bytes = byte_size(body)
    blob = Base.encode16(:crypto.hash(:sha, "blob #{bytes}\0" <> body), case: :lower)
    sibling = %{"rfilename" => "config.json", "blobId" => blob, "size" => bytes}
    Process.put(:hub_info, %{"sha" => "0", "siblings" => [sibling]})
    Process.put(:hub_file, body)
    fetch = &Transmuter.Hub.fetch!({:hf, "a/b"}, "config.json", [cache_dir: dir] ++ &1)
    path = fetch.([])
    assert File.read!(path) == body
    source = {:hf, "a/b", "config.json"}

    assert_received {[:transmuter, :hub, :info, :stop], _measurements,
                     %{source: {:hf, "a/b"}, result: :fetched}}

    assert_received {[:transmuter, :hub, :download, :stop], %{bytes: ^bytes},
                     %{source: ^source, revision: "main"}}

    etag_path = Path.join(dir, "a/b/.config.json.etag")
    new_etag = fn -> File.write!(etag_path, String.duplicate("0", 40)) end

    # Metadata saved before blobs=true has no "size", and a file can have no ETag. The size
    # then comes from the file. The SHA-256 of an LFS file does not need the size.
    save = fn sibling ->
      info = %{"sha" => "0", "siblings" => [sibling]}
      File.write!(Path.join(dir, "a/b/.info.json"), JSON.encode!(info))
    end

    sha256 = Base.encode16(:crypto.hash(:sha256, body), case: :lower)
    lfs = %{"rfilename" => "config.json", "lfs" => %{"sha256" => sha256}}

    for {change, opts, expected} <- [
          {fn -> :ok end, [], {:hit, :etag}},
          {fn -> File.rm!(etag_path) end, [], {:hit, :hash}},
          {fn -> :ok end, [offline: true], {:hit, :offline}},
          {new_etag, [], :download},
          {fn -> save.(Map.delete(sibling, "size")) end, [], {:hit, :etag}},
          {fn -> save.(lfs) end, [], :download},
          {fn -> save.(%{"rfilename" => "config.json"}) end, [], {:hit, :none}}
        ] do
      change.()
      assert fetch.(opts) == path

      case expected do
        {:hit, check} ->
          assert_received {[:transmuter, :hub, :hit], %{count: 1, bytes: ^bytes},
                           %{source: ^source, check: ^check}}

        :download ->
          assert_received {[:transmuter, :hub, :download, :stop], %{bytes: ^bytes}, _metadata}
      end

      refute_received {[:transmuter, :hub | _event], _measurements, _metadata}
    end
  end

  test "an error of the Hub API gives the saved metadata at once, without retries", %{
    tmp_dir: dir
  } do
    info = Transmuter.Hub.info({:hf, "a/b"}, cache_dir: dir)
    assert_received {:request, _request}
    attach([[:transmuter, :hub, :info, :stop]])
    Process.put(:hub_status, 503)
    assert Transmuter.Hub.info({:hf, "a/b"}, cache_dir: dir, refresh: true) == info
    assert_received {:request, _request}
    refute_received {:request, _request}

    assert_received {[:transmuter, :hub, :info, :stop], _measurements,
                     %{source: {:hf, "a/b"}, result: :saved}}
  end
end
