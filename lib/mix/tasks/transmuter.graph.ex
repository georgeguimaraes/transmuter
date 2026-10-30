defmodule Mix.Tasks.Transmuter.Graph do
  @shortdoc "Writes the graph view of an ONNX model to an HTML file"

  @moduledoc """
  Writes the view of `Transmuter.Graph.to_html/3` for an ONNX model to an HTML file.

      mix transmuter.graph REPO_OR_DIR [--file onnx/model.onnx] [--revision REV] [--out PATH] [--open] [--plain | --run] [--dim NAME=SIZE ...] [--backend exla|emlx|binary]

  `REPO_OR_DIR` is a repository on the Hugging Face Hub, such as `Xenova/all-MiniLM-L6-v2`, or a local directory, as in `Transmuter.Hub`.

  Without `--run`, the task writes the static view. It does not call the model.

  With `--plain`, the task loads the model with `fuse: false` and does not call it, so the view shows the graph as the exporter wrote it. `Transmuter.load!/2` makes the large `Constant` nodes params, so these values show as inputs without a producer node. `--plain` and `--run` do not go together.

  With `--run`, the task calls the model once with inputs of ones. A one, unlike a zero, is not a degenerate value for a divisor or a scale, such as the `speed` input of Kokoro. Each input gets the type and the shape that the graph declares. A string input gets `"a"` for each element, and an input that is not a tensor, such as a sequence, gets an empty optional. A symbolic dimension gets its size from `--dim`, else 1. When the call fails, use `--dim` with the sizes that the model needs, or call `Transmuter.Graph.as_graph/2` in Livebook with real inputs.

  ## Options

    * `--file` - the ONNX file in the repository. The default is `onnx/model.onnx`.
    * `--revision` - the revision of a Hub repository. The default is `main`.
    * `--out` - the path of the HTML file. The default is the name of the repository with `.html`, in the current directory.
    * `--open` - opens the file with `open` on macOS or `xdg-open` on Linux. Without it, the task prints the path.
    * `--plain` - shows the graph as the exporter wrote it, without fusion and without a call.
    * `--run` - calls the model once with inputs of ones, so the view shows the segments.
    * `--dim` - the size of a symbolic dimension, such as `--dim sequence_length=16`. Give it once for each dimension.
    * `--backend` - `exla` (the EXLA CPU), `emlx` (the EMLX CPU) or `binary` (`Nx.BinaryBackend`). The default is `exla` when EXLA is available, else `binary`.
  """

  use Mix.Task

  @switches [
    file: :string,
    revision: :string,
    out: :string,
    open: :boolean,
    run: :boolean,
    plain: :boolean,
    dim: :keep,
    backend: :string
  ]

  @impl true
  def run(args) do
    {opts, positional} = OptionParser.parse!(args, strict: @switches)

    source =
      case positional do
        [source] -> source
        _other -> Mix.raise("Give one repository or directory, see mix help transmuter.graph")
      end

    if opts[:plain] && opts[:run] do
      Mix.raise("--plain shows the graph without a call, so it does not go together with --run")
    end

    Mix.Task.run("app.start")

    file = Keyword.get(opts, :file, "onnx/model.onnx")
    hub_opts = if revision = opts[:revision], do: [revision: revision], else: []
    {backend, compile_opts} = backend(opts[:backend] || default_backend())

    model_source = if File.dir?(source), do: Path.join(source, file), else: {:hf, source, file}
    model = Transmuter.load!(model_source, [backend: backend, fuse: !opts[:plain]] ++ hub_opts)
    out = opts[:out] || Path.basename(source) <> ".html"

    html =
      if opts[:run] do
        types = Map.take(model.input_types, model.inputs)
        inputs = dummy_inputs(types, parse_dims(Keyword.get_values(opts, :dim)), backend)

        try do
          Transmuter.Graph.html(model, inputs: inputs, compile: compile_opts)
        rescue
          e ->
            Mix.shell().error("""
            The call with inputs of ones failed. The inputs had the shapes: #{describe(inputs)}
            Give the sizes that the model needs with --dim, or call Transmuter.Graph.as_graph/2 in Livebook with real inputs.
            """)

            reraise e, __STACKTRACE__
        end
      else
        Transmuter.Graph.html(model, [])
      end

    File.write!(out, html)
    if opts[:open], do: open(out), else: Mix.shell().info(Path.expand(out))
  end

  defp describe(inputs) do
    Enum.map_join(inputs, ", ", fn
      {name, t = %Nx.Tensor{}} -> "#{name} #{inspect(Nx.shape(t))}"
      {name, {:strings, shape, _values}} -> "#{name} #{inspect(shape)} strings"
      {name, :none} -> "#{name} empty"
    end)
  end

  defp default_backend, do: if(Code.ensure_loaded?(EXLA), do: "exla", else: "binary")

  defp backend("exla"), do: with_compiler({EXLA.Backend, client: :host})
  defp backend("emlx"), do: with_compiler({EMLX.Backend, device: :cpu})
  defp backend("binary"), do: with_compiler(Nx.BinaryBackend)
  defp backend(other), do: Mix.raise("--backend is exla, emlx or binary, not #{other}")

  defp with_compiler(backend), do: {backend, Transmuter.compiler_options(backend)}

  @doc false
  # Parses each --dim NAME=SIZE.
  def parse_dims(values) do
    Map.new(values, fn value ->
      with [name, size] <- String.split(value, "=", parts: 2),
           {size, ""} when size > 0 <- Integer.parse(size) do
        {name, size}
      else
        _ -> Mix.raise("--dim takes NAME=SIZE with a positive size, not #{value}")
      end
    end)
  end

  @doc false
  # Inputs of ones for the declared types of model.input_types: a tensor of ones, "a" for each
  # element of a string tensor, and an empty optional for an input that is not a tensor. A
  # symbolic dimension takes its size from dims, else 1.
  def dummy_inputs(types, dims, backend) do
    Map.new(types, fn
      {name, nil} ->
        {name, :none}

      {name, {type, declared}} ->
        shape = List.to_tuple(Enum.map(declared, &Transmuter.dim_size(&1, Map.get(dims, &1, 1))))

        if type == :string,
          do: {name, {:strings, shape, List.duplicate("a", Tuple.product(shape))}},
          else: {name, Nx.broadcast(Nx.tensor(1, type: type, backend: backend), shape)}
    end)
  end

  defp open(path) do
    command = if match?({:unix, :darwin}, :os.type()), do: "open", else: "xdg-open"
    System.cmd(command, [path])
  end
end
