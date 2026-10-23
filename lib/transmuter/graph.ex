defmodule Transmuter.Graph do
  @moduledoc """
  Shows the graph of a loaded model with what Transmuter does to it: where `Transmuter.compile/3` breaks the graph and why, which subgraphs `Transmuter.Fusion` replaced, and which values are static.

  For the graph as the exporter wrote it, load the model with `fuse: false` and call `as_graph/2` or `to_html/3` without `:inputs`:

      model = Transmuter.load!({:hf, "Xenova/all-MiniLM-L6-v2", "onnx/model.onnx"}, fuse: false)
      Transmuter.Graph.to_html(model, "minilm.html")

  `Transmuter.load!/2` makes the large `Constant` nodes params, so these values show as inputs without a producer node.

  In Livebook, `as_graph/2` shows an interactive view. It needs the optional `kino` dependency:

      model = Transmuter.load!({:hf, "Xenova/all-MiniLM-L6-v2", "onnx/model.onnx"}, backend: EXLA.Backend)
      inputs = %{"input_ids" => ids, "attention_mask" => mask, "token_type_ids" => types}

      Transmuter.Graph.as_graph(model, inputs: inputs, compile: [compiler: EXLA])

  Outside Livebook, `to_html/3` writes the same view to one HTML file, which a browser opens without a network connection:

      Transmuter.Graph.to_html(model, "minilm.html", inputs: inputs, compile: [compiler: EXLA])

  The view puts the nodes in groups by their name scope, such as `/encoder/layer.0`. Click a group to open it, and click the frame of an open group to close it. Click a node to see its details. The fill of a node gives its segment. A node with static outputs has a dashed border, a node that replaces fused nodes has a thick border, and a break node is red. Open `/encoder/layer.0/attention/self` to see the `Shape`, `Gather`, `Unsqueeze` and `Concat` nodes that make the targets of the `Reshape` nodes. They are static: they run on the host when the model compiles, and the compiled function holds only their results.

  Without `:inputs`, the view is static only: it shows the fused nodes and the nodes that fold on the host from the static values of the model, of its subgraphs and of the `:static_inputs`. It cannot show the graph breaks, and it does not follow static values from node to node, because some of them depend on the shapes of the inputs. Thus it does not mark the `Shape` chains. With `:inputs`, `overlay/2` compiles the model and calls it once, and it shows what that call did.
  """

  @compile {:no_warn_undefined, [Kino.JS, Transmuter.Graph.Kino]}

  alias Transmuter.Graph.Kino, as: GraphKino
  alias Transmuter.GraphBreaks

  @fused_key "transmuter.fused"
  @zero_counts %{nodes: 0, static: 0, fused: 0, breaks: 0, constants: 0}
  # The scripts of the view, in the order that they load. The data of the view gives them to
  # main.js, so as_graph/2 and to_html/3 load the same scripts.
  @scripts [
    "vendor/dagre-0.8.5.min.js",
    "vendor/cytoscape-3.30.2.min.js",
    "vendor/cytoscape-dagre-2.5.0.js",
    "viewer.js"
  ]

  @doc """
  Returns the nodes of the model with what Transmuter does to them.

  Each node is a map with:

    * `:id` - the index of the node in the order of the graph. The nodes of a subgraph come after their parent node
    * `:position` - the index of the node in `model.nodes`, which telemetry events give, or `nil` for a node of a subgraph
    * `:name`, `:op_type`, `:inputs` and `:outputs` - those of the ONNX node
    * `:path` - the parts of the name. The nodes of a subgraph have the path of their parent node and the name of the subgraph attribute first, such as `["optimum::if", "then_branch", ...]`
    * `:reads` - the ids of the nodes whose outputs the node reads, also the outputs of its subgraphs. The inputs and the initializers of a subgraph hide the values of the same names outside it
    * `:fused` - the nodes that `Transmuter.Fusion` replaced with this node, as `"OpType name"`
    * `:placement` - where the node runs, see `Transmuter.placement/2`, or `nil` for a node that did not run
    * `:literals` - the static inputs of a node that runs on the device. A traced segment takes them as literals
    * `:static` - whether the outputs of the node are static values, such as the output of `Shape`. Without `:inputs`, it is `true` for the nodes that fold on the host. It is `nil` for a node that did not run
    * `:shapes` - the shape and the type of each output, as `%{shape: [1, 8, 384], type: "f32"}`, or `nil` for an output that is not a tensor. It is `nil` without `:inputs` and for a node that did not run
    * `:segment` - the index of the segment of the node in `:segments`, or `nil`
    * `:break` - `%{reason: String.t(), after: integer}` for a node where the graph breaks, after the segment of index `:after`, or `nil`

  The result also has `:segments`, a list of `%{start: integer, stop: integer | :end, compiled: boolean, reason: String.t() | nil}`. It is empty without `:inputs`.

  ## Options

    * `:inputs` - the inputs of one call. Then the function compiles the model, calls it once and records what the call did. The default is `nil`, which gives the static view.
    * `:static_inputs` - the static inputs, as for `Transmuter.compile/3`. The default is `%{}`.
    * `:compile` - the options of `Transmuter.compile/3` for the call, such as `[compiler: EXLA]`. The default is `[]`.

  An unknown option raises `ArgumentError`.
  """
  def overlay(model, opts \\ []) do
    opts = Keyword.validate!(opts, [:inputs, static_inputs: %{}, compile: []])
    {inputs, static_inputs} = {opts[:inputs], opts[:static_inputs]}
    nodes = flatten(model.nodes)

    {segments, placement} =
      if inputs do
        {{_outputs, segments}, records} =
          Transmuter.with_record(fn ->
            GraphBreaks.run_once(model, inputs, static_inputs, opts[:compile])
          end)

        {segments, &recorded(&1, records)}
      else
        static = &(Map.has_key?(model.static, &1) or Map.has_key?(static_inputs, &1))
        {[], &static_view(&1, static)}
      end

    nodes =
      Enum.map(nodes, fn entry ->
        {where, flags, static, shapes} = placement.(entry)

        literals =
          if where == :device,
            do: for({name, true} <- Enum.zip(entry.inputs, flags), name != "", do: name),
            else: []

        entry
        |> Map.merge(%{placement: where, literals: literals, static: static, shapes: shapes})
        |> Map.merge(locate(entry, segments))
        |> Map.drop([:node, :top, :key, :sources])
      end)

    %{nodes: nodes, segments: segments}
  end

  @doc """
  Returns the groups of `overlay`, from `overlay/2`, as a tree.

  The groups follow the name scopes of the nodes. A group is a map with `:id`, `:kind` (`"group"`), `:name`, `:scope` (such as `"/encoder/layer.0"`), `:children`, `:segments` (the segments of its nodes) and `:counts`: the numbers of its `:nodes`, of its `:static` and `:fused` nodes, of its `:breaks` and of its hidden `:constants`.

  A node is a map with the fields of `overlay/2`, an `:id` such as `"n12"`, `:kind` (`"node"`) and a `:label`: its name relative to the scope of its group, or `nil` when the exporter generated the name from the op type, such as `Shape_6488`. Its `:reads` are the ids of the nodes that it reads, so a view can draw each edge between the shown groups that hold its ends.

  ## Options

    * `:max_members` - a group with more nodes directly in its scope than this splits them, in the order of the graph, into groups such as `"ops 1-40"`. Exports without name scopes and large flat scopes, such as the encoder of Kokoro, then still give small groups. The default is `40`.
    * `:constants` - whether to keep the `Constant` nodes. Else each group counts the `Constant` nodes in its scope. The default is `false`.
  """
  def tree(%{nodes: nodes}, opts \\ []) do
    opts = Keyword.validate!(opts, max_members: 40, constants: false)
    max_members = opts[:max_members]
    hide? = &(&1.op_type == "Constant" and not opts[:constants])
    group([], Enum.map(nodes, &{&1.path, &1}), max_members, hide?)
  end

  # members are the nodes under scope, each with its path below the scope.
  defp group(scope, members, max_members, hide?) do
    {direct, nested} = Enum.split_with(members, fn {rest, _node} -> length(rest) == 1 end)
    {hidden, shown} = Enum.split_with(direct, fn {_rest, node} -> hide?.(node) end)

    subgroups =
      nested
      |> Enum.group_by(fn {[part | _], _node} -> part end, fn {[_ | rest], node} ->
        {rest, node}
      end)
      |> Enum.map(fn {part, members} -> group(scope ++ [part], members, max_members, hide?) end)

    leaves =
      if length(shown) > max_members do
        shown
        |> Enum.sort_by(fn {_rest, node} -> node.id end)
        |> Enum.chunk_every(max_members)
        |> Enum.with_index()
        |> Enum.map(fn {chunk, i} ->
          first = i * max_members + 1
          name = "ops #{first}-#{first + length(chunk) - 1}"

          group(scope ++ [name], chunk, max_members, hide?)
        end)
      else
        Enum.map(shown, fn {[name], node} -> leaf(node, name) end)
      end

    children = Enum.sort_by(subgroups ++ leaves, & &1.first)

    counts =
      Enum.reduce(children, %{@zero_counts | constants: length(hidden)}, &add_counts(&2, &1))

    ids = Enum.map(children, & &1.first) ++ Enum.map(hidden, &elem(&1, 1).id)

    %{
      id: "g:" <> scope(scope),
      kind: "group",
      name: List.last(scope, "/"),
      scope: scope(scope),
      children: children,
      counts: counts,
      segments: children |> Enum.flat_map(&segments_of/1) |> Enum.uniq() |> Enum.sort(),
      first: Enum.min(ids)
    }
  end

  defp leaf(node, name) do
    node
    |> Map.drop([:path])
    |> Map.merge(%{
      id: "n#{node.id}",
      kind: "node",
      label: if(not generated_name?(node.op_type, name), do: name),
      reads: Enum.map(node.reads, &"n#{&1}"),
      first: node.id
    })
  end

  defp add_counts(counts, %{kind: "group", counts: inner}),
    do: Map.merge(counts, inner, fn _key, a, b -> a + b end)

  defp add_counts(counts, node) do
    %{
      counts
      | nodes: counts.nodes + 1,
        static: counts.static + if(node.static, do: 1, else: 0),
        fused: counts.fused + if(node.fused != [], do: 1, else: 0),
        breaks: counts.breaks + if(node.break && node.position, do: 1, else: 0)
    }
  end

  @doc false
  # Whether an exporter generated the name from the op type, such as Shape or Shape_6488.
  def generated_name?(op_type, name), do: name =~ ~r/^#{Regex.escape(op_type)}(_\d+)?$/

  defp segments_of(%{kind: "group", segments: segments}), do: segments
  defp segments_of(%{segment: nil}), do: []
  defp segments_of(%{segment: segment}), do: [segment]

  defp scope(path), do: "/" <> Enum.join(path, "/")

  @doc """
  Shows the graph of the model in Livebook, as a `Kino.JS` view.

  It needs the optional `kino` dependency. The options of `tree/2` go to it, and the other options go to `overlay/2`, which raises on an unknown option. The view loads no file from the network.
  """
  def as_graph(model, opts \\ []) do
    unless Code.ensure_loaded?(Kino.JS) do
      raise ArgumentError,
            ~s(Transmuter.Graph.as_graph/2 needs the optional kino dependency, add {:kino, "~> 0.14"} to your deps)
    end

    GraphKino.new(data(model, opts))
  end

  @doc """
  Writes the view of `as_graph/2` to one HTML file at `path`, with the scripts and the data in the file. It needs no Kino and no network connection.

  The options are those of `as_graph/2`.
  """
  def to_html(model, path, opts \\ []), do: File.write!(path, html(model, opts))

  @doc false
  # The page of to_html/3.
  def html(model, opts) do
    data = data(model, opts)
    scripts = Enum.map(data.scripts, &File.read!(asset(&1)))
    json = data |> JSON.encode!() |> String.replace("</", "<\\/")

    """
    <!doctype html>
    <html>
    <head>
    <meta charset="utf-8">
    <title>#{escape(title(model))}</title>
    <style>
    html, body { margin: 0; height: 100%; }
    #{File.read!(asset("viewer.css"))}
    </style>
    </head>
    <body>
    <div id="graph" style="height: 100vh"></div>
    #{Enum.map_join(scripts, "\n", &"<script>\n#{&1}\n</script>")}
    <script id="data" type="application/json">#{json}</script>
    <script>
    TransmuterGraphView.render(document.getElementById("graph"), JSON.parse(document.getElementById("data").textContent));
    </script>
    </body>
    </html>
    """
  end

  @doc false
  def asset(name), do: Path.join([:code.priv_dir(:transmuter), "graph_view", name])

  @doc false
  def scripts, do: @scripts

  defp data(model, opts) do
    {tree_opts, opts} = Keyword.split(opts, [:max_members, :constants])
    overlay = overlay(model, opts)

    segments =
      Enum.with_index(overlay.segments, fn segment, index ->
        stop = if segment.stop == :end, do: "end", else: Enum.at(model.nodes, segment.stop).name
        Map.merge(segment, %{index: index, stop: stop})
      end)

    %{title: title(model), scripts: @scripts, segments: segments, tree: tree(overlay, tree_opts)}
  end

  defp title(model) do
    case model[:source] do
      {:hf, repo, file} -> "#{repo} #{file}"
      path when is_binary(path) -> Path.basename(path)
      _other -> "model"
    end
  end

  defp escape(text),
    do:
      String.replace(text, ["&", "<", ">"], &%{"&" => "&amp;", "<" => "&lt;", ">" => "&gt;"}[&1])

  # The nodes of the graph and of its subgraphs, each node before the nodes of its subgraphs.
  # A name resolves in env to the node that computes it, or to an input or an initializer of an
  # enclosing subgraph, which hides the values of the same name outside it. Names that env does
  # not hold are values of the main graph.
  defp flatten(nodes) do
    {entries, _count, _env} = flatten(nodes, %{prefix: [], path: [], top: nil, env: %{}}, 0)
    entries
  end

  defp flatten(nodes, scope, count) do
    {entries, count, env} =
      nodes
      |> Enum.with_index()
      |> Enum.reduce({[], count, scope.env}, fn {node, index}, {entries, count, env} ->
        id = count
        path = scope.prefix ++ name_path(node)
        top = scope.top || index

        {children, count, outputs} =
          for a <- node.attribute,
              graph <- List.wrap(a.g) ++ a.graphs,
              reduce: {[], count + 1, []} do
            {children, count, outputs} ->
              inner = %{
                prefix: path ++ [a.name],
                path: scope.path ++ [{node.output, a.name}],
                top: top,
                env: Map.merge(env, locals(graph))
              }

              {inner_entries, count, inner_env} = flatten(graph.node, inner, count)

              {children ++ inner_entries, count,
               outputs ++ Enum.map(graph.output, &inner_env[&1.name])}
          end

        sources = Enum.map(node.input, &env[&1])

        entry = %{
          id: id,
          position: if(scope.top == nil, do: index),
          top: top,
          key: Transmuter.record_key(scope.path, node),
          name: node.name,
          op_type: node.op_type,
          inputs: node.input,
          outputs: node.output,
          path: path,
          sources: sources,
          reads: for({:node, read} <- sources ++ outputs, uniq: true, do: read),
          fused: fused(node),
          node: node
        }

        env = Enum.reduce(node.output, env, &Map.put(&2, &1, {:node, id}))
        {Enum.reverse(children, [entry | entries]), count, env}
      end)

    {Enum.reverse(entries), count, env}
  end

  # The inputs of a subgraph are not static. Its initializers are static, because a run puts
  # them on the host.
  defp locals(graph) do
    inputs = Map.new(graph.input, &{&1.name, {:local, false}})
    Map.merge(inputs, Map.new(graph.initializer, &{&1.name, {:local, true}}))
  end

  defp name_path(node) do
    case String.split(node.name || "", "/", trim: true) do
      [] -> [hd(node.output ++ [node.op_type])]
      parts -> parts
    end
  end

  defp fused(node) do
    case Enum.find(node.metadata_props, &(&1.key == @fused_key)) do
      nil -> []
      entry -> String.split(entry.value, ", ")
    end
  end

  # Without inputs, a value is static only when the model, the static inputs or an enclosing
  # subgraph gives it.
  defp static_view(entry, static) do
    flags =
      Enum.zip_with(entry.inputs, entry.sources, fn
        "", _source -> true
        _name, {:local, static?} -> static?
        _name, {:node, _id} -> false
        name, nil -> static.(name)
      end)

    placement = Transmuter.placement(entry.node, flags)
    {placement, flags, placement == :host, nil}
  end

  # A node of an eager segment takes no literals.
  defp recorded(entry, records) do
    case records[entry.key] do
      {placement, flags, true, static, shapes} -> {placement, flags, static, shapes}
      {placement, _flags, false, static, shapes} -> {placement, [], static, shapes}
      nil -> {nil, [], nil, nil}
    end
  end

  # The segment of a top-level node, or the break at it. A node of a subgraph has the
  # location of its top-level node: the nodes in the subgraph of a break node ran in the break.
  defp locate(%{top: top}, segments) do
    case Enum.find_index(segments, &(&1.stop == top)) do
      nil ->
        segment =
          Enum.find_index(segments, &(&1.start <= top and (&1.stop == :end or top < &1.stop)))

        %{segment: segment, break: nil}

      index ->
        %{segment: nil, break: %{reason: Enum.at(segments, index).reason, after: index}}
    end
  end
end

if Code.ensure_loaded?(Kino.JS) do
  defmodule Transmuter.Graph.Kino do
    @moduledoc false
    use Kino.JS, assets_path: "priv/graph_view"

    def new(data), do: Kino.JS.new(__MODULE__, data)
  end
end
