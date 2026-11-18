defmodule Transmuter.Generation.MergedDecoder do
  @moduledoc """
  Moves a decoder without `position_ids` that takes its positions from the length of its past, such as a merged decoder of optimum or Apertus, from a cache of exact length (`:grow`) to a buffer, see "Merged decoders" in `Transmuter.Generation`.

  Such a decoder takes the length L of its cache from the shape of the past. With a buffer of C rows, L is C, which is right for the shapes and the causal masks: the keys are the buffer and the new rows, and query i of a step sees the keys up to C + i. Only the positions need the valid rows v. The rewrite changes only the branch with cache of the `If` on `use_cache_branch`, because the branches of the exports have nodes with the same names. A decoder without an `If` gets the rewrite on its graph when a node other than GroupQueryAttention reads the past, as the rotary positions of Apertus. GroupQueryAttention writes its new rows into the buffer and hides its free rows, so it needs no key mask, and no other node may read its present. The rewrite adds two inputs:

    * `transmuter.position_offset`, v - C. Each position read, a `Gather` on axis 0 or a `Slice` on axis 0 with step 1 of a 2-D static table whose index is a shift of L (L plus values without L), gets it added to its index. Each cast of a shift of L to a float, such as the position of a rotary angle, gets it added to its integer input, so the float values see v and not C. The integer uses of L keep C, because they index the buffer. The offset has one value for each row, of shape `{batch, 1}`, when the value that takes the offset at every read has the batch on axis 0, or a read has positions that meet the tokens of each row, which have the batch first (the table reads of Marian, Whisper, TrOCR and Florence-2, whose output adds to the token embeddings, and the `Mul` of M2M100 and NLLB by the padding mask), and no later `Reshape` or `Expand` has a static first dimension of 1. A value has the batch on axis 0 when it is the index of a `Gather` or the input of a float cast with a leading axis of size 1: an `Unsqueeze` on axis 0 or a `Reshape` to a shape that starts with 1 or -1, then `Unsqueeze` on other axes, `Cast` and `Identity`. The offset then takes the rank of that value, as `{batch, 1, 1}` for the `[1, 1, n]` positions of the rotary angles of Apertus, so it gives the batch to axis 0 and keeps the other axes. Then the rows of a step can have different valid rows and gap rows, so Generation decodes the rows of prompts of different lengths together. Otherwise the offset is one scalar for all rows, which keeps the shape of each value that it is added to, and Generation runs the rows of each prompt length as a group of their own, without gap rows, so that buffer row j holds position j. `:per_row_offset` in the returned model tells which.

    * `transmuter.key_mask`, the attention mask of `:append`, 1 for the valid rows and the new rows and 0 for the gap rows and the free rows. Each `AttentionCore` whose keys read a buffer gets it as input 8, so the free rows get no weight, also when the decoder takes no attention mask, as in Whisper. The key mask is an input only when a node reads it. The rows of a buffer may reach only the keys and the values of these cores, the present cache and Concat, Reshape, Transpose (with the batch first), Unsqueeze, Squeeze, Identity and Cast. Another reader, such as an attention that `Transmuter.Fusion` did not replace, keeps the decoder in `:grow`.

  A walk follows L forward through the branch. A value that depends on L is a shift of L, another integer, a boolean of a comparison, or the shape of a tensor whose length follows the buffer: the buffer, the attention mask, and their results through Concat, Cast, Identity and Transpose. A dimension of such a shape is a shift of L when it is the length, and no L when it is another dimension. These uses keep the decoder in `:grow`: a cast to a float of another value of L, a 2-D static table read on axis 0 with another index of L, and the other reads of static tables at a value of L that is not a shift. A static table that a shift of L indexes in another way, such as the causal table of GPT-2, counts the rows of the buffer, so the buffer and the new token must fit its axes: its smallest axis minus 1 is the largest capacity (1023 for GPT-2). A `Slice` from 0 to a shift of L with step 1 of a tensor that the graph made, whose content has no L or only comparisons of L, is a prefix: with a buffer it keeps C + n rows in place of v + n, which hold every row that a step reads, such as the causal mask of moonshine, which compares the positions of L with the key index. The rows of a prefix are rows of the buffer, the same for each row of the batch, so a prefix allows an offset for each row: the key mask hides a gap row as it hides a free row. A prefix of a static table, such as the rotary table of the merged llama2.c that its nested `If` nodes squeeze, keeps the decoder in `:grow`.

  A read whose value has no batch axis, such as the `[K]` positions of a `Range` or the `[n]` input of a float cast, takes an offset with one axis more than that value, `{batch, 1, ..., 1}`, so the batch is a new axis 0. Each later node takes that axis, until the value meets a value of the tokens of the same rank, or a value whose only uses are the masks of `AttentionCore` nodes, which have the batch on axis 0. There, a `Squeeze` removes a static first axis of size 1 of the value, as the `[1, H, n, K]` bias of T5. Unsqueeze, Squeeze, Transpose and Slice shift their axes. Elementwise ops take static values of at most the same rank, and a `Gather` of a static table at such a value gives a value with the batch. Another op, or a read from a subgraph, keeps one offset for all rows.

  A `Range` from 0 to a shift of L, with a step without L, gives the positions of all keys of a step, as the relative positions of T5. With a buffer, its index is the row of the buffer, and the new rows come after the C rows of the buffer. So the rewrite keeps the `Range` and maps each of its values to a position through the key mask: the cumulative count of the valid rows, minus 1, and the offset after the last key. With an offset for each row, the count runs over the key mask of each row, so the positions have the shape `{batch, keys}`, and a gap row does not move them. With one offset for all rows, the rows of a step have no gap rows, and the count runs over the union of the rows of the step: a row that has ended has a prefix of the valid rows of the active rows, so the union is the mask of the active rows. A valid row gets its position and a new row v + i. The `Range` keeps its start, limit and step, so its output keeps its length. The walk follows these positions through moves, arithmetic and comparisons with values without L, float casts and reads of static tables, such as the bucket table of T5. A use that mixes them with other values of L, such as the causal mask of moonshine, which compares the key index with the positions of L, keeps that range as an index of the buffer, and the walk runs again (at most once for each range).

  A `Mul` of a shift of L by a value without L also makes positions, as the positions of M2M100 and NLLB, (cumsum(mask) + L) * mask + padding_idx. The offset goes on its shift input, so a real token gets v + cumsum and a padding token keeps padding_idx, and the walk follows the result as positions. A position is a position only in position uses: arithmetic and comparisons with values without L, float casts, and reads of static tables. A use as a size or a bound (the shape of a Reshape, Expand or Tile, a Slice start or end, a Range or ConstantOfShape input) or as an index into a tensor that is not a static table needs the index of the buffer, so it sends the range or the `Mul` back to the index space, and the walk runs again.
  """

  require Logger

  alias Transmuter.Onnx.{AttributeProto, GraphProto, NodeProto}

  @position_offset "transmuter.position_offset"
  @key_mask "transmuter.key_mask"

  @doc false
  def position_offset, do: @position_offset

  @doc false
  def key_mask, do: @key_mask

  @doc """
  Returns `model` with the position offset and the key mask in the branch with cache, its
  largest capacity in `:max_capacity`, the shape of its offset in `:per_row_offset` and in
  `:causal` whether a step of several tokens on the branch with cache gives the logits of each
  token (see causal?/3), or `model` itself when the rules do not hold. When the rows of a step
  need one length of the cache, `:one_length` names the node that needs it. `step_rows` is the
  most tokens of a step, which the causal tables must fit with the largest capacity (see
  extend_tables/4).
  """
  def rewrite(model, step_rows) do
    with false <- "position_ids" in model.inputs,
         {branch, graph} <- cache_graph(model),
         {:ok, cores} <- buffer_cores(model, graph),
         {:ok, [_ | _] = reads, limit, kinds} <- past_uses(model, graph) do
      {one_offset, ranks, lift} = one_offset(model, graph, reads)
      per_row? = one_offset == nil
      entries = Map.new(reads ++ cores, &read_entry/1)

      model
      |> offset_reads(branch, graph, entries, per_row?, ranks, lift)
      |> Map.put(:max_capacity, limit && limit - 1)
      |> Map.put(:per_row_offset, per_row?)
      |> Map.put(:one_length, one_offset)
      |> Map.put(:causal, causal?(graph, cores, kinds))
      |> extend_tables(graph, kinds, limit && limit - 1 + step_rows)
    else
      {:bail, node} ->
        exact_cache(model, node)

      node = %NodeProto{} ->
        exact_cache(model, node)

      _other ->
        model
    end
  end

  # A step of n tokens at a capacity of C rows reads C + n keys, so each causal table that a
  # Slice reads gets `rows` rows, the largest capacity and the most tokens of a step, plus the
  # largest part past L of the start of the Slice. The table is a tril (see causal?/1), and a
  # larger tril gives the same values at every key of the smaller one. The largest capacity
  # stays at the rows of the table of the export, which also bound the positions. A causal
  # table of the export is large, so it is a param (see Transmuter.load!/2).
  defp extend_tables(model, _graph, _kinds, nil), do: model

  defp extend_tables(model, graph, kinds, rows) do
    tables =
      for node = %{op_type: "Slice", input: [name, starts | _]} <- graph.node,
          match?({:causal, _, _}, kinds[hd(node.output)]),
          table = model.params[name],
          size = rows + shift_k(kinds[starts]),
          Nx.axis_size(table, -1) < size,
          do: {name, size}

    params =
      tables
      |> Enum.group_by(&elem(&1, 0), &elem(&1, 1))
      |> Map.new(fn {name, sizes} ->
        rows = Enum.max(sizes)
        table = model.params[name]

        rank = Nx.rank(table)
        shape = table |> Nx.shape() |> put_elem(rank - 2, rows) |> put_elem(rank - 1, rows)
        tril = Nx.greater_equal(Nx.iota({rows, rows}, axis: 0), Nx.iota({rows, rows}, axis: 1))
        value = tril |> Nx.as_type(Nx.type(table)) |> Nx.reshape(shape)
        {name, if(model.backend, do: Nx.backend_transfer(value, model.backend), else: value)}
      end)

    %{model | params: Map.merge(model.params, params)}
  end

  # Whether each core whose keys read a buffer masks the keys after each query, so a step of
  # several tokens on the branch with cache gives the logits of each token: its mask or its
  # keys to keep (inputs 3 and 4) follow from a causal table or a comparison of L, as in GPT-2,
  # Florence-2 and moonshine. The exports of Whisper and Marian trace this branch for one token
  # and have no such mask.
  defp causal?(graph, cores, kinds) do
    seeds = for {name, kind} <- kinds, kind == :bool or match?({:causal, _, _}, kind), do: name

    causal =
      Enum.reduce(graph.node, MapSet.new(seeds), fn node, causal ->
        if node.op_type not in ["Shape", "Size"] and
             Enum.any?(node.input, &MapSet.member?(causal, &1)),
           do: MapSet.union(causal, MapSet.new(node.output)),
           else: causal
      end)

    Enum.all?(cores, fn core ->
      Enum.any?(Enum.slice(core.input, 3, 2), &MapSet.member?(causal, &1))
    end)
  end

  # The edits of lift/4 when no read needs them.
  @no_lift %{edits: %{}, static: %{}, rows: MapSet.new(), ranks: %{}}

  # The offset is one value for each row, of shape {batch, 1}, when the rows of a step can have
  # different valid rows: the value that takes the offset at each read has the batch on axis 0
  # (see batch_rank/3) or meets the tokens, no later Reshape or Expand has a static first
  # dimension of 1. Otherwise it is one value for all rows, and Generation runs one group for
  # each prompt length. Returns nil for an offset for each row, else the reason for one offset,
  # and the rank of the value at each read whose axis 0 is the batch.
  defp one_offset(model, graph, reads) do
    scope = static_scope(model, graph)
    producers = for node <- graph.node, output <- node.output, into: %{}, do: {output, node}

    ranks =
      for read <- reads, rank = batch_rank(read, producers, scope), into: %{}, do: {read, rank}

    batch? = &(Map.has_key?(ranks, &1) or meets_tokens?(&1, graph))
    lift = lift(model, graph, scope, Enum.reject(reads, batch?))

    reason =
      cond do
        match?({:bail, _node}, lift) ->
          {"a read that needs one offset for all rows", elem(lift, 1)}

        node = batch_reshape(graph, reads, scope) ->
          {"a reshape to a batch of 1", node}

        true ->
          nil
      end

    case reason do
      nil ->
        {nil, Map.merge(ranks, lift.ranks), lift}

      {what, node} ->
        reason =
          "its position offset is one value for all rows, because of #{what} at #{where(node)}"

        Logger.debug(
          "generation runs one group for each prompt length of the merged decoder: #{reason}"
        )

        {reason, ranks, @no_lift}
    end
  end

  # The ops whose output has the shape of their broadcast inputs, and the ops of one input
  # whose output has the shape of that input.
  @broadcast ~w(Add Sub Mul Div Pow Mod Min Max Where Equal Less LessOrEqual Greater
                GreaterOrEqual And Or Xor)
  @same_dims ~w(Cast CastLike Identity Neg Abs Log Exp Sqrt Sin Cos Floor Ceil Not Sigmoid Tanh
                Relu)

  # Lifts the reads whose positions have no batch axis, such as the relative positions of T5,
  # Range(0, L + n) of shape [K]. The offset {batch, 1} gives each such position the batch as a
  # new axis 0, and each later node takes that axis: its axes and its perm shift by 1, and a
  # Gather of a static table at lifted positions gives a lifted output. A lifted value meets
  # the batch at a node whose other input is a value of the tokens, or whose output reaches
  # only the masks of AttentionCore nodes (input 3, through Cast and Identity), which the
  # graph broadcasts with the batch on axis 0, as the relative bias of T5 that adds to the
  # mask of each core. When the lifted value has a static 1 on its first axis there, as the
  # [1, H, n, K] bias of T5, a Squeeze removes that axis, so the batch takes its place.
  # Returns the edits of the lifted nodes, the new static values and the Range reads that map
  # the rows of the buffer for each row, or {:bail, node} for a node that the lift does not
  # follow.
  defp lift(_model, _graph, _scope, []), do: @no_lift

  defp lift(model, graph, scope, reads) do
    dims = position_dims(graph, scope, model.input_types)
    sources = Enum.map(reads, &read_node/1)
    takes = Map.new(sources, &{&1, offset_value(&1, dims)})

    if Enum.all?(sources, &(match?([_ | _], takes[&1]) and is_list(dims[hd(&1.output)]))) do
      lifted = Map.new(sources, &{hd(&1.output), dims[hd(&1.output)]})
      tokens = token_values(graph)
      masks = mask_values(graph)

      walk = %{scope: scope, tokens: tokens, masks: masks, sources: sources, dims: dims}

      graph.node
      |> Enum.reduce_while({lifted, @no_lift}, &lift_step(&1, &2, walk))
      |> case do
        {:bail, node} ->
          {:bail, node}

        {lifted, acc} ->
          if Enum.any?(
               model.outputs ++ Enum.map(graph.output, & &1.name),
               &Map.has_key?(lifted, &1)
             ),
             do: {:bail, hd(sources)},
             else: %{
               acc
               | rows: MapSet.new(for(%{op_type: "Range"} = n <- sources, do: n)),
                 ranks: Map.new(takes, fn {source, d} -> {source, length(d) + 1} end)
             }
      end
    else
      {:bail, Enum.find(sources, &(not match?([_ | _], takes[&1]))) || hd(sources)}
    end
  end

  # The dims of the value that takes the offset of a read: the positions of a Range (see
  # row_positions/2), the input of a float cast or the index of a Gather. The offset takes one
  # more axis than this value, {batch, 1, ..., 1}, so the batch is a new axis 0 for each rank.
  defp offset_value(%{op_type: "Range", output: [out]}, dims), do: dims[out]
  defp offset_value(%{op_type: "Cast", input: [value]}, dims), do: dims[value]
  defp offset_value(%{op_type: "Gather", input: [_table, index]}, dims), do: dims[index]
  defp offset_value(_read, _dims), do: nil

  # One node of the walk of lift/4: a node without lifted inputs, or a source, stays. A node
  # that reads a lifted value from a subgraph, such as the branch of an If, stops the walk.
  defp lift_step(node, {lifted, acc}, walk) do
    reads = Enum.filter(Transmuter.dependencies(node), &Map.has_key?(lifted, &1))

    cond do
      node in walk.sources or reads == [] ->
        {:cont, {lifted, acc}}

      reads -- node.input != [] ->
        {:halt, {:bail, node}}

      true ->
        lift_read(node, lifted, acc, walk)
    end
  end

  defp lift_read(node, lifted, acc, walk) do
    case lift_node(node, lifted, walk) do
      {:ok, outputs, nodes, static} ->
        edits = Map.put(acc.edits, node, nodes)
        acc = %{acc | edits: edits, static: Map.merge(acc.static, static)}
        {:cont, {Map.merge(lifted, outputs), acc}}

      :bail ->
        {:halt, {:bail, node}}
    end
  end

  # The values whose only uses are the masks of AttentionCore nodes (input 3), through Cast
  # and Identity.
  defp mask_values(graph) do
    uses = fn value ->
      for node <- graph.node,
          value in Transmuter.dependencies(node),
          do: {node, Enum.find_index(node.input, &(&1 == value))}
    end

    graph.node
    |> Enum.flat_map(& &1.output)
    |> Enum.filter(&mask_value?(&1, uses))
    |> MapSet.new()
  end

  defp mask_value?(value, uses) do
    case uses.(value) do
      [] ->
        false

      list ->
        Enum.all?(list, fn
          {%{op_type: "AttentionCore"}, 3} ->
            true

          {%{op_type: op, output: [out]}, 0} when op in ["Cast", "Identity"] ->
            mask_value?(out, uses)

          _other ->
            false
        end)
    end
  end

  # The lift of one node with lifted inputs (see lift/4): the dims of its lifted outputs before
  # the lift, its nodes after the lift and their new static values, or :bail.
  # Only a ConstantOfShape may read the shape of a lifted value.
  defp lift_node(node = %{op_type: op}, lifted, walk) when op != "ConstantOfShape" do
    if Enum.any?(node.input, &match?({:shape, _dims}, lifted[&1])),
      do: :bail,
      else: lift_values(node, lifted, walk)
  end

  defp lift_node(node = %{op_type: "ConstantOfShape", input: [shape]}, lifted, _walk) do
    case lifted[shape] do
      {:shape, d} -> {:ok, %{hd(node.output) => d}, [node], %{}}
      _other -> :bail
    end
  end

  defp lift_values(node = %{op_type: op}, lifted, _walk) when op in @same_dims do
    if hd(node.input) in Map.keys(lifted) and length(Map.keys(Map.take(lifted, node.input))) == 1,
      do: {:ok, %{hd(node.output) => lifted[hd(node.input)]}, [node], %{}},
      else: :bail
  end

  defp lift_values(node = %{op_type: op}, lifted, walk) when op in @broadcast do
    lifted_dims = for name <- node.input, d = lifted[name], uniq: true, do: d
    ranks = lifted_dims |> Enum.map(&length/1) |> Enum.uniq()
    others = Enum.reject(node.input, &(Map.has_key?(lifted, &1) or &1 == ""))
    static? = &(&1 in walk.scope.static)

    cond do
      length(ranks) != 1 ->
        :bail

      Enum.all?(others, static?) ->
        [rank] = ranks
        partner = Enum.map(others, &static_dims(walk.scope, &1))

        if Enum.all?(partner, &(&1 != nil and length(&1) <= rank)),
          do: {:ok, %{hd(node.output) => broadcast_dims(lifted_dims ++ partner)}, [node], %{}},
          else: :bail

      length(Enum.uniq(Enum.filter(node.input, &Map.has_key?(lifted, &1)))) == 1 ->
        join(node, lifted, others, walk)

      true ->
        :bail
    end
  end

  defp lift_values(node = %{op_type: op}, lifted, walk) when op in ["Unsqueeze", "Squeeze"] do
    [data | rest] = node.input
    d = lifted[data]
    axes = axes_of(node, walk.scope)

    if d == nil or axes == nil or Enum.any?(rest, &Map.has_key?(lifted, &1)) do
      :bail
    else
      dims = if op == "Unsqueeze", do: unsqueeze_dims(d, axes), else: squeeze_dims(d, axes)
      {node, static} = with_axes(node, Enum.map(axes, &if(&1 >= 0, do: &1 + 1, else: &1)))
      {:ok, %{hd(node.output) => dims}, [node], static}
    end
  end

  defp lift_values(node = %{op_type: "Transpose", input: [data]}, lifted, _walk) do
    perm = Enum.find_value(node.attribute, &(&1.name == "perm" && &1.ints))
    d = lifted[data]

    if perm && length(perm) == length(d) do
      attribute = %AttributeProto{
        name: "perm",
        type: :INTS,
        ints: [0 | Enum.map(perm, &(&1 + 1))]
      }

      node = %{node | attribute: [attribute | Enum.reject(node.attribute, &(&1.name == "perm"))]}
      {:ok, %{hd(node.output) => Enum.map(perm, &Enum.at(d, &1))}, [node], %{}}
    else
      :bail
    end
  end

  defp lift_values(node = %{op_type: "Gather", input: [table, index]}, lifted, walk) do
    axis = Enum.find_value(node.attribute, 0, &(&1.name == "axis" && &1.i))

    with true <- axis == 0 and table in walk.scope.static and not Map.has_key?(lifted, table),
         d when d != nil <- lifted[index],
         [_rows | tail] <- static_dims(walk.scope, table) do
      {:ok, %{hd(node.output) => d ++ tail}, [node], %{}}
    else
      _other -> :bail
    end
  end

  defp lift_values(
         node = %{op_type: "Slice", input: [data, starts, ends, axes | rest]},
         lifted,
         walk
       ) do
    d = lifted[data]
    values = static_list(walk.scope, axes, node)

    if d && values && not Enum.any?([starts, ends, axes | rest], &Map.has_key?(lifted, &1)) do
      shifted = Enum.map(values, &if(&1 >= 0, do: &1 + 1, else: &1))
      name = @position_offset <> ".lift." <> hd(node.output) <> ".axes"
      static = %{name => Nx.tensor(shifted, type: :s64, backend: Nx.BinaryBackend)}
      node = %{node | input: [data, starts, ends, name | rest]}
      dims = slice_dims(d, values)
      {:ok, %{hd(node.output) => dims}, [node], static}
    else
      :bail
    end
  end

  # The shape of a lifted value is lifted, and only a ConstantOfShape may read it.
  defp lift_values(node = %{op_type: "Shape", input: [data]}, lifted, _walk),
    do: {:ok, %{hd(node.output) => {:shape, lifted[data]}}, [node], %{}}

  defp lift_values(_node, _lifted, _walk), do: :bail

  # The node where a lifted value meets a value with the batch on axis 0: a value of the tokens,
  # or a node whose output reaches only the masks of the cores. With a static 1 on the first
  # axis of the lifted value, a Squeeze removes it there.
  defp join(node, lifted, others, walk) do
    [name] = Enum.filter(node.input, &Map.has_key?(lifted, &1)) |> Enum.uniq()
    d = lifted[name]
    squeeze? = match?([1 | _], d)
    rank = if squeeze?, do: length(d), else: length(d) + 1

    tokens? =
      Enum.any?(others, &MapSet.member?(walk.tokens, &1)) and
        Enum.all?(others, &(is_list(walk.dims[&1]) and length(walk.dims[&1]) == rank))

    mask? = MapSet.member?(walk.masks, hd(node.output)) and rank == 4

    cond do
      not is_list(d) or not (tokens? or mask?) ->
        :bail

      squeeze? ->
        squeezed = @position_offset <> ".lift." <> hd(node.output) <> ".squeezed"
        axes = @position_offset <> ".lift." <> hd(node.output) <> ".axes"

        squeeze = %NodeProto{
          op_type: "Squeeze",
          name: squeezed,
          input: [name, axes],
          output: [squeezed]
        }

        renamed = rename_input(node, name, squeezed)

        {:ok, %{}, [squeeze, renamed],
         %{axes => Nx.tensor([1], type: :s64, backend: Nx.BinaryBackend)}}

      true ->
        {:ok, %{}, [node], %{}}
    end
  end

  # The dims of the values of positions, from the Range nodes and the static values: 1 for an
  # axis of size 1, :x for another axis. A Shape gives {:shape, dims}, which only a
  # ConstantOfShape reads. The inputs of the graph take the dims of their declared shapes.
  defp position_dims(graph, scope, input_types) do
    inputs =
      Map.new(input_types, fn {name, {_type, shape}} ->
        {name, Enum.map(shape, &if(&1 == 1, do: 1, else: :x))}
      end)

    Enum.reduce(graph.node, inputs, fn node, dims ->
      get = fn name ->
        case dims[name] do
          nil -> static_dims(scope, name)
          {:shape, _} -> nil
          d -> d
        end
      end

      case node_dims(node, get, dims, scope) do
        nil -> dims
        d -> Map.put(dims, hd(node.output), d)
      end
    end)
  end

  defp node_dims(%{op_type: "Range"}, _get, _dims, _scope), do: [:x]

  defp node_dims(%{op_type: op, input: [input | _]}, get, _dims, _scope) when op in @same_dims,
    do: get.(input)

  defp node_dims(%{op_type: op, input: inputs}, get, _dims, _scope) when op in @broadcast do
    all = Enum.map(Enum.reject(inputs, &(&1 == "")), get)
    if Enum.all?(all, &(&1 != nil)), do: broadcast_dims(all)
  end

  defp node_dims(node = %{op_type: op, input: [data | _]}, get, _dims, scope)
       when op in ["Unsqueeze", "Squeeze"] do
    d = get.(data)
    axes = axes_of(node, scope)

    cond do
      d == nil or axes == nil -> nil
      op == "Unsqueeze" -> unsqueeze_dims(d, axes)
      true -> squeeze_dims(d, axes)
    end
  end

  defp node_dims(node = %{op_type: "Transpose", input: [data]}, get, _dims, _scope) do
    perm = Enum.find_value(node.attribute, &(&1.name == "perm" && &1.ints))
    d = get.(data)
    if d && perm && length(perm) == length(d), do: Enum.map(perm, &Enum.at(d, &1))
  end

  defp node_dims(node = %{op_type: "Gather", input: [data, index]}, get, _dims, _scope) do
    axis = Enum.find_value(node.attribute, 0, &(&1.name == "axis" && &1.i))
    {d, i} = {get.(data), get.(index)}

    if d && i && axis >= 0 && axis < length(d),
      do: Enum.take(d, axis) ++ i ++ Enum.drop(d, axis + 1)
  end

  defp node_dims(
         node = %{op_type: "Slice", input: [data, _starts, _ends, axes | _]},
         get,
         _dims,
         scope
       ) do
    d = get.(data)
    values = d && static_list(scope, axes, node)
    if values, do: slice_dims(d, values)
  end

  defp node_dims(%{op_type: "Shape", input: [data]}, get, _dims, _scope) do
    if d = get.(data), do: {:shape, d}
  end

  defp node_dims(%{op_type: "ConstantOfShape", input: [shape]}, _get, dims, _scope) do
    case dims[shape] do
      {:shape, d} -> d
      _other -> nil
    end
  end

  defp node_dims(_node, _get, _dims, _scope), do: nil

  defp static_dims(scope, name) do
    case scope.values[name] do
      nil -> nil
      value -> value |> Nx.shape() |> Tuple.to_list() |> Enum.map(&if(&1 == 1, do: 1, else: :x))
    end
  end

  # The dims of the broadcast of values, aligned from the last axis.
  defp broadcast_dims(all) do
    rank = all |> Enum.map(&length/1) |> Enum.max()
    padded = Enum.map(all, &(List.duplicate(1, rank - length(&1)) ++ &1))

    padded
    |> Enum.zip_with(& &1)
    |> Enum.map(&if(Enum.all?(&1, fn d -> d == 1 end), do: 1, else: :x))
  end

  defp unsqueeze_dims(d, axes) do
    rank = length(d) + length(axes)

    axes
    |> Enum.map(&if(&1 < 0, do: &1 + rank, else: &1))
    |> Enum.sort()
    |> Enum.reduce(d, &List.insert_at(&2, &1, 1))
  end

  defp squeeze_dims(d, axes) do
    axes = Enum.map(axes, &if(&1 < 0, do: &1 + length(d), else: &1))
    for {dim, i} <- Enum.with_index(d), i not in axes, do: dim
  end

  defp slice_dims(d, axes) do
    axes = Enum.map(axes, &if(&1 < 0, do: &1 + length(d), else: &1))
    for {dim, i} <- Enum.with_index(d), do: if(i in axes and dim != 1, do: :x, else: dim)
  end

  # The axes of an Unsqueeze or a Squeeze, from its static input or its attribute.
  defp axes_of(node = %{input: [_data, axes]}, scope) when axes != "",
    do: static_list(scope, axes, node)

  defp axes_of(node, _scope), do: Enum.find_value(node.attribute, &(&1.name == "axes" && &1.ints))

  # The node with new axes: a new static value for an axes input, so a constant that other
  # nodes share keeps its value, or a new attribute.
  defp with_axes(node = %{input: [data, axes]}, values) when axes != "" do
    name = @position_offset <> ".lift." <> hd(node.output) <> ".axes"

    {%{node | input: [data, name]},
     %{name => Nx.tensor(values, type: :s64, backend: Nx.BinaryBackend)}}
  end

  defp with_axes(node, values) do
    attribute = %AttributeProto{name: "axes", type: :INTS, ints: values}
    {%{node | attribute: [attribute | Enum.reject(node.attribute, &(&1.name == "axes"))]}, %{}}
  end

  # The rank of the value that takes the offset of a read, when its axis 0 is the batch: the
  # index of a Gather on axis 0, or the input of a float cast, with a leading axis of size 1.
  # The offset then has that rank, {batch, 1, ..., 1}, so it gives the batch to axis 0 and
  # keeps the other axes, as the [1, 1, n] positions of the rotary angles of Apertus.
  defp batch_rank(node = %NodeProto{op_type: "Gather", input: [_table, index]}, producers, scope) do
    axis = Enum.find_value(node.attribute, 0, &(&1.name == "axis" && &1.i))
    if axis == 0, do: leading_rank(index, producers, scope)
  end

  defp batch_rank(%NodeProto{op_type: "Cast", input: [value]}, producers, scope),
    do: leading_rank(value, producers, scope)

  defp batch_rank(_read, _producers, _scope), do: nil

  # The rank of a value whose axis 0 is a leading axis of size 1: an Unsqueeze on axis 0 or a
  # Reshape to a shape whose first static value is 1 or -1, with more values after it, then
  # Unsqueeze on other axes, Cast and Identity, which keep axis 0. nil for another value.
  defp leading_rank(value, producers, scope) do
    node = producers[value]
    list = &static_list(scope, &1, node)

    case node do
      %{op_type: op, input: [input]} when op in ["Cast", "Identity"] ->
        leading_rank(input, producers, scope)

      %{op_type: "Unsqueeze", input: [input | _]} ->
        with axes when axes != nil <- unsqueeze_axes(node, list),
             leading? = 0 in axes,
             rank when rank != nil <-
               if(leading?,
                 do: rank(input, producers, scope),
                 else: leading_rank(input, producers, scope)
               ),
             do: rank + length(axes)

      %{op_type: "Reshape", input: [_data, shape]} ->
        if leading_one?(shape, producers, list), do: shape_length(shape, producers, list)

      _other ->
        nil
    end
  end

  # The rank of a value of positions: a Range, and the Unsqueeze, Reshape, Cast and Identity
  # of it.
  defp rank(value, producers, scope) do
    node = producers[value]
    list = &static_list(scope, &1, node)

    case node do
      %{op_type: "Range"} ->
        1

      %{op_type: op, input: [input]} when op in ["Cast", "Identity"] ->
        rank(input, producers, scope)

      %{op_type: "Unsqueeze", input: [input | _]} ->
        axes = unsqueeze_axes(node, list)
        rank = rank(input, producers, scope)
        if axes && rank, do: rank + length(axes)

      %{op_type: "Reshape", input: [_data, shape]} ->
        shape_length(shape, producers, list)

      _other ->
        nil
    end
  end

  # The axes of an Unsqueeze, from its input or its attribute.
  defp unsqueeze_axes(node, list) do
    case node.input do
      [_data, axes] -> list.(axes)
      [_data] -> Enum.find_value(node.attribute, &(&1.name == "axes" && &1.ints))
    end
  end

  # The number of values of a shape: a static list, or a Concat of static lists and of
  # Unsqueezes of scalars.
  defp shape_length(shape, producers, list) do
    case {list.(shape), producers[shape]} do
      {[_ | _] = values, _producer} ->
        length(values)

      {nil, %{op_type: "Concat", input: parts}} ->
        lengths =
          Enum.map(parts, fn part ->
            case {list.(part), producers[part]} do
              {[_ | _] = values, _} -> length(values)
              {nil, %{op_type: "Unsqueeze"} = node} -> length(unsqueeze_axes(node, list) || [])
              _other -> nil
            end
          end)

        if Enum.all?(lengths, &(&1 && &1 > 0)), do: Enum.sum(lengths)

      _other ->
        nil
    end
  end

  # A read whose positions meet the tokens of each row, which have the batch first: a table
  # read whose output only goes into Add, Sub or Mul with a value of the tokens, through Cast,
  # Identity or DequantizeLinear, as the positions of Marian, Whisper and TrOCR that add to the
  # token embeddings, or a Mul whose
  # other input is a value of the tokens, as the padding mask of M2M100. Its index or its
  # shift input then takes the offset of each row, {batch, 1}: an index of the token positions
  # gets the batch first, and an index with a first axis of the batch or 1 keeps its shape,
  # as the sum with the tokens is valid in the graph.
  defp meets_tokens?({node, input}, graph),
    do: Enum.any?(List.delete_at(node.input, input), &MapSet.member?(token_values(graph), &1))

  defp meets_tokens?(node = %NodeProto{op_type: op}, graph) when op in ["Gather", "Slice"],
    do: adds_to_tokens?(hd(node.output), graph, token_values(graph))

  defp meets_tokens?(_read, _graph), do: false

  @same_shape ~w(Cast Identity DequantizeLinear)

  defp adds_to_tokens?(value, graph, tokens) do
    uses = Enum.filter(graph.node, &(value in &1.input))

    uses != [] and
      Enum.all?(uses, fn use ->
        cond do
          use.op_type in @same_shape and hd(use.input) == value ->
            adds_to_tokens?(hd(use.output), graph, tokens)

          use.op_type in ["Add", "Sub", "Mul"] ->
            Enum.any?(List.delete(use.input, value), &MapSet.member?(tokens, &1))

          true ->
            false
        end
      end)
  end

  # The values that follow the tokens of each row, input_ids or inputs_embeds, without the
  # shapes and the sizes, which have no batch axis.
  defp token_values(graph) do
    Enum.reduce(graph.node, MapSet.new(["input_ids", "inputs_embeds"]), fn node, tokens ->
      if node.op_type not in ["Shape", "Size"] and
           Enum.any?(node.input, &MapSet.member?(tokens, &1)),
         do: MapSet.union(tokens, MapSet.new(node.output)),
         else: tokens
    end)
  end

  defp leading_one?(shape, producers, list) do
    case {list.(shape), producers[shape]} do
      {[first, _ | _], _producer} -> first in [1, -1]
      {nil, %{op_type: "Concat", input: [first, _ | _]}} -> list.(first) in [[1], [-1]]
      _other -> false
    end
  end

  # A Reshape or Expand after a read, to a static shape whose first value is 1.
  defp batch_reshape(graph, reads, scope) do
    after_reads = MapSet.new(reads, &hd(read_node(&1).output))

    graph.node
    |> Enum.reduce_while(after_reads, fn node, after_reads ->
      cond do
        not Enum.any?(node.input, &MapSet.member?(after_reads, &1)) ->
          {:cont, after_reads}

        node.op_type in ["Reshape", "Expand"] and
            match?([1 | _], static_list(scope, Enum.at(node.input, 1), node)) ->
          {:halt, node}

        true ->
          {:cont, MapSet.union(after_reads, MapSet.new(node.output))}
      end
    end)
    |> case do
      %NodeProto{} = node -> node
      _after_reads -> nil
    end
  end

  defp read_node({node, _input}), do: node
  defp read_node(node), do: node

  # A read is a node, or a node and the input that takes the offset (see candidate/3).
  defp read_entry({node, input}), do: {node, input}
  defp read_entry(node), do: {node, nil}

  defp exact_cache(model, node) do
    reason =
      "it keeps a cache of exact length, because #{where(node)} reads the cache or its length"

    Logger.debug("generation: #{reason}")
    Map.put(model, :one_length, reason)
  end

  # The comparisons give booleans, whose casts are masks.
  @comparisons ~w(Equal Less LessOrEqual Greater GreaterOrEqual Not And Or Xor)

  # The inputs of an operator that carry values. The other inputs are shapes, axes or
  # indices, which do not carry the value of L into the output.
  @value_inputs %{
    "Reshape" => [0],
    "Expand" => [0],
    "Tile" => [0],
    "Unsqueeze" => [0],
    "Squeeze" => [0],
    "Slice" => [0],
    "Gather" => [0],
    "GatherElements" => [0],
    "Pad" => [0],
    "ConstantOfShape" => []
  }

  # The operators whose inputs are sizes or bounds, which need the index of the buffer.
  @sizes ~w(Range ConstantOfShape)

  # The operators that keep L shifted: their value is L plus values that do not depend on L.
  @shifts ~w(Gather Cast Unsqueeze Squeeze Reshape Expand Identity Slice)

  # The position reads of the branch with cache, and the smallest axis of the static tables
  # that a length of the buffer indexes, such as the position table of Marian and the causal
  # table of GPT-2, less the largest part past L of the index, which the rows of the buffer and
  # the new token must fit. The values that depend on L have a kind: {:shift, k} for L plus
  # other values, of which k is the largest static part (see kind/3), :int for other integers,
  # :bool for comparisons and {:dims, axis, rank} for the shape of a tensor whose length
  # follows the buffer. The lengths map gives the axis of that length from the end and the
  # rank, or :unknown: axis 2 of 4 for the buffers of the cache, axis 1 of 2 for the attention
  # mask, and the axes after Concat, Cast, Identity and Transpose.
  defp past_uses(model, graph, index_ranges \\ MapSet.new()) do
    scope = static_scope(model, graph)
    lengths = Map.new(buffers(model), &{&1, {-2, 4}})

    lengths =
      if "attention_mask" in model.inputs,
        do: Map.put(lengths, "attention_mask", {-1, 2}),
        else: lengths

    graph.node
    |> Enum.reduce_while({[], %{}, lengths, nil}, fn node, {reads, kinds, lengths, limit} ->
      lengths = Map.merge(lengths, track(node, lengths, kinds))

      case past_use(%{scope: scope, lengths: lengths, index: index_ranges}, node, kinds) do
        {:read, node, rows} ->
          {:cont, {[node | reads], kinds, lengths, min(limit, rows)}}

        {:positions, node, outputs} ->
          {:cont, {[node | reads], Map.merge(kinds, outputs), lengths, limit}}

        {:kinds, outputs} ->
          {:cont, {reads, Map.merge(kinds, outputs), lengths, limit}}

        {:limit, size, outputs} ->
          {:cont, {reads, Map.merge(kinds, outputs), lengths, min(limit, size)}}

        bail ->
          {:halt, bail}
      end
    end)
    |> case do
      {:bail, _node} = bail -> bail
      {:mixed, ranges, node} -> index_space(model, graph, index_ranges, ranges, node)
      {reads, kinds, _lengths, limit} -> {:ok, reads, limit, kinds}
    end
  end

  # A use mixes key positions with L, so their ranges stay in the index space of the buffer,
  # and the walk runs again. Each range changes at most once: a range that mixes again stops
  # the rewrite.
  defp index_space(model, graph, index_ranges, ranges, node) do
    if MapSet.subset?(ranges, index_ranges),
      do: {:bail, node},
      else: past_uses(model, graph, MapSet.union(index_ranges, ranges))
  end

  # The axis of the length in the outputs of a node that reads a tensor of the lengths map or
  # takes its shape from L. An attention gives the shape of its queries, and a Gather the shape
  # of its index.
  defp track(node, lengths, kinds) do
    data = if node.op_type == "Gather", do: Enum.take(node.input, 1), else: node.input
    axes = for name <- data, Map.has_key?(lengths, name), uniq: true, do: lengths[name]

    axis =
      case {node.op_type, axes} do
        {op, _} when op in ["Shape", "Size", "AttentionCore", "GroupQueryAttention"] -> nil
        {_op, []} -> if shaped?(node, kinds), do: :unknown
        {op, [{_, _} = axis]} when op in ["Concat", "Cast", "Identity"] -> axis
        {"Transpose", [{_, _} = axis]} -> transposed(node, axis)
        _other -> :unknown
      end

    if axis, do: outputs(node, axis), else: %{}
  end

  defp transposed(node, {axis, rank}) do
    perm = Enum.find_value(node.attribute, &(&1.name == "perm" && &1.ints))

    if perm && length(perm) == rank,
      do: {Enum.find_index(perm, &(&1 == axis + rank)) - rank, rank},
      else: :unknown
  end

  # Whether the shape of the output of a node depends on L through its shape arguments. A
  # slice or a range between two shifts of L has a length without L.
  defp shaped?(node, kinds) do
    dep? = &Map.has_key?(kinds, &1)
    same? = &((not dep?.(&1) and not dep?.(&2)) or (shift?(kinds[&1]) and shift?(kinds[&2])))

    case node do
      %{op_type: op, input: [_ | args]} when op in ["Reshape", "Expand", "Tile", "Pad"] ->
        Enum.any?(args, dep?)

      %{op_type: "ConstantOfShape", input: [shape]} ->
        dep?.(shape)

      %{op_type: "Range", input: [start, limit | _]} ->
        not same?.(start, limit)

      %{op_type: "Slice", input: [_, starts, ends | _]} ->
        not same?.(starts, ends)

      _other ->
        false
    end
  end

  # The static values of the branch: the params and the static values of the model that the
  # branch does not make again, and the tensors of its initializers and Constant nodes.
  defp static_scope(model, graph) do
    made = for node <- graph.node, node.op_type != "Constant", name <- node.output, do: name

    local =
      Map.merge(
        Map.new(graph.initializer, &{&1.name, Transmuter.Ops.tensor(&1)}),
        for(
          %{op_type: "Constant", output: [name], attribute: [%{t: t}]} <- graph.node,
          t != nil,
          into: %{},
          do: {name, Transmuter.Ops.tensor(t)}
        )
      )

    values = model.params |> Map.merge(model.static) |> Map.drop(made) |> Map.merge(local)
    %{values: values, static: MapSet.new(Map.keys(values))}
  end

  # The values of a static input of a node as a list, or nil when the input is not static. Only
  # a small value, such as a shape, an axis, a start or a step, becomes a list: a walk that
  # reads a weight, as a list of millions of terms, raises.
  defp static_list(scope, name, node) do
    value = scope.values[name]

    cond do
      value == nil ->
        nil

      Nx.rank(value) <= 1 and Nx.size(value) <= 16 ->
        Nx.to_flat_list(value)

      true ->
        raise ArgumentError,
              "the rewrite reads the static input #{name} of #{node.op_type} node " <>
                "#{inspect(node.name)} as a list, but its shape #{inspect(Nx.shape(value))} " <>
                "is not a shape, an axis or an index"
    end
  end

  defp past_use(walk, node = %{op_type: op, input: [source | _]}, _kinds)
       when op in ["Shape", "Size"] do
    sliced? = Enum.any?(node.attribute, &(&1.name in ["start", "end"]))

    case walk.lengths[source] do
      nil -> {:kinds, %{}}
      _length when sliced? -> {:bail, node}
      {axis, rank} when op == "Shape" -> {:kinds, outputs(node, {:dims, axis, rank})}
      _other -> {:kinds, outputs(node, :int)}
    end
  end

  defp past_use(walk = %{scope: scope}, node, kinds) do
    deps = for {name, i} <- Enum.with_index(node.input), kind = kinds[name], do: {i, kind}
    outer? = Enum.any?(Transmuter.dependencies(node) -- node.input, &Map.has_key?(kinds, &1))

    cond do
      outer? ->
        {:bail, node}

      deps == [] ->
        {:kinds, %{}}

      # The total length and the counts of the rows of GroupQueryAttention index the buffer.
      node.op_type == "GroupQueryAttention" and Enum.all?(deps, &(elem(&1, 0) in [5, 6])) ->
        {:kinds, %{}}

      key_positions?(walk, node, deps) ->
        {:positions, node, outputs(node, {:position, MapSet.new(node.output), range_end(deps)})}

      input = candidate(walk, node, deps) ->
        {_input, {:shift, k}} = List.keyfind(deps, input, 0)
        {:positions, {node, input}, outputs(node, {:position, MapSet.new(node.output), k + 1})}

      node.op_type in ["Gather", "Slice", "GatherElements", "GatherND"] ->
        gather_use(scope, node, kinds, deps)

      node.op_type == "If" ->
        {:bail, node}

      true ->
        value_use(scope, node, deps)
    end
  end

  # A read of a static table, a dimension of a shape, a slice of a causal table, a prefix of a
  # mask, or L as an index into another tensor, such as a rotary table that the graph makes,
  # which stops the rewrite.
  defp gather_use(scope, node, kinds, deps) do
    data = kinds[hd(node.input)]
    table_index? = &(&1 == 1 and node.op_type == "Gather" and hd(node.input) in scope.static)
    bounds = for {i, {:position, r, _k}} <- deps, i > 0, not table_index?.(i), do: r

    cond do
      bounds != [] ->
        {:mixed, Enum.reduce(bounds, &MapSet.union/2), node}

      hd(node.input) in scope.static ->
        static_read(scope, node, kinds)

      match?({:dims, _, _}, data) ->
        dims(scope, node, kinds)

      match?({:causal, _, _}, data) and node.op_type == "Slice" ->
        causal_slice(scope, node, kinds, Tuple.delete_at(data, 0))

      data in [nil, :bool] and prefix?(scope, node, kinds) ->
        {:kinds, if(data, do: outputs(node, data), else: %{})}

      Enum.any?(deps, &(elem(&1, 0) > 0)) ->
        {:bail, node}

      true ->
        value_use(scope, node, deps)
    end
  end

  # A Slice from 0 to a shift of L with step 1, a prefix. With a buffer it keeps C + n rows in
  # place of v + n, which hold every row that a step reads, and it shifts nothing. Its output
  # keeps the kind of its data. Its rows are rows of the buffer, the same for each row of the
  # batch, so it allows an offset for each row: the key mask hides a gap row as it hides a free
  # row, and a read of the prefix at a shift of L stops the rewrite (see gather_use/4).
  defp prefix?(scope, node = %{op_type: "Slice", input: [_data, starts, ends | rest]}, kinds) do
    steps = Enum.at(rest, 1, "")
    list = &static_list(scope, &1, node)
    all? = fn name, value -> list.(name) != nil and Enum.all?(list.(name), &(&1 == value)) end
    all?.(starts, 0) and shift?(kinds[ends]) and (steps == "" or all?.(steps, 1))
  end

  defp prefix?(_scope, _node, _kinds), do: false

  # A dimension of a shape: the length is a shift of L, another dimension does not depend on it.
  defp dims(scope, node = %{op_type: op, input: [shape | rest]}, kinds) do
    {:dims, axis, rank} = kinds[shape]
    list = &static_list(scope, &1, node)
    from_end = &if(&1 >= 0 and &1 < rank, do: &1 - rank, else: &1)

    picked =
      case {op, Enum.map(rest, list)} do
        {"Gather", [[index]]} ->
          if from_end.(index) == axis, do: {:shift, 0}

        {"Slice", [[start], [stop] | axes]} when axes in [[], [[0]]] ->
          {start, stop} = {from_end.(start), if(stop >= rank, do: 0, else: from_end.(stop))}

          if start <= axis and stop > axis,
            do: if(start == axis and stop == axis + 1, do: {:shift, 0}, else: :int)

        _other ->
          :int
      end

    {:kinds, if(picked, do: outputs(node, picked), else: %{})}
  end

  # The kind of the outputs of a node that reads values of L.
  defp value_use(scope, node, deps) do
    inputs = Map.get(@value_inputs, node.op_type, Enum.to_list(0..(length(node.input) - 1)//1))
    value_kinds = for {i, kind} <- deps, i in inputs, do: kind

    # Comparisons and causal tables are masks, which hold for the rows of a buffer.
    values = Enum.reject(value_kinds, &(&1 == :bool or match?({:causal, _, _}, &1)))
    {positions, others} = Enum.split_with(values, &match?({:position, _, _}, &1))

    ranges =
      Enum.reduce(positions, MapSet.new(), fn {:position, r, _k}, acc -> MapSet.union(acc, r) end)

    bounds =
      for {i, {:position, r, _k}} <- deps,
          i not in inputs or node.op_type in @sizes,
          do: r

    cond do
      bounds != [] -> {:mixed, Enum.reduce(bounds, &MapSet.union/2), node}
      positions != [] and others != [] -> {:mixed, ranges, node}
      positions != [] -> position_use(scope, node, ranges, deps)
      node.op_type in @comparisons -> {:kinds, outputs(node, :bool)}
      values == [] -> {:kinds, %{}}
      node.op_type == "Cast" and float_cast?(node) -> float_use(node, deps)
      true -> {:kinds, outputs(node, kind(scope, node, deps))}
    end
  end

  defp outputs(node, kind), do: Map.new(node.output, &{&1, kind})

  # The operators that make a position from a shift of L: a Mul by a value without L, as the
  # positions of M2M100 and NLLB, (cumsum(mask) + L) * mask + padding_idx. The offset goes on
  # the shift input, so a real token gets v + cumsum and a padding token keeps padding_idx.
  # The cumulative count of a real token is at least 1, so its position is at least k + 1 past
  # L for a shift {:shift, k}.
  # Gives the index of the shift input, unless an earlier walk sent this node back to the index
  # space (see value_use/2).
  @candidates ~w(Mul)

  defp candidate(walk, node = %{op_type: op}, [{input, {:shift, _k}}]) when op in @candidates do
    if length(node.input) == 2 and not MapSet.member?(walk.index, hd(node.output)), do: input
  end

  defp candidate(_walk, _node, _deps), do: nil

  # The positions of all keys of a step: a Range from 0 to a shift of L with a step without L,
  # unless an earlier walk found that a use compares or mixes it with L (see value_use/2), as
  # the causal mask of moonshine compares the key index with the positions of L.
  defp key_positions?(walk, node = %{op_type: "Range", input: [start, _limit | _]}, deps) do
    match?([{1, {:shift, _k}}], deps) and static_list(walk.scope, start, node) == [0] and
      not MapSet.member?(walk.index, hd(node.output))
  end

  defp key_positions?(_walk, _node, _deps), do: false

  # A use of key positions (see key_positions?/3) alone: a float cast or a comparison ends
  # them, and other operators keep them positions. Positions keep their largest part past L
  # through moves and the addition of a static value, and lose it through other operators,
  # such as the clamp of the buckets of T5, so a read of a table at them gives no limit.
  defp position_use(scope, node, ranges, deps) do
    ks = for {_i, {:position, _r, k}} <- deps, do: k

    k =
      cond do
        nil in ks -> nil
        node.op_type in @shifts -> Enum.max(ks)
        node.op_type == "Add" -> added(scope, node, deps, Enum.max(ks))
        true -> nil
      end

    cond do
      node.op_type in @comparisons -> {:kinds, outputs(node, :bool)}
      node.op_type == "Cast" and float_cast?(node) -> {:kinds, %{}}
      true -> {:kinds, outputs(node, {:position, ranges, k})}
    end
  end

  # A float cast of a shift of L gets the offset on its input (see offset_reads/4), so its
  # float value no longer depends on the buffer. A float cast of another value of L stops the
  # rewrite.
  defp float_use(node, deps) do
    if Enum.all?(deps, &shift?(elem(&1, 1))), do: {:read, node, nil}, else: {:bail, node}
  end

  # L + c, c + L and L - c are shifts, c - L is not. The largest part past L, k, adds the
  # largest value of a static c, or subtracts its smallest. A c that is not static, such as the
  # number of tokens of the step, adds nothing: a position read and a causal slice read the
  # first token of a step at k past L, and the limit of the table counts that token.
  defp kind(scope, node = %{op_type: "Add"}, deps = [{_i, {:shift, k}}]),
    do: {:shift, added(scope, node, deps, k)}

  defp kind(scope, node = %{op_type: "Sub"}, [{0, {:shift, k}}]) do
    case static_list(scope, Enum.at(node.input, 1), node) do
      nil -> {:shift, k}
      values -> {:shift, k - Enum.min(values)}
    end
  end

  # A range from L, whose step does not depend on L, starts at its start.
  defp kind(_scope, %{op_type: "Range"}, deps) do
    case List.keyfind(deps, 0, 0) do
      {0, shift = {:shift, _k}} -> if List.keymember?(deps, 2, 0), do: :int, else: shift
      _other -> :int
    end
  end

  defp kind(_scope, %{op_type: op}, deps) when op in @shifts do
    if Enum.all?(deps, &shift?(elem(&1, 1))),
      do: {:shift, deps |> Enum.map(&shift_k(elem(&1, 1))) |> Enum.max()},
      else: :int
  end

  defp kind(_scope, _node, _deps), do: :int

  # k plus the largest value of the static input of an Add whose other input is the one in
  # `deps`, or k when that input is not static.
  defp added(scope, node, [{i, _kind}], k) do
    case static_list(scope, Enum.at(node.input, 1 - i), node) do
      nil -> k
      values -> k + Enum.max(values)
    end
  end

  defp added(_scope, _node, _deps, _k), do: nil

  defp shift?(kind), do: match?({:shift, _k}, kind)

  defp shift_k({:shift, k}), do: k
  defp shift_k(nil), do: 0

  # The largest part past L of the values of a Range from 0 to a shift of L: the part of its
  # limit, because the first token of a step is the last value of the range.
  defp range_end(deps) do
    {1, {:shift, k}} = List.keyfind(deps, 1, 0)
    k
  end

  defp float_cast?(node) do
    # FLOAT, FLOAT16, DOUBLE and BFLOAT16 of TensorProto.DataType.
    Enum.any?(node.attribute, &(&1.name == "to" and &1.i in [1, 10, 11, 16]))
  end

  # A position read, or a Slice of a causal table (1 where the column is at most the row, as
  # the bias of GPT-2) on its last two axes. The rows of a position table and the rows and
  # columns of a causal table count the rows of the buffer, which is right while the buffer and
  # the new token fit the table. Another read of a static table at L, such as a local attention
  # window, stops the rewrite.
  defp static_read(scope, node, kinds) do
    table = scope.values[hd(node.input)]

    cond do
      Nx.rank(table) == 2 and position_index?(scope, node, kinds) ->
        {:read, node, Nx.axis_size(table, 0) - shift_k(kinds[Enum.at(node.input, 1)])}

      node.op_type == "Gather" and match?({:position, _, _}, kinds[Enum.at(node.input, 1)]) ->
        position_table(node, table, kinds[Enum.at(node.input, 1)])

      node.op_type == "Slice" and causal?(table) ->
        causal_slice(scope, node, kinds, {Nx.rank(table), Nx.axis_size(table, -1)})

      true ->
        {:bail, node}
    end
  end

  # A Slice of a causal table, or of a slice of it, on its last two axes, from and to shifts of L.
  defp causal_slice(scope, node = %{input: [_, starts, ends, axes | _]}, kinds, {rank, size}) do
    axes = static_list(scope, axes, node)
    last = [rank - 2, rank - 1, -2, -1]

    if (kinds[starts] == nil or shift?(kinds[starts])) and
         (kinds[ends] == nil or shift?(kinds[ends])) and axes != nil and
         Enum.all?(axes, &(&1 in last)),
       do: {:limit, size - shift_k(kinds[starts]), outputs(node, {:causal, rank, size})},
       else: {:bail, node}
  end

  defp causal_slice(_scope, node, _kinds, _table), do: {:bail, node}

  defp causal?(table) do
    case table |> Nx.shape() |> Tuple.to_list() |> Enum.split(-2) do
      {lead, [size, size]} ->
        tril = Nx.greater_equal(Nx.iota({size, size}, axis: 0), Nx.iota({size, size}, axis: 1))

        Enum.all?(lead, &(&1 == 1)) and
          table
          |> Nx.backend_copy(Nx.BinaryBackend)
          |> Nx.reshape({size, size})
          |> Nx.not_equal(0)
          |> Nx.equal(tril)
          |> Nx.all()
          |> Nx.to_number() == 1

      _other ->
        false
    end
  end

  # A Gather of a static table on axis 0 at positions, such as the position table of M2M100:
  # with the largest part past L of the positions, the rows of the table limit the buffer.
  defp position_table(node, table, {:position, _ranges, k}) do
    axis = Enum.find_value(node.attribute, 0, &(&1.name == "axis" && &1.i))

    if axis == 0 and k != nil,
      do: {:limit, Nx.axis_size(table, 0) - k, %{}},
      else: {:kinds, %{}}
  end

  # The index of a position read: a Gather on axis 0, or a Slice on axis 0 with step 1, whose
  # index or start is a shift of L.
  defp position_index?(_scope, node = %{op_type: "Gather", input: [_table, index]}, kinds) do
    axis = Enum.find_value(node.attribute, 0, &(&1.name == "axis" && &1.i))
    axis == 0 and shift?(kinds[index])
  end

  defp position_index?(scope, node = %{op_type: "Slice", input: [_, starts, ends | rest]}, kinds) do
    [axes, steps] = Enum.take(rest ++ ["", ""], 2)
    list = &static_list(scope, &1, node)

    shift?(kinds[starts]) and shift?(kinds[ends]) and list.(axes) == [0] and
      (steps == "" or list.(steps) == [1])
  end

  defp position_index?(_scope, _node, _kinds), do: false

  # The AttentionCore nodes of the branch with cache whose keys read a buffer (see buffers/1).
  # The rows of a buffer may reach only the keys and the values of these cores, the present
  # cache and the operators that move them, so the key mask hides each free row. Another
  # reader, such as an attention that Fusion did not replace, keeps the decoder in :grow.
  @moves ~w(Concat Reshape Transpose Unsqueeze Squeeze Identity Cast)

  defp buffer_cores(model, graph) do
    graph.node
    |> Enum.reduce_while({[], MapSet.new(buffers(model))}, fn node, {cores, rows} ->
      case buffer_use(node, rows) do
        :core -> {:cont, {[node | cores], rows}}
        :moves -> {:cont, {cores, MapSet.union(rows, MapSet.new(node.output))}}
        :none -> {:cont, {cores, rows}}
        bail -> {:halt, bail}
      end
    end)
    |> case do
      {:bail, _node} = bail -> bail
      {cores, _rows} -> {:ok, cores}
    end
  end

  defp buffer_use(node, rows) do
    read = for {name, i} <- Enum.with_index(node.input), MapSet.member?(rows, name), do: i
    outer? = Enum.any?(Transmuter.dependencies(node) -- node.input, &MapSet.member?(rows, &1))

    cond do
      outer? ->
        {:bail, node}

      read == [] or node.op_type in ["Shape", "Size"] ->
        :none

      node.op_type == "AttentionCore" and read -- [1, 2] == [] ->
        if(1 in read, do: :core, else: :none)

      # GroupQueryAttention writes its new rows into the buffer and hides its free rows.
      node.op_type == "GroupQueryAttention" and read -- [3, 4] == [] ->
        :none

      # The folded heads of a core must keep the batch first, as the key mask does.
      node.op_type == "Transpose" and
          Enum.any?(node.attribute, &(&1.name == "perm" and hd(&1.ints) != 0)) ->
        {:bail, node}

      node.op_type in @moves ->
        :moves

      true ->
        {:bail, node}
    end
  end

  # The graph of the positions: the branch with cache of a merged decoder, which alone reads the
  # past, or the graph of a decoder without an If whose nodes other than GroupQueryAttention
  # read the past, such as the rotary positions of Apertus. GroupQueryAttention counts its rows
  # from the mask (see Transmuter.Generation), so another node may not read its present.
  defp cache_graph(model) do
    reads? = fn node, names -> Enum.any?(Transmuter.dependencies(node), names) end
    past? = &String.starts_with?(&1, "past_key_values.")
    gqas = Enum.filter(model.nodes, &(&1.op_type == "GroupQueryAttention"))
    presents = MapSet.new(Enum.flat_map(gqas, &Enum.drop(&1.output, 1)))
    others = model.nodes -- gqas

    case cache_if(model) do
      {branch, graph} ->
        Enum.find(model.nodes -- [branch], &reads?.(&1, past?)) || {branch, graph}

      nil ->
        cond do
          node = Enum.find(others, &reads?.(&1, fn name -> name in presents end)) -> node
          Enum.any?(others, &reads?.(&1, past?)) -> {nil, %GraphProto{node: model.nodes}}
          true -> nil
        end
    end
  end

  # The If node of a merged decoder and its branch with cache, which alone reads the past. The
  # branches of the exports of optimum have nodes with the same names, so a rewrite changes
  # only this branch.
  defp cache_if(model) do
    with node = %{} <-
           Enum.find(model.nodes, &(&1.op_type == "If" and hd(&1.input) == "use_cache_branch")) do
      {node, Enum.find_value(node.attribute, &(&1.name == "then_branch" && &1.g))}
    end
  end

  # Adds the offset to the index of each position read and to the input of each float cast of
  # a shift of L. The reads are found by value, as in cut/3.
  #
  # An offset for each row, {batch, 1}, takes the rank of the value of a read whose axis 0 is
  # the batch (see batch_rank/3): an Unsqueeze at the start of the branch gives the offset of
  # each such rank above 2, as {batch, 1, 1} for the [1, 1, n] positions of Apertus.
  #
  # The reads of positions without a batch axis take the edits of lift/4, and a Range read
  # maps the rows of the buffer to positions for each row (see row_positions/2).
  defp offset_reads(model, branch, graph, reads, per_row?, ranks, lift) do
    name = &(@position_offset <> "." <> &1)
    constant = &Nx.tensor(&1, type: :s64, backend: Nx.BinaryBackend)
    ranks = if per_row?, do: Map.filter(ranks, fn {_read, rank} -> rank > 2 end), else: %{}
    ranked = &name.("rank" <> Integer.to_string(&1))
    axes = &name.("axes" <> Integer.to_string(&1))
    used = ranks |> Map.values() |> Enum.uniq() |> Enum.sort()

    unsqueezes =
      for rank <- used do
        %NodeProto{
          op_type: "Unsqueeze",
          name: ranked.(rank),
          input: [@position_offset, axes.(rank)],
          output: [ranked.(rank)]
        }
      end

    axes = Map.new(used, &{axes.(&1), constant.(Enum.to_list(2..(&1 - 1)))})

    nodes =
      Enum.flat_map(
        graph.node,
        fn node ->
          added =
            case Map.fetch(reads, node) do
              {:ok, nil} ->
                if MapSet.member?(lift.rows, node),
                  do: row_positions(node, name),
                  else: shift_read(node, name)

              {:ok, input} ->
                offset_input(node, input, name)

              :error ->
                Map.get(lift.edits, node, [node])
            end

          case Map.fetch(ranks, node) do
            {:ok, rank} -> Enum.map(added, &rename_input(&1, @position_offset, ranked.(rank)))
            :error -> added
          end
        end
      )

    nodes = unsqueezes ++ nodes

    # The key mask is an input when a node reads it. GroupQueryAttention hides its free rows
    # itself, so Apertus takes only the offset.
    masked? = Enum.any?(nodes, &(@key_mask in &1.input))
    added = if masked?, do: [@position_offset, @key_mask], else: [@position_offset]
    offset = if(per_row?, do: ["batch_size", 1], else: [])
    types = %{@position_offset => {:s64, offset}, @key_mask => {:s64, ["batch_size", "keys"]}}

    %{
      model
      | nodes: replace_graph(model.nodes, branch, nodes),
        static:
          model.static
          |> Map.merge(%{name.("zero") => constant.(0), name.("one") => constant.(1)})
          |> Map.merge(axes)
          |> Map.merge(lift.static),
        inputs: model.inputs ++ added,
        input_types: Map.merge(model.input_types, Map.take(types, added))
    }
  end

  defp rename_input(node, from, to),
    do: %{node | input: Enum.map(node.input, &if(&1 == from, do: to, else: &1))}

  defp replace_graph(_nodes, nil, graph_nodes), do: graph_nodes

  defp replace_graph(nodes, branch, graph_nodes) do
    attribute =
      Enum.map(
        branch.attribute,
        &if(&1.name == "then_branch", do: %{&1 | g: %{&1.g | node: graph_nodes}}, else: &1)
      )

    Enum.map(nodes, &if(&1 == branch, do: %{branch | attribute: attribute}, else: &1))
  end

  defp offset_input(node, input, name) do
    shifted = name.(hd(node.output) <> ".shift")

    [
      %NodeProto{
        op_type: "Add",
        name: shifted,
        input: [Enum.at(node.input, input), @position_offset],
        output: [shifted]
      },
      %{node | input: List.replace_at(node.input, input, shifted)}
    ]
  end

  defp shift_read(node = %{op_type: "AttentionCore"}, _name) do
    padding = List.duplicate("", max(9 - length(node.input), 0))
    [%{node | input: List.replace_at(node.input ++ padding, 8, @key_mask)}]
  end

  defp shift_read(node = %{op_type: "Gather", input: [table, index]}, name) do
    shifted = name.(hd(node.output))

    [
      %NodeProto{
        op_type: "Add",
        name: shifted,
        input: [index, @position_offset],
        output: [shifted]
      },
      %{node | input: [table, shifted]}
    ]
  end

  # A Slice becomes a Gather of the range of its start and end. A Slice from a start with the
  # offset, a value of the call, would have a size that the compiler cannot know, which breaks
  # the graph. The range of the static start and end keeps the size static.
  defp shift_read(node = %{op_type: "Slice", input: [table, starts, ends | _]}, name) do
    out = hd(node.output)
    value = &name.(out <> "." <> &1)

    [
      %NodeProto{
        op_type: "Gather",
        name: value.("start"),
        input: [starts, name.("zero")],
        output: [value.("start")]
      },
      %NodeProto{
        op_type: "Gather",
        name: value.("end"),
        input: [ends, name.("zero")],
        output: [value.("end")]
      },
      %NodeProto{
        op_type: "Range",
        name: value.("range"),
        input: [value.("start"), value.("end"), name.("one")],
        output: [value.("range")]
      },
      %NodeProto{
        op_type: "Add",
        name: value.("index"),
        input: [value.("range"), @position_offset],
        output: [value.("index")]
      },
      %NodeProto{
        op_type: "Gather",
        name: node.name,
        input: [table, value.("index")],
        output: [out]
      }
    ]
  end

  # With one offset for all rows, each value of the Range is a row of the buffer and the new
  # rows, which maps to its position through the key mask: the cumulative count of the valid
  # rows, minus 1, over the union of the rows (see row_positions/2 for an offset for each row). A row that has ended has a prefix of the valid rows of the active rows, so the
  # union is the mask of the active rows, and a free row gets a position that the key mask
  # hides. An index after the last key continues from the last new row with the offset. The
  # Range keeps its start, limit and step, so the output keeps its length.
  defp shift_read(node = %{op_type: "Range", input: [start | _]}, name) do
    value = &name.(hd(node.output) <> "." <> &1)

    step = fn op, inputs, out, attribute ->
      %NodeProto{
        op_type: op,
        name: value.(out),
        input: inputs,
        output: [value.(out)],
        attribute: attribute
      }
    end

    [
      %{node | output: [value.("index")]},
      step.("ReduceMax", [@key_mask, name.("zero")], "valid", [
        %AttributeProto{name: "keepdims", type: :INT, i: 0}
      ]),
      step.("CumSum", [value.("valid"), name.("zero")], "count", []),
      step.("Sub", [value.("count"), name.("one")], "map", []),
      step.("Size", [value.("map")], "keys", []),
      step.("Sub", [value.("keys"), name.("one")], "last", []),
      step.("Min", [value.("index"), value.("last")], "clamped", []),
      step.("Gather", [value.("map"), value.("clamped")], "mapped", []),
      step.("Less", [value.("index"), value.("keys")], "inside", []),
      step.("Add", [value.("index"), @position_offset], "after", []),
      step.("Where", [value.("inside"), value.("mapped"), value.("after")], "positions", []),
      %NodeProto{
        op_type: "CastLike",
        name: node.name,
        input: [value.("positions"), start],
        output: node.output
      }
    ]
  end

  defp shift_read(node = %{op_type: "Cast", input: [value]}, name) do
    shifted = name.(hd(node.output))

    [
      %NodeProto{
        op_type: "Add",
        name: shifted,
        input: [value, @position_offset],
        output: [shifted]
      },
      %{node | input: [shifted]}
    ]
  end

  # The Range of shift_read/2 for an offset for each row, where the rows of a step have gap
  # rows: the cumulative count of the valid rows of each row, minus 1, maps each row of the
  # buffer and of the new rows to its position, and an index after the last key continues
  # from the offset of the row. The positions have the batch on axis 0, {batch, keys}.
  defp row_positions(node = %{op_type: "Range", input: [start | _]}, name) do
    value = &name.(hd(node.output) <> "." <> &1)

    step = fn op, inputs, out, attribute ->
      %NodeProto{
        op_type: op,
        name: value.(out),
        input: inputs,
        output: [value.(out)],
        attribute: attribute
      }
    end

    axis = [%AttributeProto{name: "axis", type: :INT, i: 1}]

    [
      %{node | output: [value.("index")]},
      step.("CumSum", [@key_mask, name.("one")], "count", []),
      step.("Sub", [value.("count"), name.("one")], "map", []),
      step.("Shape", [value.("map")], "shape", []),
      step.("Gather", [value.("shape"), name.("one")], "keys", []),
      step.("Sub", [value.("keys"), name.("one")], "last", []),
      step.("Min", [value.("index"), value.("last")], "clamped", []),
      step.("Gather", [value.("map"), value.("clamped")], "mapped", axis),
      step.("Less", [value.("index"), value.("keys")], "inside", []),
      step.("Add", [value.("index"), @position_offset], "after", []),
      step.("Where", [value.("inside"), value.("mapped"), value.("after")], "positions", []),
      %NodeProto{
        op_type: "CastLike",
        name: node.name,
        input: [value.("positions"), start],
        output: node.output
      }
    ]
  end

  @doc """
  Returns the cache entries that `:append` and `:in_place` keep in buffers: all of them, or in a merged decoder those whose present on the branch with cache reads their past (the self-attention cache). The cross-attention cache of an encoder-decoder export gives an empty present there, so it keeps its exact length.
  """
  def buffers(model) do
    cache = Enum.filter(model.inputs, &String.starts_with?(&1, "past_key_values."))

    case cache_if(model) do
      nil ->
        cache

      {node, graph} ->
        producers = for n <- graph.node, output <- n.output, into: %{}, do: {output, n}

        for name <- cache,
            index =
              Enum.find_index(
                node.output,
                &(&1 == "present." <> String.replace_prefix(name, "past_key_values.", ""))
              ),
            index != nil,
            reads?([Enum.at(graph.output, index).name], name, producers, MapSet.new()),
            do: name
    end
  end

  # The inputs whose values a cross function can read.
  @cross_inputs ~w(encoder_hidden_states encoder_attention_mask)

  @doc """
  Returns the cross function of a merged decoder: a model of the nodes of its branch without cache that give the present of each cache entry outside `buffers/1`, the cross-attention cache, and its outputs are the names of these presents. The values of this cache may read only `encoder_hidden_states` and `encoder_attention_mask`. Other inputs, such as the tokens, may give only shapes, such as the batch size in the cross-attention of Marian, so the caller gives them as templates of one row and one token. Returns `nil` for a decoder without such a cache, and for one whose cache reads the values of another input.
  """
  def cross(model) do
    with {node, _graph} <- cache_if(model),
         cross = Enum.map(cross_cache(model), &("present." <> &1)),
         [_ | _] <- cross do
      without = Enum.find_value(node.attribute, &(&1.name == "else_branch" && &1.g))

      outputs =
        for {name, index} <- Enum.with_index(node.output),
            name in cross,
            do: {name, Enum.at(without.output, index).name}

      nodes = Enum.take_while(model.nodes, &(&1 != node)) ++ without.node
      names = MapSet.new(outputs, &elem(&1, 1))

      local =
        Map.new(without.initializer, fn t ->
          {t.name, t |> Transmuter.Ops.tensor() |> Transmuter.upcast(model.compute_type)}
        end)

      known = ["" | Map.keys(model.params) ++ Map.keys(model.static) ++ Map.keys(local)]
      value = &Transmuter.value_dependencies/1
      values = reads(slice(nodes, names, value), value, known)
      nodes = slice(nodes, names, &Transmuter.dependencies/1)
      inputs = reads(nodes, &Transmuter.dependencies/1, known)

      # The branch names its outputs, which can differ from the outputs of the If.
      renames =
        for {outer, inner} <- outputs,
            inner != outer,
            do: %NodeProto{op_type: "Identity", name: outer, input: [inner], output: [outer]}

      if values -- @cross_inputs == [] and inputs -- model.inputs == [] do
        %{
          model
          | nodes: nodes ++ renames,
            static: Map.merge(model.static, local),
            inputs: inputs,
            input_types: Map.take(model.input_types, inputs),
            outputs: cross,
            output_types: Map.take(model.output_types, cross)
        }
      end
    else
      _other -> nil
    end
  end

  @doc """
  Returns the inputs of `model` that a step on the branch with cache reads: the inputs of the nodes outside the `If` and of its branch with cache. Without an `If`, all inputs.
  """
  def cache_reads(model) do
    case cache_if(model) do
      nil ->
        model.inputs

      {node, graph} ->
        nodes = List.delete(model.nodes, node) ++ graph.node
        Enum.filter(reads(nodes, &Transmuter.dependencies/1, []), &(&1 in model.inputs))
    end
  end

  # The cache entries outside the buffers, without the "past_key_values." prefix.
  defp cross_cache(model) do
    buffers = buffers(model)
    for "past_key_values." <> name = full <- model.inputs, full not in buffers, do: name
  end

  # The nodes of `nodes`, in order, that the values of `names` need through `dependencies`.
  defp slice(nodes, names, dependencies) do
    {kept, _names} =
      nodes
      |> Enum.reverse()
      |> Enum.reduce({[], names}, fn node, {kept, names} ->
        if Enum.any?(node.output, &MapSet.member?(names, &1)),
          do: {[node | kept], MapSet.union(names, MapSet.new(dependencies.(node)))},
          else: {kept, names}
      end)

    kept
  end

  # The names that `nodes` read through `dependencies` and do not make, other than `known`.
  defp reads(nodes, dependencies, known) do
    made = Enum.flat_map(nodes, & &1.output)
    Enum.uniq(Enum.flat_map(nodes, dependencies)) -- (made ++ known)
  end

  defp where(nil), do: "a graph input"
  defp where(node), do: "#{node.op_type} #{node.name}"

  # Whether the values of a name of `names` depend on the values of `target`.
  defp reads?([], _target, _producers, _seen), do: false

  defp reads?([name | rest], target, producers, seen) do
    node = producers[name]

    cond do
      name == target ->
        true

      name in seen or node == nil ->
        reads?(rest, target, producers, seen)

      true ->
        reads?(
          Transmuter.value_dependencies(node) ++ rest,
          target,
          producers,
          MapSet.put(seen, name)
        )
    end
  end
end
