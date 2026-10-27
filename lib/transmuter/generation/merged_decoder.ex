defmodule Transmuter.Generation.MergedDecoder do
  @moduledoc """
  Moves a merged decoder of optimum without `position_ids` from a cache of exact length (`:grow`) to a buffer (`:append`), see "Merged decoders" in `Transmuter.Generation`.

  Such a decoder takes the length L of its cache from the shape of the past. With a buffer of C rows, L is C, which is right for the shapes and the causal masks: the keys are the buffer and the new rows, and query i of a step sees the keys up to C + i. Only the positions need the valid rows v. The rewrite changes only the branch with cache of the `If` on `use_cache_branch`, because the branches of the exports have nodes with the same names. It adds two inputs:

    * `transmuter.position_offset`, v - C. Each position read, a `Gather` on axis 0 or a `Slice` on axis 0 with step 1 of a 2-D static table whose index is a shift of L (L plus values without L), gets it added to its index. Each cast of a shift of L to a float, such as the position of a rotary angle, gets it added to its integer input, so the float values see v and not C. The integer uses of L keep C, because they index the buffer. The offset has one value for each row, of shape `{batch, 1}`, when every read is a `Gather` of a position table whose index has a leading axis of size 1 (as the positions of GPT-2), no later `Reshape` or `Expand` has a static first dimension of 1, and no prefix (below) takes the rows of the buffer as positions. Then the rows of a step can have different valid rows and gap rows, so Generation decodes the rows of prompts of different lengths together. Otherwise the offset is one scalar for all rows, which keeps the shape of each value that it is added to, and Generation runs the rows of each prompt length as a group of their own, without gap rows, so that buffer row j holds position j. `:per_row_offset` in the returned model tells which.

    * `transmuter.key_mask`, the attention mask of `:append`, 1 for the valid rows and the new rows and 0 for the gap rows and the free rows. Each `AttentionCore` whose keys read a buffer gets it as input 8, so the free rows get no weight, also when the decoder takes no attention mask, as in Whisper. The rows of a buffer may reach only the keys and the values of these cores, the present cache and Concat, Reshape, Transpose (with the batch first), Unsqueeze, Squeeze, Identity and Cast. Another reader, such as an attention that `Transmuter.Fusion` did not replace, keeps the decoder in `:grow`.

  A walk follows L forward through the branch. A value that depends on L is a shift of L, another integer, a boolean of a comparison, or the shape of a tensor whose length follows the buffer: the buffer, the attention mask, and their results through Concat, Cast, Identity and Transpose. A dimension of such a shape is a shift of L when it is the length, and no L when it is another dimension. These uses keep the decoder in `:grow`: a cast to a float of another value of L, a 2-D static table read on axis 0 with another index of L, and the other reads of static tables at a value of L that is not a shift. A static table that a shift of L indexes in another way, such as the causal table of GPT-2, counts the rows of the buffer, so the buffer and the new token must fit its axes: its smallest axis minus 1 is the largest capacity (1023 for GPT-2). A `Slice` from 0 to a shift of L with step 1 of a tensor that the graph made, whose content has no L or only comparisons of L, is a prefix: with a buffer it keeps C + n rows in place of v + n, which hold every row that a step reads, such as the causal mask of moonshine, which compares the positions of L with the key index. A prefix of a static table, such as the rotary table of the merged llama2.c that its nested `If` nodes squeeze, keeps the decoder in `:grow`.

  A `Range` from 0 to a shift of L, with a step without L, gives the positions of all keys of a step, as the relative positions of T5. With a buffer, its index is the row of the buffer, and the new rows come after the C rows of the buffer. So the rewrite keeps the `Range` and maps each of its values to a position through the key mask: the cumulative count of the valid rows, minus 1, over the union of the rows of the step, and the offset after the last key. A row that has ended has a prefix of the valid rows of the active rows, so the union is the mask of the active rows, and a valid row gets its position and a new row v + i. The `Range` keeps its start, limit and step, so its output keeps its length. The walk follows these positions through moves, arithmetic and comparisons with values without L, float casts and reads of static tables, such as the bucket table of T5. A use that mixes them with other values of L, such as the causal mask of moonshine, which compares the key index with the positions of L, keeps that range as an index of the buffer, and the walk runs again (at most once for each range).

  A `Mul` of a shift of L by a value without L also makes positions, as the positions of M2M100 and NLLB, (cumsum(mask) + L) * mask + padding_idx. The offset goes on its shift input, so a real token gets v + cumsum and a padding token keeps padding_idx, and the walk follows the result as positions. A position is a position only in position uses: arithmetic and comparisons with values without L, float casts, and reads of static tables. A use as a size or a bound (the shape of a Reshape, Expand or Tile, a Slice start or end, a Range or ConstantOfShape input) or as an index into a tensor that is not a static table needs the index of the buffer, so it sends the range or the `Mul` back to the index space, and the walk runs again.
  """

  require Logger

  alias Transmuter.Onnx.{AttributeProto, NodeProto}

  @position_offset "transmuter.position_offset"
  @key_mask "transmuter.key_mask"

  @doc false
  def position_offset, do: @position_offset

  @doc false
  def key_mask, do: @key_mask

  @doc """
  Returns `model` with the position offset and the key mask in the branch with cache, its
  largest capacity in `:max_capacity` and the shape of its offset in `:per_row_offset`, or
  `model` itself when the rules do not hold.
  """
  def rewrite(model) do
    gqa? = Enum.any?(model.nodes, &(&1.op_type == "GroupQueryAttention"))
    past? = &String.starts_with?(&1, "past_key_values.")

    with false <- "position_ids" in model.inputs or gqa?,
         {branch, graph} <- cache_if(model),
         nil <-
           Enum.find(model.nodes -- [branch], &Enum.any?(Transmuter.dependencies(&1), past?)),
         {:ok, cores} <- buffer_cores(model, graph),
         {:ok, [_ | _] = reads, limit, prefixes} <- past_uses(model, graph) do
      per_row? = per_row?(model, graph, reads, prefixes)

      model
      |> offset_reads(branch, graph, Map.new(reads ++ cores, &read_entry/1), per_row?)
      |> Map.put(:max_capacity, limit && limit - 1)
      |> Map.put(:per_row_offset, per_row?)
    else
      {:bail, node} ->
        exact_cache(model, node)

      node = %NodeProto{} ->
        exact_cache(model, node)

      _other ->
        model
    end
  end

  # The offset is one value for each row, of shape {batch, 1}, when the rows of a step can have
  # different valid rows: every read is a Gather of a position table whose index has a leading
  # axis of size 1, so the offset gives the index the batch and keeps its rank, no later Reshape
  # or Expand has a static first dimension of 1, and no prefix takes the rows of the buffer as
  # positions. Otherwise it is one value for all rows, and Generation runs one group for each
  # prompt length.
  defp per_row?(model, graph, reads, prefixes) do
    scope = static_scope(model, graph)
    producers = for node <- graph.node, output <- node.output, into: %{}, do: {output, node}

    reason =
      cond do
        prefixes != [] ->
          {"a prefix of the rows of the buffer", hd(prefixes)}

        read = Enum.find(reads, &(not batch_index?(&1, producers, scope))) ->
          {"a read that needs one offset for all rows", read_node(read)}

        node = batch_reshape(graph, reads, scope) ->
          {"a reshape to a batch of 1", node}

        true ->
          nil
      end

    case reason do
      nil ->
        true

      {what, node} ->
        Logger.debug(
          "generation runs one group for each prompt length of the merged decoder, because of #{what} at #{where(node)}"
        )

        false
    end
  end

  # A Gather on axis 0 whose index has a leading axis of size 1: an Unsqueeze on axis 0, or a
  # Reshape to a shape whose first static value is 1 or -1, with more values after it.
  defp batch_index?(
         node = %NodeProto{op_type: "Gather", input: [_table, index]},
         producers,
         scope
       ) do
    axis = Enum.find_value(node.attribute, 0, &(&1.name == "axis" && &1.i))
    list = &(scope.values[&1] && Nx.to_flat_list(scope.values[&1]))

    axis == 0 and
      case producers[index] do
        %{op_type: "Unsqueeze", input: [_data, axes]} ->
          list.(axes) == [0]

        %{op_type: "Unsqueeze", attribute: attribute} ->
          Enum.any?(attribute, &(&1.name == "axes" and &1.ints == [0]))

        %{op_type: "Reshape", input: [_data, shape]} ->
          leading_one?(shape, producers, list)

        _other ->
          false
      end
  end

  defp batch_index?(_read, _producers, _scope), do: false

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
      shape = Enum.at(node.input, 1, "")
      first = scope.values[shape] && scope.values[shape] |> Nx.to_flat_list() |> List.first()

      cond do
        not Enum.any?(node.input, &MapSet.member?(after_reads, &1)) -> {:cont, after_reads}
        node.op_type in ["Reshape", "Expand"] and first == 1 -> {:halt, node}
        true -> {:cont, MapSet.union(after_reads, MapSet.new(node.output))}
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
    Logger.debug(
      "generation keeps a cache of exact length, because #{where(node)} reads the cache or its length"
    )

    model
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

  # The position reads of the branch with cache, and the smallest axis of the other static
  # tables that a length of the buffer indexes, such as the causal table of GPT-2, which the
  # rows of the buffer and the new token must fit. The values that depend on L have a kind:
  # :shift for L plus other values, :int for other integers, :bool for comparisons and
  # {:dims, axis, rank} for the shape of a tensor whose length follows the buffer. The lengths
  # map gives the axis of that length from the end and the rank, or :unknown: axis 2 of 4 for
  # the buffers of the cache, axis 1 of 2 for the attention mask, and the axes after Concat,
  # Cast, Identity and Transpose.
  defp past_uses(model, graph, index_ranges \\ MapSet.new()) do
    scope = static_scope(model, graph)
    lengths = Map.new(buffers(model), &{&1, {-2, 4}})

    lengths =
      if "attention_mask" in model.inputs,
        do: Map.put(lengths, "attention_mask", {-1, 2}),
        else: lengths

    graph.node
    |> Enum.reduce_while({[], %{}, lengths, nil, []}, fn node,
                                                         {reads, kinds, lengths, limit, prefixes} ->
      lengths = Map.merge(lengths, track(node, lengths, kinds))

      case past_use(%{scope: scope, lengths: lengths, index: index_ranges}, node, kinds) do
        {:read, node} ->
          {:cont, {[node | reads], kinds, lengths, limit, prefixes}}

        {:positions, node, outputs} ->
          {:cont, {[node | reads], Map.merge(kinds, outputs), lengths, limit, prefixes}}

        {:kinds, outputs} ->
          {:cont, {reads, Map.merge(kinds, outputs), lengths, limit, prefixes}}

        {:prefix, outputs} ->
          {:cont, {reads, Map.merge(kinds, outputs), lengths, limit, [node | prefixes]}}

        {:limit, size, outputs} ->
          {:cont, {reads, Map.merge(kinds, outputs), lengths, min(limit, size), prefixes}}

        bail ->
          {:halt, bail}
      end
    end)
    |> case do
      {:bail, _node} = bail -> bail
      {:mixed, ranges, node} -> index_space(model, graph, index_ranges, ranges, node)
      {reads, _kinds, _lengths, limit, prefixes} -> {:ok, reads, limit, prefixes}
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
  # takes its shape from L. An attention core gives the shape of its queries, and a Gather the
  # shape of its index.
  defp track(node, lengths, kinds) do
    data = if node.op_type == "Gather", do: Enum.take(node.input, 1), else: node.input
    axes = for name <- data, Map.has_key?(lengths, name), uniq: true, do: lengths[name]

    axis =
      case {node.op_type, axes} do
        {op, _} when op in ["Shape", "Size", "AttentionCore"] -> nil
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
    same? = &((not dep?.(&1) and not dep?.(&2)) or (kinds[&1] == :shift and kinds[&2] == :shift))

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

      key_positions?(walk, node, deps) ->
        {:positions, node, outputs(node, {:position, MapSet.new(node.output)})}

      input = candidate(walk, node, deps) ->
        {:positions, {node, input}, outputs(node, {:position, MapSet.new(node.output)})}

      node.op_type in ["Gather", "Slice", "GatherElements", "GatherND"] ->
        gather_use(scope, node, kinds, deps)

      node.op_type == "If" ->
        {:bail, node}

      true ->
        value_use(node, deps)
    end
  end

  # A read of a static table, a dimension of a shape, a slice of a causal table, a prefix of a
  # mask, or L as an index into another tensor, such as a rotary table that the graph makes,
  # which stops the rewrite.
  defp gather_use(scope, node, kinds, deps) do
    data = kinds[hd(node.input)]
    table_index? = &(&1 == 1 and node.op_type == "Gather" and hd(node.input) in scope.static)
    bounds = for {i, {:position, r}} <- deps, i > 0, not table_index?.(i), do: r

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
        {:prefix, if(data, do: outputs(node, data), else: %{})}

      Enum.any?(deps, &(elem(&1, 0) > 0)) ->
        {:bail, node}

      true ->
        value_use(node, deps)
    end
  end

  # A Slice from 0 to a shift of L with step 1, a prefix. With a buffer it keeps C + n rows in
  # place of v + n, which hold every row that a step reads, and it shifts nothing. Its output
  # keeps the kind of its data.
  defp prefix?(scope, %{op_type: "Slice", input: [_data, starts, ends | rest]}, kinds) do
    steps = Enum.at(rest, 1, "")
    list = &(scope.values[&1] && Nx.to_flat_list(scope.values[&1]))
    all? = fn name, value -> list.(name) != nil and Enum.all?(list.(name), &(&1 == value)) end
    all?.(starts, 0) and kinds[ends] == :shift and (steps == "" or all?.(steps, 1))
  end

  defp prefix?(_scope, _node, _kinds), do: false

  # A dimension of a shape: the length is a shift of L, another dimension does not depend on it.
  defp dims(scope, node = %{op_type: op, input: [shape | rest]}, kinds) do
    {:dims, axis, rank} = kinds[shape]
    list = &(scope.values[&1] && Nx.to_flat_list(scope.values[&1]))
    from_end = &if(&1 >= 0 and &1 < rank, do: &1 - rank, else: &1)

    picked =
      case {op, Enum.map(rest, list)} do
        {"Gather", [[index]]} ->
          if from_end.(index) == axis, do: :shift

        {"Slice", [[start], [stop] | axes]} when axes in [[], [[0]]] ->
          {start, stop} = {from_end.(start), if(stop >= rank, do: 0, else: from_end.(stop))}

          if start <= axis and stop > axis,
            do: if(start == axis and stop == axis + 1, do: :shift, else: :int)

        _other ->
          :int
      end

    {:kinds, if(picked, do: outputs(node, picked), else: %{})}
  end

  # The kind of the outputs of a node that reads values of L.
  defp value_use(node, deps) do
    inputs = Map.get(@value_inputs, node.op_type, Enum.to_list(0..(length(node.input) - 1)//1))
    value_kinds = for {i, kind} <- deps, i in inputs, do: kind

    # Comparisons and causal tables are masks, which hold for the rows of a buffer.
    values = Enum.reject(value_kinds, &(&1 == :bool or match?({:causal, _, _}, &1)))
    {positions, others} = Enum.split_with(values, &match?({:position, _}, &1))

    ranges =
      Enum.reduce(positions, MapSet.new(), fn {:position, r}, acc -> MapSet.union(acc, r) end)

    bounds =
      for {i, {:position, r}} <- deps,
          i not in inputs or node.op_type in @sizes,
          do: r

    cond do
      bounds != [] -> {:mixed, Enum.reduce(bounds, &MapSet.union/2), node}
      positions != [] and others != [] -> {:mixed, ranges, node}
      positions != [] -> position_use(node, ranges)
      node.op_type in @comparisons -> {:kinds, outputs(node, :bool)}
      values == [] -> {:kinds, %{}}
      node.op_type == "Cast" and float_cast?(node) -> float_use(node, deps)
      true -> {:kinds, outputs(node, kind(node, deps))}
    end
  end

  defp outputs(node, kind), do: Map.new(node.output, &{&1, kind})

  # The operators that make a position from a shift of L: a Mul by a value without L, as the
  # positions of M2M100 and NLLB, (cumsum(mask) + L) * mask + padding_idx. The offset goes on
  # the shift input, so a real token gets v + cumsum and a padding token keeps padding_idx.
  # Gives the index of the shift input, unless an earlier walk sent this node back to the index
  # space (see value_use/2).
  @candidates ~w(Mul)

  defp candidate(walk, node = %{op_type: op}, [{input, :shift}]) when op in @candidates do
    if length(node.input) == 2 and not MapSet.member?(walk.index, hd(node.output)), do: input
  end

  defp candidate(_walk, _node, _deps), do: nil

  # The positions of all keys of a step: a Range from 0 to a shift of L with a step without L,
  # unless an earlier walk found that a use compares or mixes it with L (see value_use/2), as
  # the causal mask of moonshine compares the key index with the positions of L.
  defp key_positions?(walk, node = %{op_type: "Range", input: [start, _limit | _]}, deps) do
    value = walk.scope.values[start]

    deps == [{1, :shift}] and value != nil and Nx.to_flat_list(value) == [0] and
      not MapSet.member?(walk.index, hd(node.output))
  end

  defp key_positions?(_walk, _node, _deps), do: false

  # A use of key positions (see key_positions?/3) alone: a float cast or a comparison ends
  # them, and other operators keep them positions.
  defp position_use(node, ranges) do
    cond do
      node.op_type in @comparisons -> {:kinds, outputs(node, :bool)}
      node.op_type == "Cast" and float_cast?(node) -> {:kinds, %{}}
      true -> {:kinds, outputs(node, {:position, ranges})}
    end
  end

  # A float cast of a shift of L gets the offset on its input (see offset_reads/4), so its
  # float value no longer depends on the buffer. A float cast of another value of L stops the
  # rewrite.
  defp float_use(node, deps) do
    if Enum.all?(deps, &(elem(&1, 1) == :shift)), do: {:read, node}, else: {:bail, node}
  end

  # L + c, c + L and L - c are shifts, c - L is not.
  defp kind(%{op_type: "Add"}, [{_i, :shift}]), do: :shift
  defp kind(%{op_type: "Sub"}, [{0, :shift}]), do: :shift

  # A range from L, whose step does not depend on L.
  defp kind(%{op_type: "Range"}, deps) do
    if List.keyfind(deps, 0, 0) == {0, :shift} and not List.keymember?(deps, 2, 0),
      do: :shift,
      else: :int
  end

  defp kind(%{op_type: op}, deps) when op in @shifts do
    if Enum.all?(deps, &(elem(&1, 1) == :shift)), do: :shift, else: :int
  end

  defp kind(_node, _deps), do: :int

  defp float_cast?(node) do
    # FLOAT, FLOAT16, DOUBLE and BFLOAT16 of TensorProto.DataType.
    Enum.any?(node.attribute, &(&1.name == "to" and &1.i in [1, 10, 11, 16]))
  end

  # A position read, or a Slice of a causal table (1 where the column is at most the row, as
  # the bias of GPT-2) on its last two axes. Its rows and columns count the rows of the buffer,
  # which is right while the buffer and the new token fit the table. Another read of a static
  # table at L, such as a local attention window, stops the rewrite.
  defp static_read(scope, node, kinds) do
    table = scope.values[hd(node.input)]

    cond do
      Nx.rank(table) == 2 and position_index?(scope, node, kinds) ->
        {:read, node}

      node.op_type == "Gather" and match?({:position, _}, kinds[Enum.at(node.input, 1)]) ->
        {:kinds, %{}}

      node.op_type == "Slice" and causal?(table) ->
        causal_slice(scope, node, kinds, {Nx.rank(table), Nx.axis_size(table, -1)})

      true ->
        {:bail, node}
    end
  end

  # A Slice of a causal table, or of a slice of it, on its last two axes, from and to shifts of L.
  defp causal_slice(scope, node = %{input: [_, starts, ends, axes | _]}, kinds, {rank, size}) do
    axes = scope.values[axes] && Nx.to_flat_list(scope.values[axes])
    last = [rank - 2, rank - 1, -2, -1]

    if kinds[starts] in [nil, :shift] and kinds[ends] in [nil, :shift] and axes != nil and
         Enum.all?(axes, &(&1 in last)),
       do: {:limit, size, outputs(node, {:causal, rank, size})},
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

  # The index of a position read: a Gather on axis 0, or a Slice on axis 0 with step 1, whose
  # index or start is a shift of L.
  defp position_index?(_scope, node = %{op_type: "Gather", input: [_table, index]}, kinds) do
    axis = Enum.find_value(node.attribute, 0, &(&1.name == "axis" && &1.i))
    axis == 0 and kinds[index] == :shift
  end

  defp position_index?(scope, %{op_type: "Slice", input: [_, starts, ends | rest]}, kinds) do
    [axes, steps] = Enum.take(rest ++ ["", ""], 2)
    list = &(scope.values[&1] && Nx.to_flat_list(scope.values[&1]))

    kinds[starts] == :shift and kinds[ends] == :shift and list.(axes) == [0] and
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
  defp offset_reads(model, branch, graph, reads, per_row?) do
    name = &(@position_offset <> "." <> &1)
    constant = &Nx.tensor(&1, type: :s64, backend: Nx.BinaryBackend)

    nodes =
      Enum.flat_map(
        graph.node,
        fn node ->
          case Map.fetch(reads, node) do
            {:ok, nil} -> shift_read(node, name)
            {:ok, input} -> offset_input(node, input, name)
            :error -> [node]
          end
        end
      )

    graph = %{graph | node: nodes}

    attribute =
      Enum.map(branch.attribute, &if(&1.name == "then_branch", do: %{&1 | g: graph}, else: &1))

    %{
      model
      | nodes:
          Enum.map(model.nodes, &if(&1 == branch, do: %{branch | attribute: attribute}, else: &1)),
        static:
          Map.merge(model.static, %{name.("zero") => constant.(0), name.("one") => constant.(1)}),
        inputs: model.inputs ++ [@position_offset, @key_mask],
        input_types:
          Map.merge(model.input_types, %{
            @position_offset => {:s64, if(per_row?, do: ["batch_size", 1], else: [])},
            @key_mask => {:s64, ["batch_size", "keys"]}
          })
    }
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

  # Each value of the Range is a row of the buffer and the new rows, which maps to its position
  # through the key mask: the cumulative count of the valid rows, minus 1, over the union of
  # the rows. A row that has ended has a prefix of the valid rows of the active rows, so the
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
