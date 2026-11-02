defmodule Transmuter.Ops.Attention do
  @moduledoc """
  The `Nx.block/4` of the attention core: `softmax(q · k * scale + masks) · v` for q `{B, N, S, H}` and k and v `{B, N_kv, T, H}`. The inputs are q, k, v and the masks that `masks` names, in that order: `:bias` (a float mask to add, or a u8 mask with 1 for the keys to keep) and `:hidden` (a u8 mask with 1 for the hidden keys). The output is the attention, in the type of q.

  The default function is the attention core of `Transmuter.Ops.Transformer`, with the kernels of `EMLX.Fast` on EMLX. Every backend and compiler runs it, except for one case: on CUDA, EXLA runs the block as the fused flash attention of cuDNN (`__cudnn$fmhaSoftmax` and `__cudnn$fmhaScaleBiasSoftmax`, as `jax._src.cudnn.fused_attention_stablehlo` lowers them) when q, k and v are f16 or bf16 of one type with a head size of at most 128 and a multiple of 8, there is no softcap and no hidden mask, a bias has the full key axis and a query axis of 1 or S, and a causal mask is aligned to the first query (S = T) or hides nothing (S = 1). The kernel accumulates in f32 and gives f16 or bf16, so its output can differ from the Nx core in f32 by one unit in the last place.

  The cuDNN targets read an opaque string config and return a workspace after the output, so the `EXLA.CustomCall` implementation exists only when `EXLA.CustomCall.Spec` has the `:api_version` and `:extra_results` fields. With an EXLA without them, EXLA runs the default function.
  """

  defstruct [:scale, :softcap, :causal, masks: []]

  alias EXLA.CustomCall.Spec

  if Code.ensure_loaded?(Spec) and
       Enum.all?([:api_version, :extra_results], &Map.has_key?(Spec.__struct__(), &1)) do
    defimpl EXLA.CustomCall do
      alias EXLA.CustomCall.Spec

      def call(attention, %{shape: out_shape}, [q, k, v | masks], %{platform: :cuda}) do
        {b, heads, s, _head_size} = q.shape
        t = elem(k.shape, 2)
        bias = if attention.masks == [:bias], do: hd(masks)

        if kernel_options?(attention, s, t) and kernel_inputs?(q, k, v, out_shape) and
             kernel_bias?(bias, {b, heads, s, t}) do
          {:ok, spec(attention, q, bias, {b, heads, s, t})}
        else
          :skip
        end
      end

      def call(_attention, _out, _args, _client), do: :skip

      # The kernel has no softcap and no mask of hidden keys, and its causal mask is aligned
      # to the first query.
      defp kernel_options?(attention, s, t),
        do:
          attention.softcap == 0 and attention.masks in [[], [:bias]] and
            (not attention.causal or s == t or s == 1)

      # f16 or bf16 of one type, a head size of at most 128 and a multiple of 8, and key and
      # value heads that group the query heads.
      defp kernel_inputs?(q, k, v, out_shape) do
        {_b, heads, _s, head_size} = q.shape

        q.type in [{:f, 16}, {:bf, 16}] and k.type == q.type and v.type == q.type and
          v.shape == k.shape and out_shape == q.shape and rem(heads, elem(k.shape, 1)) == 0 and
          head_size <= 128 and rem(head_size, 8) == 0
      end

      # cuDNN takes a float bias of rank 4 with the full key axis. The batch, head and query
      # axes can be 1.
      defp kernel_bias?(nil, _scores), do: true

      defp kernel_bias?(%{type: {kind, _}, shape: {bb, bn, bs, bt}}, {b, heads, s, t}),
        do: kind in [:f, :bf] and bb in [1, b] and bn in [1, heads] and bs in [1, s] and bt == t

      defp kernel_bias?(_bias, _scores), do: false

      defp spec(attention, q, bias, {b, heads, s, t}) do
        layout = "dense<[3, 2, 1, 0]> : tensor<4xindex>"
        operands = if bias, do: 4, else: 3

        %Spec{
          call_target_name:
            if(bias, do: "__cudnn$fmhaScaleBiasSoftmax", else: "__cudnn$fmhaSoftmax"),
          api_version: 2,
          backend_config: JSON.encode!(config(attention, q.type, {b, heads, s, t})),
          mlir_attributes: [
            {"operand_layouts",
             "[" <> Enum.map_join(1..operands, ", ", fn _ -> layout end) <> "]"},
            {"result_layouts", "[#{layout}, dense<0> : tensor<1xindex>]"}
          ],
          # cuDNN takes the workspace that XLA allocates as the last result. XLA gives it
          # the size of the cuDNN graph, and an empty workspace must stay empty.
          extra_results: [{{:u, 8}, {0}}],
          operand_element_types: List.duplicate(q.type, operands)
        }
      end

      # The GpuBackendConfig of jax._src.cudnn.fused_attention_stablehlo for the BNTH
      # layout, in inference. The causal mask of cuDNN is aligned to the first query, which
      # is the alignment of the attention core when S = T. When S = 1 it hides nothing.
      defp config(attention, type, {b, heads, s, t}) do
        dims = fn lc, rc ->
          %{
            lhs_contracting_dimensions: [to_string(lc)],
            rhs_contracting_dimensions: [to_string(rc)],
            lhs_batch_dimensions: ["0", "1"],
            rhs_batch_dimensions: ["0", "1"]
          }
        end

        %{
          operation_queue_id: "0",
          wait_on_operation_queues: [],
          cudnn_fmha_backend_config: %{
            algorithm: %{
              algo_id: "0",
              math_type: "TENSOR_OP_MATH",
              tuning_knobs: %{"17" => "1", "24" => "0"},
              is_cudnn_frontend: true,
              workspace_size: "0"
            },
            fmha_scale: attention.scale * 1.0,
            intermediate_tensor_shape: %{
              element_type: if(type == {:f, 16}, do: "F16", else: "BF16"),
              dimensions: Enum.map([b, heads, s, t], &to_string/1),
              tuple_shapes: [],
              layout: %{minor_to_major: ["3", "2", "1", "0"]},
              is_dynamic_dimension: [false, false, false, false]
            },
            is_flash_attention: true,
            mask_type: if(attention.causal and s == t and s > 1, do: "CAUSAL", else: "NO_MASK"),
            bmm1_dot_dimension_numbers: dims.(3, 3),
            bmm2_dot_dimension_numbers: dims.(3, 2),
            dropout_rate: 0.0,
            seed: 42,
            sliding_window_length: 0,
            max_seg_per_batch: 1,
            is_paged_attention: false
          }
        }
      end
    end
  end
end
