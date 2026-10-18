"""Builds single-op ONNX cases with ORT reference outputs, in the layout of prepare.py.

Each case is a small graph around one operator, so a mismatch points at that operator.
Usage: op_cases.py cases_dir
"""

import json
import os
import sys
from collections import Counter

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

cases_dir = sys.argv[1]
rng = np.random.default_rng(0)
NX = {"int64": "s64", "int32": "s32", "float32": "f32", "uint8": "u8", "int8": "s8", "bool": "u8", "float16": "f16"}
MS = "com.microsoft"


def f32(*shape, scale=1.0):
    return (rng.standard_normal(shape) * scale).astype(np.float32)


def case(name, nodes, inputs, initializers=None, outputs=None, opset=23, ms=False, static=()):
    """inputs/initializers: dict name -> array. outputs: list of names (defaults to last node outputs)."""
    initializers = initializers or {}
    outputs = outputs or [o for o in nodes[-1].output if o]
    graph = helper.make_graph(
        nodes,
        name,
        [helper.make_tensor_value_info(k, helper.np_dtype_to_tensor_dtype(v.dtype), v.shape) for k, v in inputs.items()],
        [onnx.ValueInfoProto(name=o) for o in outputs],
        [numpy_helper.from_array(v, k) for k, v in initializers.items()],
    )
    imports = [helper.make_opsetid("", opset)] + ([helper.make_opsetid(MS, 1)] if ms else [])
    model = helper.make_model(graph, opset_imports=imports, ir_version=10)
    path = os.path.join(cases_dir, name)
    os.makedirs(path, exist_ok=True)
    onnx_path = os.path.join(path, "model.onnx")
    onnx.save(model, onnx_path)
    meta = {"repo": "op/" + name, "task": "op", "file": "model.onnx", "onnx_path": onnx_path, "unoptimized_ref": True,
            "ops": dict(Counter((f"{n.domain}:" if n.domain else "") + n.op_type for n in nodes)), "static_inputs": list(static)}
    try:
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        opts.log_severity_level = 3
        sess = ort.InferenceSession(onnx_path, opts, providers=["CPUExecutionProvider"])
        outs = sess.run(outputs, inputs)
        save = lambda a, f: (np.ascontiguousarray(a.astype(np.uint8) if a.dtype == np.bool_ else a).tofile(os.path.join(path, f)), {"dtype": NX[str(a.dtype)], "shape": list(a.shape)})[1]
        meta["inputs"] = [{"name": k, "file": f"in_{i}.bin", **save(v, f"in_{i}.bin")} for i, (k, v) in enumerate(inputs.items())]
        meta["outputs"] = [{"name": k, "file": f"out_{i}.bin", **save(np.asarray(v), f"out_{i}.bin")} for i, (k, v) in enumerate(zip(outputs, outs))]
        meta["ort_ok"] = True
    except Exception as e:
        meta["ort_ok"] = False
        meta["error"] = str(e).splitlines()[0][:300]
    json.dump(meta, open(os.path.join(path, "meta.json"), "w"), indent=1)
    print(("OK   " if meta["ort_ok"] else "FAIL ") + name, meta.get("error", ""), flush=True)


def node(op, inputs, outputs, domain=None, **attrs):
    return helper.make_node(op, inputs, outputs, domain=domain, **attrs)


# ---------- quantization ----------

def pack(q, bits):
    """Pack unsigned ints < 2**bits along the last axis, lowest bits first."""
    per = 8 // bits
    q = q.reshape(*q.shape[:-1], -1, per).astype(np.uint8)
    return sum((q[..., i] << (bits * i)) for i in range(per)).astype(np.uint8)


for bits, K, N, bs, zp_kind, bias in [(4, 64, 24, 16, "packed", False), (4, 72, 16, 32, "none", True), (8, 64, 8, 32, "packed", False), (4, 64, 8, 16, "float", False), (2, 64, 8, 16, "packed", False)]:
    kb = -(-K // bs)
    q = rng.integers(0, 2**bits, (N, kb * bs))
    B = pack(q, bits).reshape(N, kb, bs * bits // 8)
    scales = (rng.random((N * kb,)) * 0.1 + 0.01).astype(np.float32)
    inits = {"B": B, "scales": scales}
    names = ["A", "B", "scales"]
    if zp_kind == "packed":
        zq = rng.integers(0, 2**bits, (N, -(-kb * bits // 8) * 8 // bits))
        inits["zp"] = pack(zq, bits).reshape(-1)
        names.append("zp")
    elif zp_kind == "float":
        inits["zp"] = (rng.random((N * kb,)) * 2**bits).astype(np.float32)
        names.append("zp")
    if bias:
        names += [""] * (4 - len(names)) + ["", "bias"] if len(names) < 5 else ["bias"]
        inits["bias"] = f32(N)
    case(f"MatMulNBits_b{bits}_bs{bs}_{zp_kind}{'_bias' if bias else ''}",
         [node("MatMulNBits", names, ["Y"], domain=MS, K=K, N=N, bits=bits, block_size=bs)],
         {"A": f32(2, 3, K)}, inits, ms=True)

for bits, bs in [(4, 16), (8, 32)]:
    V, D = 50, 64
    q = rng.integers(0, 2**bits, (V, D))
    data = pack(q, bits)
    scales = (rng.random((V, D // bs)) * 0.1 + 0.01).astype(np.float32)
    zq = rng.integers(0, 2**bits, (V, D // bs))
    zp = pack(zq, bits) if bits < 8 else zq.astype(np.uint8)
    case(f"GatherBlockQuantized_b{bits}",
         [node("GatherBlockQuantized", ["data", "indices", "scales", "zp"], ["Y"], domain=MS, gather_axis=0, quantize_axis=1, block_size=bs, bits=bits)],
         {"indices": np.array([[3, 0, 49, 7]], dtype=np.int64)}, {"data": data, "scales": scales, "zp": zp}, ms=True)

case("DequantizeLinear_per_tensor", [node("DequantizeLinear", ["x", "s", "z"], ["y"])],
     {"x": rng.integers(0, 255, (3, 4)).astype(np.uint8)}, {"s": np.array(0.05, np.float32), "z": np.array(128, np.uint8)})
case("DequantizeLinear_per_axis", [node("DequantizeLinear", ["x", "s", "z"], ["y"], axis=1)],
     {"x": rng.integers(-128, 127, (3, 4)).astype(np.int8)}, {"s": (rng.random(4) * 0.1).astype(np.float32), "z": np.array([0, 1, -2, 3], np.int8)})
case("DequantizeLinear_blocked", [node("DequantizeLinear", ["x", "s", "z"], ["y"], axis=1, block_size=4)],
     {"x": rng.integers(0, 255, (3, 8)).astype(np.uint8)}, {"s": (rng.random((3, 2)) * 0.1).astype(np.float32), "z": rng.integers(0, 255, (3, 2)).astype(np.uint8)})
case("QuantizeLinear_uint8", [node("QuantizeLinear", ["x", "s", "z"], ["y"])],
     {"x": f32(3, 4, scale=3)}, {"s": np.array(0.05, np.float32), "z": np.array(128, np.uint8)})
case("QuantizeLinear_int8_axis", [node("QuantizeLinear", ["x", "s", "z"], ["y"], axis=1)],
     {"x": f32(3, 4, scale=3)}, {"s": (rng.random(4) * 0.1 + 0.01).astype(np.float32), "z": np.array([0, 1, -2, 3], np.int8)})
case("DynamicQuantizeLinear", [node("DynamicQuantizeLinear", ["x"], ["y", "s", "z"])], {"x": f32(4, 6, scale=2)})
case("MatMulInteger", [node("MatMulInteger", ["a", "b", "az", "bz"], ["y"])],
     {"a": rng.integers(0, 255, (2, 3, 16)).astype(np.uint8)}, {"b": rng.integers(-128, 127, (16, 5)).astype(np.int8), "az": np.array(120, np.uint8), "bz": np.array(3, np.int8)})
case("MatMulInteger_dynamic_quant", [
    node("DynamicQuantizeLinear", ["x"], ["xq", "xs", "xz"]),
    node("MatMulInteger", ["xq", "w", "xz"], ["acc"]),
    node("Cast", ["acc"], ["accf"], to=TensorProto.FLOAT),
    node("Mul", ["xs", "ws"], ["scale"]),
    node("Mul", ["accf", "scale"], ["y"]),
], {"x": f32(2, 16)}, {"w": rng.integers(-128, 127, (16, 4)).astype(np.int8), "ws": np.array(0.02, np.float32)})

# ---------- ONNX standard attention family (opset 23) ----------

B, S, P, H, Hd = 2, 5, 3, 4, 8
case("Attention_4d", [node("Attention", ["q", "k", "v"], ["y"])],
     {"q": f32(B, H, S, Hd), "k": f32(B, H, S, Hd), "v": f32(B, H, S, Hd)})
case("Attention_3d_gqa_causal", [node("Attention", ["q", "k", "v"], ["y"], q_num_heads=4, kv_num_heads=2, is_causal=1)],
     {"q": f32(B, S, 4 * Hd), "k": f32(B, S, 2 * Hd), "v": f32(B, S, 2 * Hd)})
case("Attention_bool_mask_past_causal", [node("Attention", ["q", "k", "v", "mask", "pk", "pv"], ["y", "present_k", "present_v"], is_causal=1, softcap=30.0)],
     {"q": f32(B, H, S, Hd), "k": f32(B, H, S, Hd), "v": f32(B, H, S, Hd), "mask": rng.random((B, 1, S, P + S)) > 0.2,
      "pk": f32(B, H, P, Hd), "pv": f32(B, H, P, Hd)})
case("Attention_float_mask_scale", [node("Attention", ["q", "k", "v", "mask"], ["y"], scale=0.3)],
     {"q": f32(B, H, S, Hd), "k": f32(B, H, 7, Hd), "v": f32(B, H, 7, Hd), "mask": f32(1, 1, S, 7)})

pos = np.array([[0, 1, 2, 3, 4], [2, 3, 4, 5, 6]], dtype=np.int64)
cos = np.cos(np.arange(16)[:, None] * np.arange(Hd // 2)[None] * 0.1).astype(np.float32)
sin = np.sin(np.arange(16)[:, None] * np.arange(Hd // 2)[None] * 0.1).astype(np.float32)
case("RotaryEmbedding_4d", [node("RotaryEmbedding", ["x", "cos", "sin", "pos"], ["y"])], {"x": f32(B, H, S, Hd), "pos": pos}, {"cos": cos, "sin": sin})
case("RotaryEmbedding_3d_interleaved", [node("RotaryEmbedding", ["x", "cos", "sin", "pos"], ["y"], interleaved=1, num_heads=H)], {"x": f32(B, S, H * Hd), "pos": pos}, {"cos": cos, "sin": sin})
case("RotaryEmbedding_partial", [node("RotaryEmbedding", ["x", "cos", "sin", "pos"], ["y"], rotary_embedding_dim=4)], {"x": f32(B, H, S, Hd), "pos": pos}, {"cos": cos[:, :2].copy(), "sin": sin[:, :2].copy()})
case("RotaryEmbedding_no_positions", [node("RotaryEmbedding", ["x", "cos", "sin"], ["y"])], {"x": f32(B, H, S, Hd)}, {"cos": cos[pos], "sin": sin[pos]})
case("RotaryEmbedding_ms_interleaved", [node("RotaryEmbedding", ["x", "pos", "cos", "sin"], ["y"], domain=MS, interleaved=1)], {"x": f32(B, S, H * Hd), "pos": pos}, {"cos": cos, "sin": sin}, ms=True)
case("RMSNormalization", [node("RMSNormalization", ["x", "w"], ["y"], epsilon=1e-6)], {"x": f32(2, 3, 16)}, {"w": f32(16)})
case("RMSNormalization_axis", [node("RMSNormalization", ["x", "w"], ["y"], axis=1)], {"x": f32(2, 3, 4)}, {"w": f32(3, 4)})

# ---------- ORT contrib BERT-style ops ----------

hidden = H * Hd
mask2d = np.array([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]], dtype=np.int32)
case("ms_Attention_mask2d", [node("Attention", ["x", "w", "b", "mask"], ["y"], domain=MS, num_heads=H)],
     {"x": f32(B, S, hidden), "mask": mask2d}, {"w": f32(hidden, 3 * hidden, scale=0.2), "b": f32(3 * hidden)}, ms=True)
case("ms_Attention_lengths_unidirectional", [node("Attention", ["x", "w", "b", "mask"], ["y"], domain=MS, num_heads=H, unidirectional=1)],
     {"x": f32(B, S, hidden), "mask": np.array([5, 3], dtype=np.int32)}, {"w": f32(hidden, 3 * hidden, scale=0.2), "b": f32(3 * hidden)}, ms=True)
case("ms_Attention_bias", [node("Attention", ["x", "w", "b", "", "", "bias"], ["y"], domain=MS, num_heads=H)],
     {"x": f32(B, S, hidden), "bias": f32(1, H, S, S)}, {"w": f32(hidden, 3 * hidden, scale=0.2), "b": f32(3 * hidden)}, ms=True)
case("SkipLayerNormalization", [node("SkipLayerNormalization", ["x", "skip", "g", "beta", "bias"], ["y", "", "", "sum"], domain=MS, epsilon=1e-5)],
     {"x": f32(B, S, 16), "skip": f32(B, S, 16)}, {"g": f32(16), "beta": f32(16), "bias": f32(16)}, ms=True)
case("EmbedLayerNormalization", [node("EmbedLayerNormalization", ["ids", "seg", "we", "pe", "se", "g", "beta", "mask"], ["y", "mask_index"], domain=MS, epsilon=1e-5)],
     {"ids": rng.integers(0, 30, (B, S)).astype(np.int32), "seg": rng.integers(0, 2, (B, S)).astype(np.int32), "mask": mask2d},
     {"we": f32(30, 16), "pe": f32(16, 16), "se": f32(2, 16), "g": f32(16), "beta": f32(16)}, ms=True)
case("FastGelu", [node("FastGelu", ["x", "b"], ["y"], domain=MS)], {"x": f32(3, 16)}, {"b": f32(16)}, ms=True)
case("BiasGelu", [node("BiasGelu", ["x", "b"], ["y"], domain=MS)], {"x": f32(3, 16)}, {"b": f32(16)}, ms=True)
case("QuickGelu", [node("QuickGelu", ["x"], ["y"], domain=MS, alpha=1.5)], {"x": f32(3, 16)}, ms=True)
case("ms_Gelu", [node("Gelu", ["x"], ["y"], domain=MS)], {"x": f32(3, 16)}, ms=True)

# GQA with batch 2, a left-aligned static cache of different lengths, interleaved rotary
# (ORT's rotary kernel needs a head size that is a multiple of 16).
G = 16
gqa_cos = np.cos(np.arange(32)[:, None] * np.arange(G // 2)[None] * 0.1).astype(np.float32)
gqa_sin = np.sin(np.arange(32)[:, None] * np.arange(G // 2)[None] * 0.1).astype(np.float32)
case("GroupQueryAttention_batch_rotary_interleaved",
     [node("GroupQueryAttention", ["q", "k", "v", "pk", "pv", "seqlens", "total", "cos", "sin"], ["y", "present_k", "present_v"], domain=MS,
           num_heads=4, kv_num_heads=2, do_rotary=1, rotary_interleaved=1)],
     {"q": f32(B, 1, 4 * G), "k": f32(B, 1, 2 * G), "v": f32(B, 1, 2 * G), "pk": f32(B, 2, 6, G), "pv": f32(B, 2, 6, G),
      "seqlens": np.array([6, 4], dtype=np.int32), "total": np.array(7, dtype=np.int32)},
     {"cos": gqa_cos, "sin": gqa_sin}, ms=True, static=("total",))

# ---------- vision, diffusion and misc ----------

case("ConvTranspose_groups", [node("ConvTranspose", ["x", "w", "b"], ["y"], group=2, strides=[2, 2], pads=[1, 1, 0, 0], output_padding=[1, 1])],
     {"x": f32(1, 4, 5, 5)}, {"w": f32(4, 3, 3, 3), "b": f32(6)})
case("Resize_linear_antialias", [node("Resize", ["x", "", "scales"], ["y"], mode="linear", antialias=1)],
     {"x": f32(1, 2, 16, 16)}, {"scales": np.array([1, 1, 0.4, 0.6], np.float32)})
case("Resize_cubic_antialias", [node("Resize", ["x", "", "", "sizes"], ["y"], mode="cubic", antialias=1)],
     {"x": f32(1, 2, 16, 16)}, {"sizes": np.array([1, 2, 7, 5], np.int64)})
case("GroupNormalization", [node("GroupNormalization", ["x", "s", "b"], ["y"], num_groups=4, epsilon=1e-5)], {"x": f32(2, 8, 5, 5)}, {"s": f32(8), "b": f32(8)}, opset=21)
case("ScatterElements_add", [node("ScatterElements", ["d", "i", "u"], ["y"], axis=1, reduction="add")],
     {"d": f32(3, 5), "u": f32(3, 2)}, {"i": np.array([[0, 4], [1, 1], [-1, 2]], np.int64)})
case("ScatterElements_none", [node("ScatterElements", ["d", "i", "u"], ["y"], axis=0)],
     {"d": f32(4, 3), "u": f32(2, 3)}, {"i": np.array([[1, 0, 3], [2, 3, 0]], np.int64)})
case("GatherND", [node("GatherND", ["d", "i"], ["y"])], {"d": f32(3, 4, 5)}, {"i": np.array([[0, 1], [2, 3], [1, -1]], np.int64)})
case("GatherND_batch", [node("GatherND", ["d", "i"], ["y"], batch_dims=1)], {"d": f32(2, 4, 5)}, {"i": np.array([[[1], [3]], [[0], [2]]], np.int64)})
case("PRelu", [node("PRelu", ["x", "s"], ["y"])], {"x": f32(2, 3, 4)}, {"s": f32(3, 1)})
case("Mish", [node("Mish", ["x"], ["y"])], {"x": f32(3, 7)})
case("Selu", [node("Selu", ["x"], ["y"])], {"x": f32(3, 7)})
case("Softsign", [node("Softsign", ["x"], ["y"])], {"x": f32(3, 7)})
case("ThresholdedRelu", [node("ThresholdedRelu", ["x"], ["y"], alpha=0.5)], {"x": f32(3, 7)})
case("LpNormalization", [node("LpNormalization", ["x"], ["y"], axis=1, p=2)], {"x": f32(3, 7)})
case("SpaceToDepth", [node("SpaceToDepth", ["x"], ["y"], blocksize=2)], {"x": f32(1, 3, 4, 6)})

# ---------- sequence and signal ----------

T, In, Hs = 6, 5, 4
case("LSTM_forward", [node("LSTM", ["x", "w", "r", "b"], ["y", "yh", "yc"], hidden_size=Hs)],
     {"x": f32(T, 2, In)}, {"w": f32(1, 4 * Hs, In, scale=0.5), "r": f32(1, 4 * Hs, Hs, scale=0.5), "b": f32(1, 8 * Hs)})
case("LSTM_bidirectional_initial", [node("LSTM", ["x", "w", "r", "b", "", "h0", "c0"], ["y", "yh", "yc"], hidden_size=Hs, direction="bidirectional")],
     {"x": f32(T, 2, In), "h0": f32(2, 2, Hs), "c0": f32(2, 2, Hs)}, {"w": f32(2, 4 * Hs, In, scale=0.5), "r": f32(2, 4 * Hs, Hs, scale=0.5), "b": f32(2, 8 * Hs)})
case("GRU_forward", [node("GRU", ["x", "w", "r", "b"], ["y", "yh"], hidden_size=Hs)],
     {"x": f32(T, 2, In)}, {"w": f32(1, 3 * Hs, In, scale=0.5), "r": f32(1, 3 * Hs, Hs, scale=0.5), "b": f32(1, 6 * Hs)})
case("GRU_linear_before_reset_bidirectional", [node("GRU", ["x", "w", "r", "b"], ["y", "yh"], hidden_size=Hs, direction="bidirectional", linear_before_reset=1)],
     {"x": f32(T, 2, In)}, {"w": f32(2, 3 * Hs, In, scale=0.5), "r": f32(2, 3 * Hs, Hs, scale=0.5), "b": f32(2, 6 * Hs)})
case("STFT_onesided_window", [node("STFT", ["x", "step", "win"], ["y"])],
     {"x": f32(2, 64, 1)}, {"step": np.array(8, np.int64), "win": np.hanning(16).astype(np.float32)}, opset=17)
case("STFT_two_sided_length", [node("STFT", ["x", "step", "", "len"], ["y"], onesided=0)],
     {"x": f32(1, 40, 1)}, {"step": np.array(10, np.int64), "len": np.array(8, np.int64)}, opset=17)
case("DFT_real_onesided", [node("DFT", ["x", "", "axis"], ["y"], onesided=1)], {"x": f32(2, 12, 1)}, {"axis": np.array(1, np.int64)}, opset=20)
case("DFT_complex_inverse", [node("DFT", ["x", "", "axis"], ["y"], inverse=1)], {"x": f32(2, 8, 2)}, {"axis": np.array(1, np.int64)}, opset=20)

# Loop with a static trip count: running sum and a scan output
body = helper.make_graph(
    [node("Add", ["acc_in", "x"], ["acc_out"]), node("Identity", ["cond_in"], ["cond_out"]),
     node("Cast", ["i"], ["i_f"], to=TensorProto.FLOAT), node("Mul", ["acc_out", "i_f"], ["scan"])],
    "body",
    [helper.make_tensor_value_info("i", TensorProto.INT64, []), helper.make_tensor_value_info("cond_in", TensorProto.BOOL, []),
     helper.make_tensor_value_info("acc_in", TensorProto.FLOAT, [3])],
    [helper.make_tensor_value_info("cond_out", TensorProto.BOOL, []), helper.make_tensor_value_info("acc_out", TensorProto.FLOAT, [3]),
     helper.make_tensor_value_info("scan", TensorProto.FLOAT, [3])])
case("Loop_static_trip_count", [node("Loop", ["M", "cond", "acc0"], ["acc", "scans"], body=body)],
     {"x": f32(3), "acc0": f32(3)}, {"M": np.array(4, np.int64), "cond": np.array(True)})
