"""Builds the op-level test fixtures: one small ONNX graph per operator variant, with
its inputs and reference outputs, in the layout of bench/coverage/prepare.py.

The reference is ONNX Runtime with graph optimizations off, or the ONNX reference
evaluator for operators that ONNX Runtime does not have yet. test/transmuter/ops_test.exs
runs every fixture, so a mismatch points at one operator.

Usage: uv run --with onnx --with onnxruntime --with numpy python -I test/support/op_cases.py [output_dir]
The default output_dir is test/fixtures/ops.
"""

import json
import os
import sys
import zlib
from collections import Counter

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

cases_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fixtures", "ops")
rng = np.random.default_rng(0)
NX = {"int64": "s64", "int32": "s32", "float32": "f32", "uint8": "u8", "int8": "s8", "bool": "u8", "float16": "f16", "uint32": "u32", "uint16": "u16", "uint64": "u64", "int16": "s16"}
MS = "com.microsoft"


def f32(*shape, scale=1.0):
    return (rng.standard_normal(shape) * scale).astype(np.float32)


def rng_for(name):
    """A generator seeded by the case name, so a new case does not change the data of other cases."""
    return np.random.default_rng(zlib.crc32(name.encode()))


def tensor_type(a):
    return TensorProto.STRING if a.dtype.kind in "OUS" else helper.np_dtype_to_tensor_dtype(a.dtype)


def value_info(name, v):
    """A list is a sequence of tensors, any other value is a tensor."""
    if isinstance(v, list):
        return helper.make_tensor_sequence_value_info(name, tensor_type(np.asarray(v[0])) if v else TensorProto.FLOAT, None)
    return helper.make_tensor_value_info(name, tensor_type(v), v.shape)


def case(name, nodes, inputs, initializers=None, outputs=None, opset=23, ms=False, static=(), eager=False, atol=None,
         output_infos=None, extra=None):
    """inputs/initializers: dict name -> value. A value is an array (a string array for STRING tensors) or a
    list of arrays (a sequence). outputs: list of names (defaults to last node outputs). output_infos: dict
    name -> ValueInfoProto for outputs that need a type, such as optionals. eager: the test runs the model
    without compiling. atol: an absolute tolerance instead of the default one. extra: more meta.json keys."""
    initializers = initializers or {}
    outputs = outputs or [o for o in nodes[-1].output if o]
    output_infos = output_infos or {}
    graph = helper.make_graph(
        nodes,
        name,
        [value_info(k, v) for k, v in inputs.items()],
        [output_infos.get(o, onnx.ValueInfoProto(name=o)) for o in outputs],
        [numpy_helper.from_array(v, k) for k, v in initializers.items()],
    )
    imports = [helper.make_opsetid("", opset)] + ([helper.make_opsetid(MS, 1)] if ms else [])
    model = helper.make_model(graph, opset_imports=imports, ir_version=10)
    path = os.path.join(cases_dir, name)
    os.makedirs(path, exist_ok=True)
    onnx_path = os.path.join(path, "model.onnx")
    onnx.save(model, onnx_path)
    meta = {"repo": "op/" + name, "task": "op", "file": "model.onnx", "onnx_path": "model.onnx", "unoptimized_ref": True,
            "ops": dict(Counter((f"{n.domain}:" if n.domain else "") + n.op_type for n in nodes)), "static_inputs": list(static)}
    if eager:
        meta["eager"] = True
    if atol is not None:
        meta["atol"] = atol
    meta.update(extra or {})

    def save(v, stem):
        if v is None:
            return {"none": True}
        if isinstance(v, list):
            return {"sequence": [save(e, f"{stem}_{j}") for j, e in enumerate(v)]}
        a = np.asarray(v)
        if a.dtype.kind in "OUS":
            return {"dtype": "string", "shape": list(a.shape), "values": [x.decode() if isinstance(x, bytes) else str(x) for x in a.reshape(-1)]}
        np.ascontiguousarray(a.astype(np.uint8) if a.dtype == np.bool_ else a).tofile(os.path.join(path, stem + ".bin"))
        return {"file": stem + ".bin", "dtype": NX[str(a.dtype)], "shape": list(a.shape)}

    try:
        try:
            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            opts.log_severity_level = 3
            sess = ort.InferenceSession(onnx_path, opts, providers=["CPUExecutionProvider"])
            outs = sess.run(outputs, inputs)
            meta["reference"] = "onnxruntime"
        except Exception as ort_error:
            # ONNX Runtime does not have the newest operators yet, so ONNX's reference evaluator is the reference.
            from onnx.reference import ReferenceEvaluator
            if ms:
                raise ort_error
            outs = ReferenceEvaluator(model).run(outputs, inputs)
            meta["reference"] = "onnx.reference: " + str(ort_error).splitlines()[0][:120]
        meta["inputs"] = [{"name": k, **save(v, f"in_{i}")} for i, (k, v) in enumerate(inputs.items())]
        meta["outputs"] = [{"name": k, **save(v, f"out_{i}")} for i, (k, v) in enumerate(zip(outputs, outs))]
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

# ---------- group 1: elementwise, reductions, pooling, windows ----------

pos = (rng.random((3, 5)) * 0.9 + 0.05).astype(np.float32)
for op in ["Acos", "Asin", "Atanh"]:
    case(op, [node(op, ["x"], ["y"])], {"x": (pos * 2 - 1).astype(np.float32)})
for op in ["Tan", "Cosh", "Sinh", "Asinh"]:
    case(op, [node(op, ["x"], ["y"])], {"x": f32(3, 5)})
case("Acosh", [node("Acosh", ["x"], ["y"])], {"x": (pos + 1.5).astype(np.float32)})
ints = lambda *shape: rng.integers(0, 200, shape).astype(np.int32)
for op in ["BitwiseAnd", "BitwiseOr", "BitwiseXor"]:
    case(op, [node(op, ["a", "b"], ["y"])], {"a": ints(3, 4), "b": ints(3, 4)}, opset=18)
case("BitwiseNot", [node("BitwiseNot", ["a"], ["y"])], {"a": ints(3, 4)}, opset=18)
case("BitShift_left", [node("BitShift", ["a", "b"], ["y"], direction="LEFT")], {"a": rng.integers(0, 50, (3, 4)).astype(np.uint8), "b": rng.integers(0, 3, (3, 4)).astype(np.uint8)})
case("BitShift_right", [node("BitShift", ["a", "b"], ["y"], direction="RIGHT")], {"a": rng.integers(0, 250, (3, 4)).astype(np.uint32), "b": rng.integers(0, 5, (3, 4)).astype(np.uint32)})
case("Celu", [node("Celu", ["x"], ["y"], alpha=0.7)], {"x": f32(3, 6)}, opset=12)
case("Swish", [node("Swish", ["x"], ["y"], alpha=1.3)], {"x": f32(3, 6)}, opset=24)
case("SwiGLU", [node("SwiGLU", ["a", "b"], ["y"])], {"a": f32(3, 6), "b": f32(3, 6)}, opset=28)
case("Shrink", [node("Shrink", ["x"], ["y"], bias=0.3, lambd=0.4)], {"x": f32(3, 6)})
case("Mean", [node("Mean", ["a", "b", "c"], ["y"])], {"a": f32(3, 4), "b": f32(1, 4), "c": f32(3, 1)})
case("Sum", [node("Sum", ["a", "b"], ["y"])], {"a": f32(2, 3, 4), "b": f32(4)})
case("Dropout_inference", [node("Dropout", ["x"], ["y", "mask"])], {"x": f32(3, 4)})
case("LogSoftmax", [node("LogSoftmax", ["x"], ["y"], axis=1)], {"x": f32(2, 5, 3)})
case("LogSoftmax_opset11_flatten", [node("LogSoftmax", ["x"], ["y"], axis=1)], {"x": f32(2, 5, 3)}, opset=11)
case("Hardmax", [node("Hardmax", ["x"], ["y"], axis=-1)], {"x": f32(3, 7)})
case("CumProd_exclusive_reverse", [node("CumProd", ["x", "axis"], ["y"], exclusive=1, reverse=1)], {"x": (pos + 0.5).astype(np.float32)}, {"axis": np.array(1, np.int64)}, opset=26)
for op in ["ReduceL1", "ReduceLogSumExp"]:
    case(op, [node(op, ["x", "axes"], ["y"], keepdims=0)], {"x": f32(3, 4, 5)}, {"axes": np.array([1, 2], np.int64)})
case("ReduceLogSum", [node("ReduceLogSum", ["x", "axes"], ["y"])], {"x": (pos + 0.1).astype(np.float32)}, {"axes": np.array([1], np.int64)})
case("LRN", [node("LRN", ["x"], ["y"], size=3, alpha=1e-3, beta=0.75, bias=2.0)], {"x": f32(2, 6, 5, 5)})
case("LpPool", [node("LpPool", ["x"], ["y"], kernel_shape=[3, 3], strides=[2, 2], pads=[1, 1, 1, 1], p=3)], {"x": f32(1, 2, 7, 7)})
case("GlobalLpPool", [node("GlobalLpPool", ["x"], ["y"], p=2)], {"x": f32(2, 3, 5, 5)}, opset=18)
case("MeanVarianceNormalization", [node("MeanVarianceNormalization", ["x"], ["y"])], {"x": f32(2, 3, 4, 4)})
case("ReverseSequence", [node("ReverseSequence", ["x", "lens"], ["y"], batch_axis=1, time_axis=0)], {"x": f32(5, 3, 2), "lens": np.array([5, 2, 3], np.int64)})
case("Det", [node("Det", ["x"], ["y"])], {"x": f32(2, 3, 3)})
case("RNN_bidirectional", [node("RNN", ["x", "w", "r", "b"], ["y", "yh"], hidden_size=4, direction="bidirectional")],
     {"x": f32(5, 2, 3)}, {"w": f32(2, 4, 3, scale=0.5), "r": f32(2, 4, 4, scale=0.5), "b": f32(2, 8)})
for op in ["HannWindow", "HammingWindow", "BlackmanWindow"]:
    case(op + "_periodic", [node(op, ["n"], ["y"])], {}, {"n": np.array(16, np.int64)}, opset=17)
    case(op + "_symmetric", [node(op, ["n"], ["y"], periodic=0)], {}, {"n": np.array(10, np.int64)}, opset=17)
case("MelWeightMatrix", [node("MelWeightMatrix", ["bins", "dft", "rate", "low", "high"], ["y"])], {},
     {"bins": np.array(8, np.int64), "dft": np.array(64, np.int64), "rate": np.array(16000, np.int64), "low": np.array(20.0, np.float32), "high": np.array(7600.0, np.float32)}, opset=17)

# ---------- group 2: vision, quantized, Scan ----------

case("AffineGrid_2d", [node("AffineGrid", ["theta", "size"], ["y"])], {"theta": f32(2, 2, 3)}, {"size": np.array([2, 3, 4, 5], np.int64)}, opset=20)
case("AffineGrid_3d_align", [node("AffineGrid", ["theta", "size"], ["y"], align_corners=1)], {"theta": f32(1, 3, 4)}, {"size": np.array([1, 1, 3, 4, 5], np.int64)}, opset=20)
case("CenterCropPad", [node("CenterCropPad", ["x", "shape"], ["y"], axes=[1, 2])], {"x": f32(2, 7, 4)}, {"shape": np.array([4, 7], np.int64)}, opset=18)
case("Col2Im", [node("Col2Im", ["x", "image", "block"], ["y"], strides=[1, 2], pads=[1, 0, 1, 1])],
     {"x": f32(2, 3 * 2 * 3, 5 * 3)}, {"image": np.array([4, 6], np.int64), "block": np.array([2, 3], np.int64)}, opset=18)
rois = np.array([[1.0, 1.5, 6.0, 7.0], [0.0, 0.0, 9.0, 9.0], [2.5, 3.0, 4.0, 8.5]], np.float32)
case("RoiAlign_avg", [node("RoiAlign", ["x", "rois", "idx"], ["y"], output_height=3, output_width=2, sampling_ratio=2, spatial_scale=0.5)],
     {"x": f32(2, 3, 10, 10), "rois": rois * 2, "idx": np.array([0, 1, 0], np.int64)}, opset=16)
# Adaptive sampling (sampling_ratio 0) depends on the ROI sizes, so the ROIs are static.
case("RoiAlign_adaptive_avg", [node("RoiAlign", ["x", "rois", "idx"], ["y"], output_height=2, output_width=2, sampling_ratio=0)],
     {"x": f32(1, 2, 10, 10), "rois": rois, "idx": np.array([0, 0, 0], np.int64)}, opset=16, static=("rois",))
# ORT warns that its max mode is wrong for sampling ratios other than 1.
case("RoiAlign_max", [node("RoiAlign", ["x", "rois", "idx"], ["y"], output_height=2, output_width=3, sampling_ratio=1, mode="max")],
     {"x": f32(1, 2, 10, 10), "rois": rois, "idx": np.array([0, 0, 0], np.int64)}, opset=16)
case("MaxRoiPool", [node("MaxRoiPool", ["x", "rois"], ["y"], pooled_shape=[2, 3], spatial_scale=1.0)],
     {"x": f32(2, 3, 8, 8), "rois": np.array([[0, 1, 1, 5, 6], [1, 0, 2, 7, 7]], np.float32)}, opset=18)
K = 3
case("DeformConv", [node("DeformConv", ["x", "w", "offset", "b", "mask"], ["y"], kernel_shape=[K, K], pads=[1, 1, 1, 1], strides=[1, 1])],
     {"x": f32(1, 4, 6, 6), "offset": f32(1, 2 * K * K, 6, 6, scale=0.7), "mask": rng.random((1, K * K, 6, 6)).astype(np.float32)},
     {"w": f32(5, 4, K, K, scale=0.3), "b": f32(5)}, opset=19)
case("DeformConv_groups", [node("DeformConv", ["x", "w", "offset"], ["y"], kernel_shape=[2, 2], group=2, offset_group=2, strides=[2, 2])],
     {"x": f32(1, 4, 7, 7), "offset": f32(1, 2 * 2 * 4, 3, 3, scale=0.7)}, {"w": f32(6, 2, 2, 2, scale=0.3)}, opset=19)
q = lambda *shape: rng.integers(0, 255, shape).astype(np.uint8)
case("QLinearMatMul", [node("QLinearMatMul", ["a", "as", "az", "b", "bs", "bz", "ys", "yz"], ["y"])],
     {"a": q(2, 3, 8)}, {"as": np.array(0.02, np.float32), "az": np.array(120, np.uint8), "b": q(8, 4), "bs": np.array(0.03, np.float32),
                         "bz": np.array(128, np.uint8), "ys": np.array(0.5, np.float32), "yz": np.array(100, np.uint8)})
case("QLinearConv_per_channel", [node("QLinearConv", ["x", "xs", "xz", "w", "ws", "wz", "ys", "yz", "b"], ["y"], pads=[1, 1, 1, 1])],
     {"x": q(1, 3, 6, 6)}, {"xs": np.array(0.02, np.float32), "xz": np.array(110, np.uint8), "w": rng.integers(-60, 60, (4, 3, 3, 3)).astype(np.int8),
                            "ws": (rng.random(4) * 0.02 + 0.005).astype(np.float32), "wz": np.zeros(4, np.int8), "ys": np.array(0.15, np.float32),
                            "yz": np.array(128, np.uint8), "b": rng.integers(-500, 500, 4).astype(np.int32)})
scan_body = helper.make_graph(
    [node("Add", ["sum_in", "x_t"], ["sum_out"]), node("Mul", ["sum_out", "x_t"], ["scan_t"])],
    "scan_body",
    [helper.make_tensor_value_info("sum_in", TensorProto.FLOAT, [3]), helper.make_tensor_value_info("x_t", TensorProto.FLOAT, [3])],
    [helper.make_tensor_value_info("sum_out", TensorProto.FLOAT, [3]), helper.make_tensor_value_info("scan_t", TensorProto.FLOAT, [3])])
case("Scan_reverse", [node("Scan", ["init", "xs"], ["final", "ys"], body=scan_body, num_scan_inputs=1, scan_input_directions=[1], scan_output_axes=[1])],
     {"init": f32(3), "xs": f32(5, 3)})
case("QLinearAdd_broadcast", [node("QLinearAdd", ["a", "as", "az", "b", "bs", "bz", "cs", "cz"], ["c"], domain=MS)],
     {"a": q(2, 3, 8)}, {"as": np.array(0.02, np.float32), "az": np.array(120, np.uint8), "b": q(8), "bs": np.array(0.03, np.float32),
                         "bz": np.array(128, np.uint8), "cs": np.array(0.04, np.float32), "cz": np.array(100, np.uint8)}, ms=True)
s8 = lambda *shape: rng.integers(-128, 127, shape).astype(np.int8)
case("QLinearMul_int8", [node("QLinearMul", ["a", "as", "az", "b", "bs", "bz", "cs", "cz"], ["c"], domain=MS)],
     {"a": s8(3, 8), "b": s8(3, 8)}, {"as": np.array(0.02, np.float32), "az": np.array(-5, np.int8), "bs": np.array(0.03, np.float32),
                                      "bz": np.array(3, np.int8), "cs": np.array(0.05, np.float32), "cz": np.array(-10, np.int8)}, ms=True)
# Scales and zero points from the first dense layer of the ONNX Model Zoo vgg16-12-int8, where values land on rounding ties.
case("QLinearAdd_ties", [node("QLinearAdd", ["a", "as", "az", "b", "bs", "bz", "cs", "cz"], ["c"], domain=MS)],
     {"a": q(64, 4096)}, {"as": np.array(2.4697657 * 0.05, np.float32), "az": np.array(152, np.uint8), "b": q(4096), "bs": np.array(0.008398832 * 0.05, np.float32),
                          "bz": np.array(128, np.uint8), "cs": np.array(0.05, np.float32), "cz": np.array(0, np.uint8)}, ms=True)
# MLX uses Winograd on the GPU for 3x3 kernels when C and O are multiples of 32, C + O >= 256 and
# N * H * W >= 4096. Winograd is not exact for integer sums, so this case needs the tap-wise integer conv.
case("ConvInteger_winograd_shape", [node("ConvInteger", ["x", "w", "xz", "wz"], ["y"], pads=[1, 1, 1, 1])],
     {"x": q(1, 224, 64, 64)}, {"w": q(32, 224, 3, 3), "xz": np.array(0, np.uint8), "wz": np.array(128, np.uint8)})
case("ConvInteger_groups_strides_dilations", [node("ConvInteger", ["x", "w", "xz"], ["y"], group=2, strides=[2, 1], dilations=[1, 2], pads=[1, 0, 1, 2])],
     {"x": q(1, 4, 9, 9)}, {"w": q(6, 2, 3, 3), "xz": np.array(100, np.uint8)})
case("Conv_auto_pad_same_upper", [node("Conv", ["x", "w"], ["y"], auto_pad="SAME_UPPER", strides=[2, 2])], {"x": f32(1, 2, 8, 8)}, {"w": f32(3, 2, 3, 3)})
case("Conv_auto_pad_same_lower", [node("Conv", ["x", "w"], ["y"], auto_pad="SAME_LOWER", strides=[2, 2])], {"x": f32(1, 2, 8, 8)}, {"w": f32(3, 2, 3, 3)})
case("Identity_strings", [node("Identity", ["x"], ["y"])], {"x": np.array([["a", "bc"], ["", "déf"]], dtype=object)}, eager=True)

# ---------- random ----------
# Values cannot match ONNX Runtime, so ops_test.exs checks the reference shape and dtype and the
# statistics in "distribution" over at least 10,000 samples.
def uniform_stats(low, high):
    return {"low": low, "high": high, "mean": (low + high) / 2, "std": (high - low) / np.sqrt(12)}


case("RandomNormal_mean_scale", [node("RandomNormal", [], ["y"], shape=[100, 100], mean=2.0, scale=3.0)], {},
     extra={"distribution": {"mean": 2.0, "std": 3.0}})
case("RandomNormalLike", [node("RandomNormalLike", ["x"], ["y"], mean=-1.0, scale=0.5)],
     {"x": rng_for("RandomNormalLike").standard_normal((4, 50, 50)).astype(np.float32)},
     extra={"distribution": {"mean": -1.0, "std": 0.5}})
case("RandomUniform_low_high", [node("RandomUniform", [], ["y"], shape=[100, 120], low=-2.0, high=4.0)], {},
     extra={"distribution": uniform_stats(-2.0, 4.0)})
case("RandomUniformLike_dtype", [node("RandomUniformLike", ["x"], ["y"], dtype=TensorProto.FLOAT)],
     {"x": rng_for("RandomUniformLike_dtype").integers(0, 9, (100, 100)).astype(np.int32)},
     extra={"distribution": uniform_stats(0.0, 1.0)})
p = rng_for("Bernoulli").uniform(0, 1, (100, 100)).astype(np.float32)
case("Bernoulli", [node("Bernoulli", ["p"], ["y"])], {"p": p}, extra={"distribution": {"rate": float(p.mean())}})
case("Bernoulli_bool", [node("Bernoulli", ["p"], ["y"], dtype=TensorProto.BOOL)], {"p": np.full((100, 100), 0.3, np.float32)},
     extra={"distribution": {"rate": 0.3}})
case("RandomNormal_seed", [node("RandomNormal", [], ["y"], shape=[100, 100], seed=7.0)], {},
     extra={"distribution": {"mean": 0.0, "std": 1.0}})
case("RandomUniform_two_nodes", [node("RandomUniform", [], ["a"], shape=[100, 100]), node("RandomUniform", [], ["b"], shape=[100, 100])],
     {}, outputs=["a", "b"], extra={"distribution": uniform_stats(0.0, 1.0)})
weights = np.array([[1, 2, 3, 4], [7, 1, 1, 1]], np.float32)
# ONNX Runtime 1.31 has Multinomial only up to opset 21, and the reference evaluator does not have it.
case("Multinomial", [node("Multinomial", ["x"], ["y"], sample_size=10000)], {"x": np.log(weights) + np.float32(5)}, opset=21,
     extra={"distribution": {"frequencies": (weights / weights.sum(axis=1, keepdims=True)).tolist()}})
