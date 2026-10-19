"""Builds the op-level test fixtures: one small ONNX graph per operator variant, with
its inputs and reference outputs, in the layout of bench/coverage/record_references.py.

The reference is ONNX Runtime with graph optimizations off, or the ONNX reference
evaluator for operators that ONNX Runtime does not have yet. test/transmuter/ops_test.exs
runs every fixture, so a mismatch points at one operator.

Usage: uv run --with onnx --with onnxruntime --with numpy --with pillow python -I test/support/make_op_fixtures.py [output_dir]
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
            meta["reference"] = "onnx.reference: " + str(ort_error).replace(onnx_path, "model.onnx").splitlines()[0][:120]
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

# ---------- sequences and optionals ----------

# A sequence cannot leave a compiled function, so cases that end in a sequence run eagerly.
seq_info = lambda name, t=TensorProto.FLOAT: {name: helper.make_tensor_sequence_value_info(name, t, None)}
r = rng_for("sequences")
sf32 = lambda *shape: r.standard_normal(shape).astype(np.float32)
case("SequenceConstruct_ConcatFromSequence", [node("SequenceConstruct", ["a", "b", "c"], ["s"]), node("ConcatFromSequence", ["s"], ["y"], axis=1)],
     {"a": sf32(2, 3), "b": sf32(2, 1), "c": sf32(2, 4)})
case("ConcatFromSequence_new_axis", [node("SequenceConstruct", ["a", "b", "c"], ["s"]), node("ConcatFromSequence", ["s"], ["y"], axis=-1, new_axis=1)],
     {"a": sf32(2, 3), "b": sf32(2, 3), "c": sf32(2, 3)})
case("SequenceAt_negative", [node("SequenceConstruct", ["a", "b", "c"], ["s"]), node("SequenceAt", ["s", "i"], ["y"])],
     {"a": sf32(2, 3), "b": sf32(4), "c": sf32(1, 2)}, {"i": np.array(-2, np.int64)})
case("SequenceEmpty_Insert_Length", [node("SequenceEmpty", [], ["s0"], dtype=TensorProto.FLOAT), node("SequenceInsert", ["s0", "a"], ["s1"]),
                                     node("SequenceInsert", ["s1", "b", "zero"], ["s2"]), node("SequenceLength", ["s2"], ["n"]),
                                     node("ConcatFromSequence", ["s2"], ["y"], axis=0)],
     {"a": sf32(2, 3), "b": sf32(1, 3)}, {"zero": np.array(0, np.int64)}, outputs=["n", "y"])
case("SequenceEmpty", [node("SequenceEmpty", [], ["s"])], {}, eager=True, output_infos=seq_info("s"))
case("SequenceInsert_negative", [node("SequenceInsert", ["s", "t", "i"], ["y"])],
     {"s": [sf32(2), sf32(3), sf32(4)], "t": sf32(5), "i": np.array(-1, np.int64)}, eager=True, output_infos=seq_info("y"))
case("SequenceErase_negative", [node("SequenceErase", ["s", "i"], ["y"])],
     {"s": [sf32(2), sf32(3), sf32(4)], "i": np.array(-3, np.int64)}, eager=True, output_infos=seq_info("y"))
case("SequenceErase_default", [node("SequenceErase", ["s"], ["y"])],
     {"s": [sf32(2), sf32(3), sf32(4)]}, eager=True, output_infos=seq_info("y"))
case("SplitToSequence_scalar", [node("SplitToSequence", ["x", "split"], ["y"], axis=0)],
     {"x": sf32(7, 3)}, {"split": np.array(3, np.int64)}, eager=True, output_infos=seq_info("y"))
case("SplitToSequence_list", [node("SplitToSequence", ["x", "split"], ["y"], axis=-1)],
     {"x": sf32(2, 5)}, {"split": np.array([1, 4], np.int64)}, eager=True, output_infos=seq_info("y"))
case("SplitToSequence_keepdims", [node("SplitToSequence", ["x"], ["y"], axis=1)], {"x": sf32(2, 3, 2)}, eager=True, output_infos=seq_info("y"))
case("SplitToSequence_no_keepdims", [node("SplitToSequence", ["x"], ["y"], axis=1, keepdims=0)], {"x": sf32(2, 3, 2)}, eager=True, output_infos=seq_info("y"))
case("SplitToSequence_ConcatFromSequence", [node("SplitToSequence", ["x"], ["s"], axis=1, keepdims=0), node("ConcatFromSequence", ["s"], ["y"], axis=1, new_axis=1)],
     {"x": sf32(2, 3, 2)})

square_body = helper.make_graph(
    [node("Mul", ["x", "x"], ["sq"]), node("Shape", ["x"], ["shape"])], "square_body",
    [helper.make_tensor_value_info("x", TensorProto.FLOAT, None)],
    [helper.make_tensor_value_info("sq", TensorProto.FLOAT, None), helper.make_tensor_value_info("shape", TensorProto.INT64, None)])
case("SequenceMap_one_input", [node("SequenceMap", ["s"], ["squares", "shapes"], body=square_body)],
     {"s": [sf32(2), sf32(3, 2), sf32(1)]}, eager=True, opset=17, output_infos={**seq_info("squares"), **seq_info("shapes", TensorProto.INT64)})
add_body = helper.make_graph(
    [node("Add", ["x", "t"], ["y"])], "add_body",
    [helper.make_tensor_value_info("x", TensorProto.FLOAT, None), helper.make_tensor_value_info("t", TensorProto.FLOAT, None)],
    [helper.make_tensor_value_info("y", TensorProto.FLOAT, None)])
case("SequenceMap_sequence_and_tensor", [node("SequenceMap", ["s", "t"], ["y"], body=add_body)],
     {"s": [sf32(2, 3), sf32(1, 3), sf32(4, 3)], "t": sf32(3)}, eager=True, opset=17, output_infos=seq_info("y"))
case("SequenceMap_compiled", [node("SplitToSequence", ["x"], ["s"], axis=0), node("SequenceMap", ["s", "t"], ["m"], body=add_body),
                              node("ConcatFromSequence", ["m"], ["y"], axis=0)],
     {"x": sf32(3, 4), "t": sf32(4)}, opset=17)

optional_float = helper.make_optional_type_proto(helper.make_tensor_type_proto(TensorProto.FLOAT, None))
case("Optional_with_value", [node("Optional", ["x"], ["o"]), node("OptionalHasElement", ["o"], ["has"]), node("OptionalGetElement", ["o"], ["y"])],
     {"x": sf32(2, 3)}, outputs=["has", "y"])
case("Optional_empty", [node("Optional", [], ["o"], type=helper.make_tensor_type_proto(TensorProto.FLOAT, None)), node("OptionalHasElement", ["o"], ["has"])],
     {}, outputs=["o", "has"], eager=True, output_infos={"o": helper.make_value_info("o", optional_float)})
case("OptionalHasElement_missing_input", [node("OptionalHasElement", [], ["has"])], {}, opset=18)

# ---------- the seven other operators ----------

r = rng_for("Upsample_nearest_opset7")
case("Upsample_nearest_opset7", [node("Upsample", ["x"], ["y"], scales=[1.0, 1.0, 1.5, 2.5])], {"x": r.standard_normal((1, 2, 3, 4)).astype(np.float32)}, opset=7)
r = rng_for("Upsample_linear_opset9")
case("Upsample_linear_opset9", [node("Upsample", ["x", "scales"], ["y"], mode="linear")],
     {"x": r.standard_normal((1, 2, 3, 4)).astype(np.float32)}, {"scales": np.array([1, 1, 2.5, 1.5], np.float32)}, opset=9)
r = rng_for("Scatter_opset10")
case("Scatter_opset10", [node("Scatter", ["d", "i", "u"], ["y"], axis=1)],
     {"d": r.standard_normal((3, 5)).astype(np.float32), "u": r.standard_normal((3, 2)).astype(np.float32)},
     {"i": np.array([[0, 4], [3, 1], [2, 0]], np.int64)}, opset=10)


def max_pool(x, **attrs):
    """MaxPool values and flat indices from ONNX Runtime, as MaxUnpool inputs."""
    g = helper.make_graph([node("MaxPool", ["x"], ["y", "i"], **attrs)], "pool", [value_info("x", x)], [onnx.ValueInfoProto(name="y"), onnx.ValueInfoProto(name="i")])
    model = helper.make_model(g, opset_imports=[helper.make_opsetid("", 22)], ir_version=10)
    y, i = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"]).run(None, {"x": x})
    return {"x": y, "i": i}


r = rng_for("MaxUnpool")
case("MaxUnpool", [node("MaxUnpool", ["x", "i"], ["y"], kernel_shape=[2, 2], strides=[2, 2])],
     max_pool(r.standard_normal((2, 3, 4, 4)).astype(np.float32), kernel_shape=[2, 2], strides=[2, 2]))
r = rng_for("MaxUnpool_pads")
case("MaxUnpool_pads", [node("MaxUnpool", ["x", "i"], ["y"], kernel_shape=[3, 3], strides=[2, 2], pads=[1, 1, 1, 1])],
     max_pool(r.standard_normal((2, 2, 5, 5)).astype(np.float32), kernel_shape=[3, 3], strides=[2, 2], pads=[1, 1, 1, 1]))
r = rng_for("MaxUnpool_output_shape")
case("MaxUnpool_output_shape", [node("MaxUnpool", ["x", "i", "shape"], ["y"], kernel_shape=[2, 2], strides=[2, 2])],
     max_pool(r.standard_normal((1, 2, 5, 5)).astype(np.float32), kernel_shape=[2, 2], strides=[2, 2]), {"shape": np.array([1, 2, 5, 5], np.int64)})
r = rng_for("BitCast_float_to_int32")
case("BitCast_float_to_int32", [node("BitCast", ["x"], ["y"], to=TensorProto.INT32)], {"x": r.standard_normal((3, 4)).astype(np.float32)}, opset=26)
r = rng_for("BitCast_uint8_to_int8")
case("BitCast_uint8_to_int8", [node("BitCast", ["x"], ["y"], to=TensorProto.INT8)], {"x": r.integers(0, 256, (3, 4)).astype(np.uint8)}, opset=26)
r = rng_for("TensorScatter_linear")
case("TensorScatter_linear", [node("TensorScatter", ["cache", "update", "idx"], ["y"])],
     {"cache": r.standard_normal((2, 3, 8, 4)).astype(np.float32), "update": r.standard_normal((2, 3, 3, 4)).astype(np.float32), "idx": np.array([1, 5], np.int64)}, opset=24)
r = rng_for("TensorScatter_no_indices")
case("TensorScatter_no_indices", [node("TensorScatter", ["cache", "update"], ["y"])],
     {"cache": r.standard_normal((2, 6, 3)).astype(np.float32), "update": r.standard_normal((2, 2, 3)).astype(np.float32)}, opset=24)
r = rng_for("TensorScatter_circular_axis")
case("TensorScatter_circular_axis", [node("TensorScatter", ["cache", "update", "idx"], ["y"], mode="circular", axis=1)],
     {"cache": r.standard_normal((2, 6, 2, 3)).astype(np.float32), "update": r.standard_normal((2, 4, 2, 3)).astype(np.float32), "idx": np.array([4, 7], np.int64)}, opset=24)
r = rng_for("CausalConvWithState")
case("CausalConvWithState", [node("CausalConvWithState", ["x", "w"], ["y", "state"])],
     {"x": r.standard_normal((2, 4, 6)).astype(np.float32)}, {"w": r.standard_normal((4, 1, 3)).astype(np.float32)}, opset=27)
# The input is shorter than k - 1, so the present state keeps part of the past state.
r = rng_for("CausalConvWithState_state_bias_silu")
case("CausalConvWithState_state_bias_silu", [node("CausalConvWithState", ["x", "w", "b", "past"], ["y", "state"], activation="silu")],
     {"x": r.standard_normal((2, 4, 2)).astype(np.float32), "past": r.standard_normal((2, 4, 3)).astype(np.float32)},
     {"w": r.standard_normal((4, 1, 4)).astype(np.float32), "b": r.standard_normal(4).astype(np.float32)}, opset=27)


def linear_attention(name, rule, q_heads, kv_heads, past=False, decay=None, beta=None, **attrs):
    """decay and beta are the last dimension of those inputs, or None to leave them out."""
    r = rng_for(name)
    b, t, d_k, d_v = 2, 5, 4, 3
    keys = r.standard_normal((b, t, kv_heads, d_k)).astype(np.float32)
    keys /= np.linalg.norm(keys, axis=-1, keepdims=True)
    inputs = {"q": r.standard_normal((b, t, q_heads * d_k)).astype(np.float32), "k": keys.reshape(b, t, -1),
              "v": r.standard_normal((b, t, kv_heads * d_v)).astype(np.float32)}
    if past:
        inputs["past"] = r.standard_normal((b, kv_heads, d_k, d_v)).astype(np.float32)
    if decay:
        inputs["decay"] = (-r.random((b, t, decay)) * 0.5).astype(np.float32)
    if beta:
        inputs["beta"] = r.random((b, t, beta)).astype(np.float32)
    names = ["q", "k", "v"] + [n if n in inputs else "" for n in ["past", "decay", "beta"]]
    while names[-1] == "":
        names.pop()
    case(name, [node("LinearAttention", names, ["y", "state"], update_rule=rule, q_num_heads=q_heads, kv_num_heads=kv_heads, **attrs)], inputs, opset=27)


linear_attention("LinearAttention_linear", "linear", 2, 2)
linear_attention("LinearAttention_gated_key_decay_past", "gated", 2, 2, past=True, decay=2 * 4)
linear_attention("LinearAttention_delta_gqa_scale", "delta", 4, 2, beta=1, scale=0.5)
linear_attention("LinearAttention_gated_delta_gqa_past", "gated_delta", 4, 2, past=True, decay=2, beta=2)

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


# ---------- strings and ImageDecoder ----------

def strings(values):
    return np.array(values, dtype=object)


def image_bytes(image, fmt):
    import io

    from PIL import Image
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format=fmt)
    return np.frombuffer(buffer.getvalue(), np.uint8)


case("StringNormalizer_lower_stopwords", [node("StringNormalizer", ["x"], ["y"], case_change_action="LOWER", stopwords=["the", "And"])],
     {"x": strings(["The", "Cat", "and", "THE", "Hat", "a b", ""])}, opset=10, eager=True)
case("StringNormalizer_upper_case_sensitive", [node("StringNormalizer", ["x"], ["y"], case_change_action="UPPER", is_case_sensitive=1,
     stopwords=["monday"], locale="en_US")], {"x": strings([["monday", "Monday", "straße", "Ünïcode"]])}, opset=10, eager=True)
case("StringNormalizer_all_stopwords_1d", [node("StringNormalizer", ["x"], ["y"], stopwords=["monday"])],
     {"x": strings(["monday", "MONDAY"])}, opset=10, eager=True)
case("StringNormalizer_all_stopwords_2d", [node("StringNormalizer", ["x"], ["y"], case_change_action="LOWER", stopwords=["monday"])],
     {"x": strings([["Monday", "monday"]])}, opset=10, eager=True)
case("StringConcat_broadcast", [node("StringConcat", ["x", "y"], ["z"])],
     {"x": strings([["a"], ["é"]]), "y": strings(["-1", "", "😀"])}, opset=20, eager=True)
case("StringConcat_scalar", [node("StringConcat", ["x", "y"], ["z"])],
     {"x": strings([["a", "b"], ["c", ""]]), "y": strings("!")}, opset=20, eager=True)
case("StringSplit_delimiter_maxsplit", [node("StringSplit", ["x"], ["y", "z"], delimiter="::", maxsplit=2)],
     {"x": strings([["a::b::c::d", "::x::", ""], ["a:b", "a::::b", "::"]])}, opset=20, eager=True)
case("StringSplit_whitespace", [node("StringSplit", ["x"], ["y", "z"])],
     {"x": strings(["  hello   world ! ", "", "one", "a b"])}, opset=20, eager=True)
case("StringSplit_whitespace_maxsplit", [node("StringSplit", ["x"], ["y", "z"], delimiter="", maxsplit=1)],
     {"x": strings(["  a  b  c ", "x", "   ", "p q"])}, opset=20, eager=True)
case("StringSplit_count", [node("StringSplit", ["x"], ["y", "z"], delimiter=",")],
     {"x": strings([["a,b", ",,,", "x"], ["", "1,2,3,4", "q,"]])}, outputs=["z"], opset=20)
case("RegexFullMatch_basic", [node("RegexFullMatch", ["x"], ["y"], pattern=r"[a-z]+@\w+\.(com|org)|\d{3}|é.")],
     {"x": strings([["abc@mail.com", "abc@mail.net", "123"], ["1234", "٣٣٣", "éa"], ["ab@x_y.org", "Ab@x.com", "é\n"]])}, opset=20)


def tfidf(name, x, mode, pool, counts, indexes, eager=False, **attrs):
    key = "pool_strings" if x.dtype == object else "pool_int64s"
    case(name, [node("TfIdfVectorizer", ["x"], ["y"], mode=mode, ngram_counts=counts, ngram_indexes=indexes, **{key: pool}, **attrs)],
         {"x": x}, opset=9, eager=eager, static=() if eager else ("x",))


r = rng_for("TfIdfVectorizer_tf_int_skip")
tfidf("TfIdfVectorizer_tf_int_skip", r.integers(0, 5, (2, 10)).astype(np.int32), "TF", [1, 2, 3, 1, 2, 2, 3, 3, 3, 4, 1], [0, 3],
      [6, 0, 5, 1, 4, 2, 7], min_gram_length=1, max_gram_length=2, max_skip_count=2)
r = rng_for("TfIdfVectorizer_idf_weights_unigrams")
tfidf("TfIdfVectorizer_idf_weights_unigrams", r.integers(0, 4, 12).astype(np.int64), "IDF", [0, 1, 3, 0, 1, 1, 3], [0, 3],
      [2, 0, 4, 1, 3], min_gram_length=1, max_gram_length=1, max_skip_count=0, weights=[0.5, 1.5, 2.0, 0.25, 3.0])
r = rng_for("TfIdfVectorizer_idf_eager")
tfidf("TfIdfVectorizer_idf_eager", r.integers(0, 3, 14).astype(np.int64), "IDF", [0, 1, 0, 1, 1, 2, 2, 2, 0, 1, 2, 2, 1, 0], [0, 2, 8],
      [0, 1, 2, 3, 4, 5, 6, 7], eager=True, min_gram_length=2, max_gram_length=3, max_skip_count=1)
r = rng_for("TfIdfVectorizer_tfidf_strings")
tfidf("TfIdfVectorizer_tfidf_strings", r.choice(["a", "b", "c", "dé"], (3, 7)).astype(object), "TFIDF",
      ["a", "dé", "a", "b", "b", "a", "dé", "c"], [0, 2], [3, 0, 1, 4, 2], min_gram_length=1, max_gram_length=2,
      max_skip_count=1, weights=[0.5, 2.0, 1.25, 3.0, 0.75])
r = rng_for("TfIdfVectorizer_tf_strings_1d")
tfidf("TfIdfVectorizer_tf_strings_1d", r.choice(["x", "y", "z"], 9).astype(object), "TF",
      ["x", "z", "x", "y", "z", "z"], [0, 2], [0, 1, 2, 3], min_gram_length=1, max_gram_length=2, max_skip_count=0)

r = rng_for("ImageDecoder")
picture = np.repeat(np.repeat(r.integers(0, 256, (6, 5, 3), dtype=np.uint8), 4, axis=0), 4, axis=1)
for fmt, atol in [("PNG", None), ("JPEG", 2)]:
    for pixel_format in ["RGB", "BGR", "Grayscale"]:
        case(f"ImageDecoder_{fmt.lower()}_{pixel_format.lower()}", [node("ImageDecoder", ["data"], ["image"], pixel_format=pixel_format)],
             {"data": image_bytes(picture, fmt)}, opset=20, static=("data",), atol=atol)
case("ImageDecoder_gray_png_grayscale", [node("ImageDecoder", ["data"], ["image"], pixel_format="Grayscale")],
     {"data": image_bytes(picture[:, :, 0], "PNG")}, opset=20, eager=True)

# ---------- group 3: data-dependent outputs and losses ----------

# Boxes 1 and 2 have flipped corners, 6 and 7 have an IoU of exactly 0.5 and 8 ties with 0 on the score.
nms_boxes = np.array([[[0, 0, 1, 1], [1, 1.1, 0, 0.1], [0, 0.9, 1, -0.1], [0, 10, 1, 11], [0, 10.1, 1, 11.1],
                       [0, 100, 1, 101], [0, 20, 1, 21], [0, 20, 2, 21], [0, 0.05, 1, 1.05]]], np.float32)
nms_scores = np.array([[[0.9, 0.75, 0.6, 0.95, 0.5, 0.3, 0.8, 0.7, 0.9]]], np.float32)


def nms_limits(m, iou, score=None):
    limits = {"m": np.array([m], np.int64), "iou": np.array([iou], np.float32)}
    return limits if score is None else {**limits, "score": np.array([score], np.float32)}


case("NonMaxSuppression_corners_ties", [node("NonMaxSuppression", ["boxes", "scores", "m", "iou"], ["y"])],
     {"boxes": nms_boxes, "scores": nms_scores}, nms_limits(10, 0.5), static=("boxes", "scores"))
case("NonMaxSuppression_score_threshold_max", [node("NonMaxSuppression", ["boxes", "scores", "m", "iou", "score"], ["y"])],
     {"boxes": nms_boxes, "scores": nms_scores}, nms_limits(3, 0.4, 0.6), static=("boxes", "scores"))
case("NonMaxSuppression_no_max_boxes", [node("NonMaxSuppression", ["boxes", "scores"], ["y"])],
     {"boxes": nms_boxes, "scores": nms_scores}, static=("boxes", "scores"))
r = rng_for("NonMaxSuppression_center_point_box")
case("NonMaxSuppression_center_point_box", [node("NonMaxSuppression", ["boxes", "scores", "m", "iou", "score"], ["y"], center_point_box=1)],
     {"boxes": np.concatenate([r.random((1, 30, 2)) * 10, r.random((1, 30, 2)) * 4 + 0.5], axis=2).astype(np.float32),
      "scores": r.random((1, 2, 30)).astype(np.float32)}, nms_limits(8, 0.3, 0.2), static=("boxes", "scores"))
r = rng_for("NonMaxSuppression_batches_classes")
case("NonMaxSuppression_batches_classes", [node("NonMaxSuppression", ["boxes", "scores", "m", "iou", "score"], ["y"])],
     {"boxes": (r.random((2, 40, 4)) * 10).astype(np.float32), "scores": np.round(r.random((2, 3, 40)), 1).astype(np.float32)},
     nms_limits(5, 0.2, 0.3), static=("boxes", "scores"))

unique_outputs = ["y", "indices", "inverse", "counts"]
case("Unique_sorted_flatten", [node("Unique", ["x"], unique_outputs)],
     {"x": np.array([[2, 1, 1.5], [3, 1, 2], [-1, 1.5, 0]], np.float32)}, static=("x",))
case("Unique_unsorted_flatten", [node("Unique", ["x"], unique_outputs, sorted=0)],
     {"x": np.array([4, 2, 4, 9, 2, 2, -3], np.int64)}, static=("x",))
case("Unique_sorted_axis", [node("Unique", ["x"], unique_outputs, axis=0)],
     {"x": np.array([[1, 0, 0], [1, 0, 0], [2, 3, 4], [0, 5, 1], [2, 3, 4]], np.float32)}, static=("x",))
case("Unique_unsorted_negative_axis", [node("Unique", ["x"], unique_outputs, axis=-1, sorted=0)],
     {"x": np.array([[[1, 1, 0, 1], [2, 2, 3, 2]], [[5, 5, 1, 5], [0, 0, 4, 0]]], np.int64)}, static=("x",))

r = rng_for("Compress")
case("Compress_axis_short_condition", [node("Compress", ["x", "c"], ["y"], axis=-1)],
     {"x": r.standard_normal((2, 3, 6)).astype(np.float32), "c": np.array([0, 1, 1, 0, 1], bool)}, static=("c",))
case("Compress_axis_0", [node("Compress", ["x", "c"], ["y"], axis=0)],
     {"x": r.standard_normal((4, 3)).astype(np.float32), "c": np.array([1, 0, 0, 1], bool)}, static=("c",))
case("Compress_flatten", [node("Compress", ["x", "c"], ["y"])],
     {"x": r.standard_normal((3, 4)).astype(np.float32), "c": np.array([1, 0, 1, 1, 0, 0, 1], bool)}, static=("c",))

r = rng_for("losses")


def loss_x(*shape):
    return r.standard_normal(shape).astype(np.float32)


def loss_t(classes, *shape, ignore=None):
    t = r.integers(0, classes, shape)
    return (t if ignore is None else np.where(r.random(shape) < 0.3, ignore, t)).astype(np.int64)


case("NegativeLogLikelihoodLoss_mean", [node("NegativeLogLikelihoodLoss", ["x", "t"], ["y"])], {"x": loss_x(4, 5), "t": loss_t(5, 4)})
case("NegativeLogLikelihoodLoss_sum_weight", [node("NegativeLogLikelihoodLoss", ["x", "t", "w"], ["y"], reduction="sum")],
     {"x": loss_x(4, 5), "t": loss_t(5, 4), "w": r.random(5).astype(np.float32)})
case("NegativeLogLikelihoodLoss_mean_weight_ignore", [node("NegativeLogLikelihoodLoss", ["x", "t", "w"], ["y"], ignore_index=2)],
     {"x": loss_x(3, 4, 6), "t": loss_t(4, 3, 6, ignore=2), "w": r.random(4).astype(np.float32)})
case("NegativeLogLikelihoodLoss_none_ignore_4d", [node("NegativeLogLikelihoodLoss", ["x", "t"], ["y"], reduction="none", ignore_index=-100)],
     {"x": loss_x(2, 3, 4, 5), "t": loss_t(3, 2, 4, 5, ignore=-100)})
case("NegativeLogLikelihoodLoss_mean_ignore_int32", [node("NegativeLogLikelihoodLoss", ["x", "t"], ["y"], ignore_index=0)],
     {"x": loss_x(6, 3), "t": np.array([0, 1, 2, 0, 2, 2], np.int32)})
case("SoftmaxCrossEntropyLoss_mean", [node("SoftmaxCrossEntropyLoss", ["x", "t"], ["y"])], {"x": loss_x(4, 5), "t": loss_t(5, 4)})
case("SoftmaxCrossEntropyLoss_sum_log_prob", [node("SoftmaxCrossEntropyLoss", ["x", "t"], ["y", "log_prob"], reduction="sum")],
     {"x": loss_x(3, 4, 2), "t": loss_t(4, 3, 2)})
case("SoftmaxCrossEntropyLoss_mean_weight_ignore_4d", [node("SoftmaxCrossEntropyLoss", ["x", "t", "w"], ["y", "log_prob"], ignore_index=-1)],
     {"x": loss_x(2, 3, 4, 5), "t": loss_t(3, 2, 4, 5, ignore=-1), "w": r.random(3).astype(np.float32)})
case("SoftmaxCrossEntropyLoss_none_weight", [node("SoftmaxCrossEntropyLoss", ["x", "t", "w"], ["y"], reduction="none")],
     {"x": loss_x(5, 4), "t": loss_t(4, 5), "w": r.random(4).astype(np.float32)})
case("SoftmaxCrossEntropyLoss_mean_ignore", [node("SoftmaxCrossEntropyLoss", ["x", "t"], ["y"], ignore_index=1)],
     {"x": loss_x(6, 3), "t": loss_t(3, 6, ignore=1)})

# RoiAlign before opset 16 has no coordinate_transformation_mode and does not shift by half a pixel.
r = rng_for("RoiAlign_opset10")
case("RoiAlign_opset10", [node("RoiAlign", ["x", "rois", "idx"], ["y"], output_height=3, output_width=2, sampling_ratio=2, spatial_scale=0.5)],
     {"x": r.standard_normal((2, 3, 10, 10)).astype(np.float32), "rois": rois * 2, "idx": np.array([0, 1, 0], np.int64)}, opset=10)

# A zero-size value keeps its shape through Slice, Transpose and Unsqueeze, as Mask R-CNN needs.
r = rng_for("Shape_of_zero_size")
case("Shape_of_zero_size", [
    node("Slice", ["x", "starts", "ends", "axes"], ["empty"]),
    node("Shape", ["empty"], ["shape"]),
    node("Gather", ["shape", "last"], ["size"]),
    node("Transpose", ["empty"], ["moved"], perm=[2, 0, 1]),
    node("Unsqueeze", ["moved", "axes"], ["wider"]),
    node("Shape", ["wider"], ["wider_shape"]),
], {"x": r.standard_normal((3, 4, 5)).astype(np.float32)},
     {"starts": np.array([2], np.int64), "ends": np.array([2], np.int64), "axes": np.array([1], np.int64), "last": np.array(2, np.int64)},
     outputs=["shape", "size", "wider_shape"])

# Index operators with no indices. In the spline clamp of the VITS duration predictor,
# NonZero finds no values out of bounds, so the gather is empty and the scatter keeps x.
r = rng_for("Empty_indices")
case("Empty_indices_GatherND_ScatterND", [
    node("Greater", ["x", "bound"], ["outside"]), node("NonZero", ["outside"], ["nz"]), node("Transpose", ["nz"], ["idx"]),
    node("GatherND", ["x", "idx"], ["picked"]), node("Mul", ["picked", "half"], ["clamped"]),
    node("ScatterND", ["x", "idx", "clamped"], ["y"]),
], {"x": r.standard_normal((1, 2, 5)).astype(np.float32)}, {"bound": np.array(10, np.float32), "half": np.array(0.5, np.float32)})
case("Empty_indices_GatherElements_OneHot", [
    node("Slice", ["all", "zero", "zero", "one"], ["idx"]), node("GatherElements", ["x", "idx"], ["g"], axis=1),
    node("Shape", ["g"], ["g_shape"]), node("Slice", ["row", "zero", "zero"], ["hot_idx"]),
    node("OneHot", ["hot_idx", "depth", "values"], ["hot"]), node("Shape", ["hot"], ["hot_shape"]),
], {"x": r.standard_normal((2, 3)).astype(np.float32)},
     {"all": np.zeros((2, 3), np.int64), "row": np.array([1, 2], np.int64), "zero": np.array([0], np.int64), "one": np.array([1], np.int64),
      "depth": np.array(4, np.int64), "values": np.array([0, 1], np.float32)},
     outputs=["g_shape", "hot_shape"])
# SAM 2 adds a bias to the embeddings of an empty box input, then concatenates them with the points.
# A MatMul over an empty axis gives zeros.
case("Empty_elementwise_Concat", [
    node("Slice", ["x", "zero", "zero", "one"], ["no_boxes"]), node("Add", ["no_boxes", "bias"], ["shifted"]),
    node("Relu", ["shifted"], ["active"]), node("Mul", ["active", "scale"], ["scaled"]), node("Shape", ["scaled"], ["empty_shape"]),
    node("Concat", ["scaled", "x"], ["y"], axis=1),
    node("MatMul", ["no_boxes", "w"], ["projected"]), node("Shape", ["projected"], ["matmul_shape"]),
    node("Slice", ["x", "zero", "zero", "two"], ["no_features"]), node("Slice", ["w", "zero", "zero"], ["no_rows"]),
    node("MatMul", ["no_features", "no_rows"], ["zeros"]),
], {"x": r.standard_normal((1, 3, 4)).astype(np.float32)},
     {"zero": np.array([0], np.int64), "one": np.array([1], np.int64), "two": np.array([2], np.int64), "bias": np.array(0.5, np.float32),
      "scale": r.standard_normal((1, 1, 4)).astype(np.float32), "w": r.standard_normal((4, 2)).astype(np.float32)},
     outputs=["empty_shape", "y", "matmul_shape", "zeros"])
# Operators that make a zero-size output from inputs that are not zero size.
case("Empty_outputs_from_sizes", [
    node("Range", ["zero_s", "zero_s", "one_s"], ["no_steps"]), node("Shape", ["no_steps"], ["range_shape"]),
    node("Expand", ["row", "to_empty"], ["expanded"]), node("Shape", ["expanded"], ["expand_shape"]),
    node("Tile", ["x", "no_repeats"], ["tiled"]), node("Shape", ["tiled"], ["tile_shape"]),
    node("Split", ["x", "sizes"], ["none_of_it", "all_of_it"], axis=1), node("Shape", ["none_of_it"], ["split_shape"]),
    node("TopK", ["x", "zero"], ["top", "top_idx"]), node("Shape", ["top"], ["topk_shape"]),
], {"x": r.standard_normal((2, 3)).astype(np.float32)},
     {"zero_s": np.array(0, np.int64), "one_s": np.array(1, np.int64), "to_empty": np.array([0, 3], np.int64), "row": np.ones((1, 3), np.float32),
      "no_repeats": np.array([1, 0], np.int64), "sizes": np.array([0, 3], np.int64), "zero": np.array([0], np.int64)},
     outputs=["range_shape", "expand_shape", "tile_shape", "split_shape", "all_of_it", "topk_shape"])

# ---------- graph breaks ----------

# These graphs compile with no static inputs. A value computed from the inputs decides a
# shape or a branch, so Transmuter.compile/3 breaks the graph there.
r = rng_for("graph breaks")
pos_f32 = lambda *shape: (r.random(shape) + 0.5).astype(np.float32)
INT64 = TensorProto.INT64
case("Break_Range_computed_limit",
     [node("ReduceSum", ["x"], ["s"], keepdims=0), node("Cast", ["s"], ["n"], to=INT64),
      node("Range", ["zero", "n", "one"], ["idx"]), node("Cast", ["idx"], ["f"], to=TensorProto.FLOAT), node("Mul", ["f", "s"], ["y"])],
     {"x": pos_f32(6)}, {"zero": np.array(0, np.int64), "one": np.array(1, np.int64)})
case("Break_NonZero_computed", [node("Greater", ["x", "zero"], ["mask"]), node("NonZero", ["mask"], ["y"])],
     {"x": (r.standard_normal((3, 4))).astype(np.float32)}, {"zero": np.array(0.0, np.float32)})
case("Break_TopK_computed_k",
     [node("ReduceMax", ["k_in"], ["k"], keepdims=1), node("TopK", ["x", "k"], ["values", "indices"])],
     {"x": r.standard_normal((2, 9)).astype(np.float32), "k_in": np.array([2, 4, 3], np.int64)})
loop_body = helper.make_graph(
    [node("Add", ["v_in", "one_f"], ["v_out"]), node("Identity", ["cond_in"], ["cond_out"]), node("Mul", ["v_out", "v_out"], ["sq"])],
    "loop_body",
    [helper.make_tensor_value_info("i", INT64, []), helper.make_tensor_value_info("cond_in", TensorProto.BOOL, []),
     helper.make_tensor_value_info("v_in", TensorProto.FLOAT, [2])],
    [helper.make_tensor_value_info("cond_out", TensorProto.BOOL, []), helper.make_tensor_value_info("v_out", TensorProto.FLOAT, [2]),
     helper.make_tensor_value_info("sq", TensorProto.FLOAT, [2])],
    [numpy_helper.from_array(np.array(1.0, np.float32), "one_f")])
case("Break_Loop_computed_trip_count",
     [node("ReduceSum", ["counts"], ["trips"], keepdims=0), node("Loop", ["trips", "", "v0"], ["v", "squares"], body=loop_body)],
     {"counts": np.array([1, 2, 1], np.int64), "v0": r.standard_normal(2).astype(np.float32)})
case("Break_two_breaks",
     [node("ReduceSum", ["x"], ["s"], keepdims=0), node("Cast", ["s"], ["n"], to=INT64),
      node("Range", ["zero", "n", "one"], ["idx"]), node("Cast", ["idx"], ["f"], to=TensorProto.FLOAT),
      node("Greater", ["f", "half"], ["mask"]), node("NonZero", ["mask"], ["nz"]), node("Cast", ["nz"], ["y"], to=TensorProto.FLOAT)],
     {"x": pos_f32(5)}, {"zero": np.array(0, np.int64), "one": np.array(1, np.int64), "half": np.array(1.5, np.float32)})
then_branch = helper.make_graph(
    [node("Cast", ["s"], ["n"], to=INT64), node("Range", ["zero", "n", "one"], ["idx"]), node("Cast", ["idx"], ["out"], to=TensorProto.FLOAT)],
    "then_branch", [], [helper.make_tensor_value_info("out", TensorProto.FLOAT, None)])
else_branch = helper.make_graph(
    [node("Constant", [], ["out"], value=numpy_helper.from_array(np.array([-1.0], np.float32)))],
    "else_branch", [], [helper.make_tensor_value_info("out", TensorProto.FLOAT, None)])
case("Break_inside_If",
     [node("ReduceSum", ["x"], ["s"], keepdims=0), node("Greater", ["s", "zero_f"], ["cond"]),
      node("If", ["cond"], ["y"], then_branch=then_branch, else_branch=else_branch)],
     {"x": pos_f32(4)}, {"zero": np.array(0, np.int64), "one": np.array(1, np.int64), "zero_f": np.array(0.0, np.float32)})
# 24 nodes after the break that read the traced sum, so the segment after it compiles.
tail = []
for i in range(12):
    tail += [node("Mul", [f"t{i}", "scale" if i else "s"], [f"m{i}"]), node("Add", [f"m{i}", "shift"], [f"t{i + 1}"])]
case("Break_Range_long_tail",
     [node("ReduceSum", ["x"], ["s"], keepdims=0), node("Cast", ["s"], ["n"], to=INT64),
      node("Range", ["zero", "n", "one"], ["idx"]), node("Cast", ["idx"], ["t0"], to=TensorProto.FLOAT)] + tail +
     [node("Identity", ["t12"], ["y"])],
     {"x": pos_f32(6)}, {"zero": np.array(0, np.int64), "one": np.array(1, np.int64),
                         "scale": np.array(0.9, np.float32), "shift": np.array(0.25, np.float32)})

# ---------- long recurrences and loops ----------

# More than 16 steps, so a compiled model runs these as a while loop.
r = rng_for("long recurrences")
rf32 = lambda *shape, scale=0.5: (r.standard_normal(shape) * scale).astype(np.float32)
T, In, Hs = 40, 5, 4
case("LSTM_long_bidirectional_lengths",
     [node("LSTM", ["x", "w", "r", "b", "lengths", "h0", "c0"], ["y", "yh", "yc"], hidden_size=Hs, direction="bidirectional")],
     {"x": rf32(T, 3, In, scale=1.0), "h0": rf32(2, 3, Hs), "c0": rf32(2, 3, Hs)},
     {"w": rf32(2, 4 * Hs, In), "r": rf32(2, 4 * Hs, Hs), "b": rf32(2, 8 * Hs), "lengths": np.array([40, 23, 1], np.int32)})
case("GRU_long_reverse", [node("GRU", ["x", "w", "r", "b"], ["y", "yh"], hidden_size=Hs, direction="reverse")],
     {"x": rf32(T, 2, In, scale=1.0)}, {"w": rf32(1, 3 * Hs, In), "r": rf32(1, 3 * Hs, Hs), "b": rf32(1, 6 * Hs)})
case("RNN_long_forward", [node("RNN", ["x", "w", "r", "b"], ["y", "yh"], hidden_size=Hs)],
     {"x": rf32(T, 2, In, scale=1.0)}, {"w": rf32(1, Hs, In), "r": rf32(1, Hs, Hs), "b": rf32(1, 2 * Hs)})

# Long Loop and Scan bodies that read an outer value. A slice by the Loop counter needs
# a static counter, so that loop unrolls.
case("Loop_long_outer_read", [node("Loop", ["M", "cond", "acc0"], ["acc", "scans"], body=body)],
     {"x": rf32(3), "acc0": rf32(3)}, {"M": np.array(40, np.int64), "cond": np.array(True)})
long_scan_body = helper.make_graph(
    [node("Mul", ["sum_in", "decay"], ["kept"]), node("Add", ["kept", "x_t"], ["sum_out"]), node("Mul", ["sum_out", "x_t"], ["scan_t"])],
    "long_scan_body",
    [helper.make_tensor_value_info("sum_in", TensorProto.FLOAT, [3]), helper.make_tensor_value_info("x_t", TensorProto.FLOAT, [3])],
    [helper.make_tensor_value_info("sum_out", TensorProto.FLOAT, [3]), helper.make_tensor_value_info("scan_t", TensorProto.FLOAT, [3])])
case("Scan_long_outer_read",
     [node("Scan", ["init", "xs"], ["final", "ys"], body=long_scan_body, num_scan_inputs=1, scan_input_directions=[1],
           scan_output_axes=[1], scan_output_directions=[1])],
     {"init": rf32(3), "xs": rf32(T, 3, scale=1.0), "decay": rf32(3)})
slice_body = helper.make_graph(
    [node("Unsqueeze", ["i", "axis0"], ["start"]), node("Add", ["start", "one_i"], ["end"]),
     node("Slice", ["xs", "start", "end"], ["row"]), node("Add", ["acc_in", "row"], ["acc_out"]), node("Identity", ["cond_in"], ["cond_out"])],
    "slice_body",
    [helper.make_tensor_value_info("i", INT64, []), helper.make_tensor_value_info("cond_in", TensorProto.BOOL, []),
     helper.make_tensor_value_info("acc_in", TensorProto.FLOAT, [1, 3])],
    [helper.make_tensor_value_info("cond_out", TensorProto.BOOL, []), helper.make_tensor_value_info("acc_out", TensorProto.FLOAT, [1, 3])],
    [numpy_helper.from_array(np.array([0], np.int64), "axis0"), numpy_helper.from_array(np.array([1], np.int64), "one_i")])
case("Loop_long_counter_slice", [node("Loop", ["M", "", "acc0"], ["acc"], body=slice_body)],
     {"xs": rf32(20, 3), "acc0": rf32(1, 3)}, {"M": np.array(20, np.int64)})
