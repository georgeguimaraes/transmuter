"""Download each corpus file, generate inputs, run ORT and save the reference outputs.

Usage: record_references.py corpus.json cases_dir
"""

import json
import os
import re
import subprocess
import sys
import traceback
from collections import Counter

import numpy as np
import onnx
import onnxruntime as ort
from huggingface_hub import hf_hub_download

corpus_path, cases_dir = sys.argv[1], sys.argv[2]
corpus = json.load(open(corpus_path))
rng = np.random.default_rng(0)

NP_TO_NX = {"int64": "s64", "int32": "s32", "float32": "f32", "float16": "f16", "bool": "u8", "uint8": "u8", "int8": "s8", "float64": "f64"}
ORT_TO_NP = {"tensor(int64)": np.int64, "tensor(int32)": np.int32, "tensor(float)": np.float32, "tensor(float16)": np.float16, "tensor(bool)": np.bool_, "tensor(uint8)": np.uint8, "tensor(int8)": np.int8, "tensor(double)": np.float64}


def try_json(repo, name):
    try:
        return json.load(open(hf_hub_download(repo, name)))
    except Exception:
        return {}


def op_inventory(graph, counter):
    for node in graph.node:
        counter[(node.domain or "ai.onnx", node.op_type)] += 1
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.GRAPH:
                op_inventory(attr.g, counter)
            for g in attr.graphs:
                op_inventory(g, counter)
    return counter


def image_size(config, pre):
    for key in (["crop_size", "size"] if pre.get("do_center_crop", True) else ["size"]):
        v = pre.get(key)
        if isinstance(v, int):
            return v, v
        if isinstance(v, dict):
            if "height" in v:
                return v["height"], v["width"]
            if "shortest_edge" in v:
                return v["shortest_edge"], v["shortest_edge"]
    for c in [config, config.get("vision_config", {}), config.get("encoder", {})]:
        if "image_size" in c:
            s = c["image_size"]
            return (s, s) if isinstance(s, int) else tuple(s)
    return 224, 224


def dim_value(param, ctx):
    # Optimum writes a product of axes as "batch_size x num_heads".
    param = param.replace(" x ", " * ")
    names = re.findall(r"[A-Za-z_][A-Za-z_0-9]*", param)
    env = {}
    for n in names:
        low = n.lower()
        if n in ctx:
            env[n] = ctx[n]
        elif "batch" in low:
            env[n] = 1
        elif "height" in low:
            env[n] = ctx["height"]
        elif "width" in low:
            env[n] = ctx["width"]
        elif "channel" in low:
            env[n] = 3
        elif "past" in low and "encoder" in low:
            env[n] = ctx["encoder_sequence_length"]
        elif "past" in low:
            env[n] = ctx["past"]
        elif "total" in low:
            env[n] = ctx["past"] + ctx["seq"]
        elif "encoder" in low:
            env[n] = ctx["encoder_sequence_length"]
        elif "frame" in low:
            env[n] = ctx["num_frames"]
        else:
            env[n] = ctx["seq"]
    return int(eval(param, {}, env))


def kv_heads(text):
    """The heads of the KV cache, as Optimum counts them: 1 for the multi-query attention of Falcon."""
    if text.get("multi_query") and not text.get("new_decoder_architecture"):
        return 1
    return text.get("num_kv_heads") or text.get("num_key_value_heads") or text.get("num_attention_heads") or text.get("n_head", 1)


def make_inputs(sess, config, pre, seq, past, dims=None):
    if not any("past" in i.name for i in sess.get_inputs()):
        past = 0
    h, w = image_size(config, pre)
    text = config.get("text_config", config)
    vocab = min(text.get("vocab_size", 100) or 100, 1000)
    ctx = {"seq": seq, "past": past, "height": h, "width": w, "encoder_sequence_length": 8, "num_frames": config.get("num_frames", 8), "num_heads": kv_heads(text)}
    if config.get("model_type") == "whisper":
        ctx["encoder_sequence_length"] = config.get("max_source_positions", 1500)
    ctx.update(dims or {})
    feeds = {}
    for i in sess.get_inputs():
        name, dtype = i.name, ORT_TO_NP[i.type]
        # An unnamed last axis is a sequence; other unnamed axes are the batch.
        shape = [d if isinstance(d, int) else (ctx["seq"] if j == len(i.shape) - 1 else 1) if d is None else dim_value(d, ctx) for j, d in enumerate(i.shape)]
        if name == "input_features" and config.get("model_type") == "whisper" and not all(isinstance(d, int) for d in i.shape[1:]):
            shape = [1, config["num_mel_bins"], 2 * config["max_source_positions"]]
        if name == "past_values" and "context_length" in config:
            shape[1] = config["context_length"]
        if name in ("input_values", "audio_values", "input_audio") and len(shape) == 2:
            shape = [1, 16000]
        if dtype == np.bool_:
            feeds[name] = np.ones(shape, dtype=np.bool_)
        elif np.issubdtype(dtype, np.integer):
            if "mask" in name:
                feeds[name] = np.ones(shape, dtype=dtype)
            elif name in ("sr", "sample_rate", "sampling_rate"):
                feeds[name] = np.full(shape, 16000, dtype=dtype)
            elif "position_ids" in name:
                feeds[name] = np.broadcast_to(np.arange(past, past + shape[-1]), shape).astype(dtype).copy()
            elif "input_ids" in name and config.get("model_type") == "grounding-dino":
                ids = rng.integers(2000, 3000, shape)
                ids[..., 0], ids[..., -2], ids[..., -1] = 101, 1012, 102
                feeds[name] = ids.astype(dtype)
            elif "input_ids" in name:
                feeds[name] = rng.integers(1, vocab, shape).astype(dtype)
            else:
                feeds[name] = np.zeros(shape, dtype=dtype)
        else:
            if name == "speed":
                feeds[name] = np.ones(shape, dtype=dtype)
            elif "pixel" in name:
                feeds[name] = rng.uniform(-1, 1, shape).astype(dtype)
            else:
                feeds[name] = (rng.standard_normal(shape) * 0.5).astype(dtype)
    return feeds


def exact_model(path):
    """ORT runs MatMulNBits with accuracy_level 4 on int8 activations, an approximation.
    Returns a copy with accuracy_level 0, so the reference is the exact dequantized model."""
    model = onnx.load(path, load_external_data=False)
    nodes = [n for n in model.graph.node if n.op_type == "MatMulNBits" and any(a.name == "accuracy_level" and a.i != 0 for a in n.attribute)]
    if not nodes:
        return path
    model = onnx.load(path)
    for n in model.graph.node:
        for a in n.attribute:
            if n.op_type == "MatMulNBits" and a.name == "accuracy_level":
                a.i = 0
    out = os.path.join(os.path.dirname(os.path.abspath(cases_dir)), "exact_models", os.path.basename(os.path.dirname(path)) + "_" + os.path.basename(path))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    onnx.save(model, out, save_as_external_data=True, location=os.path.basename(out) + ".data")
    return out


def save(arr, path):
    arr = np.ascontiguousarray(arr)
    if arr.dtype == np.bool_:
        arr = arr.astype(np.uint8)
    arr.tofile(path)
    return {"dtype": NP_TO_NX[str(arr.dtype)], "shape": list(arr.shape)}


def external_files(path):
    """External data files named by the model's initializers, relative to the model."""
    model = onnx.load(path, load_external_data=False)
    return sorted({e.value for t in model.graph.initializer for e in t.external_data if e.key == "location"})


# The clone flag of cp: -c on macOS, and --reflink=auto of GNU cp elsewhere, which copies when
# the file system cannot clone.
CLONE = "-c" if sys.platform == "darwin" else "--reflink=auto"


def fetch(entry, file, slug):
    """Returns a local model path. Local exports are used as they are. Hub files with
    external data are cloned into a real directory, because ORT refuses external data
    reached through the HF cache symlinks (APFS clones are copy-on-write)."""
    if "local_dir" in entry:
        return os.path.join(entry["local_dir"], file)
    repo = entry["repo"]
    path = hf_hub_download(repo, file)
    data = external_files(path)
    if not data:
        return path
    local = os.path.join(os.path.dirname(os.path.abspath(cases_dir)), "models", slug)
    os.makedirs(local, exist_ok=True)
    folder = os.path.dirname(file)
    for f in [os.path.basename(file)] + data:
        target = os.path.join(local, f)
        if not os.path.exists(target):
            source = path if f == os.path.basename(file) else hf_hub_download(repo, os.path.join(folder, f) if folder else f)
            subprocess.run(["cp", CLONE, os.path.realpath(source), target], check=True)
    return os.path.join(local, os.path.basename(file))


def local_json(directory, name):
    try:
        return json.load(open(os.path.join(directory, name)))
    except Exception:
        return {}


for entry in corpus:
    repo = entry["repo"]
    if "local_dir" in entry:
        config, pre = local_json(entry["local_dir"], "config.json"), local_json(entry["local_dir"], "preprocessor_config.json")
    else:
        config, pre = try_json(repo, "config.json"), try_json(repo, "preprocessor_config.json")

    for file in entry["files"]:
        slug = (repo + "__" + file).replace("/", "__")
        case = os.path.join(cases_dir, slug)
        if os.path.exists(os.path.join(case, "meta.json")):
            continue
        os.makedirs(case, exist_ok=True)
        meta = {"repo": repo, "task": entry["task"], "file": file, "downloads": entry.get("downloads", 0)}
        try:
            path = fetch(entry, file, slug)
            meta["onnx_path"] = path
            model = onnx.load(path, load_external_data=False)
            meta["opset"] = {o.domain or "ai.onnx": o.version for o in model.opset_import}
            meta["ops"] = {f"{d}:{op}" if d != "ai.onnx" else op: n for (d, op), n in op_inventory(model.graph, Counter()).items()}
            del model

            # Unoptimized ORT: the default optimizer mis-folds some exports (timesformer, mobileclip_s0) by up to 0.67 or 1751 absolute.
            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            sess = ort.InferenceSession(exact_model(path), opts, providers=["CPUExecutionProvider"])
            errors = []
            for seq, past in [(8, 3), (1, 3), (8, 8), (77, 3)]:
                try:
                    feeds = make_inputs(sess, config, pre, seq, past, entry.get("dims"))
                    outs = sess.run(None, feeds)
                    break
                except Exception as e:
                    errors.append(str(e).splitlines()[0][:300])
            else:
                raise RuntimeError("ORT failed for all input shapes: " + " | ".join(errors))

            meta["inputs"] = [{"name": k, "file": f"in_{n}.bin", **save(v, os.path.join(case, f"in_{n}.bin"))} for n, (k, v) in enumerate(feeds.items())]
            meta["outputs"] = [{"name": o.name, "file": f"out_{n}.bin", **save(v, os.path.join(case, f"out_{n}.bin"))} for n, (o, v) in enumerate(zip(sess.get_outputs(), outs))]
            meta["ort_ok"] = True
            meta["unoptimized_ref"] = True
        except Exception as e:
            meta["ort_ok"] = False
            meta["error"] = f"{type(e).__name__}: {str(e).splitlines()[0][:400] if str(e) else ''}"
            traceback.print_exc()
        json.dump(meta, open(os.path.join(case, "meta.json"), "w"), indent=1)
        print(f"{'OK  ' if meta['ort_ok'] else 'FAIL'} {repo} {file} {meta.get('error', '')}", flush=True)
