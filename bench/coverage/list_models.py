"""Print a Markdown table of the tested models. Usage: list_models.py cases_dir [cases_dir ...]"""
import json, math, os, re, sys
from collections import defaultdict

import onnx
from onnx import helper

PRECISION = re.compile(r"(_no_gather)?_(q4f16|q4|q8|fp16|int8|uint8|quantized|bnb4)$")
# Precision variants differ from ONNX Runtime by rounding noise, see "Precision and quantization".
NOISY = re.compile(r"_(q4f16|fp16|int8|uint8|quantized)$")
INT4_TYPES = {onnx.TensorProto.INT4, onnx.TensorProto.UINT4}
DECODER = re.compile(r"^decoder(_with_past)?_model(_merged)?$")


def graphs(graph):
    yield graph
    for node in graph.node:
        for attr in node.attribute:
            if attr.g.node:
                yield from graphs(attr.g)


def parameters(path):
    model = onnx.load(path, load_external_data=False)
    sizes, types, skip, count = {}, {}, set(), 0
    for g in graphs(model.graph):
        for t in g.initializer:
            sizes[t.name], types[t.name] = math.prod(t.dims), t.data_type
        for node in g.node:
            attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
            if node.op_type == "Constant" and "value" in attrs:
                sizes[node.output[0]] = math.prod(attrs["value"].dims)
            elif node.op_type == "MatMulNBits":
                count += attrs["K"] * attrs["N"]
                skip.update(node.input[1:4])
            elif node.op_type == "GatherBlockQuantized":
                packed = types.get(node.input[0]) not in INT4_TYPES
                count += sizes.get(node.input[0], 0) * (8 // attrs.get("bits", 4) if packed else 1)
                skip.update(node.input[:3])
    return count + sum(n for name, n in sizes.items() if name not in skip)


def stem(file):
    return os.path.splitext(file)[0].removeprefix("onnx/")


def component(file):
    name = PRECISION.sub("", stem(file))
    return "decoder" if DECODER.match(os.path.basename(name)) else name


def human(n):
    for unit, size in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if n >= size:
            return f"{n / size:.3g}{unit}"
    return str(n)


RANDOM_OPS = {"RandomNormal", "RandomNormalLike", "RandomUniform", "RandomUniformLike", "Bernoulli", "Multinomial"}
LABELS = {"match": "match", "noise": "within noise", "random": "random", "timeout": "times out", "differs": "differ, see below"}


def outcome(meta, result):
    if result["status"] == "pass":
        return "match"
    if "timeout" in (result.get("error") or ""):
        return "timeout"
    if any(op.split(":")[-1] in RANDOM_OPS for op in meta.get("ops", {})):
        return "random"
    return "noise" if NOISY.search(stem(meta["file"])) else "differs"


def result(cases):
    outcomes = [(stem(meta["file"]), outcome(meta, result)) for meta, result in cases]
    if all(o == "match" for _, o in outcomes):
        return "matches"
    if len(outcomes) == 1:
        return {"match": "matches", "differs": "differs, see below"}.get(outcomes[0][1], LABELS[outcomes[0][1]])
    groups = [(", ".join(f for f, o in outcomes if o == key), label) for key, label in LABELS.items()]
    return "; ".join(f"{files} {label}" for files, label in groups if files)


repos = {}
for cases_dir in sys.argv[1:]:
    for slug in sorted(os.listdir(cases_dir)):
        meta_path = os.path.join(cases_dir, slug, "meta.json")
        result_path = os.path.join(cases_dir, slug, "result.json")
        if not (os.path.exists(meta_path) and os.path.exists(result_path)):
            continue
        meta, run = json.load(open(meta_path)), json.load(open(result_path))
        if meta.get("ort_ok"):
            repo = repos.setdefault(meta["repo"], {"task": meta["task"], "exported": meta.get("exported"), "files": {}})
            repo["files"][meta["file"]] = (meta, run)

print("| Model | Task | Parameters | Files | Compared with ONNX Runtime |")
print("|---|---|---|---|---|")
for name, repo in sorted(repos.items(), key=lambda r: (r[1]["task"], r[0].lower())):
    cases = list(repo["files"].values())
    per_component = defaultdict(int)
    for meta, _status in cases:
        c = component(meta["file"])
        per_component[c] = max(per_component[c], parameters(meta["onnx_path"]))
    files = ", ".join(stem(m["file"]) for m, _ in cases)
    if name.startswith("zoo/"):
        link = f"{name.removeprefix('zoo/')} (ONNX Model Zoo)"
    else:
        link = f"[{name}](https://huggingface.co/{name})" if "/" in name else name
        link += " (exported with Optimum)" if repo["exported"] else ""
    print(f"| {link} | {repo['task']} | {human(sum(per_component.values()))} | {files} | {result(cases)} |")
