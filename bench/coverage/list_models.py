"""Print the tables of the tested models for the README. Usage: list_models.py [cases_dir ...]

tested_models.json keeps one entry for each model that a coverage run compared with ONNX Runtime.
The models of the given case dirs get new entries, and the entries of the other models stay.
end_to_end.json keeps the checks that ran a model from input to final result. A check with a
"repo" goes in the row of that tested model. A "difference" or a "note" links to its README heading.
"""
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
LABELS = {"match": "match", "noise": "within noise", "random": "random", "timeout": "times out", "differs": "differ"}


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
        return {"match": "matches", "differs": "differs"}.get(outcomes[0][1], LABELS[outcomes[0][1]])
    groups = [(", ".join(f for f, o in outcomes if o == key), label) for key, label in LABELS.items()]
    return ". ".join(f"{files} {label}" for files, label in groups if files)


HERE = os.path.dirname(os.path.abspath(__file__))
TESTED = os.path.join(HERE, "tested_models.json")
END_TO_END = os.path.join(HERE, "end_to_end.json")

AREAS = [
    ("Language models", ["text-generation"]),
    ("Text encoders", ["feature-extraction", "sentence-similarity", "fill-mask", "text-classification",
                       "token-classification", "question-answering", "zero-shot-classification", "multiple-choice"]),
    ("Translation and summarization", ["translation", "summarization"]),
    ("Speech and audio", ["automatic-speech-recognition", "text-to-speech", "audio-classification", "voice-activity-detection"]),
    ("Vision", ["image-classification", "image-feature-extraction", "face-recognition", "object-detection", "image-segmentation",
                "mask-generation", "depth-estimation", "keypoint-detection", "image-to-image", "inpainting"]),
    ("Vision-language models", ["image-text-to-text", "image-to-text", "zero-shot-image-classification", "zero-shot-object-detection"]),
    ("Image generation", ["text-to-image"]),
    ("Time series", ["time-series-forecasting"]),
    ("Video", ["video-classification"]),
]
TASK_ORDER = {task: (area, i) for area, tasks in AREAS for i, task in enumerate(tasks)}


def entry(name, repo):
    cases = list(repo["files"].values())
    per_component = defaultdict(int)
    for meta, _status in cases:
        c = component(meta["file"])
        per_component[c] = max(per_component[c], parameters(meta["onnx_path"]))
    files = [stem(m["file"]) for m, _ in cases]
    return {"repo": name, "task": repo["task"], "exported": bool(repo["exported"]),
            "parameters": human(sum(per_component.values())), "files": files, "result": result(cases)}


def link(entry):
    name = entry["repo"]
    if name.startswith("zoo/"):
        return f"{name.removeprefix('zoo/')} (ONNX Model Zoo)"
    text = f"[{name}](https://huggingface.co/{name})" if "/" in name else name
    return text + (" (exported with Optimum)" if entry["exported"] else "")


def anchor(heading):
    return re.sub(r"[^a-z0-9 _-]", "", heading.lower()).replace(" ", "-")


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

tested = {e["repo"]: e for e in json.load(open(TESTED))}
for name, repo in repos.items():
    old = tested.get(name, {})
    tested[name] = entry(name, repo) | ({"difference": old["difference"]} if "difference" in old else {})
if repos:
    with open(TESTED, "w") as f:
        json.dump(sorted(tested.values(), key=lambda e: e["repo"].lower()), f, indent=1)
        f.write("\n")

rows = {name: {"model": link(e), "task": e["task"], "parameters": e["parameters"], "files": ", ".join(e["files"]),
               "tensors": e["result"] + (f", see [Known differences](#{e['difference']})" if "difference" in e else ""),
               "end_to_end": ""} for name, e in tested.items()}
notes = []
for check in json.load(open(END_TO_END)):
    row = rows[check["repo"]] if "repo" in check else rows.setdefault(check["model"], {
        "model": check["model"], "task": check["task"], "parameters": "", "files": "", "tensors": ""})
    row["end_to_end"] = check["check"]
    if "note" in check:
        row["end_to_end"] += f" See [{check['name']}](#{anchor(check['name'])})."
        notes.append(check)

print("### Tested models")
print()
print('"Within noise" means a precision variant that differs from ONNX Runtime by its rounding noise. '
      '"Random" means a model that samples random values. "End to end" means a run from input to final result '
      "that agrees with ONNX Runtime.")
area_rows = defaultdict(list)
for row in rows.values():
    area, order = TASK_ORDER[row["task"]]
    area_rows[area].append((order, row["model"].lstrip("[").lower(), row))
for area, _tasks in AREAS:
    group = sorted(area_rows[area], key=lambda r: r[:2])
    print()
    print("<details>")
    print(f"<summary>{area}: {len(group)} model{'s' if len(group) != 1 else ''}</summary>")
    print()
    print("| Model | Task | Parameters | Files | Tensors against ONNX Runtime | End to end |")
    print("|---|---|---|---|---|---|")
    for _order, _name, r in group:
        print(f"| {r['model']} | {r['task']} | {r['parameters']} | {r['files']} | {r['tensors']} | {r['end_to_end']} |")
    print()
    print("</details>")
print()
print("### End-to-end notes")
for check in notes:
    print()
    print(f"#### {check['name']}")
    print()
    print(f"{check['check']} {check['note']}")
