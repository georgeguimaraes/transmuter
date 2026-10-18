import json
import re
import sys
import urllib.request
from collections import defaultdict

PER_TASK = int(sys.argv[1]) if len(sys.argv) > 1 else 4
MAX_BYTES = 900 * 1024 * 1024
VARIANT = re.compile(r"_(quantized|fp16|q4|q4f16|int8|uint8|bnb4|q8|bf16)\.onnx$")


def get(url):
    with urllib.request.urlopen(url) as r:
        return json.load(r)


candidates = []
for author in ["onnx-community", "Xenova"]:
    candidates += get(f"https://huggingface.co/api/models?author={author}&sort=downloads&direction=-1&limit=1000")

by_task = defaultdict(list)
for m in sorted(candidates, key=lambda m: -m.get("downloads", 0)):
    task = m.get("pipeline_tag")
    if task and len(by_task[task]) < PER_TASK * 4:
        by_task[task].append(m["id"])

corpus = []
for task, ids in sorted(by_task.items()):
    picked = 0
    for repo in ids:
        if picked == PER_TASK:
            break
        info = get(f"https://huggingface.co/api/models/{repo}?blobs=true")
        files = {s["rfilename"]: s.get("size", 0) for s in info.get("siblings", [])}
        if any(f.endswith((".onnx_data", ".onnx.data")) or "_data" in f.split("/")[-1] for f in files):
            continue
        onnx = {f: size for f, size in files.items() if f.startswith("onnx/") and f.endswith(".onnx") and not VARIANT.search(f)}
        if not onnx or sum(onnx.values()) > MAX_BYTES:
            continue
        corpus.append({"repo": repo, "task": task, "downloads": info.get("downloads", 0), "files": sorted(onnx)})
        picked += 1
        print(f"{task:32} {repo:60} {sum(onnx.values()) / 1e6:7.0f}MB {sorted(onnx)}", flush=True)

json.dump(corpus, open(sys.argv[2] if len(sys.argv) > 2 else "corpus.json", "w"), indent=1)
print(len(corpus), "repos", sum(len(c["files"]) for c in corpus), "files")
