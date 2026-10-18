"""Summarize the coverage run. Usage: report.py cases_dir [report.json]"""

import glob
import json
import os
import re
import sys
from collections import Counter, defaultdict

here = os.path.dirname(os.path.abspath(__file__))
cases_dir = sys.argv[1]
lib = "".join(open(f).read() for f in glob.glob(os.path.join(here, "..", "..", "lib", "**", "*.ex"), recursive=True))
supported = set(re.findall(r'op_type(?::| ==) "(\w+)"', lib)) | set(re.findall(r'"(\w+)"', " ".join(re.findall(r'(?:when \w+|op_type) in \[([^\]]*)\]', lib))))

files = []
for slug in sorted(os.listdir(cases_dir)):
    meta_path = os.path.join(cases_dir, slug, "meta.json")
    if not os.path.exists(meta_path):
        continue
    meta = json.load(open(meta_path))
    result_path = os.path.join(cases_dir, slug, "result.json")
    result = json.load(open(result_path)) if os.path.exists(result_path) else None
    missing = sorted({op.split(":")[-1] for op in meta.get("ops", {})} - supported)
    if not meta.get("ort_ok"):
        status = "harness"
    elif missing:
        status = "missing_ops"
    elif result is None:
        status = "pending"
    else:
        status = result["status"]
    files.append({"repo": meta["repo"], "task": meta["task"], "file": meta["file"], "status": status, "missing": missing, "result": result, "meta_error": meta.get("error"), "ops": meta.get("ops", {})})

repos = defaultdict(list)
for f in files:
    repos[(f["task"], f["repo"])].append(f)

print(f"{len(repos)} repos, {len(files)} files")
print("file status:", dict(Counter(f["status"] for f in files)))
repo_pass = [r for r, fs in repos.items() if all(f["status"] == "pass" for f in fs)]
repo_partial = [r for r, fs in repos.items() if any(f["status"] == "pass" for f in fs) and r not in repo_pass]
testable = [r for r, fs in repos.items() if not all(f["status"] == "harness" for f in fs)]
print(f"repos fully passing: {len(repo_pass)}/{len(testable)} testable, partial: {len(repo_partial)}")

print("\n## Per repo")
for (task, repo), fs in sorted(repos.items()):
    parts = []
    for f in fs:
        name = f["file"].removeprefix("onnx/").removesuffix(".onnx")
        detail = ""
        if f["status"] == "missing_ops":
            detail = " " + ",".join(f["missing"])
        elif f["status"] in ("error", "crash"):
            detail = " " + (f["result"].get("error") or "")[:160].replace("\n", " ")
        elif f["status"] == "mismatch":
            bad = [d for d in f["result"]["diffs"] if not d["ok"]]
            detail = " " + "; ".join(f"{d['name']} abs={d.get('max_abs', '?'):.3g} ref={d.get('max_ref', 0):.3g}" if "max_abs" in d else f"{d['name']} shape {d['shape']}!={d['expected']}" for d in bad[:2])
        elif f["status"] == "harness":
            detail = " " + (f["meta_error"] or "")[:120]
        elif f["status"] == "pass":
            detail = f" {f['result']['run_ms']}ms"
        parts.append(f"{name}: {f['status']}{detail}")
    print(f"{task:30} {repo:58} | " + " | ".join(parts))

print("\n## Missing ops: files blocked (repos blocked)")
blocked_files, blocked_repos = Counter(), defaultdict(set)
for f in files:
    if f["status"] == "missing_ops":
        for op in f["missing"]:
            blocked_files[op] += 1
            blocked_repos[op].add(f["repo"])
print(", ".join(f"{op} {n} ({len(blocked_repos[op])})" for op, n in blocked_files.most_common()))

print("\n## Greedy unlock order (files with no other missing op)")
remaining = [set(f["missing"]) for f in files if f["status"] == "missing_ops"]
done = set()
while remaining:
    op, _ = Counter(op for m in remaining for op in m - done).most_common(1)[0]
    done.add(op)
    unlocked = sum(1 for m in remaining if m <= done)
    remaining = [m for m in remaining if not m <= done]
    print(f"+{op}: unlocks {unlocked}, {len(remaining)} files still blocked")

if len(sys.argv) > 2:
    json.dump({"files": files}, open(sys.argv[2], "w"), indent=1)
