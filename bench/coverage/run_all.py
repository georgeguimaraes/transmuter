"""Run every ORT-validated case without missing ops through the Elixir importer.

Usage: run_all.py cases_dir [--rerun]
"""

import glob
import json
import os
import re
import subprocess
import sys

here = os.path.dirname(os.path.abspath(__file__))
cases_dir = sys.argv[1]
rerun = "--rerun" in sys.argv
lib = "".join(open(f).read() for f in glob.glob(os.path.join(here, "..", "..", "lib", "transmuter", "**", "*.ex"), recursive=True))
supported = set(re.findall(r'op_type(?::| ==) "(\w+)"', lib)) | set(re.findall(r'"(\w+)"', " ".join(re.findall(r'when \w+ in \[([^\]]*)\]', lib)))) | {"If", "Loop"}


def missing_ops(meta):
    return sorted({op.split(":")[-1] for op in meta.get("ops", {})} - supported)


for slug in sorted(os.listdir(cases_dir)):
    case = os.path.abspath(os.path.join(cases_dir, slug))
    meta_path, result_path = os.path.join(case, "meta.json"), os.path.join(case, "result.json")
    if not os.path.exists(meta_path):
        continue
    meta = json.load(open(meta_path))
    if not meta.get("ort_ok") or missing_ops(meta):
        continue
    if os.path.exists(result_path) and not rerun:
        continue
    if os.path.exists(result_path):
        os.remove(result_path)
    try:
        proc = subprocess.run(["mix", "run", os.path.join(here, "run_case.exs"), case], cwd=os.path.join(here, "..", ".."), capture_output=True, text=True, timeout=900)
        tail = (proc.stdout + proc.stderr).strip().splitlines()[-3:]
    except subprocess.TimeoutExpired:
        tail = ["timeout after 900s"]
    if not os.path.exists(result_path):
        json.dump({"status": "crash", "error": " / ".join(tail)[:600]}, open(result_path, "w"))
    print(json.load(open(result_path))["status"], meta["repo"], meta["file"], flush=True)
