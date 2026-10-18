"""Recompute ORT reference outputs with optimizations disabled for cases prepared before that change."""
import json
import os
import sys

import numpy as np
import onnxruntime as ort

NX = {"s64": np.int64, "s32": np.int32, "f32": np.float32, "u8": np.uint8, "f16": np.float16}
cases = sys.argv[1]
for slug in sorted(os.listdir(cases)):
    case = os.path.join(cases, slug)
    mp = os.path.join(case, "meta.json")
    if not os.path.exists(mp):
        continue
    meta = json.load(open(mp))
    if not meta.get("ort_ok") or meta.get("unoptimized_ref"):
        continue
    feeds = {}
    for i in meta["inputs"]:
        a = np.fromfile(os.path.join(case, i["file"]), dtype=NX[i["dtype"]]).reshape(i["shape"])
        feeds[i["name"]] = a.astype(np.bool_) if i["name"] == "use_cache_branch" else a
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    opts.log_severity_level = 3
    try:
        outs = ort.InferenceSession(meta["onnx_path"], opts, providers=["CPUExecutionProvider"]).run(None, feeds)
    except Exception as e:
        for f in os.listdir(case):
            os.remove(os.path.join(case, f))
        print(f"REGENERATE {meta['repo']} {meta['file']}: {str(e).splitlines()[0][:150]}", flush=True)
        continue
    worst = 0.0
    for spec, out in zip(meta["outputs"], outs):
        path = os.path.join(case, spec["file"])
        old = np.fromfile(path, dtype=NX[spec["dtype"]]).astype(np.float64)
        out = np.ascontiguousarray(out).astype(np.uint8 if out.dtype == np.bool_ else out.dtype)
        if old.size == out.size and old.size:
            worst = max(worst, float(np.nanmax(np.abs(old - out.reshape(-1).astype(np.float64)))))
        out.tofile(path)
    meta["unoptimized_ref"] = True
    json.dump(meta, open(mp, "w"), indent=1)
    rp = os.path.join(case, "result.json")
    if os.path.exists(rp):
        os.remove(rp)
    print(f"{worst:10.3g} {meta['repo']} {meta['file']}", flush=True)
