"""Dump per-node output stats from ORT for one case. Usage: trace_ort.py case_dir"""
import json
import os
import sys

import numpy as np
import onnx
import onnx.numpy_helper
import onnxruntime as ort

case = sys.argv[1]
meta = json.load(open(os.path.join(case, "meta.json")))
model = onnx.load(os.path.join(case, meta["onnx_path"]))
NX = {"s64": np.int64, "s32": np.int32, "f32": np.float32, "u8": np.uint8, "f16": np.float16}
feeds = {}
for i in meta["inputs"]:
    a = np.fromfile(os.path.join(case, i["file"]), dtype=NX[i["dtype"]]).reshape(i["shape"])
    feeds[i["name"]] = a.astype(np.bool_) if i["name"] == "use_cache_branch" else a


def inline_ifs(nodes, known, initializers):
    """Replaces each If whose condition is a known value with the nodes of the branch that the
    value selects, and an Identity from each branch output to the If output of another name. The initializers
    of the branch go to `initializers`. The nodes inside the branch then have outputs of the
    main graph, as find_divergence.exs walks the branch."""
    result = []
    for node in nodes:
        if node.op_type == "If" and node.input[0] in known:
            name = "then_branch" if np.any(known[node.input[0]]) else "else_branch"
            branch = next(a.g for a in node.attribute if a.name == name)
            present = {t.name for t in initializers}
            initializers.extend(t for t in branch.initializer if t.name not in present)
            known.update(bools(branch.initializer))
            result += inline_ifs(branch.node, known, initializers)
            # A branch output can have the name of the If output, as in the merged decoders.
            result += [onnx.helper.make_node("Identity", [b.name], [o], name=f"{node.name}/{o}") for b, o in zip(branch.output, node.output) if b.name != o]
        else:
            result.append(node)
    return result


def bools(initializers):
    return {t.name: onnx.numpy_helper.to_array(t) for t in initializers if t.data_type == onnx.TensorProto.BOOL}


# The new nodes are copies, because the branches belong to the If nodes that the graph drops.
inlined = [onnx.NodeProto.FromString(n.SerializeToString()) for n in inline_ifs(model.graph.node, dict(feeds) | bools(model.graph.initializer), model.graph.initializer)]
del model.graph.node[:]
model.graph.node.extend(inlined)
names = [o for n in model.graph.node for o in n.output if o]
existing = {o.name for o in model.graph.output}
for name in names:
    if name not in existing:
        model.graph.output.append(onnx.ValueInfoProto(name=name))
opts = ort.SessionOptions()
opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
sess = ort.InferenceSession(model.SerializeToString(), opts, providers=["CPUExecutionProvider"])
outs = dict(zip([o.name for o in sess.get_outputs()], sess.run(None, feeds)))
stats = {}
for name in names:
    a = np.clip(np.nan_to_num(np.asarray(outs[name], dtype=np.float64), nan=0.0, posinf=3e38, neginf=-3e38), -3e38, 3e38)
    flat = a.reshape(-1)
    idx = [(k * 2654435761) % flat.size for k in range(1024)] if flat.size else []
    raw = np.asarray(outs[name])
    # An integer sum is exact: int64 for the narrow types, Python integers for the 64-bit types.
    if np.issubdtype(raw.dtype, np.integer):
        abssum = int(np.abs(raw.reshape(-1).astype(np.int64 if raw.dtype.itemsize < 8 else object)).sum())
    else:
        abssum = float(np.abs(a).sum())
    stats[name] = {"shape": list(a.shape), "absmax": float(np.abs(a).max()) if a.size else 0.0, "abssum": abssum, "sample": [float(flat[i]) for i in idx]}
json.dump(stats, open(os.path.join(case, "ort_trace.json"), "w"))
# DUMP=name,name saves those node outputs in full, for find_divergence.exs OVERRIDE.
for name in filter(None, os.environ.get("DUMP", "").split(",")):
    a = np.ascontiguousarray(np.asarray(outs[name], dtype=np.float32))
    a.tofile(os.path.join(case, "full_" + name.replace("/", "_") + ".bin"))
    print("dumped", name, a.shape)
print(len(stats), "node outputs")
