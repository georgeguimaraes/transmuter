"""Prints ORT reference values for test/transmuter_test.exs. Inputs follow the same patterns as the tests.

Usage: uv run --with onnxruntime --with numpy --with huggingface_hub python -I test/support/reference.py
"""

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download


def pattern(shape):
    n = int(np.prod(shape))
    return (np.sin(np.arange(n, dtype=np.float32) * np.float32(0.01)) * np.float32(0.5)).reshape(shape)


def run(repo, file, feeds):
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(hf_hub_download(repo, file), opts, providers=["CPUExecutionProvider"])
    return dict(zip([o.name for o in sess.get_outputs()], sess.run(None, feeds)))


def show(label, a):
    print(label, np.round(a, 4).tolist())


ids = np.array([[101, 7592, 2088, 2003, 1037, 3231, 102]], dtype=np.int64)
out = run("Xenova/all-MiniLM-L6-v2", "onnx/model.onnx", {"input_ids": ids, "attention_mask": np.ones_like(ids), "token_type_ids": np.zeros_like(ids)})
show("minilm last_hidden_state[0, 0..2, 0..2]", out["last_hidden_state"][0, 0:3, 0:3])

out = run("Xenova/segformer-b0-finetuned-ade-512-512", "onnx/model.onnx", {"pixel_values": pattern((1, 3, 512, 512))})
show("segformer logits[0, 0..2, 0, 0..2]", out["logits"][0, 0:3, 0, 0:3])

sess_inputs = ort.InferenceSession(hf_hub_download("Xenova/whisper-tiny.en", "onnx/decoder_model_merged.onnx"), providers=["CPUExecutionProvider"]).get_inputs()
feeds = {"input_ids": np.array([[1000]], dtype=np.int64), "use_cache_branch": np.array([True]), "encoder_hidden_states": pattern((1, 1500, 384))}
for i in sess_inputs:
    if i.name.startswith("past_key_values"):
        feeds[i.name] = pattern((1, 6, 1500 if ".encoder." in i.name else 3, 64))
out = run("Xenova/whisper-tiny.en", "onnx/decoder_model_merged.onnx", feeds)
show("whisper merged logits[0, 0, 0..4]", out["logits"][0, 0, 0:5])
