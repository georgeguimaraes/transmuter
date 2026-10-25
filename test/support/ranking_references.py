"""Prints the reference values of the text ranking tests in test/transmuter/text_test.exs, from CrossEncoder.rank of sentence-transformers 6.2.0 and from ONNX Runtime.

Usage: uv run --with "sentence-transformers==6.2.0" --with "transformers>=5" --with torch --with onnxruntime --with onnx python -I test/support/ranking_references.py

Sentence-transformers 6.2.0 needs transformers 5, and test/support/pipeline_references.py needs transformers 4, so this script has its own uv run line.
"""

import gc
import tempfile

# The downloads go into this directory, and Python deletes it at exit.
scratch = tempfile.TemporaryDirectory()

import numpy as np
import onnx
import onnxruntime as ort
from huggingface_hub import snapshot_download
from sentence_transformers import CrossEncoder

query = "How many people live in Berlin?"
documents = [
    "New York City is famous for the Metropolitan Museum of Art.",
    "Berlin has a population of 3,520,031 registered inhabitants.",
    "Berlin is the capital of Germany.",
    "the quick brown fox jumps over the lazy dog " * 60,
]

# The configuration of ms-marco-MiniLM-L-6-v2 has the activation Identity, so the scores are logits.
marco = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2", device="cpu")
print("ms-marco", [(r["corpus_id"], round(float(r["score"]), 4)) for r in marco.rank(query, documents[:2])])
del marco

# Qwen3-Reranker on PyTorch gives the order, in bfloat16, the dtype of the checkpoint. The texts have at most 128 tokens, as in the test,
# so the long document is cut, and the end of the template replaces its last tokens. The token
# ids come from the preprocessing of sentence-transformers, with the default prompt, for each
# pair alone, so they have no padding.
reranker = CrossEncoder("Qwen/Qwen3-Reranker-0.6B", device="cpu")
reranker.max_seq_length = 128
print("qwen3-reranker rank", [(r["corpus_id"], round(float(r["score"]), 4)) for r in reranker.rank(query, documents)])
prompt = reranker.prompts[reranker.default_prompt_name]
ids = [reranker.preprocess([[query, d]], prompt=prompt)["input_ids"][0].tolist() for d in documents]
for row in ids:
    print("qwen3-reranker ids", len(row), row)
true_id, false_id = reranker[1].true_token_id, reranker[1].false_token_id
del reranker
gc.collect()

# The scores come from ONNX Runtime on the q4 export, with the token ids above, an empty cache
# and the positions from 0. As in pipeline_references.py, MatMulNBits gets accuracy_level 0 and
# the graph optimizations are off.
directory = snapshot_download("onnx-community/Qwen3-Reranker-0.6B-ONNX", allow_patterns=["onnx/model_q4.onnx"], local_dir=tempfile.mkdtemp(dir=scratch.name))
model = onnx.load(f"{directory}/onnx/model_q4.onnx")
for node in model.graph.node:
    for attribute in node.attribute:
        if node.op_type == "MatMulNBits" and attribute.name == "accuracy_level":
            attribute.i = 0
options = ort.SessionOptions()
options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
session = ort.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"])
del model
cache = {i.name: np.zeros((1, i.shape[1], 0, i.shape[3]), np.float32) for i in session.get_inputs() if i.name.startswith("past_key_values.")}
scores = []
for row in ids:
    n = len(row)
    feeds = {"input_ids": np.array([row], np.int64), "attention_mask": np.ones((1, n), np.int64), "position_ids": np.arange(n)[None], **cache}
    logits = session.run(["logits"], feeds)[0][0, -1]
    scores.append(round(float(logits[true_id] - logits[false_id]), 4))
print("qwen3-reranker onnxruntime q4", scores)
