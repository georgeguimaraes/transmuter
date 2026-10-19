"""Prints the reference values of test/transmuter/text_test.exs, from the pipelines of Python transformers and sentence-transformers.

Usage: uv run --with transformers --with sentence-transformers --with torch python -I test/support/pipeline_references.py
"""

from sentence_transformers import SentenceTransformer
from transformers import pipeline

for repo in ["sentence-transformers/all-MiniLM-L6-v2", "BAAI/bge-small-en-v1.5"]:
    embedding = SentenceTransformer(repo, device="cpu").encode(["The cat sat on the mat."])[0]
    print(repo, [round(float(x), 4) for x in embedding[:4]])

long = "the quick brown fox jumps over the lazy dog " * 60
embedding = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu").encode([long])[0]
print("all-MiniLM-L6-v2, 541 tokens", [round(float(x), 4) for x in embedding[:4]])

rerank = pipeline("text-classification", model="cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")
query = "How many people live in Berlin?"
for passage in ["Berlin has a population of 3,520,031 registered inhabitants.", "New York City is famous for the Metropolitan Museum of Art."]:
    print("rerank", rerank({"text": query, "text_pair": passage}, top_k=None))

# Transmuter gives byte offsets, so the character offsets of Python become byte offsets here.
ner = pipeline("token-classification", model="dslim/bert-base-NER", device="cpu")
text = "Hugging Face Inc. is a company based in New York City. Its headquarters are in DUMBO, São Paulo."
for strategy in ["simple", "first"]:
    entities = ner(text, aggregation_strategy=strategy)
    print(strategy, [(len(text[: e["start"]].encode()), len(text[: e["end"]].encode()), e["entity_group"], e["word"], round(float(e["score"]), 5)) for e in entities])

print("unknown token", [(e["entity_group"], e["word"]) for e in ner("My friend Angela☃Merkel lives in Berlin.", aggregation_strategy="simple")])

fill = pipeline("fill-mask", model="distilbert/distilroberta-base", device="cpu")
print("fill", [(r["token_str"], round(r["score"], 5)) for r in fill("Paris is the <mask> of France.")[:2]])

# Generation: greedy, 12 new tokens.
from transformers import AutoModelForCausalLM, AutoTokenizer

messages = [{"role": "user", "content": "What is the capital of France?"}]
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)
output = model.generate(**inputs, max_new_tokens=12, do_sample=False)
print("smollm2", repr(tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)))

tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
inputs = tokenizer("The history of the printing press begins", return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=12, do_sample=False)
print("gpt2", repr(tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)))

tokenizer = AutoTokenizer.from_pretrained("onnx-community/Qwen3-0.6B-ONNX")
print("qwen3 template", repr(tokenizer.apply_chat_template([{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"}], tokenize=False, add_generation_prompt=True, enable_thinking=False)))

# Gemma 3 counts the valid cache rows from the length of the attention mask. ONNX Runtime
# with a mask of the exact length and a cache that grows from 0 rows, greedy.
import tempfile

import numpy as np
import onnxruntime as ort
from huggingface_hub import snapshot_download

repo = "onnx-community/gemma-3-270m-it-ONNX"
tokenizer = AutoTokenizer.from_pretrained(repo)
ids = tokenizer.apply_chat_template([{"role": "user", "content": "Write a long story about a fox who lives near a river."}], add_generation_prompt=True)
if not isinstance(ids, list):
    ids = ids["input_ids"]
# ONNX Runtime does not read external data through the symbolic links of the Hub cache.
directory = snapshot_download(repo, allow_patterns=["onnx/model_q4.onnx*"], local_dir=tempfile.mkdtemp())
session = ort.InferenceSession(f"{directory}/onnx/model_q4.onnx", providers=["CPUExecutionProvider"])
past = {i.name: np.zeros((1, i.shape[1], 0, i.shape[3]), np.float32) for i in session.get_inputs() if i.name.startswith("past_key_values")}
names = [o.name for o in session.get_outputs()]
tokens, step = [], ids
for _ in range(12):
    feeds = {"input_ids": np.array([step]), "attention_mask": np.ones((1, len(ids) + len(tokens)), np.int64), **past}
    outputs = dict(zip(names, session.run(None, feeds)))
    tokens.append(int(outputs["logits"][0, -1].argmax()))
    past = {name.replace("present", "past_key_values"): value for name, value in outputs.items() if name.startswith("present")}
    step = tokens[-1:]
print("gemma3", repr(tokenizer.decode(tokens, skip_special_tokens=True)))
