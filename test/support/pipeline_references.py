"""Prints the reference values of test/transmuter/text_test.exs, test/transmuter/vision_test.exs and test/transmuter/audio_test.exs, from the pipelines of Python transformers and sentence-transformers.

Usage: uv run --with "transformers<5" --with sentence-transformers --with torch --with onnxruntime --with pillow --with sentencepiece --with protobuf --with sacremoses --with "datasets<4" --with soundfile --with librosa --with torchvision python -I test/support/pipeline_references.py

Transformers 5 removed the question answering pipeline, so the script uses transformers 4. The processor of Qwen2-VL loads a video processor, which needs torchvision.
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

zero_shot = pipeline("zero-shot-classification", model="typeform/distilbert-base-uncased-mnli", device="cpu")
text = "I have a problem with my iphone that needs to be resolved asap!!"
for labels, multi_label in [(["urgent", "not urgent", "phone", "tablet", "computer"], False), (["urgent", "not urgent", "phone", "tablet", "computer"], True), (["urgent"], False)]:
    r = zero_shot(text, candidate_labels=labels, multi_label=multi_label)
    print("zero-shot", multi_label, [(label, round(score, 5)) for label, score in zip(r["labels"], r["scores"])])

# The answer of the long context is in the third of five windows. The text has "é", so the byte
# offsets are not the character offsets.
qa = pipeline("question-answering", model="distilbert/distilbert-base-cased-distilled-squad", device="cpu")
filler = "The café on the corner sells bread, coffee and small cakes to the people of the town every morning. " * 30
for question, context in [("Where is the Eiffel Tower?", "The Eiffel Tower is a wrought-iron lattice tower in Paris"), ("What is the code word?", filler + "The code word for the meeting is Zanzibar. " + filler)]:
    a = qa(question=question, context=context)
    print("qa", (a["answer"], len(context[: a["start"]].encode()), len(context[: a["end"]].encode()), round(a["score"], 6)))

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

# Vision, on test/fixtures/cats.png, a 320x240 copy of the COCO image
# http://images.cocodataset.org/val2017/000000039769.jpg. PNG decoders give the same pixels,
# so the inputs of Python and Transmuter are equal. The image processors are the slow ones,
# which use Pillow, as Transmuter does. The fast ones use torchvision and differ by up to one
# step of 1/255 before normalization.
from PIL import Image
from transformers import AutoImageProcessor, CLIPVisionModelWithProjection

image = Image.open("test/fixtures/cats.png")
processor = lambda repo: AutoImageProcessor.from_pretrained(repo, use_fast=False)  # noqa: E731

# pixel_values[0] at (channel, y, x), and the mean of the absolute values. ViT resizes to
# 224x224, CLIP resizes the short side to 224 and crops the center.
positions = [(0, 0, 0), (1, 100, 37), (2, 223, 223), (0, 57, 190), (1, 150, 120)]
for repo in ["google/vit-base-patch16-224", "openai/clip-vit-base-patch32"]:
    values = processor(repo)(image, return_tensors="np")["pixel_values"][0]
    print(repo, "pixel_values", [round(float(values[p]), 4) for p in positions], round(float(np.abs(values).mean()), 4))

for repo in ["google/vit-base-patch16-224", "facebook/convnext-tiny-224"]:
    classify = pipeline("image-classification", model=repo, image_processor=processor(repo), device="cpu")
    print(repo, [(p["label"], round(p["score"], 6)) for p in classify(image, top_k=5)])

clip = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
embedding = clip(**processor("openai/clip-vit-base-patch32")(image, return_tensors="pt")).image_embeds[0].detach().numpy()
print("clip image_embeds", [round(float(x), 4) for x in embedding[:4]], round(float(np.linalg.norm(embedding)), 4))

# The DINOv2 export interpolates the position embeddings in another way than transformers
# does now, so its reference is ONNX Runtime: the first token of last_hidden_state.
directory = snapshot_download("Xenova/dinov2-small", allow_patterns=["onnx/model.onnx"], local_dir=tempfile.mkdtemp())
session = ort.InferenceSession(f"{directory}/onnx/model.onnx", providers=["CPUExecutionProvider"])
pixel_values = processor("facebook/dinov2-small")(image, return_tensors="np")["pixel_values"]
cls = session.run(None, {"pixel_values": pixel_values})[0][0, 0]
print("dinov2 onnxruntime", [round(float(x), 4) for x in cls[:4]], round(float(np.linalg.norm(cls)), 4))

labels = ["cats", "dogs", "a remote control", "a couch"]
for repo in ["openai/clip-vit-base-patch32", "google/siglip-base-patch16-224"]:
    classify = pipeline("zero-shot-image-classification", model=repo, image_processor=processor(repo), device="cpu")
    print(repo, [(p["label"], round(p["score"], 6)) for p in classify(image, candidate_labels=labels)])

# Image-to-text: generate of the original PyTorch models of the Xenova exports, greedy, as
# Transmuter.Vision.image_to_text/2. The configurations have no max_length, so the text has
# at most 20 tokens with the start token, the default of GenerationConfig.
# test/fixtures/handwriting.png is a line of text that Pillow drew with the Caveat font
# (SIL Open Font License 1.1, github.com/google/fonts).
from transformers import VisionEncoderDecoderModel

for repo, path in [("nlpconnect/vit-gpt2-image-captioning", "test/fixtures/cats.png"), ("microsoft/trocr-small-handwritten", "test/fixtures/handwriting.png")]:
    pixel_values = processor(repo)(Image.open(path), return_tensors="pt").pixel_values
    output = VisionEncoderDecoderModel.from_pretrained(repo).generate(pixel_values, do_sample=False, num_beams=1)
    print(repo, repr(AutoTokenizer.from_pretrained(repo).batch_decode(output, skip_special_tokens=True)[0]))

# Text-to-text: greedy, as Transmuter.Text.text_to_text/2 with strategy: :greedy.
from transformers import AutoModelForSeq2SeqLM

sentence = "Transmuter runs ONNX models on any Nx backend, with no code for each architecture."
for repo, text in [("google-t5/t5-small", "translate English to German: The house is wonderful."), ("google-t5/t5-small", "translate English to German: " + sentence), ("Helsinki-NLP/opus-mt-en-de", sentence)]:
    tokenizer = AutoTokenizer.from_pretrained(repo)
    model = AutoModelForSeq2SeqLM.from_pretrained(repo)
    output = model.generate(**tokenizer(text, return_tensors="pt"), do_sample=False, num_beams=1, max_new_tokens=64)
    print(repo, repr(tokenizer.decode(output[0], skip_special_tokens=True)))

# Speech-to-text of test/fixtures/librispeech_0.f32, which this part writes. datasets 4
# decodes audio with torchcodec, so the script uses datasets 3 with soundfile.
from datasets import load_dataset
from transformers import WhisperForConditionalGeneration, WhisperProcessor

audio = np.asarray(load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")[0]["audio"]["array"], dtype=np.float32)
audio.astype("<f4").tofile("test/fixtures/librispeech_0.f32")

for repo, options in [("openai/whisper-tiny.en", {}), ("openai/whisper-base", {"language": "en", "task": "transcribe"}), ("openai/whisper-base", {"language": "fr", "task": "transcribe"})]:
    processor = WhisperProcessor.from_pretrained(repo)
    features = processor(audio, sampling_rate=16000, return_tensors="pt").input_features
    if repo == "openai/whisper-tiny.en":
        print("features", features[0, 0, :5].tolist(), features[0, 40, 100:103].tolist(), float(features.mean()))
    output = WhisperForConditionalGeneration.from_pretrained(repo).generate(features, do_sample=False, num_beams=1, **options)
    print(repo, options, repr(processor.batch_decode(output, skip_special_tokens=True)[0]))

# Long audio: the first 5 samples, 62.46 s, in test/fixtures/librispeech_long.f32, which this
# part writes. The pipeline runs windows of 30 s with 5 s of context on each side.
dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
long = np.concatenate([np.asarray(dataset[i]["audio"]["array"], dtype=np.float32) for i in range(5)])
long.astype("<f4").tofile("test/fixtures/librispeech_long.f32")

for repo, options in [("openai/whisper-tiny.en", {}), ("openai/whisper-base", {"language": "en", "task": "transcribe"})]:
    asr = pipeline("automatic-speech-recognition", model=repo, chunk_length_s=30, device="cpu")
    for timestamps in [False, True]:
        result = asr(long.copy(), return_timestamps=timestamps, generate_kwargs={"do_sample": False, "num_beams": 1, **options})
        print(repo, "long", "timestamps" if timestamps else "text", result.get("chunks") or repr(result["text"]))

asr = pipeline("automatic-speech-recognition", model="openai/whisper-tiny.en", chunk_length_s=30, device="cpu")
print("short timestamps", asr(audio.copy(), return_timestamps=True, generate_kwargs={"do_sample": False, "num_beams": 1})["chunks"])

# Image-text-to-text with SmolVLM. The image processor is the slow one, so the pixel values
# are those of Transmuter. The reference of the default files of Transmuter (the q4 vision
# encoder and decoder and the 8-bit embeddings) is ONNX Runtime on the same files: the
# features of the image tokens replace their embeddings, and the decoder runs greedy with a
# cache that grows from 0 rows. The PyTorch model gives another text, also with the fp32
# ONNX files, because the vision encoder of the export gives other features than
# transformers 4.57 (a mean absolute difference of about 1.9 on each tile). ONNX Runtime and
# Transmuter give the same text with the fp32 files.
import hashlib

from transformers import AutoImageProcessor, AutoProcessor

repo = "HuggingFaceTB/SmolVLM-256M-Instruct"
image = Image.open("test/fixtures/cats.png")
processor = AutoProcessor.from_pretrained(repo, image_processor=AutoImageProcessor.from_pretrained(repo, use_fast=False))

# The prompt and the pixel values of a wide image, a tall image, and an image that gets
# larger than a multiple of 512 before it splits: pixel_values[0] at (tile, channel, y, x),
# the mean of the absolute values, and the number and a SHA-256 of the int64 input ids.
chat = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Can you describe this image?"}]}]
prompt = processor.apply_chat_template(chat, add_generation_prompt=True)
positions = [(0, 0, 0, 0), (1, 1, 100, 37), (-1, 2, 511, 511), (2, 0, 257, 190)]
for name, picture in [("wide", image), ("tall", image.transpose(Image.Transpose.TRANSPOSE)), ("crop", image.crop((0, 0, 320, 100)))]:
    inputs = processor(text=prompt, images=[picture], return_tensors="np")
    ids, values = inputs["input_ids"][0].astype("<i8"), inputs["pixel_values"][0]
    print("idefics3", name, values.shape, [round(float(values[p]), 4) for p in positions], round(float(np.abs(values).mean()), 4), len(ids), hashlib.sha256(ids.tobytes()).hexdigest()[:16])

chats = {
    "one image": ([image], [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Can you describe this image?"}]}]),
    "tall image": ([image.transpose(Image.Transpose.TRANSPOSE)], [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "What animals are in the image?"}]}]),
    "two images": ([image, image.crop((0, 0, 320, 100))], [{"role": "user", "content": [{"type": "image"}, {"type": "image"}, {"type": "text", "text": "How are these two images different?"}]}]),
    "text": ([], [{"role": "user", "content": [{"type": "text", "text": "What is the capital of France?"}]}]),
}

directory = snapshot_download(repo, allow_patterns=["onnx/vision_encoder_q4.onnx*", "onnx/embed_tokens_quantized.onnx*", "onnx/decoder_model_merged_q4.onnx*"], local_dir=tempfile.mkdtemp())
session = lambda name: ort.InferenceSession(f"{directory}/onnx/{name}.onnx", providers=["CPUExecutionProvider"])  # noqa: E731
vision, embed, decoder = session("vision_encoder_q4"), session("embed_tokens_quantized"), session("decoder_model_merged_q4")
names = [o.name for o in decoder.get_outputs()]
image_token, eos = processor.tokenizer.convert_tokens_to_ids(["<image>", "<end_of_utterance>"])

for name, (images, messages) in chats.items():
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=images or None, return_tensors="np")
    ids = inputs["input_ids"]
    embeds = embed.run(None, {"input_ids": ids})[0]
    if images:
        features = vision.run(None, {"pixel_values": inputs["pixel_values"], "pixel_attention_mask": inputs["pixel_attention_mask"].astype(bool)})[0]
        embeds[ids == image_token] = features.reshape(-1, embeds.shape[-1])
    past = {i.name: np.zeros((1, i.shape[1], 0, i.shape[3]), np.float32) for i in decoder.get_inputs() if i.name.startswith("past_key_values")}
    tokens, length = [], ids.shape[1]
    for _ in range(24):
        n = embeds.shape[1]
        feeds = {"inputs_embeds": embeds, "attention_mask": np.ones((1, length), np.int64), "position_ids": np.arange(length - n, length)[None], **past}
        outputs = dict(zip(names, decoder.run(None, feeds)))
        tokens.append(int(outputs["logits"][0, -1].argmax()))
        if tokens[-1] == eos:
            break
        past = {k.replace("present", "past_key_values"): v for k, v in outputs.items() if k.startswith("present")}
        embeds = embed.run(None, {"input_ids": np.array([tokens[-1:]])})[0]
        length += 1
    print("smolvlm onnxruntime", name, repr(processor.tokenizer.decode(tokens, skip_special_tokens=True)))

# Image-text-to-text with Qwen2-VL. The image processor is the slow one, so the pixel values
# are those of Transmuter. The reference is ONNX Runtime on the files of the test: the fp32
# vision encoder, the fp16 embeddings and the q4 decoder. The export of the vision encoder
# computes the rotary positions of the patches for one image, so each image runs alone. The
# decoder takes the 3-D positions of M-RoPE from get_rope_index of transformers, and each new
# token gets its index plus the rope delta of the prompt on all three axes. With the 8-bit
# vision encoder of the default variants, ONNX Runtime and Transmuter give different texts.
# The PyTorch model in fp32 needs about 9 GB, so the script does not run it.
from types import SimpleNamespace

import torch
from transformers import AutoConfig, GenerationConfig
from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLModel

repo = "onnx-community/Qwen2-VL-2B-Instruct"
image = Image.open("test/fixtures/cats.png")
# Transformers 4.57 rejects the size of this preprocessor_config.json, {"min_pixels", "max_pixels"},
# so the size comes from its min_pixels and max_pixels, which take precedence in Python.
size = {"shortest_edge": 3136, "longest_edge": 12845056}
processor = AutoProcessor.from_pretrained(repo, image_processor=AutoImageProcessor.from_pretrained(repo, use_fast=False, size=size))
config = AutoConfig.from_pretrained(repo)

# The patches, the grid and the prompt of a wide and a tall image: pixel_values at (patch,
# value), the mean of the absolute values, and the number and a SHA-256 of the int64 input ids.
chat = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Can you describe this image?"}]}]
prompt = processor.apply_chat_template(chat, add_generation_prompt=True)
positions = [(0, 0), (1, 100), (-1, 1175), (200, 777)]
for name, picture in [("wide", image), ("tall", image.transpose(Image.Transpose.TRANSPOSE))]:
    inputs = processor(text=prompt, images=[picture], return_tensors="np")
    ids, values = inputs["input_ids"][0].astype("<i8"), inputs["pixel_values"]
    print("qwen2-vl", name, values.shape, inputs["image_grid_thw"].tolist(), [round(float(values[p]), 4) for p in positions], round(float(np.abs(values).mean()), 4), len(ids), hashlib.sha256(ids.tobytes()).hexdigest()[:16])

chats = {
    "one image": ([image], [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Can you describe this image?"}]}]),
    "tall image": ([image.transpose(Image.Transpose.TRANSPOSE)], [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "What animals are in the image?"}]}]),
    "text": ([], [{"role": "user", "content": [{"type": "text", "text": "What is the capital of France?"}]}]),
}

directory = snapshot_download(repo, allow_patterns=["onnx/vision_encoder.onnx*", "onnx/embed_tokens_fp16.onnx", "onnx/decoder_model_merged_q4.onnx"], local_dir=tempfile.mkdtemp())
session = lambda name: ort.InferenceSession(f"{directory}/onnx/{name}.onnx", providers=["CPUExecutionProvider"])  # noqa: E731
vision, embed, decoder = session("vision_encoder"), session("embed_tokens_fp16"), session("decoder_model_merged_q4")
names = [o.name for o in decoder.get_outputs()]
eos = GenerationConfig.from_pretrained(repo).eos_token_id
rope = SimpleNamespace(config=config)

for name, (images, messages) in chats.items():
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=images or None, return_tensors="np")
    ids = inputs["input_ids"]
    embeds = embed.run(None, {"input_ids": ids})[0].astype(np.float32)
    grid = inputs.get("image_grid_thw")
    if images:
        counts = np.cumsum([0] + [int(np.prod(g)) for g in grid])
        features = [vision.run(None, {"pixel_values": inputs["pixel_values"][counts[i]:counts[i + 1]], "grid_thw": grid[i : i + 1]})[0] for i in range(len(grid))]
        embeds[ids == config.image_token_id] = np.concatenate(features)
    position_ids, delta = Qwen2VLModel.get_rope_index(rope, torch.tensor(ids), None if grid is None else torch.tensor(grid), None, torch.ones(ids.shape, dtype=torch.long))
    position_ids, delta = position_ids.numpy(), int(delta[0, 0])
    past = {i.name: np.zeros((1, i.shape[1], 0, i.shape[3]), np.float32) for i in decoder.get_inputs() if i.name.startswith("past_key_values")}
    tokens, length = [], ids.shape[1]
    for _ in range(24):
        feeds = {"inputs_embeds": embeds, "attention_mask": np.ones((1, length), np.int64), "position_ids": position_ids, **past}
        outputs = dict(zip(names, decoder.run(None, feeds)))
        tokens.append(int(outputs["logits"][0, -1].argmax()))
        if tokens[-1] in eos:
            break
        past = {k.replace("present", "past_key_values"): v for k, v in outputs.items() if k.startswith("present")}
        embeds = embed.run(None, {"input_ids": np.array([tokens[-1:]])})[0].astype(np.float32)
        position_ids = np.full((3, 1, 1), length + delta, np.int64)
        length += 1
    print("qwen2-vl onnxruntime", name, repr(processor.tokenizer.decode(tokens, skip_special_tokens=True)))
