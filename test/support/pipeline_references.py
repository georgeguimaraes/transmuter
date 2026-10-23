"""Prints the reference values of test/transmuter/text_test.exs, test/transmuter/vision_test.exs and test/transmuter/audio_test.exs, from the pipelines of Python transformers and sentence-transformers.

Usage: uv run --with "transformers<5" --with sentence-transformers --with torch --with onnxruntime --with onnx --with pillow --with sentencepiece --with protobuf --with sacremoses --with "datasets<4" --with soundfile --with librosa --with torchvision --with timm python -I test/support/pipeline_references.py

Transformers 5 removed the question answering pipeline, so the script uses transformers 4. The processor of Qwen2-VL loads a video processor, which needs torchvision. DETR needs timm.
"""

from sentence_transformers import SentenceTransformer
from transformers import pipeline

for repo in ["sentence-transformers/all-MiniLM-L6-v2", "BAAI/bge-small-en-v1.5"]:
    embedding = SentenceTransformer(repo, device="cpu").encode(["The cat sat on the mat."])[0]
    print(repo, [round(float(x), 4) for x in embedding[:4]])

long = "the quick brown fox jumps over the lazy dog " * 60
embedding = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu").encode([long])[0]
print("all-MiniLM-L6-v2, 541 tokens", [round(float(x), 4) for x in embedding[:4]])

# EmbeddingGemma without a prompt. Transmuter runs onnx-community/embeddinggemma-300m-ONNX,
# whose output sentence_embedding includes the Dense layers and the normalization. The
# reference is ONNX Runtime on that export, because the export differs from sentence-transformers
# on google/embeddinggemma-300m: by up to 5.6e-4 on the short text, and by up to 0.027 (cosine
# 0.983) on the 543 tokens of the long text. The export has no sliding window of 512 tokens:
# sentence-transformers with sliding_window 4096 is 1.3e-3 from it. The script prints both.
# google/embeddinggemma-300m is gated: accept its license on the Hub and log in with hf auth login.
import tempfile

import numpy as np
import onnxruntime as ort
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

repo = "onnx-community/embeddinggemma-300m-ONNX"
gemma = SentenceTransformer("google/embeddinggemma-300m", device="cpu")
tokenizer = AutoTokenizer.from_pretrained(repo)
directory = snapshot_download(repo, allow_patterns=["onnx/model.onnx*"], local_dir=tempfile.mkdtemp())
session = ort.InferenceSession(f"{directory}/onnx/model.onnx", providers=["CPUExecutionProvider"])
for name, text in [("short", "The cat sat on the mat."), ("543 tokens", long)]:
    inputs = {k: v.astype(np.int64) for k, v in tokenizer([text], return_tensors="np").items()}
    embedding = session.run(["sentence_embedding"], inputs)[0][0]
    print("embeddinggemma onnxruntime", name, [round(float(x), 4) for x in embedding[:4]])
    print("embeddinggemma sentence-transformers", name, [round(float(x), 4) for x in gemma.encode([text])[0][:4]])

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

# Generation with the logits processors of generation_config.json, greedy. The n-grams include
# the prompt. The empty prompt of Transmuter is the bos token, a prompt of one token, so the
# forced bos token is the first new token.
import torch

output = model.generate(**inputs, max_new_tokens=12, do_sample=False, no_repeat_ngram_size=2)
print("gpt2 no_repeat_ngram_size=2", repr(tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)))
bos = torch.tensor([[tokenizer.bos_token_id]])
output = model.generate(input_ids=bos, attention_mask=torch.ones_like(bos), max_new_tokens=8, do_sample=False, forced_bos_token_id=464)
print("gpt2 forced_bos_token_id=464", repr(tokenizer.decode(output[0, 1:], skip_special_tokens=True)))

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)
output = model.generate(**inputs, max_new_tokens=16, do_sample=False, min_new_tokens=12)
print("smollm2 min_new_tokens=12", repr(tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)))

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

# Image-to-text: the image-to-text pipeline with the original PyTorch models of the Xenova
# exports, greedy, as Transmuter.Vision.image_to_text/2. The configurations have no
# max_length, so the pipeline gives at most 256 new tokens.
# test/fixtures/handwriting.png and handwriting_long.png are lines of text that Pillow drew
# with the Caveat font (SIL Open Font License 1.1, github.com/google/fonts) at size 56.
# handwriting_long.png has the text "We walked along the river and talked about the garden
# until the sun went down behind the hills and trees.", which is more than the 20 new tokens
# that generate gives without max_new_tokens.
for repo, path in [("nlpconnect/vit-gpt2-image-captioning", "test/fixtures/cats.png"), ("microsoft/trocr-small-handwritten", "test/fixtures/handwriting.png"), ("microsoft/trocr-small-handwritten", "test/fixtures/handwriting_long.png")]:
    caption = pipeline("image-to-text", model=repo, device="cpu")
    print(repo, path, repr(caption(Image.open(path), generate_kwargs={"do_sample": False, "num_beams": 1})[0]["generated_text"]))

# Text-to-text: greedy, as Transmuter.Text.text_to_text/2 with strategy: :greedy.
from transformers import AutoModelForSeq2SeqLM

sentence = "Transmuter runs ONNX models on any Nx backend, with no code for each architecture."
for repo, text in [("google-t5/t5-small", "translate English to German: The house is wonderful."), ("google-t5/t5-small", "translate English to German: " + sentence), ("Helsinki-NLP/opus-mt-en-de", sentence)]:
    tokenizer = AutoTokenizer.from_pretrained(repo)
    model = AutoModelForSeq2SeqLM.from_pretrained(repo)
    output = model.generate(**tokenizer(text, return_tensors="pt"), do_sample=False, num_beams=1, max_new_tokens=64)
    print(repo, repr(tokenizer.decode(output[0], skip_special_tokens=True)))

# Translation with NLLB and M2M100: the translation pipeline, greedy. The pipeline gives src_lang
# to the tokenizer and the token of tgt_lang to generate in forced_bos_token_id. The
# configuration of m2m100_418M has num_beams=5, so the call sets num_beams=1. M2M100 has only
# the slow tokenizer, which reads sentencepiece.bpe.model and vocab.json.
french = "La vie est belle et le soleil brille sur la ville."
for repo, languages in [("facebook/nllb-200-distilled-600M", [("eng_Latn", "fra_Latn"), ("fra_Latn", "deu_Latn")]), ("facebook/m2m100_418M", [("en", "fr"), ("fr", "de")])]:
    translator = pipeline("translation", model=repo, device="cpu")
    for (src, tgt), text in zip(languages, [sentence, french]):
        tokenizer = AutoTokenizer.from_pretrained(repo, src_lang=src)
        target = tokenizer.convert_tokens_to_ids(tgt) if "nllb" in repo else tokenizer.get_lang_id(tgt)
        print(repo, src, tgt, "ids", tokenizer(text)["input_ids"], "forced_bos_token_id", target)
        print(repo, src, tgt, repr(translator(text, src_lang=src, tgt_lang=tgt, do_sample=False, num_beams=1)[0]["translation_text"]))

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

# Word timestamps of the short and the long audio: (text, start, end) of each word.
for name, samples in [("short", audio), ("long", long)]:
    result = asr(samples.copy(), return_timestamps="word", generate_kwargs={"do_sample": False, "num_beams": 1})
    print(name, "words", [(c["text"], c["timestamp"][0], c["timestamp"][1]) for c in result["chunks"]])

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

# Object detection: the slow image processor, the PyTorch model and
# post_process_object_detection with the threshold of the pipeline, 0.5. The pipeline
# truncates the boxes to integers, so the script prints the boxes before that. DETR uses
# the ResNet-50 of timm. The reference of RF-DETR is in test/support/rf_detr_reference.py,
# because only transformers 5 has RF-DETR.
from transformers import AutoModelForObjectDetection

image = Image.open("test/fixtures/cats.png")
for repo in ["facebook/detr-resnet-50", "hustvl/yolos-tiny"]:
    processor = AutoImageProcessor.from_pretrained(repo, use_fast=False)
    model = AutoModelForObjectDetection.from_pretrained(repo)
    for name, picture in [("wide", image), ("tall", image.transpose(Image.Transpose.TRANSPOSE))]:
        inputs = processor(picture, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        result = processor.post_process_object_detection(outputs, threshold=0.5, target_sizes=[(picture.height, picture.width)])[0]
        print(repo, name, tuple(inputs["pixel_values"].shape), [(model.config.id2label[int(label)], round(float(score), 6), [round(float(x), 4) for x in box]) for score, label, box in zip(result["scores"], result["labels"], result["boxes"])])

# The resized sizes of DETR, YOLOS and DPT. A long side over longest_edge gives the short side
# longest_edge times the aspect ratio, rounded half to even: 1333 * 400 / 800 is 666.5. DPT
# also rounds half to even: the 35x518 image keeps the scale 1, and 35 / 14 is 2.5.
for repo in ["facebook/detr-resnet-50", "hustvl/yolos-tiny", "onnx-community/depth-anything-v2-small"]:
    processor = AutoImageProcessor.from_pretrained(repo, use_fast=False)
    print(repo, "sizes", [tuple(processor(Image.new("RGB", (width, height)), return_tensors="np")["pixel_values"].shape[2:]) for height, width in [(240, 320), (100, 1000), (1000, 100), (400, 800), (800, 600), (35, 518)]])

# Depth estimation: the slow DPT image processor and the depth-estimation pipeline, on the wide
# and the tall image. The script prints pixel_values[0] of the wide image at (channel, y, x)
# and the mean of the absolute values, then predicted_depth and the 8-bit depth at (y, x) and
# their means. The ONNX file of onnx-community/depth-anything-v2-small gives the depth of
# the PyTorch model within 2e-5. The ONNX file of Xenova/depth-anything-small-hf gives a
# depth up to 0.9% away from LiheYoung/depth-anything-small-hf, so its reference is ONNX Runtime
# with the post-processing of the pipeline: a bicubic resize to the image, then the scale to
# 8 bits with the minimum and the maximum.
from transformers import AutoModelForDepthEstimation

image = Image.open("test/fixtures/cats.png")
processor = AutoImageProcessor.from_pretrained("onnx-community/depth-anything-v2-small", use_fast=False)
values = processor(image, return_tensors="np")["pixel_values"][0]
print("dpt pixel_values", values.shape, [round(float(values[p]), 4) for p in positions], round(float(np.abs(values).mean()), 4))


def print_depth(name, depth, u8):
    points = [(0, 0), (100, 37), (57, 190), (150, 120), (239, 239)]
    print(name, depth.shape, [round(float(depth[p]), 4) for p in points], round(float(depth.mean()), 4), [int(u8[p]) for p in points], round(float(u8.mean()), 4))


estimate = pipeline("depth-estimation", model=AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf"), image_processor=processor, device="cpu")
directory = snapshot_download("Xenova/depth-anything-small-hf", allow_patterns=["onnx/model.onnx"], local_dir=tempfile.mkdtemp())
session = ort.InferenceSession(f"{directory}/onnx/model.onnx", providers=["CPUExecutionProvider"])
for name, picture in [("wide", image), ("tall", image.transpose(Image.Transpose.TRANSPOSE))]:
    result = estimate(picture)
    print_depth(f"depth-anything-v2-small {name}", result["predicted_depth"].numpy(), np.array(result["depth"]))
    predicted = session.run(None, {"pixel_values": processor(picture, return_tensors="np")["pixel_values"]})[0]
    depth = torch.nn.functional.interpolate(torch.from_numpy(predicted)[None], size=(picture.height, picture.width), mode="bicubic", align_corners=False).squeeze().numpy()
    print_depth(f"depth-anything-small-hf onnxruntime {name}", depth, ((depth - depth.min()) / (depth.max() - depth.min()) * 255).astype("uint8"))

# Text-to-speech without noise, as Transmuter.Audio.text_to_speech/2 with noise: false. VITS
# runs with noise_scale and noise_scale_duration at 0. The decoder prenet of SpeechT5 applies
# dropout also in eval, so here the dropout keeps all values, as the export does when its
# RandomUniformLike operators give their upper bound. With this dropout, SpeechT5 reads "Hi."
# in 19 steps, before maxlen (25 steps). The script writes the audio to
# test/fixtures/mms_tts_eng.f32 and test/fixtures/speecht5_hi.f32. The speaker embedding in
# test/fixtures/speaker_embeddings.f32 is speaker_embeddings.bin of the transformers.js docs
# (huggingface.co/datasets/Xenova/transformers.js-docs).
import onnx
from onnx import helper, numpy_helper
from transformers import SpeechT5ForTextToSpeech, SpeechT5HifiGan, SpeechT5Processor, VitsModel
from transformers.models.speecht5 import modeling_speecht5

text = "Hello, my dog is cute."
tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")
print("vits ids", tokenizer("Hello World! It's 3 o'clock, café? ÀB_x–y").input_ids)
model = VitsModel.from_pretrained("facebook/mms-tts-eng")
model.noise_scale = model.noise_scale_duration = 0.0
with torch.no_grad():
    waveform = model(**tokenizer(text, return_tensors="pt")).waveform[0].numpy()
waveform.astype("<f4").tofile("test/fixtures/mms_tts_eng.f32")
print("vits", model.config.sampling_rate, len(waveform), float(np.abs(waveform).mean()))

modeling_speecht5.SpeechT5SpeechDecoderPrenet._consistent_dropout = lambda self, x, p: x / (1 - p)
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
speaker = np.fromfile("test/fixtures/speaker_embeddings.f32", "<f4")[None]
print("speecht5 ids", processor(text="Hello World! It's 3 o'clock, café? ÀB_x–y")["input_ids"])
ids = processor(text="Hi.", return_tensors="pt")["input_ids"]
with torch.no_grad():
    speech = model.generate_speech(ids, torch.tensor(speaker), vocoder=vocoder).numpy()
speech.astype("<f4").tofile("test/fixtures/speecht5_hi.f32")
print("speecht5", vocoder.config.sampling_rate, len(speech), float(np.abs(speech).mean()))


# ONNX Runtime on the fp32 export of SpeechT5, with the dropout that keeps all values and the
# stop rule of Python, gives the same length and a waveform within about 1e-4 of PyTorch.
def keep_all(graph):
    nodes = []
    for node in graph.node:
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.GRAPH:
                keep_all(attribute.g)
        if node.op_type == "RandomUniformLike":
            name = node.output[0]
            one = numpy_helper.from_array(np.array(1.0, np.float32))
            nodes += [helper.make_node("Shape", [node.input[0]], [name + "_shape"]), helper.make_node("Constant", [], [name + "_one"], value=one), helper.make_node("Expand", [name + "_one", name + "_shape"], [name])]
        else:
            nodes.append(node)
    del graph.node[:]
    graph.node.extend(nodes)


def keep_all_session(path):
    graph = onnx.load(path)
    keep_all(graph.graph)
    return ort.InferenceSession(graph.SerializeToString(), providers=["CPUExecutionProvider"])


directory = snapshot_download("Xenova/speecht5_tts", allow_patterns=["onnx/encoder_model.onnx", "onnx/decoder_model_merged.onnx", "onnx/decoder_postnet_and_vocoder.onnx"], local_dir=tempfile.mkdtemp())
encoder, decoder, vocoder = (keep_all_session(f"{directory}/onnx/{name}.onnx") for name in ["encoder_model", "decoder_model_merged", "decoder_postnet_and_vocoder"])
hidden, mask = encoder.run(None, {"input_ids": ids.numpy()})
names = [o.name for o in decoder.get_outputs()]
past = {i.name: np.zeros((1, i.shape[1], 0, i.shape[3]), np.float32) for i in decoder.get_inputs() if i.name.startswith("past_key_values")}
sequence, spectrum = np.zeros((1, 1, 80), np.float32), []
for step in range(1, hidden.shape[1] * 20 // 2 + 1):
    feeds = {"output_sequence": sequence, "encoder_hidden_states": hidden, "encoder_attention_mask": mask, "speaker_embeddings": speaker, "use_cache_branch": np.array([step > 1]), **past}
    outputs = dict(zip(names, decoder.run(None, feeds)))
    spectrum.append(outputs["spectrum"])
    if outputs["prob"].sum() >= 0.5:
        break
    sequence = outputs["output_sequence_out"]
    past.update({k.replace("present", "past_key_values"): v for k, v in outputs.items() if k.startswith("present") and v.shape[2] > 0})
waveform = vocoder.run(None, {"spectrogram": np.concatenate(spectrum)})[0].reshape(-1)
print("speecht5 onnxruntime", len(spectrum), len(waveform), float(np.abs(waveform - speech).max()) if len(waveform) == len(speech) else None)

# Audio classification of test/fixtures/librispeech_0.f32 (5.86 s) and librispeech_long.f32
# (62.46 s). The command above does not install torchaudio, so ASTFeatureExtractor takes its
# numpy path, as Transmuter does. The 5.86 s give 584 frames, so the features end with frames
# of zeros, and the 62.46 s keep the first 1024 frames. The script prints input_values[0] at
# (frame, mel) or at (sample) and the mean, then the top 5 of the pipeline. The pipeline gives
# all labels without top_k, so the script passes top_k=5. The PyTorch model of the wav2vec2
# export (DunnBC22/wav2vec2-base-Speech_Emotion_Recognition) is no longer on the Hub, so its
# reference is ONNX Runtime with the post-processing of the pipeline: a softmax and the top 5.
from transformers import AutoConfig, AutoFeatureExtractor
from transformers.utils import is_speech_available

assert not is_speech_available()
audio = np.fromfile("test/fixtures/librispeech_0.f32", "<f4")
long = np.fromfile("test/fixtures/librispeech_long.f32", "<f4")
extractor = AutoFeatureExtractor.from_pretrained("Xenova/ast-finetuned-audioset-10-10-0.4593")
classify = pipeline("audio-classification", model="MIT/ast-finetuned-audioset-10-10-0.4593", feature_extractor=extractor, device="cpu")
for name, samples, points in [("short", audio, [(0, 0), (100, 37), (583, 127), (584, 0), (1023, 127)]), ("long", long, [(0, 0), (500, 64), (1023, 127)])]:
    values = extractor(samples, sampling_rate=16000, return_tensors="np")["input_values"][0]
    print("ast features", name, values.shape, [round(float(values[p]), 5) for p in points], round(float(values.mean()), 5))
    print("ast", name, [(p["label"], round(p["score"], 6)) for p in classify(samples.copy(), top_k=5)])

repo = "onnx-community/wav2vec2-base-Speech_Emotion_Recognition-ONNX"
extractor = AutoFeatureExtractor.from_pretrained(repo)
id2label = AutoConfig.from_pretrained(repo).id2label
directory = snapshot_download(repo, allow_patterns=["onnx/model.onnx"], local_dir=tempfile.mkdtemp())
session = ort.InferenceSession(f"{directory}/onnx/model.onnx", providers=["CPUExecutionProvider"])
for name, samples in [("short", audio), ("cut", audio[:32000])]:
    values = extractor(samples, sampling_rate=16000, return_tensors="np")["input_values"]
    print("wav2vec2 features", name, values.shape, [round(float(values[0, i]), 5) for i in [0, 1000, len(samples) - 1]])
    scores = torch.from_numpy(session.run(None, {"input_values": values})[0][0]).softmax(-1)
    top = scores.topk(5)
    print("wav2vec2", name, [(id2label[i], round(s, 6)) for s, i in zip(top.values.tolist(), top.indices.tolist())])

# Zero-shot audio classification with CLAP. The model takes 48 kHz audio, so librosa resamples
# test/fixtures/librispeech_0.f32 from 16 kHz, and the script writes the result to
# test/fixtures/librispeech_0_48k.f32. The 5.86 s are shorter than the 10 s of the model, so
# repeatpad gives the audio once and zeros. The first 2 s repeat 5 times and fill the 10 s.
# The script prints input_features[0, 0] at (frame, mel) and the mean, then the scores of
# the zero-shot-audio-classification pipeline with the PyTorch model, then ONNX Runtime on the
# fp32 and the q8 files with the same inputs: a softmax of logits_per_audio.
import librosa

audio = librosa.resample(np.fromfile("test/fixtures/librispeech_0.f32", "<f4"), orig_sr=16000, target_sr=48000).astype("<f4")
audio.tofile("test/fixtures/librispeech_0_48k.f32")
clap_inputs = [("short", audio), ("cut", audio[:96000])]
labels = ["a man speaking", "music", "a dog barking", "rain"]
extractor = AutoFeatureExtractor.from_pretrained("Xenova/clap-htsat-unfused")
for name, samples in clap_inputs:
    values = extractor(samples, sampling_rate=48000, return_tensors="np")["input_features"][0, 0]
    points = [(0, 0), (100, 37), (281, 63), (290, 10), (600, 20), (1000, 63)]
    print("clap features", name, values.shape, [round(float(values[p]), 4) for p in points], round(float(values.mean()), 4))

classify = pipeline("zero-shot-audio-classification", model="laion/clap-htsat-unfused", device="cpu")
for name, samples in clap_inputs:
    print("clap", name, [(p["label"], round(p["score"], 6)) for p in classify(samples.copy(), candidate_labels=labels)])

tokenizer = AutoTokenizer.from_pretrained("Xenova/clap-htsat-unfused")
text = dict(tokenizer([f"This is a sound of {label}." for label in labels], padding=True, return_tensors="np"))
for file in ["model", "model_quantized"]:
    directory = snapshot_download("Xenova/clap-htsat-unfused", allow_patterns=[f"onnx/{file}.onnx"], local_dir=tempfile.mkdtemp())
    session = ort.InferenceSession(f"{directory}/onnx/{file}.onnx", providers=["CPUExecutionProvider"])
    for name, samples in clap_inputs:
        features = extractor(samples, sampling_rate=48000, return_tensors="np")["input_features"]
        logits = session.run(["logits_per_audio"], {**text, "input_features": features})[0][0]
        scores = torch.from_numpy(logits).softmax(-1).tolist()
        print("clap onnxruntime", file, name, sorted([(label, round(s, 6)) for label, s in zip(labels, scores)], key=lambda x: -x[1]))

# Image segmentation: the slow Segformer image processor of the Xenova export and the
# image-segmentation pipeline with the PyTorch model, on the wide and the tall image. The
# script prints pixel_values[0] of the wide image at (channel, y, x) and the mean of the
# absolute values, then the label and the pixel count of each segment. It writes the label
# id of each pixel to test/fixtures/cats_segformer_wide.png and cats_segformer_tall.png. The
# ONNX file gives the same label map in ONNX Runtime.
from transformers import AutoModelForSemanticSegmentation

image = Image.open("test/fixtures/cats.png")
pictures = [("wide", image), ("tall", image.transpose(Image.Transpose.TRANSPOSE))]
processor = AutoImageProcessor.from_pretrained("Xenova/segformer-b0-finetuned-ade-512-512", use_fast=False)
values = processor(image, return_tensors="np")["pixel_values"][0]
print("segformer pixel_values", values.shape, [round(float(values[p]), 4) for p in positions], round(float(np.abs(values).mean()), 4))
model = AutoModelForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
segment = pipeline("image-segmentation", model=model, image_processor=processor, device="cpu")
label2id = {label: i for i, label in model.config.id2label.items()}
for name, picture in pictures:
    segments = segment(picture)
    labels = np.zeros((picture.height, picture.width), np.uint8)
    for s in segments:
        labels[np.array(s["mask"]) == 255] = label2id[s["label"]]
    Image.fromarray(labels).save(f"test/fixtures/cats_segformer_{name}.png")
    print("segformer", name, [(s["label"], s["score"], int((np.array(s["mask"]) == 255).sum())) for s in segments])

# Background removal. Python transformers has no background-removal pipeline, so the
# reference is ONNX Runtime with the slow image processor and the post-processing of the
# BackgroundRemovalPipeline of transformers.js: a sigmoid when a value is farther than 1e-5
# from 0..1, then 255 times the value, truncated to 8 bits. Then Pillow resizes the mask to
# the image with bilinear, as Transmuter does. transformers.js resizes with sharp in Node,
# whose affine bilinear does not filter a smaller image and does not center the pixels, or
# with the canvas in the browser. The script writes the masks to test/fixtures/cats_ormbg_wide.png
# and cats_ormbg_tall.png and prints the mask at (y, x) and its mean.
processor = AutoImageProcessor.from_pretrained("onnx-community/ormbg-ONNX", use_fast=False)
directory = snapshot_download("onnx-community/ormbg-ONNX", allow_patterns=["onnx/model.onnx"], local_dir=tempfile.mkdtemp())
session = ort.InferenceSession(f"{directory}/onnx/model.onnx", providers=["CPUExecutionProvider"])
for name, picture in pictures:
    alphas = session.run(None, {"pixel_values": processor(picture, return_tensors="np")["pixel_values"]})[0][0]
    if ((alphas < -1e-5) | (alphas > 1 + 1e-5)).any():
        alphas = torch.from_numpy(alphas).sigmoid().numpy()
    mask = Image.fromarray((alphas[0] * np.float32(255)).astype(np.uint8))
    mask = mask.resize((picture.width, picture.height), Image.Resampling.BILINEAR)
    mask.save(f"test/fixtures/cats_ormbg_{name}.png")
    values = np.array(mask)
    print("ormbg", name, values.shape, [int(values[p]) for p in [(0, 0), (100, 37), (57, 190), (150, 120), (239, 239)]], round(float(values.mean()), 4))

# Image-to-image with Swin2SR: the image-to-image pipeline with the PyTorch models of the
# Xenova exports, on a 64x48 crop of the image for x2 and a 59x50 crop for x4. Swin2SRImageProcessor
# pads each side with the symmetric mode of numpy to the next multiple of 8, and a side that is
# already a multiple gets 8 more pixels. The pipeline gives the upscaled padded image. The
# script prints the size of that image, and writes its region of the crop, scale * height
# by scale * width, to test/fixtures/cats_swin2sr_x2.png and cats_swin2sr_x4.png.
image = Image.open("test/fixtures/cats.png")
for repo, name, box in [("caidas/swin2SR-classical-sr-x2-64", "x2", (110, 75, 174, 123)), ("caidas/swin2SR-realworld-sr-x4-64-bsrgan-psnr", "x4", (130, 70, 189, 120))]:
    upscale = pipeline("image-to-image", model=repo, device="cpu")
    picture = image.crop(box)
    result = upscale(picture)
    scale = upscale.model.config.upscale
    region = result.crop((0, 0, scale * picture.width, scale * picture.height))
    region.save(f"test/fixtures/cats_swin2sr_{name}.png")
    values = np.array(region)
    print("swin2sr", name, picture.size, result.size, values.shape, [values[p].tolist() for p in [(0, 0), (37, 100), (values.shape[0] - 1, values.shape[1] - 1)]], round(float(values.mean()), 4))

# Zero-shot object detection with the labels "cat", "remote control" and "couch", on the wide
# and the tall image. The script prints pixel_values[0] of the slow OWL-ViT and OWLv2 image
# processors of the Xenova exports at (channel, y, x) and the mean of the absolute values,
# for the wide image and, for OWLv2, for a 6x6 tile of it (1920x1440). The resize of OWLv2
# blurs that tile with sigma 0.5 before it interpolates.
# OWL-ViT and OWLv2: the zero-shot-object-detection pipeline with the PyTorch models and the
# threshold of the pipeline, 0.1. The pipeline runs the model once for each label and
# truncates the boxes to integers, so here _get_bounding_box keeps the floats. The script
# also prints the largest difference of the scores between one run with all labels as text
# queries, as Transmuter does, and one run for each label.
# Grounding DINO: the ONNX export computes the phrases of the text (the attention mask and
# the positions between the special tokens) for one phrase and a period. The pipeline with
# the PyTorch model tokenizes the label alone, such as "cat", and ONNX Runtime fails on that
# text. So the text of a label is the label in lowercase and a period, as
# _merge_candidate_labels_text of GroundingDinoProcessor makes it for a list of one label.
# The export selects 900 of 13294 boxes with TopK, and its scores differ from PyTorch by up
# to 6e-4, so the reference is ONNX Runtime with the post-processing of the pipeline: the
# sigmoid of the highest logit over the tokens, the boxes times the size of the image, the
# threshold 0.1, and a stable sort by score.
from transformers import AutoModelForZeroShotObjectDetection, AutoTokenizer, GroundingDinoImageProcessor
from transformers.models.grounding_dino.processing_grounding_dino import _merge_candidate_labels_text
from transformers.pipelines.zero_shot_object_detection import ZeroShotObjectDetectionPipeline

image = Image.open("test/fixtures/cats.png")
pictures = [("wide", image), ("tall", image.transpose(Image.Transpose.TRANSPOSE))]
labels = ["cat", "remote control", "couch"]
ZeroShotObjectDetectionPipeline._get_bounding_box = lambda self, box: dict(zip(["xmin", "ymin", "xmax", "ymax"], box.tolist()))


def print_detections(name, detections):
    print(name, [(d["label"], round(d["score"], 6), [round(x, 4) for x in d["box"].values()]) for d in detections])


# GroundingDinoImageProcessor with shortest_edge and longest_edge resizes as DETR, with
# get_size_with_aspect_ratio. The export uses a fixed 800x800, so this processor has the
# default values and the size of IDEA-Research/grounding-dino-tiny. The 259x100 crop of the
# image gets 515x1333, and the rule of shortest_edge alone gives 514x1333.
values = GroundingDinoImageProcessor(size={"shortest_edge": 800, "longest_edge": 1333})(image.crop((0, 0, 259, 100)), return_tensors="np")["pixel_values"][0]
print("grounding-dino shortest_edge 800 pixel_values", values.shape, [round(float(values[p]), 4) for p in positions], round(float(np.abs(values).mean()), 4))

tile = Image.fromarray(np.tile(np.array(image), (6, 6, 1)))
for repo, export in [("google/owlvit-base-patch32", "Xenova/owlvit-base-patch32"), ("google/owlv2-base-patch16-ensemble", "Xenova/owlv2-base-patch16-ensemble")]:
    processor = AutoImageProcessor.from_pretrained(export, use_fast=False)
    tokenizer = AutoTokenizer.from_pretrained(export)
    for name, picture in [("wide", image), ("tile", tile)][: 2 if "owlv2" in repo else 1]:
        values = processor(picture, return_tensors="np")["pixel_values"][0]
        print(export, name, "pixel_values", values.shape, [round(float(values[p]), 4) for p in positions], round(float(np.abs(values).mean()), 4))
    model = AutoModelForZeroShotObjectDetection.from_pretrained(repo)
    detect = pipeline("zero-shot-object-detection", model=model, tokenizer=tokenizer, image_processor=processor, device="cpu")
    for name, picture in pictures:
        print_detections(f"{export} {name}", detect(picture, candidate_labels=labels))
        pixel_values = processor(picture, return_tensors="pt")["pixel_values"]
        with torch.no_grad():
            one_run = model(pixel_values=pixel_values, **tokenizer(labels, padding=True, return_tensors="pt")).logits[0]
            each = torch.stack([model(pixel_values=pixel_values, **tokenizer(label, return_tensors="pt")).logits[0, :, 0] for label in labels], 1)
        print(export, name, "one run and one run for each label, largest score difference", float((one_run.sigmoid() - each.sigmoid()).abs().max()))

processor = AutoImageProcessor.from_pretrained("onnx-community/grounding-dino-tiny-ONNX", use_fast=False)
tokenizer = AutoTokenizer.from_pretrained("onnx-community/grounding-dino-tiny-ONNX")
directory = snapshot_download("onnx-community/grounding-dino-tiny-ONNX", allow_patterns=["onnx/model.onnx"], local_dir=tempfile.mkdtemp())
session = ort.InferenceSession(f"{directory}/onnx/model.onnx", providers=["CPUExecutionProvider"])
for name, picture in pictures:
    inputs = {k: v.numpy() for k, v in processor(picture, return_tensors="pt").items()}
    detections = []
    for label in labels:
        logits, boxes = session.run(["logits", "pred_boxes"], {**inputs, **tokenizer(_merge_candidate_labels_text([label]), return_tensors="np")})
        outputs = SimpleNamespace(logits=torch.from_numpy(logits), pred_boxes=torch.from_numpy(boxes))
        result = processor.post_process_object_detection(outputs, threshold=0.1, target_sizes=[(picture.height, picture.width)])[0]
        detections += [{"label": label, "score": float(s), "box": dict(zip(["xmin", "ymin", "xmax", "ymax"], b.tolist()))} for s, b in zip(result["scores"], result["boxes"])]
    print_detections(f"grounding-dino onnxruntime {name}", sorted(detections, key=lambda d: d["score"], reverse=True))

# Mask generation with prompts. The mask-generation pipeline of Python transformers segments
# the whole image with a grid of points, so the reference is the PyTorch model with the
# processor and post_process_masks, one prompt for each call, with the 3 masks of
# multimask_output. SAM 1 (nielsr/slimsam-77-uniform, the model of the Xenova export) uses the
# slow SamImageProcessor. SAM 2.1 (facebook/sam2.1-hiera-tiny) has only Sam2ImageProcessorFast,
# which resizes 8-bit images with torchvision. The script prints pixel_values[0] at (channel,
# y, x) and the mean of the absolute values, for the wide image and, for SAM 2.1, for the 6x6
# tile (1920x1440), and the SHA-256 of the 8-bit image of the tile after the resize, from the
# pixel values. Then, for each image and prompt, it prints the iou_scores, and writes the
# masks to test/fixtures/sam_<model>_<image>.png, the 3 masks of each prompt one below the
# other, with 0 and 255. The Xenova export of SAM 1 has no box input. The script also prints
# the largest number of mask pixels that differ when ONNX Runtime runs the exports with the
# same processors.
from transformers import Sam2Model, Sam2Processor, SamModel, SamProcessor

prompts = [
    {"points": [[80, 120]], "labels": [1]},
    {"points": [[80, 120], [200, 60], [150, 200]], "labels": [1, 1, 0]},
    {"box": [40, 60, 200, 230]},
    {"points": [[80, 120]], "labels": [1], "box": [40, 60, 200, 230]},
]

for name, export, processor, model in [
    ("slimsam", "Xenova/slimsam-77-uniform", SamProcessor.from_pretrained("Xenova/slimsam-77-uniform"), SamModel.from_pretrained("nielsr/slimsam-77-uniform")),
    ("sam2", "onnx-community/sam2.1-hiera-tiny-ONNX", Sam2Processor.from_pretrained("onnx-community/sam2.1-hiera-tiny-ONNX"), Sam2Model.from_pretrained("facebook/sam2.1-hiera-tiny")),
]:
    for picture_name, picture in [("wide", image), ("tile", tile)][: 2 if name == "sam2" else 1]:
        values = processor(images=picture, return_tensors="pt")["pixel_values"][0].numpy()
        print(name, picture_name, "pixel_values", values.shape, [round(float(values[p]), 4) for p in positions], round(float(np.abs(values).mean()), 4))
        if picture_name == "tile":
            mean, std = (np.array(processor.image_processor.image_mean)[:, None, None], np.array(processor.image_processor.image_std)[:, None, None])
            resized = np.round((values * std + mean) * 255).astype(np.uint8)
            print(name, "tile, SHA-256 of the 8-bit resized image", hashlib.sha256(resized.tobytes()).hexdigest())
    directory = snapshot_download(export, allow_patterns=["onnx/vision_encoder.onnx*", "onnx/prompt_encoder_mask_decoder.onnx*"], local_dir=tempfile.mkdtemp())
    encoder = ort.InferenceSession(f"{directory}/onnx/vision_encoder.onnx", providers=["CPUExecutionProvider"])
    decoder = ort.InferenceSession(f"{directory}/onnx/prompt_encoder_mask_decoder.onnx", providers=["CPUExecutionProvider"])
    for picture_name, picture in pictures:
        masks, ort_differences = [], 0
        for prompt in prompts[: 4 if name == "sam2" else 2]:
            kwargs = {}
            if "points" in prompt:
                kwargs.update(input_points=[[prompt["points"]]], input_labels=[[prompt["labels"]]])
            if "box" in prompt:
                kwargs.update(input_boxes=[[prompt["box"]]])
            inputs = processor(images=picture, return_tensors="pt", **kwargs)
            sizes = [inputs["original_sizes"]] + ([inputs["reshaped_input_sizes"]] if name == "slimsam" else [])
            with torch.no_grad():
                output = model(**{k: v.clone() for k, v in inputs.items() if k not in ("original_sizes", "reshaped_input_sizes")}, multimask_output=True)
            mask = processor.post_process_masks(output.pred_masks, *sizes)[0][0].numpy()
            masks.append(mask)
            print(name, picture_name, prompt, "iou_scores", [round(float(x), 5) for x in output.iou_scores[0, 0]])

            embeddings = dict(zip([o.name for o in encoder.get_outputs()], encoder.run(None, {"pixel_values": inputs["pixel_values"].numpy()})))
            feeds = {k: v for k, v in embeddings.items() if k in [i.name for i in decoder.get_inputs()]}
            feeds["input_points"] = inputs["input_points"].numpy().astype(np.float32) if "points" in prompt else np.zeros((1, 1, 0, 2), np.float32)
            feeds["input_labels"] = inputs["input_labels"].numpy() if "points" in prompt else np.zeros((1, 1, 0), np.int64)
            if name == "sam2":
                feeds["input_boxes"] = inputs["input_boxes"].numpy() if "box" in prompt else np.zeros((1, 0, 4), np.float32)
            ort_masks = decoder.run(["pred_masks"], feeds)[0]
            ort_mask = processor.post_process_masks(torch.from_numpy(ort_masks), *sizes)[0][0].numpy()
            ort_differences = max(ort_differences, int((ort_mask != mask).sum(axis=(1, 2)).max()))
        Image.fromarray(np.concatenate([m.reshape(-1, m.shape[-1]) for m in masks]).astype(np.uint8) * 255).save(f"test/fixtures/sam_{name}_{picture_name}.png")
        print(name, picture_name, "largest number of different pixels in a mask of ONNX Runtime", ort_differences)

# Beam search, as Transmuter.Generation with :num_beams.
from transformers import GenerationMixin
from transformers.generation.logits_process import WhisperTimeStampLogitsProcessor

# Beam search: generate with num_beams and do_sample=False. The script also prints the margin
# of each input: the score of the best beam minus the score of the second beam. A small margin
# shows a result that small numeric differences can change. Whisper runs the generate of
# GenerationMixin with the prompt of the Whisper generate, because the Whisper generate gives
# no beam scores. Both give the same text.
def beams(model, inputs, **options):
    output = GenerationMixin.generate(model, **inputs, do_sample=False, num_return_sequences=options["num_beams"], output_scores=True, return_dict_in_generate=True, **options)
    scores = output.sequences_scores.view(-1, options["num_beams"])
    return output.sequences[:: options["num_beams"]], [round(float(s[0] - s[1]), 5) for s in scores]

# Each case gives another text than greedy, so it fails when the beams fall back to greedy.
# "What is 2 + 2?" ends with eos at the eighth token while other beams still run. The
# options of "Say yes or no." and of the T5 inputs give other texts than the defaults.
def greedy(model, inputs, **options):
    return model.generate(**inputs, do_sample=False, num_beams=1, **options)

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
for question, options in [("What is 2 + 2?", {}), ("Say yes or no.", {}), ("Say yes or no.", {"length_penalty": 2.0}), ("Say yes or no.", {"length_penalty": 2.0, "early_stopping": "never"})]:
    inputs = tokenizer.apply_chat_template([{"role": "user", "content": question}], add_generation_prompt=True, return_tensors="pt", return_dict=True)
    n = inputs["input_ids"].shape[1]
    tokens = 12 if question == "What is 2 + 2?" else 16
    sequences, margins = beams(model, inputs, num_beams=3, max_new_tokens=tokens, **options)
    text = tokenizer.decode(sequences[0, n:], skip_special_tokens=True)
    print("smollm2 beams", repr(question), options, repr(text), margins, "greedy", repr(tokenizer.decode(greedy(model, inputs, max_new_tokens=tokens)[0, n:], skip_special_tokens=True)))

tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
inputs = tokenizer("The history of the printing press begins", return_tensors="pt")
sequences, margins = beams(model, inputs, num_beams=3, max_new_tokens=12)
print("gpt2 beams", repr(tokenizer.decode(sequences[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)), margins)

# Python runs each T5 text alone, without padding. The T5 text keeps the spaces of the tokens,
# as the text2text-generation pipeline decodes with clean_up_tokenization_spaces=False.
# Marian runs with the settings of its generation_config.json: 4 beams, bad_words_ids,
# forced_eos_token_id and renormalize_logits.
sentence = "Transmuter runs ONNX models on any Nx backend, with no code for each architecture."
tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-small")
for options in [{}, {"early_stopping": True}, {"length_penalty": -1.0}]:
    for text in ["summarize: Scientists found a new species of frog in the rainforest. The frog is very small and has bright colors. It lives near streams and eats insects.", "translate English to French: The weather is nice today, so we walk to the park."]:
        inputs = tokenizer(text, return_tensors="pt")
        sequences, margins = beams(model, inputs, num_beams=3, max_new_tokens=48, **options)
        decode = lambda ids: repr(tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False))
        print("t5 beams", options, decode(sequences[0]), margins, "greedy", decode(greedy(model, inputs, max_new_tokens=48)[0]))

tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")
model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-de")
sequences, margins = beams(model, tokenizer(sentence, return_tensors="pt"), num_beams=model.generation_config.num_beams, max_new_tokens=64)
print("marian beams", model.generation_config.num_beams, repr(tokenizer.decode(sequences[0], skip_special_tokens=True)), margins)

# With 5 new tokens, forced_eos_token_id gives the fifth token of each beam.
sequences, margins = beams(model, tokenizer(sentence, return_tensors="pt"), num_beams=model.generation_config.num_beams, max_new_tokens=5)
print("marian beams, 5 tokens", sequences[0].tolist(), repr(tokenizer.decode(sequences[0], skip_special_tokens=True)), margins)

# Whisper with 3 beams on test/fixtures/librispeech_0.f32: whisper-base in French, and
# whisper-tiny.en with timestamps.
audio = np.fromfile("test/fixtures/librispeech_0.f32", dtype="<f4")
processor = WhisperProcessor.from_pretrained("openai/whisper-base")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
features = processor(audio, sampling_rate=16000, return_tensors="pt").input_features
config = model.generation_config
prompt = torch.tensor([[config.decoder_start_token_id, config.lang_to_id["<|fr|>"], config.task_to_id["transcribe"], config.no_timestamps_token_id]])
sequences, margins = beams(model, {"input_features": features, "decoder_input_ids": prompt}, num_beams=3, max_new_tokens=444)
print("whisper-base fr beams", repr(processor.decode(sequences[0], skip_special_tokens=True)), margins)
print("whisper-base fr generate", repr(processor.batch_decode(model.generate(features, do_sample=False, num_beams=3, language="fr", task="transcribe"), skip_special_tokens=True)[0]))

processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")
features = processor(audio, sampling_rate=16000, return_tensors="pt").input_features
config = model.generation_config
prompt = torch.tensor([[config.decoder_start_token_id]])
timestamps = [WhisperTimeStampLogitsProcessor(config, begin_index=1)]
sequences, margins = beams(model, {"input_features": features, "decoder_input_ids": prompt, "logits_processor": timestamps}, num_beams=3, max_new_tokens=447)
print("whisper-tiny.en timestamps beams", repr(processor.decode(sequences[0], decode_with_timestamps=True)), margins)
print("whisper-tiny.en timestamps generate", repr(processor.batch_decode(model.generate(features, do_sample=False, num_beams=3, return_timestamps=True), decode_with_timestamps=True)[0]))
