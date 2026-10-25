"""Prints the reference values of the SAM 3 tracker mask generation test in test/transmuter/vision_test.exs, and writes its masks to test/fixtures/sam_sam3_<image>.png.

Usage: uv run --with "transformers>=5" --with torch --with torchvision --with onnxruntime --with onnx --with pillow python -I test/support/sam3_tracker_reference.py

Only transformers 5 has the SAM 3 image processor, and test/support/pipeline_references.py uses transformers 4, so this script is separate. The PyTorch weights (facebook/sam3) are gated, so the reference is ONNX Runtime on the fp32 export of onnx-community/sam3-tracker-ONNX, with graph optimizations off. The processor is that of the export: Sam2Processor with Sam3ImageProcessorFast, which resizes to 1008x1008 with torchvision, and post_process_masks. For each image and prompt, the script prints the iou_scores and writes the 3 masks of each prompt one below the other, with 0 and 255. It also prints the largest number of mask pixels that change when the decoder logits move by 1e-4, the noise of the reference.
"""

import hashlib
import tempfile

# The downloads go into this directory, and Python deletes it at exit.
scratch = tempfile.TemporaryDirectory()

import numpy as np
import onnxruntime as ort
import torch
from huggingface_hub import snapshot_download
from PIL import Image
from transformers import Sam2Processor

export = "onnx-community/sam3-tracker-ONNX"
processor = Sam2Processor.from_pretrained(export)
print("sam3", type(processor.image_processor).__name__)

# pixel_values[0] at (channel, y, x) and the mean of the absolute values, for the image and
# its 6x6 tile (1920x1440), and the SHA-256 of the 8-bit image of the tile after the resize.
image = Image.open("test/fixtures/cats.png")
tile = Image.fromarray(np.tile(np.array(image), (6, 6, 1)))
positions = [(0, 0, 0), (1, 100, 37), (2, 223, 223), (0, 57, 190), (1, 150, 120)]
for picture_name, picture in [("wide", image), ("tile", tile)]:
    values = processor(images=picture, return_tensors="pt")["pixel_values"][0].numpy()
    print("sam3", picture_name, "pixel_values", values.shape, [round(float(values[p]), 4) for p in positions], round(float(np.abs(values).mean()), 4))
resized = np.round((values * 0.5 + 0.5) * 255).astype(np.uint8)
print("sam3 tile, SHA-256 of the 8-bit resized image", hashlib.sha256(resized.tobytes()).hexdigest())

directory = snapshot_download(export, allow_patterns=["onnx/vision_encoder.onnx*", "onnx/prompt_encoder_mask_decoder.onnx*"], local_dir=tempfile.mkdtemp(dir=scratch.name))
options = ort.SessionOptions()
options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
encoder = ort.InferenceSession(f"{directory}/onnx/vision_encoder.onnx", options, providers=["CPUExecutionProvider"])
decoder = ort.InferenceSession(f"{directory}/onnx/prompt_encoder_mask_decoder.onnx", options, providers=["CPUExecutionProvider"])

prompts = [
    {"points": [[80, 120]], "labels": [1]},
    {"points": [[80, 120], [200, 60], [150, 200]], "labels": [1, 1, 0]},
    {"box": [40, 60, 200, 230]},
    {"points": [[80, 120]], "labels": [1], "box": [40, 60, 200, 230]},
]

for picture_name, picture in [("wide", image), ("tall", image.transpose(Image.Transpose.TRANSPOSE))]:
    pixel_values = processor(images=picture, return_tensors="pt")["pixel_values"]
    embeddings = dict(zip([o.name for o in encoder.get_outputs()], encoder.run(None, {"pixel_values": pixel_values.numpy()})))
    masks, noise = [], 0
    for prompt in prompts:
        kwargs = {}
        if "points" in prompt:
            kwargs.update(input_points=[[prompt["points"]]], input_labels=[[prompt["labels"]]])
        if "box" in prompt:
            kwargs.update(input_boxes=[[prompt["box"]]])
        inputs = processor(images=picture, return_tensors="pt", **kwargs)
        feeds = {k: v for k, v in embeddings.items() if k in [i.name for i in decoder.get_inputs()]}
        feeds["input_points"] = inputs["input_points"].numpy().astype(np.float32) if "points" in prompt else np.zeros((1, 1, 0, 2), np.float32)
        feeds["input_labels"] = inputs["input_labels"].numpy() if "points" in prompt else np.zeros((1, 1, 0), np.int64)
        feeds["input_boxes"] = inputs["input_boxes"].numpy() if "box" in prompt else np.zeros((1, 0, 4), np.float32)
        logits, scores = decoder.run(["pred_masks", "iou_scores"], feeds)
        mask = processor.post_process_masks(torch.from_numpy(logits), inputs["original_sizes"])[0][0].numpy()
        masks.append(mask)
        for shift in [-1e-4, 1e-4]:
            moved = processor.post_process_masks(torch.from_numpy(logits + shift), inputs["original_sizes"])[0][0].numpy()
            noise = max(noise, int((moved != mask).sum(axis=(1, 2)).max()))
        print("sam3", picture_name, prompt, "iou_scores", [round(float(x), 5) for x in scores[0, 0]])
    Image.fromarray(np.concatenate([m.reshape(-1, m.shape[-1]) for m in masks]).astype(np.uint8) * 255).save(f"test/fixtures/sam_sam3_{picture_name}.png")
    print("sam3", picture_name, "largest number of mask pixels that change with a logit shift of 1e-4", noise)
