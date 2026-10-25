"""Prints the reference values of RF-DETR object detection in test/transmuter/vision_test.exs.

Usage: uv run --with "transformers>=5" --with torch --with torchvision --with onnxruntime --with pillow python -I test/support/rf_detr_reference.py

Only transformers 5 has RF-DETR, and test/support/pipeline_references.py uses transformers 4, so this script is separate. The model is the PyTorch model of Roboflow/rf-detr-nano, which has the weights of onnx-community/rfdetr_nano-ONNX. The image processor of RF-DETR uses torchvision, whose resize differs from Pillow by up to one step of 1/255. So the pixel values come from the Pillow image processor of RT-DETR, with the size, the mean and the std of the configuration of Roboflow/rf-detr-nano. The post-processing is that of the image processor of RF-DETR, with the threshold of the pipeline, 0.5.

The script also runs ONNX Runtime on the fp32 export of onnx-community. Its preprocessor_config.json sets do_normalize to false, but its graph does not normalize the pixels. With normalized pixels, its logits are those of PyTorch.
"""

import tempfile

# The downloads go into this directory, and Python deletes it at exit.
scratch = tempfile.TemporaryDirectory()

import numpy as np
import onnxruntime as ort
import torch
from huggingface_hub import snapshot_download
from PIL import Image
from transformers import RfDetrForObjectDetection
from transformers.models.rf_detr.image_processing_rf_detr import RfDetrImageProcessor
from transformers.models.rt_detr.image_processing_pil_rt_detr import RTDetrImageProcessorPil

processor = RTDetrImageProcessorPil(size={"height": 384, "width": 384}, do_normalize=True, image_mean=[0.485, 0.456, 0.406], image_std=[0.229, 0.224, 0.225], do_pad=False)
model = RfDetrForObjectDetection.from_pretrained("Roboflow/rf-detr-nano")
post = RfDetrImageProcessor()

directory = snapshot_download("onnx-community/rfdetr_nano-ONNX", allow_patterns=["onnx/model.onnx"], local_dir=tempfile.mkdtemp(dir=scratch.name))
session = ort.InferenceSession(f"{directory}/onnx/model.onnx", providers=["CPUExecutionProvider"])

image = Image.open("test/fixtures/cats.png")
for name, picture in [("wide", image), ("tall", image.transpose(Image.Transpose.TRANSPOSE))]:
    pixel_values = processor(picture, return_tensors="pt")["pixel_values"]
    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)
    result = post.post_process_object_detection(outputs, threshold=0.5, target_sizes=[(picture.height, picture.width)])[0]
    print("Roboflow/rf-detr-nano", name, [(model.config.id2label[int(label)], round(float(score), 6), [round(float(x), 4) for x in box]) for score, label, box in zip(result["scores"], result["labels"], result["boxes"])])
    logits = session.run(["logits"], {"pixel_values": pixel_values.numpy()})[0]
    print("onnxruntime max abs difference of the logits", float(np.abs(logits - outputs.logits.numpy()).max()))
