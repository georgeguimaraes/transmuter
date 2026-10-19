# Vision and Multimodal

This guide shows how to run image classification, zero-shot image-text models, and vision-language model pipelines with Transmuter.

## Image preprocessing

Most vision models expect a float tensor of shape `{batch, channels, height, width}` with normalized pixel values. The preprocessing steps (resize, crop, rescale, normalize) are not in the ONNX file. You must do them before you give the tensor to the model.

Each model gives its preprocessing in `preprocessor_config.json`. The values differ between models. For example, ViT uses a mean and std of `0.5`, CLIP uses `[0.4815, 0.4578, 0.4082]` and `[0.2686, 0.2613, 0.2758]`, and mobileclip does not normalize. Read the file with `Transmuter.Hub.json/3`:

```elixir
source = {:hf, "Xenova/vit-base-patch16-224"}
config = Transmuter.Hub.json(source, "preprocessor_config.json")
# %{"size" => %{"height" => 224, "width" => 224}, "rescale_factor" => 0.00392156862745098,
#   "image_mean" => [0.5, 0.5, 0.5], "image_std" => [0.5, 0.5, 0.5], "do_normalize" => true, ...}
```

Use [StbImage](https://hex.pm/packages/stb_image) to load the image and resize it to `size`:

```elixir
%{"height" => height, "width" => width} = config["size"]

pixel_values =
  "image.jpg"
  |> StbImage.read_file!(channels: 3)
  |> StbImage.resize(height, width)
  |> StbImage.to_nx()
  # {height, width, 3} -> {1, 3, height, width}
  |> Nx.transpose(axes: [2, 0, 1])
  |> Nx.new_axis(0)
  |> Nx.as_type(:f32)
  |> Nx.multiply(config["rescale_factor"])

pixel_values =
  if config["do_normalize"] do
    mean = config["image_mean"] |> Nx.tensor() |> Nx.reshape({1, 3, 1, 1})
    std = config["image_std"] |> Nx.tensor() |> Nx.reshape({1, 3, 1, 1})
    pixel_values |> Nx.subtract(mean) |> Nx.divide(std)
  else
    pixel_values
  end
```

Some models give `size` as `%{"shortest_edge" => n}` and a `crop_size`. For these models, resize so that the short side is `n` and keep the aspect ratio. Then take the center `crop_size` region.

## Image classification

Run a ViT model on an image:

```elixir
model = Transmuter.load!(
  {:hf, "Xenova/vit-base-patch16-224", "onnx/model.onnx"},
  backend: EMLX.Backend
)

run = Transmuter.compile(model, %{"pixel_values" => pixel_values}, compiler: EMLX)
outputs = run.(%{"pixel_values" => pixel_values})
logits = outputs["logits"]
# Shape: {1, 1000}, one score per ImageNet class.
```

## Zero-shot image classification with CLIP

[CLIP](https://huggingface.co/Xenova/mobileclip_s0) encodes images and text into a shared embedding space. Zero-shot classification compares the image embedding with the text embeddings of candidate labels.

The mobileclip export has two ONNX files: `vision_model.onnx` and `text_model.onnx`. Load them separately:

```elixir
vision_model = Transmuter.load!(
  {:hf, "Xenova/mobileclip_s0", "onnx/vision_model.onnx"},
  backend: EMLX.Backend
)

text_model = Transmuter.load!(
  {:hf, "Xenova/mobileclip_s0", "onnx/text_model.onnx"},
  backend: EMLX.Backend
)

# Vision input: {batch, 3, 256, 256}. The preprocessor config of mobileclip resizes the
# short side to 256, crops the center 256x256 and rescales to [0, 1] with no normalization.
vision_inputs = %{"pixel_values" => pixel_values}

# Text input: {num_labels, 77}. CLIP uses a fixed context length of 77 tokens.
# Pad each label to length 77 with zero (the padding token id).
text_inputs = %{"input_ids" => input_ids}  # input_ids shape {num_labels, 77}, type :s64

vision_run = Transmuter.compile(vision_model, vision_inputs, compiler: EMLX)
text_run = Transmuter.compile(text_model, text_inputs, compiler: EMLX)

image_embeds = vision_run.(vision_inputs)["image_embeds"]  # {1, 512}
text_embeds = text_run.(text_inputs)["text_embeds"]        # {num_labels, 512}

# Cosine similarity between image and each text embedding.
normalize = &Nx.divide(&1, Nx.LinAlg.norm(&1, axes: [-1], keep_axes: true))
similarity = Nx.dot(normalize.(image_embeds), Nx.transpose(normalize.(text_embeds)))  # {1, num_labels}
```

## Vision-language model pipeline

A VLM (vision-language model) combines a vision encoder with a decoder. The typical pipeline has these components, each as a separate ONNX file:

1. **Vision encoder** (`vision_encoder.onnx`): encodes the image into patch embeddings.
2. **Token embedding** (`embed_tokens.onnx`): encodes the text tokens.
3. **Merged decoder** (`decoder_model_merged.onnx`): generates tokens from the combined image and text embeddings.

Load and run each component:

```elixir
vision_encoder = Transmuter.load!(
  {:hf, repo, "onnx/vision_encoder.onnx"},
  backend: EMLX.Backend
)

embed_tokens = Transmuter.load!(
  {:hf, repo, "onnx/embed_tokens.onnx"},
  backend: EMLX.Backend
)

decoder = Transmuter.load!(
  {:hf, repo, "onnx/decoder_model_merged.onnx"},
  backend: EMLX.Backend
)

image_features = Transmuter.run(vision_encoder, vision_encoder.params, %{"pixel_values" => pixel_values})
token_embeds = Transmuter.run(embed_tokens, embed_tokens.params, %{"input_ids" => input_ids})

# Merge and pass to the decoder.
```

Check `model.inputs` and `model.outputs` for the exact tensor names in your export. Florence-2 and LLaVA exports follow this pattern.

## SAM (Segment Anything Model)

[SAM 2.1](https://huggingface.co/onnx-community/sam2.1-hiera-tiny-ONNX) generates masks from point or box prompts. Its export has two ONNX files: a vision encoder and a prompt encoder with mask decoder.

The tested version gives masks with IoU 1.0 compared with ONNX Runtime for prompts of 1 point, 3 points, a box, and a box with a point.

```elixir
vision_encoder = Transmuter.load!(
  {:hf, "onnx-community/sam2.1-hiera-tiny-ONNX", "onnx/vision_encoder.onnx"},
  backend: EMLX.Backend
)

mask_decoder = Transmuter.load!(
  {:hf, "onnx-community/sam2.1-hiera-tiny-ONNX", "onnx/prompt_encoder_mask_decoder.onnx"},
  backend: EMLX.Backend
)
```

## Depth estimation

Depth Anything v2 takes a pixel tensor and returns a depth map of the same spatial size. It has no graph breaks:

```elixir
model = Transmuter.load!(
  {:hf, "onnx-community/depth-anything-v2-small", "onnx/model.onnx"},
  backend: EMLX.Backend
)

run = Transmuter.compile(model, %{"pixel_values" => pixel_values}, compiler: EMLX)
depth = run.(%{"pixel_values" => pixel_values})["predicted_depth"]
```

## Object detection with RF-DETR

[RF-DETR](https://huggingface.co/onnx-community/rfdetr_nano-ONNX) is a real-time object detector. Its export takes a `pixel_values` tensor of shape `{batch, 3, 544, 544}` and returns 300 proposals per image:

```elixir
model = Transmuter.load!(
  {:hf, "onnx-community/rfdetr_nano-ONNX", "onnx/model.onnx"},
  backend: EMLX.Backend
)

# Input must be {batch, 3, 544, 544}: the Swin window requires an even patch count.
run = Transmuter.compile(model, %{"pixel_values" => pixel_values}, compiler: EMLX)
outputs = run.(%{"pixel_values" => pixel_values})
pred_boxes = outputs["pred_boxes"]  # {batch, 300, 4}, normalized cx/cy/w/h
logits = outputs["logits"]          # {batch, 300, 91}, one score per COCO class
```
