defmodule Transmuter.Vision do
  @moduledoc """
  Servings for image tasks. Each function loads a model repository (the ONNX files, `preprocessor_config.json`, the configuration and, for text, the tokenizer) and returns an `Nx.Serving` that takes images and gives results in the shapes of Bumblebee.

      serving = Transmuter.Vision.image_classification({:hf, "Xenova/vit-base-patch16-224"})
      image = StbImage.read_file!("cat.jpg")
      Nx.Serving.run(serving, image)
      #=> %{predictions: [%{label: "Egyptian cat", score: 0.94}, ...]}

  An image is a `StbImage` struct or a tensor of shape `{height, width, channels}`, usually of type `:u8`. To read image files, add the optional `stb_image` dependency: `{:stb_image, "~> 1.0"}`.

  `Transmuter.Featurizer` prepares the images as the image processors of Python transformers do. The resize gives the same pixels as Pillow.

  ## Options

  All tasks take these options:

    * `:compile` - `:batch_size`. The default is `1`. The serving compiles the model for the image size of `preprocessor_config.json`. When the size depends on the image, such as with `shortest_edge` and no crop, the batch size must be 1. The serving then runs each image alone, and the model compiles each new image size on its first call.

    * `:defn_options` - options of `Transmuter.compile/3`, such as `compiler: EMLX`.

    * `:backend` and `:compute_type` - options of `Transmuter.load!/2`, such as `backend: EMLX.Backend`. The featurizer also uses `:backend` to resize the images. Without it, the featurizer uses `Nx.default_backend/0`.

    * `:variant` - the variant of the ONNX file, such as `:q4`. See `Transmuter.Repo`.

    * `:cache_dir`, `:revision` and `:offline` - see `Transmuter.Hub`.

    * `:model` and `:featurizer` - a model from `Transmuter.load!/2` and a featurizer from `Transmuter.Featurizer.load!/2`, to use in place of the files of the repository.

  Run a serving with `Nx.Serving.run/2`, or start it under your supervision tree and call `Nx.Serving.batched_run/2`. See `Transmuter.Text` for an example.
  """

  @doc """
  Builds a serving that classifies images.

  The input is an image or a list of images. The result is `%{predictions: [%{label: label, score: score}]}`, from the highest score.

  The labels come from `id2label` in `config.json`. As in Python transformers, the scores are a sigmoid when the model has one label or `problem_type` is `"multi_label_classification"`, and else a softmax.

  ## Options

    * `:top_k` - the number of predictions, or `nil` for all. The default is `5`.

    * `:scores_function` - `:softmax`, `:sigmoid` or `:none` for the logits.

  See the module documentation for the other options.
  """
  defdelegate image_classification(source, opts \\ []), to: Transmuter.Vision.ImageClassification

  @doc """
  Builds a serving that turns an image into an embedding.

  The input is an image or a list of images. The result is `%{embedding: tensor}`, or a list of them for a list.

  The embedding is the first of these model outputs:

    * `image_embeds` - the projected embedding of the image tower of CLIP or SigLIP. The serving uses `vision_model.onnx` when the repository has it.
    * `pooler_output` - the pooled output of an image encoder.
    * `last_hidden_state` - the first token of the output, which is the pooled output of DINOv2.

  ## Options

    * `:embedding_processor` - `:l2_norm` or `nil`. The default is `nil`.

  See the module documentation for the other options.
  """
  defdelegate image_embedding(source, opts \\ []), to: Transmuter.Vision.ImageEmbedding

  @doc """
  Builds a serving that classifies images with labels that you give, such as CLIP and SigLIP.

  `labels` is a list of strings. The input is an image or a list of images. The result is `%{predictions: [%{label: label, score: score}]}`, from the highest score.

  The serving uses `model.onnx`, which has the text tower, the image tower, and the logit scale and bias of the model. `vision_model.onnx` and `text_model.onnx` do not have the logit scale and bias, so the serving does not use them. Each label goes into the hypothesis template, and the text of all labels goes into each batch. The text is padded to `model_max_length` of `tokenizer_config.json`.

  As in Python transformers, the scores are a sigmoid for SigLIP models and a softmax over the labels for other models.

  ## Options

    * `:hypothesis_template` - the text for each label, with `{}` in the place of the label. The default is `"This is a photo of {}."`, as in Python transformers.

    * `:top_k` - the number of predictions, or `nil` for all. The default is `5`.

    * `:tokenizer` - a tokenizer from `Transmuter.Tokenizer.load!/2`, to use in place of the tokenizer of the repository.

  The text needs the optional `tokenizers` dependency: add `{:tokenizers, "~> 0.5"}` to your deps. See the module documentation for the other options.
  """
  defdelegate zero_shot_image_classification(source, labels, opts \\ []),
    to: Transmuter.Vision.ZeroShotImageClassification

  @doc """
  Builds a serving that finds objects in images, with a DETR, YOLOS, RT-DETR or RF-DETR model.

  The input is an image or a list of images. The result is `%{detections: [%{label: label, score: score, box: box}]}`. A box is `%{xmin: x, ymin: y, xmax: x, ymax: y}`, in pixels of the input image. The coordinates are floats. They can be a little outside the image, as in Python transformers. The pipeline of Python transformers truncates them to integers.

      serving = Transmuter.Vision.object_detection({:hf, "Xenova/detr-resnet-50"})
      Nx.Serving.run(serving, image)
      #=> %{detections: [%{label: "remote", score: 0.997, box: %{xmin: 19.9, ymin: 35.1, xmax: 87.3, ymax: 59.2}}, ...]}

  The labels come from `id2label` in `config.json`. The scores and the boxes are those of `post_process_object_detection` of Python transformers, and `model_type` in `config.json` selects the method:

    * `"detr"` and `"yolos"` - each query gives the class with the highest softmax score. The last class is "no object", and the softmax does not give it. The detections are in the order of the queries.

    * `"rt_detr"`, `"rt_detr_v2"` and `"rf_detr"` - each pair of a query and a class gets a sigmoid score. The `:top_k` pairs with the highest scores are the detections, from the highest score.

  DETR and YOLOS resize the short side of the image to `shortest_edge` and keep the aspect ratio, so the size of the pixel values depends on the image. Then each image is a batch of its own, and the model compiles each new size on its first call. YOLOS also rounds each side down to a multiple of 16, as in Python transformers. The DETR exports of Xenova take a pixel mask of 64x64. The serving gives a mask of ones, because the images of a batch are not padded.

  The RF-DETR exports of onnx-community, such as `onnx-community/rfdetr_nano-ONNX`, set `do_normalize` to `false` in `preprocessor_config.json`, but their graphs do not normalize the pixels. The serving normalizes the pixels of RF-DETR with `image_mean` and `image_std`, as the RF-DETR models of Python transformers do.

  The default variant is fp32, which gives the scores of the PyTorch models. The q8 files quantize the activations when the model runs, so their scores can be different in each runtime. See [Backends and performance](guides/backends_and_performance.md).

  ## Options

    * `:threshold` - the minimum score of a detection. A detection has a higher score than the threshold. The default is `0.5`, as in Python transformers.

    * `:top_k` - the number of pairs of a query and a class that RT-DETR and RF-DETR keep before the threshold. The default is the number of queries, as in Python transformers. DETR and YOLOS do not use this option.

    * `:compile` - `:batch_size`. The default is `1`. When the size of the images depends on the image, as with DETR and YOLOS, the batch size must be `1`.

  See the module documentation for the other options.
  """
  defdelegate object_detection(source, opts \\ []), to: Transmuter.Vision.ObjectDetection

  @doc """
  Builds a serving that finds objects in images with labels that you give, such as OWL-ViT, OWLv2 and Grounding DINO.

  `labels` is a list of strings. The input is an image or a list of images. The result is `%{detections: [%{label: label, score: score, box: box}]}`, from the highest score, as the `zero-shot-object-detection` pipeline of Python transformers gives. A box is `%{xmin: x, ymin: y, xmax: x, ymax: y}`, in pixels of the input image. The coordinates are floats. The pipeline of Python transformers truncates them to integers.

      serving = Transmuter.Vision.zero_shot_object_detection({:hf, "Xenova/owlv2-base-patch16-ensemble"}, ["cat", "remote control"])
      Nx.Serving.run(serving, image)
      #=> %{detections: [%{label: "remote control", score: 0.747, box: %{xmin: 167.3, ymin: 38.5, xmax: 185.0, ymax: 93.7}}, ...]}

  The pipeline of Python transformers runs the model once for each label. The score of a box for a label is the sigmoid of the highest logit of the box, and each box with a score higher than `:threshold` is a detection. So one box can be a detection of more than one label. `model_type` in `config.json` selects how the serving runs the model:

    * `"owlvit"` and `"owlv2"` - the labels are the text queries of one model run. Each query gets its own logit for each box, so the scores are the same as with one run for each label. The images go into batches. OWLv2 pads the image to a square, and its boxes are relative to that square, as in Python transformers.

    * `"grounding-dino"` - the text and the image go through the same layers, so the model runs once for each label and image. The ONNX exports of Grounding DINO compute the phrases of the text for one phrase and a period. So the text of a label is the label without spaces at the ends, in lowercase, and a period, as `GroundingDinoProcessor` of Python transformers makes it for a list of labels. The pipeline of Python transformers with the PyTorch model gives the label alone, without the period, so its scores and boxes are different. A label cannot have `"."` or `"?"`. The score of a box is the sigmoid of its highest logit over the tokens of the text.

  The featurizer prepares the images as the image processors of Python transformers, see `Transmuter.Featurizer`. OWLv2 rescales the image, pads it to a square with gray, and resizes it with a gaussian blur and a linear interpolation.

  ## Options

    * `:threshold` - the minimum score of a detection. A detection has a higher score than the threshold. The default is `0.1`, as in Python transformers.

    * `:top_k` - the number of detections with the highest scores to keep, or `nil` for all. The default is `nil`.

    * `:compile` - `:batch_size`. The default is `1`. Grounding DINO runs each image alone, so its batch size must be `1`.

    * `:tokenizer` - a tokenizer from `Transmuter.Tokenizer.load!/2`, to use in place of the tokenizer of the repository.

  The text needs the optional `tokenizers` dependency: add `{:tokenizers, "~> 0.5"}` to your deps. See the module documentation for the other options.
  """
  defdelegate zero_shot_object_detection(source, labels, opts \\ []),
    to: Transmuter.Vision.ZeroShotObjectDetection

  @doc """
  Builds a serving that estimates the depth of each pixel of an image, such as Depth Anything.

  The input is an image or a list of images. The result is `%{predicted_depth: tensor, depth: tensor}`, as the `depth-estimation` pipeline of Python transformers gives. Both tensors have the shape `{height, width}` of the input image.

      serving = Transmuter.Vision.depth_estimation({:hf, "onnx-community/depth-anything-v2-small"})
      Nx.Serving.run(serving, image)
      #=> %{predicted_depth: #Nx.Tensor<f32[240][320]>, depth: #Nx.Tensor<u8[240][320]>}

  `predicted_depth` is the output of the model, resized to the image with the bicubic `torch.nn.functional.interpolate` of Python transformers (see `Transmuter.Serving.interpolate/3`). Depth Anything gives a relative depth, which is larger for nearer pixels. `depth` is an image of 8 bits: the minimum of `predicted_depth` is 0 and the maximum is 255.

  The featurizer resizes the image as the DPT image processor of Python transformers. With `keep_aspect_ratio`, both sides take the scale of `size` that is nearer to 1, and each side becomes the nearest multiple of `ensure_multiple_of`. So the size of the pixel values depends on the image, and the model compiles each new image size on its first call.

  ## Options

    * `:compile` - `:batch_size` must be `1`, the default. The depth map has the size of the image, so each image is a batch of its own.

  See the module documentation for the other options.
  """
  defdelegate depth_estimation(source, opts \\ []), to: Transmuter.Vision.DepthEstimation

  @doc """
  Builds a serving that gives the label of each pixel of an image, with a semantic segmentation model such as Segformer.

  The input is an image or a list of images. The result is `%{segments: [%{label: label, score: nil, mask: tensor}]}`, as the `image-segmentation` pipeline of Python transformers gives for semantic segmentation. Each mask is a `:u8` tensor of shape `{height, width}` of the input image, with 255 for the pixels of its label and 0 for the other pixels.

      serving = Transmuter.Vision.image_segmentation({:hf, "Xenova/segformer-b0-finetuned-ade-512-512"})
      Nx.Serving.run(serving, image)
      #=> %{segments: [%{label: "wall", score: nil, mask: #Nx.Tensor<u8[240][320]>}, ...]}

  As `post_process_semantic_segmentation` of Python transformers, the logits of the model are resized to the image with the bilinear `torch.nn.functional.interpolate` (see `Transmuter.Serving.interpolate/3`), and each pixel gets the label with the highest logit. Each label that some pixel has gives one segment, in the order of the label ids. The labels come from `id2label` in `config.json`. Semantic segmentation gives no scores, so the score is `nil`.

  The model must give `logits` of shape `{batch, labels, height, width}`. Panoptic and instance segmentation models, such as DETR and Mask2Former, give mask outputs, and the serving raises an `ArgumentError` for them. `do_reduce_labels` of `preprocessor_config.json` changes only the segmentation maps for training, so the featurizer does not use it.

  See the module documentation for the options.
  """
  defdelegate image_segmentation(source, opts \\ []), to: Transmuter.Vision.ImageSegmentation

  @doc """
  Builds a serving that gives the mask of the foreground of an image, with a model such as ORMBG.

  The input is an image or a list of images. The result is `%{mask: tensor}`, a `:u8` tensor of shape `{height, width}` of the input image. A pixel of the foreground is near 255, and a pixel of the background is near 0.

      serving = Transmuter.Vision.background_removal({:hf, "onnx-community/ormbg-ONNX"})
      %{mask: mask} = Nx.Serving.run(serving, image)

  Python transformers has no pipeline for this task, so the serving follows the `background-removal` pipeline of transformers.js. The mask is the first output of the model. When a value of the output is outside 0..1 by more than 1.0e-5, the mask is the sigmoid of the output. The mask is 255 times the value, truncated to 8 bits. Then the bilinear resize of Pillow (see `Transmuter.Featurizer.resize/3`) gives the mask the size of the image. In Node, transformers.js resizes with the affine bilinear of sharp, which does not filter when the image gets smaller. So on the same output, its mask can be different at the edges of the foreground. On a 320x240 image and the 1024x1024 output of ORMBG, the difference is up to 100 steps.

  The pipeline of transformers.js gives an image with an alpha channel. To make this image, add the mask to the pixels as the fourth channel:

      pixels = image |> StbImage.to_nx() |> Transmuter.Featurizer.rgb()
      rgba = Nx.concatenate([pixels, Nx.new_axis(mask, 2)], axis: 2)
      StbImage.write_file!(StbImage.from_nx(rgba), "cats.png")

  ## Options

    * `:compile` - `:batch_size`. The default is `1`. The ONNX file of `onnx-community/ormbg-ONNX` has a fixed batch size of 1, so the batch size must be `1`.

  See the module documentation for the other options.
  """
  defdelegate background_removal(source, opts \\ []), to: Transmuter.Vision.BackgroundRemoval

  @doc """
  Builds a serving that gives the text of an image with an encoder-decoder model, such as a caption or handwritten text.

  The input is an image, `%{image: image, seed: seed}` or a list of them. The result is `%{results: [%{text: text}]}`, as in Bumblebee.

  The repository must have `onnx/encoder_model.onnx` and `onnx/decoder_model_merged.onnx`, as the `VisionEncoderDecoderModel` exports of Xenova, such as `Xenova/vit-gpt2-image-captioning` and `Xenova/trocr-small-handwritten`. The default variant is fp32. The fp16 decoder of `Xenova/vit-gpt2-image-captioning` is broken: its `If` branches give outputs that no node of the branch computes. ONNX Runtime also rejects it, and `Transmuter.load!/2` raises an `ArgumentError`. Use the fp32 or q8 files of that repository. The encoder reads `pixel_values` from `Transmuter.Featurizer`. The decoder starts from `decoder_start_token_id` and reads the output of the encoder in each step.

      serving = Transmuter.Vision.image_to_text({:hf, "Xenova/vit-gpt2-image-captioning"})
      Nx.Serving.run(serving, image)
      #=> %{results: [%{text: "a cat laying on a blanket next to a cat laying on a bed "}]}

  The settings of generation come from `generation_config.json`, as in `Transmuter.Text.generation/2`. The serving does not do beam search. It ignores `num_beams`.

  ## Options

    * `:max_new_tokens` - the maximum number of new tokens. The default is that of the `image-to-text` pipeline of transformers (see `Transmuter.Generation.max_new_tokens/3`): `max_new_tokens` of `generation_config.json`, else 256 when `max_length` is not in the configuration or is 20, else `max_length` minus 1 for the start token. So the default for ViT-GPT2 and TrOCR is 256 new tokens.

    * `:compile` - `:batch_size`, the number of images that generate together. The default is `1`.

    * `:strategy`, `:temperature`, `:top_k`, `:top_p`, `:repetition_penalty`, `:seed`, `:stream` and `:stream_done` - as in `Transmuter.Text.generation/2`.

    * `:tokenizer` - a tokenizer from `Transmuter.Tokenizer.load!/2`, to use in place of the tokenizer of the repository.

  The text needs the optional `tokenizers` dependency: add `{:tokenizers, "~> 0.5"}` to your deps. The other options are those of the module documentation, but `:model` does not apply, because the task has two models.
  """
  defdelegate image_to_text(source, opts \\ []), to: Transmuter.Vision.ImageToText

  @doc """
  Builds a serving that answers a chat about images with a decoder-only vision language model, such as SmolVLM or Qwen2-VL.

  The input is `%{messages: messages}` or a list of them. Each message has `:role` and `:content`, as in the chat format of Python transformers. The content is a list of parts: `%{type: "image", image: image}` and `%{type: "text", text: text}`. Give a list of parts also for a message without images, because the chat template of SmolVLM reads the type of the first part. The input can have `:seed`. The result is `%{results: [%{text: text, token_summary: summary}]}`, as in `Transmuter.Text.generation/2`.

      serving = Transmuter.Vision.image_text_to_text({:hf, "HuggingFaceTB/SmolVLM-256M-Instruct"})

      chat = %{
        messages: [
          %{role: "user", content: [%{type: "image", image: image}, %{type: "text", text: "Can you describe this image?"}]}
        ]
      }

      Nx.Serving.run(serving, chat)
      #=> %{results: [%{text: " In the image we can see a cat sleeping on a pink bed. ...", token_summary: %{...}}]}

  The repository must have `onnx/vision_encoder.onnx`, `onnx/embed_tokens.onnx` and `onnx/decoder_model_merged.onnx`, as the `ImageTextToText` exports of transformers.js. The default variants are q4 for the vision encoder and the decoder, and q8 for the embeddings. The `dtype` of `transformers.js_config` in `config.json` comes before these defaults, see `Transmuter.Repo`. The serving supports these values of `image_processor_type` in `preprocessor_config.json`:

    * `Idefics3ImageProcessor`, such as SmolVLM. `Transmuter.Vision.Idefics3Processor` splits each image into tiles of 512x512 pixels.

    * `Qwen2VLImageProcessor`, such as Qwen2-VL. `Transmuter.Vision.Qwen2VLProcessor` resizes each image to multiples of 28 pixels and splits it into patches of 14x14 pixels. The decoder takes the 3-D positions of M-RoPE. The configuration of `onnx-community/Qwen2-VL-2B-Instruct` picks the 8-bit vision encoder, which gives other features in Transmuter than in ONNX Runtime. For the text of ONNX Runtime, use `variant: %{vision_encoder: :fp32}`.

  The chat template renders each image as a placeholder, and the processor replaces it with the image tokens of the image. The vision encoder turns the pixel values of the images into features, and the features replace the embeddings of the image tokens in the prompt. Then the decoder generates as in `Transmuter.Text.generation/2`. A chat that ends with an assistant message continues that message.

  ## Options

    * `:compile` - `:batch_size`, the number of chats that generate together. The default is `1`. Chats go into one batch when the number of their tiles (SmolVLM) or patches (Qwen2-VL) rounds up to the same power of two. A chat can have up to 65536 tiles or patches. An image of SmolVLM gives up to 17 tiles, and an image of Qwen2-VL gives up to `max_pixels / 196` patches.

    * `:max_new_tokens`, `:strategy`, `:temperature`, `:top_k`, `:top_p`, `:repetition_penalty`, `:seed`, `:template_options`, `:stream` and `:stream_done` - as in `Transmuter.Text.generation/2`.

    * `:tokenizer` - a tokenizer from `Transmuter.Tokenizer.load!/2`, to use in place of the tokenizer of the repository.

  The serving needs the optional `tokenizers` and `otera` dependencies. The other options are those of the module documentation, but `:model` and `:featurizer` do not apply, because the task has three models and its own image processor.
  """
  defdelegate image_text_to_text(source, opts \\ []), to: Transmuter.Vision.ImageTextToText
end
