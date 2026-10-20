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

    * `:compile` - `:batch_size`. The default is `1`. The serving compiles the model for one image size, from `preprocessor_config.json`.

    * `:defn_options` - options of `Transmuter.compile/3`, such as `compiler: EMLX`.

    * `:backend` and `:compute_type` - options of `Transmuter.load!/2`, such as `backend: EMLX.Backend`. The featurizer also uses `:backend` to resize the images. Without it, the featurizer uses `Nx.default_backend/0`.

    * `:variant` - the variant of the ONNX file, such as `:q4`. See `Transmuter.Repo`.

    * `:cache_dir` - see `Transmuter.Hub`.

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
  Builds a serving that gives the text of an image with an encoder-decoder model, such as a caption or handwritten text.

  The input is an image, `%{image: image, seed: seed}` or a list of them. The result is `%{results: [%{text: text}]}`, as in Bumblebee.

  The repository must have `onnx/encoder_model.onnx` and `onnx/decoder_model_merged.onnx`, as the `VisionEncoderDecoderModel` exports of Xenova, such as `Xenova/vit-gpt2-image-captioning` and `Xenova/trocr-small-handwritten`. The default variant is fp32. The encoder reads `pixel_values` from `Transmuter.Featurizer`. The decoder starts from `decoder_start_token_id` and reads the output of the encoder in each step.

      serving = Transmuter.Vision.image_to_text({:hf, "Xenova/vit-gpt2-image-captioning"})
      Nx.Serving.run(serving, image)
      #=> %{results: [%{text: "a cat laying on a blanket next to a cat laying on a bed "}]}

  The settings of generation come from `generation_config.json`, as in `Transmuter.Text.generation/2`. The serving does not do beam search. It ignores `num_beams`.

  ## Options

    * `:max_new_tokens` - the maximum number of new tokens. The default is `max_new_tokens` of `generation_config.json`, or else `max_length` minus 1 for the start token. When the configuration has no `max_length`, the value is 20, as in Python transformers. So the default for ViT-GPT2 and TrOCR is 19 new tokens.

    * `:compile` - `:batch_size`, the number of images that generate together. The default is `1`.

    * `:strategy`, `:temperature`, `:top_k`, `:top_p`, `:repetition_penalty`, `:seed`, `:stream` and `:stream_done` - as in `Transmuter.Text.generation/2`.

    * `:tokenizer` - a tokenizer from `Transmuter.Tokenizer.load!/2`, to use in place of the tokenizer of the repository.

  The text needs the optional `tokenizers` dependency: add `{:tokenizers, "~> 0.5"}` to your deps. The other options are those of the module documentation, but `:model` does not apply, because the task has two models.
  """
  defdelegate image_to_text(source, opts \\ []), to: Transmuter.Vision.ImageToText

  @doc """
  Builds a serving that answers a chat about images with a decoder-only vision language model, such as SmolVLM.

  The input is `%{messages: messages}` or a list of them. Each message has `:role` and `:content`, as in the chat format of Python transformers. The content is a list of parts: `%{type: "image", image: image}` and `%{type: "text", text: text}`. Give a list of parts also for a message without images, because the chat template of SmolVLM reads the type of the first part. The input can have `:seed`. The result is `%{results: [%{text: text, token_summary: summary}]}`, as in `Transmuter.Text.generation/2`.

      serving = Transmuter.Vision.image_text_to_text({:hf, "HuggingFaceTB/SmolVLM-256M-Instruct"})

      chat = %{
        messages: [
          %{role: "user", content: [%{type: "image", image: image}, %{type: "text", text: "Can you describe this image?"}]}
        ]
      }

      Nx.Serving.run(serving, chat)
      #=> %{results: [%{text: " In the image we can see a cat sleeping on a pink bed. ...", token_summary: %{...}}]}

  The repository must have `onnx/vision_encoder.onnx`, `onnx/embed_tokens.onnx` and `onnx/decoder_model_merged.onnx`, as the `ImageTextToText` exports of transformers.js. The default variants are q4 for the vision encoder and the decoder, and q8 for the embeddings. The serving supports models with `Idefics3ImageProcessor` in `preprocessor_config.json`, such as SmolVLM.

  The chat template renders each image as a placeholder. `Transmuter.Vision.Idefics3Processor` splits each image into tiles of 512x512 pixels and replaces its placeholder with the image tokens of the tiles. The vision encoder turns the tiles into features, and the features replace the embeddings of the image tokens in the prompt. Then the decoder generates as in `Transmuter.Text.generation/2`. A chat that ends with an assistant message continues that message.

  ## Options

    * `:compile` - `:batch_size`, the number of chats that generate together. The default is `1`. Chats go into one batch when the number of their tiles rounds up to the same power of two. A chat can have up to 64 tiles, and an image of SmolVLM gives up to 17 tiles.

    * `:max_new_tokens`, `:strategy`, `:temperature`, `:top_k`, `:top_p`, `:repetition_penalty`, `:seed`, `:template_options`, `:stream` and `:stream_done` - as in `Transmuter.Text.generation/2`.

    * `:tokenizer` - a tokenizer from `Transmuter.Tokenizer.load!/2`, to use in place of the tokenizer of the repository.

  The serving needs the optional `tokenizers` and `otera` dependencies. The other options are those of the module documentation, but `:model` and `:featurizer` do not apply, because the task has three models and its own image processor.
  """
  defdelegate image_text_to_text(source, opts \\ []), to: Transmuter.Vision.ImageTextToText
end
