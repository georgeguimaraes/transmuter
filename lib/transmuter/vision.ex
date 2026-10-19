defmodule Transmuter.Vision do
  @moduledoc """
  Servings for image tasks. Each function loads a model repository (the ONNX file, `preprocessor_config.json` and the configuration) and returns an `Nx.Serving` that takes images and gives results in the shapes of Bumblebee.

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
end
