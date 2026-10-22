defmodule Transmuter.Vision.ImageSegmentation do
  @moduledoc """
  The image segmentation task. See `Transmuter.Vision.image_segmentation/2`.

  The featurizer gives one image size, so the images go into batches, and the model gives logits of one shape for all images. The segmentation has the size of the input image, so the client resizes the logits of each image on the `:backend` of the options, takes the label of each pixel and makes the masks.
  """

  alias Transmuter.{Featurizer, Serving}

  @doc false
  def image_segmentation(source, opts) do
    opts = Keyword.validate!(opts, Serving.image_options())
    {repo, model} = Serving.load_model!(source, opts, :fp32)
    semantic!(model)
    backend = opts[:backend] || Nx.default_backend()

    Serving.image(
      model,
      opts[:featurizer] || Featurizer.load!(repo),
      opts,
      fn outputs, _inputs -> %{logits: outputs["logits"]} end,
      fn [%{logits: logits}], [image] ->
        {height, width, _channels} = Nx.shape(image)
        %{segments: segments(repo, logits, {height, width}, backend)}
      end
    )
  end

  # The pipeline of Python transformers prefers the panoptic and the instance
  # post-processing when the image processor has them. These models, such as DETR and
  # Mask2Former, give mask outputs, and the semantic models give only logits.
  defp semantic!(model) do
    if "logits" not in model.outputs or Enum.any?(model.outputs, &(&1 =~ "mask")) do
      raise ArgumentError,
            "image_segmentation supports semantic segmentation models, such as Segformer, whose output is logits of shape {batch, labels, height, width}. " <>
              "Panoptic and instance segmentation are not supported yet. The model gives the outputs #{inspect(model.outputs)}"
    end
  end

  # As post_process_semantic_segmentation of Python transformers and the image-segmentation
  # pipeline: a bilinear resize of the logits to the image, the label with the highest
  # logit, and one mask of 0 and 255 for each label that is present, in the order of the
  # label ids.
  defp segments(repo, logits, size, backend) do
    labels = Nx.axis_size(logits, 0)

    map =
      logits
      |> Nx.backend_transfer(backend)
      |> Serving.interpolate(size, :bilinear)
      |> Nx.argmax(axis: 0)

    masks = Nx.equal(Nx.new_axis(map, 0), Nx.iota({labels, 1, 1}, backend: backend))

    present =
      for {1, label} <- Enum.with_index(Nx.to_flat_list(Nx.any(masks, axes: [1, 2]))), do: label

    masks =
      masks
      |> Nx.take(Nx.tensor(present, backend: backend))
      |> Nx.multiply(255)
      |> Nx.backend_transfer(Nx.BinaryBackend)

    for {label, i} <- Enum.with_index(present),
        do: %{label: Serving.label(repo, label), score: nil, mask: masks[i]}
  end
end
