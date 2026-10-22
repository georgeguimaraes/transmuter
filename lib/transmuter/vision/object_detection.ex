defmodule Transmuter.Vision.ObjectDetection do
  @moduledoc """
  The object detection task. See `Transmuter.Vision.object_detection/2`.

  `model_type` in `config.json` selects the scores of `post_process_object_detection` in Python transformers: a softmax for DETR and YOLOS, and a sigmoid with top-k for RT-DETR and RF-DETR. The serving computes the scores, the labels and the corners of the boxes on the device, with one row for each query or top-k pair, so the outputs have the same shape for all image sizes. The client keeps the rows above the threshold and multiplies the corners by the size of the input image.
  """

  alias Transmuter.{Featurizer, Hub, Serving}

  @methods %{
    "detr" => :softmax,
    "yolos" => :softmax,
    "rt_detr" => :sigmoid,
    "rt_detr_v2" => :sigmoid,
    "rf_detr" => :sigmoid
  }

  @doc false
  def object_detection(source, opts) do
    opts = Keyword.validate!(opts, Serving.image_options() ++ [:top_k, threshold: 0.5])
    {repo, model} = Serving.load_model!(source, opts, :fp32)
    method = method!(repo)
    {threshold, top_k} = {opts[:threshold], opts[:top_k]}

    Serving.image(
      model,
      opts[:featurizer] || featurizer(repo),
      opts,
      fn outputs, _inputs -> detections(method, outputs, top_k) end,
      fn [row], [image] ->
        {height, width, _channels} = Nx.shape(image)
        %{detections: keep(repo, row, threshold, {width, height})}
      end
    )
  end

  defp method!(repo) do
    model_type = repo.config["model_type"]

    @methods[model_type] ||
      raise ArgumentError,
            "object_detection supports the model types #{Enum.join(Enum.sort(Map.keys(@methods)), ", ")}, got: #{inspect(model_type)}"
  end

  # The RF-DETR exports of onnx-community set do_normalize to false in
  # preprocessor_config.json, but their graphs do not normalize the pixels. The RF-DETR
  # models of Python transformers take normalized pixels, with the image_mean and
  # image_std of the configuration.
  defp featurizer(repo = %{config: %{"model_type" => "rf_detr"}}) do
    config = Hub.json(repo.source, "preprocessor_config.json", repo.opts)
    Featurizer.new(Map.put(config, "do_normalize", true))
  end

  defp featurizer(repo), do: Featurizer.load!(repo)

  defp detections(:softmax, %{"logits" => logits, "pred_boxes" => boxes}, _top_k) do
    classes = Nx.axis_size(logits, 2)
    scores = logits |> Serving.softmax() |> Nx.slice_along_axis(0, classes - 1, axis: 2)

    %{
      scores: Nx.reduce_max(scores, axes: [2]),
      labels: Nx.argmax(scores, axis: 2),
      boxes: corners(boxes)
    }
  end

  defp detections(:sigmoid, %{"logits" => logits, "pred_boxes" => boxes}, top_k) do
    {batch, queries, classes} = Nx.shape(logits)
    k = min(top_k || queries, queries * classes)

    {scores, index} =
      logits |> Nx.sigmoid() |> Nx.reshape({batch, queries * classes}) |> Nx.top_k(k: k)

    query = index |> Nx.quotient(classes) |> Nx.new_axis(2) |> Nx.broadcast({batch, k, 4})

    %{
      scores: scores,
      labels: Nx.remainder(index, classes),
      boxes: Nx.take_along_axis(corners(boxes), query, axis: 1)
    }
  end

  @doc """
  Turns boxes of shape `{batch, boxes, 4}` from `(center_x, center_y, width, height)` into `(xmin, ymin, xmax, ymax)`, as `center_to_corners_format` of Python transformers.
  """
  def corners(boxes) do
    [center_x, center_y, width, height] =
      for i <- 0..3, do: Nx.slice_along_axis(boxes, i, 1, axis: 2)

    half_width = Nx.multiply(width, 0.5)
    half_height = Nx.multiply(height, 0.5)

    Nx.concatenate(
      [
        Nx.subtract(center_x, half_width),
        Nx.subtract(center_y, half_height),
        Nx.add(center_x, half_width),
        Nx.add(center_y, half_height)
      ],
      axis: 2
    )
  end

  defp keep(repo, %{scores: scores, labels: labels, boxes: boxes}, threshold, {width, height}) do
    [scores, labels, boxes]
    |> Enum.map(&Nx.to_list/1)
    |> Enum.zip()
    |> Enum.filter(fn {score, _label, _box} -> score > threshold end)
    |> Enum.map(fn {score, label, [xmin, ymin, xmax, ymax]} ->
      %{
        label: Serving.label(repo, label),
        score: score,
        box: %{xmin: xmin * width, ymin: ymin * height, xmax: xmax * width, ymax: ymax * height}
      }
    end)
  end
end
