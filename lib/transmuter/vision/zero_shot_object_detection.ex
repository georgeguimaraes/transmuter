defmodule Transmuter.Vision.ZeroShotObjectDetection do
  @moduledoc """
  The zero-shot object detection task. See `Transmuter.Vision.zero_shot_object_detection/3`.

  The `zero-shot-object-detection` pipeline of Python transformers runs the model once for each label and keeps the boxes whose score is higher than the threshold. `model_type` in `config.json` selects how the serving runs the model:

    * `"owlvit"` and `"owlv2"` - each text query gets its own score for each box, and the queries do not change the boxes. So the labels are the text queries of one model run, and the images go into batches through `Transmuter.Serving.image/6`.

    * `"grounding-dino"` - the text tokens and the image go through the same layers, so the boxes depend on the text. Each image is a batch of its own (`Transmuter.Serving.each/2`), and the model runs once for each label.

  The serving gives the scores and the corners of the boxes of each label, and the client keeps the boxes above the threshold.
  """

  alias Transmuter.{Featurizer, Serving, Tokenizer}
  alias Transmuter.Vision.ObjectDetection

  @doc false
  def zero_shot_object_detection(source, labels, opts) do
    opts =
      Keyword.validate!(opts, Serving.image_options() ++ [:tokenizer, :top_k, threshold: 0.1])

    {repo, model} =
      Serving.load_model!(source, [outputs: ["logits", "pred_boxes"]] ++ opts, :fp32)

    tokenizer = Serving.tokenizer(repo, opts)
    featurizer = opts[:featurizer] || Featurizer.load!(repo)

    case repo.config["model_type"] do
      type when type in ["owlvit", "owlv2"] ->
        owl(model, tokenizer, featurizer, labels, opts)

      "grounding-dino" ->
        grounding_dino(model, tokenizer, featurizer, labels, opts)

      type ->
        raise ArgumentError,
              "zero_shot_object_detection supports the model types grounding-dino, owlv2 and owlvit, got: #{inspect(type)}"
    end
  end

  defp owl(model, tokenizer, featurizer, labels, opts) do
    # The model reshapes the text queries to {batch, queries, length}, so each image of a
    # batch gets the queries of all labels.
    batch_size = Keyword.get(opts[:compile] || [], :batch_size, 1)
    {text, _encodings} = Tokenizer.encode(tokenizer, labels)
    text = Map.new(text, fn {name, tensor} -> {name, Nx.tile(tensor, [batch_size, 1])} end)

    Serving.image(
      model,
      featurizer,
      opts,
      fn %{"logits" => logits, "pred_boxes" => boxes}, _inputs ->
        {batch, patches, queries} = Nx.shape(logits)

        %{
          scores: logits |> Nx.sigmoid() |> Nx.transpose(axes: [0, 2, 1]),
          boxes:
            boxes
            |> ObjectDetection.corners()
            |> Nx.new_axis(1)
            |> Nx.broadcast({batch, queries, patches, 4})
        }
      end,
      fn [row], [image] ->
        {height, width, _channels} = Nx.shape(image)
        %{detections: detections(row, labels, scale(featurizer, {height, width}), opts)}
      end,
      text
    )
  end

  # OWLv2 pads the image to a square, so the boxes are relative to the side of the square,
  # as _scale_boxes of Owlv2ImageProcessor.
  defp scale(%Featurizer{pad: nil}, {height, width}), do: {width, height}
  defp scale(_featurizer, {height, width}), do: {max(height, width), max(height, width)}

  defp grounding_dino(model, tokenizer, featurizer, labels, opts) do
    if Keyword.get(opts[:compile] || [], :batch_size, 1) != 1,
      do:
        raise(
          ArgumentError,
          "Grounding DINO runs the model once for each label, so each image is a batch of its own. Use batch_size: 1"
        )

    texts = Enum.map(labels, &grounding_dino_text(tokenizer, &1))
    lengths = texts |> Enum.map(&Nx.axis_size(&1["input_ids"], 1)) |> Enum.uniq() |> length()
    backend = opts[:backend] || Nx.default_backend()

    preprocess = fn image ->
      {height, width, _channels} = Nx.shape(image)

      pixels =
        featurizer
        |> Featurizer.process_on_host([image], backend: backend)

      %{
        "pixel_values" => pixels[0],
        "size" => Nx.tensor([height, width], backend: Nx.BinaryBackend)
      }
    end

    init = fn defn_options ->
      {height, width} = Serving.compile_size(featurizer)
      zero = Nx.tensor(0, type: :f32, backend: Nx.BinaryBackend)
      pixels = Nx.broadcast(zero, {1, 3, height, width})
      template = Map.merge(Serving.image_inputs(model, pixels), hd(texts))
      run = Transmuter.compile(model, template, [cache_size: lengths] ++ defn_options)

      fn %{"pixel_values" => pixels, "size" => size} ->
        inputs = Serving.image_inputs(model, pixels)
        outputs = Enum.map(texts, &run.(Map.merge(inputs, &1)))

        # As post_process_object_detection of GroundingDinoImageProcessor: the score of a
        # box is the sigmoid of its highest logit over the text tokens.
        scores = for %{"logits" => logits} <- outputs, do: Nx.reduce_max(logits, axes: [2])
        boxes = for %{"pred_boxes" => boxes} <- outputs, do: ObjectDetection.corners(boxes)

        %{
          "scores" => scores |> Nx.stack(axis: 1) |> Nx.sigmoid(),
          "boxes" => Nx.stack(boxes, axis: 1),
          "size" => size
        }
      end
    end

    Serving.each(
      %{
        validate: &Featurizer.tensor/1,
        preprocess: preprocess,
        init: init,
        postprocess: fn %{"scores" => scores, "boxes" => boxes, "size" => size} ->
          [height, width] = Nx.to_flat_list(size)
          row = %{scores: scores, boxes: boxes}
          %{detections: detections(row, labels, {width, height}, opts)}
        end
      },
      opts
    )
  end

  # The text of one label as _merge_candidate_labels_text of GroundingDinoProcessor in Python
  # transformers makes it: the label without spaces at the ends, in lowercase, and a period.
  # The model reads the tokens between two special tokens, such as the period, as one
  # phrase. The ONNX exports compute the phrases for a text of one phrase and a period, so a
  # label cannot have more of them. The pipeline with PyTorch tokenizes the label alone.
  defp grounding_dino_text(tokenizer, label) do
    if String.contains?(label, [".", "?"]) do
      raise ArgumentError,
            ~s(a label of Grounding DINO cannot have "." or "?", because these characters end a phrase, got: #{inspect(label)})
    end

    text = (label |> String.trim() |> String.downcase()) <> "."
    {inputs, _encodings} = Tokenizer.encode(tokenizer, [text])
    inputs
  end

  # As the postprocess of the zero-shot-object-detection pipeline: the boxes of each label
  # whose score is higher than the threshold, from the highest score. The sort is stable,
  # so equal scores stay in the order of the labels and the boxes, as in Python.
  defp detections(%{scores: scores, boxes: boxes}, labels, {scale_x, scale_y}, opts) do
    detections =
      for {label, label_scores, label_boxes} <-
            Enum.zip([labels, Nx.to_list(scores), Nx.to_list(boxes)]),
          {score, [xmin, ymin, xmax, ymax]} <- Enum.zip(label_scores, label_boxes),
          score > opts[:threshold] do
        box = %{
          xmin: xmin * scale_x,
          ymin: ymin * scale_y,
          xmax: xmax * scale_x,
          ymax: ymax * scale_y
        }

        %{label: label, score: score, box: box}
      end

    detections = Enum.sort_by(detections, & &1.score, :desc)
    if opts[:top_k], do: Enum.take(detections, opts[:top_k]), else: detections
  end
end
