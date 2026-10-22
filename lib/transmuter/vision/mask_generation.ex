defmodule Transmuter.Vision.MaskGeneration do
  @moduledoc """
  The mask generation task with prompts, such as SAM. See `Transmuter.Vision.mask_generation/2`.

  The repository has two models, as the SAM exports of transformers.js: `vision_encoder` turns the pixel values into image embeddings, and `prompt_encoder_mask_decoder` turns the embeddings and one prompt into masks. The masks have the size of the image, so each image is a batch of its own (`Transmuter.Serving.each/2`). The encoder runs once for each image, and the decoder runs once for each prompt.

  The client puts the prompts of an image into tensors: the points `{prompts, points, 2}` and the labels `{prompts, points}`, with the label -10 after the points of a shorter prompt, and the boxes `{prompts, 4}` with a flag for each prompt that has a box. The serving gives the decoder the points of each prompt without the padding, because the decoder of SAM attends also to a padding point, so the padding changes the masks. The decoder compiles once for each number of points with or without a box.
  """

  alias Transmuter.{Featurizer, Hub, Repo, Serving}

  # The compiled decoder keeps this many prompt shapes.
  @prompt_shapes 32

  # The label and the coordinates of a padding point, as point_pad_value of the SAM
  # processors of Python transformers.
  @padding -10

  @doc false
  def mask_generation(source, opts) do
    opts = Keyword.validate!(opts, Serving.image_options() -- [:model])
    batch_size!(opts[:compile])
    repo = Repo.load!(source, Keyword.take(opts, [:variant | Hub.options()]))
    model_opts = Keyword.take(opts, [:backend, :compute_type])
    encoder = Repo.load_component!(repo, "vision_encoder", :fp32, model_opts)
    decoder_opts = [outputs: ["iou_scores", "pred_masks"]] ++ model_opts
    decoder = Repo.load_component!(repo, "prompt_encoder_mask_decoder", :fp32, decoder_opts)
    featurizer = opts[:featurizer] || Featurizer.load!(repo)
    boxes? = "input_boxes" in decoder.inputs
    backend = opts[:backend] || Nx.default_backend()

    preprocess = fn %{image: image, prompts: prompts} ->
      {height, width, _channels} = Nx.shape(image)
      {resized_height, resized_width} = Featurizer.image_size(featurizer, {height, width})

      pixels =
        featurizer
        |> Featurizer.process([image], backend: backend)
        |> Nx.backend_transfer(Nx.BinaryBackend)

      prompts
      |> prompt_tensors({resized_width / width, resized_height / height})
      |> Map.merge(%{
        "pixel_values" => pixels[0],
        "size" =>
          Nx.tensor([height, width, resized_height, resized_width], backend: Nx.BinaryBackend)
      })
    end

    init = fn defn_options ->
      {height, width} = Featurizer.size(featurizer)
      pixels = Nx.broadcast(Nx.tensor(0.0, backend: Nx.BinaryBackend), {1, 3, height, width})
      encode = Transmuter.compile(encoder, %{"pixel_values" => pixels}, defn_options)

      # The template of the decoder has the embeddings of an image and a prompt of one point.
      [prompt] =
        [%{points: [[0, 0]], labels: [1], box: nil}]
        |> prompt_tensors({1, 1})
        |> Map.new(fn {name, tensor} -> {name, Nx.new_axis(tensor, 0)} end)
        |> prompts(boxes?)

      embeddings = Map.take(encode.(%{"pixel_values" => pixels}), decoder.inputs)
      compile_options = [cache_size: @prompt_shapes] ++ defn_options
      decode = Transmuter.compile(decoder, Map.merge(embeddings, prompt), compile_options)

      fn inputs ->
        [height, width, resized_height, resized_width] = Nx.to_flat_list(inputs["size"])

        embeddings =
          Map.take(encode.(%{"pixel_values" => inputs["pixel_values"]}), decoder.inputs)

        {masks, scores} =
          inputs
          |> prompts(boxes?)
          |> Enum.map(fn prompt ->
            %{"iou_scores" => scores, "pred_masks" => masks} =
              decode.(Map.merge(embeddings, prompt))

            resized = {resized_height, resized_width}
            {masks(masks[0][0], featurizer.pad, resized, {height, width}), scores[0][0]}
          end)
          |> Enum.unzip()

        %{
          "masks" => masks |> Nx.stack() |> Nx.new_axis(0),
          "iou_scores" => scores |> Nx.stack() |> Nx.new_axis(0)
        }
      end
    end

    Serving.each(
      %{
        validate: &validate(&1, boxes?),
        preprocess: preprocess,
        init: init,
        postprocess: fn %{"masks" => masks, "iou_scores" => scores} ->
          {prompts, count, _height, _width} = Nx.shape(masks)

          %{
            masks: for(i <- 0..(prompts - 1), do: for(j <- 0..(count - 1), do: masks[i][j])),
            iou_scores: Nx.to_list(scores)
          }
        end
      },
      opts
    )
  end

  defp batch_size!(compile) do
    if Keyword.validate!(compile || [], batch_size: 1)[:batch_size] != 1,
      do:
        raise(
          ArgumentError,
          "the masks have the size of the image, so each image is a batch of its own. Use batch_size: 1"
        )
  end

  defp validate(%{image: image, prompts: prompts = [_ | _]}, boxes?) do
    with {:ok, image} <- Featurizer.tensor(image),
         {:ok, prompts} <- valid_prompts(prompts, boxes?, []),
         do: {:ok, %{image: image, prompts: prompts}}
  end

  defp validate(input, _boxes?),
    do:
      {:error,
       "expected %{image: image, prompts: prompts} with a list of one or more prompts, got: #{inspect(input)}"}

  defp valid_prompts([], _boxes?, valid), do: {:ok, Enum.reverse(valid)}

  defp valid_prompts([prompt | rest], boxes?, valid) do
    with {:ok, prompt} <- prompt(prompt, boxes?),
         do: valid_prompts(rest, boxes?, [prompt | valid])
  end

  defp prompt(prompt = %{}, boxes?) do
    points = Map.get(prompt, :points, [])
    labels = Map.get(prompt, :labels, if(is_list(points), do: List.duplicate(1, length(points))))
    box = Map.get(prompt, :box)

    cond do
      not (is_list(points) and Enum.all?(points, &numbers?(&1, 2))) ->
        {:error, "the points of a prompt must be a list of [x, y], got: #{inspect(points)}"}

      not (is_list(labels) and Enum.all?(labels, &(&1 in [0, 1]))) ->
        {:error, "the labels of a prompt must be a list of 0 and 1, got: #{inspect(labels)}"}

      length(labels) != length(points) ->
        {:error, "a prompt must have one label for each point, got: #{inspect(prompt)}"}

      points == [] and box == nil ->
        {:error, "a prompt must have points or a box, got: #{inspect(prompt)}"}

      box != nil and not boxes? ->
        {:error, "the decoder of this model has no input_boxes, so a prompt cannot have a box"}

      box != nil and not numbers?(box, 4) ->
        {:error, "the box of a prompt must be [x1, y1, x2, y2], got: #{inspect(box)}"}

      true ->
        {:ok, %{points: points, labels: labels, box: box}}
    end
  end

  defp prompt(prompt, _boxes?),
    do: {:error, "expected a prompt map with :points, :labels or :box, got: #{inspect(prompt)}"}

  defp numbers?(list, count),
    do: is_list(list) and length(list) == count and Enum.all?(list, &is_number/1)

  # As the SAM processors of Python transformers, the points and the corners of the boxes
  # are scaled to the resized image.
  defp prompt_tensors(prompts, {scale_x, scale_y}) do
    width = prompts |> Enum.map(&length(&1.points)) |> Enum.max() |> max(1)
    pad = &(&1 ++ List.duplicate(&2, width - length(&1)))
    scale = fn points -> for [x, y] <- points, do: [x * scale_x, y * scale_y] end
    box = fn box -> if box, do: box |> Enum.chunk_every(2) |> scale.() |> Enum.concat() end
    tensor = &Nx.tensor(&1, type: &2, backend: Nx.BinaryBackend)

    %{
      "points" =>
        tensor.(for(p <- prompts, do: pad.(scale.(p.points), [@padding, @padding])), :f32),
      "labels" => tensor.(for(p <- prompts, do: pad.(p.labels, @padding)), :s64),
      "boxes" => tensor.(for(p <- prompts, do: box.(p.box) || [0, 0, 0, 0]), :f32),
      "has_box" => tensor.(for(p <- prompts, do: if(p.box, do: 1, else: 0)), :u8)
    }
  end

  # The prompt inputs of the decoder for each prompt, from the tensors of the client with a
  # batch axis of size 1. A prompt without points or without a box gives a zero-size input
  # to a decoder that has input_boxes, as SAM 2.1.
  defp prompts(inputs, boxes?) do
    has_box = Nx.to_flat_list(inputs["has_box"])

    for {labels, i} <- Enum.with_index(Nx.to_list(inputs["labels"][0])) do
      count = Enum.count(labels, &(&1 != @padding))

      points =
        if count > 0,
          do: %{
            "input_points" => Nx.slice(inputs["points"], [0, i, 0, 0], [1, 1, count, 2]),
            "input_labels" => Nx.slice(inputs["labels"], [0, i, 0], [1, 1, count])
          },
          else: %{"input_points" => {:empty, {1, 1, 0, 2}}, "input_labels" => {:empty, {1, 1, 0}}}

      box =
        cond do
          not boxes? ->
            %{}

          Enum.at(has_box, i) == 1 ->
            %{"input_boxes" => Nx.slice(inputs["boxes"], [0, i, 0], [1, 1, 4])}

          true ->
            %{"input_boxes" => {:empty, {1, 0, 4}}}
        end

      Map.merge(points, box)
    end
  end

  # As post_process_masks of the SAM image processors of Python transformers: a bilinear
  # resize of the masks to the image, and the pixels above 0. SamImageProcessor pads the
  # image to pad_size, so its masks first get that size, lose the padding, and then get the
  # size of the image.
  defp masks(masks, {{_, _} = padded, _value}, {height, width}, size) do
    count = Nx.axis_size(masks, 0)

    masks
    |> Serving.interpolate(padded, :bilinear)
    |> Nx.slice([0, 0, 0], [count, height, width])
    |> masks(nil, nil, size)
  end

  defp masks(masks, _pad, _resized, size) do
    masks
    |> Serving.interpolate(size, :bilinear)
    |> Nx.greater(0)
    |> Nx.multiply(255)
  end
end
