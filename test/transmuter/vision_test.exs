defmodule Transmuter.VisionTest do
  use ExUnit.Case, async: true

  # Reference values come from Python transformers, see test/support/pipeline_references.py.
  # The first run downloads the models.
  @moduletag :hub
  @moduletag timeout: 900_000
  @moduletag ci_partition: 16

  @opts [
    backend: Transmuter.TestBackend.backend(),
    defn_options: Transmuter.TestBackend.compiler_opts()
  ]

  alias Transmuter.{ChatTemplate, Featurizer, Tokenizer, Vision}
  alias Transmuter.Vision.{Florence2Processor, Idefics3Processor, Qwen2VLProcessor}

  @image StbImage.read_file!(Path.expand("../fixtures/cats.png", __DIR__))
  @labels ["cats", "dogs", "a remote control", "a couch"]
  @detection_labels ["cat", "remote control", "couch"]

  @tag ci_partition: 17
  test "the featurizer gives the pixel values of the image processors" do
    # ViT resizes to 224x224 with bilinear. CLIP resizes the short side to 224 with bicubic
    # and crops the center. DPT resizes to multiples of 14 near 518, with bicubic. Segformer
    # resizes to 512x512 with bilinear. OWL-ViT resizes to 768x768 with bicubic. OWLv2 pads
    # the image to a square and resizes it to 960x960 with the resize of scikit-image, which
    # blurs the 6x6 tile of the image (1920x1440) with sigma 0.5 before it interpolates.
    # Grounding DINO with shortest_edge 800 resizes as DETR. The 259x100 crop of the image
    # gets 515x1333, with the short side rounded half to even. The rule of shortest_edge
    # alone gives 514x1333. DETR pads to its pad_size with a pixel mask, so the featurizer
    # gives no padding for DETR. SAM resizes the long side to 1024 and pads to 1024x1024 with
    # zeros. SAM 2.1 resizes to 1024x1024 with torchvision, whose 8-bit weights have fewer
    # fraction bits than those of Pillow. On the tile, 27177 values of Pillow are one step
    # away.
    tile = Nx.tile(StbImage.to_nx(@image), [6, 6, 1])
    crop = Nx.slice(StbImage.to_nx(@image), [0, 0, 0], [100, 259, 3])

    grounding_dino =
      Featurizer.new(%{
        "image_processor_type" => "GroundingDinoImageProcessor",
        "size" => %{"shortest_edge" => 800, "longest_edge" => 1333},
        "image_mean" => [0.485, 0.456, 0.406],
        "image_std" => [0.229, 0.224, 0.225]
      })

    detr =
      Featurizer.new(%{
        "image_processor_type" => "DetrImageProcessor",
        "size" => %{"shortest_edge" => 800, "longest_edge" => 1333},
        "do_pad" => true,
        "pad_size" => %{"height" => 1333, "width" => 1333},
        "image_mean" => [0.485, 0.456, 0.406],
        "image_std" => [0.229, 0.224, 0.225]
      })

    for {featurizer, image, {height, width}, expected, mean} <- [
          {"Xenova/vit-base-patch16-224", @image, {224, 224},
           [0.1137, 0.2941, -0.5608, 0.4902, -0.2863], 0.4666},
          {"Xenova/clip-vit-base-patch32", @image, {224, 224},
           [0.6165, -0.8516, 0.8519, -0.639, -0.4314], 0.8819},
          {"onnx-community/depth-anything-v2-small", @image, {392, 518},
           [0.2967, -1.3179, 1.1934, 1.4269, 0.9055], 1.0595},
          {"Xenova/segformer-b0-finetuned-ade-512-512", @image, {512, 512},
           [0.2967, 1.6758, -1.8044, -0.902, 0.8704], 1.0497},
          {"Xenova/owlvit-base-patch32", @image, {768, 768},
           [0.2661, -1.617, 0.3399, 0.1201, 1.6547], 0.8948},
          {"Xenova/owlv2-base-patch16-ensemble", @image, {960, 960},
           [0.2986, -1.527, 1.3496, 0.198, 1.4762], 0.7126},
          {"Xenova/owlv2-base-patch16-ensemble", tile, {960, 960},
           [0.3245, -0.2768, 0.8887, 0.9414, -0.7501], 0.7012},
          {grounding_dino, crop, {515, 1333}, [0.2967, -1.9132, 2.1868, 0.6734, -1.8256], 0.9977},
          {detr, crop, {515, 1333}, [0.2967, -1.9132, 2.1868, 0.6734, -1.8256], 0.9977},
          {"Xenova/slimsam-77-uniform", @image, {1024, 1024},
           [0.2967, -1.8782, 1.3328, 0.2624, 1.063], 0.7873},
          {"onnx-community/sam2.1-hiera-tiny-ONNX", @image, {1024, 1024},
           [0.2967, -2.0357, -1.5256, 0.6392, -1.6856], 1.0497},
          {"onnx-community/sam2.1-hiera-tiny-ONNX", tile, {1024, 1024},
           [0.3652, 0.8704, -0.2532, 1.6324, -0.3025], 1.0332}
        ] do
      featurizer =
        if is_binary(featurizer), do: Featurizer.load!({:hf, featurizer}), else: featurizer

      values = Featurizer.process(featurizer, [image], backend: @opts[:backend])
      values = Nx.backend_transfer(values, Nx.BinaryBackend)

      assert Nx.shape(values) == {1, 3, height, width}

      got =
        for [c, y, x] <- [[0, 0, 0], [1, 100, 37], [2, 223, 223], [0, 57, 190], [1, 150, 120]],
            do: Nx.to_number(values[[0, c, y, x]])

      assert_all_close(got, expected)
      assert_in_delta Nx.to_number(Nx.mean(Nx.abs(values))), mean, 1.0e-4
    end

    # The 8-bit image of the tile after the resize, from the pixel values of SAM 2.1.
    featurizer = Featurizer.load!({:hf, "onnx-community/sam2.1-hiera-tiny-ONNX"})
    {mean, std} = featurizer.normalize

    resized =
      featurizer
      |> Featurizer.process([tile], backend: @opts[:backend])
      |> Nx.backend_transfer(Nx.BinaryBackend)
      |> Nx.multiply(Nx.reshape(Nx.tensor(std), {1, 3, 1, 1}))
      |> Nx.add(Nx.reshape(Nx.tensor(mean), {1, 3, 1, 1}))
      |> Nx.multiply(255)
      |> Nx.round()
      |> Nx.as_type(:u8)

    assert Base.encode16(:crypto.hash(:sha256, Nx.to_binary(resized)), case: :lower) ==
             "f88aa51e6de09c5b1eb3810433b6583ec3933dc6e49938abf71d0bef2f1f0289"
  end

  @tag :smoke
  @tag ci_partition: 17
  test "image classification matches the pipeline" do
    for {repo, expected} <- [
          {"Xenova/vit-base-patch16-224",
           [
             {"Egyptian cat", 0.940545},
             {"tabby, tabby cat", 0.036794},
             {"tiger cat", 0.013316},
             {"lynx, catamount", 0.003089},
             {"Siamese cat, Siamese", 0.000671}
           ]},
          # ConvNeXt resizes the short side to 224 / crop_pct, 256, and crops 224x224.
          {"Xenova/convnext-tiny-224",
           [
             {"tabby, tabby cat", 0.555257},
             {"tiger cat", 0.302854},
             {"Egyptian cat", 0.008957},
             {"pillow", 0.004712},
             {"quilt, comforter, comfort, puff", 0.003517}
           ]}
        ] do
      serving = Vision.image_classification({:hf, repo}, @opts)
      # A tensor and a StbImage give the same result.
      [image, tensor] = Nx.Serving.run(serving, [@image, StbImage.to_nx(@image)])
      assert image == tensor
      assert_predictions(image, expected)
    end
  end

  @tag ci_partition: 17
  test "an image embedding is the projected embedding or the first token" do
    for {repo, expected, norm} <- [
          {"Xenova/clip-vit-base-patch32", [-0.0767, 0.1363, -0.2939, 0.0325], 10.8946},
          {"Xenova/dinov2-small", [-2.5502, -2.1046, -3.4216, -2.483], 49.1476}
        ] do
      serving = Vision.image_embedding({:hf, repo}, @opts)
      %{embedding: embedding} = Nx.Serving.run(serving, @image)
      embedding = Nx.backend_transfer(embedding, Nx.BinaryBackend)
      assert_all_close(Nx.to_list(embedding[0..3]), expected)
      assert_in_delta Nx.to_number(Nx.LinAlg.norm(embedding)), norm, 1.0e-3
    end
  end

  @tag ci_partition: 17
  test "zero-shot classification is a softmax for CLIP and a sigmoid for SigLIP" do
    for {repo, expected, sum} <- [
          {"Xenova/clip-vit-base-patch32",
           [
             {"cats", 0.984218},
             {"a remote control", 0.009486},
             {"a couch", 0.003834},
             {"dogs", 0.002462}
           ], 1.0},
          # SigLIP removes the punctuation of the text and pads it to 64 tokens.
          {"Xenova/siglip-base-patch16-224",
           [
             {"cats", 0.052535},
             {"a couch", 0.0001},
             {"dogs", 2.7e-05},
             {"a remote control", 1.5e-05}
           ], 0.0527}
        ] do
      serving = Vision.zero_shot_image_classification({:hf, repo}, @labels, @opts)
      result = Nx.Serving.run(serving, @image)
      assert_predictions(result, expected)
      assert_in_delta Enum.sum(Enum.map(result.predictions, & &1.score)), sum, 1.0e-4
    end
  end

  @handwriting StbImage.read_file!(Path.expand("../fixtures/handwriting.png", __DIR__))
  @trocr {:hf, "Xenova/trocr-small-handwritten"}

  # The reference is generate of the PyTorch models. The 8-bit TrOCR gives the text of
  # PyTorch. The 8-bit ViT-GPT2 gives another caption, also in ONNX Runtime, so its test
  # uses fp32. The tokenizer config of the TrOCR export has clean_up_tokenization_spaces
  # true, so the text has "noon." and not "noon .".
  @tag ci_partition: 15
  test "image to text matches generate of transformers" do
    for {source, variant, image, text} <- [
          {{:hf, "Xenova/vit-gpt2-image-captioning"}, :fp32, @image,
           "a cat laying on a blanket next to a cat laying on a bed "},
          {@trocr, :q8, @handwriting, "Meet me at the station at noon."}
        ] do
      serving = Vision.image_to_text(source, [variant: variant] ++ @opts)
      # A tensor and a StbImage give the same result.
      results = Nx.Serving.run(serving, [image, StbImage.to_nx(image)])
      assert results == List.duplicate(%{results: [%{text: text}]}, 2)
    end
  end

  @handwriting_long StbImage.read_file!(Path.expand("../fixtures/handwriting_long.png", __DIR__))

  # The rows of a batch stop at the end token at different steps, and the second row has more
  # than the 20 new tokens of generate in transformers. Both rows are clear text, so the
  # rounding of a batch does not change a token.
  @tag ci_partition: 2
  test "a batch of images gives the text of each image alone, and a stream joins to it" do
    opts = [variant: :q8] ++ @opts
    single = Vision.image_to_text(@trocr, opts)
    alone = for image <- [@handwriting, @handwriting_long], do: Nx.Serving.run(single, image)

    assert Enum.map(alone, fn %{results: [%{text: text}]} -> text end) == [
             "Meet me at the station at noon.",
             "We walked along the river and talked about the garden until the sun went down behind the hills and trees."
           ]

    batched = Vision.image_to_text(@trocr, [compile: [batch_size: 2]] ++ opts)
    assert Nx.Serving.run(batched, [@handwriting, @handwriting_long]) == alone

    [%{results: [%{text: text}]} | _] = alone
    stream = Vision.image_to_text(@trocr, [stream: true] ++ opts)
    assert Enum.join(Nx.Serving.run(stream, @handwriting)) == text
  end

  @florence_tiny {:hf, "Xenova/tiny-random-Florence2ForConditionalGeneration"}

  # The ids of test/fixtures/florence2.json come from the beam search of transformers on the
  # base model, see test/support/pipeline_references.py. The tiny export has the tokenizer and
  # the task tables of the base model. Generation gives the ids without the eos token.
  @tag ci_partition: 15
  test "the Florence-2 processor parses the answers as post_process_generation" do
    repo = Transmuter.Repo.load!(@florence_tiny)
    processor = Florence2Processor.load!(repo)
    tokenizer = Tokenizer.load!(repo)

    for {input, ids, expected} <- florence_cases() do
      text = Tokenizer.decode(tokenizer, Enum.drop(ids, -1), skip_special_tokens: false)
      {height, width, _channels} = Nx.shape(StbImage.to_nx(input.image))
      result = Florence2Processor.parse(processor, input.task, text, {height, width})
      assert_florence(result, expected)
    end
  end

  # The tiny export has a zero lm_head and zero image features, so its logits are 0 for each
  # input, and the ids are those of the processors and the lowest id of a tie. So the test
  # checks that the four models run together, alone and in a batch with two tasks, and not
  # their values. Without a task, the task is "<CAPTION>". The ids [0, 0, 0, 1, 0, 0] are
  # those of ONNX Runtime with the processors of transformers, as text "<s><s><s><pad><s><s>".
  @tag ci_partition: 15
  test "Florence-2 runs the vision encoder, the embeddings, the encoder and the decoder" do
    opts = [num_beams: 1, max_new_tokens: 20] ++ @opts

    inputs = [
      @image,
      %{image: @image, task: "<OD>"},
      %{image: @image, task: "<CAPTION_TO_PHRASE_GROUNDING>", text: "two cats"}
    ]

    expected = [
      %{results: [%{text: "<pad>"}]},
      %{results: [%{text: "<pad>", boxes: [], labels: []}]},
      %{results: [%{text: "<pad>", boxes: [], labels: []}]}
    ]

    assert Nx.Serving.run(Vision.image_to_text(@florence_tiny, opts), inputs) == expected
    batched = Vision.image_to_text(@florence_tiny, [compile: [batch_size: 2]] ++ opts)
    assert Nx.Serving.run(batched, inputs) == expected

    assert_raise ArgumentError, ~r/the task <CAPTION> takes no :text/, fn ->
      Nx.Serving.run(Vision.image_to_text(@florence_tiny, opts), %{image: @image, text: "cats"})
    end
  end

  @doc false
  def florence_cases do
    "../fixtures/florence2.json"
    |> Path.expand(__DIR__)
    |> File.read!()
    |> JSON.decode!()
    |> Enum.map(fn case ->
      image = StbImage.read_file!(Path.expand("../fixtures/#{case["image"]}", __DIR__))
      input = %{image: image, task: case["task"], text: case["text"]}
      text = case["decoded"] |> String.replace(["<s>", "</s>"], "") |> String.trim()
      {input, case["ids"], florence_result(text, case["result"])}
    end)
  end

  defp florence_result(_text, text) when is_binary(text), do: %{text: text}

  defp florence_result(text, %{"bboxes" => boxes, "labels" => labels}) do
    boxes = for [x1, y1, x2, y2] <- boxes, do: %{xmin: x1, ymin: y1, xmax: x2, ymax: y2}
    %{text: text, boxes: boxes, labels: labels}
  end

  defp florence_result(text, %{"quad_boxes" => boxes, "labels" => labels}),
    do: %{text: text, quad_boxes: boxes, labels: labels}

  defp florence_result(text, %{"polygons" => polygons, "labels" => labels}),
    do: %{text: text, polygons: polygons, labels: labels}

  # Python rounds the coordinates to f32, so they agree within 1e-6 relative.
  @doc false
  def assert_florence(got, expected) when is_map(got) and not is_struct(got) do
    assert Map.keys(got) == Map.keys(expected)
    for {key, value} <- got, do: assert_florence(value, expected[key])
  end

  def assert_florence(got, expected) when is_list(got) do
    assert length(got) == length(expected)
    Enum.zip_with(got, expected, &assert_florence/2)
  end

  def assert_florence(got, expected) when is_float(got),
    do: assert_in_delta(got, expected, 1.0e-6 * abs(expected))

  def assert_florence(got, expected), do: assert(got == expected)

  @smolvlm {:hf, "HuggingFaceTB/SmolVLM-256M-Instruct"}

  @tag ci_partition: 15
  test "the Idefics3 processor gives the tiles and the prompt of Python transformers" do
    repo = Transmuter.Repo.load!(@smolvlm)
    processor = Idefics3Processor.load!(repo)
    tokenizer = Tokenizer.load!(repo)
    image = StbImage.to_nx(@image)
    content = [%{type: "image"}, %{type: "text", text: "Can you describe this image?"}]
    prompt = ChatTemplate.render(repo, [%{role: "user", content: content}])

    # A wide image, a tall image, and an image that gets larger again to a multiple of 512
    # before it splits. The prompt is checked by its length and a SHA-256 of its ids.
    for {image, tiles, expected, mean, count, sha} <- [
          {image, 13, [0.1059, -0.7725, -0.7333, -0.9765], 0.4757, 876, "176c4ea8b7b148c7"},
          {Nx.transpose(image, axes: [1, 0, 2]), 13, [0.1059, -0.1608, -0.7333, 0.349], 0.4757,
           877, "a493c96a7687a9a8"},
          {image[0..99], 9, [0.1059, -0.7098, 0.2549, -0.4588], 0.4885, 611, "89410d97ee15e8d2"}
        ] do
      {%{"pixel_values" => pixels}, text} =
        Idefics3Processor.process(processor, image, @opts[:backend])

      pixels = Nx.backend_transfer(pixels, Nx.BinaryBackend)
      assert Nx.shape(pixels) == {tiles, 3, 512, 512}

      got =
        for index <- [[0, 0, 0, 0], [1, 1, 100, 37], [-1, 2, 511, 511], [2, 0, 257, 190]],
            do: Nx.to_number(pixels[index])

      assert_all_close(got, expected)
      assert_in_delta Nx.to_number(Nx.mean(Nx.abs(pixels))), mean, 1.0e-4

      {:ok, encoding} =
        Tokenizers.Tokenizer.encode(tokenizer.native, String.replace(prompt, "<image>", text))

      ids = Tokenizers.Encoding.get_ids(encoding)
      digest = :crypto.hash(:sha256, for(id <- ids, into: <<>>, do: <<id::little-64>>))

      assert {length(ids), binary_part(Base.encode16(digest, case: :lower), 0, 16)} ==
               {count, sha}
    end
  end

  @qwen2_vl {:hf, "onnx-community/Qwen2-VL-2B-Instruct"}

  @tag ci_partition: 15
  test "the Qwen2-VL processor gives the patches and the prompt of Python transformers" do
    repo = Transmuter.Repo.load!(@qwen2_vl)
    processor = Qwen2VLProcessor.load!(repo)
    tokenizer = Tokenizer.load!(repo)
    image = StbImage.to_nx(@image)
    content = [%{type: "image"}, %{type: "text", text: "Can you describe this image?"}]
    prompt = ChatTemplate.render(repo, [%{role: "user", content: content}])

    # A wide and a tall image. Both have 99 image tokens, so their prompts are equal.
    for {image, grid, expected} <- [
          {image, [1, 18, 22], [0.2661, 0.5435, -0.8403, -0.2963]},
          {Nx.transpose(image, axes: [1, 0, 2]), [1, 22, 18], [0.2661, -0.201, -0.8403, -1.0317]}
        ] do
      {tensors, text} = Qwen2VLProcessor.process(processor, image, @opts[:backend])
      pixels = Nx.backend_transfer(tensors["pixel_values"], Nx.BinaryBackend)
      assert Nx.shape(pixels) == {396, 1176}
      assert Nx.to_list(tensors["image_grid_thw"]) == [grid]

      got =
        for index <- [[0, 0], [1, 100], [-1, 1175], [200, 777]], do: Nx.to_number(pixels[index])

      assert_all_close(got, expected)
      assert_in_delta Nx.to_number(Nx.mean(Nx.abs(pixels))), 0.8945, 1.0e-4

      {:ok, encoding} =
        Tokenizers.Tokenizer.encode(
          tokenizer.native,
          String.replace(prompt, "<|image_pad|>", text)
        )

      ids = Tokenizers.Encoding.get_ids(encoding)
      digest = :crypto.hash(:sha256, for(id <- ids, into: <<>>, do: <<id::little-64>>))

      assert {length(ids), binary_part(Base.encode16(digest, case: :lower), 0, 16)} ==
               {126, "5879c4bd9e8deede"}
    end
  end

  test "the featurizer gives the sizes of the DETR, YOLOS and DPT image processors" do
    # A long side over longest_edge gives the short side longest_edge times the aspect ratio,
    # rounded half to even: 1333 * 400 / 800 is 666.5. YOLOS rounds each side down to a
    # multiple of 16. DPT takes the scale nearer to 1 for both sides and rounds each side half
    # to even to a multiple of 14: 35x518 keeps the scale 1, and 35 / 14 is 2.5.
    for {repo, expected} <- [
          {"Xenova/detr-resnet-50",
           [{800, 1066}, {133, 1333}, {1333, 133}, {666, 1333}, {1066, 800}, {90, 1333}]},
          {"Xenova/yolos-tiny",
           [{512, 672}, {128, 1328}, {1328, 128}, {512, 1024}, {672, 512}, {80, 1328}]},
          {"onnx-community/depth-anything-v2-small",
           [{392, 518}, {56, 518}, {518, 56}, {518, 1036}, {686, 518}, {28, 518}]}
        ] do
      featurizer = Featurizer.load!({:hf, repo})

      got =
        for size <- [{240, 320}, {100, 1000}, {1000, 100}, {400, 800}, {800, 600}, {35, 518}] do
          image = Nx.broadcast(Nx.tensor(0, type: :u8), Tuple.insert_at(size, 2, 3))
          {1, 3, height, width} = Nx.shape(Featurizer.process(featurizer, [image], @opts))
          {height, width}
        end

      assert got == expected, "#{repo}: expected #{inspect(expected)}, got #{inspect(got)}"
    end
  end

  # The references of DETR and YOLOS are the PyTorch models with
  # post_process_object_detection of transformers 4, see test/support/pipeline_references.py.
  # The reference of RF-DETR is the PyTorch model of transformers 5, see
  # test/support/rf_detr_reference.py. DETR and YOLOS give the detections in the order of the
  # queries, and RF-DETR from the highest score.
  test "object detection matches post_process_object_detection of transformers" do
    for {repo, wide, tall} <- [
          {"Xenova/detr-resnet-50",
           [
             {"remote", 0.997378, [19.8952, 35.1192, 87.331, 59.1798]},
             {"remote", 0.995471, [166.5936, 35.2691, 184.6295, 94.831]},
             {"couch", 0.995002, [-0.0737, -0.0021, 319.921, 239.9825]},
             {"cat", 0.996962, [6.5825, 25.1499, 156.9784, 236.0207]},
             {"cat", 0.997858, [172.6578, 12.0069, 320.4138, 184.9281]}
           ],
           [
             {"remote", 0.981368, [39.0159, 165.7224, 97.1045, 185.2333]},
             {"remote", 0.996639, [35.4754, 19.0578, 58.4294, 88.7652]},
             {"cat", 0.997509, [8.6708, 172.2435, 185.811, 320.2794]},
             {"cat", 0.994645, [26.4559, 5.6095, 237.1201, 156.2772]},
             {"couch", 0.993239, [-0.0561, 0.1082, 239.9409, 320.1053]}
           ]},
          {"Xenova/yolos-tiny",
           [
             {"remote", 0.847386, [24.1551, 36.6842, 89.2268, 59.8285]},
             {"remote", 0.945534, [168.4989, 39.3963, 185.3595, 96.2932]},
             {"cat", 0.974007, [6.1349, 26.4885, 155.0585, 235.7913]},
             {"bed", 0.679433, [0.0896, 0.0021, 320.0895, 239.8154]},
             {"remote", 0.990069, [20.9094, 36.1831, 87.784, 59.9306]},
             {"cat", 0.886771, [167.2931, 0.238, 319.4196, 184.5449]}
           ],
           [
             {"cat", 0.985624, [29.0007, 4.5455, 236.0907, 156.2661]},
             {"bed", 0.97155, [0.1085, 0.1047, 239.9882, 319.8357]},
             {"remote", 0.998535, [35.7993, 20.1866, 59.2976, 92.7396]},
             {"remote", 0.989537, [37.5054, 168.2169, 96.157, 186.4062]},
             {"cat", 0.996023, [12.7713, 172.1163, 185.2852, 319.4268]}
           ]},
          {"onnx-community/rfdetr_nano-ONNX",
           [
             {"cat", 0.962052, [7.2289, 27.1561, 158.3699, 236.7962]},
             {"cat", 0.948672, [173.543, 12.8385, 319.5366, 186.8357]},
             {"remote", 0.910824, [167.0068, 38.483, 185.5564, 93.4559]},
             {"remote", 0.881908, [19.8815, 36.7721, 87.671, 59.1429]},
             {"couch", 0.659234, [-0.2202, -0.1711, 320.8276, 239.657]}
           ],
           [
             {"cat", 0.948363, [12.6889, 172.1027, 185.9526, 319.9852]},
             {"cat", 0.944986, [27.1592, 7.6413, 236.8518, 160.4658]},
             {"remote", 0.874704, [36.6841, 19.3162, 58.9057, 88.8499]},
             {"remote", 0.662767, [38.3403, 167.271, 94.4649, 185.7012]}
           ]}
        ] do
      serving = Vision.object_detection({:hf, repo}, @opts)
      tall_image = Nx.transpose(StbImage.to_nx(@image), axes: [1, 0, 2])
      # A tensor and a StbImage give the same result.
      [image, tensor, tall_result] =
        Nx.Serving.run(serving, [@image, StbImage.to_nx(@image), tall_image])

      assert image == tensor
      assert_detections(repo, image, wide)
      assert_detections(repo, tall_result, tall)
    end
  end

  # The serving compiles the size of a 640x480 image when it starts, which is also the size of
  # the wide image. The tall image compiles on its first call, and its second call finds it.
  test "object detection compiles each image size once" do
    serving = Vision.object_detection({:hf, "Xenova/yolos-tiny"}, @opts)
    name = :"detection_#{System.unique_integer([:positive])}"
    start_supervised!({Nx.Serving, serving: serving, name: name})

    test_pid = self()
    handler = "detection-#{inspect(test_pid)}"

    :telemetry.attach(handler, [:transmuter, :cache, :miss], &__MODULE__.forward_miss/4, test_pid)

    on_exit(fn -> :telemetry.detach(handler) end)

    tall = Nx.transpose(StbImage.to_nx(@image), axes: [1, 0, 2])
    images = [@image, tall, @image, tall]
    results = Nx.Serving.batched_run(name, images)
    misses = receive_misses()
    :telemetry.detach(handler)

    assert Enum.uniq(misses) == misses

    pixels =
      for {0, shapes} <- misses, {"pixel_values", {shape, _type}} <- shapes, do: shape

    assert pixels == [{1, 3, 672, 512}]
    assert results == Nx.Serving.run(serving, images)
  end

  def forward_miss(_event, _measurements, metadata, pid) do
    if match?({:hf, "Xenova/yolos-tiny", _file}, metadata.source),
      do: send(pid, {:miss, metadata.position, metadata.shapes})
  end

  defp receive_misses do
    receive do
      {:miss, position, shapes} -> [{position, shapes} | receive_misses()]
    after
      0 -> []
    end
  end

  # The reference is the zero-shot-object-detection pipeline of transformers, which runs the
  # model once for each label. The serving runs the labels as the text queries of one run.
  # On the Metal GPU, the batch of 2 changes the score of one box by 1.1e-4, so the scores
  # have a tolerance of 2e-4.
  test "zero-shot object detection with OWL-ViT matches the pipeline" do
    serving =
      Vision.zero_shot_object_detection(
        {:hf, "Xenova/owlvit-base-patch32"},
        @detection_labels,
        [compile: [batch_size: 2]] ++ @opts
      )

    wide = [
      {"remote control", 0.233325, [19.3323, 35.9036, 89.429, 58.3903]},
      {"cat", 0.230558, [3.6999, 26.5906, 157.97, 234.9424]},
      {"cat", 0.171058, [168.0633, 12.8512, 316.6584, 182.1349]}
    ]

    tall = [
      {"cat", 0.521507, [13.008, 167.1858, 183.7693, 320.0486]},
      {"remote control", 0.511203, [35.5097, 18.7584, 58.6417, 85.7964]},
      {"cat", 0.484385, [26.2655, 2.954, 237.4741, 149.898]},
      {"remote control", 0.317606, [40.8807, 168.3273, 100.3745, 186.2379]},
      {"couch", 0.228428, [0.5833, 0.7704, 239.8784, 319.3692]},
      {"remote control", 0.19914, [39.5121, 167.4139, 96.1983, 186.4206]},
      {"remote control", 0.129362, [25.6609, 9.7424, 72.2205, 179.5552]},
      {"remote control", 0.121832, [24.6725, -0.1015, 67.9243, 86.3368]}
    ]

    assert_zero_shot_detections(serving, wide, tall, 2.0e-4)
  end

  @doc """
  Runs `serving` on the wide image, its tensor and the tall image in one list, and on the tall image alone. A tensor and a `StbImage` give the same result, and so does an image alone.

  The detections come from the highest score. Grounding DINO can swap two detections with near scores, so the comparison with the reference sorts the detections by label and `xmin`. The boxes have a tolerance of 1e-3 of the side of the image, 0.32 pixels.
  """
  def assert_zero_shot_detections(serving, wide, tall, score_delta) do
    tall_image = Nx.transpose(StbImage.to_nx(@image), axes: [1, 0, 2])

    [image, tensor, tall_result] =
      Nx.Serving.run(serving, [@image, StbImage.to_nx(@image), tall_image])

    assert image == tensor
    assert Nx.Serving.run(serving, tall_image) == tall_result

    for {%{detections: detections}, expected} <- [{image, wide}, {tall_result, tall}] do
      assert Enum.sort_by(detections, & &1.score, :desc) == detections
      got = Enum.sort_by(detections, &{&1.label, &1.box.xmin})
      expected = Enum.sort_by(expected, fn {label, _score, [xmin | _]} -> {label, xmin} end)
      assert Enum.map(got, & &1.label) == Enum.map(expected, &elem(&1, 0))

      for {detection, {_label, score, box}} <- Enum.zip(got, expected) do
        assert_in_delta detection.score, score, score_delta, inspect(detection)
        %{xmin: xmin, ymin: ymin, xmax: xmax, ymax: ymax} = detection.box

        for {g, e} <- Enum.zip([xmin, ymin, xmax, ymax], box),
            do: assert_in_delta(g, e, 0.32, inspect(detection))
      end
    end
  end

  # The reference of Depth Anything V2 is the depth-estimation pipeline of transformers. The
  # ONNX file of Xenova/depth-anything-small-hf gives a depth up to 0.9% away from its PyTorch
  # model, so its reference is ONNX Runtime with the post-processing of the pipeline. Each row
  # has predicted_depth and the 8-bit depth at (y, x) and their means. The 8-bit depth
  # truncates, so a value can be one step away.
  test "depth estimation matches the pipeline" do
    points = [[0, 0], [100, 37], [57, 190], [150, 120], [239, 239]]

    for {repo, wide, tall} <- [
          {"onnx-community/depth-anything-v2-small",
           {[2.3001, 2.576, 2.4813, 2.3204, 3.7449], 2.5175, [104, 123, 117, 105, 205], 119.047},
           {[2.7954, 1.4816, 2.8761, 1.9743, 3.4688], 2.6133, [130, 34, 136, 70, 179], 116.8685}},
          {"Xenova/depth-anything-small-hf",
           {[7.5774, 10.0037, 8.7434, 9.9116, 14.4565], 9.8135, [81, 137, 108, 135, 240],
            132.5709},
           {[8.6462, 5.4587, 11.3492, 7.6599, 12.2932], 8.9703, [112, 33, 178, 87, 201], 119.6214}}
        ] do
      serving = Vision.depth_estimation({:hf, repo}, @opts)
      tall_image = Nx.transpose(StbImage.to_nx(@image), axes: [1, 0, 2])
      # A tensor and a StbImage give the same result.
      [image, tensor, tall_result] =
        Nx.Serving.run(serving, [@image, StbImage.to_nx(@image), tall_image])

      assert image == tensor

      for {result, size, {depth, mean, u8, u8_mean}} <- [
            {image, {240, 320}, wide},
            {tall_result, {320, 240}, tall}
          ] do
        assert {Nx.shape(result.predicted_depth), Nx.type(result.depth)} == {size, {:u, 8}}

        for {point, d, u} <- Enum.zip([points, depth, u8]) do
          assert_in_delta Nx.to_number(result.predicted_depth[point]), d, 1.0e-3 * d, repo
          assert_in_delta Nx.to_number(result.depth[point]), u, 1, repo
        end

        assert_in_delta Nx.to_number(Nx.mean(result.predicted_depth)), mean, 1.0e-3 * mean
        assert_in_delta Nx.to_number(Nx.mean(result.depth)), u8_mean, 0.01
      end
    end
  end

  # The reference is the image-segmentation pipeline of transformers. The fixtures have the
  # label id of each pixel. The images have different sizes and go in one batch.
  test "image segmentation matches the pipeline" do
    repo = {:hf, "Xenova/segformer-b0-finetuned-ade-512-512"}
    serving = Vision.image_segmentation(repo, [compile: [batch_size: 2]] ++ @opts)

    # The panoptic DETR exports give logits, pred_boxes and pred_masks.
    assert_raise ArgumentError, ~r/Panoptic and instance segmentation are not supported/, fn ->
      model = %{outputs: ["logits", "pred_boxes", "pred_masks"]}
      Vision.image_segmentation(repo, [model: model] ++ @opts)
    end

    tall = Nx.transpose(StbImage.to_nx(@image), axes: [1, 0, 2])
    [wide_result, tall_result] = Nx.Serving.run(serving, [@image, tall])
    assert Nx.Serving.run(serving, tall) == tall_result

    %{"id2label" => id2label} = Transmuter.Repo.load!(repo).config

    ids = Map.new(id2label, fn {id, label} -> {label, String.to_integer(id)} end)

    for {result, name, labels} <- [
          {wide_result, "wide", ~w(wall floor person sofa base box signboard ball)},
          {tall_result, "tall",
           ["wall", "floor", "bed ", "person", "table", "seat", "cushion", "box", "towel"] ++
             ~w(plaything bag blanket)}
        ] do
      assert Enum.map(result.segments, & &1.label) == labels
      expected = fixture("cats_segformer_#{name}.png")

      got =
        Enum.reduce(result.segments, Nx.broadcast(Nx.tensor(0, type: :u8), expected), fn
          %{label: label, score: nil, mask: mask}, got ->
            assert Nx.type(mask) == {:u, 8}
            assert MapSet.new(Nx.to_flat_list(mask)) == MapSet.new([0, 255])
            Nx.select(mask, ids[label], got)
        end)

      assert Nx.to_number(Nx.sum(Nx.not_equal(got, expected))) == 0, name
    end
  end

  # The reference is ONNX Runtime with the post-processing of transformers.js and the
  # bilinear resize of Pillow. The mask truncates to 8 bits, so a value can be one step away.
  # The ONNX file has a fixed batch size of 1.
  test "background removal matches the post-processing of transformers.js" do
    repo = {:hf, "onnx-community/ormbg-ONNX"}
    serving = Vision.background_removal(repo, @opts)

    assert_raise ArgumentError, ~r/fixed batch size of 1/, fn ->
      Vision.background_removal(repo, [compile: [batch_size: 2]] ++ @opts)
    end

    tall = Nx.transpose(StbImage.to_nx(@image), axes: [1, 0, 2])
    [wide_result, tall_result] = Nx.Serving.run(serving, [@image, tall])
    assert Nx.Serving.run(serving, tall) == tall_result

    for {%{mask: mask}, name} <- [{wide_result, "wide"}, {tall_result, "tall"}] do
      expected = fixture("cats_ormbg_#{name}.png")
      assert {Nx.shape(mask), Nx.type(mask)} == {Nx.shape(expected), {:u, 8}}
      difference = Nx.abs(Nx.subtract(Nx.as_type(mask, :s16), Nx.as_type(expected, :s16)))

      assert Nx.to_number(Nx.reduce_max(difference)) <= 1, name
      assert Nx.to_number(Nx.sum(difference)) <= 10, name
    end
  end

  # The reference is SamModel of transformers with SamProcessor and post_process_masks, one
  # prompt for each call. The export has no box input.
  test "mask generation with SAM matches transformers" do
    serving = Vision.mask_generation({:hf, "Xenova/slimsam-77-uniform"}, @opts)

    assert_masks(
      serving,
      "slimsam",
      Enum.take(sam_prompts(), 2),
      {
        [[0.87823, 0.86798, 0.60964], [0.98281, 0.94075, 0.90734]],
        [[0.81632, 0.80129, 0.68658], [0.91979, 0.93776, 0.89779]]
      }
    )

    for {prompt, message} <- [
          {%{box: [40, 60, 200, 230]}, ~r/has no input_boxes/},
          {%{points: [[80, 120]], labels: [2]}, ~r/must be a list of 0 and 1/},
          {%{points: [[80, 120]], labels: [1, 0]}, ~r/one label for each point/},
          {%{labels: []}, ~r/must have points or a box/}
        ] do
      assert_raise ArgumentError, message, fn ->
        Nx.Serving.run(serving, %{image: @image, prompts: [prompt]})
      end
    end
  end

  @doc """
  The prompts of the SAM tests: a point with the default label, three points, a box, and a box with a point.
  """
  def sam_prompts do
    [
      %{points: [[80, 120]]},
      %{points: [[80, 120], [200, 60], [150, 200]], labels: [1, 1, 0]},
      %{box: [40, 60, 200, 230]},
      %{points: [[80, 120]], labels: [1], box: [40, 60, 200, 230]}
    ]
  end

  @doc """
  Runs `serving` on the wide and the tall image with `prompts`, and compares the masks with the fixtures `sam_<name>_wide.png` and `sam_<name>_tall.png`, which have the 3 masks of each prompt one below the other. The scores must be within 1e-4. The logits of the decoder differ from PyTorch by about 1e-4, also in ONNX Runtime, and up to 5 pixels of a mask of the test have a resized logit within 1e-4 of 0. So a mask can have up to 5 pixels that are not in the fixture.
  """
  def assert_masks(serving, name, prompts, {wide, tall}) do
    tall_image = Nx.transpose(StbImage.to_nx(@image), axes: [1, 0, 2])

    results =
      Nx.Serving.run(serving, [
        %{image: @image, prompts: prompts},
        %{image: tall_image, prompts: prompts}
      ])

    for {result, picture, scores} <- Enum.zip([results, ["wide", "tall"], [wide, tall]]) do
      expected = fixture("sam_#{name}_#{picture}.png")
      {_, width} = Nx.shape(expected)
      expected = Nx.reshape(expected, {length(prompts), 3, :auto, width})

      for {{masks, mask_scores}, i} <- Enum.with_index(Enum.zip(result.masks, result.iou_scores)) do
        assert_all_close(mask_scores, Enum.at(scores, i))

        for {mask, j} <- Enum.with_index(masks) do
          assert Nx.type(mask) == {:u, 8}
          different = Nx.to_number(Nx.sum(Nx.not_equal(mask, expected[i][j])))

          assert different <= 5,
                 "#{name} #{picture} prompt #{i} mask #{j}: #{different} pixels"
        end
      end
    end
  end

  defp fixture(name) do
    Path.expand("../fixtures/#{name}", __DIR__)
    |> StbImage.read_file!()
    |> StbImage.to_nx()
    |> Nx.squeeze(axes: [2])
  end

  # Box coordinates can be near 0, so their tolerance is 1e-3 of the side of the image, 0.32
  # pixels. Scores have a tolerance of 1e-3 relative.
  defp assert_detections(repo, %{detections: detections}, expected) do
    assert Enum.map(detections, & &1.label) == Enum.map(expected, &elem(&1, 0)), repo

    for {detection, {_label, score, box}} <- Enum.zip(detections, expected) do
      assert_in_delta detection.score, score, 1.0e-3 * score, repo
      %{xmin: xmin, ymin: ymin, xmax: xmax, ymax: ymax} = detection.box

      for {g, e} <- Enum.zip([xmin, ymin, xmax, ymax], box),
          do: assert_in_delta(g, e, 0.32, "#{repo}: #{inspect(detection)}")
    end
  end

  defp assert_predictions(%{predictions: predictions}, expected) do
    assert Enum.map(predictions, & &1.label) == Enum.map(expected, &elem(&1, 0))

    for {prediction, {_label, score}} <- Enum.zip(predictions, expected),
        do: assert_in_delta(prediction.score, score, 1.0e-4)
  end

  defp assert_all_close(got, expected) do
    for {g, e} <- Enum.zip(got, expected) do
      assert_in_delta g, e, 1.0e-4, "expected #{inspect(expected)}, got #{inspect(got)}"
    end
  end
end

# --only ci_partition:N in CI would also include :large_model tests of VisionTest, because an
# included tag wins over an excluded one. So these tests are in a module without a
# :ci_partition tag, and CI leaves them out.
defmodule Transmuter.VisionLargeModelTest do
  use ExUnit.Case, async: true

  @moduletag :hub
  @moduletag timeout: 900_000

  @opts [
    backend: Transmuter.TestBackend.backend(),
    defn_options: Transmuter.TestBackend.compiler_opts()
  ]

  alias Transmuter.{Vision, VisionTest}

  @image StbImage.read_file!(Path.expand("../fixtures/cats.png", __DIR__))
  @smolvlm {:hf, "HuggingFaceTB/SmolVLM-256M-Instruct"}
  @qwen2_vl {:hf, "onnx-community/Qwen2-VL-2B-Instruct"}

  # The reference is the beam search of transformers on the export's image features, see
  # test/support/pipeline_references.py. Python runs each prompt without padding, and the
  # serving pads it to 32 tokens. The fp32 files have 1.1 GB, so CI leaves this test out.
  @tag :large_model
  test "Florence-2 matches generate and post_process_generation of transformers" do
    cases = VisionTest.florence_cases()
    inputs = Enum.map(cases, &elem(&1, 0))
    serving = Vision.image_to_text({:hf, "onnx-community/Florence-2-base-ft"}, @opts)
    results = Nx.Serving.run(serving, inputs)

    for {%{results: [result]}, {_input, _ids, expected}} <- Enum.zip(results, cases),
        do: VisionTest.assert_florence(result, expected)

    # The <OD> and the grounding prompts pad to the same length, so they run in one batch.
    two = Enum.slice(inputs, 2, 2)

    batched =
      Vision.image_to_text(
        {:hf, "onnx-community/Florence-2-base-ft"},
        [compile: [batch_size: 2]] ++ @opts
      )

    assert Nx.Serving.run(batched, two) == Enum.slice(results, 2, 2)
  end

  # The reference of OWLv2 is the zero-shot-object-detection pipeline of transformers. On the
  # Linux runner, OWLv2 runs a ViT of 3600 patches for each image, so CI leaves this test out.
  # The reference of Grounding DINO is ONNX Runtime with the post-processing of the pipeline,
  # and the text of each label is the label and a period, see
  # test/support/pipeline_references.py. Grounding DINO selects 900 of 13294 boxes with TopK.
  # In ONNX Runtime, a change of 1e-7 in the pixel values changes its scores by up to 5e-4, so
  # its scores have a tolerance of 1e-3.
  @tag :large_model
  test "zero-shot object detection with OWLv2 and Grounding DINO matches the references" do
    labels = ["cat", "remote control", "couch"]
    dino = {:hf, "onnx-community/grounding-dino-tiny-ONNX"}

    for {repo, compile, wide, tall, score_delta} <- [
          {{:hf, "Xenova/owlv2-base-patch16-ensemble"}, [batch_size: 2],
           [
             {"remote control", 0.747254, [167.3377, 38.4727, 185.0484, 93.7467]},
             {"remote control", 0.734983, [19.3393, 36.3176, 88.0444, 58.4288]},
             {"cat", 0.687935, [2.9749, 26.82, 159.7625, 240.4006]},
             {"cat", 0.635032, [171.2533, 10.9012, 321.0749, 184.4348]},
             {"couch", 0.502501, [2.5106, 1.0772, 319.8392, 241.2461]},
             {"remote control", 0.135166, [134.4929, 98.0621, 150.8824, 119.7911]}
           ],
           [
             {"remote control", 0.720599, [38.6238, 167.0137, 93.768, 184.777]},
             {"cat", 0.644321, [27.5364, 3.5487, 238.057, 157.1026]},
             {"remote control", 0.622598, [36.3688, 20.3372, 58.93, 88.0172]},
             {"cat", 0.62235, [12.1107, 171.5322, 185.6642, 321.4313]},
             {"couch", 0.400949, [1.6723, 4.3198, 239.3741, 320.0118]},
             {"couch", 0.137685, [1.0749, 4.5949, 64.3031, 318.7564]},
             {"cat", 0.100644, [27.771, 87.9667, 119.2509, 157.6453]}
           ], 1.0e-4},
          {dino, [batch_size: 1],
           [
             {"couch", 0.885685, [0.0104, 0.033, 320.0104, 240.0329]},
             {"cat", 0.760616, [4.0827, 26.7547, 158.4818, 237.3714]},
             {"cat", 0.722298, [172.7025, 11.7282, 320.021, 186.0622]},
             {"remote control", 0.443257, [19.0494, 35.2339, 88.6779, 58.9769]},
             {"remote control", 0.241131, [165.9793, 37.3847, 185.2825, 93.6804]},
             {"remote control", 0.222448, [10.4043, 28.5515, 99.0912, 65.8146]},
             {"remote control", 0.203506, [160.3972, 34.5093, 191.4096, 95.9579]},
             {"remote control", 0.168397, [169.5572, 11.0481, 319.8499, 186.3529]},
             {"remote control", 0.145843, [4.1858, 25.7082, 158.8907, 235.9273]},
             {"remote control", 0.108425, [3.5725, 21.0415, 112.5312, 75.9585]},
             {"remote control", 0.101172, [150.7831, 28.8857, 202.0638, 103.9519]}
           ],
           [
             {"couch", 0.902724, [0.0258, 0.0261, 240.0258, 320.0261]},
             {"cat", 0.765693, [12.2327, 172.6787, 186.1916, 320.0205]},
             {"cat", 0.765443, [26.9563, 5.2032, 237.2716, 158.719]},
             {"remote control", 0.452206, [35.4916, 18.5911, 59.0669, 88.5096]},
             {"remote control", 0.237786, [37.9201, 165.442, 93.7996, 185.4498]},
             {"remote control", 0.225241, [26.6928, 12.765, 67.9435, 95.7055]},
             {"remote control", 0.205441, [26.7666, 4.7704, 237.7252, 158.7924]},
             {"remote control", 0.162937, [33.1304, 158.4051, 99.3104, 191.4754]},
             {"cat", 0.116407, [12.134, 4.8482, 237.6781, 319.4389]},
             {"remote control", 0.111291, [12.2848, 165.5046, 186.8294, 319.4321]}
           ], 1.0e-3}
        ] do
      serving = Vision.zero_shot_object_detection(repo, labels, [compile: compile] ++ @opts)
      VisionTest.assert_zero_shot_detections(serving, wide, tall, score_delta)
    end

    # A period ends a phrase of Grounding DINO.
    assert_raise ArgumentError, ~r/cannot have "." or "\?"/, fn ->
      Vision.zero_shot_object_detection(dino, ["cat. dog"], [model: %{}] ++ @opts)
    end
  end

  # The reference is Sam2Model of transformers with Sam2Processor and post_process_masks, one
  # prompt for each call. A prompt without points or without a box gives a zero-size input
  # to the decoder.
  @tag :large_model
  test "mask generation with SAM 2.1 matches transformers" do
    serving = Vision.mask_generation({:hf, "onnx-community/sam2.1-hiera-tiny-ONNX"}, @opts)

    VisionTest.assert_masks(
      serving,
      "sam2",
      VisionTest.sam_prompts(),
      {[
         [0.58498, 0.10027, 0.97088],
         [0.93111, 0.53209, 0.88527],
         [0.21746, 0.26922, 0.44355],
         [0.63941, 0.26945, 0.80429]
       ],
       [
         [0.89877, 0.08282, 0.8433],
         [0.92574, 0.36652, 0.7604],
         [0.59683, 0.23924, 0.81182],
         [0.57428, 0.50855, 0.82845]
       ]}
    )
  end

  # The reference is ONNX Runtime on the same files: the q4 vision encoder and decoder and
  # the 8-bit embeddings. The batch has rows of 13, 13 and 22 tiles and a row without images.
  # The vision encoder runs on 109 tiles. On the Linux runner (4 vCPU), 13 tiles take about
  # 41 s, so CI leaves this test out.
  @tag :large_model
  test "image-text-to-text matches ONNX Runtime alone, in a batch and in a stream" do
    tall = Nx.transpose(StbImage.to_nx(@image), axes: [1, 0, 2])
    crop = StbImage.to_nx(@image)[0..99]

    chats =
      for content <- [
            [
              %{type: "image", image: @image},
              %{type: "text", text: "Can you describe this image?"}
            ],
            [
              %{type: "image", image: tall},
              %{type: "text", text: "What animals are in the image?"}
            ],
            [
              %{type: "image", image: @image},
              %{type: "image", image: crop},
              %{type: "text", text: "How are these two images different?"}
            ],
            [%{type: "text", text: "What is the capital of France?"}]
          ],
          do: %{messages: [%{role: "user", content: content}]}

    expected = [
      " In the image we can see a cat sleeping on a pink bed. On the right there are two white and grey color",
      " There are two cats in the image.",
      " The first image shows a cat sleeping on a pink couch, while the second image shows a cat sleeping on a pink and",
      " The capital of France is Paris."
    ]

    opts = [strategy: :greedy, max_new_tokens: 24] ++ @opts
    alone = Nx.Serving.run(Vision.image_text_to_text(@smolvlm, opts), chats)
    assert Enum.map(alone, fn %{results: [%{text: text}]} -> text end) == expected

    batched = Vision.image_text_to_text(@smolvlm, [compile: [batch_size: 2]] ++ opts)
    assert Nx.Serving.run(batched, chats) == alone

    stream = Vision.image_text_to_text(@smolvlm, [stream: true] ++ opts)
    assert Enum.join(Nx.Serving.run(stream, hd(chats))) == hd(expected)
  end

  # Python has no reference for this export. Both chats have 13 tiles, so they run in one
  # batch, and each beam of a chat must read the image features of its chat.
  @tag :large_model
  test "image-text-to-text with beams gives the text of each chat alone in a batch" do
    tall = Nx.transpose(StbImage.to_nx(@image), axes: [1, 0, 2])

    chats =
      for {image, text} <- [
            {@image, "Can you describe this image?"},
            {tall, "What animals are in the image?"}
          ],
          do: %{
            messages: [
              %{
                role: "user",
                content: [%{type: "image", image: image}, %{type: "text", text: text}]
              }
            ]
          }

    opts = [num_beams: 3, max_new_tokens: 12] ++ @opts
    alone = Nx.Serving.run(Vision.image_text_to_text(@smolvlm, opts), chats)
    batched = Vision.image_text_to_text(@smolvlm, [compile: [batch_size: 2]] ++ opts)
    assert Nx.Serving.run(batched, chats) == alone
  end

  # The reference is ONNX Runtime on the same files: the fp32 vision encoder, the fp16
  # embeddings and the q4 decoder, with the 3-D positions of get_rope_index. The 8-bit vision
  # encoder of the default variants gives another text in Transmuter than in ONNX Runtime. The
  # model has 2B parameters, so CI leaves this test out.
  @tag :large_model
  test "image-text-to-text with Qwen2-VL matches ONNX Runtime alone and in a batch" do
    tall = Nx.transpose(StbImage.to_nx(@image), axes: [1, 0, 2])

    chats =
      for content <- [
            [
              %{type: "image", image: @image},
              %{type: "text", text: "Can you describe this image?"}
            ],
            [
              %{type: "image", image: tall},
              %{type: "text", text: "What animals are in the image?"}
            ],
            [%{type: "text", text: "What is the capital of France?"}]
          ],
          do: %{messages: [%{role: "user", content: content}]}

    expected = [
      "In the image, there are two cats resting on a pink couch. The cat on the left is a tabby with",
      "There are two cats in the image.",
      "The capital of France is Paris."
    ]

    # The two chats with images have the same number of patches, so they go into one batch.
    # The first chat also runs alone, with a padding row.
    opts =
      [
        variant: %{vision_encoder: :fp32},
        strategy: :greedy,
        max_new_tokens: 24,
        compile: [batch_size: 2]
      ] ++ @opts

    serving = Vision.image_text_to_text(@qwen2_vl, opts)
    results = Nx.Serving.run(serving, chats)
    assert Enum.map(results, fn %{results: [%{text: text}]} -> text end) == expected
    assert Nx.Serving.run(serving, hd(chats)) == hd(results)
  end

  # The reference is the image-to-image pipeline of transformers, without the padding, on a
  # 64x48 crop (sides that are multiples of 8, which get 8 more pixels) and a 59x50 crop.
  # The 8-bit values round, so a value can be one step away. CI takes no new models until
  # its download budget allows them, so the test is :large_model.
  @tag :large_model
  test "image to image with Swin2SR matches the pipeline" do
    for {repo, name, {x, y, width, height}, scale} <- [
          {"Xenova/swin2SR-classical-sr-x2-64", "x2", {110, 75, 64, 48}, 2},
          {"Xenova/swin2SR-realworld-sr-x4-64-bsrgan-psnr", "x4", {130, 70, 59, 50}, 4}
        ] do
      serving = Vision.image_to_image({:hf, repo}, @opts)
      crop = StbImage.to_nx(@image)[[y..(y + height - 1), x..(x + width - 1)]]
      %{image: image} = Nx.Serving.run(serving, crop)

      expected =
        Path.expand("../fixtures/cats_swin2sr_#{name}.png", __DIR__)
        |> StbImage.read_file!()
        |> StbImage.to_nx()

      assert {Nx.shape(image), Nx.type(image)} == {{scale * height, scale * width, 3}, {:u, 8}}
      assert Nx.shape(expected) == Nx.shape(image)
      difference = Nx.abs(Nx.subtract(Nx.as_type(image, :s16), Nx.as_type(expected, :s16)))
      differing = Nx.to_number(Nx.sum(Nx.not_equal(difference, 0)))
      assert Nx.to_number(Nx.reduce_max(difference)) <= 1, name
      assert differing <= 20, "#{name}: #{differing} values are one step away"
    end
  end
end
