defmodule Transmuter.VisionTest do
  use ExUnit.Case, async: true

  # Reference values come from Python transformers, see test/support/pipeline_references.py.
  # The first run downloads the models.
  @moduletag :hub
  @moduletag timeout: 900_000
  @moduletag ci_partition: 6

  @opts [
    backend: Transmuter.TestBackend.backend(),
    defn_options: Transmuter.TestBackend.compiler_opts()
  ]

  alias Transmuter.{ChatTemplate, Featurizer, Tokenizer, Vision}
  alias Transmuter.Vision.{Idefics3Processor, Qwen2VLProcessor}

  @image StbImage.read_file!(Path.expand("../fixtures/cats.png", __DIR__))
  @labels ["cats", "dogs", "a remote control", "a couch"]

  test "the featurizer gives the pixel values of the Pillow image processors" do
    # ViT resizes to 224x224 with bilinear. CLIP resizes the short side to 224 with bicubic
    # and crops the center.
    for {repo, expected, mean} <- [
          {"Xenova/vit-base-patch16-224", [0.1137, 0.2941, -0.5608, 0.4902, -0.2863], 0.4666},
          {"Xenova/clip-vit-base-patch32", [0.6165, -0.8516, 0.8519, -0.639, -0.4314], 0.8819}
        ] do
      featurizer = Featurizer.load!({:hf, repo})
      values = Featurizer.process(featurizer, [@image], backend: @opts[:backend])
      values = Nx.backend_transfer(values, Nx.BinaryBackend)

      assert Nx.shape(values) == {1, 3, 224, 224}

      got =
        for [c, y, x] <- [[0, 0, 0], [1, 100, 37], [2, 223, 223], [0, 57, 190], [1, 150, 120]],
            do: Nx.to_number(values[[0, c, y, x]])

      assert_all_close(got, expected)
      assert_in_delta Nx.to_number(Nx.mean(Nx.abs(values))), mean, 1.0e-4
    end
  end

  @tag :smoke
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
  # uses fp32.
  test "image to text matches generate of transformers" do
    for {source, variant, image, text} <- [
          {{:hf, "Xenova/vit-gpt2-image-captioning"}, :fp32, @image,
           "a cat laying on a blanket next to a cat laying on a bed "},
          {@trocr, :q8, @handwriting, "Meet me at the station at noon ."}
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
  test "a batch of images gives the text of each image alone, and a stream joins to it" do
    opts = [variant: :q8] ++ @opts
    single = Vision.image_to_text(@trocr, opts)
    alone = for image <- [@handwriting, @handwriting_long], do: Nx.Serving.run(single, image)

    assert Enum.map(alone, fn %{results: [%{text: text}]} -> text end) == [
             "Meet me at the station at noon .",
             "We walked along the river and talked about the garden until the sun went down behind the hills and trees ."
           ]

    batched = Vision.image_to_text(@trocr, [compile: [batch_size: 2]] ++ opts)
    assert Nx.Serving.run(batched, [@handwriting, @handwriting_long]) == alone

    [%{results: [%{text: text}]} | _] = alone
    stream = Vision.image_to_text(@trocr, [stream: true] ++ opts)
    assert Enum.join(Nx.Serving.run(stream, @handwriting)) == text
  end

  @smolvlm {:hf, "HuggingFaceTB/SmolVLM-256M-Instruct"}

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

  test "the featurizer gives the sizes of the DETR and YOLOS image processors" do
    # A long side over longest_edge gives the short side longest_edge times the aspect ratio,
    # rounded half to even: 1333 * 400 / 800 is 666.5. YOLOS rounds each side down to a
    # multiple of 16.
    for {repo, expected} <- [
          {"Xenova/detr-resnet-50",
           [{800, 1066}, {133, 1333}, {1333, 133}, {666, 1333}, {1066, 800}]},
          {"Xenova/yolos-tiny", [{512, 672}, {128, 1328}, {1328, 128}, {512, 1024}, {672, 512}]}
        ] do
      featurizer = Featurizer.load!({:hf, repo})

      got =
        for size <- [{240, 320}, {100, 1000}, {1000, 100}, {400, 800}, {800, 600}] do
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

# --only ci_partition:6 in CI would also include :large_model tests of VisionTest, because an
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

  alias Transmuter.Vision

  @image StbImage.read_file!(Path.expand("../fixtures/cats.png", __DIR__))
  @smolvlm {:hf, "HuggingFaceTB/SmolVLM-256M-Instruct"}
  @qwen2_vl {:hf, "onnx-community/Qwen2-VL-2B-Instruct"}

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
end
