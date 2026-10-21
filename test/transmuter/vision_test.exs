defmodule Transmuter.VisionTest do
  use ExUnit.Case, async: true

  # Reference values come from Python transformers, see test/support/pipeline_references.py.
  # The first run downloads the models.
  @moduletag timeout: 900_000

  @device (System.get_env("TRANSMUTER_TEST_DEVICE") ||
             if(match?({:unix, :darwin}, :os.type()), do: "gpu", else: "cpu"))
          |> String.to_existing_atom()
  @opts [
    backend: {EMLX.Backend, device: @device},
    defn_options: [compiler: EMLX, device: @device]
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

  # The rows of a batch stop at different steps.
  test "a batch of images gives the text of each image alone, and a stream joins to it" do
    opts = [variant: :q8] ++ @opts
    single = Vision.image_to_text(@trocr, opts)
    alone = for image <- [@handwriting, @image], do: Nx.Serving.run(single, image)
    batched = Vision.image_to_text(@trocr, [compile: [batch_size: 2]] ++ opts)
    assert Nx.Serving.run(batched, [@handwriting, @image]) == alone

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

  # The reference is ONNX Runtime on the same files: the q4 vision encoder and decoder and
  # the 8-bit embeddings. The batch has rows of 13, 13 and 22 tiles and a row without images.
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
