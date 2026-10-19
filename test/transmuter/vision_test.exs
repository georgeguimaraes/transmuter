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

  alias Transmuter.{Featurizer, Vision}

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
