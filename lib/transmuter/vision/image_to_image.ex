defmodule Transmuter.Vision.ImageToImage do
  @moduledoc """
  The image-to-image task. See `Transmuter.Vision.image_to_image/2`.

  The output image has a size that depends on the input image, so the serving runs each image as a batch of its own (`Transmuter.Serving.each/2`). The model, the crop and the conversion to 8 bits run on the device.
  """

  alias Transmuter.{Featurizer, Serving}

  # The compiled model keeps this many image sizes.
  @image_sizes 64

  @doc false
  def image_to_image(source, opts) do
    opts = Keyword.validate!(opts, Serving.image_options())

    Serving.batch_size_one!(
      opts[:compile],
      "the output image has a size that depends on the image"
    )

    {repo, model} = Serving.load_model!(source, [outputs: ["reconstruction"]] ++ opts, :fp32)
    featurizer = opts[:featurizer] || Featurizer.load!(repo)
    backend = opts[:backend] || Nx.default_backend()
    scale = repo.config["upscale"] || 1

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
      # On CUDA, the compile time of Swin2SR grows with the image size, because the XLA
      # autotuner runs its kernels at that size: 8.5 s for 72x56 pixel values and 49 s for
      # 144x112. So the serving starts with the pixel values of the smallest image, and
      # compiles the size of each image on its first call.
      pixel = Nx.broadcast(Nx.tensor(0, type: :u8, backend: Nx.BinaryBackend), {1, 1, 3})
      template = %{"pixel_values" => Featurizer.process_on_host(featurizer, [pixel])}
      run = Transmuter.compile(model, template, [cache_size: @image_sizes] ++ defn_options)

      fn %{"pixel_values" => pixels, "size" => size} ->
        [height, width] = Nx.to_flat_list(size)
        reconstruction = run.(%{"pixel_values" => pixels})["reconstruction"]
        %{"image" => image(reconstruction, scale * height, scale * width)}
      end
    end

    task = %{
      validate: &Featurizer.tensor/1,
      preprocess: preprocess,
      init: init,
      postprocess: fn %{"image" => image} -> %{image: image} end
    }

    Serving.each(task, opts)
  end

  # The region of the input image, without the padding of the featurizer. Then as the
  # image-to-image pipeline of Python transformers: the values clamped to 0..1, times 255,
  # rounded to 8 bits, with the channels last.
  defp image(reconstruction, height, width) do
    reconstruction
    |> Nx.slice([0, 0, 0, 0], [1, 3, height, width])
    |> Nx.clip(0, 1)
    |> Nx.multiply(255)
    |> Nx.round()
    |> Nx.as_type(:u8)
    |> Nx.transpose(axes: [0, 2, 3, 1])
  end
end
