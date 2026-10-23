defmodule Transmuter.Vision.DepthEstimation do
  @moduledoc """
  The depth estimation task. See `Transmuter.Vision.depth_estimation/2`.

  The depth map has the size of the input image, so the results of two images can have different shapes. The serving runs each image as a batch of its own (`Transmuter.Serving.each/2`). The model, the bicubic resize and the scale to 8 bits run on the device.
  """

  alias Transmuter.{Featurizer, Serving}

  # The compiled model keeps this many image sizes.
  @image_sizes 64

  @doc false
  def depth_estimation(source, opts) do
    opts = Keyword.validate!(opts, Serving.image_options())
    Serving.batch_size_one!(opts[:compile], "the depth map has the size of the image")
    {repo, model} = Serving.load_model!(source, [outputs: ["predicted_depth"]] ++ opts, :fp32)
    featurizer = opts[:featurizer] || Featurizer.load!(repo)
    backend = opts[:backend] || Nx.default_backend()

    preprocess = fn image ->
      {height, width, _channels} = Nx.shape(image)

      pixels =
        featurizer
        |> Featurizer.process([image], backend: backend)
        |> Nx.backend_transfer(Nx.BinaryBackend)

      %{
        "pixel_values" => pixels[0],
        "size" => Nx.tensor([height, width], backend: Nx.BinaryBackend)
      }
    end

    init = fn defn_options ->
      {height, width} = Serving.compile_size(featurizer)
      zero = Nx.tensor(0, type: :f32, backend: Nx.BinaryBackend)
      template = %{"pixel_values" => Nx.broadcast(zero, {1, 3, height, width})}
      run = Transmuter.compile(model, template, [cache_size: @image_sizes] ++ defn_options)

      fn %{"pixel_values" => pixels, "size" => size} ->
        [height, width] = Nx.to_flat_list(size)

        depth =
          run.(%{"pixel_values" => pixels})["predicted_depth"]
          |> Serving.interpolate({height, width}, :bicubic)

        %{"predicted_depth" => depth, "depth" => image(depth)}
      end
    end

    task = %{
      validate: &Featurizer.tensor/1,
      preprocess: preprocess,
      init: init,
      postprocess: fn %{"predicted_depth" => depth, "depth" => image} ->
        %{predicted_depth: depth, depth: image}
      end
    }

    Serving.each(task, opts)
  end

  # As the depth-estimation pipeline of Python transformers: the depth scaled to 0..1 with
  # its minimum and maximum, times 255, truncated to 8 bits.
  defp image(depth) do
    min = Nx.reduce_min(depth)

    depth
    |> Nx.subtract(min)
    |> Nx.divide(Nx.subtract(Nx.reduce_max(depth), min))
    |> Nx.multiply(255)
    |> Nx.as_type(:u8)
  end
end
