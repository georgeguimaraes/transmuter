defmodule Transmuter.Audio.Featurizer do
  @moduledoc """
  Turns audio into the `input_values` input of an audio model, with the `preprocessor_config.json` of a model repository.

      featurizer = Transmuter.Audio.Featurizer.load!({:hf, "Xenova/ast-finetuned-audioset-10-10-0.4593"})
      Transmuter.Audio.Featurizer.process(featurizer, [audio], backend: EMLX.Backend)
      #=> #Nx.Tensor<f32[1][1024][128]>

  The audio is a tensor of PCM samples `{samples}` at the sampling rate of the model, of any float type. `feature_extractor_type` in the configuration selects the steps of the feature extractor of Python transformers:

    * `"Wav2Vec2FeatureExtractor"` - the samples. When `do_normalize` is `true`, each audio gets a mean of 0 and a variance of 1: `(x - mean) / sqrt(variance + 1.0e-7)`. The result has the shape `{batch, samples}`, so the shape depends on the audio.

    * `"ASTFeatureExtractor"` - the log-mel filter bank of Kaldi. The result has the shape `{batch, max_length, num_mel_bins}` for all audio. The steps are those of the numpy path of Python transformers:

      1. Cut the audio into frames of 400 samples every 160 samples. The last frame ends in the audio.
      2. In each frame, subtract the mean, apply a preemphasis of 0.97 and multiply by a symmetric Hann window.
      3. Take the power spectrum of 512 points, with zeros after the 400 samples.
      4. Multiply by `num_mel_bins` mel filters from 20 Hz to half of the sampling rate. The filters are triangles on the Kaldi mel scale.
      5. Take the natural log of the values, at least 1.1920929e-7.
      6. Keep the first `max_length` frames. Shorter audio gets frames of zeros after its end.
      7. When `do_normalize` is `true`, give `(x - mean) / (2 * std)` with `mean` and `std` of the configuration.

  When torchaudio is installed, Python transformers computes the AST features with `torchaudio.compliance.kaldi.fbank` in place of the numpy path.
  """

  alias Transmuter.{Hub, Repo}
  alias Transmuter.Ops.Signal

  defstruct [:type, :sampling_rate, :normalize, :frames, :mean, :std, :window, :dft, :filters]

  # The constants of ASTFeatureExtractor.
  @frame_length 400
  @hop_length 160
  @fft_length 512
  @preemphasis 0.97
  @mel_floor 1.192092955078125e-07

  @doc """
  Loads the featurizer of a `Transmuter.Repo` or of a source (`{:hf, repo}` or a local directory) from its `preprocessor_config.json`.

  Options go to `Transmuter.Repo.load!/2` when the argument is a source.
  """
  def load!(repo_or_source, opts \\ [])

  def load!(repo = %Repo{}, _opts) do
    config =
      Hub.json(repo.source, "preprocessor_config.json", repo.opts) ||
        raise ArgumentError, "#{inspect(repo.source)} has no preprocessor_config.json"

    new(config)
  end

  def load!(source, opts), do: load!(Repo.load!(source, opts))

  @doc """
  Builds a featurizer from the map of a `preprocessor_config.json`, with the defaults of Python transformers.
  """
  def new(config = %{"feature_extractor_type" => "Wav2Vec2FeatureExtractor"}) do
    %__MODULE__{
      type: :wav2vec2,
      sampling_rate: Map.get(config, "sampling_rate", 16_000),
      normalize: Map.get(config, "do_normalize", true)
    }
  end

  def new(config = %{"feature_extractor_type" => "ASTFeatureExtractor"}) do
    rate = Map.get(config, "sampling_rate", 16_000)
    bins = div(@fft_length, 2) + 1

    %__MODULE__{
      type: :ast,
      sampling_rate: rate,
      normalize: Map.get(config, "do_normalize", true),
      frames: Map.get(config, "max_length", 1024),
      mean: Map.get(config, "mean", -4.2677393),
      std: Map.get(config, "std", 4.5689974),
      window: hann_window(@frame_length, false),
      dft: dft(@frame_length, @fft_length, bins),
      filters:
        mel_filters(bins, Map.get(config, "num_mel_bins", 128), rate,
          min_frequency: 20,
          max_frequency: div(rate, 2),
          mel_scale: :kaldi,
          triangularize_in_mel_space: true
        )
    }
  end

  def new(config) do
    raise ArgumentError,
          "expected feature_extractor_type in preprocessor_config.json to be ASTFeatureExtractor or Wav2Vec2FeatureExtractor, got: #{inspect(config["feature_extractor_type"])}"
  end

  @doc """
  Returns the shape of the features of one audio that `process/3` gives, or `nil` when the shape depends on the audio.
  """
  def shape(%__MODULE__{type: :ast, frames: frames, filters: filters}),
    do: {frames, Nx.axis_size(filters, 1)}

  def shape(%__MODULE__{type: :wav2vec2}), do: nil

  @doc """
  Turns a list of audio into the features `{batch, ...}` on the `:backend` of `opts`. Without `:backend`, the featurizer uses `Nx.default_backend/0`. For a featurizer whose `shape/1` is `nil`, the audio of a list must have one length.
  """
  def process(featurizer, audio, opts \\ []) do
    features(featurizer, audio, opts[:backend] || Nx.default_backend())
  end

  defp features(%__MODULE__{type: :wav2vec2, normalize: normalize?}, audio, backend) do
    values = audio |> Nx.stack() |> Nx.backend_copy(backend) |> Nx.as_type(:f32)

    if normalize? do
      mean = Nx.mean(values, axes: [1], keep_axes: true)
      variance = Nx.variance(values, axes: [1], keep_axes: true)
      values |> Nx.subtract(mean) |> Nx.divide(Nx.sqrt(Nx.add(variance, 1.0e-7)))
    else
      values
    end
  end

  defp features(featurizer = %__MODULE__{type: :ast, frames: frames}, audio, backend) do
    on_device = &Nx.backend_copy(&1, backend)
    samples = @frame_length + @hop_length * (frames - 1)
    counts = Enum.map(audio, &min(frame_count(Nx.size(&1)), frames))
    signal = audio |> Enum.map(&fit(&1, samples)) |> Nx.stack() |> on_device.()

    framed =
      signal
      |> Nx.as_type(:f32)
      |> Nx.new_axis(2)
      |> Signal.frame(@frame_length, @hop_length, frames)
      |> Nx.squeeze(axes: [3])

    centered = Nx.subtract(framed, Nx.mean(framed, axes: [2], keep_axes: true))
    first = Nx.slice_along_axis(centered, 0, 1, axis: 2)
    previous = Nx.slice_along_axis(centered, 0, @frame_length - 1, axis: 2)
    rest = Nx.slice_along_axis(centered, 1, @frame_length - 1, axis: 2)

    emphasized =
      Nx.concatenate(
        [
          Nx.multiply(first, 1 - @preemphasis),
          Nx.subtract(rest, Nx.multiply(previous, @preemphasis))
        ],
        axis: 2
      )

    windowed = Nx.multiply(emphasized, on_device.(featurizer.window))
    {cos, sin} = featurizer.dft
    real = Nx.dot(windowed, [2], on_device.(cos), [0])
    imaginary = Nx.dot(windowed, [2], on_device.(sin), [0])
    power = Nx.add(Nx.pow(real, 2), Nx.pow(imaginary, 2))

    log =
      power
      |> Nx.dot([2], on_device.(featurizer.filters), [0])
      |> Nx.max(@mel_floor)
      |> Nx.log()

    counts = counts |> Nx.tensor(backend: backend) |> Nx.reshape({:auto, 1, 1})
    present = Nx.less(Nx.iota({1, frames, 1}, backend: backend), counts)
    features = Nx.multiply(log, present)

    if featurizer.normalize,
      do: features |> Nx.subtract(featurizer.mean) |> Nx.divide(2 * featurizer.std),
      else: features
  end

  # The frames of 400 samples every 160 samples that end in the audio.
  defp frame_count(samples) when samples < @frame_length, do: 0
  defp frame_count(samples), do: div(samples - @frame_length, @hop_length) + 1

  # Cuts or pads the audio with zeros to `samples`. Nx.slice and Nx.pad on Nx.BinaryBackend
  # go through a list of the samples, so the audio is cut from the binary. Zero bits are the
  # zero of each type.
  defp fit(audio, samples) do
    type = {_kind, bits} = Nx.type(audio)
    binary = Nx.to_binary(audio)
    take = min(bit_size(binary), samples * bits)
    <<kept::bitstring-size(^take), _::bitstring>> = binary
    Nx.from_binary(<<kept::bitstring, 0::size(samples * bits - take)>>, type)
  end

  # The real and imaginary parts {size, bins} of the DFT of n points of a frame of `size`
  # samples, with zeros after the frame.
  defp dft(size, n, bins) do
    angles = for k <- 0..(size - 1), j <- 0..(bins - 1), do: 2 * :math.pi() * rem(k * j, n) / n

    for fun <- [&:math.cos/1, &(-:math.sin(&1))] do
      angles
      |> Enum.map(fun)
      |> Nx.tensor(type: :f32, backend: Nx.BinaryBackend)
      |> Nx.reshape({size, bins})
    end
    |> List.to_tuple()
  end

  @doc """
  Returns the Hann window of `size` samples as an f32 tensor. A periodic window, as `torch.hann_window`, divides by `size`. A symmetric window, as `numpy.hanning`, divides by `size - 1`.
  """
  def hann_window(size, periodic?) do
    denominator = if periodic?, do: size, else: size - 1

    for(i <- 0..(size - 1), do: 0.5 - 0.5 * :math.cos(2 * :math.pi() * i / denominator))
    |> Nx.tensor(type: :f32, backend: Nx.BinaryBackend)
  end

  @doc """
  Returns the mel filters `{bins, mels}` of `mel_filter_bank` in Python transformers, for `bins` frequency bins from 0 Hz to half of `sampling_rate`. Filter `m` is a triangle that rises from point `m` to point `m + 1` and falls to point `m + 2`. The points are evenly spaced on the mel scale from `:min_frequency` to `:max_frequency`.

  ## Options

    * `:min_frequency` and `:max_frequency` - in Hz.

    * `:mel_scale` - `:slaney` or `:kaldi`.

    * `:norm` - `:slaney` multiplies each filter by 2 divided by its width, so each filter has about the same energy. The default is `nil`.

    * `:triangularize_in_mel_space` - `true` makes the triangles on the mel scale, as Kaldi does. The default is `false`, which makes the triangles in Hz.
  """
  def mel_filters(bins, mels, sampling_rate, opts) do
    opts =
      Keyword.validate!(opts, [
        :min_frequency,
        :max_frequency,
        :mel_scale,
        norm: nil,
        triangularize_in_mel_space: false
      ])

    scale = opts[:mel_scale]
    low = hz_to_mel(opts[:min_frequency], scale)
    high = hz_to_mel(opts[:max_frequency], scale)
    mel_points = for i <- 0..(mels + 1), do: low + (high - low) * i / (mels + 1)

    {points, frequencies} =
      if opts[:triangularize_in_mel_space] do
        width = sampling_rate / ((bins - 1) * 2)
        {mel_points, for(bin <- 0..(bins - 1), do: hz_to_mel(width * bin, scale))}
      else
        hz = for bin <- 0..(bins - 1), do: div(sampling_rate, 2) * bin / (bins - 1)
        {Enum.map(mel_points, &mel_to_hz(&1, scale)), hz}
      end

    points = List.to_tuple(points)

    rows =
      for frequency <- frequencies do
        for m <- 0..(mels - 1) do
          {low, center, high} = {elem(points, m), elem(points, m + 1), elem(points, m + 2)}
          rise = (frequency - low) / (center - low)
          fall = (high - frequency) / (high - center)
          max(0.0, min(rise, fall)) * norm(opts[:norm], high - low)
        end
      end

    Nx.tensor(rows, type: :f32, backend: Nx.BinaryBackend)
  end

  defp norm(nil, _width), do: 1.0
  defp norm(:slaney, width), do: 2.0 / width

  defp hz_to_mel(hz, :kaldi), do: 1127.0 * :math.log(1.0 + hz / 700.0)

  defp hz_to_mel(hz, :slaney) when hz >= 1000.0,
    do: 15.0 + :math.log(hz / 1000.0) * 27.0 / :math.log(6.4)

  defp hz_to_mel(hz, :slaney), do: 3.0 * hz / 200.0

  defp mel_to_hz(mel, :kaldi), do: 700.0 * (:math.exp(mel / 1127.0) - 1.0)

  defp mel_to_hz(mel, :slaney) when mel >= 15.0,
    do: 1000.0 * :math.exp(:math.log(6.4) / 27.0 * (mel - 15.0))

  defp mel_to_hz(mel, :slaney), do: 200.0 * mel / 3.0

  @doc """
  Accepts a tensor of PCM samples `{samples}` as one audio, and moves it to `Nx.BinaryBackend`.
  """
  def tensor(audio = %Nx.Tensor{shape: {_samples}}),
    do: {:ok, Nx.backend_transfer(audio, Nx.BinaryBackend)}

  def tensor(input),
    do: {:error, "expected a tensor of PCM samples {samples}, got: #{inspect(input)}"}
end
