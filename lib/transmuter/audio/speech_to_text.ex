defmodule Transmuter.Audio.SpeechToText do
  @moduledoc """
  The speech-to-text task of Whisper. See `Transmuter.Audio.speech_to_text/2`.

  The featurizer does the steps of `WhisperFeatureExtractor` in Python transformers, with the settings of `preprocessor_config.json`:

    1. Pad the audio with zeros to 30 s (`n_samples`).
    2. Pad 200 samples (`n_fft / 2`) at each end with a reflection of the audio.
    3. Take the power spectrum of frames of 400 samples (`n_fft`) every 160 samples (`hop_length`), with a periodic Hann window.
    4. Remove the last frame, so that 30 s give 3000 frames.
    5. Multiply by the mel filters with the Slaney scale and norm from 0 Hz to 8000 Hz (`feature_size` filters, 80 or 128).
    6. Take the log10 of the values, at least 1.0e-10, clamp to the maximum minus 8, and give `(x + 4) / 4`.

  The serving compiles the featurizer and the encoder, so each 30 s of audio runs one compiled shape.
  """

  alias Transmuter.{Generation, Hub, Serving, Tokenizer}
  alias Transmuter.Ops.Signal

  @doc false
  def speech_to_text(source, opts) do
    opts =
      Keyword.validate!(opts, [
        :variant,
        :cache_dir,
        :tokenizer,
        :backend,
        :compute_type,
        :max_new_tokens,
        :strategy,
        :temperature,
        :top_k,
        :top_p,
        :repetition_penalty,
        :seed,
        :language,
        task: :transcribe,
        defn_options: []
      ])

    {repo, encoder, decoder} = Serving.load_encoder_decoder!(source, opts)
    ctx = Generation.context(repo, decoder, opts, ["encoder_hidden_states"])
    tokenizer = opts[:tokenizer] || Tokenizer.load!(repo)
    featurizer = featurizer(repo)
    prompt = prompt(repo.generation_config, opts[:language], opts[:task])
    [input] = encoder.inputs

    init = fn defn_options ->
      features = Nx.Defn.jit(&features(&1, &2, &3, featurizer.hop_length), defn_options)
      frames = div(featurizer.n_samples, featurizer.hop_length)
      shape = {1, featurizer.feature_size, frames}
      template = Nx.broadcast(Nx.tensor(0.0, backend: Nx.BinaryBackend), shape)
      encode = Transmuter.compile(encoder, %{input => template}, defn_options)

      fn %{"audio" => audio} ->
        features = features.(audio, featurizer.filters, featurizer.window)
        %{"last_hidden_state" => hidden} = encode.(%{input => features})
        %{"encoder_hidden_states" => hidden}
      end
    end

    preprocess = fn audio ->
      seed = opts[:seed] || :rand.uniform(2 ** 32)
      padding = featurizer.n_samples - Nx.size(audio)

      %{
        "audio" => Nx.pad(Nx.as_type(audio, :f32), 0.0, [{0, padding, 0}]),
        "prompt" => Nx.tensor(prompt, type: :s64),
        "seed" => Nx.tensor(seed, type: :s64)
      }
    end

    postprocess = fn ids, _inputs ->
      text = Tokenizer.decode(tokenizer, ids)
      %{chunks: [%{text: text, start_timestamp_seconds: nil, end_timestamp_seconds: nil}]}
    end

    validate = &validate(&1, featurizer)
    Serving.generation(ctx, init, validate, preprocess, postprocess, opts[:defn_options])
  end

  @doc """
  Returns the settings of the featurizer of `repo` from `preprocessor_config.json`, with the defaults of `WhisperFeatureExtractor`. `:filters` are the mel filters {bins, mels} and `:window` is the Hann window, as `Nx.BinaryBackend` tensors.
  """
  def featurizer(repo) do
    config =
      Map.merge(
        %{
          "feature_size" => 80,
          "sampling_rate" => 16_000,
          "hop_length" => 160,
          "n_fft" => 400,
          "n_samples" => 480_000
        },
        Hub.json(repo.source, "preprocessor_config.json", repo.opts) || %{}
      )

    n_fft = config["n_fft"]
    window = for i <- 0..(n_fft - 1), do: 0.5 - 0.5 * :math.cos(2 * :math.pi() * i / n_fft)

    %{
      feature_size: config["feature_size"],
      sampling_rate: config["sampling_rate"],
      hop_length: config["hop_length"],
      n_samples: config["n_samples"],
      filters: mel_filters(n_fft, config["feature_size"], config["sampling_rate"]),
      window: Nx.tensor(window, type: :f32, backend: Nx.BinaryBackend)
    }
  end

  # mel_filter_bank of transformers with the Slaney scale and norm, from 0 Hz to 8000 Hz as in
  # WhisperFeatureExtractor. Filter m rises from point m to point m + 1 and falls to m + 2.
  defp mel_filters(n_fft, mels, sampling_rate) do
    bins = div(n_fft, 2) + 1
    max_mel = hz_to_mel(8000.0)
    points = List.to_tuple(for i <- 0..(mels + 1), do: mel_to_hz(max_mel * i / (mels + 1)))

    rows =
      for bin <- 0..(bins - 1) do
        hz = div(sampling_rate, 2) * bin / (bins - 1)

        for m <- 0..(mels - 1) do
          {low, center, high} = {elem(points, m), elem(points, m + 1), elem(points, m + 2)}
          rise = (hz - low) / (center - low)
          fall = (high - hz) / (high - center)
          max(0.0, min(rise, fall)) * 2.0 / (high - low)
        end
      end

    Nx.tensor(rows, type: :f32, backend: Nx.BinaryBackend)
  end

  defp hz_to_mel(hz) when hz >= 1000.0, do: 15.0 + :math.log(hz / 1000.0) * 27.0 / :math.log(6.4)
  defp hz_to_mel(hz), do: 3.0 * hz / 200.0

  defp mel_to_hz(mel) when mel >= 15.0,
    do: 1000.0 * :math.exp(:math.log(6.4) / 27.0 * (mel - 15.0))

  defp mel_to_hz(mel), do: 200.0 * mel / 3.0

  @doc """
  Returns the log-mel features {batch, mels, frames} of `audio` {batch, samples}, with the `filters` and the `window` of `featurizer/1`. The audio must already have its length of 30 s.
  """
  def features(audio, filters, window, hop_length) do
    n_fft = Nx.axis_size(window, 0)
    pad = div(n_fft, 2)
    samples = Nx.axis_size(audio, 1)
    left = audio |> Nx.slice_along_axis(1, pad, axis: 1) |> Nx.reverse(axes: [1])
    right = audio |> Nx.slice_along_axis(samples - pad - 1, pad, axis: 1) |> Nx.reverse(axes: [1])
    signal = Nx.concatenate([left, audio, right], axis: 1) |> Nx.new_axis(2)

    power =
      signal |> Signal.stft(window, n_fft, hop_length, true) |> Nx.pow(2) |> Nx.sum(axes: [3])

    power = Nx.slice_along_axis(power, 0, Nx.axis_size(power, 1) - 1, axis: 1)
    mel = power |> Nx.dot([2], filters, [0]) |> Nx.transpose(axes: [0, 2, 1])

    log = mel |> Nx.max(1.0e-10) |> Nx.log() |> Nx.divide(:math.log(10))
    floor = log |> Nx.reduce_max(axes: [1, 2], keep_axes: true) |> Nx.subtract(8.0)
    log |> Nx.max(floor) |> Nx.add(4.0) |> Nx.divide(4.0)
  end

  # The prompt of transformers without timestamps: the start token, the language and the task
  # of a multilingual model, and the token that turns timestamps off.
  defp prompt(gen, language, task) do
    tokens =
      if gen["is_multilingual"] do
        language = language || "en"

        [
          gen["lang_to_id"]["<|#{language}|>"] ||
            raise(ArgumentError, "the model does not know the language #{inspect(language)}"),
          gen["task_to_id"]["#{task}"] ||
            raise(
              ArgumentError,
              "expected :task to be :transcribe or :translate, got: #{inspect(task)}"
            )
        ]
      else
        unless language in [nil, "en"] and task == :transcribe do
          raise ArgumentError,
                "the model is English-only, so it does not take a :language or :task other than English transcription"
        end

        []
      end

    [gen["decoder_start_token_id"]] ++ tokens ++ [gen["no_timestamps_token_id"]]
  end

  defp validate(audio = %Nx.Tensor{shape: {samples}}, featurizer) do
    if samples <= featurizer.n_samples do
      {:ok, Nx.backend_transfer(audio, Nx.BinaryBackend)}
    else
      seconds = div(featurizer.n_samples, featurizer.sampling_rate)

      {:error,
       "the audio has #{samples} samples, and speech-to-text takes up to #{seconds} s (#{featurizer.n_samples} samples). Split longer audio into parts"}
    end
  end

  defp validate(input, _featurizer),
    do: {:error, "expected a tensor of PCM samples {samples}, got: #{inspect(input)}"}
end
