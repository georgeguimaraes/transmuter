defmodule Transmuter.Audio.TextToSpeech do
  @moduledoc """
  The text-to-speech task of VITS (MMS-TTS) and SpeechT5. See `Transmuter.Audio.text_to_speech/2`.

  VITS runs in one pass of its model, as `VitsModel` of Python transformers. The `RandomNormalLike` operators of the export add the noise of `noise_scale` and `noise_scale_duration`, so the seed of the request decides the audio.

  SpeechT5 does the steps of `_generate_speech` in Python transformers:

    1. The encoder runs once on the text.
    2. The decoder makes one step of the spectrogram at a time, from a step of zeros. A step gives `reduction_factor` frames of `num_mel_bins` values (`spectrum`), the sequence with its last frame added (`output_sequence_out`), and the probability that the speech ends (`prob`). The next step takes this sequence and the KV cache.
    3. The loop stops when the sum of `prob` is 0.5 or more, or after `maxlen` steps. `maxlen` is the length of the encoder output times 20 divided by `reduction_factor`, as `maxlenratio` 20 in Python. The `minlenratio` of Python is 0, so the loop has no minimum.
    4. The postnet and the vocoder (`decoder_postnet_and_vocoder`) turn the frames into audio.

  The prenet of the decoder applies dropout in each step, as in Python, so the seed of the request also decides the audio of SpeechT5. Each step draws from its own key, made from the seed and the number of the step.

  The serving runs each text as a batch of its own (`Transmuter.Serving.each/2`). The predicted durations of VITS decide the length of its audio, and the export gives no length for each row of a batch, so the audio of a padded row cannot be cut from a batch. The SpeechT5 exports take a batch of one text.
  """

  alias Transmuter.{Hub, Repo, Serving, Tokenizer}

  # The compiled models keep this many shapes: the text lengths and the audio lengths.
  @shapes 64

  # The decoder of SpeechT5 compiles a shape for each step, and keeps this many.
  @steps 1024

  # The stop rule of _generate_speech in Python transformers, with its defaults.
  @threshold 0.5
  @maxlenratio 20

  @doc false
  def text_to_speech(source, opts) do
    opts =
      Keyword.validate!(
        opts,
        [
          :variant,
          :tokenizer,
          :backend,
          :compute_type,
          :seed,
          :speaker_embedding,
          noise: true,
          defn_options: []
        ] ++ Hub.options()
      )

    repo = Repo.load!(source, Keyword.take(opts, [:variant | Hub.options()]))
    tokenizer = opts[:tokenizer] || Tokenizer.load!(repo)
    model_opts = Keyword.take(opts, [:backend, :compute_type, :noise])

    {init, speaker} =
      case repo.config["model_type"] do
        "vits" ->
          {vits(repo, model_opts), nil}

        "speecht5" ->
          speecht5(repo, model_opts)

        other ->
          raise ArgumentError, "text_to_speech runs VITS and SpeechT5, not #{inspect(other)}"
      end

    rate = sampling_rate(repo)

    preprocess = fn %{text: text, seed: seed, speaker_embedding: embedding} ->
      {%{"input_ids" => ids}, _encodings} = Tokenizer.encode(tokenizer, [text])

      tensors = %{
        "input_ids" => ids[0],
        "seed" => Nx.tensor(seed || :rand.uniform(2 ** 32), type: :s64, backend: Nx.BinaryBackend)
      }

      if embedding, do: Map.put(tensors, "speaker_embeddings", embedding), else: tensors
    end

    postprocess = fn %{"audio" => audio} -> %{audio: audio, sampling_rate: rate} end

    task = %{
      validate: &validate(&1, opts, speaker),
      preprocess: preprocess,
      init: init,
      postprocess: postprocess
    }

    Serving.each(task, opts)
  end

  defp validate(text, opts, speaker) when is_binary(text),
    do: validate(%{text: text}, opts, speaker)

  defp validate(input = %{text: text}, opts, speaker) when is_binary(text) do
    embedding = Map.get(input, :speaker_embedding, opts[:speaker_embedding])
    seed = Map.get(input, :seed, opts[:seed])

    with {:ok, embedding} <- speaker_embedding(embedding, speaker) do
      {:ok, %{text: text, seed: seed, speaker_embedding: embedding}}
    end
  end

  defp validate(input, _opts, _speaker),
    do: {:error, "expected a string or a map with :text, got: #{inspect(input)}"}

  defp speaker_embedding(nil, nil), do: {:ok, nil}
  defp speaker_embedding(_embedding, nil), do: {:error, "the model takes no speaker embedding"}

  defp speaker_embedding(nil, size),
    do: {:error, "the model needs a :speaker_embedding, a tensor of #{size} values"}

  defp speaker_embedding(embedding = %Nx.Tensor{shape: {size}}, size),
    do: {:ok, embedding |> Nx.as_type(:f32) |> Nx.backend_transfer(Nx.BinaryBackend)}

  defp speaker_embedding(embedding, size),
    do: {:error, "expected a :speaker_embedding of #{size} values, got: #{inspect(embedding)}"}

  defp sampling_rate(repo) do
    repo.config["sampling_rate"] ||
      (Hub.json(repo.source, "preprocessor_config.json", repo.opts) || %{})["sampling_rate"] ||
      raise ArgumentError, "the repository gives no sampling_rate"
  end

  defp vits(repo, model_opts) do
    model = Repo.load_component!(repo, "model", :fp32, model_opts)

    fn defn_options ->
      backend = Nx.Defn.to_backend(defn_options)
      opts = [cache_size: @shapes] ++ defn_options

      run =
        Transmuter.compile(
          model,
          Map.new(model.inputs, &{&1, template(model, &1, {1, 32})}),
          opts
        )

      fn %{"input_ids" => ids, "seed" => seed} ->
        inputs = %{
          "input_ids" => ids,
          "attention_mask" =>
            Nx.broadcast(Nx.tensor(1, backend: Nx.BinaryBackend), Nx.shape(ids)),
          Transmuter.random_input() => key(seed, backend)
        }

        %{"audio" => run.(inputs)["waveform"] |> Nx.as_type(:f32)}
      end
    end
  end

  defp speecht5(repo, model_opts) do
    load = &Repo.load_component!(repo, &1, :fp32, model_opts)

    {encoder, decoder, vocoder} =
      {load.("encoder_model"), load.("decoder_model_merged"),
       load.("decoder_postnet_and_vocoder")}

    {_type, [_batch, speaker]} = decoder.input_types["speaker_embeddings"]
    reduction = repo.config["reduction_factor"]
    cache = Enum.filter(decoder.inputs, &String.starts_with?(&1, "past_key_values."))

    init = fn defn_options ->
      backend = Nx.Defn.to_backend(defn_options)
      compile = &Transmuter.compile(&1, &2, &3 ++ defn_options)
      text = %{"input_ids" => template(encoder, "input_ids", {1, 32})}
      encode = compile.(encoder, text, cache_size: @shapes)
      frames = %{"spectrogram" => template(vocoder, "spectrogram", {64, 80})}
      vocode = compile.(vocoder, frames, cache_size: @shapes)

      # The first step takes the branch without cache, and the later steps the branch with
      # it. Each branch is a static value of use_cache_branch. The symbolic dimensions of
      # the templates are 1.
      inputs =
        Map.new(decoder.inputs -- ["use_cache_branch"], fn name ->
          {_type, dims} = decoder.input_types[name]
          {name, template(decoder, name, List.to_tuple(Enum.map(dims, &(&1 || 1))))}
        end)

      branches =
        Map.new([false, true], fn cached? ->
          flag = Nx.tensor([if(cached?, do: 1, else: 0)], type: :u8, backend: Nx.BinaryBackend)
          static = %{"use_cache_branch" => flag}
          {cached?, compile.(decoder, inputs, static_inputs: static, cache_size: @steps)}
        end)

      # An eager fold_in is many small calls to the device.
      fold_in = Nx.Defn.jit(&Transmuter.Random.fold_in/2, defn_options)

      fn %{"input_ids" => ids, "speaker_embeddings" => embedding, "seed" => seed} ->
        %{"encoder_outputs" => hidden, "encoder_attention_mask" => mask} =
          encode.(%{"input_ids" => ids})

        # The first step does not read the cache, so the cache starts with one row of zeros.
        inputs =
          Map.merge(
            Map.take(inputs, ["output_sequence" | cache]),
            %{
              "encoder_hidden_states" => hidden,
              "encoder_attention_mask" => mask,
              "speaker_embeddings" => embedding
            }
          )
          |> Map.new(fn {name, t} -> {name, Nx.backend_transfer(t, backend)} end)

        key = key(seed, backend)

        step = fn i, inputs ->
          branches[i > 1].(Map.put(inputs, Transmuter.random_input(), fold_in.(key, i)))
        end

        maxlen = trunc(Nx.axis_size(hidden, 1) * @maxlenratio / reduction)
        spectrum = decode(step, cache, inputs, 1, maxlen, [])
        %{"waveform" => waveform} = vocode.(%{"spectrogram" => Nx.concatenate(spectrum)})
        %{"audio" => waveform |> Nx.as_type(:f32) |> Nx.reshape({1, :auto})}
      end
    end

    {init, speaker}
  end

  # Step i of the decoder gives frames 2i - 1 and 2i, until the sum of the stop probabilities
  # of a step is 0.5 or more, or until maxlen. A step gives an empty present for the cache of
  # the encoder, so that cache keeps its value from the first step.
  defp decode(step, cache, inputs, i, maxlen, spectrum) do
    outputs = step.(i, inputs)
    spectrum = [outputs["spectrum"] | spectrum]

    if i >= maxlen or Nx.to_number(Nx.sum(outputs["prob"])) >= @threshold do
      Enum.reverse(spectrum)
    else
      present =
        for "past_key_values." <> rest = name <- cache,
            Map.has_key?(outputs, "present." <> rest),
            into: %{},
            do: {name, outputs["present." <> rest]}

      inputs =
        inputs
        |> Map.merge(present)
        |> Map.put("output_sequence", outputs["output_sequence_out"])

      decode(step, cache, inputs, i + 1, maxlen, spectrum)
    end
  end

  defp key(seed, backend),
    do:
      seed |> Nx.reshape({}) |> Nx.to_number() |> Nx.Random.key() |> Nx.backend_transfer(backend)

  # An input of `shape` and of the declared type. Integer inputs are ones, so a template of
  # ids and masks is a valid text.
  defp template(model, name, shape) do
    {type, _dims} = model.input_types[name]
    value = if Nx.Type.integer?(Nx.Type.normalize!(type)), do: 1, else: 0
    Nx.broadcast(Nx.tensor(value, type: type, backend: Nx.BinaryBackend), shape)
  end
end
