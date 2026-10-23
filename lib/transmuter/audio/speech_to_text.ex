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

  Long audio follows `AutomaticSpeechRecognitionPipeline` of transformers with `chunk_length_s` and `stride_length_s`:

    1. The client splits the audio into windows (`chunk_iter`). It cuts the chunks of a stream into the same windows as the whole audio, because a window waits for the first sample after it, which tells that it is not the last. Each window is one row of a batch of the serving, with its stride.
    2. The serving transcribes each window as the `generate` function of `WhisperForConditionalGeneration` does. When the tokens of a window end with two timestamps, the last segment is not complete. The serving removes the tokens after these timestamps, moves the start of the features to the last timestamp, fills the end with zeros, and transcribes again. When these timestamps are at the start of the features, the window is done with all its tokens, as when the tokens have no two timestamps together.
    3. The client merges the tokens of the windows as `_decode_asr` of the Whisper tokenizer does. It removes the timestamps in the context of each window, and joins the overlap of two token sequences at the middle of their best alignment (`_find_longest_common_sequence`).

  With timestamps, the processor of `WhisperTimeStampLogitsProcessor` selects the tokens. With word timestamps, `Transmuter.Audio.WordTimestamps` gives the end time of each token after each pass, and the merge compares these times in the overlaps, as `_decode_asr` does with `token_timestamps`.
  """

  alias Transmuter.Audio.{Featurizer, WordTimestamps}
  alias Transmuter.{Generation, GraphBreaks, Hub, Serving, Tokenizer}

  # The bytes of an f32 sample of Featurizer.audio/2.
  @bytes 4

  @doc false
  def speech_to_text(source, opts) do
    opts =
      Keyword.validate!(
        opts,
        [
          :variant,
          :tokenizer,
          :backend,
          :compute_type,
          :max_new_tokens,
          :strategy,
          :temperature,
          :top_k,
          :top_p,
          :repetition_penalty,
          :num_beams,
          :length_penalty,
          :early_stopping,
          :no_repeat_ngram_size,
          :min_length,
          :min_new_tokens,
          :seed,
          :language,
          :chunk_num_seconds,
          :context_num_seconds,
          :timestamps,
          task: :transcribe,
          compile: [],
          defn_options: []
        ] ++ Hub.options()
      )

    compile = Keyword.validate!(opts[:compile], batch_size: 1)
    mode = opts[:timestamps]

    unless mode in [nil, :segments, :words] do
      raise ArgumentError,
            "expected :timestamps to be nil, :segments or :words, got: #{inspect(mode)}"
    end

    outputs =
      if mode == :words,
        do: &(Generation.outputs(&1) ++ WordTimestamps.outputs(&1)),
        else: &Generation.outputs/1

    {repo, encoder, decoder} = Serving.load_encoder_decoder!(source, opts, outputs)
    gen = repo.generation_config
    tokenizer = opts[:tokenizer] || Tokenizer.load!(repo)
    featurizer = featurizer(repo)
    settings = settings(repo, featurizer, opts)
    prompt = prompt(gen, opts[:language], opts[:task], mode != nil)
    [input] = encoder.inputs

    max_new_tokens = Generation.max_new_tokens(repo, opts, length(prompt))
    opts = Keyword.put(opts, :max_new_tokens, max_new_tokens)
    generator = %{decoder | outputs: Generation.outputs(decoder.outputs)}
    ctx = Generation.context(repo, generator, opts, ["encoder_hidden_states"])

    ctx =
      if mode,
        do:
          Map.update!(
            ctx,
            :processors,
            &(&1 ++ [timestamp_processor(settings, gen["max_initial_timestamp_index"])])
          ),
        else: ctx

    aligner = aligner(mode, repo, decoder, settings, length(prompt), max_new_tokens)

    init = fn defn_options ->
      features = GraphBreaks.jit(&features(&1, &2, &3, featurizer.hop_length), defn_options)
      shape = {compile[:batch_size], featurizer.feature_size, settings.frames}
      template = Nx.broadcast(Nx.tensor(0.0, backend: Nx.BinaryBackend), shape)
      encode = Transmuter.compile(encoder, %{input => template}, defn_options)
      align = aligner.(defn_options)

      encode = fn features ->
        %{"last_hidden_state" => hidden} = encode.(%{input => features})
        %{"encoder_hidden_states" => hidden}
      end

      # A row goes out as its stride, then each token with its end time in feature frames,
      # as ids. The frames of the audio are the frames of the attention mask of
      # WhisperFeatureExtractor, which word timestamps read.
      fn %{"audio" => audio, "stride" => strides}, prompts, generate ->
        features = features.(audio, featurizer.filters, featurizer.window)
        strides = Nx.to_list(strides)
        hop = featurizer.hop_length
        frames = for [size, _left, _right] <- strides, do: div(size + hop - 1, hop)
        seeks = Enum.map(prompts, fn _ -> 0 end)

        pass = fn input, prompts, seeks ->
          inputs = encode.(input)
          tokens = generate.(inputs, prompts)
          times = align.(inputs, prompts, tokens, seeks, frames)
          Enum.zip_with(tokens, times, &Enum.zip/2)
        end

        for {row, stride} <-
              Enum.zip(transcribe(features, prompts, seeks, settings, pass), strides),
            do: stride ++ Enum.flat_map(row, &Tuple.to_list/1)
      end
    end

    # The windows of a tensor or a file are a list, so an error in them raises in the caller.
    # Zero bits are the zero of f32.
    preprocess = fn chunks ->
      seed = opts[:seed] || :rand.uniform(2 ** 32)

      runs =
        chunks
        |> windows(settings.chunk, settings.context)
        |> Stream.map(fn {samples, stride = {size, _left, _right}} ->
          padding = (featurizer.n_samples - size) * @bytes

          audio =
            Nx.from_binary(<<samples::binary, 0::size(padding)-unit(8)>>, :f32,
              backend: Nx.BinaryBackend
            )

          stride = Nx.tensor(Tuple.to_list(stride), backend: Nx.BinaryBackend)
          %{prompt: prompt, seed: seed, tensors: %{"audio" => audio, "stride" => stride}}
        end)

      if is_list(chunks), do: Enum.to_list(runs), else: runs
    end

    frame_seconds = settings.precision / settings.frames_per_timestamp
    # The language of the text. A translation is English.
    language =
      if gen["is_multilingual"] and opts[:task] != :translate,
        do: opts[:language] || "en",
        else: "en"

    postprocess = fn rows, _runs ->
      windows =
        Enum.map(rows, fn [size, left, right | ids] ->
          tokens =
            for [token, time] <- Enum.chunk_every(ids, 2), do: {token, time * frame_seconds}

          {tokens, Enum.map([size, left, right], &(&1 / featurizer.sampling_rate))}
        end)

      %{chunks: result(mode, merge(windows, settings), tokenizer, language)}
    end

    task = %{
      validate: &Featurizer.audio(&1, featurizer.sampling_rate),
      preprocess: preprocess,
      init: init,
      postprocess: postprocess
    }

    Serving.generation(ctx, tokenizer, task, opts)
  end

  # Without word timestamps, each token ends at the start of the window.
  defp aligner(:words, repo, decoder, settings, prompt_length, max_new_tokens) do
    WordTimestamps.aligner(repo, decoder, %{
      prompt_length: prompt_length,
      max_new_tokens: max_new_tokens,
      frames_per_timestamp: settings.frames_per_timestamp,
      positions: repo.config["max_target_positions"]
    })
  end

  defp aligner(_mode, _repo, _decoder, _settings, _prompt_length, _max_new_tokens) do
    fn _defn_options ->
      fn _inputs, _prompts, tokens, _seeks, _frames ->
        for row <- tokens, do: List.duplicate(0, length(row))
      end
    end
  end

  # The chunks of the result from the merged chunks {start, end, elements}.
  defp result(nil, chunks, tokenizer, _language) do
    text = Enum.map_join(chunks, fn {_start, _stop, elements} -> text(tokenizer, elements) end)
    [%{text: text, start_timestamp_seconds: nil, end_timestamp_seconds: nil}]
  end

  defp result(:segments, chunks, tokenizer, _language) do
    for {start, stop, elements} <- chunks do
      %{
        text: text(tokenizer, elements),
        start_timestamp_seconds: start,
        end_timestamp_seconds: stop
      }
    end
  end

  defp result(:words, chunks, tokenizer, language) do
    for {_start, _stop, elements} <- chunks,
        {text, start, stop} <- WordTimestamps.words(tokenizer, elements, language),
        do: %{text: text, start_timestamp_seconds: start, end_timestamp_seconds: stop}
  end

  defp text(tokenizer, elements),
    do: Tokenizer.decode(tokenizer, Enum.map(elements, &elem(&1, 0)))

  # The windows in samples, as chunk_length_s and stride_length_s of transformers, and the
  # settings of the timestamps. A timestamp is 0.02 s, and a feature frame is 0.01 s.
  defp settings(repo, featurizer, opts) do
    rate = featurizer.sampling_rate
    seconds = featurizer.n_samples / rate
    positions = repo.config["max_source_positions"]
    frames = div(featurizer.n_samples, featurizer.hop_length)
    chunk_seconds = opts[:chunk_num_seconds] || seconds
    chunk = round(chunk_seconds * rate)
    context = round((opts[:context_num_seconds] || chunk_seconds / 6) * rate)

    unless chunk <= featurizer.n_samples and chunk - 2 * context > 0 do
      raise ArgumentError,
            "expected :chunk_num_seconds to be at most #{seconds} and more than twice :context_num_seconds"
    end

    %{
      chunk: chunk,
      context: context,
      frames: frames,
      frames_per_timestamp: div(frames, positions),
      positions: positions,
      precision: seconds / positions,
      eos: repo.generation_config["eos_token_id"],
      timestamp_begin: repo.generation_config["no_timestamps_token_id"] + 1
    }
  end

  @doc false
  # chunk_iter of transformers on the chunks of the audio: windows of `chunk` samples every
  # `chunk - 2 * context` samples, as {samples, {size, left context, right context}}. The
  # first window has no left context and the last has no right context. A window is not the
  # last when a sample follows it, so a full window waits for the next sample. At the end,
  # the samples that are left are the last window when they are more than its left context.
  # The state is the samples from the start of the next window, their count and the left
  # context of that window. The first chunk is the samples without a copy, and the windows
  # are parts of the samples.
  def windows(chunks, chunk, context) do
    Stream.transform(
      chunks,
      fn -> {"", 0, 0} end,
      fn audio, {pending, count, left} ->
        samples = Nx.to_binary(audio)
        pending = if pending == "", do: samples, else: pending <> samples
        count = count + Nx.size(audio)

        if count > chunk,
          do: cut(pending, count, left, chunk, context, []),
          else: {[], {pending, count, left}}
      end,
      fn state = {pending, count, left} ->
        last = if count > left, do: [{pending, {count, left, 0}}], else: []
        {last, state}
      end,
      fn _state -> :ok end
    )
  end

  defp cut(samples, count, left, chunk, context, windows) when count > chunk do
    step = chunk - 2 * context
    window = {binary_part(samples, 0, chunk * @bytes), {chunk, left, context}}
    rest = binary_part(samples, step * @bytes, (count - step) * @bytes)
    cut(rest, count - step, context, chunk, context, [window | windows])
  end

  defp cut(samples, count, left, _chunk, _context, windows),
    do: {Enum.reverse(windows), {samples, count, left}}

  @doc false
  # The loop of generate in WhisperForConditionalGeneration over the 3000 frames of the window
  # of each row. Tokens that end with two timestamps leave the last segment incomplete, so
  # the next pass of that row starts at the last timestamp, and the features after the frames
  # are zeros. A row without a prompt is done. A pass gives the tokens of each row as
  # {token, time}.
  def transcribe(features, prompts, seeks, settings, pass) do
    if Enum.all?(prompts, &(&1 == [])) do
      Enum.map(prompts, fn _ -> [] end)
    else
      frames = settings.frames

      input =
        features
        |> Nx.to_batched(1)
        |> Enum.zip_with(seeks, fn row, seek ->
          row
          |> Nx.slice_along_axis(seek, frames - seek, axis: 2)
          |> Nx.pad(0.0, [{0, 0, 0}, {0, 0, 0}, {0, seek, 0}])
        end)
        |> Nx.concatenate()

      {kept, next_seeks} =
        pass.(input, prompts, seeks)
        |> Enum.zip_with(seeks, &next_pass(&1, &2, settings))
        |> Enum.unzip()

      prompts = Enum.zip_with(prompts, next_seeks, &if(&2 < frames, do: &1, else: []))
      seeks = Enum.map(next_seeks, &if(&1 < frames, do: &1, else: 0))
      Enum.zip_with(kept, transcribe(features, prompts, seeks, settings, pass), &++/2)
    end
  end

  # The tokens to keep from a pass and the seek of the next pass. The seek moves to the first
  # of the last two timestamps together, or to the start of the window without them. When
  # the seek does not move or the tokens end with a single timestamp, the pass keeps all its
  # tokens and the seek is the number of frames: the row is done.
  defp next_pass(tokens, seek, settings = %{frames: frames, timestamp_begin: begin}) do
    timestamps = Enum.map(tokens, fn {token, _time} -> token >= begin end)
    pairs = for {[true, true], i} <- Enum.with_index(Enum.chunk_every(timestamps, 2, 1), 1), do: i
    {last, _time} = if pairs == [], do: {begin, 0}, else: Enum.at(tokens, List.last(pairs) - 1)
    shift = (last - begin) * settings.frames_per_timestamp

    if shift > 0 and Enum.take(timestamps, -2) != [false, true] do
      {Enum.take(tokens, List.last(pairs) + 1), seek + shift}
    else
      {tokens, frames}
    end
  end

  # WhisperTimeStampLogitsProcessor. The state gives the bounds of the step:
  # [below, from, above] removes the tokens before `below`, the timestamps before `from` and
  # the tokens after `above`. Then, when the timestamps together are more probable than each
  # text token, the step selects a timestamp.
  defp timestamp_processor(%{timestamp_begin: begin, eos: eos}, max_initial) do
    state = fn %{generated: generated} ->
      Nx.tensor(bounds(generated, begin, eos, max_initial), type: :s32, backend: Nx.BinaryBackend)
    end

    process = fn logits, bounds ->
      ids = Nx.iota(Nx.shape(logits))
      min = Nx.Constants.min_finite(:f32)
      timestamp? = Nx.greater_equal(ids, begin)

      removed =
        ids
        |> Nx.less(bounds[0])
        |> Nx.logical_or(Nx.equal(ids, begin - 1))
        |> Nx.logical_or(Nx.logical_and(timestamp?, Nx.less(ids, bounds[1])))
        |> Nx.logical_or(Nx.greater(ids, bounds[2]))

      logits = Nx.select(removed, min, logits)
      logprobs = Nx.subtract(logits, Nx.logsumexp(logits))
      timestamps = Nx.logsumexp(Nx.select(timestamp?, logprobs, min))
      text = Nx.reduce_max(Nx.select(timestamp?, min, logprobs))

      Nx.select(
        Nx.logical_and(Nx.greater(timestamps, text), Nx.logical_not(timestamp?)),
        min,
        logits
      )
    end

    {state, process}
  end

  @no_limit 2 ** 31 - 1

  # The first token is a timestamp of at most `max_initial`.
  defp bounds([], begin, _eos, nil), do: [begin, begin, @no_limit]
  defp bounds([], begin, _eos, max_initial), do: [begin, begin, begin + max_initial]

  # Timestamps come in pairs, except before eos, and do not decrease. A pair is followed by
  # text, a single timestamp by a timestamp or eos.
  defp bounds(generated = [last | rest], begin, eos, _max_initial) do
    last? = last >= begin
    penultimate? = match?([], rest) or hd(rest) >= begin
    single? = last? and not penultimate?

    from =
      case Enum.find(generated, &(&1 >= begin)) do
        nil -> begin
        timestamp when single? -> timestamp
        timestamp -> timestamp + 1
      end

    [
      if(single?, do: eos, else: 0),
      from,
      if(last? and penultimate?, do: begin - 1, else: @no_limit)
    ]
  end

  # _decode_asr of the Whisper tokenizer: the windows are {tokens, [size, left, right]} in
  # seconds, and a token is {id, end time in the window}. Returns the chunks as
  # {start, end, elements}, with timestamps in seconds. An element is a text token with its
  # start and end, {id, {start, end}}.
  defp merge(windows, settings) do
    state = %{chunks: [], start: nil, previous: [], offset: 0.0, skip: false}
    state = Enum.reduce(windows, state, &merge_window(&1, &2, settings))

    chunks =
      if state.previous == [],
        do: state.chunks,
        else: [{state.start, nil, longest_common_sequence(state.previous)} | state.chunks]

    Enum.reverse(chunks)
  end

  defp merge_window({tokens, [size, left, right]}, state, settings) do
    %{timestamp_begin: begin, precision: precision} = settings
    state = %{state | offset: state.offset - left}
    first = if left != 0, do: left / precision + begin, else: begin
    {ids, times} = Enum.unzip(tokens)

    # Timestamps in the right context do not end a chunk, and the last timestamp never does.
    last =
      if right != 0 do
        ids
        |> Enum.reverse()
        |> Enum.filter(&(&1 >= begin))
        |> Enum.reduce_while(nil, fn token, last ->
          if last != nil and (token - begin) * precision < size - right,
            do: {:halt, last},
            else: {:cont, token}
        end)
      end

    # A token starts at the end of the token before it in the window.
    stamps =
      Enum.zip_with([0.0 | times], times, fn start, stop ->
        {Float.round(start + state.offset, 2), Float.round(stop + state.offset, 2)}
      end)

    window = %{
      ids: List.to_tuple(ids),
      stamps: List.to_tuple(stamps),
      first: first,
      last: last,
      max: 0.0,
      penultimate: 0.0,
      before: 0.0,
      current: []
    }

    {state, window} =
      ids
      |> Enum.with_index()
      |> Enum.reduce({state, window}, &merge_token(&1, &2, settings))

    state = %{state | offset: state.offset + (size - right)}

    cond do
      window.current != [] ->
        %{state | previous: state.previous ++ [Enum.reverse(window.current)]}

      Enum.all?(state.previous, &(&1 == [])) ->
        %{state | start: nil, previous: []}

      true ->
        state
    end
  end

  defp merge_token({token, i}, {state, window}, settings) do
    %{timestamp_begin: begin, eos: eos} = settings

    cond do
      token >= begin -> merge_timestamp(token, i, state, window, settings)
      token >= eos -> {state, window}
      true -> {state, %{window | current: [{token, elem(window.stamps, i)} | window.current]}}
    end
  end

  # The tokens of a window can have several passes of generation, and the timestamps of each
  # pass start again from 0. `before` is the length of the passes before.
  defp merge_timestamp(token, i, state, window, settings) do
    %{timestamp_begin: begin, precision: precision, positions: positions} = settings
    timestamp = (token - begin) * precision

    window =
      cond do
        timestamp >= window.max ->
          window

        i >= 2 and not (elem(window.ids, i - 1) >= begin and elem(window.ids, i - 2) >= begin) ->
          %{window | before: window.before + precision * positions}

        true ->
          %{window | max: window.penultimate, before: window.before + window.penultimate}
      end

    window = %{window | penultimate: window.max, max: timestamp}
    time = Float.round(timestamp + state.offset + window.before, 2)

    cond do
      window.last && token >= window.last ->
        {%{state | skip: true}, window}

      state.skip or (state.previous != [] and token < window.first) ->
        {%{state | skip: false}, window}

      state.start == nil ->
        {%{state | start: time}, window}

      time == state.start ->
        {state, window}

      true ->
        tokens = longest_common_sequence(state.previous ++ [Enum.reverse(window.current)])
        chunks = [{state.start, time, tokens} | state.chunks]
        {%{state | chunks: chunks, start: nil, previous: []}, %{window | current: []}}
    end
  end

  # _find_longest_common_sequence of the Whisper tokenizer: slides each sequence over the
  # merged tokens, takes the alignment with the most matches (more than one), and cuts the
  # overlap at its middle. Two elements match when their ids are equal and the
  # {start, end} of the left one is not after the right one. Without word timestamps, each
  # element has the offset of its window, which grows from window to window, so only the
  # ids decide.
  defp longest_common_sequence([first | rest]) do
    {merged, last} =
      Enum.reduce(rest, {[], first}, fn right, {merged, left} ->
        {left_start, left_stop, right_start, right_stop} = best_overlap(left, right)
        left_part = Enum.take(left, div(left_stop + left_start, 2))
        {[left_part | merged], Enum.drop(right, div(right_stop + right_start, 2))}
      end)

    Enum.concat(Enum.reverse([last | merged]))
  end

  defp best_overlap(left, right) do
    {l, r} = {length(left), length(right)}
    {left_ids, right_ids} = {List.to_tuple(left), List.to_tuple(right)}

    {_score, indices} =
      Enum.reduce(1..(l + r - 1)//1, {0.0, {l, l, 0, 0}}, fn i, {best, indices} ->
        {left_start, left_stop} = {max(0, l - i), min(l, l + r - i)}
        right_start = max(0, i - l)

        matches =
          Enum.count(0..(left_stop - left_start - 1)//1, fn k ->
            {id, stamp} = elem(left_ids, left_start + k)
            {other_id, other_stamp} = elem(right_ids, right_start + k)
            id == other_id and stamp <= other_stamp
          end)

        score = matches / i + i / 10_000.0

        if matches > 1 and score > best,
          do: {score, {left_start, left_stop, right_start, min(r, i)}},
          else: {best, indices}
      end)

    indices
  end

  @doc """
  Returns the settings of the featurizer of `repo` from `preprocessor_config.json`, with the defaults of `WhisperFeatureExtractor`. `:filters` are the mel filters {bins, mels} and `:window` is the periodic Hann window, as `Nx.BinaryBackend` tensors.
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
    bins = div(n_fft, 2) + 1

    %{
      feature_size: config["feature_size"],
      sampling_rate: config["sampling_rate"],
      hop_length: config["hop_length"],
      n_samples: config["n_samples"],
      filters:
        Featurizer.mel_filters(bins, config["feature_size"], config["sampling_rate"],
          min_frequency: 0.0,
          max_frequency: 8000.0,
          mel_scale: :slaney,
          norm: :slaney
        ),
      window: Featurizer.hann_window(n_fft, true)
    }
  end

  @doc """
  Returns the log-mel features {batch, mels, frames} of `audio` {batch, samples}, with the `filters` and the `window` of `featurizer/1`. The audio must already have its length of 30 s.
  """
  def features(audio, filters, window, hop_length) do
    power = Featurizer.power_spectrogram(audio, window, hop_length)
    power = Nx.slice_along_axis(power, 0, Nx.axis_size(power, 1) - 1, axis: 1)
    mel = power |> Nx.dot([2], filters, [0]) |> Nx.transpose(axes: [0, 2, 1])

    log = mel |> Nx.max(1.0e-10) |> Nx.log() |> Nx.divide(:math.log(10))
    floor = log |> Nx.reduce_max(axes: [1, 2], keep_axes: true) |> Nx.subtract(8.0)
    log |> Nx.max(floor) |> Nx.add(4.0) |> Nx.divide(4.0)
  end

  # The prompt of transformers: the start token, the language and the task of a multilingual
  # model, and without timestamps the token that turns them off.
  defp prompt(gen, language, task, timestamps?) do
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

    [gen["decoder_start_token_id"]] ++
      tokens ++ if(timestamps?, do: [], else: [gen["no_timestamps_token_id"]])
  end
end
