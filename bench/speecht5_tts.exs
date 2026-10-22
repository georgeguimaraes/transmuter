# Text to speech with SpeechT5: the encoder, one decoder step for each two frames, and the vocoder.
# Usage: mix run bench/speecht5_tts.exs [repeats] [emlx|exla]
#
# emlx (the default) runs on the GPU of a Mac, and exla runs on the host client.
#
# The serving runs in its own process, so the compiled shapes stay between calls. The first call compiles,
# and the later calls show the time of a warm serving. The telemetry of the decoder gives the number of
# steps and the time of the loop: from the first decoder call to the vocoder call.

repo = "Xenova/speecht5_tts"
text = "Hello, my dog is cute. It likes to play in the park."
seed = 42
{repeats, compiler} =
  case System.argv() do
    [] -> {3, "emlx"}
    [repeats] -> {String.to_integer(repeats), "emlx"}
    [repeats, compiler] -> {String.to_integer(repeats), compiler}
  end

{backend, defn_options} =
  case compiler do
    "emlx" -> {{EMLX.Backend, device: :gpu}, [compiler: EMLX, device: :gpu]}
    "exla" -> {{EXLA.Backend, client: :host}, [compiler: EXLA, client: :host]}
  end

speaker = Nx.from_binary(File.read!("test/fixtures/speaker_embeddings.f32"), :f32)

serving =
  Transmuter.Audio.text_to_speech({:hf, repo},
    speaker_embedding: speaker,
    seed: seed,
    backend: backend,
    defn_options: defn_options
  )

table = :ets.new(:speecht5_bench, [:public, :duplicate_bag])

record = fn event, measurements, %{source: {:hf, _repo, file}}, _config ->
  :ets.insert(table, {event, Path.basename(file, ".onnx"), measurements, System.monotonic_time()})
end

:telemetry.attach_many(
  "speecht5-bench",
  [[:transmuter, :call, :start], [:transmuter, :call, :stop], [:transmuter, :segment, :compile, :stop]],
  record,
  nil
)

ms = &System.convert_time_unit(&1, :native, :microsecond) / 1000

{us, {:ok, _pid}} = :timer.tc(fn -> Nx.Serving.start_link(serving: serving, name: SpeechT5Bench) end)
IO.puts("start (loads and compiles the template shapes): #{div(us, 1000)}ms")

speak = fn label ->
  :ets.delete_all_objects(table)
  {us, %{audio: audio}} = :timer.tc(fn -> Nx.Serving.batched_run(SpeechT5Bench, text) end)
  events = :ets.tab2list(table)

  starts =
    events
    |> Enum.filter(&match?({[:transmuter, :call, :start], _model, _m, _time}, &1))
    |> Enum.group_by(&elem(&1, 1), &elem(&1, 3))

  steps = length(starts["decoder_model_merged"])
  loop = Enum.min(starts["decoder_postnet_and_vocoder"]) - Enum.min(starts["decoder_model_merged"])
  compiles = for {[:transmuter, :segment, :compile, :stop], _model, %{duration: d}, _time} <- events, do: d
  hash = :crypto.hash(:sha256, Nx.to_binary(audio)) |> Base.encode16() |> binary_part(0, 16)

  IO.puts(
    "#{label}: #{div(us, 1000)}ms, #{steps} steps, #{Float.round(ms.(loop) / steps, 1)}ms per step, " <>
      "#{length(compiles)} compiles in #{round(ms.(Enum.sum(compiles)))}ms, " <>
      "#{Float.round(Nx.size(audio) / 16_000, 2)} s of audio, sha256 #{hash}"
  )
end

speak.("first call")
for i <- 1..repeats, do: speak.("repeat #{i}")
