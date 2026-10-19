# Text to speech with Kokoro-82M on EMLX, compiled with graph breaks.
# Usage: mix run bench/kokoro_tts.exs [output.wav]
#
# The length of the audio comes from the predicted duration of each phoneme, so each
# new length compiles the decoder again. This example rounds the total duration up to
# a multiple of 8 frames (200 ms) by making the final pause longer. Then sentences of
# a similar length share one compiled decoder. The rounding changes the audio a little,
# because the decoder normalizes over the whole utterance, including the longer pause.

defmodule KokoroBuckets do
  alias Transmuter.Onnx.{AttributeProto, NodeProto}

  # The int64 phoneme durations, shape {1, tokens}, in the Kokoro v1.0 ONNX export.
  @durations "/encoder/Cast_output_0"

  # Adds nodes that round the total duration up to a multiple of `bucket` frames by
  # adding the difference to the last token, the end-of-text pad.
  def round_durations(model, bucket) do
    constants = %{
      "bucket.minus_one" => bucket - 1,
      "bucket.size" => bucket,
      "bucket.zero" => [0],
      "bucket.last" => [-1],
      "bucket.end" => [2 ** 62],
      "bucket.axis" => [1]
    }

    added = [
      node("ReduceSum", [@durations], "bucket.total", keepdims: 0),
      node("Add", ["bucket.total", "bucket.minus_one"], "bucket.up"),
      node("Div", ["bucket.up", "bucket.size"], "bucket.count"),
      node("Mul", ["bucket.count", "bucket.size"], "bucket.rounded"),
      node("Sub", ["bucket.rounded", "bucket.total"], "bucket.extra"),
      node("Slice", [@durations, "bucket.zero", "bucket.last", "bucket.axis"], "bucket.head"),
      node("Slice", [@durations, "bucket.last", "bucket.end", "bucket.axis"], "bucket.tail"),
      node("Add", ["bucket.tail", "bucket.extra"], "bucket.longer_tail"),
      node("Concat", ["bucket.head", "bucket.longer_tail"], "bucket.durations", axis: 1)
    ]

    first_reader = Enum.find_index(model.nodes, &(@durations in &1.input))

    nodes =
      model.nodes
      |> Enum.map(fn n ->
        %{n | input: Enum.map(n.input, &if(&1 == @durations, do: "bucket.durations", else: &1))}
      end)
      |> List.insert_at(first_reader, added)
      |> List.flatten()

    static =
      Map.new(constants, fn {k, v} -> {k, Nx.tensor(v, type: :s64, backend: Nx.BinaryBackend)} end)

    %{model | nodes: nodes, static: Map.merge(model.static, static)}
  end

  defp node(op, inputs, output, attrs \\ []) do
    attributes =
      for {name, value} <- attrs, do: %AttributeProto{name: to_string(name), type: :INT, i: value}

    %NodeProto{op_type: op, input: inputs, output: [output], name: output, attribute: attributes}
  end
end

repo = "onnx-community/Kokoro-82M-v1.0-ONNX"
backend = {EMLX.Backend, device: :gpu}
output = List.first(System.argv(), "kokoro.wav")

# MLX keeps freed buffers in a cache. Each audio length frees buffers of new sizes,
# and without a limit this cache grew to 11.7 GB on a 16 GB Mac and made calls slow.
EMLX.set_cache_limit(1_000_000_000)

download = fn file ->
  Req.get!("https://huggingface.co/#{repo}/resolve/main/#{file}", decode_body: false).body
end

# Kokoro reads phonemes. misaki or espeak make them from text. These are for
# "Hello from Elixir. This voice runs on Nx, without Python."
phonemes = "həlˈO fɹʌm əlˈɪksəɹ. ðˌɪs vˈYs ɹˈʌnz ˌɔn ˌɛnˈɛks, wɪðˈWt pˈIθˌɑn."
vocab = download.("tokenizer.json") |> JSON.decode!() |> get_in(["model", "vocab"])
ids = [0] ++ for(char <- String.codepoints(phonemes), id = vocab[char], do: id) ++ [0]

# A voice holds one style vector for each phoneme count.
style =
  download.("voices/af_heart.bin")
  |> Nx.from_binary(:f32, backend: backend)
  |> Nx.reshape({:auto, 256})
  |> Nx.slice_along_axis(length(ids) - 2, 1, axis: 0)

model =
  {:hf, repo, "onnx/model.onnx"}
  |> Transmuter.load!(backend: backend)
  |> KokoroBuckets.round_durations(8)

inputs = fn speed ->
  %{
    "input_ids" => Nx.tensor([ids], type: :s64, backend: backend),
    "style" => style,
    "speed" => Nx.tensor([speed], backend: backend)
  }
end

speak = Transmuter.compile(model, inputs.(1.0), compiler: EMLX)

# Other speeds give other lengths. Lengths in one 200 ms bucket reuse one compiled decoder.
for speed <- [1.0, 1.05, 1.1, 1.0, 1.05, 1.1] do
  {us, out} =
    :timer.tc(fn -> speak.(inputs.(speed))["waveform"] |> Nx.backend_copy(Nx.BinaryBackend) end)

  IO.puts(
    "speed #{speed}: #{Float.round(Nx.size(out) / 24_000, 2)} s of audio in #{div(us, 1000)} ms"
  )
end

waveform = speak.(inputs.(1.0))["waveform"] |> Nx.backend_copy(Nx.BinaryBackend)
pcm = waveform |> Nx.clip(-1, 1) |> Nx.multiply(32_767) |> Nx.as_type(:s16) |> Nx.to_binary()

header =
  <<"RIFF", 36 + byte_size(pcm)::little-32, "WAVE", "fmt ", 16::little-32, 1::little-16,
    1::little-16, 24_000::little-32, 48_000::little-32, 2::little-16, 16::little-16, "data",
    byte_size(pcm)::little-32>>

File.write!(output, header <> pcm)
IO.puts("wrote #{output}")
