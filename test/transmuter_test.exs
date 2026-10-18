defmodule TransmuterTest do
  use ExUnit.Case, async: true

  # Reference values come from ONNX Runtime with graph optimizations disabled,
  # see test/support/reference.py. The first run downloads the models.
  @moduletag timeout: 900_000

  # Metal on macOS; the CPU device elsewhere, such as Linux CI. The EMLX compiler
  # takes the device too, because it defaults to the GPU.
  # TRANSMUTER_TEST_DEVICE=cpu runs the CI configuration on a Mac.
  @device (System.get_env("TRANSMUTER_TEST_DEVICE") ||
             if(match?({:unix, :darwin}, :os.type()), do: "gpu", else: "cpu"))
          |> String.to_existing_atom()
  @backend {EMLX.Backend, device: @device}
  @compiler [compiler: EMLX, device: @device]

  test "runs a BERT encoder" do
    model = load("Xenova/all-MiniLM-L6-v2", "onnx/model.onnx")
    ids = Nx.tensor([[101, 7592, 2088, 2003, 1037, 3231, 102]], type: :s64, backend: @backend)

    inputs = %{
      "input_ids" => ids,
      "attention_mask" => Nx.broadcast(Nx.tensor(1, type: :s64, backend: @backend), {1, 7}),
      "token_type_ids" => Nx.broadcast(Nx.tensor(0, type: :s64, backend: @backend), {1, 7})
    }

    %{"last_hidden_state" => hidden} = run(model, inputs)

    assert_all_close(hidden[[0, 0..2, 0..2]], [
      [0.2091, -0.0291, 0.0011],
      [-0.4452, 0.2465, 0.5265],
      [0.0783, 0.2770, -0.2609]
    ])
  end

  test "resizes with interpolation matrices" do
    model = load("Xenova/segformer-b0-finetuned-ade-512-512", "onnx/model.onnx")
    %{"logits" => logits} = run(model, %{"pixel_values" => pattern({1, 3, 512, 512})})

    assert_all_close(logits[[0, 0..2, 0, 0..2]], [
      [-5.9847, -6.1833, -6.5003],
      [-7.3063, -7.5704, -7.7874],
      [-9.4081, -9.2398, -9.7014]
    ])
  end

  test "selects an If branch from a static input" do
    model = load("Xenova/whisper-tiny.en", "onnx/decoder_model_merged.onnx")

    past =
      for name <- model.inputs, String.starts_with?(name, "past_key_values"), into: %{} do
        {name, pattern({1, 6, if(name =~ ".encoder.", do: 1500, else: 3), 64})}
      end

    inputs =
      Map.merge(past, %{
        "input_ids" => Nx.tensor([[1000]], type: :s64, backend: @backend),
        "encoder_hidden_states" => pattern({1, 1500, 384})
      })

    flag = Nx.tensor([1], type: :u8, backend: Nx.BinaryBackend)
    %{"logits" => logits} = run(model, inputs, %{"use_cache_branch" => flag})
    assert_all_close(logits[[0, 0, 0..4]], [11.7904, 7.4783, 7.9252, 8.0604, 6.4908])

    assert_raise RuntimeError, ~r/If needs a static condition/, fn ->
      run(model, Map.put(inputs, "use_cache_branch", Nx.backend_copy(flag, @backend)))
    end
  end

  test "generates with a static KV cache like transformers" do
    model = load("HuggingFaceTB/SmolLM2-135M-Instruct", "onnx/model.onnx")

    prompt = [
      1,
      9690,
      198,
      2683,
      359,
      253,
      5356,
      5646,
      11_173,
      3365,
      3511,
      308,
      34_519,
      28,
      7018,
      411,
      407,
      19_712,
      8182,
      2,
      198,
      1,
      4093,
      198,
      36_971,
      281,
      827,
      8545,
      1701,
      260,
      6376,
      314,
      4461,
      30,
      2,
      198,
      1,
      520,
      9531,
      198
    ]

    max_len = 64

    # The KV buffer is as long as the attention mask, so GroupQueryAttention writes
    # new keys in place and every decode step has the same shapes.
    cache =
      for name <- model.inputs, String.starts_with?(name, "past_key_values"), into: %{} do
        {name, Nx.broadcast(Nx.tensor(0.0, backend: @backend), {1, 3, max_len, 64})}
      end

    step_inputs = fn ids, pos, cache ->
      n = pos + length(ids)

      Map.merge(cache, %{
        "input_ids" => Nx.tensor([ids], type: :s64, backend: @backend),
        "attention_mask" =>
          Nx.tensor([Enum.map(0..(max_len - 1), &if(&1 < n, do: 1, else: 0))],
            type: :s64,
            backend: @backend
          ),
        "position_ids" => Nx.tensor([Enum.to_list(pos..(n - 1))], type: :s64, backend: @backend)
      })
    end

    prefill = Transmuter.compile(model, step_inputs.(prompt, 0, cache), @compiler)
    decode = Transmuter.compile(model, step_inputs.([0], length(prompt), cache), @compiler)

    next_cache = fn out ->
      for {"present." <> rest, t} <- out, into: %{}, do: {"past_key_values." <> rest, t}
    end

    next_token = fn out -> out["logits"][[0, -1]] |> Nx.argmax() |> Nx.to_number() end

    out = prefill.(step_inputs.(prompt, 0, cache))

    {tokens, _} =
      Enum.reduce(1..7, {[next_token.(out)], next_cache.(out)}, fn i, {tokens, cache} ->
        out = decode.(step_inputs.([hd(tokens)], length(prompt) + i - 1, cache))
        {[next_token.(out) | tokens], next_cache.(out)}
      end)

    assert Enum.reverse(tokens) == [504, 6376, 4541, 4461, 975, 260, 2591, 506]
  end

  # The BERT pooler gathers the first token on axis 1, a path whose gradient EMLX 0.5
  # scatters to the wrong place. Autodiff must match central finite differences.
  test "computes gradients that match finite differences" do
    model = load("Xenova/ms-marco-MiniLM-L-6-v2", "onnx/model.onnx", trainable: true)
    params = model.params

    # Two sequences: with a batch of 1 the misplaced gradient happens to land correctly.
    ids =
      Nx.tensor(
        [
          [101, 2054, 2003, 1041, 102, 3793, 2003, 1037, 2653, 102],
          [101, 3007, 1997, 2605, 102, 3000, 2003, 1996, 3007, 102]
        ],
        type: :s64,
        backend: @backend
      )

    segments = Nx.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], type: :s64, backend: @backend)

    inputs = %{
      "input_ids" => ids,
      "attention_mask" => Nx.broadcast(Nx.tensor(1, type: :s64, backend: @backend), {2, 10}),
      "token_type_ids" => Nx.broadcast(segments, {2, 10})
    }

    score = fn p, i -> Transmuter.run(model, p, i)["logits"] |> Nx.sum() end
    forward = Nx.Defn.jit(score, @compiler)

    {_, grads} =
      Nx.Defn.jit(&Nx.Defn.value_and_grad(&1, fn p -> score.(p, &2) end), @compiler).(
        params,
        inputs
      )

    name = "bert.embeddings.LayerNorm.weight"

    bump = fn delta ->
      Map.update!(
        params,
        name,
        &Nx.indexed_add(
          &1,
          Nx.tensor([[1]], backend: @backend),
          Nx.tensor([delta], backend: @backend)
        )
      )
    end

    numeric =
      (Nx.to_number(forward.(bump.(1.0e-2), inputs)) -
         Nx.to_number(forward.(bump.(-1.0e-2), inputs))) / 2.0e-2

    autodiff = grads[name] |> Nx.backend_copy(Nx.BinaryBackend) |> Nx.to_flat_list() |> Enum.at(1)

    assert_in_delta autodiff, numeric, 0.02 * abs(numeric) + 1.0e-3
  end

  test "returns a string output only without compiling" do
    model = Transmuter.load!(Path.expand("fixtures/ops/Identity_strings/model.onnx", __DIR__))
    strings = %{"x" => {:strings, {2}, ["a", "b"]}}

    assert Transmuter.run(model, model.params, %{}, strings) == %{
             "y" => {:strings, {2}, ["a", "b"]}
           }

    assert_raise ArgumentError, ~r/output y is a string tensor/, fn ->
      Transmuter.compile(model, %{}, [static_inputs: strings] ++ @compiler).(%{})
    end
  end

  defp load(repo, file, opts \\ []) do
    Transmuter.load!({:hf, repo, file}, [backend: @backend] ++ opts)
  end

  defp run(model, inputs, static_inputs \\ %{}) do
    Transmuter.compile(model, inputs, [static_inputs: static_inputs] ++ @compiler).(inputs)
  end

  # sin(i * 0.01) * 0.5, the same pattern that test/support/reference.py feeds to ORT.
  defp pattern(shape) do
    {Tuple.product(shape)}
    |> Nx.iota(type: :f32, backend: @backend)
    |> Nx.multiply(0.01)
    |> Nx.sin()
    |> Nx.multiply(0.5)
    |> Nx.reshape(shape)
  end

  defp assert_all_close(tensor, expected) do
    expected = Nx.tensor(expected)
    tensor = Nx.backend_copy(tensor, Nx.BinaryBackend)

    assert Nx.to_number(Nx.all_close(tensor, expected, atol: 1.0e-4, rtol: 1.0e-4)) == 1,
           "expected #{inspect(Nx.to_list(expected))}, got #{inspect(Nx.to_list(tensor))}"
  end
end
