defmodule Transmuter.TestBackend do
  @moduledoc false

  # TRANSMUTER_TEST_BACKEND is emlx (the default) or exla. TRANSMUTER_TEST_DEVICE is gpu or
  # cpu, and defaults to Metal on macOS and the CPU elsewhere, such as Linux CI.
  # TRANSMUTER_TEST_DEVICE=cpu runs the CI configuration on a Mac.

  def compiler do
    case System.get_env("TRANSMUTER_TEST_BACKEND", "emlx") do
      "emlx" -> EMLX
      "exla" -> EXLA
    end
  end

  def backend, do: {Module.concat(compiler(), Backend), device_opts()}

  # The EMLX compiler takes the device too, because it defaults to the GPU.
  def compiler_opts, do: [compiler: compiler()] ++ device_opts()

  defp device_opts do
    case {compiler(), device()} do
      {EMLX, device} -> [device: device]
      {EXLA, :gpu} -> [client: :cuda]
      {EXLA, :cpu} -> [client: :host]
    end
  end

  defp device do
    default = if match?({:unix, :darwin}, :os.type()), do: "gpu", else: "cpu"

    case System.get_env("TRANSMUTER_TEST_DEVICE", default) do
      "gpu" -> :gpu
      "cpu" -> :cpu
    end
  end
end

# EMLX 0.5 creates its compile cache, a named ETS table, in the first process that
# compiles, and ETS deletes a table when its owner exits. A compile here makes this
# long-lived process the owner, so async tests that finish do not delete the cache.
if Transmuter.TestBackend.compiler() == EMLX do
  Nx.Defn.jit(&Nx.add(&1, 1), Transmuter.TestBackend.compiler_opts()).(Nx.tensor(1.0))
end

ExUnit.start(exclude: if(Transmuter.TestBackend.compiler() == EXLA, do: [], else: [:exla]))
