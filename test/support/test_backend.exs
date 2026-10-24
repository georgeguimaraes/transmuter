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
