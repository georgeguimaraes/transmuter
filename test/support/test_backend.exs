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

  # The memory of the test device in GB: the GPU of the CUDA client, from nvidia-smi, because
  # EXLA gives only the allocated and the peak bytes of a client. Else the RAM of the host,
  # which EMLX shares with the GPU of a Mac. test_helper.exs reads it once.
  def device_memory_gb do
    bytes = if device_opts() == [client: :cuda], do: cuda_memory(), else: host_memory()
    bytes / 1024 ** 3
  end

  # nvidia-smi does not read CUDA_VISIBLE_DEVICES, so it gets the index of the first visible
  # GPU, which is the device of the client.
  defp cuda_memory do
    path =
      System.find_executable("nvidia-smi") ||
        raise "the tests on CUDA read the memory of the GPU with nvidia-smi, which is not in PATH"

    index = "CUDA_VISIBLE_DEVICES" |> System.get_env("0") |> String.split(",") |> hd()
    args = ["--query-gpu=memory.total", "--format=csv,noheader,nounits", "-i", index]
    {mib, 0} = System.cmd(path, args)
    String.to_integer(String.trim(mib)) * 1024 ** 2
  end

  defp host_memory do
    case :os.type() do
      {:unix, :darwin} ->
        {bytes, 0} = System.cmd("sysctl", ["-n", "hw.memsize"])
        String.to_integer(String.trim(bytes))

      {:unix, _linux} ->
        [kib] =
          Regex.run(~r/MemTotal:\s+(\d+) kB/, File.read!("/proc/meminfo"),
            capture: :all_but_first
          )

        String.to_integer(kib) * 1024
    end
  end

  # Tags the next test with a need of the test machine, and skips it when the machine does
  # not have it (see unmet/1). The tag is the need: :exla, :ffmpeg or device_memory: gigabytes.
  # A skip and not an exclude filter, because a test that --only selects by name runs
  # whatever the exclude filters say.
  defmacro needs(need) do
    quote do
      @tag unquote(need)

      if reason = Transmuter.TestBackend.unmet(unquote(need)) do
        @tag skip: reason
      end
    end
  end

  # needs/1 for all tests of a module.
  defmacro module_needs(need) do
    quote do
      @moduletag unquote(need)

      if reason = Transmuter.TestBackend.unmet(unquote(need)) do
        @moduletag skip: reason
      end
    end
  end

  # The reason to skip a test with a need that the test machine does not have, or nil.
  def unmet(:exla) do
    if compiler() != EXLA, do: "needs the EXLA compiler, the tests run on #{inspect(compiler())}"
  end

  # The tests that read audio files need ffmpeg, which CI does not install.
  def unmet(:ffmpeg) do
    if !System.find_executable("ffmpeg"), do: "needs ffmpeg, which is not in PATH"
  end

  def unmet(device_memory: gigabytes) do
    available = Application.fetch_env!(:transmuter, :test_device_memory_gb)

    if gigabytes > available,
      do: "needs #{gigabytes} GB of device memory, the device has #{Float.round(available, 1)} GB"
  end

  defp device do
    default = if match?({:unix, :darwin}, :os.type()), do: "gpu", else: "cpu"

    case System.get_env("TRANSMUTER_TEST_DEVICE", default) do
      "gpu" -> :gpu
      "cpu" -> :cpu
    end
  end
end
