Code.require_file("support/test_backend.exs", __DIR__)

# EMLX.CompileCache keeps each compiled closure without a limit by default, and each
# Transmuter.compile adds a closure that no later compile reads. The cache reads this limit
# on each insert, so the limit applies to the cache that already runs.
Application.put_env(:emlx, EMLX.CompileCache, max_items: 128)
Application.put_env(:transmuter, :progress_bar_enabled, false)

# On CUDA, the client of EXLA takes 90 percent of the GPU memory for its arena when it starts.
# When the VITS test compiles its decoder, the autotuner of XLA takes about 8.6 GB of that arena
# as scratch for the convolutions, and the async audio tests that run at the same time can run
# out of memory. Without the arena, the client takes memory when it needs it, and audio_test
# peaks at about 9 GB on a GPU of 12 GB.
if Transmuter.TestBackend.compiler() == EXLA do
  clients = Application.fetch_env!(:exla, :clients)
  cuda = Keyword.put(clients[:cuda], :preallocate, false)
  Application.put_env(:exla, :clients, Keyword.put(clients, :cuda, cuda))
end

# EMLX 0.5 creates its compile cache, a named ETS table, in the first process that
# compiles, and ETS deletes a table when its owner exits. A compile here makes this
# long-lived process the owner, so async tests that finish do not delete the cache.
#
# MLX keeps freed buffers in a cache without a limit by default. The prompt of a vision
# language model runs in parts of different sizes, so few buffers are used again, and the
# cache grew to 9.4 GB in the LLaVA-Interleave test (live data at most 3.6 GB), near the 16 GB
# of a Mac. A limit of 1 GB keeps the buffers of the decode steps, which have one size.
if Transmuter.TestBackend.compiler() == EMLX do
  EMLX.set_cache_limit(1_000_000_000)
  Nx.Defn.jit(&Nx.add(&1, 1), Transmuter.TestBackend.compiler_opts()).(Nx.tensor(1.0))
end

# The device memory for the device_memory need of Transmuter.TestBackend.needs/1.
Application.put_env(
  :transmuter,
  :test_device_memory_gb,
  Transmuter.TestBackend.device_memory_gb()
)

ExUnit.start()

# The EMLX job of CI sets HANG_DIR. See Transmuter.HangWatchdog.
if dir = System.get_env("HANG_DIR") do
  Code.require_file("support/hang_watchdog.exs", __DIR__)
  Transmuter.HangWatchdog.start(dir)
end
