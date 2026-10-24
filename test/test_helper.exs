Code.require_file("support/test_backend.exs", __DIR__)

# EMLX.CompileCache keeps each compiled closure without a limit by default, and each
# Transmuter.compile adds a closure that no later compile reads. The cache reads this limit
# on each insert, so the limit applies to the cache that already runs.
Application.put_env(:emlx, EMLX.CompileCache, max_items: 128)
Application.put_env(:transmuter, :progress_bar_enabled, false)

# EMLX 0.5 creates its compile cache, a named ETS table, in the first process that
# compiles, and ETS deletes a table when its owner exits. A compile here makes this
# long-lived process the owner, so async tests that finish do not delete the cache.
if Transmuter.TestBackend.compiler() == EMLX do
  Nx.Defn.jit(&Nx.add(&1, 1), Transmuter.TestBackend.compiler_opts()).(Nx.tensor(1.0))
end

# Tests that read audio files need ffmpeg, which CI does not install.
exclude =
  if(Transmuter.TestBackend.compiler() == EXLA, do: [], else: [:exla]) ++
    if System.find_executable("ffmpeg"), do: [], else: [:ffmpeg]

ExUnit.start(exclude: exclude)

# The EMLX job of CI sets HANG_DIR. See Transmuter.HangWatchdog.
if dir = System.get_env("HANG_DIR") do
  Code.require_file("support/hang_watchdog.exs", __DIR__)
  Transmuter.HangWatchdog.start(dir)
end
