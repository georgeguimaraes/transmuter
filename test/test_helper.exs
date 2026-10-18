# EMLX 0.5 creates its compile cache, a named ETS table, in the first process that
# compiles, and ETS deletes a table when its owner exits. A compile here makes this
# long-lived process the owner, so async tests that finish do not delete the cache.
device = if match?({:unix, :darwin}, :os.type()), do: :gpu, else: :cpu

device =
  String.to_existing_atom(System.get_env("TRANSMUTER_TEST_DEVICE") || Atom.to_string(device))

Nx.Defn.jit(&Nx.add(&1, 1), compiler: EMLX, device: device).(Nx.tensor(1.0))

ExUnit.start()
