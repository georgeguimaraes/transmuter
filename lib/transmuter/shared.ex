defmodule Transmuter.Shared do
  @moduledoc """
  Keeps large terms that many processes read, such as compiled functions, in `:persistent_term` while the process that owns them lives.

  A process copies each term that it reads from ETS, and each closure that it gets from another process. Nx.Serving runs each batch in a new process, and the compiled function of a step of Gemma 3 holds about 1 MB of terms. A read from `:persistent_term` does not copy the term.

  A scope belongs to the process that makes it. When that process exits, a small process erases the terms of the scope. An erase makes each process check its heap for references to the term, so only terms that live long go here. The small process also owns the ETS tables of the scope (see `table/2`), so they end with the scope.

  A scope can also end before its owner, with `erase_scope/1`. `collect/1` gives the scopes that a function makes, so the code that runs the function can end them, for example at the end of a call of `Nx.Serving.run/2` that compiles in the caller.
  """

  @collect {__MODULE__, :collect}

  @doc false
  def scope do
    owner = self()

    watcher =
      spawn(fn ->
        ref = Process.monitor(owner)

        receive do
          {:DOWN, ^ref, :process, _pid, _reason} -> erase_terms({owner, self()})
        end
      end)

    scope = {owner, watcher}
    if scopes = Process.get(@collect), do: Process.put(@collect, [scope | scopes])
    scope
  end

  @doc false
  # Runs `fun` and returns its result and the scopes that it made in this process.
  def collect(fun) do
    previous = Process.put(@collect, [])

    try do
      result = fun.()
      {result, Process.get(@collect)}
    catch
      kind, reason ->
        Enum.each(Process.get(@collect), &erase_scope/1)
        :erlang.raise(kind, reason, __STACKTRACE__)
    after
      if previous, do: Process.put(@collect, previous), else: Process.delete(@collect)
    end
  end

  @doc false
  # A new ETS table of the scope with `options`. The small process of the scope owns it, so
  # the table ends with the scope.
  def table({_owner, watcher}, options) do
    table = :ets.new(__MODULE__, options)
    true = :ets.give_away(table, watcher, nil)
    table
  end

  @doc false
  def get(scope, key), do: :persistent_term.get({__MODULE__, scope, key}, nil)

  # The owner can exit while the term goes in, after its terms were erased, so the term
  # goes out again. A scope whose owner is gone keeps no terms.
  @doc false
  def put(scope = {owner, _watcher}, key, term) do
    if Process.alive?(owner) do
      :persistent_term.put({__MODULE__, scope, key}, term)
      unless Process.alive?(owner), do: erase(scope, key)
    end

    :ok
  end

  @doc false
  # Ends the scope: erases its terms and stops its small process, which ends its tables.
  def erase_scope(scope = {_owner, watcher}) do
    erase_terms(scope)
    Process.exit(watcher, :kill)
    :ok
  end

  defp erase_terms(scope) do
    for {key = {__MODULE__, ^scope, _key}, _term} <- :persistent_term.get(),
        do: :persistent_term.erase(key)

    :ok
  end

  @doc false
  def erase(scope, key) do
    :persistent_term.erase({__MODULE__, scope, key})
    :ok
  end
end
