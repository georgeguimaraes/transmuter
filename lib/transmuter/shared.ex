defmodule Transmuter.Shared do
  @moduledoc """
  Keeps large terms that many processes read, such as compiled functions, in `:persistent_term` while the process that owns them lives.

  A process copies each term that it reads from ETS, and each closure that it gets from another process. Nx.Serving runs each batch in a new process, and the compiled function of a step of Gemma 3 holds about 1 MB of terms. A read from `:persistent_term` does not copy the term.

  A scope belongs to the process that makes it. When that process exits, a small process erases the terms of the scope. An erase makes each process check its heap for references to the term, so only terms that live long go here.
  """

  @doc false
  def scope do
    owner = self()
    scope = {owner, make_ref()}

    spawn(fn ->
      ref = Process.monitor(owner)

      receive do
        {:DOWN, ^ref, :process, _pid, _reason} -> erase_scope(scope)
      end
    end)

    scope
  end

  @doc false
  def get(scope, key), do: :persistent_term.get({__MODULE__, scope, key}, nil)

  # The owner can exit while the term goes in, after its terms were erased, so the term
  # goes out again. A scope whose owner is gone keeps no terms.
  @doc false
  def put(scope = {owner, _ref}, key, term) do
    if Process.alive?(owner) do
      :persistent_term.put({__MODULE__, scope, key}, term)
      unless Process.alive?(owner), do: erase(scope, key)
    end

    :ok
  end

  @doc false
  # Erases all terms of the scope.
  def erase_scope(scope) do
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
