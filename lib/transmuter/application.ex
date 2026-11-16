defmodule Transmuter.Application do
  @moduledoc """
  Starts the `:pg` scope `Transmuter.PG`. A `Transmuter.ContinuousBatching` server with a name joins the group of its name in this scope, so a caller on another node of the cluster finds it.
  """

  use Application

  @impl true
  def start(_type, _args) do
    children = [%{id: Transmuter.PG, start: {:pg, :start_link, [Transmuter.PG]}}]
    Supervisor.start_link(children, strategy: :one_for_one, name: Transmuter.Supervisor)
  end
end
