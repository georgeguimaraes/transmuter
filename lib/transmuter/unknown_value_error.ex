defmodule Transmuter.UnknownValueError do
  @moduledoc """
  Raised when an operator needs the value of a tensor that is only known when the
  model runs, for example the limit of a `Range` or the input of a `NonZero`.
  """

  defexception [:value, :node, :op_type]

  @impl true
  def message(error) do
    at = if error.node, do: "#{error.op_type} #{error.node}: ", else: ""

    at <>
      "#{error.value} is only known when the model runs. Pass the graph inputs it depends " <>
      "on as static_inputs, or call Transmuter.run/4 without compiling"
  end
end
