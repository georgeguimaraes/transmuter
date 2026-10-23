defmodule Transmuter.UnknownValueError do
  @moduledoc """
  Raised when an operator needs the value of a tensor that is only known when
  the model runs.

  Examples include the limit of a `Range`, the input of a `NonZero`, the
  condition of an `If`, and the scores of a `NonMaxSuppression`.

  Transmuter catches this error during compilation and creates a graph break at
  the operator. The break operator runs eagerly on the real values, and
  compilation continues after it with the shapes the values decide.

  If Transmuter cannot handle the break, it re-raises the error. To fix it:

  - Pass the tensor as a static input (`:static_inputs` option of
    `Transmuter.compile/3`).
  - Call `Transmuter.run/4` without compilation.

  ## Fields

    * `:value` - the name of the tensor whose value was needed
    * `:op_type` - the operator type that needed the value
    * `:node` - the node name in the graph, or `nil` when not available
    * `:expr` - the id of the traced expression whose value was needed, or `nil`. `Transmuter.compile/3` finds the input of the node that it is.
  """

  defexception [:value, :node, :op_type, :expr]

  @impl true
  def message(error) do
    at = if error.node, do: "#{error.op_type} #{error.node}: ", else: ""

    at <>
      "#{error.value} is only known when the model runs. Pass the graph inputs it depends " <>
      "on as static_inputs, or call Transmuter.run/4 without compiling"
  end
end
