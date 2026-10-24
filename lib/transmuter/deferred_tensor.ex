defmodule Transmuter.DeferredTensor do
  @moduledoc false
  # An initializer in the external data file that `Transmuter.load!/2` with a backend reads
  # only when it moves the param to the backend, so the file data of all params is never in
  # memory at once. It has the shape and the type of the tensor that `Transmuter.Ops.tensor/2`
  # gives. Such a value exists only in `Transmuter.load!/2`, between the read of the
  # initializers and the move to the backend. The move reads it with `value/1`, and no other
  # code sees it.

  alias Transmuter.Ops

  defstruct [:shape, :type, :proto, :dir]

  @doc false
  # The deferred tensor of an external initializer, or nil for a string or an empty tensor.
  def new(proto = %{data_location: :EXTERNAL}, dir) do
    case Ops.tensor_shape_type(proto) do
      {shape, type} -> %__MODULE__{shape: shape, type: type, proto: proto, dir: dir}
      nil -> nil
    end
  end

  def new(_proto, _dir), do: nil

  @doc false
  # The tensor of a param: a deferred tensor reads it from the file.
  def value(%__MODULE__{proto: proto, dir: dir}), do: Ops.tensor(proto, dir)
  def value(tensor), do: tensor

  @doc false
  # The shape and the type of a tensor or of a deferred tensor.
  def shape_type(%__MODULE__{shape: shape, type: type}), do: {shape, type}
  def shape_type(tensor = %Nx.Tensor{}), do: {Nx.shape(tensor), Nx.type(tensor)}
end
