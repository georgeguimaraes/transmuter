defmodule Transmuter.Ops.Elementwise do
  @moduledoc false

  # Elementwise operators: arithmetic, comparisons, logic, bitwise, trigonometry and
  # activations.

  import Transmuter.Ops, only: [attr: 2, attr: 3, broadcast_shape: 1]

  @op_types ~w(Add Sub Mul Div Pow Sqrt Erf Tanh Abs Exp Sigmoid Mod Equal Less LessOrEqual
               Greater GreaterOrEqual Not And Or Xor Neg Sin Cos Atan Log Floor Ceil Round Sign
               Reciprocal IsNaN IsInf Relu LeakyRelu Softplus HardSigmoid HardSwish Gelu Max Min
               Clip Where FastGelu BiasGelu QuickGelu PRelu Mish Softsign Selu ThresholdedRelu
               Elu Acos Acosh Asin Asinh Atanh Tan Cosh Sinh BitwiseAnd BitwiseOr BitwiseXor
               BitwiseNot BitShift Celu Swish SwiGLU Shrink Sum Mean)

  @doc false
  def op_types, do: @op_types

  @doc false
  def apply(node, inputs), do: node |> op(inputs) |> List.wrap()

  defp op(%{op_type: "Add"}, [a, b]), do: Nx.add(a, b)
  defp op(%{op_type: "Sub"}, [a, b]), do: Nx.subtract(a, b)
  defp op(%{op_type: "Mul"}, [a, b]), do: Nx.multiply(a, b)

  defp op(%{op_type: "Div"}, [a, b]) do
    if Nx.Type.integer?(Nx.type(a)) and Nx.Type.integer?(Nx.type(b)),
      do: Nx.quotient(a, b),
      else: Nx.divide(a, b)
  end

  # The output has the type of the base, also for an integer base with a float exponent.
  defp op(%{op_type: "Pow"}, [a, b]), do: a |> Nx.pow(b) |> Nx.as_type(Nx.type(a))
  defp op(%{op_type: "Sqrt"}, [a]), do: Nx.sqrt(a)
  defp op(%{op_type: "Erf"}, [a]), do: Nx.erf(a)
  defp op(%{op_type: "Tanh"}, [a]), do: Nx.tanh(a)

  defp op(%{op_type: "Abs"}, [a]), do: Nx.abs(a)
  defp op(%{op_type: "Exp"}, [a]), do: Nx.exp(a)
  defp op(%{op_type: "Sigmoid"}, [a]), do: Nx.sigmoid(a)

  defp op(node = %{op_type: "Mod"}, [a, b]) do
    r = Nx.remainder(a, b)

    if attr(node, "fmod", 0) == 1,
      do: r,
      else:
        Nx.select(
          Nx.logical_and(Nx.not_equal(r, 0), Nx.not_equal(Nx.sign(r), Nx.sign(b))),
          Nx.add(r, b),
          r
        )
  end

  defp op(%{op_type: "Equal"}, [a, b]), do: Nx.equal(a, b)
  defp op(%{op_type: "Less"}, [a, b]), do: Nx.less(a, b)
  defp op(%{op_type: "LessOrEqual"}, [a, b]), do: Nx.less_equal(a, b)
  defp op(%{op_type: "Greater"}, [a, b]), do: Nx.greater(a, b)
  defp op(%{op_type: "GreaterOrEqual"}, [a, b]), do: Nx.greater_equal(a, b)
  defp op(%{op_type: "Not"}, [a]), do: Nx.logical_not(a)
  defp op(%{op_type: "And"}, [a, b]), do: Nx.logical_and(a, b)
  defp op(%{op_type: "Or"}, [a, b]), do: Nx.logical_or(a, b)
  defp op(%{op_type: "Xor"}, [a, b]), do: Nx.logical_xor(a, b)
  defp op(%{op_type: "Neg"}, [a]), do: Nx.negate(a)
  defp op(%{op_type: "Sin"}, [a]), do: Nx.sin(a)
  defp op(%{op_type: "Cos"}, [a]), do: Nx.cos(a)
  defp op(%{op_type: "Atan"}, [a]), do: Nx.atan(a)
  defp op(%{op_type: "Log"}, [a]), do: Nx.log(a)
  defp op(%{op_type: "Floor"}, [a]), do: Nx.floor(a)
  defp op(%{op_type: "Ceil"}, [a]), do: Nx.ceil(a)
  defp op(%{op_type: "Round"}, [a]), do: Nx.round(a)
  defp op(%{op_type: "Sign"}, [a]), do: Nx.sign(a)
  defp op(%{op_type: "Reciprocal"}, [a]), do: Nx.divide(1, a)
  defp op(%{op_type: "IsNaN"}, [a]), do: Nx.is_nan(a)
  defp op(%{op_type: "IsInf"}, [a]), do: Nx.is_infinity(a)
  defp op(%{op_type: "Relu"}, [a]), do: Nx.max(a, 0)

  defp op(node = %{op_type: "LeakyRelu"}, [a]),
    do: Nx.select(Nx.less(a, 0), Nx.multiply(a, attr(node, "alpha", 0.01)), a)

  defp op(%{op_type: "Softplus"}, [a]), do: Nx.log1p(Nx.exp(a))

  defp op(node = %{op_type: "HardSigmoid"}, [a]),
    do:
      a
      |> Nx.multiply(attr(node, "alpha", 0.2))
      |> Nx.add(attr(node, "beta", 0.5))
      |> Nx.clip(0, 1)

  defp op(%{op_type: "HardSwish"}, [a]),
    do: Nx.multiply(a, a |> Nx.divide(6) |> Nx.add(0.5) |> Nx.clip(0, 1))

  defp op(node = %{op_type: "Gelu"}, [a]),
    do: if(attr(node, "approximate", "none") == "tanh", do: gelu_tanh(a), else: gelu(a))

  defp op(%{op_type: "Max"}, inputs), do: Enum.reduce(inputs, &Nx.max(&2, &1))
  defp op(%{op_type: "Min"}, inputs), do: Enum.reduce(inputs, &Nx.min(&2, &1))

  defp op(node = %{op_type: "Clip"}, [a | bounds]) do
    {lo, hi} =
      case bounds do
        [] -> {attr(node, "min"), attr(node, "max")}
        [lo] -> {lo, nil}
        [lo, hi] -> {lo, hi}
      end

    a = if lo != nil, do: Nx.max(a, Nx.as_type(lo, Nx.type(a))), else: a
    if hi != nil, do: Nx.min(a, Nx.as_type(hi, Nx.type(a))), else: a
  end

  defp op(%{op_type: "Where"}, [c, x, y]) do
    shape = broadcast_shape([Nx.shape(c), Nx.shape(x), Nx.shape(y)])
    Nx.select(Nx.broadcast(c, shape), Nx.broadcast(x, shape), Nx.broadcast(y, shape))
  end

  defp op(%{op_type: "FastGelu"}, [x | bias]), do: gelu_tanh(add_optional(x, List.first(bias)))
  defp op(%{op_type: "BiasGelu"}, [x, bias]), do: gelu(Nx.add(x, bias))

  defp op(node = %{op_type: "QuickGelu"}, [x]),
    do: Nx.multiply(x, Nx.sigmoid(Nx.multiply(x, attr(node, "alpha", 1.702))))

  defp op(%{op_type: "PRelu"}, [x, slope]),
    do:
      Nx.select(Nx.less(x, 0), Nx.multiply(x, slope), x)
      |> Nx.broadcast(broadcast_shape([Nx.shape(x), Nx.shape(slope)]))

  defp op(%{op_type: "Mish"}, [x]), do: Nx.multiply(x, Nx.tanh(Nx.log1p(Nx.exp(x))))
  defp op(%{op_type: "Softsign"}, [x]), do: Nx.divide(x, Nx.add(1, Nx.abs(x)))

  defp op(node = %{op_type: "Selu"}, [x]) do
    alpha = attr(node, "alpha", 1.67326319217681884765625)
    gamma = attr(node, "gamma", 1.05070102214813232421875)

    Nx.multiply(
      gamma,
      Nx.select(Nx.greater(x, 0), x, Nx.multiply(alpha, Nx.subtract(Nx.exp(x), 1)))
    )
  end

  defp op(node = %{op_type: "ThresholdedRelu"}, [x]),
    do: Nx.select(Nx.greater(x, attr(node, "alpha", 1.0)), x, Nx.tensor(0, type: Nx.type(x)))

  defp op(node = %{op_type: "Elu"}, [a]),
    do:
      Nx.select(
        Nx.less(a, 0),
        Nx.multiply(attr(node, "alpha", 1.0), Nx.subtract(Nx.exp(a), 1)),
        a
      )

  defp op(%{op_type: "Acos"}, [a]), do: Nx.acos(a)
  defp op(%{op_type: "Acosh"}, [a]), do: Nx.acosh(a)
  defp op(%{op_type: "Asin"}, [a]), do: Nx.asin(a)
  defp op(%{op_type: "Asinh"}, [a]), do: Nx.asinh(a)
  defp op(%{op_type: "Atanh"}, [a]), do: Nx.atanh(a)
  defp op(%{op_type: "Tan"}, [a]), do: Nx.tan(a)
  defp op(%{op_type: "Cosh"}, [a]), do: Nx.cosh(a)
  defp op(%{op_type: "Sinh"}, [a]), do: Nx.sinh(a)
  defp op(%{op_type: "BitwiseAnd"}, [a, b]), do: Nx.bitwise_and(a, b)
  defp op(%{op_type: "BitwiseOr"}, [a, b]), do: Nx.bitwise_or(a, b)
  defp op(%{op_type: "BitwiseXor"}, [a, b]), do: Nx.bitwise_xor(a, b)
  defp op(%{op_type: "BitwiseNot"}, [a]), do: Nx.bitwise_not(a)

  defp op(node = %{op_type: "BitShift"}, [a, b]) do
    if attr(node, "direction") == "LEFT",
      do: Nx.left_shift(a, b),
      else: Nx.right_shift(a, b)
  end

  defp op(node = %{op_type: "Celu"}, [x]) do
    alpha = attr(node, "alpha", 1.0)

    Nx.add(
      Nx.max(x, 0),
      Nx.min(0, Nx.multiply(alpha, Nx.subtract(Nx.exp(Nx.divide(x, alpha)), 1)))
    )
  end

  defp op(node = %{op_type: "Swish"}, [x]), do: swish(x, attr(node, "alpha", 1.0))

  defp op(node = %{op_type: "SwiGLU"}, [a, b]),
    do: Nx.multiply(swish(a, attr(node, "alpha", 1.0)), b)

  defp op(node = %{op_type: "Shrink"}, [x]) do
    {bias, lambd} = {attr(node, "bias", 0.0), attr(node, "lambd", 0.5)}
    zero = Nx.tensor(0, type: Nx.type(x))

    Nx.select(
      Nx.less(x, -lambd),
      Nx.add(x, bias),
      Nx.select(Nx.greater(x, lambd), Nx.subtract(x, bias), zero)
    )
  end

  defp op(%{op_type: "Sum"}, inputs), do: Enum.reduce(inputs, &Nx.add(&2, &1))

  defp op(%{op_type: "Mean"}, inputs),
    do: inputs |> Enum.reduce(&Nx.add(&2, &1)) |> Nx.divide(length(inputs))

  defp swish(x, alpha), do: Nx.multiply(x, Nx.sigmoid(Nx.multiply(x, alpha)))

  defp add_optional(t, nil), do: t
  defp add_optional(t, other), do: Nx.add(t, other)

  defp gelu(x),
    do: Nx.multiply(Nx.multiply(x, 0.5), Nx.add(1, Nx.erf(Nx.divide(x, :math.sqrt(2)))))

  defp gelu_tanh(x) do
    inner =
      Nx.multiply(:math.sqrt(2 / :math.pi()), Nx.add(x, Nx.multiply(0.044715, Nx.pow(x, 3))))

    Nx.multiply(Nx.multiply(x, 0.5), Nx.add(1, Nx.tanh(inner)))
  end
end
