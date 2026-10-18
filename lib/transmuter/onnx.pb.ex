# credo:disable-for-this-file
[
  defmodule Transmuter.Onnx.AttributeProto.AttributeType do
    @moduledoc false
    (
      defstruct []

      (
        @spec default() :: :UNDEFINED
        def default() do
          :UNDEFINED
        end
      )

      @spec encode(atom() | String.t()) :: integer() | atom()
      [
        (
          def encode(:UNDEFINED) do
            0
          end

          def encode("UNDEFINED") do
            0
          end
        ),
        (
          def encode(:FLOAT) do
            1
          end

          def encode("FLOAT") do
            1
          end
        ),
        (
          def encode(:INT) do
            2
          end

          def encode("INT") do
            2
          end
        ),
        (
          def encode(:STRING) do
            3
          end

          def encode("STRING") do
            3
          end
        ),
        (
          def encode(:TENSOR) do
            4
          end

          def encode("TENSOR") do
            4
          end
        ),
        (
          def encode(:GRAPH) do
            5
          end

          def encode("GRAPH") do
            5
          end
        ),
        (
          def encode(:SPARSE_TENSOR) do
            11
          end

          def encode("SPARSE_TENSOR") do
            11
          end
        ),
        (
          def encode(:TYPE_PROTO) do
            13
          end

          def encode("TYPE_PROTO") do
            13
          end
        ),
        (
          def encode(:FLOATS) do
            6
          end

          def encode("FLOATS") do
            6
          end
        ),
        (
          def encode(:INTS) do
            7
          end

          def encode("INTS") do
            7
          end
        ),
        (
          def encode(:STRINGS) do
            8
          end

          def encode("STRINGS") do
            8
          end
        ),
        (
          def encode(:TENSORS) do
            9
          end

          def encode("TENSORS") do
            9
          end
        ),
        (
          def encode(:GRAPHS) do
            10
          end

          def encode("GRAPHS") do
            10
          end
        ),
        (
          def encode(:SPARSE_TENSORS) do
            12
          end

          def encode("SPARSE_TENSORS") do
            12
          end
        ),
        (
          def encode(:TYPE_PROTOS) do
            14
          end

          def encode("TYPE_PROTOS") do
            14
          end
        )
      ]

      def encode(x) do
        x
      end

      @spec decode(integer()) :: atom() | integer()
      [
        def decode(0) do
          :UNDEFINED
        end,
        def decode(1) do
          :FLOAT
        end,
        def decode(2) do
          :INT
        end,
        def decode(3) do
          :STRING
        end,
        def decode(4) do
          :TENSOR
        end,
        def decode(5) do
          :GRAPH
        end,
        def decode(6) do
          :FLOATS
        end,
        def decode(7) do
          :INTS
        end,
        def decode(8) do
          :STRINGS
        end,
        def decode(9) do
          :TENSORS
        end,
        def decode(10) do
          :GRAPHS
        end,
        def decode(11) do
          :SPARSE_TENSOR
        end,
        def decode(12) do
          :SPARSE_TENSORS
        end,
        def decode(13) do
          :TYPE_PROTO
        end,
        def decode(14) do
          :TYPE_PROTOS
        end
      ]

      def decode(x) do
        x
      end

      @spec constants() :: [{integer(), atom()}]
      def constants() do
        [
          {0, :UNDEFINED},
          {1, :FLOAT},
          {2, :INT},
          {3, :STRING},
          {4, :TENSOR},
          {5, :GRAPH},
          {11, :SPARSE_TENSOR},
          {13, :TYPE_PROTO},
          {6, :FLOATS},
          {7, :INTS},
          {8, :STRINGS},
          {9, :TENSORS},
          {10, :GRAPHS},
          {12, :SPARSE_TENSORS},
          {14, :TYPE_PROTOS}
        ]
      end

      @spec has_constant?(any()) :: boolean()
      (
        [
          def has_constant?(:UNDEFINED) do
            true
          end,
          def has_constant?(:FLOAT) do
            true
          end,
          def has_constant?(:INT) do
            true
          end,
          def has_constant?(:STRING) do
            true
          end,
          def has_constant?(:TENSOR) do
            true
          end,
          def has_constant?(:GRAPH) do
            true
          end,
          def has_constant?(:SPARSE_TENSOR) do
            true
          end,
          def has_constant?(:TYPE_PROTO) do
            true
          end,
          def has_constant?(:FLOATS) do
            true
          end,
          def has_constant?(:INTS) do
            true
          end,
          def has_constant?(:STRINGS) do
            true
          end,
          def has_constant?(:TENSORS) do
            true
          end,
          def has_constant?(:GRAPHS) do
            true
          end,
          def has_constant?(:SPARSE_TENSORS) do
            true
          end,
          def has_constant?(:TYPE_PROTOS) do
            true
          end
        ]

        def has_constant?(_) do
          false
        end
      )
    )
  end,
  defmodule Transmuter.Onnx.OperatorStatus do
    @moduledoc false
    (
      defstruct []

      (
        @spec default() :: :EXPERIMENTAL
        def default() do
          :EXPERIMENTAL
        end
      )

      @spec encode(atom() | String.t()) :: integer() | atom()
      [
        (
          def encode(:EXPERIMENTAL) do
            0
          end

          def encode("EXPERIMENTAL") do
            0
          end
        ),
        (
          def encode(:STABLE) do
            1
          end

          def encode("STABLE") do
            1
          end
        )
      ]

      def encode(x) do
        x
      end

      @spec decode(integer()) :: atom() | integer()
      [
        def decode(0) do
          :EXPERIMENTAL
        end,
        def decode(1) do
          :STABLE
        end
      ]

      def decode(x) do
        x
      end

      @spec constants() :: [{integer(), atom()}]
      def constants() do
        [{0, :EXPERIMENTAL}, {1, :STABLE}]
      end

      @spec has_constant?(any()) :: boolean()
      (
        [
          def has_constant?(:EXPERIMENTAL) do
            true
          end,
          def has_constant?(:STABLE) do
            true
          end
        ]

        def has_constant?(_) do
          false
        end
      )
    )
  end,
  defmodule Transmuter.Onnx.TensorProto.DataLocation do
    @moduledoc false
    (
      defstruct []

      (
        @spec default() :: :DEFAULT
        def default() do
          :DEFAULT
        end
      )

      @spec encode(atom() | String.t()) :: integer() | atom()
      [
        (
          def encode(:DEFAULT) do
            0
          end

          def encode("DEFAULT") do
            0
          end
        ),
        (
          def encode(:EXTERNAL) do
            1
          end

          def encode("EXTERNAL") do
            1
          end
        )
      ]

      def encode(x) do
        x
      end

      @spec decode(integer()) :: atom() | integer()
      [
        def decode(0) do
          :DEFAULT
        end,
        def decode(1) do
          :EXTERNAL
        end
      ]

      def decode(x) do
        x
      end

      @spec constants() :: [{integer(), atom()}]
      def constants() do
        [{0, :DEFAULT}, {1, :EXTERNAL}]
      end

      @spec has_constant?(any()) :: boolean()
      (
        [
          def has_constant?(:DEFAULT) do
            true
          end,
          def has_constant?(:EXTERNAL) do
            true
          end
        ]

        def has_constant?(_) do
          false
        end
      )
    )
  end,
  defmodule Transmuter.Onnx.TensorProto.DataType do
    @moduledoc false
    (
      defstruct []

      (
        @spec default() :: :UNDEFINED
        def default() do
          :UNDEFINED
        end
      )

      @spec encode(atom() | String.t()) :: integer() | atom()
      [
        (
          def encode(:UNDEFINED) do
            0
          end

          def encode("UNDEFINED") do
            0
          end
        ),
        (
          def encode(:FLOAT) do
            1
          end

          def encode("FLOAT") do
            1
          end
        ),
        (
          def encode(:UINT8) do
            2
          end

          def encode("UINT8") do
            2
          end
        ),
        (
          def encode(:INT8) do
            3
          end

          def encode("INT8") do
            3
          end
        ),
        (
          def encode(:UINT16) do
            4
          end

          def encode("UINT16") do
            4
          end
        ),
        (
          def encode(:INT16) do
            5
          end

          def encode("INT16") do
            5
          end
        ),
        (
          def encode(:INT32) do
            6
          end

          def encode("INT32") do
            6
          end
        ),
        (
          def encode(:INT64) do
            7
          end

          def encode("INT64") do
            7
          end
        ),
        (
          def encode(:STRING) do
            8
          end

          def encode("STRING") do
            8
          end
        ),
        (
          def encode(:BOOL) do
            9
          end

          def encode("BOOL") do
            9
          end
        ),
        (
          def encode(:FLOAT16) do
            10
          end

          def encode("FLOAT16") do
            10
          end
        ),
        (
          def encode(:DOUBLE) do
            11
          end

          def encode("DOUBLE") do
            11
          end
        ),
        (
          def encode(:UINT32) do
            12
          end

          def encode("UINT32") do
            12
          end
        ),
        (
          def encode(:UINT64) do
            13
          end

          def encode("UINT64") do
            13
          end
        ),
        (
          def encode(:COMPLEX64) do
            14
          end

          def encode("COMPLEX64") do
            14
          end
        ),
        (
          def encode(:COMPLEX128) do
            15
          end

          def encode("COMPLEX128") do
            15
          end
        ),
        (
          def encode(:BFLOAT16) do
            16
          end

          def encode("BFLOAT16") do
            16
          end
        ),
        (
          def encode(:FLOAT8E4M3FN) do
            17
          end

          def encode("FLOAT8E4M3FN") do
            17
          end
        ),
        (
          def encode(:FLOAT8E4M3FNUZ) do
            18
          end

          def encode("FLOAT8E4M3FNUZ") do
            18
          end
        ),
        (
          def encode(:FLOAT8E5M2) do
            19
          end

          def encode("FLOAT8E5M2") do
            19
          end
        ),
        (
          def encode(:FLOAT8E5M2FNUZ) do
            20
          end

          def encode("FLOAT8E5M2FNUZ") do
            20
          end
        ),
        (
          def encode(:UINT4) do
            21
          end

          def encode("UINT4") do
            21
          end
        ),
        (
          def encode(:INT4) do
            22
          end

          def encode("INT4") do
            22
          end
        ),
        (
          def encode(:FLOAT4E2M1) do
            23
          end

          def encode("FLOAT4E2M1") do
            23
          end
        ),
        (
          def encode(:FLOAT8E8M0) do
            24
          end

          def encode("FLOAT8E8M0") do
            24
          end
        ),
        (
          def encode(:UINT2) do
            25
          end

          def encode("UINT2") do
            25
          end
        ),
        (
          def encode(:INT2) do
            26
          end

          def encode("INT2") do
            26
          end
        ),
        (
          def encode(:FLOAT6E2M3) do
            27
          end

          def encode("FLOAT6E2M3") do
            27
          end
        ),
        (
          def encode(:FLOAT6E3M2) do
            28
          end

          def encode("FLOAT6E3M2") do
            28
          end
        )
      ]

      def encode(x) do
        x
      end

      @spec decode(integer()) :: atom() | integer()
      [
        def decode(0) do
          :UNDEFINED
        end,
        def decode(1) do
          :FLOAT
        end,
        def decode(2) do
          :UINT8
        end,
        def decode(3) do
          :INT8
        end,
        def decode(4) do
          :UINT16
        end,
        def decode(5) do
          :INT16
        end,
        def decode(6) do
          :INT32
        end,
        def decode(7) do
          :INT64
        end,
        def decode(8) do
          :STRING
        end,
        def decode(9) do
          :BOOL
        end,
        def decode(10) do
          :FLOAT16
        end,
        def decode(11) do
          :DOUBLE
        end,
        def decode(12) do
          :UINT32
        end,
        def decode(13) do
          :UINT64
        end,
        def decode(14) do
          :COMPLEX64
        end,
        def decode(15) do
          :COMPLEX128
        end,
        def decode(16) do
          :BFLOAT16
        end,
        def decode(17) do
          :FLOAT8E4M3FN
        end,
        def decode(18) do
          :FLOAT8E4M3FNUZ
        end,
        def decode(19) do
          :FLOAT8E5M2
        end,
        def decode(20) do
          :FLOAT8E5M2FNUZ
        end,
        def decode(21) do
          :UINT4
        end,
        def decode(22) do
          :INT4
        end,
        def decode(23) do
          :FLOAT4E2M1
        end,
        def decode(24) do
          :FLOAT8E8M0
        end,
        def decode(25) do
          :UINT2
        end,
        def decode(26) do
          :INT2
        end,
        def decode(27) do
          :FLOAT6E2M3
        end,
        def decode(28) do
          :FLOAT6E3M2
        end
      ]

      def decode(x) do
        x
      end

      @spec constants() :: [{integer(), atom()}]
      def constants() do
        [
          {0, :UNDEFINED},
          {1, :FLOAT},
          {2, :UINT8},
          {3, :INT8},
          {4, :UINT16},
          {5, :INT16},
          {6, :INT32},
          {7, :INT64},
          {8, :STRING},
          {9, :BOOL},
          {10, :FLOAT16},
          {11, :DOUBLE},
          {12, :UINT32},
          {13, :UINT64},
          {14, :COMPLEX64},
          {15, :COMPLEX128},
          {16, :BFLOAT16},
          {17, :FLOAT8E4M3FN},
          {18, :FLOAT8E4M3FNUZ},
          {19, :FLOAT8E5M2},
          {20, :FLOAT8E5M2FNUZ},
          {21, :UINT4},
          {22, :INT4},
          {23, :FLOAT4E2M1},
          {24, :FLOAT8E8M0},
          {25, :UINT2},
          {26, :INT2},
          {27, :FLOAT6E2M3},
          {28, :FLOAT6E3M2}
        ]
      end

      @spec has_constant?(any()) :: boolean()
      (
        [
          def has_constant?(:UNDEFINED) do
            true
          end,
          def has_constant?(:FLOAT) do
            true
          end,
          def has_constant?(:UINT8) do
            true
          end,
          def has_constant?(:INT8) do
            true
          end,
          def has_constant?(:UINT16) do
            true
          end,
          def has_constant?(:INT16) do
            true
          end,
          def has_constant?(:INT32) do
            true
          end,
          def has_constant?(:INT64) do
            true
          end,
          def has_constant?(:STRING) do
            true
          end,
          def has_constant?(:BOOL) do
            true
          end,
          def has_constant?(:FLOAT16) do
            true
          end,
          def has_constant?(:DOUBLE) do
            true
          end,
          def has_constant?(:UINT32) do
            true
          end,
          def has_constant?(:UINT64) do
            true
          end,
          def has_constant?(:COMPLEX64) do
            true
          end,
          def has_constant?(:COMPLEX128) do
            true
          end,
          def has_constant?(:BFLOAT16) do
            true
          end,
          def has_constant?(:FLOAT8E4M3FN) do
            true
          end,
          def has_constant?(:FLOAT8E4M3FNUZ) do
            true
          end,
          def has_constant?(:FLOAT8E5M2) do
            true
          end,
          def has_constant?(:FLOAT8E5M2FNUZ) do
            true
          end,
          def has_constant?(:UINT4) do
            true
          end,
          def has_constant?(:INT4) do
            true
          end,
          def has_constant?(:FLOAT4E2M1) do
            true
          end,
          def has_constant?(:FLOAT8E8M0) do
            true
          end,
          def has_constant?(:UINT2) do
            true
          end,
          def has_constant?(:INT2) do
            true
          end,
          def has_constant?(:FLOAT6E2M3) do
            true
          end,
          def has_constant?(:FLOAT6E3M2) do
            true
          end
        ]

        def has_constant?(_) do
          false
        end
      )
    )
  end,
  defmodule Transmuter.Onnx.Version do
    @moduledoc false
    (
      defstruct []

      (
        @spec default() :: :_START_VERSION
        def default() do
          :_START_VERSION
        end
      )

      @spec encode(atom() | String.t()) :: integer() | atom()
      [
        (
          def encode(:_START_VERSION) do
            0
          end

          def encode("_START_VERSION") do
            0
          end
        ),
        (
          def encode(:IR_VERSION_2017_10_10) do
            1
          end

          def encode("IR_VERSION_2017_10_10") do
            1
          end
        ),
        (
          def encode(:IR_VERSION_2017_10_30) do
            2
          end

          def encode("IR_VERSION_2017_10_30") do
            2
          end
        ),
        (
          def encode(:IR_VERSION_2017_11_3) do
            3
          end

          def encode("IR_VERSION_2017_11_3") do
            3
          end
        ),
        (
          def encode(:IR_VERSION_2019_1_22) do
            4
          end

          def encode("IR_VERSION_2019_1_22") do
            4
          end
        ),
        (
          def encode(:IR_VERSION_2019_3_18) do
            5
          end

          def encode("IR_VERSION_2019_3_18") do
            5
          end
        ),
        (
          def encode(:IR_VERSION_2019_9_19) do
            6
          end

          def encode("IR_VERSION_2019_9_19") do
            6
          end
        ),
        (
          def encode(:IR_VERSION_2020_5_8) do
            7
          end

          def encode("IR_VERSION_2020_5_8") do
            7
          end
        ),
        (
          def encode(:IR_VERSION_2021_7_30) do
            8
          end

          def encode("IR_VERSION_2021_7_30") do
            8
          end
        ),
        (
          def encode(:IR_VERSION_2023_5_5) do
            9
          end

          def encode("IR_VERSION_2023_5_5") do
            9
          end
        ),
        (
          def encode(:IR_VERSION_2024_3_25) do
            10
          end

          def encode("IR_VERSION_2024_3_25") do
            10
          end
        ),
        (
          def encode(:IR_VERSION_2025_05_12) do
            11
          end

          def encode("IR_VERSION_2025_05_12") do
            11
          end
        ),
        (
          def encode(:IR_VERSION_2025_08_26) do
            12
          end

          def encode("IR_VERSION_2025_08_26") do
            12
          end
        ),
        (
          def encode(:IR_VERSION_2025_11_06) do
            13
          end

          def encode("IR_VERSION_2025_11_06") do
            13
          end
        ),
        (
          def encode(:IR_VERSION) do
            14
          end

          def encode("IR_VERSION") do
            14
          end
        )
      ]

      def encode(x) do
        x
      end

      @spec decode(integer()) :: atom() | integer()
      [
        def decode(0) do
          :_START_VERSION
        end,
        def decode(1) do
          :IR_VERSION_2017_10_10
        end,
        def decode(2) do
          :IR_VERSION_2017_10_30
        end,
        def decode(3) do
          :IR_VERSION_2017_11_3
        end,
        def decode(4) do
          :IR_VERSION_2019_1_22
        end,
        def decode(5) do
          :IR_VERSION_2019_3_18
        end,
        def decode(6) do
          :IR_VERSION_2019_9_19
        end,
        def decode(7) do
          :IR_VERSION_2020_5_8
        end,
        def decode(8) do
          :IR_VERSION_2021_7_30
        end,
        def decode(9) do
          :IR_VERSION_2023_5_5
        end,
        def decode(10) do
          :IR_VERSION_2024_3_25
        end,
        def decode(11) do
          :IR_VERSION_2025_05_12
        end,
        def decode(12) do
          :IR_VERSION_2025_08_26
        end,
        def decode(13) do
          :IR_VERSION_2025_11_06
        end,
        def decode(14) do
          :IR_VERSION
        end
      ]

      def decode(x) do
        x
      end

      @spec constants() :: [{integer(), atom()}]
      def constants() do
        [
          {0, :_START_VERSION},
          {1, :IR_VERSION_2017_10_10},
          {2, :IR_VERSION_2017_10_30},
          {3, :IR_VERSION_2017_11_3},
          {4, :IR_VERSION_2019_1_22},
          {5, :IR_VERSION_2019_3_18},
          {6, :IR_VERSION_2019_9_19},
          {7, :IR_VERSION_2020_5_8},
          {8, :IR_VERSION_2021_7_30},
          {9, :IR_VERSION_2023_5_5},
          {10, :IR_VERSION_2024_3_25},
          {11, :IR_VERSION_2025_05_12},
          {12, :IR_VERSION_2025_08_26},
          {13, :IR_VERSION_2025_11_06},
          {14, :IR_VERSION}
        ]
      end

      @spec has_constant?(any()) :: boolean()
      (
        [
          def has_constant?(:_START_VERSION) do
            true
          end,
          def has_constant?(:IR_VERSION_2017_10_10) do
            true
          end,
          def has_constant?(:IR_VERSION_2017_10_30) do
            true
          end,
          def has_constant?(:IR_VERSION_2017_11_3) do
            true
          end,
          def has_constant?(:IR_VERSION_2019_1_22) do
            true
          end,
          def has_constant?(:IR_VERSION_2019_3_18) do
            true
          end,
          def has_constant?(:IR_VERSION_2019_9_19) do
            true
          end,
          def has_constant?(:IR_VERSION_2020_5_8) do
            true
          end,
          def has_constant?(:IR_VERSION_2021_7_30) do
            true
          end,
          def has_constant?(:IR_VERSION_2023_5_5) do
            true
          end,
          def has_constant?(:IR_VERSION_2024_3_25) do
            true
          end,
          def has_constant?(:IR_VERSION_2025_05_12) do
            true
          end,
          def has_constant?(:IR_VERSION_2025_08_26) do
            true
          end,
          def has_constant?(:IR_VERSION_2025_11_06) do
            true
          end,
          def has_constant?(:IR_VERSION) do
            true
          end
        ]

        def has_constant?(_) do
          false
        end
      )
    )
  end,
  defmodule Transmuter.Onnx.AttributeProto do
    @moduledoc false
    defstruct name: nil,
              f: nil,
              i: nil,
              s: nil,
              t: nil,
              g: nil,
              floats: [],
              ints: [],
              strings: [],
              tensors: [],
              graphs: [],
              doc_string: nil,
              tp: nil,
              type_protos: [],
              type: nil,
              ref_attr_name: nil,
              sparse_tensor: nil,
              sparse_tensors: [],
              __uf__: []

    (
      (
        @spec encode(struct) :: {:ok, iodata} | {:error, any}
        def encode(msg) do
          try do
            {:ok, encode!(msg)}
          rescue
            e in [Protox.EncodingError, Protox.RequiredFieldsError] -> {:error, e}
          end
        end

        @spec encode!(struct) :: iodata | no_return
        def encode!(msg) do
          []
          |> encode_name(msg)
          |> encode_f(msg)
          |> encode_i(msg)
          |> encode_s(msg)
          |> encode_t(msg)
          |> encode_g(msg)
          |> encode_floats(msg)
          |> encode_ints(msg)
          |> encode_strings(msg)
          |> encode_tensors(msg)
          |> encode_graphs(msg)
          |> encode_doc_string(msg)
          |> encode_tp(msg)
          |> encode_type_protos(msg)
          |> encode_type(msg)
          |> encode_ref_attr_name(msg)
          |> encode_sparse_tensor(msg)
          |> encode_sparse_tensors(msg)
          |> encode_unknown_fields(msg)
        end
      )

      []

      [
        defp encode_name(acc, msg) do
          try do
            case msg.name do
              nil -> acc
              _ -> [acc, "\n", Protox.Encode.encode_string(msg.name)]
            end
          rescue
            ArgumentError ->
              reraise Protox.EncodingError.new(:name, "invalid field value"), __STACKTRACE__
          end
        end,
        defp encode_f(acc, msg) do
          try do
            case msg.f do
              nil -> acc
              _ -> [acc, "\x15", Protox.Encode.encode_float(msg.f)]
            end
          rescue
            ArgumentError ->
              reraise Protox.EncodingError.new(:f, "invalid field value"), __STACKTRACE__
          end
        end,
        defp encode_i(acc, msg) do
          try do
            case msg.i do
              nil -> acc
              _ -> [acc, "\x18", Protox.Encode.encode_int64(msg.i)]
            end
          rescue
            ArgumentError ->
              reraise Protox.EncodingError.new(:i, "invalid field value"), __STACKTRACE__
          end
        end,
        defp encode_s(acc, msg) do
          try do
            case msg.s do
              nil -> acc
              _ -> [acc, "\"", Protox.Encode.encode_bytes(msg.s)]
            end
          rescue
            ArgumentError ->
              reraise Protox.EncodingError.new(:s, "invalid field value"), __STACKTRACE__
          end
        end,
        defp encode_t(acc, msg) do
          try do
            case msg.t do
              nil -> acc
              _ -> [acc, "*", Protox.Encode.encode_message(msg.t)]
            end
          rescue
            ArgumentError ->
              reraise Protox.EncodingError.new(:t, "invalid field value"), __STACKTRACE__
          end
        end,
        defp encode_g(acc, msg) do
          try do
            case msg.g do
              nil -> acc
              _ -> [acc, "2", Protox.Encode.encode_message(msg.g)]
            end
          rescue
            ArgumentError ->
              reraise Protox.EncodingError.new(:g, "invalid field value"), __STACKTRACE__
          end
        end,
        defp encode_floats(acc, msg) do
          try do
            case msg.floats do
              [] ->
                acc

              values ->
                [
                  acc,
                  Enum.reduce(values, [], fn value, acc ->
                    [acc, "=", Protox.Encode.encode_float(value)]
                  end)
                ]
            end
          rescue
            ArgumentError ->
              reraise Protox.EncodingError.new(:floats, "invalid field value"), __STACKTRACE__
          end
        end,
        defp encode_ints(acc, msg) do
          try do
            case msg.ints do
              [] ->
                acc

              values ->
                [
                  acc,
                  Enum.reduce(values, [], fn value, acc ->
                    [acc, "@", Protox.Encode.encode_int64(value)]
                  end)
                ]
            end
          rescue
            ArgumentError ->
              reraise Protox.EncodingError.new(:ints, "invalid field value"), __STACKTRACE__
          end
        end,
        defp encode_strings(acc, msg) do
          try do
            case msg.strings do
              [] ->
                acc

              values ->
                [
                  acc,
                  Enum.reduce(values, [], fn value, acc ->
                    [acc, "J", Protox.Encode.encode_bytes(value)]
                  end)
                ]
            end
          rescue
            ArgumentError ->
              reraise Protox.EncodingError.new(:strings, "invalid field value"), __STACKTRACE__
          end
        end,
        defp encode_tensors(acc, msg) do
          try do
            case msg.tensors do
              [] ->
                acc

              values ->
                [
                  acc,
                  Enum.reduce(values, [], fn value, acc ->
                    [acc, "R", Protox.Encode.encode_message(value)]
                  end)
                ]
            end
          rescue
            ArgumentError ->
              reraise Protox.EncodingError.new(:tensors, "invalid field value"), __STACKTRACE__
          end
        end,
        defp encode_graphs(acc, msg) do
          try do
            case msg.graphs do
              [] ->
                acc

              values ->
                [
                  acc,
                  Enum.reduce(values, [], fn value, acc ->
                    [acc, "Z", Protox.Encode.encode_message(value)]
                  end)
                ]
            end
          rescue
            ArgumentError ->
              reraise Protox.EncodingError.new(:graphs, "invalid field value"), __STACKTRACE__
          end
        end,
        defp encode_doc_string(acc, msg) do
          try do
            case msg.doc_string do
              nil -> acc
              _ -> [acc, "j", Protox.Encode.encode_string(msg.doc_string)]
            end
          rescue
            ArgumentError ->
              reraise Protox.EncodingError.new(:doc_string, "invalid field value"), __STACKTRACE__
          end
        end,
        defp encode_tp(acc, msg) do
          try do
            case msg.tp do
              nil -> acc
              _ -> [acc, "r", Protox.Encode.encode_message(msg.tp)]
            end
          rescue
            ArgumentError ->
              reraise Protox.EncodingError.new(:tp, "invalid field value"), __STACKTRACE__
          end
        end,
        defp encode_type_protos(acc, msg) do
          try do
            case msg.type_protos do
              [] ->
                acc

              values ->
                [
                  acc,
                  Enum.reduce(values, [], fn value, acc ->
                    [acc, "z", Protox.Encode.encode_message(value)]
                  end)
                ]
            end
          rescue
            ArgumentError ->
              reraise Protox.EncodingError.new(:type_protos, "invalid field value"),
                      __STACKTRACE__
          end
        end,
        defp encode_type(acc, msg) do
          try do
            case msg.type do
              nil ->
                acc

              _ ->
                [
                  acc,
                  "\xA0\x01",
                  msg.type
                  |> Transmuter.Onnx.AttributeProto.AttributeType.encode()
                  |> Protox.Encode.encode_enum()
                ]
            end
          rescue
            ArgumentError ->
              reraise Protox.EncodingError.new(:type, "invalid field value"), __STACKTRACE__
          end
        end,
        defp encode_ref_attr_name(acc, msg) do
          try do
            case msg.ref_attr_name do
              nil -> acc
              _ -> [acc, "\xAA\x01", Protox.Encode.encode_string(msg.ref_attr_name)]
            end
          rescue
            ArgumentError ->
              reraise Protox.EncodingError.new(:ref_attr_name, "invalid field value"),
                      __STACKTRACE__
          end
        end,
        defp encode_sparse_tensor(acc, msg) do
          try do
            case msg.sparse_tensor do
              nil -> acc
              _ -> [acc, "\xB2\x01", Protox.Encode.encode_message(msg.sparse_tensor)]
            end
          rescue
            ArgumentError ->
              reraise Protox.EncodingError.new(:sparse_tensor, "invalid field value"),
                      __STACKTRACE__
          end
        end,
        defp encode_sparse_tensors(acc, msg) do
          try do
            case msg.sparse_tensors do
              [] ->
                acc

              values ->
                [
                  acc,
                  Enum.reduce(values, [], fn value, acc ->
                    [acc, "\xBA\x01", Protox.Encode.encode_message(value)]
                  end)
                ]
            end
          rescue
            ArgumentError ->
              reraise Protox.EncodingError.new(:sparse_tensors, "invalid field value"),
                      __STACKTRACE__
          end
        end
      ]

      defp encode_unknown_fields(acc, msg) do
        Enum.reduce(msg.__struct__.unknown_fields(msg), acc, fn {tag, wire_type, bytes}, acc ->
          case wire_type do
            0 ->
              [acc, Protox.Encode.make_key_bytes(tag, :int32), bytes]

            1 ->
              [acc, Protox.Encode.make_key_bytes(tag, :double), bytes]

            2 ->
              len_bytes = bytes |> byte_size() |> Protox.Varint.encode()
              [acc, Protox.Encode.make_key_bytes(tag, :packed), len_bytes, bytes]

            5 ->
              [acc, Protox.Encode.make_key_bytes(tag, :float), bytes]
          end
        end)
      end
    )

    (
      (
        @spec decode(binary) :: {:ok, struct} | {:error, any}
        def decode(bytes) do
          try do
            {:ok, decode!(bytes)}
          rescue
            e in [Protox.DecodingError, Protox.IllegalTagError, Protox.RequiredFieldsError] ->
              {:error, e}
          end
        end

        (
          @spec decode!(binary) :: struct | no_return
          def decode!(bytes) do
            parse_key_value(bytes, struct(Transmuter.Onnx.AttributeProto))
          end
        )
      )

      (
        @spec parse_key_value(binary, struct) :: struct
        defp parse_key_value(<<>>, msg) do
          msg
        end

        defp parse_key_value(bytes, msg) do
          {field, rest} =
            case Protox.Decode.parse_key(bytes) do
              {0, _, _} ->
                raise %Protox.IllegalTagError{}

              {1, _, bytes} ->
                {len, bytes} = Protox.Varint.decode(bytes)
                {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)
                {[name: Protox.Decode.validate_string!(delimited)], rest}

              {2, _, bytes} ->
                {value, rest} = Protox.Decode.parse_float(bytes)
                {[f: value], rest}

              {3, _, bytes} ->
                {value, rest} = Protox.Decode.parse_int64(bytes)
                {[i: value], rest}

              {4, _, bytes} ->
                {len, bytes} = Protox.Varint.decode(bytes)
                {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)
                {[s: delimited], rest}

              {5, _, bytes} ->
                {len, bytes} = Protox.Varint.decode(bytes)
                {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)

                {[
                   t:
                     Protox.MergeMessage.merge(
                       msg.t,
                       Transmuter.Onnx.TensorProto.decode!(delimited)
                     )
                 ], rest}

              {6, _, bytes} ->
                {len, bytes} = Protox.Varint.decode(bytes)
                {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)

                {[
                   g:
                     Protox.MergeMessage.merge(
                       msg.g,
                       Transmuter.Onnx.GraphProto.decode!(delimited)
                     )
                 ], rest}

              {7, 2, bytes} ->
                {len, bytes} = Protox.Varint.decode(bytes)
                {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)
                {[floats: msg.floats ++ Protox.Decode.parse_repeated_float([], delimited)], rest}

              {7, _, bytes} ->
                {value, rest} = Protox.Decode.parse_float(bytes)
                {[floats: msg.floats ++ [value]], rest}

              {8, 2, bytes} ->
                {len, bytes} = Protox.Varint.decode(bytes)
                {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)
                {[ints: msg.ints ++ Protox.Decode.parse_repeated_int64([], delimited)], rest}

              {8, _, bytes} ->
                {value, rest} = Protox.Decode.parse_int64(bytes)
                {[ints: msg.ints ++ [value]], rest}

              {9, _, bytes} ->
                {len, bytes} = Protox.Varint.decode(bytes)
                {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)
                {[strings: msg.strings ++ [delimited]], rest}

              {10, _, bytes} ->
                {len, bytes} = Protox.Varint.decode(bytes)
                {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)
                {[tensors: msg.tensors ++ [Transmuter.Onnx.TensorProto.decode!(delimited)]], rest}

              {11, _, bytes} ->
                {len, bytes} = Protox.Varint.decode(bytes)
                {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)
                {[graphs: msg.graphs ++ [Transmuter.Onnx.GraphProto.decode!(delimited)]], rest}

              {13, _, bytes} ->
                {len, bytes} = Protox.Varint.decode(bytes)
                {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)
                {[doc_string: Protox.Decode.validate_string!(delimited)], rest}

              {14, _, bytes} ->
                {len, bytes} = Protox.Varint.decode(bytes)
                {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)

                {[
                   tp:
                     Protox.MergeMessage.merge(
                       msg.tp,
                       Transmuter.Onnx.TypeProto.decode!(delimited)
                     )
                 ], rest}

              {15, _, bytes} ->
                {len, bytes} = Protox.Varint.decode(bytes)
                {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)

                {[type_protos: msg.type_protos ++ [Transmuter.Onnx.TypeProto.decode!(delimited)]],
                 rest}

              {20, _, bytes} ->
                {value, rest} =
                  Protox.Decode.parse_enum(bytes, Transmuter.Onnx.AttributeProto.AttributeType)

                {[type: value], rest}

              {21, _, bytes} ->
                {len, bytes} = Protox.Varint.decode(bytes)
                {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)
                {[ref_attr_name: Protox.Decode.validate_string!(delimited)], rest}

              {22, _, bytes} ->
                {len, bytes} = Protox.Varint.decode(bytes)
                {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)

                {[
                   sparse_tensor:
                     Protox.MergeMessage.merge(
                       msg.sparse_tensor,
                       Transmuter.Onnx.SparseTensorProto.decode!(delimited)
                     )
                 ], rest}

              {23, _, bytes} ->
                {len, bytes} = Protox.Varint.decode(bytes)
                {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)

                {[
                   sparse_tensors:
                     msg.sparse_tensors ++ [Transmuter.Onnx.SparseTensorProto.decode!(delimited)]
                 ], rest}

              {tag, wire_type, rest} ->
                {value, rest} = Protox.Decode.parse_unknown(tag, wire_type, rest)

                {[
                   {msg.__struct__.unknown_fields_name(),
                    [value | msg.__struct__.unknown_fields(msg)]}
                 ], rest}
            end

          msg_updated = struct(msg, field)
          parse_key_value(rest, msg_updated)
        end
      )

      []
    )

    (
      @spec json_decode(iodata(), keyword()) :: {:ok, struct()} | {:error, any()}
      def json_decode(input, opts \\ []) do
        try do
          {:ok, json_decode!(input, opts)}
        rescue
          e in Protox.JsonDecodingError -> {:error, e}
        end
      end

      @spec json_decode!(iodata(), keyword()) :: struct() | no_return()
      def json_decode!(input, opts \\ []) do
        {json_library_wrapper, json_library} = Protox.JsonLibrary.get_library(opts, :decode)

        Protox.JsonDecode.decode!(
          input,
          Transmuter.Onnx.AttributeProto,
          &json_library_wrapper.decode!(json_library, &1)
        )
      end

      @spec json_encode(struct(), keyword()) :: {:ok, iodata()} | {:error, any()}
      def json_encode(msg, opts \\ []) do
        try do
          {:ok, json_encode!(msg, opts)}
        rescue
          e in Protox.JsonEncodingError -> {:error, e}
        end
      end

      @spec json_encode!(struct(), keyword()) :: iodata() | no_return()
      def json_encode!(msg, opts \\ []) do
        {json_library_wrapper, json_library} = Protox.JsonLibrary.get_library(opts, :encode)
        Protox.JsonEncode.encode!(msg, &json_library_wrapper.encode!(json_library, &1))
      end
    )

    (
      @deprecated "Use fields_defs()/0 instead"
      @spec defs() :: %{
              required(non_neg_integer) => {atom, Protox.Types.kind(), Protox.Types.type()}
            }
      def defs() do
        %{
          1 => {:name, {:scalar, ""}, :string},
          2 => {:f, {:scalar, 0.0}, :float},
          3 => {:i, {:scalar, 0}, :int64},
          4 => {:s, {:scalar, ""}, :bytes},
          5 => {:t, {:scalar, nil}, {:message, Transmuter.Onnx.TensorProto}},
          6 => {:g, {:scalar, nil}, {:message, Transmuter.Onnx.GraphProto}},
          7 => {:floats, :unpacked, :float},
          8 => {:ints, :unpacked, :int64},
          9 => {:strings, :unpacked, :bytes},
          10 => {:tensors, :unpacked, {:message, Transmuter.Onnx.TensorProto}},
          11 => {:graphs, :unpacked, {:message, Transmuter.Onnx.GraphProto}},
          13 => {:doc_string, {:scalar, ""}, :string},
          14 => {:tp, {:scalar, nil}, {:message, Transmuter.Onnx.TypeProto}},
          15 => {:type_protos, :unpacked, {:message, Transmuter.Onnx.TypeProto}},
          20 =>
            {:type, {:scalar, :UNDEFINED}, {:enum, Transmuter.Onnx.AttributeProto.AttributeType}},
          21 => {:ref_attr_name, {:scalar, ""}, :string},
          22 => {:sparse_tensor, {:scalar, nil}, {:message, Transmuter.Onnx.SparseTensorProto}},
          23 => {:sparse_tensors, :unpacked, {:message, Transmuter.Onnx.SparseTensorProto}}
        }
      end

      @deprecated "Use fields_defs()/0 instead"
      @spec defs_by_name() :: %{
              required(atom) => {non_neg_integer, Protox.Types.kind(), Protox.Types.type()}
            }
      def defs_by_name() do
        %{
          doc_string: {13, {:scalar, ""}, :string},
          f: {2, {:scalar, 0.0}, :float},
          floats: {7, :unpacked, :float},
          g: {6, {:scalar, nil}, {:message, Transmuter.Onnx.GraphProto}},
          graphs: {11, :unpacked, {:message, Transmuter.Onnx.GraphProto}},
          i: {3, {:scalar, 0}, :int64},
          ints: {8, :unpacked, :int64},
          name: {1, {:scalar, ""}, :string},
          ref_attr_name: {21, {:scalar, ""}, :string},
          s: {4, {:scalar, ""}, :bytes},
          sparse_tensor: {22, {:scalar, nil}, {:message, Transmuter.Onnx.SparseTensorProto}},
          sparse_tensors: {23, :unpacked, {:message, Transmuter.Onnx.SparseTensorProto}},
          strings: {9, :unpacked, :bytes},
          t: {5, {:scalar, nil}, {:message, Transmuter.Onnx.TensorProto}},
          tensors: {10, :unpacked, {:message, Transmuter.Onnx.TensorProto}},
          tp: {14, {:scalar, nil}, {:message, Transmuter.Onnx.TypeProto}},
          type:
            {20, {:scalar, :UNDEFINED}, {:enum, Transmuter.Onnx.AttributeProto.AttributeType}},
          type_protos: {15, :unpacked, {:message, Transmuter.Onnx.TypeProto}}
        }
      end
    )

    (
      @spec fields_defs() :: list(Protox.Field.t())
      def fields_defs() do
        [
          %{
            __struct__: Protox.Field,
            json_name: "name",
            kind: {:scalar, ""},
            label: :optional,
            name: :name,
            tag: 1,
            type: :string
          },
          %{
            __struct__: Protox.Field,
            json_name: "f",
            kind: {:scalar, 0.0},
            label: :optional,
            name: :f,
            tag: 2,
            type: :float
          },
          %{
            __struct__: Protox.Field,
            json_name: "i",
            kind: {:scalar, 0},
            label: :optional,
            name: :i,
            tag: 3,
            type: :int64
          },
          %{
            __struct__: Protox.Field,
            json_name: "s",
            kind: {:scalar, ""},
            label: :optional,
            name: :s,
            tag: 4,
            type: :bytes
          },
          %{
            __struct__: Protox.Field,
            json_name: "t",
            kind: {:scalar, nil},
            label: :optional,
            name: :t,
            tag: 5,
            type: {:message, Transmuter.Onnx.TensorProto}
          },
          %{
            __struct__: Protox.Field,
            json_name: "g",
            kind: {:scalar, nil},
            label: :optional,
            name: :g,
            tag: 6,
            type: {:message, Transmuter.Onnx.GraphProto}
          },
          %{
            __struct__: Protox.Field,
            json_name: "floats",
            kind: :unpacked,
            label: :repeated,
            name: :floats,
            tag: 7,
            type: :float
          },
          %{
            __struct__: Protox.Field,
            json_name: "ints",
            kind: :unpacked,
            label: :repeated,
            name: :ints,
            tag: 8,
            type: :int64
          },
          %{
            __struct__: Protox.Field,
            json_name: "strings",
            kind: :unpacked,
            label: :repeated,
            name: :strings,
            tag: 9,
            type: :bytes
          },
          %{
            __struct__: Protox.Field,
            json_name: "tensors",
            kind: :unpacked,
            label: :repeated,
            name: :tensors,
            tag: 10,
            type: {:message, Transmuter.Onnx.TensorProto}
          },
          %{
            __struct__: Protox.Field,
            json_name: "graphs",
            kind: :unpacked,
            label: :repeated,
            name: :graphs,
            tag: 11,
            type: {:message, Transmuter.Onnx.GraphProto}
          },
          %{
            __struct__: Protox.Field,
            json_name: "docString",
            kind: {:scalar, ""},
            label: :optional,
            name: :doc_string,
            tag: 13,
            type: :string
          },
          %{
            __struct__: Protox.Field,
            json_name: "tp",
            kind: {:scalar, nil},
            label: :optional,
            name: :tp,
            tag: 14,
            type: {:message, Transmuter.Onnx.TypeProto}
          },
          %{
            __struct__: Protox.Field,
            json_name: "typeProtos",
            kind: :unpacked,
            label: :repeated,
            name: :type_protos,
            tag: 15,
            type: {:message, Transmuter.Onnx.TypeProto}
          },
          %{
            __struct__: Protox.Field,
            json_name: "type",
            kind: {:scalar, :UNDEFINED},
            label: :optional,
            name: :type,
            tag: 20,
            type: {:enum, Transmuter.Onnx.AttributeProto.AttributeType}
          },
          %{
            __struct__: Protox.Field,
            json_name: "refAttrName",
            kind: {:scalar, ""},
            label: :optional,
            name: :ref_attr_name,
            tag: 21,
            type: :string
          },
          %{
            __struct__: Protox.Field,
            json_name: "sparseTensor",
            kind: {:scalar, nil},
            label: :optional,
            name: :sparse_tensor,
            tag: 22,
            type: {:message, Transmuter.Onnx.SparseTensorProto}
          },
          %{
            __struct__: Protox.Field,
            json_name: "sparseTensors",
            kind: :unpacked,
            label: :repeated,
            name: :sparse_tensors,
            tag: 23,
            type: {:message, Transmuter.Onnx.SparseTensorProto}
          }
        ]
      end

      [
        @spec(field_def(atom) :: {:ok, Protox.Field.t()} | {:error, :no_such_field}),
        (
          def field_def(:name) do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "name",
               kind: {:scalar, ""},
               label: :optional,
               name: :name,
               tag: 1,
               type: :string
             }}
          end

          def field_def("name") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "name",
               kind: {:scalar, ""},
               label: :optional,
               name: :name,
               tag: 1,
               type: :string
             }}
          end

          []
        ),
        (
          def field_def(:f) do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "f",
               kind: {:scalar, 0.0},
               label: :optional,
               name: :f,
               tag: 2,
               type: :float
             }}
          end

          def field_def("f") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "f",
               kind: {:scalar, 0.0},
               label: :optional,
               name: :f,
               tag: 2,
               type: :float
             }}
          end

          []
        ),
        (
          def field_def(:i) do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "i",
               kind: {:scalar, 0},
               label: :optional,
               name: :i,
               tag: 3,
               type: :int64
             }}
          end

          def field_def("i") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "i",
               kind: {:scalar, 0},
               label: :optional,
               name: :i,
               tag: 3,
               type: :int64
             }}
          end

          []
        ),
        (
          def field_def(:s) do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "s",
               kind: {:scalar, ""},
               label: :optional,
               name: :s,
               tag: 4,
               type: :bytes
             }}
          end

          def field_def("s") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "s",
               kind: {:scalar, ""},
               label: :optional,
               name: :s,
               tag: 4,
               type: :bytes
             }}
          end

          []
        ),
        (
          def field_def(:t) do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "t",
               kind: {:scalar, nil},
               label: :optional,
               name: :t,
               tag: 5,
               type: {:message, Transmuter.Onnx.TensorProto}
             }}
          end

          def field_def("t") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "t",
               kind: {:scalar, nil},
               label: :optional,
               name: :t,
               tag: 5,
               type: {:message, Transmuter.Onnx.TensorProto}
             }}
          end

          []
        ),
        (
          def field_def(:g) do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "g",
               kind: {:scalar, nil},
               label: :optional,
               name: :g,
               tag: 6,
               type: {:message, Transmuter.Onnx.GraphProto}
             }}
          end

          def field_def("g") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "g",
               kind: {:scalar, nil},
               label: :optional,
               name: :g,
               tag: 6,
               type: {:message, Transmuter.Onnx.GraphProto}
             }}
          end

          []
        ),
        (
          def field_def(:floats) do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "floats",
               kind: :unpacked,
               label: :repeated,
               name: :floats,
               tag: 7,
               type: :float
             }}
          end

          def field_def("floats") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "floats",
               kind: :unpacked,
               label: :repeated,
               name: :floats,
               tag: 7,
               type: :float
             }}
          end

          []
        ),
        (
          def field_def(:ints) do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "ints",
               kind: :unpacked,
               label: :repeated,
               name: :ints,
               tag: 8,
               type: :int64
             }}
          end

          def field_def("ints") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "ints",
               kind: :unpacked,
               label: :repeated,
               name: :ints,
               tag: 8,
               type: :int64
             }}
          end

          []
        ),
        (
          def field_def(:strings) do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "strings",
               kind: :unpacked,
               label: :repeated,
               name: :strings,
               tag: 9,
               type: :bytes
             }}
          end

          def field_def("strings") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "strings",
               kind: :unpacked,
               label: :repeated,
               name: :strings,
               tag: 9,
               type: :bytes
             }}
          end

          []
        ),
        (
          def field_def(:tensors) do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "tensors",
               kind: :unpacked,
               label: :repeated,
               name: :tensors,
               tag: 10,
               type: {:message, Transmuter.Onnx.TensorProto}
             }}
          end

          def field_def("tensors") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "tensors",
               kind: :unpacked,
               label: :repeated,
               name: :tensors,
               tag: 10,
               type: {:message, Transmuter.Onnx.TensorProto}
             }}
          end

          []
        ),
        (
          def field_def(:graphs) do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "graphs",
               kind: :unpacked,
               label: :repeated,
               name: :graphs,
               tag: 11,
               type: {:message, Transmuter.Onnx.GraphProto}
             }}
          end

          def field_def("graphs") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "graphs",
               kind: :unpacked,
               label: :repeated,
               name: :graphs,
               tag: 11,
               type: {:message, Transmuter.Onnx.GraphProto}
             }}
          end

          []
        ),
        (
          def field_def(:doc_string) do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "docString",
               kind: {:scalar, ""},
               label: :optional,
               name: :doc_string,
               tag: 13,
               type: :string
             }}
          end

          def field_def("docString") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "docString",
               kind: {:scalar, ""},
               label: :optional,
               name: :doc_string,
               tag: 13,
               type: :string
             }}
          end

          def field_def("doc_string") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "docString",
               kind: {:scalar, ""},
               label: :optional,
               name: :doc_string,
               tag: 13,
               type: :string
             }}
          end
        ),
        (
          def field_def(:tp) do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "tp",
               kind: {:scalar, nil},
               label: :optional,
               name: :tp,
               tag: 14,
               type: {:message, Transmuter.Onnx.TypeProto}
             }}
          end

          def field_def("tp") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "tp",
               kind: {:scalar, nil},
               label: :optional,
               name: :tp,
               tag: 14,
               type: {:message, Transmuter.Onnx.TypeProto}
             }}
          end

          []
        ),
        (
          def field_def(:type_protos) do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "typeProtos",
               kind: :unpacked,
               label: :repeated,
               name: :type_protos,
               tag: 15,
               type: {:message, Transmuter.Onnx.TypeProto}
             }}
          end

          def field_def("typeProtos") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "typeProtos",
               kind: :unpacked,
               label: :repeated,
               name: :type_protos,
               tag: 15,
               type: {:message, Transmuter.Onnx.TypeProto}
             }}
          end

          def field_def("type_protos") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "typeProtos",
               kind: :unpacked,
               label: :repeated,
               name: :type_protos,
               tag: 15,
               type: {:message, Transmuter.Onnx.TypeProto}
             }}
          end
        ),
        (
          def field_def(:type) do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "type",
               kind: {:scalar, :UNDEFINED},
               label: :optional,
               name: :type,
               tag: 20,
               type: {:enum, Transmuter.Onnx.AttributeProto.AttributeType}
             }}
          end

          def field_def("type") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "type",
               kind: {:scalar, :UNDEFINED},
               label: :optional,
               name: :type,
               tag: 20,
               type: {:enum, Transmuter.Onnx.AttributeProto.AttributeType}
             }}
          end

          []
        ),
        (
          def field_def(:ref_attr_name) do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "refAttrName",
               kind: {:scalar, ""},
               label: :optional,
               name: :ref_attr_name,
               tag: 21,
               type: :string
             }}
          end

          def field_def("refAttrName") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "refAttrName",
               kind: {:scalar, ""},
               label: :optional,
               name: :ref_attr_name,
               tag: 21,
               type: :string
             }}
          end

          def field_def("ref_attr_name") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "refAttrName",
               kind: {:scalar, ""},
               label: :optional,
               name: :ref_attr_name,
               tag: 21,
               type: :string
             }}
          end
        ),
        (
          def field_def(:sparse_tensor) do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "sparseTensor",
               kind: {:scalar, nil},
               label: :optional,
               name: :sparse_tensor,
               tag: 22,
               type: {:message, Transmuter.Onnx.SparseTensorProto}
             }}
          end

          def field_def("sparseTensor") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "sparseTensor",
               kind: {:scalar, nil},
               label: :optional,
               name: :sparse_tensor,
               tag: 22,
               type: {:message, Transmuter.Onnx.SparseTensorProto}
             }}
          end

          def field_def("sparse_tensor") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "sparseTensor",
               kind: {:scalar, nil},
               label: :optional,
               name: :sparse_tensor,
               tag: 22,
               type: {:message, Transmuter.Onnx.SparseTensorProto}
             }}
          end
        ),
        (
          def field_def(:sparse_tensors) do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "sparseTensors",
               kind: :unpacked,
               label: :repeated,
               name: :sparse_tensors,
               tag: 23,
               type: {:message, Transmuter.Onnx.SparseTensorProto}
             }}
          end

          def field_def("sparseTensors") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "sparseTensors",
               kind: :unpacked,
               label: :repeated,
               name: :sparse_tensors,
               tag: 23,
               type: {:message, Transmuter.Onnx.SparseTensorProto}
             }}
          end

          def field_def("sparse_tensors") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "sparseTensors",
               kind: :unpacked,
               label: :repeated,
               name: :sparse_tensors,
               tag: 23,
               type: {:message, Transmuter.Onnx.SparseTensorProto}
             }}
          end
        ),
        def field_def(_) do
          {:error, :no_such_field}
        end
      ]
    )

    (
      @spec unknown_fields(struct) :: [{non_neg_integer, Protox.Types.tag(), binary}]
      def unknown_fields(msg) do
        msg.__uf__
      end

      @spec unknown_fields_name() :: :__uf__
      def unknown_fields_name() do
        :__uf__
      end

      @spec clear_unknown_fields(struct) :: struct
      def clear_unknown_fields(msg) do
        struct!(msg, [{unknown_fields_name(), []}])
      end
    )

    (
      @spec required_fields() :: []
      def required_fields() do
        []
      end
    )

    (
      @spec syntax() :: atom()
      def syntax() do
        :proto2
      end
    )

    [
      @spec(default(atom) :: {:ok, boolean | integer | String.t() | float} | {:error, atom}),
      def default(:name) do
        {:ok, ""}
      end,
      def default(:f) do
        {:ok, 0.0}
      end,
      def default(:i) do
        {:ok, 0}
      end,
      def default(:s) do
        {:ok, ""}
      end,
      def default(:t) do
        {:ok, nil}
      end,
      def default(:g) do
        {:ok, nil}
      end,
      def default(:floats) do
        {:error, :no_default_value}
      end,
      def default(:ints) do
        {:error, :no_default_value}
      end,
      def default(:strings) do
        {:error, :no_default_value}
      end,
      def default(:tensors) do
        {:error, :no_default_value}
      end,
      def default(:graphs) do
        {:error, :no_default_value}
      end,
      def default(:doc_string) do
        {:ok, ""}
      end,
      def default(:tp) do
        {:ok, nil}
      end,
      def default(:type_protos) do
        {:error, :no_default_value}
      end,
      def default(:type) do
        {:ok, :UNDEFINED}
      end,
      def default(:ref_attr_name) do
        {:ok, ""}
      end,
      def default(:sparse_tensor) do
        {:ok, nil}
      end,
      def default(:sparse_tensors) do
        {:error, :no_default_value}
      end,
      def default(_) do
        {:error, :no_such_field}
      end
    ]

    (
      @spec file_options() :: struct()
      def file_options() do
        file_options = %{
          __struct__: Protox.Google.Protobuf.FileOptions,
          __uf__: [],
          cc_enable_arenas: nil,
          cc_generic_services: nil,
          csharp_namespace: nil,
          deprecated: nil,
          go_package: nil,
          java_generate_equals_and_hash: nil,
          java_generic_services: nil,
          java_multiple_files: nil,
          java_outer_classname: nil,
          java_package: nil,
          java_string_check_utf8: nil,
          objc_class_prefix: nil,
          optimize_for: :LITE_RUNTIME,
          php_class_prefix: nil,
          php_generic_services: nil,
          php_metadata_namespace: nil,
          php_namespace: nil,
          py_generic_services: nil,
          ruby_package: nil,
          swift_prefix: nil,
          uninterpreted_option: []
        }

        case function_exported?(Google.Protobuf.FileOptions, :decode!, 1) do
          true ->
            bytes =
              file_options
              |> Protox.Google.Protobuf.FileOptions.encode!()
              |> :binary.list_to_bin()

            apply(Google.Protobuf.FileOptions, :decode!, [bytes])

          false ->
            file_options
        end
      end
    )
  end,
  defmodule Transmuter.Onnx.DeviceConfigurationProto do
    @moduledoc false
    defstruct name: nil, num_devices: nil, device: [], __uf__: []

    (
      (
        @spec encode(struct) :: {:ok, iodata} | {:error, any}
        def encode(msg) do
          try do
            {:ok, encode!(msg)}
          rescue
            e in [Protox.EncodingError, Protox.RequiredFieldsError] -> {:error, e}
          end
        end

        @spec encode!(struct) :: iodata | no_return
        def encode!(msg) do
          []
          |> encode_name(msg)
          |> encode_num_devices(msg)
          |> encode_device(msg)
          |> encode_unknown_fields(msg)
        end
      )

      []

      [
        defp encode_name(acc, msg) do
          try do
            case msg.name do
              nil -> acc
              _ -> [acc, "\n", Protox.Encode.encode_string(msg.name)]
            end
          rescue
            ArgumentError ->
              reraise Protox.EncodingError.new(:name, "invalid field value"), __STACKTRACE__
          end
        end,
        defp encode_num_devices(acc, msg) do
          try do
            case msg.num_devices do
              nil -> acc
              _ -> [acc, "\x10", Protox.Encode.encode_int32(msg.num_devices)]
            end
          rescue
            ArgumentError ->
              reraise Protox.EncodingError.new(:num_devices, "invalid field value"),
                      __STACKTRACE__
          end
        end,
        defp encode_device(acc, msg) do
          try do
            case msg.device do
              [] ->
                acc

              values ->
                [
                  acc,
                  Enum.reduce(values, [], fn value, acc ->
                    [acc, "\x1A", Protox.Encode.encode_string(value)]
                  end)
                ]
            end
          rescue
            ArgumentError ->
              reraise Protox.EncodingError.new(:device, "invalid field value"), __STACKTRACE__
          end
        end
      ]

      defp encode_unknown_fields(acc, msg) do
        Enum.reduce(msg.__struct__.unknown_fields(msg), acc, fn {tag, wire_type, bytes}, acc ->
          case wire_type do
            0 ->
              [acc, Protox.Encode.make_key_bytes(tag, :int32), bytes]

            1 ->
              [acc, Protox.Encode.make_key_bytes(tag, :double), bytes]

            2 ->
              len_bytes = bytes |> byte_size() |> Protox.Varint.encode()
              [acc, Protox.Encode.make_key_bytes(tag, :packed), len_bytes, bytes]

            5 ->
              [acc, Protox.Encode.make_key_bytes(tag, :float), bytes]
          end
        end)
      end
    )

    (
      (
        @spec decode(binary) :: {:ok, struct} | {:error, any}
        def decode(bytes) do
          try do
            {:ok, decode!(bytes)}
          rescue
            e in [Protox.DecodingError, Protox.IllegalTagError, Protox.RequiredFieldsError] ->
              {:error, e}
          end
        end

        (
          @spec decode!(binary) :: struct | no_return
          def decode!(bytes) do
            parse_key_value(bytes, struct(Transmuter.Onnx.DeviceConfigurationProto))
          end
        )
      )

      (
        @spec parse_key_value(binary, struct) :: struct
        defp parse_key_value(<<>>, msg) do
          msg
        end

        defp parse_key_value(bytes, msg) do
          {field, rest} =
            case Protox.Decode.parse_key(bytes) do
              {0, _, _} ->
                raise %Protox.IllegalTagError{}

              {1, _, bytes} ->
                {len, bytes} = Protox.Varint.decode(bytes)
                {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)
                {[name: Protox.Decode.validate_string!(delimited)], rest}

              {2, _, bytes} ->
                {value, rest} = Protox.Decode.parse_int32(bytes)
                {[num_devices: value], rest}

              {3, _, bytes} ->
                {len, bytes} = Protox.Varint.decode(bytes)
                {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)
                {[device: msg.device ++ [Protox.Decode.validate_string!(delimited)]], rest}

              {tag, wire_type, rest} ->
                {value, rest} = Protox.Decode.parse_unknown(tag, wire_type, rest)

                {[
                   {msg.__struct__.unknown_fields_name(),
                    [value | msg.__struct__.unknown_fields(msg)]}
                 ], rest}
            end

          msg_updated = struct(msg, field)
          parse_key_value(rest, msg_updated)
        end
      )

      []
    )

    (
      @spec json_decode(iodata(), keyword()) :: {:ok, struct()} | {:error, any()}
      def json_decode(input, opts \\ []) do
        try do
          {:ok, json_decode!(input, opts)}
        rescue
          e in Protox.JsonDecodingError -> {:error, e}
        end
      end

      @spec json_decode!(iodata(), keyword()) :: struct() | no_return()
      def json_decode!(input, opts \\ []) do
        {json_library_wrapper, json_library} = Protox.JsonLibrary.get_library(opts, :decode)

        Protox.JsonDecode.decode!(
          input,
          Transmuter.Onnx.DeviceConfigurationProto,
          &json_library_wrapper.decode!(json_library, &1)
        )
      end

      @spec json_encode(struct(), keyword()) :: {:ok, iodata()} | {:error, any()}
      def json_encode(msg, opts \\ []) do
        try do
          {:ok, json_encode!(msg, opts)}
        rescue
          e in Protox.JsonEncodingError -> {:error, e}
        end
      end

      @spec json_encode!(struct(), keyword()) :: iodata() | no_return()
      def json_encode!(msg, opts \\ []) do
        {json_library_wrapper, json_library} = Protox.JsonLibrary.get_library(opts, :encode)
        Protox.JsonEncode.encode!(msg, &json_library_wrapper.encode!(json_library, &1))
      end
    )

    (
      @deprecated "Use fields_defs()/0 instead"
      @spec defs() :: %{
              required(non_neg_integer) => {atom, Protox.Types.kind(), Protox.Types.type()}
            }
      def defs() do
        %{
          1 => {:name, {:scalar, ""}, :string},
          2 => {:num_devices, {:scalar, 0}, :int32},
          3 => {:device, :unpacked, :string}
        }
      end

      @deprecated "Use fields_defs()/0 instead"
      @spec defs_by_name() :: %{
              required(atom) => {non_neg_integer, Protox.Types.kind(), Protox.Types.type()}
            }
      def defs_by_name() do
        %{
          device: {3, :unpacked, :string},
          name: {1, {:scalar, ""}, :string},
          num_devices: {2, {:scalar, 0}, :int32}
        }
      end
    )

    (
      @spec fields_defs() :: list(Protox.Field.t())
      def fields_defs() do
        [
          %{
            __struct__: Protox.Field,
            json_name: "name",
            kind: {:scalar, ""},
            label: :optional,
            name: :name,
            tag: 1,
            type: :string
          },
          %{
            __struct__: Protox.Field,
            json_name: "numDevices",
            kind: {:scalar, 0},
            label: :optional,
            name: :num_devices,
            tag: 2,
            type: :int32
          },
          %{
            __struct__: Protox.Field,
            json_name: "device",
            kind: :unpacked,
            label: :repeated,
            name: :device,
            tag: 3,
            type: :string
          }
        ]
      end

      [
        @spec(field_def(atom) :: {:ok, Protox.Field.t()} | {:error, :no_such_field}),
        (
          def field_def(:name) do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "name",
               kind: {:scalar, ""},
               label: :optional,
               name: :name,
               tag: 1,
               type: :string
             }}
          end

          def field_def("name") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "name",
               kind: {:scalar, ""},
               label: :optional,
               name: :name,
               tag: 1,
               type: :string
             }}
          end

          []
        ),
        (
          def field_def(:num_devices) do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "numDevices",
               kind: {:scalar, 0},
               label: :optional,
               name: :num_devices,
               tag: 2,
               type: :int32
             }}
          end

          def field_def("numDevices") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "numDevices",
               kind: {:scalar, 0},
               label: :optional,
               name: :num_devices,
               tag: 2,
               type: :int32
             }}
          end

          def field_def("num_devices") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "numDevices",
               kind: {:scalar, 0},
               label: :optional,
               name: :num_devices,
               tag: 2,
               type: :int32
             }}
          end
        ),
        (
          def field_def(:device) do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "device",
               kind: :unpacked,
               label: :repeated,
               name: :device,
               tag: 3,
               type: :string
             }}
          end

          def field_def("device") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "device",
               kind: :unpacked,
               label: :repeated,
               name: :device,
               tag: 3,
               type: :string
             }}
          end

          []
        ),
        def field_def(_) do
          {:error, :no_such_field}
        end
      ]
    )

    (
      @spec unknown_fields(struct) :: [{non_neg_integer, Protox.Types.tag(), binary}]
      def unknown_fields(msg) do
        msg.__uf__
      end

      @spec unknown_fields_name() :: :__uf__
      def unknown_fields_name() do
        :__uf__
      end

      @spec clear_unknown_fields(struct) :: struct
      def clear_unknown_fields(msg) do
        struct!(msg, [{unknown_fields_name(), []}])
      end
    )

    (
      @spec required_fields() :: []
      def required_fields() do
        []
      end
    )

    (
      @spec syntax() :: atom()
      def syntax() do
        :proto2
      end
    )

    [
      @spec(default(atom) :: {:ok, boolean | integer | String.t() | float} | {:error, atom}),
      def default(:name) do
        {:ok, ""}
      end,
      def default(:num_devices) do
        {:ok, 0}
      end,
      def default(:device) do
        {:error, :no_default_value}
      end,
      def default(_) do
        {:error, :no_such_field}
      end
    ]

    (
      @spec file_options() :: struct()
      def file_options() do
        file_options = %{
          __struct__: Protox.Google.Protobuf.FileOptions,
          __uf__: [],
          cc_enable_arenas: nil,
          cc_generic_services: nil,
          csharp_namespace: nil,
          deprecated: nil,
          go_package: nil,
          java_generate_equals_and_hash: nil,
          java_generic_services: nil,
          java_multiple_files: nil,
          java_outer_classname: nil,
          java_package: nil,
          java_string_check_utf8: nil,
          objc_class_prefix: nil,
          optimize_for: :LITE_RUNTIME,
          php_class_prefix: nil,
          php_generic_services: nil,
          php_metadata_namespace: nil,
          php_namespace: nil,
          py_generic_services: nil,
          ruby_package: nil,
          swift_prefix: nil,
          uninterpreted_option: []
        }

        case function_exported?(Google.Protobuf.FileOptions, :decode!, 1) do
          true ->
            bytes =
              file_options
              |> Protox.Google.Protobuf.FileOptions.encode!()
              |> :binary.list_to_bin()

            apply(Google.Protobuf.FileOptions, :decode!, [bytes])

          false ->
            file_options
        end
      end
    )
  end,
  defmodule Transmuter.Onnx.FunctionProto do
    @moduledoc false
    defstruct name: nil,
              input: [],
              output: [],
              attribute: [],
              node: [],
              doc_string: nil,
              opset_import: [],
              domain: nil,
              attribute_proto: [],
              value_info: [],
              overload: nil,
              metadata_props: [],
              __uf__: []

    (
      (
        @spec encode(struct) :: {:ok, iodata} | {:error, any}
        def encode(msg) do
          try do
            {:ok, encode!(msg)}
          rescue
            e in [Protox.EncodingError, Protox.RequiredFieldsError] -> {:error, e}
          end
        end

        @spec encode!(struct) :: iodata | no_return
        def encode!(msg) do
          []
          |> encode_name(msg)
          |> encode_input(msg)
          |> encode_output(msg)
          |> encode_attribute(msg)
          |> encode_node(msg)
          |> encode_doc_string(msg)
          |> encode_opset_import(msg)
          |> encode_domain(msg)
          |> encode_attribute_proto(msg)
          |> encode_value_info(msg)
          |> encode_overload(msg)
          |> encode_metadata_props(msg)
          |> encode_unknown_fields(msg)
        end
      )

      []

      [
        defp encode_name(acc, msg) do
          try do
            case msg.name do
              nil -> acc
              _ -> [acc, "\n", Protox.Encode.encode_string(msg.name)]
            end
          rescue
            ArgumentError ->
              reraise Protox.EncodingError.new(:name, "invalid field value"), __STACKTRACE__
          end
        end,
        defp encode_input(acc, msg) do
          try do
            case msg.input do
              [] ->
                acc

              values ->
                [
                  acc,
                  Enum.reduce(values, [], fn value, acc ->
                    [acc, "\"", Protox.Encode.encode_string(value)]
                  end)
                ]
            end
          rescue
            ArgumentError ->
              reraise Protox.EncodingError.new(:input, "invalid field value"), __STACKTRACE__
          end
        end,
        defp encode_output(acc, msg) do
          try do
            case msg.output do
              [] ->
                acc

              values ->
                [
                  acc,
                  Enum.reduce(values, [], fn value, acc ->
                    [acc, "*", Protox.Encode.encode_string(value)]
                  end)
                ]
            end
          rescue
            ArgumentError ->
              reraise Protox.EncodingError.new(:output, "invalid field value"), __STACKTRACE__
          end
        end,
        defp encode_attribute(acc, msg) do
          try do
            case msg.attribute do
              [] ->
                acc

              values ->
                [
                  acc,
                  Enum.reduce(values, [], fn value, acc ->
                    [acc, "2", Protox.Encode.encode_string(value)]
                  end)
                ]
            end
          rescue
            ArgumentError ->
              reraise Protox.EncodingError.new(:attribute, "invalid field value"), __STACKTRACE__
          end
        end,
        defp encode_node(acc, msg) do
          try do
            case msg.node do
              [] ->
                acc

              values ->
                [
                  acc,
                  Enum.reduce(values, [], fn value, acc ->
                    [acc, ":", Protox.Encode.encode_message(value)]
                  end)
                ]
            end
          rescue
            ArgumentError ->
              reraise Protox.EncodingError.new(:node, "invalid field value"), __STACKTRACE__
          end
        end,
        defp encode_doc_string(acc, msg) do
          try do
            case msg.doc_string do
              nil -> acc
              _ -> [acc, "B", Protox.Encode.encode_string(msg.doc_string)]
            end
          rescue
            ArgumentError ->
              reraise Protox.EncodingError.new(:doc_string, "invalid field value"), __STACKTRACE__
          end
        end,
        defp encode_opset_import(acc, msg) do
          try do
            case msg.opset_import do
              [] ->
                acc

              values ->
                [
                  acc,
                  Enum.reduce(values, [], fn value, acc ->
                    [acc, "J", Protox.Encode.encode_message(value)]
                  end)
                ]
            end
          rescue
            ArgumentError ->
              reraise Protox.EncodingError.new(:opset_import, "invalid field value"),
                      __STACKTRACE__
          end
        end,
        defp encode_domain(acc, msg) do
          try do
            case msg.domain do
              nil -> acc
              _ -> [acc, "R", Protox.Encode.encode_string(msg.domain)]
            end
          rescue
            ArgumentError ->
              reraise Protox.EncodingError.new(:domain, "invalid field value"), __STACKTRACE__
          end
        end,
        defp encode_attribute_proto(acc, msg) do
          try do
            case msg.attribute_proto do
              [] ->
                acc

              values ->
                [
                  acc,
                  Enum.reduce(values, [], fn value, acc ->
                    [acc, "Z", Protox.Encode.encode_message(value)]
                  end)
                ]
            end
          rescue
            ArgumentError ->
              reraise Protox.EncodingError.new(:attribute_proto, "invalid field value"),
                      __STACKTRACE__
          end
        end,
        defp encode_value_info(acc, msg) do
          try do
            case msg.value_info do
              [] ->
                acc

              values ->
                [
                  acc,
                  Enum.reduce(values, [], fn value, acc ->
                    [acc, "b", Protox.Encode.encode_message(value)]
                  end)
                ]
            end
          rescue
            ArgumentError ->
              reraise Protox.EncodingError.new(:value_info, "invalid field value"), __STACKTRACE__
          end
        end,
        defp encode_overload(acc, msg) do
          try do
            case msg.overload do
              nil -> acc
              _ -> [acc, "j", Protox.Encode.encode_string(msg.overload)]
            end
          rescue
            ArgumentError ->
              reraise Protox.EncodingError.new(:overload, "invalid field value"), __STACKTRACE__
          end
        end,
        defp encode_metadata_props(acc, msg) do
          try do
            case msg.metadata_props do
              [] ->
                acc

              values ->
                [
                  acc,
                  Enum.reduce(values, [], fn value, acc ->
                    [acc, "r", Protox.Encode.encode_message(value)]
                  end)
                ]
            end
          rescue
            ArgumentError ->
              reraise Protox.EncodingError.new(:metadata_props, "invalid field value"),
                      __STACKTRACE__
          end
        end
      ]

      defp encode_unknown_fields(acc, msg) do
        Enum.reduce(msg.__struct__.unknown_fields(msg), acc, fn {tag, wire_type, bytes}, acc ->
          case wire_type do
            0 ->
              [acc, Protox.Encode.make_key_bytes(tag, :int32), bytes]

            1 ->
              [acc, Protox.Encode.make_key_bytes(tag, :double), bytes]

            2 ->
              len_bytes = bytes |> byte_size() |> Protox.Varint.encode()
              [acc, Protox.Encode.make_key_bytes(tag, :packed), len_bytes, bytes]

            5 ->
              [acc, Protox.Encode.make_key_bytes(tag, :float), bytes]
          end
        end)
      end
    )

    (
      (
        @spec decode(binary) :: {:ok, struct} | {:error, any}
        def decode(bytes) do
          try do
            {:ok, decode!(bytes)}
          rescue
            e in [Protox.DecodingError, Protox.IllegalTagError, Protox.RequiredFieldsError] ->
              {:error, e}
          end
        end

        (
          @spec decode!(binary) :: struct | no_return
          def decode!(bytes) do
            parse_key_value(bytes, struct(Transmuter.Onnx.FunctionProto))
          end
        )
      )

      (
        @spec parse_key_value(binary, struct) :: struct
        defp parse_key_value(<<>>, msg) do
          msg
        end

        defp parse_key_value(bytes, msg) do
          {field, rest} =
            case Protox.Decode.parse_key(bytes) do
              {0, _, _} ->
                raise %Protox.IllegalTagError{}

              {1, _, bytes} ->
                {len, bytes} = Protox.Varint.decode(bytes)
                {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)
                {[name: Protox.Decode.validate_string!(delimited)], rest}

              {4, _, bytes} ->
                {len, bytes} = Protox.Varint.decode(bytes)
                {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)
                {[input: msg.input ++ [Protox.Decode.validate_string!(delimited)]], rest}

              {5, _, bytes} ->
                {len, bytes} = Protox.Varint.decode(bytes)
                {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)
                {[output: msg.output ++ [Protox.Decode.validate_string!(delimited)]], rest}

              {6, _, bytes} ->
                {len, bytes} = Protox.Varint.decode(bytes)
                {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)
                {[attribute: msg.attribute ++ [Protox.Decode.validate_string!(delimited)]], rest}

              {7, _, bytes} ->
                {len, bytes} = Protox.Varint.decode(bytes)
                {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)
                {[node: msg.node ++ [Transmuter.Onnx.NodeProto.decode!(delimited)]], rest}

              {8, _, bytes} ->
                {len, bytes} = Protox.Varint.decode(bytes)
                {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)
                {[doc_string: Protox.Decode.validate_string!(delimited)], rest}

              {9, _, bytes} ->
                {len, bytes} = Protox.Varint.decode(bytes)
                {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)

                {[
                   opset_import:
                     msg.opset_import ++ [Transmuter.Onnx.OperatorSetIdProto.decode!(delimited)]
                 ], rest}

              {10, _, bytes} ->
                {len, bytes} = Protox.Varint.decode(bytes)
                {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)
                {[domain: Protox.Decode.validate_string!(delimited)], rest}

              {11, _, bytes} ->
                {len, bytes} = Protox.Varint.decode(bytes)
                {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)

                {[
                   attribute_proto:
                     msg.attribute_proto ++ [Transmuter.Onnx.AttributeProto.decode!(delimited)]
                 ], rest}

              {12, _, bytes} ->
                {len, bytes} = Protox.Varint.decode(bytes)
                {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)

                {[
                   value_info:
                     msg.value_info ++ [Transmuter.Onnx.ValueInfoProto.decode!(delimited)]
                 ], rest}

              {13, _, bytes} ->
                {len, bytes} = Protox.Varint.decode(bytes)
                {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)
                {[overload: Protox.Decode.validate_string!(delimited)], rest}

              {14, _, bytes} ->
                {len, bytes} = Protox.Varint.decode(bytes)
                {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)

                {[
                   metadata_props:
                     msg.metadata_props ++
                       [Transmuter.Onnx.StringStringEntryProto.decode!(delimited)]
                 ], rest}

              {tag, wire_type, rest} ->
                {value, rest} = Protox.Decode.parse_unknown(tag, wire_type, rest)

                {[
                   {msg.__struct__.unknown_fields_name(),
                    [value | msg.__struct__.unknown_fields(msg)]}
                 ], rest}
            end

          msg_updated = struct(msg, field)
          parse_key_value(rest, msg_updated)
        end
      )

      []
    )

    (
      @spec json_decode(iodata(), keyword()) :: {:ok, struct()} | {:error, any()}
      def json_decode(input, opts \\ []) do
        try do
          {:ok, json_decode!(input, opts)}
        rescue
          e in Protox.JsonDecodingError -> {:error, e}
        end
      end

      @spec json_decode!(iodata(), keyword()) :: struct() | no_return()
      def json_decode!(input, opts \\ []) do
        {json_library_wrapper, json_library} = Protox.JsonLibrary.get_library(opts, :decode)

        Protox.JsonDecode.decode!(
          input,
          Transmuter.Onnx.FunctionProto,
          &json_library_wrapper.decode!(json_library, &1)
        )
      end

      @spec json_encode(struct(), keyword()) :: {:ok, iodata()} | {:error, any()}
      def json_encode(msg, opts \\ []) do
        try do
          {:ok, json_encode!(msg, opts)}
        rescue
          e in Protox.JsonEncodingError -> {:error, e}
        end
      end

      @spec json_encode!(struct(), keyword()) :: iodata() | no_return()
      def json_encode!(msg, opts \\ []) do
        {json_library_wrapper, json_library} = Protox.JsonLibrary.get_library(opts, :encode)
        Protox.JsonEncode.encode!(msg, &json_library_wrapper.encode!(json_library, &1))
      end
    )

    (
      @deprecated "Use fields_defs()/0 instead"
      @spec defs() :: %{
              required(non_neg_integer) => {atom, Protox.Types.kind(), Protox.Types.type()}
            }
      def defs() do
        %{
          1 => {:name, {:scalar, ""}, :string},
          4 => {:input, :unpacked, :string},
          5 => {:output, :unpacked, :string},
          6 => {:attribute, :unpacked, :string},
          7 => {:node, :unpacked, {:message, Transmuter.Onnx.NodeProto}},
          8 => {:doc_string, {:scalar, ""}, :string},
          9 => {:opset_import, :unpacked, {:message, Transmuter.Onnx.OperatorSetIdProto}},
          10 => {:domain, {:scalar, ""}, :string},
          11 => {:attribute_proto, :unpacked, {:message, Transmuter.Onnx.AttributeProto}},
          12 => {:value_info, :unpacked, {:message, Transmuter.Onnx.ValueInfoProto}},
          13 => {:overload, {:scalar, ""}, :string},
          14 => {:metadata_props, :unpacked, {:message, Transmuter.Onnx.StringStringEntryProto}}
        }
      end

      @deprecated "Use fields_defs()/0 instead"
      @spec defs_by_name() :: %{
              required(atom) => {non_neg_integer, Protox.Types.kind(), Protox.Types.type()}
            }
      def defs_by_name() do
        %{
          attribute: {6, :unpacked, :string},
          attribute_proto: {11, :unpacked, {:message, Transmuter.Onnx.AttributeProto}},
          doc_string: {8, {:scalar, ""}, :string},
          domain: {10, {:scalar, ""}, :string},
          input: {4, :unpacked, :string},
          metadata_props: {14, :unpacked, {:message, Transmuter.Onnx.StringStringEntryProto}},
          name: {1, {:scalar, ""}, :string},
          node: {7, :unpacked, {:message, Transmuter.Onnx.NodeProto}},
          opset_import: {9, :unpacked, {:message, Transmuter.Onnx.OperatorSetIdProto}},
          output: {5, :unpacked, :string},
          overload: {13, {:scalar, ""}, :string},
          value_info: {12, :unpacked, {:message, Transmuter.Onnx.ValueInfoProto}}
        }
      end
    )

    (
      @spec fields_defs() :: list(Protox.Field.t())
      def fields_defs() do
        [
          %{
            __struct__: Protox.Field,
            json_name: "name",
            kind: {:scalar, ""},
            label: :optional,
            name: :name,
            tag: 1,
            type: :string
          },
          %{
            __struct__: Protox.Field,
            json_name: "input",
            kind: :unpacked,
            label: :repeated,
            name: :input,
            tag: 4,
            type: :string
          },
          %{
            __struct__: Protox.Field,
            json_name: "output",
            kind: :unpacked,
            label: :repeated,
            name: :output,
            tag: 5,
            type: :string
          },
          %{
            __struct__: Protox.Field,
            json_name: "attribute",
            kind: :unpacked,
            label: :repeated,
            name: :attribute,
            tag: 6,
            type: :string
          },
          %{
            __struct__: Protox.Field,
            json_name: "node",
            kind: :unpacked,
            label: :repeated,
            name: :node,
            tag: 7,
            type: {:message, Transmuter.Onnx.NodeProto}
          },
          %{
            __struct__: Protox.Field,
            json_name: "docString",
            kind: {:scalar, ""},
            label: :optional,
            name: :doc_string,
            tag: 8,
            type: :string
          },
          %{
            __struct__: Protox.Field,
            json_name: "opsetImport",
            kind: :unpacked,
            label: :repeated,
            name: :opset_import,
            tag: 9,
            type: {:message, Transmuter.Onnx.OperatorSetIdProto}
          },
          %{
            __struct__: Protox.Field,
            json_name: "domain",
            kind: {:scalar, ""},
            label: :optional,
            name: :domain,
            tag: 10,
            type: :string
          },
          %{
            __struct__: Protox.Field,
            json_name: "attributeProto",
            kind: :unpacked,
            label: :repeated,
            name: :attribute_proto,
            tag: 11,
            type: {:message, Transmuter.Onnx.AttributeProto}
          },
          %{
            __struct__: Protox.Field,
            json_name: "valueInfo",
            kind: :unpacked,
            label: :repeated,
            name: :value_info,
            tag: 12,
            type: {:message, Transmuter.Onnx.ValueInfoProto}
          },
          %{
            __struct__: Protox.Field,
            json_name: "overload",
            kind: {:scalar, ""},
            label: :optional,
            name: :overload,
            tag: 13,
            type: :string
          },
          %{
            __struct__: Protox.Field,
            json_name: "metadataProps",
            kind: :unpacked,
            label: :repeated,
            name: :metadata_props,
            tag: 14,
            type: {:message, Transmuter.Onnx.StringStringEntryProto}
          }
        ]
      end

      [
        @spec(field_def(atom) :: {:ok, Protox.Field.t()} | {:error, :no_such_field}),
        (
          def field_def(:name) do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "name",
               kind: {:scalar, ""},
               label: :optional,
               name: :name,
               tag: 1,
               type: :string
             }}
          end

          def field_def("name") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "name",
               kind: {:scalar, ""},
               label: :optional,
               name: :name,
               tag: 1,
               type: :string
             }}
          end

          []
        ),
        (
          def field_def(:input) do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "input",
               kind: :unpacked,
               label: :repeated,
               name: :input,
               tag: 4,
               type: :string
             }}
          end

          def field_def("input") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "input",
               kind: :unpacked,
               label: :repeated,
               name: :input,
               tag: 4,
               type: :string
             }}
          end

          []
        ),
        (
          def field_def(:output) do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "output",
               kind: :unpacked,
               label: :repeated,
               name: :output,
               tag: 5,
               type: :string
             }}
          end

          def field_def("output") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "output",
               kind: :unpacked,
               label: :repeated,
               name: :output,
               tag: 5,
               type: :string
             }}
          end

          []
        ),
        (
          def field_def(:attribute) do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "attribute",
               kind: :unpacked,
               label: :repeated,
               name: :attribute,
               tag: 6,
               type: :string
             }}
          end

          def field_def("attribute") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "attribute",
               kind: :unpacked,
               label: :repeated,
               name: :attribute,
               tag: 6,
               type: :string
             }}
          end

          []
        ),
        (
          def field_def(:node) do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "node",
               kind: :unpacked,
               label: :repeated,
               name: :node,
               tag: 7,
               type: {:message, Transmuter.Onnx.NodeProto}
             }}
          end

          def field_def("node") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "node",
               kind: :unpacked,
               label: :repeated,
               name: :node,
               tag: 7,
               type: {:message, Transmuter.Onnx.NodeProto}
             }}
          end

          []
        ),
        (
          def field_def(:doc_string) do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "docString",
               kind: {:scalar, ""},
               label: :optional,
               name: :doc_string,
               tag: 8,
               type: :string
             }}
          end

          def field_def("docString") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "docString",
               kind: {:scalar, ""},
               label: :optional,
               name: :doc_string,
               tag: 8,
               type: :string
             }}
          end

          def field_def("doc_string") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "docString",
               kind: {:scalar, ""},
               label: :optional,
               name: :doc_string,
               tag: 8,
               type: :string
             }}
          end
        ),
        (
          def field_def(:opset_import) do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "opsetImport",
               kind: :unpacked,
               label: :repeated,
               name: :opset_import,
               tag: 9,
               type: {:message, Transmuter.Onnx.OperatorSetIdProto}
             }}
          end

          def field_def("opsetImport") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "opsetImport",
               kind: :unpacked,
               label: :repeated,
               name: :opset_import,
               tag: 9,
               type: {:message, Transmuter.Onnx.OperatorSetIdProto}
             }}
          end

          def field_def("opset_import") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "opsetImport",
               kind: :unpacked,
               label: :repeated,
               name: :opset_import,
               tag: 9,
               type: {:message, Transmuter.Onnx.OperatorSetIdProto}
             }}
          end
        ),
        (
          def field_def(:domain) do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "domain",
               kind: {:scalar, ""},
               label: :optional,
               name: :domain,
               tag: 10,
               type: :string
             }}
          end

          def field_def("domain") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "domain",
               kind: {:scalar, ""},
               label: :optional,
               name: :domain,
               tag: 10,
               type: :string
             }}
          end

          []
        ),
        (
          def field_def(:attribute_proto) do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "attributeProto",
               kind: :unpacked,
               label: :repeated,
               name: :attribute_proto,
               tag: 11,
               type: {:message, Transmuter.Onnx.AttributeProto}
             }}
          end

          def field_def("attributeProto") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "attributeProto",
               kind: :unpacked,
               label: :repeated,
               name: :attribute_proto,
               tag: 11,
               type: {:message, Transmuter.Onnx.AttributeProto}
             }}
          end

          def field_def("attribute_proto") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "attributeProto",
               kind: :unpacked,
               label: :repeated,
               name: :attribute_proto,
               tag: 11,
               type: {:message, Transmuter.Onnx.AttributeProto}
             }}
          end
        ),
        (
          def field_def(:value_info) do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "valueInfo",
               kind: :unpacked,
               label: :repeated,
               name: :value_info,
               tag: 12,
               type: {:message, Transmuter.Onnx.ValueInfoProto}
             }}
          end

          def field_def("valueInfo") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "valueInfo",
               kind: :unpacked,
               label: :repeated,
               name: :value_info,
               tag: 12,
               type: {:message, Transmuter.Onnx.ValueInfoProto}
             }}
          end

          def field_def("value_info") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "valueInfo",
               kind: :unpacked,
               label: :repeated,
               name: :value_info,
               tag: 12,
               type: {:message, Transmuter.Onnx.ValueInfoProto}
             }}
          end
        ),
        (
          def field_def(:overload) do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "overload",
               kind: {:scalar, ""},
               label: :optional,
               name: :overload,
               tag: 13,
               type: :string
             }}
          end

          def field_def("overload") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "overload",
               kind: {:scalar, ""},
               label: :optional,
               name: :overload,
               tag: 13,
               type: :string
             }}
          end

          []
        ),
        (
          def field_def(:metadata_props) do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "metadataProps",
               kind: :unpacked,
               label: :repeated,
               name: :metadata_props,
               tag: 14,
               type: {:message, Transmuter.Onnx.StringStringEntryProto}
             }}
          end

          def field_def("metadataProps") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "metadataProps",
               kind: :unpacked,
               label: :repeated,
               name: :metadata_props,
               tag: 14,
               type: {:message, Transmuter.Onnx.StringStringEntryProto}
             }}
          end

          def field_def("metadata_props") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "metadataProps",
               kind: :unpacked,
               label: :repeated,
               name: :metadata_props,
               tag: 14,
               type: {:message, Transmuter.Onnx.StringStringEntryProto}
             }}
          end
        ),
        def field_def(_) do
          {:error, :no_such_field}
        end
      ]
    )

    (
      @spec unknown_fields(struct) :: [{non_neg_integer, Protox.Types.tag(), binary}]
      def unknown_fields(msg) do
        msg.__uf__
      end

      @spec unknown_fields_name() :: :__uf__
      def unknown_fields_name() do
        :__uf__
      end

      @spec clear_unknown_fields(struct) :: struct
      def clear_unknown_fields(msg) do
        struct!(msg, [{unknown_fields_name(), []}])
      end
    )

    (
      @spec required_fields() :: []
      def required_fields() do
        []
      end
    )

    (
      @spec syntax() :: atom()
      def syntax() do
        :proto2
      end
    )

    [
      @spec(default(atom) :: {:ok, boolean | integer | String.t() | float} | {:error, atom}),
      def default(:name) do
        {:ok, ""}
      end,
      def default(:input) do
        {:error, :no_default_value}
      end,
      def default(:output) do
        {:error, :no_default_value}
      end,
      def default(:attribute) do
        {:error, :no_default_value}
      end,
      def default(:node) do
        {:error, :no_default_value}
      end,
      def default(:doc_string) do
        {:ok, ""}
      end,
      def default(:opset_import) do
        {:error, :no_default_value}
      end,
      def default(:domain) do
        {:ok, ""}
      end,
      def default(:attribute_proto) do
        {:error, :no_default_value}
      end,
      def default(:value_info) do
        {:error, :no_default_value}
      end,
      def default(:overload) do
        {:ok, ""}
      end,
      def default(:metadata_props) do
        {:error, :no_default_value}
      end,
      def default(_) do
        {:error, :no_such_field}
      end
    ]

    (
      @spec file_options() :: struct()
      def file_options() do
        file_options = %{
          __struct__: Protox.Google.Protobuf.FileOptions,
          __uf__: [],
          cc_enable_arenas: nil,
          cc_generic_services: nil,
          csharp_namespace: nil,
          deprecated: nil,
          go_package: nil,
          java_generate_equals_and_hash: nil,
          java_generic_services: nil,
          java_multiple_files: nil,
          java_outer_classname: nil,
          java_package: nil,
          java_string_check_utf8: nil,
          objc_class_prefix: nil,
          optimize_for: :LITE_RUNTIME,
          php_class_prefix: nil,
          php_generic_services: nil,
          php_metadata_namespace: nil,
          php_namespace: nil,
          py_generic_services: nil,
          ruby_package: nil,
          swift_prefix: nil,
          uninterpreted_option: []
        }

        case function_exported?(Google.Protobuf.FileOptions, :decode!, 1) do
          true ->
            bytes =
              file_options
              |> Protox.Google.Protobuf.FileOptions.encode!()
              |> :binary.list_to_bin()

            apply(Google.Protobuf.FileOptions, :decode!, [bytes])

          false ->
            file_options
        end
      end
    )
  end,
  defmodule Transmuter.Onnx.GraphProto do
    @moduledoc false
    defstruct node: [],
              name: nil,
              initializer: [],
              doc_string: nil,
              input: [],
              output: [],
              value_info: [],
              quantization_annotation: [],
              sparse_initializer: [],
              metadata_props: [],
              __uf__: []

    (
      (
        @spec encode(struct) :: {:ok, iodata} | {:error, any}
        def encode(msg) do
          try do
            {:ok, encode!(msg)}
          rescue
            e in [Protox.EncodingError, Protox.RequiredFieldsError] -> {:error, e}
          end
        end

        @spec encode!(struct) :: iodata | no_return
        def encode!(msg) do
          []
          |> encode_node(msg)
          |> encode_name(msg)
          |> encode_initializer(msg)
          |> encode_doc_string(msg)
          |> encode_input(msg)
          |> encode_output(msg)
          |> encode_value_info(msg)
          |> encode_quantization_annotation(msg)
          |> encode_sparse_initializer(msg)
          |> encode_metadata_props(msg)
          |> encode_unknown_fields(msg)
        end
      )

      []

      [
        defp encode_node(acc, msg) do
          try do
            case msg.node do
              [] ->
                acc

              values ->
                [
                  acc,
                  Enum.reduce(values, [], fn value, acc ->
                    [acc, "\n", Protox.Encode.encode_message(value)]
                  end)
                ]
            end
          rescue
            ArgumentError ->
              reraise Protox.EncodingError.new(:node, "invalid field value"), __STACKTRACE__
          end
        end,
        defp encode_name(acc, msg) do
          try do
            case msg.name do
              nil -> acc
              _ -> [acc, "\x12", Protox.Encode.encode_string(msg.name)]
            end
          rescue
            ArgumentError ->
              reraise Protox.EncodingError.new(:name, "invalid field value"), __STACKTRACE__
          end
        end,
        defp encode_initializer(acc, msg) do
          try do
            case msg.initializer do
              [] ->
                acc

              values ->
                [
                  acc,
                  Enum.reduce(values, [], fn value, acc ->
                    [acc, "*", Protox.Encode.encode_message(value)]
                  end)
                ]
            end
          rescue
            ArgumentError ->
              reraise Protox.EncodingError.new(:initializer, "invalid field value"),
                      __STACKTRACE__
          end
        end,
        defp encode_doc_string(acc, msg) do
          try do
            case msg.doc_string do
              nil -> acc
              _ -> [acc, "R", Protox.Encode.encode_string(msg.doc_string)]
            end
          rescue
            ArgumentError ->
              reraise Protox.EncodingError.new(:doc_string, "invalid field value"), __STACKTRACE__
          end
        end,
        defp encode_input(acc, msg) do
          try do
            case msg.input do
              [] ->
                acc

              values ->
                [
                  acc,
                  Enum.reduce(values, [], fn value, acc ->
                    [acc, "Z", Protox.Encode.encode_message(value)]
                  end)
                ]
            end
          rescue
            ArgumentError ->
              reraise Protox.EncodingError.new(:input, "invalid field value"), __STACKTRACE__
          end
        end,
        defp encode_output(acc, msg) do
          try do
            case msg.output do
              [] ->
                acc

              values ->
                [
                  acc,
                  Enum.reduce(values, [], fn value, acc ->
                    [acc, "b", Protox.Encode.encode_message(value)]
                  end)
                ]
            end
          rescue
            ArgumentError ->
              reraise Protox.EncodingError.new(:output, "invalid field value"), __STACKTRACE__
          end
        end,
        defp encode_value_info(acc, msg) do
          try do
            case msg.value_info do
              [] ->
                acc

              values ->
                [
                  acc,
                  Enum.reduce(values, [], fn value, acc ->
                    [acc, "j", Protox.Encode.encode_message(value)]
                  end)
                ]
            end
          rescue
            ArgumentError ->
              reraise Protox.EncodingError.new(:value_info, "invalid field value"), __STACKTRACE__
          end
        end,
        defp encode_quantization_annotation(acc, msg) do
          try do
            case msg.quantization_annotation do
              [] ->
                acc

              values ->
                [
                  acc,
                  Enum.reduce(values, [], fn value, acc ->
                    [acc, "r", Protox.Encode.encode_message(value)]
                  end)
                ]
            end
          rescue
            ArgumentError ->
              reraise Protox.EncodingError.new(:quantization_annotation, "invalid field value"),
                      __STACKTRACE__
          end
        end,
        defp encode_sparse_initializer(acc, msg) do
          try do
            case msg.sparse_initializer do
              [] ->
                acc

              values ->
                [
                  acc,
                  Enum.reduce(values, [], fn value, acc ->
                    [acc, "z", Protox.Encode.encode_message(value)]
                  end)
                ]
            end
          rescue
            ArgumentError ->
              reraise Protox.EncodingError.new(:sparse_initializer, "invalid field value"),
                      __STACKTRACE__
          end
        end,
        defp encode_metadata_props(acc, msg) do
          try do
            case msg.metadata_props do
              [] ->
                acc

              values ->
                [
                  acc,
                  Enum.reduce(values, [], fn value, acc ->
                    [acc, "\x82\x01", Protox.Encode.encode_message(value)]
                  end)
                ]
            end
          rescue
            ArgumentError ->
              reraise Protox.EncodingError.new(:metadata_props, "invalid field value"),
                      __STACKTRACE__
          end
        end
      ]

      defp encode_unknown_fields(acc, msg) do
        Enum.reduce(msg.__struct__.unknown_fields(msg), acc, fn {tag, wire_type, bytes}, acc ->
          case wire_type do
            0 ->
              [acc, Protox.Encode.make_key_bytes(tag, :int32), bytes]

            1 ->
              [acc, Protox.Encode.make_key_bytes(tag, :double), bytes]

            2 ->
              len_bytes = bytes |> byte_size() |> Protox.Varint.encode()
              [acc, Protox.Encode.make_key_bytes(tag, :packed), len_bytes, bytes]

            5 ->
              [acc, Protox.Encode.make_key_bytes(tag, :float), bytes]
          end
        end)
      end
    )

    (
      (
        @spec decode(binary) :: {:ok, struct} | {:error, any}
        def decode(bytes) do
          try do
            {:ok, decode!(bytes)}
          rescue
            e in [Protox.DecodingError, Protox.IllegalTagError, Protox.RequiredFieldsError] ->
              {:error, e}
          end
        end

        (
          @spec decode!(binary) :: struct | no_return
          def decode!(bytes) do
            parse_key_value(bytes, struct(Transmuter.Onnx.GraphProto))
          end
        )
      )

      (
        @spec parse_key_value(binary, struct) :: struct
        defp parse_key_value(<<>>, msg) do
          msg
        end

        defp parse_key_value(bytes, msg) do
          {field, rest} =
            case Protox.Decode.parse_key(bytes) do
              {0, _, _} ->
                raise %Protox.IllegalTagError{}

              {1, _, bytes} ->
                {len, bytes} = Protox.Varint.decode(bytes)
                {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)
                {[node: msg.node ++ [Transmuter.Onnx.NodeProto.decode!(delimited)]], rest}

              {2, _, bytes} ->
                {len, bytes} = Protox.Varint.decode(bytes)
                {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)
                {[name: Protox.Decode.validate_string!(delimited)], rest}

              {5, _, bytes} ->
                {len, bytes} = Protox.Varint.decode(bytes)
                {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)

                {[
                   initializer:
                     msg.initializer ++ [Transmuter.Onnx.TensorProto.decode!(delimited)]
                 ], rest}

              {10, _, bytes} ->
                {len, bytes} = Protox.Varint.decode(bytes)
                {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)
                {[doc_string: Protox.Decode.validate_string!(delimited)], rest}

              {11, _, bytes} ->
                {len, bytes} = Protox.Varint.decode(bytes)
                {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)
                {[input: msg.input ++ [Transmuter.Onnx.ValueInfoProto.decode!(delimited)]], rest}

              {12, _, bytes} ->
                {len, bytes} = Protox.Varint.decode(bytes)
                {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)

                {[output: msg.output ++ [Transmuter.Onnx.ValueInfoProto.decode!(delimited)]],
                 rest}

              {13, _, bytes} ->
                {len, bytes} = Protox.Varint.decode(bytes)
                {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)

                {[
                   value_info:
                     msg.value_info ++ [Transmuter.Onnx.ValueInfoProto.decode!(delimited)]
                 ], rest}

              {14, _, bytes} ->
                {len, bytes} = Protox.Varint.decode(bytes)
                {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)

                {[
                   quantization_annotation:
                     msg.quantization_annotation ++
                       [Transmuter.Onnx.TensorAnnotation.decode!(delimited)]
                 ], rest}

              {15, _, bytes} ->
                {len, bytes} = Protox.Varint.decode(bytes)
                {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)

                {[
                   sparse_initializer:
                     msg.sparse_initializer ++
                       [Transmuter.Onnx.SparseTensorProto.decode!(delimited)]
                 ], rest}

              {16, _, bytes} ->
                {len, bytes} = Protox.Varint.decode(bytes)
                {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)

                {[
                   metadata_props:
                     msg.metadata_props ++
                       [Transmuter.Onnx.StringStringEntryProto.decode!(delimited)]
                 ], rest}

              {tag, wire_type, rest} ->
                {value, rest} = Protox.Decode.parse_unknown(tag, wire_type, rest)

                {[
                   {msg.__struct__.unknown_fields_name(),
                    [value | msg.__struct__.unknown_fields(msg)]}
                 ], rest}
            end

          msg_updated = struct(msg, field)
          parse_key_value(rest, msg_updated)
        end
      )

      []
    )

    (
      @spec json_decode(iodata(), keyword()) :: {:ok, struct()} | {:error, any()}
      def json_decode(input, opts \\ []) do
        try do
          {:ok, json_decode!(input, opts)}
        rescue
          e in Protox.JsonDecodingError -> {:error, e}
        end
      end

      @spec json_decode!(iodata(), keyword()) :: struct() | no_return()
      def json_decode!(input, opts \\ []) do
        {json_library_wrapper, json_library} = Protox.JsonLibrary.get_library(opts, :decode)

        Protox.JsonDecode.decode!(
          input,
          Transmuter.Onnx.GraphProto,
          &json_library_wrapper.decode!(json_library, &1)
        )
      end

      @spec json_encode(struct(), keyword()) :: {:ok, iodata()} | {:error, any()}
      def json_encode(msg, opts \\ []) do
        try do
          {:ok, json_encode!(msg, opts)}
        rescue
          e in Protox.JsonEncodingError -> {:error, e}
        end
      end

      @spec json_encode!(struct(), keyword()) :: iodata() | no_return()
      def json_encode!(msg, opts \\ []) do
        {json_library_wrapper, json_library} = Protox.JsonLibrary.get_library(opts, :encode)
        Protox.JsonEncode.encode!(msg, &json_library_wrapper.encode!(json_library, &1))
      end
    )

    (
      @deprecated "Use fields_defs()/0 instead"
      @spec defs() :: %{
              required(non_neg_integer) => {atom, Protox.Types.kind(), Protox.Types.type()}
            }
      def defs() do
        %{
          1 => {:node, :unpacked, {:message, Transmuter.Onnx.NodeProto}},
          2 => {:name, {:scalar, ""}, :string},
          5 => {:initializer, :unpacked, {:message, Transmuter.Onnx.TensorProto}},
          10 => {:doc_string, {:scalar, ""}, :string},
          11 => {:input, :unpacked, {:message, Transmuter.Onnx.ValueInfoProto}},
          12 => {:output, :unpacked, {:message, Transmuter.Onnx.ValueInfoProto}},
          13 => {:value_info, :unpacked, {:message, Transmuter.Onnx.ValueInfoProto}},
          14 =>
            {:quantization_annotation, :unpacked, {:message, Transmuter.Onnx.TensorAnnotation}},
          15 => {:sparse_initializer, :unpacked, {:message, Transmuter.Onnx.SparseTensorProto}},
          16 => {:metadata_props, :unpacked, {:message, Transmuter.Onnx.StringStringEntryProto}}
        }
      end

      @deprecated "Use fields_defs()/0 instead"
      @spec defs_by_name() :: %{
              required(atom) => {non_neg_integer, Protox.Types.kind(), Protox.Types.type()}
            }
      def defs_by_name() do
        %{
          doc_string: {10, {:scalar, ""}, :string},
          initializer: {5, :unpacked, {:message, Transmuter.Onnx.TensorProto}},
          input: {11, :unpacked, {:message, Transmuter.Onnx.ValueInfoProto}},
          metadata_props: {16, :unpacked, {:message, Transmuter.Onnx.StringStringEntryProto}},
          name: {2, {:scalar, ""}, :string},
          node: {1, :unpacked, {:message, Transmuter.Onnx.NodeProto}},
          output: {12, :unpacked, {:message, Transmuter.Onnx.ValueInfoProto}},
          quantization_annotation: {14, :unpacked, {:message, Transmuter.Onnx.TensorAnnotation}},
          sparse_initializer: {15, :unpacked, {:message, Transmuter.Onnx.SparseTensorProto}},
          value_info: {13, :unpacked, {:message, Transmuter.Onnx.ValueInfoProto}}
        }
      end
    )

    (
      @spec fields_defs() :: list(Protox.Field.t())
      def fields_defs() do
        [
          %{
            __struct__: Protox.Field,
            json_name: "node",
            kind: :unpacked,
            label: :repeated,
            name: :node,
            tag: 1,
            type: {:message, Transmuter.Onnx.NodeProto}
          },
          %{
            __struct__: Protox.Field,
            json_name: "name",
            kind: {:scalar, ""},
            label: :optional,
            name: :name,
            tag: 2,
            type: :string
          },
          %{
            __struct__: Protox.Field,
            json_name: "initializer",
            kind: :unpacked,
            label: :repeated,
            name: :initializer,
            tag: 5,
            type: {:message, Transmuter.Onnx.TensorProto}
          },
          %{
            __struct__: Protox.Field,
            json_name: "docString",
            kind: {:scalar, ""},
            label: :optional,
            name: :doc_string,
            tag: 10,
            type: :string
          },
          %{
            __struct__: Protox.Field,
            json_name: "input",
            kind: :unpacked,
            label: :repeated,
            name: :input,
            tag: 11,
            type: {:message, Transmuter.Onnx.ValueInfoProto}
          },
          %{
            __struct__: Protox.Field,
            json_name: "output",
            kind: :unpacked,
            label: :repeated,
            name: :output,
            tag: 12,
            type: {:message, Transmuter.Onnx.ValueInfoProto}
          },
          %{
            __struct__: Protox.Field,
            json_name: "valueInfo",
            kind: :unpacked,
            label: :repeated,
            name: :value_info,
            tag: 13,
            type: {:message, Transmuter.Onnx.ValueInfoProto}
          },
          %{
            __struct__: Protox.Field,
            json_name: "quantizationAnnotation",
            kind: :unpacked,
            label: :repeated,
            name: :quantization_annotation,
            tag: 14,
            type: {:message, Transmuter.Onnx.TensorAnnotation}
          },
          %{
            __struct__: Protox.Field,
            json_name: "sparseInitializer",
            kind: :unpacked,
            label: :repeated,
            name: :sparse_initializer,
            tag: 15,
            type: {:message, Transmuter.Onnx.SparseTensorProto}
          },
          %{
            __struct__: Protox.Field,
            json_name: "metadataProps",
            kind: :unpacked,
            label: :repeated,
            name: :metadata_props,
            tag: 16,
            type: {:message, Transmuter.Onnx.StringStringEntryProto}
          }
        ]
      end

      [
        @spec(field_def(atom) :: {:ok, Protox.Field.t()} | {:error, :no_such_field}),
        (
          def field_def(:node) do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "node",
               kind: :unpacked,
               label: :repeated,
               name: :node,
               tag: 1,
               type: {:message, Transmuter.Onnx.NodeProto}
             }}
          end

          def field_def("node") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "node",
               kind: :unpacked,
               label: :repeated,
               name: :node,
               tag: 1,
               type: {:message, Transmuter.Onnx.NodeProto}
             }}
          end

          []
        ),
        (
          def field_def(:name) do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "name",
               kind: {:scalar, ""},
               label: :optional,
               name: :name,
               tag: 2,
               type: :string
             }}
          end

          def field_def("name") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "name",
               kind: {:scalar, ""},
               label: :optional,
               name: :name,
               tag: 2,
               type: :string
             }}
          end

          []
        ),
        (
          def field_def(:initializer) do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "initializer",
               kind: :unpacked,
               label: :repeated,
               name: :initializer,
               tag: 5,
               type: {:message, Transmuter.Onnx.TensorProto}
             }}
          end

          def field_def("initializer") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "initializer",
               kind: :unpacked,
               label: :repeated,
               name: :initializer,
               tag: 5,
               type: {:message, Transmuter.Onnx.TensorProto}
             }}
          end

          []
        ),
        (
          def field_def(:doc_string) do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "docString",
               kind: {:scalar, ""},
               label: :optional,
               name: :doc_string,
               tag: 10,
               type: :string
             }}
          end

          def field_def("docString") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "docString",
               kind: {:scalar, ""},
               label: :optional,
               name: :doc_string,
               tag: 10,
               type: :string
             }}
          end

          def field_def("doc_string") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "docString",
               kind: {:scalar, ""},
               label: :optional,
               name: :doc_string,
               tag: 10,
               type: :string
             }}
          end
        ),
        (
          def field_def(:input) do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "input",
               kind: :unpacked,
               label: :repeated,
               name: :input,
               tag: 11,
               type: {:message, Transmuter.Onnx.ValueInfoProto}
             }}
          end

          def field_def("input") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "input",
               kind: :unpacked,
               label: :repeated,
               name: :input,
               tag: 11,
               type: {:message, Transmuter.Onnx.ValueInfoProto}
             }}
          end

          []
        ),
        (
          def field_def(:output) do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "output",
               kind: :unpacked,
               label: :repeated,
               name: :output,
               tag: 12,
               type: {:message, Transmuter.Onnx.ValueInfoProto}
             }}
          end

          def field_def("output") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "output",
               kind: :unpacked,
               label: :repeated,
               name: :output,
               tag: 12,
               type: {:message, Transmuter.Onnx.ValueInfoProto}
             }}
          end

          []
        ),
        (
          def field_def(:value_info) do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "valueInfo",
               kind: :unpacked,
               label: :repeated,
               name: :value_info,
               tag: 13,
               type: {:message, Transmuter.Onnx.ValueInfoProto}
             }}
          end

          def field_def("valueInfo") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "valueInfo",
               kind: :unpacked,
               label: :repeated,
               name: :value_info,
               tag: 13,
               type: {:message, Transmuter.Onnx.ValueInfoProto}
             }}
          end

          def field_def("value_info") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "valueInfo",
               kind: :unpacked,
               label: :repeated,
               name: :value_info,
               tag: 13,
               type: {:message, Transmuter.Onnx.ValueInfoProto}
             }}
          end
        ),
        (
          def field_def(:quantization_annotation) do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "quantizationAnnotation",
               kind: :unpacked,
               label: :repeated,
               name: :quantization_annotation,
               tag: 14,
               type: {:message, Transmuter.Onnx.TensorAnnotation}
             }}
          end

          def field_def("quantizationAnnotation") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "quantizationAnnotation",
               kind: :unpacked,
               label: :repeated,
               name: :quantization_annotation,
               tag: 14,
               type: {:message, Transmuter.Onnx.TensorAnnotation}
             }}
          end

          def field_def("quantization_annotation") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "quantizationAnnotation",
               kind: :unpacked,
               label: :repeated,
               name: :quantization_annotation,
               tag: 14,
               type: {:message, Transmuter.Onnx.TensorAnnotation}
             }}
          end
        ),
        (
          def field_def(:sparse_initializer) do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "sparseInitializer",
               kind: :unpacked,
               label: :repeated,
               name: :sparse_initializer,
               tag: 15,
               type: {:message, Transmuter.Onnx.SparseTensorProto}
             }}
          end

          def field_def("sparseInitializer") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "sparseInitializer",
               kind: :unpacked,
               label: :repeated,
               name: :sparse_initializer,
               tag: 15,
               type: {:message, Transmuter.Onnx.SparseTensorProto}
             }}
          end

          def field_def("sparse_initializer") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "sparseInitializer",
               kind: :unpacked,
               label: :repeated,
               name: :sparse_initializer,
               tag: 15,
               type: {:message, Transmuter.Onnx.SparseTensorProto}
             }}
          end
        ),
        (
          def field_def(:metadata_props) do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "metadataProps",
               kind: :unpacked,
               label: :repeated,
               name: :metadata_props,
               tag: 16,
               type: {:message, Transmuter.Onnx.StringStringEntryProto}
             }}
          end

          def field_def("metadataProps") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "metadataProps",
               kind: :unpacked,
               label: :repeated,
               name: :metadata_props,
               tag: 16,
               type: {:message, Transmuter.Onnx.StringStringEntryProto}
             }}
          end

          def field_def("metadata_props") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "metadataProps",
               kind: :unpacked,
               label: :repeated,
               name: :metadata_props,
               tag: 16,
               type: {:message, Transmuter.Onnx.StringStringEntryProto}
             }}
          end
        ),
        def field_def(_) do
          {:error, :no_such_field}
        end
      ]
    )

    (
      @spec unknown_fields(struct) :: [{non_neg_integer, Protox.Types.tag(), binary}]
      def unknown_fields(msg) do
        msg.__uf__
      end

      @spec unknown_fields_name() :: :__uf__
      def unknown_fields_name() do
        :__uf__
      end

      @spec clear_unknown_fields(struct) :: struct
      def clear_unknown_fields(msg) do
        struct!(msg, [{unknown_fields_name(), []}])
      end
    )

    (
      @spec required_fields() :: []
      def required_fields() do
        []
      end
    )

    (
      @spec syntax() :: atom()
      def syntax() do
        :proto2
      end
    )

    [
      @spec(default(atom) :: {:ok, boolean | integer | String.t() | float} | {:error, atom}),
      def default(:node) do
        {:error, :no_default_value}
      end,
      def default(:name) do
        {:ok, ""}
      end,
      def default(:initializer) do
        {:error, :no_default_value}
      end,
      def default(:doc_string) do
        {:ok, ""}
      end,
      def default(:input) do
        {:error, :no_default_value}
      end,
      def default(:output) do
        {:error, :no_default_value}
      end,
      def default(:value_info) do
        {:error, :no_default_value}
      end,
      def default(:quantization_annotation) do
        {:error, :no_default_value}
      end,
      def default(:sparse_initializer) do
        {:error, :no_default_value}
      end,
      def default(:metadata_props) do
        {:error, :no_default_value}
      end,
      def default(_) do
        {:error, :no_such_field}
      end
    ]

    (
      @spec file_options() :: struct()
      def file_options() do
        file_options = %{
          __struct__: Protox.Google.Protobuf.FileOptions,
          __uf__: [],
          cc_enable_arenas: nil,
          cc_generic_services: nil,
          csharp_namespace: nil,
          deprecated: nil,
          go_package: nil,
          java_generate_equals_and_hash: nil,
          java_generic_services: nil,
          java_multiple_files: nil,
          java_outer_classname: nil,
          java_package: nil,
          java_string_check_utf8: nil,
          objc_class_prefix: nil,
          optimize_for: :LITE_RUNTIME,
          php_class_prefix: nil,
          php_generic_services: nil,
          php_metadata_namespace: nil,
          php_namespace: nil,
          py_generic_services: nil,
          ruby_package: nil,
          swift_prefix: nil,
          uninterpreted_option: []
        }

        case function_exported?(Google.Protobuf.FileOptions, :decode!, 1) do
          true ->
            bytes =
              file_options
              |> Protox.Google.Protobuf.FileOptions.encode!()
              |> :binary.list_to_bin()

            apply(Google.Protobuf.FileOptions, :decode!, [bytes])

          false ->
            file_options
        end
      end
    )
  end,
  defmodule Transmuter.Onnx.IntIntListEntryProto do
    @moduledoc false
    defstruct key: nil, value: [], __uf__: []

    (
      (
        @spec encode(struct) :: {:ok, iodata} | {:error, any}
        def encode(msg) do
          try do
            {:ok, encode!(msg)}
          rescue
            e in [Protox.EncodingError, Protox.RequiredFieldsError] -> {:error, e}
          end
        end

        @spec encode!(struct) :: iodata | no_return
        def encode!(msg) do
          [] |> encode_key(msg) |> encode_value(msg) |> encode_unknown_fields(msg)
        end
      )

      []

      [
        defp encode_key(acc, msg) do
          try do
            case msg.key do
              nil -> acc
              _ -> [acc, "\b", Protox.Encode.encode_int64(msg.key)]
            end
          rescue
            ArgumentError ->
              reraise Protox.EncodingError.new(:key, "invalid field value"), __STACKTRACE__
          end
        end,
        defp encode_value(acc, msg) do
          try do
            case msg.value do
              [] ->
                acc

              values ->
                [
                  acc,
                  Enum.reduce(values, [], fn value, acc ->
                    [acc, "\x10", Protox.Encode.encode_int64(value)]
                  end)
                ]
            end
          rescue
            ArgumentError ->
              reraise Protox.EncodingError.new(:value, "invalid field value"), __STACKTRACE__
          end
        end
      ]

      defp encode_unknown_fields(acc, msg) do
        Enum.reduce(msg.__struct__.unknown_fields(msg), acc, fn {tag, wire_type, bytes}, acc ->
          case wire_type do
            0 ->
              [acc, Protox.Encode.make_key_bytes(tag, :int32), bytes]

            1 ->
              [acc, Protox.Encode.make_key_bytes(tag, :double), bytes]

            2 ->
              len_bytes = bytes |> byte_size() |> Protox.Varint.encode()
              [acc, Protox.Encode.make_key_bytes(tag, :packed), len_bytes, bytes]

            5 ->
              [acc, Protox.Encode.make_key_bytes(tag, :float), bytes]
          end
        end)
      end
    )

    (
      (
        @spec decode(binary) :: {:ok, struct} | {:error, any}
        def decode(bytes) do
          try do
            {:ok, decode!(bytes)}
          rescue
            e in [Protox.DecodingError, Protox.IllegalTagError, Protox.RequiredFieldsError] ->
              {:error, e}
          end
        end

        (
          @spec decode!(binary) :: struct | no_return
          def decode!(bytes) do
            parse_key_value(bytes, struct(Transmuter.Onnx.IntIntListEntryProto))
          end
        )
      )

      (
        @spec parse_key_value(binary, struct) :: struct
        defp parse_key_value(<<>>, msg) do
          msg
        end

        defp parse_key_value(bytes, msg) do
          {field, rest} =
            case Protox.Decode.parse_key(bytes) do
              {0, _, _} ->
                raise %Protox.IllegalTagError{}

              {1, _, bytes} ->
                {value, rest} = Protox.Decode.parse_int64(bytes)
                {[key: value], rest}

              {2, 2, bytes} ->
                {len, bytes} = Protox.Varint.decode(bytes)
                {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)
                {[value: msg.value ++ Protox.Decode.parse_repeated_int64([], delimited)], rest}

              {2, _, bytes} ->
                {value, rest} = Protox.Decode.parse_int64(bytes)
                {[value: msg.value ++ [value]], rest}

              {tag, wire_type, rest} ->
                {value, rest} = Protox.Decode.parse_unknown(tag, wire_type, rest)

                {[
                   {msg.__struct__.unknown_fields_name(),
                    [value | msg.__struct__.unknown_fields(msg)]}
                 ], rest}
            end

          msg_updated = struct(msg, field)
          parse_key_value(rest, msg_updated)
        end
      )

      []
    )

    (
      @spec json_decode(iodata(), keyword()) :: {:ok, struct()} | {:error, any()}
      def json_decode(input, opts \\ []) do
        try do
          {:ok, json_decode!(input, opts)}
        rescue
          e in Protox.JsonDecodingError -> {:error, e}
        end
      end

      @spec json_decode!(iodata(), keyword()) :: struct() | no_return()
      def json_decode!(input, opts \\ []) do
        {json_library_wrapper, json_library} = Protox.JsonLibrary.get_library(opts, :decode)

        Protox.JsonDecode.decode!(
          input,
          Transmuter.Onnx.IntIntListEntryProto,
          &json_library_wrapper.decode!(json_library, &1)
        )
      end

      @spec json_encode(struct(), keyword()) :: {:ok, iodata()} | {:error, any()}
      def json_encode(msg, opts \\ []) do
        try do
          {:ok, json_encode!(msg, opts)}
        rescue
          e in Protox.JsonEncodingError -> {:error, e}
        end
      end

      @spec json_encode!(struct(), keyword()) :: iodata() | no_return()
      def json_encode!(msg, opts \\ []) do
        {json_library_wrapper, json_library} = Protox.JsonLibrary.get_library(opts, :encode)
        Protox.JsonEncode.encode!(msg, &json_library_wrapper.encode!(json_library, &1))
      end
    )

    (
      @deprecated "Use fields_defs()/0 instead"
      @spec defs() :: %{
              required(non_neg_integer) => {atom, Protox.Types.kind(), Protox.Types.type()}
            }
      def defs() do
        %{1 => {:key, {:scalar, 0}, :int64}, 2 => {:value, :unpacked, :int64}}
      end

      @deprecated "Use fields_defs()/0 instead"
      @spec defs_by_name() :: %{
              required(atom) => {non_neg_integer, Protox.Types.kind(), Protox.Types.type()}
            }
      def defs_by_name() do
        %{key: {1, {:scalar, 0}, :int64}, value: {2, :unpacked, :int64}}
      end
    )

    (
      @spec fields_defs() :: list(Protox.Field.t())
      def fields_defs() do
        [
          %{
            __struct__: Protox.Field,
            json_name: "key",
            kind: {:scalar, 0},
            label: :optional,
            name: :key,
            tag: 1,
            type: :int64
          },
          %{
            __struct__: Protox.Field,
            json_name: "value",
            kind: :unpacked,
            label: :repeated,
            name: :value,
            tag: 2,
            type: :int64
          }
        ]
      end

      [
        @spec(field_def(atom) :: {:ok, Protox.Field.t()} | {:error, :no_such_field}),
        (
          def field_def(:key) do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "key",
               kind: {:scalar, 0},
               label: :optional,
               name: :key,
               tag: 1,
               type: :int64
             }}
          end

          def field_def("key") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "key",
               kind: {:scalar, 0},
               label: :optional,
               name: :key,
               tag: 1,
               type: :int64
             }}
          end

          []
        ),
        (
          def field_def(:value) do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "value",
               kind: :unpacked,
               label: :repeated,
               name: :value,
               tag: 2,
               type: :int64
             }}
          end

          def field_def("value") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "value",
               kind: :unpacked,
               label: :repeated,
               name: :value,
               tag: 2,
               type: :int64
             }}
          end

          []
        ),
        def field_def(_) do
          {:error, :no_such_field}
        end
      ]
    )

    (
      @spec unknown_fields(struct) :: [{non_neg_integer, Protox.Types.tag(), binary}]
      def unknown_fields(msg) do
        msg.__uf__
      end

      @spec unknown_fields_name() :: :__uf__
      def unknown_fields_name() do
        :__uf__
      end

      @spec clear_unknown_fields(struct) :: struct
      def clear_unknown_fields(msg) do
        struct!(msg, [{unknown_fields_name(), []}])
      end
    )

    (
      @spec required_fields() :: []
      def required_fields() do
        []
      end
    )

    (
      @spec syntax() :: atom()
      def syntax() do
        :proto2
      end
    )

    [
      @spec(default(atom) :: {:ok, boolean | integer | String.t() | float} | {:error, atom}),
      def default(:key) do
        {:ok, 0}
      end,
      def default(:value) do
        {:error, :no_default_value}
      end,
      def default(_) do
        {:error, :no_such_field}
      end
    ]

    (
      @spec file_options() :: struct()
      def file_options() do
        file_options = %{
          __struct__: Protox.Google.Protobuf.FileOptions,
          __uf__: [],
          cc_enable_arenas: nil,
          cc_generic_services: nil,
          csharp_namespace: nil,
          deprecated: nil,
          go_package: nil,
          java_generate_equals_and_hash: nil,
          java_generic_services: nil,
          java_multiple_files: nil,
          java_outer_classname: nil,
          java_package: nil,
          java_string_check_utf8: nil,
          objc_class_prefix: nil,
          optimize_for: :LITE_RUNTIME,
          php_class_prefix: nil,
          php_generic_services: nil,
          php_metadata_namespace: nil,
          php_namespace: nil,
          py_generic_services: nil,
          ruby_package: nil,
          swift_prefix: nil,
          uninterpreted_option: []
        }

        case function_exported?(Google.Protobuf.FileOptions, :decode!, 1) do
          true ->
            bytes =
              file_options
              |> Protox.Google.Protobuf.FileOptions.encode!()
              |> :binary.list_to_bin()

            apply(Google.Protobuf.FileOptions, :decode!, [bytes])

          false ->
            file_options
        end
      end
    )
  end,
  defmodule Transmuter.Onnx.ModelProto do
    @moduledoc false
    defstruct ir_version: nil,
              producer_name: nil,
              producer_version: nil,
              domain: nil,
              model_version: nil,
              doc_string: nil,
              graph: nil,
              opset_import: [],
              metadata_props: [],
              training_info: [],
              functions: [],
              configuration: [],
              __uf__: []

    (
      (
        @spec encode(struct) :: {:ok, iodata} | {:error, any}
        def encode(msg) do
          try do
            {:ok, encode!(msg)}
          rescue
            e in [Protox.EncodingError, Protox.RequiredFieldsError] -> {:error, e}
          end
        end

        @spec encode!(struct) :: iodata | no_return
        def encode!(msg) do
          []
          |> encode_ir_version(msg)
          |> encode_producer_name(msg)
          |> encode_producer_version(msg)
          |> encode_domain(msg)
          |> encode_model_version(msg)
          |> encode_doc_string(msg)
          |> encode_graph(msg)
          |> encode_opset_import(msg)
          |> encode_metadata_props(msg)
          |> encode_training_info(msg)
          |> encode_functions(msg)
          |> encode_configuration(msg)
          |> encode_unknown_fields(msg)
        end
      )

      []

      [
        defp encode_ir_version(acc, msg) do
          try do
            case msg.ir_version do
              nil -> acc
              _ -> [acc, "\b", Protox.Encode.encode_int64(msg.ir_version)]
            end
          rescue
            ArgumentError ->
              reraise Protox.EncodingError.new(:ir_version, "invalid field value"), __STACKTRACE__
          end
        end,
        defp encode_producer_name(acc, msg) do
          try do
            case msg.producer_name do
              nil -> acc
              _ -> [acc, "\x12", Protox.Encode.encode_string(msg.producer_name)]
            end
          rescue
            ArgumentError ->
              reraise Protox.EncodingError.new(:producer_name, "invalid field value"),
                      __STACKTRACE__
          end
        end,
        defp encode_producer_version(acc, msg) do
          try do
            case msg.producer_version do
              nil -> acc
              _ -> [acc, "\x1A", Protox.Encode.encode_string(msg.producer_version)]
            end
          rescue
            ArgumentError ->
              reraise Protox.EncodingError.new(:producer_version, "invalid field value"),
                      __STACKTRACE__
          end
        end,
        defp encode_domain(acc, msg) do
          try do
            case msg.domain do
              nil -> acc
              _ -> [acc, "\"", Protox.Encode.encode_string(msg.domain)]
            end
          rescue
            ArgumentError ->
              reraise Protox.EncodingError.new(:domain, "invalid field value"), __STACKTRACE__
          end
        end,
        defp encode_model_version(acc, msg) do
          try do
            case msg.model_version do
              nil -> acc
              _ -> [acc, "(", Protox.Encode.encode_int64(msg.model_version)]
            end
          rescue
            ArgumentError ->
              reraise Protox.EncodingError.new(:model_version, "invalid field value"),
                      __STACKTRACE__
          end
        end,
        defp encode_doc_string(acc, msg) do
          try do
            case msg.doc_string do
              nil -> acc
              _ -> [acc, "2", Protox.Encode.encode_string(msg.doc_string)]
            end
          rescue
            ArgumentError ->
              reraise Protox.EncodingError.new(:doc_string, "invalid field value"), __STACKTRACE__
          end
        end,
        defp encode_graph(acc, msg) do
          try do
            case msg.graph do
              nil -> acc
              _ -> [acc, ":", Protox.Encode.encode_message(msg.graph)]
            end
          rescue
            ArgumentError ->
              reraise Protox.EncodingError.new(:graph, "invalid field value"), __STACKTRACE__
          end
        end,
        defp encode_opset_import(acc, msg) do
          try do
            case msg.opset_import do
              [] ->
                acc

              values ->
                [
                  acc,
                  Enum.reduce(values, [], fn value, acc ->
                    [acc, "B", Protox.Encode.encode_message(value)]
                  end)
                ]
            end
          rescue
            ArgumentError ->
              reraise Protox.EncodingError.new(:opset_import, "invalid field value"),
                      __STACKTRACE__
          end
        end,
        defp encode_metadata_props(acc, msg) do
          try do
            case msg.metadata_props do
              [] ->
                acc

              values ->
                [
                  acc,
                  Enum.reduce(values, [], fn value, acc ->
                    [acc, "r", Protox.Encode.encode_message(value)]
                  end)
                ]
            end
          rescue
            ArgumentError ->
              reraise Protox.EncodingError.new(:metadata_props, "invalid field value"),
                      __STACKTRACE__
          end
        end,
        defp encode_training_info(acc, msg) do
          try do
            case msg.training_info do
              [] ->
                acc

              values ->
                [
                  acc,
                  Enum.reduce(values, [], fn value, acc ->
                    [acc, "\xA2\x01", Protox.Encode.encode_message(value)]
                  end)
                ]
            end
          rescue
            ArgumentError ->
              reraise Protox.EncodingError.new(:training_info, "invalid field value"),
                      __STACKTRACE__
          end
        end,
        defp encode_functions(acc, msg) do
          try do
            case msg.functions do
              [] ->
                acc

              values ->
                [
                  acc,
                  Enum.reduce(values, [], fn value, acc ->
                    [acc, "\xCA\x01", Protox.Encode.encode_message(value)]
                  end)
                ]
            end
          rescue
            ArgumentError ->
              reraise Protox.EncodingError.new(:functions, "invalid field value"), __STACKTRACE__
          end
        end,
        defp encode_configuration(acc, msg) do
          try do
            case msg.configuration do
              [] ->
                acc

              values ->
                [
                  acc,
                  Enum.reduce(values, [], fn value, acc ->
                    [acc, "\xD2\x01", Protox.Encode.encode_message(value)]
                  end)
                ]
            end
          rescue
            ArgumentError ->
              reraise Protox.EncodingError.new(:configuration, "invalid field value"),
                      __STACKTRACE__
          end
        end
      ]

      defp encode_unknown_fields(acc, msg) do
        Enum.reduce(msg.__struct__.unknown_fields(msg), acc, fn {tag, wire_type, bytes}, acc ->
          case wire_type do
            0 ->
              [acc, Protox.Encode.make_key_bytes(tag, :int32), bytes]

            1 ->
              [acc, Protox.Encode.make_key_bytes(tag, :double), bytes]

            2 ->
              len_bytes = bytes |> byte_size() |> Protox.Varint.encode()
              [acc, Protox.Encode.make_key_bytes(tag, :packed), len_bytes, bytes]

            5 ->
              [acc, Protox.Encode.make_key_bytes(tag, :float), bytes]
          end
        end)
      end
    )

    (
      (
        @spec decode(binary) :: {:ok, struct} | {:error, any}
        def decode(bytes) do
          try do
            {:ok, decode!(bytes)}
          rescue
            e in [Protox.DecodingError, Protox.IllegalTagError, Protox.RequiredFieldsError] ->
              {:error, e}
          end
        end

        (
          @spec decode!(binary) :: struct | no_return
          def decode!(bytes) do
            parse_key_value(bytes, struct(Transmuter.Onnx.ModelProto))
          end
        )
      )

      (
        @spec parse_key_value(binary, struct) :: struct
        defp parse_key_value(<<>>, msg) do
          msg
        end

        defp parse_key_value(bytes, msg) do
          {field, rest} =
            case Protox.Decode.parse_key(bytes) do
              {0, _, _} ->
                raise %Protox.IllegalTagError{}

              {1, _, bytes} ->
                {value, rest} = Protox.Decode.parse_int64(bytes)
                {[ir_version: value], rest}

              {2, _, bytes} ->
                {len, bytes} = Protox.Varint.decode(bytes)
                {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)
                {[producer_name: Protox.Decode.validate_string!(delimited)], rest}

              {3, _, bytes} ->
                {len, bytes} = Protox.Varint.decode(bytes)
                {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)
                {[producer_version: Protox.Decode.validate_string!(delimited)], rest}

              {4, _, bytes} ->
                {len, bytes} = Protox.Varint.decode(bytes)
                {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)
                {[domain: Protox.Decode.validate_string!(delimited)], rest}

              {5, _, bytes} ->
                {value, rest} = Protox.Decode.parse_int64(bytes)
                {[model_version: value], rest}

              {6, _, bytes} ->
                {len, bytes} = Protox.Varint.decode(bytes)
                {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)
                {[doc_string: Protox.Decode.validate_string!(delimited)], rest}

              {7, _, bytes} ->
                {len, bytes} = Protox.Varint.decode(bytes)
                {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)

                {[
                   graph:
                     Protox.MergeMessage.merge(
                       msg.graph,
                       Transmuter.Onnx.GraphProto.decode!(delimited)
                     )
                 ], rest}

              {8, _, bytes} ->
                {len, bytes} = Protox.Varint.decode(bytes)
                {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)

                {[
                   opset_import:
                     msg.opset_import ++ [Transmuter.Onnx.OperatorSetIdProto.decode!(delimited)]
                 ], rest}

              {14, _, bytes} ->
                {len, bytes} = Protox.Varint.decode(bytes)
                {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)

                {[
                   metadata_props:
                     msg.metadata_props ++
                       [Transmuter.Onnx.StringStringEntryProto.decode!(delimited)]
                 ], rest}

              {20, _, bytes} ->
                {len, bytes} = Protox.Varint.decode(bytes)
                {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)

                {[
                   training_info:
                     msg.training_info ++ [Transmuter.Onnx.TrainingInfoProto.decode!(delimited)]
                 ], rest}

              {25, _, bytes} ->
                {len, bytes} = Protox.Varint.decode(bytes)
                {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)

                {[functions: msg.functions ++ [Transmuter.Onnx.FunctionProto.decode!(delimited)]],
                 rest}

              {26, _, bytes} ->
                {len, bytes} = Protox.Varint.decode(bytes)
                {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)

                {[
                   configuration:
                     msg.configuration ++
                       [Transmuter.Onnx.DeviceConfigurationProto.decode!(delimited)]
                 ], rest}

              {tag, wire_type, rest} ->
                {value, rest} = Protox.Decode.parse_unknown(tag, wire_type, rest)

                {[
                   {msg.__struct__.unknown_fields_name(),
                    [value | msg.__struct__.unknown_fields(msg)]}
                 ], rest}
            end

          msg_updated = struct(msg, field)
          parse_key_value(rest, msg_updated)
        end
      )

      []
    )

    (
      @spec json_decode(iodata(), keyword()) :: {:ok, struct()} | {:error, any()}
      def json_decode(input, opts \\ []) do
        try do
          {:ok, json_decode!(input, opts)}
        rescue
          e in Protox.JsonDecodingError -> {:error, e}
        end
      end

      @spec json_decode!(iodata(), keyword()) :: struct() | no_return()
      def json_decode!(input, opts \\ []) do
        {json_library_wrapper, json_library} = Protox.JsonLibrary.get_library(opts, :decode)

        Protox.JsonDecode.decode!(
          input,
          Transmuter.Onnx.ModelProto,
          &json_library_wrapper.decode!(json_library, &1)
        )
      end

      @spec json_encode(struct(), keyword()) :: {:ok, iodata()} | {:error, any()}
      def json_encode(msg, opts \\ []) do
        try do
          {:ok, json_encode!(msg, opts)}
        rescue
          e in Protox.JsonEncodingError -> {:error, e}
        end
      end

      @spec json_encode!(struct(), keyword()) :: iodata() | no_return()
      def json_encode!(msg, opts \\ []) do
        {json_library_wrapper, json_library} = Protox.JsonLibrary.get_library(opts, :encode)
        Protox.JsonEncode.encode!(msg, &json_library_wrapper.encode!(json_library, &1))
      end
    )

    (
      @deprecated "Use fields_defs()/0 instead"
      @spec defs() :: %{
              required(non_neg_integer) => {atom, Protox.Types.kind(), Protox.Types.type()}
            }
      def defs() do
        %{
          1 => {:ir_version, {:scalar, 0}, :int64},
          2 => {:producer_name, {:scalar, ""}, :string},
          3 => {:producer_version, {:scalar, ""}, :string},
          4 => {:domain, {:scalar, ""}, :string},
          5 => {:model_version, {:scalar, 0}, :int64},
          6 => {:doc_string, {:scalar, ""}, :string},
          7 => {:graph, {:scalar, nil}, {:message, Transmuter.Onnx.GraphProto}},
          8 => {:opset_import, :unpacked, {:message, Transmuter.Onnx.OperatorSetIdProto}},
          14 => {:metadata_props, :unpacked, {:message, Transmuter.Onnx.StringStringEntryProto}},
          20 => {:training_info, :unpacked, {:message, Transmuter.Onnx.TrainingInfoProto}},
          25 => {:functions, :unpacked, {:message, Transmuter.Onnx.FunctionProto}},
          26 => {:configuration, :unpacked, {:message, Transmuter.Onnx.DeviceConfigurationProto}}
        }
      end

      @deprecated "Use fields_defs()/0 instead"
      @spec defs_by_name() :: %{
              required(atom) => {non_neg_integer, Protox.Types.kind(), Protox.Types.type()}
            }
      def defs_by_name() do
        %{
          configuration: {26, :unpacked, {:message, Transmuter.Onnx.DeviceConfigurationProto}},
          doc_string: {6, {:scalar, ""}, :string},
          domain: {4, {:scalar, ""}, :string},
          functions: {25, :unpacked, {:message, Transmuter.Onnx.FunctionProto}},
          graph: {7, {:scalar, nil}, {:message, Transmuter.Onnx.GraphProto}},
          ir_version: {1, {:scalar, 0}, :int64},
          metadata_props: {14, :unpacked, {:message, Transmuter.Onnx.StringStringEntryProto}},
          model_version: {5, {:scalar, 0}, :int64},
          opset_import: {8, :unpacked, {:message, Transmuter.Onnx.OperatorSetIdProto}},
          producer_name: {2, {:scalar, ""}, :string},
          producer_version: {3, {:scalar, ""}, :string},
          training_info: {20, :unpacked, {:message, Transmuter.Onnx.TrainingInfoProto}}
        }
      end
    )

    (
      @spec fields_defs() :: list(Protox.Field.t())
      def fields_defs() do
        [
          %{
            __struct__: Protox.Field,
            json_name: "irVersion",
            kind: {:scalar, 0},
            label: :optional,
            name: :ir_version,
            tag: 1,
            type: :int64
          },
          %{
            __struct__: Protox.Field,
            json_name: "producerName",
            kind: {:scalar, ""},
            label: :optional,
            name: :producer_name,
            tag: 2,
            type: :string
          },
          %{
            __struct__: Protox.Field,
            json_name: "producerVersion",
            kind: {:scalar, ""},
            label: :optional,
            name: :producer_version,
            tag: 3,
            type: :string
          },
          %{
            __struct__: Protox.Field,
            json_name: "domain",
            kind: {:scalar, ""},
            label: :optional,
            name: :domain,
            tag: 4,
            type: :string
          },
          %{
            __struct__: Protox.Field,
            json_name: "modelVersion",
            kind: {:scalar, 0},
            label: :optional,
            name: :model_version,
            tag: 5,
            type: :int64
          },
          %{
            __struct__: Protox.Field,
            json_name: "docString",
            kind: {:scalar, ""},
            label: :optional,
            name: :doc_string,
            tag: 6,
            type: :string
          },
          %{
            __struct__: Protox.Field,
            json_name: "graph",
            kind: {:scalar, nil},
            label: :optional,
            name: :graph,
            tag: 7,
            type: {:message, Transmuter.Onnx.GraphProto}
          },
          %{
            __struct__: Protox.Field,
            json_name: "opsetImport",
            kind: :unpacked,
            label: :repeated,
            name: :opset_import,
            tag: 8,
            type: {:message, Transmuter.Onnx.OperatorSetIdProto}
          },
          %{
            __struct__: Protox.Field,
            json_name: "metadataProps",
            kind: :unpacked,
            label: :repeated,
            name: :metadata_props,
            tag: 14,
            type: {:message, Transmuter.Onnx.StringStringEntryProto}
          },
          %{
            __struct__: Protox.Field,
            json_name: "trainingInfo",
            kind: :unpacked,
            label: :repeated,
            name: :training_info,
            tag: 20,
            type: {:message, Transmuter.Onnx.TrainingInfoProto}
          },
          %{
            __struct__: Protox.Field,
            json_name: "functions",
            kind: :unpacked,
            label: :repeated,
            name: :functions,
            tag: 25,
            type: {:message, Transmuter.Onnx.FunctionProto}
          },
          %{
            __struct__: Protox.Field,
            json_name: "configuration",
            kind: :unpacked,
            label: :repeated,
            name: :configuration,
            tag: 26,
            type: {:message, Transmuter.Onnx.DeviceConfigurationProto}
          }
        ]
      end

      [
        @spec(field_def(atom) :: {:ok, Protox.Field.t()} | {:error, :no_such_field}),
        (
          def field_def(:ir_version) do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "irVersion",
               kind: {:scalar, 0},
               label: :optional,
               name: :ir_version,
               tag: 1,
               type: :int64
             }}
          end

          def field_def("irVersion") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "irVersion",
               kind: {:scalar, 0},
               label: :optional,
               name: :ir_version,
               tag: 1,
               type: :int64
             }}
          end

          def field_def("ir_version") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "irVersion",
               kind: {:scalar, 0},
               label: :optional,
               name: :ir_version,
               tag: 1,
               type: :int64
             }}
          end
        ),
        (
          def field_def(:producer_name) do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "producerName",
               kind: {:scalar, ""},
               label: :optional,
               name: :producer_name,
               tag: 2,
               type: :string
             }}
          end

          def field_def("producerName") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "producerName",
               kind: {:scalar, ""},
               label: :optional,
               name: :producer_name,
               tag: 2,
               type: :string
             }}
          end

          def field_def("producer_name") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "producerName",
               kind: {:scalar, ""},
               label: :optional,
               name: :producer_name,
               tag: 2,
               type: :string
             }}
          end
        ),
        (
          def field_def(:producer_version) do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "producerVersion",
               kind: {:scalar, ""},
               label: :optional,
               name: :producer_version,
               tag: 3,
               type: :string
             }}
          end

          def field_def("producerVersion") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "producerVersion",
               kind: {:scalar, ""},
               label: :optional,
               name: :producer_version,
               tag: 3,
               type: :string
             }}
          end

          def field_def("producer_version") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "producerVersion",
               kind: {:scalar, ""},
               label: :optional,
               name: :producer_version,
               tag: 3,
               type: :string
             }}
          end
        ),
        (
          def field_def(:domain) do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "domain",
               kind: {:scalar, ""},
               label: :optional,
               name: :domain,
               tag: 4,
               type: :string
             }}
          end

          def field_def("domain") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "domain",
               kind: {:scalar, ""},
               label: :optional,
               name: :domain,
               tag: 4,
               type: :string
             }}
          end

          []
        ),
        (
          def field_def(:model_version) do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "modelVersion",
               kind: {:scalar, 0},
               label: :optional,
               name: :model_version,
               tag: 5,
               type: :int64
             }}
          end

          def field_def("modelVersion") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "modelVersion",
               kind: {:scalar, 0},
               label: :optional,
               name: :model_version,
               tag: 5,
               type: :int64
             }}
          end

          def field_def("model_version") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "modelVersion",
               kind: {:scalar, 0},
               label: :optional,
               name: :model_version,
               tag: 5,
               type: :int64
             }}
          end
        ),
        (
          def field_def(:doc_string) do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "docString",
               kind: {:scalar, ""},
               label: :optional,
               name: :doc_string,
               tag: 6,
               type: :string
             }}
          end

          def field_def("docString") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "docString",
               kind: {:scalar, ""},
               label: :optional,
               name: :doc_string,
               tag: 6,
               type: :string
             }}
          end

          def field_def("doc_string") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "docString",
               kind: {:scalar, ""},
               label: :optional,
               name: :doc_string,
               tag: 6,
               type: :string
             }}
          end
        ),
        (
          def field_def(:graph) do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "graph",
               kind: {:scalar, nil},
               label: :optional,
               name: :graph,
               tag: 7,
               type: {:message, Transmuter.Onnx.GraphProto}
             }}
          end

          def field_def("graph") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "graph",
               kind: {:scalar, nil},
               label: :optional,
               name: :graph,
               tag: 7,
               type: {:message, Transmuter.Onnx.GraphProto}
             }}
          end

          []
        ),
        (
          def field_def(:opset_import) do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "opsetImport",
               kind: :unpacked,
               label: :repeated,
               name: :opset_import,
               tag: 8,
               type: {:message, Transmuter.Onnx.OperatorSetIdProto}
             }}
          end

          def field_def("opsetImport") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "opsetImport",
               kind: :unpacked,
               label: :repeated,
               name: :opset_import,
               tag: 8,
               type: {:message, Transmuter.Onnx.OperatorSetIdProto}
             }}
          end

          def field_def("opset_import") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "opsetImport",
               kind: :unpacked,
               label: :repeated,
               name: :opset_import,
               tag: 8,
               type: {:message, Transmuter.Onnx.OperatorSetIdProto}
             }}
          end
        ),
        (
          def field_def(:metadata_props) do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "metadataProps",
               kind: :unpacked,
               label: :repeated,
               name: :metadata_props,
               tag: 14,
               type: {:message, Transmuter.Onnx.StringStringEntryProto}
             }}
          end

          def field_def("metadataProps") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "metadataProps",
               kind: :unpacked,
               label: :repeated,
               name: :metadata_props,
               tag: 14,
               type: {:message, Transmuter.Onnx.StringStringEntryProto}
             }}
          end

          def field_def("metadata_props") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "metadataProps",
               kind: :unpacked,
               label: :repeated,
               name: :metadata_props,
               tag: 14,
               type: {:message, Transmuter.Onnx.StringStringEntryProto}
             }}
          end
        ),
        (
          def field_def(:training_info) do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "trainingInfo",
               kind: :unpacked,
               label: :repeated,
               name: :training_info,
               tag: 20,
               type: {:message, Transmuter.Onnx.TrainingInfoProto}
             }}
          end

          def field_def("trainingInfo") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "trainingInfo",
               kind: :unpacked,
               label: :repeated,
               name: :training_info,
               tag: 20,
               type: {:message, Transmuter.Onnx.TrainingInfoProto}
             }}
          end

          def field_def("training_info") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "trainingInfo",
               kind: :unpacked,
               label: :repeated,
               name: :training_info,
               tag: 20,
               type: {:message, Transmuter.Onnx.TrainingInfoProto}
             }}
          end
        ),
        (
          def field_def(:functions) do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "functions",
               kind: :unpacked,
               label: :repeated,
               name: :functions,
               tag: 25,
               type: {:message, Transmuter.Onnx.FunctionProto}
             }}
          end

          def field_def("functions") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "functions",
               kind: :unpacked,
               label: :repeated,
               name: :functions,
               tag: 25,
               type: {:message, Transmuter.Onnx.FunctionProto}
             }}
          end

          []
        ),
        (
          def field_def(:configuration) do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "configuration",
               kind: :unpacked,
               label: :repeated,
               name: :configuration,
               tag: 26,
               type: {:message, Transmuter.Onnx.DeviceConfigurationProto}
             }}
          end

          def field_def("configuration") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "configuration",
               kind: :unpacked,
               label: :repeated,
               name: :configuration,
               tag: 26,
               type: {:message, Transmuter.Onnx.DeviceConfigurationProto}
             }}
          end

          []
        ),
        def field_def(_) do
          {:error, :no_such_field}
        end
      ]
    )

    (
      @spec unknown_fields(struct) :: [{non_neg_integer, Protox.Types.tag(), binary}]
      def unknown_fields(msg) do
        msg.__uf__
      end

      @spec unknown_fields_name() :: :__uf__
      def unknown_fields_name() do
        :__uf__
      end

      @spec clear_unknown_fields(struct) :: struct
      def clear_unknown_fields(msg) do
        struct!(msg, [{unknown_fields_name(), []}])
      end
    )

    (
      @spec required_fields() :: []
      def required_fields() do
        []
      end
    )

    (
      @spec syntax() :: atom()
      def syntax() do
        :proto2
      end
    )

    [
      @spec(default(atom) :: {:ok, boolean | integer | String.t() | float} | {:error, atom}),
      def default(:ir_version) do
        {:ok, 0}
      end,
      def default(:producer_name) do
        {:ok, ""}
      end,
      def default(:producer_version) do
        {:ok, ""}
      end,
      def default(:domain) do
        {:ok, ""}
      end,
      def default(:model_version) do
        {:ok, 0}
      end,
      def default(:doc_string) do
        {:ok, ""}
      end,
      def default(:graph) do
        {:ok, nil}
      end,
      def default(:opset_import) do
        {:error, :no_default_value}
      end,
      def default(:metadata_props) do
        {:error, :no_default_value}
      end,
      def default(:training_info) do
        {:error, :no_default_value}
      end,
      def default(:functions) do
        {:error, :no_default_value}
      end,
      def default(:configuration) do
        {:error, :no_default_value}
      end,
      def default(_) do
        {:error, :no_such_field}
      end
    ]

    (
      @spec file_options() :: struct()
      def file_options() do
        file_options = %{
          __struct__: Protox.Google.Protobuf.FileOptions,
          __uf__: [],
          cc_enable_arenas: nil,
          cc_generic_services: nil,
          csharp_namespace: nil,
          deprecated: nil,
          go_package: nil,
          java_generate_equals_and_hash: nil,
          java_generic_services: nil,
          java_multiple_files: nil,
          java_outer_classname: nil,
          java_package: nil,
          java_string_check_utf8: nil,
          objc_class_prefix: nil,
          optimize_for: :LITE_RUNTIME,
          php_class_prefix: nil,
          php_generic_services: nil,
          php_metadata_namespace: nil,
          php_namespace: nil,
          py_generic_services: nil,
          ruby_package: nil,
          swift_prefix: nil,
          uninterpreted_option: []
        }

        case function_exported?(Google.Protobuf.FileOptions, :decode!, 1) do
          true ->
            bytes =
              file_options
              |> Protox.Google.Protobuf.FileOptions.encode!()
              |> :binary.list_to_bin()

            apply(Google.Protobuf.FileOptions, :decode!, [bytes])

          false ->
            file_options
        end
      end
    )
  end,
  defmodule Transmuter.Onnx.NodeDeviceConfigurationProto do
    @moduledoc false
    defstruct configuration_id: nil, sharding_spec: [], pipeline_stage: nil, __uf__: []

    (
      (
        @spec encode(struct) :: {:ok, iodata} | {:error, any}
        def encode(msg) do
          try do
            {:ok, encode!(msg)}
          rescue
            e in [Protox.EncodingError, Protox.RequiredFieldsError] -> {:error, e}
          end
        end

        @spec encode!(struct) :: iodata | no_return
        def encode!(msg) do
          []
          |> encode_configuration_id(msg)
          |> encode_sharding_spec(msg)
          |> encode_pipeline_stage(msg)
          |> encode_unknown_fields(msg)
        end
      )

      []

      [
        defp encode_configuration_id(acc, msg) do
          try do
            case msg.configuration_id do
              nil -> acc
              _ -> [acc, "\n", Protox.Encode.encode_string(msg.configuration_id)]
            end
          rescue
            ArgumentError ->
              reraise Protox.EncodingError.new(:configuration_id, "invalid field value"),
                      __STACKTRACE__
          end
        end,
        defp encode_sharding_spec(acc, msg) do
          try do
            case msg.sharding_spec do
              [] ->
                acc

              values ->
                [
                  acc,
                  Enum.reduce(values, [], fn value, acc ->
                    [acc, "\x12", Protox.Encode.encode_message(value)]
                  end)
                ]
            end
          rescue
            ArgumentError ->
              reraise Protox.EncodingError.new(:sharding_spec, "invalid field value"),
                      __STACKTRACE__
          end
        end,
        defp encode_pipeline_stage(acc, msg) do
          try do
            case msg.pipeline_stage do
              nil -> acc
              _ -> [acc, "\x18", Protox.Encode.encode_int32(msg.pipeline_stage)]
            end
          rescue
            ArgumentError ->
              reraise Protox.EncodingError.new(:pipeline_stage, "invalid field value"),
                      __STACKTRACE__
          end
        end
      ]

      defp encode_unknown_fields(acc, msg) do
        Enum.reduce(msg.__struct__.unknown_fields(msg), acc, fn {tag, wire_type, bytes}, acc ->
          case wire_type do
            0 ->
              [acc, Protox.Encode.make_key_bytes(tag, :int32), bytes]

            1 ->
              [acc, Protox.Encode.make_key_bytes(tag, :double), bytes]

            2 ->
              len_bytes = bytes |> byte_size() |> Protox.Varint.encode()
              [acc, Protox.Encode.make_key_bytes(tag, :packed), len_bytes, bytes]

            5 ->
              [acc, Protox.Encode.make_key_bytes(tag, :float), bytes]
          end
        end)
      end
    )

    (
      (
        @spec decode(binary) :: {:ok, struct} | {:error, any}
        def decode(bytes) do
          try do
            {:ok, decode!(bytes)}
          rescue
            e in [Protox.DecodingError, Protox.IllegalTagError, Protox.RequiredFieldsError] ->
              {:error, e}
          end
        end

        (
          @spec decode!(binary) :: struct | no_return
          def decode!(bytes) do
            parse_key_value(bytes, struct(Transmuter.Onnx.NodeDeviceConfigurationProto))
          end
        )
      )

      (
        @spec parse_key_value(binary, struct) :: struct
        defp parse_key_value(<<>>, msg) do
          msg
        end

        defp parse_key_value(bytes, msg) do
          {field, rest} =
            case Protox.Decode.parse_key(bytes) do
              {0, _, _} ->
                raise %Protox.IllegalTagError{}

              {1, _, bytes} ->
                {len, bytes} = Protox.Varint.decode(bytes)
                {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)
                {[configuration_id: Protox.Decode.validate_string!(delimited)], rest}

              {2, _, bytes} ->
                {len, bytes} = Protox.Varint.decode(bytes)
                {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)

                {[
                   sharding_spec:
                     msg.sharding_spec ++ [Transmuter.Onnx.ShardingSpecProto.decode!(delimited)]
                 ], rest}

              {3, _, bytes} ->
                {value, rest} = Protox.Decode.parse_int32(bytes)
                {[pipeline_stage: value], rest}

              {tag, wire_type, rest} ->
                {value, rest} = Protox.Decode.parse_unknown(tag, wire_type, rest)

                {[
                   {msg.__struct__.unknown_fields_name(),
                    [value | msg.__struct__.unknown_fields(msg)]}
                 ], rest}
            end

          msg_updated = struct(msg, field)
          parse_key_value(rest, msg_updated)
        end
      )

      []
    )

    (
      @spec json_decode(iodata(), keyword()) :: {:ok, struct()} | {:error, any()}
      def json_decode(input, opts \\ []) do
        try do
          {:ok, json_decode!(input, opts)}
        rescue
          e in Protox.JsonDecodingError -> {:error, e}
        end
      end

      @spec json_decode!(iodata(), keyword()) :: struct() | no_return()
      def json_decode!(input, opts \\ []) do
        {json_library_wrapper, json_library} = Protox.JsonLibrary.get_library(opts, :decode)

        Protox.JsonDecode.decode!(
          input,
          Transmuter.Onnx.NodeDeviceConfigurationProto,
          &json_library_wrapper.decode!(json_library, &1)
        )
      end

      @spec json_encode(struct(), keyword()) :: {:ok, iodata()} | {:error, any()}
      def json_encode(msg, opts \\ []) do
        try do
          {:ok, json_encode!(msg, opts)}
        rescue
          e in Protox.JsonEncodingError -> {:error, e}
        end
      end

      @spec json_encode!(struct(), keyword()) :: iodata() | no_return()
      def json_encode!(msg, opts \\ []) do
        {json_library_wrapper, json_library} = Protox.JsonLibrary.get_library(opts, :encode)
        Protox.JsonEncode.encode!(msg, &json_library_wrapper.encode!(json_library, &1))
      end
    )

    (
      @deprecated "Use fields_defs()/0 instead"
      @spec defs() :: %{
              required(non_neg_integer) => {atom, Protox.Types.kind(), Protox.Types.type()}
            }
      def defs() do
        %{
          1 => {:configuration_id, {:scalar, ""}, :string},
          2 => {:sharding_spec, :unpacked, {:message, Transmuter.Onnx.ShardingSpecProto}},
          3 => {:pipeline_stage, {:scalar, 0}, :int32}
        }
      end

      @deprecated "Use fields_defs()/0 instead"
      @spec defs_by_name() :: %{
              required(atom) => {non_neg_integer, Protox.Types.kind(), Protox.Types.type()}
            }
      def defs_by_name() do
        %{
          configuration_id: {1, {:scalar, ""}, :string},
          pipeline_stage: {3, {:scalar, 0}, :int32},
          sharding_spec: {2, :unpacked, {:message, Transmuter.Onnx.ShardingSpecProto}}
        }
      end
    )

    (
      @spec fields_defs() :: list(Protox.Field.t())
      def fields_defs() do
        [
          %{
            __struct__: Protox.Field,
            json_name: "configurationId",
            kind: {:scalar, ""},
            label: :optional,
            name: :configuration_id,
            tag: 1,
            type: :string
          },
          %{
            __struct__: Protox.Field,
            json_name: "shardingSpec",
            kind: :unpacked,
            label: :repeated,
            name: :sharding_spec,
            tag: 2,
            type: {:message, Transmuter.Onnx.ShardingSpecProto}
          },
          %{
            __struct__: Protox.Field,
            json_name: "pipelineStage",
            kind: {:scalar, 0},
            label: :optional,
            name: :pipeline_stage,
            tag: 3,
            type: :int32
          }
        ]
      end

      [
        @spec(field_def(atom) :: {:ok, Protox.Field.t()} | {:error, :no_such_field}),
        (
          def field_def(:configuration_id) do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "configurationId",
               kind: {:scalar, ""},
               label: :optional,
               name: :configuration_id,
               tag: 1,
               type: :string
             }}
          end

          def field_def("configurationId") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "configurationId",
               kind: {:scalar, ""},
               label: :optional,
               name: :configuration_id,
               tag: 1,
               type: :string
             }}
          end

          def field_def("configuration_id") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "configurationId",
               kind: {:scalar, ""},
               label: :optional,
               name: :configuration_id,
               tag: 1,
               type: :string
             }}
          end
        ),
        (
          def field_def(:sharding_spec) do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "shardingSpec",
               kind: :unpacked,
               label: :repeated,
               name: :sharding_spec,
               tag: 2,
               type: {:message, Transmuter.Onnx.ShardingSpecProto}
             }}
          end

          def field_def("shardingSpec") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "shardingSpec",
               kind: :unpacked,
               label: :repeated,
               name: :sharding_spec,
               tag: 2,
               type: {:message, Transmuter.Onnx.ShardingSpecProto}
             }}
          end

          def field_def("sharding_spec") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "shardingSpec",
               kind: :unpacked,
               label: :repeated,
               name: :sharding_spec,
               tag: 2,
               type: {:message, Transmuter.Onnx.ShardingSpecProto}
             }}
          end
        ),
        (
          def field_def(:pipeline_stage) do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "pipelineStage",
               kind: {:scalar, 0},
               label: :optional,
               name: :pipeline_stage,
               tag: 3,
               type: :int32
             }}
          end

          def field_def("pipelineStage") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "pipelineStage",
               kind: {:scalar, 0},
               label: :optional,
               name: :pipeline_stage,
               tag: 3,
               type: :int32
             }}
          end

          def field_def("pipeline_stage") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "pipelineStage",
               kind: {:scalar, 0},
               label: :optional,
               name: :pipeline_stage,
               tag: 3,
               type: :int32
             }}
          end
        ),
        def field_def(_) do
          {:error, :no_such_field}
        end
      ]
    )

    (
      @spec unknown_fields(struct) :: [{non_neg_integer, Protox.Types.tag(), binary}]
      def unknown_fields(msg) do
        msg.__uf__
      end

      @spec unknown_fields_name() :: :__uf__
      def unknown_fields_name() do
        :__uf__
      end

      @spec clear_unknown_fields(struct) :: struct
      def clear_unknown_fields(msg) do
        struct!(msg, [{unknown_fields_name(), []}])
      end
    )

    (
      @spec required_fields() :: []
      def required_fields() do
        []
      end
    )

    (
      @spec syntax() :: atom()
      def syntax() do
        :proto2
      end
    )

    [
      @spec(default(atom) :: {:ok, boolean | integer | String.t() | float} | {:error, atom}),
      def default(:configuration_id) do
        {:ok, ""}
      end,
      def default(:sharding_spec) do
        {:error, :no_default_value}
      end,
      def default(:pipeline_stage) do
        {:ok, 0}
      end,
      def default(_) do
        {:error, :no_such_field}
      end
    ]

    (
      @spec file_options() :: struct()
      def file_options() do
        file_options = %{
          __struct__: Protox.Google.Protobuf.FileOptions,
          __uf__: [],
          cc_enable_arenas: nil,
          cc_generic_services: nil,
          csharp_namespace: nil,
          deprecated: nil,
          go_package: nil,
          java_generate_equals_and_hash: nil,
          java_generic_services: nil,
          java_multiple_files: nil,
          java_outer_classname: nil,
          java_package: nil,
          java_string_check_utf8: nil,
          objc_class_prefix: nil,
          optimize_for: :LITE_RUNTIME,
          php_class_prefix: nil,
          php_generic_services: nil,
          php_metadata_namespace: nil,
          php_namespace: nil,
          py_generic_services: nil,
          ruby_package: nil,
          swift_prefix: nil,
          uninterpreted_option: []
        }

        case function_exported?(Google.Protobuf.FileOptions, :decode!, 1) do
          true ->
            bytes =
              file_options
              |> Protox.Google.Protobuf.FileOptions.encode!()
              |> :binary.list_to_bin()

            apply(Google.Protobuf.FileOptions, :decode!, [bytes])

          false ->
            file_options
        end
      end
    )
  end,
  defmodule Transmuter.Onnx.NodeProto do
    @moduledoc false
    defstruct input: [],
              output: [],
              name: nil,
              op_type: nil,
              attribute: [],
              doc_string: nil,
              domain: nil,
              overload: nil,
              metadata_props: [],
              device_configurations: [],
              __uf__: []

    (
      (
        @spec encode(struct) :: {:ok, iodata} | {:error, any}
        def encode(msg) do
          try do
            {:ok, encode!(msg)}
          rescue
            e in [Protox.EncodingError, Protox.RequiredFieldsError] -> {:error, e}
          end
        end

        @spec encode!(struct) :: iodata | no_return
        def encode!(msg) do
          []
          |> encode_input(msg)
          |> encode_output(msg)
          |> encode_name(msg)
          |> encode_op_type(msg)
          |> encode_attribute(msg)
          |> encode_doc_string(msg)
          |> encode_domain(msg)
          |> encode_overload(msg)
          |> encode_metadata_props(msg)
          |> encode_device_configurations(msg)
          |> encode_unknown_fields(msg)
        end
      )

      []

      [
        defp encode_input(acc, msg) do
          try do
            case msg.input do
              [] ->
                acc

              values ->
                [
                  acc,
                  Enum.reduce(values, [], fn value, acc ->
                    [acc, "\n", Protox.Encode.encode_string(value)]
                  end)
                ]
            end
          rescue
            ArgumentError ->
              reraise Protox.EncodingError.new(:input, "invalid field value"), __STACKTRACE__
          end
        end,
        defp encode_output(acc, msg) do
          try do
            case msg.output do
              [] ->
                acc

              values ->
                [
                  acc,
                  Enum.reduce(values, [], fn value, acc ->
                    [acc, "\x12", Protox.Encode.encode_string(value)]
                  end)
                ]
            end
          rescue
            ArgumentError ->
              reraise Protox.EncodingError.new(:output, "invalid field value"), __STACKTRACE__
          end
        end,
        defp encode_name(acc, msg) do
          try do
            case msg.name do
              nil -> acc
              _ -> [acc, "\x1A", Protox.Encode.encode_string(msg.name)]
            end
          rescue
            ArgumentError ->
              reraise Protox.EncodingError.new(:name, "invalid field value"), __STACKTRACE__
          end
        end,
        defp encode_op_type(acc, msg) do
          try do
            case msg.op_type do
              nil -> acc
              _ -> [acc, "\"", Protox.Encode.encode_string(msg.op_type)]
            end
          rescue
            ArgumentError ->
              reraise Protox.EncodingError.new(:op_type, "invalid field value"), __STACKTRACE__
          end
        end,
        defp encode_attribute(acc, msg) do
          try do
            case msg.attribute do
              [] ->
                acc

              values ->
                [
                  acc,
                  Enum.reduce(values, [], fn value, acc ->
                    [acc, "*", Protox.Encode.encode_message(value)]
                  end)
                ]
            end
          rescue
            ArgumentError ->
              reraise Protox.EncodingError.new(:attribute, "invalid field value"), __STACKTRACE__
          end
        end,
        defp encode_doc_string(acc, msg) do
          try do
            case msg.doc_string do
              nil -> acc
              _ -> [acc, "2", Protox.Encode.encode_string(msg.doc_string)]
            end
          rescue
            ArgumentError ->
              reraise Protox.EncodingError.new(:doc_string, "invalid field value"), __STACKTRACE__
          end
        end,
        defp encode_domain(acc, msg) do
          try do
            case msg.domain do
              nil -> acc
              _ -> [acc, ":", Protox.Encode.encode_string(msg.domain)]
            end
          rescue
            ArgumentError ->
              reraise Protox.EncodingError.new(:domain, "invalid field value"), __STACKTRACE__
          end
        end,
        defp encode_overload(acc, msg) do
          try do
            case msg.overload do
              nil -> acc
              _ -> [acc, "B", Protox.Encode.encode_string(msg.overload)]
            end
          rescue
            ArgumentError ->
              reraise Protox.EncodingError.new(:overload, "invalid field value"), __STACKTRACE__
          end
        end,
        defp encode_metadata_props(acc, msg) do
          try do
            case msg.metadata_props do
              [] ->
                acc

              values ->
                [
                  acc,
                  Enum.reduce(values, [], fn value, acc ->
                    [acc, "J", Protox.Encode.encode_message(value)]
                  end)
                ]
            end
          rescue
            ArgumentError ->
              reraise Protox.EncodingError.new(:metadata_props, "invalid field value"),
                      __STACKTRACE__
          end
        end,
        defp encode_device_configurations(acc, msg) do
          try do
            case msg.device_configurations do
              [] ->
                acc

              values ->
                [
                  acc,
                  Enum.reduce(values, [], fn value, acc ->
                    [acc, "R", Protox.Encode.encode_message(value)]
                  end)
                ]
            end
          rescue
            ArgumentError ->
              reraise Protox.EncodingError.new(:device_configurations, "invalid field value"),
                      __STACKTRACE__
          end
        end
      ]

      defp encode_unknown_fields(acc, msg) do
        Enum.reduce(msg.__struct__.unknown_fields(msg), acc, fn {tag, wire_type, bytes}, acc ->
          case wire_type do
            0 ->
              [acc, Protox.Encode.make_key_bytes(tag, :int32), bytes]

            1 ->
              [acc, Protox.Encode.make_key_bytes(tag, :double), bytes]

            2 ->
              len_bytes = bytes |> byte_size() |> Protox.Varint.encode()
              [acc, Protox.Encode.make_key_bytes(tag, :packed), len_bytes, bytes]

            5 ->
              [acc, Protox.Encode.make_key_bytes(tag, :float), bytes]
          end
        end)
      end
    )

    (
      (
        @spec decode(binary) :: {:ok, struct} | {:error, any}
        def decode(bytes) do
          try do
            {:ok, decode!(bytes)}
          rescue
            e in [Protox.DecodingError, Protox.IllegalTagError, Protox.RequiredFieldsError] ->
              {:error, e}
          end
        end

        (
          @spec decode!(binary) :: struct | no_return
          def decode!(bytes) do
            parse_key_value(bytes, struct(Transmuter.Onnx.NodeProto))
          end
        )
      )

      (
        @spec parse_key_value(binary, struct) :: struct
        defp parse_key_value(<<>>, msg) do
          msg
        end

        defp parse_key_value(bytes, msg) do
          {field, rest} =
            case Protox.Decode.parse_key(bytes) do
              {0, _, _} ->
                raise %Protox.IllegalTagError{}

              {1, _, bytes} ->
                {len, bytes} = Protox.Varint.decode(bytes)
                {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)
                {[input: msg.input ++ [Protox.Decode.validate_string!(delimited)]], rest}

              {2, _, bytes} ->
                {len, bytes} = Protox.Varint.decode(bytes)
                {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)
                {[output: msg.output ++ [Protox.Decode.validate_string!(delimited)]], rest}

              {3, _, bytes} ->
                {len, bytes} = Protox.Varint.decode(bytes)
                {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)
                {[name: Protox.Decode.validate_string!(delimited)], rest}

              {4, _, bytes} ->
                {len, bytes} = Protox.Varint.decode(bytes)
                {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)
                {[op_type: Protox.Decode.validate_string!(delimited)], rest}

              {5, _, bytes} ->
                {len, bytes} = Protox.Varint.decode(bytes)
                {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)

                {[
                   attribute: msg.attribute ++ [Transmuter.Onnx.AttributeProto.decode!(delimited)]
                 ], rest}

              {6, _, bytes} ->
                {len, bytes} = Protox.Varint.decode(bytes)
                {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)
                {[doc_string: Protox.Decode.validate_string!(delimited)], rest}

              {7, _, bytes} ->
                {len, bytes} = Protox.Varint.decode(bytes)
                {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)
                {[domain: Protox.Decode.validate_string!(delimited)], rest}

              {8, _, bytes} ->
                {len, bytes} = Protox.Varint.decode(bytes)
                {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)
                {[overload: Protox.Decode.validate_string!(delimited)], rest}

              {9, _, bytes} ->
                {len, bytes} = Protox.Varint.decode(bytes)
                {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)

                {[
                   metadata_props:
                     msg.metadata_props ++
                       [Transmuter.Onnx.StringStringEntryProto.decode!(delimited)]
                 ], rest}

              {10, _, bytes} ->
                {len, bytes} = Protox.Varint.decode(bytes)
                {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)

                {[
                   device_configurations:
                     msg.device_configurations ++
                       [Transmuter.Onnx.NodeDeviceConfigurationProto.decode!(delimited)]
                 ], rest}

              {tag, wire_type, rest} ->
                {value, rest} = Protox.Decode.parse_unknown(tag, wire_type, rest)

                {[
                   {msg.__struct__.unknown_fields_name(),
                    [value | msg.__struct__.unknown_fields(msg)]}
                 ], rest}
            end

          msg_updated = struct(msg, field)
          parse_key_value(rest, msg_updated)
        end
      )

      []
    )

    (
      @spec json_decode(iodata(), keyword()) :: {:ok, struct()} | {:error, any()}
      def json_decode(input, opts \\ []) do
        try do
          {:ok, json_decode!(input, opts)}
        rescue
          e in Protox.JsonDecodingError -> {:error, e}
        end
      end

      @spec json_decode!(iodata(), keyword()) :: struct() | no_return()
      def json_decode!(input, opts \\ []) do
        {json_library_wrapper, json_library} = Protox.JsonLibrary.get_library(opts, :decode)

        Protox.JsonDecode.decode!(
          input,
          Transmuter.Onnx.NodeProto,
          &json_library_wrapper.decode!(json_library, &1)
        )
      end

      @spec json_encode(struct(), keyword()) :: {:ok, iodata()} | {:error, any()}
      def json_encode(msg, opts \\ []) do
        try do
          {:ok, json_encode!(msg, opts)}
        rescue
          e in Protox.JsonEncodingError -> {:error, e}
        end
      end

      @spec json_encode!(struct(), keyword()) :: iodata() | no_return()
      def json_encode!(msg, opts \\ []) do
        {json_library_wrapper, json_library} = Protox.JsonLibrary.get_library(opts, :encode)
        Protox.JsonEncode.encode!(msg, &json_library_wrapper.encode!(json_library, &1))
      end
    )

    (
      @deprecated "Use fields_defs()/0 instead"
      @spec defs() :: %{
              required(non_neg_integer) => {atom, Protox.Types.kind(), Protox.Types.type()}
            }
      def defs() do
        %{
          1 => {:input, :unpacked, :string},
          2 => {:output, :unpacked, :string},
          3 => {:name, {:scalar, ""}, :string},
          4 => {:op_type, {:scalar, ""}, :string},
          5 => {:attribute, :unpacked, {:message, Transmuter.Onnx.AttributeProto}},
          6 => {:doc_string, {:scalar, ""}, :string},
          7 => {:domain, {:scalar, ""}, :string},
          8 => {:overload, {:scalar, ""}, :string},
          9 => {:metadata_props, :unpacked, {:message, Transmuter.Onnx.StringStringEntryProto}},
          10 =>
            {:device_configurations, :unpacked,
             {:message, Transmuter.Onnx.NodeDeviceConfigurationProto}}
        }
      end

      @deprecated "Use fields_defs()/0 instead"
      @spec defs_by_name() :: %{
              required(atom) => {non_neg_integer, Protox.Types.kind(), Protox.Types.type()}
            }
      def defs_by_name() do
        %{
          attribute: {5, :unpacked, {:message, Transmuter.Onnx.AttributeProto}},
          device_configurations:
            {10, :unpacked, {:message, Transmuter.Onnx.NodeDeviceConfigurationProto}},
          doc_string: {6, {:scalar, ""}, :string},
          domain: {7, {:scalar, ""}, :string},
          input: {1, :unpacked, :string},
          metadata_props: {9, :unpacked, {:message, Transmuter.Onnx.StringStringEntryProto}},
          name: {3, {:scalar, ""}, :string},
          op_type: {4, {:scalar, ""}, :string},
          output: {2, :unpacked, :string},
          overload: {8, {:scalar, ""}, :string}
        }
      end
    )

    (
      @spec fields_defs() :: list(Protox.Field.t())
      def fields_defs() do
        [
          %{
            __struct__: Protox.Field,
            json_name: "input",
            kind: :unpacked,
            label: :repeated,
            name: :input,
            tag: 1,
            type: :string
          },
          %{
            __struct__: Protox.Field,
            json_name: "output",
            kind: :unpacked,
            label: :repeated,
            name: :output,
            tag: 2,
            type: :string
          },
          %{
            __struct__: Protox.Field,
            json_name: "name",
            kind: {:scalar, ""},
            label: :optional,
            name: :name,
            tag: 3,
            type: :string
          },
          %{
            __struct__: Protox.Field,
            json_name: "opType",
            kind: {:scalar, ""},
            label: :optional,
            name: :op_type,
            tag: 4,
            type: :string
          },
          %{
            __struct__: Protox.Field,
            json_name: "attribute",
            kind: :unpacked,
            label: :repeated,
            name: :attribute,
            tag: 5,
            type: {:message, Transmuter.Onnx.AttributeProto}
          },
          %{
            __struct__: Protox.Field,
            json_name: "docString",
            kind: {:scalar, ""},
            label: :optional,
            name: :doc_string,
            tag: 6,
            type: :string
          },
          %{
            __struct__: Protox.Field,
            json_name: "domain",
            kind: {:scalar, ""},
            label: :optional,
            name: :domain,
            tag: 7,
            type: :string
          },
          %{
            __struct__: Protox.Field,
            json_name: "overload",
            kind: {:scalar, ""},
            label: :optional,
            name: :overload,
            tag: 8,
            type: :string
          },
          %{
            __struct__: Protox.Field,
            json_name: "metadataProps",
            kind: :unpacked,
            label: :repeated,
            name: :metadata_props,
            tag: 9,
            type: {:message, Transmuter.Onnx.StringStringEntryProto}
          },
          %{
            __struct__: Protox.Field,
            json_name: "deviceConfigurations",
            kind: :unpacked,
            label: :repeated,
            name: :device_configurations,
            tag: 10,
            type: {:message, Transmuter.Onnx.NodeDeviceConfigurationProto}
          }
        ]
      end

      [
        @spec(field_def(atom) :: {:ok, Protox.Field.t()} | {:error, :no_such_field}),
        (
          def field_def(:input) do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "input",
               kind: :unpacked,
               label: :repeated,
               name: :input,
               tag: 1,
               type: :string
             }}
          end

          def field_def("input") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "input",
               kind: :unpacked,
               label: :repeated,
               name: :input,
               tag: 1,
               type: :string
             }}
          end

          []
        ),
        (
          def field_def(:output) do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "output",
               kind: :unpacked,
               label: :repeated,
               name: :output,
               tag: 2,
               type: :string
             }}
          end

          def field_def("output") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "output",
               kind: :unpacked,
               label: :repeated,
               name: :output,
               tag: 2,
               type: :string
             }}
          end

          []
        ),
        (
          def field_def(:name) do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "name",
               kind: {:scalar, ""},
               label: :optional,
               name: :name,
               tag: 3,
               type: :string
             }}
          end

          def field_def("name") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "name",
               kind: {:scalar, ""},
               label: :optional,
               name: :name,
               tag: 3,
               type: :string
             }}
          end

          []
        ),
        (
          def field_def(:op_type) do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "opType",
               kind: {:scalar, ""},
               label: :optional,
               name: :op_type,
               tag: 4,
               type: :string
             }}
          end

          def field_def("opType") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "opType",
               kind: {:scalar, ""},
               label: :optional,
               name: :op_type,
               tag: 4,
               type: :string
             }}
          end

          def field_def("op_type") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "opType",
               kind: {:scalar, ""},
               label: :optional,
               name: :op_type,
               tag: 4,
               type: :string
             }}
          end
        ),
        (
          def field_def(:attribute) do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "attribute",
               kind: :unpacked,
               label: :repeated,
               name: :attribute,
               tag: 5,
               type: {:message, Transmuter.Onnx.AttributeProto}
             }}
          end

          def field_def("attribute") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "attribute",
               kind: :unpacked,
               label: :repeated,
               name: :attribute,
               tag: 5,
               type: {:message, Transmuter.Onnx.AttributeProto}
             }}
          end

          []
        ),
        (
          def field_def(:doc_string) do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "docString",
               kind: {:scalar, ""},
               label: :optional,
               name: :doc_string,
               tag: 6,
               type: :string
             }}
          end

          def field_def("docString") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "docString",
               kind: {:scalar, ""},
               label: :optional,
               name: :doc_string,
               tag: 6,
               type: :string
             }}
          end

          def field_def("doc_string") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "docString",
               kind: {:scalar, ""},
               label: :optional,
               name: :doc_string,
               tag: 6,
               type: :string
             }}
          end
        ),
        (
          def field_def(:domain) do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "domain",
               kind: {:scalar, ""},
               label: :optional,
               name: :domain,
               tag: 7,
               type: :string
             }}
          end

          def field_def("domain") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "domain",
               kind: {:scalar, ""},
               label: :optional,
               name: :domain,
               tag: 7,
               type: :string
             }}
          end

          []
        ),
        (
          def field_def(:overload) do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "overload",
               kind: {:scalar, ""},
               label: :optional,
               name: :overload,
               tag: 8,
               type: :string
             }}
          end

          def field_def("overload") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "overload",
               kind: {:scalar, ""},
               label: :optional,
               name: :overload,
               tag: 8,
               type: :string
             }}
          end

          []
        ),
        (
          def field_def(:metadata_props) do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "metadataProps",
               kind: :unpacked,
               label: :repeated,
               name: :metadata_props,
               tag: 9,
               type: {:message, Transmuter.Onnx.StringStringEntryProto}
             }}
          end

          def field_def("metadataProps") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "metadataProps",
               kind: :unpacked,
               label: :repeated,
               name: :metadata_props,
               tag: 9,
               type: {:message, Transmuter.Onnx.StringStringEntryProto}
             }}
          end

          def field_def("metadata_props") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "metadataProps",
               kind: :unpacked,
               label: :repeated,
               name: :metadata_props,
               tag: 9,
               type: {:message, Transmuter.Onnx.StringStringEntryProto}
             }}
          end
        ),
        (
          def field_def(:device_configurations) do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "deviceConfigurations",
               kind: :unpacked,
               label: :repeated,
               name: :device_configurations,
               tag: 10,
               type: {:message, Transmuter.Onnx.NodeDeviceConfigurationProto}
             }}
          end

          def field_def("deviceConfigurations") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "deviceConfigurations",
               kind: :unpacked,
               label: :repeated,
               name: :device_configurations,
               tag: 10,
               type: {:message, Transmuter.Onnx.NodeDeviceConfigurationProto}
             }}
          end

          def field_def("device_configurations") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "deviceConfigurations",
               kind: :unpacked,
               label: :repeated,
               name: :device_configurations,
               tag: 10,
               type: {:message, Transmuter.Onnx.NodeDeviceConfigurationProto}
             }}
          end
        ),
        def field_def(_) do
          {:error, :no_such_field}
        end
      ]
    )

    (
      @spec unknown_fields(struct) :: [{non_neg_integer, Protox.Types.tag(), binary}]
      def unknown_fields(msg) do
        msg.__uf__
      end

      @spec unknown_fields_name() :: :__uf__
      def unknown_fields_name() do
        :__uf__
      end

      @spec clear_unknown_fields(struct) :: struct
      def clear_unknown_fields(msg) do
        struct!(msg, [{unknown_fields_name(), []}])
      end
    )

    (
      @spec required_fields() :: []
      def required_fields() do
        []
      end
    )

    (
      @spec syntax() :: atom()
      def syntax() do
        :proto2
      end
    )

    [
      @spec(default(atom) :: {:ok, boolean | integer | String.t() | float} | {:error, atom}),
      def default(:input) do
        {:error, :no_default_value}
      end,
      def default(:output) do
        {:error, :no_default_value}
      end,
      def default(:name) do
        {:ok, ""}
      end,
      def default(:op_type) do
        {:ok, ""}
      end,
      def default(:attribute) do
        {:error, :no_default_value}
      end,
      def default(:doc_string) do
        {:ok, ""}
      end,
      def default(:domain) do
        {:ok, ""}
      end,
      def default(:overload) do
        {:ok, ""}
      end,
      def default(:metadata_props) do
        {:error, :no_default_value}
      end,
      def default(:device_configurations) do
        {:error, :no_default_value}
      end,
      def default(_) do
        {:error, :no_such_field}
      end
    ]

    (
      @spec file_options() :: struct()
      def file_options() do
        file_options = %{
          __struct__: Protox.Google.Protobuf.FileOptions,
          __uf__: [],
          cc_enable_arenas: nil,
          cc_generic_services: nil,
          csharp_namespace: nil,
          deprecated: nil,
          go_package: nil,
          java_generate_equals_and_hash: nil,
          java_generic_services: nil,
          java_multiple_files: nil,
          java_outer_classname: nil,
          java_package: nil,
          java_string_check_utf8: nil,
          objc_class_prefix: nil,
          optimize_for: :LITE_RUNTIME,
          php_class_prefix: nil,
          php_generic_services: nil,
          php_metadata_namespace: nil,
          php_namespace: nil,
          py_generic_services: nil,
          ruby_package: nil,
          swift_prefix: nil,
          uninterpreted_option: []
        }

        case function_exported?(Google.Protobuf.FileOptions, :decode!, 1) do
          true ->
            bytes =
              file_options
              |> Protox.Google.Protobuf.FileOptions.encode!()
              |> :binary.list_to_bin()

            apply(Google.Protobuf.FileOptions, :decode!, [bytes])

          false ->
            file_options
        end
      end
    )
  end,
  defmodule Transmuter.Onnx.OperatorSetIdProto do
    @moduledoc false
    defstruct domain: nil, version: nil, __uf__: []

    (
      (
        @spec encode(struct) :: {:ok, iodata} | {:error, any}
        def encode(msg) do
          try do
            {:ok, encode!(msg)}
          rescue
            e in [Protox.EncodingError, Protox.RequiredFieldsError] -> {:error, e}
          end
        end

        @spec encode!(struct) :: iodata | no_return
        def encode!(msg) do
          [] |> encode_domain(msg) |> encode_version(msg) |> encode_unknown_fields(msg)
        end
      )

      []

      [
        defp encode_domain(acc, msg) do
          try do
            case msg.domain do
              nil -> acc
              _ -> [acc, "\n", Protox.Encode.encode_string(msg.domain)]
            end
          rescue
            ArgumentError ->
              reraise Protox.EncodingError.new(:domain, "invalid field value"), __STACKTRACE__
          end
        end,
        defp encode_version(acc, msg) do
          try do
            case msg.version do
              nil -> acc
              _ -> [acc, "\x10", Protox.Encode.encode_int64(msg.version)]
            end
          rescue
            ArgumentError ->
              reraise Protox.EncodingError.new(:version, "invalid field value"), __STACKTRACE__
          end
        end
      ]

      defp encode_unknown_fields(acc, msg) do
        Enum.reduce(msg.__struct__.unknown_fields(msg), acc, fn {tag, wire_type, bytes}, acc ->
          case wire_type do
            0 ->
              [acc, Protox.Encode.make_key_bytes(tag, :int32), bytes]

            1 ->
              [acc, Protox.Encode.make_key_bytes(tag, :double), bytes]

            2 ->
              len_bytes = bytes |> byte_size() |> Protox.Varint.encode()
              [acc, Protox.Encode.make_key_bytes(tag, :packed), len_bytes, bytes]

            5 ->
              [acc, Protox.Encode.make_key_bytes(tag, :float), bytes]
          end
        end)
      end
    )

    (
      (
        @spec decode(binary) :: {:ok, struct} | {:error, any}
        def decode(bytes) do
          try do
            {:ok, decode!(bytes)}
          rescue
            e in [Protox.DecodingError, Protox.IllegalTagError, Protox.RequiredFieldsError] ->
              {:error, e}
          end
        end

        (
          @spec decode!(binary) :: struct | no_return
          def decode!(bytes) do
            parse_key_value(bytes, struct(Transmuter.Onnx.OperatorSetIdProto))
          end
        )
      )

      (
        @spec parse_key_value(binary, struct) :: struct
        defp parse_key_value(<<>>, msg) do
          msg
        end

        defp parse_key_value(bytes, msg) do
          {field, rest} =
            case Protox.Decode.parse_key(bytes) do
              {0, _, _} ->
                raise %Protox.IllegalTagError{}

              {1, _, bytes} ->
                {len, bytes} = Protox.Varint.decode(bytes)
                {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)
                {[domain: Protox.Decode.validate_string!(delimited)], rest}

              {2, _, bytes} ->
                {value, rest} = Protox.Decode.parse_int64(bytes)
                {[version: value], rest}

              {tag, wire_type, rest} ->
                {value, rest} = Protox.Decode.parse_unknown(tag, wire_type, rest)

                {[
                   {msg.__struct__.unknown_fields_name(),
                    [value | msg.__struct__.unknown_fields(msg)]}
                 ], rest}
            end

          msg_updated = struct(msg, field)
          parse_key_value(rest, msg_updated)
        end
      )

      []
    )

    (
      @spec json_decode(iodata(), keyword()) :: {:ok, struct()} | {:error, any()}
      def json_decode(input, opts \\ []) do
        try do
          {:ok, json_decode!(input, opts)}
        rescue
          e in Protox.JsonDecodingError -> {:error, e}
        end
      end

      @spec json_decode!(iodata(), keyword()) :: struct() | no_return()
      def json_decode!(input, opts \\ []) do
        {json_library_wrapper, json_library} = Protox.JsonLibrary.get_library(opts, :decode)

        Protox.JsonDecode.decode!(
          input,
          Transmuter.Onnx.OperatorSetIdProto,
          &json_library_wrapper.decode!(json_library, &1)
        )
      end

      @spec json_encode(struct(), keyword()) :: {:ok, iodata()} | {:error, any()}
      def json_encode(msg, opts \\ []) do
        try do
          {:ok, json_encode!(msg, opts)}
        rescue
          e in Protox.JsonEncodingError -> {:error, e}
        end
      end

      @spec json_encode!(struct(), keyword()) :: iodata() | no_return()
      def json_encode!(msg, opts \\ []) do
        {json_library_wrapper, json_library} = Protox.JsonLibrary.get_library(opts, :encode)
        Protox.JsonEncode.encode!(msg, &json_library_wrapper.encode!(json_library, &1))
      end
    )

    (
      @deprecated "Use fields_defs()/0 instead"
      @spec defs() :: %{
              required(non_neg_integer) => {atom, Protox.Types.kind(), Protox.Types.type()}
            }
      def defs() do
        %{1 => {:domain, {:scalar, ""}, :string}, 2 => {:version, {:scalar, 0}, :int64}}
      end

      @deprecated "Use fields_defs()/0 instead"
      @spec defs_by_name() :: %{
              required(atom) => {non_neg_integer, Protox.Types.kind(), Protox.Types.type()}
            }
      def defs_by_name() do
        %{domain: {1, {:scalar, ""}, :string}, version: {2, {:scalar, 0}, :int64}}
      end
    )

    (
      @spec fields_defs() :: list(Protox.Field.t())
      def fields_defs() do
        [
          %{
            __struct__: Protox.Field,
            json_name: "domain",
            kind: {:scalar, ""},
            label: :optional,
            name: :domain,
            tag: 1,
            type: :string
          },
          %{
            __struct__: Protox.Field,
            json_name: "version",
            kind: {:scalar, 0},
            label: :optional,
            name: :version,
            tag: 2,
            type: :int64
          }
        ]
      end

      [
        @spec(field_def(atom) :: {:ok, Protox.Field.t()} | {:error, :no_such_field}),
        (
          def field_def(:domain) do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "domain",
               kind: {:scalar, ""},
               label: :optional,
               name: :domain,
               tag: 1,
               type: :string
             }}
          end

          def field_def("domain") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "domain",
               kind: {:scalar, ""},
               label: :optional,
               name: :domain,
               tag: 1,
               type: :string
             }}
          end

          []
        ),
        (
          def field_def(:version) do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "version",
               kind: {:scalar, 0},
               label: :optional,
               name: :version,
               tag: 2,
               type: :int64
             }}
          end

          def field_def("version") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "version",
               kind: {:scalar, 0},
               label: :optional,
               name: :version,
               tag: 2,
               type: :int64
             }}
          end

          []
        ),
        def field_def(_) do
          {:error, :no_such_field}
        end
      ]
    )

    (
      @spec unknown_fields(struct) :: [{non_neg_integer, Protox.Types.tag(), binary}]
      def unknown_fields(msg) do
        msg.__uf__
      end

      @spec unknown_fields_name() :: :__uf__
      def unknown_fields_name() do
        :__uf__
      end

      @spec clear_unknown_fields(struct) :: struct
      def clear_unknown_fields(msg) do
        struct!(msg, [{unknown_fields_name(), []}])
      end
    )

    (
      @spec required_fields() :: []
      def required_fields() do
        []
      end
    )

    (
      @spec syntax() :: atom()
      def syntax() do
        :proto2
      end
    )

    [
      @spec(default(atom) :: {:ok, boolean | integer | String.t() | float} | {:error, atom}),
      def default(:domain) do
        {:ok, ""}
      end,
      def default(:version) do
        {:ok, 0}
      end,
      def default(_) do
        {:error, :no_such_field}
      end
    ]

    (
      @spec file_options() :: struct()
      def file_options() do
        file_options = %{
          __struct__: Protox.Google.Protobuf.FileOptions,
          __uf__: [],
          cc_enable_arenas: nil,
          cc_generic_services: nil,
          csharp_namespace: nil,
          deprecated: nil,
          go_package: nil,
          java_generate_equals_and_hash: nil,
          java_generic_services: nil,
          java_multiple_files: nil,
          java_outer_classname: nil,
          java_package: nil,
          java_string_check_utf8: nil,
          objc_class_prefix: nil,
          optimize_for: :LITE_RUNTIME,
          php_class_prefix: nil,
          php_generic_services: nil,
          php_metadata_namespace: nil,
          php_namespace: nil,
          py_generic_services: nil,
          ruby_package: nil,
          swift_prefix: nil,
          uninterpreted_option: []
        }

        case function_exported?(Google.Protobuf.FileOptions, :decode!, 1) do
          true ->
            bytes =
              file_options
              |> Protox.Google.Protobuf.FileOptions.encode!()
              |> :binary.list_to_bin()

            apply(Google.Protobuf.FileOptions, :decode!, [bytes])

          false ->
            file_options
        end
      end
    )
  end,
  defmodule Transmuter.Onnx.ShardedDimProto do
    @moduledoc false
    defstruct axis: nil, simple_sharding: [], __uf__: []

    (
      (
        @spec encode(struct) :: {:ok, iodata} | {:error, any}
        def encode(msg) do
          try do
            {:ok, encode!(msg)}
          rescue
            e in [Protox.EncodingError, Protox.RequiredFieldsError] -> {:error, e}
          end
        end

        @spec encode!(struct) :: iodata | no_return
        def encode!(msg) do
          [] |> encode_axis(msg) |> encode_simple_sharding(msg) |> encode_unknown_fields(msg)
        end
      )

      []

      [
        defp encode_axis(acc, msg) do
          try do
            case msg.axis do
              nil -> acc
              _ -> [acc, "\b", Protox.Encode.encode_int64(msg.axis)]
            end
          rescue
            ArgumentError ->
              reraise Protox.EncodingError.new(:axis, "invalid field value"), __STACKTRACE__
          end
        end,
        defp encode_simple_sharding(acc, msg) do
          try do
            case msg.simple_sharding do
              [] ->
                acc

              values ->
                [
                  acc,
                  Enum.reduce(values, [], fn value, acc ->
                    [acc, "\x12", Protox.Encode.encode_message(value)]
                  end)
                ]
            end
          rescue
            ArgumentError ->
              reraise Protox.EncodingError.new(:simple_sharding, "invalid field value"),
                      __STACKTRACE__
          end
        end
      ]

      defp encode_unknown_fields(acc, msg) do
        Enum.reduce(msg.__struct__.unknown_fields(msg), acc, fn {tag, wire_type, bytes}, acc ->
          case wire_type do
            0 ->
              [acc, Protox.Encode.make_key_bytes(tag, :int32), bytes]

            1 ->
              [acc, Protox.Encode.make_key_bytes(tag, :double), bytes]

            2 ->
              len_bytes = bytes |> byte_size() |> Protox.Varint.encode()
              [acc, Protox.Encode.make_key_bytes(tag, :packed), len_bytes, bytes]

            5 ->
              [acc, Protox.Encode.make_key_bytes(tag, :float), bytes]
          end
        end)
      end
    )

    (
      (
        @spec decode(binary) :: {:ok, struct} | {:error, any}
        def decode(bytes) do
          try do
            {:ok, decode!(bytes)}
          rescue
            e in [Protox.DecodingError, Protox.IllegalTagError, Protox.RequiredFieldsError] ->
              {:error, e}
          end
        end

        (
          @spec decode!(binary) :: struct | no_return
          def decode!(bytes) do
            parse_key_value(bytes, struct(Transmuter.Onnx.ShardedDimProto))
          end
        )
      )

      (
        @spec parse_key_value(binary, struct) :: struct
        defp parse_key_value(<<>>, msg) do
          msg
        end

        defp parse_key_value(bytes, msg) do
          {field, rest} =
            case Protox.Decode.parse_key(bytes) do
              {0, _, _} ->
                raise %Protox.IllegalTagError{}

              {1, _, bytes} ->
                {value, rest} = Protox.Decode.parse_int64(bytes)
                {[axis: value], rest}

              {2, _, bytes} ->
                {len, bytes} = Protox.Varint.decode(bytes)
                {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)

                {[
                   simple_sharding:
                     msg.simple_sharding ++
                       [Transmuter.Onnx.SimpleShardedDimProto.decode!(delimited)]
                 ], rest}

              {tag, wire_type, rest} ->
                {value, rest} = Protox.Decode.parse_unknown(tag, wire_type, rest)

                {[
                   {msg.__struct__.unknown_fields_name(),
                    [value | msg.__struct__.unknown_fields(msg)]}
                 ], rest}
            end

          msg_updated = struct(msg, field)
          parse_key_value(rest, msg_updated)
        end
      )

      []
    )

    (
      @spec json_decode(iodata(), keyword()) :: {:ok, struct()} | {:error, any()}
      def json_decode(input, opts \\ []) do
        try do
          {:ok, json_decode!(input, opts)}
        rescue
          e in Protox.JsonDecodingError -> {:error, e}
        end
      end

      @spec json_decode!(iodata(), keyword()) :: struct() | no_return()
      def json_decode!(input, opts \\ []) do
        {json_library_wrapper, json_library} = Protox.JsonLibrary.get_library(opts, :decode)

        Protox.JsonDecode.decode!(
          input,
          Transmuter.Onnx.ShardedDimProto,
          &json_library_wrapper.decode!(json_library, &1)
        )
      end

      @spec json_encode(struct(), keyword()) :: {:ok, iodata()} | {:error, any()}
      def json_encode(msg, opts \\ []) do
        try do
          {:ok, json_encode!(msg, opts)}
        rescue
          e in Protox.JsonEncodingError -> {:error, e}
        end
      end

      @spec json_encode!(struct(), keyword()) :: iodata() | no_return()
      def json_encode!(msg, opts \\ []) do
        {json_library_wrapper, json_library} = Protox.JsonLibrary.get_library(opts, :encode)
        Protox.JsonEncode.encode!(msg, &json_library_wrapper.encode!(json_library, &1))
      end
    )

    (
      @deprecated "Use fields_defs()/0 instead"
      @spec defs() :: %{
              required(non_neg_integer) => {atom, Protox.Types.kind(), Protox.Types.type()}
            }
      def defs() do
        %{
          1 => {:axis, {:scalar, 0}, :int64},
          2 => {:simple_sharding, :unpacked, {:message, Transmuter.Onnx.SimpleShardedDimProto}}
        }
      end

      @deprecated "Use fields_defs()/0 instead"
      @spec defs_by_name() :: %{
              required(atom) => {non_neg_integer, Protox.Types.kind(), Protox.Types.type()}
            }
      def defs_by_name() do
        %{
          axis: {1, {:scalar, 0}, :int64},
          simple_sharding: {2, :unpacked, {:message, Transmuter.Onnx.SimpleShardedDimProto}}
        }
      end
    )

    (
      @spec fields_defs() :: list(Protox.Field.t())
      def fields_defs() do
        [
          %{
            __struct__: Protox.Field,
            json_name: "axis",
            kind: {:scalar, 0},
            label: :optional,
            name: :axis,
            tag: 1,
            type: :int64
          },
          %{
            __struct__: Protox.Field,
            json_name: "simpleSharding",
            kind: :unpacked,
            label: :repeated,
            name: :simple_sharding,
            tag: 2,
            type: {:message, Transmuter.Onnx.SimpleShardedDimProto}
          }
        ]
      end

      [
        @spec(field_def(atom) :: {:ok, Protox.Field.t()} | {:error, :no_such_field}),
        (
          def field_def(:axis) do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "axis",
               kind: {:scalar, 0},
               label: :optional,
               name: :axis,
               tag: 1,
               type: :int64
             }}
          end

          def field_def("axis") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "axis",
               kind: {:scalar, 0},
               label: :optional,
               name: :axis,
               tag: 1,
               type: :int64
             }}
          end

          []
        ),
        (
          def field_def(:simple_sharding) do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "simpleSharding",
               kind: :unpacked,
               label: :repeated,
               name: :simple_sharding,
               tag: 2,
               type: {:message, Transmuter.Onnx.SimpleShardedDimProto}
             }}
          end

          def field_def("simpleSharding") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "simpleSharding",
               kind: :unpacked,
               label: :repeated,
               name: :simple_sharding,
               tag: 2,
               type: {:message, Transmuter.Onnx.SimpleShardedDimProto}
             }}
          end

          def field_def("simple_sharding") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "simpleSharding",
               kind: :unpacked,
               label: :repeated,
               name: :simple_sharding,
               tag: 2,
               type: {:message, Transmuter.Onnx.SimpleShardedDimProto}
             }}
          end
        ),
        def field_def(_) do
          {:error, :no_such_field}
        end
      ]
    )

    (
      @spec unknown_fields(struct) :: [{non_neg_integer, Protox.Types.tag(), binary}]
      def unknown_fields(msg) do
        msg.__uf__
      end

      @spec unknown_fields_name() :: :__uf__
      def unknown_fields_name() do
        :__uf__
      end

      @spec clear_unknown_fields(struct) :: struct
      def clear_unknown_fields(msg) do
        struct!(msg, [{unknown_fields_name(), []}])
      end
    )

    (
      @spec required_fields() :: []
      def required_fields() do
        []
      end
    )

    (
      @spec syntax() :: atom()
      def syntax() do
        :proto2
      end
    )

    [
      @spec(default(atom) :: {:ok, boolean | integer | String.t() | float} | {:error, atom}),
      def default(:axis) do
        {:ok, 0}
      end,
      def default(:simple_sharding) do
        {:error, :no_default_value}
      end,
      def default(_) do
        {:error, :no_such_field}
      end
    ]

    (
      @spec file_options() :: struct()
      def file_options() do
        file_options = %{
          __struct__: Protox.Google.Protobuf.FileOptions,
          __uf__: [],
          cc_enable_arenas: nil,
          cc_generic_services: nil,
          csharp_namespace: nil,
          deprecated: nil,
          go_package: nil,
          java_generate_equals_and_hash: nil,
          java_generic_services: nil,
          java_multiple_files: nil,
          java_outer_classname: nil,
          java_package: nil,
          java_string_check_utf8: nil,
          objc_class_prefix: nil,
          optimize_for: :LITE_RUNTIME,
          php_class_prefix: nil,
          php_generic_services: nil,
          php_metadata_namespace: nil,
          php_namespace: nil,
          py_generic_services: nil,
          ruby_package: nil,
          swift_prefix: nil,
          uninterpreted_option: []
        }

        case function_exported?(Google.Protobuf.FileOptions, :decode!, 1) do
          true ->
            bytes =
              file_options
              |> Protox.Google.Protobuf.FileOptions.encode!()
              |> :binary.list_to_bin()

            apply(Google.Protobuf.FileOptions, :decode!, [bytes])

          false ->
            file_options
        end
      end
    )
  end,
  defmodule Transmuter.Onnx.ShardingSpecProto do
    @moduledoc false
    defstruct tensor_name: nil,
              device: [],
              index_to_device_group_map: [],
              sharded_dim: [],
              __uf__: []

    (
      (
        @spec encode(struct) :: {:ok, iodata} | {:error, any}
        def encode(msg) do
          try do
            {:ok, encode!(msg)}
          rescue
            e in [Protox.EncodingError, Protox.RequiredFieldsError] -> {:error, e}
          end
        end

        @spec encode!(struct) :: iodata | no_return
        def encode!(msg) do
          []
          |> encode_tensor_name(msg)
          |> encode_device(msg)
          |> encode_index_to_device_group_map(msg)
          |> encode_sharded_dim(msg)
          |> encode_unknown_fields(msg)
        end
      )

      []

      [
        defp encode_tensor_name(acc, msg) do
          try do
            case msg.tensor_name do
              nil -> acc
              _ -> [acc, "\n", Protox.Encode.encode_string(msg.tensor_name)]
            end
          rescue
            ArgumentError ->
              reraise Protox.EncodingError.new(:tensor_name, "invalid field value"),
                      __STACKTRACE__
          end
        end,
        defp encode_device(acc, msg) do
          try do
            case msg.device do
              [] ->
                acc

              values ->
                [
                  acc,
                  Enum.reduce(values, [], fn value, acc ->
                    [acc, "\x10", Protox.Encode.encode_int64(value)]
                  end)
                ]
            end
          rescue
            ArgumentError ->
              reraise Protox.EncodingError.new(:device, "invalid field value"), __STACKTRACE__
          end
        end,
        defp encode_index_to_device_group_map(acc, msg) do
          try do
            case msg.index_to_device_group_map do
              [] ->
                acc

              values ->
                [
                  acc,
                  Enum.reduce(values, [], fn value, acc ->
                    [acc, "\x1A", Protox.Encode.encode_message(value)]
                  end)
                ]
            end
          rescue
            ArgumentError ->
              reraise Protox.EncodingError.new(:index_to_device_group_map, "invalid field value"),
                      __STACKTRACE__
          end
        end,
        defp encode_sharded_dim(acc, msg) do
          try do
            case msg.sharded_dim do
              [] ->
                acc

              values ->
                [
                  acc,
                  Enum.reduce(values, [], fn value, acc ->
                    [acc, "\"", Protox.Encode.encode_message(value)]
                  end)
                ]
            end
          rescue
            ArgumentError ->
              reraise Protox.EncodingError.new(:sharded_dim, "invalid field value"),
                      __STACKTRACE__
          end
        end
      ]

      defp encode_unknown_fields(acc, msg) do
        Enum.reduce(msg.__struct__.unknown_fields(msg), acc, fn {tag, wire_type, bytes}, acc ->
          case wire_type do
            0 ->
              [acc, Protox.Encode.make_key_bytes(tag, :int32), bytes]

            1 ->
              [acc, Protox.Encode.make_key_bytes(tag, :double), bytes]

            2 ->
              len_bytes = bytes |> byte_size() |> Protox.Varint.encode()
              [acc, Protox.Encode.make_key_bytes(tag, :packed), len_bytes, bytes]

            5 ->
              [acc, Protox.Encode.make_key_bytes(tag, :float), bytes]
          end
        end)
      end
    )

    (
      (
        @spec decode(binary) :: {:ok, struct} | {:error, any}
        def decode(bytes) do
          try do
            {:ok, decode!(bytes)}
          rescue
            e in [Protox.DecodingError, Protox.IllegalTagError, Protox.RequiredFieldsError] ->
              {:error, e}
          end
        end

        (
          @spec decode!(binary) :: struct | no_return
          def decode!(bytes) do
            parse_key_value(bytes, struct(Transmuter.Onnx.ShardingSpecProto))
          end
        )
      )

      (
        @spec parse_key_value(binary, struct) :: struct
        defp parse_key_value(<<>>, msg) do
          msg
        end

        defp parse_key_value(bytes, msg) do
          {field, rest} =
            case Protox.Decode.parse_key(bytes) do
              {0, _, _} ->
                raise %Protox.IllegalTagError{}

              {1, _, bytes} ->
                {len, bytes} = Protox.Varint.decode(bytes)
                {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)
                {[tensor_name: Protox.Decode.validate_string!(delimited)], rest}

              {2, 2, bytes} ->
                {len, bytes} = Protox.Varint.decode(bytes)
                {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)
                {[device: msg.device ++ Protox.Decode.parse_repeated_int64([], delimited)], rest}

              {2, _, bytes} ->
                {value, rest} = Protox.Decode.parse_int64(bytes)
                {[device: msg.device ++ [value]], rest}

              {3, _, bytes} ->
                {len, bytes} = Protox.Varint.decode(bytes)
                {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)

                {[
                   index_to_device_group_map:
                     msg.index_to_device_group_map ++
                       [Transmuter.Onnx.IntIntListEntryProto.decode!(delimited)]
                 ], rest}

              {4, _, bytes} ->
                {len, bytes} = Protox.Varint.decode(bytes)
                {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)

                {[
                   sharded_dim:
                     msg.sharded_dim ++ [Transmuter.Onnx.ShardedDimProto.decode!(delimited)]
                 ], rest}

              {tag, wire_type, rest} ->
                {value, rest} = Protox.Decode.parse_unknown(tag, wire_type, rest)

                {[
                   {msg.__struct__.unknown_fields_name(),
                    [value | msg.__struct__.unknown_fields(msg)]}
                 ], rest}
            end

          msg_updated = struct(msg, field)
          parse_key_value(rest, msg_updated)
        end
      )

      []
    )

    (
      @spec json_decode(iodata(), keyword()) :: {:ok, struct()} | {:error, any()}
      def json_decode(input, opts \\ []) do
        try do
          {:ok, json_decode!(input, opts)}
        rescue
          e in Protox.JsonDecodingError -> {:error, e}
        end
      end

      @spec json_decode!(iodata(), keyword()) :: struct() | no_return()
      def json_decode!(input, opts \\ []) do
        {json_library_wrapper, json_library} = Protox.JsonLibrary.get_library(opts, :decode)

        Protox.JsonDecode.decode!(
          input,
          Transmuter.Onnx.ShardingSpecProto,
          &json_library_wrapper.decode!(json_library, &1)
        )
      end

      @spec json_encode(struct(), keyword()) :: {:ok, iodata()} | {:error, any()}
      def json_encode(msg, opts \\ []) do
        try do
          {:ok, json_encode!(msg, opts)}
        rescue
          e in Protox.JsonEncodingError -> {:error, e}
        end
      end

      @spec json_encode!(struct(), keyword()) :: iodata() | no_return()
      def json_encode!(msg, opts \\ []) do
        {json_library_wrapper, json_library} = Protox.JsonLibrary.get_library(opts, :encode)
        Protox.JsonEncode.encode!(msg, &json_library_wrapper.encode!(json_library, &1))
      end
    )

    (
      @deprecated "Use fields_defs()/0 instead"
      @spec defs() :: %{
              required(non_neg_integer) => {atom, Protox.Types.kind(), Protox.Types.type()}
            }
      def defs() do
        %{
          1 => {:tensor_name, {:scalar, ""}, :string},
          2 => {:device, :unpacked, :int64},
          3 =>
            {:index_to_device_group_map, :unpacked,
             {:message, Transmuter.Onnx.IntIntListEntryProto}},
          4 => {:sharded_dim, :unpacked, {:message, Transmuter.Onnx.ShardedDimProto}}
        }
      end

      @deprecated "Use fields_defs()/0 instead"
      @spec defs_by_name() :: %{
              required(atom) => {non_neg_integer, Protox.Types.kind(), Protox.Types.type()}
            }
      def defs_by_name() do
        %{
          device: {2, :unpacked, :int64},
          index_to_device_group_map:
            {3, :unpacked, {:message, Transmuter.Onnx.IntIntListEntryProto}},
          sharded_dim: {4, :unpacked, {:message, Transmuter.Onnx.ShardedDimProto}},
          tensor_name: {1, {:scalar, ""}, :string}
        }
      end
    )

    (
      @spec fields_defs() :: list(Protox.Field.t())
      def fields_defs() do
        [
          %{
            __struct__: Protox.Field,
            json_name: "tensorName",
            kind: {:scalar, ""},
            label: :optional,
            name: :tensor_name,
            tag: 1,
            type: :string
          },
          %{
            __struct__: Protox.Field,
            json_name: "device",
            kind: :unpacked,
            label: :repeated,
            name: :device,
            tag: 2,
            type: :int64
          },
          %{
            __struct__: Protox.Field,
            json_name: "indexToDeviceGroupMap",
            kind: :unpacked,
            label: :repeated,
            name: :index_to_device_group_map,
            tag: 3,
            type: {:message, Transmuter.Onnx.IntIntListEntryProto}
          },
          %{
            __struct__: Protox.Field,
            json_name: "shardedDim",
            kind: :unpacked,
            label: :repeated,
            name: :sharded_dim,
            tag: 4,
            type: {:message, Transmuter.Onnx.ShardedDimProto}
          }
        ]
      end

      [
        @spec(field_def(atom) :: {:ok, Protox.Field.t()} | {:error, :no_such_field}),
        (
          def field_def(:tensor_name) do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "tensorName",
               kind: {:scalar, ""},
               label: :optional,
               name: :tensor_name,
               tag: 1,
               type: :string
             }}
          end

          def field_def("tensorName") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "tensorName",
               kind: {:scalar, ""},
               label: :optional,
               name: :tensor_name,
               tag: 1,
               type: :string
             }}
          end

          def field_def("tensor_name") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "tensorName",
               kind: {:scalar, ""},
               label: :optional,
               name: :tensor_name,
               tag: 1,
               type: :string
             }}
          end
        ),
        (
          def field_def(:device) do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "device",
               kind: :unpacked,
               label: :repeated,
               name: :device,
               tag: 2,
               type: :int64
             }}
          end

          def field_def("device") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "device",
               kind: :unpacked,
               label: :repeated,
               name: :device,
               tag: 2,
               type: :int64
             }}
          end

          []
        ),
        (
          def field_def(:index_to_device_group_map) do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "indexToDeviceGroupMap",
               kind: :unpacked,
               label: :repeated,
               name: :index_to_device_group_map,
               tag: 3,
               type: {:message, Transmuter.Onnx.IntIntListEntryProto}
             }}
          end

          def field_def("indexToDeviceGroupMap") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "indexToDeviceGroupMap",
               kind: :unpacked,
               label: :repeated,
               name: :index_to_device_group_map,
               tag: 3,
               type: {:message, Transmuter.Onnx.IntIntListEntryProto}
             }}
          end

          def field_def("index_to_device_group_map") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "indexToDeviceGroupMap",
               kind: :unpacked,
               label: :repeated,
               name: :index_to_device_group_map,
               tag: 3,
               type: {:message, Transmuter.Onnx.IntIntListEntryProto}
             }}
          end
        ),
        (
          def field_def(:sharded_dim) do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "shardedDim",
               kind: :unpacked,
               label: :repeated,
               name: :sharded_dim,
               tag: 4,
               type: {:message, Transmuter.Onnx.ShardedDimProto}
             }}
          end

          def field_def("shardedDim") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "shardedDim",
               kind: :unpacked,
               label: :repeated,
               name: :sharded_dim,
               tag: 4,
               type: {:message, Transmuter.Onnx.ShardedDimProto}
             }}
          end

          def field_def("sharded_dim") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "shardedDim",
               kind: :unpacked,
               label: :repeated,
               name: :sharded_dim,
               tag: 4,
               type: {:message, Transmuter.Onnx.ShardedDimProto}
             }}
          end
        ),
        def field_def(_) do
          {:error, :no_such_field}
        end
      ]
    )

    (
      @spec unknown_fields(struct) :: [{non_neg_integer, Protox.Types.tag(), binary}]
      def unknown_fields(msg) do
        msg.__uf__
      end

      @spec unknown_fields_name() :: :__uf__
      def unknown_fields_name() do
        :__uf__
      end

      @spec clear_unknown_fields(struct) :: struct
      def clear_unknown_fields(msg) do
        struct!(msg, [{unknown_fields_name(), []}])
      end
    )

    (
      @spec required_fields() :: []
      def required_fields() do
        []
      end
    )

    (
      @spec syntax() :: atom()
      def syntax() do
        :proto2
      end
    )

    [
      @spec(default(atom) :: {:ok, boolean | integer | String.t() | float} | {:error, atom}),
      def default(:tensor_name) do
        {:ok, ""}
      end,
      def default(:device) do
        {:error, :no_default_value}
      end,
      def default(:index_to_device_group_map) do
        {:error, :no_default_value}
      end,
      def default(:sharded_dim) do
        {:error, :no_default_value}
      end,
      def default(_) do
        {:error, :no_such_field}
      end
    ]

    (
      @spec file_options() :: struct()
      def file_options() do
        file_options = %{
          __struct__: Protox.Google.Protobuf.FileOptions,
          __uf__: [],
          cc_enable_arenas: nil,
          cc_generic_services: nil,
          csharp_namespace: nil,
          deprecated: nil,
          go_package: nil,
          java_generate_equals_and_hash: nil,
          java_generic_services: nil,
          java_multiple_files: nil,
          java_outer_classname: nil,
          java_package: nil,
          java_string_check_utf8: nil,
          objc_class_prefix: nil,
          optimize_for: :LITE_RUNTIME,
          php_class_prefix: nil,
          php_generic_services: nil,
          php_metadata_namespace: nil,
          php_namespace: nil,
          py_generic_services: nil,
          ruby_package: nil,
          swift_prefix: nil,
          uninterpreted_option: []
        }

        case function_exported?(Google.Protobuf.FileOptions, :decode!, 1) do
          true ->
            bytes =
              file_options
              |> Protox.Google.Protobuf.FileOptions.encode!()
              |> :binary.list_to_bin()

            apply(Google.Protobuf.FileOptions, :decode!, [bytes])

          false ->
            file_options
        end
      end
    )
  end,
  defmodule Transmuter.Onnx.SimpleShardedDimProto do
    @moduledoc false
    defstruct dim: nil, num_shards: nil, __uf__: []

    (
      (
        @spec encode(struct) :: {:ok, iodata} | {:error, any}
        def encode(msg) do
          try do
            {:ok, encode!(msg)}
          rescue
            e in [Protox.EncodingError, Protox.RequiredFieldsError] -> {:error, e}
          end
        end

        @spec encode!(struct) :: iodata | no_return
        def encode!(msg) do
          [] |> encode_dim(msg) |> encode_num_shards(msg) |> encode_unknown_fields(msg)
        end
      )

      [
        defp encode_dim(acc, msg) do
          case msg.dim do
            nil -> acc
            {:dim_value, _field_value} -> encode_dim_value(acc, msg)
            {:dim_param, _field_value} -> encode_dim_param(acc, msg)
          end
        end
      ]

      [
        defp encode_dim_value(acc, msg) do
          try do
            {_, child_field_value} = msg.dim
            [acc, "\b", Protox.Encode.encode_int64(child_field_value)]
          rescue
            ArgumentError ->
              reraise Protox.EncodingError.new(:dim_value, "invalid field value"), __STACKTRACE__
          end
        end,
        defp encode_dim_param(acc, msg) do
          try do
            {_, child_field_value} = msg.dim
            [acc, "\x12", Protox.Encode.encode_string(child_field_value)]
          rescue
            ArgumentError ->
              reraise Protox.EncodingError.new(:dim_param, "invalid field value"), __STACKTRACE__
          end
        end,
        defp encode_num_shards(acc, msg) do
          try do
            case msg.num_shards do
              nil -> acc
              _ -> [acc, "\x18", Protox.Encode.encode_int64(msg.num_shards)]
            end
          rescue
            ArgumentError ->
              reraise Protox.EncodingError.new(:num_shards, "invalid field value"), __STACKTRACE__
          end
        end
      ]

      defp encode_unknown_fields(acc, msg) do
        Enum.reduce(msg.__struct__.unknown_fields(msg), acc, fn {tag, wire_type, bytes}, acc ->
          case wire_type do
            0 ->
              [acc, Protox.Encode.make_key_bytes(tag, :int32), bytes]

            1 ->
              [acc, Protox.Encode.make_key_bytes(tag, :double), bytes]

            2 ->
              len_bytes = bytes |> byte_size() |> Protox.Varint.encode()
              [acc, Protox.Encode.make_key_bytes(tag, :packed), len_bytes, bytes]

            5 ->
              [acc, Protox.Encode.make_key_bytes(tag, :float), bytes]
          end
        end)
      end
    )

    (
      (
        @spec decode(binary) :: {:ok, struct} | {:error, any}
        def decode(bytes) do
          try do
            {:ok, decode!(bytes)}
          rescue
            e in [Protox.DecodingError, Protox.IllegalTagError, Protox.RequiredFieldsError] ->
              {:error, e}
          end
        end

        (
          @spec decode!(binary) :: struct | no_return
          def decode!(bytes) do
            parse_key_value(bytes, struct(Transmuter.Onnx.SimpleShardedDimProto))
          end
        )
      )

      (
        @spec parse_key_value(binary, struct) :: struct
        defp parse_key_value(<<>>, msg) do
          msg
        end

        defp parse_key_value(bytes, msg) do
          {field, rest} =
            case Protox.Decode.parse_key(bytes) do
              {0, _, _} ->
                raise %Protox.IllegalTagError{}

              {1, _, bytes} ->
                {value, rest} = Protox.Decode.parse_int64(bytes)
                {[dim: {:dim_value, value}], rest}

              {2, _, bytes} ->
                {len, bytes} = Protox.Varint.decode(bytes)
                {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)
                {[dim: {:dim_param, Protox.Decode.validate_string!(delimited)}], rest}

              {3, _, bytes} ->
                {value, rest} = Protox.Decode.parse_int64(bytes)
                {[num_shards: value], rest}

              {tag, wire_type, rest} ->
                {value, rest} = Protox.Decode.parse_unknown(tag, wire_type, rest)

                {[
                   {msg.__struct__.unknown_fields_name(),
                    [value | msg.__struct__.unknown_fields(msg)]}
                 ], rest}
            end

          msg_updated = struct(msg, field)
          parse_key_value(rest, msg_updated)
        end
      )

      []
    )

    (
      @spec json_decode(iodata(), keyword()) :: {:ok, struct()} | {:error, any()}
      def json_decode(input, opts \\ []) do
        try do
          {:ok, json_decode!(input, opts)}
        rescue
          e in Protox.JsonDecodingError -> {:error, e}
        end
      end

      @spec json_decode!(iodata(), keyword()) :: struct() | no_return()
      def json_decode!(input, opts \\ []) do
        {json_library_wrapper, json_library} = Protox.JsonLibrary.get_library(opts, :decode)

        Protox.JsonDecode.decode!(
          input,
          Transmuter.Onnx.SimpleShardedDimProto,
          &json_library_wrapper.decode!(json_library, &1)
        )
      end

      @spec json_encode(struct(), keyword()) :: {:ok, iodata()} | {:error, any()}
      def json_encode(msg, opts \\ []) do
        try do
          {:ok, json_encode!(msg, opts)}
        rescue
          e in Protox.JsonEncodingError -> {:error, e}
        end
      end

      @spec json_encode!(struct(), keyword()) :: iodata() | no_return()
      def json_encode!(msg, opts \\ []) do
        {json_library_wrapper, json_library} = Protox.JsonLibrary.get_library(opts, :encode)
        Protox.JsonEncode.encode!(msg, &json_library_wrapper.encode!(json_library, &1))
      end
    )

    (
      @deprecated "Use fields_defs()/0 instead"
      @spec defs() :: %{
              required(non_neg_integer) => {atom, Protox.Types.kind(), Protox.Types.type()}
            }
      def defs() do
        %{
          1 => {:dim_value, {:oneof, :dim}, :int64},
          2 => {:dim_param, {:oneof, :dim}, :string},
          3 => {:num_shards, {:scalar, 0}, :int64}
        }
      end

      @deprecated "Use fields_defs()/0 instead"
      @spec defs_by_name() :: %{
              required(atom) => {non_neg_integer, Protox.Types.kind(), Protox.Types.type()}
            }
      def defs_by_name() do
        %{
          dim_param: {2, {:oneof, :dim}, :string},
          dim_value: {1, {:oneof, :dim}, :int64},
          num_shards: {3, {:scalar, 0}, :int64}
        }
      end
    )

    (
      @spec fields_defs() :: list(Protox.Field.t())
      def fields_defs() do
        [
          %{
            __struct__: Protox.Field,
            json_name: "dimValue",
            kind: {:oneof, :dim},
            label: :optional,
            name: :dim_value,
            tag: 1,
            type: :int64
          },
          %{
            __struct__: Protox.Field,
            json_name: "dimParam",
            kind: {:oneof, :dim},
            label: :optional,
            name: :dim_param,
            tag: 2,
            type: :string
          },
          %{
            __struct__: Protox.Field,
            json_name: "numShards",
            kind: {:scalar, 0},
            label: :optional,
            name: :num_shards,
            tag: 3,
            type: :int64
          }
        ]
      end

      [
        @spec(field_def(atom) :: {:ok, Protox.Field.t()} | {:error, :no_such_field}),
        (
          def field_def(:dim_value) do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "dimValue",
               kind: {:oneof, :dim},
               label: :optional,
               name: :dim_value,
               tag: 1,
               type: :int64
             }}
          end

          def field_def("dimValue") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "dimValue",
               kind: {:oneof, :dim},
               label: :optional,
               name: :dim_value,
               tag: 1,
               type: :int64
             }}
          end

          def field_def("dim_value") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "dimValue",
               kind: {:oneof, :dim},
               label: :optional,
               name: :dim_value,
               tag: 1,
               type: :int64
             }}
          end
        ),
        (
          def field_def(:dim_param) do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "dimParam",
               kind: {:oneof, :dim},
               label: :optional,
               name: :dim_param,
               tag: 2,
               type: :string
             }}
          end

          def field_def("dimParam") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "dimParam",
               kind: {:oneof, :dim},
               label: :optional,
               name: :dim_param,
               tag: 2,
               type: :string
             }}
          end

          def field_def("dim_param") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "dimParam",
               kind: {:oneof, :dim},
               label: :optional,
               name: :dim_param,
               tag: 2,
               type: :string
             }}
          end
        ),
        (
          def field_def(:num_shards) do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "numShards",
               kind: {:scalar, 0},
               label: :optional,
               name: :num_shards,
               tag: 3,
               type: :int64
             }}
          end

          def field_def("numShards") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "numShards",
               kind: {:scalar, 0},
               label: :optional,
               name: :num_shards,
               tag: 3,
               type: :int64
             }}
          end

          def field_def("num_shards") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "numShards",
               kind: {:scalar, 0},
               label: :optional,
               name: :num_shards,
               tag: 3,
               type: :int64
             }}
          end
        ),
        def field_def(_) do
          {:error, :no_such_field}
        end
      ]
    )

    (
      @spec unknown_fields(struct) :: [{non_neg_integer, Protox.Types.tag(), binary}]
      def unknown_fields(msg) do
        msg.__uf__
      end

      @spec unknown_fields_name() :: :__uf__
      def unknown_fields_name() do
        :__uf__
      end

      @spec clear_unknown_fields(struct) :: struct
      def clear_unknown_fields(msg) do
        struct!(msg, [{unknown_fields_name(), []}])
      end
    )

    (
      @spec required_fields() :: []
      def required_fields() do
        []
      end
    )

    (
      @spec syntax() :: atom()
      def syntax() do
        :proto2
      end
    )

    [
      @spec(default(atom) :: {:ok, boolean | integer | String.t() | float} | {:error, atom}),
      def default(:dim_value) do
        {:error, :no_default_value}
      end,
      def default(:dim_param) do
        {:error, :no_default_value}
      end,
      def default(:num_shards) do
        {:ok, 0}
      end,
      def default(_) do
        {:error, :no_such_field}
      end
    ]

    (
      @spec file_options() :: struct()
      def file_options() do
        file_options = %{
          __struct__: Protox.Google.Protobuf.FileOptions,
          __uf__: [],
          cc_enable_arenas: nil,
          cc_generic_services: nil,
          csharp_namespace: nil,
          deprecated: nil,
          go_package: nil,
          java_generate_equals_and_hash: nil,
          java_generic_services: nil,
          java_multiple_files: nil,
          java_outer_classname: nil,
          java_package: nil,
          java_string_check_utf8: nil,
          objc_class_prefix: nil,
          optimize_for: :LITE_RUNTIME,
          php_class_prefix: nil,
          php_generic_services: nil,
          php_metadata_namespace: nil,
          php_namespace: nil,
          py_generic_services: nil,
          ruby_package: nil,
          swift_prefix: nil,
          uninterpreted_option: []
        }

        case function_exported?(Google.Protobuf.FileOptions, :decode!, 1) do
          true ->
            bytes =
              file_options
              |> Protox.Google.Protobuf.FileOptions.encode!()
              |> :binary.list_to_bin()

            apply(Google.Protobuf.FileOptions, :decode!, [bytes])

          false ->
            file_options
        end
      end
    )
  end,
  defmodule Transmuter.Onnx.SparseTensorProto do
    @moduledoc false
    defstruct values: nil, indices: nil, dims: [], __uf__: []

    (
      (
        @spec encode(struct) :: {:ok, iodata} | {:error, any}
        def encode(msg) do
          try do
            {:ok, encode!(msg)}
          rescue
            e in [Protox.EncodingError, Protox.RequiredFieldsError] -> {:error, e}
          end
        end

        @spec encode!(struct) :: iodata | no_return
        def encode!(msg) do
          []
          |> encode_values(msg)
          |> encode_indices(msg)
          |> encode_dims(msg)
          |> encode_unknown_fields(msg)
        end
      )

      []

      [
        defp encode_values(acc, msg) do
          try do
            case msg.values do
              nil -> acc
              _ -> [acc, "\n", Protox.Encode.encode_message(msg.values)]
            end
          rescue
            ArgumentError ->
              reraise Protox.EncodingError.new(:values, "invalid field value"), __STACKTRACE__
          end
        end,
        defp encode_indices(acc, msg) do
          try do
            case msg.indices do
              nil -> acc
              _ -> [acc, "\x12", Protox.Encode.encode_message(msg.indices)]
            end
          rescue
            ArgumentError ->
              reraise Protox.EncodingError.new(:indices, "invalid field value"), __STACKTRACE__
          end
        end,
        defp encode_dims(acc, msg) do
          try do
            case msg.dims do
              [] ->
                acc

              values ->
                [
                  acc,
                  Enum.reduce(values, [], fn value, acc ->
                    [acc, "\x18", Protox.Encode.encode_int64(value)]
                  end)
                ]
            end
          rescue
            ArgumentError ->
              reraise Protox.EncodingError.new(:dims, "invalid field value"), __STACKTRACE__
          end
        end
      ]

      defp encode_unknown_fields(acc, msg) do
        Enum.reduce(msg.__struct__.unknown_fields(msg), acc, fn {tag, wire_type, bytes}, acc ->
          case wire_type do
            0 ->
              [acc, Protox.Encode.make_key_bytes(tag, :int32), bytes]

            1 ->
              [acc, Protox.Encode.make_key_bytes(tag, :double), bytes]

            2 ->
              len_bytes = bytes |> byte_size() |> Protox.Varint.encode()
              [acc, Protox.Encode.make_key_bytes(tag, :packed), len_bytes, bytes]

            5 ->
              [acc, Protox.Encode.make_key_bytes(tag, :float), bytes]
          end
        end)
      end
    )

    (
      (
        @spec decode(binary) :: {:ok, struct} | {:error, any}
        def decode(bytes) do
          try do
            {:ok, decode!(bytes)}
          rescue
            e in [Protox.DecodingError, Protox.IllegalTagError, Protox.RequiredFieldsError] ->
              {:error, e}
          end
        end

        (
          @spec decode!(binary) :: struct | no_return
          def decode!(bytes) do
            parse_key_value(bytes, struct(Transmuter.Onnx.SparseTensorProto))
          end
        )
      )

      (
        @spec parse_key_value(binary, struct) :: struct
        defp parse_key_value(<<>>, msg) do
          msg
        end

        defp parse_key_value(bytes, msg) do
          {field, rest} =
            case Protox.Decode.parse_key(bytes) do
              {0, _, _} ->
                raise %Protox.IllegalTagError{}

              {1, _, bytes} ->
                {len, bytes} = Protox.Varint.decode(bytes)
                {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)

                {[
                   values:
                     Protox.MergeMessage.merge(
                       msg.values,
                       Transmuter.Onnx.TensorProto.decode!(delimited)
                     )
                 ], rest}

              {2, _, bytes} ->
                {len, bytes} = Protox.Varint.decode(bytes)
                {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)

                {[
                   indices:
                     Protox.MergeMessage.merge(
                       msg.indices,
                       Transmuter.Onnx.TensorProto.decode!(delimited)
                     )
                 ], rest}

              {3, 2, bytes} ->
                {len, bytes} = Protox.Varint.decode(bytes)
                {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)
                {[dims: msg.dims ++ Protox.Decode.parse_repeated_int64([], delimited)], rest}

              {3, _, bytes} ->
                {value, rest} = Protox.Decode.parse_int64(bytes)
                {[dims: msg.dims ++ [value]], rest}

              {tag, wire_type, rest} ->
                {value, rest} = Protox.Decode.parse_unknown(tag, wire_type, rest)

                {[
                   {msg.__struct__.unknown_fields_name(),
                    [value | msg.__struct__.unknown_fields(msg)]}
                 ], rest}
            end

          msg_updated = struct(msg, field)
          parse_key_value(rest, msg_updated)
        end
      )

      []
    )

    (
      @spec json_decode(iodata(), keyword()) :: {:ok, struct()} | {:error, any()}
      def json_decode(input, opts \\ []) do
        try do
          {:ok, json_decode!(input, opts)}
        rescue
          e in Protox.JsonDecodingError -> {:error, e}
        end
      end

      @spec json_decode!(iodata(), keyword()) :: struct() | no_return()
      def json_decode!(input, opts \\ []) do
        {json_library_wrapper, json_library} = Protox.JsonLibrary.get_library(opts, :decode)

        Protox.JsonDecode.decode!(
          input,
          Transmuter.Onnx.SparseTensorProto,
          &json_library_wrapper.decode!(json_library, &1)
        )
      end

      @spec json_encode(struct(), keyword()) :: {:ok, iodata()} | {:error, any()}
      def json_encode(msg, opts \\ []) do
        try do
          {:ok, json_encode!(msg, opts)}
        rescue
          e in Protox.JsonEncodingError -> {:error, e}
        end
      end

      @spec json_encode!(struct(), keyword()) :: iodata() | no_return()
      def json_encode!(msg, opts \\ []) do
        {json_library_wrapper, json_library} = Protox.JsonLibrary.get_library(opts, :encode)
        Protox.JsonEncode.encode!(msg, &json_library_wrapper.encode!(json_library, &1))
      end
    )

    (
      @deprecated "Use fields_defs()/0 instead"
      @spec defs() :: %{
              required(non_neg_integer) => {atom, Protox.Types.kind(), Protox.Types.type()}
            }
      def defs() do
        %{
          1 => {:values, {:scalar, nil}, {:message, Transmuter.Onnx.TensorProto}},
          2 => {:indices, {:scalar, nil}, {:message, Transmuter.Onnx.TensorProto}},
          3 => {:dims, :unpacked, :int64}
        }
      end

      @deprecated "Use fields_defs()/0 instead"
      @spec defs_by_name() :: %{
              required(atom) => {non_neg_integer, Protox.Types.kind(), Protox.Types.type()}
            }
      def defs_by_name() do
        %{
          dims: {3, :unpacked, :int64},
          indices: {2, {:scalar, nil}, {:message, Transmuter.Onnx.TensorProto}},
          values: {1, {:scalar, nil}, {:message, Transmuter.Onnx.TensorProto}}
        }
      end
    )

    (
      @spec fields_defs() :: list(Protox.Field.t())
      def fields_defs() do
        [
          %{
            __struct__: Protox.Field,
            json_name: "values",
            kind: {:scalar, nil},
            label: :optional,
            name: :values,
            tag: 1,
            type: {:message, Transmuter.Onnx.TensorProto}
          },
          %{
            __struct__: Protox.Field,
            json_name: "indices",
            kind: {:scalar, nil},
            label: :optional,
            name: :indices,
            tag: 2,
            type: {:message, Transmuter.Onnx.TensorProto}
          },
          %{
            __struct__: Protox.Field,
            json_name: "dims",
            kind: :unpacked,
            label: :repeated,
            name: :dims,
            tag: 3,
            type: :int64
          }
        ]
      end

      [
        @spec(field_def(atom) :: {:ok, Protox.Field.t()} | {:error, :no_such_field}),
        (
          def field_def(:values) do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "values",
               kind: {:scalar, nil},
               label: :optional,
               name: :values,
               tag: 1,
               type: {:message, Transmuter.Onnx.TensorProto}
             }}
          end

          def field_def("values") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "values",
               kind: {:scalar, nil},
               label: :optional,
               name: :values,
               tag: 1,
               type: {:message, Transmuter.Onnx.TensorProto}
             }}
          end

          []
        ),
        (
          def field_def(:indices) do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "indices",
               kind: {:scalar, nil},
               label: :optional,
               name: :indices,
               tag: 2,
               type: {:message, Transmuter.Onnx.TensorProto}
             }}
          end

          def field_def("indices") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "indices",
               kind: {:scalar, nil},
               label: :optional,
               name: :indices,
               tag: 2,
               type: {:message, Transmuter.Onnx.TensorProto}
             }}
          end

          []
        ),
        (
          def field_def(:dims) do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "dims",
               kind: :unpacked,
               label: :repeated,
               name: :dims,
               tag: 3,
               type: :int64
             }}
          end

          def field_def("dims") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "dims",
               kind: :unpacked,
               label: :repeated,
               name: :dims,
               tag: 3,
               type: :int64
             }}
          end

          []
        ),
        def field_def(_) do
          {:error, :no_such_field}
        end
      ]
    )

    (
      @spec unknown_fields(struct) :: [{non_neg_integer, Protox.Types.tag(), binary}]
      def unknown_fields(msg) do
        msg.__uf__
      end

      @spec unknown_fields_name() :: :__uf__
      def unknown_fields_name() do
        :__uf__
      end

      @spec clear_unknown_fields(struct) :: struct
      def clear_unknown_fields(msg) do
        struct!(msg, [{unknown_fields_name(), []}])
      end
    )

    (
      @spec required_fields() :: []
      def required_fields() do
        []
      end
    )

    (
      @spec syntax() :: atom()
      def syntax() do
        :proto2
      end
    )

    [
      @spec(default(atom) :: {:ok, boolean | integer | String.t() | float} | {:error, atom}),
      def default(:values) do
        {:ok, nil}
      end,
      def default(:indices) do
        {:ok, nil}
      end,
      def default(:dims) do
        {:error, :no_default_value}
      end,
      def default(_) do
        {:error, :no_such_field}
      end
    ]

    (
      @spec file_options() :: struct()
      def file_options() do
        file_options = %{
          __struct__: Protox.Google.Protobuf.FileOptions,
          __uf__: [],
          cc_enable_arenas: nil,
          cc_generic_services: nil,
          csharp_namespace: nil,
          deprecated: nil,
          go_package: nil,
          java_generate_equals_and_hash: nil,
          java_generic_services: nil,
          java_multiple_files: nil,
          java_outer_classname: nil,
          java_package: nil,
          java_string_check_utf8: nil,
          objc_class_prefix: nil,
          optimize_for: :LITE_RUNTIME,
          php_class_prefix: nil,
          php_generic_services: nil,
          php_metadata_namespace: nil,
          php_namespace: nil,
          py_generic_services: nil,
          ruby_package: nil,
          swift_prefix: nil,
          uninterpreted_option: []
        }

        case function_exported?(Google.Protobuf.FileOptions, :decode!, 1) do
          true ->
            bytes =
              file_options
              |> Protox.Google.Protobuf.FileOptions.encode!()
              |> :binary.list_to_bin()

            apply(Google.Protobuf.FileOptions, :decode!, [bytes])

          false ->
            file_options
        end
      end
    )
  end,
  defmodule Transmuter.Onnx.StringStringEntryProto do
    @moduledoc false
    defstruct key: nil, value: nil, __uf__: []

    (
      (
        @spec encode(struct) :: {:ok, iodata} | {:error, any}
        def encode(msg) do
          try do
            {:ok, encode!(msg)}
          rescue
            e in [Protox.EncodingError, Protox.RequiredFieldsError] -> {:error, e}
          end
        end

        @spec encode!(struct) :: iodata | no_return
        def encode!(msg) do
          [] |> encode_key(msg) |> encode_value(msg) |> encode_unknown_fields(msg)
        end
      )

      []

      [
        defp encode_key(acc, msg) do
          try do
            case msg.key do
              nil -> acc
              _ -> [acc, "\n", Protox.Encode.encode_string(msg.key)]
            end
          rescue
            ArgumentError ->
              reraise Protox.EncodingError.new(:key, "invalid field value"), __STACKTRACE__
          end
        end,
        defp encode_value(acc, msg) do
          try do
            case msg.value do
              nil -> acc
              _ -> [acc, "\x12", Protox.Encode.encode_string(msg.value)]
            end
          rescue
            ArgumentError ->
              reraise Protox.EncodingError.new(:value, "invalid field value"), __STACKTRACE__
          end
        end
      ]

      defp encode_unknown_fields(acc, msg) do
        Enum.reduce(msg.__struct__.unknown_fields(msg), acc, fn {tag, wire_type, bytes}, acc ->
          case wire_type do
            0 ->
              [acc, Protox.Encode.make_key_bytes(tag, :int32), bytes]

            1 ->
              [acc, Protox.Encode.make_key_bytes(tag, :double), bytes]

            2 ->
              len_bytes = bytes |> byte_size() |> Protox.Varint.encode()
              [acc, Protox.Encode.make_key_bytes(tag, :packed), len_bytes, bytes]

            5 ->
              [acc, Protox.Encode.make_key_bytes(tag, :float), bytes]
          end
        end)
      end
    )

    (
      (
        @spec decode(binary) :: {:ok, struct} | {:error, any}
        def decode(bytes) do
          try do
            {:ok, decode!(bytes)}
          rescue
            e in [Protox.DecodingError, Protox.IllegalTagError, Protox.RequiredFieldsError] ->
              {:error, e}
          end
        end

        (
          @spec decode!(binary) :: struct | no_return
          def decode!(bytes) do
            parse_key_value(bytes, struct(Transmuter.Onnx.StringStringEntryProto))
          end
        )
      )

      (
        @spec parse_key_value(binary, struct) :: struct
        defp parse_key_value(<<>>, msg) do
          msg
        end

        defp parse_key_value(bytes, msg) do
          {field, rest} =
            case Protox.Decode.parse_key(bytes) do
              {0, _, _} ->
                raise %Protox.IllegalTagError{}

              {1, _, bytes} ->
                {len, bytes} = Protox.Varint.decode(bytes)
                {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)
                {[key: Protox.Decode.validate_string!(delimited)], rest}

              {2, _, bytes} ->
                {len, bytes} = Protox.Varint.decode(bytes)
                {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)
                {[value: Protox.Decode.validate_string!(delimited)], rest}

              {tag, wire_type, rest} ->
                {value, rest} = Protox.Decode.parse_unknown(tag, wire_type, rest)

                {[
                   {msg.__struct__.unknown_fields_name(),
                    [value | msg.__struct__.unknown_fields(msg)]}
                 ], rest}
            end

          msg_updated = struct(msg, field)
          parse_key_value(rest, msg_updated)
        end
      )

      []
    )

    (
      @spec json_decode(iodata(), keyword()) :: {:ok, struct()} | {:error, any()}
      def json_decode(input, opts \\ []) do
        try do
          {:ok, json_decode!(input, opts)}
        rescue
          e in Protox.JsonDecodingError -> {:error, e}
        end
      end

      @spec json_decode!(iodata(), keyword()) :: struct() | no_return()
      def json_decode!(input, opts \\ []) do
        {json_library_wrapper, json_library} = Protox.JsonLibrary.get_library(opts, :decode)

        Protox.JsonDecode.decode!(
          input,
          Transmuter.Onnx.StringStringEntryProto,
          &json_library_wrapper.decode!(json_library, &1)
        )
      end

      @spec json_encode(struct(), keyword()) :: {:ok, iodata()} | {:error, any()}
      def json_encode(msg, opts \\ []) do
        try do
          {:ok, json_encode!(msg, opts)}
        rescue
          e in Protox.JsonEncodingError -> {:error, e}
        end
      end

      @spec json_encode!(struct(), keyword()) :: iodata() | no_return()
      def json_encode!(msg, opts \\ []) do
        {json_library_wrapper, json_library} = Protox.JsonLibrary.get_library(opts, :encode)
        Protox.JsonEncode.encode!(msg, &json_library_wrapper.encode!(json_library, &1))
      end
    )

    (
      @deprecated "Use fields_defs()/0 instead"
      @spec defs() :: %{
              required(non_neg_integer) => {atom, Protox.Types.kind(), Protox.Types.type()}
            }
      def defs() do
        %{1 => {:key, {:scalar, ""}, :string}, 2 => {:value, {:scalar, ""}, :string}}
      end

      @deprecated "Use fields_defs()/0 instead"
      @spec defs_by_name() :: %{
              required(atom) => {non_neg_integer, Protox.Types.kind(), Protox.Types.type()}
            }
      def defs_by_name() do
        %{key: {1, {:scalar, ""}, :string}, value: {2, {:scalar, ""}, :string}}
      end
    )

    (
      @spec fields_defs() :: list(Protox.Field.t())
      def fields_defs() do
        [
          %{
            __struct__: Protox.Field,
            json_name: "key",
            kind: {:scalar, ""},
            label: :optional,
            name: :key,
            tag: 1,
            type: :string
          },
          %{
            __struct__: Protox.Field,
            json_name: "value",
            kind: {:scalar, ""},
            label: :optional,
            name: :value,
            tag: 2,
            type: :string
          }
        ]
      end

      [
        @spec(field_def(atom) :: {:ok, Protox.Field.t()} | {:error, :no_such_field}),
        (
          def field_def(:key) do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "key",
               kind: {:scalar, ""},
               label: :optional,
               name: :key,
               tag: 1,
               type: :string
             }}
          end

          def field_def("key") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "key",
               kind: {:scalar, ""},
               label: :optional,
               name: :key,
               tag: 1,
               type: :string
             }}
          end

          []
        ),
        (
          def field_def(:value) do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "value",
               kind: {:scalar, ""},
               label: :optional,
               name: :value,
               tag: 2,
               type: :string
             }}
          end

          def field_def("value") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "value",
               kind: {:scalar, ""},
               label: :optional,
               name: :value,
               tag: 2,
               type: :string
             }}
          end

          []
        ),
        def field_def(_) do
          {:error, :no_such_field}
        end
      ]
    )

    (
      @spec unknown_fields(struct) :: [{non_neg_integer, Protox.Types.tag(), binary}]
      def unknown_fields(msg) do
        msg.__uf__
      end

      @spec unknown_fields_name() :: :__uf__
      def unknown_fields_name() do
        :__uf__
      end

      @spec clear_unknown_fields(struct) :: struct
      def clear_unknown_fields(msg) do
        struct!(msg, [{unknown_fields_name(), []}])
      end
    )

    (
      @spec required_fields() :: []
      def required_fields() do
        []
      end
    )

    (
      @spec syntax() :: atom()
      def syntax() do
        :proto2
      end
    )

    [
      @spec(default(atom) :: {:ok, boolean | integer | String.t() | float} | {:error, atom}),
      def default(:key) do
        {:ok, ""}
      end,
      def default(:value) do
        {:ok, ""}
      end,
      def default(_) do
        {:error, :no_such_field}
      end
    ]

    (
      @spec file_options() :: struct()
      def file_options() do
        file_options = %{
          __struct__: Protox.Google.Protobuf.FileOptions,
          __uf__: [],
          cc_enable_arenas: nil,
          cc_generic_services: nil,
          csharp_namespace: nil,
          deprecated: nil,
          go_package: nil,
          java_generate_equals_and_hash: nil,
          java_generic_services: nil,
          java_multiple_files: nil,
          java_outer_classname: nil,
          java_package: nil,
          java_string_check_utf8: nil,
          objc_class_prefix: nil,
          optimize_for: :LITE_RUNTIME,
          php_class_prefix: nil,
          php_generic_services: nil,
          php_metadata_namespace: nil,
          php_namespace: nil,
          py_generic_services: nil,
          ruby_package: nil,
          swift_prefix: nil,
          uninterpreted_option: []
        }

        case function_exported?(Google.Protobuf.FileOptions, :decode!, 1) do
          true ->
            bytes =
              file_options
              |> Protox.Google.Protobuf.FileOptions.encode!()
              |> :binary.list_to_bin()

            apply(Google.Protobuf.FileOptions, :decode!, [bytes])

          false ->
            file_options
        end
      end
    )
  end,
  defmodule Transmuter.Onnx.TensorAnnotation do
    @moduledoc false
    defstruct tensor_name: nil, quant_parameter_tensor_names: [], __uf__: []

    (
      (
        @spec encode(struct) :: {:ok, iodata} | {:error, any}
        def encode(msg) do
          try do
            {:ok, encode!(msg)}
          rescue
            e in [Protox.EncodingError, Protox.RequiredFieldsError] -> {:error, e}
          end
        end

        @spec encode!(struct) :: iodata | no_return
        def encode!(msg) do
          []
          |> encode_tensor_name(msg)
          |> encode_quant_parameter_tensor_names(msg)
          |> encode_unknown_fields(msg)
        end
      )

      []

      [
        defp encode_tensor_name(acc, msg) do
          try do
            case msg.tensor_name do
              nil -> acc
              _ -> [acc, "\n", Protox.Encode.encode_string(msg.tensor_name)]
            end
          rescue
            ArgumentError ->
              reraise Protox.EncodingError.new(:tensor_name, "invalid field value"),
                      __STACKTRACE__
          end
        end,
        defp encode_quant_parameter_tensor_names(acc, msg) do
          try do
            case msg.quant_parameter_tensor_names do
              [] ->
                acc

              values ->
                [
                  acc,
                  Enum.reduce(values, [], fn value, acc ->
                    [acc, "\x12", Protox.Encode.encode_message(value)]
                  end)
                ]
            end
          rescue
            ArgumentError ->
              reraise Protox.EncodingError.new(
                        :quant_parameter_tensor_names,
                        "invalid field value"
                      ),
                      __STACKTRACE__
          end
        end
      ]

      defp encode_unknown_fields(acc, msg) do
        Enum.reduce(msg.__struct__.unknown_fields(msg), acc, fn {tag, wire_type, bytes}, acc ->
          case wire_type do
            0 ->
              [acc, Protox.Encode.make_key_bytes(tag, :int32), bytes]

            1 ->
              [acc, Protox.Encode.make_key_bytes(tag, :double), bytes]

            2 ->
              len_bytes = bytes |> byte_size() |> Protox.Varint.encode()
              [acc, Protox.Encode.make_key_bytes(tag, :packed), len_bytes, bytes]

            5 ->
              [acc, Protox.Encode.make_key_bytes(tag, :float), bytes]
          end
        end)
      end
    )

    (
      (
        @spec decode(binary) :: {:ok, struct} | {:error, any}
        def decode(bytes) do
          try do
            {:ok, decode!(bytes)}
          rescue
            e in [Protox.DecodingError, Protox.IllegalTagError, Protox.RequiredFieldsError] ->
              {:error, e}
          end
        end

        (
          @spec decode!(binary) :: struct | no_return
          def decode!(bytes) do
            parse_key_value(bytes, struct(Transmuter.Onnx.TensorAnnotation))
          end
        )
      )

      (
        @spec parse_key_value(binary, struct) :: struct
        defp parse_key_value(<<>>, msg) do
          msg
        end

        defp parse_key_value(bytes, msg) do
          {field, rest} =
            case Protox.Decode.parse_key(bytes) do
              {0, _, _} ->
                raise %Protox.IllegalTagError{}

              {1, _, bytes} ->
                {len, bytes} = Protox.Varint.decode(bytes)
                {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)
                {[tensor_name: Protox.Decode.validate_string!(delimited)], rest}

              {2, _, bytes} ->
                {len, bytes} = Protox.Varint.decode(bytes)
                {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)

                {[
                   quant_parameter_tensor_names:
                     msg.quant_parameter_tensor_names ++
                       [Transmuter.Onnx.StringStringEntryProto.decode!(delimited)]
                 ], rest}

              {tag, wire_type, rest} ->
                {value, rest} = Protox.Decode.parse_unknown(tag, wire_type, rest)

                {[
                   {msg.__struct__.unknown_fields_name(),
                    [value | msg.__struct__.unknown_fields(msg)]}
                 ], rest}
            end

          msg_updated = struct(msg, field)
          parse_key_value(rest, msg_updated)
        end
      )

      []
    )

    (
      @spec json_decode(iodata(), keyword()) :: {:ok, struct()} | {:error, any()}
      def json_decode(input, opts \\ []) do
        try do
          {:ok, json_decode!(input, opts)}
        rescue
          e in Protox.JsonDecodingError -> {:error, e}
        end
      end

      @spec json_decode!(iodata(), keyword()) :: struct() | no_return()
      def json_decode!(input, opts \\ []) do
        {json_library_wrapper, json_library} = Protox.JsonLibrary.get_library(opts, :decode)

        Protox.JsonDecode.decode!(
          input,
          Transmuter.Onnx.TensorAnnotation,
          &json_library_wrapper.decode!(json_library, &1)
        )
      end

      @spec json_encode(struct(), keyword()) :: {:ok, iodata()} | {:error, any()}
      def json_encode(msg, opts \\ []) do
        try do
          {:ok, json_encode!(msg, opts)}
        rescue
          e in Protox.JsonEncodingError -> {:error, e}
        end
      end

      @spec json_encode!(struct(), keyword()) :: iodata() | no_return()
      def json_encode!(msg, opts \\ []) do
        {json_library_wrapper, json_library} = Protox.JsonLibrary.get_library(opts, :encode)
        Protox.JsonEncode.encode!(msg, &json_library_wrapper.encode!(json_library, &1))
      end
    )

    (
      @deprecated "Use fields_defs()/0 instead"
      @spec defs() :: %{
              required(non_neg_integer) => {atom, Protox.Types.kind(), Protox.Types.type()}
            }
      def defs() do
        %{
          1 => {:tensor_name, {:scalar, ""}, :string},
          2 =>
            {:quant_parameter_tensor_names, :unpacked,
             {:message, Transmuter.Onnx.StringStringEntryProto}}
        }
      end

      @deprecated "Use fields_defs()/0 instead"
      @spec defs_by_name() :: %{
              required(atom) => {non_neg_integer, Protox.Types.kind(), Protox.Types.type()}
            }
      def defs_by_name() do
        %{
          quant_parameter_tensor_names:
            {2, :unpacked, {:message, Transmuter.Onnx.StringStringEntryProto}},
          tensor_name: {1, {:scalar, ""}, :string}
        }
      end
    )

    (
      @spec fields_defs() :: list(Protox.Field.t())
      def fields_defs() do
        [
          %{
            __struct__: Protox.Field,
            json_name: "tensorName",
            kind: {:scalar, ""},
            label: :optional,
            name: :tensor_name,
            tag: 1,
            type: :string
          },
          %{
            __struct__: Protox.Field,
            json_name: "quantParameterTensorNames",
            kind: :unpacked,
            label: :repeated,
            name: :quant_parameter_tensor_names,
            tag: 2,
            type: {:message, Transmuter.Onnx.StringStringEntryProto}
          }
        ]
      end

      [
        @spec(field_def(atom) :: {:ok, Protox.Field.t()} | {:error, :no_such_field}),
        (
          def field_def(:tensor_name) do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "tensorName",
               kind: {:scalar, ""},
               label: :optional,
               name: :tensor_name,
               tag: 1,
               type: :string
             }}
          end

          def field_def("tensorName") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "tensorName",
               kind: {:scalar, ""},
               label: :optional,
               name: :tensor_name,
               tag: 1,
               type: :string
             }}
          end

          def field_def("tensor_name") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "tensorName",
               kind: {:scalar, ""},
               label: :optional,
               name: :tensor_name,
               tag: 1,
               type: :string
             }}
          end
        ),
        (
          def field_def(:quant_parameter_tensor_names) do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "quantParameterTensorNames",
               kind: :unpacked,
               label: :repeated,
               name: :quant_parameter_tensor_names,
               tag: 2,
               type: {:message, Transmuter.Onnx.StringStringEntryProto}
             }}
          end

          def field_def("quantParameterTensorNames") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "quantParameterTensorNames",
               kind: :unpacked,
               label: :repeated,
               name: :quant_parameter_tensor_names,
               tag: 2,
               type: {:message, Transmuter.Onnx.StringStringEntryProto}
             }}
          end

          def field_def("quant_parameter_tensor_names") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "quantParameterTensorNames",
               kind: :unpacked,
               label: :repeated,
               name: :quant_parameter_tensor_names,
               tag: 2,
               type: {:message, Transmuter.Onnx.StringStringEntryProto}
             }}
          end
        ),
        def field_def(_) do
          {:error, :no_such_field}
        end
      ]
    )

    (
      @spec unknown_fields(struct) :: [{non_neg_integer, Protox.Types.tag(), binary}]
      def unknown_fields(msg) do
        msg.__uf__
      end

      @spec unknown_fields_name() :: :__uf__
      def unknown_fields_name() do
        :__uf__
      end

      @spec clear_unknown_fields(struct) :: struct
      def clear_unknown_fields(msg) do
        struct!(msg, [{unknown_fields_name(), []}])
      end
    )

    (
      @spec required_fields() :: []
      def required_fields() do
        []
      end
    )

    (
      @spec syntax() :: atom()
      def syntax() do
        :proto2
      end
    )

    [
      @spec(default(atom) :: {:ok, boolean | integer | String.t() | float} | {:error, atom}),
      def default(:tensor_name) do
        {:ok, ""}
      end,
      def default(:quant_parameter_tensor_names) do
        {:error, :no_default_value}
      end,
      def default(_) do
        {:error, :no_such_field}
      end
    ]

    (
      @spec file_options() :: struct()
      def file_options() do
        file_options = %{
          __struct__: Protox.Google.Protobuf.FileOptions,
          __uf__: [],
          cc_enable_arenas: nil,
          cc_generic_services: nil,
          csharp_namespace: nil,
          deprecated: nil,
          go_package: nil,
          java_generate_equals_and_hash: nil,
          java_generic_services: nil,
          java_multiple_files: nil,
          java_outer_classname: nil,
          java_package: nil,
          java_string_check_utf8: nil,
          objc_class_prefix: nil,
          optimize_for: :LITE_RUNTIME,
          php_class_prefix: nil,
          php_generic_services: nil,
          php_metadata_namespace: nil,
          php_namespace: nil,
          py_generic_services: nil,
          ruby_package: nil,
          swift_prefix: nil,
          uninterpreted_option: []
        }

        case function_exported?(Google.Protobuf.FileOptions, :decode!, 1) do
          true ->
            bytes =
              file_options
              |> Protox.Google.Protobuf.FileOptions.encode!()
              |> :binary.list_to_bin()

            apply(Google.Protobuf.FileOptions, :decode!, [bytes])

          false ->
            file_options
        end
      end
    )
  end,
  defmodule Transmuter.Onnx.TensorProto do
    @moduledoc false
    defstruct dims: [],
              data_type: nil,
              segment: nil,
              float_data: [],
              int32_data: [],
              string_data: [],
              int64_data: [],
              name: nil,
              raw_data: nil,
              double_data: [],
              uint64_data: [],
              doc_string: nil,
              external_data: [],
              data_location: nil,
              metadata_props: [],
              __uf__: []

    (
      (
        @spec encode(struct) :: {:ok, iodata} | {:error, any}
        def encode(msg) do
          try do
            {:ok, encode!(msg)}
          rescue
            e in [Protox.EncodingError, Protox.RequiredFieldsError] -> {:error, e}
          end
        end

        @spec encode!(struct) :: iodata | no_return
        def encode!(msg) do
          []
          |> encode_dims(msg)
          |> encode_data_type(msg)
          |> encode_segment(msg)
          |> encode_float_data(msg)
          |> encode_int32_data(msg)
          |> encode_string_data(msg)
          |> encode_int64_data(msg)
          |> encode_name(msg)
          |> encode_raw_data(msg)
          |> encode_double_data(msg)
          |> encode_uint64_data(msg)
          |> encode_doc_string(msg)
          |> encode_external_data(msg)
          |> encode_data_location(msg)
          |> encode_metadata_props(msg)
          |> encode_unknown_fields(msg)
        end
      )

      []

      [
        defp encode_dims(acc, msg) do
          try do
            case msg.dims do
              [] ->
                acc

              values ->
                [
                  acc,
                  Enum.reduce(values, [], fn value, acc ->
                    [acc, "\b", Protox.Encode.encode_int64(value)]
                  end)
                ]
            end
          rescue
            ArgumentError ->
              reraise Protox.EncodingError.new(:dims, "invalid field value"), __STACKTRACE__
          end
        end,
        defp encode_data_type(acc, msg) do
          try do
            case msg.data_type do
              nil -> acc
              _ -> [acc, "\x10", Protox.Encode.encode_int32(msg.data_type)]
            end
          rescue
            ArgumentError ->
              reraise Protox.EncodingError.new(:data_type, "invalid field value"), __STACKTRACE__
          end
        end,
        defp encode_segment(acc, msg) do
          try do
            case msg.segment do
              nil -> acc
              _ -> [acc, "\x1A", Protox.Encode.encode_message(msg.segment)]
            end
          rescue
            ArgumentError ->
              reraise Protox.EncodingError.new(:segment, "invalid field value"), __STACKTRACE__
          end
        end,
        defp encode_float_data(acc, msg) do
          try do
            case msg.float_data do
              [] ->
                acc

              values ->
                [
                  acc,
                  "\"",
                  (
                    {bytes, len} =
                      Enum.reduce(values, {[], 0}, fn value, {acc, len} ->
                        value_bytes = :binary.list_to_bin([Protox.Encode.encode_float(value)])
                        {[acc, value_bytes], len + byte_size(value_bytes)}
                      end)

                    [Protox.Varint.encode(len), bytes]
                  )
                ]
            end
          rescue
            ArgumentError ->
              reraise Protox.EncodingError.new(:float_data, "invalid field value"), __STACKTRACE__
          end
        end,
        defp encode_int32_data(acc, msg) do
          try do
            case msg.int32_data do
              [] ->
                acc

              values ->
                [
                  acc,
                  "*",
                  (
                    {bytes, len} =
                      Enum.reduce(values, {[], 0}, fn value, {acc, len} ->
                        value_bytes = :binary.list_to_bin([Protox.Encode.encode_int32(value)])
                        {[acc, value_bytes], len + byte_size(value_bytes)}
                      end)

                    [Protox.Varint.encode(len), bytes]
                  )
                ]
            end
          rescue
            ArgumentError ->
              reraise Protox.EncodingError.new(:int32_data, "invalid field value"), __STACKTRACE__
          end
        end,
        defp encode_string_data(acc, msg) do
          try do
            case msg.string_data do
              [] ->
                acc

              values ->
                [
                  acc,
                  Enum.reduce(values, [], fn value, acc ->
                    [acc, "2", Protox.Encode.encode_bytes(value)]
                  end)
                ]
            end
          rescue
            ArgumentError ->
              reraise Protox.EncodingError.new(:string_data, "invalid field value"),
                      __STACKTRACE__
          end
        end,
        defp encode_int64_data(acc, msg) do
          try do
            case msg.int64_data do
              [] ->
                acc

              values ->
                [
                  acc,
                  ":",
                  (
                    {bytes, len} =
                      Enum.reduce(values, {[], 0}, fn value, {acc, len} ->
                        value_bytes = :binary.list_to_bin([Protox.Encode.encode_int64(value)])
                        {[acc, value_bytes], len + byte_size(value_bytes)}
                      end)

                    [Protox.Varint.encode(len), bytes]
                  )
                ]
            end
          rescue
            ArgumentError ->
              reraise Protox.EncodingError.new(:int64_data, "invalid field value"), __STACKTRACE__
          end
        end,
        defp encode_name(acc, msg) do
          try do
            case msg.name do
              nil -> acc
              _ -> [acc, "B", Protox.Encode.encode_string(msg.name)]
            end
          rescue
            ArgumentError ->
              reraise Protox.EncodingError.new(:name, "invalid field value"), __STACKTRACE__
          end
        end,
        defp encode_raw_data(acc, msg) do
          try do
            case msg.raw_data do
              nil -> acc
              _ -> [acc, "J", Protox.Encode.encode_bytes(msg.raw_data)]
            end
          rescue
            ArgumentError ->
              reraise Protox.EncodingError.new(:raw_data, "invalid field value"), __STACKTRACE__
          end
        end,
        defp encode_double_data(acc, msg) do
          try do
            case msg.double_data do
              [] ->
                acc

              values ->
                [
                  acc,
                  "R",
                  (
                    {bytes, len} =
                      Enum.reduce(values, {[], 0}, fn value, {acc, len} ->
                        value_bytes = :binary.list_to_bin([Protox.Encode.encode_double(value)])
                        {[acc, value_bytes], len + byte_size(value_bytes)}
                      end)

                    [Protox.Varint.encode(len), bytes]
                  )
                ]
            end
          rescue
            ArgumentError ->
              reraise Protox.EncodingError.new(:double_data, "invalid field value"),
                      __STACKTRACE__
          end
        end,
        defp encode_uint64_data(acc, msg) do
          try do
            case msg.uint64_data do
              [] ->
                acc

              values ->
                [
                  acc,
                  "Z",
                  (
                    {bytes, len} =
                      Enum.reduce(values, {[], 0}, fn value, {acc, len} ->
                        value_bytes = :binary.list_to_bin([Protox.Encode.encode_uint64(value)])
                        {[acc, value_bytes], len + byte_size(value_bytes)}
                      end)

                    [Protox.Varint.encode(len), bytes]
                  )
                ]
            end
          rescue
            ArgumentError ->
              reraise Protox.EncodingError.new(:uint64_data, "invalid field value"),
                      __STACKTRACE__
          end
        end,
        defp encode_doc_string(acc, msg) do
          try do
            case msg.doc_string do
              nil -> acc
              _ -> [acc, "b", Protox.Encode.encode_string(msg.doc_string)]
            end
          rescue
            ArgumentError ->
              reraise Protox.EncodingError.new(:doc_string, "invalid field value"), __STACKTRACE__
          end
        end,
        defp encode_external_data(acc, msg) do
          try do
            case msg.external_data do
              [] ->
                acc

              values ->
                [
                  acc,
                  Enum.reduce(values, [], fn value, acc ->
                    [acc, "j", Protox.Encode.encode_message(value)]
                  end)
                ]
            end
          rescue
            ArgumentError ->
              reraise Protox.EncodingError.new(:external_data, "invalid field value"),
                      __STACKTRACE__
          end
        end,
        defp encode_data_location(acc, msg) do
          try do
            case msg.data_location do
              nil ->
                acc

              _ ->
                [
                  acc,
                  "p",
                  msg.data_location
                  |> Transmuter.Onnx.TensorProto.DataLocation.encode()
                  |> Protox.Encode.encode_enum()
                ]
            end
          rescue
            ArgumentError ->
              reraise Protox.EncodingError.new(:data_location, "invalid field value"),
                      __STACKTRACE__
          end
        end,
        defp encode_metadata_props(acc, msg) do
          try do
            case msg.metadata_props do
              [] ->
                acc

              values ->
                [
                  acc,
                  Enum.reduce(values, [], fn value, acc ->
                    [acc, "\x82\x01", Protox.Encode.encode_message(value)]
                  end)
                ]
            end
          rescue
            ArgumentError ->
              reraise Protox.EncodingError.new(:metadata_props, "invalid field value"),
                      __STACKTRACE__
          end
        end
      ]

      defp encode_unknown_fields(acc, msg) do
        Enum.reduce(msg.__struct__.unknown_fields(msg), acc, fn {tag, wire_type, bytes}, acc ->
          case wire_type do
            0 ->
              [acc, Protox.Encode.make_key_bytes(tag, :int32), bytes]

            1 ->
              [acc, Protox.Encode.make_key_bytes(tag, :double), bytes]

            2 ->
              len_bytes = bytes |> byte_size() |> Protox.Varint.encode()
              [acc, Protox.Encode.make_key_bytes(tag, :packed), len_bytes, bytes]

            5 ->
              [acc, Protox.Encode.make_key_bytes(tag, :float), bytes]
          end
        end)
      end
    )

    (
      (
        @spec decode(binary) :: {:ok, struct} | {:error, any}
        def decode(bytes) do
          try do
            {:ok, decode!(bytes)}
          rescue
            e in [Protox.DecodingError, Protox.IllegalTagError, Protox.RequiredFieldsError] ->
              {:error, e}
          end
        end

        (
          @spec decode!(binary) :: struct | no_return
          def decode!(bytes) do
            parse_key_value(bytes, struct(Transmuter.Onnx.TensorProto))
          end
        )
      )

      (
        @spec parse_key_value(binary, struct) :: struct
        defp parse_key_value(<<>>, msg) do
          msg
        end

        defp parse_key_value(bytes, msg) do
          {field, rest} =
            case Protox.Decode.parse_key(bytes) do
              {0, _, _} ->
                raise %Protox.IllegalTagError{}

              {1, 2, bytes} ->
                {len, bytes} = Protox.Varint.decode(bytes)
                {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)
                {[dims: msg.dims ++ Protox.Decode.parse_repeated_int64([], delimited)], rest}

              {1, _, bytes} ->
                {value, rest} = Protox.Decode.parse_int64(bytes)
                {[dims: msg.dims ++ [value]], rest}

              {2, _, bytes} ->
                {value, rest} = Protox.Decode.parse_int32(bytes)
                {[data_type: value], rest}

              {3, _, bytes} ->
                {len, bytes} = Protox.Varint.decode(bytes)
                {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)

                {[
                   segment:
                     Protox.MergeMessage.merge(
                       msg.segment,
                       Transmuter.Onnx.TensorProto.Segment.decode!(delimited)
                     )
                 ], rest}

              {4, 2, bytes} ->
                {len, bytes} = Protox.Varint.decode(bytes)
                {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)

                {[
                   float_data: msg.float_data ++ Protox.Decode.parse_repeated_float([], delimited)
                 ], rest}

              {4, _, bytes} ->
                {value, rest} = Protox.Decode.parse_float(bytes)
                {[float_data: msg.float_data ++ [value]], rest}

              {5, 2, bytes} ->
                {len, bytes} = Protox.Varint.decode(bytes)
                {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)

                {[
                   int32_data: msg.int32_data ++ Protox.Decode.parse_repeated_int32([], delimited)
                 ], rest}

              {5, _, bytes} ->
                {value, rest} = Protox.Decode.parse_int32(bytes)
                {[int32_data: msg.int32_data ++ [value]], rest}

              {6, _, bytes} ->
                {len, bytes} = Protox.Varint.decode(bytes)
                {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)
                {[string_data: msg.string_data ++ [delimited]], rest}

              {7, 2, bytes} ->
                {len, bytes} = Protox.Varint.decode(bytes)
                {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)

                {[
                   int64_data: msg.int64_data ++ Protox.Decode.parse_repeated_int64([], delimited)
                 ], rest}

              {7, _, bytes} ->
                {value, rest} = Protox.Decode.parse_int64(bytes)
                {[int64_data: msg.int64_data ++ [value]], rest}

              {8, _, bytes} ->
                {len, bytes} = Protox.Varint.decode(bytes)
                {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)
                {[name: Protox.Decode.validate_string!(delimited)], rest}

              {9, _, bytes} ->
                {len, bytes} = Protox.Varint.decode(bytes)
                {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)
                {[raw_data: delimited], rest}

              {10, 2, bytes} ->
                {len, bytes} = Protox.Varint.decode(bytes)
                {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)

                {[
                   double_data:
                     msg.double_data ++ Protox.Decode.parse_repeated_double([], delimited)
                 ], rest}

              {10, _, bytes} ->
                {value, rest} = Protox.Decode.parse_double(bytes)
                {[double_data: msg.double_data ++ [value]], rest}

              {11, 2, bytes} ->
                {len, bytes} = Protox.Varint.decode(bytes)
                {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)

                {[
                   uint64_data:
                     msg.uint64_data ++ Protox.Decode.parse_repeated_uint64([], delimited)
                 ], rest}

              {11, _, bytes} ->
                {value, rest} = Protox.Decode.parse_uint64(bytes)
                {[uint64_data: msg.uint64_data ++ [value]], rest}

              {12, _, bytes} ->
                {len, bytes} = Protox.Varint.decode(bytes)
                {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)
                {[doc_string: Protox.Decode.validate_string!(delimited)], rest}

              {13, _, bytes} ->
                {len, bytes} = Protox.Varint.decode(bytes)
                {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)

                {[
                   external_data:
                     msg.external_data ++
                       [Transmuter.Onnx.StringStringEntryProto.decode!(delimited)]
                 ], rest}

              {14, _, bytes} ->
                {value, rest} =
                  Protox.Decode.parse_enum(bytes, Transmuter.Onnx.TensorProto.DataLocation)

                {[data_location: value], rest}

              {16, _, bytes} ->
                {len, bytes} = Protox.Varint.decode(bytes)
                {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)

                {[
                   metadata_props:
                     msg.metadata_props ++
                       [Transmuter.Onnx.StringStringEntryProto.decode!(delimited)]
                 ], rest}

              {tag, wire_type, rest} ->
                {value, rest} = Protox.Decode.parse_unknown(tag, wire_type, rest)

                {[
                   {msg.__struct__.unknown_fields_name(),
                    [value | msg.__struct__.unknown_fields(msg)]}
                 ], rest}
            end

          msg_updated = struct(msg, field)
          parse_key_value(rest, msg_updated)
        end
      )

      []
    )

    (
      @spec json_decode(iodata(), keyword()) :: {:ok, struct()} | {:error, any()}
      def json_decode(input, opts \\ []) do
        try do
          {:ok, json_decode!(input, opts)}
        rescue
          e in Protox.JsonDecodingError -> {:error, e}
        end
      end

      @spec json_decode!(iodata(), keyword()) :: struct() | no_return()
      def json_decode!(input, opts \\ []) do
        {json_library_wrapper, json_library} = Protox.JsonLibrary.get_library(opts, :decode)

        Protox.JsonDecode.decode!(
          input,
          Transmuter.Onnx.TensorProto,
          &json_library_wrapper.decode!(json_library, &1)
        )
      end

      @spec json_encode(struct(), keyword()) :: {:ok, iodata()} | {:error, any()}
      def json_encode(msg, opts \\ []) do
        try do
          {:ok, json_encode!(msg, opts)}
        rescue
          e in Protox.JsonEncodingError -> {:error, e}
        end
      end

      @spec json_encode!(struct(), keyword()) :: iodata() | no_return()
      def json_encode!(msg, opts \\ []) do
        {json_library_wrapper, json_library} = Protox.JsonLibrary.get_library(opts, :encode)
        Protox.JsonEncode.encode!(msg, &json_library_wrapper.encode!(json_library, &1))
      end
    )

    (
      @deprecated "Use fields_defs()/0 instead"
      @spec defs() :: %{
              required(non_neg_integer) => {atom, Protox.Types.kind(), Protox.Types.type()}
            }
      def defs() do
        %{
          1 => {:dims, :unpacked, :int64},
          2 => {:data_type, {:scalar, 0}, :int32},
          3 => {:segment, {:scalar, nil}, {:message, Transmuter.Onnx.TensorProto.Segment}},
          4 => {:float_data, :packed, :float},
          5 => {:int32_data, :packed, :int32},
          6 => {:string_data, :unpacked, :bytes},
          7 => {:int64_data, :packed, :int64},
          8 => {:name, {:scalar, ""}, :string},
          9 => {:raw_data, {:scalar, ""}, :bytes},
          10 => {:double_data, :packed, :double},
          11 => {:uint64_data, :packed, :uint64},
          12 => {:doc_string, {:scalar, ""}, :string},
          13 => {:external_data, :unpacked, {:message, Transmuter.Onnx.StringStringEntryProto}},
          14 =>
            {:data_location, {:scalar, :DEFAULT},
             {:enum, Transmuter.Onnx.TensorProto.DataLocation}},
          16 => {:metadata_props, :unpacked, {:message, Transmuter.Onnx.StringStringEntryProto}}
        }
      end

      @deprecated "Use fields_defs()/0 instead"
      @spec defs_by_name() :: %{
              required(atom) => {non_neg_integer, Protox.Types.kind(), Protox.Types.type()}
            }
      def defs_by_name() do
        %{
          data_location:
            {14, {:scalar, :DEFAULT}, {:enum, Transmuter.Onnx.TensorProto.DataLocation}},
          data_type: {2, {:scalar, 0}, :int32},
          dims: {1, :unpacked, :int64},
          doc_string: {12, {:scalar, ""}, :string},
          double_data: {10, :packed, :double},
          external_data: {13, :unpacked, {:message, Transmuter.Onnx.StringStringEntryProto}},
          float_data: {4, :packed, :float},
          int32_data: {5, :packed, :int32},
          int64_data: {7, :packed, :int64},
          metadata_props: {16, :unpacked, {:message, Transmuter.Onnx.StringStringEntryProto}},
          name: {8, {:scalar, ""}, :string},
          raw_data: {9, {:scalar, ""}, :bytes},
          segment: {3, {:scalar, nil}, {:message, Transmuter.Onnx.TensorProto.Segment}},
          string_data: {6, :unpacked, :bytes},
          uint64_data: {11, :packed, :uint64}
        }
      end
    )

    (
      @spec fields_defs() :: list(Protox.Field.t())
      def fields_defs() do
        [
          %{
            __struct__: Protox.Field,
            json_name: "dims",
            kind: :unpacked,
            label: :repeated,
            name: :dims,
            tag: 1,
            type: :int64
          },
          %{
            __struct__: Protox.Field,
            json_name: "dataType",
            kind: {:scalar, 0},
            label: :optional,
            name: :data_type,
            tag: 2,
            type: :int32
          },
          %{
            __struct__: Protox.Field,
            json_name: "segment",
            kind: {:scalar, nil},
            label: :optional,
            name: :segment,
            tag: 3,
            type: {:message, Transmuter.Onnx.TensorProto.Segment}
          },
          %{
            __struct__: Protox.Field,
            json_name: "floatData",
            kind: :packed,
            label: :repeated,
            name: :float_data,
            tag: 4,
            type: :float
          },
          %{
            __struct__: Protox.Field,
            json_name: "int32Data",
            kind: :packed,
            label: :repeated,
            name: :int32_data,
            tag: 5,
            type: :int32
          },
          %{
            __struct__: Protox.Field,
            json_name: "stringData",
            kind: :unpacked,
            label: :repeated,
            name: :string_data,
            tag: 6,
            type: :bytes
          },
          %{
            __struct__: Protox.Field,
            json_name: "int64Data",
            kind: :packed,
            label: :repeated,
            name: :int64_data,
            tag: 7,
            type: :int64
          },
          %{
            __struct__: Protox.Field,
            json_name: "name",
            kind: {:scalar, ""},
            label: :optional,
            name: :name,
            tag: 8,
            type: :string
          },
          %{
            __struct__: Protox.Field,
            json_name: "rawData",
            kind: {:scalar, ""},
            label: :optional,
            name: :raw_data,
            tag: 9,
            type: :bytes
          },
          %{
            __struct__: Protox.Field,
            json_name: "doubleData",
            kind: :packed,
            label: :repeated,
            name: :double_data,
            tag: 10,
            type: :double
          },
          %{
            __struct__: Protox.Field,
            json_name: "uint64Data",
            kind: :packed,
            label: :repeated,
            name: :uint64_data,
            tag: 11,
            type: :uint64
          },
          %{
            __struct__: Protox.Field,
            json_name: "docString",
            kind: {:scalar, ""},
            label: :optional,
            name: :doc_string,
            tag: 12,
            type: :string
          },
          %{
            __struct__: Protox.Field,
            json_name: "externalData",
            kind: :unpacked,
            label: :repeated,
            name: :external_data,
            tag: 13,
            type: {:message, Transmuter.Onnx.StringStringEntryProto}
          },
          %{
            __struct__: Protox.Field,
            json_name: "dataLocation",
            kind: {:scalar, :DEFAULT},
            label: :optional,
            name: :data_location,
            tag: 14,
            type: {:enum, Transmuter.Onnx.TensorProto.DataLocation}
          },
          %{
            __struct__: Protox.Field,
            json_name: "metadataProps",
            kind: :unpacked,
            label: :repeated,
            name: :metadata_props,
            tag: 16,
            type: {:message, Transmuter.Onnx.StringStringEntryProto}
          }
        ]
      end

      [
        @spec(field_def(atom) :: {:ok, Protox.Field.t()} | {:error, :no_such_field}),
        (
          def field_def(:dims) do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "dims",
               kind: :unpacked,
               label: :repeated,
               name: :dims,
               tag: 1,
               type: :int64
             }}
          end

          def field_def("dims") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "dims",
               kind: :unpacked,
               label: :repeated,
               name: :dims,
               tag: 1,
               type: :int64
             }}
          end

          []
        ),
        (
          def field_def(:data_type) do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "dataType",
               kind: {:scalar, 0},
               label: :optional,
               name: :data_type,
               tag: 2,
               type: :int32
             }}
          end

          def field_def("dataType") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "dataType",
               kind: {:scalar, 0},
               label: :optional,
               name: :data_type,
               tag: 2,
               type: :int32
             }}
          end

          def field_def("data_type") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "dataType",
               kind: {:scalar, 0},
               label: :optional,
               name: :data_type,
               tag: 2,
               type: :int32
             }}
          end
        ),
        (
          def field_def(:segment) do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "segment",
               kind: {:scalar, nil},
               label: :optional,
               name: :segment,
               tag: 3,
               type: {:message, Transmuter.Onnx.TensorProto.Segment}
             }}
          end

          def field_def("segment") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "segment",
               kind: {:scalar, nil},
               label: :optional,
               name: :segment,
               tag: 3,
               type: {:message, Transmuter.Onnx.TensorProto.Segment}
             }}
          end

          []
        ),
        (
          def field_def(:float_data) do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "floatData",
               kind: :packed,
               label: :repeated,
               name: :float_data,
               tag: 4,
               type: :float
             }}
          end

          def field_def("floatData") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "floatData",
               kind: :packed,
               label: :repeated,
               name: :float_data,
               tag: 4,
               type: :float
             }}
          end

          def field_def("float_data") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "floatData",
               kind: :packed,
               label: :repeated,
               name: :float_data,
               tag: 4,
               type: :float
             }}
          end
        ),
        (
          def field_def(:int32_data) do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "int32Data",
               kind: :packed,
               label: :repeated,
               name: :int32_data,
               tag: 5,
               type: :int32
             }}
          end

          def field_def("int32Data") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "int32Data",
               kind: :packed,
               label: :repeated,
               name: :int32_data,
               tag: 5,
               type: :int32
             }}
          end

          def field_def("int32_data") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "int32Data",
               kind: :packed,
               label: :repeated,
               name: :int32_data,
               tag: 5,
               type: :int32
             }}
          end
        ),
        (
          def field_def(:string_data) do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "stringData",
               kind: :unpacked,
               label: :repeated,
               name: :string_data,
               tag: 6,
               type: :bytes
             }}
          end

          def field_def("stringData") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "stringData",
               kind: :unpacked,
               label: :repeated,
               name: :string_data,
               tag: 6,
               type: :bytes
             }}
          end

          def field_def("string_data") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "stringData",
               kind: :unpacked,
               label: :repeated,
               name: :string_data,
               tag: 6,
               type: :bytes
             }}
          end
        ),
        (
          def field_def(:int64_data) do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "int64Data",
               kind: :packed,
               label: :repeated,
               name: :int64_data,
               tag: 7,
               type: :int64
             }}
          end

          def field_def("int64Data") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "int64Data",
               kind: :packed,
               label: :repeated,
               name: :int64_data,
               tag: 7,
               type: :int64
             }}
          end

          def field_def("int64_data") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "int64Data",
               kind: :packed,
               label: :repeated,
               name: :int64_data,
               tag: 7,
               type: :int64
             }}
          end
        ),
        (
          def field_def(:name) do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "name",
               kind: {:scalar, ""},
               label: :optional,
               name: :name,
               tag: 8,
               type: :string
             }}
          end

          def field_def("name") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "name",
               kind: {:scalar, ""},
               label: :optional,
               name: :name,
               tag: 8,
               type: :string
             }}
          end

          []
        ),
        (
          def field_def(:raw_data) do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "rawData",
               kind: {:scalar, ""},
               label: :optional,
               name: :raw_data,
               tag: 9,
               type: :bytes
             }}
          end

          def field_def("rawData") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "rawData",
               kind: {:scalar, ""},
               label: :optional,
               name: :raw_data,
               tag: 9,
               type: :bytes
             }}
          end

          def field_def("raw_data") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "rawData",
               kind: {:scalar, ""},
               label: :optional,
               name: :raw_data,
               tag: 9,
               type: :bytes
             }}
          end
        ),
        (
          def field_def(:double_data) do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "doubleData",
               kind: :packed,
               label: :repeated,
               name: :double_data,
               tag: 10,
               type: :double
             }}
          end

          def field_def("doubleData") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "doubleData",
               kind: :packed,
               label: :repeated,
               name: :double_data,
               tag: 10,
               type: :double
             }}
          end

          def field_def("double_data") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "doubleData",
               kind: :packed,
               label: :repeated,
               name: :double_data,
               tag: 10,
               type: :double
             }}
          end
        ),
        (
          def field_def(:uint64_data) do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "uint64Data",
               kind: :packed,
               label: :repeated,
               name: :uint64_data,
               tag: 11,
               type: :uint64
             }}
          end

          def field_def("uint64Data") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "uint64Data",
               kind: :packed,
               label: :repeated,
               name: :uint64_data,
               tag: 11,
               type: :uint64
             }}
          end

          def field_def("uint64_data") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "uint64Data",
               kind: :packed,
               label: :repeated,
               name: :uint64_data,
               tag: 11,
               type: :uint64
             }}
          end
        ),
        (
          def field_def(:doc_string) do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "docString",
               kind: {:scalar, ""},
               label: :optional,
               name: :doc_string,
               tag: 12,
               type: :string
             }}
          end

          def field_def("docString") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "docString",
               kind: {:scalar, ""},
               label: :optional,
               name: :doc_string,
               tag: 12,
               type: :string
             }}
          end

          def field_def("doc_string") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "docString",
               kind: {:scalar, ""},
               label: :optional,
               name: :doc_string,
               tag: 12,
               type: :string
             }}
          end
        ),
        (
          def field_def(:external_data) do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "externalData",
               kind: :unpacked,
               label: :repeated,
               name: :external_data,
               tag: 13,
               type: {:message, Transmuter.Onnx.StringStringEntryProto}
             }}
          end

          def field_def("externalData") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "externalData",
               kind: :unpacked,
               label: :repeated,
               name: :external_data,
               tag: 13,
               type: {:message, Transmuter.Onnx.StringStringEntryProto}
             }}
          end

          def field_def("external_data") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "externalData",
               kind: :unpacked,
               label: :repeated,
               name: :external_data,
               tag: 13,
               type: {:message, Transmuter.Onnx.StringStringEntryProto}
             }}
          end
        ),
        (
          def field_def(:data_location) do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "dataLocation",
               kind: {:scalar, :DEFAULT},
               label: :optional,
               name: :data_location,
               tag: 14,
               type: {:enum, Transmuter.Onnx.TensorProto.DataLocation}
             }}
          end

          def field_def("dataLocation") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "dataLocation",
               kind: {:scalar, :DEFAULT},
               label: :optional,
               name: :data_location,
               tag: 14,
               type: {:enum, Transmuter.Onnx.TensorProto.DataLocation}
             }}
          end

          def field_def("data_location") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "dataLocation",
               kind: {:scalar, :DEFAULT},
               label: :optional,
               name: :data_location,
               tag: 14,
               type: {:enum, Transmuter.Onnx.TensorProto.DataLocation}
             }}
          end
        ),
        (
          def field_def(:metadata_props) do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "metadataProps",
               kind: :unpacked,
               label: :repeated,
               name: :metadata_props,
               tag: 16,
               type: {:message, Transmuter.Onnx.StringStringEntryProto}
             }}
          end

          def field_def("metadataProps") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "metadataProps",
               kind: :unpacked,
               label: :repeated,
               name: :metadata_props,
               tag: 16,
               type: {:message, Transmuter.Onnx.StringStringEntryProto}
             }}
          end

          def field_def("metadata_props") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "metadataProps",
               kind: :unpacked,
               label: :repeated,
               name: :metadata_props,
               tag: 16,
               type: {:message, Transmuter.Onnx.StringStringEntryProto}
             }}
          end
        ),
        def field_def(_) do
          {:error, :no_such_field}
        end
      ]
    )

    (
      @spec unknown_fields(struct) :: [{non_neg_integer, Protox.Types.tag(), binary}]
      def unknown_fields(msg) do
        msg.__uf__
      end

      @spec unknown_fields_name() :: :__uf__
      def unknown_fields_name() do
        :__uf__
      end

      @spec clear_unknown_fields(struct) :: struct
      def clear_unknown_fields(msg) do
        struct!(msg, [{unknown_fields_name(), []}])
      end
    )

    (
      @spec required_fields() :: []
      def required_fields() do
        []
      end
    )

    (
      @spec syntax() :: atom()
      def syntax() do
        :proto2
      end
    )

    [
      @spec(default(atom) :: {:ok, boolean | integer | String.t() | float} | {:error, atom}),
      def default(:dims) do
        {:error, :no_default_value}
      end,
      def default(:data_type) do
        {:ok, 0}
      end,
      def default(:segment) do
        {:ok, nil}
      end,
      def default(:float_data) do
        {:error, :no_default_value}
      end,
      def default(:int32_data) do
        {:error, :no_default_value}
      end,
      def default(:string_data) do
        {:error, :no_default_value}
      end,
      def default(:int64_data) do
        {:error, :no_default_value}
      end,
      def default(:name) do
        {:ok, ""}
      end,
      def default(:raw_data) do
        {:ok, ""}
      end,
      def default(:double_data) do
        {:error, :no_default_value}
      end,
      def default(:uint64_data) do
        {:error, :no_default_value}
      end,
      def default(:doc_string) do
        {:ok, ""}
      end,
      def default(:external_data) do
        {:error, :no_default_value}
      end,
      def default(:data_location) do
        {:ok, :DEFAULT}
      end,
      def default(:metadata_props) do
        {:error, :no_default_value}
      end,
      def default(_) do
        {:error, :no_such_field}
      end
    ]

    (
      @spec file_options() :: struct()
      def file_options() do
        file_options = %{
          __struct__: Protox.Google.Protobuf.FileOptions,
          __uf__: [],
          cc_enable_arenas: nil,
          cc_generic_services: nil,
          csharp_namespace: nil,
          deprecated: nil,
          go_package: nil,
          java_generate_equals_and_hash: nil,
          java_generic_services: nil,
          java_multiple_files: nil,
          java_outer_classname: nil,
          java_package: nil,
          java_string_check_utf8: nil,
          objc_class_prefix: nil,
          optimize_for: :LITE_RUNTIME,
          php_class_prefix: nil,
          php_generic_services: nil,
          php_metadata_namespace: nil,
          php_namespace: nil,
          py_generic_services: nil,
          ruby_package: nil,
          swift_prefix: nil,
          uninterpreted_option: []
        }

        case function_exported?(Google.Protobuf.FileOptions, :decode!, 1) do
          true ->
            bytes =
              file_options
              |> Protox.Google.Protobuf.FileOptions.encode!()
              |> :binary.list_to_bin()

            apply(Google.Protobuf.FileOptions, :decode!, [bytes])

          false ->
            file_options
        end
      end
    )
  end,
  defmodule Transmuter.Onnx.TensorProto.Segment do
    @moduledoc false
    defstruct begin: nil, end: nil, __uf__: []

    (
      (
        @spec encode(struct) :: {:ok, iodata} | {:error, any}
        def encode(msg) do
          try do
            {:ok, encode!(msg)}
          rescue
            e in [Protox.EncodingError, Protox.RequiredFieldsError] -> {:error, e}
          end
        end

        @spec encode!(struct) :: iodata | no_return
        def encode!(msg) do
          [] |> encode_begin(msg) |> encode_end(msg) |> encode_unknown_fields(msg)
        end
      )

      []

      [
        defp encode_begin(acc, msg) do
          try do
            case msg.begin do
              nil -> acc
              _ -> [acc, "\b", Protox.Encode.encode_int64(msg.begin)]
            end
          rescue
            ArgumentError ->
              reraise Protox.EncodingError.new(:begin, "invalid field value"), __STACKTRACE__
          end
        end,
        defp encode_end(acc, msg) do
          try do
            case msg.end do
              nil -> acc
              _ -> [acc, "\x10", Protox.Encode.encode_int64(msg.end)]
            end
          rescue
            ArgumentError ->
              reraise Protox.EncodingError.new(:end, "invalid field value"), __STACKTRACE__
          end
        end
      ]

      defp encode_unknown_fields(acc, msg) do
        Enum.reduce(msg.__struct__.unknown_fields(msg), acc, fn {tag, wire_type, bytes}, acc ->
          case wire_type do
            0 ->
              [acc, Protox.Encode.make_key_bytes(tag, :int32), bytes]

            1 ->
              [acc, Protox.Encode.make_key_bytes(tag, :double), bytes]

            2 ->
              len_bytes = bytes |> byte_size() |> Protox.Varint.encode()
              [acc, Protox.Encode.make_key_bytes(tag, :packed), len_bytes, bytes]

            5 ->
              [acc, Protox.Encode.make_key_bytes(tag, :float), bytes]
          end
        end)
      end
    )

    (
      (
        @spec decode(binary) :: {:ok, struct} | {:error, any}
        def decode(bytes) do
          try do
            {:ok, decode!(bytes)}
          rescue
            e in [Protox.DecodingError, Protox.IllegalTagError, Protox.RequiredFieldsError] ->
              {:error, e}
          end
        end

        (
          @spec decode!(binary) :: struct | no_return
          def decode!(bytes) do
            parse_key_value(bytes, struct(Transmuter.Onnx.TensorProto.Segment))
          end
        )
      )

      (
        @spec parse_key_value(binary, struct) :: struct
        defp parse_key_value(<<>>, msg) do
          msg
        end

        defp parse_key_value(bytes, msg) do
          {field, rest} =
            case Protox.Decode.parse_key(bytes) do
              {0, _, _} ->
                raise %Protox.IllegalTagError{}

              {1, _, bytes} ->
                {value, rest} = Protox.Decode.parse_int64(bytes)
                {[begin: value], rest}

              {2, _, bytes} ->
                {value, rest} = Protox.Decode.parse_int64(bytes)
                {[end: value], rest}

              {tag, wire_type, rest} ->
                {value, rest} = Protox.Decode.parse_unknown(tag, wire_type, rest)

                {[
                   {msg.__struct__.unknown_fields_name(),
                    [value | msg.__struct__.unknown_fields(msg)]}
                 ], rest}
            end

          msg_updated = struct(msg, field)
          parse_key_value(rest, msg_updated)
        end
      )

      []
    )

    (
      @spec json_decode(iodata(), keyword()) :: {:ok, struct()} | {:error, any()}
      def json_decode(input, opts \\ []) do
        try do
          {:ok, json_decode!(input, opts)}
        rescue
          e in Protox.JsonDecodingError -> {:error, e}
        end
      end

      @spec json_decode!(iodata(), keyword()) :: struct() | no_return()
      def json_decode!(input, opts \\ []) do
        {json_library_wrapper, json_library} = Protox.JsonLibrary.get_library(opts, :decode)

        Protox.JsonDecode.decode!(
          input,
          Transmuter.Onnx.TensorProto.Segment,
          &json_library_wrapper.decode!(json_library, &1)
        )
      end

      @spec json_encode(struct(), keyword()) :: {:ok, iodata()} | {:error, any()}
      def json_encode(msg, opts \\ []) do
        try do
          {:ok, json_encode!(msg, opts)}
        rescue
          e in Protox.JsonEncodingError -> {:error, e}
        end
      end

      @spec json_encode!(struct(), keyword()) :: iodata() | no_return()
      def json_encode!(msg, opts \\ []) do
        {json_library_wrapper, json_library} = Protox.JsonLibrary.get_library(opts, :encode)
        Protox.JsonEncode.encode!(msg, &json_library_wrapper.encode!(json_library, &1))
      end
    )

    (
      @deprecated "Use fields_defs()/0 instead"
      @spec defs() :: %{
              required(non_neg_integer) => {atom, Protox.Types.kind(), Protox.Types.type()}
            }
      def defs() do
        %{1 => {:begin, {:scalar, 0}, :int64}, 2 => {:end, {:scalar, 0}, :int64}}
      end

      @deprecated "Use fields_defs()/0 instead"
      @spec defs_by_name() :: %{
              required(atom) => {non_neg_integer, Protox.Types.kind(), Protox.Types.type()}
            }
      def defs_by_name() do
        %{begin: {1, {:scalar, 0}, :int64}, end: {2, {:scalar, 0}, :int64}}
      end
    )

    (
      @spec fields_defs() :: list(Protox.Field.t())
      def fields_defs() do
        [
          %{
            __struct__: Protox.Field,
            json_name: "begin",
            kind: {:scalar, 0},
            label: :optional,
            name: :begin,
            tag: 1,
            type: :int64
          },
          %{
            __struct__: Protox.Field,
            json_name: "end",
            kind: {:scalar, 0},
            label: :optional,
            name: :end,
            tag: 2,
            type: :int64
          }
        ]
      end

      [
        @spec(field_def(atom) :: {:ok, Protox.Field.t()} | {:error, :no_such_field}),
        (
          def field_def(:begin) do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "begin",
               kind: {:scalar, 0},
               label: :optional,
               name: :begin,
               tag: 1,
               type: :int64
             }}
          end

          def field_def("begin") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "begin",
               kind: {:scalar, 0},
               label: :optional,
               name: :begin,
               tag: 1,
               type: :int64
             }}
          end

          []
        ),
        (
          def field_def(:end) do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "end",
               kind: {:scalar, 0},
               label: :optional,
               name: :end,
               tag: 2,
               type: :int64
             }}
          end

          def field_def("end") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "end",
               kind: {:scalar, 0},
               label: :optional,
               name: :end,
               tag: 2,
               type: :int64
             }}
          end

          []
        ),
        def field_def(_) do
          {:error, :no_such_field}
        end
      ]
    )

    (
      @spec unknown_fields(struct) :: [{non_neg_integer, Protox.Types.tag(), binary}]
      def unknown_fields(msg) do
        msg.__uf__
      end

      @spec unknown_fields_name() :: :__uf__
      def unknown_fields_name() do
        :__uf__
      end

      @spec clear_unknown_fields(struct) :: struct
      def clear_unknown_fields(msg) do
        struct!(msg, [{unknown_fields_name(), []}])
      end
    )

    (
      @spec required_fields() :: []
      def required_fields() do
        []
      end
    )

    (
      @spec syntax() :: atom()
      def syntax() do
        :proto2
      end
    )

    [
      @spec(default(atom) :: {:ok, boolean | integer | String.t() | float} | {:error, atom}),
      def default(:begin) do
        {:ok, 0}
      end,
      def default(:end) do
        {:ok, 0}
      end,
      def default(_) do
        {:error, :no_such_field}
      end
    ]

    (
      @spec file_options() :: struct()
      def file_options() do
        file_options = %{
          __struct__: Protox.Google.Protobuf.FileOptions,
          __uf__: [],
          cc_enable_arenas: nil,
          cc_generic_services: nil,
          csharp_namespace: nil,
          deprecated: nil,
          go_package: nil,
          java_generate_equals_and_hash: nil,
          java_generic_services: nil,
          java_multiple_files: nil,
          java_outer_classname: nil,
          java_package: nil,
          java_string_check_utf8: nil,
          objc_class_prefix: nil,
          optimize_for: :LITE_RUNTIME,
          php_class_prefix: nil,
          php_generic_services: nil,
          php_metadata_namespace: nil,
          php_namespace: nil,
          py_generic_services: nil,
          ruby_package: nil,
          swift_prefix: nil,
          uninterpreted_option: []
        }

        case function_exported?(Google.Protobuf.FileOptions, :decode!, 1) do
          true ->
            bytes =
              file_options
              |> Protox.Google.Protobuf.FileOptions.encode!()
              |> :binary.list_to_bin()

            apply(Google.Protobuf.FileOptions, :decode!, [bytes])

          false ->
            file_options
        end
      end
    )
  end,
  defmodule Transmuter.Onnx.TensorShapeProto do
    @moduledoc false
    defstruct dim: [], __uf__: []

    (
      (
        @spec encode(struct) :: {:ok, iodata} | {:error, any}
        def encode(msg) do
          try do
            {:ok, encode!(msg)}
          rescue
            e in [Protox.EncodingError, Protox.RequiredFieldsError] -> {:error, e}
          end
        end

        @spec encode!(struct) :: iodata | no_return
        def encode!(msg) do
          [] |> encode_dim(msg) |> encode_unknown_fields(msg)
        end
      )

      []

      [
        defp encode_dim(acc, msg) do
          try do
            case msg.dim do
              [] ->
                acc

              values ->
                [
                  acc,
                  Enum.reduce(values, [], fn value, acc ->
                    [acc, "\n", Protox.Encode.encode_message(value)]
                  end)
                ]
            end
          rescue
            ArgumentError ->
              reraise Protox.EncodingError.new(:dim, "invalid field value"), __STACKTRACE__
          end
        end
      ]

      defp encode_unknown_fields(acc, msg) do
        Enum.reduce(msg.__struct__.unknown_fields(msg), acc, fn {tag, wire_type, bytes}, acc ->
          case wire_type do
            0 ->
              [acc, Protox.Encode.make_key_bytes(tag, :int32), bytes]

            1 ->
              [acc, Protox.Encode.make_key_bytes(tag, :double), bytes]

            2 ->
              len_bytes = bytes |> byte_size() |> Protox.Varint.encode()
              [acc, Protox.Encode.make_key_bytes(tag, :packed), len_bytes, bytes]

            5 ->
              [acc, Protox.Encode.make_key_bytes(tag, :float), bytes]
          end
        end)
      end
    )

    (
      (
        @spec decode(binary) :: {:ok, struct} | {:error, any}
        def decode(bytes) do
          try do
            {:ok, decode!(bytes)}
          rescue
            e in [Protox.DecodingError, Protox.IllegalTagError, Protox.RequiredFieldsError] ->
              {:error, e}
          end
        end

        (
          @spec decode!(binary) :: struct | no_return
          def decode!(bytes) do
            parse_key_value(bytes, struct(Transmuter.Onnx.TensorShapeProto))
          end
        )
      )

      (
        @spec parse_key_value(binary, struct) :: struct
        defp parse_key_value(<<>>, msg) do
          msg
        end

        defp parse_key_value(bytes, msg) do
          {field, rest} =
            case Protox.Decode.parse_key(bytes) do
              {0, _, _} ->
                raise %Protox.IllegalTagError{}

              {1, _, bytes} ->
                {len, bytes} = Protox.Varint.decode(bytes)
                {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)

                {[
                   dim: msg.dim ++ [Transmuter.Onnx.TensorShapeProto.Dimension.decode!(delimited)]
                 ], rest}

              {tag, wire_type, rest} ->
                {value, rest} = Protox.Decode.parse_unknown(tag, wire_type, rest)

                {[
                   {msg.__struct__.unknown_fields_name(),
                    [value | msg.__struct__.unknown_fields(msg)]}
                 ], rest}
            end

          msg_updated = struct(msg, field)
          parse_key_value(rest, msg_updated)
        end
      )

      []
    )

    (
      @spec json_decode(iodata(), keyword()) :: {:ok, struct()} | {:error, any()}
      def json_decode(input, opts \\ []) do
        try do
          {:ok, json_decode!(input, opts)}
        rescue
          e in Protox.JsonDecodingError -> {:error, e}
        end
      end

      @spec json_decode!(iodata(), keyword()) :: struct() | no_return()
      def json_decode!(input, opts \\ []) do
        {json_library_wrapper, json_library} = Protox.JsonLibrary.get_library(opts, :decode)

        Protox.JsonDecode.decode!(
          input,
          Transmuter.Onnx.TensorShapeProto,
          &json_library_wrapper.decode!(json_library, &1)
        )
      end

      @spec json_encode(struct(), keyword()) :: {:ok, iodata()} | {:error, any()}
      def json_encode(msg, opts \\ []) do
        try do
          {:ok, json_encode!(msg, opts)}
        rescue
          e in Protox.JsonEncodingError -> {:error, e}
        end
      end

      @spec json_encode!(struct(), keyword()) :: iodata() | no_return()
      def json_encode!(msg, opts \\ []) do
        {json_library_wrapper, json_library} = Protox.JsonLibrary.get_library(opts, :encode)
        Protox.JsonEncode.encode!(msg, &json_library_wrapper.encode!(json_library, &1))
      end
    )

    (
      @deprecated "Use fields_defs()/0 instead"
      @spec defs() :: %{
              required(non_neg_integer) => {atom, Protox.Types.kind(), Protox.Types.type()}
            }
      def defs() do
        %{1 => {:dim, :unpacked, {:message, Transmuter.Onnx.TensorShapeProto.Dimension}}}
      end

      @deprecated "Use fields_defs()/0 instead"
      @spec defs_by_name() :: %{
              required(atom) => {non_neg_integer, Protox.Types.kind(), Protox.Types.type()}
            }
      def defs_by_name() do
        %{dim: {1, :unpacked, {:message, Transmuter.Onnx.TensorShapeProto.Dimension}}}
      end
    )

    (
      @spec fields_defs() :: list(Protox.Field.t())
      def fields_defs() do
        [
          %{
            __struct__: Protox.Field,
            json_name: "dim",
            kind: :unpacked,
            label: :repeated,
            name: :dim,
            tag: 1,
            type: {:message, Transmuter.Onnx.TensorShapeProto.Dimension}
          }
        ]
      end

      [
        @spec(field_def(atom) :: {:ok, Protox.Field.t()} | {:error, :no_such_field}),
        (
          def field_def(:dim) do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "dim",
               kind: :unpacked,
               label: :repeated,
               name: :dim,
               tag: 1,
               type: {:message, Transmuter.Onnx.TensorShapeProto.Dimension}
             }}
          end

          def field_def("dim") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "dim",
               kind: :unpacked,
               label: :repeated,
               name: :dim,
               tag: 1,
               type: {:message, Transmuter.Onnx.TensorShapeProto.Dimension}
             }}
          end

          []
        ),
        def field_def(_) do
          {:error, :no_such_field}
        end
      ]
    )

    (
      @spec unknown_fields(struct) :: [{non_neg_integer, Protox.Types.tag(), binary}]
      def unknown_fields(msg) do
        msg.__uf__
      end

      @spec unknown_fields_name() :: :__uf__
      def unknown_fields_name() do
        :__uf__
      end

      @spec clear_unknown_fields(struct) :: struct
      def clear_unknown_fields(msg) do
        struct!(msg, [{unknown_fields_name(), []}])
      end
    )

    (
      @spec required_fields() :: []
      def required_fields() do
        []
      end
    )

    (
      @spec syntax() :: atom()
      def syntax() do
        :proto2
      end
    )

    [
      @spec(default(atom) :: {:ok, boolean | integer | String.t() | float} | {:error, atom}),
      def default(:dim) do
        {:error, :no_default_value}
      end,
      def default(_) do
        {:error, :no_such_field}
      end
    ]

    (
      @spec file_options() :: struct()
      def file_options() do
        file_options = %{
          __struct__: Protox.Google.Protobuf.FileOptions,
          __uf__: [],
          cc_enable_arenas: nil,
          cc_generic_services: nil,
          csharp_namespace: nil,
          deprecated: nil,
          go_package: nil,
          java_generate_equals_and_hash: nil,
          java_generic_services: nil,
          java_multiple_files: nil,
          java_outer_classname: nil,
          java_package: nil,
          java_string_check_utf8: nil,
          objc_class_prefix: nil,
          optimize_for: :LITE_RUNTIME,
          php_class_prefix: nil,
          php_generic_services: nil,
          php_metadata_namespace: nil,
          php_namespace: nil,
          py_generic_services: nil,
          ruby_package: nil,
          swift_prefix: nil,
          uninterpreted_option: []
        }

        case function_exported?(Google.Protobuf.FileOptions, :decode!, 1) do
          true ->
            bytes =
              file_options
              |> Protox.Google.Protobuf.FileOptions.encode!()
              |> :binary.list_to_bin()

            apply(Google.Protobuf.FileOptions, :decode!, [bytes])

          false ->
            file_options
        end
      end
    )
  end,
  defmodule Transmuter.Onnx.TensorShapeProto.Dimension do
    @moduledoc false
    defstruct value: nil, denotation: nil, __uf__: []

    (
      (
        @spec encode(struct) :: {:ok, iodata} | {:error, any}
        def encode(msg) do
          try do
            {:ok, encode!(msg)}
          rescue
            e in [Protox.EncodingError, Protox.RequiredFieldsError] -> {:error, e}
          end
        end

        @spec encode!(struct) :: iodata | no_return
        def encode!(msg) do
          [] |> encode_value(msg) |> encode_denotation(msg) |> encode_unknown_fields(msg)
        end
      )

      [
        defp encode_value(acc, msg) do
          case msg.value do
            nil -> acc
            {:dim_value, _field_value} -> encode_dim_value(acc, msg)
            {:dim_param, _field_value} -> encode_dim_param(acc, msg)
          end
        end
      ]

      [
        defp encode_dim_value(acc, msg) do
          try do
            {_, child_field_value} = msg.value
            [acc, "\b", Protox.Encode.encode_int64(child_field_value)]
          rescue
            ArgumentError ->
              reraise Protox.EncodingError.new(:dim_value, "invalid field value"), __STACKTRACE__
          end
        end,
        defp encode_dim_param(acc, msg) do
          try do
            {_, child_field_value} = msg.value
            [acc, "\x12", Protox.Encode.encode_string(child_field_value)]
          rescue
            ArgumentError ->
              reraise Protox.EncodingError.new(:dim_param, "invalid field value"), __STACKTRACE__
          end
        end,
        defp encode_denotation(acc, msg) do
          try do
            case msg.denotation do
              nil -> acc
              _ -> [acc, "\x1A", Protox.Encode.encode_string(msg.denotation)]
            end
          rescue
            ArgumentError ->
              reraise Protox.EncodingError.new(:denotation, "invalid field value"), __STACKTRACE__
          end
        end
      ]

      defp encode_unknown_fields(acc, msg) do
        Enum.reduce(msg.__struct__.unknown_fields(msg), acc, fn {tag, wire_type, bytes}, acc ->
          case wire_type do
            0 ->
              [acc, Protox.Encode.make_key_bytes(tag, :int32), bytes]

            1 ->
              [acc, Protox.Encode.make_key_bytes(tag, :double), bytes]

            2 ->
              len_bytes = bytes |> byte_size() |> Protox.Varint.encode()
              [acc, Protox.Encode.make_key_bytes(tag, :packed), len_bytes, bytes]

            5 ->
              [acc, Protox.Encode.make_key_bytes(tag, :float), bytes]
          end
        end)
      end
    )

    (
      (
        @spec decode(binary) :: {:ok, struct} | {:error, any}
        def decode(bytes) do
          try do
            {:ok, decode!(bytes)}
          rescue
            e in [Protox.DecodingError, Protox.IllegalTagError, Protox.RequiredFieldsError] ->
              {:error, e}
          end
        end

        (
          @spec decode!(binary) :: struct | no_return
          def decode!(bytes) do
            parse_key_value(bytes, struct(Transmuter.Onnx.TensorShapeProto.Dimension))
          end
        )
      )

      (
        @spec parse_key_value(binary, struct) :: struct
        defp parse_key_value(<<>>, msg) do
          msg
        end

        defp parse_key_value(bytes, msg) do
          {field, rest} =
            case Protox.Decode.parse_key(bytes) do
              {0, _, _} ->
                raise %Protox.IllegalTagError{}

              {1, _, bytes} ->
                {value, rest} = Protox.Decode.parse_int64(bytes)
                {[value: {:dim_value, value}], rest}

              {2, _, bytes} ->
                {len, bytes} = Protox.Varint.decode(bytes)
                {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)
                {[value: {:dim_param, Protox.Decode.validate_string!(delimited)}], rest}

              {3, _, bytes} ->
                {len, bytes} = Protox.Varint.decode(bytes)
                {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)
                {[denotation: Protox.Decode.validate_string!(delimited)], rest}

              {tag, wire_type, rest} ->
                {value, rest} = Protox.Decode.parse_unknown(tag, wire_type, rest)

                {[
                   {msg.__struct__.unknown_fields_name(),
                    [value | msg.__struct__.unknown_fields(msg)]}
                 ], rest}
            end

          msg_updated = struct(msg, field)
          parse_key_value(rest, msg_updated)
        end
      )

      []
    )

    (
      @spec json_decode(iodata(), keyword()) :: {:ok, struct()} | {:error, any()}
      def json_decode(input, opts \\ []) do
        try do
          {:ok, json_decode!(input, opts)}
        rescue
          e in Protox.JsonDecodingError -> {:error, e}
        end
      end

      @spec json_decode!(iodata(), keyword()) :: struct() | no_return()
      def json_decode!(input, opts \\ []) do
        {json_library_wrapper, json_library} = Protox.JsonLibrary.get_library(opts, :decode)

        Protox.JsonDecode.decode!(
          input,
          Transmuter.Onnx.TensorShapeProto.Dimension,
          &json_library_wrapper.decode!(json_library, &1)
        )
      end

      @spec json_encode(struct(), keyword()) :: {:ok, iodata()} | {:error, any()}
      def json_encode(msg, opts \\ []) do
        try do
          {:ok, json_encode!(msg, opts)}
        rescue
          e in Protox.JsonEncodingError -> {:error, e}
        end
      end

      @spec json_encode!(struct(), keyword()) :: iodata() | no_return()
      def json_encode!(msg, opts \\ []) do
        {json_library_wrapper, json_library} = Protox.JsonLibrary.get_library(opts, :encode)
        Protox.JsonEncode.encode!(msg, &json_library_wrapper.encode!(json_library, &1))
      end
    )

    (
      @deprecated "Use fields_defs()/0 instead"
      @spec defs() :: %{
              required(non_neg_integer) => {atom, Protox.Types.kind(), Protox.Types.type()}
            }
      def defs() do
        %{
          1 => {:dim_value, {:oneof, :value}, :int64},
          2 => {:dim_param, {:oneof, :value}, :string},
          3 => {:denotation, {:scalar, ""}, :string}
        }
      end

      @deprecated "Use fields_defs()/0 instead"
      @spec defs_by_name() :: %{
              required(atom) => {non_neg_integer, Protox.Types.kind(), Protox.Types.type()}
            }
      def defs_by_name() do
        %{
          denotation: {3, {:scalar, ""}, :string},
          dim_param: {2, {:oneof, :value}, :string},
          dim_value: {1, {:oneof, :value}, :int64}
        }
      end
    )

    (
      @spec fields_defs() :: list(Protox.Field.t())
      def fields_defs() do
        [
          %{
            __struct__: Protox.Field,
            json_name: "dimValue",
            kind: {:oneof, :value},
            label: :optional,
            name: :dim_value,
            tag: 1,
            type: :int64
          },
          %{
            __struct__: Protox.Field,
            json_name: "dimParam",
            kind: {:oneof, :value},
            label: :optional,
            name: :dim_param,
            tag: 2,
            type: :string
          },
          %{
            __struct__: Protox.Field,
            json_name: "denotation",
            kind: {:scalar, ""},
            label: :optional,
            name: :denotation,
            tag: 3,
            type: :string
          }
        ]
      end

      [
        @spec(field_def(atom) :: {:ok, Protox.Field.t()} | {:error, :no_such_field}),
        (
          def field_def(:dim_value) do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "dimValue",
               kind: {:oneof, :value},
               label: :optional,
               name: :dim_value,
               tag: 1,
               type: :int64
             }}
          end

          def field_def("dimValue") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "dimValue",
               kind: {:oneof, :value},
               label: :optional,
               name: :dim_value,
               tag: 1,
               type: :int64
             }}
          end

          def field_def("dim_value") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "dimValue",
               kind: {:oneof, :value},
               label: :optional,
               name: :dim_value,
               tag: 1,
               type: :int64
             }}
          end
        ),
        (
          def field_def(:dim_param) do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "dimParam",
               kind: {:oneof, :value},
               label: :optional,
               name: :dim_param,
               tag: 2,
               type: :string
             }}
          end

          def field_def("dimParam") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "dimParam",
               kind: {:oneof, :value},
               label: :optional,
               name: :dim_param,
               tag: 2,
               type: :string
             }}
          end

          def field_def("dim_param") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "dimParam",
               kind: {:oneof, :value},
               label: :optional,
               name: :dim_param,
               tag: 2,
               type: :string
             }}
          end
        ),
        (
          def field_def(:denotation) do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "denotation",
               kind: {:scalar, ""},
               label: :optional,
               name: :denotation,
               tag: 3,
               type: :string
             }}
          end

          def field_def("denotation") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "denotation",
               kind: {:scalar, ""},
               label: :optional,
               name: :denotation,
               tag: 3,
               type: :string
             }}
          end

          []
        ),
        def field_def(_) do
          {:error, :no_such_field}
        end
      ]
    )

    (
      @spec unknown_fields(struct) :: [{non_neg_integer, Protox.Types.tag(), binary}]
      def unknown_fields(msg) do
        msg.__uf__
      end

      @spec unknown_fields_name() :: :__uf__
      def unknown_fields_name() do
        :__uf__
      end

      @spec clear_unknown_fields(struct) :: struct
      def clear_unknown_fields(msg) do
        struct!(msg, [{unknown_fields_name(), []}])
      end
    )

    (
      @spec required_fields() :: []
      def required_fields() do
        []
      end
    )

    (
      @spec syntax() :: atom()
      def syntax() do
        :proto2
      end
    )

    [
      @spec(default(atom) :: {:ok, boolean | integer | String.t() | float} | {:error, atom}),
      def default(:dim_value) do
        {:error, :no_default_value}
      end,
      def default(:dim_param) do
        {:error, :no_default_value}
      end,
      def default(:denotation) do
        {:ok, ""}
      end,
      def default(_) do
        {:error, :no_such_field}
      end
    ]

    (
      @spec file_options() :: struct()
      def file_options() do
        file_options = %{
          __struct__: Protox.Google.Protobuf.FileOptions,
          __uf__: [],
          cc_enable_arenas: nil,
          cc_generic_services: nil,
          csharp_namespace: nil,
          deprecated: nil,
          go_package: nil,
          java_generate_equals_and_hash: nil,
          java_generic_services: nil,
          java_multiple_files: nil,
          java_outer_classname: nil,
          java_package: nil,
          java_string_check_utf8: nil,
          objc_class_prefix: nil,
          optimize_for: :LITE_RUNTIME,
          php_class_prefix: nil,
          php_generic_services: nil,
          php_metadata_namespace: nil,
          php_namespace: nil,
          py_generic_services: nil,
          ruby_package: nil,
          swift_prefix: nil,
          uninterpreted_option: []
        }

        case function_exported?(Google.Protobuf.FileOptions, :decode!, 1) do
          true ->
            bytes =
              file_options
              |> Protox.Google.Protobuf.FileOptions.encode!()
              |> :binary.list_to_bin()

            apply(Google.Protobuf.FileOptions, :decode!, [bytes])

          false ->
            file_options
        end
      end
    )
  end,
  defmodule Transmuter.Onnx.TrainingInfoProto do
    @moduledoc false
    defstruct initialization: nil,
              algorithm: nil,
              initialization_binding: [],
              update_binding: [],
              __uf__: []

    (
      (
        @spec encode(struct) :: {:ok, iodata} | {:error, any}
        def encode(msg) do
          try do
            {:ok, encode!(msg)}
          rescue
            e in [Protox.EncodingError, Protox.RequiredFieldsError] -> {:error, e}
          end
        end

        @spec encode!(struct) :: iodata | no_return
        def encode!(msg) do
          []
          |> encode_initialization(msg)
          |> encode_algorithm(msg)
          |> encode_initialization_binding(msg)
          |> encode_update_binding(msg)
          |> encode_unknown_fields(msg)
        end
      )

      []

      [
        defp encode_initialization(acc, msg) do
          try do
            case msg.initialization do
              nil -> acc
              _ -> [acc, "\n", Protox.Encode.encode_message(msg.initialization)]
            end
          rescue
            ArgumentError ->
              reraise Protox.EncodingError.new(:initialization, "invalid field value"),
                      __STACKTRACE__
          end
        end,
        defp encode_algorithm(acc, msg) do
          try do
            case msg.algorithm do
              nil -> acc
              _ -> [acc, "\x12", Protox.Encode.encode_message(msg.algorithm)]
            end
          rescue
            ArgumentError ->
              reraise Protox.EncodingError.new(:algorithm, "invalid field value"), __STACKTRACE__
          end
        end,
        defp encode_initialization_binding(acc, msg) do
          try do
            case msg.initialization_binding do
              [] ->
                acc

              values ->
                [
                  acc,
                  Enum.reduce(values, [], fn value, acc ->
                    [acc, "\x1A", Protox.Encode.encode_message(value)]
                  end)
                ]
            end
          rescue
            ArgumentError ->
              reraise Protox.EncodingError.new(:initialization_binding, "invalid field value"),
                      __STACKTRACE__
          end
        end,
        defp encode_update_binding(acc, msg) do
          try do
            case msg.update_binding do
              [] ->
                acc

              values ->
                [
                  acc,
                  Enum.reduce(values, [], fn value, acc ->
                    [acc, "\"", Protox.Encode.encode_message(value)]
                  end)
                ]
            end
          rescue
            ArgumentError ->
              reraise Protox.EncodingError.new(:update_binding, "invalid field value"),
                      __STACKTRACE__
          end
        end
      ]

      defp encode_unknown_fields(acc, msg) do
        Enum.reduce(msg.__struct__.unknown_fields(msg), acc, fn {tag, wire_type, bytes}, acc ->
          case wire_type do
            0 ->
              [acc, Protox.Encode.make_key_bytes(tag, :int32), bytes]

            1 ->
              [acc, Protox.Encode.make_key_bytes(tag, :double), bytes]

            2 ->
              len_bytes = bytes |> byte_size() |> Protox.Varint.encode()
              [acc, Protox.Encode.make_key_bytes(tag, :packed), len_bytes, bytes]

            5 ->
              [acc, Protox.Encode.make_key_bytes(tag, :float), bytes]
          end
        end)
      end
    )

    (
      (
        @spec decode(binary) :: {:ok, struct} | {:error, any}
        def decode(bytes) do
          try do
            {:ok, decode!(bytes)}
          rescue
            e in [Protox.DecodingError, Protox.IllegalTagError, Protox.RequiredFieldsError] ->
              {:error, e}
          end
        end

        (
          @spec decode!(binary) :: struct | no_return
          def decode!(bytes) do
            parse_key_value(bytes, struct(Transmuter.Onnx.TrainingInfoProto))
          end
        )
      )

      (
        @spec parse_key_value(binary, struct) :: struct
        defp parse_key_value(<<>>, msg) do
          msg
        end

        defp parse_key_value(bytes, msg) do
          {field, rest} =
            case Protox.Decode.parse_key(bytes) do
              {0, _, _} ->
                raise %Protox.IllegalTagError{}

              {1, _, bytes} ->
                {len, bytes} = Protox.Varint.decode(bytes)
                {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)

                {[
                   initialization:
                     Protox.MergeMessage.merge(
                       msg.initialization,
                       Transmuter.Onnx.GraphProto.decode!(delimited)
                     )
                 ], rest}

              {2, _, bytes} ->
                {len, bytes} = Protox.Varint.decode(bytes)
                {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)

                {[
                   algorithm:
                     Protox.MergeMessage.merge(
                       msg.algorithm,
                       Transmuter.Onnx.GraphProto.decode!(delimited)
                     )
                 ], rest}

              {3, _, bytes} ->
                {len, bytes} = Protox.Varint.decode(bytes)
                {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)

                {[
                   initialization_binding:
                     msg.initialization_binding ++
                       [Transmuter.Onnx.StringStringEntryProto.decode!(delimited)]
                 ], rest}

              {4, _, bytes} ->
                {len, bytes} = Protox.Varint.decode(bytes)
                {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)

                {[
                   update_binding:
                     msg.update_binding ++
                       [Transmuter.Onnx.StringStringEntryProto.decode!(delimited)]
                 ], rest}

              {tag, wire_type, rest} ->
                {value, rest} = Protox.Decode.parse_unknown(tag, wire_type, rest)

                {[
                   {msg.__struct__.unknown_fields_name(),
                    [value | msg.__struct__.unknown_fields(msg)]}
                 ], rest}
            end

          msg_updated = struct(msg, field)
          parse_key_value(rest, msg_updated)
        end
      )

      []
    )

    (
      @spec json_decode(iodata(), keyword()) :: {:ok, struct()} | {:error, any()}
      def json_decode(input, opts \\ []) do
        try do
          {:ok, json_decode!(input, opts)}
        rescue
          e in Protox.JsonDecodingError -> {:error, e}
        end
      end

      @spec json_decode!(iodata(), keyword()) :: struct() | no_return()
      def json_decode!(input, opts \\ []) do
        {json_library_wrapper, json_library} = Protox.JsonLibrary.get_library(opts, :decode)

        Protox.JsonDecode.decode!(
          input,
          Transmuter.Onnx.TrainingInfoProto,
          &json_library_wrapper.decode!(json_library, &1)
        )
      end

      @spec json_encode(struct(), keyword()) :: {:ok, iodata()} | {:error, any()}
      def json_encode(msg, opts \\ []) do
        try do
          {:ok, json_encode!(msg, opts)}
        rescue
          e in Protox.JsonEncodingError -> {:error, e}
        end
      end

      @spec json_encode!(struct(), keyword()) :: iodata() | no_return()
      def json_encode!(msg, opts \\ []) do
        {json_library_wrapper, json_library} = Protox.JsonLibrary.get_library(opts, :encode)
        Protox.JsonEncode.encode!(msg, &json_library_wrapper.encode!(json_library, &1))
      end
    )

    (
      @deprecated "Use fields_defs()/0 instead"
      @spec defs() :: %{
              required(non_neg_integer) => {atom, Protox.Types.kind(), Protox.Types.type()}
            }
      def defs() do
        %{
          1 => {:initialization, {:scalar, nil}, {:message, Transmuter.Onnx.GraphProto}},
          2 => {:algorithm, {:scalar, nil}, {:message, Transmuter.Onnx.GraphProto}},
          3 =>
            {:initialization_binding, :unpacked,
             {:message, Transmuter.Onnx.StringStringEntryProto}},
          4 => {:update_binding, :unpacked, {:message, Transmuter.Onnx.StringStringEntryProto}}
        }
      end

      @deprecated "Use fields_defs()/0 instead"
      @spec defs_by_name() :: %{
              required(atom) => {non_neg_integer, Protox.Types.kind(), Protox.Types.type()}
            }
      def defs_by_name() do
        %{
          algorithm: {2, {:scalar, nil}, {:message, Transmuter.Onnx.GraphProto}},
          initialization: {1, {:scalar, nil}, {:message, Transmuter.Onnx.GraphProto}},
          initialization_binding:
            {3, :unpacked, {:message, Transmuter.Onnx.StringStringEntryProto}},
          update_binding: {4, :unpacked, {:message, Transmuter.Onnx.StringStringEntryProto}}
        }
      end
    )

    (
      @spec fields_defs() :: list(Protox.Field.t())
      def fields_defs() do
        [
          %{
            __struct__: Protox.Field,
            json_name: "initialization",
            kind: {:scalar, nil},
            label: :optional,
            name: :initialization,
            tag: 1,
            type: {:message, Transmuter.Onnx.GraphProto}
          },
          %{
            __struct__: Protox.Field,
            json_name: "algorithm",
            kind: {:scalar, nil},
            label: :optional,
            name: :algorithm,
            tag: 2,
            type: {:message, Transmuter.Onnx.GraphProto}
          },
          %{
            __struct__: Protox.Field,
            json_name: "initializationBinding",
            kind: :unpacked,
            label: :repeated,
            name: :initialization_binding,
            tag: 3,
            type: {:message, Transmuter.Onnx.StringStringEntryProto}
          },
          %{
            __struct__: Protox.Field,
            json_name: "updateBinding",
            kind: :unpacked,
            label: :repeated,
            name: :update_binding,
            tag: 4,
            type: {:message, Transmuter.Onnx.StringStringEntryProto}
          }
        ]
      end

      [
        @spec(field_def(atom) :: {:ok, Protox.Field.t()} | {:error, :no_such_field}),
        (
          def field_def(:initialization) do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "initialization",
               kind: {:scalar, nil},
               label: :optional,
               name: :initialization,
               tag: 1,
               type: {:message, Transmuter.Onnx.GraphProto}
             }}
          end

          def field_def("initialization") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "initialization",
               kind: {:scalar, nil},
               label: :optional,
               name: :initialization,
               tag: 1,
               type: {:message, Transmuter.Onnx.GraphProto}
             }}
          end

          []
        ),
        (
          def field_def(:algorithm) do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "algorithm",
               kind: {:scalar, nil},
               label: :optional,
               name: :algorithm,
               tag: 2,
               type: {:message, Transmuter.Onnx.GraphProto}
             }}
          end

          def field_def("algorithm") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "algorithm",
               kind: {:scalar, nil},
               label: :optional,
               name: :algorithm,
               tag: 2,
               type: {:message, Transmuter.Onnx.GraphProto}
             }}
          end

          []
        ),
        (
          def field_def(:initialization_binding) do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "initializationBinding",
               kind: :unpacked,
               label: :repeated,
               name: :initialization_binding,
               tag: 3,
               type: {:message, Transmuter.Onnx.StringStringEntryProto}
             }}
          end

          def field_def("initializationBinding") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "initializationBinding",
               kind: :unpacked,
               label: :repeated,
               name: :initialization_binding,
               tag: 3,
               type: {:message, Transmuter.Onnx.StringStringEntryProto}
             }}
          end

          def field_def("initialization_binding") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "initializationBinding",
               kind: :unpacked,
               label: :repeated,
               name: :initialization_binding,
               tag: 3,
               type: {:message, Transmuter.Onnx.StringStringEntryProto}
             }}
          end
        ),
        (
          def field_def(:update_binding) do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "updateBinding",
               kind: :unpacked,
               label: :repeated,
               name: :update_binding,
               tag: 4,
               type: {:message, Transmuter.Onnx.StringStringEntryProto}
             }}
          end

          def field_def("updateBinding") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "updateBinding",
               kind: :unpacked,
               label: :repeated,
               name: :update_binding,
               tag: 4,
               type: {:message, Transmuter.Onnx.StringStringEntryProto}
             }}
          end

          def field_def("update_binding") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "updateBinding",
               kind: :unpacked,
               label: :repeated,
               name: :update_binding,
               tag: 4,
               type: {:message, Transmuter.Onnx.StringStringEntryProto}
             }}
          end
        ),
        def field_def(_) do
          {:error, :no_such_field}
        end
      ]
    )

    (
      @spec unknown_fields(struct) :: [{non_neg_integer, Protox.Types.tag(), binary}]
      def unknown_fields(msg) do
        msg.__uf__
      end

      @spec unknown_fields_name() :: :__uf__
      def unknown_fields_name() do
        :__uf__
      end

      @spec clear_unknown_fields(struct) :: struct
      def clear_unknown_fields(msg) do
        struct!(msg, [{unknown_fields_name(), []}])
      end
    )

    (
      @spec required_fields() :: []
      def required_fields() do
        []
      end
    )

    (
      @spec syntax() :: atom()
      def syntax() do
        :proto2
      end
    )

    [
      @spec(default(atom) :: {:ok, boolean | integer | String.t() | float} | {:error, atom}),
      def default(:initialization) do
        {:ok, nil}
      end,
      def default(:algorithm) do
        {:ok, nil}
      end,
      def default(:initialization_binding) do
        {:error, :no_default_value}
      end,
      def default(:update_binding) do
        {:error, :no_default_value}
      end,
      def default(_) do
        {:error, :no_such_field}
      end
    ]

    (
      @spec file_options() :: struct()
      def file_options() do
        file_options = %{
          __struct__: Protox.Google.Protobuf.FileOptions,
          __uf__: [],
          cc_enable_arenas: nil,
          cc_generic_services: nil,
          csharp_namespace: nil,
          deprecated: nil,
          go_package: nil,
          java_generate_equals_and_hash: nil,
          java_generic_services: nil,
          java_multiple_files: nil,
          java_outer_classname: nil,
          java_package: nil,
          java_string_check_utf8: nil,
          objc_class_prefix: nil,
          optimize_for: :LITE_RUNTIME,
          php_class_prefix: nil,
          php_generic_services: nil,
          php_metadata_namespace: nil,
          php_namespace: nil,
          py_generic_services: nil,
          ruby_package: nil,
          swift_prefix: nil,
          uninterpreted_option: []
        }

        case function_exported?(Google.Protobuf.FileOptions, :decode!, 1) do
          true ->
            bytes =
              file_options
              |> Protox.Google.Protobuf.FileOptions.encode!()
              |> :binary.list_to_bin()

            apply(Google.Protobuf.FileOptions, :decode!, [bytes])

          false ->
            file_options
        end
      end
    )
  end,
  defmodule Transmuter.Onnx.TypeProto do
    @moduledoc false
    defstruct value: nil, denotation: nil, __uf__: []

    (
      (
        @spec encode(struct) :: {:ok, iodata} | {:error, any}
        def encode(msg) do
          try do
            {:ok, encode!(msg)}
          rescue
            e in [Protox.EncodingError, Protox.RequiredFieldsError] -> {:error, e}
          end
        end

        @spec encode!(struct) :: iodata | no_return
        def encode!(msg) do
          [] |> encode_value(msg) |> encode_denotation(msg) |> encode_unknown_fields(msg)
        end
      )

      [
        defp encode_value(acc, msg) do
          case msg.value do
            nil -> acc
            {:tensor_type, _field_value} -> encode_tensor_type(acc, msg)
            {:sequence_type, _field_value} -> encode_sequence_type(acc, msg)
            {:map_type, _field_value} -> encode_map_type(acc, msg)
            {:opaque_type, _field_value} -> encode_opaque_type(acc, msg)
            {:sparse_tensor_type, _field_value} -> encode_sparse_tensor_type(acc, msg)
            {:optional_type, _field_value} -> encode_optional_type(acc, msg)
          end
        end
      ]

      [
        defp encode_tensor_type(acc, msg) do
          try do
            {_, child_field_value} = msg.value
            [acc, "\n", Protox.Encode.encode_message(child_field_value)]
          rescue
            ArgumentError ->
              reraise Protox.EncodingError.new(:tensor_type, "invalid field value"),
                      __STACKTRACE__
          end
        end,
        defp encode_sequence_type(acc, msg) do
          try do
            {_, child_field_value} = msg.value
            [acc, "\"", Protox.Encode.encode_message(child_field_value)]
          rescue
            ArgumentError ->
              reraise Protox.EncodingError.new(:sequence_type, "invalid field value"),
                      __STACKTRACE__
          end
        end,
        defp encode_map_type(acc, msg) do
          try do
            {_, child_field_value} = msg.value
            [acc, "*", Protox.Encode.encode_message(child_field_value)]
          rescue
            ArgumentError ->
              reraise Protox.EncodingError.new(:map_type, "invalid field value"), __STACKTRACE__
          end
        end,
        defp encode_denotation(acc, msg) do
          try do
            case msg.denotation do
              nil -> acc
              _ -> [acc, "2", Protox.Encode.encode_string(msg.denotation)]
            end
          rescue
            ArgumentError ->
              reraise Protox.EncodingError.new(:denotation, "invalid field value"), __STACKTRACE__
          end
        end,
        defp encode_opaque_type(acc, msg) do
          try do
            {_, child_field_value} = msg.value
            [acc, ":", Protox.Encode.encode_message(child_field_value)]
          rescue
            ArgumentError ->
              reraise Protox.EncodingError.new(:opaque_type, "invalid field value"),
                      __STACKTRACE__
          end
        end,
        defp encode_sparse_tensor_type(acc, msg) do
          try do
            {_, child_field_value} = msg.value
            [acc, "B", Protox.Encode.encode_message(child_field_value)]
          rescue
            ArgumentError ->
              reraise Protox.EncodingError.new(:sparse_tensor_type, "invalid field value"),
                      __STACKTRACE__
          end
        end,
        defp encode_optional_type(acc, msg) do
          try do
            {_, child_field_value} = msg.value
            [acc, "J", Protox.Encode.encode_message(child_field_value)]
          rescue
            ArgumentError ->
              reraise Protox.EncodingError.new(:optional_type, "invalid field value"),
                      __STACKTRACE__
          end
        end
      ]

      defp encode_unknown_fields(acc, msg) do
        Enum.reduce(msg.__struct__.unknown_fields(msg), acc, fn {tag, wire_type, bytes}, acc ->
          case wire_type do
            0 ->
              [acc, Protox.Encode.make_key_bytes(tag, :int32), bytes]

            1 ->
              [acc, Protox.Encode.make_key_bytes(tag, :double), bytes]

            2 ->
              len_bytes = bytes |> byte_size() |> Protox.Varint.encode()
              [acc, Protox.Encode.make_key_bytes(tag, :packed), len_bytes, bytes]

            5 ->
              [acc, Protox.Encode.make_key_bytes(tag, :float), bytes]
          end
        end)
      end
    )

    (
      (
        @spec decode(binary) :: {:ok, struct} | {:error, any}
        def decode(bytes) do
          try do
            {:ok, decode!(bytes)}
          rescue
            e in [Protox.DecodingError, Protox.IllegalTagError, Protox.RequiredFieldsError] ->
              {:error, e}
          end
        end

        (
          @spec decode!(binary) :: struct | no_return
          def decode!(bytes) do
            parse_key_value(bytes, struct(Transmuter.Onnx.TypeProto))
          end
        )
      )

      (
        @spec parse_key_value(binary, struct) :: struct
        defp parse_key_value(<<>>, msg) do
          msg
        end

        defp parse_key_value(bytes, msg) do
          {field, rest} =
            case Protox.Decode.parse_key(bytes) do
              {0, _, _} ->
                raise %Protox.IllegalTagError{}

              {1, _, bytes} ->
                {len, bytes} = Protox.Varint.decode(bytes)
                {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)

                {[
                   case msg.value do
                     {:tensor_type, previous_value} ->
                       {:value,
                        {:tensor_type,
                         Protox.MergeMessage.merge(
                           previous_value,
                           Transmuter.Onnx.TypeProto.Tensor.decode!(delimited)
                         )}}

                     _ ->
                       {:value,
                        {:tensor_type, Transmuter.Onnx.TypeProto.Tensor.decode!(delimited)}}
                   end
                 ], rest}

              {4, _, bytes} ->
                {len, bytes} = Protox.Varint.decode(bytes)
                {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)

                {[
                   case msg.value do
                     {:sequence_type, previous_value} ->
                       {:value,
                        {:sequence_type,
                         Protox.MergeMessage.merge(
                           previous_value,
                           Transmuter.Onnx.TypeProto.Sequence.decode!(delimited)
                         )}}

                     _ ->
                       {:value,
                        {:sequence_type, Transmuter.Onnx.TypeProto.Sequence.decode!(delimited)}}
                   end
                 ], rest}

              {5, _, bytes} ->
                {len, bytes} = Protox.Varint.decode(bytes)
                {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)

                {[
                   case msg.value do
                     {:map_type, previous_value} ->
                       {:value,
                        {:map_type,
                         Protox.MergeMessage.merge(
                           previous_value,
                           Transmuter.Onnx.TypeProto.Map.decode!(delimited)
                         )}}

                     _ ->
                       {:value, {:map_type, Transmuter.Onnx.TypeProto.Map.decode!(delimited)}}
                   end
                 ], rest}

              {6, _, bytes} ->
                {len, bytes} = Protox.Varint.decode(bytes)
                {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)
                {[denotation: Protox.Decode.validate_string!(delimited)], rest}

              {7, _, bytes} ->
                {len, bytes} = Protox.Varint.decode(bytes)
                {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)

                {[
                   case msg.value do
                     {:opaque_type, previous_value} ->
                       {:value,
                        {:opaque_type,
                         Protox.MergeMessage.merge(
                           previous_value,
                           Transmuter.Onnx.TypeProto.Opaque.decode!(delimited)
                         )}}

                     _ ->
                       {:value,
                        {:opaque_type, Transmuter.Onnx.TypeProto.Opaque.decode!(delimited)}}
                   end
                 ], rest}

              {8, _, bytes} ->
                {len, bytes} = Protox.Varint.decode(bytes)
                {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)

                {[
                   case msg.value do
                     {:sparse_tensor_type, previous_value} ->
                       {:value,
                        {:sparse_tensor_type,
                         Protox.MergeMessage.merge(
                           previous_value,
                           Transmuter.Onnx.TypeProto.SparseTensor.decode!(delimited)
                         )}}

                     _ ->
                       {:value,
                        {:sparse_tensor_type,
                         Transmuter.Onnx.TypeProto.SparseTensor.decode!(delimited)}}
                   end
                 ], rest}

              {9, _, bytes} ->
                {len, bytes} = Protox.Varint.decode(bytes)
                {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)

                {[
                   case msg.value do
                     {:optional_type, previous_value} ->
                       {:value,
                        {:optional_type,
                         Protox.MergeMessage.merge(
                           previous_value,
                           Transmuter.Onnx.TypeProto.Optional.decode!(delimited)
                         )}}

                     _ ->
                       {:value,
                        {:optional_type, Transmuter.Onnx.TypeProto.Optional.decode!(delimited)}}
                   end
                 ], rest}

              {tag, wire_type, rest} ->
                {value, rest} = Protox.Decode.parse_unknown(tag, wire_type, rest)

                {[
                   {msg.__struct__.unknown_fields_name(),
                    [value | msg.__struct__.unknown_fields(msg)]}
                 ], rest}
            end

          msg_updated = struct(msg, field)
          parse_key_value(rest, msg_updated)
        end
      )

      []
    )

    (
      @spec json_decode(iodata(), keyword()) :: {:ok, struct()} | {:error, any()}
      def json_decode(input, opts \\ []) do
        try do
          {:ok, json_decode!(input, opts)}
        rescue
          e in Protox.JsonDecodingError -> {:error, e}
        end
      end

      @spec json_decode!(iodata(), keyword()) :: struct() | no_return()
      def json_decode!(input, opts \\ []) do
        {json_library_wrapper, json_library} = Protox.JsonLibrary.get_library(opts, :decode)

        Protox.JsonDecode.decode!(
          input,
          Transmuter.Onnx.TypeProto,
          &json_library_wrapper.decode!(json_library, &1)
        )
      end

      @spec json_encode(struct(), keyword()) :: {:ok, iodata()} | {:error, any()}
      def json_encode(msg, opts \\ []) do
        try do
          {:ok, json_encode!(msg, opts)}
        rescue
          e in Protox.JsonEncodingError -> {:error, e}
        end
      end

      @spec json_encode!(struct(), keyword()) :: iodata() | no_return()
      def json_encode!(msg, opts \\ []) do
        {json_library_wrapper, json_library} = Protox.JsonLibrary.get_library(opts, :encode)
        Protox.JsonEncode.encode!(msg, &json_library_wrapper.encode!(json_library, &1))
      end
    )

    (
      @deprecated "Use fields_defs()/0 instead"
      @spec defs() :: %{
              required(non_neg_integer) => {atom, Protox.Types.kind(), Protox.Types.type()}
            }
      def defs() do
        %{
          1 => {:tensor_type, {:oneof, :value}, {:message, Transmuter.Onnx.TypeProto.Tensor}},
          4 => {:sequence_type, {:oneof, :value}, {:message, Transmuter.Onnx.TypeProto.Sequence}},
          5 => {:map_type, {:oneof, :value}, {:message, Transmuter.Onnx.TypeProto.Map}},
          6 => {:denotation, {:scalar, ""}, :string},
          7 => {:opaque_type, {:oneof, :value}, {:message, Transmuter.Onnx.TypeProto.Opaque}},
          8 =>
            {:sparse_tensor_type, {:oneof, :value},
             {:message, Transmuter.Onnx.TypeProto.SparseTensor}},
          9 => {:optional_type, {:oneof, :value}, {:message, Transmuter.Onnx.TypeProto.Optional}}
        }
      end

      @deprecated "Use fields_defs()/0 instead"
      @spec defs_by_name() :: %{
              required(atom) => {non_neg_integer, Protox.Types.kind(), Protox.Types.type()}
            }
      def defs_by_name() do
        %{
          denotation: {6, {:scalar, ""}, :string},
          map_type: {5, {:oneof, :value}, {:message, Transmuter.Onnx.TypeProto.Map}},
          opaque_type: {7, {:oneof, :value}, {:message, Transmuter.Onnx.TypeProto.Opaque}},
          optional_type: {9, {:oneof, :value}, {:message, Transmuter.Onnx.TypeProto.Optional}},
          sequence_type: {4, {:oneof, :value}, {:message, Transmuter.Onnx.TypeProto.Sequence}},
          sparse_tensor_type:
            {8, {:oneof, :value}, {:message, Transmuter.Onnx.TypeProto.SparseTensor}},
          tensor_type: {1, {:oneof, :value}, {:message, Transmuter.Onnx.TypeProto.Tensor}}
        }
      end
    )

    (
      @spec fields_defs() :: list(Protox.Field.t())
      def fields_defs() do
        [
          %{
            __struct__: Protox.Field,
            json_name: "tensorType",
            kind: {:oneof, :value},
            label: :optional,
            name: :tensor_type,
            tag: 1,
            type: {:message, Transmuter.Onnx.TypeProto.Tensor}
          },
          %{
            __struct__: Protox.Field,
            json_name: "sequenceType",
            kind: {:oneof, :value},
            label: :optional,
            name: :sequence_type,
            tag: 4,
            type: {:message, Transmuter.Onnx.TypeProto.Sequence}
          },
          %{
            __struct__: Protox.Field,
            json_name: "mapType",
            kind: {:oneof, :value},
            label: :optional,
            name: :map_type,
            tag: 5,
            type: {:message, Transmuter.Onnx.TypeProto.Map}
          },
          %{
            __struct__: Protox.Field,
            json_name: "denotation",
            kind: {:scalar, ""},
            label: :optional,
            name: :denotation,
            tag: 6,
            type: :string
          },
          %{
            __struct__: Protox.Field,
            json_name: "opaqueType",
            kind: {:oneof, :value},
            label: :optional,
            name: :opaque_type,
            tag: 7,
            type: {:message, Transmuter.Onnx.TypeProto.Opaque}
          },
          %{
            __struct__: Protox.Field,
            json_name: "sparseTensorType",
            kind: {:oneof, :value},
            label: :optional,
            name: :sparse_tensor_type,
            tag: 8,
            type: {:message, Transmuter.Onnx.TypeProto.SparseTensor}
          },
          %{
            __struct__: Protox.Field,
            json_name: "optionalType",
            kind: {:oneof, :value},
            label: :optional,
            name: :optional_type,
            tag: 9,
            type: {:message, Transmuter.Onnx.TypeProto.Optional}
          }
        ]
      end

      [
        @spec(field_def(atom) :: {:ok, Protox.Field.t()} | {:error, :no_such_field}),
        (
          def field_def(:tensor_type) do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "tensorType",
               kind: {:oneof, :value},
               label: :optional,
               name: :tensor_type,
               tag: 1,
               type: {:message, Transmuter.Onnx.TypeProto.Tensor}
             }}
          end

          def field_def("tensorType") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "tensorType",
               kind: {:oneof, :value},
               label: :optional,
               name: :tensor_type,
               tag: 1,
               type: {:message, Transmuter.Onnx.TypeProto.Tensor}
             }}
          end

          def field_def("tensor_type") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "tensorType",
               kind: {:oneof, :value},
               label: :optional,
               name: :tensor_type,
               tag: 1,
               type: {:message, Transmuter.Onnx.TypeProto.Tensor}
             }}
          end
        ),
        (
          def field_def(:sequence_type) do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "sequenceType",
               kind: {:oneof, :value},
               label: :optional,
               name: :sequence_type,
               tag: 4,
               type: {:message, Transmuter.Onnx.TypeProto.Sequence}
             }}
          end

          def field_def("sequenceType") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "sequenceType",
               kind: {:oneof, :value},
               label: :optional,
               name: :sequence_type,
               tag: 4,
               type: {:message, Transmuter.Onnx.TypeProto.Sequence}
             }}
          end

          def field_def("sequence_type") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "sequenceType",
               kind: {:oneof, :value},
               label: :optional,
               name: :sequence_type,
               tag: 4,
               type: {:message, Transmuter.Onnx.TypeProto.Sequence}
             }}
          end
        ),
        (
          def field_def(:map_type) do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "mapType",
               kind: {:oneof, :value},
               label: :optional,
               name: :map_type,
               tag: 5,
               type: {:message, Transmuter.Onnx.TypeProto.Map}
             }}
          end

          def field_def("mapType") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "mapType",
               kind: {:oneof, :value},
               label: :optional,
               name: :map_type,
               tag: 5,
               type: {:message, Transmuter.Onnx.TypeProto.Map}
             }}
          end

          def field_def("map_type") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "mapType",
               kind: {:oneof, :value},
               label: :optional,
               name: :map_type,
               tag: 5,
               type: {:message, Transmuter.Onnx.TypeProto.Map}
             }}
          end
        ),
        (
          def field_def(:denotation) do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "denotation",
               kind: {:scalar, ""},
               label: :optional,
               name: :denotation,
               tag: 6,
               type: :string
             }}
          end

          def field_def("denotation") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "denotation",
               kind: {:scalar, ""},
               label: :optional,
               name: :denotation,
               tag: 6,
               type: :string
             }}
          end

          []
        ),
        (
          def field_def(:opaque_type) do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "opaqueType",
               kind: {:oneof, :value},
               label: :optional,
               name: :opaque_type,
               tag: 7,
               type: {:message, Transmuter.Onnx.TypeProto.Opaque}
             }}
          end

          def field_def("opaqueType") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "opaqueType",
               kind: {:oneof, :value},
               label: :optional,
               name: :opaque_type,
               tag: 7,
               type: {:message, Transmuter.Onnx.TypeProto.Opaque}
             }}
          end

          def field_def("opaque_type") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "opaqueType",
               kind: {:oneof, :value},
               label: :optional,
               name: :opaque_type,
               tag: 7,
               type: {:message, Transmuter.Onnx.TypeProto.Opaque}
             }}
          end
        ),
        (
          def field_def(:sparse_tensor_type) do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "sparseTensorType",
               kind: {:oneof, :value},
               label: :optional,
               name: :sparse_tensor_type,
               tag: 8,
               type: {:message, Transmuter.Onnx.TypeProto.SparseTensor}
             }}
          end

          def field_def("sparseTensorType") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "sparseTensorType",
               kind: {:oneof, :value},
               label: :optional,
               name: :sparse_tensor_type,
               tag: 8,
               type: {:message, Transmuter.Onnx.TypeProto.SparseTensor}
             }}
          end

          def field_def("sparse_tensor_type") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "sparseTensorType",
               kind: {:oneof, :value},
               label: :optional,
               name: :sparse_tensor_type,
               tag: 8,
               type: {:message, Transmuter.Onnx.TypeProto.SparseTensor}
             }}
          end
        ),
        (
          def field_def(:optional_type) do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "optionalType",
               kind: {:oneof, :value},
               label: :optional,
               name: :optional_type,
               tag: 9,
               type: {:message, Transmuter.Onnx.TypeProto.Optional}
             }}
          end

          def field_def("optionalType") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "optionalType",
               kind: {:oneof, :value},
               label: :optional,
               name: :optional_type,
               tag: 9,
               type: {:message, Transmuter.Onnx.TypeProto.Optional}
             }}
          end

          def field_def("optional_type") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "optionalType",
               kind: {:oneof, :value},
               label: :optional,
               name: :optional_type,
               tag: 9,
               type: {:message, Transmuter.Onnx.TypeProto.Optional}
             }}
          end
        ),
        def field_def(_) do
          {:error, :no_such_field}
        end
      ]
    )

    (
      @spec unknown_fields(struct) :: [{non_neg_integer, Protox.Types.tag(), binary}]
      def unknown_fields(msg) do
        msg.__uf__
      end

      @spec unknown_fields_name() :: :__uf__
      def unknown_fields_name() do
        :__uf__
      end

      @spec clear_unknown_fields(struct) :: struct
      def clear_unknown_fields(msg) do
        struct!(msg, [{unknown_fields_name(), []}])
      end
    )

    (
      @spec required_fields() :: []
      def required_fields() do
        []
      end
    )

    (
      @spec syntax() :: atom()
      def syntax() do
        :proto2
      end
    )

    [
      @spec(default(atom) :: {:ok, boolean | integer | String.t() | float} | {:error, atom}),
      def default(:tensor_type) do
        {:error, :no_default_value}
      end,
      def default(:sequence_type) do
        {:error, :no_default_value}
      end,
      def default(:map_type) do
        {:error, :no_default_value}
      end,
      def default(:denotation) do
        {:ok, ""}
      end,
      def default(:opaque_type) do
        {:error, :no_default_value}
      end,
      def default(:sparse_tensor_type) do
        {:error, :no_default_value}
      end,
      def default(:optional_type) do
        {:error, :no_default_value}
      end,
      def default(_) do
        {:error, :no_such_field}
      end
    ]

    (
      @spec file_options() :: struct()
      def file_options() do
        file_options = %{
          __struct__: Protox.Google.Protobuf.FileOptions,
          __uf__: [],
          cc_enable_arenas: nil,
          cc_generic_services: nil,
          csharp_namespace: nil,
          deprecated: nil,
          go_package: nil,
          java_generate_equals_and_hash: nil,
          java_generic_services: nil,
          java_multiple_files: nil,
          java_outer_classname: nil,
          java_package: nil,
          java_string_check_utf8: nil,
          objc_class_prefix: nil,
          optimize_for: :LITE_RUNTIME,
          php_class_prefix: nil,
          php_generic_services: nil,
          php_metadata_namespace: nil,
          php_namespace: nil,
          py_generic_services: nil,
          ruby_package: nil,
          swift_prefix: nil,
          uninterpreted_option: []
        }

        case function_exported?(Google.Protobuf.FileOptions, :decode!, 1) do
          true ->
            bytes =
              file_options
              |> Protox.Google.Protobuf.FileOptions.encode!()
              |> :binary.list_to_bin()

            apply(Google.Protobuf.FileOptions, :decode!, [bytes])

          false ->
            file_options
        end
      end
    )
  end,
  defmodule Transmuter.Onnx.TypeProto.Map do
    @moduledoc false
    defstruct key_type: nil, value_type: nil, __uf__: []

    (
      (
        @spec encode(struct) :: {:ok, iodata} | {:error, any}
        def encode(msg) do
          try do
            {:ok, encode!(msg)}
          rescue
            e in [Protox.EncodingError, Protox.RequiredFieldsError] -> {:error, e}
          end
        end

        @spec encode!(struct) :: iodata | no_return
        def encode!(msg) do
          [] |> encode_key_type(msg) |> encode_value_type(msg) |> encode_unknown_fields(msg)
        end
      )

      []

      [
        defp encode_key_type(acc, msg) do
          try do
            case msg.key_type do
              nil -> acc
              _ -> [acc, "\b", Protox.Encode.encode_int32(msg.key_type)]
            end
          rescue
            ArgumentError ->
              reraise Protox.EncodingError.new(:key_type, "invalid field value"), __STACKTRACE__
          end
        end,
        defp encode_value_type(acc, msg) do
          try do
            case msg.value_type do
              nil -> acc
              _ -> [acc, "\x12", Protox.Encode.encode_message(msg.value_type)]
            end
          rescue
            ArgumentError ->
              reraise Protox.EncodingError.new(:value_type, "invalid field value"), __STACKTRACE__
          end
        end
      ]

      defp encode_unknown_fields(acc, msg) do
        Enum.reduce(msg.__struct__.unknown_fields(msg), acc, fn {tag, wire_type, bytes}, acc ->
          case wire_type do
            0 ->
              [acc, Protox.Encode.make_key_bytes(tag, :int32), bytes]

            1 ->
              [acc, Protox.Encode.make_key_bytes(tag, :double), bytes]

            2 ->
              len_bytes = bytes |> byte_size() |> Protox.Varint.encode()
              [acc, Protox.Encode.make_key_bytes(tag, :packed), len_bytes, bytes]

            5 ->
              [acc, Protox.Encode.make_key_bytes(tag, :float), bytes]
          end
        end)
      end
    )

    (
      (
        @spec decode(binary) :: {:ok, struct} | {:error, any}
        def decode(bytes) do
          try do
            {:ok, decode!(bytes)}
          rescue
            e in [Protox.DecodingError, Protox.IllegalTagError, Protox.RequiredFieldsError] ->
              {:error, e}
          end
        end

        (
          @spec decode!(binary) :: struct | no_return
          def decode!(bytes) do
            parse_key_value(bytes, struct(Transmuter.Onnx.TypeProto.Map))
          end
        )
      )

      (
        @spec parse_key_value(binary, struct) :: struct
        defp parse_key_value(<<>>, msg) do
          msg
        end

        defp parse_key_value(bytes, msg) do
          {field, rest} =
            case Protox.Decode.parse_key(bytes) do
              {0, _, _} ->
                raise %Protox.IllegalTagError{}

              {1, _, bytes} ->
                {value, rest} = Protox.Decode.parse_int32(bytes)
                {[key_type: value], rest}

              {2, _, bytes} ->
                {len, bytes} = Protox.Varint.decode(bytes)
                {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)

                {[
                   value_type:
                     Protox.MergeMessage.merge(
                       msg.value_type,
                       Transmuter.Onnx.TypeProto.decode!(delimited)
                     )
                 ], rest}

              {tag, wire_type, rest} ->
                {value, rest} = Protox.Decode.parse_unknown(tag, wire_type, rest)

                {[
                   {msg.__struct__.unknown_fields_name(),
                    [value | msg.__struct__.unknown_fields(msg)]}
                 ], rest}
            end

          msg_updated = struct(msg, field)
          parse_key_value(rest, msg_updated)
        end
      )

      []
    )

    (
      @spec json_decode(iodata(), keyword()) :: {:ok, struct()} | {:error, any()}
      def json_decode(input, opts \\ []) do
        try do
          {:ok, json_decode!(input, opts)}
        rescue
          e in Protox.JsonDecodingError -> {:error, e}
        end
      end

      @spec json_decode!(iodata(), keyword()) :: struct() | no_return()
      def json_decode!(input, opts \\ []) do
        {json_library_wrapper, json_library} = Protox.JsonLibrary.get_library(opts, :decode)

        Protox.JsonDecode.decode!(
          input,
          Transmuter.Onnx.TypeProto.Map,
          &json_library_wrapper.decode!(json_library, &1)
        )
      end

      @spec json_encode(struct(), keyword()) :: {:ok, iodata()} | {:error, any()}
      def json_encode(msg, opts \\ []) do
        try do
          {:ok, json_encode!(msg, opts)}
        rescue
          e in Protox.JsonEncodingError -> {:error, e}
        end
      end

      @spec json_encode!(struct(), keyword()) :: iodata() | no_return()
      def json_encode!(msg, opts \\ []) do
        {json_library_wrapper, json_library} = Protox.JsonLibrary.get_library(opts, :encode)
        Protox.JsonEncode.encode!(msg, &json_library_wrapper.encode!(json_library, &1))
      end
    )

    (
      @deprecated "Use fields_defs()/0 instead"
      @spec defs() :: %{
              required(non_neg_integer) => {atom, Protox.Types.kind(), Protox.Types.type()}
            }
      def defs() do
        %{
          1 => {:key_type, {:scalar, 0}, :int32},
          2 => {:value_type, {:scalar, nil}, {:message, Transmuter.Onnx.TypeProto}}
        }
      end

      @deprecated "Use fields_defs()/0 instead"
      @spec defs_by_name() :: %{
              required(atom) => {non_neg_integer, Protox.Types.kind(), Protox.Types.type()}
            }
      def defs_by_name() do
        %{
          key_type: {1, {:scalar, 0}, :int32},
          value_type: {2, {:scalar, nil}, {:message, Transmuter.Onnx.TypeProto}}
        }
      end
    )

    (
      @spec fields_defs() :: list(Protox.Field.t())
      def fields_defs() do
        [
          %{
            __struct__: Protox.Field,
            json_name: "keyType",
            kind: {:scalar, 0},
            label: :optional,
            name: :key_type,
            tag: 1,
            type: :int32
          },
          %{
            __struct__: Protox.Field,
            json_name: "valueType",
            kind: {:scalar, nil},
            label: :optional,
            name: :value_type,
            tag: 2,
            type: {:message, Transmuter.Onnx.TypeProto}
          }
        ]
      end

      [
        @spec(field_def(atom) :: {:ok, Protox.Field.t()} | {:error, :no_such_field}),
        (
          def field_def(:key_type) do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "keyType",
               kind: {:scalar, 0},
               label: :optional,
               name: :key_type,
               tag: 1,
               type: :int32
             }}
          end

          def field_def("keyType") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "keyType",
               kind: {:scalar, 0},
               label: :optional,
               name: :key_type,
               tag: 1,
               type: :int32
             }}
          end

          def field_def("key_type") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "keyType",
               kind: {:scalar, 0},
               label: :optional,
               name: :key_type,
               tag: 1,
               type: :int32
             }}
          end
        ),
        (
          def field_def(:value_type) do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "valueType",
               kind: {:scalar, nil},
               label: :optional,
               name: :value_type,
               tag: 2,
               type: {:message, Transmuter.Onnx.TypeProto}
             }}
          end

          def field_def("valueType") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "valueType",
               kind: {:scalar, nil},
               label: :optional,
               name: :value_type,
               tag: 2,
               type: {:message, Transmuter.Onnx.TypeProto}
             }}
          end

          def field_def("value_type") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "valueType",
               kind: {:scalar, nil},
               label: :optional,
               name: :value_type,
               tag: 2,
               type: {:message, Transmuter.Onnx.TypeProto}
             }}
          end
        ),
        def field_def(_) do
          {:error, :no_such_field}
        end
      ]
    )

    (
      @spec unknown_fields(struct) :: [{non_neg_integer, Protox.Types.tag(), binary}]
      def unknown_fields(msg) do
        msg.__uf__
      end

      @spec unknown_fields_name() :: :__uf__
      def unknown_fields_name() do
        :__uf__
      end

      @spec clear_unknown_fields(struct) :: struct
      def clear_unknown_fields(msg) do
        struct!(msg, [{unknown_fields_name(), []}])
      end
    )

    (
      @spec required_fields() :: []
      def required_fields() do
        []
      end
    )

    (
      @spec syntax() :: atom()
      def syntax() do
        :proto2
      end
    )

    [
      @spec(default(atom) :: {:ok, boolean | integer | String.t() | float} | {:error, atom}),
      def default(:key_type) do
        {:ok, 0}
      end,
      def default(:value_type) do
        {:ok, nil}
      end,
      def default(_) do
        {:error, :no_such_field}
      end
    ]

    (
      @spec file_options() :: struct()
      def file_options() do
        file_options = %{
          __struct__: Protox.Google.Protobuf.FileOptions,
          __uf__: [],
          cc_enable_arenas: nil,
          cc_generic_services: nil,
          csharp_namespace: nil,
          deprecated: nil,
          go_package: nil,
          java_generate_equals_and_hash: nil,
          java_generic_services: nil,
          java_multiple_files: nil,
          java_outer_classname: nil,
          java_package: nil,
          java_string_check_utf8: nil,
          objc_class_prefix: nil,
          optimize_for: :LITE_RUNTIME,
          php_class_prefix: nil,
          php_generic_services: nil,
          php_metadata_namespace: nil,
          php_namespace: nil,
          py_generic_services: nil,
          ruby_package: nil,
          swift_prefix: nil,
          uninterpreted_option: []
        }

        case function_exported?(Google.Protobuf.FileOptions, :decode!, 1) do
          true ->
            bytes =
              file_options
              |> Protox.Google.Protobuf.FileOptions.encode!()
              |> :binary.list_to_bin()

            apply(Google.Protobuf.FileOptions, :decode!, [bytes])

          false ->
            file_options
        end
      end
    )
  end,
  defmodule Transmuter.Onnx.TypeProto.Opaque do
    @moduledoc false
    defstruct domain: nil, name: nil, __uf__: []

    (
      (
        @spec encode(struct) :: {:ok, iodata} | {:error, any}
        def encode(msg) do
          try do
            {:ok, encode!(msg)}
          rescue
            e in [Protox.EncodingError, Protox.RequiredFieldsError] -> {:error, e}
          end
        end

        @spec encode!(struct) :: iodata | no_return
        def encode!(msg) do
          [] |> encode_domain(msg) |> encode_name(msg) |> encode_unknown_fields(msg)
        end
      )

      []

      [
        defp encode_domain(acc, msg) do
          try do
            case msg.domain do
              nil -> acc
              _ -> [acc, "\n", Protox.Encode.encode_string(msg.domain)]
            end
          rescue
            ArgumentError ->
              reraise Protox.EncodingError.new(:domain, "invalid field value"), __STACKTRACE__
          end
        end,
        defp encode_name(acc, msg) do
          try do
            case msg.name do
              nil -> acc
              _ -> [acc, "\x12", Protox.Encode.encode_string(msg.name)]
            end
          rescue
            ArgumentError ->
              reraise Protox.EncodingError.new(:name, "invalid field value"), __STACKTRACE__
          end
        end
      ]

      defp encode_unknown_fields(acc, msg) do
        Enum.reduce(msg.__struct__.unknown_fields(msg), acc, fn {tag, wire_type, bytes}, acc ->
          case wire_type do
            0 ->
              [acc, Protox.Encode.make_key_bytes(tag, :int32), bytes]

            1 ->
              [acc, Protox.Encode.make_key_bytes(tag, :double), bytes]

            2 ->
              len_bytes = bytes |> byte_size() |> Protox.Varint.encode()
              [acc, Protox.Encode.make_key_bytes(tag, :packed), len_bytes, bytes]

            5 ->
              [acc, Protox.Encode.make_key_bytes(tag, :float), bytes]
          end
        end)
      end
    )

    (
      (
        @spec decode(binary) :: {:ok, struct} | {:error, any}
        def decode(bytes) do
          try do
            {:ok, decode!(bytes)}
          rescue
            e in [Protox.DecodingError, Protox.IllegalTagError, Protox.RequiredFieldsError] ->
              {:error, e}
          end
        end

        (
          @spec decode!(binary) :: struct | no_return
          def decode!(bytes) do
            parse_key_value(bytes, struct(Transmuter.Onnx.TypeProto.Opaque))
          end
        )
      )

      (
        @spec parse_key_value(binary, struct) :: struct
        defp parse_key_value(<<>>, msg) do
          msg
        end

        defp parse_key_value(bytes, msg) do
          {field, rest} =
            case Protox.Decode.parse_key(bytes) do
              {0, _, _} ->
                raise %Protox.IllegalTagError{}

              {1, _, bytes} ->
                {len, bytes} = Protox.Varint.decode(bytes)
                {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)
                {[domain: Protox.Decode.validate_string!(delimited)], rest}

              {2, _, bytes} ->
                {len, bytes} = Protox.Varint.decode(bytes)
                {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)
                {[name: Protox.Decode.validate_string!(delimited)], rest}

              {tag, wire_type, rest} ->
                {value, rest} = Protox.Decode.parse_unknown(tag, wire_type, rest)

                {[
                   {msg.__struct__.unknown_fields_name(),
                    [value | msg.__struct__.unknown_fields(msg)]}
                 ], rest}
            end

          msg_updated = struct(msg, field)
          parse_key_value(rest, msg_updated)
        end
      )

      []
    )

    (
      @spec json_decode(iodata(), keyword()) :: {:ok, struct()} | {:error, any()}
      def json_decode(input, opts \\ []) do
        try do
          {:ok, json_decode!(input, opts)}
        rescue
          e in Protox.JsonDecodingError -> {:error, e}
        end
      end

      @spec json_decode!(iodata(), keyword()) :: struct() | no_return()
      def json_decode!(input, opts \\ []) do
        {json_library_wrapper, json_library} = Protox.JsonLibrary.get_library(opts, :decode)

        Protox.JsonDecode.decode!(
          input,
          Transmuter.Onnx.TypeProto.Opaque,
          &json_library_wrapper.decode!(json_library, &1)
        )
      end

      @spec json_encode(struct(), keyword()) :: {:ok, iodata()} | {:error, any()}
      def json_encode(msg, opts \\ []) do
        try do
          {:ok, json_encode!(msg, opts)}
        rescue
          e in Protox.JsonEncodingError -> {:error, e}
        end
      end

      @spec json_encode!(struct(), keyword()) :: iodata() | no_return()
      def json_encode!(msg, opts \\ []) do
        {json_library_wrapper, json_library} = Protox.JsonLibrary.get_library(opts, :encode)
        Protox.JsonEncode.encode!(msg, &json_library_wrapper.encode!(json_library, &1))
      end
    )

    (
      @deprecated "Use fields_defs()/0 instead"
      @spec defs() :: %{
              required(non_neg_integer) => {atom, Protox.Types.kind(), Protox.Types.type()}
            }
      def defs() do
        %{1 => {:domain, {:scalar, ""}, :string}, 2 => {:name, {:scalar, ""}, :string}}
      end

      @deprecated "Use fields_defs()/0 instead"
      @spec defs_by_name() :: %{
              required(atom) => {non_neg_integer, Protox.Types.kind(), Protox.Types.type()}
            }
      def defs_by_name() do
        %{domain: {1, {:scalar, ""}, :string}, name: {2, {:scalar, ""}, :string}}
      end
    )

    (
      @spec fields_defs() :: list(Protox.Field.t())
      def fields_defs() do
        [
          %{
            __struct__: Protox.Field,
            json_name: "domain",
            kind: {:scalar, ""},
            label: :optional,
            name: :domain,
            tag: 1,
            type: :string
          },
          %{
            __struct__: Protox.Field,
            json_name: "name",
            kind: {:scalar, ""},
            label: :optional,
            name: :name,
            tag: 2,
            type: :string
          }
        ]
      end

      [
        @spec(field_def(atom) :: {:ok, Protox.Field.t()} | {:error, :no_such_field}),
        (
          def field_def(:domain) do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "domain",
               kind: {:scalar, ""},
               label: :optional,
               name: :domain,
               tag: 1,
               type: :string
             }}
          end

          def field_def("domain") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "domain",
               kind: {:scalar, ""},
               label: :optional,
               name: :domain,
               tag: 1,
               type: :string
             }}
          end

          []
        ),
        (
          def field_def(:name) do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "name",
               kind: {:scalar, ""},
               label: :optional,
               name: :name,
               tag: 2,
               type: :string
             }}
          end

          def field_def("name") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "name",
               kind: {:scalar, ""},
               label: :optional,
               name: :name,
               tag: 2,
               type: :string
             }}
          end

          []
        ),
        def field_def(_) do
          {:error, :no_such_field}
        end
      ]
    )

    (
      @spec unknown_fields(struct) :: [{non_neg_integer, Protox.Types.tag(), binary}]
      def unknown_fields(msg) do
        msg.__uf__
      end

      @spec unknown_fields_name() :: :__uf__
      def unknown_fields_name() do
        :__uf__
      end

      @spec clear_unknown_fields(struct) :: struct
      def clear_unknown_fields(msg) do
        struct!(msg, [{unknown_fields_name(), []}])
      end
    )

    (
      @spec required_fields() :: []
      def required_fields() do
        []
      end
    )

    (
      @spec syntax() :: atom()
      def syntax() do
        :proto2
      end
    )

    [
      @spec(default(atom) :: {:ok, boolean | integer | String.t() | float} | {:error, atom}),
      def default(:domain) do
        {:ok, ""}
      end,
      def default(:name) do
        {:ok, ""}
      end,
      def default(_) do
        {:error, :no_such_field}
      end
    ]

    (
      @spec file_options() :: struct()
      def file_options() do
        file_options = %{
          __struct__: Protox.Google.Protobuf.FileOptions,
          __uf__: [],
          cc_enable_arenas: nil,
          cc_generic_services: nil,
          csharp_namespace: nil,
          deprecated: nil,
          go_package: nil,
          java_generate_equals_and_hash: nil,
          java_generic_services: nil,
          java_multiple_files: nil,
          java_outer_classname: nil,
          java_package: nil,
          java_string_check_utf8: nil,
          objc_class_prefix: nil,
          optimize_for: :LITE_RUNTIME,
          php_class_prefix: nil,
          php_generic_services: nil,
          php_metadata_namespace: nil,
          php_namespace: nil,
          py_generic_services: nil,
          ruby_package: nil,
          swift_prefix: nil,
          uninterpreted_option: []
        }

        case function_exported?(Google.Protobuf.FileOptions, :decode!, 1) do
          true ->
            bytes =
              file_options
              |> Protox.Google.Protobuf.FileOptions.encode!()
              |> :binary.list_to_bin()

            apply(Google.Protobuf.FileOptions, :decode!, [bytes])

          false ->
            file_options
        end
      end
    )
  end,
  defmodule Transmuter.Onnx.TypeProto.Optional do
    @moduledoc false
    defstruct elem_type: nil, __uf__: []

    (
      (
        @spec encode(struct) :: {:ok, iodata} | {:error, any}
        def encode(msg) do
          try do
            {:ok, encode!(msg)}
          rescue
            e in [Protox.EncodingError, Protox.RequiredFieldsError] -> {:error, e}
          end
        end

        @spec encode!(struct) :: iodata | no_return
        def encode!(msg) do
          [] |> encode_elem_type(msg) |> encode_unknown_fields(msg)
        end
      )

      []

      [
        defp encode_elem_type(acc, msg) do
          try do
            case msg.elem_type do
              nil -> acc
              _ -> [acc, "\n", Protox.Encode.encode_message(msg.elem_type)]
            end
          rescue
            ArgumentError ->
              reraise Protox.EncodingError.new(:elem_type, "invalid field value"), __STACKTRACE__
          end
        end
      ]

      defp encode_unknown_fields(acc, msg) do
        Enum.reduce(msg.__struct__.unknown_fields(msg), acc, fn {tag, wire_type, bytes}, acc ->
          case wire_type do
            0 ->
              [acc, Protox.Encode.make_key_bytes(tag, :int32), bytes]

            1 ->
              [acc, Protox.Encode.make_key_bytes(tag, :double), bytes]

            2 ->
              len_bytes = bytes |> byte_size() |> Protox.Varint.encode()
              [acc, Protox.Encode.make_key_bytes(tag, :packed), len_bytes, bytes]

            5 ->
              [acc, Protox.Encode.make_key_bytes(tag, :float), bytes]
          end
        end)
      end
    )

    (
      (
        @spec decode(binary) :: {:ok, struct} | {:error, any}
        def decode(bytes) do
          try do
            {:ok, decode!(bytes)}
          rescue
            e in [Protox.DecodingError, Protox.IllegalTagError, Protox.RequiredFieldsError] ->
              {:error, e}
          end
        end

        (
          @spec decode!(binary) :: struct | no_return
          def decode!(bytes) do
            parse_key_value(bytes, struct(Transmuter.Onnx.TypeProto.Optional))
          end
        )
      )

      (
        @spec parse_key_value(binary, struct) :: struct
        defp parse_key_value(<<>>, msg) do
          msg
        end

        defp parse_key_value(bytes, msg) do
          {field, rest} =
            case Protox.Decode.parse_key(bytes) do
              {0, _, _} ->
                raise %Protox.IllegalTagError{}

              {1, _, bytes} ->
                {len, bytes} = Protox.Varint.decode(bytes)
                {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)

                {[
                   elem_type:
                     Protox.MergeMessage.merge(
                       msg.elem_type,
                       Transmuter.Onnx.TypeProto.decode!(delimited)
                     )
                 ], rest}

              {tag, wire_type, rest} ->
                {value, rest} = Protox.Decode.parse_unknown(tag, wire_type, rest)

                {[
                   {msg.__struct__.unknown_fields_name(),
                    [value | msg.__struct__.unknown_fields(msg)]}
                 ], rest}
            end

          msg_updated = struct(msg, field)
          parse_key_value(rest, msg_updated)
        end
      )

      []
    )

    (
      @spec json_decode(iodata(), keyword()) :: {:ok, struct()} | {:error, any()}
      def json_decode(input, opts \\ []) do
        try do
          {:ok, json_decode!(input, opts)}
        rescue
          e in Protox.JsonDecodingError -> {:error, e}
        end
      end

      @spec json_decode!(iodata(), keyword()) :: struct() | no_return()
      def json_decode!(input, opts \\ []) do
        {json_library_wrapper, json_library} = Protox.JsonLibrary.get_library(opts, :decode)

        Protox.JsonDecode.decode!(
          input,
          Transmuter.Onnx.TypeProto.Optional,
          &json_library_wrapper.decode!(json_library, &1)
        )
      end

      @spec json_encode(struct(), keyword()) :: {:ok, iodata()} | {:error, any()}
      def json_encode(msg, opts \\ []) do
        try do
          {:ok, json_encode!(msg, opts)}
        rescue
          e in Protox.JsonEncodingError -> {:error, e}
        end
      end

      @spec json_encode!(struct(), keyword()) :: iodata() | no_return()
      def json_encode!(msg, opts \\ []) do
        {json_library_wrapper, json_library} = Protox.JsonLibrary.get_library(opts, :encode)
        Protox.JsonEncode.encode!(msg, &json_library_wrapper.encode!(json_library, &1))
      end
    )

    (
      @deprecated "Use fields_defs()/0 instead"
      @spec defs() :: %{
              required(non_neg_integer) => {atom, Protox.Types.kind(), Protox.Types.type()}
            }
      def defs() do
        %{1 => {:elem_type, {:scalar, nil}, {:message, Transmuter.Onnx.TypeProto}}}
      end

      @deprecated "Use fields_defs()/0 instead"
      @spec defs_by_name() :: %{
              required(atom) => {non_neg_integer, Protox.Types.kind(), Protox.Types.type()}
            }
      def defs_by_name() do
        %{elem_type: {1, {:scalar, nil}, {:message, Transmuter.Onnx.TypeProto}}}
      end
    )

    (
      @spec fields_defs() :: list(Protox.Field.t())
      def fields_defs() do
        [
          %{
            __struct__: Protox.Field,
            json_name: "elemType",
            kind: {:scalar, nil},
            label: :optional,
            name: :elem_type,
            tag: 1,
            type: {:message, Transmuter.Onnx.TypeProto}
          }
        ]
      end

      [
        @spec(field_def(atom) :: {:ok, Protox.Field.t()} | {:error, :no_such_field}),
        (
          def field_def(:elem_type) do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "elemType",
               kind: {:scalar, nil},
               label: :optional,
               name: :elem_type,
               tag: 1,
               type: {:message, Transmuter.Onnx.TypeProto}
             }}
          end

          def field_def("elemType") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "elemType",
               kind: {:scalar, nil},
               label: :optional,
               name: :elem_type,
               tag: 1,
               type: {:message, Transmuter.Onnx.TypeProto}
             }}
          end

          def field_def("elem_type") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "elemType",
               kind: {:scalar, nil},
               label: :optional,
               name: :elem_type,
               tag: 1,
               type: {:message, Transmuter.Onnx.TypeProto}
             }}
          end
        ),
        def field_def(_) do
          {:error, :no_such_field}
        end
      ]
    )

    (
      @spec unknown_fields(struct) :: [{non_neg_integer, Protox.Types.tag(), binary}]
      def unknown_fields(msg) do
        msg.__uf__
      end

      @spec unknown_fields_name() :: :__uf__
      def unknown_fields_name() do
        :__uf__
      end

      @spec clear_unknown_fields(struct) :: struct
      def clear_unknown_fields(msg) do
        struct!(msg, [{unknown_fields_name(), []}])
      end
    )

    (
      @spec required_fields() :: []
      def required_fields() do
        []
      end
    )

    (
      @spec syntax() :: atom()
      def syntax() do
        :proto2
      end
    )

    [
      @spec(default(atom) :: {:ok, boolean | integer | String.t() | float} | {:error, atom}),
      def default(:elem_type) do
        {:ok, nil}
      end,
      def default(_) do
        {:error, :no_such_field}
      end
    ]

    (
      @spec file_options() :: struct()
      def file_options() do
        file_options = %{
          __struct__: Protox.Google.Protobuf.FileOptions,
          __uf__: [],
          cc_enable_arenas: nil,
          cc_generic_services: nil,
          csharp_namespace: nil,
          deprecated: nil,
          go_package: nil,
          java_generate_equals_and_hash: nil,
          java_generic_services: nil,
          java_multiple_files: nil,
          java_outer_classname: nil,
          java_package: nil,
          java_string_check_utf8: nil,
          objc_class_prefix: nil,
          optimize_for: :LITE_RUNTIME,
          php_class_prefix: nil,
          php_generic_services: nil,
          php_metadata_namespace: nil,
          php_namespace: nil,
          py_generic_services: nil,
          ruby_package: nil,
          swift_prefix: nil,
          uninterpreted_option: []
        }

        case function_exported?(Google.Protobuf.FileOptions, :decode!, 1) do
          true ->
            bytes =
              file_options
              |> Protox.Google.Protobuf.FileOptions.encode!()
              |> :binary.list_to_bin()

            apply(Google.Protobuf.FileOptions, :decode!, [bytes])

          false ->
            file_options
        end
      end
    )
  end,
  defmodule Transmuter.Onnx.TypeProto.Sequence do
    @moduledoc false
    defstruct elem_type: nil, __uf__: []

    (
      (
        @spec encode(struct) :: {:ok, iodata} | {:error, any}
        def encode(msg) do
          try do
            {:ok, encode!(msg)}
          rescue
            e in [Protox.EncodingError, Protox.RequiredFieldsError] -> {:error, e}
          end
        end

        @spec encode!(struct) :: iodata | no_return
        def encode!(msg) do
          [] |> encode_elem_type(msg) |> encode_unknown_fields(msg)
        end
      )

      []

      [
        defp encode_elem_type(acc, msg) do
          try do
            case msg.elem_type do
              nil -> acc
              _ -> [acc, "\n", Protox.Encode.encode_message(msg.elem_type)]
            end
          rescue
            ArgumentError ->
              reraise Protox.EncodingError.new(:elem_type, "invalid field value"), __STACKTRACE__
          end
        end
      ]

      defp encode_unknown_fields(acc, msg) do
        Enum.reduce(msg.__struct__.unknown_fields(msg), acc, fn {tag, wire_type, bytes}, acc ->
          case wire_type do
            0 ->
              [acc, Protox.Encode.make_key_bytes(tag, :int32), bytes]

            1 ->
              [acc, Protox.Encode.make_key_bytes(tag, :double), bytes]

            2 ->
              len_bytes = bytes |> byte_size() |> Protox.Varint.encode()
              [acc, Protox.Encode.make_key_bytes(tag, :packed), len_bytes, bytes]

            5 ->
              [acc, Protox.Encode.make_key_bytes(tag, :float), bytes]
          end
        end)
      end
    )

    (
      (
        @spec decode(binary) :: {:ok, struct} | {:error, any}
        def decode(bytes) do
          try do
            {:ok, decode!(bytes)}
          rescue
            e in [Protox.DecodingError, Protox.IllegalTagError, Protox.RequiredFieldsError] ->
              {:error, e}
          end
        end

        (
          @spec decode!(binary) :: struct | no_return
          def decode!(bytes) do
            parse_key_value(bytes, struct(Transmuter.Onnx.TypeProto.Sequence))
          end
        )
      )

      (
        @spec parse_key_value(binary, struct) :: struct
        defp parse_key_value(<<>>, msg) do
          msg
        end

        defp parse_key_value(bytes, msg) do
          {field, rest} =
            case Protox.Decode.parse_key(bytes) do
              {0, _, _} ->
                raise %Protox.IllegalTagError{}

              {1, _, bytes} ->
                {len, bytes} = Protox.Varint.decode(bytes)
                {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)

                {[
                   elem_type:
                     Protox.MergeMessage.merge(
                       msg.elem_type,
                       Transmuter.Onnx.TypeProto.decode!(delimited)
                     )
                 ], rest}

              {tag, wire_type, rest} ->
                {value, rest} = Protox.Decode.parse_unknown(tag, wire_type, rest)

                {[
                   {msg.__struct__.unknown_fields_name(),
                    [value | msg.__struct__.unknown_fields(msg)]}
                 ], rest}
            end

          msg_updated = struct(msg, field)
          parse_key_value(rest, msg_updated)
        end
      )

      []
    )

    (
      @spec json_decode(iodata(), keyword()) :: {:ok, struct()} | {:error, any()}
      def json_decode(input, opts \\ []) do
        try do
          {:ok, json_decode!(input, opts)}
        rescue
          e in Protox.JsonDecodingError -> {:error, e}
        end
      end

      @spec json_decode!(iodata(), keyword()) :: struct() | no_return()
      def json_decode!(input, opts \\ []) do
        {json_library_wrapper, json_library} = Protox.JsonLibrary.get_library(opts, :decode)

        Protox.JsonDecode.decode!(
          input,
          Transmuter.Onnx.TypeProto.Sequence,
          &json_library_wrapper.decode!(json_library, &1)
        )
      end

      @spec json_encode(struct(), keyword()) :: {:ok, iodata()} | {:error, any()}
      def json_encode(msg, opts \\ []) do
        try do
          {:ok, json_encode!(msg, opts)}
        rescue
          e in Protox.JsonEncodingError -> {:error, e}
        end
      end

      @spec json_encode!(struct(), keyword()) :: iodata() | no_return()
      def json_encode!(msg, opts \\ []) do
        {json_library_wrapper, json_library} = Protox.JsonLibrary.get_library(opts, :encode)
        Protox.JsonEncode.encode!(msg, &json_library_wrapper.encode!(json_library, &1))
      end
    )

    (
      @deprecated "Use fields_defs()/0 instead"
      @spec defs() :: %{
              required(non_neg_integer) => {atom, Protox.Types.kind(), Protox.Types.type()}
            }
      def defs() do
        %{1 => {:elem_type, {:scalar, nil}, {:message, Transmuter.Onnx.TypeProto}}}
      end

      @deprecated "Use fields_defs()/0 instead"
      @spec defs_by_name() :: %{
              required(atom) => {non_neg_integer, Protox.Types.kind(), Protox.Types.type()}
            }
      def defs_by_name() do
        %{elem_type: {1, {:scalar, nil}, {:message, Transmuter.Onnx.TypeProto}}}
      end
    )

    (
      @spec fields_defs() :: list(Protox.Field.t())
      def fields_defs() do
        [
          %{
            __struct__: Protox.Field,
            json_name: "elemType",
            kind: {:scalar, nil},
            label: :optional,
            name: :elem_type,
            tag: 1,
            type: {:message, Transmuter.Onnx.TypeProto}
          }
        ]
      end

      [
        @spec(field_def(atom) :: {:ok, Protox.Field.t()} | {:error, :no_such_field}),
        (
          def field_def(:elem_type) do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "elemType",
               kind: {:scalar, nil},
               label: :optional,
               name: :elem_type,
               tag: 1,
               type: {:message, Transmuter.Onnx.TypeProto}
             }}
          end

          def field_def("elemType") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "elemType",
               kind: {:scalar, nil},
               label: :optional,
               name: :elem_type,
               tag: 1,
               type: {:message, Transmuter.Onnx.TypeProto}
             }}
          end

          def field_def("elem_type") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "elemType",
               kind: {:scalar, nil},
               label: :optional,
               name: :elem_type,
               tag: 1,
               type: {:message, Transmuter.Onnx.TypeProto}
             }}
          end
        ),
        def field_def(_) do
          {:error, :no_such_field}
        end
      ]
    )

    (
      @spec unknown_fields(struct) :: [{non_neg_integer, Protox.Types.tag(), binary}]
      def unknown_fields(msg) do
        msg.__uf__
      end

      @spec unknown_fields_name() :: :__uf__
      def unknown_fields_name() do
        :__uf__
      end

      @spec clear_unknown_fields(struct) :: struct
      def clear_unknown_fields(msg) do
        struct!(msg, [{unknown_fields_name(), []}])
      end
    )

    (
      @spec required_fields() :: []
      def required_fields() do
        []
      end
    )

    (
      @spec syntax() :: atom()
      def syntax() do
        :proto2
      end
    )

    [
      @spec(default(atom) :: {:ok, boolean | integer | String.t() | float} | {:error, atom}),
      def default(:elem_type) do
        {:ok, nil}
      end,
      def default(_) do
        {:error, :no_such_field}
      end
    ]

    (
      @spec file_options() :: struct()
      def file_options() do
        file_options = %{
          __struct__: Protox.Google.Protobuf.FileOptions,
          __uf__: [],
          cc_enable_arenas: nil,
          cc_generic_services: nil,
          csharp_namespace: nil,
          deprecated: nil,
          go_package: nil,
          java_generate_equals_and_hash: nil,
          java_generic_services: nil,
          java_multiple_files: nil,
          java_outer_classname: nil,
          java_package: nil,
          java_string_check_utf8: nil,
          objc_class_prefix: nil,
          optimize_for: :LITE_RUNTIME,
          php_class_prefix: nil,
          php_generic_services: nil,
          php_metadata_namespace: nil,
          php_namespace: nil,
          py_generic_services: nil,
          ruby_package: nil,
          swift_prefix: nil,
          uninterpreted_option: []
        }

        case function_exported?(Google.Protobuf.FileOptions, :decode!, 1) do
          true ->
            bytes =
              file_options
              |> Protox.Google.Protobuf.FileOptions.encode!()
              |> :binary.list_to_bin()

            apply(Google.Protobuf.FileOptions, :decode!, [bytes])

          false ->
            file_options
        end
      end
    )
  end,
  defmodule Transmuter.Onnx.TypeProto.SparseTensor do
    @moduledoc false
    defstruct elem_type: nil, shape: nil, __uf__: []

    (
      (
        @spec encode(struct) :: {:ok, iodata} | {:error, any}
        def encode(msg) do
          try do
            {:ok, encode!(msg)}
          rescue
            e in [Protox.EncodingError, Protox.RequiredFieldsError] -> {:error, e}
          end
        end

        @spec encode!(struct) :: iodata | no_return
        def encode!(msg) do
          [] |> encode_elem_type(msg) |> encode_shape(msg) |> encode_unknown_fields(msg)
        end
      )

      []

      [
        defp encode_elem_type(acc, msg) do
          try do
            case msg.elem_type do
              nil -> acc
              _ -> [acc, "\b", Protox.Encode.encode_int32(msg.elem_type)]
            end
          rescue
            ArgumentError ->
              reraise Protox.EncodingError.new(:elem_type, "invalid field value"), __STACKTRACE__
          end
        end,
        defp encode_shape(acc, msg) do
          try do
            case msg.shape do
              nil -> acc
              _ -> [acc, "\x12", Protox.Encode.encode_message(msg.shape)]
            end
          rescue
            ArgumentError ->
              reraise Protox.EncodingError.new(:shape, "invalid field value"), __STACKTRACE__
          end
        end
      ]

      defp encode_unknown_fields(acc, msg) do
        Enum.reduce(msg.__struct__.unknown_fields(msg), acc, fn {tag, wire_type, bytes}, acc ->
          case wire_type do
            0 ->
              [acc, Protox.Encode.make_key_bytes(tag, :int32), bytes]

            1 ->
              [acc, Protox.Encode.make_key_bytes(tag, :double), bytes]

            2 ->
              len_bytes = bytes |> byte_size() |> Protox.Varint.encode()
              [acc, Protox.Encode.make_key_bytes(tag, :packed), len_bytes, bytes]

            5 ->
              [acc, Protox.Encode.make_key_bytes(tag, :float), bytes]
          end
        end)
      end
    )

    (
      (
        @spec decode(binary) :: {:ok, struct} | {:error, any}
        def decode(bytes) do
          try do
            {:ok, decode!(bytes)}
          rescue
            e in [Protox.DecodingError, Protox.IllegalTagError, Protox.RequiredFieldsError] ->
              {:error, e}
          end
        end

        (
          @spec decode!(binary) :: struct | no_return
          def decode!(bytes) do
            parse_key_value(bytes, struct(Transmuter.Onnx.TypeProto.SparseTensor))
          end
        )
      )

      (
        @spec parse_key_value(binary, struct) :: struct
        defp parse_key_value(<<>>, msg) do
          msg
        end

        defp parse_key_value(bytes, msg) do
          {field, rest} =
            case Protox.Decode.parse_key(bytes) do
              {0, _, _} ->
                raise %Protox.IllegalTagError{}

              {1, _, bytes} ->
                {value, rest} = Protox.Decode.parse_int32(bytes)
                {[elem_type: value], rest}

              {2, _, bytes} ->
                {len, bytes} = Protox.Varint.decode(bytes)
                {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)

                {[
                   shape:
                     Protox.MergeMessage.merge(
                       msg.shape,
                       Transmuter.Onnx.TensorShapeProto.decode!(delimited)
                     )
                 ], rest}

              {tag, wire_type, rest} ->
                {value, rest} = Protox.Decode.parse_unknown(tag, wire_type, rest)

                {[
                   {msg.__struct__.unknown_fields_name(),
                    [value | msg.__struct__.unknown_fields(msg)]}
                 ], rest}
            end

          msg_updated = struct(msg, field)
          parse_key_value(rest, msg_updated)
        end
      )

      []
    )

    (
      @spec json_decode(iodata(), keyword()) :: {:ok, struct()} | {:error, any()}
      def json_decode(input, opts \\ []) do
        try do
          {:ok, json_decode!(input, opts)}
        rescue
          e in Protox.JsonDecodingError -> {:error, e}
        end
      end

      @spec json_decode!(iodata(), keyword()) :: struct() | no_return()
      def json_decode!(input, opts \\ []) do
        {json_library_wrapper, json_library} = Protox.JsonLibrary.get_library(opts, :decode)

        Protox.JsonDecode.decode!(
          input,
          Transmuter.Onnx.TypeProto.SparseTensor,
          &json_library_wrapper.decode!(json_library, &1)
        )
      end

      @spec json_encode(struct(), keyword()) :: {:ok, iodata()} | {:error, any()}
      def json_encode(msg, opts \\ []) do
        try do
          {:ok, json_encode!(msg, opts)}
        rescue
          e in Protox.JsonEncodingError -> {:error, e}
        end
      end

      @spec json_encode!(struct(), keyword()) :: iodata() | no_return()
      def json_encode!(msg, opts \\ []) do
        {json_library_wrapper, json_library} = Protox.JsonLibrary.get_library(opts, :encode)
        Protox.JsonEncode.encode!(msg, &json_library_wrapper.encode!(json_library, &1))
      end
    )

    (
      @deprecated "Use fields_defs()/0 instead"
      @spec defs() :: %{
              required(non_neg_integer) => {atom, Protox.Types.kind(), Protox.Types.type()}
            }
      def defs() do
        %{
          1 => {:elem_type, {:scalar, 0}, :int32},
          2 => {:shape, {:scalar, nil}, {:message, Transmuter.Onnx.TensorShapeProto}}
        }
      end

      @deprecated "Use fields_defs()/0 instead"
      @spec defs_by_name() :: %{
              required(atom) => {non_neg_integer, Protox.Types.kind(), Protox.Types.type()}
            }
      def defs_by_name() do
        %{
          elem_type: {1, {:scalar, 0}, :int32},
          shape: {2, {:scalar, nil}, {:message, Transmuter.Onnx.TensorShapeProto}}
        }
      end
    )

    (
      @spec fields_defs() :: list(Protox.Field.t())
      def fields_defs() do
        [
          %{
            __struct__: Protox.Field,
            json_name: "elemType",
            kind: {:scalar, 0},
            label: :optional,
            name: :elem_type,
            tag: 1,
            type: :int32
          },
          %{
            __struct__: Protox.Field,
            json_name: "shape",
            kind: {:scalar, nil},
            label: :optional,
            name: :shape,
            tag: 2,
            type: {:message, Transmuter.Onnx.TensorShapeProto}
          }
        ]
      end

      [
        @spec(field_def(atom) :: {:ok, Protox.Field.t()} | {:error, :no_such_field}),
        (
          def field_def(:elem_type) do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "elemType",
               kind: {:scalar, 0},
               label: :optional,
               name: :elem_type,
               tag: 1,
               type: :int32
             }}
          end

          def field_def("elemType") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "elemType",
               kind: {:scalar, 0},
               label: :optional,
               name: :elem_type,
               tag: 1,
               type: :int32
             }}
          end

          def field_def("elem_type") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "elemType",
               kind: {:scalar, 0},
               label: :optional,
               name: :elem_type,
               tag: 1,
               type: :int32
             }}
          end
        ),
        (
          def field_def(:shape) do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "shape",
               kind: {:scalar, nil},
               label: :optional,
               name: :shape,
               tag: 2,
               type: {:message, Transmuter.Onnx.TensorShapeProto}
             }}
          end

          def field_def("shape") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "shape",
               kind: {:scalar, nil},
               label: :optional,
               name: :shape,
               tag: 2,
               type: {:message, Transmuter.Onnx.TensorShapeProto}
             }}
          end

          []
        ),
        def field_def(_) do
          {:error, :no_such_field}
        end
      ]
    )

    (
      @spec unknown_fields(struct) :: [{non_neg_integer, Protox.Types.tag(), binary}]
      def unknown_fields(msg) do
        msg.__uf__
      end

      @spec unknown_fields_name() :: :__uf__
      def unknown_fields_name() do
        :__uf__
      end

      @spec clear_unknown_fields(struct) :: struct
      def clear_unknown_fields(msg) do
        struct!(msg, [{unknown_fields_name(), []}])
      end
    )

    (
      @spec required_fields() :: []
      def required_fields() do
        []
      end
    )

    (
      @spec syntax() :: atom()
      def syntax() do
        :proto2
      end
    )

    [
      @spec(default(atom) :: {:ok, boolean | integer | String.t() | float} | {:error, atom}),
      def default(:elem_type) do
        {:ok, 0}
      end,
      def default(:shape) do
        {:ok, nil}
      end,
      def default(_) do
        {:error, :no_such_field}
      end
    ]

    (
      @spec file_options() :: struct()
      def file_options() do
        file_options = %{
          __struct__: Protox.Google.Protobuf.FileOptions,
          __uf__: [],
          cc_enable_arenas: nil,
          cc_generic_services: nil,
          csharp_namespace: nil,
          deprecated: nil,
          go_package: nil,
          java_generate_equals_and_hash: nil,
          java_generic_services: nil,
          java_multiple_files: nil,
          java_outer_classname: nil,
          java_package: nil,
          java_string_check_utf8: nil,
          objc_class_prefix: nil,
          optimize_for: :LITE_RUNTIME,
          php_class_prefix: nil,
          php_generic_services: nil,
          php_metadata_namespace: nil,
          php_namespace: nil,
          py_generic_services: nil,
          ruby_package: nil,
          swift_prefix: nil,
          uninterpreted_option: []
        }

        case function_exported?(Google.Protobuf.FileOptions, :decode!, 1) do
          true ->
            bytes =
              file_options
              |> Protox.Google.Protobuf.FileOptions.encode!()
              |> :binary.list_to_bin()

            apply(Google.Protobuf.FileOptions, :decode!, [bytes])

          false ->
            file_options
        end
      end
    )
  end,
  defmodule Transmuter.Onnx.TypeProto.Tensor do
    @moduledoc false
    defstruct elem_type: nil, shape: nil, __uf__: []

    (
      (
        @spec encode(struct) :: {:ok, iodata} | {:error, any}
        def encode(msg) do
          try do
            {:ok, encode!(msg)}
          rescue
            e in [Protox.EncodingError, Protox.RequiredFieldsError] -> {:error, e}
          end
        end

        @spec encode!(struct) :: iodata | no_return
        def encode!(msg) do
          [] |> encode_elem_type(msg) |> encode_shape(msg) |> encode_unknown_fields(msg)
        end
      )

      []

      [
        defp encode_elem_type(acc, msg) do
          try do
            case msg.elem_type do
              nil -> acc
              _ -> [acc, "\b", Protox.Encode.encode_int32(msg.elem_type)]
            end
          rescue
            ArgumentError ->
              reraise Protox.EncodingError.new(:elem_type, "invalid field value"), __STACKTRACE__
          end
        end,
        defp encode_shape(acc, msg) do
          try do
            case msg.shape do
              nil -> acc
              _ -> [acc, "\x12", Protox.Encode.encode_message(msg.shape)]
            end
          rescue
            ArgumentError ->
              reraise Protox.EncodingError.new(:shape, "invalid field value"), __STACKTRACE__
          end
        end
      ]

      defp encode_unknown_fields(acc, msg) do
        Enum.reduce(msg.__struct__.unknown_fields(msg), acc, fn {tag, wire_type, bytes}, acc ->
          case wire_type do
            0 ->
              [acc, Protox.Encode.make_key_bytes(tag, :int32), bytes]

            1 ->
              [acc, Protox.Encode.make_key_bytes(tag, :double), bytes]

            2 ->
              len_bytes = bytes |> byte_size() |> Protox.Varint.encode()
              [acc, Protox.Encode.make_key_bytes(tag, :packed), len_bytes, bytes]

            5 ->
              [acc, Protox.Encode.make_key_bytes(tag, :float), bytes]
          end
        end)
      end
    )

    (
      (
        @spec decode(binary) :: {:ok, struct} | {:error, any}
        def decode(bytes) do
          try do
            {:ok, decode!(bytes)}
          rescue
            e in [Protox.DecodingError, Protox.IllegalTagError, Protox.RequiredFieldsError] ->
              {:error, e}
          end
        end

        (
          @spec decode!(binary) :: struct | no_return
          def decode!(bytes) do
            parse_key_value(bytes, struct(Transmuter.Onnx.TypeProto.Tensor))
          end
        )
      )

      (
        @spec parse_key_value(binary, struct) :: struct
        defp parse_key_value(<<>>, msg) do
          msg
        end

        defp parse_key_value(bytes, msg) do
          {field, rest} =
            case Protox.Decode.parse_key(bytes) do
              {0, _, _} ->
                raise %Protox.IllegalTagError{}

              {1, _, bytes} ->
                {value, rest} = Protox.Decode.parse_int32(bytes)
                {[elem_type: value], rest}

              {2, _, bytes} ->
                {len, bytes} = Protox.Varint.decode(bytes)
                {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)

                {[
                   shape:
                     Protox.MergeMessage.merge(
                       msg.shape,
                       Transmuter.Onnx.TensorShapeProto.decode!(delimited)
                     )
                 ], rest}

              {tag, wire_type, rest} ->
                {value, rest} = Protox.Decode.parse_unknown(tag, wire_type, rest)

                {[
                   {msg.__struct__.unknown_fields_name(),
                    [value | msg.__struct__.unknown_fields(msg)]}
                 ], rest}
            end

          msg_updated = struct(msg, field)
          parse_key_value(rest, msg_updated)
        end
      )

      []
    )

    (
      @spec json_decode(iodata(), keyword()) :: {:ok, struct()} | {:error, any()}
      def json_decode(input, opts \\ []) do
        try do
          {:ok, json_decode!(input, opts)}
        rescue
          e in Protox.JsonDecodingError -> {:error, e}
        end
      end

      @spec json_decode!(iodata(), keyword()) :: struct() | no_return()
      def json_decode!(input, opts \\ []) do
        {json_library_wrapper, json_library} = Protox.JsonLibrary.get_library(opts, :decode)

        Protox.JsonDecode.decode!(
          input,
          Transmuter.Onnx.TypeProto.Tensor,
          &json_library_wrapper.decode!(json_library, &1)
        )
      end

      @spec json_encode(struct(), keyword()) :: {:ok, iodata()} | {:error, any()}
      def json_encode(msg, opts \\ []) do
        try do
          {:ok, json_encode!(msg, opts)}
        rescue
          e in Protox.JsonEncodingError -> {:error, e}
        end
      end

      @spec json_encode!(struct(), keyword()) :: iodata() | no_return()
      def json_encode!(msg, opts \\ []) do
        {json_library_wrapper, json_library} = Protox.JsonLibrary.get_library(opts, :encode)
        Protox.JsonEncode.encode!(msg, &json_library_wrapper.encode!(json_library, &1))
      end
    )

    (
      @deprecated "Use fields_defs()/0 instead"
      @spec defs() :: %{
              required(non_neg_integer) => {atom, Protox.Types.kind(), Protox.Types.type()}
            }
      def defs() do
        %{
          1 => {:elem_type, {:scalar, 0}, :int32},
          2 => {:shape, {:scalar, nil}, {:message, Transmuter.Onnx.TensorShapeProto}}
        }
      end

      @deprecated "Use fields_defs()/0 instead"
      @spec defs_by_name() :: %{
              required(atom) => {non_neg_integer, Protox.Types.kind(), Protox.Types.type()}
            }
      def defs_by_name() do
        %{
          elem_type: {1, {:scalar, 0}, :int32},
          shape: {2, {:scalar, nil}, {:message, Transmuter.Onnx.TensorShapeProto}}
        }
      end
    )

    (
      @spec fields_defs() :: list(Protox.Field.t())
      def fields_defs() do
        [
          %{
            __struct__: Protox.Field,
            json_name: "elemType",
            kind: {:scalar, 0},
            label: :optional,
            name: :elem_type,
            tag: 1,
            type: :int32
          },
          %{
            __struct__: Protox.Field,
            json_name: "shape",
            kind: {:scalar, nil},
            label: :optional,
            name: :shape,
            tag: 2,
            type: {:message, Transmuter.Onnx.TensorShapeProto}
          }
        ]
      end

      [
        @spec(field_def(atom) :: {:ok, Protox.Field.t()} | {:error, :no_such_field}),
        (
          def field_def(:elem_type) do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "elemType",
               kind: {:scalar, 0},
               label: :optional,
               name: :elem_type,
               tag: 1,
               type: :int32
             }}
          end

          def field_def("elemType") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "elemType",
               kind: {:scalar, 0},
               label: :optional,
               name: :elem_type,
               tag: 1,
               type: :int32
             }}
          end

          def field_def("elem_type") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "elemType",
               kind: {:scalar, 0},
               label: :optional,
               name: :elem_type,
               tag: 1,
               type: :int32
             }}
          end
        ),
        (
          def field_def(:shape) do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "shape",
               kind: {:scalar, nil},
               label: :optional,
               name: :shape,
               tag: 2,
               type: {:message, Transmuter.Onnx.TensorShapeProto}
             }}
          end

          def field_def("shape") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "shape",
               kind: {:scalar, nil},
               label: :optional,
               name: :shape,
               tag: 2,
               type: {:message, Transmuter.Onnx.TensorShapeProto}
             }}
          end

          []
        ),
        def field_def(_) do
          {:error, :no_such_field}
        end
      ]
    )

    (
      @spec unknown_fields(struct) :: [{non_neg_integer, Protox.Types.tag(), binary}]
      def unknown_fields(msg) do
        msg.__uf__
      end

      @spec unknown_fields_name() :: :__uf__
      def unknown_fields_name() do
        :__uf__
      end

      @spec clear_unknown_fields(struct) :: struct
      def clear_unknown_fields(msg) do
        struct!(msg, [{unknown_fields_name(), []}])
      end
    )

    (
      @spec required_fields() :: []
      def required_fields() do
        []
      end
    )

    (
      @spec syntax() :: atom()
      def syntax() do
        :proto2
      end
    )

    [
      @spec(default(atom) :: {:ok, boolean | integer | String.t() | float} | {:error, atom}),
      def default(:elem_type) do
        {:ok, 0}
      end,
      def default(:shape) do
        {:ok, nil}
      end,
      def default(_) do
        {:error, :no_such_field}
      end
    ]

    (
      @spec file_options() :: struct()
      def file_options() do
        file_options = %{
          __struct__: Protox.Google.Protobuf.FileOptions,
          __uf__: [],
          cc_enable_arenas: nil,
          cc_generic_services: nil,
          csharp_namespace: nil,
          deprecated: nil,
          go_package: nil,
          java_generate_equals_and_hash: nil,
          java_generic_services: nil,
          java_multiple_files: nil,
          java_outer_classname: nil,
          java_package: nil,
          java_string_check_utf8: nil,
          objc_class_prefix: nil,
          optimize_for: :LITE_RUNTIME,
          php_class_prefix: nil,
          php_generic_services: nil,
          php_metadata_namespace: nil,
          php_namespace: nil,
          py_generic_services: nil,
          ruby_package: nil,
          swift_prefix: nil,
          uninterpreted_option: []
        }

        case function_exported?(Google.Protobuf.FileOptions, :decode!, 1) do
          true ->
            bytes =
              file_options
              |> Protox.Google.Protobuf.FileOptions.encode!()
              |> :binary.list_to_bin()

            apply(Google.Protobuf.FileOptions, :decode!, [bytes])

          false ->
            file_options
        end
      end
    )
  end,
  defmodule Transmuter.Onnx.ValueInfoProto do
    @moduledoc false
    defstruct name: nil, type: nil, doc_string: nil, metadata_props: [], __uf__: []

    (
      (
        @spec encode(struct) :: {:ok, iodata} | {:error, any}
        def encode(msg) do
          try do
            {:ok, encode!(msg)}
          rescue
            e in [Protox.EncodingError, Protox.RequiredFieldsError] -> {:error, e}
          end
        end

        @spec encode!(struct) :: iodata | no_return
        def encode!(msg) do
          []
          |> encode_name(msg)
          |> encode_type(msg)
          |> encode_doc_string(msg)
          |> encode_metadata_props(msg)
          |> encode_unknown_fields(msg)
        end
      )

      []

      [
        defp encode_name(acc, msg) do
          try do
            case msg.name do
              nil -> acc
              _ -> [acc, "\n", Protox.Encode.encode_string(msg.name)]
            end
          rescue
            ArgumentError ->
              reraise Protox.EncodingError.new(:name, "invalid field value"), __STACKTRACE__
          end
        end,
        defp encode_type(acc, msg) do
          try do
            case msg.type do
              nil -> acc
              _ -> [acc, "\x12", Protox.Encode.encode_message(msg.type)]
            end
          rescue
            ArgumentError ->
              reraise Protox.EncodingError.new(:type, "invalid field value"), __STACKTRACE__
          end
        end,
        defp encode_doc_string(acc, msg) do
          try do
            case msg.doc_string do
              nil -> acc
              _ -> [acc, "\x1A", Protox.Encode.encode_string(msg.doc_string)]
            end
          rescue
            ArgumentError ->
              reraise Protox.EncodingError.new(:doc_string, "invalid field value"), __STACKTRACE__
          end
        end,
        defp encode_metadata_props(acc, msg) do
          try do
            case msg.metadata_props do
              [] ->
                acc

              values ->
                [
                  acc,
                  Enum.reduce(values, [], fn value, acc ->
                    [acc, "\"", Protox.Encode.encode_message(value)]
                  end)
                ]
            end
          rescue
            ArgumentError ->
              reraise Protox.EncodingError.new(:metadata_props, "invalid field value"),
                      __STACKTRACE__
          end
        end
      ]

      defp encode_unknown_fields(acc, msg) do
        Enum.reduce(msg.__struct__.unknown_fields(msg), acc, fn {tag, wire_type, bytes}, acc ->
          case wire_type do
            0 ->
              [acc, Protox.Encode.make_key_bytes(tag, :int32), bytes]

            1 ->
              [acc, Protox.Encode.make_key_bytes(tag, :double), bytes]

            2 ->
              len_bytes = bytes |> byte_size() |> Protox.Varint.encode()
              [acc, Protox.Encode.make_key_bytes(tag, :packed), len_bytes, bytes]

            5 ->
              [acc, Protox.Encode.make_key_bytes(tag, :float), bytes]
          end
        end)
      end
    )

    (
      (
        @spec decode(binary) :: {:ok, struct} | {:error, any}
        def decode(bytes) do
          try do
            {:ok, decode!(bytes)}
          rescue
            e in [Protox.DecodingError, Protox.IllegalTagError, Protox.RequiredFieldsError] ->
              {:error, e}
          end
        end

        (
          @spec decode!(binary) :: struct | no_return
          def decode!(bytes) do
            parse_key_value(bytes, struct(Transmuter.Onnx.ValueInfoProto))
          end
        )
      )

      (
        @spec parse_key_value(binary, struct) :: struct
        defp parse_key_value(<<>>, msg) do
          msg
        end

        defp parse_key_value(bytes, msg) do
          {field, rest} =
            case Protox.Decode.parse_key(bytes) do
              {0, _, _} ->
                raise %Protox.IllegalTagError{}

              {1, _, bytes} ->
                {len, bytes} = Protox.Varint.decode(bytes)
                {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)
                {[name: Protox.Decode.validate_string!(delimited)], rest}

              {2, _, bytes} ->
                {len, bytes} = Protox.Varint.decode(bytes)
                {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)

                {[
                   type:
                     Protox.MergeMessage.merge(
                       msg.type,
                       Transmuter.Onnx.TypeProto.decode!(delimited)
                     )
                 ], rest}

              {3, _, bytes} ->
                {len, bytes} = Protox.Varint.decode(bytes)
                {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)
                {[doc_string: Protox.Decode.validate_string!(delimited)], rest}

              {4, _, bytes} ->
                {len, bytes} = Protox.Varint.decode(bytes)
                {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)

                {[
                   metadata_props:
                     msg.metadata_props ++
                       [Transmuter.Onnx.StringStringEntryProto.decode!(delimited)]
                 ], rest}

              {tag, wire_type, rest} ->
                {value, rest} = Protox.Decode.parse_unknown(tag, wire_type, rest)

                {[
                   {msg.__struct__.unknown_fields_name(),
                    [value | msg.__struct__.unknown_fields(msg)]}
                 ], rest}
            end

          msg_updated = struct(msg, field)
          parse_key_value(rest, msg_updated)
        end
      )

      []
    )

    (
      @spec json_decode(iodata(), keyword()) :: {:ok, struct()} | {:error, any()}
      def json_decode(input, opts \\ []) do
        try do
          {:ok, json_decode!(input, opts)}
        rescue
          e in Protox.JsonDecodingError -> {:error, e}
        end
      end

      @spec json_decode!(iodata(), keyword()) :: struct() | no_return()
      def json_decode!(input, opts \\ []) do
        {json_library_wrapper, json_library} = Protox.JsonLibrary.get_library(opts, :decode)

        Protox.JsonDecode.decode!(
          input,
          Transmuter.Onnx.ValueInfoProto,
          &json_library_wrapper.decode!(json_library, &1)
        )
      end

      @spec json_encode(struct(), keyword()) :: {:ok, iodata()} | {:error, any()}
      def json_encode(msg, opts \\ []) do
        try do
          {:ok, json_encode!(msg, opts)}
        rescue
          e in Protox.JsonEncodingError -> {:error, e}
        end
      end

      @spec json_encode!(struct(), keyword()) :: iodata() | no_return()
      def json_encode!(msg, opts \\ []) do
        {json_library_wrapper, json_library} = Protox.JsonLibrary.get_library(opts, :encode)
        Protox.JsonEncode.encode!(msg, &json_library_wrapper.encode!(json_library, &1))
      end
    )

    (
      @deprecated "Use fields_defs()/0 instead"
      @spec defs() :: %{
              required(non_neg_integer) => {atom, Protox.Types.kind(), Protox.Types.type()}
            }
      def defs() do
        %{
          1 => {:name, {:scalar, ""}, :string},
          2 => {:type, {:scalar, nil}, {:message, Transmuter.Onnx.TypeProto}},
          3 => {:doc_string, {:scalar, ""}, :string},
          4 => {:metadata_props, :unpacked, {:message, Transmuter.Onnx.StringStringEntryProto}}
        }
      end

      @deprecated "Use fields_defs()/0 instead"
      @spec defs_by_name() :: %{
              required(atom) => {non_neg_integer, Protox.Types.kind(), Protox.Types.type()}
            }
      def defs_by_name() do
        %{
          doc_string: {3, {:scalar, ""}, :string},
          metadata_props: {4, :unpacked, {:message, Transmuter.Onnx.StringStringEntryProto}},
          name: {1, {:scalar, ""}, :string},
          type: {2, {:scalar, nil}, {:message, Transmuter.Onnx.TypeProto}}
        }
      end
    )

    (
      @spec fields_defs() :: list(Protox.Field.t())
      def fields_defs() do
        [
          %{
            __struct__: Protox.Field,
            json_name: "name",
            kind: {:scalar, ""},
            label: :optional,
            name: :name,
            tag: 1,
            type: :string
          },
          %{
            __struct__: Protox.Field,
            json_name: "type",
            kind: {:scalar, nil},
            label: :optional,
            name: :type,
            tag: 2,
            type: {:message, Transmuter.Onnx.TypeProto}
          },
          %{
            __struct__: Protox.Field,
            json_name: "docString",
            kind: {:scalar, ""},
            label: :optional,
            name: :doc_string,
            tag: 3,
            type: :string
          },
          %{
            __struct__: Protox.Field,
            json_name: "metadataProps",
            kind: :unpacked,
            label: :repeated,
            name: :metadata_props,
            tag: 4,
            type: {:message, Transmuter.Onnx.StringStringEntryProto}
          }
        ]
      end

      [
        @spec(field_def(atom) :: {:ok, Protox.Field.t()} | {:error, :no_such_field}),
        (
          def field_def(:name) do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "name",
               kind: {:scalar, ""},
               label: :optional,
               name: :name,
               tag: 1,
               type: :string
             }}
          end

          def field_def("name") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "name",
               kind: {:scalar, ""},
               label: :optional,
               name: :name,
               tag: 1,
               type: :string
             }}
          end

          []
        ),
        (
          def field_def(:type) do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "type",
               kind: {:scalar, nil},
               label: :optional,
               name: :type,
               tag: 2,
               type: {:message, Transmuter.Onnx.TypeProto}
             }}
          end

          def field_def("type") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "type",
               kind: {:scalar, nil},
               label: :optional,
               name: :type,
               tag: 2,
               type: {:message, Transmuter.Onnx.TypeProto}
             }}
          end

          []
        ),
        (
          def field_def(:doc_string) do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "docString",
               kind: {:scalar, ""},
               label: :optional,
               name: :doc_string,
               tag: 3,
               type: :string
             }}
          end

          def field_def("docString") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "docString",
               kind: {:scalar, ""},
               label: :optional,
               name: :doc_string,
               tag: 3,
               type: :string
             }}
          end

          def field_def("doc_string") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "docString",
               kind: {:scalar, ""},
               label: :optional,
               name: :doc_string,
               tag: 3,
               type: :string
             }}
          end
        ),
        (
          def field_def(:metadata_props) do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "metadataProps",
               kind: :unpacked,
               label: :repeated,
               name: :metadata_props,
               tag: 4,
               type: {:message, Transmuter.Onnx.StringStringEntryProto}
             }}
          end

          def field_def("metadataProps") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "metadataProps",
               kind: :unpacked,
               label: :repeated,
               name: :metadata_props,
               tag: 4,
               type: {:message, Transmuter.Onnx.StringStringEntryProto}
             }}
          end

          def field_def("metadata_props") do
            {:ok,
             %{
               __struct__: Protox.Field,
               json_name: "metadataProps",
               kind: :unpacked,
               label: :repeated,
               name: :metadata_props,
               tag: 4,
               type: {:message, Transmuter.Onnx.StringStringEntryProto}
             }}
          end
        ),
        def field_def(_) do
          {:error, :no_such_field}
        end
      ]
    )

    (
      @spec unknown_fields(struct) :: [{non_neg_integer, Protox.Types.tag(), binary}]
      def unknown_fields(msg) do
        msg.__uf__
      end

      @spec unknown_fields_name() :: :__uf__
      def unknown_fields_name() do
        :__uf__
      end

      @spec clear_unknown_fields(struct) :: struct
      def clear_unknown_fields(msg) do
        struct!(msg, [{unknown_fields_name(), []}])
      end
    )

    (
      @spec required_fields() :: []
      def required_fields() do
        []
      end
    )

    (
      @spec syntax() :: atom()
      def syntax() do
        :proto2
      end
    )

    [
      @spec(default(atom) :: {:ok, boolean | integer | String.t() | float} | {:error, atom}),
      def default(:name) do
        {:ok, ""}
      end,
      def default(:type) do
        {:ok, nil}
      end,
      def default(:doc_string) do
        {:ok, ""}
      end,
      def default(:metadata_props) do
        {:error, :no_default_value}
      end,
      def default(_) do
        {:error, :no_such_field}
      end
    ]

    (
      @spec file_options() :: struct()
      def file_options() do
        file_options = %{
          __struct__: Protox.Google.Protobuf.FileOptions,
          __uf__: [],
          cc_enable_arenas: nil,
          cc_generic_services: nil,
          csharp_namespace: nil,
          deprecated: nil,
          go_package: nil,
          java_generate_equals_and_hash: nil,
          java_generic_services: nil,
          java_multiple_files: nil,
          java_outer_classname: nil,
          java_package: nil,
          java_string_check_utf8: nil,
          objc_class_prefix: nil,
          optimize_for: :LITE_RUNTIME,
          php_class_prefix: nil,
          php_generic_services: nil,
          php_metadata_namespace: nil,
          php_namespace: nil,
          py_generic_services: nil,
          ruby_package: nil,
          swift_prefix: nil,
          uninterpreted_option: []
        }

        case function_exported?(Google.Protobuf.FileOptions, :decode!, 1) do
          true ->
            bytes =
              file_options
              |> Protox.Google.Protobuf.FileOptions.encode!()
              |> :binary.list_to_bin()

            apply(Google.Protobuf.FileOptions, :decode!, [bytes])

          false ->
            file_options
        end
      end
    )
  end
]
