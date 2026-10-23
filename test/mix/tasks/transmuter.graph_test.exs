defmodule Mix.Tasks.Transmuter.GraphTest do
  use ExUnit.Case, async: true

  alias Mix.Tasks.Transmuter.Graph

  test "dummy inputs take the declared types and shapes, with sizes for symbolic dims" do
    types = %{
      "input_ids" => {:s64, ["batch", "sequence_length"]},
      "pixel_values" => {:f32, ["batch", 3, nil]},
      "text" => {:string, [2]},
      "states" => nil
    }

    inputs = Graph.dummy_inputs(types, %{"sequence_length" => 16}, Nx.BinaryBackend)

    assert {Nx.type(inputs["input_ids"]), Nx.shape(inputs["input_ids"])} == {{:s, 64}, {1, 16}}

    assert {Nx.type(inputs["pixel_values"]), Nx.shape(inputs["pixel_values"])} ==
             {{:f, 32}, {1, 3, 1}}

    assert Nx.to_flat_list(inputs["pixel_values"]) == [1.0, 1.0, 1.0]
    assert inputs["text"] == {:strings, {2}, ["a", "a"]}
    assert inputs["states"] == :none
  end

  test "--dim takes NAME=SIZE" do
    assert Graph.parse_dims(["batch=2", "sequence_length=16"]) == %{
             "batch" => 2,
             "sequence_length" => 16
           }

    for bad <- ["batch", "batch=0", "batch=two", "batch=2x"] do
      assert_raise Mix.Error, fn -> Graph.parse_dims([bad]) end
    end
  end
end
