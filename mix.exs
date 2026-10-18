defmodule Transmuter.MixProject do
  use Mix.Project

  @version "0.1.0"
  @source_url "https://github.com/georgeguimaraes/transmuter"

  def project do
    [
      app: :transmuter,
      version: @version,
      elixir: "~> 1.18",
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      description: "Runs ONNX models, such as Hugging Face Hub exports, on any Nx backend.",
      package: package(),
      docs: docs(),
      source_url: @source_url
    ]
  end

  def application do
    [extra_applications: [:logger]]
  end

  defp deps do
    [
      {:nx, "~> 1.0"},
      {:protox, "~> 1.7"},
      {:req, "~> 0.5"},
      {:emlx, "~> 0.5", optional: true},
      {:stb_image, "~> 1.0", optional: true},
      {:tokenizers, "~> 0.5", only: [:dev, :test]},
      {:polaris, "~> 0.1", only: [:dev, :test]},
      {:credo, "~> 1.7", only: [:dev, :test], runtime: false},
      {:ex_doc, "~> 0.34", only: :dev, runtime: false}
    ]
  end

  defp package do
    [
      licenses: ["Apache-2.0"],
      links: %{"GitHub" => @source_url},
      files: ~w(lib priv mix.exs README.md LICENSE)
    ]
  end

  defp docs do
    [main: "readme", extras: ["README.md"], source_ref: "v#{@version}"]
  end
end
