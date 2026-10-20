defmodule Transmuter.MixProject do
  use Mix.Project

  @version "1.0.1"
  @source_url "https://github.com/georgeguimaraes/transmuter"

  def project do
    [
      app: :transmuter,
      version: @version,
      elixir: "~> 1.18",
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      description:
        "Runs Hugging Face ONNX models on any Nx backend, with no per-architecture code.",
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
      {:protox, "~> 2.1"},
      {:req, "~> 0.5"},
      {:telemetry, "~> 1.0"},
      {:emlx, "~> 0.5", optional: true},
      {:stb_image, "~> 1.0", optional: true},
      {:tokenizers, "~> 0.5", optional: true},
      {:otera, "~> 0.1", optional: true},
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
    [
      main: "Transmuter",
      extras: [
        "README.md": [title: "Overview"],
        "guides/getting_started.md": [title: "Getting started"],
        "guides/pipelines.md": [title: "Pipelines"],
        "guides/graph_breaks.md": [title: "Compiling and graph breaks"],
        "guides/language_models.md": [title: "Language models with a KV cache"],
        "guides/speech_and_audio.md": [title: "Speech and audio"],
        "guides/vision_and_multimodal.md": [title: "Vision and multimodal"],
        "guides/backends_and_performance.md": [title: "Backends and performance"],
        "guides/supported_operators.md": [title: "Supported operators and data types"]
      ],
      groups_for_extras: [
        Guides: ~w(
          guides/getting_started.md
          guides/pipelines.md
          guides/graph_breaks.md
          guides/language_models.md
          guides/speech_and_audio.md
          guides/vision_and_multimodal.md
          guides/backends_and_performance.md
          guides/supported_operators.md
        )
      ],
      groups_for_modules: [
        Pipelines: [
          Transmuter.Text,
          Transmuter.Vision,
          Transmuter.Audio,
          Transmuter.Tokenizer,
          Transmuter.Featurizer,
          Transmuter.ChatTemplate
        ],
        "Public API": [
          Transmuter,
          Transmuter.EMLX,
          Transmuter.Hub,
          Transmuter.Repo,
          Transmuter.Telemetry,
          Transmuter.Telemetry.Logger,
          Transmuter.UnknownValueError
        ],
        Internals: [
          Transmuter.Serving,
          Transmuter.Generation,
          Transmuter.Text.FillMask,
          Transmuter.Text.Generation,
          Transmuter.Text.QuestionAnswering,
          Transmuter.Text.TextClassification,
          Transmuter.Text.TextEmbedding,
          Transmuter.Text.TextToText,
          Transmuter.Text.TokenClassification,
          Transmuter.Text.ZeroShotClassification,
          Transmuter.Vision.ImageClassification,
          Transmuter.Vision.ImageEmbedding,
          Transmuter.Vision.ZeroShotImageClassification,
          Transmuter.Vision.ImageToText,
          Transmuter.Vision.ImageTextToText,
          Transmuter.Vision.Idefics3Processor,
          Transmuter.Vision.Qwen2VLProcessor,
          Transmuter.Audio.SpeechToText,
          Transmuter.Fusion,
          Transmuter.GraphBreaks,
          Transmuter.Shared,
          Transmuter.Ops,
          Transmuter.Ops.Elementwise,
          Transmuter.Ops.LinAlg,
          Transmuter.Ops.MoE,
          Transmuter.Ops.NN,
          Transmuter.Ops.Quantized,
          Transmuter.Ops.Random,
          Transmuter.Ops.Reduction,
          Transmuter.Ops.Sequence,
          Transmuter.Ops.Shape,
          Transmuter.Ops.Signal,
          Transmuter.Ops.Text,
          Transmuter.Ops.Transformer,
          Transmuter.Ops.Vision
        ]
      ],
      source_ref: "v#{@version}"
    ]
  end
end
