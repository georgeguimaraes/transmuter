defmodule Transmuter.MixProject do
  use Mix.Project

  @version "3.0.0"
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
    [mod: {Transmuter.Application, []}, extra_applications: [:logger]]
  end

  defp deps do
    [
      {:nx, "~> 1.0"},
      {:protox, "~> 2.1"},
      {:req, "~> 0.5"},
      {:telemetry, "~> 1.3"},
      {:exla, "~> 1.0", optional: true},
      {:emlx, "~> 0.5", optional: true},
      # EMLX 0.5.0 and EXLA 1.0.0 ask for fine ~> 0.1, but they do not compile with fine 0.2.
      # An optional dep does not limit a fine that EMLX or EXLA bring to an application.
      {:fine, "~> 0.1.6", runtime: false},
      {:stb_image, "~> 1.0", optional: true},
      {:tokenizers, "~> 0.5", optional: true},
      {:otera, "~> 0.1", optional: true},
      {:kino, "~> 0.14", optional: true},
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
      assets: %{"guides/assets" => "guides/assets"},
      extras: [
        "README.md": [title: "Overview"],
        "guides/getting_started.md": [title: "Getting started"],
        "guides/pipelines.md": [title: "Pipelines"],
        "guides/graph_breaks.md": [title: "Compiling and graph breaks"],
        "guides/language_models.md": [title: "Language models with a KV cache"],
        "guides/speech_and_audio.md": [title: "Speech and audio"],
        "guides/vision_and_multimodal.md": [title: "Vision and multimodal"],
        "guides/backends_and_performance.md": [title: "Backends and performance"],
        "guides/distributed.md": [title: "Distributed nodes"],
        "guides/telemetry.md": [title: "Telemetry in production"],
        "guides/supported_operators.md": [title: "Supported operators and data types"],
        "notebooks/chat_server.livemd": [title: "A chat server with ContinuousBatching"],
        "notebooks/embeddings_and_reranking.livemd": [
          title: "Search with embeddings and reranking"
        ],
        "notebooks/speech_to_text.livemd": [title: "Speech to text"]
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
          guides/distributed.md
          guides/telemetry.md
          guides/supported_operators.md
        ),
        Notebooks: ~w(
          notebooks/chat_server.livemd
          notebooks/embeddings_and_reranking.livemd
          notebooks/speech_to_text.livemd
        )
      ],
      groups_for_modules: [
        Pipelines: [
          Transmuter.Text,
          Transmuter.Vision,
          Transmuter.Audio,
          Transmuter.Tokenizer,
          Transmuter.Featurizer,
          Transmuter.Audio.Featurizer,
          Transmuter.ChatTemplate
        ],
        "Public API": [
          Transmuter,
          Transmuter.EMLX,
          Transmuter.Graph,
          Transmuter.Hub,
          Transmuter.HubError,
          Transmuter.Repo,
          Transmuter.Telemetry,
          Transmuter.Telemetry.Logger,
          Transmuter.UnknownValueError
        ],
        Internals: [
          Transmuter.Application,
          Transmuter.Serving,
          Transmuter.Generation,
          Transmuter.Generation.MergedDecoder,
          Transmuter.Text.FillMask,
          Transmuter.Text.Generation,
          Transmuter.Text.QuestionAnswering,
          Transmuter.Text.TextClassification,
          Transmuter.Text.TextEmbedding,
          Transmuter.Text.TextRanking,
          Transmuter.Text.TextToText,
          Transmuter.Text.TokenClassification,
          Transmuter.Text.ZeroShotClassification,
          Transmuter.Vision.ImageClassification,
          Transmuter.Vision.ImageEmbedding,
          Transmuter.Vision.ZeroShotImageClassification,
          Transmuter.Vision.ImageToText,
          Transmuter.Vision.ImageTextToText,
          Transmuter.Vision.ObjectDetection,
          Transmuter.Vision.ZeroShotObjectDetection,
          Transmuter.Vision.DepthEstimation,
          Transmuter.Vision.ImageSegmentation,
          Transmuter.Vision.BackgroundRemoval,
          Transmuter.Vision.MaskGeneration,
          Transmuter.Vision.Idefics3Processor,
          Transmuter.Vision.Qwen2VLProcessor,
          Transmuter.Audio.SpeechToText,
          Transmuter.Audio.WordTimestamps,
          Transmuter.Audio.AudioClassification,
          Transmuter.Audio.TextToSpeech,
          Transmuter.Fusion,
          Transmuter.GraphBreaks,
          Transmuter.Shared,
          Transmuter.Random,
          Transmuter.Ops,
          Transmuter.Ops.Attention,
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
