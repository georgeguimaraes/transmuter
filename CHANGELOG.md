# Changelog

## [1.0.1](https://github.com/georgeguimaraes/transmuter/compare/v1.0.0...v1.0.1) (2026-10-20)


### Bug Fixes

* Let concurrent downloads of one Hub file succeed ([8900c12](https://github.com/georgeguimaraes/transmuter/commit/8900c1247df60378e968f2e334159eabe58c6590))


### Continuous Integration

* Save the model cache only when the cached models change ([24462a1](https://github.com/georgeguimaraes/transmuter/commit/24462a183fcb903daa299c8f89eed77b8f7b2e85))

## [1.0.0](https://github.com/georgeguimaraes/transmuter/compare/v0.2.0...v1.0.0) (2026-10-20)


### Features

* Add image servings in Transmuter.Vision ([0295431](https://github.com/georgeguimaraes/transmuter/commit/0295431959177c7b433180497c010a6831289ea0))
* Add speech-to-text with Whisper ([a5e93dc](https://github.com/georgeguimaraes/transmuter/commit/a5e93dced04f75d6c87e503b487f6422b1bbb439))
* Add text-to-text generation for encoder-decoder models ([de0f08d](https://github.com/georgeguimaraes/transmuter/commit/de0f08d6856148ac182d08be4132fb5c9f104665))
* Add zero-shot classification and question answering servings ([d5ebf37](https://github.com/georgeguimaraes/transmuter/commit/d5ebf370b9c9c6d58795f98b10767f1ce4172ea9))
* Generate for a batch of inputs ([36d7246](https://github.com/georgeguimaraes/transmuter/commit/36d72461dda628e23d2fff7a916a2899c38d947f))
* Let one input of a text serving give more than one row ([1cff559](https://github.com/georgeguimaraes/transmuter/commit/1cff55952b240353040ebb15fcdb53563c8e705a))
* Render chat templates with Otera ([2f1189e](https://github.com/georgeguimaraes/transmuter/commit/2f1189ee41c63ecbea2a099f37d45173002df0db))
* Run half-precision exports in native f16 by default ([8693d6b](https://github.com/georgeguimaraes/transmuter/commit/8693d6b8231a440b565ac01734cc57c29ce4a1aa))
* Stream the tokens of generation ([f922a7d](https://github.com/georgeguimaraes/transmuter/commit/f922a7d549dd1e56bb7bd4afe9d2237b7ac036ee))
* Transcribe long audio with Whisper and give timestamps ([516996b](https://github.com/georgeguimaraes/transmuter/commit/516996b30c25cfccab91661c54ee6980053c1a00))


### Bug Fixes

* Add the Otera changes that commit 2f1189e did not include ([489baf8](https://github.com/georgeguimaraes/transmuter/commit/489baf89af95ce1529b657f11505493fc26384c9))


### Miscellaneous

* Release 1.0.0 ([6a53633](https://github.com/georgeguimaraes/transmuter/commit/6a53633185e4939b6cfe610b7367421b2ba8660e))
* Update protox to 2.1 and remove decimal ([790b843](https://github.com/georgeguimaraes/transmuter/commit/790b843e3ee1ab15c86c936f53ff0beb33dd6c7b))


### Documentation

* Correct the claims that recent changes made out of date ([9d25e8a](https://github.com/georgeguimaraes/transmuter/commit/9d25e8a9fce3131d81963eb5b0788d330c8c295a))
* Give the measured drift of the Qwen2-VL vision encoder in native f16 ([dd97638](https://github.com/georgeguimaraes/transmuter/commit/dd976385be60e44a3b17b0cd388a870f92d4952b))
* Point the getting started guide to the servings ([5f76652](https://github.com/georgeguimaraes/transmuter/commit/5f766525e4514b3f68341188f7c3ddf9c0fd5e2b))
* Say that the grow mode compiles each cache length ([64f1fea](https://github.com/georgeguimaraes/transmuter/commit/64f1feae4e6f6b77f6ceed2c2ce7edef04b1a0b3))


### Code Refactoring

* Move the generation loop into Transmuter.Generation ([e18be59](https://github.com/georgeguimaraes/transmuter/commit/e18be592b0d37d05618b557de63d83b102624ebf))


### Tests

* Give the concurrent streams test 60 s on slow runners ([3f3999c](https://github.com/georgeguimaraes/transmuter/commit/3f3999c83fe2758ea940c54ce3190df39879b869))


### Performance Improvements

* Build the large host tensors of a request from binaries ([43bfbe4](https://github.com/georgeguimaraes/transmuter/commit/43bfbe4e78af80d2ec54b41079659bb0a97fd9cf))
* Compile a graph segment from the templates of its arguments ([4568251](https://github.com/georgeguimaraes/transmuter/commit/456825102c18ae01b9f13d1f74798c5d72d04a73))
* Compile each cache length of the grow mode ([bf85c93](https://github.com/georgeguimaraes/transmuter/commit/bf85c930aa6594b7073b404af8b04beba211f86e))
* Find the arguments of a graph segment in one pass ([f985617](https://github.com/georgeguimaraes/transmuter/commit/f985617cfe2f13988e8bda13d43610c2f1aa3672))
* Fuse the normalizations and attention cores that exports write as separate operators ([5e1ffea](https://github.com/georgeguimaraes/transmuter/commit/5e1ffea6fc4ff2fcd35524df6b4aad91f96a7252))
* Keep the batch functions of a serving in persistent_term ([473e816](https://github.com/georgeguimaraes/transmuter/commit/473e816c00a2f818a33bfaaffb02085bd238191f))
* Keep the compiled closure of a graph segment small ([c0ff088](https://github.com/georgeguimaraes/transmuter/commit/c0ff0885789f21554aebfcd7779a33ce78461d29))
* Keep the compiled segments of a graph in persistent_term ([a738a87](https://github.com/georgeguimaraes/transmuter/commit/a738a87707e638f61fa1f4b10cac998c260b9428))
* Make the empty KV cache on the device ([1abfac6](https://github.com/georgeguimaraes/transmuter/commit/1abfac65eeb510ea14588d61dfd055d957488168))
* Run the attention core as a fused EMLX kernel ([babf726](https://github.com/georgeguimaraes/transmuter/commit/babf726b40c795297d71d1fafd0beae708e1c499))
* Run the normalization operators as fused EMLX kernels ([65b8265](https://github.com/georgeguimaraes/transmuter/commit/65b8265bdf7730dc599384fd25b2bfc611801af4))
* Start each generation task with a heap of 2 MB ([0bafca2](https://github.com/georgeguimaraes/transmuter/commit/0bafca2a60a9ee97ff18f5139a3699a20f5d6a73))
* Write the appended KV cache rows in one compiled call ([864e81a](https://github.com/georgeguimaraes/transmuter/commit/864e81a880bb1e68d3d8238672ef644578d7fdf9))

## [0.2.0](https://github.com/georgeguimaraes/transmuter/compare/v0.1.1...v0.2.0) (2026-10-19)


### Features

* Add chat templates and text generation ([089cb95](https://github.com/georgeguimaraes/transmuter/commit/089cb959dee1605f97e4e6aed1a33302da1113ab))
* Add com.microsoft MoE and QMoE operators ([9b5d562](https://github.com/georgeguimaraes/transmuter/commit/9b5d56299c3c51be1f0b8e0e0a3fcb0177fc0e7a))
* Add text task servings with a tokenizer ([97e7157](https://github.com/georgeguimaraes/transmuter/commit/97e715730e361fd167bbc4f7dac1f29f5b404404))
* Choose the compute type in find_divergence.exs ([83ac43c](https://github.com/georgeguimaraes/transmuter/commit/83ac43c1e60c6c44985284f37e5c2ed56a79caa3))
* Emit telemetry events for load, compile, run and graph breaks ([eb33cce](https://github.com/georgeguimaraes/transmuter/commit/eb33cce3a8d1bdc6e03d5063c0db4a9af33e5966))
* Load FP8 tensors ([ab71f57](https://github.com/georgeguimaraes/transmuter/commit/ab71f57e175451246c50ed827b68bc6e77faa5f9))
* Read the files and variants of a model repository ([1a9a042](https://github.com/georgeguimaraes/transmuter/commit/1a9a0424425dd83225025383c8e14584cecfc565))
* Set the divergence threshold in find_divergence.exs ([a583afa](https://github.com/georgeguimaraes/transmuter/commit/a583afaf3b39aef2f94efe3cf0017a5266f369f2))


### Bug Fixes

* Keep Hub errors out of the cache and work offline ([6b649ab](https://github.com/georgeguimaraes/transmuter/commit/6b649ab4b5e8d5bc431bbfccfcd1f5574efd28cc))
* Read external data in Constant values and subgraphs ([0f907e4](https://github.com/georgeguimaraes/transmuter/commit/0f907e4ff0933e31e81d507d8a7c49650176ac97))
* Report the cases that run_cases.py skips ([bf00308](https://github.com/georgeguimaraes/transmuter/commit/bf0030870289a137ec1153610fad9ea95214e1d1))
* Run half-precision exports in f32 by default ([d035284](https://github.com/georgeguimaraes/transmuter/commit/d035284557d68f9c5c1d761ff535f78ee19c51d5))
* Run If, Loop and Scan nodes in find_divergence.exs ([7f13ded](https://github.com/georgeguimaraes/transmuter/commit/7f13ded94715a314c6a49b219d374cc402053115))


### Miscellaneous

* Set the next release to 0.2.0 ([d080a25](https://github.com/georgeguimaraes/transmuter/commit/d080a254d0b7fbb1c2b468ad6882be5487f42166))


### Documentation

* Add guides and document the public API ([2a8c14d](https://github.com/georgeguimaraes/transmuter/commit/2a8c14dfdcd06a3c5141cea08ac1217fa5b757f0))
* Add the models tested today to the README ([fd31449](https://github.com/georgeguimaraes/transmuter/commit/fd31449311149bcbe117bcfc79dd1d71e9f5f502))
* Document the internal helpers in [@doc](https://github.com/doc) ([c8f1b6d](https://github.com/georgeguimaraes/transmuter/commit/c8f1b6d0b40323da021294c6aa4f929d448ff641))
* Give the drift of a deep graph in native f16 ([f677d59](https://github.com/georgeguimaraes/transmuter/commit/f677d5951a63ef5fa4ad1aba4553257e26926dff))
* Link ONNX to its site in the README ([ffa788e](https://github.com/georgeguimaraes/transmuter/commit/ffa788ea9a3cac0555770f13232db9761d1195d4))
* Read the image preprocessing from preprocessor_config.json ([1b76de4](https://github.com/georgeguimaraes/transmuter/commit/1b76de4d08ec48ee2a03e22d2e39c0ea225bc978))
* Show the text servings first and add a pipelines guide ([3c1740c](https://github.com/georgeguimaraes/transmuter/commit/3c1740c51513dc7274093be277d488d3477a88ac))
* Write the module comments as moduledocs ([ed92a1f](https://github.com/georgeguimaraes/transmuter/commit/ed92a1f59afc4b8e9f5d422bf9a7834aaab01c0f))

## [0.1.1](https://github.com/georgeguimaraes/transmuter/compare/v0.1.0...v0.1.1) (2026-10-19)


### Features

* Add a :cache_size option to compile/3 ([1754311](https://github.com/georgeguimaraes/transmuter/commit/1754311ae6ab38a482a4b9e8b2791307fd4c9337))
* Add a Kokoro text-to-speech example with duration buckets ([25993a5](https://github.com/georgeguimaraes/transmuter/commit/25993a53978a753569b76323ea970c74383da018))
* Add NegativeLogLikelihoodLoss and SoftmaxCrossEntropyLoss ([0e199a0](https://github.com/georgeguimaraes/transmuter/commit/0e199a0a7d1bf7bd630dbecc9794e06a6a214f92))
* Add NonMaxSuppression, Unique and Compress ([ac8fd3d](https://github.com/georgeguimaraes/transmuter/commit/ac8fd3d6089bc612dd50beefe0907be5605c980d))
* Add string operators and ImageDecoder ([fe4bb7e](https://github.com/georgeguimaraes/transmuter/commit/fe4bb7e656ba8bcd1d5d1f0339d070babd45152c))
* Add the QLinearAdd and QLinearMul contrib operators ([d0c18ff](https://github.com/georgeguimaraes/transmuter/commit/d0c18ff7bfd48848e91db47edc8075a99b977c1c))
* Add the random operators ([26309ec](https://github.com/georgeguimaraes/transmuter/commit/26309ecb0486dcbe433413844ce66b955207054c))
* Add Upsample, Scatter, MaxUnpool, BitCast, TensorScatter, CausalConvWithState and LinearAttention ([93748e0](https://github.com/georgeguimaraes/transmuter/commit/93748e0a64b06b3b873263b532feae6a94b2ea73))
* Break the graph at values that are only known at run time ([0a99e9b](https://github.com/georgeguimaraes/transmuter/commit/0a99e9b2015ba9d45985acb15f603479f2aefbbb))
* Evict the least recently used segments after a graph break ([e727dab](https://github.com/georgeguimaraes/transmuter/commit/e727dabc33cdd7f1fc99ccc00faa1a27213012d4))
* Hold strings, sequences and empty optionals in the graph ([d248a79](https://github.com/georgeguimaraes/transmuter/commit/d248a79ab6dd6dad929768215367eeee321e9c33))
* Keep the shape of zero-size values ([39b4a80](https://github.com/georgeguimaraes/transmuter/commit/39b4a80c256256a6a58d6adf5156423d4c6380c8))
* Let a compiled function take inputs of other shapes ([c5930f9](https://github.com/georgeguimaraes/transmuter/commit/c5930f914adf456b88490b5e7c5aac9209ce0e85))
* Pass zero-size values through shape and index operators ([e90688e](https://github.com/georgeguimaraes/transmuter/commit/e90688e3106c6ed7f71ab800c69b33932e57ca40))
* Raise one UnknownValueError for values that are only known at run time ([2710f35](https://github.com/georgeguimaraes/transmuter/commit/2710f352623b22137f9f99898e0fc8eecd0ebb09))
* Support ONNX sequence and optional operators ([5df99f0](https://github.com/georgeguimaraes/transmuter/commit/5df99f0428f8bb97a44fc0c23ffca68eaaaed070))


### Bug Fixes

* Accept a Loop trip count of shape [1] ([f34e686](https://github.com/georgeguimaraes/transmuter/commit/f34e6864227e9a18f8aa3059658a5a53b785a0d4))
* Format the random distribution check the same way on Elixir 1.18 ([e75215f](https://github.com/georgeguimaraes/transmuter/commit/e75215f0cf705c33373f4b2585b6b8620eba3de3))
* Make find_divergence.exs exact for integer outputs and overrides ([e5d074b](https://github.com/georgeguimaraes/transmuter/commit/e5d074b7e010177e2a6a50a171dba5abbe0a77ba))
* Make integer convolutions exact and support Conv auto_pad SAME ([bea89b4](https://github.com/georgeguimaraes/transmuter/commit/bea89b4a1b660eb2c27d1b13ab41a35e829f1a95))
* Match ONNX Runtime for QLinearAdd, QLinearMul and QLinearConv ([4e53ed7](https://github.com/georgeguimaraes/transmuter/commit/4e53ed78904d08220106a1cc61267a35f776a6f0))
* Pass zero-size values through index, elementwise and MatMul operators ([ff0ec30](https://github.com/georgeguimaraes/transmuter/commit/ff0ec305e22b199f57f0b72ed8f2159f63b6ff4e))
* Put literal values in the EMLX compile cache key ([11371fe](https://github.com/georgeguimaraes/transmuter/commit/11371feb8ccbb461367b5682142b61f99b4fa7cf))
* Remove the 4.8 kHz tone from the Kokoro example ([d29be98](https://github.com/georgeguimaraes/transmuter/commit/d29be9880693bc81a5c0c8a0f128e2f15b6a4ff4))
* Run long Loop and Scan nodes as Nx while loops when compiled ([fa4a298](https://github.com/georgeguimaraes/transmuter/commit/fa4a298cd62d1f1555d40ea2722b86222d40b92f))
* Run long recurrences as Nx while loops when compiled ([f1b0501](https://github.com/georgeguimaraes/transmuter/commit/f1b0501d169bd7ea4c73f4380ea5a53ad7b2034f))
* Run the Florence-2 vision encoder and the SmolVLM prefill ([84342cd](https://github.com/georgeguimaraes/transmuter/commit/84342cdaea403813545783c7628aec9bcaabb74b))
* Use output_half_pixel for RoiAlign before opset 16 ([247e8aa](https://github.com/georgeguimaraes/transmuter/commit/247e8aae6c52ef6edb39b84441068978027f3d5d))


### Miscellaneous

* Compare whole integer tensors in the coverage debug tools ([1c51520](https://github.com/georgeguimaraes/transmuter/commit/1c5152046c08940fdc50f88da1efc26e31a0b2ee))
* Format mix.exs ([1f3dbe7](https://github.com/georgeguimaraes/transmuter/commit/1f3dbe7836f0db4de5079c298062a23a1df4c6ec))
* Keep the checkout path out of the op fixture metadata ([8e51aa7](https://github.com/georgeguimaraes/transmuter/commit/8e51aa74674f801328939ad73c475cad0cad6896))
* Name the coverage and fixture scripts by what they do ([5ab0488](https://github.com/georgeguimaraes/transmuter/commit/5ab0488b7c3e2fa5aa1988cb382969f69aa9ee56))
* Remove comments that restate the code ([059702e](https://github.com/georgeguimaraes/transmuter/commit/059702e44bbdaec2ce81f1dc9f33bc9c44275e5f))


### Documentation

* Add a gold coin to the README title ([8c200d0](https://github.com/georgeguimaraes/transmuter/commit/8c200d09cd232fe6fd2218178edff5fa73b8be94))
* Add a Kokoro section to the README ([3f54b81](https://github.com/georgeguimaraes/transmuter/commit/3f54b81cad1ec054b5203760b339557f00acb32f))
* Describe strings, sequences, random values and zero-size tensors ([33af035](https://github.com/georgeguimaraes/transmuter/commit/33af0355d8355a42f6bc14c2ee68ffd55a406c36))
* Explain graph breaks and the two models that do not match value by value ([5f9f61e](https://github.com/georgeguimaraes/transmuter/commit/5f9f61e3c7f9cfff5c6c6ea9bdc8f78988a83c27))
* Explain that Transmuter implements the ONNX operators in Nx ([fb8215c](https://github.com/georgeguimaraes/transmuter/commit/fb8215cde42d509bb590afb0e3a48e482c7d2834))
* Explain the model outputs and the operator tests in the README ([27dd91e](https://github.com/georgeguimaraes/transmuter/commit/27dd91e4eaf94c5b1fcd921d9a6d829fcda221b6))
* Keep 64 compiled decoders in the Kokoro example ([72a6f15](https://github.com/georgeguimaraes/transmuter/commit/72a6f1576d4a9d061be7e4759e4fcbe7ebe8f133))
* Limit the MLX buffer cache and update the Kokoro timings ([00cd10e](https://github.com/georgeguimaraes/transmuter/commit/00cd10e916663c1c6632fcdd6dea819582e7409a))
* List the tested models in the README ([ed4d912](https://github.com/georgeguimaraes/transmuter/commit/ed4d91289f5f064d72f2b2116827667768e7d0f7))
* Say that EMLX runs compiled functions on the GPU on a Mac ([012056e](https://github.com/georgeguimaraes/transmuter/commit/012056ea009920f4ab9b698e4124e98a8967d180))
* Update the Kokoro timings after the performance changes ([278abfd](https://github.com/georgeguimaraes/transmuter/commit/278abfd7546e8e968bc3d23738c3a55667398d1a))
* Use the profile tagline in the README, Hex description and moduledoc ([5e71e50](https://github.com/georgeguimaraes/transmuter/commit/5e71e50f3eee34d0473a263c439c4e132e4fd13e))


### Code Refactoring

* Share one normalize helper across the normalization ops ([d653d3d](https://github.com/georgeguimaraes/transmuter/commit/d653d3d2c42545ecf45fa3ca99792163673b0850))
* Split the operators in ops.ex into modules ([65c5664](https://github.com/georgeguimaraes/transmuter/commit/65c56643609735ac06b2ea2ad04dae41c34943dc))


### Tests

* Add one test for each operator variant ([16e7f96](https://github.com/georgeguimaraes/transmuter/commit/16e7f96e2fa3fd41ce6f297bd810b87b066077a8))


### Performance Improvements

* Frame STFT and pad with slices instead of gathers ([b981f1e](https://github.com/georgeguimaraes/transmuter/commit/b981f1eaad5cf4e7f34b219d499b5b257109d5cd))
* Multiply a matrix with a batch of matrices without a broadcast ([08c3c45](https://github.com/georgeguimaraes/transmuter/commit/08c3c45dde233962bd49047a596ff7da4fad20d2))
* Run a grouped ConvTranspose as one grouped conv ([ef8f50a](https://github.com/georgeguimaraes/transmuter/commit/ef8f50af9d1bacc603e746712271a9c74ce5eff0))
* Share keys and values across grouped query heads without copies ([3ba6f92](https://github.com/georgeguimaraes/transmuter/commit/3ba6f929ce536b12115ce4f2be03a4952d0d0d3e))
