# Changelog

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
