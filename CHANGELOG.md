# Changelog

## [2.0.0](https://github.com/georgeguimaraes/transmuter/compare/v1.0.1...v2.0.0) (2026-10-24)


### ⚠ BREAKING CHANGES

* Transmuter.EMLX.params/2 is removed. Call Transmuter.EMLX.move/2, which returns the model with its params on EMLX and the packed MatMulNBits nodes changed to match, and use that model: `model = Transmuter.EMLX.move(model, device: :gpu)`.
* Transmuter.Generation.generate/5 returns each token as a list, [id] or [id, logprob] with token_logprobs: true. Transmuter.Audio.WordTimestamps.words/3 returns {text, elements} for each word. The :init function of Transmuter.Serving.generation/4 returns lists of integers, f32 values and lists of them, and :postprocess gets them as a flat list of integers, with each f32 value as its bits.

### Features

* Add a prepare step to the servings of Transmuter.Serving ([cf9ea1e](https://github.com/georgeguimaraes/transmuter/commit/cf9ea1ef4ace3622f567343c528ff6458e1d5637))
* Add audio classification to Transmuter.Audio ([3735e6a](https://github.com/georgeguimaraes/transmuter/commit/3735e6a5f1f76d897c725da3b07167a22eb36094))
* Add background removal to Transmuter.Vision ([d12b5b6](https://github.com/georgeguimaraes/transmuter/commit/d12b5b6f3a88f084c4aee91e69470168b5d7d231))
* Add beam search to Transmuter.Generation ([bc98483](https://github.com/georgeguimaraes/transmuter/commit/bc98483df06e42b4cbf882d0802d28f9f8eb3706))
* Add depth estimation to Transmuter.Vision ([4efc993](https://github.com/georgeguimaraes/transmuter/commit/4efc9931245d5ca8c5c0933bbb4fbaebbdeb47e6))
* Add EXLA as optional backend dependency ([03bf9a6](https://github.com/georgeguimaraes/transmuter/commit/03bf9a684914ad36aaa0bb40e9a339d19b2a5079))
* Add Florence-2 to Transmuter.Vision.image_to_text ([a978950](https://github.com/georgeguimaraes/transmuter/commit/a9789508ae65133891ac06770fbdbff3a8bbee8e))
* Add image segmentation to Transmuter.Vision ([5bd24a8](https://github.com/georgeguimaraes/transmuter/commit/5bd24a89ceac1e66353eab120ecd98f16c454df6))
* Add image to image to Transmuter.Vision ([0dd2f9c](https://github.com/georgeguimaraes/transmuter/commit/0dd2f9c80d09d3f79221533b4f48667f9d881525))
* Add mask generation to Transmuter.Vision ([81949e9](https://github.com/georgeguimaraes/transmuter/commit/81949e9c18dbcdf575c994bcd7dbdfeb21a41441))
* Add mix transmuter.graph to write the graph view of a model ([a0aec6e](https://github.com/georgeguimaraes/transmuter/commit/a0aec6e5720d2d811d18dfa3551f6be435830cee))
* Add NLLB and M2M100 translation to text_to_text ([738b2a2](https://github.com/georgeguimaraes/transmuter/commit/738b2a206fe7267e13a099b181f5f80c77b49813))
* Add offline mode to Transmuter.Hub ([3c7a29e](https://github.com/georgeguimaraes/transmuter/commit/3c7a29e938f646b30a50c3f6e7f3565f78a83db6))
* Add the :revision option to Transmuter.Hub ([5b37438](https://github.com/georgeguimaraes/transmuter/commit/5b3743816df415d0815ba07001898a6082a8f956))
* Add the Gemma 3 image processor for image-text-to-text ([0bb2314](https://github.com/georgeguimaraes/transmuter/commit/0bb23140b7737379298c0fde9b2f8678d1af5adf))
* Add the logits processors that read the tokens of a row ([94bb905](https://github.com/georgeguimaraes/transmuter/commit/94bb9050d6804545dbd713c05417cc38c79063e6))
* Add the Pixtral processor of Mistral 3 models ([b077c08](https://github.com/georgeguimaraes/transmuter/commit/b077c08462d3eacd7d6e2472a10d7ec218e90d0d))
* Add Transmuter.Graph to show the graph with what Transmuter does ([3e11cba](https://github.com/georgeguimaraes/transmuter/commit/3e11cbaad14aa7445763d55e9811827c7a146846))
* Add Transmuter.Text.text_ranking/2 to rank documents for a query ([04a107d](https://github.com/georgeguimaraes/transmuter/commit/04a107d225acb8054d5145f4d6ca7634e49b8c8d))
* Add word timestamps to Transmuter.Audio.speech_to_text ([fe27084](https://github.com/georgeguimaraes/transmuter/commit/fe27084aa7453063e8db2a7ac3f0383a762cc36f))
* Add zero-shot audio classification to Transmuter.Audio ([8819473](https://github.com/georgeguimaraes/transmuter/commit/8819473f7ca731aab872e19f55d1a504b9c034f0))
* Add zero-shot entity recognition with GLiNER ([603a5dc](https://github.com/georgeguimaraes/transmuter/commit/603a5dc8f4baeec375d1fa8920b9ac31b5956118))
* Add zero-shot object detection to Transmuter.Vision ([7d85d86](https://github.com/georgeguimaraes/transmuter/commit/7d85d867dd45ac77cb80bc68294a57e9646b7811))
* Answer chats about images with image_text_to_text ([4211aba](https://github.com/georgeguimaraes/transmuter/commit/4211abaa2190f1dad6a2aa14a502a1ce53d1a7d1))
* Answer chats about images with Qwen2-VL ([a27cca6](https://github.com/georgeguimaraes/transmuter/commit/a27cca6068c640bbb1af315a7c5978fe09ee92b5))
* Caption images with image_to_text ([45b4492](https://github.com/georgeguimaraes/transmuter/commit/45b4492993f84a6d69edaa2969fd869f5c980b0d))
* Clean up tokenization spaces in Tokenizer.decode ([db63c5d](https://github.com/georgeguimaraes/transmuter/commit/db63c5da33b2b71a96ae3591511b6afe1b2d7e99))
* Detect objects with object_detection ([b8be3ee](https://github.com/georgeguimaraes/transmuter/commit/b8be3eef42de7569c9408ee07bb8879227aa899f))
* Download a Hub file again when its ETag changes ([3120e24](https://github.com/georgeguimaraes/transmuter/commit/3120e24e71fb785196bf95034dadff925471fe91))
* Emit telemetry events for the Hub cache and the EXLA disk cache ([1cb3c0a](https://github.com/georgeguimaraes/transmuter/commit/1cb3c0a73ccda7fbdfab48a000c42a834a0054bf))
* Generate from token embeddings with image features ([83b992c](https://github.com/georgeguimaraes/transmuter/commit/83b992c373917132bd2d03733186b6a680a59ce9))
* Give fixed values from random operators with noise: false ([b753222](https://github.com/georgeguimaraes/transmuter/commit/b753222c9261c580fb4e9d48a6e50aee1de66d56))
* Give Generation the positions of each prompt ([ac5c6aa](https://github.com/georgeguimaraes/transmuter/commit/ac5c6aa9570b928e2db12fcba610989aa627e548))
* Give the log-probability of each generated token ([c5e3550](https://github.com/georgeguimaraes/transmuter/commit/c5e355038318ff60cc456bda435b5750ac99a741))
* Give the ONNX attributes of each node in the graph view ([b42da56](https://github.com/georgeguimaraes/transmuter/commit/b42da5609e4ad8c614b8357675b6bf062c8dc89c))
* Give the reason of a graph break in the segment compile event ([2735fff](https://github.com/georgeguimaraes/transmuter/commit/2735fff1fa1d2224dcfef9cb9af94a2938d91316))
* Give Transmuter.Featurizer the SAM image processors ([6e46550](https://github.com/georgeguimaraes/transmuter/commit/6e4655098befdf7ebd6f684823460e6699a7496b))
* Give UnknownValueError the id of the traced expression ([1fe99a5](https://github.com/georgeguimaraes/transmuter/commit/1fe99a5950e07259f04227c7cf8635a546ad4ec8))
* Keep only the graph outputs that the outputs option names ([7a6303e](https://github.com/georgeguimaraes/transmuter/commit/7a6303ee5c0caae3bce5c5a41a6a2056a9a56657))
* Keep promoted values out of the segment cache key ([e0747d6](https://github.com/georgeguimaraes/transmuter/commit/e0747d682031febc36b6db3390c40d4368b600fa))
* Keep the special tokens in Transmuter.Tokenizer.decode/3 on request ([6a176b3](https://github.com/georgeguimaraes/transmuter/commit/6a176b3e59bb33d94f21e39bba4b3e97c449fb5a))
* Prepare the images and the prompt of Qwen3-VL ([bcbb7bb](https://github.com/georgeguimaraes/transmuter/commit/bcbb7bbe5e3d505d92221da2b45a3c1b36016999))
* Raise Transmuter.HubError for an error status of the Hub ([6ac4d85](https://github.com/georgeguimaraes/transmuter/commit/6ac4d85d3bed7a2c7f8283a33ce12e6987afb337))
* Read the grid input name of the Qwen2-VL vision encoder from its graph ([c76a4e7](https://github.com/georgeguimaraes/transmuter/commit/c76a4e77cc2bd9689e6f1a0d2093d7d2bcf5c1c1))
* Read the Hugging Face token file and the proxy variables in Transmuter.Hub ([ab95cd1](https://github.com/georgeguimaraes/transmuter/commit/ab95cd13af48a1e6d4c07a7297040c9e25f23bac))
* Record the replaced nodes in each fused node ([9c512e1](https://github.com/georgeguimaraes/transmuter/commit/9c512e189bd69ce0aaa704fdc6093c4f0fd3e442))
* Resize images as the DETR and YOLOS image processors ([3603803](https://github.com/georgeguimaraes/transmuter/commit/36038037fe4071db9b343666952c1599a6b91d41))
* Restore the end of a chat template when truncation cuts a text ([33afb9e](https://github.com/georgeguimaraes/transmuter/commit/33afb9eb088b28283355ce0a77f7459615df07e6))
* Run Gemma 3 in image-text-to-text ([74deca8](https://github.com/georgeguimaraes/transmuter/commit/74deca8c9aaa52aa81b4c82666adfda4a1e2ebdd))
* Run images of any size in the image servings ([9b9ef01](https://github.com/georgeguimaraes/transmuter/commit/9b9ef01fd0b2808ddc3073c99e099bee9e95ecef))
* Run LLaVA in image-text-to-text ([ce9e9c3](https://github.com/georgeguimaraes/transmuter/commit/ce9e9c36ba7731fe509bba838d3f67e83558f22c))
* Run Ministral 3 in image_text_to_text ([fe9ae8a](https://github.com/georgeguimaraes/transmuter/commit/fe9ae8a7f4764e29c7e846ea1b1269a7fb49dabe))
* Run the coverage cases with EXLA and add the coverage batch corpus ([4c21805](https://github.com/georgeguimaraes/transmuter/commit/4c21805d81d1f9058b314e3fd0ebb5133e327143))
* Run the text tasks on decoders with a KV cache ([eba3a2a](https://github.com/georgeguimaraes/transmuter/commit/eba3a2a824de7e01412185f1317fc9c02b3ea36d))
* Show a progress bar for Hub downloads ([bbb5d28](https://github.com/georgeguimaraes/transmuter/commit/bbb5d281d9cbc69d6b7111311a74c2645d09670f))
* Show the graph as exported with mix transmuter.graph --plain ([ba05448](https://github.com/georgeguimaraes/transmuter/commit/ba054489048f2abfa2f38008fd72f8cdc934d581))
* Speak text with text_to_speech ([76108bc](https://github.com/georgeguimaraes/transmuter/commit/76108bcc6706000835399db320d8439163f307a2))
* Split images into tiles as the Idefics3 processor ([4c833cb](https://github.com/georgeguimaraes/transmuter/commit/4c833cb4d363123f61001d088ddcab412b1e7ecf))
* Take audio files and streams in the audio tasks ([a96771b](https://github.com/georgeguimaraes/transmuter/commit/a96771b12dc0ca3fd8320efa3e5fba13dd8933d4))
* Trace the nodes inside an If branch in the coverage divergence tools ([9bc4afc](https://github.com/georgeguimaraes/transmuter/commit/9bc4afcbb566736814a1cbb3e537939def67b15d))
* Use the cleanup value of the Python pipeline in text tasks ([361dde2](https://github.com/georgeguimaraes/transmuter/commit/361dde2720b070431e72dd4fccf909ec27d75270))
* Use the sentence_embedding output of EmbeddingGemma in text_embedding ([52a4b3f](https://github.com/georgeguimaraes/transmuter/commit/52a4b3fe8c2f5feee935ef25418f64622a7afbd0))


### Bug Fixes

* Add the exla and xla entries to mix.lock ([4189421](https://github.com/georgeguimaraes/transmuter/commit/41894217a09cb53fc6ffbc8a4d0d4ee7b098345f))
* Apply the review of the GLiNER serving ([b38de44](https://github.com/georgeguimaraes/transmuter/commit/b38de4493bf79e8ec5e369de781ad586a5856d74))
* Attention QK output mode and MultiHeadAttention key padding masks ([f31f3d1](https://github.com/georgeguimaraes/transmuter/commit/f31f3d19f6a85aa910c5d4a7cbc5b2e24e54ad16))
* Check language tokens and map unknown pieces for every caller of the tokenizer ([2d18f6b](https://github.com/georgeguimaraes/transmuter/commit/2d18f6b5413c941e4c49f7682faefb312ab41893))
* Check the image tokens of a chat and encode the Gemma 3 images in one call ([e834da3](https://github.com/georgeguimaraes/transmuter/commit/e834da39f3d480b004b2a93c0c4d16e6f7b2bdb8))
* Compile the word timestamp aligner with GraphBreaks.jit ([f41c079](https://github.com/georgeguimaraes/transmuter/commit/f41c079548aaa53ccb6540248fdc31061145943b))
* Compute the MaxRoiPool bin size with integer operations ([f497430](https://github.com/georgeguimaraes/transmuter/commit/f497430e83898e632f96d3876868bd8c2f28e98b))
* Copy a learned host value only when it follows from static values ([ae36a0c](https://github.com/georgeguimaraes/transmuter/commit/ae36a0c9281df78842ec700a22c80368504efd14))
* Correct Cast to bool, Round, IsInf, Softplus and Mod ([a8874fa](https://github.com/georgeguimaraes/transmuter/commit/a8874fa8beb3161d8dfb4a62d01d6dd4d1bb7fd7))
* Einsum with ellipsis broadcasting, diagonals and implicit output ([655d9b4](https://github.com/georgeguimaraes/transmuter/commit/655d9b47d075d25d8167b66a3ef5dfc1ec271441))
* Encode each Gemma 3 image alone and check the image token at build ([480ab32](https://github.com/georgeguimaraes/transmuter/commit/480ab32b2cfd47e45f74cd6ca9c22c2bab88d397))
* Exact integer sums for long contractions, and DynamicQuantizeLinear and DequantizeLinear edge values ([a6fb6ea](https://github.com/georgeguimaraes/transmuter/commit/a6fb6ea8be3b886054496de849dcfd0caa11c013))
* Follow GRAPHS attributes in the dependencies and the random check of a node ([4add796](https://github.com/georgeguimaraes/transmuter/commit/4add796fc268770db9bbdfee7f1902169a08b991))
* Follow ProcessorMixin and AutoProcessor for LLaVA and the other processors ([b09b1ba](https://github.com/georgeguimaraes/transmuter/commit/b09b1bac28baa7594e5cd1d2c55ebccded402c1c))
* Follow sentence-transformers for the text of text_ranking/2 ([c0d92c7](https://github.com/georgeguimaraes/transmuter/commit/c0d92c770e5d872a571a5c6301e10977f395a8ec))
* Follow the pipelines of transformers for the default max_new_tokens ([b3b4e05](https://github.com/georgeguimaraes/transmuter/commit/b3b4e05e555b6ef0351c91bbb32f1d8b66a19d7c))
* Follow the Pixtral processors of Python for alpha, do_resize and the image token ([e72bdee](https://github.com/georgeguimaraes/transmuter/commit/e72bdee135eee5bbef441769a4086f566fced03a))
* Follow the review of the cache telemetry ([72e84a6](https://github.com/georgeguimaraes/transmuter/commit/72e84a684eb9cd9680f3fea55e1fb6f6881a244c))
* Follow the review of Transmuter.EMLX.move ([1166329](https://github.com/georgeguimaraes/transmuter/commit/11663297991df0f131a10cccfeb7551bdc5daad7))
* Give 20 new tokens when max_length has the default value ([a8330d3](https://github.com/georgeguimaraes/transmuter/commit/a8330d31c1ed5c49f5a725d09de34d3190736108))
* Give each compiled segment its own EXLA disk cache file ([d3d089c](https://github.com/georgeguimaraes/transmuter/commit/d3d089c2b236cc1416274045c4324d88a4065f41))
* Give Grounding DINO the DETR resize in Transmuter.Featurizer ([469795b](https://github.com/georgeguimaraes/transmuter/commit/469795bb28e87e8514a64c0e885e319418e3d03f))
* Give the Pixtral processor the defaults of its Python classes ([7e09d29](https://github.com/georgeguimaraes/transmuter/commit/7e09d296a902b1330f569e51d8a421c9d3230335))
* Induce a break on a value that a node computed from arguments ([88f2166](https://github.com/georgeguimaraes/transmuter/commit/88f21668003a5b2cdea0a2e97a8c184defa6d803))
* Keep nearest Resize exact for integers and split the MultiHeadAttention bias by hidden sizes ([3390ee1](https://github.com/georgeguimaraes/transmuter/commit/3390ee19596d89763c118bcf1c5bc610c92be402))
* Keep the empty elements of a promoted sequence ([ed25b10](https://github.com/georgeguimaraes/transmuter/commit/ed25b10fd7e03d8aedf131fcad1854b14af857eb))
* Keep the large vision tests out of EXLA partition 2 ([af97d7f](https://github.com/georgeguimaraes/transmuter/commit/af97d7f5f6d89d0cf8626f575a18cb411c3a5197))
* Keep the MaxRoiPool bin edges exact when XLA compiles the op ([5e52729](https://github.com/georgeguimaraes/transmuter/commit/5e52729e561741152f983413726412c085156ced))
* Keep the non-tensor elements of a sequence that leaves a segment ([2b819fc](https://github.com/georgeguimaraes/transmuter/commit/2b819fc5163a0f098600f7b2304f3dc41f76edde))
* Keep the word buckets of GLiNER up to max_len ([03bafc0](https://github.com/georgeguimaraes/transmuter/commit/03bafc0c2c3e2ef08a1e716ccd1214579db16a00))
* Keep traced values out of the errors of a graph node ([28bd975](https://github.com/georgeguimaraes/transmuter/commit/28bd9755fa79c12083542ed69fdbec4e96b7370a))
* Keep values that follow only from shapes static ([cfe489d](https://github.com/georgeguimaraes/transmuter/commit/cfe489d827c73e554d437918cedc5e39d43003aa))
* Learn host values for each segment position ([10aeb22](https://github.com/georgeguimaraes/transmuter/commit/10aeb224a846a10846c68f5b1aa4db1b6af62e1b))
* Load tensors of 512 MiB and 1 GiB on OTP 27 ([7f07254](https://github.com/georgeguimaraes/transmuter/commit/7f072544c9ee525ee491eab4092ff97cc8bd4708))
* LSTM, GRU and RNN attributes: clip, input_forget, peepholes, activations and layout ([88517b9](https://github.com/georgeguimaraes/transmuter/commit/88517b974eacf65edf186c98c6157e3cfdddb895))
* MaxRoiPool memory and rounding, and MeanVarianceNormalization variance ([49508e2](https://github.com/georgeguimaraes/transmuter/commit/49508e25c2a4c3f6f08ae105a365517d30b05aeb))
* Pack MatMulNBits weights on EMLX without host math or unused copies ([a3bc0a8](https://github.com/georgeguimaraes/transmuter/commit/a3bc0a869428461760d1f39a02809069178907f3))
* Pad on the right with a user tokenizer in all servings ([76a5b2d](https://github.com/georgeguimaraes/transmuter/commit/76a5b2d140fb2ebda3edd1cf6b39e7c007f193bd))
* Pad with wrap and negative pads, scatter reductions and empty sizes ([f9ec1bb](https://github.com/georgeguimaraes/transmuter/commit/f9ec1bbddcca59c72f78eaf46ebfca85023a0edb))
* Pass the random key to compiled models ([08edd66](https://github.com/georgeguimaraes/transmuter/commit/08edd66945738fe1c7fd440c6552a638e4ee9214))
* Pool and ConvTranspose padding with auto_pad, dilations and output_shape ([e2ec096](https://github.com/georgeguimaraes/transmuter/commit/e2ec096e073eaed286d8d9fa3b68151cb50c2640))
* Put the large text tests in one Transmuter.TextLargeModelTest module ([9fe0384](https://github.com/georgeguimaraes/transmuter/commit/9fe03847698a7693a64f4928539509a932adaf05))
* Raise input errors of audio and generation in the caller ([19f058c](https://github.com/georgeguimaraes/transmuter/commit/19f058c9c14fb5ac1e0bcb13ba87a9c4ca207efa))
* Raise when an op reads one value from an input with several ([be701e7](https://github.com/georgeguimaraes/transmuter/commit/be701e7d22a75e2b5a4c93b3b0d1b630dc9aea68))
* Read product axes and [@op](https://github.com/op)_types lists in the coverage harness ([e36be9b](https://github.com/georgeguimaraes/transmuter/commit/e36be9bfc66a740d00a70d9390e067ed145d7c95))
* Read the fc3 and zero point inputs of MoE and QMoE ([3071491](https://github.com/georgeguimaraes/transmuter/commit/3071491de0472b3cb10c76049736c13286f81f1d))
* Read the tokenizers of the VITS and SpeechT5 exports ([1a4a9f8](https://github.com/georgeguimaraes/transmuter/commit/1a4a9f885d35af99234289d514f323eb52c0b7f9))
* Reduce empty axes, shift exclusive cumulative ops and sort unsigned TopK ([7b11dfb](https://github.com/georgeguimaraes/transmuter/commit/7b11dfbb674a530b882bcc0fb9fd3229f7b75b86))
* Reject a subgraph output that the subgraph does not compute ([f92710a](https://github.com/georgeguimaraes/transmuter/commit/f92710ac8b703a6ea706ac247ccad43b759bf2c4))
* Remove a stray conflict marker from the op fixture script ([da7aecc](https://github.com/georgeguimaraes/transmuter/commit/da7aecc4c570874df1887ca8c923a4cc484546ae))
* Remove the eager marks of segment positions ([152af7f](https://github.com/georgeguimaraes/transmuter/commit/152af7fb4e543880ffffb339e7d19627a321f992))
* Remove the memo of real breaks and upload only promoted values ([b3ffbca](https://github.com/georgeguimaraes/transmuter/commit/b3ffbca20d2a0248e49b6400fe0483d738993f50))
* Resize axes, roi and integer inputs, and GridSample reflection and nearest ties ([fe124e1](https://github.com/georgeguimaraes/transmuter/commit/fe124e10be7afed1b2824163916b3c2ab1d4cf87))
* Restore the suffix on each row of the largest length, as sentence-transformers ([c70d61d](https://github.com/georgeguimaraes/transmuter/commit/c70d61d7dd9db1ac36d857b1ef3f4945edab06fd))
* Retry a transient error of the Hub API when no metadata is saved ([d612e32](https://github.com/georgeguimaraes/transmuter/commit/d612e32d99600b306caccda2473dfc956accae0a))
* Run a long Loop over a large folded value as a while loop ([e80947d](https://github.com/georgeguimaraes/transmuter/commit/e80947d62476a65249975b7e43a622699c725baf))
* Scatter 64-bit integers on the EMLX GPU ([6ad9dcb](https://github.com/georgeguimaraes/transmuter/commit/6ad9dcb7a4577607d1b719cefe3e648fc15bbe05))
* Share the test backend with the coverage scripts and make their sums exact ([5dd033d](https://github.com/georgeguimaraes/transmuter/commit/5dd033d6114ae51a957a2d4e89e76df105bc5ac7))
* Split a large scope of the graph view into groups of equal size ([b4500a8](https://github.com/georgeguimaraes/transmuter/commit/b4500a8136d8df25315ccd3efeeca6dc3c1231b5))
* Stop the long-form Whisper loop when the seek does not move ([59b1020](https://github.com/georgeguimaraes/transmuter/commit/59b10202d89d3e3b6bb4bdef0562b1c9c3523602))
* Subgraph initializers get compute_type, Bernoulli draws in f32, and MultiHeadAttention reads its bias ([dd6884b](https://github.com/georgeguimaraes/transmuter/commit/dd6884b7372b7f68a622d7a744897a18dd25796d))
* Take the default resample of the image processor class ([5e6cc3d](https://github.com/georgeguimaraes/transmuter/commit/5e6cc3d5cf89862e10b918488f49b32157b62d81))
* Take the generation settings from config.json without generation_config.json ([f50484a](https://github.com/georgeguimaraes/transmuter/commit/f50484aae83d0a918f8c3716072faf632441cb6f))
* Turn off the compile cache of EXLA for compiled segments ([547e415](https://github.com/georgeguimaraes/transmuter/commit/547e415f10522c4ccac9ef101c4ec8ba64e5279a))
* Update find_divergence.exs for the current run_nodes and integer sums ([09ec23e](https://github.com/georgeguimaraes/transmuter/commit/09ec23ee02a1975795244d972a267ea6af788885))
* Use the clone flag of cp only on macOS in record_references.py ([09113e8](https://github.com/georgeguimaraes/transmuter/commit/09113e8d01da6b2fb88e0dd657656a4d50beea13))


### Miscellaneous

* Add a bench for SpeechT5 text to speech ([b2b7bc5](https://github.com/georgeguimaraes/transmuter/commit/b2b7bc563c892e4cbab240d3279ea4766d09198e))
* **ci:** Pin georgeguimaraes/workflows to v1.0.1 ([09e8ea1](https://github.com/georgeguimaraes/transmuter/commit/09e8ea1c9f7a9f5021e3d8ed17349c1313b7340a))
* **ci:** Pin GitHub Actions to commit SHAs ([f0bfdba](https://github.com/georgeguimaraes/transmuter/commit/f0bfdbaf4c5c189b21677c77c077a6ff73e4f6cd))
* Run the SpeechT5 bench on EXLA or EMLX ([f1b0e3a](https://github.com/georgeguimaraes/transmuter/commit/f1b0e3ac578ee928e61ac6770d8fe41402f83d02))
* Update castore to 1.0.22 ([2df1a7e](https://github.com/georgeguimaraes/transmuter/commit/2df1a7e92169afbb5cfdbd287226780bea29f7f7))
* Update otera to 0.2.0 ([244dedf](https://github.com/georgeguimaraes/transmuter/commit/244dedf332c481f889d188b1fa16c0ca0edbf503))


### Documentation

* Add LLaVA-Interleave to the README ([d168eae](https://github.com/georgeguimaraes/transmuter/commit/d168eaedbe745917dd94162eb95ebe8fb9eef8d4))
* Add the coverage of Apertus, Qwen3-MoE, GPT-J, Falcon and BigBird ([8728f40](https://github.com/georgeguimaraes/transmuter/commit/8728f40ba9a424f16d6e4a9af88dfd401fc3f629))
* Add the Gemma 3 results to the README ([4f5e48d](https://github.com/georgeguimaraes/transmuter/commit/4f5e48d366c1ed02797dd8de8a6259c011bc9214))
* Add the Ministral 3 results to the README ([977b87f](https://github.com/georgeguimaraes/transmuter/commit/977b87fc0cdcbd69350ef8ff408a59d46e0fcaad))
* Add the Qwen3-VL results to the README ([b5dcc6c](https://github.com/georgeguimaraes/transmuter/commit/b5dcc6c3b72ea9f35ba9aa82b549345ca3947b1e))
* Describe file and stream input of the audio tasks ([9af9477](https://github.com/georgeguimaraes/transmuter/commit/9af9477443c970c543058f6afd5f652413961a9a))
* Describe the channel-attention scale of the Florence-2 export in the README ([b22eda4](https://github.com/georgeguimaraes/transmuter/commit/b22eda43189d656b16973035c9d74c86addfbe87))
* Document Qwen2-VL in image_text_to_text ([5c81fcb](https://github.com/georgeguimaraes/transmuter/commit/5c81fcb14c8b38d3f86c1f3fcd8e5d8431f751f4))
* Document why q8 outputs differ from ONNX Runtime ([3f70fee](https://github.com/georgeguimaraes/transmuter/commit/3f70feeb3c6c22928fe879909827d743f4cc7cfa))
* Explain only the falcon int8 difference and name the x86_64 JIT ([f5be2c5](https://github.com/georgeguimaraes/transmuter/commit/f5be2c50fde069131b7985f564f6d9f977c3cedb))
* Explain TF32 on NVIDIA GPUs with EXLA ([f8f38f2](https://github.com/georgeguimaraes/transmuter/commit/f8f38f2cd6eda8b8bac727d6a3c095cab4daaf33))
* Explain the slovak-gpt-j int8 difference ([b9149a8](https://github.com/georgeguimaraes/transmuter/commit/b9149a877261271451085b57135e3e5a094eb363))
* Give the models and the docs link of each task in the README ([44c7cf0](https://github.com/georgeguimaraes/transmuter/commit/44c7cf09682197fd5fe8e6cd30be34aa141f19d0))
* Give the variants of Qwen3-VL that match ONNX Runtime, and the tolerance of its test ([39dbc28](https://github.com/georgeguimaraes/transmuter/commit/39dbc28985595001583f7d468aef9a82caa54b3b))
* List the tasks of the README by module ([2578ab5](https://github.com/georgeguimaraes/transmuter/commit/2578ab53cc9a663180d448c8ae8e6f39f7cfaafa))
* List Transmuter.Vision.DepthEstimation in the internal modules ([94ccf97](https://github.com/georgeguimaraes/transmuter/commit/94ccf97fc83b40fa47a51d17643be9dec25201d5))
* Move the Kokoro section of the README to the Speech and Audio guide ([e650266](https://github.com/georgeguimaraes/transmuter/commit/e650266b911821dc038ecbc583a9f3ecfb5874a6))
* Name SigLIP and SigLIP2 in the image tasks of the README ([02683b6](https://github.com/georgeguimaraes/transmuter/commit/02683b687432ddb043133fb0b386adc4b65eb19a))
* Name the backend and the device of the Ministral 3 coverage run ([e2d8daa](https://github.com/georgeguimaraes/transmuter/commit/e2d8daa8894289d816e7f0a6b522f437267a22f8))
* Record the Gemma 4 real-prompt check and the arm64 falcon result ([da3f623](https://github.com/georgeguimaraes/transmuter/commit/da3f623a34ff73f7843585a62d93372795dbe4e8))
* Say how GLiNER offsets follow a word without tokens ([82d8d8a](https://github.com/georgeguimaraes/transmuter/commit/82d8d8a48bebbe3d3f927eb42748a2e2e4feaa21))
* Say that dynamic 8-bit quantization depends on the batch ([87e8eca](https://github.com/georgeguimaraes/transmuter/commit/87e8eca1594eef11d4149305edf5e6f18f4aa8e0))
* Say that the first calls can compile again while the plan learns host values ([b9042a3](https://github.com/georgeguimaraes/transmuter/commit/b9042a33a280c85811530a5e67617a93b20a4735))
* Say that the Qwen2-VL vision encoder compiles for each image size ([7727d8f](https://github.com/georgeguimaraes/transmuter/commit/7727d8f5bebb7b07f66884f239928c200694cd70))
* Separate the falcon and GPT-J int8 notes and name the reference machine ([5670a50](https://github.com/georgeguimaraes/transmuter/commit/5670a50558bd85b703ccb058fded99438c083144))
* Show the graph view in the README with a screenshot of Kokoro ([4dac7aa](https://github.com/georgeguimaraes/transmuter/commit/4dac7aa4349b798c901a2de19c08346a4a167418))


### Code Refactoring

* Clean up MoE, dead code and unclear errors in the ops ([5538da6](https://github.com/georgeguimaraes/transmuter/commit/5538da6022d489930a1db15112a24f1f37e167f7))
* Give back the input of each row of a generation serving ([cd81b05](https://github.com/georgeguimaraes/transmuter/commit/cd81b055748aac3f5a0e7d6eed187595a841e8b7))
* Move the lookup of sentence-transformers files to Transmuter.Repo ([3298ac0](https://github.com/georgeguimaraes/transmuter/commit/3298ac07867f4afaf1767e8a8e26d54b0ef9d829))
* Put the placement decision of run_nodes in one function, with a recorder ([7fdb2ba](https://github.com/georgeguimaraes/transmuter/commit/7fdb2bab6ad265d0346e4129bbc0d84e187e2133))
* Raise on a speech-to-text row without its stride ([9ae406d](https://github.com/georgeguimaraes/transmuter/commit/9ae406dc3e35f40f7ca647f620ca8e718730ea29))
* Read the window attributes of convolutions and pools once ([8f7a261](https://github.com/georgeguimaraes/transmuter/commit/8f7a261936272421fb6f2070cf1a3257c31f7ee4))
* Select the grid input of the Qwen2-VL encoder by its name ([f59eac5](https://github.com/georgeguimaraes/transmuter/commit/f59eac581b9d1e2f91e6db62f5647245c80648e3))
* Share one attention core between the attention ops ([b09079c](https://github.com/georgeguimaraes/transmuter/commit/b09079cbc31c038c40b7b701380f0cae1e2d9ff6))
* Share the input and axis helpers of the ops ([eaf0484](https://github.com/georgeguimaraes/transmuter/commit/eaf0484c83fe90465c79383394e861cc603dc10e))
* Share the max_length and image size rules of the tasks ([bb0a9ca](https://github.com/georgeguimaraes/transmuter/commit/bb0a9ca3a2a449517af64d8800317abe04099dbd))
* Split the Nx path of AttentionCore from the kernel path ([54bd629](https://github.com/georgeguimaraes/transmuter/commit/54bd629bc8d40802b4a9ae02fe1aa251c0b4adfc))
* Unpack and dequantize packed weights in one module ([6e6a34f](https://github.com/georgeguimaraes/transmuter/commit/6e6a34f3f6a6410a850f2b29fa10b523515a9418))


### Tests

* Add the ONNX Runtime reference of Ministral 3 to the reference script ([763f79e](https://github.com/georgeguimaraes/transmuter/commit/763f79e7f8c783ccd7fc28d332fd58e1c9bc064f))
* Check a Qwen3-VL chat with two images against ONNX Runtime ([0fe4082](https://github.com/georgeguimaraes/transmuter/commit/0fe40824083cdc6bbcd82fb629ef916e73a5e4c6))
* Check eager segments and real breaks after a retry ([5a524f5](https://github.com/georgeguimaraes/transmuter/commit/5a524f5996901705f8d96ad8ee81cc54b3bc5712))
* Check host values for each segment position ([4bda697](https://github.com/georgeguimaraes/transmuter/commit/4bda69745eedc0163f0cb27b88163d1d13c5c561))
* Check image-text-to-text with a tiny Gemma 3 against ONNX Runtime ([dfc1b86](https://github.com/georgeguimaraes/transmuter/commit/dfc1b86cab202d6d9321bad1c21a6d873e9e63c4))
* Check image-text-to-text with Gemma 3 4B against ONNX Runtime ([983b802](https://github.com/georgeguimaraes/transmuter/commit/983b802b8b67eba2466ae77cc45b953e894e6cf8))
* Check image-text-to-text with Ministral 3 against ONNX Runtime ([be5e1d7](https://github.com/georgeguimaraes/transmuter/commit/be5e1d7a15bb0147a8a0008148ba04a9b309afd0))
* Check image-text-to-text with Qwen3-VL against ONNX Runtime ([58f1389](https://github.com/georgeguimaraes/transmuter/commit/58f1389d0b634e869bdf48628437c3cbad0e3a51))
* Check real break records and host values of device data ([e2e0326](https://github.com/georgeguimaraes/transmuter/commit/e2e032652d85d589921d3633f141d3367e41baec))
* Check that a GLiNER text without words gives no entities ([52c8911](https://github.com/georgeguimaraes/transmuter/commit/52c89115416639605bcdc3b1ed96cf2f5e964b0a))
* Check that a scatter on device data keeps positions as arguments ([e9b50b2](https://github.com/georgeguimaraes/transmuter/commit/e9b50b2f6c5aa035d2908aca217fed8ec189418f))
* Check the batches of a list, and a list with a stream ([692e4fb](https://github.com/georgeguimaraes/transmuter/commit/692e4fb5d756c439a11a9d24aa08020304ef0f60))
* Check the segment key for promoted positions ([d6839ec](https://github.com/georgeguimaraes/transmuter/commit/d6839ecca85c0af55c1001d05943b3e3ad8ab5a5))
* Check the type of the segmentation masks ([09e3bd3](https://github.com/georgeguimaraes/transmuter/commit/09e3bd334aaa7889300d953b86a3896c0acd0be7))
* Compare the Gemma 3 processor with Python transformers ([0d046e5](https://github.com/georgeguimaraes/transmuter/commit/0d046e585e7148dfe070bc41c72233767f98fb55))
* Give the TrOCR batch test a second line of handwriting ([990cac1](https://github.com/georgeguimaraes/transmuter/commit/990cac16dea754cad6a2e678a95109162551c230))
* Limit EMLX.CompileCache to 128 closures in the tests ([26ee642](https://github.com/georgeguimaraes/transmuter/commit/26ee6420ec9a66df24cefda17aeccc484695b378))
* Limit the MLX buffer cache of EMLX test runs to 1 GB ([f0ee1fe](https://github.com/georgeguimaraes/transmuter/commit/f0ee1fe90204778ce81376f23ddabcd306bd07e8))
* Run the LLaVA-Interleave chats alone and in a batch with one serving ([4260400](https://github.com/georgeguimaraes/transmuter/commit/426040078f77bdd6c17f6829f12029f46d03660c))
* Run the SmolVLM test only on local machines ([a1946c6](https://github.com/georgeguimaraes/transmuter/commit/a1946c6e6e306df7fe37ebd8e1eafa94b29d1b12))
* Run the SpeechT5 test only on local machines ([dbfd715](https://github.com/georgeguimaraes/transmuter/commit/dbfd715f226aa80afc766e4397ddd1aa565afc00))
* Select the test backend with TRANSMUTER_TEST_BACKEND ([dcb4f1c](https://github.com/georgeguimaraes/transmuter/commit/dcb4f1ce9fe8b8e297ccd46c899d6ef56fc98d3d))
* Tag the tests that need the Hub, and one smoke test of each pipeline ([25cc724](https://github.com/georgeguimaraes/transmuter/commit/25cc724dd7c4e7b56f48201017cf6942b0a0ae50))
* Test mask generation with the SAM 3 tracker ([4df034c](https://github.com/georgeguimaraes/transmuter/commit/4df034c4788cdba0c13bfdab7839047655aebef7))
* Test that Apertus generates the greedy text of ONNX Runtime ([3de750a](https://github.com/georgeguimaraes/transmuter/commit/3de750afef90511819e53d891a55405798300db0))


### Continuous Integration

* Cache the MLX archive in the EXLA jobs ([355f08f](https://github.com/georgeguimaraes/transmuter/commit/355f08f4769b9108ff4e5ab906b05991ef950511))
* Download the MLX archive with curl in the emlx job ([90c8876](https://github.com/georgeguimaraes/transmuter/commit/90c887619f5fccadcac2251979249af0bb0e74d3))
* Download the MLX archive with curl in the EXLA jobs ([4992488](https://github.com/georgeguimaraes/transmuter/commit/4992488eeb892009203fe28a561b288296e1818e))
* Download the release files of the deps with curl ([0ec55c0](https://github.com/georgeguimaraes/transmuter/commit/0ec55c075ae635e2f3e0fa162c5ab6508e0cfe4d))
* Download the test models without a model cache ([6772f66](https://github.com/georgeguimaraes/transmuter/commit/6772f665ce6d6d39c577c057cf5f88c67dbee508))
* Give each slow EXLA module its own partition ([cc32096](https://github.com/georgeguimaraes/transmuter/commit/cc32096ddc0ea2ce1d3da39fd51c51108d9af1a1))
* Install OpenBLAS from cached packages ([de153b0](https://github.com/georgeguimaraes/transmuter/commit/de153b01a7418d477768832afe2d8eb76ace8837))
* Keep the checksum of the MLX archive in the cache, and fix a README link ([6c9551e](https://github.com/georgeguimaraes/transmuter/commit/6c9551e09643a2d2b6796f52542c6143c5be0d12))
* Put the slowest EXLA tests in their own partitions ([f1dd18d](https://github.com/georgeguimaraes/transmuter/commit/f1dd18d9df3f497843a4a6870d46e5642e8b9814))
* Record the state of a stuck EMLX call in the EMLX job ([90c1f60](https://github.com/georgeguimaraes/transmuter/commit/90c1f60fd37285e22e6702177cdfb205db36e587))
* Run all tests on EXLA, and the fast tests on EMLX on the Metal GPU ([d168550](https://github.com/georgeguimaraes/transmuter/commit/d168550619d6983bffe02b62c541e6b6fc5e4ad3))
* Run Credo in the test environment ([8613a88](https://github.com/georgeguimaraes/transmuter/commit/8613a88832213efea6c5f448aa40637da7c0adee))
* Run the tests on EXLA on the CPU ([08d30cc](https://github.com/georgeguimaraes/transmuter/commit/08d30cc460aa1ebe3c459b84f082660f9269c3c4))
* Run the tests one at a time ([c095e2e](https://github.com/georgeguimaraes/transmuter/commit/c095e2e35f715ab98f64cacd512a3d35fc986162))
* Run the vision tests in EXLA partition 2 ([d67ea42](https://github.com/georgeguimaraes/transmuter/commit/d67ea427319be98e235c15a216c8ba8c78155a20))
* Show the slowest EXLA tests and modules in each partition ([c8188d5](https://github.com/georgeguimaraes/transmuter/commit/c8188d526b695ad470a4b343d5f15fbe501628b4))
* Split the EXLA tests into 17 partitions by test time ([8b8b49c](https://github.com/georgeguimaraes/transmuter/commit/8b8b49c5fa64d7e215b89e0aeacdb4536979de91))
* Split the EXLA tests into 3 partitions ([eea8217](https://github.com/georgeguimaraes/transmuter/commit/eea8217604b83a74144d7113d4efd134e0bac550))


### Performance Improvements

* Align all ROIs of a sampling grid in one pass ([90dc636](https://github.com/georgeguimaraes/transmuter/commit/90dc6368146574ededb0e6a69e1cc5f0eab892a8))
* Build large folded values on the device in a traced function ([6398a11](https://github.com/georgeguimaraes/transmuter/commit/6398a1164b83fbe7f0f11a4cf06065a64f8bc86b))
* Build the constants of eager operators on the backend of the model ([613836b](https://github.com/georgeguimaraes/transmuter/commit/613836b5def449a53ffb22b341c6c7293633fe83))
* Compute patch embeddings as one matrix product ([508b50f](https://github.com/georgeguimaraes/transmuter/commit/508b50f35e3340b3afb78f3f81e67bd982cc6164))
* Compute TopK of one value as an arg reduction ([c3b7feb](https://github.com/georgeguimaraes/transmuter/commit/c3b7feb870380b88f10e31c410561235d736240d))
* Convert half-precision params to compute_type on the backend ([4ab1841](https://github.com/georgeguimaraes/transmuter/commit/4ab1841520bfe325117a4206626ffeb8afa3a1b0))
* Dequantize MatMulNBits weights once on the EMLX CPU ([06e63dd](https://github.com/georgeguimaraes/transmuter/commit/06e63ddf24192800f174d07eb82214a84737220d))
* Digest each static value of a graph segment once per call ([78d4d3a](https://github.com/georgeguimaraes/transmuter/commit/78d4d3a4355d4dddd22608e724926001d7452e06))
* Draw random values without while loops on EMLX ([467f04a](https://github.com/georgeguimaraes/transmuter/commit/467f04a690dc6fe4d92e3d9cb53c094728020d2b))
* Fold each static node once per call ([6819e56](https://github.com/georgeguimaraes/transmuter/commit/6819e56b842f01b972782332df6ee97a40ce85be))
* Free the file binary of a model after a backend takes its params ([c1a7f9c](https://github.com/georgeguimaraes/transmuter/commit/c1a7f9cf90e332235691bb370b0623b3b684ed33))
* Fuse attention cores with a Reshape pair after the softmax ([85f9199](https://github.com/georgeguimaraes/transmuter/commit/85f91998de8a70375e95d1495af187f0ca9dbb36))
* Fuse the attention core of GPT-2 exports ([ac64e7d](https://github.com/georgeguimaraes/transmuter/commit/ac64e7dc7e6f440db28152255afaa6476be83acf))
* Give the servings only the model outputs that they read ([9bd4934](https://github.com/georgeguimaraes/transmuter/commit/9bd493492e58d830ce0a5e80b69a5dfc948f7f42))
* Give the static inputs of a traced node to the trace as literals ([9892eb9](https://github.com/georgeguimaraes/transmuter/commit/9892eb9c671b735ccf6715aa76d6a71a078933fe))
* Keep MatMulNBits weights packed for blocks of a multiple of 128 ([b774000](https://github.com/georgeguimaraes/transmuter/commit/b7740007052cda7e938a1e1bfb8ee87d4d6991fc))
* Keep the outputs of Shape and Size static in eager graph segments ([f0e419b](https://github.com/georgeguimaraes/transmuter/commit/f0e419b4ac8a7e3fc378ccaa04eebfad0021ca15))
* Key the fold memo on the value of a tensor attribute ([1ba4ab0](https://github.com/georgeguimaraes/transmuter/commit/1ba4ab033005bf0ba2e5f6d6794675a6d3fef71c))
* Make large Constant nodes and subgraph initializers params ([fa49973](https://github.com/georgeguimaraes/transmuter/commit/fa4997307a9d2f31b45f294cac50c6fb88270963))
* Make the outer values of a break node static ([bfd749b](https://github.com/georgeguimaraes/transmuter/commit/bfd749bf8d77a49897b3ffb81e0d0a55a80353d1))
* Move static values with more than 4096 elements off the host ([c83a566](https://github.com/georgeguimaraes/transmuter/commit/c83a566ae25aa6d70df44c62224cc93a170f45df))
* Normalize instances and groups with the EMLX layer norm kernel ([c8fca83](https://github.com/georgeguimaraes/transmuter/commit/c8fca83d741316d5a68284ee788ef20a71a4561e))
* Read external initializers when they move to the backend ([d92ed6e](https://github.com/georgeguimaraes/transmuter/commit/d92ed6eb0e076e574267ccf166b8d68f903731b3))
* Resize by an integer scale from shifted copies of the input ([b851eee](https://github.com/georgeguimaraes/transmuter/commit/b851eee37bc73d5b91cc839071f39bd49e8fa349))
* Run attention cores with the heads in the batch as one kernel ([8611b62](https://github.com/georgeguimaraes/transmuter/commit/8611b623aab03ce77ad5deb8e3e9a356de49a902))
* Run every expert on every token when MoE routes many tokens ([f1501d7](https://github.com/georgeguimaraes/transmuter/commit/f1501d731c12afe5f9cf244b8a3a8e41042cb863))
* Run the fused normalization and attention kernels in f16 and bf16 ([0c7ffc3](https://github.com/georgeguimaraes/transmuter/commit/0c7ffc3e9159069dc54f1d817e9e5e96ecfee171))
* Trace each graph segment once ([0c8aa3e](https://github.com/georgeguimaraes/transmuter/commit/0c8aa3e08ed27f80789eb675bda3a9bd70ae739f))
* Trace graph segments from the plan of the call ([d44f99d](https://github.com/georgeguimaraes/transmuter/commit/d44f99dc042a692dd54622b2b7aaa952d6ef5751))
* Write the GroupQueryAttention cache with one select ([5f98528](https://github.com/georgeguimaraes/transmuter/commit/5f98528b9f3c5b7804b962a9bd75b802aea05b09))

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
