# PR Review: Enable DeepSeek-V2-Lite-Chat-FP8

Link to PR: https://github.com/vllm-project/tpu-inference/pull/1729

Adds support for FP8 quantized Deepseek V2 models, with main changes focused on the attention and gmm modules of tpu-inference.

Bug: dimension mis-match in the attention module and tile dimension not divisible by 128

Multi-head latent attention (MLA) implementation deferred to a later PR - could draft what this would look like

Terms:
* Tile: related to blocks and warps as scheduling units in a GPU?
* Quantization: reducing the numerical precision of weights and biases to improve memory footprint and latency with minimal degradation in accuracy.
* MLA: 
* GMM: general matrix multiplication
* Attention head dimension:
