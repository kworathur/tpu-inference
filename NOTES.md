# PR Review: Enable DeepSeek-V2-Lite-Chat-FP8

Link to PR: https://github.com/vllm-project/tpu-inference/pull/1729

Commit hash: 768519152ac6c021fc2fe0277c28bdc258789369

Adds support for FP8 quantized Deepseek V2 models, with main changes focused on the attention and gmm modules of tpu-inference.

## Bugs fixed by this PR

Bug #1: Dimension mis-match in the attention module where the the query 2D matrix is actually a 3D tensor

Fairly certain this is NOT related to FP8 quantization but could be related to DeepSeek v2 being an MoE model.

Bug #2: Another dimension mis-match in the fused MoE kernel that is common to all models.

Caused by DeepSeek using k=6 experts in its MoE architecture, which means matrix dims are not multiples of 128.

From DeepSeek v2 paper:

>Among the routed experts, 6 experts will be activated for each token.

Why 128? XLA (the ML compiler) prefers creating 128 x 128 tiles to perfectly fill the MXU that is found in each TPU (MXU supports fast matrix multiplication using systolic arrays)

MoE calculation: obtain a matrix of logits from the top k=6 experts, which are then multiplied? and then passed through fully-connected layers to obtain the final activations from the LLM.


Note that DeepSeek being a quantized model does not seem to be relevant to either bug.


## Fixes

Tiling math introduced in this PR:

In get_default_gmm_block_sizes(), 

-> round_up_to_multiple_of_128_within_limit()


Terms:
* Tiling: code transformation step that involves breaking matrix-matrix multiplications into smaller units of work that run efficiently on the MXU
* Quantization: reducing the numerical precision of weights and biases to improve memory footprint and latency with minimal degradation in accuracy.
* MLA: 
* Just-in-time compilation:
* GMM: general matrix multiplication
* Attention head dimension:

Future work:
- Multi-head latent attention (MLA) implementation deferred to a later PR - could draft what this would look like

Additional learning:
- How SparseCore accelerates sparse matrix multiplications
