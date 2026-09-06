# PR Review: Enable DeepSeek-V2-Lite-Chat-FP8

Link to PR: https://github.com/vllm-project/tpu-inference/pull/1729

Commit hash: 768519152ac6c021fc2fe0277c28bdc258789369

Adds support for FP8 quantized Deepseek V2 models, with main changes focused on the attention and gmm modules of tpu-inference.

## Bugs fixed by this PR

Bug #1: Dimension mis-match in the attention module where the the query 2D matrix is actually a 3D tensor.

This bug is not related to FP8 quantization, nor is it related to DeepSeek v2 using an MoE architecture. The decision to adapt RoPE from DeepSeek v1 to the new multi-head latent attention (MLA) introduced by DeepSeek v2 is the main cause for this bug. 

Aside on RoPE: RoPE is used because it encodes relative position rather than absolute position, which is sensitive to translation. Using absolute positioning embedding (e.g. using a sinusoidal function as in original Transformer paper) means that the linear attention function has limited capacity to express relative distance. This is because the attention mechanism relies entirely on query and key projection matrices for encoding "distance" between positional embeddings, making it error-prone on unseen positions [1]. RoPE exchanges additive positional encoding for a matrix multiplication, which preserves attention scores under translation.

Now that we are compressing our keys and caching only their compressed representations, we ideally want to avoid re-computing k_t for previous tokens. Fortunately, the attention scores (q_t)^T * k_t^c can be computed nicely in terms of W^UK and W^Q as 

h_t^T (W^Q)^T W^UK c_t^KV 

Notably, we can reuse the same W^Q and W^UK for every attention computation. When we use RoPE, we add a matmul between rotation matrices in between the W^Q and W^UK matmul. This is because we have to compute 

k_t^R = R_theta_i * W^UK * c_t^KV
and 

(q_t^R)^T =  h_t^T * (W^Q)^T *  R_theta_j^T

Where i and j are indices denoting the positions of the key and query respectively. 

Using RoPE as described above means k_t^R is going to depend on i. This means we first have to perform the up-projection of c_t^KV, then apply ROPE. This is no better than computing k_t on every token, which is what MLA attempts to avoid.

What decoupling RoPE means: Use compressed queries and attention input to compute q^R and k^R vectors that have rotation transformations applied to them using the standard RoPE operator. These vectors are then concatenated with the compressed query and key vectors before being provided as input to the attention mechanism. 

Q: When we say the decoupled RoPE query and key are shared, does it mean shared across attention heads, whereas the compressed queries and keys are unique per head?



 concatenating RoPE query to each query (this concatenation cannot be represented with only two dims). Why do we need to perform concatenation - why can't we apply RoPE to the query itself?

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


## References

[1] RoPE: Addressing the Position Encoding Flaw in Transformer Models https://swtheking.notion.site/a08016ff028f44f7ab7e366ed682efa0?v=b5a87525670a4e7a95a0d2671d81e5e6