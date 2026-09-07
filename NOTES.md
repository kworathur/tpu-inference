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


## Issue #3483 plan (AI-generated)

Here is a week plan for [issue 3483](https://github.com/vllm-project/tpu-inference/issues/3483). Keep the PR as **KV cache accounting**, not a GDN kernel.

Post on the issue first: you are taking it, and you will fix spec bytes, not full Qwen3.5 serving.

---

## Goal

Make `get_kv_cache_spec()` report the right bytes per token for a hybrid model.

The bug report: **596 GiB for 65536 tokens ≈ 9.7 MB/token**. Dense GQA is about **0.3 MB/token**. Your job is to find which spec path inflates that number, then add a test that fails on `main` and passes on your branch.

Do **not** try to run `orcarouter/Qwen3.8-27B-Uncensored` at 64K as the first proof. Use unit tests.

---

## Day 1: Reproduce the math, do not code yet

### Code reading (in this order)

1. Read `tpu_inference/layers/vllm/attention.py` `get_kv_cache_shape` (about lines 45–54). Note `TPU_HEAD_SIZE_ALIGNMENT = 128` and `num_kv_heads * 2`.
2. Read `tpu_inference/utils.py` `align_to` and `get_padded_head_dim`.
3. Read all of `tpu_inference/runner/kv_cache.py`. Focus on `get_kv_cache_shape_with_mesh` and `get_attention_page_size_bytes`.
4. Read `KVCacheManager.get_kv_cache_spec` in `tpu_inference/runner/kv_cache_manager.py` (lines 95–201). There are **two** paths:
   - empty `static_forward_context`: every layer gets `FullAttentionSpec`
   - registered `Attention` modules: only those modules get a spec
5. Read `_create_attention_spec` in the same file (lines 60–93). Note MLA padding of `kv_lora_rank` and `qk_rope_head_dim` **separately** to 128, then sum.
6. Read `tpu_inference/platforms/tpu_platform.py` `support_hybrid_kv_cache` (returns `True`).

### Exercise: write the byte formula

On paper, compute bytes per token for:

- GQA: `num_layers * 2 * num_kv_heads * padded_head_dim * 2` (bf16)
- MLA: `num_layers * (align(d_c, 128) + align(d_h^R, 128)) * 2`
- Hybrid: only **full-attention** layers scale with sequence length. GDN state is **O(1)** in sequence length.

Then compute 9.7 MB/token backward. Ask: how many heads, layers, and padded dims would you need to hit that number if every layer were full MHA?

### Supplemental reading

- DeepSeek-V2 paper, Table 1 (you already know this). Use it as the model for “cache per token.”
- Issue text: [3483](https://github.com/vllm-project/tpu-inference/issues/3483)
- vLLM `KVCacheSpec` types: `FullAttentionSpec`, `SlidingWindowSpec`, `MLAAttentionSpec` (imported in `kv_cache_manager.py`)
- Related hybrid work: [PR 3220](https://github.com/vllm-project/tpu-inference/pull/3220) (Gemma4 sliding window). Read the spec change, not the whole kernel.

---

## Day 2: Read the tests that already exist

1. Read `tests/runner/test_kv_cache.py` end to end. Copy the mesh fixture and `get_attention_page_size_bytes` asserts.
2. Read `tests/runner/test_kv_cache_manager.py`:
   - `test_get_kv_cache_spec` with mixed full / sliding-window layers (about lines 250–298)
   - `test_get_kv_cache_spec_with_compilation_cfg_mla` (about lines 300–329). This is the pattern for a new hybrid test: mock `static_forward_context`, call `get_kv_cache_spec()`, assert types and `page_size_padded`.
3. Skim `tests/e2e/test_hybrid_kvcache.py`. That test needs Gemma 27B and 4 chips. Do **not** start there.

Run the unit tests once on current `main` so you know the baseline:

```bash
python -m pytest -s -v tests/runner/test_kv_cache.py tests/runner/test_kv_cache_manager.py
```

If JAX has no device, the tests skip. Use the same mesh pattern the file already uses (`jax.local_devices()[:1]`).

---

## Day 3: Failing test first

Add one test in `tests/runner/test_kv_cache_manager.py`. Mirror `_setup_runner` and the compilation-config test.

Suggested asserts (adjust after you see the real Qwen3.5 layout):

1. A model with N layers, only N/4 of them full attention, must **not** emit N `FullAttentionSpec`s.
2. Bytes per token from `sum(spec.page_size_bytes / spec.block_size for spec in specs.values())` must stay far below 9.7e6.
3. If GDN / linear layers have no paged cache, they must be absent from the spec dict, same as encoder-only layers (see lines 191–194).

Name the test after the bug, for example `test_get_kv_cache_spec_hybrid_does_not_treat_all_layers_as_full_attn`.

Run only that test. Confirm it **fails on `main`**. If it already passes, your hypothesis is wrong. Then inspect the empty-`static_forward_context` branch (lines 118–133). That branch labels every layer as full attention. That is the most likely inflate path for flax / unregistered hybrid models.

---

## Day 4: Smallest fix that makes the test pass

Stay inside:

- `tpu_inference/runner/kv_cache_manager.py`
- `tpu_inference/runner/kv_cache.py` only if page-size math is wrong
- the new unit test

Likely fix directions (pick one after Day 3):

1. Empty context path: do not call `_create_attention_spec` for layers that are not decoder attention.
2. Registered-attention path: skip or use a non-sequence spec for GDN modules. Do not pad GDN state as `head_size` with `get_padded_head_dim`.
3. Do not use one `representative_spec` for all tensors if groups have different page sizes (`initialize_kv_cache` around line 237).

Keep MLA padding as it is unless the test proves it is the 9.7 MB bug. MLA padding (`512+64` → `512+128`) is about 11%, not 30×.

Do not change RPA kernels this week.

---

## Day 5: Test workflow before the PR

Numbered run:

1. Run `pre-commit run --all-files` (see `CONTRIBUTING.md`).
2. Run `python -m pytest -s -v tests/runner/test_kv_cache.py tests/runner/test_kv_cache_manager.py`.
3. Run the new test twice: once to show it would fail without the fix (git stash the production change if needed), once with the fix.
4. If you have a TPU: optional smoke of `tests/e2e/test_hybrid_kvcache.py` is extra, not required for the first PR.
5. Do **not** add a Buildkite e2e job unless a maintainer asks.

In the PR body, paste:

- the byte formula
- before/after `page_size_padded` for a mock hybrid config
- `FIXES: #3483`
- what you did **not** implement (no GDN kernel, no 64K Qwen3.5 serve)

Need **two** reviews. Code owners for runner: `@kyuyeunk @jrplatin @wenxindongwork @sixiang-google @mrjunwan-lang`. Add the `ready` label only when tests pass.

If you push after an approve, ask the reviewer to re-approve the new head. That is how PR 3035 stalled.

---

## Day 6–7: Review loop

Expect questions about:

- flax vs torchax (`static_forward_context` empty vs filled)
- whether GDN should have a new spec type vs “no cache”
- whether `support_hybrid_kv_cache()` is enough for vLLM’s hybrid manager

Answer with the test, not with a 27B serve log, unless they ask.

If review says “also serve Qwen3.5,” treat that as a follow-up PR.

---

## Out of scope this week

- Implementing Gated DeltaNet on TPU
- Issue 3126 (FP8 RPA cache)
- Rebasing DeepSeek load test 1852 / PR 3035
- Changing `TPU_HEAD_SIZE_ALIGNMENT` globally

---

## Daily checkpoint

At the end of each day you should have one artifact:

1. Written byte formulas and a named suspect branch in `get_kv_cache_spec`
2. Notes from existing tests
3. A failing unit test
4. A passing unit test plus a one-file fix
5. PR open with `FIXES: #3483`
6. Review replies

If Day 3 does not produce a failing test, stop coding. Comment on 3483 with the formulas and ask `@kyuyeunk` which spec path they expect for Qwen3.x GDN. That is still progress.