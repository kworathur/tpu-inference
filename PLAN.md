
## Issue #3483 plan (updated after jt226ub comment)

Here is a week plan for [issue 3483](https://github.com/vllm-project/tpu-inference/issues/3483). Keep the PR as **KV cache accounting**, not a GDN kernel.

You already claimed the issue. Post a short follow-up: you will price **one page size per KV-cache group**, not one page size for the whole model. Accept jt226ub's offer to test a hybrid GDN model with side caches. Do not promise daily updates.

---

## Goal

Make hybrid KV-cache **page-size padding** match vLLM's grouping, so the bytes-per-token estimate is neither ~9.7 MB (too high) nor silently too low.

The bug report: **596 GiB for 65536 tokens ≈ 9.7 MB/token** on `vllm serve` of a Qwen3.x hybrid GDN model. Dense GQA is about **0.3 MB/token**.

Do **not** try to run `orcarouter/Qwen3.8-27B-Uncensored` at 64K as the first proof. Use unit tests. jt226ub can run the real hybrid harness later.

### Current diagnosis (jt226ub, 2026-09-08)

The inflate is **not** “every layer is `FullAttentionSpec`.” The `vllm serve` path fills `static_forward_context`. That path already emits mixed `FullAttentionSpec` / `MambaSpec` (see `test_get_kv_cache_spec_hybrid_mamba_cache_config_updates`).

The real site is `update_mamba_page_size_padded` (`kv_cache_manager.py` ~147–296):

* It samples **one** attention geometry and one mamba geometry for the **whole model**.
* It asserts every `Attention` layer shares `num_kv_heads` and `head_size` (~196).
* It sets `_hybrid_uniform_page_size_bytes` and `mamba_page_size_padded` to
  `num_attn_groups × attn_page + num_mamba_groups × mamba_unpadded`.
* `_create_attention_spec` then writes that same padded size onto every attention spec.
* Layers that own their own spec (Mamba via `get_kv_cache_spec`, DSv4, extra side caches) skip that geometry sample, so their bytes may never enter the padded page size.

vLLM's `_check_enough_kv_cache_memory` then **sums** `max_memory_usage_bytes` over all layers. Each layer reports the **group** footprint, so the estimate can be:

1. **Too high** (this issue): every layer is priced as the full attn+mamba group.
2. **Too low** (jt226ub): side caches / second attention geometries are allocated against a page size that never included them. `num_blocks` and the real array count disagree until `num_gpu_blocks_override`.

Padding to a **common** page size by `max` is also the expensive part: a correct estimate can still waste HBM if groups do not get their own size.

jt226ub's suggested shape: compute one page size **per KV-cache group** (after grouping, from that group's geometry). That also composes with dual-pool work in [#3517](https://github.com/vllm-project/tpu-inference/pull/3517) (merged). A single corrected global page size would fix 3483's recurrent-vs-attention gap and still misprice mixed attention geometries or side caches.

Related: [#3081](https://github.com/vllm-project/tpu-inference/issues/3081) hits the grouping side of the same “one geometry” assumption.

The empty-`static_forward_context` flax path can still treat linear layers as full attention (`layer_types` is read, but `_create_attention_spec` is still used). That is a **secondary** path. This ticket's serve log is torchax/vLLM.

---

## Day 1: Reproduce the math, do not code yet

### Code reading (in this order)

1. Read `update_mamba_page_size_padded` in `tpu_inference/runner/kv_cache_manager.py` (lines 145–290). Trace:
   * the all-attention-same-shape assert
   * the vLLM group-size heuristic (`max < 1.5 × min` → `max_count`, else `min_count`)
   * `uniform_page_size_bytes` and `_hybrid_uniform_page_size_bytes`
2. Read `_create_attention_spec` (lines 104–143). Note it applies `_hybrid_uniform_page_size_bytes` when set.
3. Read `get_kv_cache_spec` registered-module branch (lines 590–690):
   * `update_mamba_page_size_padded(layers)` before spec creation
   * Mamba / DSv4 layers that call `attn_module.get_kv_cache_spec` and `continue` (the “own spec” bypass)
4. Skim `_maybe_set_compact_mamba_num_blocks_override` and how `#3517` dual-pool sizing uses the same group counts.
5. Read `get_attention_page_size_bytes` in `tpu_inference/runner/kv_cache.py`. You need this for the attn_page term.
6. Optional background only: `flash_attn.py` `get_kv_cache_shape`, `align_to` / `get_padded_head_dim`. Do not start the flax empty-context loop unless Day 3's test already passes.

### Exercise: write the byte formula

On paper, for a hybrid layout like **16 full-attn + 68 GDN** (issue-like 1-in-4) and for **1 attn + 3 mamba per group** (Qwen3.5):

* Unpadded attn page: `get_attention_page_size_bytes(...)`
* Unpadded mamba page: `prod(shape) × dtype_size`
* **Current code:** every layer's `page_size_padded` = `num_attn_groups × attn_page + num_mamba_groups × mamba_page`
* **vLLM check:** `sum_over_layers(page_size_padded / block_size)` ≈ reported bytes/token

Show that summing the group footprint over every layer can land near **9.7 MB/token**. Also write the **per-group** formula: each group pays only its own attn+mamba pages, not the model-wide max.

### Supplemental reading

* Issue text: [3483](https://github.com/vllm-project/tpu-inference/issues/3483)
* jt226ub comment on that issue (2026-09-08)
* Dual-pool PR: [3517](https://github.com/vllm-project/tpu-inference/pull/3517)
* vLLM grouping heuristic duplicated in the docstring of `update_mamba_page_size_padded` (from `vllm/v1/core/kv_cache_utils.py::_get_kv_cache_groups_uniform_page_size`)
* Existing hybrid unit test: `test_get_kv_cache_spec_hybrid_mamba_cache_config_updates`

---

## Day 2: Read the tests that already exist

1. Read `tests/runner/test_kv_cache.py` for mesh fixtures and `get_attention_page_size_bytes`.
2. Read `tests/runner/test_kv_cache_manager.py` hybrid tests:
   * `test_get_kv_cache_spec_hybrid_mamba_cache_config_updates` (~1202): 1:1 attn/mamba, **already expects one uniform page size on both specs**. That test will need updating if you stop using a model-wide pad.
   * `test_update_mamba_page_size_padded` / 10:30 attn:mamba case (~1266): asserts the group-size heuristic and the single `mamba_page_size_padded`.
3. Skim `tests/e2e/test_hybrid_kvcache.py`. Needs Gemma 27B and 4 chips. Do **not** start there.

Run the unit tests once so you know the baseline:

```bash
python -m pytest -s -v tests/runner/test_kv_cache.py tests/runner/test_kv_cache_manager.py
```

If JAX has no device, the tests skip. Use the same mesh pattern the file already uses (`jax.local_devices()[:1]`).

Note which existing asserts **encode the bug** (model-wide uniform page size). Those tests must change with the fix, not stay as “green means correct.”

---

## Day 3: Failing test first

Add tests in `tests/runner/test_kv_cache_manager.py`. Mirror `_setup_runner` and the hybrid compilation-config tests.

Do **not** assert “not all specs are `FullAttentionSpec`.” That likely **already passes** on `main`.

Primary test (issue 3483 overestimate), e.g. `test_hybrid_page_size_is_per_group_not_per_model`:

1. Mock **16 Attention + 68 Mamba** (or 10:30 if that matches the existing 10:30 fixture).
2. Call `get_kv_cache_spec()` / `update_mamba_page_size_padded`.
3. Assert spec **types** are mixed (`FullAttentionSpec` vs `MambaSpec`) — this is already true; keep it as a sanity check.
4. Assert bytes per token
   `sum(spec.page_size_bytes / spec.block_size for spec in specs.values())`
   stays far below 9.7e6. On current `main` this should **fail** because every layer reports the group sum.
5. Assert attention specs in different groups are **not** forced to one model-wide `_hybrid_uniform_page_size_bytes` after the fix. Until the fix, document the current single value as the failing baseline.

Secondary test (jt226ub underestimate / mixed geometry), e.g. `test_hybrid_page_size_includes_side_cache_or_second_attn_geometry`:

* Two `Attention` modules with different `head_size` or `num_kv_heads`, **or** extra specs that bypass `update_mamba_page_size_padded`.
* Today: the same-shape assert fires, **or** side-cache bytes are missing from the padded page size.
* After the fix: each group is priced from **its** geometry; no silent drop of extra cache bytes.

If the primary test already passes on `main`, the “sum of padded group sizes” hypothesis is wrong. Stop and re-read how vLLM uses `page_size_padded` vs `page_size_bytes` before coding.

---

## Day 4: Smallest fix that makes the tests pass

Stay inside:

* `tpu_inference/runner/kv_cache_manager.py`
* `tpu_inference/runner/kv_cache.py` only if page-size math is wrong
* the new (and updated existing) unit tests

Target shape (from jt226ub; pick the smallest version that fails the new tests):

1. Build **unpadded** specs first (break the current cycle: padding depends on grouping, grouping needs specs).
2. Group by geometry (same heuristic vLLM uses, or call it once specs exist).
3. Set `page_size_padded` **per group**, not via one `_hybrid_uniform_page_size_bytes` for the model.
4. Keep `#3517` compact-mamba / dual-pool sizing: it already takes `num_attn_groups` / `num_mamba_groups`. Do not undo that. Feed it per-group page sizes instead of one uniform size if it currently consumes the uniform value.

A one-line “skip GDN in `_create_attention_spec`” will **not** fix this. Spec types are already mixed.

Keep MLA padding as it is unless a test proves it is the 9.7 MB bug. MLA padding (`512+64` → `512+128`) is about 11%, not 30×.

Do not change RPA kernels this week.

If per-group padding plus a spec-ownership protocol for side caches is too large, ship per-group padding for the 16:68 / 10:30 case and call out mixed-geometry / side caches as follow-up. Say that explicitly in the PR, because jt226ub already measured that gap.

---

## Day 5: Test workflow before the PR

Numbered run:

1. Run `pre-commit run --all-files` (see `CONTRIBUTING.md`).
2. Run `python -m pytest -s -v tests/runner/test_kv_cache.py tests/runner/test_kv_cache_manager.py`.
3. Run the new tests twice: once to show they fail without the fix (git stash the production change if needed), once with the fix.
4. Confirm the old hybrid tests still match the **new** contract (per-group pad), not the old uniform pad.
5. Optional: ping jt226ub with the branch for their GDN + side-cache harness. That is extra, not a merge gate.
6. Do **not** add a Buildkite e2e job unless a maintainer asks.

In the PR body, paste:

* the group page-size formula vs the old model-wide uniform formula
* before/after `page_size_padded` for the 16:68 (or 10:30) mock
* that the estimate can be wrong in **both** directions (cite jt226ub)
* `FIXES: #3483`
* what you did **not** implement (no GDN kernel, no 64K Qwen3.5 serve, no spec-ownership protocol unless you actually did it)
* how this composes with #3517

Need **two** reviews. Code owners for runner: `@kyuyeunk @jrplatin @wenxindongwork @sixiang-google @mrjunwan-lang`. Also mention `@jt226ub` as the downstream tester. Add the `ready` label only when tests pass.

If you push after an approve, ask the reviewer to re-approve the new head. That is how PR 3035 stalled.

---

## Day 6–7: Review loop

Expect questions about:

* why a single corrected `uniform_page_size_bytes` is not enough
* chicken-and-egg: grouping needs specs, padding used to need grouping first
* flax vs torchax (empty `static_forward_context` is secondary for this ticket)
* whether `#3517` dual-pool still gets the right `num_gpu_blocks_override`
* whether mixed attention geometries should assert or split groups

Answer with the tests, not with a 27B serve log, unless they ask. If jt226ub posts a harness result, include it.

If review says “also serve Qwen3.5,” treat that as a follow-up PR.

---

## Out of scope this week

* Implementing Gated DeltaNet on TPU
* A spec-ownership protocol instead of the `isinstance` whitelist (jt226ub's separate write-up; related to #3081)
* Issue 3126 (FP8 RPA cache)
* Rebasing DeepSeek load test 1852 / PR 3035
* Changing `TPU_HEAD_SIZE_ALIGNMENT` globally
* Replacing `#3517` dual-pool logic

---

## Daily checkpoint

At the end of each day you should have one artifact:

1. Written formulas: model-wide uniform pad vs per-group pad, and how the sum over layers hits ~9.7 MB/token
2. Notes from existing hybrid tests, including which asserts currently **require** the uniform pad
3. A failing unit test for per-group vs per-model page size (and mixed geometry if you take that slice)
4. A passing test plus a `kv_cache_manager.py` fix; old hybrid tests updated to the new contract
5. PR open with `FIXES: #3483` and a follow-up comment pointing jt226ub at the branch
6. Review replies

If Day 3 does not produce a failing test, stop coding. Comment on 3483 with the formulas and ask `@kyuyeunk` / jt226ub how vLLM's memory check uses `page_size_padded` for hybrid groups. That is still progress.
