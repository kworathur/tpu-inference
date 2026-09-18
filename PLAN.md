## 8. What a fix should change

Do **not** teach `MambaSpec.max_memory_usage_bytes` to ignore
`max_model_len` on TPU. It already does, except in unsupported mode
`"all"`.

Do change **where bytes are attributed**:

1. Keep `KVCacheSpec.page_size_bytes` as the **real** per-layer page
   (attention page or unpadded Mamba state).
2. After grouping, compute `bytes_per_block` from those real pages
   **per group / per pool**, not from a model-wide
   `num_groups × mixed page`.
3. For TPU per-layer arrays, size `num_blocks` so block IDs fit each
   layer's leading dim **without** lying that every attention token
   also stores every Mamba group.
4. Put extra specs (DeepSeek indexer, compressor, SWA, other side
   caches) into the same grouping/pricing path. Do not special-case
   them only in `is_cache_for_ds_v4`.
5. If compact-mamba stays, publish `num_gpu_blocks_override` (or an
   equivalent) on something EngineCore already receives (specs or
   `determine_available_memory()`), so #3544 cannot resurrect #3483
   on Ray.

`test_get_kv_cache_spec_hybrid_mamba_cache_config_updates` today
**asserts the padded lie**. A correct fix should replace that with
tests that:

- real attention page ≠ real Mamba page
- one max-length request's `max_memory_usage_bytes` is
  `O(seq × attn_layers × attn_page + mamba_layers × mamba_page)`
- extra non-Attention specs still appear in `bytes_per_block`