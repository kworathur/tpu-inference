# Issue Review: KV Cache Size Accounting Bugs

Issue URL: https://github.com/vllm-project/tpu-inference/issues/3483

## Which models are affected by the bug?

Hybrid Gated DeltaNet models [1], which include both full attention and a variation of the Mamba layers proposed in [2]; for these models, KV cache size can be *overestimated*. DeepSeek v4 models have layers that are not included in the hybrid KV cache manager's accounting routine; for these models, KV cache size can be *underestimated* [4].

## Root cause of the bug:

vLLM's Hybrid KV Cache Manager allocates memory for all layer types from a single pool [5]. Memory is allocated in units of pages; **page size** is defined as the physical size of a block. Each block stores `block_size` tokens, each consuming `kv_hidden` bytes. Page size is then `block_size` *`kv_hidden`* `num_layers`, which means that page size can vary by attention type. But allocating pages from a single pool means we must use a uniform page size across all layers.

**KV cache groups** are the solution to this problem. vLLM will allocate memory for a group of `group_size` identical layers at a time, rather than call the allocator for each layer. All groups have the same number of layers, so the page size remains uniform across groups. Note that the `num_blocks` per layer (and by extension group) can and should vary in the case of hybrid models. The value of `group_size` should be chosen strategically to reduce the amount of padding in each group [6]. vLLM then divides physical memory into `group_size` buffers, each storing `num_groups` layers, one from each group (group slice). The buffers are also referred to as `KVCacheTensor`s.

However, the TPU backend for vLLM does not support storing tensors from different layer types in the same buffer. This is because `jax.Array` (used for allocating buffers) are strongly typed. As a workaround, tpu-inference allocates buffers *per layer*, rather than *per group slice*. The number of block IDs that can be used to index a group's blocks outnumbers the number of block IDs used to index a layer's blocks. In other words, the amount of memory needed to store a layer's blocks is less than the amount of memory needed to store multiple layer's blocks.

This discrepancy prompts a workaround in tpu-inference (see `update_mamba_page_size_padded()`) that pads the page size of all layers to be

$$\text{uniform\_page\_size\_bytes} = \text{num\_attention\_groups} *\text{attention\_page\_size} \\ + \text{num\_mamba\_groups}* \text{unpadded\_mamba\_page\_size}$$

which is exactly the size of one `KVCacheTensor` and satisfies the uniform page size constraint from before.

In the vLLM core, we call `get_kv_cache_specs()` which calls `get_kv_cache_spec()` on each worker to obtain layer name -> `KVCacheSpec` mappings. Each `KVCacheSpec` stores the padded `page_size_bytes` set in `update_mamba_page_size_padded()`.

Next, we call `get_kv_cache_configs()`, which handles the following:

1) merges KV cache specs across workers into a global layer name -> `KVCacheSpec` mapping.
2) grouping layers in the model that have the same `KVCacheSpec` to create a list of `KVCacheGroupSpec`s

In the failing run with Qwen, the hybrid KV cache manager was enabled. Under this setting, vLLM unifies the KVCacheSpec page size, which appears to be a no-op since the `page_size_bytes` has already been standardized by `update_mamba_page_size_padded` and `_create_attention_spec`. If the `page_size_bytes` were not standardized, vLLM will pad the Mamba page size to the largest page size. Unlike full attention layers, Mamba page size does not scale with block_size.

After ensuring a uniform page size across all specs, vLLM gets the KV cache groups using the exact same heuristic as in tpu-inference to obtain a list of `KVCacheGroupSpec`s, one for each cache group (see `_get_kv_cache_groups_uniform_page_size()` line 1516).

3) assigning all or some of a KV cache groups layers to each worker (projection) and then checking there is enough memory on each worker to store its part of the KV cache

Projection: the layer->KVCacheSpec mapping for each worker is passed as a list argument to `get_kv_cache_configs()`. `KVCacheGroupSpecs` are assigned to workers depending on whether tensor and/or pipeline parallelism is enabled. Since the failing run only had tensor parallelism enabled, each worker should receive every `KVCacheGroupSpec` and store part of the KV cache for each layer. 

Memory checks: The available memory in bytes for each worker is defined in the `available_memory` array and optionally can be overridden by `num_gpu_blocks_override`. In the absence of an override, vLLM  subtracts the sum of all `page_size_bytes` over layers in a group from each worker's available memory[7]. This sum is exactly the size of the *entire* KV cache, which seems like a mistake considering that no worker is storing the entire KV cache in this tensor parallel mode of execution.

After this initial accounting for used memory, vLLM does another pass over all workers, calling `_check_enough_kv_cache_memory()` with `_max_memory_usage_bytes_from_groups()` as the function for computing needed memory.  First it computes the same sum of `page_size_bytes` over all layers in a KV cache group, and stores it as `bytes_per_block`. Then it divides a group spec's `max_memory_usage_bytes` by its `page_size_bytes` to get the number of blocks for the group. Finally it returns `bytes_per_block` * `num_blocks`. Although Mamba's `max_memory_usage_bytes` should not scale with `max_model_len` according to the issue poster, the implementation on line 22 of MambaSpec seems to suggest otherwise:

```python
max_model_len = vllm_config.model_config.max_model_len
            return (
                cdiv(max_model_len, self.block_size) + self.num_speculative_blocks
            ) * self.page_size_bytes
```

So the needed memory for the worker is clearly an overestimate, and since we already subtracted the full KV cache size from each worker's available memory, there is simply not enough memory to allocate and vLLM returns the following error:

> ​ ValueError: To serve at least one request with the model's max seq len (65536), (596.12 GiB KV cache is needed, which is larger than the available KV cache memory (54.55 GiB). Based on the available memory, the estimated maximum model length is 5904.

The rest of `get_kv_cache_configs()` is not run, but this is what would happen if the error weren't raised: 

From each worker's available memory, we subtract the max sum of `page_size_bytes` of each layer in a group over all groups. Note in our case, all groups have the same `page_size_bytes` (due to padding) so the max is just the sum of `page_size_bytes` for any of the groups. Take for example, a model with 10 full attn + 30 mamba, group_size is 10 and there are four groups. Each page_size_padded sums bytes of one layer from each four groups. Add 10 `page_size_padded` to get the actual number of bytes consumed by the KV cache.

Finally, we compute  the number of blocks allocated to the KV cache using the lines below

```python
num_blocks = available_memory // bytes_per_block
num_blocks = may_override_num_blocks(vllm_config, num_blocks)
```

Once we have determined the number of blocks, we can create the `KVCacheConfig`s and call `initialize_from_config()` in the engine core.

After getting all KV cache configs, the TPUWorker is initialized from a `KVCacheConfig` that includes the number of blocks:

```python
@dataclass
class KVCacheConfig:
    """
    The KV cache configuration of a model.
    """

    num_blocks: int
    """The number of KV cache blocks"""
```

TODO: Read test_get_kv_cache_spec_hybrid_mamba_cache_config_updates

## References

[1] [Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/abs/2412.06464)

[2] [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)

[3] [Hybrid KV Cache Manager in vLLM](https://docs.vllm.ai/en/latest/design/hybrid_kv_cache_manager/?h=)

[4] The part of the code where deepseek layers are omitted from cache size calculation is line 64 of `tpu_inference/runner/kv_cache_manager.py`

[5] [Single memory pool allocation](https://docs.vllm.ai/en/latest/design/hybrid_kv_cache_manager/?h=#definitions:~:text=We%20use%20a%20single%20memory%20pool%20for%20all%20layer%20types)

[6] The current heuristic for determining group size can be found on line 262 of `kv_cache_manager.py`.

[7] (technically the group with max sum is selected, but in this case they're all the same)