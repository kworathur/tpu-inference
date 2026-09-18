# Issue Review: KV Cache Size Accounting Bugs

Issue URL: https://github.com/vllm-project/tpu-inference/issues/3483

## Which models are affected by the bug?

Hybrid Gated DeltaNet models [1], which include both full attention and a variation of the Mamba layers proposed in [2]; for these models, reported KV cache size can be *overestimated*. Extra KV caches (such as DeepSeek indexer and compressor) provide their own spec; for these models, KV cache size can be *underestimated* [4].

## Root cause of the bug:

vLLM's Hybrid KV Cache Manager allocates memory for all layer types from a single pool [5]. Memory is allocated in units of pages, each measuring a physical block in memory; the physical block size is determined by `block_size` * `kv_hidden`, multiplied by a number of layers (all the same type, not necessarily all model layers). Thus, we can end up with different page sizes per attention type, which is not allowed since we allocate pages from a single pool.

**KV cache groups** are the solution to this problem. A group is a collection of identical layers, and the number of layers per group (`group_size`) is uniform across all groups (after padding). The value of `group_size` should be chosen strategically to reduce the amount of padding in each group [6]. Observe that if we standardize `page_size_bytes` (size of a single layer's block), then we can achieve the same physical page size across groups. Every attention type has its own `page_size_bytes`, requiring a trick to make this value consistent across attention types. 

vLLM allocates a single buffer to store KV cache state of multiple groups. Within this buffer, a physical block stores a page from each layer in a group. By using a layer's position for strided access within a block, all layers in a group can share the same logical block IDs. Different groups cannot share the same logical block IDs, however, since that would result in groups overwriting each other's data in the shared buffer. Notably, if groups have different dtypes for their data, vLLM on GPU can return a view for part of the buffer to the requested dtype, known as **overlaying**.

However, the TPU backend for vLLM does not support overlaying tensors with different dtypes in the same buffer. This is because `jax.Array` (used for allocating buffers) are strongly typed. As a workaround, tpu-inference allocates buffers *per layer*, rather than *per group slice*. Clearly, the number of block IDs needed for all groups to have separate slots of the same buffer outnumbers the number of block IDs needed for a layer to store all of its pages in its own buffer. To prevent out of bound block IDs, tpu-inference pads the `page_size_bytes` of all layers to be [8]

$$\text{uniform\_page\_size\_bytes} = \text{num\_attention\_groups} *\text{attention\_page\_size} \\ + \text{num\_mamba\_groups}* \text{unpadded\_mamba\_page\_size}$$

This trick achieves a uniform page size while sizing the number of block IDs (block pool size) to fit in the bounds of the per-layer buffers. For example, in the unit test for the padding formula above [9], we have a model with a 1:1 ratio of full attention to Mamba layers. The unpadded mamba page size is $3 * 12288 * 2 + 64 * 128 * 128 * 4$ bytes, which is product of dimensions in convolutional state times number of bytes to store a bf16 data type, plus the product of dimensions in the recurrent state times number of bytes to store a float32 data type. The attention page size is $1081344$ bytes. We pad every layer's page size to be `attention_page_size` + `unpadded_mamba_page_size`

When vLLM computes the total number of blocks to allocate for a KV cache group, it will divide its `max_memory_usage_bytes` (computed by vLLM), by this padded page size. This padded page size works by giving the illusion that a page holds multiple pages, one from each KV cache group. 

In the vLLM core, we call `get_kv_cache_specs()` which calls `get_kv_cache_spec()` on each worker to obtain layer name -> `KVCacheSpec` mappings. Each `KVCacheSpec` stores the padded `page_size_bytes` set in `update_mamba_page_size_padded()`.

Next, we call `get_kv_cache_configs()`, which handles the following:

1) merges KV cache specs across workers into a global layer name -> `KVCacheSpec` mapping.
2) grouping layers in the model that have the same `KVCacheSpec` to create a list of `KVCacheGroupSpec`s

In the failing run with Qwen, the hybrid KV cache manager was enabled. Under this setting, vLLM unifies the KVCacheSpec page size, which appears to be a no-op since the `page_size_bytes` has already been standardized by `update_mamba_page_size_padded` and `_create_attention_spec`.

After ensuring a uniform page size across all specs, vLLM gets the KV cache groups using the exact same heuristic as in tpu-inference to obtain a list of `KVCacheGroupSpec`s, one for each cache group (see `_get_kv_cache_groups_uniform_page_size()` line 1516).

3) assigning all or some of a KV cache groups layers to each worker (projection) and then checking there is enough memory on each worker to store its part of the KV cache

Projection: the layer->KVCacheSpec mapping for each worker is passed as a list argument to `get_kv_cache_configs()`. `KVCacheGroupSpecs` are assigned to workers depending on whether tensor and/or pipeline parallelism is enabled. Since the failing run only had tensor parallelism enabled, each worker should receive every `KVCacheGroupSpec` and run JAX SPMD on its mesh. 

Memory checks: The available memory in bytes for each worker is defined in the `available_memory` array and optionally can be overridden by `num_gpu_blocks_override`. In the absence of an override, vLLM  subtracts the size of a null block from each worker's available memory [7]. The size of this null block is the sum of `page_size_bytes` over layers in a group, which lines up with our intuition that `page_size_bytes` measures the size of one layer's block.

After this initial accounting for null blocks, vLLM does another pass over all workers, calling `_check_enough_kv_cache_memory()` with `_max_memory_usage_bytes_from_groups()` as the function for computing needed memory.  First it computes the same sum of `page_size_bytes` over all layers in a KV cache group, and stores it as `bytes_per_block`. Then the function divides a group spec's `max_memory_usage_bytes` by its `page_size_bytes` to get the number of blocks for the group. Finally it returns `bytes_per_block` * `num_blocks` as the amount of memory needed to store the KV cache. 

Each attention spec implements their own `max_memory_usage_bytes`; for `MambaSpec`, this is a constant value unless the mamba_cache_mode is "all", which TPU does not support. But Mamba bytes (included in block_size_bytes) are over-billed when we multiply them by sequence length, since mamba state should be constant w.r.t to sequence length.

So the needed memory for the worker is an overestimate and vastly exceeds the amount of available device memory. So vLLM returns the following error:

> ​ ValueError: To serve at least one request with the model's max seq len (65536), (596.12 GiB KV cache is needed, which is larger than the available KV cache memory (54.55 GiB). Based on the available memory, the estimated maximum model length is 5904.

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

[4] The part of the code where deepseek layers are defined to provide their own layers is line 68 of `tpu_inference/runner/kv_cache_manager.py`

[5] [Single memory pool allocation](https://docs.vllm.ai/en/latest/design/hybrid_kv_cache_manager/?h=#definitions:~:text=We%20use%20a%20single%20memory%20pool%20for%20all%20layer%20types)

[6] The current heuristic for determining group size can be found on line 262 of `kv_cache_manager.py`.

[7] (technically the group with max sum is selected, but in this case they're all the same)

[8] See `update_mamba_page_size_padded()` for the formula.

[9] Relevant unit test is `test_get_kv_cache_spec_hybrid_mamba_cache_config_updates`