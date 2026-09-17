# Issue Review: KV Cache Size Accounting Bugs

Issue URL: https://github.com/vllm-project/tpu-inference/issues/3483

## Which models are affected by the bug?

Hybrid Gated DeltaNet models [1], which include both full attention and a variation of the Mamba layers proposed in [2]; for these models, KV cache size can be *overestimated*. DeepSeek v4 models have layers that are not included in the hybrid KV cache manager's accounting routine; for these models, KV cache size can be *underestimated* [4].

## Root cause of the bug:

vLLM's Hybrid KV Cache Manager allocates memory for all layer types from a single pool [5]. Memory is allocated in units of pages; **page size** is defined as the physical size of a block. Each block stores `block_size` tokens, each consuming `kv_hidden` bytes. Page size is then `block_size` * `kv_hidden`  * `num_layers`, which means that page size can vary by attention type. But allocating pages from a single pool means we must use a uniform page size across all layers.

**KV cache groups** are the solution to this problem. vLLM will allocate memory for a group of `group_size` identical layers at a time, rather than call the allocator for each layer. All groups have the same number of layers, so the page size remains uniform across groups. Note that the `num_blocks` per layer (and by extension group) can and should vary in the case of hybrid models. The value of `group_size` should be chosen strategically to reduce the amount of padding in each group [6]. vLLM then divides physical memory into `group_size` buffers, each storing `num_groups` layers, one from each group (group slice). The buffers are also referred to as `KVCacheTensor`s.

However, the TPU backend for vLLM does not support storing tensors from different layer types in the same buffer. This is because `jax.Array` (used for allocating buffers) are strongly typed. As a workaround, tpu-inference allocates buffers *per layer*, rather than *per group slice*. The number of block IDs that can be used to index a group's blocks outnumbers the number of block IDs used to index a layer's blocks. In other words, the amount of memory needed to store a layer's blocks is less than the amount of memory needed to store multiple layer's blocks.

This discrepancy prompts a workaround in tpu-inference (see `update_mamba_page_size_padded()`) that pads the page size of all layers to be

$$\text{uniform\_page\_size\_bytes} = \text{num\_attention\_groups} *\text{attention\_page\_size} \\ + \text{num\_mamba\_groups}* \text{unpadded\_mamba\_page\_size}$$

which is exactly the size of one `KVCacheTensor` and satisfies the uniform page size constraint from before. 

In the vLLM core, we call `get_kv_cache_specs()` which calls `get_kv_cache_spec()` on each worker to obtain layer name -> `KVCacheSpec` mappings. Each `KVCacheSpec` stores the padded `page_size_bytes` set in `update_mamba_page_size_padded()`. 

Next, we call `get_kv_cache_configs()`, which handles the following:

1) merges KV cache specs across workers into a global layer name -> `KVCacheSpec` mapping.
2) grouping layers in the model that have the same `KVCacheSpec` to create a list of `KVCacheGroupSpec`s 

In the failing run with Qwen, the hybrid KV cache manager was enabled. Under this setting, vLLM unifies the kv cache spec page size, which appears to be a no-op since the `page_size_bytes` has already been standardized by `update_mamba_page_size_padded` and `_create_attention_spec`. If the `page_size_bytes` were not standardized, vLLM will pad the Mamba page size to the largest page size. Unlike full attention layers, Mamba page size does not scale with block_size.

After ensuring a uniform page size across all specs, vLLM gets the KV cache groups using the exact same heuristic as in tpu-inference to obtain a list of `KVCacheGroupSpec`s, one for each cache group (see `_get_kv_cache_groups_uniform_page_size()` line 1516).

3) assigning all or some of a KV cache groups layers to each worker (projection) and then checking there is enough memory on each worker to store its part of the KV cache

The layer->KVCacheSpec mapping for each worker is passed as a list argument to get_kv_cache_configs. The failing run in the issue only used tensor parallelism but not pipeline parallelism. This means each worker should receive part of each of the model's layers since tensor parallelism splits the model width-wise. 


4) checking that there is enough memory on each worker to store its part of the KV cache

This step is where the error in this issue comes from. One of the arguments to get_kv_cache_configs is an array `available_memory`, containing the number of bytes available per worker. 

If no num_gpu_blocks_override is set:

From each worker's available memory, we subtract the max sum of `page_size_bytes` of each layer in a group over all groups. Note in our case, all groups have the same `page_size_bytes` (due to padding) so the max is just the sum of `page_size_bytes` for any of the groups. Take for example, a model with 10 full attn + 30 mamba, group_size is 10 and there are four groups. Each page_size_padded sums bytes of one layer from each four groups. Add 10 `page_size_padded` to get the actual number of bytes consumed by the KV cache. 

Up to this point, the code seems to be working properly. 






If there is no num_gpu_blocks_override set, we subtract _pool_bytes_per_block(groups) bytes from each worker. 


To check if there is enough memory, vLLM first initializes an array available_memory , 

check_memory = [
        avail_mem - _pool_bytes_per_block(groups) if groups else avail_mem
        for groups, avail_mem in zip(projected_groups_per_worker, available_memory)
    ]






When does the total number of blocks differ from the number of allocated blocks in get_kv_cache_configs()?

_max_memory_usage_bytes_from_groups computes a total num_blocks that is different than `_get_kv_cache_bytes_per_block()`
5) 
grouping per-layer KV cache specs and ensuring they all have the same page size



 in which we call `get_kv_cache_config_from_groups()`. This function is responsible for computing `num_blocks`. To compute `num_blocks`, the function `get_kv_cache_config_from_groups()` first calls `_get_kv_cache_bytes_per_block()`

`_get_kv_cache_bytes_per_block()` then  Sanity check: physical block in memory stores page_size_bytes from each of the layers in a group, which matches the [equation](https://docs.vllm.ai/en/latest/design/hybrid_kv_cache_manager/#definitions:~:text=page%20size%3A%20the%20physical%20memory%20size%20of%20a%20block%2C%20defined%20as%3A) in the vLLM docs.

Finally, we compute  the number of blocks allocated to the KV cache using the lines below

```python
num_blocks = available_memory // bytes_per_block
num_blocks = may_override_num_blocks(vllm_config, num_blocks)
```

Once we have determined the number of blocks, 

The error in the original issue comes from the _check_enough_kv_cache_memory function, 

Finally, we call initialize_from_config in the engine core

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

Read test_get_kv_cache_spec_hybrid_mamba_cache_config_updates in its entirety before dumping variables on a TPU.


`get_kv_cache_spec()` in class KVCacheManager calls `update_mamba_page_size_padded()` (where the eqn above is applied) then `_create_attention_spec()` for each of the layers in the model. `_create_attention_spec()` returns a `KVCacheSpec` for each layer. `get_kv_cache_spec()` then returns a dictionary mapping layer names to their KVCacheSpec.


get_kv_cache_spec in a TPU worker drills down to KVCacheManager's get_kv_cache_spec
TPUWorker
    TPUModelRunner
        KVCacheManager

```python
@dataclass(frozen=True)
class KVCacheSpec:
    """
    A base class for specifying the KV cache format of one layer.
    """

    # number of tokens in a block
    block_size: int
    ...
    page_size_bytes: int # where the padded page size is stored
    max_memory_usage_bytes: int
```

KVCacheSpec also includes `page_size_bytes`, the size of a block for one layer. The `max_memory_usage_bytes` for a layer. 




Extra detail 

 All layers in a group can share the same block table, but the same logical block ID will resolve to different physical block IDs for each layer.






However, this padding has unintended effects. When vLLM accounts for the KV cache size of a model, it sums the page size (which is already padded to account for the full KV cache size), and then sums the max memory usage of each layer, producing a vast overestimate of the true KV cache size.

## References

[1] [Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/abs/2412.06464)

[2] [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)

[3] [Hybrid KV Cache Manager in vLLM](https://docs.vllm.ai/en/latest/design/hybrid_kv_cache_manager/?h=)

[4] The part of the code where deepseek layers are omitted from cache size calculation is line 64 of `tpu_inference/runner/kv_cache_manager.py`

[5] [Single memory pool allocation](https://docs.vllm.ai/en/latest/design/hybrid_kv_cache_manager/?h=#definitions:~:text=We%20use%20a%20single%20memory%20pool%20for%20all%20layer%20types)

[6] The current heuristic for determining group size can be found on line 262 of `kv_cache_manager.py`.

## Code Tracing Experiment

## Understanding get_kv_cache_shape_with_mesh

get_kv_cache_shape_with_mesh

line-by-line plain english description:

(1) call get_mesh_shape_product with the mesh and the named axes ('model', 'expert') to get the number of devices to shard attention heads over.

(2) call get_mesh_shape_product with the mesh and the named axes ('dcp', 'pcp'). dcp and pcp refer to decode context parallelism and prefill context parallelism respectively. The return value is the number of DCP processes times the number of PCP processes but only one will be > 1 in practice.

We determine the physical block size by scaling the block size by the result of step (2). This allows us to allocate memory for the KV cache that will be sharded across the PCP or DCP devices. That means that every device receives a different part of the same logical KV cache block.

MLA branch is out of scope for this change, so we will focus on explaining the MHA/GQA branch. In this branch, we compute the KV cache shape using RPA implementation of get_kv_cache_shape. This gives us the size of the KV cache, padded to 32 bit words,

Finally we return a 5-tuple representing the shape of the KV cache to be allocated in GPU memory.

Mamba = alternative to attention layer in LLM that uses state space search

## Plain english description of update_mamba_page_size_padded

A KV cache group in vLLM spans multiple layers. In the comments, "type" refers to type of layer in an LLM. Group sizing math in `update_mamba_page_size_padded` works as follows:

if the greater of {num_attn, num_mamba} is within factor of 1.5 of the smllaer of {num_attn, num_mamba} then set group_size to be the greater of {num_attn, num_mamba}.
Effect: group size is 1 for every type

Otherwise, set group_size to smaller of {num_attn, num_mamba}.
Effect: no padding, but the larger of the two might have multiple groups.

## Simple Explanation of the Bug

vLLM counts the number of bytes needed to store the KV cache in order to allocate sufficient GPU memory for it. While the KV cache is organized in physical memory in units of blocks, vLLM uses pages (which store bytes for one block of one layer) to perform KV cache accounting.

Currently, vLLM assumes a common page size across all attention layers; this assumption breaks for hybrid architectures, such as ones that use full attention + GDN layers like in the bug report. To compute the uniform page size for these architectures, vLLM uses a heuristic that presently looks at two types of layers: full attention layers and mamba layers.

This heuristic sets the group size to be max(num_attn, num_mamba) when the two counts are sufficiently close. This group size then determines the page size in line 269 of update_mamba_page_size_padded. In this way, the page size ends up being determined by the layer geometry that appears more in the LLM's architecture.

There are two seperate mis-estimates that can arise as a result of this oversimplified accounting.

1) The KV cache resource manager looks at the max page size when determining if sufficient GPU memory exists, leading to an overestimate of the true KV cache size.
2) Layers that don't fall in the {full attention, mamba} set assume the same page size as all other layers, which can be estimate.

The mechanism to opt-out could make estimates accurate even as new layers are added, as the current opt-out only whitelists four DeepSeek v4 and mamba layers.

Why groups should be used to compute page_size instead of the model layers. A group is homogenous in that it stores KV caches of a certain type. A hybrid model can have different types of layers, on the other hand.

The uniform_page_size_bytes formula from update_mamba_page_size_padded:

uniform_page_size_bytes = (num_attn_groups *attn_page_size_bytes +
                                   num_mamba_groups* unpadded_mamba_page_size)

In Qwen, there are 10 attention layers and 30 mamba layers. So num_attn_groups = 1 and num_mamba_groups = 3

One block of one layer has a number of bytes given by the formula above. But notice that the formula above
sums page sizes for both kinds of layers.

Q: Why does TPU want a padded page size at all?

A: Each mixed tensor stores layers from different groups (4 shared layers per mixed tensor in Qwen). While the layers in a mixed tensor share the same physical tensor, they can be individually indexed using each layer's block_table. However, layers of different types cannot be overlaid on the same bytes in TPUs, so the TPU actually allocates a physical array per-layer. Project maintainers have mentioned this should be changed in the future (see jacobplatin comment on line 788 of).

In the current setup, the number of slots per layer is outnumbered by block IDs 4 to 1 (for Qwen). The solution is to pad the page size for each layer to be the same size as a single mixed tensor. This is said to correct num_blocks to match the actual physical tensor size allocated by the TPU but I'm still not sure about this.

This page size padding happens inside get_kv_cache_spec, and later the num_blocks get set in initialize_kv_cache

vllm expects page sizes to be unified? Does one page size per KV-cache group break this precondition?

also jacobplatin suggests: "we should not be replicating the kv cache for each layer" (tpu_inference/runner/kv_cache_manager.py, line 788)
