# KV cache assignment via indirect scatter

**Status:** proposed — implementation CPU-complete, **blocked on device** (see
[Device status](#device-status))
**Date:** 2026-08-12
**Target:** `hf-adapters` — replace int-specialized slice-assignment KV writes
with a single tensor-indexed scatter.

## Problem

`kv_cache_update` wrote K/V by native slice assignment at a Python-int
`cache_position`:

```python
key_cache[:, :, cache_position : cache_position + seq_len, :] = k_write
```

`is_filling`, `token_index`, and `cache_position` are Python scalars threaded
through `torch.compile(..., dynamic=False)` blocks, so Dynamo specializes on
them. Consequences:

1. **One compiled binary per distinct `cache_position`.** The decode loop's fill
   arm walks `token_index` 1..63, so a model compiles up to `BLOCK_SIZE` fill
   binaries plus an expansion binary — per layer shape. The original docstring
   already flagged this; `torch_spyre` bumps Dynamo's `cache_size_limit` to 1024
   to survive it.
2. **The fill step computes 64 rows to keep 1.** It feeds a full `BLOCK_SIZE`
   input, projects Q/K/V for all 64 positions, then discards 63 K/V rows.

Measured (granite-3.3-8b fp16, 2026-08-12): decode ~783 ms kernel vs prefill
~396 ms; 19.0 ms/layer decode vs 9.3 ms/layer prefill; decode barely moved
(783→718 ms) when `seq_len` dropped 512→128, because it always pushes
`bs × 64` queries regardless of prompt length.

## Design

### Write: one tensor-indexed scatter, in place

```python
def kv_cache_update(k, v, key_cache, value_cache, cache_index, source_index=None):
    if source_index is not None:
        k = k.index_select(2, source_index)
        v = v.index_select(2, source_index)
    key_cache.index_copy_(2, cache_index, k)
    value_cache.index_copy_(2, cache_index, v)
    return key_cache, value_cache
```

`cache_index` is a device tensor, so its value is not a compile-time constant:
**one binary serves every position** (measured `unique_graphs == 1`).

Three properties are load-bearing and were each verified on hardware:

- **In place, not out of place.** `index_copy` (no underscore) allocates a fresh
  cache per layer per step, and the copy comes back with the **default** device
  layout instead of the pinned one (measured: `[576, 8, 2, 1, 64]` in,
  `[8, 576, 2, 1, 64]` out). Feeding that to the next step's scatter would
  silently write to wrong rows. `index_copy_` keeps the pin across sequential
  writes and still compiles to one binary.
- **`k`/`v` keep their `[B, n_kv, n, head_dim]` shape.** Both this and a
  position-major `[n, B*n_kv, hd]` boundary measured free (relayouts=0,
  graphs=1), so the shape that requires no call-site changes wins.
- **`source_index` preserves the old fill semantics exactly.** The previous code
  wrote `k[:, :, token_index, :]` — an offset from the **front** of the block,
  *not* its tail. Those coincide only at `token_index == BLOCK_SIZE - 1`.
  `source_index` encodes the wasteful compute-64-store-1 behaviour deliberately;
  narrowing the decode step is explicitly out of scope (below).

### Cache layout: logical shape unchanged, device layout pinned

Indirect access requires the indexed dimension at **device position 0**. The
`SpyreTensorLayout` decouples device layout from logical shape, so the cache
keeps its natural `[B, n_kv, max_cache_len, head_dim]` shape — `index_copy_`
writes dim 2, and SDPA reads the cache directly with **no views**:

```python
device_size = [max_cache_len, n_kv, sticks, batch_size, eps]   # L outermost
stride_map  = [head_dim, max_cache_len * head_dim, eps,
               n_kv * max_cache_len * head_dim, 1]
```

The batch sits between the stick-count dim and the elems-per-stick dim.

**Logical permutation cannot substitute for the pin.** Moving the indexed dim to
logical position 0 makes the device position *worse*:

| logical shape | indexed dim | resulting `device_size` | indirect device pos |
|---|---|---|---|
| `[B, n_kv, L, hd]` | 2 | `[8, 576, 2, 1, 64]` | 1 |
| `[L, n_kv, hd]` | 0 | `[8, 2, 576, 64]` | 2 |
| `[L, B, n_kv, hd]` | 0 | `[4, 8, 2, 576, 64]` | 3 |

`device_layout=` requires a live RuntimeContext, so the allocator primes
torch-spyre autoload with `torch.empty(1, device=DEVICE)` first (mirroring
`_move_to_spyre_with_layout`) and falls back to a plain `torch.zeros` off-device.

### Read: unchanged

Attention still reads the full `max_cache_len` window with the additive mask
suppressing unwritten and padding columns. `build_prefill_mask`,
`build_expansion_mask`, and `add_causal_sliding_window_band` are **untouched**,
and cache-coordinate semantics (column `c` holds absolute position
`c - prompt_offset`) are preserved. Keeping the mask contract fixed is what
bounds this change.

### Decode loop

`make_cache_index(start, length, device)` expresses all three shapes:

| Step | before | after |
|---|---|---|
| Prefill | `is_filling=False, token_index=0, cache_position=0` | `cache_index=make_cache_index(0, padded_len)` |
| Fill | `is_filling=True, token_index=t, cache_position=fill_pos` | `cache_index=make_cache_index(fill_pos, 1)`, `source_index=make_cache_index(t, 1)` |
| Expansion | `is_filling=False, token_index=0, cache_position=cur-64` | `cache_index=make_cache_index(cur-64, BLOCK_SIZE)` |

`tokens_in_block` / `decode_pos` / `current_cache_len` bookkeeping and
`decode_block_walk` are unchanged.

### Sliding-window adapters need `block_base`, not the write position

Gemma 3 and Gemma 4 used the removed triple for something other than the KV
write: the sliding-window band's query coordinates, as
`block_base = cache_position - token_index`. That is now derived from the write
args:

```python
block_base = int(cache_index[0])
if source_index is not None:
    block_base -= int(source_index[0])
```

Subtracting `source_index` matters — on a fill step `cache_index[0]` is the
written slot, which differs from the block base by `tokens_in_block`.

## Scope

Converted: `hf_common` (`make_standard_gqa_block`, `make_decoder_block`, the
backbone/forward wrappers, `generate`, the embedding prefill path), adapters
`hf_gemma3`, `hf_gemma4`, `hf_gpt2`, `hf_gpt_neo`, `hf_gpt_neox`, `hf_granite`,
`hf_granite_vision`, `hf_granitemoehybrid`, `hf_olmo2`, `hf_phi3`, `hf_qwen3`,
`hf_smollm3`, the VLM arms (`hf_gemma4_mm`, `hf_mistral3_vision_mm`,
`hf_granite_vision_mm`), three test harnesses, and
`scripts/profile_prefill_decode_spyre.py`.

`_dspark_common.py` has no `kv_cache_update` call (one-shot concat-KV drafter) —
unaffected.

### Out of scope

- Paged KV / block tables / continuous batching.
- Narrowing the SDPA read window to live slots only.
- Reducing the decode step from 64 queries to 1. Now *unblocked* by this change
  (the write no longer forces a block) but it touches the mask builders and
  `decode_block_walk`, so it lands separately. This is where the remaining
  19.0 ms/layer decode cost lives.
- **Hoisting the Gemma sliding-window mask out of the forward pass.** Gemma 3 and
  Gemma 4 are the only adapters that read a scalar out of `cache_index`
  (`int(cache_index[0])`, for the band's `block_base`), which syncs from the
  device. Accepted as-is for now: it runs once per step rather than per layer, in
  eager code outside the compiled block, and
  `add_causal_sliding_window_band` already round-trips the whole mask through CPU
  by necessity (Spyre's Inductor backend rejects int64 compare-to-constant and
  bool intermediates), so the scalar read is noise beside it. The real fix is to
  build the per-layer-type masks *before* the forward pass — the decode loop
  already knows every step's `block_base` in advance — which removes the sync and
  the per-step mask construction together. That is a mask-construction change, and
  this PR's boundary is "mask builders unchanged", so it lands separately.

## Device status

**The scatter does not currently lower inside a real decoder block.** Bisected on
Qwen3-0.6B, all else equal:

| Write | Layout pin | Result |
|---|---|---|
| old slice assignment | off | **passes** |
| `index_copy_` | off | **fails** |
| `index_copy_` | on | **fails** |

So the layout pin is *not* the cause; `index_copy_` in the real block is:

```
InductorError: Incompatible host_size and dim_order
  propagate_layouts.py:1017 _is_supported_layout
  via _multi_arg_pointwise_layouts, output FixedLayout size=[1, 8, 1, 128]
                                            stride=[1024, 128, 1024, 1]
  failing STL: size=[1, 64, 8, 2, 1, 64] dim_order=[3, 5, 2, 4, 1]
```

Six size entries against five `dim_order` entries. `_is_supported_layout`
projects the output's `dim_order` onto each input with
`[d - rank_diff for d in dim_order if d >= rank_diff]`, which can drop entries
and break the `len(host_size) == len(dim_order)` invariant asserted at
`spyre_tensor_impl.cpp:144`.

Four hypotheses were tested standalone and **rejected** (each passed): device
layout rank, `k` as a transposed view, `seq_len == 1`, and `apply_rope_matmul` in
the k path. The trigger requires something the real block supplies that those
synthetic graphs do not; a downward bisection from the real model is in progress.

Related: torch-spyre#3705 (unpinned indirect *scatter* silently writes to wrong
rows — a separate, already-filed bug).

## CPU verification

- `tests/cpu/test_kv_cache_scatter.py` (new): 13 tests — shapes, heterogeneous
  `_spyre_kv_shapes`, `B=1`/`B=4`, MQA, `head_dim` 256, non-contiguous indices,
  sequential decode writes, in-place semantics, equivalence with the old slice
  write, and fill-step equivalence at `tokens_in_block ∈ {0, 1, 17, 63}`.
- `tests/cpu/test_adapter_cpu_accuracy.py`: 17 pass. The one failure
  (`google/gemma-4-12b`) is pre-existing — a transformers per-layer-attribute
  config error that reproduces identically on the unmodified baseline.
- Wider CPU suite: 66 pass (`test_vlm_e2e_cpu.py` needs `PIL`, not installed).

## Risks

| Risk | Mitigation |
|---|---|
| Lost layout pin silently corrupts K/V (torch-spyre#3705) | `assert_kv_cache_layout` guard; in-place write keeps the pin. Note CPU cannot detect this — device-only |
| fp16 device round trip differs ~1 ULP from IEEE fp16 | Expected, not a defect: a bare `.to("spyre").cpu()` with no scatter shows the same 1 ULP on ~50% of elements. bf16 is exact |
| Gemma 3/4 sliding-window coordinates | `block_base` derivation above; covered by CPU accuracy tests |
| `index_copy` with duplicate indices is order-undefined | `make_cache_index` produces distinct indices by construction |
