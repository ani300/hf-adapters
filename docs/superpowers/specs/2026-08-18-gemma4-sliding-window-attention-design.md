# Sliding-window attention for Gemma 4 via `spyre::sliding_window_attention`

**Status:** proposed — no implementation yet
**Date:** 2026-08-18
**Target:** `hf-adapters` — replace the band-masked full-cache SDPA on Gemma 4's
sliding layers with `spyre::sliding_window_attention` from
[torch-spyre#3405](https://github.com/torch-spyre/torch-spyre/pull/3405), plus a
`valid_start` extension to that op.

## Problem

Gemma 4's sliding layers score the **whole** KV cache and then throw most of it
away behind an additive band. `hf_gemma4._build_layer_masks` intersects the base
causal mask with `add_causal_sliding_window_band` and hands the result to
`F.scaled_dot_product_attention` over the full-length cache.

For `google/gemma-4-12b`: `sliding_window=1024`, **40 of 48 layers** are
`sliding_attention`, sliding `head_dim=256`, `num_key_value_heads=8`,
`num_attention_heads=16`, `max_position_embeddings=262144`. At a context of `L`
tokens a decode step reads `8192 * L` bytes of K+V per sliding layer, of which
only the newest 1024 rows can ever contribute.

PR #3405 adds `spyre::sliding_window_attention`, which reads only the window:

| shape | masked full attention | the op |
|---|---|---|
| decode `Lq=1, Lkv=4096, W=64` | `[1, 4096]` | `[1, 64]` |
| prefill `Lq=Lkv=512, W=64` | `[512, 512]` | 8 x `[64, 128]` |

Three obstacles stand between that op and Gemma 4.

### 1. The op's geometry is trace-time integers

`sliding_window_attention(query, key, value, window_size, is_causal, scale,
cache_seqlen, buffer_origin)` takes `cache_seqlen` and `buffer_origin` as Python
`int`s, and the band comes from `spyre::window_band_mask`, also built on CPU from
`int`s. Under `torch.compile(dynamic=False)` every distinct value is a distinct
compiled graph. Gemma 4 compiles ~2 variants per block today; a naive
integration multiplies that by the number of decode positions.

This is the same failure the KV write hit before hf-adapters#330, which fixed it
by making `cache_index` a **tensor**. That escape is not available here: the op's
whole design rests on read offsets being trace-time constants ("The Python loop
is unrolled at trace time, so every read offset is a constant").

### 2. The op has no mask input, and `generate` left-pads

`generate` block-pads prompts to a `BLOCK_SIZE` multiple with real tokens
**right-aligned**, so columns `[0, prompt_offsets[b])` hold K/V computed from pad
tokens. Today `build_prefill_mask` / `build_decode_mask` mask them. The op cannot:
windowing is an offset plus a length. Those columns sit inside the window and
would be attended, silently.

### 3. The obvious acceptance test is already red

Device token-compare for `google/gemma-4-12b` fails 0/5 and diverges at prefill
(recorded 2026-08-17 on the `kv-cache-indirect-scatter` branch; both gemma4
entries are in `NON_BLOCKING_CAUSAL_MODELS`). "Does Gemma 4 still produce good
tokens" cannot tell us whether the SWA replacement is correct.

## Design

### Call the op in buffer-relative coordinates

From `SlidingWindowPlan`:

- `read_start(qi) = min(floor_stick(window_start_logical - buffer_origin),
  cache_capacity - buffer_width)` — so when `cache_capacity == buffer_width`,
  **`read_start` is identically 0**.
- the band depends only on
  `delta = (cache_seqlen - seqlen_q) - buffer_origin`.

So the compiled graph is invariant exactly when `cache_seqlen - buffer_origin`
and `seqlen_q` are pinned. Everything below follows from arranging that. The
`min(..., cache_capacity - buffer_width)` clamp shows the op was designed for
this case.

### Phase 1 — no-roll, test-only

Keep the existing full-length cache and `kv_cache_update` untouched; pass
`buffer_origin=0` and `cache_seqlen = int(cache_index[0]) + Lq`. The op
explicitly supports a still-filling, zero-filled cache, so this is nearly a
drop-in: sliding layers stop building the band and call the op.

**This phase is a correctness harness, not a step toward performance.**
`read_start` stick-floors on a growing `cache_seqlen`, so it recompiles once per
64 decode steps without bound, and the cache does not shrink. Its only job is to
compare op-vs-band numerics on short generations with one variable changed. It is
never a default and carries no perf claim.

### Phase 2 — the anchored compact buffer

Sliding layers get `cache_capacity = buffer_width = round_up_64(W + 64)`:
**1088 rows** for Gemma 4 (`W=1024`), 576 for Gemma 3 (`W=512`), replacing
`prompt + generation`.

**Invariant.** At the start of each 64-token stick period, physical rows
`[0, 1024)` hold the most recent 1024 tokens, all real; rows `[1024, 1088)` are
empty and receive the next 64 tokens. Token `m` of the period is written at row
`1024 + m` (via `kv_cache_update`, whose `cache_index` is already a tensor). After
the 64th write, rows `[64, 1088)` shift down to `[0, 1024)` and the invariant is
restored.

**The op is called with `cache_seqlen = 1088`, `buffer_origin = 0`,
`seqlen_q = 64`** — all three constant for the whole generation. Therefore
`query_blocking(64)` gives `q_block = 64` and a single Q block, `read_start = 0`,
and `delta = 1024`: there is **one decode graph**.

Decode presents a full 64-row query stick rather than one row: the real token's
query goes at index `m` (matching its K row at `1024 + m`), the other 63 rows are
zeros, and the output row is recovered with `index_select` on a **1-element
tensor index** so that `m` never enters the graph as a constant.

Why the surrounding rows are safe without any new masking:

- Rows above `1024 + m` are unwritten zeros at *later* coordinates, so the causal
  band already excludes them from the real row (`delta < 0`).
- Rows below are all real, because the shift happens *before* a stick of writes
  rather than after. Hence **`valid_start = 0` in the steady state** — no
  per-step variant.

**Prefill stays one-shot.** A `P`-token prompt needs
`cache_capacity >= round_up_64(W + P)` for a single call, so sliding layers
prefill into a prompt-sized buffer (`max(round_up_64(P), 1088)`; the floor
matters for short prompts) with `cache_seqlen = P`, `buffer_origin = 0`, and
`valid_start = prompt_offsets`, which is constant across prefill and so costs one
graph. At the prefill/decode boundary the buffer is **compacted once**: the last
1024 rows are copied into rows `[0, 1024)` of a fresh 1088-row buffer and the
prompt-sized allocation is dropped. Steady-state decode memory is 1088 rows
regardless of context length; peak memory is unchanged from today.

When `P < 1024` the compaction right-aligns the prompt at rows `[1024 - P, 1024)`
and `valid_start = 1024 - P`, eroded by 64 at each shift — at most
`ceil((1024 - P) / 64) <= 16` transient variants, each reused for 64 steps.
For `P >= 1024`, exactly one.

### Graph count and cost, compared

| Integration | Decode graphs / block | Shift traffic | Verdict |
|---|---|---|---|
| Phase 1 (full cache, no roll) | 1 per 64 steps, unbounded | none | harness only |
| Compact, shift 1 row per step | 1 | full buffer read+write **every** step | traffic exceeds the saving |
| Compact, shift 64 rows per 64 steps, `Lq=1` | 64 | 1/64 of the above | ~32x today's compile time |
| **Anchored compact buffer** | **1** (2 with the evict branch) | 1/64 | chosen |

Order-of-magnitude arithmetic, to be **measured, not assumed**. Per decode step
across 40 sliding layers, K+V read: `8192 B/token/layer`, so ~2.7 GB at a
context of 8192 tokens versus ~356 MB windowed (1088 rows) — about 7.5x less.
The amortized shift adds ~11 MB/step. The 64-row query stick adds ~46 GFLOP/step
(scores plus `P·V`, 16 heads x 64 rows x 1088 columns x 256), which is small
against a memory-bound step but **not free**, and it widens the score tile to
`64 x 1088` in fp16 (~139 KB per head-tile), which is LX pressure worth watching.

### Eviction primitive

`aten::roll` has no Spyre lowering, no Spyre decomposition and no CPU fallback
(stated in `kv_window`'s docstring). The shift uses the primitive
`kv_cache_update` already trusts:

```python
tail = cache.index_select(2, src_index)      # rows [64, 1088)
cache.index_copy_(2, dst_index, tail)        # into rows [0, 1024)
```

In place on the destination, so the pinned device layout survives — the property
`kv_cache_update`'s docstring calls load-bearing (a fresh tensor comes back with
the **default** layout and silently writes wrong rows, torch-spyre#3705). Cost is
one temp of ~4 MB for K and ~4 MB for V, per layer, per 64 steps.

Rejected: `torch.slice_scatter` (has a Spyre lowering at `lowering.py:1289`, but
is functional and so returns an unpinned tensor — exactly the #3705 failure).
Documented fallback: ping-pong between two compact buffers, both allocated
pinned, at 2x steady-state memory.

Eviction is a `bool` argument to the compiled block, so decode costs **2 graphs
per sliding block** rather than 64. Making it unconditional with tensor indices
would give 1 graph but forces a full-buffer copy every step.

### `valid_start` — the torch-spyre change

Add `valid_start: Optional[list[int]]` (first non-pad column per batch entry) to
`spyre::sliding_window_attention`, its `register_fake`, and
`spyre_sliding_window_attention`; thread it to `spyre::window_band_mask`, whose
predicate becomes:

```python
allowed = (0 <= delta < window_size) & (column >= valid_start[b])
```

Still built on CPU, still from `int`s only — no device mask tensor, no per-block
slice, no extra memory traffic. The band stays `[1, 1, q_block, buffer_width]`
when the values are uniform and widens to `[B, 1, q_block, buffer_width]` only
when they differ. `rejection_reason` validates `len(valid_start) == batch` and
`0 <= v <= cache_seqlen`. `block_is_fully_attended` must return `False` whenever
any `valid_start` is non-zero.

Rejected: a general additive `attn_mask` tensor. It is more general but adds a
device slice per block, defeats the `block_is_fully_attended` fast path, and
would force the adapter's mask builders into physical coordinates for phase 2 —
while still not expressing the one case that actually needs generality (see
Out of scope).

### Code boundaries

| File | Change |
|---|---|
| `hf_adapters/swa_attention.py` (new) | `sliding_window_attention(...)` dispatcher — `spyre` to the op, `cpu` to a masked reference over the same compact buffer. `SlidingWindowCache`: capacity, live rows, shift schedule, compaction, and the `cache_seqlen`/`buffer_origin`/`valid_start` derivation. |
| `hf_adapters/hf_common.py` | `kv_cache_shapes` / `allocate_kv_caches` gain a **per-layer capacity**: sliding layers want 1088 while global layers want `prompt + generation`. Today `max_cache_len` is one value for every layer. |
| `hf_adapters/hf_gemma4.py` | `Gemma4Attention` gains `is_sliding`; sliding layers call the dispatcher instead of `F.scaled_dot_product_attention`; `_build_layer_masks` stops building the sliding band on the op path; shift and post-prefill compaction happen in `_run_blocks_over_embeds`, which already holds the caches and `cache_index`. |
| `hf_adapters/hf_gemma3.py` | The same replacement, as the green control. |

`generate()` is untouched. This is hf-adapters' **first** `torch.ops.spyre.*`
call, so the coupling to a torch-spyre-private API stays in one file;
`hf_common.py` is already 2744 lines.

## Testing

1. **CPU bookkeeping** — `tests/cpu/test_swa_cache.py`. Integer tests for
   capacity, live rows, shift schedule and `valid_start` (including the
   `P < 1024` erosion), plus the compact-buffer reference against HF's own
   sliding mask over a synthetic sequence, batch-1 and ragged-batch. No device.
   Catches every off-by-one cheaply, and is the only place the phase-2 arithmetic
   is exhaustively covered.
2. **Layer-level A/B on device** — the **primary gate**. One Gemma 4 sliding
   block built from a shrunken `Gemma4TextConfig` with random weights; identical
   inputs *and* identical cache contents; band-masked SDPA versus the anchored op.
   Sweep: prefill `Lq` in {64, 512}; decode across a stick boundary; decode
   across a shift; `valid_start` in {0, 17}. Isolates SWA from the pre-existing
   Gemma 4 divergence and needs no healthy checkpoint.
3. **Gemma 3 1B control** — port the dispatcher into `hf_gemma3` and require
   `tests/spyre/test_e2e_token_compare_spyre.py -k gemma-3-1b` to stay green. It
   is cached locally at `/tmp/models/huggingface_cache/hub` and is **not** in
   `NON_BLOCKING_CAUSAL_MODELS`, so unlike Gemma 4 it is a trustworthy end-to-end
   baseline.
4. **Gemma 4, non-gating** — record device token-compare before and after. It
   stays red for the pre-existing prefill divergence; the requirement is no *new*
   failure mode (NaN, `Unsupported`, hang).

Tolerances: fp16 device comparisons use the tolerances the neighbouring suites
use (`rtol=1e-2`, `atol=1e-3`); bit-exactness must not be asserted, since the
windowed softmax reduces `buffer_width` terms where the reference reduces
`seqlen_kv` and `sum` blocks differently at different widths.

## Scope

Phase 1 and phase 2 land in that order, in this repo, for `hf_gemma4` and
`hf_gemma3`. The `valid_start` change is developed on a local branch off
`swa-window-roll` and offered upstream as a stacked PR (or patch) to
@abhishekkunuru6-cmyk once the Gemma 4 evidence exists — #3405 is someone else's
open PR and is not modified uninvited.

### Out of scope

- **`hf_gemma4_mm`.** Its bidirectional vision overlay *widens* attention, and an
  additive mask can only remove; `is_causal=False` raises `Unsupported` in the op
  today. The VLM keeps the band-masked path.
- **Chunked prefill.** It would shrink only the transient prefill allocation, and
  it is a large change to `generate()` that also affects global layers. The
  compaction step delivers the steady-state memory win without it.
- **bfloat16.** `gemma4_base` is bf16 and every test in #3405 is fp16. Flagged to
  verify, not to fix.
- **Bidirectional windows.** Not implemented in the op.

## Risks

- **`head_dim=256` is untested in #3405** — every test there uses 64. Gemma 4's
  sliding layers are 256, so `kv_window`'s transposed slice at `E=256` is the
  first thing to smoke, before anything else.
- **`torch_spyre` does not currently import in this workspace** (backend
  extension load failure); `source torch-spyre-docs/scripts/dev-env.sh` first.
- **LX pressure from the 64-row query stick** (`64 x 1088` fp16 score tile).
  If it does not fit, the fallback is `Lq=1` with 64 graph variants, which
  changes the phase-2 economics and would need re-deciding.
- **The self-referential shift** (`index_select` from a cache that
  `index_copy_` then writes) may fuse in a way that reads already-overwritten
  rows. Ping-pong buffers are the documented fallback. The CPU bookkeeping test
  cannot catch this; the layer-level A/B across a shift is what does.
- **Perf numbers in this document are arithmetic, not measurements.** No
  performance claim should be repeated until it is measured on device.
- **`key.shape == value.shape` is required** by `check_window_read`. Gemma 4
  sliding layers satisfy it (256/256); Gemma 3 does too. Any future model with
  `v_head_dim != head_dim` cannot use this path.

## Milestones

0. Unblock: `dev-env.sh`; fetch `swa-window-roll`; smoke `kv_window` and
   `sliding_window_attention` at `head_dim=256`, GQA 16/8, fp16.
1. `valid_start` in torch-spyre on the local branch, with tests in that repo's
   style.
2. `swa_attention.py` dispatcher plus the CPU bookkeeping suite (phase 1, CPU).
3. Phase-1 device A/B on the shrunken Gemma 4 block.
4. Phase 2: per-layer capacity, compaction, shift, tensor-index output gather;
   extend the A/B to cover a shift.
5. Gemma 3 1B control green; Gemma 4 non-gating record.
6. Upstream the `valid_start` patch with the evidence attached.
