# Gemma-4 MoE: phase-dispatched two-method architecture

**Date:** 2026-08-20
**File:** `hf_adapters/hf_gemma4_moe.py` (adapter-only; no torch-spyre edit)
**Branch:** `pr3892-moe` (working tree; not committed until decode is exercisable)

## Problem

`hf_gemma4_moe.py` currently carries **four** selectable device FFN formulations,
chosen by mutually-exclusive module-global booleans and dispatched in
`Gemma4MoEBlock.forward` by a 4-way branch. Only two are wanted going forward, and
they should be selected by **execution phase**, not by a global flag set at import:

- **Prefill** (`seq_len > 1`, T=512): the dense all-experts **persistent** path
  (`spyre_hint({"E":128})` coarse-tile counted loop). Already benchmarked — warm
  7.3 s, 2.3× faster than chunked, emits `' the'`.
- **Decode** (`seq_len == 1`): the **loop-on-topk** path that gathers only the K=8
  activated expert rows per token, reworked onto the bespoke
  `spyre::index_mask(probs, kth)` router mask and the `fp32 → int32` idx2addr
  gather/scatter chain.

The other two formulations — **split** (host/device split) and **chunked**
(`_MOE_EC=32` mask-reduce) — are superseded and are removed.

This reduces three device formulations (persistent / chunked / split; loop-on-topk was
scaffold-only) to **two live methods**, dispatched by `seq_len`.

## Decisions (confirmed with user)

1. **Two methods, kept:** persistent (prefill) + loop-on-topk (decode). Deleted: split
   + chunked. ("we're going from 3 → 2 methods"; "persistent + loop-on-topk".)
2. **Dispatch by `seq_len`** inside `Gemma4MoEBlock.forward`: `seq_len > 1` → persistent,
   `seq_len == 1` → loop-on-topk. Both compiled regions coexist in `__init__`. No mode
   global gates the choice.
3. **Decode written ahead of its ops.** `spyre::index_mask` and the special `fp32→int32`
   cast op are not yet in `pr3892-moe`. Decode code is written to be ready for them;
   tracing the decode path will error at the missing op until torch-spyre ships them —
   **expected, not a regression**.
4. **Hard dispatch for decode.** `forward` routes `seq_len == 1` to the decode region
   unconditionally; it aborts at the missing op with a clear error. No temporary
   persistent fallback for decode (no dead branch to maintain).
5. **Whole-layer compile is a SEPARATE follow-up**, after this consolidation is verified.
   Not in scope for this change. (See "Deferred" below.)
6. **Keep in working tree** — do not commit until decode is exercisable end-to-end. The
   verified persistent prefill state is preserved by not touching the persistent body.
7. **KV `index_copy_` needs no change.** Debugger ground truth (2026-08-20): the KV
   scatter already lowers on-device inside `torch.compile` (core decomp rewrites
   `index_copy_` → `index_put` → Spyre Scatter); the `aten.index_copy.out is falling
   back to cpu` warning is purely an **eager-mode** artifact. No torch-spyre edit; do
   NOT remove the fallback entry (it governs eager dispatch + the compile-guard only).

## Current structure (line numbers, `hf_gemma4_moe.py`)

**KEEP — persistent (prefill):**
- `_moe_route_persistent_packed` (794) — router; calls `_moe_route_padded`
- `_pack_persistent_expert_weight` (825)
- `_moe_expert_persistent` (848) — the coarse-tile body; `.clone()` fix applied here
- `_moe_ffn_persistent` (903)

**KEEP — loop-on-topk (decode):**
- `_compiled_moe_loop_region` (447) — already keeps `[T,K]`, already has the B5–B8
  idx2addr chain (idx → expand-64 → restickify → fp32 → `[:32]`)
- `_moe_ffn_loop` (641) — **host `index_add` combine (lines ~681-685) moves to device**

**KEEP — shared router (used by persistent):**
- `_moe_route_padded` (696) — **`torch.where` threshold → `spyre::index_mask`**

**DELETE — split:**
- `_moe_route` (181), `_moe_permute` (203), `_grouped_gemm_4a` (228),
  `_group_offsets` (246), `_grouped_gemm_4b` (268), `_grouped_gemm` (317),
  `_moe_ffn` (337), `_moe_ffn_loop_ref` (374), `_compiled_moe_device_region` (426),
  `_compiled_device_gather` (534), `_moe_ffn_split` (545)

**DELETE — chunked:**
- `_moe_expert_chunk` (756), `_moe_ffn_chunked` (951)

**DELETE — globals** (`_MOE_GEMM_4B` 117, `_MOE_LOOP_ON_TOPK` 130,
`_MOE_CHUNKED_ONDEVICE` 161, `_MOE_PERSISTENT_ONDEVICE` 166, `_MOE_EC` 171):
the four mode-select booleans no longer gate anything once dispatch is by `seq_len`,
and `_MOE_GEMM_4B`/`_MOE_EC` belonged to deleted methods.

**KEEP — globals** (resolved by grep, not conditional):
- `_MOE_MAX_K` (98) — asserted in `prepare_for_spyre`.
- `_MOE_TILE` (122) — read at line 1167, passed to `_moe_ffn_loop` (decode). Survives.
- `_MOE_PADW` (177) + `_MOE_PAD_NEG` (178) — the topk-pad fix inside `_moe_route_padded`
  (padded before topk); still used by the surviving router. Re-home next to
  `_moe_route_padded`.

## Changes

### A. `Gemma4MoEBlock.forward` — seq_len dispatch (line 1089)

Replace the 4-way mode-global branch (lines ~1129-1180) with:

```python
seq_len = hidden_states.shape[1]          # [B, S, H]
if seq_len > 1:                            # prefill
    ffn_out = _moe_ffn_persistent(..., self._compiled_persistent_route,
                                   self._compiled_persistent, ...)
else:                                      # decode (seq_len == 1)
    ffn_out = _moe_ffn_loop(..., self._compiled_loop, ...)
```

Exact call signatures come from the surviving `_moe_ffn_persistent` /
`_moe_ffn_loop` definitions; no arg reshaping beyond what those already expect.
Delete the mutual-exclusion assert (lines ~1252-1255 region) — no globals to reconcile.

### B. `Gemma4MoEBlock.__init__` — compile handles (line 1034)

Keep: `_compiled_persistent_route` (1081), `_compiled_persistent`, `_compiled_loop`
(1076), plus the shared attention/MLP handles.
Delete: `_compiled_route` (1079, `torch.compile(_moe_route_padded)`) — grep-confirmed
used ONLY by the chunked branch (line 1152), which is deleted; the persistent router
uses `_compiled_persistent_route` and `_moe_ffn_loop` runs its router INSIDE
`_compiled_loop`, so nothing else references it. Also delete `_compiled_chunk`,
`_compiled_gather`, `_compiled_expert`, `_compiled_device_region`, and any split-only
handles. (The `_moe_route_padded` *function* survives — only this standalone compile
of it is dead.)

### C. `prepare_for_spyre` — weight materialization (line 1190)

Keep the persistent branch (pre-packs `[K,E,N]` backing via
`_pack_persistent_expert_weight`) and the loop-on-topk branch (HBM-resident
`gate_up_dev`/`down_dev` + `per_expert_scale`). Delete the chunked branch (per-chunk
offset-0 expert weights) and the split else-branch (host-resident experts). Both
surviving branches now run **unconditionally** for their compiled region — no longer
guarded by a mode global; both weight sets are always materialized so both regions can
run (prefill uses persistent weights, decode uses loop weights).

### D. Decode rework — `_compiled_moe_loop_region` + `_moe_ffn_loop`

Per plan `temporal-stargazing-taco.md`. `_compiled_moe_loop_region` already keeps
`[T,K]` and already has the idx2addr chain, so the remaining work is:

- **D1.** `_moe_ffn_loop` host combine (lines ~681-685,
  `row_out.cpu().float()... out.index_add(0, token_of_row, row_out)`) → on-device.
  **Simplification (verified from the region):** `token_of_row[t,k] == t` (line 530,
  `token_ids[:, None].expand(T, K)`), so the `index_add` never accumulates *across*
  tokens — only within a token's own K rows. It is therefore a plain **K-axis
  reduction**, not an indirect scatter:
  ```python
  out = row_out.sum(dim=1)   # [T,K,H] -> [T,H], on device
  ```
  Move this `sum(dim=1)` inside the compiled region (`_compiled_moe_loop_region` returns
  `[T,H]` directly instead of `[T,K,H]`), and drop the host `.cpu()...index_add` and the
  `token_of_row` return entirely. No idx2addr scatter is needed for the combine — the
  idx2addr chain is still needed for the two expert-weight `index_select`s (lines
  521-522), which are unaffected. This fully removes the host round-trip.
- **D2.** `_moe_route_padded` (line ~751): `torch.where(probs >= kth, probs, zeros)` →
  `torch.ops.spyre.index_mask(probs, kth)`. Update the surrounding comment (lines
  ~684-711) to reference `index_mask`. This is shared with the persistent router, so
  the persistent prefill path also picks up `index_mask` — **semantics identical**, but
  it means persistent prefill will ALSO error until `index_mask` lands. (See risk below.)

### E. Module docstring (lines 26-71)

Rewrite the "four selectable formulations" section to describe the two phase-dispatched
methods. Remove the split/chunked/loop-flag prose.

## Implementation order (staged for a green checkpoint)

The `index_mask` swap (D2) is shared by prefill and decode, so applying it early would
block the one verification that works today. Stage the work so prefill stays verifiable
until the last step:

1. **Delete** split + chunked (functions, globals, `__init__` handles including
   `_compiled_route`, `forward` branches, `prepare_for_spyre` branches, mutual-exclusion
   assert). `_moe_route_padded` still on `torch.where`.
2. **Dispatch** — rewrite `forward` to branch on `seq_len` (A).
3. **Decode combine → device** (D1): region returns `[T,H]` via `sum(dim=1)`; drop the
   host `index_add` + `token_of_row`.
4. **Verify prefill** (step 3 below) — must still emit `' the'` at ~7.3 s. Green
   checkpoint. Prefill uses the persistent router, still on `torch.where`, so it traces.
5. **`index_mask` swap** (D2) — the LAST edit. Leaves the tree in the "written ahead"
   state; both prefill and decode routers now depend on the unlanded op, as intended.

## Files to modify

- `hf_adapters/hf_gemma4_moe.py` — all of the above. Adapter-only; Python; no rebuild.

## Verification

1. **Import/parse sanity:** `python -c "import torch, torch_spyre; import hf_adapters.hf_gemma4_moe"`
   — confirms the file parses after deletions and no deleted symbol is still referenced.
2. **Grep for dangling refs:** `grep -n "_moe_ffn_split\|_moe_ffn_chunked\|_MOE_EC\|_MOE_CHUNKED\|_compiled_chunk\|_grouped_gemm\|_moe_permute\|_moe_route\b" hf_gemma4_moe.py`
   → must return only the surviving `_moe_route_padded` / `_moe_route_persistent_packed`.
3. **Prefill re-verification (the real gate):** run the persistent A/B harness prefill
   leg (`repros/gemma4_moe/ab_persistent_vs_chunked.py persistent`, NEW_TOKENS=1,
   PREFILL_TOKENS=512, `HF_HUB_CACHE=/mnt/models/hf_cache/hub HF_HUB_OFFLINE=1`). Must
   still emit `' the'` at ~7.3 s warm. **Blocked by D2** until `index_mask` lands — see
   risk. If `index_mask` is not yet available, temporarily keep the `torch.where` form
   to run this verification, then swap to `index_mask` as the last edit.
4. **Decode dispatch reaches the right region:** trace a `seq_len == 1` step; confirm it
   enters `_moe_ffn_loop` and aborts at the missing `index_mask`/`fp32→int32` op with a
   clear message (expected until ops land) — NOT at a shape/dispatch bug.

## Risk: `index_mask` is shared by prefill and decode

Swapping `torch.where` → `index_mask` in `_moe_route_padded` (D2) affects the persistent
prefill router too, so once applied, prefill also fails to trace until `index_mask`
lands. Handled by the staging above: D2 is the last edit, after prefill is verified
green on `torch.where`. This is the intended end state ("written ahead"), not a defect —
but it means after step 5 there is no traceable path until the ops ship.

## Deferred (separate follow-up, not this change)

- **Whole-layer persistent compile:** fold attention + norms + dense-MLP + router +
  persistent body + KV scatter into one `torch.compile`'d block. Now unblocked (KV
  scatter lowers on device when compiled). Applies to the prefill/persistent leg only;
  decode stays its own region. To be specced separately after this consolidation
  verifies.

## Notes

- Prefill-only A/B discipline (NEW_TOKENS=1, PREFILL_TOKENS=512).
- No GitHub push; keep in working tree.
- Recursive-delete guard: use `mv` not `rm -rf` if clearing `/tmp/torchinductor_aviros`.
