# Gemma-4 MoE Phase-Dispatch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Consolidate the Gemma-4 MoE adapter's four device FFN formulations to two, dispatched by `seq_len` (persistent for prefill, loop-on-topk for decode), and move the decode combine on-device.

**Architecture:** `Gemma4MoEBlock.forward` picks the FFN method at runtime by sequence length instead of by an import-time module global. The persistent dense-all-experts path (verified working) serves prefill (`seq_len > 1`); the loop-on-topk gather path serves decode (`seq_len == 1`). The split and chunked formulations and their helpers/globals/handles are deleted. The decode path's host `index_add` combine collapses to an on-device K-axis reduction. The `spyre::index_mask` router swap is staged last so a green checkpoint exists before the tree depends on unlanded ops.

**Tech Stack:** Python; PyTorch; torch-spyre `_inductor` OOT backend; HuggingFace Transformers monkey-patch adapter. No C++/rebuild (Python-only). No new dependencies.

**Spec:** `docs/superpowers/specs/2026-08-20-gemma4-moe-phase-dispatch-design.md`

## Global Constraints

- **Adapter-only.** Every change is in `hf_adapters/hf_gemma4_moe.py`. No torch-spyre edit. Python-only — no rebuild.
- **All-device intent.** Nothing gather/scatter/FFN runs on host. The decode combine must end on-device.
- **Keep in working tree.** Do NOT commit the source changes until decode is exercisable end-to-end (i.e. after the ops land). The design spec is already committed (`20b6cd4`). Task commits below are LOCAL checkpoint commits on branch `gemma4-moe-persistent-moe-ffn`; do not push. Sign with `git commit -s`.
- **No GitHub push** without explicit approval. Force-push is blocked by the harness.
- **`import torch` before any `import torch_spyre`** (circular-import guard).
- **Prefill-only A/B discipline:** NEW_TOKENS=1, PREFILL_TOKENS=512.
- **Spyre test env:** `HF_HUB_CACHE=/mnt/models/hf_cache/hub HF_HUB_OFFLINE=1`; model `google/gemma-4-26B-A4B-it`; host `aviros-spyre-test` (no pod).
- **Recursive-delete guard:** never `rm -rf`; use `mv /tmp/torchinductor_aviros /tmp/torchinductor_aviros.old.$$` to clear the inductor cache.
- **`index_mask` staging:** the `torch.where` → `torch.ops.spyre.index_mask` swap (Task 6) is the LAST edit. Until then `_moe_route_padded` stays on `torch.where` so the persistent prefill path traces and can be verified.

---

## File Structure

Only one source file changes: `hf_adapters/hf_gemma4_moe.py`. No new files. The tasks are ordered so the file stays importable after each one and prefill stays verifiable until the final op-dependent swap.

Task order (matches the spec's staging):
1. Delete the split formulation (functions + handles + branches).
2. Delete the chunked formulation (functions + globals + branches).
3. Collapse the surviving mode globals; rewrite the mutual-exclusion assert region.
4. Rewrite `forward` to dispatch on `seq_len`; make `prepare_for_spyre` materialize BOTH weight sets unconditionally.
5. Move the decode combine on-device (host `index_add` → `sum(dim=1)`).
6. Swap `torch.where` → `index_mask` (LAST; leaves tree in "written ahead" state).

There is no unit-test harness for this file (it requires the Spyre card); verification is by import-parse, dangling-ref grep, and the on-card prefill A/B. Each task therefore ends with an import + grep check and a checkpoint commit, and Task 4 ends with the on-card prefill run.

---

### Task 1: Delete the split formulation

**Files:**
- Modify: `hf_adapters/hf_gemma4_moe.py`

**Interfaces:**
- Consumes: nothing (pure deletion).
- Produces: removes symbols `_moe_route`, `_moe_permute`, `_grouped_gemm_4a`, `_group_offsets`, `_grouped_gemm_4b`, `_grouped_gemm`, `_moe_ffn`, `_moe_ffn_loop_ref`, `_compiled_moe_device_region`, `_compiled_device_gather`, `_moe_ffn_split`; removes `__init__` handles `_compiled_gather`, `_compiled_expert`; removes the `else:` split branch in `forward` and the split `else:` branch in `prepare_for_spyre`.

- [ ] **Step 1: Delete the split helper + FFN functions**

Delete these function definitions in full (current line ranges; verify by name before cutting):
- `_moe_route` (181), `_moe_permute` (203), `_grouped_gemm_4a` (228), `_group_offsets` (246), `_grouped_gemm_4b` (268), `_grouped_gemm` (317), `_moe_ffn` (337), `_moe_ffn_loop_ref` (374), `_compiled_moe_device_region` (426), `_compiled_device_gather` (534), `_moe_ffn_split` (545).

Do NOT delete `_compiled_moe_loop_region` (447) or `_moe_ffn_loop` (641) — those are the decode path, kept.

- [ ] **Step 2: Delete the split `__init__` compile handles**

In `Gemma4MoEBlock.__init__`, delete these two handle assignments (currently lines 1070-1075):

```python
        self._compiled_gather = torch.compile(
            _compiled_device_gather, dynamic=False
        )
        self._compiled_expert = torch.compile(
            _compiled_moe_device_region, dynamic=False
        )
```

- [ ] **Step 3: Delete the split `forward` branch**

In `Gemma4MoEBlock.forward`, delete the `else:` split branch (currently lines 1170-1180):

```python
        else:
            moe_out = _moe_ffn_split(
                flat,
                x_moe,
                self.router,
                self._compiled_gather,
                self._compiled_expert,
                self._spyre_gate_up_t,
                self._spyre_down_t,
                self._moe_k,
            )  # [T,H]
```

(The `if/elif/elif` chain remains for now; Task 4 replaces the whole chain. Leaving no `else` here is fine temporarily because Task 4 immediately rewrites this block.)

- [ ] **Step 4: Delete the split `prepare_for_spyre` branch**

In `prepare_for_spyre`, delete the `else:` split weight branch (currently lines 1420-1422):

```python
        else:
            block._spyre_gate_up_t = gate_up_t.cpu()  # [E,H,2M] host-resident
            block._spyre_down_t = down_t.cpu()  # [E,M,H] host-resident
```

- [ ] **Step 5: Verify import + no dangling split refs**

Run:
```bash
cd /mnt/devel/inductor_src/hf-adapters
python -c "import torch, torch_spyre; import hf_adapters.hf_gemma4_moe" && echo IMPORT_OK
grep -n "_moe_ffn_split\|_moe_permute\|_grouped_gemm\|_group_offsets\|_moe_ffn_loop_ref\|_compiled_device_gather\|_compiled_moe_device_region\|_compiled_gather\|_compiled_expert\|_spyre_gate_up_t\|_spyre_down_t\|def _moe_route\b\|_moe_route(" hf_adapters/hf_gemma4_moe.py
```
Expected: `IMPORT_OK`. The grep must return NOTHING (all split symbols gone). Note: `_moe_route_padded` and `_moe_route_persistent_packed` are DIFFERENT names — the `def _moe_route\b` / `_moe_route(` patterns are anchored so they will not match those; if any line prints, a split ref survived — fix it.

- [ ] **Step 6: Commit**

```bash
git add hf_adapters/hf_gemma4_moe.py
git commit -s -m "refactor(gemma4-moe): delete split FFN formulation

Remove the host/device-split MoE path (_moe_ffn_split and its grouped-GEMM
helpers, device-gather/expert-region compiles, and the host-resident expert
stacks). Superseded by the persistent (prefill) and loop-on-topk (decode)
methods. Working-tree checkpoint; not pushed.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Delete the chunked formulation

**Files:**
- Modify: `hf_adapters/hf_gemma4_moe.py`

**Interfaces:**
- Consumes: nothing (pure deletion).
- Produces: removes symbols `_moe_expert_chunk`, `_moe_ffn_chunked`; removes `__init__` handle `_compiled_chunk`; removes the `_MOE_CHUNKED_ONDEVICE` branch in `forward` and in `prepare_for_spyre`; removes globals `_MOE_GEMM_4B`, `_MOE_EC` (chunked-only). Keeps `_MOE_PADW`/`_MOE_PAD_NEG` (used by the surviving router).

- [ ] **Step 1: Delete the chunked FFN functions**

Delete these function definitions in full:
- `_moe_expert_chunk` (756), `_moe_ffn_chunked` (951).

- [ ] **Step 2: Delete the chunked `__init__` compile handles**

In `Gemma4MoEBlock.__init__`, delete these two handle assignments (currently lines 1079-1080):

```python
        self._compiled_route = torch.compile(_moe_route_padded, dynamic=False)
        self._compiled_chunk = torch.compile(_moe_expert_chunk, dynamic=False)
```

`_compiled_route` (standalone `torch.compile(_moe_route_padded)`) is grep-confirmed used ONLY by the chunked branch — the persistent router uses `_compiled_persistent_route` and `_moe_ffn_loop` runs its router inside `_compiled_loop`. The `_moe_route_padded` *function* survives.

- [ ] **Step 3: Delete the chunked `forward` branch**

In `Gemma4MoEBlock.forward`, delete the `elif _MOE_CHUNKED_ONDEVICE:` branch (currently lines 1143-1157):

```python
        elif _MOE_CHUNKED_ONDEVICE:
            # ALL-DEVICE: router + expert GEMMs + weight + sum-over-experts all
            # lower; host does only the chunk-loop glue (inside _moe_ffn_chunked)
            # threading a device-resident accumulator. Per-chunk offset-0 expert
            # weights were pre-materialized in prepare_for_spyre.
            moe_out = _moe_ffn_chunked(
                flat,
                x_moe,
                self.router,
                self._compiled_route,
                self._compiled_chunk,
                self._spyre_moe_chunks,
                self._moe_k,
                self._moe_rms_eps,
            )  # [T,H]
```

- [ ] **Step 4: Delete the chunked `prepare_for_spyre` branch**

In `prepare_for_spyre`, delete the `elif _MOE_CHUNKED_ONDEVICE:` weight branch (currently lines 1349-1392 — the de-fuse/chunk/one-hot block ending at the router-weight moves). Delete from `elif _MOE_CHUNKED_ONDEVICE:` through the closing `router.per_expert_scale = torch.nn.Parameter(...)` of that branch (line 1392), inclusive.

- [ ] **Step 5: Delete the chunked-only globals**

Delete the `_MOE_GEMM_4B` global + its comment block (currently lines 100-117) and the `_MOE_EC` global + its comment (currently lines 168-171). Do NOT delete `_MOE_PADW` (177) or `_MOE_PAD_NEG` (178) — the surviving `_moe_route_padded` still pads before topk.

Also delete the chunked-specific asserts in `prepare_for_spyre` (currently lines 1262-1266):

```python
    if _MOE_CHUNKED_ONDEVICE:
        assert E % _MOE_EC == 0, (
            f"all-device chunked MoE needs num_experts ({E}) divisible by "
            f"_MOE_EC ({_MOE_EC})."
        )
```

Leave the `_MOE_PADW` assert (1256-1261) for now — Task 3 adjusts its guard condition.

- [ ] **Step 6: Verify import + no dangling chunked refs**

Run:
```bash
cd /mnt/devel/inductor_src/hf-adapters
python -c "import torch, torch_spyre; import hf_adapters.hf_gemma4_moe" && echo IMPORT_OK
grep -n "_moe_ffn_chunked\|_moe_expert_chunk\|_compiled_chunk\|_compiled_route\b\|_spyre_moe_chunks\|_MOE_GEMM_4B\|_MOE_EC" hf_adapters/hf_gemma4_moe.py
```
Expected: `IMPORT_OK`; grep returns NOTHING.

- [ ] **Step 7: Commit**

```bash
git add hf_adapters/hf_gemma4_moe.py
git commit -s -m "refactor(gemma4-moe): delete chunked FFN formulation

Remove the all-device chunked mask-reduce MoE path (_moe_ffn_chunked,
_moe_expert_chunk, the standalone router compile, and the per-chunk offset-0
expert-weight materialization) plus its _MOE_GEMM_4B/_MOE_EC globals.
Working-tree checkpoint; not pushed.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Collapse the surviving mode globals + mutual-exclusion assert

**Files:**
- Modify: `hf_adapters/hf_gemma4_moe.py`

**Interfaces:**
- Consumes: nothing.
- Produces: removes globals `_MOE_LOOP_ON_TOPK`, `_MOE_CHUNKED_ONDEVICE`, `_MOE_PERSISTENT_ONDEVICE` (dispatch is now by `seq_len`, not by flag); keeps `_MOE_TILE`, `_MOE_PADW`, `_MOE_PAD_NEG`, `_MOE_MAX_K`; replaces the mutual-exclusion assert with a `_MOE_PADW`-only assert. This task leaves `forward` still referencing the deleted flags — **Task 4 rewrites `forward` in the same session; run Task 3 and Task 4 back-to-back.** The import check at the end of Task 3 only parses the module (flag refs inside `forward` are not evaluated at import), so it passes; the flags are fully gone after Task 4.

- [ ] **Step 1: Delete the mode-select globals**

Delete these globals and their comment blocks:
- `_MOE_LOOP_ON_TOPK` (130) + comment (124-130).
- `_MOE_CHUNKED_ONDEVICE` (161) + comment (132-161).
- `_MOE_PERSISTENT_ONDEVICE` (166) + comment (163-166).

Keep `_MOE_TILE` (122) — used by the decode `_moe_ffn_loop` call. Keep `_MOE_MAX_K` (98), `_MOE_PADW` (177), `_MOE_PAD_NEG` (178).

- [ ] **Step 2: Replace the mutual-exclusion assert block**

In `prepare_for_spyre`, replace the mutual-exclusion assert + the `_MOE_PADW` guard (currently lines 1242-1261) with just the `_MOE_PADW` guard, unconditional (both surviving methods' router pads before topk):

```python
    # The router pads its logits to a non-pow2 width before topk (topk-pad fix,
    # shared by the prefill persistent router and the decode loop router).
    E = cfg.num_experts
    assert _MOE_PADW > E and (_MOE_PADW & (_MOE_PADW - 1)) != 0, (
        f"_MOE_PADW ({_MOE_PADW}) must exceed num_experts ({E}) and be "
        "non-power-of-two (topk-pad fix)."
    )
```

Remove the `enabled_modes = sum(...)` block, the `assert enabled_modes <= 1` block, and the `if _MOE_PERSISTENT_ONDEVICE or _MOE_CHUNKED_ONDEVICE:` guard that wrapped the old `_MOE_PADW` assert.

- [ ] **Step 3: Verify module still parses**

Run:
```bash
cd /mnt/devel/inductor_src/hf-adapters
python -c "import torch, torch_spyre; import hf_adapters.hf_gemma4_moe" && echo IMPORT_OK
grep -n "_MOE_PERSISTENT_ONDEVICE\|_MOE_CHUNKED_ONDEVICE\|_MOE_LOOP_ON_TOPK\|enabled_modes" hf_adapters/hf_gemma4_moe.py
```
Expected: `IMPORT_OK` (module-level parse succeeds; the deleted flags are only referenced inside `forward`/`prepare_for_spyre` bodies which Task 4 fixes). Grep may still show the flag names INSIDE `forward` (lines ~1129, ~1158) — that is expected and removed in Task 4. It must NOT show them at module scope (no bare `_MOE_..._ONDEVICE = ...` line) or in `prepare_for_spyre` (`enabled_modes` gone).

- [ ] **Step 4: Do NOT commit yet**

`forward` still references deleted flags; committing here would leave a broken forward. Proceed directly to Task 4 and commit at Task 4's end.

---

### Task 4: seq_len dispatch + unconditional weight materialization

**Files:**
- Modify: `hf_adapters/hf_gemma4_moe.py`

**Interfaces:**
- Consumes: `_moe_ffn_persistent(x_router, x_expert, router, compiled_route, compiled_persistent, gate, up, down, route_identity, K, eps)` and `_moe_ffn_loop(x_router, x_expert, router, compiled_loop, gate_up_dev, down_dev, K, tile, eps)` — both unchanged from their current definitions.
- Produces: `Gemma4MoEBlock.forward` dispatches by `seq_len`; `prepare_for_spyre` materializes BOTH the persistent weight set (`_spyre_persistent_gate/up/down`, `_spyre_persistent_route_identity`) AND the loop weight set (`_spyre_gate_up_dev`, `_spyre_down_dev`) on every block, with the router weights moved to device once.

- [ ] **Step 1: Rewrite the `forward` FFN dispatch**

`seq_len` is already bound at the top of the FFN section (`bsz, seq_len, hidden = h.shape`, current line 1113). Replace the entire `if _MOE_PERSISTENT_ONDEVICE: ... elif _MOE_LOOP_ON_TOPK: ...` chain (currently lines 1129-1169, after Tasks 1-2 removed the chunked/split arms) with a `seq_len` branch:

```python
        # Phase dispatch: prefill (seq_len > 1) runs the dense all-experts
        # persistent path; decode (seq_len == 1) runs the loop-on-topk gather
        # path. Both compiled regions and both expert-weight sets are built once
        # (prepare_for_spyre / __init__); the choice is per-forward by shape, not
        # an import-time flag.
        if seq_len > 1:
            moe_out = _moe_ffn_persistent(
                flat,
                x_moe,
                self.router,
                self._compiled_persistent_route,
                self._compiled_persistent,
                self._spyre_persistent_gate,
                self._spyre_persistent_up,
                self._spyre_persistent_down,
                self._spyre_persistent_route_identity,
                self._moe_k,
                self._moe_rms_eps,
            )  # [T,H]
        else:
            moe_out = _moe_ffn_loop(
                flat,
                x_moe,
                self.router,
                self._compiled_loop,
                self._spyre_gate_up_dev,
                self._spyre_down_dev,
                self._moe_k,
                _MOE_TILE,
                self._moe_rms_eps,
            )  # [T,H]
```

- [ ] **Step 2: Make `prepare_for_spyre` materialize BOTH weight sets unconditionally**

Replace the `if _MOE_PERSISTENT_ONDEVICE: ... elif _MOE_LOOP_ON_TOPK: ...` weight chain (currently lines 1325-1419, after Tasks 1-2 removed the chunked/split arms) with an unconditional block that builds both sets and moves the router once:

```python
        # Both FFN methods coexist (dispatch is per-forward by seq_len), so
        # materialize BOTH expert-weight layouts on every block.
        #
        # Persistent (prefill): K-major [K,E,N] contiguous backing exposed as
        # logical [E,K,N] expert-major views for the hinted matmul body.
        M = gate_up_t.shape[2] // 2
        block._spyre_persistent_gate = _pack_persistent_expert_weight(
            gate_up_t[:, :, :M]
        )
        block._spyre_persistent_up = _pack_persistent_expert_weight(
            gate_up_t[:, :, M:]
        )
        block._spyre_persistent_down = _pack_persistent_expert_weight(down_t)
        block._spyre_persistent_route_identity = torch.eye(
            64, dtype=gate_up_t.dtype
        ).to("spyre")

        # Loop-on-topk (decode): whole stacks HBM-resident, E outermost
        # ([E,H,2M]/[E,M,H] — indexed dim at device position 0 for the on-device
        # index_select, stick-correct for the bmm weight operand).
        block._spyre_gate_up_dev = gate_up_t.to("spyre")  # [E,H,2M]
        block._spyre_down_dev = down_t.to("spyre")  # [E,M,H]

        # The router runs on-device in BOTH paths (scale-free norm + [H] scale +
        # proj + padded topk + per_expert_scale), so its weights are device-
        # resident. Reassign the Parameter object (a cross-backend param.data set
        # raises on the type change).
        router.proj.weight = torch.nn.Parameter(
            router.proj.weight.data.to("spyre"), requires_grad=False
        )
        router.scale = torch.nn.Parameter(
            router.scale.data.to("spyre"), requires_grad=False
        )
        router.per_expert_scale = torch.nn.Parameter(
            router.per_expert_scale.data.to("spyre"), requires_grad=False
        )
```

- [ ] **Step 3: Verify import + no flag refs anywhere**

Run:
```bash
cd /mnt/devel/inductor_src/hf-adapters
python -c "import torch, torch_spyre; import hf_adapters.hf_gemma4_moe" && echo IMPORT_OK
grep -n "_MOE_PERSISTENT_ONDEVICE\|_MOE_CHUNKED_ONDEVICE\|_MOE_LOOP_ON_TOPK\|enabled_modes" hf_adapters/hf_gemma4_moe.py
```
Expected: `IMPORT_OK`; grep returns NOTHING (flags fully gone from `forward` and `prepare_for_spyre`).

- [ ] **Step 4: On-card prefill re-verification (the real gate)**

The router is still on `torch.where` (Task 6 not yet applied), so the persistent prefill path traces. Clear the inductor cache (guard-safe `mv`, never `rm -rf`) and run the persistent A/B prefill leg:

```bash
cd /mnt/devel/inductor_src/hf-adapters
mv /tmp/torchinductor_aviros /tmp/torchinductor_aviros.old.$$ 2>/dev/null || true
HF_HUB_CACHE=/mnt/models/hf_cache/hub HF_HUB_OFFLINE=1 \
  python -u repros/gemma4_moe/ab_persistent_vs_chunked.py persistent 2>&1 | tee /tmp/task4_prefill.log
```
Expected: compiles and runs end-to-end; emits `' the'`; warm generation ≈ 7.3 s (the pre-refactor persistent number). If it aborts or the token changes, the dispatch/materialization refactor regressed — diagnose before committing. `_untracked_*` named-dim warnings are expected and do NOT abort.

Note: the harness's `chunked` leg no longer exists (deleted). Only run the `persistent` argv. If the harness hard-requires both legs, run just the persistent path via its underlying entry (the harness sets the mode by module global, which is gone — so invoke the persistent path directly; if the harness cannot select without the global, note it and run a minimal prefill through `generate` with PREFILL_TOKENS=512, NEW_TOKENS=1 instead). Report the token and warm time either way.

- [ ] **Step 5: Commit (green checkpoint)**

```bash
git add hf_adapters/hf_gemma4_moe.py
git commit -s -m "refactor(gemma4-moe): dispatch FFN by seq_len; build both weight sets

forward picks the persistent path for prefill (seq_len>1) and loop-on-topk
for decode (seq_len==1) at runtime instead of by import-time flag. Delete the
three mode-select globals. prepare_for_spyre materializes both the persistent
K-major backing and the loop-on-topk HBM-resident stacks on every block, and
moves the router weights to device once. Persistent prefill re-verified on
card: emits ' the' at ~7.3s warm. Working-tree checkpoint; not pushed.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: Move the decode combine on-device (K-axis reduction)

**Files:**
- Modify: `hf_adapters/hf_gemma4_moe.py`

**Interfaces:**
- Consumes: `_compiled_moe_loop_region` currently returns `(row_out [T,K,H], token_of_row [T,K])`.
- Produces: `_compiled_moe_loop_region` returns `out [T,H]` (K-axis reduced on device); `_moe_ffn_loop` no longer takes `token_ids`, no `.cpu()`, no `index_add`. The region signature drops `token_ids` and its `token_of_row` return.

Rationale (from the spec): `token_of_row[t,k] == t`, so the host `index_add(0, token_of_row, row_out)` never accumulates across tokens — it only sums a token's own K rows. That is exactly `row_out.sum(dim=1)`, which lowers on device. No indirect scatter is needed for the combine. The idx2addr chain (region lines 510-512) is still needed for the two expert-weight `index_select`s — leave it.

- [ ] **Step 1: Reduce over K inside the compiled region**

In `_compiled_moe_loop_region`, replace the trailing `token_of_row` construction and return (currently lines 528-531):

```python
        row_out = row_out * w[..., None]  # [T,K,1] broadcast
    # token_of_row[t,k] = t (each of a token's K rows scatters back to token t);
    # kept in [T,K] here, flattened host-side for the eager index_add combine.
    token_of_row = token_ids[:, None].expand(T, K)  # [T,K]
    return row_out, token_of_row
```

with an on-device K-axis reduction (each token's K expert outputs are already
weighted; `token_of_row` would be `t` broadcast over K, so the scatter-combine
is a plain sum over K):

```python
        row_out = row_out * w[..., None]  # [T,K,1] broadcast
        out = row_out.sum(dim=1)  # [T,K,H] -> [T,H] on-device K-axis combine
    return out
```

Keep `out = row_out.sum(dim=1)` INSIDE the `with spyre_hint(...)` block (same
indentation as `row_out = row_out * w[..., None]`) so the reduction lowers under
the row-tile hint alongside the matmuls.

- [ ] **Step 2: Drop the now-unused `token_ids` region input**

In `_compiled_moe_loop_region`'s signature (currently lines 447-460), remove the `token_ids,` parameter. It was only used to build `token_of_row`, now gone. Update the region docstring's final line (currently line 482, "Returns (row_out[T,K,H], token_of_row[T,K])...") to "Returns the combined MoE output [T,H] on x_expert's device (K-axis reduced under the row-tile hint)."

- [ ] **Step 3: Simplify `_moe_ffn_loop` — no host round-trip**

In `_moe_ffn_loop`, remove the `token_ids` construction (currently line 659) and the host combine (currently lines 677-685). Replace the region call + host combine so the region returns the final `[T,H]` directly:

```python
def _moe_ffn_loop(x_router, x_expert, router, compiled_loop, gate_up_dev,
                  down_dev, K, tile, eps):
    """Loop-on-topk MoE FFN orchestrator (decode path).

    Unpacks the router's tensors, calls the compiled loop region (router +
    gather + on-device expert-weight index_select + bmms + K-axis combine, all
    row-tiled on device), and returns its [T,H] output. Nothing runs on host:
    the K-axis reduction that combines each token's K expert outputs lowers on
    device (token_of_row is the identity t-broadcast, so index_add degenerates
    to sum-over-K).

    Router surface (preflight-corrected): router.norm has NO .weight; the
    region applies a scale-free RMSNorm (eps INSIDE sqrt) then the [H]
    router.scale vector and the router.scalar_root_size float. Pass
    eps=config.rms_norm_eps.

    Returns the combined [T,H] MoE output on x_expert's device.
    """
    return compiled_loop(
        x_router,
        x_expert,
        router.proj.weight,
        router.scale,
        router.scalar_root_size,
        router.per_expert_scale,
        gate_up_dev,
        down_dev,
        K,
        tile,
        eps,
    )
```

The `token_ids` argument is removed from the call (it was the 9th positional arg). Confirm the remaining positional order matches the region signature after Step 2 removed `token_ids`.

- [ ] **Step 4: Verify import + no host-combine refs in the loop path**

Run:
```bash
cd /mnt/devel/inductor_src/hf-adapters
python -c "import torch, torch_spyre; import hf_adapters.hf_gemma4_moe" && echo IMPORT_OK
grep -n "token_of_row\|token_ids\|index_add" hf_adapters/hf_gemma4_moe.py
```
Expected: `IMPORT_OK`; grep returns NOTHING (the decode path's host scatter is fully gone). If `index_add` still appears, a reference survived — fix it.

- [ ] **Step 5: Commit**

```bash
git add hf_adapters/hf_gemma4_moe.py
git commit -s -m "refactor(gemma4-moe): decode combine on-device via K-axis sum

The loop-on-topk host index_add combine (row_out.cpu()...index_add) is a plain
K-axis reduction in disguise: token_of_row[t,k]==t, so it never accumulates
across tokens. Replace it with row_out.sum(dim=1) inside the compiled region
under the row-tile hint, dropping the token_ids input, the token_of_row return,
and the host round-trip. Nothing gather/scatter/FFN runs on host now.
Working-tree checkpoint; not pushed.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 6: Swap `torch.where` → `spyre::index_mask` (LAST — written ahead of the op)

**Files:**
- Modify: `hf_adapters/hf_gemma4_moe.py`

**Interfaces:**
- Consumes: `torch.ops.spyre.index_mask(probs, kth)` — NOT yet landed in `pr3892-moe`. This is written ahead; tracing any router path (prefill OR decode) errors at this call until torch-spyre ships the op. Expected, per the spec ("written ahead").
- Produces: `_moe_route_padded` uses `index_mask` for the threshold mask.

- [ ] **Step 1: Swap the threshold mask**

In `_moe_route_padded`, replace the `torch.where` threshold (currently lines 747-751):

```python
    # Keep probs where probs >= kth (the selected top-K experts), zero the rest,
    # with the [T,1] kth broadcast over the [T,E] probs. NOTE: this is the
    # threshold form of the planned torch.ops.spyre.index_mask(probs, kth); until
    # that op lands in torch-spyre, use the equivalent torch.where directly.
    mask = torch.where(probs >= kth, probs, torch.zeros_like(probs))  # [T,E]
```

with:

```python
    # Threshold mask: keep probs where probs >= kth (the selected top-K experts),
    # zero the rest. torch.ops.spyre.index_mask(probs, kth) is the device op for
    # this (kth [T,1] broadcast over the [T,E] probs); it is the drop-in for the
    # earlier torch.where form and lowers on-device.
    mask = torch.ops.spyre.index_mask(probs, kth)  # [T,E]
```

- [ ] **Step 2: Update the router docstring**

In `_moe_route_padded`'s docstring, the `index_mask` reference (currently lines 710-712) already describes `torch.ops.spyre.index_mask(probs, kth)` as the mechanism — confirm it reads correctly (it was written for this op). No change needed if it already says `index_mask`; if it still hedges "until that op lands, use torch.where", trim that clause.

- [ ] **Step 3: Verify import (op resolves at call time, not import)**

Run:
```bash
cd /mnt/devel/inductor_src/hf-adapters
python -c "import torch, torch_spyre; import hf_adapters.hf_gemma4_moe" && echo IMPORT_OK
grep -n "torch.where\|index_mask" hf_adapters/hf_gemma4_moe.py
```
Expected: `IMPORT_OK` (import does not resolve `torch.ops.spyre.index_mask`; that happens at trace time). Grep shows `index_mask` in `_moe_route_padded`, no `torch.where` in the router.

- [ ] **Step 4: Confirm the expected trace-time error (once, for the record)**

Attempt a prefill trace to confirm the ONLY failure is the missing op (not a shape/dispatch bug):

```bash
cd /mnt/devel/inductor_src/hf-adapters
mv /tmp/torchinductor_aviros /tmp/torchinductor_aviros.old.$$ 2>/dev/null || true
HF_HUB_CACHE=/mnt/models/hf_cache/hub HF_HUB_OFFLINE=1 \
  python -u repros/gemma4_moe/ab_persistent_vs_chunked.py persistent 2>&1 | tail -40 | tee /tmp/task6_trace.log
```
Expected: fails at `torch.ops.spyre.index_mask` (AttributeError / no-such-op), NOT at a shape or layout error. This confirms the "written ahead" state is clean. If it fails anywhere else, that is a real bug — diagnose. Once `index_mask` lands in torch-spyre, this same run should pass end-to-end.

- [ ] **Step 5: Commit**

```bash
git add hf_adapters/hf_gemma4_moe.py
git commit -s -m "feat(gemma4-moe): route via spyre::index_mask (written ahead of op)

Swap the router's torch.where threshold for torch.ops.spyre.index_mask(probs,
kth). The op is not yet in torch-spyre pr3892-moe, so tracing any router path
errors at this call until it lands -- intended 'written ahead' state. Shared
by the prefill persistent router and the decode loop router (identical
semantics). Working-tree checkpoint; not pushed.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:**
- 3→2 consolidation (delete split + chunked): Tasks 1, 2. ✓
- seq_len dispatch in `forward`: Task 4 Step 1. ✓
- Both weight sets materialized unconditionally: Task 4 Step 2. ✓
- Decode combine on-device: Task 5. ✓
- `index_mask` swap, staged last: Task 6. ✓
- Globals collapsed: Tasks 2 (chunked-only), 3 (mode-select). ✓
- Compile handles pruned: Tasks 1, 2. ✓
- `prepare_for_spyre` branches pruned: Tasks 1, 2, 3, 4. ✓
- KV `index_copy_` — spec says NO change (already on-device when compiled); no task, correctly. ✓
- Whole-layer compile — spec defers to a follow-up; no task, correctly. ✓
- Green-checkpoint staging (verify prefill before `index_mask`): Task 4 Step 4 verifies; Task 6 is after. ✓
- Prefill-only A/B (NEW_TOKENS=1, PREFILL_TOKENS=512): Task 4 Step 4, Global Constraints. ✓
- Keep in working tree (local commits only, no push): every commit says "not pushed"; Global Constraints. ✓

**Placeholder scan:** No TBD/TODO. Every code step shows the exact old block and the exact replacement. The one soft spot — Task 4 Step 4's "if the harness cannot select without the global" — is a real contingency with a concrete fallback (run prefill through `generate` directly), not a placeholder.

**Type consistency:**
- `_moe_ffn_loop` signature after Task 5: `(x_router, x_expert, router, compiled_loop, gate_up_dev, down_dev, K, tile, eps)` — matches the Task 4 Step 1 call (which passes `flat, x_moe, self.router, self._compiled_loop, self._spyre_gate_up_dev, self._spyre_down_dev, self._moe_k, _MOE_TILE, self._moe_rms_eps` — 9 args, order matches). ✓
- `_compiled_moe_loop_region` returns `[T,H]` after Task 5; `_moe_ffn_loop` returns it straight through; `forward` assigns to `moe_out` then `.reshape(bsz, seq_len, hidden)` (current line 1181) — `[T,H]` reshapes to `[B,S,H]` correctly (T == B*S). ✓
- `_moe_ffn_persistent` call in Task 4 Step 1 matches its current signature (11 args) exactly. ✓
- `_spyre_persistent_gate/up/down`, `_spyre_persistent_route_identity`, `_spyre_gate_up_dev`, `_spyre_down_dev` — produced in Task 4 Step 2, consumed in Task 4 Step 1. Names identical. ✓
- `_MOE_TILE` kept (Tasks 2, 3), used in Task 4 Step 1. ✓
- `_MOE_PADW`/`_MOE_PAD_NEG` kept, used by `_moe_route_padded` (unchanged) and its Task 3 assert. ✓

No gaps found.
