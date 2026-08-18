# KV cache indirect scatter — session handoff (2026-08-12)

Where the work stands, what is verified, what is not, and what to do next.

## Shipped

**Draft PR: [torch-spyre/hf-adapters#330](https://github.com/torch-spyre/hf-adapters/pull/330)**
Branch `kv-cache-indirect-scatter` on `ani300/hf-adapters`, 23 files, 3 commits:

| Commit | Contents |
|---|---|
| `755f590` | design doc |
| `847c12a` | the change itself (67 call sites across 16 adapters + 3 harnesses + profiling script) |
| (latest) | Gemma mask-sync rationale + follow-up recorded |

**Upstream issue: [torch-spyre#3705](https://github.com/torch-spyre/torch-spyre/issues/3705)**
Unpinned indirect scatter silently writes to the wrong rows. Root-caused to
`_value_bufs_for_op` (`enforce_indirect_access_layout.py:195`) iterating
`get_read_writes().reads` only, while a scatter's indirect coordinate lives on the
**write** dep — so the compliance/rotation code at lines 251-299 is unreachable and
nothing rotates the indexed dim. Includes a paired fail/pass repro.

## The blocker — SOLVED (root-caused, two working fixes)

**`index_copy_` was a red herring.** The trigger is an **`aten.index` gather whose
value operand has HIGHER host rank than the gather's output** (negative
`rank_diff`). No cache, no scatter, no SDPA, no model needed to reproduce.

It is the **fill step** only (prefill is fine) — the one forward that passes
`source_index`, so `kv_cache_update` does `k.index_select(2, source_index)` on a
rank-6 RoPE output producing rank-4. Offending node: that `index_select`, layer 0.

**Genuine Inductor bug** at `propagate_layouts.py:1013-1014`, which was written
assuming broadcast (`arg rank ≤ output rank`). Two independent defects:

1. **`rank_diff < 0` unhandled** — the filter admits everything, shifts the wrong
   way, and the arg's extra *leading* dims are never added, so the projection ends
   up with too few entries. **This exact failure is already documented and worked
   around at the call site in `_inductor/decompositions.py:1101`** — a known bug
   never fixed at source.
2. **The trailing `-1` sparse-stick marker is silently dropped** whenever
   `rank_diff >= 0` (since `-1 < rank_diff`). Doesn't crash — silently turns a
   sparse stick dense.

`aten.index` is a 2-arg `Pointwise`, so it routes to
`_multi_arg_pointwise_layouts`; a gather's value operand legitimately outranks its
output. **Not misuse.**

Defect 2 (the dropped `-1`) is a **latent correctness risk, not just a crash**: it
silently turns a sparse stick dense. It has **no test coverage anywhere in
`torch-spyre/tests/`** — worth stating in the issue, since it can be fixed blind
otherwise.

**Independently verified**: `kv_indexcopy_block_repro.py` reproduces with **no
model load** — cases 1, 2, 6 PASS and 3, 4, 5 FAIL, re-run from a clean shell.
Case 3 (rank-6 gather, no cache, no scatter) is the minimal trigger. One layer
suffices; this is not a cross-layer or graph-size effect.

### Two fixes, both verified PASSING on the real qwen3 test

| Fix | Result |
|---|---|
| **Inductor-side**: repair the projection in `_is_supported_layout` | **PASS**, correct output `' Paris. The capital of'` |
| **hf-adapters-side**: gather k/v **before** RoPE (gather `selected_freqs` on dim 1 too) | **PASS** (92.7 s) |
| `.clone()` before `index_select` | FAIL — Inductor fuses through the clone |
| `index_put_` / subscript assign instead of `index_copy_` | FAIL — identical error |
| `.contiguous()` after the gather | FAIL |

The adapter-side workaround is **bit-exact equivalent** (verified on CPU,
`max abs diff 0.0`) because RoPE is per-position, so gather and RoPE commute.

**Recommended**: fix `propagate_layouts.py:1014` to prepend the arg's extra
leading dims when `rank_diff < 0` *and* preserve the trailing `-1`; use the
gather-before-RoPE workaround as the interim hf-adapters unblock. Worth a **second
torch-spyre issue**, distinct from #3705 (that one is layout *compliance* for
stores; this is the *projection* in the pointwise layout solver).

### Original symptom, for reference

```
InductorError: Incompatible host_size and dim_order
  propagate_layouts.py:1017 _is_supported_layout
  via _multi_arg_pointwise_layouts
  output: FixedLayout size=[1, 8, 1, 128] stride=[1024, 128, 1024, 1]
  failing STL: size=[1, 64, 8, 2, 1, 64] dim_order=[3, 5, 2, 4, 1]
```

Six size entries vs five `dim_order` entries. `_is_supported_layout` projects the
output's `dim_order` onto each input with
`[d - rank_diff for d in dim_order if d >= rank_diff]`, which can drop entries and
break the `len(host_size) == len(dim_order)` invariant asserted at
`spyre_tensor_impl.cpp:144`.

**Bisected on the real model, all else equal:**

| Write | Layout pin | Result |
|---|---|---|
| old slice assignment | off | **passes** |
| `index_copy_` | off | **fails** |
| `index_copy_` | on | **fails** |

→ **The layout pin is not the cause.** `index_copy_` in the real block is.

**Ruled out as necessary** (all confirmed on device): the scatter /
`index_copy_` itself, the pinned cache layout, the SDPA consumer, RoPE
specifically, `seq_len == 1`, `k` as a transposed view, and any
multi-layer/graph-size effect. The rank-6→rank-4 gather alone is sufficient —
reproducible with a hand-built `mul → flatten → transpose → gather`, no RoPE and no
cache.

## Verification status

**Verified:**
- `tests/cpu/test_kv_cache_scatter.py` — **13/13 pass** (re-run after the final
  edits). Covers shapes, heterogeneous `_spyre_kv_shapes`, `B=1`/`B=4`, MQA,
  `head_dim` 256, non-contiguous indices, sequential decode writes, in-place
  semantics, equivalence with the old slice write, fill-step equivalence at
  `tokens_in_block ∈ {0, 1, 17, 63}`.
- `tests/cpu/test_adapter_cpu_accuracy.py` — **17 pass**, 1 pre-existing failure
  (`google/gemma-4-12b`, a transformers per-layer-attribute config error that
  reproduces identically on the unmodified baseline — verified by stashing).
- **Full CPU suite, re-run after the final signature fixes — 84 passed, 2 failed,
  2 xfailed, 3 xpassed.** Both failures are `google/gemma-4-12b`
  (`test_adapter_cpu_accuracy` and `test_load_cpu`) and both are **pre-existing**:
  a transformers per-layer-attribute config error that reproduces identically on
  the unmodified baseline. `test_vlm_e2e_cpu.py` excluded — needs `PIL`, not
  installed here.
- `pre-commit run ruff` — clean on all changed files. The 3 remaining errors in
  `scripts/profile_prefill_decode_spyre.py` (E402, E741 ×2) **pre-exist on the
  baseline**.

**NOT verified:**
- Anything on device. The blocker is root-caused with two verified fixes (above),
  but **neither is applied on this branch yet** — the branch as pushed still fails
  the qwen3 smoke test. Apply the gather-before-RoPE workaround, then re-run the
  device suites.
- `pre-commit` mypy hook — crashes with an internal error on a numpy stub
  (`Type statement is only supported in Python 3.12 and greater`), unrelated to
  this change.

## Next steps, in order

1. **Apply the gather-before-RoPE workaround** to `kv_cache_update` / the block
   `source_index` path so the PR passes on device without waiting on torch-spyre.
   Bit-exact, already verified PASS on the real qwen3 test; the working plugin form
   is at `torch-spyre/kvdiag-scratch/wa_gatherfirst.py` (already rescued out of `/tmp`).
2. **File the second torch-spyre issue** for `propagate_layouts.py:1014`
   (`rank_diff < 0` + dropped trailing `-1`), with
   `kv_indexcopy_block_repro.py` as the model-free repro and
   `torch-spyre/kvdiag-scratch/fix_rankdiff.py` as the proposed fix. Note
   `_inductor/decompositions.py:1101` already works around the same defect at a
   different call site — useful corroboration for the report.
3. **Then re-run the device suites** (smoke + token-compare, incl. a
   sliding-window model) and take the PR out of draft.
4. **Follow-ups deliberately deferred** (documented in the design doc): hoist the
   Gemma sliding-window mask out of the forward pass; reduce the decode step from
   64 queries to 1 (now unblocked by this change, and where the remaining
   19.0 ms/layer decode cost lives).

## Reusable scratch (in `/mnt/home/spyre/torch-spyre/`, untracked)

| Script | What it establishes |
|---|---|
| `kv_scatter_issue_repro.py` | #3705's paired fail/pass; exits 0 only if the bug reproduces AND pinned+gather are correct |
| `kv_pinned_matrix.py` | 13-case shape sweep, all pass pinned, `unique_graphs == 1` |
| `kv_inplace_probe.py` | out-of-place `index_copy` **loses the pin** (`[576,8,2,1,64]` → `[8,576,2,1,64]`) |
| `kv_layout_candidates.py` | logical permutation does NOT control device dim order |
| `kv_stl_decouple_probe.py` | the winning STL: logical `[B,n_kv,L,hd]`, device `L` outermost, `B` between sticks and eps |
| `kv_sdpa_layout_probe.py` | SDPA reads the pinned cache relayout-free, incl. fused scatter+attend |
| `kv_batch_permute_probe.py` | `B=4` correctness |
| `kv_scatter_precision.py` | the 1-ULP fp16 delta is the **device round trip**, not the scatter (bf16 exact) |
| `kv_real_block_repro.py`, `kv_stl_rank_probe.py` | early synthetic attempts that PASSED, i.e. rejected hypotheses |
| `kv_indexcopy_block_repro.py` | **the minimal trigger, no model load** — 6 cases; case 3 is rank-6 gather with no cache and no scatter |
| `kv_indexcopy_rankdiff_probe.py` | the A/B/C/D bisection matrix isolating negative `rank_diff` |

**In `/mnt/home/spyre/torch-spyre/kvdiag-scratch/`** (rescued from `/tmp`, untracked):

| Script | What it does |
|---|---|
| `fix_rankdiff.py` | pytest plugin proving the Inductor-side projection fix (real qwen3 PASSES) |
| `wa_gatherfirst.py` | pytest plugin with the passing gather-before-RoPE hf-adapters workaround |
| `stlprobe.py` | pytest plugin dumping the failing STL args + op/arg layouts |

## Gotchas worth not rediscovering

- **`index_copy` with duplicate indices is order-undefined.** This, not
  non-determinism, explains the varying mismatch counts in torch-spyre#3671. Use
  `randperm`, not `randint`.
- **fp16 on Spyre is not IEEE fp16.** A bare `.to("spyre").cpu()` with no scatter
  perturbs ~50% of elements by 1 ULP. Compare with tolerance or use
  exactly-representable values. bf16 is bit-exact.
- **A lost layout pin is undetectable on CPU.** `assert_kv_cache_layout` is the only
  automated guard; it uses `tensor.device_tensor_layout()` (there is no
  `device_layout` attribute — an earlier version of the guard silently no-op'd).
- **`device_layout=` needs a live RuntimeContext** — touch the device first or it
  raises `RAS::RUNTIMECONTEXT::ContextNotCreated`.
- **The Spyre device is exclusive to one process** and takes ~30s to release.
  `DeviceOpenFail` is contention, not a test result — retry.
- **Mechanical sweeps need three checks to converge.** AST scan, tests, and ruff
  each caught a distinct class of miss: single-line call forms, semantic reuse of
  the removed args (the Gemma mask coordinates), and signatures ending in extra
  params. Ruff's `F821` found the last batch, including one that would have passed
  `layer_scalar` into the wrong parameter — a silent wrong-value bug, not a crash.
