# Blocker 6 handoff — root cause confirmed in unit-BMM marking

**Status:** ROOT CAUSE CONFIRMED; minimal torch-spyre fix written, device
verification still pending.

**Last updated:** 2026-08-12.

> **Start here tomorrow.** The closeout section below is authoritative; later
> sections are retained as investigation history and any statement there that
> says the root cause is unknown is superseded. No compile/test process is
> running. The production fix has passed its focused CPU regression but has not
> yet been exercised on device.

Working directories:

- torch-spyre fix/tests: `/mnt/devel/inductor_src/torch-spyre`
- device repros/handoff: `/mnt/devel/inductor_src/hf-adapters/.claude/worktrees/hier-compile`

First command tomorrow:

```bash
cd /mnt/devel/inductor_src/torch-spyre
python -m pytest \
  tests/inductor/test_coarse_tiling.py::TestSharedWeightUnitBmmLayout -q
```

Then preserve the current diagnostic cache and run the production fix without
the diagnostic toggle:

```bash
mv /tmp/torchinductor_aviros \
  /tmp/torchinductor_aviros.unit_bmm_marker_off.20260812_1
cd /mnt/devel/inductor_src/hf-adapters/.claude/worktrees/hier-compile
PYTHONPATH="$PWD" HF_HOME=/mnt/models/hf_cache \
  python repros/hier/standalone_single_projection_cache_discriminator.py
```

Expected: K and K-cache deltas are both `0.000000`, and the generated shared
wrapper contains neither `_spyre_bmm_unit` nor `shared_weight_unit_bmm`.

### 2026-08-12 closeout: root cause confirmed

Blocker 6 is caused by **incorrect shared-weight unit-BMM marking** in
`torch_spyre/_inductor/temp_passes.py`, not LX/scratchpad corruption.

The final device bisection was:

| Shared `_1` vs inline `_0` body | max delta |
|---|---:|
| Projections + RoPE, no mutation: Q/K/V | `0.000000` each |
| Projections + cache writes, no RoPE: Q | `98.453125` |
| Projections + cache writes, no RoPE: K cache | `131.937500` |
| Projections + cache writes, no RoPE: V cache | `50.083984` |
| **One K-projection BMM + one cache slice-copy: K/cache** | **`131.937500` each** |
| Same one-BMM repro with all unit-BMM marking suppressed | **`0.000000` each** |

The minimal failing kernel is therefore one K projection followed by one
slice-copy mutation in a shared `invoke_subgraph` body. Mutation
functionalization changes the projection rewrite such that
`_mark_static_unit_batch_bmm` tags it with
`_spyre_shared_weight_unit_bmm`. `_preserve_shared_weight_unit_bmm_dim` then
injects `_spyre_bmm_unit` and rewrites the BMM operand/output physical layout.
That tag and dimension are absent from the known-good inline body. Suppressing
the marker removes the generated `_spyre_bmm_unit`, makes the shared BMM layout
match inline, and makes the device result bit-exact.

#### Minimal production fix now in the dirty torch-spyre tree

`torch_spyre/_inductor/temp_passes.py` now:

1. recognizes a BMM output with any direct rank-expanding reshape user;
2. delays `mm_to_bmm` unit-BMM marking until after use replacement, when those
   real downstream users are visible;
3. skips shared-weight unit-BMM marking for that case, while retaining it for
   ordinary rank-3 MLP BMMs.

CPU regression added to
`tests/inductor/test_coarse_tiling.py`:
`TestSharedWeightUnitBmmLayout::test_mm_to_bmm_does_not_mark_rank_expanding_output_view`.
TDD evidence:

- RED: failed because `_spyre_shared_weight_unit_bmm` was unexpectedly present.
- GREEN after the minimal fix: `1 passed in 0.69s`.
- The full `TestSharedWeightUnitBmmLayout` class run was started but interrupted
  by the user asking for a closeout; no pytest process remains.

#### New repros and artifact snapshots

- `repros/hier/standalone_projection_consumers_discriminator.py`
- `repros/hier/standalone_single_projection_cache_discriminator.py`
  - default run is the failing gate;
  - `B6_DISABLE_DIRECT_UNIT_BMM_MARKING=1` is a diagnostic-only process-local
    monkey-patch that suppresses `_mark_static_unit_batch_bmm`.
- `/tmp/torchinductor_aviros.single_projection_cache.bad.20260812_1` — minimal
  default failing `_1`/`_0` artifacts.
- `/tmp/torchinductor_aviros` — bit-exact diagnostic run with all unit-BMM
  marking suppressed; **not yet generated with the production fix**.
- `/tmp/torchinductor_aviros.projection_consumers.20260812_1` — RoPE-vs-cache
  discriminator artifacts.
- `/tmp/torchinductor_aviros.direct_pass_toggle.20260812_1` — an intentionally
  ineffective first control that disabled only `mark_direct_unit_bmm_pass`; the
  marker remained because `mm_to_bmm` calls `_mark_static_unit_batch_bmm`
  directly.

#### Exact next steps

1. Run the complete CPU class:
   `python -m pytest tests/inductor/test_coarse_tiling.py::TestSharedWeightUnitBmmLayout -q`.
2. Move `/tmp/torchinductor_aviros` aside (never delete it), then run
   `standalone_single_projection_cache_discriminator.py` **without** the
   diagnostic environment variable. Expect both deltas `0.000000` and confirm
   the generated shared wrapper contains no `_spyre_bmm_unit`.
3. Run `standalone_projection_consumers_discriminator.py`; expect both halves
   bit-exact.
4. Run the decisive original gate,
   `standalone_body_discriminator.py`; expect the former `14.613281` delta to
   become zero.
5. If all pass, run the whole-forward token comparison and the broader relevant
   torch-spyre tests. Remove diagnostic-only repro monkey-patching and old
   `B6_FOLD_DUMP` instrumentation before finalizing.

No commit was made. Preserve all inherited dirty-worktree changes.

### 2026-08-12 takeover update: PR #3683 / memory-planning lead refuted

The related nested-launch corruption fixed by
[`torch-spyre#3683`](https://github.com/torch-spyre/torch-spyre/pull/3683)
is real, but it is **not Blocker 6**:

- `InvokeSubgraph` inherits `ExternKernel`, so #3683 correctly prevents a
  parent-graph tensor from staying LX-resident across an `invoke_subgraph`
  launch. However, the failing shared attention/body wrapper invokes one
  SuperDSC bundle for the body; SDPA is decomposed inside that bundle and no
  nested compiled program runs within it.
- The existing direct-extern-user guard already keeps each hidden-state operand
  passed to `InvokeSubgraph` out of LX. More importantly, disabling LX planning
  did not change the failure.

New device controls, all using
`repros/hier/standalone_attention_discriminator.py` unless noted:

| Configuration | max delta |
|---|---:|
| Default prior run | `17.772461` |
| `LX_PLANNING=0`, 32 cores | `17.770508` |
| `LX_PLANNING=0`, `SENCORES=1` | `17.744141` |
| `LX_PLANNING=0`, `HBM_POOL_PLANNING=0`, 32 cores | `17.770508` |
| Same no-LX/no-HBM-pool config, SDPA-only discriminator | `0.000000` |

The generated LX-off wrappers contain no `allocation={'lx': ...}` entries.
The no-LX/no-HBM-pool wrappers contain neither LX nor HBM-pool allocations and
do not pass `_pool` to their kernels. Therefore **LX placement, HBM intermediate
reuse, and multicore work division are exonerated**. PR #3669 is also a
different bug: it registers eager `mul_`/`add_`/`sub_`/`div_` kernels, whereas
this path's only input mutation is the compiled KV slice assignment.

Static comparison of the 32-core LX-off attention pair found 39 aligned SDSCs.
After removing only `debug_handle_`, 32 are identical. Differences are limited
to:

- Q/K/V projection BMMs: SDSCs `3`, `7`, `24` (the shared body adds a
  degenerate `x=1` coordinate/layout dimension).
- SDPA/layout records: SDSCs `12`, `13`, `27`, `28` (including the 32-head axis
  represented as `4 x 8` in the shared body).

The identically configured SDPA-only discriminator is bit-exact, and the output
projection SDSC is identical. The next high-value device bisection is thus:

1. shared-vs-inline Q/K/V projections + RoPE + KV writes, comparing `q` and the
   mutated caches;
2. shared-vs-inline SDPA + reshape + output projection on fixed captured Q/K/V.

If both halves are clean, the defect requires fusion across that boundary.

---

## 1. The goal (unchanged, multi-session)

Fold Granite 3.3's whole forward into a single `torch.compile` graph on IBM Spyre
(`hf-adapters` + `torch-spyre`), reusing ONE decoder-block artifact across all 40
layers via `torch.compiler.nested_compile_region` → the `invoke_subgraph` HOP
(compile-once), to kill host-overhead / device-idle gaps while preserving
correctness.

Inductor's behavior with `invoke_subgraph`:
- **N=1 call** of a region → Inductor **inlines** the body → kernel suffix `_0`.
- **N≥2 calls** of the same region → Inductor keeps a **shared HOP body**
  (`auto_functionalized_subgraph_0`) reused across calls → kernel suffix `_1`.

Blockers 1–5 (guard registration, subgraph input-STL seeding, origin-in-graph,
pool alloc/dealloc, decomp patch) are resolved and in the uncommitted working
tree. See `project_torch_spyre_rebase_state` memory for the exact file list.

## 2. The blocker

The compiled whole-forward **runs** but produces **WRONG logits**. Device-proven:
the shared-HOP-body kernel `_1` is **wrong per se on a single invocation**, while
the same block inlined (N=1, kernel `_0`) is bit-correct.

**Decisive measurement** (`repros/hier/standalone_body_discriminator.py`):

```
max| _1(call1 inputs) - _0(call1 inputs) | = 14.613281
```

Both bodies evaluated on **byte-identical device tensors** (h0, selected_freqs,
dev_mask, and the SAME restored kc0/vc0 that `_1` saw at entry). The only variable
is which compiled body ran. So this is a **build-of-`_1` defect**, NOT a ≥2-call
runtime-replay effect. That discriminator is the single most important artifact —
re-run it to confirm any candidate fix actually moves the number toward 0.

## 3. What has been RULED OUT (do not re-litigate without new evidence)

Each of these was checked and is NOT the cause:

| Candidate | How it was ruled out |
|---|---|
| **Work-division fold planner** | ★ Just refuted (2026-08-12). Device dump: `_0` and `_1` produce **byte-identical** fold splits for all 7 batchmatmuls. See §4. |
| Decomp of the body | B6DIAG10: body decomposes identically |
| Codegen / OpSpec multiset / pool / layout (wrapper-diff) | 52/62 SDSC files byte-identical; bundle sig 21 params; OpSpec multiset matches |
| Parent per-call binding | each call binds distinct buffers correctly |
| Raw sequential launch | both launch orders 0.0 divergence |
| Cross-call output clobber | distinct output buffers per call |
| V.graph-scoping bug in work_division | static: `iteration_space_from_op` reads only the op's own `op_read_writes`; `V.graph` IS correctly re-scoped to the `SubgraphLowering` during subgraph codegen (wrapper.py:4186 `V.set_graph_handler(subgraph.graph)`); subgraph has its own sizevars/get_buffer/graph_inputs; NO positional-zip-into-parent-list bug (contrast Blocker 3's `V.get_real_inputs()`) |

**Over-claim warning for the next agent:** the SDSC-diff line of reasoning produced
THREE wrong conclusions in a row ("entirely extra fold dimension", "dscs_ is a
dict", "`_0` flat-32 vs `_1` 8×4"). The normalized-JSON SDSC diff is NOT a reliable
signal for `_0`/`_1` differences — it strips `debug_handle_` + index prefixes and
can mask or manufacture apparent deltas. Prove any SDSC claim on-device before
trusting it.

## 4. The fold-planner refutation (most recent work)

**Instrumentation (env-gated, currently in the tree, INERT unless `B6_FOLD_DUMP=1`):**
`torch_spyre/_inductor/work_division.py`
- import `os` (after `import math`) and `from torch._inductor.virtualized import V`
  (after the `GraphLowering` import).
- `_cost_model_matmul_planner` ~line 1565: `[B6_PLAN]` dump — graph name/id, op,
  n_dim, row_dims, m candidates, m_dim, batch_dims, k_dim, rhs_once, committed,
  `[B M N K]`, chosen b/m/n/k folds, in_coords, out_coords.
- `_cost_model_divide_op` ~line 1790: `[B6_FOLD]` dump — graph_name, graph_id, op,
  rtype, it_adj_sizes, device_coords, committed, default_splits, cost_splits.

**Run that produced the verdict:**
```bash
cd /mnt/devel/inductor_src/hf-adapters/.claude/worktrees/hier-compile
B6_FOLD_DUMP=1 PYTHONPATH="$PWD" HF_HOME=/mnt/models/hf_cache/ \
  python repros/hier/standalone_body_discriminator.py 2>&1 | tee /tmp/b6fold.log
```

**Result (in `/tmp/b6fold.log`, per-op comparison in `/tmp/b6fold_tagged.txt`):**
All 7 batchmatmuls get IDENTICAL fold splits in both builds, matched by buffer
index (the subgraph prefixes the buffer name):

| bmm buffer | cost_splits (BOTH `_0` and `_1`) |
|---|---|
| buf11, buf15, buf31, buf46, buf55, buf57 | `{d0:4, d1:8, d2:1}` |
| buf59 | `{d0:4, d1:4, d2:2}` |

Same `it_adj_sizes`, same `device_coords`, same `in_coords`/`out_coords`, same
classification (`m_dim=d0, batch_dims=[], committed={}`). The "8" is `n=8` (a split
of the N/output-feature **stick** axis `d1`), `m=4` on `d0` — **not** a head-count
fold and **not** a batch mis-classification. `_0` uses 8×4 too. **The fold planner
is exonerated.**

## 5. Surviving leads (where to look next)

The corruption enters `_1` somewhere that is NOT visible in the normalized SDSC
JSON and NOT the fold. The most concrete unexplained difference:

1. **★ Kernel fusion-provenance / name delta.** The `_1` fused attention kernel is
   named `...sum_t_transpose...scaled_dot_product_attention..._1` while `_0` is
   `..._overrideable...sum_transpose..._0`. Note `sum_t_transpose` vs
   `sum_transpose` — a `t`/transpose difference in the fusion grouping, even though
   the OpSpec **multiset** matches. This suggests the backend (dxp/deeptools)
   lowers the `_1` fusion to **different device code** (different transpose /
   contraction wiring), which would corrupt the result without changing any
   Inductor-level fold. **Chase this first.** Compare the two `spyreCodeDir`
   `bundle.mlir` BODIES (not just the signature) and the generated device assembly
   for the attention kernel between the `_0` and `_1` builds.

2. **A masked-but-material SDSC field.** Re-audit the SDSC normalize: confirm no
   operand-wiring / address / Affine field that DOES matter is being stripped along
   with `debug_handle_` and the index prefix.

3. **Input-operand layout seeding** (`_subgraph_input_stls`, propagate_layouts.py
   ~1578). The dump showed identical `device_coords` for the bmm operands, so if a
   layout delta exists it is in a property NOT captured by device_coords (e.g.
   `stride_map`, `elems_per_stick`) or on a NON-bmm op. Probe: compare
   `FixedTiledLayout.device_layout` (device_size, stride_map, elems_per_stick) of
   the attention bmm operands+output between the two builds at `collect_tensor_deps`.

Given lead #1 is a concrete, unexplained static difference in the fused-kernel
identity, it is the strongest thread — the divergence is most likely on the
**deeptools/dxp lowering side of the `_1` fusion**, not in the Inductor Python
passes (which have now been extensively exonerated).

## 6. Key artifacts and files

- **Discriminator (the gate):** `repros/hier/standalone_body_discriminator.py`
  — builds N=2 (`_1`) then N=1 (`_0`) in one process on identical device tensors.
  MODEL=`ibm-granite/granite-3.3-2b-instruct`, BLOCK_SIZE=64, prompt="The capital
  of France is".
- **Device dump:** `/tmp/b6fold.log`, `/tmp/b6fold_tagged.txt` (fold refutation).
- **Instrumentation:** `torch_spyre/_inductor/work_division.py` (env-gated
  `B6_FOLD_DUMP`; §4). **Remove when no longer needed** — it is harmless but should
  not ship.
- **Uncommitted working tree** (7 files, Blockers 1–5 + tests + a2 decomp no-op).
  See `project_torch_spyre_rebase_state` memory. base commit `e2acb51` == current
  `upstream/main` tip; nothing to rebase.
- **Plan file** (design spec, DO NOT COMMIT):
  `/mnt/home/.claude/plans/zany-questing-sundae.md`.

## 7. Constraints still in force (from the user, verbatim intent)

1. **STOP-AND-REPORT before ANY fallback / fix.** A refuted hypothesis is an
   architectural pivot — report before starting a new direction.
2. **Do not commit the design spec** (the plan file).
3. Do **not** `torch.manual_seed` (eagerly inits the busy VFIO card).
4. Do **not** kill/relaunch processes to clear VFIO "busy" (self-inflicted churn;
   launch once). See `reference_vfio_transient_busy_launch_churn` memory.
5. Preserve compile-once (one block artifact across all 40 layers).
6. Recursive-delete guard: MOVE files aside, never delete (cache: `mv
   /tmp/torchinductor_aviros` aside after `_inductor` edits, don't `rm`).

## 8. How to run on device

Card is on host `aviros-spyre-test`, single-tenant VFIO, currently free. Run from
the worktree `/mnt/devel/inductor_src/hf-adapters/.claude/worktrees/hier-compile`
with `PYTHONPATH="$PWD" HF_HOME=/mnt/models/hf_cache/`. Real weights in
`/mnt/models/hf_cache/hub`. See `project_spyre_test_host_env` memory. Python-only
`_inductor` edits need no rebuild; C++ changes need
`../torch-spyre-docs/scripts/build-torch-spyre.sh --local-pytorch` from
`torch-spyre/` (see `project_torch_spyre_rebuild`).
