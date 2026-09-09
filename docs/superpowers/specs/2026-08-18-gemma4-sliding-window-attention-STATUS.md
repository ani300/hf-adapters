# Gemma 4 sliding-window attention — status

> **SUPERSEDED (2026-08-20).** The blocker described below (§ "The blocker") is
> **resolved.** The design pivoted to a `seqlen_q=1` decode (the 64-row in-graph
> query stick was dropped), which was the trigger for `out_reuse_dim.size() == 1`.
> Gemma 3 1B now runs the op end-to-end on device (5/5 token-compare) and all 12
> tasks are done. For the current state and the upstream landing order, read
> **`2026-08-18-gemma4-sliding-window-attention-HANDOFF.md`**. The rest of this
> file is kept as the record of the blocker investigation that motivated the pivot.

**Status:** paused. Phase 1 (op over the full-length cache) is **device-validated and
working**. Phase 2 (the anchored compact buffer) is **logic-complete and CPU-verified but
blocked on device by a torch-spyre compiler limitation**, characterised below.
**Date:** 2026-08-19
**Design:** `2026-08-18-gemma4-sliding-window-attention-design.md`
**Plan:** `../plans/2026-08-18-gemma4-sliding-window-attention.md` (Tasks 1–9 of 12 done)

## What works, measured

`spyre::sliding_window_attention` is correct for Gemma 4's shapes. Device A/B of one Gemma 4
sliding layer, band-masked SDPA versus the op, both measured against a **float32 CPU
reference** (`tests/spyre/test_swa_layer_ab_spyre.py`, `SENCORES=1`):

| case | op error | band error (the path we ship) | ratio |
|---|---|---|---|
| prefill `Lq=64` | 0.0749 | 0.0845 | **0.89** — op better |
| prefill `Lq=512` | 0.0885 | 0.0992 | **0.89** — op better |
| left-padded prefill (`valid_start=[17]`) | 0.1038 | 0.0844 | 1.23 |
| decode `Lq=1` | 0.0289 | 0.0103 | 2.82 |

Output scale ≈ 2.5. On two of four shapes the op is *more* accurate than the band-masked
path it replaces.

**Why the assertion is a ratio and not a tolerance.** Two fp16 device paths cannot be
required to agree tightly: the shipped band path is itself 0.049 from float32 truth, so it
fails `rtol=1e-2, atol=1e-3`. Gemma 4 attends unscaled (`scaling == 1.0`) at
`head_dim=256`, so a random-weight softmax is nearly one-hot and reduction-order noise is
amplified. The test asserts
`op_error <= max(2.0 * band_error, sqrt(terms) * eps * scale) + 5e-3`: the op must be no
less accurate than the path it replaces, or within fp16's own floor for the reduction it
performs. A wrong-column defect would err at the output scale — 15–35x the allowance.

Also green: `head_dim=256` and the anchored 1088-row stick geometry against the op's own
CPU reference (torch-spyre `tests/inductor/test_sliding_window_attention.py`), `valid_start`
including a ragged batch, and 131 of 133 in torch-spyre's two SWA suites.

## The blocker

The anchored decode path **fails to compile**, not to agree:

```
error: sbf-ddc: DtException: out_reuse_dim.size() == 1
  dxp_standalone -d .../sdsc_fused_add_index_copy_index_select_linear_mean_mul_rsqrt_
                     sliding_window_attention_transpose_view_1_n0
```

Six device probes at Gemma 4's real geometry (`capacity=1088`, `W=1024`, `head_dim=256`),
each the same op call with one ingredient changed:

| probe | result |
|---|---|
| the stick alone, no `o_proj`/norm tail | compiles |
| stick via `index_copy` + KV write + `o_proj` + norm mean | **fails** |
| stick via broadcast-multiply instead + same tail | **fails** |
| `Lq=64` query passed **natively** + same write + same tail | **compiles** |
| `Lq=64` native + tail, no KV write | compiles |
| `Lq=64` native + write, no tail | compiles |
| `Lq=1` with `cache_seqlen=capacity` + write + tail | **fails** |

**Conclusion: expanding a 1-row query to 64 rows inside the graph is the trigger.** It fails
through two independent mechanisms — `index_copy` and broadcast-multiply — while a
natively-64-row query at the identical geometry compiles. It is not the self-referential
buffer roll (reproduces with no shift) and not the indexing op as such.

Separately, `Lq=1` with `cache_seqlen=capacity` also fails, while `Lq=1` with
`cache_seqlen=written+1` (the phase-1 geometry) compiles and runs — so phase 1 is unaffected.

Probe log: `.superpowers/sdd/2026-08-18-gemma4-sliding-window-attention/probe-anchored-compile.log`

### Why this is load-bearing

The 64-row query stick exists *only* to pin `seqlen_q` so decode compiles one graph instead
of one per position. Building it outside the graph costs either a host round-trip per
sliding layer per decode step (~21 MB/step across Gemma 4's 40 sliding layers) or running
the whole block on 64 rows, paying 64x the MLP. Neither is obviously worth it; both need
measuring.

## What is verified on CPU but unproven on device

`SlidingWindowCache`'s bookkeeping, the compaction at the prefill/decode boundary, and the
buffer roll — including a test at the **shipped** geometry (`capacity=1088`, `anchor=1024`,
prompt 1024, 65 steps, exactly one roll) and 70 anchored decode steps matching the
full-cache band path step for step. The arithmetic is sound; only its compilation is not.

## Options, not yet chosen

1. **File the compiler issue upstream**, default to phase 1, keep phase 2 reachable via
   `model._spyre_swa_mode = "anchored"` and marked device-blocked. Honest; no memory win yet.
2. **Eager-stick workaround** — split the block so the stick is built between two compiled
   regions, giving the second a natively-64-row query (which the probe shows compiles).
   Preserves one-graph decode and the memory win; costs the host round-trip above. Measure
   before committing.
3. **Fix `out_reuse_dim.size() == 1`** in the DCG scheduler. Highest ceiling and probably
   affects other decode-shaped graphs, but unbounded scope.

## Not done

Tasks 10–12: the Gemma 3 1B control (currently the **only** prospective end-to-end evidence,
since Gemma 4 12B's own device token-compare is red 0/5 for unrelated reasons and diverges at
prefill), the Gemma 4 before/after record, and offering `valid_start` upstream.

## Not measured

No latency or memory measurement was taken anywhere in this work. Every performance figure in
the design document is arithmetic, and none of it should be repeated as a result.

## Also found, worth reporting upstream

- **PR #3405 has two failing tests of its own**, unchanged by the rebase onto latest main:
  `test_query_length_not_a_multiple_of_the_block` and `test_ragged_query_and_window_together`,
  both `Lq=100`, producing `inf` with ~60% of output elements mismatched. That is the op's
  internal query front-padding path (`pad_rows=28`). This integration never enters it — every
  `Lq` it passes is 1 or a multiple of 64, so `pad_rows` is always 0.
- **The op is ~2.8x less accurate than SDPA at `Lq=1`/`M=1`**, within fp16's floor but
  consistently so. Whether that persists at `q_block=64` is unknown, because no anchored
  decode graph compiles.
