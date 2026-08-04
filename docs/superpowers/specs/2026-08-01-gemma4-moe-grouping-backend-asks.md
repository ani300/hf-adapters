# Gemma 4 MoE — Backend-Collaboration Asks for Per-Expert Grouping (Approach B)

**Date:** 2026-08-01
**Status:** Specify only — nothing in this document is implemented. No host-grouping
code and no RISC-V code has been written against this spec.
**Audience:** torch-spyre / deeptools backend engineers. This document is
self-contained — it does not assume you have read the design spec it is
extracted from.
**Source spec:** `docs/superpowers/specs/2026-08-01-gemma4-moe-loop-on-topk-design.md`
(section "Approach B — backend-collaboration grouping (specify only)" and
"Validation & top-8 restore"). This document restates that content
verbatim-faithfully as a standalone reference; it does not add new facts or
citations beyond what that spec states.
**Implementation plan:** `docs/superpowers/plans/2026-08-01-gemma4-moe-loop-on-topk.md`.

## Context in one paragraph

Gemma 4's MoE FFN (`google/gemma-4-26B-A4B-it`: 128 experts, top-8 trained,
`moe_intermediate_size=704`, `hidden_size=2816`) is being ported to run on
Spyre. A separate, already-in-progress path ("Approach A") runs the whole
MoE FFN on-device except the scatter-combine, by looping directly over the
`[T,K]` topk results with an on-device `index_select` from an HBM-resident
expert stack — no CPU grouping, no argsort. Approach A does one weight fetch
**per row**. Approach B, described here, is the follow-on: turn Approach A's
per-row weight fetches into **once-per-expert** fetches, by grouping routed
rows into contiguous per-expert segments and driving a fixed-size loop over a
single static device program that gathers one expert weight slab plus its
group of activations per iteration.

Approach B is **not implemented in the current plan** because the per-segment
device grouped program cannot be tiled until backend primitive #1 (below)
exists. Building the host-grouping side now would only reproduce the
project's existing, non-working `4B` path. **B is landed once the primitive
lands** — that is the purpose of this document: to hand the backend team a
precise, actionable spec of what's needed, so grouping can be picked up as
soon as the primitive is available.

## B-Stage 1 — grouping on host CPU

### What "grouping" means, from scratch

The router has already run. For each of the `T` tokens it picked `K` experts,
so there are `N = T·K` **(token, expert) pairs** to compute. Two flat arrays
describe them, both length `N`, in the same order:

- `token_of_row[i]` — which token the `i`-th pair belongs to (values in `0..T-1`,
  each appearing exactly `K` times).
- `expert_of_row[i]` — which expert the `i`-th pair routes to (values in
  `0..E-1`, `E = 128`; arbitrary, data-dependent, in no particular order).

Approach A computes these `N` pairs in their natural (per-token) order, and so
must fetch a fresh expert weight slab for **every row** — even when two adjacent
rows happen to use the same expert. Grouping removes that waste by physically
**reordering the rows so that all rows routed to the same expert sit next to
each other**. Once the rows are expert-contiguous, a tile of consecutive rows is
guaranteed to share one expert, so the program fetches that expert's weight slab
**once per tile** instead of once per row.

Grouping is three plain array operations — no model knowledge, just sorting a
list of integers and recording where the runs begin and end:

1. **Sort the pairs by expert.** Compute the permutation that would sort
   `expert_of_row` into non-decreasing order — i.e. for each output position,
   which input row lands there. Call it `sort_perm [N]`. (In PyTorch this is
   `sort_perm = argsort(expert_of_row)`.) Applying `sort_perm` to the row arrays
   gives the reordered views:
   - `expert_sorted = expert_of_row[sort_perm]` — now non-decreasing, e.g.
     `[0,0,0,2,2,5,5,5,5,…]`: all of expert 0's rows, then all of expert 2's,
     and so on. Experts that got no rows simply don't appear.
   - `gathered_sorted = activations[sort_perm]` — the token activations
     reordered to match, so row `i` of `gathered_sorted` is the input for the
     expert named in `expert_sorted[i]`.
   - `token_of_row_sorted = token_of_row[sort_perm]` — kept so the final result
     can be scattered back to the right token.

2. **Count how many rows each expert got.** Tally the sorted expert list into a
   length-`E` histogram: `counts[e]` = number of rows routed to expert `e`
   (`counts = bincount(expert_sorted, minlength=E)`). With top-8 over 128
   experts and a typical prompt, most counts are small and some are zero.

3. **Turn counts into segment boundaries.** Take the running (prefix) sum of the
   counts, prepended with a leading `0`, giving `group_off [E+1]`
   (`group_off[0]=0`; `group_off[1:] = cumsum(counts)`). Then expert `e` owns
   exactly the contiguous row range `group_off[e] : group_off[e+1]` of the
   sorted arrays — that half-open interval is expert `e`'s **segment**. An
   expert with `counts[e]==0` has `group_off[e]==group_off[e+1]` (an empty
   segment).

Worked micro-example (`T=3`, `K=2`, so `N=6`, and suppose `E=6`):

```
expert_of_row      = [2, 0, 2, 5, 0, 2]      # token0→{2,0}, token1→{2,5}, token2→{0,2}
sort_perm          = [1, 4, 0, 2, 5, 3]      # positions of experts 0,0,2,2,2,5
expert_sorted      = [0, 0, 2, 2, 2, 5]      # non-decreasing after the sort
counts (E=6)       = [2, 0, 3, 0, 0, 1]      # expert0:2 rows, expert2:3, expert5:1
group_off (E+1=7)  = [0, 2, 2, 5, 5, 5, 6]   # expert2 owns 2:5, expert5 owns 5:6
```

So expert 2's segment is `group_off[2]:group_off[3] = 2:5` — rows 2, 3, 4 of the
sorted block, all of which need expert 2's weights fetched exactly once.

(The adapter already ships these three steps as `_moe_permute` (step 1) and
`_group_offsets` (steps 2–3); the algorithm above is the whole of what they do —
you do not need to read that code to implement or review this spec.)

### From variable-size segments to a fixed-trip static loop

- **Fixed-tile contract:** the device loop's trip count must be a compile-time
  constant, but per-expert segment sizes are data-dependent (they change with
  every prompt). Stage 1 bridges this with a **capacity/padding scheme**: round
  each expert's segment size **up to a multiple of `TILE`** rows, inserting
  padding rows so that every segment is `TILE`-aligned and **no tile ever
  straddles two experts**. After padding, the block has `N_pad` rows and the
  loop is exactly `N_TILES = N_pad / TILE` static iterations — one expert per
  tile, known at compile time.
- **The host→device side-channel.** Two small integer tables travel from host
  to device alongside the reordered activations: `group_off [E+1]` (the segment
  boundaries above) and `tile_expert [N_TILES]` (for each padded tile, the
  single expert id it belongs to). The device program reads `tile_expert[tile]`
  to know which weight slab to fetch for the current tile; padding rows
  contribute nothing to the scatter-combine.

Device static program (single program, hint-looped over `N_TILES`):

```
with spyre_hint(tiles={"row": TILE}):
    e    = tile_expert[tile]        # ONE expert id for the whole tile
    W_gu = gate_up_dev[e]           # ONE slab fetch [H,2M] — per_tile_fixed: loaded once/tile
    seg  = gathered_sorted[rows]    # [TILE,H] one expert's group
    ... bmm / gelu-tanh SwiGLU / bmm ...
```

Win vs. Approach A: **one weight slab per tile** (expert constant within a
tile) plus `per_tile_fixed` marks it loop-invariant (loaded once — see
deliverable #3 below). Approach A fetches a slab per row; grouping fetches a
slab per `TILE` rows.

## B-Stage 2 — grouping on the RISC-V CPU inside Spyre

Move Stage-1 grouping (argsort + bincount + cumsum + capacity-pad +
offset-table build) onto the **in-Spyre RISC-V CPU**, so topk results never
round-trip to the x86 host for grouping. The **device static program is
byte-identical to Stage 1** — only the *producer* of `group_off` /
`tile_expert` / `sort_perm` moves. This defines a **RISC-V ↔ device-program
ABI**: the RISC-V code writes the offset table plus permutation into a known
HBM/scratchpad region that the static program reads, with a defined
sync/fence contract.

## The four named backend deliverables (the collaboration asks)

1. **Per-segment operand-select tiling primitive.** A way to tile the loop by
   **per-expert segment** so one static program iterates `N_TILES` times,
   each binding one `expert_id → one weight slab`. Today `tiles={...}` binds
   only to an op's **output** ranges (`wsr/coarse_tile.py:758-768`,
   `wsr/coarse_tile_hints.py`); there is no hint to make a **per-tile scalar
   `tile_expert[tile]` select the weight operand** from `group_off`. This is
   the core new primitive — either a backend **grouped-GEMM op** or a
   `group_off`-driven **per-tile operand-select hint**.
2. **Windowed HBM→scratchpad indirect-gather correctness.** Harden the
   indirect-gather execution path: `expert_w[expert_ids]` reaches SDSC but is
   `xfail` on divergence by default (indirect gather defaults to xfail on
   numeric divergence — `tests/inductor/indirect_access_common.py:413-434`)
   and the literal MoE case is skipped for output-span overflow
   (`test_moe` skipped, `tests/inductor/test_indirect_access_gather.py:447-465`).
   **This is also Approach A's dependency** — surfaced by Approach A's
   on-card gate; if Approach A's gate fails here, this ask is what unblocks
   it.
3. **`per_tile_fixed` for the weight operand.** Confirm/extend that the
   loop-invariant-load flag fires for the expert weight slab within a
   segment tile (mechanism exists: `insert_restickify.py:281-345`).
4. **RISC-V grouping ABI (Stage 2).** Memory region, layout, and
   synchronization/fence contract for RISC-V-produced `group_off` /
   `tile_expert` / `sort_perm` handed to the static device program. Ties to
   the Spyre correction-path/host-compute ordering constraints — HostCompute
   runs inline, H2D→Compute is auto-barriered; a RISC-V→device handoff needs
   an explicit fence design.

## Validation & top-8 restore (both approaches)

- Each stage is validated by the **single-layer fp16-vs-fp32 rel-err gate**
  (mean_rel < 0.02 / max_rel < 0.5), then the **e2e token-compare**
  (`tests/spyre/test_e2e_token_compare_spyre.py -k 26B-A4B`, non-blocking
  xfail during bring-up).
- **Top-8 restore criterion:** K is pinned to 4 *because* on-device
  `topk(k>4)` SIGABRT'd and grouping ops didn't lower (ledger Task 2). Once
  routing runs on-device with a working `topk` (Approach A) — or once
  grouping is RISC-V-side where `k` is unconstrained (B-Stage 2) — lift `K`
  from 4 to the trained **top-8** and re-validate. **Gate:** token-compare
  top-1 agreement must recover with K=8 plus a correct routing/grouping
  path; that recovery is the signal that the K=4 pin (not an adapter bug)
  was the cause of the current 0/5 divergence.

## Out of scope for this document

- Implementing B-Stage-1 host grouping or B-Stage-2 RISC-V grouping code —
  specified only.
- Approach A implementation details (see the design spec and implementation
  plan linked above).
- Anything about the dense (non-MoE) Gemma 4 adapter path.

## Pointers

- Design spec (full context, both approaches):
  `docs/superpowers/specs/2026-08-01-gemma4-moe-loop-on-topk-design.md`
- Implementation plan (task breakdown, Approach A build + this doc):
  `docs/superpowers/plans/2026-08-01-gemma4-moe-loop-on-topk.md`
