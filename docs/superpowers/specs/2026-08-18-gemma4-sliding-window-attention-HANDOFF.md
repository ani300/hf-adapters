# Gemma SWA custom op — landing handoff

**Date:** 2026-08-20
**Design:** `2026-08-18-gemma4-sliding-window-attention-design.md`
**Plan:** `../plans/2026-08-18-gemma4-sliding-window-attention.md` (all 12 tasks done)
**Supersedes:** the `-STATUS.md` note — its compile blocker is resolved (see §0).

This document is for the person who lands the work. It says **what must merge, in
what order, and why**, to get Gemma 3 and Gemma 4 running in hf-adapters on the
`spyre::sliding_window_attention` op. The building is done; this is a merge plan.

---

## 0. Where we are

- **Gemma 3 1B runs the op end-to-end on device** — 5/5 token-compare against HF-CPU.
  This is the green control. "If Gemma 3 works on device it works" — the op path,
  the compact buffer, the roll, and `valid_start` are all exercised by it.
- **Gemma 4 12B is red for reasons unrelated to SWA** — its own device
  token-compare is 0/5 and it diverges at *prefill*, before any sliding layer is
  the suspect. SWA is not what's blocking Gemma 4; do not tie the two together.
- **The design pivoted (2026-08-20): the 64-row query stick was dropped.** Decode
  now passes `seqlen_q=1` with `cache_seqlen=write_row+1`, sweeping ≤64 reused
  compiled graphs instead of one pinned graph. This **resolved** the
  `out_reuse_dim.size() == 1` compile blocker the STATUS file describes — that
  blocker was the in-graph 1→64 expansion, which no longer exists. The compact
  buffer + eager shift (`roll_compact_buffer`) are retained for the memory bound.
- **hf-adapters branch `gemma4-swa-op`** holds all the integration work, 84 commits
  ahead of `origin/main`. No PR opened yet (blocked on the torch-spyre deps below).

---

## 1. The dependency chain

hf-adapters depends on a torch-spyre release that contains the op **and** the
fixes the op's shipped geometry needs. Landing order is bottom-up:

```
  torch-spyre                                        hf-adapters
  ───────────                                        ───────────
  #3405  spyre::sliding_window_attention  ─┐
    ├─ valid_start (offered, patches)       │
    └─ copy_f→opaque_copy_ migration        ├──►  release ──►  gemma4-swa-op PR
  #3903  lowering fix (seqlen_q=1 tail)    ─┘                    (Gemma 3 green;
  #3882  scatter fix  ........ MERGED ✓                           Gemma 4 red,
  #3904  predicate fix ....... off critical path                  unrelated)
  #3733  bmm guard ........... superseded by #3903
```

**Critical path = the three joined by `┐┘`.** Everything else is already merged,
defensive, or off-path after the pivot.

---

## 2. What must land, in order

### Step 1 — `valid_start` into #3405  ⏳ offered, awaiting author

- **What:** an optional per-batch `valid_start` argument (first attendable column)
  so left-padding is excluded from the window, plus two Gemma-4-geometry tests
  (head_dim-256 GQA prefill; `seqlen_q=1` anchored decode).
- **Why it's required:** an offset-and-length window **cannot** skip pad columns.
  Without `valid_start` the op silently attends the left-pad K/V columns — logit
  magnitude barely moves but the **argmax flips**. It reads like a numeric bug and
  is actually a missing input. On the band path the same information rides in the
  additive mask; on the op path it must be an explicit argument.
- **Status:** offered on #3405 as 3 `git am`-ready patches
  (comment: `pull/3405#issuecomment-5358950913`). Verified 131 passed; the only 2
  failures pre-exist on `swa-window-roll` identically (the op's front-padding path,
  `pad_rows>0`, which this integration never enters). Awaiting @abhishekkunuru6-cmyk.
- **hf-adapters reads it via** `valid_start_for(model, bsz)` from
  `model._spyre_prompt_offsets`.

### Step 2 — `copy_f → opaque_copy_` migration into #3405  ❗ gap, no PR

- **What:** commit 98a982ed migrates the op's decomposition off `copy_f`.
- **Why it's required:** after upstream #3811, #3405's own decomposition won't
  compile without it. This is **#3405's code**, not a separable fix — it belongs
  *in* #3405.
- **Status:** **carried by no PR branch.** It lives only on our working branch. The
  #3405 author must be told it's needed; it is not in the `valid_start` patch set
  (that set is deliberately scoped to the feature). **This is the one action item
  with no owner yet.**

### Step 3 — lowering fix into a release: #3903  📝 DRAFT, on critical path

- **What:** fixes the DCG scheduler for a `seqlen_q==1` op fused with an
  RMSNorm+Linear tail (bundles the standalone bmm guard from #3733).
- **Why it's required:** `seqlen_q=1` + norm/linear tail **is** the shipped decode
  geometry after the pivot. Without this fix that graph fails to compile
  (`out_reuse_dim.size() == 1`). The bmm guard only *rejects* the bad shape; this
  fix makes it *work*.
- **Status:** DRAFT on `fix/matmul-contraction-two-dims`. **Must move draft→ready→
  merged.** Resolve the overlap with #3733 (which is the standalone guard #3903
  already contains) — likely close #3733 in favour of #3903, reviewer's call.

### Already landed / off critical path (no action to ship Gemma 3):

- **#3882 scatter fix — MERGED ✓.** Also no longer *triggered* post-pivot (the
  stick that needed the indexed scatter is gone), so it's belt-and-suspenders now.
- **#3904 predicate fix (reject inexpressible layout) — DRAFT, off-path.** Not
  triggered by the seqlen_q=1 path. Independent value; land on its own schedule.
- **#3733 standalone bmm guard — OPEN, superseded** by #3903's bundled copy.

### Step 4 — cut a torch-spyre release, bump hf-adapters, open the PR

Once #3405 (with `valid_start` + `copy_f`) and #3903 are merged: pin hf-adapters to
that torch-spyre, open the `gemma4-swa-op` PR (84 commits). It should go green on
the Gemma 3 1B token-compare in CI, the same control that's green on the pod.

---

## 3. Minimal set vs. everything

**Minimal set to ship Gemma 3 on the op:** Step 1 + Step 2 + Step 3. That's it.
Two of them (`valid_start`, `copy_f`) merge into #3405; one (#3903) is its own PR.

**Not required for Gemma 3, land on their own merit:** #3904 (predicate), #3733
(guard — or just fold into #3903 and close). #3882 is already merged.

**Gemma 4 12B is not in the minimal set and this op does not unblock it.** Its
divergence is at prefill and unrelated to sliding-window attention. Treat "Gemma 4
on the op" as: *the op is ready for Gemma 4's geometry* (head_dim-256 GQA is tested
and green) *the moment Gemma 4's unrelated prefill issue is fixed* — not as work
this SWA effort still owes.

---

## 4. Verification gates (don't skip)

- torch-spyre, per PR: `tests/inductor/test_sliding_window_attention.py` +
  `tests/inductor/test_kv_window.py` green except the 2 known pre-existing
  front-padding failures (`test_query_length_not_a_multiple_of_the_block`,
  `test_ragged_query_and_window_together`). `pre-commit run --all-files` clean.
  Sign off commits (`-s`). Never run Spyre tests in parallel.
- hf-adapters, on the pod: `pytest -s tests/spyre/test_e2e_token_compare_spyre.py
  -k gemma3` → 5/5. This is the gate that says the whole chain works.
- **A/B tolerance model** (device op vs. band, both vs. float32):
  `op_err <= max(2.0*band_err, sqrt(terms)*eps*scale) + 5e-3`. Gemma 4 attends
  unscaled at head_dim=256 → near-one-hot softmax amplifies reduction-order noise,
  so a *ratio*, not a fixed tolerance. A wrong-column defect errs at output scale
  (15–35× the allowance) — the gate catches it.

---

## 5. One-paragraph version

Gemma 3 already runs the op on device (green control); Gemma 4's redness is a
separate prefill problem, not SWA. To ship, three things must merge upstream:
`valid_start` and the `copy_f→opaque_copy_` migration, both into PR #3405, and the
`seqlen_q=1` lowering fix, PR #3903. `valid_start` is offered and waiting on the
#3405 author; the `copy_f` migration has **no PR and needs an owner**; #3903 is a
draft that needs to go ready. The scatter fix (#3882) already merged; the predicate
fix (#3904) and standalone bmm guard (#3733) are off the post-pivot critical path.
Then cut a torch-spyre release, bump hf-adapters, and open the `gemma4-swa-op` PR.
