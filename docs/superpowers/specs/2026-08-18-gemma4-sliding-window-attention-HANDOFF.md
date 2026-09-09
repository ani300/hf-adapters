# Gemma sliding-window attention — current integration status

**Updated:** 2026-09-09

**torch-spyre dependency:** merged PR #3405 plus follow-up PR #4423

**hf-adapters base:** upstream `main` at `ee1e8a9`

This supersedes the earlier August landing notes. The `gemma4-swa-op` work has
now been reconciled with current hf-adapters main and audited against the current
head of torch-spyre PR #3405.

## Implemented

- Dense Gemma 4, including PLE and KV-sharing E variants, uses
  `spyre::sliding_window_attention` for causal sliding layers.
- Dense Gemma 3 causal language models use the same path.
- Sliding layers allocate only the padded-prompt capacity needed for prefill,
  then compact to an anchored `window + one stick` buffer at first decode.
- Compaction is deliberately deferred until first decode, after all chunked
  prefill calls have populated the owning caches.
- Every 64 decode tokens the compact sliding caches roll into fresh pinned-layout
  allocations. Contiguous slices use torch-spyre's offset-aware device-to-device
  copy path, avoiding the eager `index_copy_` CPU fallback. Decode position,
  unwritten rows, window placement, and left padding travel in one fixed-shape
  runtime mask shared by every sliding layer, so all positions reuse one graph.
- Prefill left padding is carried through PR #3405's `valid_start` argument.
  Gemma 4 also zeros padded query rows after every block so invalid activations
  cannot poison later global-attention KV caches.
- KV-sharing consumers alias their producer's cache allocation and are rebound
  after compaction/roll; they no longer retain unused full-size cache tensors.
- The band-masked SDPA path remains available with
  `model._spyre_swa_mode = None` before adapter preparation. `"phase1"` remains a
  full-cache correctness/debug path.

## Audit results

PR #3405 already contains the formerly external `valid_start` work and the current
functional `copy_forced` API migration. It contains runtime changes through main
`875ea166`; the subsequently published `0e0f01fc` changes CI workflow naming only
and has no runtime or API delta.

The integration and follow-up torch-spyre audit fixed these gaps:

1. The old adapter compacted after the first prefill call. Current hf-adapters can
   split prefill into multiple calls, so later chunks then wrote absolute cache
   indices into a compact buffer. Compaction now occurs only at first decode.
2. A nonzero `valid_start` made padding query rows fully masked. Softmax over an
   all-`-inf` row can produce NaNs, which can contaminate later layers. PR #3405's
   band now retains a harmless diagonal for padding query rows; callers still
   discard or zero those rows.
3. Eager cache compaction and rolling used `index_copy_`, which fell back through
   CPU on Spyre. They now use contiguous sliced `copy_`, dispatched through
   torch-spyre's offset-aware `spyre::copy_from_d2d` path and verified on device.
4. Decode geometry was expressed as Python integers and compiled up to 64 graph
   variants. The op now accepts a fixed-shape runtime mask and consumes it in the
   fused windowed recurrence, yielding one decode graph without an SDPA fallback.
5. Gemma 4's four-stick head rows exposed layout and mutable online-softmax
   defects. Query/window materialization, functional SSA softmax carries, bounded
   KV chunks, and corrected hint placement now cover D=256 MHA/GQA and batches.
6. Late scratchpad graph edits could reuse an existing `opN` name after prior
   graph pruning, corrupting dependency resolution. Late registrations now choose
   an unused name, and iteration-space inference ignores ordering-only `WeakDep`s.

The PR API documentation was also corrected: long prefills are internally tiled
into 64-row query blocks. The required attention-buffer width is therefore
`round_up_64(window_size + q_block - 1)`, not
`round_up_64(window_size + full_query_length - 1)`. A left-aligned prefill cache
must additionally contain every written position through `cache_seqlen - 1`.

## Deliberate exclusions

- Gemma 3 bidirectional embedding models stay on the mask path because the op is
  causal-only.
- Gemma 4 multimodal models stay on the mask path because their bidirectional
  vision overlay can widen attention beyond a causal window.
- Gemma 4 MoE stays on its existing mask path for now. Its attention is embedded
  in separately compiled prefill/decode regions and needs device validation before
  the op and compact-cache state are threaded into that call contract.
- Gemma 2 requires attention-logit softcapping, which PR #3405 does not expose.
  GraniteSWA can use a value head dimension different from its key head dimension,
  while PR #3405 currently requires identical key/value cache shapes. ModernBERT
  is bidirectional. These adapters therefore remain on their existing mask paths.

## Compilation and performance

Anchored decode now has one fixed tensor signature instead of up to 64 Python
`cache_seqlen` variants. Together with prefill, that restores the expected two
attention shapes per block. At Gemma-like B=1, Hq=4, Hkv=2, D=256, W=1024,
Lk=1088 geometry, warm runtime-mask SWA measured 0.562 ms versus 0.677 ms for
equal-sized masked SDPA and 2.692 ms for full-cache Lk=8192 SDPA. The 1088-column
mask build and transfer measured 146.2 us once per step and is shared by all
sliding layers.

## Verification and landing

CPU coverage includes op-vs-band layer comparisons, compact-cache shifts,
left-padding, exact Gemma geometries, chunked prefill followed by first-decode
compaction, allocator-hook dispatch, and KV-sharing consumers. The focused
hf-adapters CPU suites pass 32 tests. Against a compatible local native build,
the torch-spyre SWA, KV-window/planner, and pass-utils suites pass 142 tests. The
hf-adapters Gemma 4 device A/B suite passes all five cases, including left
padding and 70 decode steps across a compact-cache roll. Gemma 4 E2B also has 5/5
CPU-vs-Spyre top-1 agreement across prefill and four decode steps.

The broader CPU generation suite reports 49 passes, one expected failure, one
unexpected pass, and two failures unrelated to this SWA path: Gemma 4 E4B has the
same final-token drift with the band fallback forced, while Gemma 4 MoE encounters
an existing CPU/Spyre fixture-device mismatch in its prefill FFN.

Landing order:

1. Land the torch-spyre runtime-mask/kernel and compiler-fix follow-up to #3405.
2. Build/release that torch-spyre revision and run the focused device suites.
3. Point hf-adapters CI at that release and land this integration.

No latency or peak-memory measurements have been taken in this refresh. The
memory reduction follows directly from per-sliding-layer cache capacity, but it
should not be quoted as a measured performance result until benchmarked.
