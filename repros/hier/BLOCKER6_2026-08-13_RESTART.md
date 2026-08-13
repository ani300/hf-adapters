# Blocker 6 restart — 2026-08-13

**Read this before the older BLOCKER6_HANDOFF.md.** This supersedes the older
claim that the first unit-BMM fix completely solved Blocker 6.

No commit was made. No cache was deleted. No device process is running.

## Environment

- torch-spyre: `/mnt/devel/inductor_src/torch-spyre`
- hf-adapters: `/mnt/devel/inductor_src/hf-adapters/.claude/worktrees/hier-compile`
- Python: `/mnt/devel/inductor_src/.venv/bin/python`
- PyTorch: `2.13.0a0+gitcf30153` from `/tmp/pytorch/torch/__init__.py`
- Sandbox bwrap is broken; Python/device runs require the approved host.
- Preserve all dirty changes and caches.

## Existing source changes

The first production fix is in:

- `torch-spyre/torch_spyre/_inductor/temp_passes.py`
- `torch-spyre/tests/inductor/test_coarse_tiling.py`

It delays unit-BMM marking until downstream users are visible and avoids
marking rank-expanding projection outputs.

Focused test:

```text
tests/inductor/test_coarse_tiling.py::TestSharedWeightUnitBmmLayout
5 passed in 1.41s
```

## Device results

All production runs used the dirty torch-spyre tree first on `PYTHONPATH`,
unset `B6_DISABLE_DIRECT_UNIT_BMM_MARKING` and `B6_FOLD_DUMP`, and used
fresh host caches.

### Minimal K projection + cache

Script: `standalone_single_projection_cache_discriminator.py`
Cache: `/tmp/b6-production.bDGaw7`

```text
K delta = 0.000000
K cache delta = 0.000000
```

The cache contains neither `_spyre_bmm_unit` nor
`shared_weight_unit_bmm`.

### Projection consumers

Script: `standalone_projection_consumers_discriminator.py`
Cache: `/tmp/b6-production.N6C000`

```text
Q / K / raw V after RoPE:             0.000000 each
raw Q / K cache / V cache after copy: 0.000000 each
```

### Original full decoder-body gate

Script: `standalone_body_discriminator.py`
Cache: `/tmp/b6-production.oeu7cE`

```text
max| _1(call1 inputs) - _0(call1 inputs) | = 8.664062
```

This fell from `14.613281`: the first fix is real but incomplete.
Whole-forward testing was stopped.

### Entire attention pipeline

Script: `standalone_attention_discriminator.py`
Cache: `/tmp/b6-production.ufVBlD`

Includes Q/K/V, RoPE, cache mutation, SDPA, and output projection.

```text
max|attention _1(call1) - _0(call1)| = 0.000000
```

The residual is in layernorm/residual/MLP, not attention.

## Residual root cause

Generated wrappers:

- Shared N=2:
  `/tmp/b6-production.oeu7cE/oj/coj34gjmtlfbepkarl2i6tejvdtlqhuo7gbdyejottspvbkpl4c4.py`
- Inline N=1:
  `/tmp/b6-production.oeu7cE/xg/cxgvsrbtgc5hpchldzgrhhbux3vcadpabvetiys2w35mwfgndnqh.py`

The shared wrapper injects `_spyre_bmm_unit`, adds
`shared_weight_unit_bmm`, and retains a leading physical size-1 axis on all
three Granite MLP projections:

- gate: `64 x 2048 -> 8192`
- up: `64 x 2048 -> 8192`
- down: `64 x 8192 -> 2048`

The inline wrapper has identical mathematical iteration sizes but no marker and
no leading unit axis. The attention output projection has marker metadata but
no injected unit dimension and is bit-exact.

Suppressing all unit-BMM marking with the existing process-local diagnostic
monkey patch, cache `/tmp/b6-production.mFzILH`, produced:

```text
max| _1(call1 inputs) - _0(call1 inputs) | = 0.000000
```

Thus the three MLP unit-BMM transformations cause the residual.

Optimization history:

- `74f6c92`: Optimize prefill MLP shared-weight unit BMM (#2550)
- `9035fb8`: Narrow shared-weight unit BMM marking (#2588)
- `611a77b`: skip preservation for higher-rank layouts (#3107)

This performance optimization must not be disabled globally.

## FX provenance

Diagnostic cache: `/tmp/b6-production.BaRgoZ`

- N=2 called `_mark_static_unit_batch_bmm` four times: attention output plus
  three MLP projections.
- N=1 never called that marker path.
- N=2 marker nodes expose only `val` metadata.
- Owning GraphModule metadata only says `post_grad_custom_post_pass`.
- No stable FX-level invoke-subgraph flag was found. Do not invent one.

## Proven lower-level fix direction

The optimization is meant to recover a unit axis squeezed out during layout
construction. Its existing test deliberately deletes the physical unit axis
before calling `_preserve_shared_weight_unit_bmm_dim`.

Shared-region MLP TensorArgs already retain an explicit physical size-1,
coordinate-0 axis. The helper wrongly rewrites that existing axis into the
active `_spyre_bmm_unit` iteration dimension.

A process-local monkey patch made the helper return the original iteration
space when a marked BMM input or output already had such a physical unit axis.
Normal squeezed-layout marking remained enabled.

Cache: `/tmp/b6-production.CRdL3k`

```text
max| _1(call1 inputs) - _0(call1 inputs) | = 0.000000
```

Strongest fix: only synthesize/preserve the unit-BMM dimension when both target
layouts have had the unit axis squeezed away. If an explicit physical unit axis
already exists, leave the OpSpec unchanged.

## Exact next steps

1. TDD in `TestSharedWeightUnitBmmLayout`: add a failing test proving
   `_preserve_shared_weight_unit_bmm_dim` does not rewrite an already-present
   rank-4 physical unit axis.
2. Keep the existing squeezed-layout optimization test passing.
3. Add the minimal early return in
   `torch_spyre/_inductor/spyre_kernel.py`; do not disable marking globally.
4. Run the focused class.
5. With fresh caches and no diagnostic patches rerun:
   single projection/cache, projection consumers, attention, and full body.
6. Require every delta to be `0.000000`.
7. Only then run whole-forward token comparison and broader tests.
8. Remove diagnostic monkey-patching and `B6_FOLD_DUMP` before finalizing.

## Constraints

- Never use `torch.manual_seed` in device repros.
- Never kill/relaunch processes to clear VFIO.
- Move caches aside; never recursively delete.
- Preserve compile-once/shared-region behavior and all dirty changes.
- No commits were made.

