# Gemma 4 Sliding-Window Attention Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the band-masked full-cache SDPA on Gemma 4's 40 sliding layers
with `spyre::sliding_window_attention`, reading only the 1024-column window from a
1088-row compact KV buffer.

**Architecture:** The op takes its geometry as trace-time integers, so the
integration is built entirely around keeping those integers constant: an
*anchored* compact buffer where `cache_seqlen`, `buffer_origin` and `seqlen_q` never
change, giving one compiled decode graph. A `valid_start` extension to the op
excludes left-pad columns the band cannot express. Phase 1 (full cache, no roll) is
built first purely as a correctness harness.

**Tech Stack:** PyTorch 2.13, `torch.compile(dynamic=False)` + Inductor,
torch-spyre custom ops (`torch.library.custom_op`), HF Transformers, pytest.

**Spec:** `docs/superpowers/specs/2026-08-18-gemma4-sliding-window-attention-design.md`

## Global Constraints

- **Two repos.** Tasks 1, 2, 11 are in `/mnt/home/spyre/torch-spyre` on branch
  `swa-3405-valid-start` (created off PR #3405's `swa-window-roll`). Tasks 3-10 are
  in `/mnt/home/spyre/hf-adapters` on branch `gemma4-swa-op`. Never mix a commit
  across repos.
- **Device setup, every shell that touches Spyre:**
  `source /mnt/home/spyre/torch-spyre-docs/scripts/dev-env.sh`. Plain
  `import torch` then registers the `spyre` device; do **not** set
  `TORCH_DEVICE_BACKEND_AUTOLOAD=0` and do **not** `import torch_spyre` separately.
  Set `HF_HOME=/tmp/models/hf_cache` and `PYTHONUNBUFFERED=1`.
- **Gemma 3 1B lives at** `/tmp/models/huggingface_cache/hub`, not
  `/tmp/models/hf_cache`. Gemma 4 12B is at `/tmp/models/hub/models--google--gemma-4-12b`.
- **`import regex`, never `import re`** in torch-spyre (pre-commit enforces it).
- **Sign off every commit** with `git commit -s` (DCO), in both repos.
- **Line length 88** (ruff), both repos.
- **Apache 2.0 header** — the 14-line Python header on every new file. Copy it from
  any neighbouring file.
- **Gemma 4 12B numbers:** `sliding_window=1024`, `head_dim=256` (sliding),
  `global_head_dim=512`, `num_attention_heads=16`, `num_key_value_heads=8`,
  40 of 48 layers `sliding_attention`, `attention_k_eq_v=True`,
  `Gemma4TextAttention.scaling == 1.0`.
- **Anchored constants for Gemma 4:** `capacity = buffer_width = 1088`,
  `cache_seqlen = 1088`, `buffer_origin = 0`, `seqlen_q = 64`, `delta = 1024`,
  `read_start = 0`. Gemma 3 1B: `window_size=512`, `capacity = 576`.
- **Never assert bit-exactness** against a masked reference: the windowed softmax
  reduces `buffer_width` terms where the reference reduces `seqlen_kv`, and `sum`
  blocks differently at different widths. Use `rtol=1e-2, atol=1e-3` for fp16
  device comparisons.
- **Perf claims require measurement.** The spec's figures are arithmetic. Do not
  repeat them as results.

## File Structure

| File | Repo | Responsibility |
|---|---|---|
| `torch_spyre/_inductor/sliding_window_plan.py` | torch-spyre | *modify* — add the torch-free `valid_start` helpers (`band_valid_start`, `band_batch`, `check_valid_start`). Stays free of torch and of the backend's error classes. |
| `torch_spyre/_inductor/customops.py` | torch-spyre | *modify* — `valid_start` parameter on `spyre::sliding_window_attention` and `spyre::window_band_mask`, plus both fakes. |
| `torch_spyre/_inductor/decompositions.py` | torch-spyre | *modify* — thread `valid_start` through `spyre_sliding_window_attention` and `_windowed_attention`; validate and raise `Unsupported`. |
| `tests/inductor/test_sliding_window_attention.py` | torch-spyre | *modify* — Gemma 4 shapes, anchored geometry, `valid_start` device tests. |
| `tests/inductor/test_kv_window.py` | torch-spyre | *modify* — pure-integer tests for the three new helpers. |
| `hf_adapters/swa_attention.py` | hf-adapters | **new** — the only file that touches `torch.ops.spyre.*`. Device dispatcher, CPU reference, `sliding_capacity`, and `SlidingWindowCache` (anchored bookkeeping: compaction, shift, `valid_start`). |
| `hf_adapters/hf_common.py` | hf-adapters | *modify* — `kv_cache_capacities`, `capacities=` on `allocate_kv_caches`, two lines in `generate`. |
| `hf_adapters/hf_gemma4.py` | hf-adapters | *modify* — `is_sliding` on `Gemma4Attention`, the SWA call path, mask suppression, shift/compaction in `_run_blocks_over_embeds`. |
| `hf_adapters/hf_gemma3.py` | hf-adapters | *modify* — the same replacement, as the green control. |
| `tests/cpu/test_swa_attention.py` | hf-adapters | **new** — CPU reference equivalence, `sliding_capacity`, dispatcher routing. |
| `tests/cpu/test_swa_cache.py` | hf-adapters | **new** — `SlidingWindowCache` arithmetic, compaction and shift on CPU tensors. |
| `tests/cpu/test_kv_cache_scatter.py` | hf-adapters | *modify* — per-layer capacity allocation. |
| `tests/_swa_helpers.py` | hf-adapters | **new** — builders shared by both test lanes: one `Gemma4Attention` at Gemma 4's shapes with random weights, identity RoPE freqs, and a fake model for layout-pinned allocation. |
| `tests/cpu/test_swa_gemma4_layer.py` | hf-adapters | **new** — phase-1 wiring: coordinates, mask suppression, KV write. |
| `tests/cpu/test_swa_anchored_cpu.py` | hf-adapters | **new** — 70 decode steps across a shift, anchored buffer versus full-cache band. |
| `tests/spyre/test_swa_layer_ab_spyre.py` | hf-adapters | **new** — the primary gate: band-masked SDPA vs the op, one Gemma 4 sliding block, on device. |

---

### Task 1: Gate — does the op work at Gemma 4's shapes?

Everything downstream depends on this. `head_dim=256` is untested in #3405 (its
widest case is `test_prefill_head_dim_128`), and `kv_window` returns a
**transposed** `[B, Hq, E, buffer_width]` slice, so `E=256` is the risk. If this
task fails, **stop and report** — do not start Task 2.

**Repo:** `/mnt/home/spyre/torch-spyre`

**Files:**
- Test: `tests/inductor/test_sliding_window_attention.py` (add to
  `TestSlidingWindowAttention` and `TestCompactCache`)

**Interfaces:**
- Consumes: PR #3405's existing test helpers in that file — `_inputs(batch, heads,
  kvheads, seqlen_q, seqlen_kv, head_dim=64)`, `_attention(q, k, v, window_size)`,
  `_compact_kv(batch, kvheads, capacity, cache_seqlen, head_dim=64)`,
  `_rolled_attention(q, k, v, window_size, cache_seqlen, buffer_origin=None)`, and
  `compare_with_cpu` / `cached_randn` from `utils_inductor`.
- Produces: nothing consumed by later tasks — this is a gate.

- [ ] **Step 1: Set up the environment and confirm the device registers**

```bash
source /mnt/home/spyre/torch-spyre-docs/scripts/dev-env.sh
cd /mnt/home/spyre/torch-spyre
python3 -c "import torch; x = torch.ones(4).to('spyre'); print((x + 1).to('cpu'))"
```

Expected: `tensor([2., 2., 2., 2.])`. If it raises
`Failed to load the backend extension: torch_spyre`, the library cascade has
regressed — re-run the `source` line and confirm `LD_LIBRARY_PATH` is populated.
Do not proceed until this prints.

- [ ] **Step 2: Check out PR #3405 and branch from it**

```bash
cd /mnt/home/spyre/torch-spyre
gh pr checkout 3405
git checkout -b swa-3405-valid-start
git log --oneline -1
```

Expected: HEAD is the tip of `swa-window-roll`. Confirm the op exists:

```bash
ls torch_spyre/_inductor/sliding_window_plan.py
```

- [ ] **Step 3: Write the two failing tests**

Add to `class TestSlidingWindowAttention`:

```python
    def test_prefill_head_dim_256_gqa(self):
        # Gemma 4's sliding layers: 16 query heads from 8 KV heads, head_dim 256,
        # W=1024. head_dim 256 is four sticks per row where the rest of this file
        # uses one or two, and kv_window hands back a transposed slice.
        query, key, value = _inputs(1, 16, 8, 512, 512, head_dim=256)
        compare_with_cpu(_attention, query, key, value, 1024, run_eager=False)
```

Add to `class TestCompactCache`:

```python
    def test_anchored_decode_stick_gemma4(self):
        # The exact geometry hf-adapters will call every decode step: a 1088-row
        # compact buffer declared exactly full, and a 64-row query stick. This is
        # the one shape that must be right for the integration to work at all.
        key, value = _compact_kv(1, 8, 1088, 1088, head_dim=256)
        query = cached_randn(
            (1, 16, 64, 256), differentiation=1, dtype=torch.float16
        )
        compare_with_cpu(
            _rolled_attention, query, key, value, 1024, 1088, run_eager=False
        )
```

- [ ] **Step 4: Run them**

```bash
cd /mnt/home/spyre/torch-spyre
SENCORES=1 python3 -m pytest tests/inductor/test_sliding_window_attention.py \
  -k "head_dim_256 or anchored" -v 2>&1 | tail -30
```

Expected: **PASS** — nothing is being implemented here, this is a probe of existing
code at new shapes. A failure is a finding, not a bug to fix in this task: capture
the full error, stop, and report which of the two shapes failed and how.

- [ ] **Step 5: Commit**

```bash
cd /mnt/home/spyre/torch-spyre
git add tests/inductor/test_sliding_window_attention.py
git commit -s -m "test(swa): cover Gemma 4's head_dim 256 and the anchored stick"
```

---

### Task 2: `valid_start` on the op

**Repo:** `/mnt/home/spyre/torch-spyre` (branch `swa-3405-valid-start`)

**Files:**
- Modify: `torch_spyre/_inductor/sliding_window_plan.py` (append three helpers)
- Modify: `torch_spyre/_inductor/customops.py` (`sliding_window_attention` + fake,
  `window_band_mask` + fake)
- Modify: `torch_spyre/_inductor/decompositions.py`
  (`spyre_sliding_window_attention`, `_windowed_attention`)
- Test: `tests/inductor/test_kv_window.py`,
  `tests/inductor/test_sliding_window_attention.py`

**Interfaces:**
- Consumes: Task 1's verified op.
- Produces:
  - `torch.ops.spyre.sliding_window_attention(query, key, value, window_size,
    is_causal=True, scale=None, cache_seqlen=None, buffer_origin=None,
    valid_start=None)` where `valid_start: Optional[list[int]]` has one **logical**
    column coordinate per batch entry; columns strictly before it are never
    attended. With `buffer_origin=0` (what hf-adapters passes) that coordinate is
    simply the physical row index.
  - `sliding_window_plan.band_valid_start(valid_start) -> list[int] | None`
  - `sliding_window_plan.band_batch(valid_start) -> int`
  - `sliding_window_plan.check_valid_start(valid_start, batch, cache_seqlen) -> str | None`

- [ ] **Step 1: Write the failing pure-integer tests**

In `tests/inductor/test_kv_window.py`, extend the import from
`torch_spyre._inductor.sliding_window_plan` with `band_batch`, `band_valid_start`
and `check_valid_start`, then add:

```python
class TestValidStart:
    """The three torch-free helpers behind the valid_start band."""

    def test_none_and_all_zero_mask_nothing(self):
        # A caller with no padding must not pay for a per-batch band.
        assert band_valid_start(None) is None
        assert band_valid_start([]) is None
        assert band_valid_start([0, 0]) is None
        assert band_batch([0, 0]) == 1

    def test_uniform_nonzero_stays_broadcast(self):
        # Same threshold for every sequence: one row of band, broadcast over batch.
        assert band_valid_start([40, 40]) == [40, 40]
        assert band_batch([40, 40]) == 1

    def test_ragged_widens_to_batch(self):
        assert band_batch([0, 40]) == 2
        assert band_batch([40, 40, 7]) == 3

    def test_check_rejects_wrong_length(self):
        assert "2 entries" in check_valid_start([0, 40], 3, 1088)

    def test_check_rejects_out_of_range(self):
        assert "outside" in check_valid_start([-1], 1, 1088)
        assert "outside" in check_valid_start([1089], 1, 1088)

    def test_check_accepts_valid(self):
        assert check_valid_start(None, 4, 1088) is None
        assert check_valid_start([0, 40], 2, 1088) is None
        assert check_valid_start([1088], 1, 1088) is None
```

- [ ] **Step 2: Run them to verify they fail**

```bash
cd /mnt/home/spyre/torch-spyre
python3 -m pytest tests/inductor/test_kv_window.py -k ValidStart -v 2>&1 | tail -15
```

Expected: collection error — `ImportError: cannot import name 'band_batch'`.

- [ ] **Step 3: Implement the three helpers**

Append to `torch_spyre/_inductor/sliding_window_plan.py`:

```python
def band_valid_start(valid_start: list[int] | None) -> list[int] | None:
    """The ``valid_start`` the band must actually apply, or None when it masks nothing.

    A caller with no left padding passes zeros rather than tracking whether it has
    any; collapsing that to None here is what keeps the band broadcast over batch
    and keeps ``block_is_fully_attended``'s skip available.
    """
    if not valid_start or max(valid_start) <= 0:
        return None
    return list(valid_start)


def band_batch(valid_start: list[int] | None) -> int:
    """Leading dimension of the band: 1 unless the threshold differs per sequence.

    The band is ``q_block x buffer_width`` per distinct threshold, so a uniform one
    stays a single broadcast row rather than one copy per batch entry.
    """
    effective = band_valid_start(valid_start)
    if effective is None or min(effective) == max(effective):
        return 1
    return len(effective)


def check_valid_start(
    valid_start: list[int] | None, batch: int, cache_seqlen: int
) -> str | None:
    """Why this ``valid_start`` is invalid, or None.

    Strings not exceptions, matching ``check_window_read`` -- this module stays
    free of torch and of the backend's error classes. Batch is a tensor property,
    so this cannot live in ``rejection_reason``, which answers placement questions
    from integers alone.
    """
    if valid_start is None:
        return None
    if len(valid_start) != batch:
        return (
            f"valid_start has {len(valid_start)} entries for a batch of {batch}"
        )
    for entry in valid_start:
        if entry < 0 or entry > cache_seqlen:
            return (
                f"valid_start={entry} outside [0, cache_seqlen={cache_seqlen}]"
            )
    return None
```

- [ ] **Step 4: Run the pure tests to verify they pass**

```bash
cd /mnt/home/spyre/torch-spyre
python3 -m pytest tests/inductor/test_kv_window.py -k ValidStart -v 2>&1 | tail -15
```

Expected: 6 passed.

- [ ] **Step 5: Commit the helpers**

```bash
cd /mnt/home/spyre/torch-spyre
git add torch_spyre/_inductor/sliding_window_plan.py tests/inductor/test_kv_window.py
git commit -s -m "feat(swa): torch-free valid_start helpers for the window band"
```

- [ ] **Step 6: Write the failing device test**

In `tests/inductor/test_sliding_window_attention.py`, add a CPU reference helper
next to `_rolled_reference`:

```python
def _reference_with_valid_start(query, key, value, window_size, valid_start):
    """``_rolled_reference`` for an exactly-full buffer, additionally excluding
    physical rows below ``valid_start`` -- the left-padding an offset-and-length
    window cannot express.

    One threshold per batch entry, so ``valid_start`` is a list even when uniform.
    """
    seqlen_q, capacity = query.size(2), key.size(2)
    rows = torch.arange(seqlen_q) + (capacity - seqlen_q)
    columns = torch.arange(capacity)
    delta = rows.unsqueeze(-1) - columns.unsqueeze(0)
    allowed = (delta >= 0) & (delta < window_size)
    starts = torch.tensor(valid_start).view(-1, 1, 1)
    allowed = allowed.unsqueeze(0) & (columns.view(1, 1, -1) >= starts)
    mask = torch.zeros(allowed.shape, dtype=query.dtype)
    mask.masked_fill_(~allowed, float("-inf"))
    return F.scaled_dot_product_attention(
        query, key, value, attn_mask=mask.unsqueeze(1)
    )


def _valid_start_attention(q, k, v, window_size, valid_start):
    """The op with an explicit valid_start on spyre, the reference on CPU."""
    if q.device.type == "spyre":
        return torch.ops.spyre.sliding_window_attention(
            q, k, v, window_size, True, None, k.size(2), 0, valid_start
        )
    return _reference_with_valid_start(q, k, v, window_size, valid_start)
```

Add to `class TestCompactCache`:

```python
    def test_valid_start_excludes_padded_columns(self):
        # 17 rows of left padding inside the window: without valid_start they are
        # attended, and the reference proves the difference is visible.
        key, value = _compact_kv(1, 8, 1088, 1088)
        query = cached_randn((1, 8, 64, 64), differentiation=1, dtype=torch.float16)
        compare_with_cpu(
            _valid_start_attention, query, key, value, 1024, [17], run_eager=False
        )

    def test_valid_start_per_sequence(self):
        # Ragged batch: the band widens to [B, 1, q, W'] only for this case.
        key, value = _compact_kv(2, 8, 1088, 1088)
        query = cached_randn((2, 8, 64, 64), differentiation=1, dtype=torch.float16)
        compare_with_cpu(
            _valid_start_attention, query, key, value, 1024, [0, 40], run_eager=False
        )

    def test_all_zero_valid_start_matches_no_valid_start(self):
        # The fast path must be numerically identical to passing nothing.
        key, value = _compact_kv(1, 8, 1088, 1088)
        query = cached_randn((1, 8, 64, 64), differentiation=1, dtype=torch.float16)

        def attention(q, k, v, window_size):
            if q.device.type == "spyre":
                return torch.ops.spyre.sliding_window_attention(
                    q, k, v, window_size, True, None, k.size(2), 0, [0]
                )
            return _rolled_reference(q, k, v, window_size, k.size(2))

        compare_with_cpu(attention, query, key, value, 1024, run_eager=False)
```

- [ ] **Step 7: Run it to verify it fails**

```bash
cd /mnt/home/spyre/torch-spyre
SENCORES=1 python3 -m pytest tests/inductor/test_sliding_window_attention.py \
  -k valid_start -v 2>&1 | tail -20
```

Expected: FAIL — the op takes 8 positional arguments, not 9.

- [ ] **Step 8: Add `valid_start` to both custom ops**

In `torch_spyre/_inductor/customops.py`, add the parameter to
`sliding_window_attention` and its `register_fake` (the fake ignores it — the
output shape is `query.size()` either way):

```python
def sliding_window_attention(  # type: ignore[empty-body]
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    window_size: int,
    is_causal: bool = True,
    scale: Optional[float] = None,
    cache_seqlen: Optional[int] = None,
    buffer_origin: Optional[int] = None,
    valid_start: Optional[list[int]] = None,
) -> torch.Tensor:
```

and document it in that docstring, after the ``buffer_origin`` paragraph:

```
    ``valid_start`` is one **logical** column coordinate per batch entry;
    columns strictly below it are never attended, whatever the window says. It
    exists for left-padded prompts, whose pad columns sit inside the window and
    which an offset-and-length window cannot otherwise exclude. ``None`` or
    all-zero costs nothing; a uniform threshold keeps the band broadcast over
    batch; only a ragged one widens it. Coordinates are logical, matching
    ``cache_seqlen``, so a caller passing ``buffer_origin=0`` passes physical
    row indices.
```

Then `window_band_mask` — add the parameter and replace the body's tail:

```python
def window_band_mask(
    read_start: int,
    q_block: int,
    buffer_width: int,
    q_row_origin: int,
    window_size: int,
    is_causal: bool,
    dtype: torch.dtype,
    device: torch.device,
    valid_start: Optional[list[int]] = None,
) -> torch.Tensor:
```

```python
    row = torch.arange(q_block, device="cpu") + q_row_origin
    column = torch.arange(buffer_width, device="cpu") + read_start
    delta = row.unsqueeze(-1) - column.unsqueeze(0)
    if is_causal:
        allowed = (delta >= 0) & (delta < window_size)
    else:
        allowed = delta.abs() < window_size
    effective = band_valid_start(valid_start)
    if effective is None:
        allowed = allowed.unsqueeze(0)
    elif min(effective) == max(effective):
        # Uniform threshold: still one broadcast row, not one per sequence.
        allowed = (allowed & (column.unsqueeze(0) >= effective[0])).unsqueeze(0)
    else:
        starts = torch.tensor(effective, device="cpu").view(-1, 1, 1)
        allowed = allowed.unsqueeze(0) & (column.view(1, 1, -1) >= starts)
    mask_cpu = torch.zeros(allowed.shape, dtype=dtype, device="cpu")
    mask_cpu.masked_fill_(~allowed, float("-inf"))
    return mask_cpu.unsqueeze(1).to(device=device)
```

Import the helper at the top of the module, beside the other
`sliding_window_plan` imports:

```python
from .sliding_window_plan import band_batch, band_valid_start
```

In that docstring, replace the `Shape:` line's
`[1, 1, q_block, buffer_width]` with a note that the leading axis is 1 unless
`valid_start` differs across the batch, in which case it is `B`. Then make
`register_fake` return the same shape the real op does:

```python
@window_band_mask.register_fake
def _(
    read_start: int,
    q_block: int,
    buffer_width: int,
    q_row_origin: int,
    window_size: int,
    is_causal: bool,
    dtype: torch.dtype,
    device: torch.device,
    valid_start: Optional[list[int]] = None,
) -> torch.Tensor:
    return torch.empty(
        band_batch(valid_start), 1, q_block, buffer_width, dtype=dtype, device=device
    )
```

- [ ] **Step 9: Thread it through the decomposition**

In `torch_spyre/_inductor/decompositions.py`, extend the import from
`.sliding_window_plan` with `band_valid_start` and `check_valid_start`. Give
`_windowed_attention` the extra parameter and use it in two places:

```python
def _windowed_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    plan: SlidingWindowPlan,
    scaling_factor: float,
    num_heads: int,
    valid_start: list[int] | None = None,
) -> torch.Tensor:
```

```python
        read_start = plan.read_start(block_index)
        k_win, v_win = torch.ops.spyre.kv_window(
            key, value, read_start, buffer_width, num_heads
        )
        # A valid_start that masks anything makes the band load-bearing even for a
        # block the window alone fully covers.
        fully_attended = (
            plan.block_is_fully_attended(block_index)
            and band_valid_start(valid_start) is None
        )
        band = (
            None
            if fully_attended
            else torch.ops.spyre.window_band_mask(
                plan.read_start_logical(block_index),
                q_block,
                buffer_width,
                plan.q_kv_offset + q_start,
                plan.window_size,
                plan.is_causal,
                query.dtype,
                query.device,
                valid_start,
            )
        )
```

In `spyre_sliding_window_attention`, add the parameter, validate it immediately
after `cache_seqlen` defaults, and pass it down:

```python
def spyre_sliding_window_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    window_size: int,
    is_causal: bool = True,
    scale: float | None = None,
    cache_seqlen: int | None = None,
    buffer_origin: int | None = None,
    valid_start: list[int] | None = None,
) -> torch.Tensor:
```

```python
    if cache_seqlen is None:
        cache_seqlen = cache_capacity

    reason = check_valid_start(valid_start, batch_size, cache_seqlen)
    if reason is not None:
        raise Unsupported(f"sliding_window_attention: {reason}")
```

```python
    output = _windowed_attention(
        query, key, value, plan, scaling_factor, num_heads, valid_start
    )
```

- [ ] **Step 10: Run the device tests to verify they pass**

```bash
cd /mnt/home/spyre/torch-spyre
SENCORES=1 python3 -m pytest tests/inductor/test_sliding_window_attention.py \
  -k valid_start -v 2>&1 | tail -20
```

Expected: 3 passed.

- [ ] **Step 11: Run the whole SWA suite for regressions**

The `unsqueeze(0)` restructure touches the no-`valid_start` path too, so the
existing tests are the regression gate.

```bash
cd /mnt/home/spyre/torch-spyre
SENCORES=1 python3 -m pytest tests/inductor/test_sliding_window_attention.py \
  tests/inductor/test_kv_window.py -v 2>&1 | tail -20
```

Expected: all pass, no new failures versus Task 1's baseline.

- [ ] **Step 12: Lint and commit**

```bash
cd /mnt/home/spyre/torch-spyre
pre-commit run --files torch_spyre/_inductor/customops.py \
  torch_spyre/_inductor/decompositions.py \
  torch_spyre/_inductor/sliding_window_plan.py \
  tests/inductor/test_sliding_window_attention.py
git add -u
git commit -s -m "feat(swa): valid_start to exclude left-pad columns from the window"
```

---

### Task 3: the dispatcher and its CPU reference

**Repo:** `/mnt/home/spyre/hf-adapters` (branch `gemma4-swa-op`) — and every task
from here on.

`spyre::sliding_window_attention` is registered for the `spyre` device only and its
eager body is empty, so the CPU test lane needs the definition computed literally.
That reference is also the specification: if it does not equal what
`add_causal_sliding_window_band` + SDPA computes today, the integration is wrong
before any device is involved.

**Files:**
- Create: `hf_adapters/swa_attention.py`
- Test: `tests/cpu/test_swa_attention.py`

**Interfaces:**
- Consumes: `hf_common.BLOCK_SIZE` (64), `hf_common.add_causal_sliding_window_band`
  and `hf_common.build_prefill_mask` (in the test only). Task 2's `valid_start`
  parameter.
- Produces:
  - `swa_attention.sliding_capacity(window_size, q_block=BLOCK_SIZE) -> int`
  - `swa_attention.sliding_window_attention(query, key_cache, value_cache, *,
    window_size, scale, cache_seqlen, buffer_origin=0, valid_start=None) -> Tensor`

- [ ] **Step 1: Write the failing tests**

Create `tests/cpu/test_swa_attention.py` with the 14-line Apache header, then:

```python
"""CPU tests for the sliding-window attention dispatcher.

``spyre::sliding_window_attention`` exists only on the spyre device, so this lane
exercises the CPU reference branch — and pins it to the band-masked SDPA the Gemma
adapters compute today, which is the equivalence the whole replacement rests on.
"""

import torch
import torch.nn.functional as F

from hf_adapters.hf_common import add_causal_sliding_window_band, build_prefill_mask
from hf_adapters.swa_attention import sliding_capacity, sliding_window_attention


def _band_masked_attention(query, key_cache, value_cache, window_size, offset):
    """What the Gemma adapters do today: causal + left-pad mask, then a band."""
    batch, _, seqlen_q, _ = query.shape
    capacity = key_cache.size(2)
    mask = build_prefill_mask(
        batch, seqlen_q, capacity, offset, dtype=query.dtype
    )
    coords = torch.arange(seqlen_q)[None, :].expand(batch, seqlen_q)
    mask = add_causal_sliding_window_band(mask, coords, window_size)
    return F.scaled_dot_product_attention(
        query, key_cache, value_cache, attn_mask=mask, enable_gqa=True
    )


def _inputs(batch=1, q_heads=4, kv_heads=2, seqlen_q=128, capacity=256, head_dim=32):
    """Query plus a full-length cache whose rows past ``seqlen_q`` stay zero."""
    torch.manual_seed(0)
    query = torch.randn(batch, q_heads, seqlen_q, head_dim)
    key_cache = torch.zeros(batch, kv_heads, capacity, head_dim)
    value_cache = torch.zeros(batch, kv_heads, capacity, head_dim)
    key_cache[:, :, :seqlen_q, :] = torch.randn(batch, kv_heads, seqlen_q, head_dim)
    value_cache[:, :, :seqlen_q, :] = torch.randn(batch, kv_heads, seqlen_q, head_dim)
    return query, key_cache, value_cache


def test_sliding_capacity_is_window_plus_one_stick():
    # Gemma 4 and Gemma 3, the two models this lands for.
    assert sliding_capacity(1024) == 1088
    assert sliding_capacity(512) == 576


def test_sliding_capacity_rounds_up_to_a_stick():
    # A window that is not a whole number of sticks still gets a stick-aligned
    # allocation: rejection_reason refuses a capacity that is not.
    assert sliding_capacity(100) == 192
    assert sliding_capacity(1) == 128  # one row of window plus a 64-row stick
    assert sliding_capacity(100) % 64 == 0


def test_reference_matches_the_band_masked_path():
    """The equivalence the replacement rests on, at phase-1 geometry."""
    query, key_cache, value_cache = _inputs()
    expected = _band_masked_attention(query, key_cache, value_cache, 64, 0)
    actual = sliding_window_attention(
        query,
        key_cache,
        value_cache,
        window_size=64,
        scale=None,
        cache_seqlen=128,
    )
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_reference_honors_valid_start_like_left_padding():
    """valid_start must reproduce build_prefill_mask's left-pad columns.

    Compared only over rows ``>= 17``. A row below the threshold has its ENTIRE
    window excluded — row ``i``'s window is ``(i - 64, i]``, and every one of those
    columns is below 17 — so it is a fully-masked row, and the two paths legitimately
    disagree there: this reference fills uniformly with ``-inf`` and so spreads weight
    over every column, while the band path's mask mixes ``_mask_fill_value`` (on the
    pad columns) with ``-inf`` (out of band) and so spreads weight over the pad
    columns only. Neither answer means anything — those query rows ARE padding, and
    their outputs are discarded — so the assertion covers the rows whose attention is
    defined, plus finiteness everywhere, which is the property that actually matters
    (a non-finite value would travel into the next layer's KV cache).
    """
    query, key_cache, value_cache = _inputs()
    expected = _band_masked_attention(query, key_cache, value_cache, 64, 17)
    actual = sliding_window_attention(
        query,
        key_cache,
        value_cache,
        window_size=64,
        scale=None,
        cache_seqlen=128,
        valid_start=[17],
    )
    torch.testing.assert_close(
        actual[:, :, 17:], expected[:, :, 17:], rtol=1e-5, atol=1e-6
    )
    assert torch.isfinite(actual).all(), "fully-masked rows must stay finite"


def test_reference_honors_per_sequence_valid_start():
    """A ragged batch: each entry gets its own threshold.

    Per-entry row slicing for the same reason as the test above: entry 1's threshold
    of 40 makes its rows ``[0, 40)`` fully masked, while entry 0's threshold of 0
    makes none of its rows fully masked.
    """
    query, key_cache, value_cache = _inputs(batch=2)
    offsets = (0, 40)
    per_entry = [
        _band_masked_attention(
            query[b : b + 1], key_cache[b : b + 1], value_cache[b : b + 1], 64, offset
        )
        for b, offset in enumerate(offsets)
    ]
    expected = torch.cat(per_entry, dim=0)
    actual = sliding_window_attention(
        query,
        key_cache,
        value_cache,
        window_size=64,
        scale=None,
        cache_seqlen=128,
        valid_start=list(offsets),
    )
    for b, offset in enumerate(offsets):
        torch.testing.assert_close(
            actual[b, :, offset:], expected[b, :, offset:], rtol=1e-5, atol=1e-6
        )
    assert torch.isfinite(actual).all(), "fully-masked rows must stay finite"


def test_explicit_scale_is_honored():
    """Gemma 4 attends unscaled (scaling == 1.0), so scale must reach SDPA."""
    query, key_cache, value_cache = _inputs()
    unscaled = sliding_window_attention(
        query, key_cache, value_cache, window_size=64, scale=1.0, cache_seqlen=128
    )
    default = sliding_window_attention(
        query, key_cache, value_cache, window_size=64, scale=None, cache_seqlen=128
    )
    assert not torch.allclose(unscaled, default)
```

- [ ] **Step 2: Run them to verify they fail**

```bash
cd /mnt/home/spyre/hf-adapters
python3 -m pytest tests/cpu/test_swa_attention.py -v 2>&1 | tail -10
```

Expected: collection error — `ModuleNotFoundError: No module named
'hf_adapters.swa_attention'`.

- [ ] **Step 3: Create the module**

`hf_adapters/swa_attention.py`, with the 14-line Apache header, then:

```python
"""Sliding-window attention for the Spyre adapters.

The only module in hf-adapters that calls ``torch.ops.spyre.*``. Gemma 3 and
Gemma 4 alternate sliding and full-attention layers; the sliding ones can read
just their window out of a compact KV buffer instead of scoring the whole cache
behind a band mask (torch-spyre#3405).

Two things make that a module rather than a call site:

1. ``spyre::sliding_window_attention`` is registered for the spyre device only and
   has an empty eager body, so CPU needs the definition computed literally.
2. The op takes its geometry as trace-time integers, so which integers a caller
   passes decides how many binaries it compiles. ``SlidingWindowCache`` exists to
   keep them constant.
"""

import torch
import torch.nn.functional as F

from hf_adapters.hf_common import BLOCK_SIZE


def sliding_capacity(window_size, q_block=BLOCK_SIZE):
    """Rows an anchored compact KV buffer allocates for ``window_size``.

    ``window_size + q_block`` rounded up to a stick: a block of ``q_block``
    query rows has staggered windows spanning ``window_size + q_block - 1``
    columns, and ``rejection_reason`` refuses a capacity that is not
    stick-aligned. 1088 for Gemma 4 (W=1024), 576 for Gemma 3 (W=512).
    """
    return -(-(window_size + q_block) // BLOCK_SIZE) * BLOCK_SIZE


def sliding_window_attention(
    query,
    key_cache,
    value_cache,
    *,
    window_size,
    scale,
    cache_seqlen,
    buffer_origin=0,
    valid_start=None,
):
    """Attend ``query`` against the window of a KV cache.

    Args:
        query: ``[B, Hq, Lq, D]``.
        key_cache, value_cache: ``[B, Hkv, capacity, D]``. GQA is expanded inside
            the op, so ``Hq`` need only be a whole multiple of ``Hkv``. Both must
            have the same shape (``check_window_read`` requires it) and the
            allocation must be **zero-filled**: a window may overshoot the written
            prefix, and an additive mask cannot rescue a NaN.
        window_size: keys per query, exclusive lower bound — row at coordinate
            ``c`` attends ``(c - window_size, c]``.
        scale: ``Q·Kᵀ`` multiplier. ``None`` means ``1/sqrt(D)``. Gemma 4 attends
            **unscaled** and must pass ``1.0``.
        cache_seqlen: tokens the cache has seen, as distinct from its allocated
            rows. Query row ``i`` sits at coordinate ``cache_seqlen - Lq + i``.
        buffer_origin: logical position held by physical row 0. Callers keeping a
            buffer-relative view (see ``SlidingWindowCache``) pass 0.
        valid_start: one logical column per batch entry, below which nothing is
            attended — left padding, which an offset-and-length window cannot
            express. ``None`` or all-zero costs nothing.

    Returns ``[B, Hq, Lq, D]``.
    """
    if query.device.type == "spyre":
        return torch.ops.spyre.sliding_window_attention(
            query,
            key_cache,
            value_cache,
            window_size,
            True,
            scale,
            cache_seqlen,
            buffer_origin,
            valid_start,
        )
    return _reference_attention(
        query,
        key_cache,
        value_cache,
        window_size,
        scale,
        cache_seqlen,
        buffer_origin,
        valid_start,
    )


def _reference_attention(
    query,
    key_cache,
    value_cache,
    window_size,
    scale,
    cache_seqlen,
    buffer_origin,
    valid_start,
):
    """The op's definition as a masked SDPA, for the CPU lane.

    Query row ``i`` is at logical coordinate ``cache_seqlen - Lq + i``; physical
    row ``j`` holds ``buffer_origin + j``. A row attends a column iff their gap is
    in ``[0, window_size)`` and the column is not below ``valid_start``.

    ``-inf`` rather than ``hf_common._mask_fill_value``: this branch runs on CPU
    only, where the dlfloat16 saturation that motivates the finite fill does not
    apply.
    """
    seqlen_q = query.size(2)
    capacity = key_cache.size(2)
    rows = torch.arange(seqlen_q, device=query.device) + (cache_seqlen - seqlen_q)
    columns = torch.arange(capacity, device=query.device) + buffer_origin
    delta = rows.unsqueeze(-1) - columns.unsqueeze(0)
    allowed = (delta >= 0) & (delta < window_size)
    if valid_start is not None and max(valid_start) > 0:
        starts = torch.tensor(valid_start, device=query.device).view(-1, 1, 1)
        allowed = allowed.unsqueeze(0) & (columns.view(1, 1, -1) >= starts)
    else:
        allowed = allowed.unsqueeze(0)
    mask = torch.zeros(allowed.shape, dtype=query.dtype, device=query.device)
    mask.masked_fill_(~allowed, float("-inf"))
    return F.scaled_dot_product_attention(
        query,
        key_cache,
        value_cache,
        attn_mask=mask.unsqueeze(1),
        dropout_p=0.0,
        scale=scale,
        enable_gqa=True,
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
cd /mnt/home/spyre/hf-adapters
python3 -m pytest tests/cpu/test_swa_attention.py -v 2>&1 | tail -12
```

Expected: 6 passed. If `test_reference_matches_the_band_masked_path` fails, the
coordinate convention is wrong — check that `build_prefill_mask` masks cache
columns above the query row (it does, via `mask[:, :, i, i + 1:]`), which is the
same thing `delta >= 0` says.

- [ ] **Step 5: Commit**

```bash
cd /mnt/home/spyre/hf-adapters
git add hf_adapters/swa_attention.py tests/cpu/test_swa_attention.py
git commit -s -m "feat(swa): sliding-window attention dispatcher with a CPU reference"
```

---

### Task 4: per-layer KV cache capacity

Sliding layers must not be allocated at `prompt + generation` and then compacted —
for a 8192-token context that is roughly 2.7 GB allocated and freed across 40
layers. `allocate_kv_caches` currently takes one `max_cache_len` for every layer.

**Files:**
- Modify: `hf_adapters/hf_common.py` (`allocate_kv_caches` around line 1203; new
  `kv_cache_capacities` beside `kv_cache_shapes` at line 1092; `generate` around
  line 1665)
- Test: `tests/cpu/test_kv_cache_scatter.py` (append)

**Interfaces:**
- Consumes: `kv_cache_shapes(model)`, `model._spyre_kv_shapes`.
- Produces:
  - `hf_common.kv_cache_capacities(model, padded_prompt_len, max_cache_len) -> list[int]`
  - `hf_common.allocate_kv_caches(model, batch_size, max_cache_len, dtype,
    device=None, capacities=None)` — `capacities[i]` overrides `max_cache_len` for
    layer `i`.
  - The model hook: `model._spyre_kv_capacity(layer_index, padded_prompt_len,
    max_cache_len) -> int`, a plain module-level function (not a lambda, so a model
    stays picklable).

- [ ] **Step 0: Repair the `_FakeModel` fixture first**

`kv_cache_shapes` gained a tensor-parallel head-count probe in `6f1cbc4`
("TP enablement", #234) that reads
`get_backbone(model).layers[0].self_attn`. `get_backbone` falls through to the
model itself for a stub object, so every `_FakeModel` without an explicit
`_spyre_kv_shapes` now raises `AttributeError: 'NoneType' object has no attribute
'layers'` — which already breaks the pre-existing
`test_allocate_keeps_attention_shapes`, and would break the tests below. Give the
fixture the one attribute that probe needs, in `tests/cpu/test_kv_cache_scatter.py`:

```python
class _FakeModel:
    """Minimal stand-in for the parts of a model allocate_kv_caches reads."""

    def __init__(self, num_layers=2, num_kv_heads=8, head_dim=128, kv_shapes=None):
        self.config = _FakeConfig(num_layers, num_kv_heads, head_dim)
        # kv_cache_shapes probes layers[0].self_attn.k_proj to recover the KV-head
        # count under tensor parallelism (hf_common, added in #234). No k_proj here,
        # so it falls back to the config count -- which is what these tests want.
        self.layers = [types.SimpleNamespace(self_attn=types.SimpleNamespace())]
        if kv_shapes is not None:
            self._spyre_kv_shapes = kv_shapes
```

and add `import types` to that file's imports.

Run `python3 -m pytest tests/cpu/test_kv_cache_scatter.py -q` and confirm the
pre-existing failure is gone before writing anything new.

- [ ] **Step 1: Write the failing tests**

Append to `tests/cpu/test_kv_cache_scatter.py`:

```python
def test_capacities_default_to_max_cache_len():
    """Without a model hook, every layer keeps the single length."""
    from hf_adapters.hf_common import kv_cache_capacities

    model = _FakeModel(num_layers=3)
    assert kv_cache_capacities(model, 128, 576) == [576, 576, 576]


def test_capacities_honor_the_model_hook():
    """Gemma 4 style: sliding layers want a compact buffer, global ones do not."""
    from hf_adapters.hf_common import kv_cache_capacities

    def capacity(layer_index, padded_prompt_len, max_cache_len):
        return 1088 if layer_index % 2 == 0 else max_cache_len

    model = _FakeModel(num_layers=4)
    model._spyre_kv_capacity = capacity
    assert kv_cache_capacities(model, 512, 576) == [1088, 576, 1088, 576]


def test_allocate_uses_per_layer_capacities():
    """The allocation, not just the arithmetic."""
    model = _FakeModel(num_layers=3, num_kv_heads=8, head_dim=128)
    keys, values = allocate_kv_caches(
        model, 2, 576, torch.float32, device="cpu", capacities=[1088, 576, 1088]
    )
    assert [k.shape[2] for k in keys] == [1088, 576, 1088]
    assert [v.shape[2] for v in values] == [1088, 576, 1088]
    for k, v in zip(keys, values):
        assert not k.any() and not v.any(), "caches must start zeroed"


def test_allocate_rejects_a_capacity_per_layer_mismatch():
    """A short list is a silent wrong-size cache; refuse it."""
    model = _FakeModel(num_layers=3)
    with pytest.raises(ValueError, match="capacities"):
        allocate_kv_caches(
            model, 1, 576, torch.float32, device="cpu", capacities=[1088]
        )
```

- [ ] **Step 2: Run them to verify they fail**

```bash
cd /mnt/home/spyre/hf-adapters
python3 -m pytest tests/cpu/test_kv_cache_scatter.py -k "capacit" -v 2>&1 | tail -12
```

Expected: `ImportError: cannot import name 'kv_cache_capacities'` and
`TypeError: allocate_kv_caches() got an unexpected keyword argument 'capacities'`.

- [ ] **Step 3: Add `kv_cache_capacities`**

In `hf_adapters/hf_common.py`, directly after `kv_cache_shapes`:

```python
def kv_cache_capacities(model, padded_prompt_len, max_cache_len):
    """Resolve the per-layer KV-cache row count.

    Most models allocate ``max_cache_len`` rows for every layer. A model whose
    layers need different lengths — Gemma 4's sliding layers keep a compact
    ``window + 64`` buffer while its global layers hold the whole generation — sets
    ``model._spyre_kv_capacity(layer_index, padded_prompt_len, max_cache_len)``.

    ``padded_prompt_len`` is passed because a sliding layer's prefill buffer is
    sized by the prompt rather than by the generation: it is compacted at the
    prefill/decode boundary and never sees the generated tail.

    Returns a list of length ``num_hidden_layers``.
    """
    num_layers = len(kv_cache_shapes(model))
    hook = getattr(model, "_spyre_kv_capacity", None)
    if hook is None:
        return [max_cache_len] * num_layers
    return [hook(i, padded_prompt_len, max_cache_len) for i in range(num_layers)]
```

- [ ] **Step 4: Add `capacities` to `allocate_kv_caches`**

Change the signature and the two allocation lines. The rest of the function, and
the layout pin, are untouched:

```python
def allocate_kv_caches(
    model, batch_size, max_cache_len, dtype, device=None, capacities=None
):
```

Add to that docstring, after the ``_spyre_kv_shapes`` paragraph:

```
    ``capacities`` overrides ``max_cache_len`` per layer (see
    ``kv_cache_capacities``); it must have one entry per layer. Every entry is
    still allocated with the same pinned device layout, so a compact cache
    scatters correctly like any other.
```

and replace the body's tail:

```python
    shapes = kv_cache_shapes(model)
    if capacities is None:
        capacities = [max_cache_len] * len(shapes)
    elif len(capacities) != len(shapes):
        raise ValueError(
            f"capacities has {len(capacities)} entries for {len(shapes)} layers"
        )
    on_spyre = torch.device(device).type == "spyre"

    def _alloc(n_kv, head_dim, rows):
        stl = (
            _cache_position_first_stl(batch_size, n_kv, rows, head_dim, dtype)
            if on_spyre
            else None
        )
        shape = (batch_size, n_kv, rows, head_dim)
        if stl is None:
            return torch.zeros(shape, dtype=dtype, device=device)
        cache: torch.Tensor = torch.empty(  # type: ignore[call-overload]
            shape,
            device=torch.device(device),
            device_layout=stl,
            dtype=dtype,
        )
        cache.zero_()
        return cache

    key_caches = [
        _alloc(n_kv, hd, rows)
        for (n_kv, hd, _vhd), rows in zip(shapes, capacities)
    ]
    value_caches = [
        _alloc(n_kv, vhd, rows)
        for (n_kv, _hd, vhd), rows in zip(shapes, capacities)
    ]
    return key_caches, value_caches
```

- [ ] **Step 5: Wire it into `generate`**

In `generate`, replace the `allocate_kv_caches` call (it currently passes four
arguments) with:

```python
    key_caches, value_caches = allocate_kv_caches(
        model,
        batch_size,
        max_cache_len,
        model_d_type,
        capacities=kv_cache_capacities(model, padded_len, max_cache_len),
    )
```

`padded_len` is already in scope from `pad_and_position` on the line above.

- [ ] **Step 6: Run the tests to verify they pass**

```bash
cd /mnt/home/spyre/hf-adapters
python3 -m pytest tests/cpu/test_kv_cache_scatter.py -v 2>&1 | tail -12
```

Expected: all pass, including the four pre-existing allocation tests — the
`_alloc` signature changed, so they are the regression gate.

- [ ] **Step 7: Run the CPU generate suite for regressions**

```bash
cd /mnt/home/spyre/hf-adapters
python3 -m pytest tests/cpu/test_generate_cpu.py -x -q 2>&1 | tail -12
```

Expected: no new failures. `generate` now calls a new function on every path.

- [ ] **Step 8: Commit**

```bash
cd /mnt/home/spyre/hf-adapters
git add hf_adapters/hf_common.py tests/cpu/test_kv_cache_scatter.py
git commit -s -m "feat(kv-cache): allow per-layer cache capacities"
```

---

### Task 5: phase 1 — wire the op into Gemma 4's sliding layers

The correctness harness: the existing full-length cache, `buffer_origin=0`, and a
`cache_seqlen` that grows. It recompiles once per 64 decode steps without bound and
shrinks nothing, so it is **opt-in only** (`model._spyre_swa_mode = "phase1"`) and
never becomes a default. Its job is to isolate op-vs-band numerics with one variable
changed.

**Files:**
- Modify: `hf_adapters/hf_gemma4.py` (`Gemma4Attention` at line 137,
  `Gemma4Block` at line 216, `prepare_gemma4_blocks` at line 268,
  `_build_layer_masks` at line 292, `_run_blocks_over_embeds` at line 322,
  `prepare_text_decoder_for_spyre` at line 444)
- Create: `tests/_swa_helpers.py`
- Test: `tests/cpu/test_swa_gemma4_layer.py`

**Interfaces:**
- Consumes: `swa_attention.sliding_window_attention` (Task 3);
  `model._spyre_prompt_offsets` (Task 4).
- Produces:
  - `Gemma4Attention(attn, num_q_heads, num_kv_heads, head_dim, is_kv_eq_v,
    is_sliding=False, window_size=None, swa_mode=None)`
  - `Gemma4Attention.forward(hidden_states, selected_freqs, attn_mask, key_cache,
    value_cache, cache_index, cache_seqlen=None, valid_start=None)`
  - `Gemma4Block.forward(hidden_states, selected_freqs, attn_mask, key_cache,
    value_cache, cache_index, layer_scalar, cache_seqlen=None, valid_start=None)`
  - `prepare_gemma4_blocks(layers, num_q_heads_per_layer, kv_shapes,
    is_kv_eq_v_per_layer, layer_types, window_size, swa_mode)`
  - `_build_layer_masks(model, attn_mask, seq_len, batch_size, block_base,
    sliding_band=True)` — with `sliding_band=False` the `"sliding_attention"` entry
    is `None`, so a mis-wired layer cannot silently attend the pad columns behind a
    stale mask.
  - `tests/_swa_helpers.py`: `make_sliding_attention(...)`, `identity_freqs(...)`,
    `FakeKVModel(...)`

- [ ] **Step 1: Write the shared test helpers**

Create `tests/_swa_helpers.py` with the 14-line Apache header, then:

```python
"""Builders shared by the CPU and Spyre sliding-window attention tests.

A real Gemma 4 checkpoint is 12B parameters and its device token-compare is
currently red for unrelated reasons, so the SWA replacement is tested through one
``Gemma4Attention`` with random weights at Gemma 4's shapes.
"""

import types

import torch
import torch.nn as nn


class HeadRMSNorm(nn.Module):
    """Per-head RMSNorm over the last dim, standing in for Gemma4RMSNorm."""

    def __init__(self, head_dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(head_dim))
        self.eps = eps

    def forward(self, x):
        variance = (x * x).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


class FakeKVModel:
    """Minimal stand-in for what ``allocate_kv_caches`` reads off a model.

    Used so the tests get caches with the pinned device layout that
    ``kv_cache_update``'s indirect scatter requires, rather than a bare
    ``torch.zeros`` that would silently write to the wrong rows.
    """

    def __init__(self, kv_shapes):
        self._spyre_kv_shapes = kv_shapes


def identity_freqs(batch, seqlen, head_dim, dtype=torch.float32):
    """``selected_freqs`` that make ``apply_rope_matmul`` a no-op.

    Shape ``[B, L, 2, 2, D/2]`` holding the 2x2 identity, so RoPE cannot
    contribute a difference to an A/B comparison of the attention itself.
    """
    half = head_dim // 2
    freqs = torch.zeros(batch, seqlen, 2, 2, half, dtype=dtype)
    freqs[:, :, 0, 0, :] = 1.0
    freqs[:, :, 1, 1, :] = 1.0
    return freqs


def make_sliding_attention(
    num_q_heads=4,
    num_kv_heads=2,
    head_dim=64,
    window_size=64,
    swa_mode=None,
    dtype=torch.float32,
    seed=0,
):
    """A ``Gemma4Attention`` wired as a sliding layer, with random weights.

    Gemma 4's sliding layers keep a separate V (``is_kv_eq_v`` is False, which is
    a global-layer property), carry per-head RMSNorm on Q/K/V, and attend
    **unscaled** (``scaling == 1.0``).
    """
    from hf_adapters.hf_gemma4 import Gemma4Attention

    torch.manual_seed(seed)
    hidden = num_q_heads * head_dim
    attn = types.SimpleNamespace(
        q_proj=nn.Linear(hidden, num_q_heads * head_dim, bias=False),
        k_proj=nn.Linear(hidden, num_kv_heads * head_dim, bias=False),
        v_proj=nn.Linear(hidden, num_kv_heads * head_dim, bias=False),
        o_proj=nn.Linear(num_q_heads * head_dim, hidden, bias=False),
        q_norm=HeadRMSNorm(head_dim),
        k_norm=HeadRMSNorm(head_dim),
        v_norm=HeadRMSNorm(head_dim),
        scaling=1.0,
    )
    module = Gemma4Attention(
        attn,
        num_q_heads,
        num_kv_heads,
        head_dim,
        False,
        is_sliding=True,
        window_size=window_size,
        swa_mode=swa_mode,
    )
    return module.to(dtype).eval()
```

- [ ] **Step 2: Write the failing test**

Create `tests/cpu/test_swa_gemma4_layer.py` with the header, then:

```python
"""CPU wiring tests for Gemma 4's sliding layer on the SWA op path.

On CPU the dispatcher takes its reference branch, so what these check is the
*wiring*: that the coordinates the adapter derives, the mask suppression, and the
KV write all line up with the band-masked path they replace. The numerics of the
op itself are a device question (tests/spyre/test_swa_layer_ab_spyre.py).
"""

import copy

import torch

from _swa_helpers import identity_freqs, make_sliding_attention
from hf_adapters.hf_common import (
    add_causal_sliding_window_band,
    build_decode_mask,
    build_prefill_mask,
    make_cache_index,
)

WINDOW = 64
CAPACITY = 256
PREFILL = 128
HEAD_DIM = 64


def _band_mask(batch, seqlen, block_base, dtype=torch.float32):
    """Exactly what _build_layer_masks builds for a sliding layer today."""
    mask = build_prefill_mask(batch, seqlen, CAPACITY, 0, dtype=dtype)
    coords = (torch.arange(seqlen)[None, :] + block_base).expand(batch, seqlen)
    return add_causal_sliding_window_band(mask, coords, WINDOW)


def _caches(batch=1, num_kv_heads=2):
    return (
        torch.zeros(batch, num_kv_heads, CAPACITY, HEAD_DIM),
        torch.zeros(batch, num_kv_heads, CAPACITY, HEAD_DIM),
    )


def test_phase1_prefill_matches_the_band_masked_path():
    torch.manual_seed(1)
    hidden = torch.randn(1, PREFILL, 4 * HEAD_DIM)
    freqs = identity_freqs(1, PREFILL, HEAD_DIM)
    index = make_cache_index(0, PREFILL)

    band = make_sliding_attention(window_size=WINDOW, swa_mode=None)
    op = copy.deepcopy(band)
    op.swa_mode = "phase1"

    band_out, band_k, band_v = band(
        hidden, freqs, _band_mask(1, PREFILL, 0), *_caches(), index
    )
    op_out, op_k, op_v = op(
        hidden, freqs, None, *_caches(), index, cache_seqlen=PREFILL, valid_start=[0]
    )

    torch.testing.assert_close(op_out, band_out, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(op_k, band_k)
    torch.testing.assert_close(op_v, band_v)


def test_phase1_decode_matches_the_band_masked_path():
    """One token at cache slot 128, reading the 64 columns behind it."""
    torch.manual_seed(2)
    key_cache, value_cache = _caches()
    key_cache[:, :, :PREFILL, :] = torch.randn(1, 2, PREFILL, HEAD_DIM)
    value_cache[:, :, :PREFILL, :] = torch.randn(1, 2, PREFILL, HEAD_DIM)

    hidden = torch.randn(1, 1, 4 * HEAD_DIM)
    freqs = identity_freqs(1, 1, HEAD_DIM)
    index = make_cache_index(PREFILL, 1)

    band = make_sliding_attention(window_size=WINDOW, swa_mode=None)
    op = copy.deepcopy(band)
    op.swa_mode = "phase1"

    band_out, _, _ = band(
        hidden,
        freqs,
        _band_mask(1, 1, PREFILL),
        key_cache.clone(),
        value_cache.clone(),
        index,
    )
    op_out, _, _ = op(
        hidden,
        freqs,
        None,
        key_cache.clone(),
        value_cache.clone(),
        index,
        cache_seqlen=PREFILL + 1,
        valid_start=[0],
    )

    torch.testing.assert_close(op_out, band_out, rtol=1e-5, atol=1e-6)


def test_global_layer_still_uses_the_mask():
    """is_sliding=False must ignore swa_mode entirely."""
    torch.manual_seed(3)
    hidden = torch.randn(1, PREFILL, 4 * HEAD_DIM)
    freqs = identity_freqs(1, PREFILL, HEAD_DIM)
    index = make_cache_index(0, PREFILL)
    mask = build_prefill_mask(1, PREFILL, CAPACITY, 0, dtype=torch.float32)

    module = make_sliding_attention(swa_mode="phase1")
    module.is_sliding = False

    out, _, _ = module(hidden, freqs, mask, *_caches(), index)
    assert torch.isfinite(out).all()
```

- [ ] **Step 3: Run it to verify it fails**

```bash
cd /mnt/home/spyre/hf-adapters
python3 -m pytest tests/cpu/test_swa_gemma4_layer.py -v 2>&1 | tail -12
```

Expected: `TypeError: Gemma4Attention.__init__() got an unexpected keyword argument
'is_sliding'`.

- [ ] **Step 4: Extend `Gemma4Attention`**

In `hf_adapters/hf_gemma4.py`, add the import:

```python
from hf_adapters.swa_attention import sliding_window_attention
```

Extend `__init__` (keep every existing line; these are additions):

```python
    def __init__(
        self,
        attn,
        num_q_heads,
        num_kv_heads,
        head_dim,
        is_kv_eq_v,
        is_sliding=False,
        window_size=None,
        swa_mode=None,
    ):
```

and after `self.scaling = attn.scaling`:

```python
        self.is_sliding = is_sliding
        self.window_size = window_size
        # None keeps the band-masked SDPA path. "phase1" reads the window out of
        # the full-length cache -- a correctness harness only: cache_seqlen grows,
        # so it recompiles once per 64 decode steps without bound.
        self.swa_mode = swa_mode
```

Replace the `F.scaled_dot_product_attention` call in `forward` (and extend the
signature) with:

```python
    def forward(
        self,
        hidden_states,
        selected_freqs,
        attn_mask,
        key_cache,
        value_cache,
        cache_index,
        cache_seqlen=None,
        valid_start=None,
    ):
```

```python
        if self.is_sliding and self.swa_mode is not None:
            # Windowing is an offset plus a length, so attn_mask is unused here
            # and arrives as None; left padding travels as valid_start instead.
            attn_out = sliding_window_attention(
                q,
                key_cache,
                value_cache,
                window_size=self.window_size,
                scale=self.scaling,
                cache_seqlen=cache_seqlen,
                buffer_origin=0,
                valid_start=valid_start,
            )
        else:
            attn_out = F.scaled_dot_product_attention(
                q,
                key_cache,
                value_cache,
                attn_mask=attn_mask,
                dropout_p=0.0,
                scale=self.scaling,
                enable_gqa=True,
            )
```

- [ ] **Step 5: Thread the two arguments through `Gemma4Block`**

Extend `Gemma4Block.__init__` to accept and forward the three new attention
arguments:

```python
    def __init__(
        self,
        layer,
        num_q_heads,
        num_kv_heads,
        head_dim,
        is_kv_eq_v,
        is_sliding=False,
        window_size=None,
        swa_mode=None,
    ):
        super().__init__()
        self.self_attn = Gemma4Attention(
            layer.self_attn,
            num_q_heads,
            num_kv_heads,
            head_dim,
            is_kv_eq_v,
            is_sliding=is_sliding,
            window_size=window_size,
            swa_mode=swa_mode,
        )
```

and its `forward`:

```python
    def forward(
        self,
        hidden_states,
        selected_freqs,
        attn_mask,
        key_cache,
        value_cache,
        cache_index,
        layer_scalar,
        cache_seqlen=None,
        valid_start=None,
    ):
        residual = hidden_states
        h = self.input_layernorm(hidden_states)
        attn_out, key_cache, value_cache = self.self_attn(
            h,
            selected_freqs,
            attn_mask,
            key_cache,
            value_cache,
            cache_index,
            cache_seqlen,
            valid_start,
        )
```

- [ ] **Step 6: Pass the layer types into `prepare_gemma4_blocks`**

```python
def prepare_gemma4_blocks(
    layers,
    num_q_heads_per_layer,
    kv_shapes,
    is_kv_eq_v_per_layer,
    layer_types,
    window_size,
    swa_mode,
):
    """Replace Gemma 4 decoder layers with registered blocks and compile them."""
    blocks = []
    for i, layer in enumerate(list(layers)):
        block = Gemma4Block(
            layer,
            num_q_heads_per_layer[i],
            kv_shapes[i][0],
            kv_shapes[i][1],
            is_kv_eq_v_per_layer[i],
            is_sliding=layer_types[i] == "sliding_attention",
            window_size=window_size,
            swa_mode=swa_mode,
        )
        layers[i] = block
        blocks.append(torch.compile(block, dynamic=False))
    return blocks
```

and at the call site in `prepare_text_decoder_for_spyre`:

```python
    model._spyre_compiled_blocks = prepare_gemma4_blocks(
        backbone.layers,
        num_q_heads_per_layer,
        kv_shapes,
        is_kv_eq_v_per_layer,
        cfg.layer_types,
        cfg.sliding_window,
        getattr(model, "_spyre_swa_mode", None),
    )
```

- [ ] **Step 7: Suppress the band and supply the coordinates**

Give `_build_layer_masks` the extra parameter, and add to its docstring that
`sliding_band=False` returns `None` for the sliding entry because the op path takes
its window from an offset and a length:

```python
def _build_layer_masks(
    model,
    attn_mask,
    seq_len,
    batch_size,
    block_base,
    sliding_band=True,
):
```

```python
    if not sliding_band:
        # The op path derives its window from cache_seqlen and window_size. Return
        # None rather than a stale band so a mis-wired layer fails visibly instead
        # of quietly attending the pad columns.
        return {"full_attention": attn_mask, "sliding_attention": None}
    cfg = text_config(model.config)
```

Add the helper at module level:

```python
def _swa_valid_start(model, batch_size):
    """First attendable cache column per sequence -- ``generate``'s left padding.

    ``generate`` stashes ``_spyre_prompt_offsets``; a caller driving the forward
    directly (the layer tests) has none.
    """
    offsets = getattr(model, "_spyre_prompt_offsets", None)
    if offsets is None:
        return [0] * batch_size
    return [int(offset) for offset in offsets]
```

In `_run_blocks_over_embeds`, replace the mask-building block and the call:

```python
    swa_mode = getattr(model, "_spyre_swa_mode", None)
    if masks is not None:
        # A caller supplying its own masks is the unified VLM adapter, whose
        # bidirectional vision overlay *widens* attention. The op cannot express
        # that (is_causal=False raises, and an additive mask only ever removes), so
        # the VLM keeps the band-masked path.
        swa_mode = None
    bsz, seq_len = h.shape[0], h.shape[1]
    # The scalar read syncs from the device; deliberately not optimized, and now
    # needed by the op path as well as by the band. Once per step, not per layer.
    block_base = int(cache_index[0]) if masks is None or swa_mode else 0

    if masks is None:
        masks = _build_layer_masks(
            model, attn_mask, seq_len, bsz, block_base, sliding_band=swa_mode is None
        )

    cache_seqlen = block_base + seq_len if swa_mode else None
    valid_start = _swa_valid_start(model, bsz) if swa_mode else None

    backbone_layers = backbone.layers
    for i, compiled_block in enumerate(model._spyre_compiled_blocks):
        lt = cfg.layer_types[i]
        h, key_caches[i], value_caches[i] = compiled_block(
            h,
            freqs[lt],
            masks[lt],
            key_caches[i],
            value_caches[i],
            cache_index,
            backbone_layers[i].layer_scalar,
            cache_seqlen,
            valid_start,
        )
```

Delete the now-duplicated `bsz, seq_len = ...` and `block_base = ...` lines from
inside the old `if masks is None:` body.

- [ ] **Step 8: Run the layer tests to verify they pass**

```bash
cd /mnt/home/spyre/hf-adapters
python3 -m pytest tests/cpu/test_swa_gemma4_layer.py -v 2>&1 | tail -12
```

Expected: 3 passed.

- [ ] **Step 9: Run the CPU accuracy suite for regressions**

The default path (`swa_mode is None`) must be byte-identical to before.

```bash
cd /mnt/home/spyre/hf-adapters
python3 -m pytest tests/cpu/test_adapter_cpu_accuracy.py -k gemma -q 2>&1 | tail -12
python3 -m pytest tests/cpu/test_generate_cpu.py -x -q 2>&1 | tail -6
```

Expected: no new failures.

- [ ] **Step 10: Commit**

```bash
cd /mnt/home/spyre/hf-adapters
git add hf_adapters/hf_gemma4.py tests/_swa_helpers.py tests/cpu/test_swa_gemma4_layer.py
git commit -s -m "feat(gemma4): opt-in sliding-window op path over the full cache"
```

---

### Task 6: phase 1 on device — the A/B gate

The primary gate. Same block, same inputs, same cache contents, band-masked SDPA
versus the op, on hardware. This is what tells us the op is correct for Gemma 4
independently of Gemma 4's pre-existing prefill divergence.

**Files:**
- Create: `tests/spyre/test_swa_layer_ab_spyre.py`

**Interfaces:**
- Consumes: `tests/_swa_helpers.py` (Task 5), `hf_common.allocate_kv_caches` with
  `capacities=` (Task 4), the op with `valid_start` (Task 2).
- Produces: nothing consumed by later tasks. Task 9 extends this file.

- [ ] **Step 1: Write the test**

Create `tests/spyre/test_swa_layer_ab_spyre.py` with the header, then:

```python
"""Device A/B: Gemma 4's sliding layer, band-masked SDPA versus the SWA op.

Gemma 4 12B currently fails device token-compare 0/5 and diverges at prefill for
reasons unrelated to sliding-window attention, so end-to-end output cannot answer
whether this replacement is correct. This test can: one layer, identical inputs,
identical cache contents, both paths measured against a common float32 reference.

**Why not compare the two device paths to each other at a tight tolerance.**
Measured on this hardware at these shapes: against a float32 CPU reference, the
band-masked path we already ship sits at max abs error 0.049 and the op at 0.067,
on outputs of scale 2.5; the two differ from each other by 0.064. None of that is
an op defect — it is fp16 reduction noise, amplified because Gemma 4 attends
unscaled (``scaling == 1.0``) at ``head_dim=256``, which makes a random-weight
softmax nearly one-hot. In the real model ``q_norm``/``k_norm`` tame the scores;
this harness has no such luxury. A device-vs-device tolerance would therefore be a
statement about fp16, not about the op — and the shipped path would fail it too.

So the assertion is the one that matters: **the op must be no less accurate than
the path it replaces.**

Run (on the Spyre pod)::

    source /mnt/home/spyre/torch-spyre-docs/scripts/dev-env.sh
    python3 -m pytest -s -vvv tests/spyre/test_swa_layer_ab_spyre.py
"""

import copy

import pytest
import torch

from _swa_helpers import FakeKVModel, identity_freqs, make_sliding_attention
from hf_adapters.hf_common import (
    add_causal_sliding_window_band,
    allocate_kv_caches,
    build_decode_mask,
    build_prefill_mask,
    make_cache_index,
)

# Gemma 4 12B's sliding layers, at one quarter the head count so a test fits.
WINDOW = 1024
HEAD_DIM = 256
Q_HEADS = 4
KV_HEADS = 2
DTYPE = torch.float16
# The op may be up to this multiple of the band path's own float32 error, plus a
# floor so the ratio stays meaningful when both errors are near zero.
ERROR_RATIO = 2.0
ERROR_FLOOR = 5e-3


def _spyre_caches(capacity):
    """Caches with the pinned device layout the indirect scatter requires."""
    model = FakeKVModel([(KV_HEADS, HEAD_DIM, HEAD_DIM)])
    keys, values = allocate_kv_caches(model, 1, capacity, DTYPE, device="spyre")
    return keys[0], values[0]


def _cpu_caches(capacity):
    return (
        torch.zeros(1, KV_HEADS, capacity, HEAD_DIM),
        torch.zeros(1, KV_HEADS, capacity, HEAD_DIM),
    )


def _run(module, hidden, freqs, mask, key_cache, value_cache, index, **kwargs):
    """One compiled forward on device. Returns (output on CPU as float32, caches)."""
    compiled = torch.compile(module, dynamic=False)
    with torch.no_grad():
        out, key_cache, value_cache = compiled(
            hidden, freqs, mask, key_cache, value_cache, index, **kwargs
        )
    return out.to("cpu").float(), key_cache, value_cache


def _run_cpu32(module, hidden, freqs, mask, key_cache, value_cache, index):
    """The same layer in float32 on CPU, eager — the closest thing to truth."""
    with torch.no_grad():
        out, key_cache, value_cache = module(
            hidden.float(), freqs.float(), mask.float(), key_cache, value_cache, index
        )
    return out, key_cache, value_cache


def _assert_no_worse(op_out, band_out, reference, rows_from=0):
    """The op must be no less accurate than the band path, against float32 truth.

    All three are ``[B, L, hidden]`` — the layer returns its ``o_proj`` output, not
    per-head attention — so ``rows_from`` slices dim 1, the query rows. It drops
    leading rows whose attention is undefined (see the left-padding test).
    """
    op_out = op_out[:, rows_from:]
    band_out = band_out[:, rows_from:]
    reference = reference[:, rows_from:]
    assert torch.isfinite(op_out).all(), "op output must be finite"
    band_error = (band_out - reference).abs().max().item()
    op_error = (op_out - reference).abs().max().item()
    allowed = ERROR_RATIO * band_error + ERROR_FLOOR
    print(
        f"\n  op {op_error:.4f} vs band {band_error:.4f} "
        f"(allowed {allowed:.4f}, ref scale {reference.abs().max().item():.3f})"
    )
    assert op_error <= allowed, (
        f"op error {op_error:.4f} exceeds {allowed:.4f} = "
        f"{ERROR_RATIO} x the band path's own error {band_error:.4f} + {ERROR_FLOOR}"
    )


def _modules():
    """A band-path module and an op-path module with identical weights, plus a
    float32 CPU twin of the band path to measure both against."""
    band = make_sliding_attention(
        Q_HEADS, KV_HEADS, HEAD_DIM, WINDOW, swa_mode=None, dtype=DTYPE
    )
    op = copy.deepcopy(band)
    op.swa_mode = "phase1"
    reference = make_sliding_attention(
        Q_HEADS, KV_HEADS, HEAD_DIM, WINDOW, swa_mode=None, dtype=torch.float32
    )
    return band.to("spyre"), op.to("spyre"), reference


@pytest.mark.parametrize("seqlen_q", [64, 512])
def test_prefill_op_no_worse_than_band_mask(seqlen_q):
    torch._dynamo.reset()
    capacity = 1088
    torch.manual_seed(3)
    hidden = torch.randn(1, seqlen_q, Q_HEADS * HEAD_DIM, dtype=DTYPE)
    freqs = identity_freqs(1, seqlen_q, HEAD_DIM, dtype=DTYPE)
    index = make_cache_index(0, seqlen_q)

    mask = build_prefill_mask(1, seqlen_q, capacity, 0, dtype=DTYPE)
    mask = add_causal_sliding_window_band(mask, torch.arange(seqlen_q)[None, :], WINDOW)

    band, op, reference = _modules()
    band_out, _, _ = _run(
        band,
        hidden.to("spyre"),
        freqs.to("spyre"),
        mask.to("spyre"),
        *_spyre_caches(capacity),
        index.to("spyre"),
    )
    op_out, _, _ = _run(
        op,
        hidden.to("spyre"),
        freqs.to("spyre"),
        None,
        *_spyre_caches(capacity),
        index.to("spyre"),
        cache_seqlen=seqlen_q,
        valid_start=[0],
    )
    ref_out, _, _ = _run_cpu32(
        reference, hidden, freqs, mask, *_cpu_caches(capacity), index
    )
    _assert_no_worse(op_out, band_out, ref_out)


def test_decode_op_no_worse_than_band_mask():
    """A real prefill, then one decode step reading the 1024 columns behind it.

    The cache is filled by running prefill through each module rather than by an
    eager ``index_copy_``: that mirrors production, and an eager index write on
    device falls back to CPU with a warning.
    """
    torch._dynamo.reset()
    capacity, written = 1088, 512
    torch.manual_seed(4)
    prompt = torch.randn(1, written, Q_HEADS * HEAD_DIM, dtype=DTYPE)
    prompt_freqs = identity_freqs(1, written, HEAD_DIM, dtype=DTYPE)
    prompt_index = make_cache_index(0, written)
    prompt_mask = build_prefill_mask(1, written, capacity, 0, dtype=DTYPE)
    prompt_mask = add_causal_sliding_window_band(
        prompt_mask, torch.arange(written)[None, :], WINDOW
    )

    band, op, reference = _modules()
    _, band_k, band_v = _run(
        band,
        prompt.to("spyre"),
        prompt_freqs.to("spyre"),
        prompt_mask.to("spyre"),
        *_spyre_caches(capacity),
        prompt_index.to("spyre"),
    )
    _, op_k, op_v = _run(
        op,
        prompt.to("spyre"),
        prompt_freqs.to("spyre"),
        None,
        *_spyre_caches(capacity),
        prompt_index.to("spyre"),
        cache_seqlen=written,
        valid_start=[0],
    )
    _, ref_k, ref_v = _run_cpu32(
        reference, prompt, prompt_freqs, prompt_mask, *_cpu_caches(capacity),
        prompt_index,
    )

    token = torch.randn(1, 1, Q_HEADS * HEAD_DIM, dtype=DTYPE)
    token_freqs = identity_freqs(1, 1, HEAD_DIM, dtype=DTYPE)
    index = make_cache_index(written, 1)
    # build_decode_mask, not build_prefill_mask: a one-token step at a non-zero
    # cache position must allow every real column up to that position, whereas
    # build_prefill_mask masks everything past the query's *relative* row index
    # and so allows only column 0. The band is combined additively, so an
    # over-restrictive base cannot be widened back — every column ends up masked.
    mask = build_decode_mask(1, capacity, written, 0, dtype=DTYPE)
    mask = add_causal_sliding_window_band(mask, torch.tensor([[written]]), WINDOW)

    band_out, _, _ = _run(
        band,
        token.to("spyre"),
        token_freqs.to("spyre"),
        mask.to("spyre"),
        band_k,
        band_v,
        index.to("spyre"),
    )
    op_out, _, _ = _run(
        op,
        token.to("spyre"),
        token_freqs.to("spyre"),
        None,
        op_k,
        op_v,
        index.to("spyre"),
        cache_seqlen=written + 1,
        valid_start=[0],
    )
    ref_out, _, _ = _run_cpu32(
        reference, token, token_freqs, mask, ref_k, ref_v, index
    )
    _assert_no_worse(op_out, band_out, ref_out)


def test_left_padding_op_no_worse_than_band_mask():
    """17 pad columns inside the window: valid_start must match the mask.

    This is the case the batch>1 Qwen3 decode bug lived in, so it is also the
    canary for window_band_mask's -inf fill (see hf_common._mask_fill_value).

    Compared from row ``offset`` on, plus finiteness everywhere. Rows below the
    threshold have their whole window excluded, so all three paths disagree there
    by construction and none of the answers is meaningful — those rows are padding
    whose outputs are discarded. What matters is that they stay finite, since a
    non-finite value would reach the next layer's KV cache.
    """
    torch._dynamo.reset()
    capacity, seqlen_q, offset = 1088, 512, 17
    torch.manual_seed(5)
    hidden = torch.randn(1, seqlen_q, Q_HEADS * HEAD_DIM, dtype=DTYPE)
    freqs = identity_freqs(1, seqlen_q, HEAD_DIM, dtype=DTYPE)
    index = make_cache_index(0, seqlen_q)

    mask = build_prefill_mask(1, seqlen_q, capacity, offset, dtype=DTYPE)
    mask = add_causal_sliding_window_band(mask, torch.arange(seqlen_q)[None, :], WINDOW)

    band, op, reference = _modules()
    band_out, _, _ = _run(
        band,
        hidden.to("spyre"),
        freqs.to("spyre"),
        mask.to("spyre"),
        *_spyre_caches(capacity),
        index.to("spyre"),
    )
    op_out, _, _ = _run(
        op,
        hidden.to("spyre"),
        freqs.to("spyre"),
        None,
        *_spyre_caches(capacity),
        index.to("spyre"),
        cache_seqlen=seqlen_q,
        valid_start=[offset],
    )
    ref_out, _, _ = _run_cpu32(
        reference, hidden, freqs, mask, *_cpu_caches(capacity), index
    )
    assert torch.isfinite(op_out).all(), "fully-masked pad rows must stay finite"
    _assert_no_worse(op_out, band_out, ref_out, rows_from=offset)
```

- [ ] **Step 2: Run it**

```bash
source /mnt/home/spyre/torch-spyre-docs/scripts/dev-env.sh
cd /mnt/home/spyre/hf-adapters
SENCORES=1 python3 -m pytest -s tests/spyre/test_swa_layer_ab_spyre.py -v 2>&1 | tail -40
```

Expected: 4 passed (two prefill shapes, decode, left padding). Each prints its
measured `op` and `band` errors against the float32 reference — record those numbers,
they are the evidence Task 11 reports.

For calibration, the numbers measured while designing this test (prefill `Lq=64`,
output scale 2.50): band path 0.0487, op 0.0670, allowed
`2 x 0.0487 + 0.005 = 0.102`. An op error in that neighbourhood is healthy.

Diagnosis if a case fails:

- **`op_error` is a large multiple of `band_error`** (say 5x or more), or `band_error`
  is near zero while the op's is not → a real op defect at this geometry. Report the
  two numbers and the shape; do not widen `ERROR_RATIO` to accommodate it. Widening
  the ratio to make a test pass converts the gate into a rubber stamp.
- **`op_error` is not finite, or the output contains NaN/Inf** → suspect
  `window_band_mask`'s `float("-inf")` fill. Host fp16 `-inf` lands on device as a
  finite `-3.35e7`, ~500x past fp16's max finite, which round-trips badly through
  fp16 materializations; `hf_common._mask_fill_value` uses `finfo(dtype).min / 2` for
  exactly this reason and cites the batch>1 Qwen3 decode bug. The fix would be to use
  a finite fill in `window_band_mask` **in the torch-spyre repo** — escalate rather
  than doing it, since that is a change to a different repo.
- **`band_error` itself is huge** (comparable to the output scale) → the band mask is
  wrong for this case, not the op. Check the prefill-vs-decode mask builder choice:
  `build_decode_mask` for a one-token step at a non-zero position,
  `build_prefill_mask` otherwise.
- **Every case fails identically** → check the float32 reference is really running the
  band path (`swa_mode=None`) and receiving the same weights, seed and inputs.

- [ ] **Step 3: Commit**

```bash
cd /mnt/home/spyre/hf-adapters
git add tests/spyre/test_swa_layer_ab_spyre.py
git commit -s -m "test(gemma4): device A/B of the SWA op against the band mask"
```

---

### Task 7: the anchored compact buffer's bookkeeping

Pure arithmetic plus two tensor copies. Everything that decides *which integers*
reach the op lives here, and every off-by-one in it is cheap to catch on CPU and
expensive to catch on device.

**Files:**
- Modify: `hf_adapters/hf_common.py` (lift `_alloc` out of `allocate_kv_caches`)
- Modify: `hf_adapters/swa_attention.py` (append)
- Test: `tests/cpu/test_swa_cache.py`

**Interfaces:**
- Consumes: `swa_attention.sliding_capacity` (Task 3), `allocate_kv_caches`'s
  layout-pinned allocation (Task 4).
- Produces:
  - `hf_common.allocate_kv_cache(batch_size, num_kv_heads, rows, head_dim, dtype,
    device) -> Tensor` — one layout-pinned, zeroed cache.
  - `swa_attention.SlidingWindowCache(window_size, capacity, write_row, valid_start)`
    with `.anchor`, `.stick_offset()`, `.needs_shift()`, `.shift()`, `.advance()`,
    and `SlidingWindowCache.after_prefill(window_size, prompt_len, offsets)`.
  - `swa_attention.compact_after_prefill(key_cache, value_cache, state, prompt_len)
    -> (Tensor, Tensor)`
  - `swa_attention.shift_indices(capacity, device) -> (Tensor, Tensor)`

- [ ] **Step 1: Write the failing tests**

Create `tests/cpu/test_swa_cache.py` with the header, then:

```python
"""CPU tests for the anchored compact KV buffer's bookkeeping.

The invariant under test, at the start of every 64-token stick period:

  * physical rows ``[0, anchor)`` hold the most recent ``anchor`` tokens, all real
  * rows ``[anchor, capacity)`` are empty and take the next 64 writes

which is what pins ``cache_seqlen == capacity``, ``buffer_origin == 0`` and
``seqlen_q == 64`` and so gives one compiled decode graph. ``anchor`` is
``capacity - 64``.
"""

import torch

from hf_adapters.swa_attention import (
    SlidingWindowCache,
    compact_after_prefill,
    shift_indices,
    sliding_capacity,
)

WINDOW = 1024
CAPACITY = 1088  # sliding_capacity(1024)
ANCHOR = 1024


def test_after_prefill_anchors_the_write_row():
    state = SlidingWindowCache.after_prefill(WINDOW, prompt_len=2048, offsets=[0])
    assert state.capacity == CAPACITY
    assert state.anchor == ANCHOR
    assert state.write_row == ANCHOR
    assert state.stick_offset() == 0
    # A prompt longer than the buffer fills rows [0, anchor) with real tokens.
    assert state.valid_start == [0]


def test_after_prefill_short_prompt_leaves_a_masked_front():
    # 100 prompt tokens cannot fill 1024 rows: the unwritten front is zeros, and
    # only valid_start can keep them out of the window.
    state = SlidingWindowCache.after_prefill(WINDOW, prompt_len=100, offsets=[0])
    assert state.valid_start == [ANCHOR - 100]
    assert state.write_row == ANCHOR


def test_after_prefill_counts_left_padding_that_survives_compaction():
    # 100 padded columns of which 17 are pad: 83 real tokens land last, so the
    # threshold is the unwritten front plus the pad that came with them.
    state = SlidingWindowCache.after_prefill(WINDOW, prompt_len=100, offsets=[17])
    assert state.valid_start == [ANCHOR - 100 + 17]


def test_after_prefill_drops_left_padding_that_falls_off_the_front():
    # A prompt longer than the buffer: the pad columns are the oldest tokens and
    # compaction discards them, so nothing needs masking.
    state = SlidingWindowCache.after_prefill(WINDOW, prompt_len=2048, offsets=[17])
    assert state.valid_start == [0]


def test_after_prefill_is_per_sequence():
    state = SlidingWindowCache.after_prefill(WINDOW, prompt_len=100, offsets=[0, 17])
    assert state.valid_start == [ANCHOR - 100, ANCHOR - 100 + 17]


def test_the_write_stick_holds_64_tokens_then_asks_for_a_shift():
    state = SlidingWindowCache.after_prefill(WINDOW, prompt_len=2048, offsets=[0])
    rows = []
    for _ in range(64):
        assert not state.needs_shift()
        rows.append(state.write_row)
        state.advance()
    assert rows == list(range(ANCHOR, CAPACITY))
    assert state.needs_shift()


def test_shift_returns_to_the_anchor_and_erodes_valid_start():
    state = SlidingWindowCache.after_prefill(WINDOW, prompt_len=100, offsets=[0])
    assert state.valid_start == [ANCHOR - 100]
    for _ in range(64):
        state.advance()
    state.shift()
    assert state.write_row == ANCHOR
    assert state.stick_offset() == 0
    # 64 of the unwritten front rows fell off, so the threshold drops by 64.
    assert state.valid_start == [ANCHOR - 100 - 64]


def test_valid_start_never_goes_negative():
    state = SlidingWindowCache.after_prefill(WINDOW, prompt_len=1020, offsets=[0])
    assert state.valid_start == [4]
    state.shift()
    assert state.valid_start == [0]


def test_stick_offset_tracks_the_position_within_the_stick():
    state = SlidingWindowCache.after_prefill(WINDOW, prompt_len=2048, offsets=[0])
    for expected in range(64):
        assert state.stick_offset() == expected
        state.advance()


def test_shift_indices_move_the_tail_down_one_stick():
    src, dst = shift_indices(CAPACITY, "cpu")
    assert src.tolist() == list(range(64, CAPACITY))
    assert dst.tolist() == list(range(0, ANCHOR))
    assert src.dtype == torch.long and dst.dtype == torch.long


def test_compaction_keeps_the_newest_rows_right_aligned_at_the_anchor():
    """The prefill/decode boundary: the tail of a big buffer into a compact one."""
    prompt_len = 2048
    big_k = torch.zeros(1, 2, prompt_len, 8)
    big_v = torch.zeros(1, 2, prompt_len, 8)
    # Row r carries the value r, so provenance is checkable.
    marks = torch.arange(prompt_len, dtype=torch.float32).view(1, 1, prompt_len, 1)
    big_k += marks
    big_v += marks + 0.5

    state = SlidingWindowCache.after_prefill(WINDOW, prompt_len, offsets=[0])
    compact_k, compact_v = compact_after_prefill(big_k, big_v, state, prompt_len)

    assert compact_k.shape == (1, 2, CAPACITY, 8)
    # Rows [0, anchor) hold prompt rows [prompt_len - anchor, prompt_len).
    assert compact_k[0, 0, 0, 0].item() == prompt_len - ANCHOR
    assert compact_k[0, 0, ANCHOR - 1, 0].item() == prompt_len - 1
    assert compact_v[0, 0, ANCHOR - 1, 0].item() == prompt_len - 1 + 0.5
    # The write stick starts empty -- the op reads it, so it must be zero, not junk.
    assert not compact_k[:, :, ANCHOR:, :].any()
    assert not compact_v[:, :, ANCHOR:, :].any()


def test_compaction_right_aligns_a_short_prompt():
    prompt_len = 100
    big_k = torch.zeros(1, 2, prompt_len, 8)
    big_v = torch.zeros(1, 2, prompt_len, 8)
    marks = torch.arange(prompt_len, dtype=torch.float32).view(1, 1, prompt_len, 1)
    big_k += marks
    big_v += marks

    state = SlidingWindowCache.after_prefill(WINDOW, prompt_len, offsets=[0])
    compact_k, _ = compact_after_prefill(big_k, big_v, state, prompt_len)

    # Real rows end at the anchor; everything before valid_start stays zero.
    assert compact_k[0, 0, ANCHOR - 1, 0].item() == prompt_len - 1
    assert compact_k[0, 0, ANCHOR - prompt_len, 0].item() == 0.0
    assert not compact_k[:, :, : state.valid_start[0], :].any()


def test_capacity_matches_sliding_capacity():
    """One source of truth for the allocation size."""
    for window in (512, 1024, 100):
        state = SlidingWindowCache.after_prefill(window, 4096, offsets=[0])
        assert state.capacity == sliding_capacity(window)
        assert state.anchor == state.capacity - 64
```

- [ ] **Step 2: Run them to verify they fail**

```bash
cd /mnt/home/spyre/hf-adapters
python3 -m pytest tests/cpu/test_swa_cache.py -v 2>&1 | tail -10
```

Expected: `ImportError: cannot import name 'SlidingWindowCache'`.

- [ ] **Step 3: Lift the pinned allocation out of `allocate_kv_caches`**

In `hf_adapters/hf_common.py`, add a module-level function just above
`allocate_kv_caches`:

```python
def allocate_kv_cache(batch_size, num_kv_heads, rows, head_dim, dtype, device):
    """One zeroed ``[B, n_kv, rows, head_dim]`` cache with a scatter-ready layout.

    On Spyre the *device* layout is pinned so the cache-position dim lands at
    device position 0, which is what ``kv_cache_update``'s indirect scatter
    requires; a cache without the pin is written to the wrong rows silently, with
    no error (torch-spyre#3705). Split out of ``allocate_kv_caches`` so a caller
    allocating one replacement cache mid-generation — the sliding-window
    compaction — gets the same pin as the initial allocation.
    """
    if torch.device(device).type != "spyre":
        return torch.zeros(
            (batch_size, num_kv_heads, rows, head_dim), dtype=dtype, device=device
        )
    stl = _cache_position_first_stl(batch_size, num_kv_heads, rows, head_dim, dtype)
    cache: torch.Tensor = torch.empty(  # type: ignore[call-overload]
        (batch_size, num_kv_heads, rows, head_dim),
        device=torch.device(device),
        device_layout=stl,
        dtype=dtype,
    )
    cache.zero_()
    return cache
```

and replace the inner `_alloc` in `allocate_kv_caches` with a call to it:

```python
    def _alloc(n_kv, head_dim, rows):
        return allocate_kv_cache(batch_size, n_kv, rows, head_dim, dtype, device)
```

- [ ] **Step 4: Implement the bookkeeping**

Append to `hf_adapters/swa_attention.py`:

```python
@dataclasses.dataclass
class SlidingWindowCache:
    """Anchored compact-buffer state for one generation's sliding layers.

    The invariant, at the start of every 64-token stick period:

      * physical rows ``[0, anchor)`` hold the most recent ``anchor`` tokens, all
        real once the buffer has filled
      * rows ``[anchor, capacity)`` are empty and take the next 64 writes

    Token ``m`` of a period is written at row ``anchor + m``; after the 64th, rows
    ``[64, capacity)`` shift down to ``[0, anchor)`` and the invariant is restored.
    The shift happens *before* a stick of writes rather than after, which is what
    keeps every row below the write cursor real and so keeps ``valid_start`` at 0
    in the steady state.

    Why this shape at all: the op takes its geometry as trace-time integers, so
    every distinct ``(cache_seqlen, buffer_origin, seqlen_q)`` is a distinct
    compiled graph. Anchoring pins all three — ``capacity``, 0, and 64 — for the
    whole generation, so decode compiles one graph (two, counting the shift
    branch) instead of one per position.
    """

    window_size: int
    capacity: int
    write_row: int
    valid_start: list

    @property
    def anchor(self):
        """First physical row of the 64-row stick currently being written."""
        return self.capacity - BLOCK_SIZE

    @classmethod
    def after_prefill(cls, window_size, prompt_len, offsets):
        """State for the first decode step, i.e. just after compaction.

        ``offsets`` is ``generate``'s per-sequence left padding, in the prefill
        buffer's coordinates. Compaction keeps the newest ``min(prompt_len,
        anchor)`` rows, right-aligned at the anchor, so:

          * a prompt longer than the buffer pushes its pad columns off the front
            and needs no threshold at all;
          * a shorter one leaves unwritten rows at the front, and any pad that
            travelled with it sits directly above them.
        """
        capacity = sliding_capacity(window_size)
        anchor = capacity - BLOCK_SIZE
        kept = min(prompt_len, anchor)
        valid_start = [
            (anchor - kept) + max(0, int(offset) - (prompt_len - kept))
            for offset in offsets
        ]
        return cls(window_size, capacity, anchor, valid_start)

    def stick_offset(self):
        """Index of the current token within its 64-row query stick."""
        return self.write_row - self.anchor

    def needs_shift(self):
        """True when the write stick is full, so the buffer must roll first."""
        return self.write_row >= self.capacity

    def shift(self):
        """Advance the bookkeeping past a 64-row roll of the buffer."""
        self.write_row = self.anchor
        self.valid_start = [max(0, start - BLOCK_SIZE) for start in self.valid_start]

    def advance(self):
        """Move the write cursor on by the one token this step wrote."""
        self.write_row += 1


def shift_indices(capacity, device):
    """``(src, dst)`` for rolling a compact buffer down by one stick.

    Rows ``[64, capacity)`` move to ``[0, capacity - 64)``. Built on CPU then
    moved, like ``make_cache_index``: ``torch.arange`` falls back to CPU on Spyre
    anyway.
    """
    src = torch.arange(BLOCK_SIZE, capacity, dtype=torch.long)
    dst = torch.arange(0, capacity - BLOCK_SIZE, dtype=torch.long)
    return src.to(device), dst.to(device)


def compact_after_prefill(key_cache, value_cache, state, prompt_len):
    """Move a prefill-sized cache's newest rows into a fresh anchored buffer.

    Prefill needs ``max(sliding_capacity(W), prompt)`` rows; decode needs only
    ``capacity``. Rather than carry the prefill allocation for the whole
    generation — at Gemma 4 12B's 40 sliding layers and an 8192-token context that
    is gigabytes — copy the newest ``min(prompt_len, anchor)`` rows into
    ``[anchor - kept, anchor)`` of a compact buffer and let the big one go.

    Returns the new ``(key_cache, value_cache)``; the caller must replace its
    references, since nothing else keeps the compact buffers alive.
    """
    anchor = state.anchor
    kept = min(prompt_len, anchor)
    device = key_cache.device
    src = torch.arange(prompt_len - kept, prompt_len, dtype=torch.long).to(device)
    dst = torch.arange(anchor - kept, anchor, dtype=torch.long).to(device)

    batch, num_kv_heads, _, head_dim = key_cache.shape
    compact_key = allocate_kv_cache(
        batch, num_kv_heads, state.capacity, head_dim, key_cache.dtype, device
    )
    compact_value = allocate_kv_cache(
        batch,
        num_kv_heads,
        state.capacity,
        value_cache.shape[3],
        value_cache.dtype,
        device,
    )
    _compact_copy(compact_key, key_cache, dst, src)
    _compact_copy(compact_value, value_cache, dst, src)
    return compact_key, compact_value


def _compact_copy(destination, source, dst_index, src_index):
    """``destination[dst] = source[src]`` along the cache-position dim, in place.

    ``index_select`` then ``index_copy_``, the pair ``kv_cache_update`` already
    relies on: in place on the destination so its pinned layout survives, where an
    out-of-place copy would come back with the default layout and be written to the
    wrong rows afterwards (torch-spyre#3705).
    """
    destination.index_copy_(2, dst_index, source.index_select(2, src_index))
    return destination
```

Add the two imports at the top of the module:

```python
import dataclasses

from hf_adapters.hf_common import BLOCK_SIZE, allocate_kv_cache
```

- [ ] **Step 5: Run the tests to verify they pass**

```bash
cd /mnt/home/spyre/hf-adapters
python3 -m pytest tests/cpu/test_swa_cache.py -v 2>&1 | tail -20
python3 -m pytest tests/cpu/test_kv_cache_scatter.py -q 2>&1 | tail -6
```

Expected: 13 passed in the first, no regressions in the second (`_alloc` was
rewritten).

- [ ] **Step 6: Commit**

```bash
cd /mnt/home/spyre/hf-adapters
git add hf_adapters/hf_common.py hf_adapters/swa_attention.py tests/cpu/test_swa_cache.py
git commit -s -m "feat(swa): anchored compact-buffer bookkeeping and compaction"
```

---

### Task 8: phase 2 — the anchored path in Gemma 4

The shipped path. Prefill keeps the phase-1 call shape (correct, and one graph);
decode runs the anchored geometry, so `cache_seqlen`, `buffer_origin` and
`seqlen_q` never change and decode compiles two graphs per block (one with the
shift branch, one without) instead of one per position.

**Files:**
- Modify: `hf_adapters/swa_attention.py` (append `AnchoredStep`, `anchored_step`)
- Modify: `hf_adapters/hf_gemma4.py` (`Gemma4Attention`, `Gemma4Block`,
  `_run_blocks_over_embeds`, `prepare_text_decoder_for_spyre`)
- Modify: `hf_adapters/hf_common.py` (`generate`, one line)
- Modify: `tests/_swa_helpers.py` (no signature change needed — `capacity` is
  derived from `window_size`)
- Test: `tests/cpu/test_swa_anchored_cpu.py`

**Interfaces:**
- Consumes: everything from Tasks 3, 5 and 7.
- Produces:
  - `swa_attention.AnchoredStep(do_shift, cache_index, stick_index, cache_seqlen,
    valid_start)` and `swa_attention.anchored_step(state, device) -> AnchoredStep`
    (mutates `state` when a shift is due; the caller calls `state.advance()` after
    the step).
  - `Gemma4Attention.forward(..., cache_seqlen=None, valid_start=None,
    stick_index=None, do_shift=False)` — `stick_index` not None selects the
    anchored 64-row query stick.
  - `Gemma4Block.forward(..., cache_seqlen=None, valid_start=None,
    stick_index=None, do_shift=False)`
  - `model._spyre_swa_state`, reset on every prefill call.
  - `generate` stashes `model._spyre_prompt_offsets = prompt_offsets`.

- [ ] **Step 1: Write the failing test**

Create `tests/cpu/test_swa_anchored_cpu.py` with the header, then:

```python
"""CPU integration test for the anchored compact buffer, across a shift.

Drives 70 decode steps through one Gemma 4 sliding layer twice: once against a
full-length cache behind a band mask (what the adapter does today), once against a
192-row anchored compact buffer that rolls at token 64. The outputs must agree at
every step, which is what says the compaction, the stick offsets, the shift and the
valid_start erosion all line up.

W=128 rather than Gemma 4's 1024 so the test crosses a shift in 70 steps; the
arithmetic is identical.
"""

import copy

import torch

from _swa_helpers import identity_freqs, make_sliding_attention
from hf_adapters.hf_common import (
    add_causal_sliding_window_band,
    build_prefill_mask,
    make_cache_index,
)
from hf_adapters.swa_attention import (
    SlidingWindowCache,
    anchored_step,
    compact_after_prefill,
)

WINDOW = 128
PROMPT = 256
STEPS = 70  # crosses the shift at token 64
FULL_CAPACITY = 384  # >= PROMPT + STEPS, stick-aligned
HEAD_DIM = 64
Q_HEADS = 4
KV_HEADS = 2


def _band_mask(seqlen, block_base):
    """What _build_layer_masks builds for a sliding layer today.

    Dispatches the way ``generate`` does: ``build_decode_mask`` for a one-token
    step at a non-zero cache position, ``build_prefill_mask`` otherwise.
    ``build_prefill_mask`` masks every column past the query's *relative* row
    index, so at a non-zero ``block_base`` it allows only column 0 — and the band
    is additive, so it cannot widen that back. Using it for decode masks every
    column and fails the comparison for a reason unrelated to the buffer.
    """
    if seqlen == 1 and block_base > 0:
        mask = build_decode_mask(1, FULL_CAPACITY, block_base, 0, dtype=torch.float32)
    else:
        mask = build_prefill_mask(1, seqlen, FULL_CAPACITY, 0, dtype=torch.float32)
    coords = torch.arange(seqlen)[None, :] + block_base
    return add_causal_sliding_window_band(mask, coords, WINDOW)


def test_anchored_decode_matches_the_full_cache_band_path():
    torch.manual_seed(11)
    band = make_sliding_attention(
        Q_HEADS, KV_HEADS, HEAD_DIM, WINDOW, swa_mode=None
    )
    op = copy.deepcopy(band)
    op.swa_mode = "anchored"

    band_k = torch.zeros(1, KV_HEADS, FULL_CAPACITY, HEAD_DIM)
    band_v = torch.zeros(1, KV_HEADS, FULL_CAPACITY, HEAD_DIM)
    # Prefill buffer for a sliding layer: max(sliding_capacity(128), PROMPT).
    op_k = torch.zeros(1, KV_HEADS, PROMPT, HEAD_DIM)
    op_v = torch.zeros(1, KV_HEADS, PROMPT, HEAD_DIM)

    hidden = torch.randn(1, PROMPT, Q_HEADS * HEAD_DIM)
    freqs = identity_freqs(1, PROMPT, HEAD_DIM)
    index = make_cache_index(0, PROMPT)

    band_out, band_k, band_v = band(
        hidden, freqs, _band_mask(PROMPT, 0), band_k, band_v, index
    )
    op_out, op_k, op_v = op(
        hidden, freqs, None, op_k, op_v, index, cache_seqlen=PROMPT, valid_start=[0]
    )
    torch.testing.assert_close(op_out, band_out, rtol=1e-5, atol=1e-6)

    state = SlidingWindowCache.after_prefill(WINDOW, PROMPT, [0])
    op_k, op_v = compact_after_prefill(op_k, op_v, state, PROMPT)
    assert op_k.shape[2] == 192

    shifts = 0
    for step_index in range(STEPS):
        slot = PROMPT + step_index
        token = torch.randn(1, 1, Q_HEADS * HEAD_DIM)
        token_freqs = identity_freqs(1, 1, HEAD_DIM)

        expected, band_k, band_v = band(
            token, token_freqs, _band_mask(1, slot), band_k, band_v,
            make_cache_index(slot, 1),
        )

        step = anchored_step(state, "cpu")
        shifts += int(step.do_shift)
        actual, op_k, op_v = op(
            token,
            token_freqs,
            None,
            op_k,
            op_v,
            step.cache_index,
            cache_seqlen=step.cache_seqlen,
            valid_start=step.valid_start,
            stick_index=step.stick_index,
            do_shift=step.do_shift,
        )
        state.advance()

        torch.testing.assert_close(
            actual, expected, rtol=1e-4, atol=1e-5, msg=f"step {step_index}"
        )

    assert shifts == 1, "70 steps must cross exactly one 64-row shift"
    assert op_k.shape[2] == 192, "the compact buffer must never grow"


def test_anchored_geometry_is_constant_across_steps():
    """The whole point: the integers the op sees never change."""
    state = SlidingWindowCache.after_prefill(1024, 4096, [0])
    seen = set()
    for _ in range(200):
        step = anchored_step(state, "cpu")
        seen.add((step.cache_seqlen, tuple(step.valid_start)))
        state.advance()
    assert seen == {(1088, (0,))}, seen
```

- [ ] **Step 2: Run it to verify it fails**

```bash
cd /mnt/home/spyre/hf-adapters
python3 -m pytest tests/cpu/test_swa_anchored_cpu.py -v 2>&1 | tail -10
```

Expected: `ImportError: cannot import name 'anchored_step'`.

- [ ] **Step 3: Add the per-step geometry helper**

Append to `hf_adapters/swa_attention.py`:

```python
@dataclasses.dataclass(frozen=True)
class AnchoredStep:
    """What one anchored decode step passes into a compiled sliding block.

    ``cache_seqlen`` and ``valid_start`` are the same values at every step of a
    steady-state generation, which is what keeps the block at one compiled graph;
    ``cache_index``, ``stick_index`` and ``do_shift`` carry the per-step part —
    the first two as tensors so the write position never becomes a graph constant,
    the third as a bool because a shift genuinely is a different graph.
    """

    do_shift: bool
    cache_index: torch.Tensor
    stick_index: torch.Tensor
    cache_seqlen: int
    valid_start: list


def anchored_step(state, device):
    """Geometry for the next anchored decode step, rolling the buffer if due.

    Mutates ``state`` when a shift is due — the tensor roll itself happens inside
    the compiled block, driven by ``do_shift``. The caller must call
    ``state.advance()`` after the step completes.
    """
    do_shift = state.needs_shift()
    if do_shift:
        state.shift()
    return AnchoredStep(
        do_shift=do_shift,
        cache_index=torch.tensor([state.write_row], dtype=torch.long).to(device),
        stick_index=torch.tensor([state.stick_offset()], dtype=torch.long).to(device),
        cache_seqlen=state.capacity,
        valid_start=list(state.valid_start),
    )
```

- [ ] **Step 4: Teach `Gemma4Attention` the stick and the shift**

In `hf_adapters/hf_gemma4.py`, extend the import:

```python
from hf_adapters.swa_attention import (
    shift_indices,
    sliding_capacity,
    sliding_window_attention,
)
```

At the end of `Gemma4Attention.__init__`, after `self.swa_mode = swa_mode`:

```python
        # Non-persistent so the roll indices never enter a state_dict. Registered
        # buffers because prepare_for_spyre runs *before* the device move, so these
        # travel with the module; int64 survives the dtype cast, which only touches
        # floating-point buffers.
        if is_sliding and window_size is not None:
            capacity = sliding_capacity(window_size)
            src, dst = shift_indices(capacity, "cpu")
            self.register_buffer("shift_src", src, persistent=False)
            self.register_buffer("shift_dst", dst, persistent=False)
```

Extend `forward`'s signature and add the shift before the KV write:

```python
    def forward(
        self,
        hidden_states,
        selected_freqs,
        attn_mask,
        key_cache,
        value_cache,
        cache_index,
        cache_seqlen=None,
        valid_start=None,
        stick_index=None,
        do_shift=False,
    ):
```

Immediately before the `kv_cache_update(...)` call:

```python
        if do_shift:
            # Roll the compact buffer down one stick so rows [0, anchor) hold the
            # most recent anchor tokens again. index_select into index_copy_ keeps
            # the destination's pinned layout, which slice_scatter would not
            # (torch-spyre#3705); aten::roll has no Spyre lowering at all.
            key_cache.index_copy_(
                2, self.shift_dst, key_cache.index_select(2, self.shift_src)
            )
            value_cache.index_copy_(
                2, self.shift_dst, value_cache.index_select(2, self.shift_src)
            )
```

Replace the sliding branch of the attention call with:

```python
        if self.is_sliding and self.swa_mode is not None:
            # Windowing is an offset plus a length, so attn_mask is unused here and
            # arrives as None; left padding travels as valid_start instead.
            if stick_index is None:
                attn_out = sliding_window_attention(
                    q,
                    key_cache,
                    value_cache,
                    window_size=self.window_size,
                    scale=self.scaling,
                    cache_seqlen=cache_seqlen,
                    buffer_origin=0,
                    valid_start=valid_start,
                )
            else:
                # Anchored decode: present the whole 64-row stick with the real
                # query at stick_index and the rest zeroed, so seqlen_q is 64 at
                # every step and the op compiles once. The padding rows attend
                # their own windows and are thrown away; index_copy/index_select
                # with a tensor index keeps the offset out of the graph.
                stick = torch.zeros(
                    (bsz, self.num_q_heads, BLOCK_SIZE, self.head_dim),
                    dtype=q.dtype,
                    device=q.device,
                )
                stick = stick.index_copy(2, stick_index, q)
                attn_full = sliding_window_attention(
                    stick,
                    key_cache,
                    value_cache,
                    window_size=self.window_size,
                    scale=self.scaling,
                    cache_seqlen=cache_seqlen,
                    buffer_origin=0,
                    valid_start=valid_start,
                )
                attn_out = attn_full.index_select(2, stick_index)
        else:
```

Import `BLOCK_SIZE` alongside the other `hf_common` names in that module's import
block.

- [ ] **Step 5: Thread the two new arguments through `Gemma4Block`**

Add `stick_index=None, do_shift=False` to `Gemma4Block.forward`'s signature and
pass them to `self.self_attn(...)` after `valid_start`.

- [ ] **Step 6: Drive it from `_run_blocks_over_embeds`**

Add the import:

```python
from hf_adapters.swa_attention import SlidingWindowCache, anchored_step, compact_after_prefill
```

Replace the geometry block written in Task 5 with:

```python
    swa_mode = getattr(model, "_spyre_swa_mode", None)
    if masks is not None:
        # A caller supplying its own masks is the unified VLM adapter, whose
        # bidirectional vision overlay *widens* attention. The op cannot express
        # that (is_causal=False raises, and an additive mask only ever removes), so
        # the VLM keeps the band-masked path.
        swa_mode = None
    bsz, seq_len = h.shape[0], h.shape[1]
    # The scalar read syncs from the device; deliberately not optimized, and needed
    # by the op path as well as by the band. Once per step, not per layer.
    block_base = int(cache_index[0]) if masks is None or swa_mode else 0

    if masks is None:
        masks = _build_layer_masks(
            model, attn_mask, seq_len, bsz, block_base, sliding_band=swa_mode is None
        )

    if seq_len > 1:
        # A prefill call starts a new generation: drop any state a previous
        # generate() on this model left behind.
        model._spyre_swa_state = None
    state = getattr(model, "_spyre_swa_state", None)

    if swa_mode and state is not None:
        # Anchored decode: constant geometry, per-step position in tensors.
        step = anchored_step(state, cache_index.device)
        swa_args = {
            "cache_seqlen": step.cache_seqlen,
            "valid_start": step.valid_start,
            "stick_index": step.stick_index,
            "do_shift": step.do_shift,
        }
        sliding_index = step.cache_index
    elif swa_mode:
        # Prefill (or the phase-1 harness): the window is read out of the
        # prompt-sized buffer at its true position.
        swa_args = {
            "cache_seqlen": block_base + seq_len,
            "valid_start": _swa_valid_start(model, bsz),
        }
        sliding_index = cache_index
    else:
        swa_args = {}
        sliding_index = cache_index

    backbone_layers = backbone.layers
    for i, compiled_block in enumerate(model._spyre_compiled_blocks):
        lt = cfg.layer_types[i]
        is_sliding = lt == "sliding_attention"
        h, key_caches[i], value_caches[i] = compiled_block(
            h,
            freqs[lt],
            masks[lt],
            key_caches[i],
            value_caches[i],
            sliding_index if is_sliding else cache_index,
            backbone_layers[i].layer_scalar,
            **(swa_args if is_sliding else {}),
        )

    if swa_mode == "anchored":
        if state is None and seq_len > 1:
            # Prefill just finished: compact every sliding layer down to the
            # anchored buffer and let the prompt-sized allocations go.
            state = SlidingWindowCache.after_prefill(
                cfg.sliding_window, seq_len, _swa_valid_start(model, bsz)
            )
            for i, layer_type in enumerate(cfg.layer_types):
                if layer_type == "sliding_attention":
                    key_caches[i], value_caches[i] = compact_after_prefill(
                        key_caches[i], value_caches[i], state, seq_len
                    )
            model._spyre_swa_state = state
        elif state is not None:
            state.advance()
```

- [ ] **Step 7: Make anchored the default and size the sliding caches**

At module level in `hf_gemma4.py`:

```python
def _gemma4_kv_capacity(layer_types, window_size, layer_index, padded_prompt_len,
                        max_cache_len):
    """Rows to allocate for one layer's KV cache.

    Global layers hold the whole generation. Sliding layers hold only what prefill
    needs — one window buffer, or the prompt if it is longer — because
    ``_run_blocks_over_embeds`` compacts them to ``sliding_capacity(window_size)``
    the moment prefill ends. A module-level function bound with
    ``functools.partial`` rather than a closure, so a prepared model stays
    picklable.
    """
    if layer_types[layer_index] != "sliding_attention":
        return max_cache_len
    return max(sliding_capacity(window_size), padded_prompt_len)
```

In `prepare_text_decoder_for_spyre`, just before the `prepare_gemma4_blocks` call:

```python
    # The sliding-window op path is the default; set model._spyre_swa_mode = None
    # before prepare to fall back to the band-masked SDPA, or "phase1" for the
    # full-cache harness.
    if not hasattr(model, "_spyre_swa_mode"):
        model._spyre_swa_mode = "anchored"
    if model._spyre_swa_mode:
        model._spyre_kv_capacity = functools.partial(
            _gemma4_kv_capacity, tuple(cfg.layer_types), cfg.sliding_window
        )
```

and add `import functools` at the top of the module.

- [ ] **Step 8: Stash the left padding in `generate`**

In `hf_adapters/hf_common.py`, in `generate`, directly after the
`pad_and_position(...)` call:

```python
    # Per-layer cache management needs the left padding: a sliding-window layer
    # cannot mask pad columns with an attention mask (see swa_attention).
    model._spyre_prompt_offsets = prompt_offsets
```

- [ ] **Step 9: Run the tests to verify they pass**

```bash
cd /mnt/home/spyre/hf-adapters
python3 -m pytest tests/cpu/test_swa_anchored_cpu.py -v 2>&1 | tail -20
```

Expected: 2 passed. If step-by-step comparison fails at **step 0**, the compaction
is misaligned (check `test_compaction_keeps_the_newest_rows_right_aligned_at_the_anchor`
still passes). If it fails at **step 64**, the shift is wrong — print
`op_k[0, 0, :, 0]` either side of the shift and confirm rows `[64, 192)` moved to
`[0, 128)`.

- [ ] **Step 10: Run the CPU regression suites**

```bash
cd /mnt/home/spyre/hf-adapters
python3 -m pytest tests/cpu/ -x -q 2>&1 | tail -20
```

Expected: no new failures. Gemma 4's CPU accuracy test now runs the anchored path
by default, so this is where a default-flip mistake surfaces.

- [ ] **Step 11: Commit**

```bash
cd /mnt/home/spyre/hf-adapters
git add hf_adapters/swa_attention.py hf_adapters/hf_gemma4.py hf_adapters/hf_common.py \
  tests/cpu/test_swa_anchored_cpu.py
git commit -s -m "feat(gemma4): anchored compact sliding-window KV buffer"
```

---

### Task 9: phase 2 on device

**Files:**
- Modify: `tests/spyre/test_swa_layer_ab_spyre.py` (append)

**Interfaces:**
- Consumes: Tasks 6, 7, 8.
- Produces: nothing.

- [ ] **Step 1: Write the test**

Append to `tests/spyre/test_swa_layer_ab_spyre.py`:

```python
def test_anchored_decode_matches_band_mask_across_a_shift():
    """The shipped geometry, on device, over a 64-row roll.

    The roll reads a cache it then writes (index_select into index_copy_ on the
    same tensor). If Inductor fuses those so the read sees already-overwritten
    rows, this is the test that catches it — the CPU lane cannot, and the
    documented fallback is a pair of ping-pong buffers.
    """
    torch._dynamo.reset()
    window, prompt, steps = 128, 256, 70
    capacity, full_capacity = 192, 384

    band = make_sliding_attention(
        Q_HEADS, KV_HEADS, HEAD_DIM, window, swa_mode=None, dtype=DTYPE
    ).to("spyre")
    op = copy.deepcopy(band)
    op.swa_mode = "anchored"
    reference = make_sliding_attention(
        Q_HEADS, KV_HEADS, HEAD_DIM, window, swa_mode=None, dtype=torch.float32
    )

    band_k, band_v = _spyre_caches(full_capacity)
    op_k, op_v = _spyre_caches(prompt)
    ref_k, ref_v = _cpu_caches(full_capacity)

    torch.manual_seed(5)
    hidden = torch.randn(1, prompt, Q_HEADS * HEAD_DIM, dtype=DTYPE).to("spyre")
    freqs = identity_freqs(1, prompt, HEAD_DIM, dtype=DTYPE).to("spyre")
    index = make_cache_index(0, prompt, "spyre")

    mask = build_prefill_mask(1, prompt, full_capacity, 0, dtype=DTYPE)
    mask = add_causal_sliding_window_band(
        mask, torch.arange(prompt)[None, :], window
    ).to("spyre")

    compiled_band = torch.compile(band, dynamic=False)
    compiled_op = torch.compile(op, dynamic=False)
    with torch.no_grad():
        _, band_k, band_v = compiled_band(
            hidden, freqs, mask, band_k, band_v, index
        )
        _, op_k, op_v = compiled_op(
            hidden, freqs, None, op_k, op_v, index,
            cache_seqlen=prompt, valid_start=[0],
        )
        _, ref_k, ref_v = _run_cpu32(
            reference,
            hidden.to("cpu"),
            freqs.to("cpu"),
            mask.to("cpu"),
            ref_k,
            ref_v,
            make_cache_index(0, prompt),
        )

    state = SlidingWindowCache.after_prefill(window, prompt, [0])
    op_k, op_v = compact_after_prefill(op_k, op_v, state, prompt)
    assert op_k.shape[2] == capacity

    shifts = 0
    for step_index in range(steps):
        slot = prompt + step_index
        token = torch.randn(1, 1, Q_HEADS * HEAD_DIM, dtype=DTYPE).to("spyre")
        token_freqs = identity_freqs(1, 1, HEAD_DIM, dtype=DTYPE).to("spyre")
        # build_decode_mask for a one-token step at a non-zero position; see the
        # note in the decode A/B above.
        band_mask = build_decode_mask(1, full_capacity, slot, 0, dtype=DTYPE)
        band_mask = add_causal_sliding_window_band(
            band_mask, torch.tensor([[slot]]), window
        ).to("spyre")

        step = anchored_step(state, "spyre")
        shifts += int(step.do_shift)
        with torch.no_grad():
            expected, band_k, band_v = compiled_band(
                token, token_freqs, band_mask, band_k, band_v,
                make_cache_index(slot, 1, "spyre"),
            )
            actual, op_k, op_v = compiled_op(
                token,
                token_freqs,
                None,
                op_k,
                op_v,
                step.cache_index,
                cache_seqlen=step.cache_seqlen,
                valid_start=step.valid_start,
                stick_index=step.stick_index,
                do_shift=step.do_shift,
            )
            # The float32 twin keeps its own full-length cache through the same
            # token stream, so every step is measured against truth rather than
            # against the other fp16 path. See _assert_no_worse.
            ref_out, ref_k, ref_v = _run_cpu32(
                reference,
                token.to("cpu"),
                token_freqs.to("cpu"),
                band_mask.to("cpu"),
                ref_k,
                ref_v,
                make_cache_index(slot, 1),
            )
        state.advance()
        _assert_no_worse(actual.to("cpu").float(), expected.to("cpu").float(), ref_out)

    assert shifts == 1
```

Add to that file's imports:

```python
from hf_adapters.swa_attention import (
    SlidingWindowCache,
    anchored_step,
    compact_after_prefill,
)
```

- [ ] **Step 2: Run it**

```bash
source /mnt/home/spyre/torch-spyre-docs/scripts/dev-env.sh
cd /mnt/home/spyre/hf-adapters
python3 -m pytest -s tests/spyre/test_swa_layer_ab_spyre.py -v 2>&1 | tail -40
```

Expected: 5 passed. Diagnosis:

- **Fails only at step 64** (`shift=True`) → the self-referential roll. Confirm by
  reading `op_k[0, 0, :, 0]` before and after that step. Fix: allocate a second
  compact buffer per sliding layer and ping-pong (copy survivors across, swap
  references) at 2x steady-state memory, replacing the in-place roll in
  `Gemma4Attention.forward`.
- **Every decode step is wrong but prefill is right** → the stick offset. Check
  that `stick_index` reaches both the `index_copy` and the `index_select`, and that
  `state.advance()` runs exactly once per step.

- [ ] **Step 3: Record the graph count**

The one-graph claim needs evidence, not assertion.

```bash
cd /mnt/home/spyre/hf-adapters
TORCH_LOGS=recompiles python3 -m pytest -s \
  tests/spyre/test_swa_layer_ab_spyre.py -k anchored 2>&1 | grep -ci recompil
```

Expected: at most 2 recompiles for the op module across all 70 steps (the shift
branch is the second). If it is ~70, the geometry is not constant — dump
`step.cache_seqlen` and `step.valid_start` per step and compare against
`test_anchored_geometry_is_constant_across_steps`. Record the number in the commit
message.

- [ ] **Step 4: Commit**

```bash
cd /mnt/home/spyre/hf-adapters
git add tests/spyre/test_swa_layer_ab_spyre.py
git commit -s -m "test(gemma4): device A/B of the anchored buffer across a shift"
```

---

### Task 10: the Gemma 3 control

Gemma 4 12B's device token-compare is red for unrelated reasons, so it cannot tell
us the replacement works end-to-end. Gemma 3 1B can: same band-mask pattern, same
`head_dim=256`, `sliding_window=512`, cached locally, and **not** in
`NON_BLOCKING_CAUSAL_MODELS`. Porting the path there and requiring it to stay green
is the end-to-end evidence.

This task also has a second consumer arrive, so the sliding branch gets extracted
rather than copied.

**Files:**
- Modify: `hf_adapters/swa_attention.py` (append `attend_sliding`,
  `roll_compact_cache`)
- Modify: `hf_adapters/hf_gemma4.py` (use the extracted helpers)
- Modify: `hf_adapters/hf_gemma3.py` (`_make_compiled_block`,
  `_run_blocks_over_embeds`-equivalent driver at line 260, `prepare_for_spyre`)

**Interfaces:**
- Consumes: Tasks 3, 7, 8.
- Produces:
  - `swa_attention.attend_sliding(query, key_cache, value_cache, *, window_size,
    scale, cache_seqlen, valid_start=None, stick_index=None) -> Tensor`
  - `swa_attention.roll_compact_cache(key_cache, value_cache, shift_src, shift_dst)
    -> (Tensor, Tensor)`

- [ ] **Step 1: Extract the two helpers**

Append to `hf_adapters/swa_attention.py`:

```python
def attend_sliding(
    query,
    key_cache,
    value_cache,
    *,
    window_size,
    scale,
    cache_seqlen,
    valid_start=None,
    stick_index=None,
):
    """One sliding layer's attention: the plain call, or the anchored stick.

    ``stick_index`` selects anchored decode: the single query row is placed at that
    index of a zeroed 64-row stick, so ``seqlen_q`` is 64 at every step and the op
    compiles once. The padding rows attend their own windows and are discarded;
    they cannot be fully masked, so they cannot produce a NaN that spreads. The
    index travels as a **tensor** so the position never becomes a graph constant —
    the same reason ``kv_cache_update`` takes a tensor ``cache_index``.
    """
    if stick_index is None:
        return sliding_window_attention(
            query,
            key_cache,
            value_cache,
            window_size=window_size,
            scale=scale,
            cache_seqlen=cache_seqlen,
            buffer_origin=0,
            valid_start=valid_start,
        )
    batch, num_heads, _, head_dim = query.shape
    stick = torch.zeros(
        (batch, num_heads, BLOCK_SIZE, head_dim),
        dtype=query.dtype,
        device=query.device,
    )
    stick = stick.index_copy(2, stick_index, query)
    attended = sliding_window_attention(
        stick,
        key_cache,
        value_cache,
        window_size=window_size,
        scale=scale,
        cache_seqlen=cache_seqlen,
        buffer_origin=0,
        valid_start=valid_start,
    )
    return attended.index_select(2, stick_index)


def valid_start_for(model, batch_size):
    """First attendable cache column per sequence -- ``generate``'s left padding.

    Moved here from ``hf_gemma4`` now that both Gemma adapters need it.
    ``generate`` stashes ``_spyre_prompt_offsets``; a caller driving a forward
    directly (the layer tests) has none.
    """
    offsets = getattr(model, "_spyre_prompt_offsets", None)
    if offsets is None:
        return [0] * batch_size
    return [int(offset) for offset in offsets]


def roll_compact_cache(key_cache, value_cache, shift_src, shift_dst):
    """Roll a compact buffer down one stick, in place, restoring the invariant.

    ``index_select`` into ``index_copy_`` keeps the destination's pinned device
    layout, which ``slice_scatter`` would not (torch-spyre#3705); ``aten::roll`` has
    no Spyre lowering at all. The read and the write touch the same tensor, so the
    device A/B across a shift is the test that matters here.
    """
    key_cache.index_copy_(2, shift_dst, key_cache.index_select(2, shift_src))
    value_cache.index_copy_(2, shift_dst, value_cache.index_select(2, shift_src))
    return key_cache, value_cache
```

- [ ] **Step 2: Switch Gemma 4 onto them**

In `Gemma4Attention.forward`, replace the `do_shift` block with:

```python
        if do_shift:
            key_cache, value_cache = roll_compact_cache(
                key_cache, value_cache, self.shift_src, self.shift_dst
            )
```

and the whole sliding branch with:

```python
        if self.is_sliding and self.swa_mode is not None:
            # Windowing is an offset plus a length, so attn_mask is unused here and
            # arrives as None; left padding travels as valid_start instead.
            attn_out = attend_sliding(
                q,
                key_cache,
                value_cache,
                window_size=self.window_size,
                scale=self.scaling,
                cache_seqlen=cache_seqlen,
                valid_start=valid_start,
                stick_index=stick_index,
            )
        else:
```

Delete `_swa_valid_start` from `hf_gemma4.py` and call
`swa_attention.valid_start_for(model, bsz)` in its place. Update the import to
`attend_sliding, roll_compact_cache, shift_indices, sliding_capacity, valid_start_for`,
dropping `sliding_window_attention` now that nothing in the module calls it
directly.

- [ ] **Step 3: Run the Gemma 4 tests to confirm the extraction changed nothing**

```bash
cd /mnt/home/spyre/hf-adapters
python3 -m pytest tests/cpu/test_swa_anchored_cpu.py tests/cpu/test_swa_gemma4_layer.py -q 2>&1 | tail -8
```

Expected: 5 passed. Commit this refactor on its own:

```bash
git add hf_adapters/swa_attention.py hf_adapters/hf_gemma4.py
git commit -s -m "refactor(swa): extract the shared sliding branch and buffer roll"
```

- [ ] **Step 4: Wire Gemma 3's block**

Gemma 3's block is a closure, not an `nn.Module`, so its roll indices cannot be
registered buffers — capture them at prepare time, already on the device, instead:

In `hf_adapters/hf_gemma3.py`, extend `_make_compiled_block`:

```python
def _make_compiled_block(
    layer,
    num_q_heads,
    num_kv_heads,
    head_dim,
    is_sliding=False,
    window_size=None,
    swa_mode=None,
):
```

after the captured modules:

```python
    # Captured on the device rather than registered as buffers: this block is a
    # closure, so a CPU tensor here would never be moved by model.to(DEVICE).
    shift_src, shift_dst = (
        shift_indices(sliding_capacity(window_size), DEVICE)
        if is_sliding and swa_mode and window_size is not None
        else (None, None)
    )
```

extend `block_forward`'s signature with the same four keyword arguments Gemma 4
uses, and replace its `F.scaled_dot_product_attention` call:

```python
    def block_forward(
        hidden_states,
        selected_freqs,
        attn_mask,
        key_cache,
        value_cache,
        cache_index,
        cache_seqlen=None,
        valid_start=None,
        stick_index=None,
        do_shift=False,
    ):
```

```python
        if do_shift:
            key_cache, value_cache = roll_compact_cache(
                key_cache, value_cache, shift_src, shift_dst
            )

        key_cache, value_cache = kv_cache_update(
            k,
            v,
            key_cache,
            value_cache,
            cache_index,
        )

        if is_sliding and swa_mode is not None:
            attn_out = attend_sliding(
                q,
                key_cache,
                value_cache,
                window_size=window_size,
                scale=scaling,
                cache_seqlen=cache_seqlen,
                valid_start=valid_start,
                stick_index=stick_index,
            )
        else:
            attn_out = F.scaled_dot_product_attention(
                q,
                key_cache,
                value_cache,
                attn_mask=attn_mask,
                dropout_p=0.0,
                scale=scaling,
                enable_gqa=True,
            )
```

with the imports:

```python
from hf_adapters.hf_common import DEVICE
from hf_adapters.swa_attention import (
    SlidingWindowCache,
    anchored_step,
    attend_sliding,
    compact_after_prefill,
    roll_compact_cache,
    shift_indices,
    sliding_capacity,
    valid_start_for,
)
```

- [ ] **Step 5: Wire Gemma 3's driver and prepare**

Gemma 3's driver is `_run_backbone_forward` (line 246), which builds `masks`
inline and loops the blocks. Two Gemma-3-specific points: its `sliding_window` is
512, so `sliding_capacity` is 576; and the **bidirectional** variant
(`use_bidirectional_attention`, the embedder path) must keep the band-masked SDPA,
because the op raises `Unsupported` for `is_causal=False` and a bidirectional
overlay *widens* attention, which no additive mask can express.

Replace everything in that function from `bsz, seq_len = ...` down to the end of
the block loop with:

```python
    swa_mode = getattr(model, "_spyre_swa_mode", None)
    if getattr(cfg, "use_bidirectional_attention", False):
        # The op is causal-only. The embedder path keeps the symmetric band.
        swa_mode = None

    bsz, seq_len = input_ids.shape[0], input_ids.shape[1]
    # These reads sync a scalar back from the device: fine here and deliberately
    # not optimized (once per step, not per layer, in eager code outside the
    # compiled block). See the long note this replaces.
    block_base = int(cache_index[0])
    query_coords = (torch.arange(seq_len)[None, :] + block_base).expand(bsz, seq_len)

    if swa_mode:
        # The op derives its window from cache_seqlen and window_size, so the
        # sliding entry is None: a mis-wired layer then fails visibly instead of
        # quietly attending the pad columns behind a stale band.
        masks = {"full_attention": attn_mask, "sliding_attention": None}
    elif getattr(cfg, "use_bidirectional_attention", False):
        masks = {
            "full_attention": attn_mask,
            "sliding_attention": _add_bidirectional_sliding_window_band(
                attn_mask, query_coords, cfg.sliding_window
            ),
        }
    else:
        masks = {
            "full_attention": attn_mask,
            "sliding_attention": add_causal_sliding_window_band(
                attn_mask, query_coords, cfg.sliding_window
            ),
        }

    if seq_len > 1:
        # A prefill call starts a new generation: drop any state a previous
        # generate() on this model left behind.
        model._spyre_swa_state = None
    state = getattr(model, "_spyre_swa_state", None)

    if swa_mode and state is not None:
        # Anchored decode: constant geometry, per-step position in tensors.
        step = anchored_step(state, cache_index.device)
        swa_args = {
            "cache_seqlen": step.cache_seqlen,
            "valid_start": step.valid_start,
            "stick_index": step.stick_index,
            "do_shift": step.do_shift,
        }
        sliding_index = step.cache_index
    elif swa_mode:
        # Prefill: the window is read out of the prompt-sized buffer at its true
        # position. generate stashes the left padding; a caller driving the forward
        # directly has none.
        swa_args = {
            "cache_seqlen": block_base + seq_len,
            "valid_start": valid_start_for(model, bsz),
        }
        sliding_index = cache_index
    else:
        swa_args = {}
        sliding_index = cache_index

    for i, compiled_block in enumerate(model._spyre_compiled_blocks):
        lt = cfg.layer_types[i]
        is_sliding = lt == "sliding_attention"
        h, key_caches[i], value_caches[i] = compiled_block(
            h,
            freqs[lt],
            masks[lt],
            key_caches[i],
            value_caches[i],
            sliding_index if is_sliding else cache_index,
            **(swa_args if is_sliding else {}),
        )

    if swa_mode == "anchored":
        if state is None and seq_len > 1:
            # Prefill just finished: compact every sliding layer down to the
            # anchored buffer and let the prompt-sized allocations go.
            state = SlidingWindowCache.after_prefill(
                cfg.sliding_window, seq_len, swa_args["valid_start"]
            )
            for i, layer_type in enumerate(cfg.layer_types):
                if layer_type == "sliding_attention":
                    key_caches[i], value_caches[i] = compact_after_prefill(
                        key_caches[i], value_caches[i], state, seq_len
                    )
            model._spyre_swa_state = state
        elif state is not None:
            state.advance()
```

In `prepare_for_spyre`, add the same two pieces Gemma 4 has. First a module-level
capacity hook:

```python
def _gemma3_kv_capacity(layer_types, window_size, layer_index, padded_prompt_len,
                        max_cache_len):
    """Rows to allocate for one layer's KV cache.

    Global layers hold the whole generation. Sliding layers hold only what prefill
    needs, because ``_run_backbone_forward`` compacts them to
    ``sliding_capacity(window_size)`` the moment prefill ends. A module-level
    function bound with ``functools.partial`` rather than a closure, so a prepared
    model stays picklable.
    """
    if layer_types[layer_index] != "sliding_attention":
        return max_cache_len
    return max(sliding_capacity(window_size), padded_prompt_len)
```

then, in `prepare_for_spyre` before the blocks are built (`import functools` at the
top of the module):

```python
    cfg = text_config(model.config)
    if not hasattr(model, "_spyre_swa_mode"):
        model._spyre_swa_mode = "anchored"
    if getattr(cfg, "use_bidirectional_attention", False):
        # Embedder path: the op is causal-only.
        model._spyre_swa_mode = None
    if model._spyre_swa_mode:
        model._spyre_kv_capacity = functools.partial(
            _gemma3_kv_capacity, tuple(cfg.layer_types), cfg.sliding_window
        )
```

and pass the three new arguments where the blocks are built:

```python
        _make_compiled_block(
            layer,
            num_q_heads,
            num_kv_heads,
            head_dim,
            is_sliding=cfg.layer_types[i] == "sliding_attention",
            window_size=cfg.sliding_window,
            swa_mode=model._spyre_swa_mode,
        )
```

matching however that call site currently enumerates the layers (add an index if it
does not already have one).

- [ ] **Step 6: Run the Gemma 3 CPU suites**

```bash
cd /mnt/home/spyre/hf-adapters
python3 -m pytest tests/cpu/ -q -k "gemma3 or gemma_3 or swa" 2>&1 | tail -12
python3 -m pytest tests/cpu/ -x -q 2>&1 | tail -8
```

Expected: no new failures, including the embedder path (which must still take the
band).

- [ ] **Step 7: Run the Gemma 3 device token-compare — the control**

```bash
source /mnt/home/spyre/torch-spyre-docs/scripts/dev-env.sh
cd /mnt/home/spyre/hf-adapters
HF_HOME=/tmp/models/huggingface_cache PYTHONUNBUFFERED=1 \
  python3 -m pytest -s tests/spyre/test_e2e_token_compare_spyre.py \
  -k "gemma-3-1b" -v 2>&1 | tail -40
```

Expected: **PASS** (5/5 tokens), matching its pre-change state. This is the
end-to-end gate. If it fails, capture `max_diff` and which token first diverges,
then compare against the same command with `_spyre_swa_mode = None` forced to
confirm the baseline is still green — a red baseline means the failure is not ours.

- [ ] **Step 8: Commit**

```bash
cd /mnt/home/spyre/hf-adapters
git add hf_adapters/hf_gemma3.py
git commit -s -m "feat(gemma3): anchored sliding-window op path on the causal LM path"
```

---

### Task 11: Gemma 4 end-to-end, non-gating

Gemma 4 12B fails device token-compare 0/5 today and diverges at prefill. The
requirement is **not** that it passes — it is that the SWA replacement introduces no
*new* failure mode, and that the graph count is what the design claims.

**Files:**
- Modify: `docs/superpowers/specs/2026-08-18-gemma4-sliding-window-attention-design.md`
  (append a "Results" section)

**Interfaces:**
- Consumes: everything above.
- Produces: the recorded evidence.

- [ ] **Step 1: Record the baseline with the op path off**

```bash
source /mnt/home/spyre/torch-spyre-docs/scripts/dev-env.sh
cd /mnt/home/spyre/hf-adapters
HF_HOME=/tmp/models/hub PYTHONUNBUFFERED=1 \
  python3 -m pytest -s tests/spyre/test_e2e_token_compare_spyre.py \
  -k "gemma-4-12b" -v 2>&1 | tee /tmp/gemma4_swa_off.log | tail -30
```

Expected: XFAIL (0/5, `max_diff` around 53.75, no NaN), matching the 2026-08-17
record. This is the comparison baseline, not a pass.

To force the op path off for this run, set `model._spyre_swa_mode = None` before
`prepare_for_spyre` — the harness loads through `AutoSpyreModelForCausalLM`, so do
it with an environment-free edit: temporarily change the default in
`prepare_text_decoder_for_spyre` to `None`, run, and revert. Note in the log which
way it ran.

- [ ] **Step 2: Record it with the op path on (the default)**

```bash
cd /mnt/home/spyre/hf-adapters
HF_HOME=/tmp/models/hub PYTHONUNBUFFERED=1 \
  python3 -m pytest -s tests/spyre/test_e2e_token_compare_spyre.py \
  -k "gemma-4-12b" -v 2>&1 | tee /tmp/gemma4_swa_on.log | tail -30
```

Expected: still XFAIL, and specifically:
- **no NaN** in the logits,
- **no `Unsupported`** raised,
- **no hang** (it completes in the same order of time),
- `max_diff` in the same magnitude as the baseline.

Any of those four appearing is a new failure mode and blocks the task.

- [ ] **Step 3: Record the smoke test**

```bash
cd /mnt/home/spyre/hf-adapters
HF_HOME=/tmp/models/hub python3 -m pytest -s \
  tests/spyre/test_e2e_smoke_spyre.py -k "gemma-4-12b" -v 2>&1 | tail -20
```

Expected: XPASS, as before. Remember that smoke XPASS is not correctness — it only
checks the output is non-empty and not all-one-token.

- [ ] **Step 4: Append the results to the spec**

Add a `## Results` section to the design document recording, with dates:
the Task 1 gate outcome at `head_dim=256`; the Task 6 and Task 9 A/B results and
tolerances; the measured recompile count from Task 9 Step 3; Gemma 3 1B's
token-compare before and after; and Gemma 4's before/after `max_diff` with the
explicit statement that it was red before and remains red for an unrelated prefill
divergence. State plainly anything that was **not** measured — in particular that no
latency or memory measurement was taken, so the spec's arithmetic remains
arithmetic.

- [ ] **Step 5: Commit**

```bash
cd /mnt/home/spyre/hf-adapters
git add docs/superpowers/specs/2026-08-18-gemma4-sliding-window-attention-design.md
git commit -s -m "docs(swa): record the sliding-window attention results"
```

---

### Task 12: offer `valid_start` upstream

**Repo:** `/mnt/home/spyre/torch-spyre` (branch `swa-3405-valid-start`)

**Interfaces:**
- Consumes: Tasks 2, 6, 9, 11.
- Produces: a PR (or patch) against `swa-window-roll`.

- [ ] **Step 1: Confirm the branch is clean and the suite is green**

```bash
source /mnt/home/spyre/torch-spyre-docs/scripts/dev-env.sh
cd /mnt/home/spyre/torch-spyre
git status -sb
pre-commit run --all-files 2>&1 | tail -20
SENCORES=1 python3 -m pytest tests/inductor/test_sliding_window_attention.py \
  tests/inductor/test_kv_window.py -q 2>&1 | tail -10
```

Expected: clean tree, hooks pass, all tests pass.

- [ ] **Step 2: Open the stacked PR**

Target `swa-window-roll`, not `main` — #3405 is @abhishekkunuru6-cmyk's PR and this
stacks on it rather than rewriting it.

```bash
cd /mnt/home/spyre/torch-spyre
gh pr create --base swa-window-roll --head swa-3405-valid-start \
  --title "Add valid_start to sliding_window_attention for left-padded prompts" \
  --body "$(cat <<'BODY'
#### What type of PR is this?

- [x] feature

#### What this PR does:

Adds an optional `valid_start` to `spyre::sliding_window_attention`: one logical
column coordinate per batch entry, below which nothing is attended.

A left-padded prompt puts pad K/V inside the window, and windowing-as-offset-plus-
length cannot exclude it. `valid_start` folds `column >= valid_start[b]` into the
band `window_band_mask` already builds on CPU from integers — no device mask
tensor, no per-block slice. `None` or all-zero costs nothing; a uniform threshold
keeps the band broadcast over batch; only a ragged one widens it to
`[B, 1, q_block, buffer_width]`.

#### Which issue(s) this PR is related to:

Stacks on #3405. Related to #3073.

#### Special notes for your reviewer:

- Validation lives in the decomposition, not `rejection_reason`: the batch size is a
  tensor property, and `sliding_window_plan.py` answers placement questions from
  integers alone.
- `block_is_fully_attended`'s skip is suppressed whenever `valid_start` masks
  anything, since the band becomes load-bearing.
- Driven by the hf-adapters Gemma 4 integration, which is what needs it; the
  `head_dim=256` and anchored-stick tests here came out of that work too.

#### Does this PR introduce a user-facing change?

Yes. `torch.ops.spyre.sliding_window_attention(..., valid_start=None)`.
BODY
)"
```

- [ ] **Step 2b: If a stacked PR is unwelcome, hand over a patch instead**

```bash
cd /mnt/home/spyre/torch-spyre
git format-patch swa-window-roll..swa-3405-valid-start -o /tmp/valid-start-patches
ls /tmp/valid-start-patches
```

Attach those to a comment on #3405 with the A/B evidence from Task 11's Results
section.

- [ ] **Step 3: Report**

Summarize for the user: the PR (or patch) link, the A/B results, the recompile
count, Gemma 3's control status, and Gemma 4's unchanged red baseline. Do not claim
a performance improvement — none was measured.

---

## Notes for the executor

- **Task 1 is a gate.** If `head_dim=256` fails, stop. Everything else assumes the
  op is correct at Gemma 4's shapes.
- **Two documented fallbacks**, both already diagnosed in the tasks that would hit
  them: `window_band_mask`'s `-inf` fill (Task 6 Step 2) and the self-referential
  roll (Task 9 Step 2). Neither is speculative — each has a named symptom and a
  named fix.
- **The `-inf` question is real.** `hf_common._mask_fill_value` exists because host
  fp16 `-inf` lands on device as a finite `-3.35e7` and corrupts attention for
  heavily left-padded rows. `window_band_mask` uses `-inf`. Its own tests pass, so
  it is not obviously broken — but Task 6's left-padding test is the first time this
  op sees the conditions that produced the batch>1 Qwen3 bug.
- **Never claim a perf result.** No task measures latency or memory. If a
  measurement is wanted, it is separate work.
