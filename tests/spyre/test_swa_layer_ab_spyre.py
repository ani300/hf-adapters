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
import math

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


def _fp16_floor(reference, terms):
    """Analytic fp16 noise floor for a ``terms``-long reduction at this output scale.

    ``sqrt(n) * eps * scale`` is the usual random-walk estimate of accumulated
    rounding in an n-term fp16 reduction. It exists because a pure ratio test
    demands the op be more accurate than fp16 allows whenever SDPA happens to be
    unusually accurate: measured at decode, the band path lands at 0.010 where
    fp16's floor for the same reduction is 0.059, so requiring 2x0.010 asks the op
    to beat its own arithmetic.
    """
    scale = reference.abs().max().item()
    return math.sqrt(terms) * torch.finfo(torch.float16).eps * scale


def _assert_no_worse(op_out, band_out, reference, terms, rows_from=0):
    """The op must beat the band path, or come within fp16's floor — whichever is
    more permissive.

    Both halves are load-bearing. The ratio catches an op that is materially less
    accurate than the path it replaces. The floor keeps the test from failing an op
    that is merely fp16-accurate on a case where SDPA got lucky.

    All three tensors are ``[B, L, hidden]`` — the layer returns its ``o_proj``
    output, not per-head attention — so ``rows_from`` slices dim 1, the query rows.
    It drops leading rows whose attention is undefined (see the left-padding test).
    ``terms`` is how many KV columns the op reduces over.
    """
    op_out = op_out[:, rows_from:]
    band_out = band_out[:, rows_from:]
    reference = reference[:, rows_from:]
    assert torch.isfinite(op_out).all(), "op output must be finite"
    band_error = (band_out - reference).abs().max().item()
    op_error = (op_out - reference).abs().max().item()
    floor = _fp16_floor(reference, terms)
    allowed = max(ERROR_RATIO * band_error, floor) + ERROR_FLOOR
    print(
        f"\n  op {op_error:.4f} vs band {band_error:.4f} "
        f"(allowed {allowed:.4f} = max({ERROR_RATIO}x band, fp16 floor {floor:.4f}) "
        f"+ {ERROR_FLOOR}, ref scale {reference.abs().max().item():.3f})"
    )
    assert op_error <= allowed, (
        f"op error {op_error:.4f} exceeds {allowed:.4f}: neither within "
        f"{ERROR_RATIO}x the band path's own error {band_error:.4f} nor within "
        f"fp16's floor {floor:.4f} for a {terms}-term reduction"
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
    # A 64-row query block spans WINDOW + 64 columns, capped by the allocation.
    _assert_no_worse(op_out, band_out, ref_out, terms=min(capacity, WINDOW + 64))


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
    # One query row has no stagger, so it spans WINDOW + 1 columns.
    _assert_no_worse(op_out, band_out, ref_out, terms=min(capacity, WINDOW + 1))


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
    _assert_no_worse(
        op_out, band_out, ref_out, terms=min(capacity, WINDOW + 64), rows_from=offset
    )
