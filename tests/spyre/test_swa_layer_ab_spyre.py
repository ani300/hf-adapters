"""Device A/B: Gemma 4's sliding layer, band-masked SDPA versus the SWA op.

Gemma 4 12B currently fails device token-compare 0/5 and diverges at prefill for
reasons unrelated to sliding-window attention, so end-to-end output cannot answer
whether this replacement is correct. This test can: one layer, identical inputs and
identical cache contents, the two paths compared directly.

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

# Gemma 4 12B's sliding layers, at one eighth the head count so a test fits.
WINDOW = 1024
HEAD_DIM = 256
Q_HEADS = 4
KV_HEADS = 2
DTYPE = torch.float16
RTOL, ATOL = 1e-2, 1e-3


def _caches(batch, capacity):
    """Caches with the pinned device layout the indirect scatter requires."""
    model = FakeKVModel([(KV_HEADS, HEAD_DIM, HEAD_DIM)])
    keys, values = allocate_kv_caches(model, batch, capacity, DTYPE, device="spyre")
    return keys[0], values[0]


def _fill(cache, rows):
    """Write pseudo-random rows into [0, rows) and leave the tail zeroed."""
    torch.manual_seed(7)
    payload = torch.randn(
        cache.shape[0], cache.shape[1], rows, cache.shape[3], dtype=DTYPE
    )
    cache.index_copy_(2, make_cache_index(0, rows, "spyre"), payload.to("spyre"))
    return cache


def _run(module, hidden, freqs, mask, key_cache, value_cache, index, **kwargs):
    compiled = torch.compile(module, dynamic=False)
    with torch.no_grad():
        out, _, _ = compiled(
            hidden, freqs, mask, key_cache, value_cache, index, **kwargs
        )
    return out.to("cpu").float()


@pytest.mark.parametrize("seqlen_q", [64, 512])
def test_prefill_op_matches_band_mask(seqlen_q):
    torch._dynamo.reset()
    capacity = 1088
    hidden = torch.randn(1, seqlen_q, Q_HEADS * HEAD_DIM, dtype=DTYPE).to("spyre")
    freqs = identity_freqs(1, seqlen_q, HEAD_DIM, dtype=DTYPE).to("spyre")
    index = make_cache_index(0, seqlen_q, "spyre")

    band = make_sliding_attention(
        Q_HEADS, KV_HEADS, HEAD_DIM, WINDOW, swa_mode=None, dtype=DTYPE
    ).to("spyre")
    op = copy.deepcopy(band)
    op.swa_mode = "phase1"

    mask = build_prefill_mask(1, seqlen_q, capacity, 0, dtype=DTYPE)
    coords = torch.arange(seqlen_q)[None, :]
    mask = add_causal_sliding_window_band(mask, coords, WINDOW).to("spyre")

    expected = _run(band, hidden, freqs, mask, *_caches(1, capacity), index)
    actual = _run(
        op,
        hidden,
        freqs,
        None,
        *_caches(1, capacity),
        index,
        cache_seqlen=seqlen_q,
        valid_start=[0],
    )
    torch.testing.assert_close(actual, expected, rtol=RTOL, atol=ATOL)


def test_decode_op_matches_band_mask():
    """One token at slot 512, reading the 1024 columns behind it."""
    torch._dynamo.reset()
    capacity, written = 1088, 512
    hidden = torch.randn(1, 1, Q_HEADS * HEAD_DIM, dtype=DTYPE).to("spyre")
    freqs = identity_freqs(1, 1, HEAD_DIM, dtype=DTYPE).to("spyre")
    index = make_cache_index(written, 1, "spyre")

    band = make_sliding_attention(
        Q_HEADS, KV_HEADS, HEAD_DIM, WINDOW, swa_mode=None, dtype=DTYPE
    ).to("spyre")
    op = copy.deepcopy(band)
    op.swa_mode = "phase1"

    # build_decode_mask, not build_prefill_mask: a one-token step at a non-zero
    # cache position must allow every real column up to that position, whereas
    # build_prefill_mask masks everything past the query's *relative* row index
    # and so allows only column 0. The band is combined additively, so an
    # over-restrictive base cannot be widened back — every column ends up masked
    # and the comparison fails for a reason unrelated to the wiring.
    mask = build_decode_mask(1, capacity, written, 0, dtype=DTYPE)
    mask = add_causal_sliding_window_band(
        mask, torch.tensor([[written]]), WINDOW
    ).to("spyre")

    band_k, band_v = _caches(1, capacity)
    op_k, op_v = _caches(1, capacity)
    expected = _run(
        band, hidden, freqs, mask, _fill(band_k, written), _fill(band_v, written), index
    )
    actual = _run(
        op,
        hidden,
        freqs,
        None,
        _fill(op_k, written),
        _fill(op_v, written),
        index,
        cache_seqlen=written + 1,
        valid_start=[0],
    )
    torch.testing.assert_close(actual, expected, rtol=RTOL, atol=ATOL)


def test_left_padding_op_matches_band_mask():
    """17 pad columns inside the window: valid_start must equal the mask.

    This is the case the batch>1 Qwen3 decode bug lived in, so it is also the
    canary for window_band_mask's -inf fill (see hf_common._mask_fill_value).

    Compared over rows ``>= offset`` only, plus finiteness everywhere. Rows below the
    threshold have their whole window excluded, and the two paths disagree there by
    construction — the op spreads weight over its window buffer, the band path over
    the full cache's pad columns. Those rows are padding whose outputs are discarded;
    what matters is that they stay finite, since a non-finite value would reach the
    next layer's KV cache.
    """
    torch._dynamo.reset()
    capacity, seqlen_q, offset = 1088, 512, 17
    hidden = torch.randn(1, seqlen_q, Q_HEADS * HEAD_DIM, dtype=DTYPE).to("spyre")
    freqs = identity_freqs(1, seqlen_q, HEAD_DIM, dtype=DTYPE).to("spyre")
    index = make_cache_index(0, seqlen_q, "spyre")

    band = make_sliding_attention(
        Q_HEADS, KV_HEADS, HEAD_DIM, WINDOW, swa_mode=None, dtype=DTYPE
    ).to("spyre")
    op = copy.deepcopy(band)
    op.swa_mode = "phase1"

    mask = build_prefill_mask(1, seqlen_q, capacity, offset, dtype=DTYPE)
    coords = torch.arange(seqlen_q)[None, :]
    mask = add_causal_sliding_window_band(mask, coords, WINDOW).to("spyre")

    expected = _run(band, hidden, freqs, mask, *_caches(1, capacity), index)
    actual = _run(
        op,
        hidden,
        freqs,
        None,
        *_caches(1, capacity),
        index,
        cache_seqlen=seqlen_q,
        valid_start=[offset],
    )
    torch.testing.assert_close(
        actual[:, :, offset:], expected[:, :, offset:], rtol=RTOL, atol=ATOL
    )
    assert torch.isfinite(actual).all(), "fully-masked pad rows must stay finite"
