# Copyright 2025 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
    BLOCK_SIZE,
    add_causal_sliding_window_band,
    build_decode_mask,
    build_prefill_mask,
    make_cache_index,
)
from hf_adapters.hf_gemma4 import _run_blocks_over_embeds
from hf_adapters.swa_attention import (
    SlidingWindowCache,
    anchored_step,
    compact_after_prefill,
    roll_compact_buffer,
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
    band = make_sliding_attention(Q_HEADS, KV_HEADS, HEAD_DIM, WINDOW, swa_mode=None)
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
            token,
            token_freqs,
            _band_mask(1, slot),
            band_k,
            band_v,
            make_cache_index(slot, 1),
        )

        step = anchored_step(state, "cpu")
        shifts += int(step.do_shift)
        if step.do_shift:
            op_k, op_v = roll_compact_buffer(op_k, op_v)
        actual, op_k, op_v = op(
            token,
            token_freqs,
            None,
            op_k,
            op_v,
            step.cache_index,
            cache_seqlen=step.cache_seqlen,
            valid_start=step.valid_start,
        )
        state.advance()

        torch.testing.assert_close(
            actual, expected, rtol=1e-4, atol=1e-5, msg=f"step {step_index}"
        )

    assert shifts == 1, "70 steps must cross exactly one 64-row shift"
    assert op_k.shape[2] == 192, "the compact buffer must never grow"


def test_anchored_geometry_stays_in_a_bounded_reused_set():
    """The stick-free trade: cache_seqlen sweeps a stick's 64 values and no more.

    Dropping the 64-row stick means seqlen_q is 1 and cache_seqlen is the query
    row's own coordinate, so it is no longer pinned. What the compact buffer still
    guarantees is that it stays *bounded* — anchor+1 through capacity, 64 distinct
    values reused for the whole generation, with valid_start at 0 throughout — so
    decode compiles a bounded, reused set of graphs rather than one per position.
    """
    window, prompt = 1024, 4096
    state = SlidingWindowCache.after_prefill(window, prompt, [0])
    anchor, capacity = state.anchor, state.capacity
    seen = set()
    for _ in range(200):
        step = anchored_step(state, "cpu")
        assert isinstance(
            step.cache_index, torch.Tensor
        ), "write position must be a tensor"
        seen.add((step.cache_seqlen, tuple(step.valid_start)))
        state.advance()
    cache_seqlens = {c for c, _ in seen}
    assert cache_seqlens == set(range(anchor + 1, capacity + 1)), cache_seqlens
    assert len(cache_seqlens) == BLOCK_SIZE, len(cache_seqlens)
    assert {v for _, v in seen} == {(0,)}, seen


def test_anchored_shift_at_the_shipped_geometry():
    """The 1088-row, 1024-anchor roll Gemma 4 actually runs, crossed once.

    W=1024 means anchor 1024, so 64 writes fill rows [1024, 1088) and the 65th
    step triggers the roll. Small head_dim and head counts keep this quick; what
    is under test is the bookkeeping at the real capacity, not the arithmetic
    intensity.
    """
    window, prompt, steps = 1024, 1024, 65
    capacity, full_capacity = 1088, 1152
    head_dim, q_heads, kv_heads = 32, 2, 1

    def band_mask(seqlen, block_base):
        if seqlen == 1 and block_base > 0:
            mask = build_decode_mask(
                1, full_capacity, block_base, 0, dtype=torch.float32
            )
        else:
            mask = build_prefill_mask(1, seqlen, full_capacity, 0, dtype=torch.float32)
        coords = torch.arange(seqlen)[None, :] + block_base
        return add_causal_sliding_window_band(mask, coords, window)

    torch.manual_seed(21)
    band = make_sliding_attention(q_heads, kv_heads, head_dim, window, swa_mode=None)
    op = copy.deepcopy(band)
    op.swa_mode = "anchored"

    band_k = torch.zeros(1, kv_heads, full_capacity, head_dim)
    band_v = torch.zeros(1, kv_heads, full_capacity, head_dim)
    op_k = torch.zeros(1, kv_heads, prompt, head_dim)
    op_v = torch.zeros(1, kv_heads, prompt, head_dim)

    hidden = torch.randn(1, prompt, q_heads * head_dim)
    freqs = identity_freqs(1, prompt, head_dim)
    index = make_cache_index(0, prompt)
    _, band_k, band_v = band(hidden, freqs, band_mask(prompt, 0), band_k, band_v, index)
    _, op_k, op_v = op(
        hidden, freqs, None, op_k, op_v, index, cache_seqlen=prompt, valid_start=[0]
    )

    state = SlidingWindowCache.after_prefill(window, prompt, [0])
    assert state.capacity == capacity and state.anchor == 1024
    assert state.valid_start == [0], "a prompt of exactly anchor rows leaves no gap"
    op_k, op_v = compact_after_prefill(op_k, op_v, state, prompt)
    assert op_k.shape[2] == capacity

    shifts = 0
    for step_index in range(steps):
        slot = prompt + step_index
        token = torch.randn(1, 1, q_heads * head_dim)
        token_freqs = identity_freqs(1, 1, head_dim)
        expected, band_k, band_v = band(
            token,
            token_freqs,
            band_mask(1, slot),
            band_k,
            band_v,
            make_cache_index(slot, 1),
        )
        step = anchored_step(state, "cpu")
        shifts += int(step.do_shift)
        if step.do_shift:
            op_k, op_v = roll_compact_buffer(op_k, op_v)
        actual, op_k, op_v = op(
            token,
            token_freqs,
            None,
            op_k,
            op_v,
            step.cache_index,
            cache_seqlen=step.cache_seqlen,
            valid_start=step.valid_start,
        )
        state.advance()
        torch.testing.assert_close(
            actual, expected, rtol=1e-4, atol=1e-5, msg=f"step {step_index}"
        )

    assert shifts == 1, f"65 steps at anchor 1024 must roll exactly once, got {shifts}"
    assert op_k.shape[2] == capacity, "the compact buffer must never grow"


class _MinimalGemma4:
    """Smallest stand-in that reaches the caller-supplied-masks guard.

    ``_run_blocks_over_embeds`` checks ``model._spyre_swa_mode`` against
    ``masks`` as its second statement, before it touches the backbone, the
    config, or anything else on the model -- so nothing else needs to be real
    here.
    """

    def __init__(self, swa_mode):
        self._spyre_swa_mode = swa_mode
        self.config = None


def test_caller_supplied_masks_reject_the_op_path():
    """The VLM's own masks and the op path cannot be combined — say so loudly.

    The module-level branch is fixed at prepare time, so a driver cannot un-enable
    it per call; the only correct answer is to refuse and name the opt-out.
    """
    import pytest

    from hf_adapters.hf_common import SpyreUnsupportedFeatureError

    model = _MinimalGemma4(swa_mode="anchored")
    with pytest.raises(SpyreUnsupportedFeatureError, match="_spyre_swa_mode"):
        _run_blocks_over_embeds(
            model,
            torch.zeros(1, 4, 8),
            torch.zeros(1, 4, dtype=torch.long),
            None,
            [],
            [],
            make_cache_index(0, 4),
            masks={"full_attention": None, "sliding_attention": None},
        )
