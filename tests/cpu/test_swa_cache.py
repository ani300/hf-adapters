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
