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

import dataclasses

import torch
import torch.nn.functional as F

from hf_adapters.hf_common import BLOCK_SIZE, allocate_kv_cache


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
