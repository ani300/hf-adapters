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
    from hf_adapters.hf_common import _mask_fill_value

    batch = query.size(0)
    seqlen_q = query.size(2)
    capacity = key_cache.size(2)
    rows = torch.arange(seqlen_q, device=query.device) + (cache_seqlen - seqlen_q)
    columns = torch.arange(capacity, device=query.device) + buffer_origin
    delta = rows.unsqueeze(-1) - columns.unsqueeze(0)
    in_window = (delta >= 0) & (delta < window_size)
    in_window = in_window.unsqueeze(0).unsqueeze(0).expand(batch, 1, -1, -1)
    in_valid = torch.ones(in_window.shape, dtype=torch.bool, device=query.device)
    if valid_start is not None and max(valid_start) > 0:
        starts = torch.tensor(valid_start, device=query.device).view(-1, 1, 1, 1)
        in_valid = columns.view(1, 1, 1, -1) >= starts
    mask = torch.zeros(in_window.shape, dtype=query.dtype, device=query.device)
    mask.masked_fill_(~in_window, float("-inf"))
    mask.masked_fill_(~in_valid & in_window, _mask_fill_value(query.dtype))
    return F.scaled_dot_product_attention(
        query,
        key_cache,
        value_cache,
        attn_mask=mask,
        dropout_p=0.0,
        scale=scale,
        enable_gqa=True,
    )
