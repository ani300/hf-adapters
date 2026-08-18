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
    """valid_start must reproduce build_prefill_mask's left-pad columns."""
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
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_reference_honors_per_sequence_valid_start():
    """A ragged batch: each entry gets its own threshold."""
    query, key_cache, value_cache = _inputs(batch=2)
    per_entry = [
        _band_masked_attention(
            query[b : b + 1], key_cache[b : b + 1], value_cache[b : b + 1], 64, offset
        )
        for b, offset in enumerate((0, 40))
    ]
    expected = torch.cat(per_entry, dim=0)
    actual = sliding_window_attention(
        query,
        key_cache,
        value_cache,
        window_size=64,
        scale=None,
        cache_seqlen=128,
        valid_start=[0, 40],
    )
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


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
