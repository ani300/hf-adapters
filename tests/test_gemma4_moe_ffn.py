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

import torch
from hf_adapters.hf_gemma4_moe import _moe_route, _moe_permute


def test_route_shapes_and_renorm():
    T, H, E, K = 4, 16, 8, 2
    x = torch.randn(T, H)
    W = torch.randn(E, H)
    scale = torch.ones(E)
    w, idx = _moe_route(x, W, scale, K)
    assert w.shape == (T, K) and idx.shape == (T, K)
    # with per_expert_scale == 1, weights renormalize to sum 1 per token
    torch.testing.assert_close(
        w.sum(-1), torch.ones(T), atol=1e-5, rtol=1e-5
    )


def test_permute_roundtrip():
    T, H, E, K = 4, 16, 8, 2
    x = torch.randn(T, H)
    idx = torch.tensor([[0, 3], [3, 1], [7, 0], [2, 2]])
    gathered, token_of_row, row_expert, sort_perm = _moe_permute(
        x, idx, K
    )
    assert gathered.shape == (T * K, H)
    # rows are sorted by expert id
    assert torch.equal(
        row_expert, torch.sort(idx.reshape(-1)).values
    )
    # gathered row r is the source token for that pair
    torch.testing.assert_close(gathered, x[token_of_row])
