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
import torch.nn.functional as F
from hf_adapters.hf_gemma4_moe import _moe_route, _moe_permute, _moe_ffn


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


def _ref_moe(x, W_router, gate_up, down, scale, K):
    # dense reference: compute all experts, select top-K, weighted sum
    T, H = x.shape
    E = W_router.shape[0]
    probs = torch.softmax(F.linear(x, W_router), dim=-1)
    w, idx = torch.topk(probs, K, dim=-1)
    w = (w / w.sum(-1, keepdim=True)) * scale[idx]
    out = torch.zeros_like(x)
    for t in range(T):
        for k in range(K):
            e = idx[t, k].item()
            g, u = F.linear(x[t], gate_up[e]).chunk(2, dim=-1)
            h = F.linear(
                F.gelu(g, approximate="tanh") * u, down[e]
            )
            out[t] += w[t, k] * h
    return out


def test_moe_ffn_matches_reference():
    T, H, E, K, M = 4, 16, 8, 2, 5
    x = torch.randn(T, H)
    W_router = torch.randn(E, H)
    gate_up = torch.randn(E, 2 * M, H)
    down = torch.randn(E, H, M)
    scale = torch.rand(E) + 0.5
    ref = _ref_moe(x, W_router, gate_up, down, scale, K)
    got = _moe_ffn(x, W_router, gate_up, down, scale, K)
    torch.testing.assert_close(got, ref, atol=1e-4, rtol=1e-4)
