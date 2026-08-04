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
from hf_adapters import hf_gemma4_moe
from hf_adapters.hf_gemma4_moe import (
    _grouped_gemm_4a,
    _grouped_gemm_4b,
    _moe_ffn,
    _moe_ffn_loop_ref,
    _moe_permute,
    _moe_route,
)


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


def test_moe_ffn_loop_ref_matches_dense_reference():
    """Loop-on-topk FFN (no grouping) equals the dense per-token top-K sum."""
    torch.manual_seed(0)
    T, H, E, M, K = 6, 16, 8, 12, 4
    x = torch.randn(T, H)
    W_router = torch.randn(E, H)
    # Pre-transposed expert weights, matching the device layout.
    gate_up_t = torch.randn(E, H, 2 * M)  # [E,H,2M]
    down_t = torch.randn(E, M, H)          # [E,M,H]
    per_expert_scale = torch.rand(E) + 0.5

    got = _moe_ffn_loop_ref(x, W_router, gate_up_t, down_t, per_expert_scale, K)

    # Independent dense reference: route, then for each token sum its K experts.
    w, idx = _moe_route(x, W_router, per_expert_scale, K)  # [T,K],[T,K]
    ref = torch.zeros(T, H)
    for t in range(T):
        for j in range(K):
            e = int(idx[t, j])
            gu = x[t] @ gate_up_t[e]            # [2M]
            g, u = gu.chunk(2, dim=-1)          # [M],[M]
            act = torch.nn.functional.gelu(g, approximate="tanh") * u
            ref[t] += w[t, j] * (act @ down_t[e])  # [H]
    assert torch.allclose(got, ref, atol=1e-4, rtol=1e-4)


def test_grouped_gemm_4a_4b_agree():
    # Option 4B (contiguous per-expert-segment slab GEMM) must be numerically
    # identical to the shipped 4A (per-row weight gather). Rows must be sorted
    # by expert (the _moe_permute invariant 4B relies on).
    torch.manual_seed(0)
    N, IN, OUT, E = 11, 6, 5, 4
    gathered = torch.randn(N, IN)
    Wstack = torch.randn(E, OUT, IN)
    row_expert = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2, 3, 3, 3])
    out_4a = _grouped_gemm_4a(gathered, Wstack, row_expert)
    out_4b = _grouped_gemm_4b(gathered, Wstack, row_expert)
    torch.testing.assert_close(out_4a, out_4b, atol=1e-5, rtol=1e-5)


def test_grouped_gemm_4b_handles_empty_experts():
    # Experts that receive no rows leave an empty segment; 4B must skip them
    # and still match 4A.
    torch.manual_seed(1)
    N, IN, OUT, E = 6, 4, 3, 5
    gathered = torch.randn(N, IN)
    Wstack = torch.randn(E, OUT, IN)
    # experts 1 and 3 get no rows
    row_expert = torch.tensor([0, 0, 2, 2, 4, 4])
    out_4a = _grouped_gemm_4a(gathered, Wstack, row_expert)
    out_4b = _grouped_gemm_4b(gathered, Wstack, row_expert)
    torch.testing.assert_close(out_4a, out_4b, atol=1e-5, rtol=1e-5)


def test_moe_ffn_matches_reference_under_4b_flag(monkeypatch):
    # The full FFN under the 4B flag must match the dense reference exactly as
    # the 4A path does (same numerics, different weight-load schedule).
    monkeypatch.setattr(hf_gemma4_moe, "_MOE_GEMM_4B", True)
    T, H, E, K, M = 4, 16, 8, 2, 5
    x = torch.randn(T, H)
    W_router = torch.randn(E, H)
    gate_up = torch.randn(E, 2 * M, H)
    down = torch.randn(E, H, M)
    scale = torch.rand(E) + 0.5
    ref = _ref_moe(x, W_router, gate_up, down, scale, K)
    got = _moe_ffn(x, W_router, gate_up, down, scale, K)
    torch.testing.assert_close(got, ref, atol=1e-4, rtol=1e-4)
