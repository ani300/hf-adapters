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

"""CPU-eager equivalence tests for the Gemma 4 MoE device regions.

``hf_gemma4_moe`` compiles three device regions, dispatched per forward by
sequence length (see the module docstring):

  * decode  -- ``_compiled_moe_loop_region`` (whole FFN, combine included)
  * prefill -- ``_moe_route_persistent_packed`` then ``_moe_expert_persistent``

Every region is plain PyTorch that also runs eagerly on CPU (``keep_by_index``
and ``spyre_hint`` are available / no-ops off device), so each test runs the
real region on CPU and compares it to an independent dense reference. The
router surface is shared by both paths: a scale-free RMSNorm on the RAW
residual, then an ``[H]`` ``router_scale``, then the ``root_size`` float, then
the projection. Passing ``router_scale=ones(H)`` and ``root_size=1.0`` reduces
that to a plain scale-free RMSNorm the references reproduce directly.
"""

import torch
import torch.nn.functional as F

from hf_adapters.hf_gemma4_moe import (
    _MOE_PAD_NEG,
    _MOE_PADW,
    _compiled_moe_loop_region,
    _moe_expert_persistent,
    _moe_route_persistent_packed,
)


def _router_probs(x_router, router_proj_w, router_scale, root_size, eps):
    """The router softmax both device regions compute internally."""
    var = x_router.pow(2).mean(-1, keepdim=True)
    normed = x_router * torch.rsqrt(var + eps)  # scale-free RMSNorm
    normed = normed * router_scale * root_size
    return torch.softmax(F.linear(normed, router_proj_w), dim=-1)  # [T,E]


def test_decode_loop_region_matches_dense_reference():
    """``_compiled_moe_loop_region`` (decode) == dense per-token top-K sum.

    The region routes on ``x_router``, gathers the top-K experts for each
    token, runs the SwiGLU expert GEMM, scales by the routing weight and
    per-expert scale, and reduces over K -- all on device. It returns [T,H].
    """
    torch.manual_seed(0)
    T, H, E, M, K = 6, 16, 8, 12, 4
    x_router = torch.randn(T, H)
    x_expert = torch.randn(T, H)
    router_proj_w = torch.randn(E, H)
    router_scale = torch.ones(H)  # neutral -> region router == scale-free norm
    gate = torch.randn(E, H, M)  # [E,H,M] device layout
    up = torch.randn(E, H, M)
    down = torch.randn(E, M, H)  # [E,M,H] device layout
    per_expert_scale = torch.rand(E) + 0.5
    # Decode gathers per_expert_scale by expert id, so the region reads the
    # host-widened sticked [E,64] source (every lane the same scalar).
    per_expert_scale_stick = per_expert_scale[:, None].expand(-1, 64).contiguous()

    got = _compiled_moe_loop_region(
        x_router, x_expert, router_proj_w, router_scale, 1.0,
        per_expert_scale_stick, gate, up, down, K, 4, 1e-6,
    )

    # Dense reference: route, then for each token sum its K weighted experts.
    probs = _router_probs(x_router, router_proj_w, router_scale, 1.0, 1e-6)
    w, idx = torch.topk(probs, K, dim=-1)
    w = w / w.sum(-1, keepdim=True)
    ref = torch.zeros(T, H)
    for t in range(T):
        for j in range(K):
            e = int(idx[t, j])
            g = x_expert[t] @ gate[e]  # [M]
            u = x_expert[t] @ up[e]  # [M]
            act = F.gelu(g, approximate="tanh") * u
            ref[t] += w[t, j] * per_expert_scale[e] * (act @ down[e])  # [H]

    torch.testing.assert_close(got, ref, atol=1e-4, rtol=1e-4)


def test_prefill_route_packed_selects_and_broadcasts():
    """``_moe_route_persistent_packed`` (prefill router) == renormed top-K.

    Returns [T,E,64]: the renormed top-K weight scaled by per_expert_scale
    (zero off the top-K), broadcast across all 64 stick lanes. keep_by_index
    is a POSITIONAL selection by the padded-topk indices.
    """
    torch.manual_seed(1)
    T, H, E, K = 6, 16, 8, 4
    x_router = torch.randn(T, H)
    router_proj_w = torch.randn(E, H)
    router_scale = torch.ones(H)
    per_expert_scale = torch.rand(E) + 0.5
    route_identity = torch.eye(64)

    routing_sticks = _moe_route_persistent_packed(
        x_router, router_proj_w, router_scale, 1.0, per_expert_scale,
        K, 1e-6, _MOE_PADW, _MOE_PAD_NEG, route_identity,
    )
    assert routing_sticks.shape == (T, E, 64)

    # Reference: pad -> topk -> keep_by_index -> renorm -> per_expert_scale.
    probs = _router_probs(x_router, router_proj_w, router_scale, 1.0, 1e-6)
    pad = torch.full((T, _MOE_PADW - E), _MOE_PAD_NEG)
    _, sel = torch.topk(torch.cat([probs, pad], dim=-1), K, dim=-1)
    mask = torch.ops.spyre.keep_by_index(probs, sel, -1, 0.0)
    w = mask / mask.sum(-1, keepdim=True)
    token_major = torch.relu(w * per_expert_scale)  # relu of nonneg == identity

    # Every lane carries the same scalar; lane 0 is the routing weight.
    torch.testing.assert_close(
        routing_sticks[..., 0], token_major, atol=1e-4, rtol=1e-4
    )
    torch.testing.assert_close(
        routing_sticks, token_major[..., None].expand(-1, -1, 64),
        atol=1e-5, rtol=1e-5,
    )
    # Exactly K experts per token are nonzero (positional selection, no leak).
    assert (routing_sticks[..., 0] > 0).sum(-1).tolist() == [K] * T


def test_prefill_expert_persistent_matches_dense_reference():
    """``_moe_expert_persistent`` == routing-weighted sum over ALL experts.

    The persistent path runs every expert densely and folds in the [T,E,1]
    routing weight (zero off the top-K), so it equals the weighted expert sum.
    """
    torch.manual_seed(2)
    T, H, E, M, K = 6, 16, 8, 12, 4
    x_expert = torch.randn(T, H)
    gate = torch.randn(E, H, M)
    up = torch.randn(E, H, M)
    down = torch.randn(E, M, H)
    routing_weight = torch.rand(T, E, 1)  # already renormed/masked upstream

    got = _moe_expert_persistent(x_expert, routing_weight, gate, up, down, K)

    ref = torch.zeros(T, H)
    for e in range(E):
        g = x_expert @ gate[e]  # [T,M]
        u = x_expert @ up[e]  # [T,M]
        act = F.gelu(g, approximate="tanh") * u
        ref += (act @ down[e]) * routing_weight[:, e, :]  # [T,H]

    torch.testing.assert_close(got, ref, atol=1e-4, rtol=1e-4)


def test_prefill_pipeline_matches_dense_topk_reference():
    """End-to-end prefill (route packed -> persistent experts) == dense top-K.

    Chains the two prefill regions exactly as ``Gemma4MoEBlock.forward`` does
    (slice lane 0 of the [T,E,64] router output to the [T,E,1] the expert path
    reads) and compares to an independent dense top-K FFN reference.
    """
    torch.manual_seed(3)
    T, H, E, M, K = 6, 16, 8, 12, 4
    x_router = torch.randn(T, H)
    x_expert = torch.randn(T, H)
    router_proj_w = torch.randn(E, H)
    router_scale = torch.ones(H)
    gate = torch.randn(E, H, M)
    up = torch.randn(E, H, M)
    down = torch.randn(E, M, H)
    per_expert_scale = torch.rand(E) + 0.5

    routing_sticks = _moe_route_persistent_packed(
        x_router, router_proj_w, router_scale, 1.0, per_expert_scale,
        K, 1e-6, _MOE_PADW, _MOE_PAD_NEG, torch.eye(64),
    )
    routing_weight = routing_sticks[..., :1]  # [T,E,1] lane-0 view
    got = _moe_expert_persistent(x_expert, routing_weight, gate, up, down, K)

    # Dense reference: route (padded topk == plain topk since pads never win),
    # renorm the top-K, scale by per_expert_scale, sum the selected experts.
    probs = _router_probs(x_router, router_proj_w, router_scale, 1.0, 1e-6)
    w, idx = torch.topk(probs, K, dim=-1)
    w = w / w.sum(-1, keepdim=True)
    ref = torch.zeros(T, H)
    for t in range(T):
        for j in range(K):
            e = int(idx[t, j])
            g = x_expert[t] @ gate[e]
            u = x_expert[t] @ up[e]
            act = F.gelu(g, approximate="tanh") * u
            ref[t] += w[t, j] * per_expert_scale[e] * (act @ down[e])

    torch.testing.assert_close(got, ref, atol=1e-4, rtol=1e-4)
