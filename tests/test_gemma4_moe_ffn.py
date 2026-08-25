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
    _STICK_SIZE,
    _compiled_moe_loop_region,
    _moe_expert_persistent,
    _moe_route_persistent_packed,
    _prepare_experts,
    _select_experts,
)

_DTYPE = torch.float16


def _router_probs(x_router, router_proj_w, router_scale, root_size, eps):
    """The router softmax both device regions compute internally."""
    var = x_router.pow(2).mean(-1, keepdim=True)
    normed = x_router * torch.rsqrt(var + eps)  # scale-free RMSNorm
    normed = normed * router_scale * root_size
    return torch.softmax(F.linear(normed, router_proj_w), dim=-1)  # [T,E]


def _separated_router_inputs(tokens, hidden, experts):
    signs = torch.ones(tokens, dtype=_DTYPE)
    signs[1::2] = -1
    x = signs[:, None].expand(tokens, hidden).contiguous()
    levels = torch.arange(experts, dtype=_DTYPE) / 16
    weight = levels[:, None].expand(experts, hidden).contiguous() / hidden
    return x, weight


def test_prepare_experts_splits_gate_up_and_reuses_projection_names(monkeypatch):
    experts = torch.nn.Module()
    gate_up = torch.arange(2 * 8 * 6, dtype=_DTYPE).reshape(2, 8, 6)
    down = torch.arange(2 * 6 * 4, dtype=_DTYPE).reshape(2, 6, 4)
    experts.gate_up_proj = torch.nn.Parameter(gate_up)
    experts.down_proj = torch.nn.Parameter(down)
    monkeypatch.setattr(
        "hf_adapters.hf_gemma4_moe._move_expert_weight", lambda weight: weight
    )

    _prepare_experts(experts)

    assert not hasattr(experts, "gate_up_proj")
    torch.testing.assert_close(experts.gate_proj, gate_up[:, :4].transpose(1, 2))
    torch.testing.assert_close(experts.up_proj, gate_up[:, 4:].transpose(1, 2))
    torch.testing.assert_close(experts.down_proj, down.transpose(1, 2))


def test_decode_loop_region_matches_dense_reference():
    """``_compiled_moe_loop_region`` (decode) == dense per-token top-K sum.

    The region routes on ``x_router``, gathers the top-K experts for each
    token, runs the SwiGLU expert GEMM, scales by the routing weight and
    per-expert scale, and reduces over K -- all on device. It returns [T,H].
    """
    torch.manual_seed(0)
    T, H, E, M, K = 6, 16, 8, 12, 4
    x_router, router_proj_w = _separated_router_inputs(T, H, E)
    x_expert = torch.randn(T, H, dtype=_DTYPE)
    router_scale = torch.ones(H, dtype=_DTYPE)  # neutral -> scale-free norm
    gate = torch.randn(E, H, M, dtype=_DTYPE)  # [E,H,M] device layout
    up = torch.randn(E, H, M, dtype=_DTYPE)
    down = torch.randn(E, M, H, dtype=_DTYPE)  # [E,M,H] device layout
    per_expert_scale = torch.rand(E, dtype=_DTYPE) + 0.5
    # Decode gathers per_expert_scale by expert id, so the region reads the
    # host-widened sticked source (every lane has the same scalar).
    per_expert_scale_stick = (
        per_expert_scale[:, None].expand(-1, _STICK_SIZE).contiguous()
    )

    got = _compiled_moe_loop_region(
        x_router,
        x_expert,
        router_proj_w,
        router_scale,
        1.0,
        per_expert_scale_stick,
        gate,
        up,
        down,
        K,
        4,
        1e-6,
    )

    # Dense reference: route, then for each token sum its K weighted experts.
    probs = _router_probs(x_router, router_proj_w, router_scale, 1.0, 1e-6)
    w, idx = torch.topk(probs, K, dim=-1)
    w = w / w.sum(-1, keepdim=True)
    ref = torch.zeros(T, H, dtype=_DTYPE)
    for t in range(T):
        for j in range(K):
            e = int(idx[t, j])
            g = x_expert[t] @ gate[e]  # [M]
            u = x_expert[t] @ up[e]  # [M]
            act = F.gelu(g, approximate="tanh") * u
            ref[t] += w[t, j] * per_expert_scale[e] * (act @ down[e])  # [H]

    torch.testing.assert_close(got, ref, atol=1e-2, rtol=1e-2)


def test_prefill_route_packed_selects_and_broadcasts():
    """``_moe_route_persistent_packed`` (prefill router) == renormed top-K.

    Returns [T,E,S]: the renormed top-K weight scaled by per_expert_scale
    (zero off the top-K), broadcast across all stick lanes. keep_by_index
    is a positional selection by the top-k indices.
    """
    torch.manual_seed(1)
    T, H, E, K = 6, 16, 8, 4
    x_router, router_proj_w = _separated_router_inputs(T, H, E)
    router_scale = torch.ones(H, dtype=_DTYPE)
    per_expert_scale = torch.rand(E, dtype=_DTYPE) + 0.5
    route_identity = torch.eye(_STICK_SIZE, dtype=_DTYPE)

    routing_sticks = _moe_route_persistent_packed(
        x_router,
        router_proj_w,
        router_scale,
        1.0,
        per_expert_scale,
        K,
        1e-6,
        route_identity,
    )
    assert routing_sticks.shape == (T, E, _STICK_SIZE)

    # Reference: topk -> keep_by_index -> renorm -> per_expert_scale.
    probs = _router_probs(x_router, router_proj_w, router_scale, 1.0, 1e-6)
    _, sel = torch.topk(probs, K, dim=-1)
    mask = torch.ops.spyre.keep_by_index(probs, sel, -1, 0.0)
    w = mask / mask.sum(-1, keepdim=True)
    token_major = torch.relu(w * per_expert_scale)  # relu of nonneg == identity

    # Every lane carries the same scalar; lane 0 is the routing weight.
    torch.testing.assert_close(
        routing_sticks[..., 0], token_major, atol=1e-3, rtol=1e-3
    )
    torch.testing.assert_close(
        routing_sticks,
        token_major[..., None].expand(-1, -1, _STICK_SIZE),
        atol=1e-5,
        rtol=1e-5,
    )
    # Exactly K experts per token are nonzero (positional selection, no leak).
    assert (routing_sticks[..., 0] > 0).sum(-1).tolist() == [K] * T


def test_prefill_expert_persistent_matches_dense_reference():
    """``_moe_expert_persistent`` == routing-weighted sum over ALL experts.

    The persistent path runs every expert densely and folds in the [T,E,1]
    routing weight (zero off the top-K), so it equals the weighted expert sum.
    """
    torch.manual_seed(2)
    T, H, E, M = 6, 16, 8, 12
    x_expert = torch.randn(T, H, dtype=_DTYPE)
    gate = torch.randn(E, H, M, dtype=_DTYPE)
    up = torch.randn(E, H, M, dtype=_DTYPE)
    down = torch.randn(E, M, H, dtype=_DTYPE)
    routing_weight = torch.rand(T, E, 1, dtype=_DTYPE)  # renormed/masked upstream

    got = _moe_expert_persistent(x_expert, routing_weight, gate, up, down)

    ref = torch.zeros(T, H, dtype=_DTYPE)
    for e in range(E):
        g = x_expert @ gate[e]  # [T,M]
        u = x_expert @ up[e]  # [T,M]
        act = F.gelu(g, approximate="tanh") * u
        ref += (act @ down[e]) * routing_weight[:, e, :]  # [T,H]

    torch.testing.assert_close(got, ref, atol=2e-2, rtol=2e-2)


def test_prefill_pipeline_matches_dense_topk_reference():
    """End-to-end prefill (route packed -> persistent experts) == dense top-K.

    Chains the two prefill regions exactly as ``Gemma4MoEBlock.forward`` does
    (slice lane 0 of the [T,E,S] router output to the [T,E,1] the expert path
    reads) and compares to an independent dense top-K FFN reference.
    """
    torch.manual_seed(3)
    T, H, E, M, K = 6, 16, 8, 12, 4
    x_router, router_proj_w = _separated_router_inputs(T, H, E)
    x_expert = torch.randn(T, H, dtype=_DTYPE)
    router_scale = torch.ones(H, dtype=_DTYPE)
    gate = torch.randn(E, H, M, dtype=_DTYPE)
    up = torch.randn(E, H, M, dtype=_DTYPE)
    down = torch.randn(E, M, H, dtype=_DTYPE)
    per_expert_scale = torch.rand(E, dtype=_DTYPE) + 0.5

    routing_sticks = _moe_route_persistent_packed(
        x_router,
        router_proj_w,
        router_scale,
        1.0,
        per_expert_scale,
        K,
        1e-6,
        torch.eye(_STICK_SIZE, dtype=_DTYPE),
    )
    routing_weight = routing_sticks[..., :1]  # [T,E,1] lane-0 view
    got = _moe_expert_persistent(x_expert, routing_weight, gate, up, down)

    # Dense reference: renorm the top-K, scale by per-expert scale, and sum.
    probs = _router_probs(x_router, router_proj_w, router_scale, 1.0, 1e-6)
    w, idx = torch.topk(probs, K, dim=-1)
    w = w / w.sum(-1, keepdim=True)
    ref = torch.zeros(T, H, dtype=_DTYPE)
    for t in range(T):
        for j in range(K):
            e = int(idx[t, j])
            g = x_expert[t] @ gate[e]
            u = x_expert[t] @ up[e]
            act = F.gelu(g, approximate="tanh") * u
            ref[t] += w[t, j] * per_expert_scale[e] * (act @ down[e])

    torch.testing.assert_close(got, ref, atol=2e-2, rtol=2e-2)


def test_single_token_topk_matches_direct_topk():
    logits = torch.arange(128, dtype=_DTYPE)[None] / 16
    probs = torch.softmax(logits, dim=-1)

    weights, indices = _select_experts(probs, 8)
    ref_weights, ref_indices = torch.topk(probs, 8, dim=-1)
    ref_weights = ref_weights / ref_weights.sum(-1, keepdim=True)

    torch.testing.assert_close(weights, ref_weights)
    assert torch.equal(indices, ref_indices)
