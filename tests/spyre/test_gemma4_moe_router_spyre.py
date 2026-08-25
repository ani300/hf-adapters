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

import pytest
import torch
import torch.nn.functional as F

from hf_adapters.hf_gemma4_moe import (
    _STICK_SIZE,
    _moe_route_persistent_packed,
    _router_probs,
    _topk,
)

pytestmark = pytest.mark.requires_spyre

_HIDDEN = 2816
_EXPERTS = 128
_TOP_K = 8
_DTYPE = torch.float16


def _router_inputs(tokens):
    signs = torch.ones(tokens, dtype=_DTYPE)
    signs[1::2] = -1
    inputs = signs[:, None].expand(tokens, _HIDDEN).contiguous()
    levels = torch.arange(_EXPERTS, dtype=_DTYPE) / 16
    weights = levels[:, None].expand(_EXPERTS, _HIDDEN).contiguous() / _HIDDEN
    return inputs, weights, torch.ones(_HIDDEN, dtype=_DTYPE)


def _reference(inputs, weights, scale):
    variance = inputs.pow(2).mean(-1, keepdim=True)
    logits = F.linear(inputs * torch.rsqrt(variance + 1e-6) * scale, weights)
    probabilities = torch.softmax(logits, dim=-1)
    topk_weights, indices = torch.topk(probabilities, _TOP_K, dim=-1)
    topk_weights = topk_weights / topk_weights.sum(-1, keepdim=True)
    return probabilities, topk_weights, indices


def test_decode_router_real_shape():
    inputs, weights, scale = _router_inputs(1)
    _, expected_weights, expected_indices = _reference(inputs, weights, scale)

    def route(x, proj, router_scale):
        probabilities = _router_probs(x, proj, router_scale, 1.0, 1e-6)
        topk_weights, indices = _topk(probabilities, _TOP_K)
        return topk_weights / topk_weights.sum(-1, keepdim=True), indices

    actual_weights, actual_indices = torch.compile(route, dynamic=False)(
        inputs.to("spyre"), weights.to("spyre"), scale.to("spyre")
    )

    torch.testing.assert_close(
        actual_weights.cpu(), expected_weights, atol=1e-2, rtol=1e-2
    )
    assert torch.equal(actual_indices.cpu().to(torch.int64), expected_indices)


def test_prefill_router_real_shape():
    inputs, weights, scale = _router_inputs(64)
    probabilities, topk_weights, indices = _reference(inputs, weights, scale)
    expert_scale = 1 + torch.arange(_EXPERTS, dtype=_DTYPE) / 1024
    expected = torch.zeros_like(probabilities).scatter(
        -1, indices, topk_weights * expert_scale[indices]
    )

    actual = torch.compile(_moe_route_persistent_packed, dynamic=False)(
        inputs.to("spyre"),
        weights.to("spyre"),
        scale.to("spyre"),
        1.0,
        expert_scale.to("spyre"),
        _TOP_K,
        1e-6,
        torch.eye(_STICK_SIZE, dtype=_DTYPE).to("spyre"),
    )[..., 0].cpu()

    assert torch.equal(actual != 0, expected != 0)
    assert torch.count_nonzero(actual, dim=-1).unique().tolist() == [_TOP_K]
    torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)
