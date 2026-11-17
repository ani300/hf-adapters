# Copyright 2026 The Torch-Spyre Authors.
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

"""Check RFC routing and accumulation under the explicitly assumed op contract."""

import pytest
import torch

from hf_adapters.sparse_moe import pack_prefill_routes, sparse_decode, sparse_prefill
from tests._sparse_moe_reference import (
    dense_reference,
    reference_for_each_tile,
    register_assumed_ops,
)


@pytest.fixture(autouse=True)
def assumed_backend(monkeypatch):
    import torch_spyre._inductor.wsr as wsr

    register_assumed_ops()
    monkeypatch.setattr(wsr, "for_each_tile", reference_for_each_tile)


def inputs(tokens=35):
    generator = torch.Generator().manual_seed(48)
    x = torch.randn(tokens, 16, generator=generator) * 0.1
    gate = torch.randn(4, 16, 32, generator=generator) * 0.1
    up = torch.randn(gate.shape, generator=generator) * 0.1
    down = torch.randn(4, 32, 16, generator=generator) * 0.1
    # Skewed routing, empty experts, and an expert crossing several tile bounds.
    score = torch.zeros(tokens, 4)
    score[:, 0] = 0.75
    score[::3, 1] = 0.25
    return x, score, gate, up, down


def test_packed_lists():
    _, score, *_ = inputs(16)
    weights, ids, counts = pack_prefill_routes(
        score, torch.arange(16, dtype=torch.int32), 8
    )
    assert counts.tolist() == [[16, 8, 8, 8]]
    for expert in range(4):
        selected = torch.where(score[:, expert] != 0)[0]
        torch.testing.assert_close(ids[: len(selected), expert].long(), selected)
        torch.testing.assert_close(
            weights[: len(selected), expert], score[selected, expert]
        )
        assert (ids[len(selected) : 16, expert] == 16).all()
        assert (weights[len(selected) : 16, expert] == 16).all()


@pytest.mark.parametrize("activation", ["silu", "gelu_tanh"])
@pytest.mark.parametrize("tokens", [3, 17, 35])
def test_prefill_matches_dense(tokens, activation):
    x, score, gate, up, down = inputs(tokens)
    out = sparse_prefill(
        x, score, gate, up, down, torch.arange(16, dtype=torch.int32), activation, 8
    )
    torch.testing.assert_close(
        out, dense_reference(x, score, gate, up, down, activation)
    )
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("activation", ["silu", "gelu_tanh"])
@pytest.mark.parametrize("scale", [1.0, -1.0, 0.0])
def test_decode_matches_dense(activation, scale):
    x, score, gate, up, down = inputs(3)
    score = score * scale
    out = sparse_decode(
        x, score, gate, up, down, torch.arange(4, dtype=torch.int32), activation
    )
    torch.testing.assert_close(
        out, dense_reference(x, score, gate, up, down, activation)
    )


@pytest.mark.parametrize("devices", [4, 8])
def test_tensor_parallel_partial_sums(devices):
    x, score, gate, up, down = inputs(17)
    outputs = [
        sparse_prefill(
            x, score, g, u, d, torch.arange(16, dtype=torch.int32), "silu", 8
        )
        for g, u, d in zip(
            gate.chunk(devices, -1), up.chunk(devices, -1), down.chunk(devices, 1)
        )
    ]
    torch.testing.assert_close(
        sum(outputs), dense_reference(x, score, gate, up, down, "silu")
    )


def test_prefill_512_token_granularity():
    x, score, gate, up, down = inputs(513)
    actual = sparse_prefill(
        x, score, gate, up, down, torch.arange(1024, dtype=torch.int32), "silu"
    )
    torch.testing.assert_close(
        actual, dense_reference(x, score, gate, up, down, "silu")
    )


@pytest.mark.parametrize("mode", ["current", "sparse_prefill", "sparse"])
@pytest.mark.parametrize("prepared", [False, True])
def test_configure_sparse_moe_constants(monkeypatch, mode, prepared):
    from types import SimpleNamespace

    from hf_adapters.hf_common import configure_sparse_moe

    monkeypatch.setenv("SPYRE_MOE_IMPL", mode)
    block = torch.nn.Module()
    experts = (
        SimpleNamespace(gate_proj=torch.empty(128, 1, 1))
        if prepared
        else SimpleNamespace(gate_up_proj=torch.empty(128, 2, 1))
    )
    configure_sparse_moe(block, experts, 8)
    assert block._sparse_prefill == (mode != "current")
    assert block._sparse_decode == (mode == "sparse")
    assert not block.state_dict()
    if mode != "current":
        torch.testing.assert_close(
            block._moe_token_ids, torch.arange(8192, dtype=torch.int32)
        )
        torch.testing.assert_close(
            block._moe_expert_ids, torch.arange(128, dtype=torch.int32)
        )
