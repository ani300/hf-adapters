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

"""Diagnostic assumptions for RFC 48; never imported by production code."""

import torch

_LIBRARIES = []


def register_assumed_ops():
    lib = torch.library.Library("spyre", "FRAGMENT")
    _LIBRARIES.append(lib)
    if not hasattr(torch.ops.spyre, "non_zero_pos_to_idx"):
        lib.define("non_zero_pos_to_idx(Tensor x, int dim) -> Tensor")

        def positions(x, dim):
            nonzero = x != 0
            return torch.where(nonzero, nonzero.cumsum(dim) - 1, x.shape[dim]).to(
                x.dtype
            )

        lib.impl("non_zero_pos_to_idx", positions, "CPU")
        torch.library.register_fake("spyre::non_zero_pos_to_idx")(
            lambda x, dim: torch.empty_like(x)
        )
    if not hasattr(torch.ops.spyre, "count_non_zero_round_up"):
        lib.define("count_non_zero_round_up(Tensor x, int dim, int g) -> Tensor")

        def counts(x, dim, g):
            count = (x != 0).sum(dim, keepdim=True)
            return (((count + g - 1) // g).clamp_min(1) * g).to(x.dtype)

        def fake_count(x, dim, g):
            shape = list(x.shape)
            shape[dim] = 1
            return x.new_empty(shape)

        lib.impl("count_non_zero_round_up", counts, "CPU")
        torch.library.register_fake("spyre::count_non_zero_round_up")(fake_count)


def reference_for_each_tile(
    body, operands, *, dims, tile_size, init=None, out_dim=None, reverse=False
):
    """Execute tile semantics on CPU without asking Dynamo to trace a scan."""
    from torch_spyre._inductor.wsr import Gather

    lengths = [
        len(spec.index) if isinstance(spec, Gather) else value.shape[spec] // tile_size
        for value, spec in zip(operands, dims)
        if spec is not None
    ]
    assert len(set(lengths)) == 1
    carry, outputs = init, []
    for step in range(lengths[0]):
        tiles = tuple(
            (
                value
                if spec is None
                else (
                    value.index_select(spec.axis, spec.index[step : step + 1])
                    if isinstance(spec, Gather)
                    else value.narrow(spec, step * tile_size, tile_size)
                )
            )
            for value, spec in zip(operands, dims)
        )
        carry, output = body(carry, tiles)
        if out_dim is not None:
            outputs.append(output)
    return carry, torch.cat(outputs, dim=out_dim) if outputs else None


def dense_reference(x, score, gate, up, down, activation):
    import torch.nn.functional as F

    gate_out = x.unsqueeze(0) @ gate
    activated = (
        F.silu(gate_out)
        if activation == "silu"
        else F.gelu(gate_out, approximate="tanh")
    )
    return ((activated * (x.unsqueeze(0) @ up) @ down) * score.T[:, :, None]).sum(0)
