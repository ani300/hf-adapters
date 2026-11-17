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

"""Experimental RFC 48 MoE paths; scan/count kernels must come from torch-spyre.

The matching spyre-inference module intentionally has the same implementation.
Dummy fp32 stick dimensions have 32 elements. Layout conversion and
index-to-address conversion belong to the backend.
"""

import torch
import torch.nn.functional as F


def pack_prefill_routes(score, token_ids, granularity=512):
    """A9–A17: pack each expert's weights and token ids, with a sink row."""
    tokens, experts = score.shape
    score = score.to(torch.float32)
    positions = torch.ops.spyre.non_zero_pos_to_idx(
        score,
        0,  # ty: ignore[invalid-argument-type]
    )
    counts = torch.ops.spyre.count_non_zero_round_up(
        score,
        0,  # ty: ignore[invalid-argument-type]
        granularity,  # ty: ignore[invalid-argument-type]
    )
    # aten.scatter requires logical int64 indices. Spyre stores these as int32.
    indices = positions.to(torch.int64).unsqueeze(-1).expand(-1, -1, 32)
    weights = torch.full(
        (tokens + 1, experts, 32), tokens, dtype=score.dtype, device=score.device
    ).scatter(0, indices, score.unsqueeze(-1).expand(-1, -1, 32).contiguous())
    ids = token_ids[:tokens, None, None].expand(tokens, experts, 32).contiguous()
    token_list = torch.full(
        (tokens + 1, experts, 32), tokens, dtype=token_ids.dtype, device=score.device
    ).scatter(0, indices, ids)
    return weights[..., 0], token_list[..., 0], counts


def _activate(gate_out, up_out, activation):
    if activation == "silu":
        return F.silu(gate_out) * up_out
    if activation == "gelu_tanh":
        return F.gelu(gate_out, approximate="tanh") * up_out
    raise ValueError(f"unsupported MoE activation: {activation!r}")


def sparse_prefill_chunk(
    x, score, gate, up, down, token_ids, activation, granularity=512
):
    """A18–A32: expert loop around a device-counted loop of token tiles.

    Input is one chunk, padded to a multiple of granularity. Both loops carry a
    fresh output buffer: functional scatter preserves for_each_tile's no-alias
    contract while expressing the RFC's gather/add/scatter accumulation.
    """
    from torch_spyre._inductor.wsr import for_each_tile

    tokens, hidden = x.shape
    weights, ids, counts = pack_prefill_routes(score, token_ids, granularity)
    x_pad = F.pad(x, (0, 0, 0, 1))
    weights = (
        weights[:tokens].T.contiguous().reshape(-1, tokens // granularity, granularity)
    )
    ids = ids[:tokens].T.contiguous().reshape(-1, tokens // granularity, granularity)
    counts = counts.reshape(-1, 1, 1).expand(-1, 1, 32).contiguous()

    def expert_body(acc, tiles):
        x_pad, weight_tiles, id_tiles, count, gate_tile, up_tile, down_tile = tiles
        # A14b supplies a device value, not an input-shape symbol. Narrowing the
        # tile arrays exposes this count to for_each_tile's shape-derived loop.
        trips = torch.sym_int(count.reshape(-1)[0].item() / granularity)
        torch._check_is_size(trips)
        torch._check(trips >= 1)
        torch._check(trips != 0)
        torch._check(trips <= weight_tiles.shape[1])
        weight_tiles = weight_tiles.squeeze(0).narrow(0, 0, trips)
        id_tiles = id_tiles.squeeze(0).narrow(0, 0, trips)
        gate_tile, up_tile, down_tile = (
            gate_tile.squeeze(0),
            up_tile.squeeze(0),
            down_tile.squeeze(0),
        )

        def token_body(partial, token_tiles):
            x_pad, weight, ids, gate_tile, up_tile, down_tile = token_tiles
            ids = ids.reshape(-1).to(torch.int64)
            exp_in = x_pad.index_select(0, ids)
            activated = _activate(exp_in @ gate_tile, exp_in @ up_tile, activation)
            weight_stick = weight.reshape(-1, 1).expand(-1, 32).contiguous()
            weighted = ((activated @ down_tile) * weight_stick[:, :1]).to(partial.dtype)
            updated = partial.index_select(0, ids) + weighted
            return partial.scatter(0, ids[:, None].expand_as(updated), updated), None

        result, _ = for_each_tile(
            token_body,
            (x_pad, weight_tiles, id_tiles, gate_tile, up_tile, down_tile),
            dims=(None, 0, 0, None, None, None),
            tile_size=1,
            init=acc,
        )
        return result, None

    result, _ = for_each_tile(
        expert_body,
        (x_pad, weights, ids, counts, gate, up, down),
        dims=(None, 0, 0, 0, 0, 0, 0),
        tile_size=1,
        init=x.new_zeros((tokens + 1, hidden)),
    )
    return result[:tokens]


def sparse_prefill(x, score, gate, up, down, token_ids, activation, granularity=512):
    """Chunk to the precomputed id capacity, and discard padding after combine."""
    from torch_spyre._inductor.wsr import for_each_tile

    tokens, hidden = x.shape
    chunk = min(
        token_ids.shape[0], ((tokens + granularity - 1) // granularity) * granularity
    )
    padded = ((tokens + chunk - 1) // chunk) * chunk
    x_tiles = F.pad(x, (0, 0, 0, padded - tokens)).reshape(-1, chunk, hidden)
    score_tiles = F.pad(score, (0, 0, 0, padded - tokens)).reshape(
        -1, chunk, gate.shape[0]
    )

    def chunk_body(carry, tiles):
        x_tile, score_tile, gate, up, down, token_ids = tiles
        out = sparse_prefill_chunk(
            x_tile.squeeze(0),
            score_tile.squeeze(0),
            gate,
            up,
            down,
            token_ids,
            activation,
            granularity,
        )
        return None, out.unsqueeze(0)

    _, result = for_each_tile(
        chunk_body,
        (x_tiles, score_tiles, gate, up, down, token_ids),
        dims=(0, 0, None, None, None, None),
        tile_size=1,
        out_dim=0,
    )
    return result.reshape(padded, hidden)[:tokens]


def pack_decode_routes(score, expert_ids):
    """B8–B16: compact the union of experts selected anywhere in the batch."""
    # Model-specific output scales can be signed; selection uses magnitude.
    selected = score.abs().amax(0).to(torch.float32).unsqueeze(-1)
    selected = selected.expand(-1, 32).contiguous()
    positions = torch.ops.spyre.non_zero_pos_to_idx(
        selected,
        0,  # ty: ignore[invalid-argument-type]
    )
    count = torch.ops.spyre.count_non_zero_round_up(
        selected,
        0,  # ty: ignore[invalid-argument-type]
        1,  # ty: ignore[invalid-argument-type]
    )
    ids = torch.zeros(
        (score.shape[1] + 1, 32), dtype=expert_ids.dtype, device=score.device
    ).scatter(
        0, positions.to(torch.int64), expert_ids[:, None].expand(-1, 32).contiguous()
    )
    return ids[:, 0], count


def sparse_decode(x, score, gate, up, down, expert_ids, activation):
    """B18–B25: whole-batch GEMMs, once for each selected expert."""
    from torch_spyre._inductor.wsr import Gather, for_each_tile

    ids, count = pack_decode_routes(score, expert_ids)
    selected = torch.sym_int(count.reshape(-1)[0].item())
    torch._check_is_size(selected)
    torch._check(selected >= 1)
    torch._check(selected != 0)
    torch._check(selected <= gate.shape[0])
    ids = ids.narrow(0, 0, selected).to(torch.int64)

    def expert_body(acc, tiles):
        x, weight, gate_tile, up_tile, down_tile = tiles
        activated = _activate(
            x @ gate_tile.squeeze(0), x @ up_tile.squeeze(0), activation
        )
        out = (activated @ down_tile.squeeze(0)) * weight
        return acc + out.to(acc.dtype), None

    result, _ = for_each_tile(
        expert_body,
        (x, score, gate, up, down),
        dims=(None, Gather(1, ids), Gather(0, ids), Gather(0, ids), Gather(0, ids)),
        tile_size=1,
        init=torch.zeros_like(x),
    )
    return result
