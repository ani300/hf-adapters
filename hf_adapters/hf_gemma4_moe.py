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

"""Spyre adapter for the sparse Gemma 4 MoE causal LM.

The attention path comes from :mod:`hf_gemma4`. Prefill routes tokens before
evaluating every expert; single-token decode gathers only the selected experts.
Both paths share one device-resident expert-weight set.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_spyre._C import get_elem_in_stick
from torch_spyre._inductor import config as spyre_config
from torch_spyre.model_utils import (
    dma_moe_expert_weight_to_spyre,
    dma_moe_per_expert_scale_to_spyre,
)

from hf_adapters.hf_common import text_config
from hf_adapters.hf_gemma4 import (
    Gemma4Attention,
    _gemma4_backbone,
    _run_backbone_forward,
    _run_forward,
    _setup_gemma4_text_decoder,
)

__all__ = ["prepare_for_spyre", "_run_forward", "_run_backbone_forward"]

_MOE_TILE = 32  # Decode gather requires tiles with at least two rows.
_STICK_SIZE = get_elem_in_stick(torch.float16)


def _router_probs(x, weight, scale, root_size, eps):
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return torch.softmax(F.linear(x * scale * root_size, weight), dim=-1)


def _select_experts(probs, top_k):
    tokens = probs.shape[0]
    topk_input = probs.expand(2, -1).contiguous() if tokens == 1 else probs
    weights, expert_indices = torch.topk(topk_input, top_k, dim=-1)
    weights = weights[:tokens]
    expert_indices = expert_indices[:tokens]
    return weights / weights.sum(-1, keepdim=True), expert_indices


def _compiled_moe_loop_region(
    x_router,
    x_expert,
    router_proj_w,
    router_scale,
    router_scalar_root_size,
    per_expert_scale,
    gate_dev,
    up_dev,
    down_dev,
    top_k,
    tile,
    eps,
):
    """Run the routed decode FFN and combine its expert outputs on device."""
    from torch_spyre._inductor.propagate_hints import spyre_hint

    T, H = x_expert.shape
    probs = _router_probs(
        x_router,
        router_proj_w,
        router_scale,
        router_scalar_root_size,
        eps,
    )
    weights, expert_indices = _select_experts(probs, top_k)

    # Widen topk's fp16 indices onto a stick before converting them to the
    # device's int32 gather indices. The layout pass inserts the restickify.
    index_stick = expert_indices[..., None].expand(T, top_k, _STICK_SIZE).contiguous()
    index_stick = index_stick.to(torch.float32)
    index_address = index_stick[..., : _STICK_SIZE // 2]
    expert_indices = index_address[..., 0].to(torch.int32)

    with spyre_hint(tiles={"row": tile}):
        # A real K-batch stride is required by the backend scheduler.
        inputs = x_expert[:, None, :].expand(T, top_k, H).contiguous()
        gate = gate_dev[expert_indices]
        up = up_dev[expert_indices]
        down = down_dev[expert_indices]

        # Explicit reductions produce rank-3 buffers; equivalent batched
        # matmuls leave rank-4 views that the layout pass cannot consume.
        gate_out = (inputs[..., None] * gate).sum(dim=2)
        up_out = (inputs[..., None] * up).sum(dim=2)
        activated = F.gelu(gate_out, approximate="tanh") * up_out
        expert_out = (activated[..., None] * down).sum(dim=2)

        # Scale on the H-carrying tensor because bare [T,K] products have no
        # legal layout. The widened source gives the gather a physical stick.
        expert_scale = per_expert_scale[expert_indices][..., :1]
        expert_out = expert_out * weights[..., None] * expert_scale
        return expert_out.sum(dim=1)


def _moe_route_persistent_packed(
    x_router,
    router_proj_w,
    router_scale,
    router_scalar_root_size,
    per_expert_scale,
    top_k,
    eps,
    route_identity,
):
    """Compute packed prefill routing weights on device."""
    probs = _router_probs(
        x_router,
        router_proj_w,
        router_scale,
        router_scalar_root_size,
        eps,
    )
    _, selected = _select_experts(probs, top_k)
    weights = torch.ops.spyre.keep_by_index(probs, selected, -1, 0.0)
    weights = weights / weights.sum(-1, keepdim=True)
    weights = weights * per_expert_scale

    # ReLU materializes the expansion; the identity BMM puts it on a stick.
    packed = torch.relu(weights.unsqueeze(-1).expand(-1, -1, _STICK_SIZE))
    return packed @ route_identity


def _moe_expert_persistent(x_expert, routing_weight, gate, up, down):
    """Evaluate every expert and sum their routed outputs on device."""
    from torch_spyre._inductor.propagate_hints import spyre_hint
    from torch_spyre._inductor.wsr.propagate_named_dims import (
        declare_tensor_dim,
        name_tensor_dims,
    )

    experts, hidden, intermediate = gate.shape
    tokens = x_expert.shape[0]
    for name, extent in (
        ("E", experts),
        ("T", tokens),
        ("H", hidden),
        ("M", intermediate),
        ("ONE", 1),
    ):
        declare_tensor_dim(name, extent)

    # clone() turns the stick slice into an owned dense [E,T,1] buffer.
    x = x_expert.unsqueeze(0)
    route = routing_weight.permute(1, 0, 2).contiguous().clone()
    name_tensor_dims(x_expert, ["T", "H"])
    name_tensor_dims(gate, ["E", "H", "M"])
    name_tensor_dims(up, ["E", "H", "M"])
    name_tensor_dims(down, ["E", "M", "H"])
    name_tensor_dims(route, ["E", "T", "ONE"])

    with spyre_hint(num_tiles_per_dim={"E": experts}, work_div={"T": 32}):
        gate_out = torch.matmul(x, gate)
        up_out = torch.matmul(x, up)
        activated = F.gelu(gate_out, approximate="tanh") * up_out
        down_out = torch.matmul(activated, down)
        return (down_out * route).sum(dim=0)


class Gemma4MoEBlock(nn.Module):
    """Gemma 4 decoder block with parallel dense and sparse FFNs."""

    def __init__(
        self,
        layer,
        num_q_heads,
        num_kv_heads,
        head_dim,
        is_kv_eq_v,
        moe_k,
    ):
        super().__init__()
        self.self_attn = Gemma4Attention(
            layer.self_attn,
            num_q_heads,
            num_kv_heads,
            head_dim,
            is_kv_eq_v,
        )
        self.mlp = layer.mlp
        self.input_layernorm = layer.input_layernorm
        self.post_attention_layernorm = layer.post_attention_layernorm
        self.pre_feedforward_layernorm = layer.pre_feedforward_layernorm
        self.post_feedforward_layernorm = layer.post_feedforward_layernorm
        self.experts = layer.experts
        self.router = layer.router
        self.post_feedforward_layernorm_1 = layer.post_feedforward_layernorm_1
        self.pre_feedforward_layernorm_2 = layer.pre_feedforward_layernorm_2
        self.post_feedforward_layernorm_2 = layer.post_feedforward_layernorm_2
        self.register_buffer(
            "layer_scalar",
            layer.layer_scalar,
            persistent="layer_scalar" not in layer._non_persistent_buffers_set,
        )
        self._moe_k = moe_k
        self._moe_rms_eps = self.router.eps
        self._compiled_mlp = torch.compile(self.mlp, dynamic=False)
        self._compiled_decode = torch.compile(_compiled_moe_loop_region, dynamic=False)
        self._compiled_prefill_router = torch.compile(
            _moe_route_persistent_packed, dynamic=False
        )
        self._compiled_prefill_experts = torch.compile(
            _moe_expert_persistent, dynamic=False
        )
        self.train(layer.training)

    def forward(
        self,
        hidden_states,
        selected_freqs,
        attn_mask,
        key_cache,
        value_cache,
        cache_index,
        layer_scalar,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_out, key_cache, value_cache = self.self_attn(
            hidden_states,
            selected_freqs,
            attn_mask,
            key_cache,
            value_cache,
            cache_index,
        )
        hidden_states = residual + self.post_attention_layernorm(attn_out)

        residual = hidden_states
        batch_size, seq_len, hidden_size = residual.shape
        dense_out = self.post_feedforward_layernorm_1(
            self._compiled_mlp(self.pre_feedforward_layernorm(residual))
        )

        # The router reads the raw residual; experts use their own normalization.
        router_input = residual.reshape(-1, hidden_size)
        expert_input = self.pre_feedforward_layernorm_2(router_input)
        experts = self.experts
        router = self.router

        if seq_len > 1:
            routing_weight = self._compiled_prefill_router(
                router_input,
                router.proj.weight,
                router.scale,
                router.scalar_root_size,
                router.per_expert_scale,
                self._moe_k,
                self._moe_rms_eps,
                router.route_identity,
            )[..., :1]
            with spyre_config.patch(
                {
                    "sencores": 32,
                    "lx_planning": True,
                    "allow_all_ops_in_lx_planning": True,
                }
            ):
                moe_out = self._compiled_prefill_experts(
                    expert_input,
                    routing_weight,
                    experts.gate_proj,
                    experts.up_proj,
                    experts.down_proj,
                )
        else:
            moe_out = self._compiled_decode(
                router_input,
                expert_input,
                router.proj.weight,
                router.scale,
                router.scalar_root_size,
                router.per_expert_scale_stick,
                experts.gate_proj,
                experts.up_proj,
                experts.down_proj,
                self._moe_k,
                _MOE_TILE,
                self._moe_rms_eps,
            )

        moe_out = moe_out.to(expert_input.dtype).reshape(
            batch_size, seq_len, hidden_size
        )
        moe_out = self.post_feedforward_layernorm_2(moe_out)
        ffn_out = self.post_feedforward_layernorm(dense_out + moe_out)
        return (residual + ffn_out) * layer_scalar, key_cache, value_cache


def _move_expert_weight(weight):
    moved = dma_moe_expert_weight_to_spyre(weight)
    return moved if moved is not None else weight.to("spyre")


def _prepare_experts(experts):
    gate_up = experts.gate_up_proj.detach().transpose(1, 2).contiguous()
    down = experts.down_proj.detach().transpose(1, 2).contiguous()
    del experts.gate_up_proj
    del experts.down_proj

    intermediate_size = gate_up.shape[2] // 2
    gate = gate_up[:, :, :intermediate_size].contiguous()
    up = gate_up[:, :, intermediate_size:].contiguous()
    experts.gate_proj = _move_expert_weight(gate)
    experts.up_proj = _move_expert_weight(up)
    experts.down_proj = _move_expert_weight(down)


def prepare_for_spyre(model):
    """Prepare a Gemma 4 MoE causal LM for Spyre in place."""
    backbone = _gemma4_backbone(model)
    cfg = text_config(model.config)

    assert getattr(cfg, "enable_moe_block", False), (
        "hf_gemma4_moe requires an MoE checkpoint (enable_moe_block=True); "
        "use hf_gemma4 for the dense variants."
    )
    moe_k = int(cfg.top_k_experts)
    num_q_heads, kv_shapes, kv_equals_v = _setup_gemma4_text_decoder(
        model, allow_moe=True
    )

    blocks = []
    for i, layer in enumerate(list(backbone.layers)):
        block = Gemma4MoEBlock(
            layer,
            num_q_heads[i],
            kv_shapes[i][0],
            kv_shapes[i][1],
            kv_equals_v[i],
            moe_k,
        )
        expert_scale = block.router.per_expert_scale.detach()
        block.router.route_identity = torch.eye(
            _STICK_SIZE, dtype=expert_scale.dtype
        ).to("spyre")
        block.router.per_expert_scale_stick = dma_moe_per_expert_scale_to_spyre(
            expert_scale
        )
        _prepare_experts(block.experts)
        backbone.layers[i] = block
        blocks.append(block)

    model._spyre_compiled_blocks = blocks
