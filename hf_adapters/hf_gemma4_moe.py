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

"""HuggingFace adapter for the sparse Gemma 4 MoE causal-LM on Spyre.

Targets ``google/gemma-4-26B-A4B-it``, whose ``Gemma4TextDecoderLayer`` runs a
dense MLP in parallel with a top-K MoE FFN. Everything attention-side is shared
with the dense ``hf_gemma4`` adapter; this module adds only the sparse FFN and
its block/prepare/forward wiring. All MoE compute runs on device.

The FFN has two device formulations, chosen per forward by sequence length
directly in ``Gemma4MoEBlock.forward``:

  * prefill (``seq_len > 1``): route once (``_moe_route_persistent_packed``),
    then run the dense all-expert value path (``_moe_expert_persistent``) under
    a coarse-tile ``spyre_hint(num_tiles_per_dim={"E": E})`` so it lowers to one
    counted device loop that accumulates over experts.
  * decode (``seq_len == 1``, ``_compiled_moe_loop_region``): experts stay
    HBM-resident; the top-K rows are gathered on-device by expert id, run
    through the per-row expert GEMM in ``[T,K,·]`` batch form, and combined by
    an on-device reduction over K.

Each of those three is compiled once in ``Gemma4MoEBlock.__init__`` and called
inline from ``forward``; the router surface lives inside every region so the
raw residual is routed without double-normalization.

Both paths read the same single expert-weight set (~42.5 GiB / 30 layers), laid
out expert-dim-outermost with the free dim on the stick by
``prepare_for_spyre`` -- simultaneously the decode gather-source layout and the
prefill matmul weight-operand layout, so neither path restickifies. Router
weights are device-resident and shared by both paths.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_spyre._inductor import config as spyre_config
from torch_spyre.model_utils import (
    dma_moe_expert_weight_to_spyre,
    dma_moe_per_expert_scale_to_spyre,
)

from hf_adapters.hf_common import text_config
from hf_adapters.hf_gemma4 import (
    Gemma4Attention,
    _gemma4_backbone,
    _run_backbone_forward,  # re-exported: block-agnostic, drives _spyre_compiled_blocks
    _run_forward,  # re-exported: block-agnostic backbone + LM head + softcap
    _setup_gemma4_text_decoder,
)

# Block-agnostic (drive ``model._spyre_compiled_blocks``); re-exported from
# ``hf_gemma4`` because ``resolve_adapter_module`` / ``generate`` look them up
# on the resolved adapter.
__all__ = ["prepare_for_spyre", "_run_forward", "_run_backbone_forward"]

# Ceiling of the Spyre topk reduction (torch-spyre #3782 raised it 4 -> 128).
_MOE_MAX_K = 128

# Row-tile size for the decode loop region's ``spyre_hint(tiles={"row": ...})``.
_MOE_TILE = 32

# Router logits are padded to a non-pow2, non-stick-multiple width before topk
# so the backend's binary-tree tiling stays stick-aligned; pad columns are
# ~-inf so they never win top-K. 160 (=E+32) is the proven value for E=128.
_MOE_PADW = 160
_MOE_PAD_NEG = -30000.0  # ~ -inf in fp16




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
    K,
    tile,
    eps,
):
    """Whole decode MoE FFN on device, combine included.

    Router (scale-free RMSNorm on the raw residual x_router, then router_scale[H]
    and the root-size scalar, then proj) -> softmax -> topk(K) -> renorm; then,
    under one ``spyre_hint(tiles={"row": tile})`` that tiles the [T,K] row axis,
    gather the per-(token,expert) weights from the HBM-resident E-outermost
    stacks, run the SwiGLU expert GEMM in [T,K,·] batch form, weight by the
    routing scalar, and reduce over K.

    Router surface: the stock router.norm is scale-free (no .weight); the gain
    is router_scale [H], applied after the norm. router_scalar_root_size is
    ``hidden_size ** -0.5`` (float); eps is config.rms_norm_eps.

    gate_dev/up_dev [E,H,M], down_dev [E,M,H], E outermost -- the shared layout
    _moe_expert_persistent reads. tile must be >= 2 (single-row gather
    SIGABRTs). Returns [T,H].
    """
    from torch_spyre._inductor.propagate_hints import spyre_hint

    T, H = x_expert.shape

    var = x_router.pow(2).mean(-1, keepdim=True)
    normed = x_router * torch.rsqrt(var + eps)  # scale-free RMSNorm
    normed = normed * router_scale * router_scalar_root_size
    probs = torch.softmax(F.linear(normed, router_proj_w), dim=-1)  # [T,E]
    w, idx = torch.topk(probs, K, dim=-1)  # values, fp16 positions [T,K]
    w = w / w.sum(-1, keepdim=True)

    # Decode index prep. topk returns positions in the input dtype (fp16), but
    # advanced indexing needs an integer index, and the backend wants the gather
    # index widened onto a stick. Expand a dummy-64 axis and .contiguous() so the
    # layout pass inserts the restickify (torch.ops.spyre.restickify is
    # pass-inserted-only and cannot be traced from user code); widen to fp32,
    # slice to one stick, take lane 0, cast to int32 (the device index width).
    idx_stick = idx[..., None].expand(T, K, 64).contiguous()  # [T,K,64] fp16
    idx_stick = idx_stick.to(torch.float32)
    idx_addr = idx_stick[..., :32]  # one fp32 stick
    idx = idx_addr[..., 0].to(torch.int32)  # [T,K] gather index

    with spyre_hint(tiles={"row": tile}):
        # The K rows for one token read the same embedding, so broadcast over K.
        # .contiguous() materializes the K-batch stride; without it the matmul
        # LHS drops its batch dim and the backend scheduler aborts on
        # inp0_reuse_dim != 1 (L3DlOpsScheduler.cpp:945).
        gathered = x_expert[:, None, :].expand(T, K, H).contiguous()  # [T,K,H]
        W_g = gate_dev[idx]  # [T,K,H,M] on-device index_select
        W_u = up_dev[idx]  # [T,K,H,M]
        W_dn = down_dev[idx]  # [T,K,M,H]
        # Each (t,k) row is a distinct vector x gathered-matrix product with no
        # shared-weight bmm form. Express it as broadcast-multiply + a GENUINE
        # .sum reduction: the reduction physically collapses the contracted axis
        # to a rank-3 buffer, whereas a matmul's [T,K,1,M] result squeezed to
        # rank-3 stays a rank-4 view and trips "Incompatible host_size and
        # dim_order" at the downstream scale (and einsum mis-derives the output
        # shape at fake-tensor inference).
        g = (gathered[..., None] * W_g).sum(dim=2)  # [T,K,M]
        u = (gathered[..., None] * W_u).sum(dim=2)  # [T,K,M]
        act = F.gelu(g, approximate="tanh") * u  # [T,K,M]
        row_out = (act[..., None] * W_dn).sum(dim=2)  # [T,K,H]
        # per_expert_scale arrives host-widened to a sticked [E,64] device tensor
        # (see dma_moe_per_expert_scale_to_spyre) so indirect access has a stick
        # to gather; every lane is the same scalar, so slice lane 0 rather than
        # reduce (a reduction over a gathered stick's innermost dim mis-lowers to
        # inf on device).
        pscale = per_expert_scale[idx][..., :1]  # [T,K,1]
        # Fold both scales into the H-carrying row_out (H on the stick); a bare
        # [T,K] product has no legal layout (K=8 not stick-divisible).
        row_out = row_out * w[..., None] * pscale  # [T,K,H]
        # Combine a token's K expert rows by an on-device reduction over K. A
        # plain sum, not a scatter: an on-device index_add is unusable here
        # (spyre_index_add requires no duplicate indices, but all K rows of a
        # token share index t).
        moe_out = row_out.sum(dim=1)  # [T,H]
    return moe_out


def _moe_route_persistent_packed(
    x_router,
    router_proj_w,
    router_scale,
    router_scalar_root_size,
    per_expert_scale,
    K,
    eps,
    pad_w,
    pad_neg,
    route_identity,
):
    """All-device prefill router: one broadcast-ready stick per route scalar.

    Router surface matches _compiled_moe_loop_region (scale-free norm on the raw
    residual, then [H] router_scale, then the root-size float, then proj). topk
    over the pow2 stick-multiple width E aborts on-card ("Incorrect chunk
    size"), so the logits are padded to the non-pow2 pad_w with pad_neg (~-inf)
    before topk; the pad columns never win. keep_by_index keeps probs at the
    top-K expert coordinates and zeros the rest -- POSITIONAL selection by topk
    index, so ties at the kth value cannot leak extra experts. The renormed
    top-K (scaled by per_expert_scale, zero elsewhere) is then broadcast onto a
    physical stick via a BMM against route_identity (I64) so the persistent
    expert path can read it lane-major. Returns [T,E,64].
    """
    T, _ = x_router.shape
    E = router_proj_w.shape[0]
    var = x_router.pow(2).mean(-1, keepdim=True)
    normed = x_router * torch.rsqrt(var + eps)  # scale-free RMSNorm
    normed = normed * router_scale * router_scalar_root_size
    probs = torch.softmax(F.linear(normed, router_proj_w), dim=-1)  # [T,E]
    pad = torch.full((T, pad_w - E), pad_neg, dtype=probs.dtype,
                     device=probs.device)
    padded = torch.cat([probs, pad], dim=-1)  # [T,pad_w]
    _, sel = torch.topk(padded, K, dim=-1)  # [T,K] expert ids (< E)
    mask = torch.ops.spyre.keep_by_index(probs, sel, -1, 0.0)  # [T,E]
    w = mask / mask.sum(-1, keepdim=True)  # renorm top-K to sum 1
    token_major = w * per_expert_scale  # [T,E]
    # Put the broadcast lane on the physical stick via a BMM against the
    # identity (values unchanged, route_identity is I64).
    expanded = torch.relu(token_major.unsqueeze(-1).expand(-1, -1, 64))
    return expanded @ route_identity


def _moe_expert_persistent(x_expert, routing_weight, gate, up, down, K):
    """Run the dense all-expert body as one coarse-tile-hinted program.

    Declares the dims, names each operand, then runs the matmuls under
    ``spyre_hint(num_tiles_per_dim={"E": E})`` (one tile per expert) so they
    lower to one counted device loop accumulating over experts.

    Shapes: x_expert [T,H]; gate/up [E,H,M]; down [E,M,H]; routing_weight
    [T,E,1]. gate/up/down arrive in the shared device layout (E outermost, free
    dim on stick) from prepare_for_spyre.
    """
    from torch_spyre._inductor.propagate_hints import spyre_hint
    from torch_spyre._inductor.wsr.propagate_named_dims import (
        declare_tensor_dim,
        name_tensor_dims,
    )

    experts, hidden, inter = gate.shape
    tokens = x_expert.shape[0]
    for name, extent in (
        ("E", experts),
        ("T", tokens),
        ("H", hidden),
        ("M", inter),
        ("ONE", 1),
    ):
        declare_tensor_dim(name, extent)

    x_singleton = x_expert.unsqueeze(0)
    # routing_weight is a size-1 slice over a 64-wide broadcast stick, so its
    # [T,E,1] view still carries 64-lane strides and the coarse-tile size
    # derivation under-counts T. .clone() forces an owned dense [E,T,1] buffer
    # (.contiguous() alone can be elided upstream).
    route = routing_weight.permute(1, 0, 2).contiguous().clone()  # [E,T,ONE]
    name_tensor_dims(x_expert, ["T", "H"])
    name_tensor_dims(gate, ["E", "H", "M"])
    name_tensor_dims(up, ["E", "H", "M"])
    name_tensor_dims(down, ["E", "M", "H"])
    name_tensor_dims(route, ["E", "T", "ONE"])
    with spyre_hint(num_tiles_per_dim={"E": experts}, work_div={"T": 32}):
        gate_out = torch.matmul(x_singleton, gate)  # [1,T,H]@[E,H,M]->[E,T,M]
        up_out = torch.matmul(x_singleton, up)
        activated = F.gelu(gate_out, approximate="tanh") * up_out
        down_out = torch.matmul(activated, down)  # [E,T,M]@[E,M,H]->[E,T,H]
        return (down_out * route).sum(dim=0)  # [T,H]


class Gemma4MoEBlock(nn.Module):
    """Registered Gemma 4 MoE decoder block.

    Mirrors the dense ``hf_gemma4.Gemma4Block`` (same class shape, 7-arg call
    signature, ``layer_scalar`` idiom) but its forward reproduces the
    ``enable_moe_block=True`` branch of ``Gemma4TextDecoderLayer.forward``: a
    dense MLP in parallel with a top-K MoE FFN, combined via
    ``post_feedforward_layernorm(h_dense + h_moe)``.

    Attention is ``Gemma4Attention`` composed verbatim (same KV path the dense
    adapter is tested on). The block is not a single ``torch.compile`` -- the
    dense MLP and the two FFN-phase regions are each compiled once in
    ``__init__`` and dispatched per forward. The router reads the RAW flattened
    residual (its own norm is inside the FFN region); the experts read a
    SEPARATE pre_ff_ln_2 norm of that residual. Expert weights (set by
    ``prepare_for_spyre``) are read fresh off ``self`` at call time.
    """

    def __init__(self, layer, num_q_heads, num_kv_heads, head_dim, is_kv_eq_v):
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
        # MoE-branch submodules / norms (stock Gemma4TextDecoderLayer names).
        self.router = layer.router
        self.post_feedforward_layernorm_1 = layer.post_feedforward_layernorm_1
        self.pre_feedforward_layernorm_2 = layer.pre_feedforward_layernorm_2
        self.post_feedforward_layernorm_2 = layer.post_feedforward_layernorm_2
        self.register_buffer(
            "layer_scalar",
            layer.layer_scalar,
            persistent="layer_scalar" not in layer._non_persistent_buffers_set,
        )
        self._moe_k = layer._spyre_moe_k
        self._moe_rms_eps = self.pre_feedforward_layernorm_2.eps

        # Compiled device regions, built once: dense MLP, plus the prefill
        # (persistent) and decode (loop) FFN regions.
        self._compiled_mlp = torch.compile(self.mlp, dynamic=False)
        self._compiled_loop = torch.compile(
            _compiled_moe_loop_region, dynamic=False
        )
        self._compiled_persistent_route = torch.compile(
            _moe_route_persistent_packed, dynamic=False
        )
        self._compiled_persistent = torch.compile(
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
        h = self.input_layernorm(hidden_states)
        attn_out, key_cache, value_cache = self.self_attn(
            h,
            selected_freqs,
            attn_mask,
            key_cache,
            value_cache,
            cache_index,
        )
        # Sandwich: norm the attention output BEFORE adding the residual.
        h = residual + self.post_attention_layernorm(attn_out)

        residual = h
        bsz, seq_len, hidden = h.shape

        # Dense branch: pre_ff_ln -> mlp -> post_ff_ln_1.
        h_dense = self.post_feedforward_layernorm_1(
            self._compiled_mlp(self.pre_feedforward_layernorm(residual))
        )

        # Sparse branch: the router reads the RAW residual (its own norm is
        # inside the FFN region); the experts read a SEPARATE pre_ff_ln_2 norm
        # of that same residual. Thread both so the router isn't
        # double-normalized.
        flat = residual.reshape(-1, hidden)  # [T,H] RAW -> router
        x_moe = self.pre_feedforward_layernorm_2(flat)  # [T,H] normed -> experts
        # Phase dispatch by shape. Both paths read the same E-outermost expert
        # stacks and the device-resident router; the router surface (scale-free
        # RMSNorm on the RAW residual, then [H] scale, then the root-size float,
        # then proj) lives inside each compiled region.
        router = self.router
        if seq_len > 1:
            # Prefill: route once to a broadcast stick, then run the persistent
            # all-expert value path (one E-counted device loop). routing_sticks
            # is [T,E,64]; slice lane 0 to the [T,E,1] the expert path reads.
            routing_sticks = self._compiled_persistent_route(
                flat,
                router.proj.weight,
                router.scale,
                router.scalar_root_size,
                router.per_expert_scale,
                self._moe_k,
                self._moe_rms_eps,
                _MOE_PADW,
                _MOE_PAD_NEG,
                self._spyre_persistent_route_identity,
            )
            routing_weight = routing_sticks[..., :1]  # [T,E,1] lane-0 view
            with spyre_config.patch(
                {
                    "sencores": 32,
                    "lx_planning": True,
                    "allow_all_ops_in_lx_planning": True,
                }
            ):
                moe_out = self._compiled_persistent(
                    x_moe,
                    routing_weight,
                    self._spyre_gate,
                    self._spyre_up,
                    self._spyre_down,
                    self._moe_k,
                )  # [T,H]
        else:
            # Decode: experts stay HBM-resident; the loop region gathers the
            # top-K rows by expert id and reduces over K on device. Decode
            # gathers per_expert_scale by expert id, so pass the host-widened
            # sticked [E,64] source, not the 1-D [E] the prefill router uses.
            moe_out = self._compiled_loop(
                flat,
                x_moe,
                router.proj.weight,
                router.scale,
                router.scalar_root_size,
                router.per_expert_scale_stick,
                self._spyre_gate,
                self._spyre_up,
                self._spyre_down,
                self._moe_k,
                _MOE_TILE,
                self._moe_rms_eps,
            ).to(dtype=x_moe.dtype)  # [T,H]
        moe_out = moe_out.reshape(bsz, seq_len, hidden)
        h_moe = self.post_feedforward_layernorm_2(moe_out)

        # Combine dense + MoE, final sandwich norm, residual, per-layer scalar.
        h = self.post_feedforward_layernorm(h_dense + h_moe)
        h = residual + h
        return h * layer_scalar, key_cache, value_cache


def prepare_for_spyre(model):
    """Apply Spyre adaptations to a Gemma 4 MoE causal-LM in-place.

    Reuses the shared attention-side prep (``_setup_gemma4_text_decoder``) and
    adds the MoE steps: assert ``enable_moe_block=True`` and the topk ceiling;
    de-fuse ``gate_up`` into ``gate``/``up`` and lay all three out in one shared
    device layout (expert dim outermost, free dim on stick) read by both FFN
    paths; move the router weights to device; build
    ``model._spyre_compiled_blocks``.
    """
    backbone = _gemma4_backbone(model)
    cfg = text_config(model.config)

    assert getattr(cfg, "enable_moe_block", False), (
        "hf_gemma4_moe requires an MoE checkpoint (enable_moe_block=True); "
        "use hf_gemma4 for the dense variants."
    )
    moe_k = int(cfg.top_k_experts)
    assert 1 <= moe_k <= _MOE_MAX_K, (
        f"top_k_experts ({moe_k}) must be in [1, {_MOE_MAX_K}]; the Spyre topk "
        f"reduction caps k at {_MOE_MAX_K}."
    )
    # The expert SwiGLU hardcodes gelu(approximate="tanh"); guard so a variant
    # with a different activation fails loudly.
    act_fn = getattr(cfg, "hidden_activation", None)
    assert act_fn == "gelu_pytorch_tanh", (
        "hf_gemma4_moe expert SwiGLU is fixed to gelu(approximate='tanh'); "
        f"config hidden_activation={act_fn!r} is unsupported."
    )

    E = cfg.num_experts
    assert _MOE_PADW > E and (_MOE_PADW & (_MOE_PADW - 1)) != 0, (
        f"_MOE_PADW ({_MOE_PADW}) must exceed num_experts ({E}) and be "
        "non-power-of-two (topk-pad fix)."
    )

    num_q_heads_per_layer, kv_shapes, is_kv_eq_v_per_layer = (
        _setup_gemma4_text_decoder(model, allow_moe=True)
    )

    # One pass per layer: build the MoE block, attach its device-resident expert
    # weights, register it back into backbone.layers[i] (so
    # _run_blocks_over_embeds reads layer_scalar off it), and collect it.
    compiled_blocks = []
    for i, layer in enumerate(list(backbone.layers)):
        layer._spyre_moe_k = moe_k  # captured by Gemma4MoEBlock.__init__
        block = Gemma4MoEBlock(
            layer,
            num_q_heads_per_layer[i],
            kv_shapes[i][0],
            kv_shapes[i][1],
            is_kv_eq_v_per_layer[i],
        )
        experts = layer.experts
        # gate_up_proj: [E,2M,H] -> [E,H,2M]; down_proj: [E,H,M] -> [E,M,H].
        gate_up_t = experts.gate_up_proj.data.transpose(1, 2).contiguous()
        down_t = experts.down_proj.data.transpose(1, 2).contiguous()
        del experts.gate_up_proj
        del experts.down_proj
        router = block.router  # shared with layer; moves below apply to both
        # De-fuse gate_up and lay all three out E-outermost with the free dim on
        # the stick (dma_moe_expert_weight_to_spyre): one shared set read by both
        # FFN paths, ~42.5 GiB / 30 layers.
        M = gate_up_t.shape[2] // 2
        gate_l = gate_up_t[:, :, :M].contiguous()  # [E,H,M]
        up_l = gate_up_t[:, :, M:].contiguous()  # [E,H,M]
        block._spyre_gate = dma_moe_expert_weight_to_spyre(gate_l)  # [E,H,M]
        block._spyre_up = dma_moe_expert_weight_to_spyre(up_l)  # [E,H,M]
        block._spyre_down = dma_moe_expert_weight_to_spyre(down_t)  # [E,M,H]
        # Fall back to a plain device move if the free dim doesn't tile into
        # sticks (won't happen for gemma-4: M=704, H=2816 both divide 64).
        if block._spyre_gate is None:
            block._spyre_gate = gate_l.to("spyre")
        if block._spyre_up is None:
            block._spyre_up = up_l.to("spyre")
        if block._spyre_down is None:
            block._spyre_down = down_t.to("spyre")

        # Identity the packed router BMMs against to move the broadcast lane
        # onto the stick.
        block._spyre_persistent_route_identity = torch.eye(
            64, dtype=gate_up_t.dtype
        ).to("spyre")

        # The router runs on-device in both paths, so its weights must be
        # device-resident. Reassign the Parameter (cross-backend param.data=...
        # raises on the type change). scalar_root_size is a float (no move).
        router.proj.weight = torch.nn.Parameter(
            router.proj.weight.data.to("spyre"), requires_grad=False
        )
        router.scale = torch.nn.Parameter(
            router.scale.data.to("spyre"), requires_grad=False
        )
        # per_expert_scale is used two ways: the prefill router broadcasts the
        # 1-D [E] tensor ([T,E]*[E]); the decode loop gathers it by expert id,
        # which needs a sticked [E,64] source (widening [E]->[E,64] in-graph is a
        # broadcast-into-stick the layout pass can't express, so widen on host).
        pes_cpu = router.per_expert_scale.data  # still on host here
        pes_stick = dma_moe_per_expert_scale_to_spyre(pes_cpu)
        if pes_stick is None:
            pes_stick = pes_cpu[:, None].expand(-1, 64).contiguous().to("spyre")
        router.per_expert_scale = torch.nn.Parameter(
            pes_cpu.to("spyre"), requires_grad=False
        )
        router.per_expert_scale_stick = torch.nn.Parameter(
            pes_stick, requires_grad=False
        )

        # The block's forward is eager glue that dispatches to the inner
        # compiled device regions (per-phase); it is not itself torch.compile'd,
        # so register the block as-is and invoke it directly.
        backbone.layers[i] = block
        compiled_blocks.append(block)

    model._spyre_compiled_blocks = compiled_blocks


# _run_backbone_forward / _run_forward are re-exported from hf_gemma4 (see the
# top import block); they are block-agnostic, so the MoE blocks need no
# MoE-specific forward.
