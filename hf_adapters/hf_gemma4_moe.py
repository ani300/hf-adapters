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

"""
HuggingFace Transformers adapter for the Gemma 4 **MoE** causal-LM on Spyre.

Targets the sparse ``google/gemma-4-26B-A4B-it`` variant, whose
``Gemma4TextDecoderLayer`` runs a dense MLP **in parallel** with a top-K
mixture-of-experts FFN when ``config.enable_moe_block=True`` (see stock
``transformers.models.gemma4.modeling_gemma4``). The dense attention half and
all attention-side Spyre prep (RMSNorm patch, per-type RoPE, KV shapes,
LM-head padding) are shared with the dense adapter ``hf_gemma4`` — this module
only adds the sparse FFN and the surrounding block/prepare/forward wiring.

The FFN runs on ONE of two device formulations, chosen per forward by the
sequence length (``Gemma4MoEBlock.forward`` dispatches on ``seq_len``):

  * **prefill** (``seq_len > 1``, ``_moe_ffn_persistent``) -- the router runs
    once, then the dense all-expert value path is emitted as an ordinary hinted
    PyTorch program: ``x @ gate``/``x @ up`` (SwiGLU) ``@ down`` for every
    expert, run under a coarse-tile ``spyre_hint(num_tiles_per_dim={"E": E})``
    so it lowers to one counted device loop (one expert body per trip, the
    running sum kept in the LX accumulator). No custom op -- plain matmuls named
    with ``declare_tensor_dim`` / ``name_tensor_dims``; the per-expert routing
    weight rides in as a named ``[E,T,1]`` operand and the outputs are summed
    over the expert axis.
  * **decode** (``seq_len == 1``, ``_moe_ffn_loop`` / ``_compiled_moe_loop_region``)
    -- experts stay HBM-resident and the top-K rows are gathered on-device with
    ``index_select`` under a row-tiled ``spyre_hint``, then the per-row expert
    GEMM (gather ``gate``/``up``/``down`` by expert id, SwiGLU, weight by the
    routing scalar) runs in ``[T,K,·]`` batch form; the host does only the
    ``index_add`` scatter-combine back to ``[T,H]``.

Both paths read the **same single expert-weight set**. ``prepare_for_spyre``
de-fuses the checkpoint's fused ``gate_up`` into separate ``gate``/``up`` and
lays each of ``gate``/``up``/``down`` out in ONE shared device layout via
``torch_spyre.model_utils.dma_moe_expert_weight_to_spyre`` -- expert dim
**outermost** with the free (output) dim on the stick
(``gate``/``up`` ``[E,H,M]``, ``down`` ``[E,M,H]``, each becoming
``[E, C, F//eps, eps]`` on device). That one layout is simultaneously the
gather-source layout the decode ``index_select`` needs (indexed dim outermost)
and the matmul weight-operand layout the prefill hint-body needs (sticked on
the free dim), so neither path restickifies. Only ONE weight set is
materialized (~42.5 GiB / 30 layers, not the old dual ~85 GiB), so it fits the
card. Router weights (``proj.weight``, ``scale``, ``per_expert_scale``) are
device-resident too; both paths share the same on-device router.

``K`` is the checkpoint's real ``config.top_k_experts`` (8 for gemma-4-26B-A4B);
``prepare_for_spyre`` asserts only the Spyre topk ceiling ``K <= 128`` (raised
from 4 by torch-spyre #3782, which splits topk's k across cores).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_spyre.model_utils import dma_moe_expert_weight_to_spyre

from hf_adapters.hf_common import text_config
from hf_adapters.hf_gemma4 import (
    Gemma4Attention,
    _gemma4_backbone,
    _run_backbone_forward,  # re-exported: block-agnostic, drives _spyre_compiled_blocks
    _run_forward,  # re-exported: block-agnostic backbone + LM head + softcap
    _setup_gemma4_text_decoder,
)

# ``_run_forward`` / ``_run_backbone_forward`` are re-exported from ``hf_gemma4``
# unchanged: since #350 they drive ``model._spyre_compiled_blocks`` and read
# ``layer_scalar`` off each registered block, so they are block-AGNOSTIC — the
# MoE blocks slot in transparently. Kept in this module's namespace because
# ``resolve_adapter_module`` / ``generate`` look them up on the resolved adapter.
__all__ = ["prepare_for_spyre", "_run_forward", "_run_backbone_forward"]

# Upper bound on the top-K the Spyre topk reduction can serve. torch-spyre
# #3782 split topk's k across cores and raised the ceiling from 4 to 128, so
# the adapter now uses the checkpoint's real ``top_k_experts`` (8 for
# gemma-4-26B-A4B) instead of the old bring-up pin, asserting only this ceiling.
_MOE_MAX_K = 128

# Row-tile size for the loop-on-topk device region (spec Approach A). The
# spyre_hint(tiles={"row": _MOE_TILE}) tiles the N=T*K row axis so the backend
# loops over ceil(N/_MOE_TILE) tiles; a tuning knob (scratchpad window size).
_MOE_TILE = 32

# topk-input pad width for the all-device router (topk-pad fix). The router
# logits width E is padded to this NON-pow2, non-stick-multiple width before
# topk so the backend's binary-tree tiling stays stick-aligned. For E=128 the
# proven value is 160 (=E+32). Pad columns are -inf so they never win top-K.
_MOE_PADW = 160
_MOE_PAD_NEG = -30000.0  # ~ -inf in fp16 (< any real softmax-prob logit)




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
    token_ids,
    K,
    tile,
    eps,
):
    """Whole MoE FFN on-device except the scatter-combine (spec Approach A).

    Router (inlined SCALE-FREE RMSNorm on the RAW residual x_router with eps
    INSIDE the sqrt, then * router_scale[H] * router_scalar_root_size, then
    proj) -> softmax -> topk(K) -> renorm -> per_expert_scale; then, under a
    single spyre_hint(tiles={"row": tile}) that tiles the [T,K] row axis (the
    hint IS the loop -- no Python for), broadcast the expert-input rows from
    x_expert over K, index_select the per-(token,expert) weights from the
    HBM-resident E-outermost stacks with the B5-B8 address-prep index
    (idx_addr [T,K,32] fp32; the backend's idx2Addr turns it into weight base
    addresses when it lowers the index_selects), two batched matmuls
    ([T,K,1,*] throughout) with a gelu-tanh SwiGLU, and weight by the router
    weight.

    Preflight-corrected router surface: the stock router.norm has NO .weight
    (Gemma4RMSNorm with_scale=False); the learnable gain is router_scale, an
    [H] vector applied AFTER the scale-free norm. router_scalar_root_size is a
    Python float (hidden_size ** -0.5). eps is config.rms_norm_eps.

    gate_dev/up_dev: [E,H,M] (stick=M), down_dev: [E,M,H] (stick=H), E outermost
    in all three -- the same shared device layout _moe_ffn_persistent reads.
    tile must be >= 2 (single-row P=1 gather SIGABRTs in dxp_standalone).
    Returns (row_out[T,K,H], token_of_row[T,K]) for the host index_add combine.
    """
    from torch_spyre._inductor.propagate_hints import spyre_hint

    T, H = x_expert.shape

    # --- router (all device-lowerable): SCALE-FREE RMSNorm (eps inside sqrt)
    # on the RAW residual, then the [H] scale vector and the root-size scalar.
    var = x_router.pow(2).mean(-1, keepdim=True)
    normed = x_router * torch.rsqrt(var + eps)  # scale-free (no gain in norm)
    normed = normed * router_scale * router_scalar_root_size  # scale is [H]
    logits = F.linear(normed, router_proj_w)  # [T,E]
    probs = torch.softmax(logits, dim=-1)
    # topk's index is kept in its native [T,K] shape (no reshape to N). The
    # spyre_topk decomposition leaves the index in fp16 (the device reduction
    # materializes positions in the input dtype; Spyre has no native int64), so
    # idx is an fp16 value that "lies" it is an index. Before it can drive an
    # on-device indirect gather it must be put in the address-prep form the
    # backend's idx2Addr step consumes (moe-implementation-notes-aug2026.md
    # B5-B8): replicate the index across a dummy 64-lane stick (B5 identity),
    # restickify that lane onto the stick (B6), widen fp16 -> fp32 for the
    # address arithmetic (B7), then slice to the 32 elements an fp32 stick holds
    # (B8). idx2Addr itself (B9-B11) is inserted by the backend when it lowers
    # the index_selects below -- we produce only the B5-B8 index, not addresses.
    w, idx = torch.topk(probs, K, dim=-1)  # [T,K] values, [T,K] fp16 indices
    w = w / w.sum(-1, keepdim=True)

    # B5-B8 index address-prep, shared by every indirect consumer below.
    idx_stick = idx[..., None].expand(T, K, 64)  # B5: replicate over dummy 64
    idx_stick = torch.ops.spyre.restickify(idx_stick)  # B6: dummy dim -> stick
    idx_addr = idx_stick.to(torch.float32)[..., :32]  # B7 widen, B8 slice -> fp32

    w = w * per_expert_scale[idx_addr]  # [T,K]

    # The K rows for one token all read the SAME token embedding, so broadcast
    # x_expert over the K axis instead of gathering with a flattened index. The
    # expert-weight index_selects consume the B5-B8 idx_addr in [T,K,32] form.
    with spyre_hint(tiles={"row": tile}):
        gathered = x_expert[:, None, :].expand(T, K, H)  # [T,K,H]
        W_g = gate_dev[idx_addr]  # [T,K,H,M] on-device index_select
        W_u = up_dev[idx_addr]  # [T,K,H,M]
        W_dn = down_dev[idx_addr]  # [T,K,M,H]
        g = torch.matmul(gathered.unsqueeze(-2), W_g)  # [T,K,1,M] batched
        u = torch.matmul(gathered.unsqueeze(-2), W_u)  # [T,K,1,M]
        act = F.gelu(g, approximate="tanh") * u  # [T,K,1,M]
        row_out = torch.matmul(act, W_dn).squeeze(-2)  # [T,K,H]
        row_out = row_out * w[..., None]  # [T,K,1] broadcast
    # token_of_row[t,k] = t (each of a token's K rows scatters back to token t);
    # kept in [T,K] here, flattened host-side for the eager index_add combine.
    token_of_row = token_ids[:, None].expand(T, K)  # [T,K]
    return row_out, token_of_row


def _moe_ffn_loop(x_router, x_expert, router, compiled_loop, gate_dev, up_dev,
                  down_dev, K, tile, eps):
    """Loop-on-topk MoE FFN orchestrator (spec Approach A).

    Unpacks the router's tensors, calls the compiled loop region (router +
    gather + on-device expert-weight index_select + bmms, row-tiled), then does
    the host index_add scatter-combine (scatter does not lower on device). The
    expert stacks are DEVICE-resident here (unlike _moe_ffn_split's host-
    resident stacks) -- the whole point of Approach A is the on-device select.

    Router surface (preflight-corrected): router.norm has NO .weight; the
    region applies a scale-free RMSNorm (eps INSIDE sqrt) then the [H]
    router.scale vector and the router.scalar_root_size float. Pass
    eps=config.rms_norm_eps.

    Returns the combined [T,H] MoE output on x_expert's device.
    """
    T, H = x_expert.shape
    token_ids = torch.arange(T, device=x_expert.device, dtype=torch.int32)
    # The router's proj.weight / scale / per_expert_scale are moved onto the
    # device in prepare_for_spyre (Approach-A flag branch) so they are already
    # device-resident here -- pass them straight through as region inputs.
    row_out, token_of_row = compiled_loop(
        x_router,
        x_expert,
        router.proj.weight,
        router.scale,
        router.scalar_root_size,
        router.per_expert_scale,
        gate_dev,
        up_dev,
        down_dev,
        token_ids,
        K,
        tile,
        eps,
    )
    # The region returns [T,K,H] / [T,K] (topk shape kept on device). Flatten to
    # [N,H] / [N] HOST-side for the eager index_add scatter-combine -- this is
    # host glue, not the device/topk-consumption path the "no reshape" rule
    # governs, and index_add does not lower on device anyway.
    row_out = row_out.cpu().float().reshape(-1, H)  # [N,H]
    token_of_row = token_of_row.cpu().long().reshape(-1)  # [N]
    out = torch.zeros(T, H, dtype=torch.float32)
    out = out.index_add(0, token_of_row, row_out)
    return out.to(dtype=x_expert.dtype, device=x_expert.device)


# ---------------------------------------------------------------------------
# ALL-DEVICE chunked FFN (spec Approach A, "nothing but glue on host").
# Gate-A5-proven (gateA5_chunked_ondevice.py, mean_rel=0.028). See the
# _MOE_CHUNKED_ONDEVICE flag docstring for the two backend workarounds this
# encodes (topk-pad + mask-reduce/accumulate instead of gather/scatter).
# ---------------------------------------------------------------------------


def _moe_route_padded(x_router, router_proj_w, router_scale,
                      router_scalar_root_size, per_expert_scale, K, eps,
                      pad_w, pad_neg):
    """All-device router: full dense [T,E] routing-weight (topk-pad fix).

    Router surface matches _compiled_moe_loop_region (preflight-corrected): the
    stock ``router.norm`` is a SCALE-FREE Gemma4RMSNorm (no ``.weight``; eps
    INSIDE the sqrt) applied to the RAW residual ``x_router``, then the [H]
    ``router_scale`` vector and the ``router_scalar_root_size`` float, then the
    router projection.

    topk over a pow2 stick-multiple width (E=128) aborts on-card
    (``Incorrect chunk size``), so the logits are padded to ``pad_w`` (non-pow2)
    with ``pad_neg`` (-inf) BEFORE topk; the pad columns never win top-K. Only
    the kth topk VALUE is used (``wv[..., -1:]``) as a threshold, applied over
    the ORIGINAL [T,E] probs via ``torch.ops.spyre.index_mask(probs, kth)`` --
    the fp16 index is never materialized (no ``customops.py`` fp16->int32 CPU
    fallback). The result is a DENSE [T,E] routing weight: zero for non-selected
    experts, ``renorm*per_expert_scale`` for the top-K. Downstream expert chunks
    turn per-expert selection into arithmetic (``(w*onehot[e]).sum``), so no
    index/gather is needed.

    Args:
        x_router: RAW flattened residual [T,H] (router's own norm applied here).
        router_proj_w: Router projection weight [E,H].
        router_scale: Post-norm gain vector [H] (router.scale).
        router_scalar_root_size: hidden_size ** -0.5 (Python float).
        per_expert_scale: Per-expert scale [E] (router.per_expert_scale).
        K: Top-K experts per token.
        eps: config.rms_norm_eps (inside the sqrt).
        pad_w: Padded topk-input width (>E, non-pow2; e.g. 160 for E=128).
        pad_neg: Pad-column fill (~ -inf fp16; e.g. -30000.0).

    Returns:
        w: Dense routing weight [T,E] (renormed top-K * per_expert_scale, zero
           elsewhere), on x_router's device.
    """
    T, _ = x_router.shape
    E = router_proj_w.shape[0]
    # scale-free RMSNorm (eps inside sqrt) on the raw residual, then [H] gain.
    var = x_router.pow(2).mean(-1, keepdim=True)
    normed = x_router * torch.rsqrt(var + eps)
    normed = normed * router_scale * router_scalar_root_size
    probs = torch.softmax(F.linear(normed, router_proj_w), dim=-1)  # [T,E]
    # topk-pad: widen to a non-pow2 width so the backend tiling stays
    # stick-aligned; pad cols are -inf -> never selected.
    pad = torch.full((T, pad_w - E), pad_neg, dtype=probs.dtype,
                     device=probs.device)
    padded = torch.cat([probs, pad], dim=-1)  # [T,pad_w]
    wv, _ = torch.topk(padded, K, dim=-1)  # [T,K]; idx<E, never materialized
    kth = wv[..., -1:]  # [T,1] kth-largest VALUE threshold
    # torch.where threshold: keep probs where probs >= kth (the selected top-K
    # experts), zero the rest. Equivalent to the torch.ops.spyre.index_mask(probs,
    # kth) device op, which is not yet registered in torch-spyre -- switch to
    # index_mask once it lands; until then this form traces on-device fine.
    mask = torch.where(probs >= kth, probs, torch.zeros_like(probs))  # [T,E]
    w = mask / mask.sum(-1, keepdim=True)  # renorm top-K to sum 1
    return w * per_expert_scale  # [T,E] dense routing weight


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
    """Produce one broadcast-ready stick per token/expert route scalar."""
    token_major = _moe_route_padded(
        x_router,
        router_proj_w,
        router_scale,
        router_scalar_root_size,
        per_expert_scale,
        K,
        eps,
        pad_w,
        pad_neg,
    )
    # Materialize the logical expansion first in the router's natural layout,
    # then use an ordinary BMM to put the broadcast lane on the physical stick.
    # The semantic values are unchanged because route_identity is I64.
    expanded = torch.relu(token_major.unsqueeze(-1).expand(-1, -1, 64))
    return expanded @ route_identity


def _moe_expert_persistent(x_expert, routing_weight, gate, up, down, K):
    """Run the dense expert body as one hinted, ordinary PyTorch program.

    The dims are declared and each operand is named, then the matmuls run under
    a coarse-tile ``spyre_hint(num_tiles_per_dim={"E": E})`` (one tile per
    expert): coarse-tiling emits one expert body inside one counted device loop
    -- stage ``[T,H]`` once, advance gate/up/down/route per expert, keep the
    running sum in the LX accumulator, drain once. No custom op.

    Shapes: ``x_expert`` [T,H]; ``gate``/``up`` logical [E,H,M]; ``down`` logical
    [E,M,H]; ``routing_weight`` [T,E,1]. gate/up/down arrive already in the
    shared device layout (E outermost, free dim on stick) from prepare_for_spyre.
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
    # routing_weight arrives as routing_sticks[..., :1]: a size-1 NARROWING slice
    # over a physically 64-wide broadcast stick, so its logical [T,E,1] view still
    # carries the 64-lane strides. Fed into the coarse-tile read-copy that way, the
    # outermost-dim size derivation (total_elems // stride) under-counts T by the
    # dropped 64 factor and rejects the T=512 tile. Materialize a genuinely dense,
    # owned [E,T,1] buffer so the derivation recovers E / T correctly. .contiguous()
    # alone on the permuted slice can be elided upstream; .clone() forces an owned
    # contiguous copy.
    route = routing_weight.permute(1, 0, 2).contiguous().clone()  # [T,E,1]->[E,T,ONE]
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


def _moe_ffn_persistent(
    x_router,
    x_expert,
    router,
    compiled_route,
    compiled_persistent,
    gate,
    up,
    down,
    route_identity,
    K,
    eps,
):
    """Run PR293 routing once, then one persistent all-expert value path."""
    from torch_spyre._inductor import config as spyre_config

    routing_sticks = compiled_route(
        x_router,
        router.proj.weight,
        router.scale,
        router.scalar_root_size,
        router.per_expert_scale,
        K,
        eps,
        _MOE_PADW,
        _MOE_PAD_NEG,
        route_identity,
    )
    # The semantic op consumes logical [T,E,1]. Lane zero is a zero-copy view
    # over one full broadcast stick per token/expert scalar.
    routing_weight = routing_sticks[..., :1]
    with spyre_config.patch(
        {
            "sencores": 32,
            "lx_planning": True,
            "allow_all_ops_in_lx_planning": True,
        }
    ):
        return compiled_persistent(
            x_expert,
            routing_weight,
            gate,
            up,
            down,
            K,
        )


class Gemma4MoEBlock(nn.Module):
    """Registered Gemma 4 **MoE** decoder block used by the Spyre adapter.

    Mirrors the dense ``hf_gemma4.Gemma4Block`` (same class shape, same 7-arg
    ``cache_index`` call signature, same ``layer_scalar`` buffer idiom) but its
    ``forward`` reproduces the ``enable_moe_block=True`` branch of the stock
    ``Gemma4TextDecoderLayer.forward``: a dense MLP **in parallel** with a
    top-K MoE FFN, combined ``post_feedforward_layernorm(h_dense + h_moe)``.

    Attention is the upstream ``Gemma4Attention`` module composed VERBATIM
    (exactly as ``Gemma4Block`` does), so KV handling (in-place
    ``kv_cache_update`` indirect scatter, #330) is the same code path the dense
    adapter is tested on. The block is NOT a single ``torch.compile`` — the MoE
    routing is host-side in the default/split mode — so it composes several
    per-region ``torch.compile`` handles built once in ``__init__``:

        # attention half -> post-attn-norm sandwich -> residual add
        residual = h
        h_dense = post_feedforward_layernorm_1(mlp(pre_feedforward_layernorm(h)))
        flat    = residual.reshape(-1, H)          # RAW residual
        # router reads flat (its own scale-free norm inside the FFN region);
        # experts read a SEPARATE pre_ff_ln_2 norm of the same flat.
        h_moe   = post_feedforward_layernorm_2(<ffn mode>(flat, pre_ff_ln_2(flat), ...))
        h       = post_feedforward_layernorm(h_dense + h_moe)
        h       = residual + h
        h       = h * layer_scalar

    The pre/post ``_2`` norms run on-device on the flattened ``[T,H]`` tensor;
    the router's internal norm runs on the raw ``flat`` (NOT the pre_ff_ln_2
    output), so the host only ever sees the small routing tensors plus the
    ``[T*K,H]`` gathered / expert-out buffers. Mode-specific expert weights
    (set by ``prepare_for_spyre``) are read fresh from ``self`` at call time,
    the same call-time-read rule the dense block uses for ``layer_scalar``.
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
        # Captured knobs. K + eps are per-layer scalars; the mode-specific
        # expert stacks are read fresh off ``self`` at call time (populated by
        # prepare_for_spyre), NOT captured here.
        self._moe_k = layer._spyre_moe_k
        # Gemma4RMSNorm exposes ``.eps`` (== config.rms_norm_eps).
        self._moe_rms_eps = self.pre_feedforward_layernorm_2.eps

        # Compiled device regions (built once per block). The dense MLP is
        # compiled as its own region so the dense branch lowers; the router,
        # gather, expert GEMM, loop, and per-chunk regions are each compiled
        # for whichever FFN mode is active.
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

        # Sparse branch: the router reads the RAW flattened residual (its own
        # scale-free norm is applied inside the FFN region), while the experts
        # consume a SEPARATE pre_ff_ln_2 normalization of that same residual
        # (stock modeling_gemma4.py). Thread them as two tensors so the router
        # input is not double-normalized. The per-layer expert weights
        # (mode-specific layout, set by prepare_for_spyre) are read fresh off
        # ``self`` at call time (like the dense block's layer_scalar).
        flat = residual.reshape(-1, hidden)  # [T,H] RAW -> router
        x_moe = self.pre_feedforward_layernorm_2(flat)  # [T,H] normed -> experts
        # Phase dispatch: prefill (seq_len > 1) runs the dense all-experts
        # persistent path; decode (seq_len == 1) runs the loop-on-topk gather
        # path. Both compiled regions and both expert-weight sets are built once
        # (prepare_for_spyre / __init__); the choice is per-forward by shape, not
        # an import-time flag.
        if seq_len > 1:
            moe_out = _moe_ffn_persistent(
                flat,
                x_moe,
                self.router,
                self._compiled_persistent_route,
                self._compiled_persistent,
                self._spyre_gate,
                self._spyre_up,
                self._spyre_down,
                self._spyre_persistent_route_identity,
                self._moe_k,
                self._moe_rms_eps,
            )  # [T,H]
        else:
            moe_out = _moe_ffn_loop(
                flat,
                x_moe,
                self.router,
                self._compiled_loop,
                self._spyre_gate,
                self._spyre_up,
                self._spyre_down,
                self._moe_k,
                _MOE_TILE,
                self._moe_rms_eps,
            )  # [T,H]
        moe_out = moe_out.reshape(bsz, seq_len, hidden)
        h_moe = self.post_feedforward_layernorm_2(moe_out)

        # Combine dense + MoE, final sandwich norm, residual, per-layer scalar.
        h = self.post_feedforward_layernorm(h_dense + h_moe)
        h = residual + h
        return h * layer_scalar, key_cache, value_cache


def prepare_for_spyre(model):
    """Apply Spyre adaptations to a Gemma 4 **MoE** causal-LM model in-place.

    Reuses the shared attention-side prep (``_setup_gemma4_text_decoder``:
    RMSNorm patch, per-type RoPE, KV shapes, ``pad_lm_head``) and adds the MoE
    layout / bring-up steps:

      * assert ``enable_moe_block=True`` and use the checkpoint's real
        ``top_k_experts`` (asserting only the Spyre topk ceiling ``<= 128``);
      * de-fuse ``gate_up`` into ``gate``/``up`` and lay each of the three
        packed expert weights out in ONE SHARED device layout, expert-dim-
        outermost with the free dim on the stick (``gate``/``up``
        ``[E,2M,H]``-derived -> ``[E,H,M]``, ``down`` ``[E,H,M]`` ->
        ``[E,M,H]``; shape rule 2 + spec §3.5), via
        ``dma_moe_expert_weight_to_spyre``. This single set is simultaneously
        the gather-source layout (expert dim outermost, consumed by the decode
        loop-on-topk index_select) and the matmul weight-operand layout
        (sticked on the free dim, consumed by the prefill persistent
        hint-body) -- both FFN paths read the SAME device-resident
        ``layer._spyre_gate`` / ``_spyre_up`` / ``_spyre_down``; router weights
        are also device-resident. Only ONE weight set is ever materialized
        (no dual/mode-dependent copies), so it fits the card; the original
        ``gate_up_proj`` / ``down_proj`` parameters are deleted;
      * build ``model._spyre_compiled_blocks`` from the MoE block factory.
    """
    backbone = _gemma4_backbone(model)
    cfg = text_config(model.config)

    assert getattr(cfg, "enable_moe_block", False), (
        "hf_gemma4_moe requires an MoE checkpoint (enable_moe_block=True); "
        "use hf_gemma4 for the dense variants."
    )
    # Use the checkpoint's real top_k_experts (no bring-up pin). The Spyre topk
    # reduction serves up to k=128 (torch-spyre #3782), so only assert that
    # ceiling; the router's topk and the device value path both run at this K.
    moe_k = int(cfg.top_k_experts)
    assert 1 <= moe_k <= _MOE_MAX_K, (
        f"top_k_experts ({moe_k}) must be in [1, {_MOE_MAX_K}]; the Spyre topk "
        f"reduction caps k at {_MOE_MAX_K}."
    )
    # The expert SwiGLU hardcodes gelu(approximate="tanh") to match this
    # checkpoint's hidden_activation. Guard so a variant with a different
    # activation fails loudly instead of computing silently-wrong output.
    act_fn = getattr(cfg, "hidden_activation", None)
    assert act_fn == "gelu_pytorch_tanh", (
        "hf_gemma4_moe expert SwiGLU is fixed to gelu(approximate='tanh'); "
        f"config hidden_activation={act_fn!r} is unsupported."
    )

    # The router pads its logits to a non-pow2 width before topk (topk-pad fix,
    # shared by the prefill persistent router and the decode loop router).
    E = cfg.num_experts
    assert _MOE_PADW > E and (_MOE_PADW & (_MOE_PADW - 1)) != 0, (
        f"_MOE_PADW ({_MOE_PADW}) must exceed num_experts ({E}) and be "
        "non-power-of-two (topk-pad fix)."
    )

    num_q_heads_per_layer, kv_shapes, is_kv_eq_v_per_layer = (
        _setup_gemma4_text_decoder(model, allow_moe=True)
    )

    # Lay out expert weights: expert-dim-outermost (already, [E,...]) and
    # PRE-TRANSPOSED so the compiled expert region needs no in-kernel transpose
    # of a large weight (shape rule 2).
    #
    # The pre-transposed expert stacks are kept on the HOST (CPU), NOT moved to
    # the device, for two reasons that both surfaced on-card at 26B:
    #
    #   1. The per-row weight select ``gate_up_t[row_expert]`` is an eager
    #      ``aten::index.Tensor`` — unsupported on the spyre backend
    #      (``NotImplementedError``). Only plain ``gather``/``bmm`` lower
    #      (spec §2.1); fancy indexing must run on CPU. Gate 2 selects on host
    #      for exactly this reason, then moves the ``[N,·]`` slice to the device.
    #   2. Even if the select lowered, keeping all 128 experts × 30 layers
    #      resident is ~46 GB fp16; the [N,H,2M]/[N,M,H] per-row gathers plus
    #      the rest of the model exhaust the card (FlexAllocator OOM).
    #
    # They are stored as PLAIN ATTRIBUTES (not ``register_buffer``) so
    # ``_move_to_spyre_with_layout``'s ``named_buffers()`` sweep never moves them
    # to the device — they stay on CPU. ``_moe_ffn_split`` selects the per-row
    # weights here on the host and moves only the small ``[N,·]`` slices to the
    # device for the compiled expert GEMM. The ORIGINAL ``gate_up_proj`` /
    # ``down_proj`` parameters are deleted (the stock ``Gemma4TextExperts.forward``
    # never runs; the split FFN uses only the transposed CPU stacks), so the
    # experts are not paid for twice on the host either.
    # One pass per layer: build the MoE block (composing upstream Gemma4Attention
    # verbatim), attach its mode-specific expert weights, register it back into
    # ``backbone.layers[i]`` (so ``_run_blocks_over_embeds`` can read
    # ``layer_scalar`` off it, exactly as ``prepare_gemma4_blocks`` does for the
    # dense path), and compile it. The expert-weight layout below targets the
    # BLOCK instance (``block._spyre_*``), read fresh by ``Gemma4MoEBlock.forward``
    # — the same call-time-read rule the dense block uses for ``layer_scalar``.
    compiled_blocks = []
    for i, layer in enumerate(list(backbone.layers)):
        # Stash the checkpoint's K on the layer so Gemma4MoEBlock.__init__ can
        # capture it (validated <= _MOE_MAX_K above).
        layer._spyre_moe_k = moe_k
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
        # The router is shared between ``layer`` and ``block`` (composed as
        # ``self.router = layer.router``), so router-weight moves below apply to
        # the block's router too.
        router = block.router
        # ONE shared expert-weight set, read by BOTH the prefill persistent
        # hint-body and the decode loop-on-topk gather. De-fuse gate_up into
        # separate gate/up, then lay each out E-outermost with the free dim on
        # the stick (dma_moe_expert_weight_to_spyre): simultaneously the
        # gather-source layout (expert dim outermost) and the matmul
        # weight-operand layout (sticked on free dim). ~42.5 GiB / 30 layers
        # (one set, not two) -- fits the card.
        M = gate_up_t.shape[2] // 2
        gate_l = gate_up_t[:, :, :M].contiguous()  # [E,H,M]
        up_l = gate_up_t[:, :, M:].contiguous()  # [E,H,M]
        block._spyre_gate = dma_moe_expert_weight_to_spyre(gate_l)  # [E,H,M]
        block._spyre_up = dma_moe_expert_weight_to_spyre(up_l)  # [E,H,M]
        block._spyre_down = dma_moe_expert_weight_to_spyre(down_t)  # [E,M,H]
        # Fall back to a plain device move if the free dim doesn't tile into
        # sticks (should not happen for gemma-4: M=704, H=2816 both divide 64).
        if block._spyre_gate is None:
            block._spyre_gate = gate_l.to("spyre")
        if block._spyre_up is None:
            block._spyre_up = up_l.to("spyre")
        if block._spyre_down is None:
            block._spyre_down = down_t.to("spyre")

        # Persistent routing identity (unrelated to expert-weight layout; the
        # packed router expands its one-hot rows against this eye).
        block._spyre_persistent_route_identity = torch.eye(
            64, dtype=gate_up_t.dtype
        ).to("spyre")

        # The router runs on-device in BOTH paths (scale-free norm + [H] scale +
        # proj + padded topk + per_expert_scale gather), so its weights must be
        # device-resident. Reassign the Parameter object (a cross-backend
        # ``param.data = ...`` set_data raises on the type change). Done once here
        # -- both methods share this router. router.scalar_root_size is a Python
        # float (no move).
        router.proj.weight = torch.nn.Parameter(
            router.proj.weight.data.to("spyre"), requires_grad=False
        )
        router.scale = torch.nn.Parameter(
            router.scale.data.to("spyre"), requires_grad=False
        )
        router.per_expert_scale = torch.nn.Parameter(
            router.per_expert_scale.data.to("spyre"), requires_grad=False
        )

        # Register the block back into the backbone (so _run_blocks_over_embeds
        # reads layer_scalar off it), then append the EAGER block to the
        # compiled-blocks list. Unlike the dense Gemma4Block (a pure-device
        # forward that prepare_gemma4_blocks wraps in torch.compile), the MoE
        # block's forward is a HYBRID: eager host orchestration (topk/argsort/
        # index_add/chunk-loop glue, all outside any spyre graph per the
        # all-device Global Constraint) around several inner torch.compile'd
        # device regions built in __init__. Wrapping the whole block in
        # torch.compile would try to trace that host work into one spyre graph.
        # So _run_blocks_over_embeds invokes the eager block, which dispatches to
        # its inner compiled regions.
        backbone.layers[i] = block
        compiled_blocks.append(block)

    model._spyre_compiled_blocks = compiled_blocks


# ``_run_backbone_forward`` and ``_run_forward`` are NOT defined here — they are
# re-exported from ``hf_gemma4`` (see the import block at the top of this file).
# Since #350 both are block-AGNOSTIC: they drive ``model._spyre_compiled_blocks``
# and read ``layer_scalar`` off each registered block, so the MoE blocks slot in
# with no MoE-specific forward. This removes the last carrier of the pre-#330
# ``is_filling/token_index/cache_position`` triple and guarantees the MoE forward
# tracks any future dense-forward change.
