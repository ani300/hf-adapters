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

The MoE routing ops (``topk``, ``argsort``, 1-D index arithmetic,
``index_add``) do not lower on the current torch-spyre backend, so the FFN is
**device/host split** (spec §2.1, verified in ``repros/gemma4_moe/
gate2_route_permute.py``):

  device (torch.compile, spyre):  router projection ; token gather
                                  ``x[token_of_row]`` ; expert grouped GEMM
                                  (bmm + gelu_tanh SwiGLU + bmm)
  host   (eager CPU):             softmax / topk / renorm / per_expert_scale
                                  ; argsort + ``token_of_row`` arithmetic
                                  ; weighted ``index_add`` combine

Two load-bearing device-shape rules (verified on-card, gate 2):

1. The row-batched expert tensors stay **3D ``[N,1,·]``** through the whole
   expert FFN — the ``squeeze(1)→chunk→unsqueeze(1)`` 2D round-trip breaks
   Spyre layout propagation ("Incompatible host_size and dim_order"). Squeeze
   only at the very end.
2. Expert weights are supplied **pre-transposed** (``gate_up`` as ``[E,H,2M]``,
   ``down`` as ``[E,M,H]``) so the compiled region has no in-kernel
   ``.transpose`` of a large weight (which forces a giant-offset restickify:
   ``L3_ADDEARIMM Immediate value out of boundary``). ``prepare_for_spyre``
   lays the experts out pre-transposed once.

``K`` is pinned to 4 for bring-up; ``prepare_for_spyre`` coerces/asserts
``config.top_k_experts == 4``.
"""

import torch
import torch.nn.functional as F

from hf_adapters.hf_common import text_config
from hf_adapters.hf_gemma4 import (
    _gemma4_attention,
    _gemma4_backbone,
    _run_blocks_over_embeds,
    _setup_gemma4_text_decoder,
)

# Top-K pinned to 4 for MoE bring-up (Global Constraints). The device grouped
# GEMM and the host route/permute path are validated at this K; a different K
# is a later-task concern.
_MOE_BRINGUP_K = 4


def _moe_route(x, W_router, per_expert_scale, K):
    """Route tokens to top-K experts with softmax and per-expert scaling.

    Args:
        x: Token embeddings [T, H]
        W_router: Expert router weights [E, H]
        per_expert_scale: Expert scaling factors [E]
        K: Number of top experts per token

    Returns:
        w: Router weights after softmax, top-K selection, renormalization,
           and per-expert scaling [T, K]
        idx: Top-K expert indices [T, K]
    """
    logits = F.linear(x, W_router)  # [T,E]
    probs = torch.softmax(logits, dim=-1)
    w, idx = torch.topk(probs, K, dim=-1)  # [T,K],[T,K]
    w = w / w.sum(-1, keepdim=True)
    w = w * per_expert_scale[idx]
    return w, idx


def _moe_permute(x, idx, K):
    """Sort token-expert pairs by expert and return gather/sort information.

    Args:
        x: Token embeddings [T, H]
        idx: Top-K expert indices [T, K]
        K: Number of experts per token (used for reconstruction)

    Returns:
        gathered: Token embeddings sorted by expert assignment [T*K, H]
        token_of_row: Token indices for each row in gathered [T*K]
        row_expert: Expert ID for each row in gathered [T*K]
        sort_perm: Permutation that sorts (token, expert) pairs by expert
                   [T*K]
    """
    flat_expert = idx.reshape(-1)  # [T*K]
    sort_perm = torch.argsort(flat_expert)  # [T*K]
    row_expert = flat_expert[sort_perm]  # [T*K] expert id per sorted row
    token_of_row = (
        torch.arange(idx.shape[0] * K, device=x.device) // K
    )[sort_perm]
    gathered = x[token_of_row]  # [T*K,H]
    return gathered, token_of_row, row_expert, sort_perm


def _grouped_gemm(gathered, Wstack, row_expert):
    """Option 4A: gather per-row weight, row-batched matmul.

    Args:
        gathered: Token embeddings sorted by expert [N, in]
        Wstack: Expert weight matrices [E, out, in]
        row_expert: Expert ID for each row in gathered [N]

    Returns:
        out: Result of gather per-row weight @ gathered [N, out]
    """
    W_row = Wstack[row_expert]  # index_select on expert dim [N,out,in]
    out = torch.bmm(
        gathered.unsqueeze(1), W_row.transpose(1, 2)
    )  # [N,1,out]
    return out.squeeze(1)  # [N,out]


def _moe_ffn(x, W_router, gate_up_proj, down_proj, per_expert_scale, K):
    """MoE FFN forward: route, permute, grouped gate_up, gelu_tanh SwiGLU,
    grouped down, weight by w, scatter_add combine.

    Args:
        x: Token embeddings [T, H]
        W_router: Expert router weights [E, H]
        gate_up_proj: Gate-up projection per expert [E, 2*M, H]
        down_proj: Down projection per expert [E, H, M]
        per_expert_scale: Expert scaling factors [E]
        K: Number of top experts per token

    Returns:
        out: MoE FFN output [T, H]
    """
    T, H = x.shape
    w, idx = _moe_route(x, W_router, per_expert_scale, K)
    (
        gathered,
        token_of_row,
        row_expert,
        sort_perm,
    ) = _moe_permute(x, idx, K)
    gate_up = _grouped_gemm(
        gathered, gate_up_proj, row_expert
    )  # [N,2M]
    g, u = gate_up.chunk(2, dim=-1)
    act = F.gelu(g, approximate="tanh") * u  # [N,M]
    expert_out = _grouped_gemm(
        act, down_proj, row_expert
    )  # [N,H]
    expert_out = expert_out * w.reshape(-1)[sort_perm].unsqueeze(-1)
    out = torch.zeros(T, H, dtype=x.dtype, device=x.device)
    out = out.index_add(0, token_of_row, expert_out)  # scatter_add combine
    return out


# ---------------------------------------------------------------------------
# Device (compiled, spyre) FFN region + host orchestrator + decoder block.
# ---------------------------------------------------------------------------


def _compiled_moe_device_region(gathered3d, gate_up_row_t, down_row_t):
    """The device portion of the expert FFN — the two grouped GEMMs + SwiGLU.

    This is the exact shape flow verified on-card in gate 2
    (``repros/gemma4_moe/gate2_route_permute.py``) and MUST stay byte-for-byte
    shape-identical to it. Everything is 3D ``[N,1,·]`` (shape rule 1) and the
    weights are pre-transposed (shape rule 2):

        gathered3d:    [N,1,H]   (per-row gathered token embedding)
        gate_up_row_t: [N,H,2M]  (per-row gate_up weight, PRE-transposed)
        down_row_t:    [N,M,H]   (per-row down weight, PRE-transposed)

    Do NOT squeeze between the two ``bmm``s — the 2D round-trip breaks Spyre
    layout propagation. Squeeze only on the final return.
    """
    gu = torch.bmm(gathered3d, gate_up_row_t)  # [N,1,2M]
    g, u = gu.chunk(2, dim=-1)  # [N,1,M] each
    act = F.gelu(g, approximate="tanh") * u  # [N,1,M]
    return torch.bmm(act, down_row_t).squeeze(1)  # [N,H]


def _compiled_device_gather(x, token_of_row):
    """On-device token gather ``x[token_of_row]`` (indirect-access op).

    Kept as its own compiled region (gate-2 template): a single-op ``[N,H]``
    gather whose row (indexed) dim is outermost by construction, so it needs no
    restickify. The routing tensor ``token_of_row`` is computed host-side and
    moved to the device for this call.
    """
    return x[token_of_row]  # [T*K, H]


def _moe_ffn_split(
    x_router,
    x_expert,
    router,
    compiled_gather,
    compiled_expert,
    gate_up_dev_t,
    down_dev_t,
    K,
):
    """Device/host-split MoE FFN (spec §2.1; gate-2 host-orchestration template).

    The router and the experts consume **two independent normalizations of the
    same raw flattened residual** (stock ``modeling_gemma4.py:1432-1435``):

      * ``x_router`` ``[T,H]`` — the **raw** flattened residual. The router's
        own internal ``self.norm`` (scale-free RMSNorm) is applied to THIS
        tensor. Do NOT pass a ``pre_feedforward_layernorm_2``-normed tensor
        here: that would double-normalize the router input
        (``router.norm ∘ pre_ff_ln_2``) and select the wrong experts.
      * ``x_expert`` ``[T,H]`` — the ``pre_feedforward_layernorm_2``-normed
        residual. The token gather (expert FFN input) reads THIS tensor.

    Both are **on the device**. Router weights and the pre-transposed expert
    weights (``gate_up_dev_t`` ``[E,H,2M]``, ``down_dev_t`` ``[E,M,H]``) stay
    resident on-device across calls; only the small routing tensors and the
    ``[T*K,H]`` gathered / expert-out buffers cross the host boundary.

    Ordering (matches gate 2):
      device: router projection (on x_router) → cpu
      host:   softmax / topk(K) / renorm / per_expert_scale ; argsort +
              token_of_row arithmetic
      device: gather token rows (from x_expert) ; expert grouped GEMM
              (3D, pre-transposed)
      host:   weighted index_add combine

    ``topk`` / ``argsort`` / index arithmetic / ``index_add`` are eager host
    CPU (never inside a spyre ``torch.compile``, per Global Constraints).
    Returns the combined ``[T,H]`` output **on the device**.
    """
    T, H = x_expert.shape

    # --- device: router projection (router.norm + router.scale + proj) -> host
    # The router's norm/scale/proj are all device-lowerable (rmsnorm + mul +
    # linear); only the softmax/topk that follow must be host. Reproduce the
    # stock Gemma4TextRouter pre-softmax math on the RAW residual (x_router),
    # then bring logits to CPU. router.norm is applied to x_router here, NOT to
    # the pre_ff_ln_2-normed x_expert (avoids the double-normalization bug).
    normed = router.norm(x_router)
    normed = normed * router.scale * router.scalar_root_size
    logits = router.proj(normed)  # [T, E]
    logits = logits.cpu().float()

    # --- host: softmax / topk / renorm / per_expert_scale (unsupported on dev)
    probs = torch.softmax(logits, dim=-1)
    w, idx = torch.topk(probs, K, dim=-1)  # [T,K], [T,K]
    w = w / w.sum(-1, keepdim=True)
    per_expert_scale = router.per_expert_scale.detach().cpu().float()
    w = w * per_expert_scale[idx]

    # --- host: argsort + token_of_row / row_expert index arithmetic
    flat_expert = idx.reshape(-1)  # [T*K]
    sort_perm = torch.argsort(flat_expert)
    row_expert = flat_expert[sort_perm]  # [T*K] expert id per sorted row
    token_of_row = (torch.arange(T * K) // K)[sort_perm].to(torch.int32)

    # --- device: gather token rows from the EXPERT input ([N,H])
    gathered = compiled_gather(x_expert, token_of_row.to(x_expert.device))

    # --- host: select + PRE-TRANSPOSE per-row expert weights.
    # gate_up_dev_t / down_dev_t are already expert-outermost, pre-transposed
    # ([E,H,2M] / [E,M,H]); indexing on the expert dim keeps that layout, so
    # the compiled expert region sees no in-kernel transpose (shape rule 2).
    # The per-row index_select happens on the device tensors directly.
    row_expert_dev = row_expert.to(gate_up_dev_t.device)
    gate_up_row_t = gate_up_dev_t[row_expert_dev]  # [N,H,2M]
    down_row_t = down_dev_t[row_expert_dev]  # [N,M,H]

    # --- device: expert grouped GEMM (rows as [N,1,H], stay 3D)
    expert_out = compiled_expert(
        gathered.unsqueeze(1), gate_up_row_t, down_row_t
    )  # [N,H]
    expert_out = expert_out.cpu().float()

    # --- host: weighted index_add combine
    row_w = w.reshape(-1)[sort_perm].unsqueeze(-1)  # [N,1]
    out = torch.zeros(T, H, dtype=torch.float32)
    out = out.index_add(0, token_of_row.long(), expert_out * row_w)
    return out.to(dtype=x_expert.dtype, device=x_expert.device)


def _make_moe_block(layer, num_q_heads, num_kv_heads, head_dim, is_kv_eq_v):
    """Build one Gemma 4 **MoE** decoder-layer block callable.

    NOT a single ``torch.compile`` of the whole block: the FFN's routing is
    host-side (spec §2.1). The block runs the **compiled** attention (shared
    ``_gemma4_attention``, wrapped here) and the **compiled** dense MLP, and
    calls ``_moe_ffn_split`` for the sparse branch, combining per the stock
    ``Gemma4TextDecoderLayer.forward`` (``enable_moe_block=True``):

        # attention half -> residual add (inside _gemma4_attention)
        residual = h
        h_dense = post_feedforward_layernorm_1(mlp(pre_feedforward_layernorm(h)))
        flat    = residual.reshape(-1, H)          # RAW residual
        # router reads flat (its own scale-free norm inside _moe_ffn_split);
        # experts read a SEPARATE pre_ff_ln_2 norm of the same flat.
        h_moe   = post_feedforward_layernorm_2(
                      _moe_ffn_split(flat, pre_feedforward_layernorm_2(flat), ...))
        h       = post_feedforward_layernorm(h_dense + h_moe)
        h       = residual + h
        h       = h * layer_scalar

    The pre/post ``_2`` norms run on-device on the flattened ``[T,H]`` tensor
    before / after the split; the router's internal norm runs on the raw
    ``flat`` (NOT the pre_ff_ln_2 output), so the host only ever sees the small
    routing tensors plus the ``[T*K,H]`` gathered / expert-out buffers.

    Matches the dense block's call signature
    (``hidden_states, selected_freqs, attn_mask, key_cache, value_cache,
    is_filling, token_index, cache_position, layer_scalar``); the MoE weights
    are captured from ``layer`` / ``prepare_for_spyre``.
    """
    attn = layer.self_attn
    q_proj = attn.q_proj
    k_proj = attn.k_proj
    v_proj = attn.v_proj  # None when is_kv_eq_v
    o_proj = attn.o_proj
    q_norm = attn.q_norm
    k_norm = attn.k_norm
    v_norm = attn.v_norm
    scaling = attn.scaling  # 1.0 for Gemma 4

    input_ln = layer.input_layernorm
    post_attn_ln = layer.post_attention_layernorm
    pre_ff_ln = layer.pre_feedforward_layernorm
    post_ff_ln = layer.post_feedforward_layernorm
    mlp = layer.mlp

    # MoE-specific submodules / norms (stock Gemma4TextDecoderLayer names).
    router = layer.router
    post_ff_ln_1 = layer.post_feedforward_layernorm_1  # dense-branch post-norm
    pre_ff_ln_2 = layer.pre_feedforward_layernorm_2  # MoE pre-norm (on residual)
    post_ff_ln_2 = layer.post_feedforward_layernorm_2  # MoE post-norm
    # Pre-transposed, device-resident expert weights laid down by
    # prepare_for_spyre (shape rule 2 + expert-dim-outermost, spec §3.5).
    gate_up_dev_t = layer._spyre_gate_up_t  # [E,H,2M]
    down_dev_t = layer._spyre_down_t  # [E,M,H]
    K = layer._spyre_moe_k

    # Compiled device regions (shared per layer; router.proj/gather/expert). The
    # dense MLP is compiled as its own region too so the dense branch lowers.
    compiled_mlp = torch.compile(mlp, dynamic=False)
    compiled_gather = torch.compile(_compiled_device_gather, dynamic=False)
    compiled_expert = torch.compile(_compiled_moe_device_region, dynamic=False)
    compiled_attn = torch.compile(_gemma4_attention, dynamic=False)

    def block_forward(
        hidden_states,
        selected_freqs,
        attn_mask,
        key_cache,
        value_cache,
        is_filling,
        token_index,
        cache_position,
        layer_scalar,
    ):
        h, key_cache, value_cache = compiled_attn(
            hidden_states,
            input_ln=input_ln,
            post_attn_ln=post_attn_ln,
            q_proj=q_proj,
            k_proj=k_proj,
            v_proj=v_proj,
            o_proj=o_proj,
            q_norm=q_norm,
            k_norm=k_norm,
            v_norm=v_norm,
            scaling=scaling,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            is_kv_eq_v=is_kv_eq_v,
            selected_freqs=selected_freqs,
            attn_mask=attn_mask,
            key_cache=key_cache,
            value_cache=value_cache,
            is_filling=is_filling,
            token_index=token_index,
            cache_position=cache_position,
        )

        residual = h
        bsz, seq_len, hidden = h.shape

        # Dense branch: pre_ff_ln -> mlp -> post_ff_ln_1.
        h_dense = post_ff_ln_1(compiled_mlp(pre_ff_ln(residual)))

        # Sparse branch: the router reads the RAW flattened residual (its own
        # scale-free norm is applied inside _moe_ffn_split), while the experts
        # consume a SEPARATE pre_ff_ln_2 normalization of that same residual
        # (stock modeling_gemma4.py:1432-1435). Thread them as two tensors so
        # the router input is not double-normalized.
        flat = residual.reshape(-1, hidden)  # [T,H] RAW -> router
        x_moe = pre_ff_ln_2(flat)  # [T,H] normed -> experts
        moe_out = _moe_ffn_split(
            flat,
            x_moe,
            router,
            compiled_gather,
            compiled_expert,
            gate_up_dev_t,
            down_dev_t,
            K,
        )  # [T,H]
        moe_out = moe_out.reshape(bsz, seq_len, hidden)
        h_moe = post_ff_ln_2(moe_out)

        # Combine dense + MoE, final sandwich norm, residual, per-layer scalar.
        h = post_ff_ln(h_dense + h_moe)
        h = residual + h
        h = h * layer_scalar
        return h, key_cache, value_cache

    return block_forward


def prepare_for_spyre(model):
    """Apply Spyre adaptations to a Gemma 4 **MoE** causal-LM model in-place.

    Reuses the shared attention-side prep (``_setup_gemma4_text_decoder``:
    RMSNorm patch, per-type RoPE, KV shapes, ``pad_lm_head``) and adds the MoE
    layout / bring-up steps:

      * assert ``enable_moe_block=True`` and coerce/assert ``top_k_experts == 4``
        (K pinned for bring-up, Global Constraints);
      * lay each layer's packed expert weights **expert-dim-outermost and
        pre-transposed** (``gate_up`` ``[E,2M,H]`` -> ``[E,H,2M]``, ``down``
        ``[E,H,M]`` -> ``[E,M,H]``; shape rule 2 + spec §3.5), register them as
        buffers, and move the whole model (router + experts + scales) to
        ``spyre`` so they stay resident;
      * build ``model._spyre_compiled_blocks`` from the MoE block factory.
    """
    backbone = _gemma4_backbone(model)
    cfg = text_config(model.config)

    assert getattr(cfg, "enable_moe_block", False), (
        "hf_gemma4_moe requires an MoE checkpoint (enable_moe_block=True); "
        "use hf_gemma4 for the dense variants."
    )
    # K pinned to 4 for bring-up. Coerce the config then assert so the router's
    # host topk and the device grouped-GEMM path both run at the validated K.
    cfg.top_k_experts = _MOE_BRINGUP_K
    assert cfg.top_k_experts == _MOE_BRINGUP_K, (
        f"MoE bring-up pins top_k_experts to {_MOE_BRINGUP_K}; "
        f"got {cfg.top_k_experts}."
    )

    num_q_heads, kv_shapes, is_kv_eq_v_per_layer = _setup_gemma4_text_decoder(
        model, allow_moe=True
    )

    # Lay out expert weights: expert-dim-outermost (already, [E,...]) and
    # PRE-TRANSPOSED so the compiled expert region needs no in-kernel transpose
    # of a large weight (shape rule 2). Registered as buffers on the layer so
    # they move to spyre with model.to("spyre") and stay resident across calls.
    for layer in backbone.layers:
        experts = layer.experts
        # gate_up_proj: [E, 2M, H] -> [E, H, 2M]; down_proj: [E, H, M] -> [E, M, H]
        gate_up_t = experts.gate_up_proj.data.transpose(1, 2).contiguous()
        down_t = experts.down_proj.data.transpose(1, 2).contiguous()
        layer.register_buffer("_spyre_gate_up_t", gate_up_t, persistent=False)
        layer.register_buffer("_spyre_down_t", down_t, persistent=False)
        layer._spyre_moe_k = _MOE_BRINGUP_K

    model._spyre_compiled_blocks = [
        _make_moe_block(
            layer,
            num_q_heads,
            kv_shapes[i][0],
            kv_shapes[i][1],
            is_kv_eq_v_per_layer[i],
        )
        for i, layer in enumerate(backbone.layers)
    ]


def _run_backbone_forward(
    model,
    input_ids,
    position_ids,
    attn_mask,
    key_caches,
    value_caches,
    is_filling,
    token_index,
    cache_position,
):
    """Gemma 4 MoE backbone: scaled embedding, per-type RoPE + masks, blocks.

    Identical to the dense backbone except the compiled blocks are the MoE
    blocks (``prepare_for_spyre`` populated ``model._spyre_compiled_blocks``).
    Delegates the block loop + final norm to the shared
    ``_run_blocks_over_embeds`` machinery.
    """
    backbone = _gemma4_backbone(model)
    h = backbone.embed_tokens(input_ids)
    return _run_blocks_over_embeds(
        model,
        h,
        position_ids,
        attn_mask,
        key_caches,
        value_caches,
        is_filling,
        token_index,
        cache_position,
    )


def _run_forward(
    model,
    input_ids,
    position_ids,
    attn_mask,
    key_caches,
    value_caches,
    is_filling,
    token_index,
    cache_position,
):
    """Gemma 4 MoE causal-LM forward: backbone + LM head + logit softcap."""
    h = _run_backbone_forward(
        model,
        input_ids,
        position_ids,
        attn_mask,
        key_caches,
        value_caches,
        is_filling,
        token_index,
        cache_position,
    )

    logits = model.lm_head(h)

    cap = text_config(model.config).final_logit_softcapping
    if cap is not None:
        logits = logits / cap
        logits = torch.tanh(logits)
        logits = logits * cap
    return logits
