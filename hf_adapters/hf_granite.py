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
HuggingFace Transformers adapter for Granite 3.3 models on Spyre.

Usage::

    from hf_adapters import AutoSpyreModelForCausalLM
    from transformers import AutoTokenizer

    model = AutoSpyreModelForCausalLM.from_pretrained(
        "/path/to/granite-3.3-8b-instruct")
    tokenizer = AutoTokenizer.from_pretrained("/path/to/granite-3.3-8b-instruct")
    outputs = model.generate(tokenizer, ["Hello!"], max_new_tokens=32)
"""

import torch

from hf_adapters.hf_common import (
    get_backbone,
    pad_lm_head,
    patch_rmsnorm,
    prepare_rope_and_heads,
    prepare_standard_gqa_region_blocks,
    text_config,
)


def _run_backbone_forward(
    model,
    input_ids,
    selected_freqs,
    attn_mask,
    key_caches,
    value_caches,
    is_filling,
    token_index,
    cache_position,
):
    """Granite 3.3 backbone: embedding * multiplier, blocks, norm.

    Takes ``selected_freqs`` (already gathered on the host by ``model._spyre_rope``)
    rather than ``position_ids``. The RoPE gather is intrinsically host-side
    (``.item()``-driven cache extend + CPU fancy-index) and must NOT be traced
    into the whole-forward graph — mirrors foundation-model-stack's ``eager_spyre``
    split, where the compiled forward consumes a ready freqs tensor.
    """
    backbone = get_backbone(model)
    h = backbone.embed_tokens(input_ids)
    h = h * backbone.embedding_multiplier

    for i, compiled_block in enumerate(model._spyre_compiled_blocks):
        # Each region block (see nested_region_block) updates key_caches[i]/
        # value_caches[i] IN PLACE and returns only ``h`` — the region wrapper
        # deliberately drops the cache buffers, since the surrounding
        # whole-forward compile turns each block into an ``invoke_subgraph`` HOP
        # call that rejects a subgraph output aliasing a subgraph input.
        h = compiled_block(
            h,
            selected_freqs,
            attn_mask,
            key_caches[i],
            value_caches[i],
            is_filling,
            token_index,
            cache_position,
        )

    h = backbone.norm(h)
    return h


def _run_forward_freqs(
    model,
    input_ids,
    selected_freqs,
    attn_mask,
    key_caches,
    value_caches,
    is_filling,
    token_index,
    cache_position,
):
    """Granite 3.3 causal-LM forward: backbone + head / scaling.

    Consumes ``selected_freqs`` directly, so the entire body is Spyre-traceable
    (no host-side RoPE gather). This is the callable wrapped by ``torch.compile``.
    """
    h = _run_backbone_forward(
        model,
        input_ids,
        selected_freqs,
        attn_mask,
        key_caches,
        value_caches,
        is_filling,
        token_index,
        cache_position,
    )
    logits = model.lm_head(h)
    return logits / text_config(model.config).logits_scaling


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
    """Eager Granite forward entry (position_ids based).

    Used by the stock token-compare test and any caller that still passes
    ``position_ids``. Performs the host-side RoPE gather here, then delegates to
    the freqs-based body. ``PrecomputedRotaryEmbedding.forward`` uses only its
    second arg for the gather, so ``input_ids`` as the first arg is a safe filler.
    """
    selected_freqs = model._spyre_rope(input_ids, position_ids)
    return _run_forward_freqs(
        model,
        input_ids,
        selected_freqs,
        attn_mask,
        key_caches,
        value_caches,
        is_filling,
        token_index,
        cache_position,
    )


def _make_compiled_run_forward(model):
    """Bind Granite's whole-forward to ``model`` and torch.compile it once.

    The compiled callable owns the embed/mul prologue, the 40-block loop (each
    block a nested_compile_region → compiled once), and the norm/head/scaling
    epilogue. It consumes ``selected_freqs`` as a graph INPUT — the host-side
    RoPE gather runs in the generate shim before this is called (see
    ``auto_spyre_model._resolve_run_forward_fn``). Signature matches that shim's
    call minus the leading ``model`` (which is closed over)."""
    def _bound(
        input_ids, selected_freqs, attn_mask,
        key_caches, value_caches, is_filling, token_index, cache_position,
    ):
        return _run_forward_freqs(
            model, input_ids, selected_freqs, attn_mask,
            key_caches, value_caches, is_filling, token_index, cache_position,
        )

    return torch.compile(_bound, dynamic=False)


def prepare_for_spyre(model):
    """Apply Spyre adaptations to Granite 3.3 model in-place."""
    from transformers.models.granite.modeling_granite import GraniteRMSNorm

    prepare_rope_and_heads(model)
    patch_rmsnorm(GraniteRMSNorm)
    pad_lm_head(model)
    model._spyre_compiled_blocks = prepare_standard_gqa_region_blocks(
        get_backbone(model).layers, True
    )
    model._spyre_run_forward = _make_compiled_run_forward(model)
