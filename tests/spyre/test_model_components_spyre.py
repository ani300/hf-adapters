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

"""Focused Spyre tests for model and adapter components.

These tests exercise hardware-specific behavior without loading a complete model.
Run them explicitly on a Spyre pod::

    pytest -s -vvv tests/spyre/test_model_components_spyre.py
"""

import pytest
import torch
import torch.nn.functional as F
from torch_spyre.model_utils import (
    dma_moe_expert_weight_to_spyre,
    dma_moe_per_expert_scale_to_spyre,
)
from transformers import GraniteConfig, PixtralVisionConfig
from transformers.models.granite.modeling_granite import (
    GraniteAttention,
    GraniteRotaryEmbedding,
)
from transformers.models.pixtral.modeling_pixtral import PixtralAttentionLayer

from hf_adapters.hf_common import (
    BLOCK_SIZE,
    _move_to_spyre_with_layout,
    apply_rope_matmul,
    pad_lm_head,
    prepare_rope_and_heads,
)
from hf_adapters.hf_gemma4_moe import _compiled_moe_loop_region
from hf_adapters.hf_pixtral_vision import (
    _make_compiled_pixtral_block,
    _pad_pixtral_heads,
)

pytestmark = pytest.mark.requires_spyre


@pytest.mark.parametrize("batch", [1, 4])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_gemma4_selected_expert_decode(batch, dtype):
    """Compile real Gemma routing and decode with different experts per row."""
    # Gemma 4 26B-A4B geometry with the TP=4 local intermediate padded from
    # 704 / 4 = 176 to 192. This test needs only one card and no checkpoint.
    experts, hidden, intermediate, top_k = 128, 2816, 192, 8
    generator = torch.Generator().manual_seed(27)
    x = torch.randn(batch, hidden, generator=generator, dtype=dtype) * 0.1
    gate = (
        torch.randn(experts, hidden, intermediate, generator=generator, dtype=dtype)
        * 0.1
    )
    up = (
        torch.randn(experts, hidden, intermediate, generator=generator, dtype=dtype)
        * 0.1
    )
    down = (
        torch.randn(experts, intermediate, hidden, generator=generator, dtype=dtype)
        * 0.1
    )
    expert_scale = (torch.rand(experts, generator=generator) + 0.5).to(dtype)

    # Well-separated logits avoid a top-k boundary changing under DL16
    # rounding. Disjoint, nonconsecutive expert sets expose row mixing and
    # incorrect scale gathers; nonuniform weights exercise normalization.
    router_input = torch.eye(batch, hidden, dtype=dtype)
    router_weight = torch.zeros(experts, hidden, dtype=dtype)
    selected = torch.randperm(experts, generator=generator)[: batch * top_k]
    selected = selected.reshape(batch, top_k)
    for row in range(batch):
        router_weight[:, row] = -2
        router_weight[selected[row], row] = torch.linspace(2, 4, top_k).to(dtype)
    router_scale = torch.linspace(0.8, 1.2, hidden).to(dtype)
    root_size = torch.tensor(hidden**-0.5, dtype=dtype)
    eps = 1e-6

    # Independent float32 mathematical reference, including RMS normalization,
    # router projection, selected-expert normalization, and per-expert scales.
    router_float = router_input.float()
    normalized = router_float * torch.rsqrt(
        router_float.square().mean(-1, keepdim=True) + eps
    )
    logits = F.linear(
        normalized * router_scale.float() * root_size.float(), router_weight.float()
    )
    weights, indices = logits.softmax(-1).topk(top_k, dim=-1)
    weights /= weights.sum(-1, keepdim=True)
    expected = torch.zeros(batch, hidden)
    for row in range(batch):
        for slot in range(top_k):
            expert = indices[row, slot]
            activated = F.gelu(
                x[row].float() @ gate[expert].float(), approximate="tanh"
            ) * (x[row].float() @ up[expert].float())
            expected[row] += (
                (activated @ down[expert].float())
                * weights[row, slot]
                * expert_scale[expert].float()
            )

    args = (
        router_input.to("spyre"),
        x.to("spyre"),
        router_weight.to("spyre"),
        router_scale.to("spyre"),
        root_size.to("spyre"),
        dma_moe_per_expert_scale_to_spyre(expert_scale),
        dma_moe_expert_weight_to_spyre(gate),
        dma_moe_expert_weight_to_spyre(up),
        dma_moe_expert_weight_to_spyre(down),
        top_k,
        32,
        64,
        eps,
    )
    compiled = torch.compile(_compiled_moe_loop_region, dynamic=False, fullgraph=True)
    with torch.inference_mode():
        actual = compiled(*args).cpu()
        repeated = compiled(*args).cpu()
    assert actual.dtype == dtype
    # Allow DL16 rounding through the complete router and three expert GEMMs,
    # including cancellation near zero in the weighted expert sum.
    torch.testing.assert_close(actual.float(), expected, atol=2e-3, rtol=0.05)
    torch.testing.assert_close(repeated, actual, atol=0, rtol=0)


class _Layer(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = GraniteAttention(config, layer_idx=0)


class _Model(torch.nn.Module):
    def __init__(self, dtype):
        super().__init__()
        # Granite 3.3 2B attention geometry. Keep only the modules needed by
        # prepare_rope_and_heads rather than constructing a complete causal LM.
        self.config = GraniteConfig(
            hidden_size=2048,
            intermediate_size=8192,
            num_hidden_layers=1,
            num_attention_heads=32,
            num_key_value_heads=8,
            vocab_size=49159,
            max_position_embeddings=131072,
            attention_multiplier=0.015625,
            rope_parameters={"rope_theta": 10000000.0, "rope_type": "default"},
        )
        self.config.head_dim = 64
        self.layers = torch.nn.ModuleList([_Layer(self.config)])
        self.rotary_emb = GraniteRotaryEmbedding(self.config)
        self.to(dtype)


class _AttentionRunner(torch.nn.Module):
    def __init__(self, attn):
        super().__init__()
        self.attn = attn

    def forward(self, hidden_states, selected_freqs):
        return _attention_forward(self.attn, hidden_states, selected_freqs)


def _attention_forward(attn, hidden_states, selected_freqs):
    batch_size, seq_len, _ = hidden_states.shape
    q = (
        attn.q_proj(hidden_states)
        .view(batch_size, seq_len, -1, attn.head_dim)
        .transpose(1, 2)
    )
    k = (
        attn.k_proj(hidden_states)
        .view(batch_size, seq_len, -1, attn.head_dim)
        .transpose(1, 2)
    )
    v = (
        attn.v_proj(hidden_states)
        .view(batch_size, seq_len, -1, attn.head_dim)
        .transpose(1, 2)
    )
    q = apply_rope_matmul(q, selected_freqs)
    k = apply_rope_matmul(k, selected_freqs)
    attn_output = F.scaled_dot_product_attention(
        q,
        k,
        v,
        dropout_p=0.0,
        scale=attn.scaling,
        enable_gqa=True,
    )
    attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, -1)
    return attn.o_proj(attn_output)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_embedding_module(dtype):
    """Exercise production-sized embedding transfer and gathers on Spyre.

    Verifies that torch-spyre's patched ``Module.to`` path accepts a large
    embedding table, selects a gather-compatible layout, and preserves CPU
    lookup results for the first, middle, and last vocabulary rows in both
    supported 16-bit dtypes.
    """

    # Gemma 4 12B text embedding geometry.
    vocab_size = 262144
    hidden_size = 3840

    torch.manual_seed(42)

    input_ids = torch.tensor([[0, vocab_size // 2, vocab_size - 1]], dtype=torch.int64)
    embedding = torch.nn.Embedding(
        num_embeddings=vocab_size,
        embedding_dim=hidden_size,
        dtype=dtype,
    )
    torch.nn.init.normal_(embedding.weight, mean=0.0, std=0.02)

    # Run the CPU reference before moving the module to Spyre.
    with torch.no_grad():
        cpu_output = embedding(input_ids)

    # Module.to("spyre") routes embeddings through load_model_to_spyre(), which
    # assigns the indirect-access layout required by the gather kernel.
    assert getattr(
        torch.nn.Module.to, "_spyre_patched", False
    ), "torch-spyre did not patch torch.nn.Module.to"
    embedding.to("spyre")

    with torch.no_grad():
        spyre_output = embedding(input_ids.to("spyre")).to("cpu")

    torch.testing.assert_close(spyre_output, cpu_output, rtol=0, atol=1e-3)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_padded_lm_head(dtype):
    """Exercise smooth-stick padding of a production-sized LM head.

    Starts from an aligned 2374-stick vocabulary whose large prime factor can
    overflow Spyre's per-core EAR during bundling. Verifies that ``pad_lm_head``
    advances it to a safe stick count, that the resulting large projection
    compiles in both supported 16-bit dtypes, that original-vocabulary logits
    match the unpadded CPU projection, and that padded logits remain exactly
    zero.
    """

    vocab_size = 151936
    hidden_size = 4096

    torch.manual_seed(42)

    model = torch.nn.Module()
    model.lm_head = torch.nn.Linear(hidden_size, vocab_size, bias=False, dtype=dtype)
    hidden_states = torch.randn(1, 1, hidden_size, dtype=dtype)

    with torch.no_grad():
        cpu_output = model.lm_head(hidden_states)

    pad_lm_head(model)
    padded_vocab_size = model.lm_head.weight.shape[0]
    assert padded_vocab_size % BLOCK_SIZE == 0
    assert padded_vocab_size > vocab_size

    model.lm_head.to("spyre")
    compiled_lm_head = torch.compile(model.lm_head, dynamic=False)

    with torch.no_grad():
        spyre_output = compiled_lm_head(hidden_states.to("spyre")).to("cpu")

    torch.testing.assert_close(
        spyre_output[..., :vocab_size], cpu_output, rtol=0, atol=2e-2
    )
    torch.testing.assert_close(
        spyre_output[..., vocab_size:],
        torch.zeros_like(spyre_output[..., vocab_size:]),
        rtol=0,
        atol=0,
    )


def test_pixtral_compiled_block():
    """Exercise a complete production-sized Pixtral encoder block on Spyre.

    Covers head padding, projections, RoPE, bidirectional SDPA, residual
    connections, and the SwiGLU MLP using Ministral-3's vision configuration and
    a representative padded image-patch sequence.
    """

    config = PixtralVisionConfig(
        hidden_size=1024,
        intermediate_size=4096,
        num_hidden_layers=1,
        num_attention_heads=16,
        image_size=1540,
        patch_size=14,
    )
    layer = PixtralAttentionLayer(config).eval().to(torch.bfloat16)

    head_dim = config.hidden_size // config.num_attention_heads
    padded_head_dim = 2 * BLOCK_SIZE
    _pad_pixtral_heads([layer], config.num_attention_heads, head_dim, padded_head_dim)
    compiled_block = _make_compiled_pixtral_block(
        layer,
        config.num_attention_heads,
        padded_head_dim,
        head_dim**-0.5,
    )
    _move_to_spyre_with_layout(layer, torch.bfloat16)

    sequence_length = 3520
    torch.manual_seed(42)
    hidden_states = torch.randn(
        1, sequence_length, config.hidden_size, dtype=torch.bfloat16
    )
    selected_freqs = torch.zeros(
        1,
        sequence_length,
        2,
        2,
        padded_head_dim // 2,
        dtype=torch.bfloat16,
    )
    selected_freqs[:, :, 0, 0, :] = 1
    selected_freqs[:, :, 1, 1, :] = 1
    attn_mask = torch.zeros(
        1, 1, sequence_length, sequence_length, dtype=torch.bfloat16
    )

    with torch.no_grad():
        output = compiled_block(
            hidden_states.to("spyre"),
            selected_freqs.to("spyre"),
            attn_mask.to("spyre"),
        )

    assert output.shape == hidden_states.shape


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_prepare_rope_and_heads_padded_attention_matches_cpu(dtype):
    """Exercise Granite head padding, RoPE preparation, and compiled attention.

    Uses Granite 3.3 2B geometry, whose 64-element heads must be padded to 128
    for Spyre's matmul-based RoPE. Verifies the projection and RoPE dimensions,
    preserved attention scaling, CPU equivalence of the padded adapter path to
    stock Hugging Face attention, production RoPE dtype propagation and cache
    prebuild during model transfer, and compiled Spyre output parity in both
    supported 16-bit dtypes.
    """

    torch.manual_seed(42)

    model = _Model(dtype).eval()
    attn = model.layers[0].self_attn
    hidden_states = torch.randn(1, BLOCK_SIZE, model.config.hidden_size, dtype=dtype)
    position_ids = torch.arange(BLOCK_SIZE).unsqueeze(0)

    with torch.no_grad():
        cos, sin = model.rotary_emb(hidden_states, position_ids)
        cpu_output, _ = attn(
            hidden_states,
            position_embeddings=(cos, sin),
        )

    freqs = torch.outer(position_ids[0].float(), model.rotary_emb.inv_freq)
    cpu_freqs = torch.stack(
        [torch.cos(freqs), -torch.sin(freqs), torch.sin(freqs), torch.cos(freqs)],
        dim=1,
    ).view(1, BLOCK_SIZE, 2, 2, model.config.head_dim // 2)
    cpu_freqs = cpu_freqs.to(dtype)

    original_scaling = attn.scaling
    prepare_rope_and_heads(model)

    padded_head_dim = 2 * BLOCK_SIZE
    assert model._spyre_head_dim == padded_head_dim
    assert attn.head_dim == padded_head_dim
    assert attn.scaling == original_scaling
    assert (
        attn.q_proj.out_features == model.config.num_attention_heads * padded_head_dim
    )
    assert (
        attn.k_proj.out_features == model.config.num_key_value_heads * padded_head_dim
    )
    assert (
        attn.v_proj.out_features == model.config.num_key_value_heads * padded_head_dim
    )
    assert attn.o_proj.in_features == model.config.num_attention_heads * padded_head_dim

    identity_padding = torch.zeros(
        1,
        BLOCK_SIZE,
        2,
        2,
        (padded_head_dim - model.config.head_dim) // 2,
        dtype=dtype,
    )
    identity_padding[:, :, 0, 0, :] = 1
    identity_padding[:, :, 1, 1, :] = 1
    padded_cpu_freqs = torch.cat([cpu_freqs, identity_padding], dim=-1)
    with torch.no_grad():
        padded_cpu_output = _attention_forward(attn, hidden_states, padded_cpu_freqs)
    torch.testing.assert_close(padded_cpu_output, cpu_output, rtol=0, atol=2e-3)

    _move_to_spyre_with_layout(model, dtype)
    spyre_freqs = model._spyre_rope(hidden_states, position_ids)
    assert spyre_freqs.shape[-1] == padded_head_dim // 2

    runner = _AttentionRunner(attn)
    compiled_runner = torch.compile(runner, dynamic=False)
    with torch.no_grad():
        spyre_output = compiled_runner(hidden_states.to("spyre"), spyre_freqs).to("cpu")

    torch.testing.assert_close(spyre_output, cpu_output, rtol=0, atol=2e-2)
