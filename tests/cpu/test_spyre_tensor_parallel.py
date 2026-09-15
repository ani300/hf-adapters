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

from types import SimpleNamespace

import torch
import torch.nn as nn
from transformers.integrations.tensor_parallel import ALL_PARALLEL_STYLES

from hf_adapters import hf_gemma4_moe
from hf_adapters.hf_common import _prefer_exact_tp_plan_entries
from hf_adapters.hf_gemma4 import spyre_tp_grouped_colwise_modules
from hf_adapters.spyre_tensor_parallel import (
    SPYRE_COLWISE,
    SPYRE_CPU_PACKED_COLWISE,
    SPYRE_CPU_ROWWISE,
    SPYRE_EMBEDDING_COLWISE,
    SPYRE_EMBEDDING_ROWWISE,
    SPYRE_GROUPED_COLWISE_PREFIX,
    SPYRE_REPLICATED_EMBEDDING,
    SPYRE_REPLICATED_LINEAR,
    SPYRE_ROWWISE,
    SpyreEmbeddingColwiseParallel,
    SpyreEmbeddingRowwiseParallel,
    prepare_spyre_tp_plan,
    register_spyre_tp_styles,
)


class _ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(64, 16)
        self.model.layers = nn.ModuleList([nn.Module(), nn.Module()])
        for layer in self.model.layers:
            layer.q_proj = nn.Linear(16, 32, bias=False)
            layer.o_proj = nn.Linear(32, 16, bias=False)
            layer.router = nn.Linear(16, 4, bias=False)
            layer.experts = nn.Module()
            layer.experts.gate_up_proj = nn.Parameter(torch.empty(2, 32, 16))
            layer.experts.down_proj = nn.Parameter(torch.empty(2, 16, 16))
        self.lm_head = nn.Linear(16, 64, bias=False)


def test_prepare_spyre_tp_plan_translates_and_covers_placement():
    model = _ToyModel()
    plan = {
        "model.embed_tokens": "embedding_rowwise",
        "model.layers.*.q_proj": "colwise",
        "model.layers.*.o_proj": "rowwise",
        "model.layers.*.experts.gate_up_proj": "packed_colwise",
        "model.layers.*.experts.down_proj": "rowwise",
    }
    staged = {
        "model.layers.*.experts.gate_up_proj",
        "model.layers.*.experts.down_proj",
    }

    result = prepare_spyre_tp_plan(
        model,
        plan,
        cpu_staged_modules=staged,
        replicated_linear_modules={"lm_head"},
        replicated_embedding_modules={"model.embed_tokens"},
    )

    assert result["model.embed_tokens"] == SPYRE_REPLICATED_EMBEDDING
    assert result["model.layers.*.q_proj"] == SPYRE_COLWISE
    assert result["model.layers.*.o_proj"] == SPYRE_ROWWISE
    assert result["model.layers.*.experts.gate_up_proj"] == SPYRE_CPU_PACKED_COLWISE
    assert result["model.layers.*.experts.down_proj"] == SPYRE_CPU_ROWWISE
    assert result["model.layers.*.router"] == SPYRE_REPLICATED_LINEAR
    assert result["lm_head"] == SPYRE_REPLICATED_LINEAR


def test_embedding_styles_reconstruct_with_their_registered_dimension():
    prepare_spyre_tp_plan(_ToyModel(), {})

    row_style = ALL_PARALLEL_STYLES[SPYRE_EMBEDDING_ROWWISE]
    col_style = ALL_PARALLEL_STYLES[SPYRE_EMBEDDING_COLWISE]
    reconstructed_row = row_style.__class__()
    reconstructed_col = col_style.__class__()

    assert isinstance(reconstructed_row, SpyreEmbeddingRowwiseParallel)
    assert reconstructed_row.embedding_dim_sharding == 0
    assert isinstance(reconstructed_col, SpyreEmbeddingColwiseParallel)
    assert reconstructed_col.embedding_dim_sharding == 1


def test_explicit_embedding_and_lm_head_styles_are_not_forced_replicated():
    result = prepare_spyre_tp_plan(
        _ToyModel(),
        {
            "model.embed_tokens": "embedding_rowwise",
            "lm_head": "colwise",
        },
    )

    assert result["model.embed_tokens"] == SPYRE_EMBEDDING_ROWWISE
    assert result["lm_head"] == SPYRE_COLWISE


def test_cpu_staged_styles_return_local_cpu_shards():
    register_spyre_tp_styles()

    class Mesh:
        shape = (2,)

        @staticmethod
        def size():
            return 2

    packed = ALL_PARALLEL_STYLES[SPYRE_CPU_PACKED_COLWISE].__class__(
        device_mesh=Mesh(), rank=1, empty_param=torch.empty(2, 8, 8)
    )
    rowwise = ALL_PARALLEL_STYLES[SPYRE_CPU_ROWWISE].__class__(
        device_mesh=Mesh(), rank=1, empty_param=torch.empty(2, 8, 8)
    )

    packed_shard = packed.shard_tensor(torch.arange(128).reshape(8, 16))
    rowwise_shard = rowwise.shard_tensor(torch.arange(128).reshape(2, 8, 8))

    assert packed_shard.device.type == "cpu"
    assert packed_shard.shape == (4, 16)
    assert rowwise_shard.device.type == "cpu"
    assert rowwise_shard.shape == (2, 8, 4)


def test_gemma4_tp4_groups_kv_only_when_a_shard_would_split_a_head():
    assert (
        hf_gemma4_moe.spyre_tp_grouped_colwise_modules
        is spyre_tp_grouped_colwise_modules
    )

    class Attention(nn.Module):
        def __init__(self, kv_heads):
            super().__init__()
            head_dim = 256
            self.q_proj = nn.Linear(16, 8 * head_dim, bias=False)
            self.k_proj = nn.Linear(16, kv_heads * head_dim, bias=False)
            self.v_proj = nn.Linear(16, kv_heads * head_dim, bias=False)
            self.o_proj = nn.Linear(8 * head_dim, 16, bias=False)

    class GemmaModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.language_model = nn.Module()
            self.model.language_model.layers = nn.ModuleList([nn.Module(), nn.Module()])
            self.model.language_model.layers[0].self_attn = Attention(kv_heads=2)
            self.model.language_model.layers[1].self_attn = Attention(kv_heads=4)
            layer_cfg = SimpleNamespace(head_dim=256)
            self.config = SimpleNamespace(
                text_config=SimpleNamespace(per_layer_config=[layer_cfg, layer_cfg])
            )

    model = GemmaModel()
    plan = {
        "model.language_model.layers.*.self_attn.q_proj": "colwise",
        "model.language_model.layers.*.self_attn.k_proj": "colwise",
        "model.language_model.layers.*.self_attn.v_proj": "colwise",
        "model.language_model.layers.*.self_attn.o_proj": "rowwise",
    }
    grouped = spyre_tp_grouped_colwise_modules(model, tp_size=4)
    result = prepare_spyre_tp_plan(model, plan, grouped_colwise_modules=grouped)

    prefix = "model.language_model.layers"
    grouped_style = f"{SPYRE_GROUPED_COLWISE_PREFIX}_2"
    assert result[f"{prefix}.0.self_attn.k_proj"] == grouped_style
    assert result[f"{prefix}.0.self_attn.v_proj"] == grouped_style
    assert list(result).index(f"{prefix}.0.self_attn.k_proj") < list(result).index(
        f"{prefix}.*.self_attn.k_proj"
    )
    assert result[f"{prefix}.*.self_attn.q_proj"] == SPYRE_COLWISE
    assert result[f"{prefix}.*.self_attn.o_proj"] == SPYRE_ROWWISE
    assert f"{prefix}.1.self_attn.k_proj" not in result
    assert f"{prefix}.1.self_attn.v_proj" not in result


def test_exact_tp_plan_entry_precedes_transformers_wildcard_lookup():
    from transformers import modeling_utils
    from transformers.integrations import tensor_parallel

    parameter = "model.layers.5.self_attn.k_proj.weight"
    plan = {
        "model.layers.*.self_attn.k_proj": SPYRE_COLWISE,
        "model.layers.5.self_attn.k_proj": SPYRE_REPLICATED_LINEAR,
    }

    assert tensor_parallel._get_parameter_tp_plan(parameter, plan) == SPYRE_COLWISE
    with _prefer_exact_tp_plan_entries():
        assert (
            tensor_parallel._get_parameter_tp_plan(parameter, plan)
            == SPYRE_REPLICATED_LINEAR
        )
        assert (
            modeling_utils._get_parameter_tp_plan(parameter, plan)
            == SPYRE_REPLICATED_LINEAR
        )
    assert tensor_parallel._get_parameter_tp_plan(parameter, plan) == SPYRE_COLWISE
