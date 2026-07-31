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
Gemma 4 MoE router and expert permutation functions.

Provides host-side (CPU/eager) routing for Gemma 4's mixture of experts
layer, consisting of expert selection and token-expert pair permutation.
"""

import torch
import torch.nn.functional as F


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
