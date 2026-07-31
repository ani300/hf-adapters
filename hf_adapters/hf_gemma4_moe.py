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
