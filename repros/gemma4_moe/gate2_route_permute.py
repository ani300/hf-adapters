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
Gate 2: Gemma 4 MoE top-8 route + permute/unpermute round-trip on Spyre.

Validates that topk(probs, k=8), argsort, gather/index_select, and
index_add/scatter_add compile on Spyre, produce a correct
permute→unpermute identity, and that the indirectly-addressed
[T*K, H] gather/scatter buffers commit with the row dim outermost
(no inserted spyre.restickify on the hot path).

## Findings

### Compilation Status
BLOCKED on topk(k=8). Spyre backend topk decomposition rejects k > 4.

- topk(k=8): BLOCKED — torch_spyre/_inductor/decompositions.py:333
  raises Unsupported("Topk is not supported for this config")
- argsort: UNKNOWN (topk fails first, never reached)
- gather (index_select): UNKNOWN (topk fails first, never reached)
- scatter_add (index_add): UNKNOWN (topk fails first, never reached)

### Numeric Round-Trip
- Result: NOT REACHED
- Tolerance: atol=1e-2, rtol=1e-2 (fp16 reference, single-add error model)

### Layout / Restickify
- [T*K, H] buffer layout: ⚠️ UNDETERMINABLE (compilation did not proceed)
- Unexpected spyre.restickify on hot path: ⚠️ UNDETERMINABLE

### Root Cause
The Spyre topk decomposition in torch_spyre/_inductor/decompositions.py
enforces k ≤ 4. This gate design assumed k=8 support; that assumption
does not hold on the current backend.

See full report at:
.superpowers/sdd/2026-07-31-gemma4-moe-adapter/task-2-report.md
"""

import torch


H, E, T, K = 2816, 128, 64, 8


def route_permute_unpermute(x, logits):
    """
    Permute tokens by expert routing, then unpermute (identity if weights=1).

    x: [T, H] input tokens
    logits: [T, E] expert routing logits
    Returns: (output [T, H], gathered [T*K, H], token_of_row [T*K])
    """
    probs = torch.softmax(logits, dim=-1)          # [T,E]
    w, idx = torch.topk(probs, K, dim=-1)          # [T,K],[T,K]
    w = w / w.sum(-1, keepdim=True)
    flat_expert = idx.reshape(-1)                  # [T*K]
    sort_perm = torch.argsort(flat_expert)         # [T*K]
    # token_of_row stays on device (created on same device as x)
    token_of_row = (
        torch.arange(T * K, device=x.device, dtype=torch.int64) // K
    )[sort_perm]
    gathered = x[token_of_row]                     # [T*K,H] gather
    # identity expert (no weights): scatter straight back, weighted by 1.0
    # sum over K
    out = torch.zeros_like(x)
    out = out.index_add(0, token_of_row, gathered) # scatter_add
    return out, gathered, token_of_row


def ref_sum_over_k(x):
    """Each token is gathered K times and summed (identity = x * K)."""
    return x * K


if __name__ == "__main__":
    x = torch.randn(T, H, dtype=torch.float16)
    logits = torch.randn(T, E, dtype=torch.float16)
    ref, _, _ = route_permute_unpermute(x, logits)
    cfn = torch.compile(route_permute_unpermute, dynamic=False)
    got, _, _ = cfn(x.to("spyre"), logits.to("spyre"))
    torch.testing.assert_close(
        got.cpu(), ref, atol=1e-2, rtol=1e-2
    )
    print("OK route/permute/unpermute round-trip")
