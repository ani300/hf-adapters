# Copyright 2024 IBM Corporation
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
"""Blocker 6 raw Q/K/V projection discriminator.

Compare a two-call shared HOP body with a one-call inline body for only the
three attention linear projections plus their view/transpose metadata.  RoPE,
KV-cache mutation, SDPA, and the output projection are absent.

No RNG seed is used. Run from the hf-adapters worktree:

    PYTHONPATH="$PWD" HF_HOME=/mnt/models/hf_cache/ \
        python repros/hier/standalone_projection_discriminator.py
"""

import math

import torch
from torch.compiler import nested_compile_region

from hf_adapters.auto_spyre_model import torch_dtype_for_model_path
from hf_adapters.hf_common import (
    DEVICE,
    get_backbone,
    move_model_to_spyre,
)
from tests.conftest import load_ref_model, resolve_adapter_module_for_test

MODEL = "ibm-granite/granite-3.3-2b-instruct"
BLOCK_SIZE = 64


@nested_compile_region
def _shared_projections(attn, hidden_states):
    bsz, seq_len, _ = hidden_states.shape
    q = (
        attn.q_proj(hidden_states)
        .view(bsz, seq_len, -1, attn.head_dim)
        .transpose(1, 2)
    )
    k = (
        attn.k_proj(hidden_states)
        .view(bsz, seq_len, -1, attn.head_dim)
        .transpose(1, 2)
    )
    v = (
        attn.v_proj(hidden_states)
        .view(bsz, seq_len, -1, attn.v_head_dim)
        .transpose(1, 2)
    )
    return q, k, v


def main():
    from transformers import AutoTokenizer

    adapter = resolve_adapter_module_for_test(MODEL)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = load_ref_model(model_path=MODEL, adapter_mod=adapter)

    input_ids = tokenizer(
        "The capital of France is", return_tensors="pt"
    )["input_ids"]
    seq_len = input_ids.shape[1]
    batch_size = 1

    dtype = torch_dtype_for_model_path(MODEL)
    move_model_to_spyre(model=model, module=adapter, dtype=dtype)

    padded_len = math.ceil(seq_len / BLOCK_SIZE) * BLOCK_SIZE
    prompt_offset = padded_len - seq_len
    if prompt_offset:
        pad = input_ids.new_zeros((batch_size, prompt_offset))
        padded_ids = torch.cat([pad, input_ids], dim=1)
    else:
        padded_ids = input_ids

    dev_ids = padded_ids.to(DEVICE)
    backbone = get_backbone(model)
    block = list(backbone.layers)[0]
    attn = block.self_attn
    with torch.no_grad():
        h0 = backbone.embed_tokens(dev_ids) * backbone.embedding_multiplier
        norm_fn = torch.compile(block.input_layernorm, dynamic=False)
        h_norm = norm_fn(h0)

    print("\n===N2=== two projection region calls -> shared HOP body", flush=True)

    def two_calls(h):
        q1, k1, v1 = _shared_projections(attn, h)
        q2, k2, v2 = _shared_projections(attn, h * 1.0)
        return q1, k1, v1, q2, k2, v2

    n2_fn = torch.compile(two_calls, dynamic=False)
    with torch.no_grad():
        n2_q, n2_k, n2_v, _q2, _k2, _v2 = n2_fn(h_norm)
    n2_hosts = tuple(t.to("cpu").float() for t in (n2_q, n2_k, n2_v))

    print("\n===N1=== one projection region call -> inline body", flush=True)

    def one_call(h):
        return _shared_projections(attn, h)

    n1_fn = torch.compile(one_call, dynamic=False)
    with torch.no_grad():
        n1_q, n1_k, n1_v = n1_fn(h_norm)
    n1_hosts = tuple(t.to("cpu").float() for t in (n1_q, n1_k, n1_v))

    deltas = [
        (n2 - n1).abs().max().item()
        for n2, n1 in zip(n2_hosts, n1_hosts)
    ]
    for name, delta in zip(("Q", "K", "V"), deltas):
        print(
            f"=== max|{name}-projection _1(call1) - _0(call1)| "
            f"= {delta:.6f} ===",
            flush=True,
        )

    failing = [
        name
        for name, delta in zip(("Q", "K", "V"), deltas)
        if delta > 0.5
    ]
    if failing:
        print(
            ">>> Raw projection region reproduces Blocker 6 in: "
            + ", ".join(failing),
            flush=True,
        )
    else:
        print(
            ">>> Raw projections are clean: the defect first appears when "
            "RoPE and/or KV-cache writes are fused.",
            flush=True,
        )


if __name__ == "__main__":
    main()
