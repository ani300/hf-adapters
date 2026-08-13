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
"""Blocker 6 projection-consumer discriminator.

The raw Q/K/V projections are bit-exact, while projections plus RoPE and cache
writes are wrong. This script separates the two consumers:

1. projections + RoPE, with no mutation;
2. projections + KV-cache writes, with no RoPE.

No RNG seed is used. Run from the hf-adapters worktree:

    PYTHONPATH="$PWD" HF_HOME=/mnt/models/hf_cache/ \
        python repros/hier/standalone_projection_consumers_discriminator.py
"""

import math

import torch
from torch.compiler import nested_compile_region

from hf_adapters.auto_spyre_model import torch_dtype_for_model_path
from hf_adapters.hf_common import (
    DEVICE,
    allocate_kv_caches,
    apply_rope_matmul,
    build_prefill_mask,
    generation_cache_len,
    get_backbone,
    get_model_dtype,
    kv_cache_update,
    move_model_to_spyre,
)
from tests.conftest import load_ref_model, resolve_adapter_module_for_test

MODEL = "ibm-granite/granite-3.3-2b-instruct"
BLOCK_SIZE = 64


def _project(attn, hidden_states):
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


@nested_compile_region
def _shared_projection_rope(attn, hidden_states, selected_freqs):
    q, k, v = _project(attn, hidden_states)
    return (
        apply_rope_matmul(q, selected_freqs),
        apply_rope_matmul(k, selected_freqs),
        v,
    )


@nested_compile_region
def _shared_projection_cache(
    attn,
    hidden_states,
    key_cache,
    value_cache,
):
    q, k, v = _project(attn, hidden_states)
    kv_cache_update(k, v, key_cache, value_cache, False, 0, 0)
    return q


def _print_deltas(label, names, n2_values, n1_values):
    deltas = [
        (n2 - n1).abs().max().item()
        for n2, n1 in zip(n2_values, n1_values)
    ]
    print(f"\n=== {label} ===", flush=True)
    for name, delta in zip(names, deltas):
        print(f"max|{name} _1(call1) - _0(call1)| = {delta:.6f}")
    return deltas


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
    mdtype = get_model_dtype(model)

    padded_len = math.ceil(seq_len / BLOCK_SIZE) * BLOCK_SIZE
    prompt_offset = padded_len - seq_len
    if prompt_offset:
        pad = input_ids.new_zeros((batch_size, prompt_offset))
        padded_ids = torch.cat([pad, input_ids], dim=1)
    else:
        padded_ids = input_ids

    position_ids = torch.zeros((batch_size, padded_len), dtype=torch.long)
    position_ids[:, prompt_offset:] = torch.arange(seq_len)
    max_cache_len = generation_cache_len(seq_len, 5)
    build_prefill_mask(
        batch_size,
        padded_len,
        max_cache_len,
        prompt_offset,
        dtype=mdtype,
    )

    dev_ids = padded_ids.to(DEVICE)
    dev_pos = position_ids.to(DEVICE)
    freqs = model._spyre_rope(dev_ids, dev_pos)

    backbone = get_backbone(model)
    block = list(backbone.layers)[0]
    attn = block.self_attn
    with torch.no_grad():
        h0 = backbone.embed_tokens(dev_ids) * backbone.embedding_multiplier
        norm_fn = torch.compile(block.input_layernorm, dynamic=False)
        h_norm = norm_fn(h0)

    print("\n=== RoPE consumer: N2 shared body ===", flush=True)

    def two_rope(h, selected_freqs):
        q1, k1, v1 = _shared_projection_rope(attn, h, selected_freqs)
        q2, k2, v2 = _shared_projection_rope(
            attn, h * 1.0, selected_freqs
        )
        return q1, k1, v1, q2, k2, v2

    n2_rope_fn = torch.compile(two_rope, dynamic=False)
    with torch.no_grad():
        n2_q, n2_k, n2_v, _q2, _k2, _v2 = n2_rope_fn(h_norm, freqs)
    n2_rope = tuple(t.to("cpu").float() for t in (n2_q, n2_k, n2_v))

    print("\n=== RoPE consumer: N1 inline body ===", flush=True)

    def one_rope(h, selected_freqs):
        return _shared_projection_rope(attn, h, selected_freqs)

    n1_rope_fn = torch.compile(one_rope, dynamic=False)
    with torch.no_grad():
        n1_q, n1_k, n1_v = n1_rope_fn(h_norm, freqs)
    n1_rope = tuple(t.to("cpu").float() for t in (n1_q, n1_k, n1_v))
    rope_deltas = _print_deltas(
        "projections + RoPE, no mutation",
        ("Q", "K", "raw V"),
        n2_rope,
        n1_rope,
    )

    kcs2, vcs2 = allocate_kv_caches(
        model, batch_size, max_cache_len, mdtype
    )
    kc0, vc0 = kcs2[0], vcs2[0]
    kc0_entry = kc0.to("cpu").clone()
    vc0_entry = vc0.to("cpu").clone()

    print("\n=== Cache consumer: N2 shared body ===", flush=True)

    def two_cache(h, kcs, vcs):
        q1 = _shared_projection_cache(attn, h, kcs[0], vcs[0])
        q2 = _shared_projection_cache(
            attn, h * 1.0, kcs[1], vcs[1]
        )
        return q1, q2

    n2_cache_fn = torch.compile(two_cache, dynamic=False)
    with torch.no_grad():
        n2_q, _n2_q2 = n2_cache_fn(h_norm, kcs2, vcs2)
    n2_cache = (
        n2_q.to("cpu").float(),
        kc0.to("cpu").float(),
        vc0.to("cpu").float(),
    )

    with torch.no_grad():
        kc0.copy_(kc0_entry.to(DEVICE))
        vc0.copy_(vc0_entry.to(DEVICE))

    print("\n=== Cache consumer: N1 inline body ===", flush=True)

    def one_cache(h, kc, vc):
        return _shared_projection_cache(attn, h, kc, vc)

    n1_cache_fn = torch.compile(one_cache, dynamic=False)
    with torch.no_grad():
        n1_q = n1_cache_fn(h_norm, kc0, vc0)
    n1_cache = (
        n1_q.to("cpu").float(),
        kc0.to("cpu").float(),
        vc0.to("cpu").float(),
    )
    cache_deltas = _print_deltas(
        "projections + cache writes, no RoPE",
        ("raw Q", "K cache", "V cache"),
        n2_cache,
        n1_cache,
    )

    if any(delta > 0.5 for delta in rope_deltas):
        print(">>> RoPE independently exposes wrong internal projections.")
    else:
        print(">>> RoPE-only consumer is clean.")
    if any(delta > 0.5 for delta in cache_deltas):
        print(">>> Cache writes independently expose wrong projections.")
    else:
        print(">>> Cache-write-only consumer is clean.")


if __name__ == "__main__":
    main()
