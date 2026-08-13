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
"""Blocker 6 back-half attention discriminator.

Prepare Granite layer 0's real Q/K/V tensors once, then compare a two-call
shared HOP body with a one-call inline body for only:

    SDPA -> transpose/reshape -> output projection

There is no mutable state in the measured region, and both builds consume the
same device tensors.  No RNG seed is used.

Run from the hf-adapters worktree:

    PYTHONPATH="$PWD" HF_HOME=/mnt/models/hf_cache/ \
        python repros/hier/standalone_sdpa_output_discriminator.py
"""

import math

import torch
import torch.nn.functional as F
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


@nested_compile_region
def _shared_sdpa_output(attn, q, k, v, attn_mask):
    attn_out = F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=attn_mask,
        dropout_p=0.0,
        scale=attn.scaling,
        enable_gqa=True,
    )
    bsz, _heads, seq_len, _head_dim = q.shape
    attn_out = attn_out.transpose(1, 2).reshape(bsz, seq_len, -1)
    return attn.o_proj(attn_out)


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
    mask = build_prefill_mask(
        batch_size,
        padded_len,
        max_cache_len,
        prompt_offset,
        dtype=mdtype,
    ).to(DEVICE)

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

    key_caches, value_caches = allocate_kv_caches(
        model, batch_size, max_cache_len, mdtype
    )
    key_cache = key_caches[0]
    value_cache = value_caches[0]

    def prepare_qkv(h, selected_freqs, kc, vc):
        bsz, length, _ = h.shape
        q = (
            attn.q_proj(h)
            .view(bsz, length, -1, attn.head_dim)
            .transpose(1, 2)
        )
        k = (
            attn.k_proj(h)
            .view(bsz, length, -1, attn.head_dim)
            .transpose(1, 2)
        )
        v = (
            attn.v_proj(h)
            .view(bsz, length, -1, attn.v_head_dim)
            .transpose(1, 2)
        )
        q = apply_rope_matmul(q, selected_freqs)
        k = apply_rope_matmul(k, selected_freqs)
        kc, vc = kv_cache_update(k, v, kc, vc, False, 0, 0)
        return q, kc, vc

    prepare_fn = torch.compile(prepare_qkv, dynamic=False)
    with torch.no_grad():
        q, k, v = prepare_fn(
            h_norm, freqs, key_cache, value_cache
        )
    print(
        f"captured q={tuple(q.shape)} k={tuple(k.shape)} "
        f"v={tuple(v.shape)} mask={tuple(mask.shape)}",
        flush=True,
    )

    print("\n===N2=== two SDPA+O region calls -> shared HOP body", flush=True)

    def two_calls(q_arg, k_arg, v_arg, mask_arg):
        out1 = _shared_sdpa_output(
            attn, q_arg, k_arg, v_arg, mask_arg
        )
        out2 = _shared_sdpa_output(
            attn, q_arg * 1.0, k_arg, v_arg, mask_arg
        )
        return out1, out2

    n2_fn = torch.compile(two_calls, dynamic=False)
    with torch.no_grad():
        n2_out1, _n2_out2 = n2_fn(q, k, v, mask)
    n2_host = n2_out1.to("cpu").float()

    print("\n===N1=== one SDPA+O region call -> inline body", flush=True)

    def one_call(q_arg, k_arg, v_arg, mask_arg):
        return _shared_sdpa_output(
            attn, q_arg, k_arg, v_arg, mask_arg
        )

    n1_fn = torch.compile(one_call, dynamic=False)
    with torch.no_grad():
        n1_out = n1_fn(q, k, v, mask)
    n1_host = n1_out.to("cpu").float()

    delta = (n2_host - n1_host).abs().max().item()
    print(
        f"\n=== max|SDPA+O _1(call1) - _0(call1)| = {delta:.6f} ===",
        flush=True,
    )
    if delta > 0.5:
        print(
            ">>> Back half reproduces Blocker 6: SDPA-to-output-projection "
            "integration is sufficient.",
            flush=True,
        )
    else:
        print(
            ">>> Back half is clean: the defect requires the QKV front half "
            "or cross-boundary fusion.",
            flush=True,
        )


if __name__ == "__main__":
    main()
