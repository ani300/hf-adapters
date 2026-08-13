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
"""Blocker 6 single-projection cache-write discriminator.

Reduce the failing projections-plus-cache-writes body to one K projection BMM
whose result is both returned and copied into a KV-cache-shaped slice. Compare
a two-call shared HOP body with a one-call inline body on identical inputs.

No RNG seed is used. Run from the hf-adapters worktree:

    PYTHONPATH="$PWD" HF_HOME=/mnt/models/hf_cache/ \
        python repros/hier/standalone_single_projection_cache_discriminator.py
"""

import math
import os

import torch
from torch.compiler import nested_compile_region

from hf_adapters.auto_spyre_model import torch_dtype_for_model_path
from hf_adapters.hf_common import (
    DEVICE,
    allocate_kv_caches,
    generation_cache_len,
    get_backbone,
    get_model_dtype,
    move_model_to_spyre,
)
from tests.conftest import load_ref_model, resolve_adapter_module_for_test

MODEL = "ibm-granite/granite-3.3-2b-instruct"
BLOCK_SIZE = 64


def _maybe_disable_direct_unit_bmm_marking():
    if os.environ.get("B6_DISABLE_DIRECT_UNIT_BMM_MARKING") != "1":
        return

    import torch_spyre._inductor.temp_passes as temp_passes

    def skip_static_unit_batch_bmm_marking(_bmm, _lhs, _rhs):
        return None

    temp_passes._mark_static_unit_batch_bmm = (  # noqa: SLF001
        skip_static_unit_batch_bmm_marking
    )
    print(
        ">>> Diagnostic toggle: disabled shared-weight unit-BMM marking",
        flush=True,
    )


@nested_compile_region
def _shared_k_projection_cache(attn, hidden_states, key_cache):
    bsz, seq_len, _ = hidden_states.shape
    k = (
        attn.k_proj(hidden_states)
        .view(bsz, seq_len, -1, attn.head_dim)
        .transpose(1, 2)
    )
    key_cache[:, :, :seq_len, :] = k
    return k


def main():
    from transformers import AutoTokenizer

    _maybe_disable_direct_unit_bmm_marking()

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

    dev_ids = padded_ids.to(DEVICE)
    backbone = get_backbone(model)
    block = list(backbone.layers)[0]
    attn = block.self_attn
    with torch.no_grad():
        h0 = backbone.embed_tokens(dev_ids) * backbone.embedding_multiplier
        norm_fn = torch.compile(block.input_layernorm, dynamic=False)
        h_norm = norm_fn(h0)

    max_cache_len = generation_cache_len(seq_len, 5)
    kcs2, _vcs2 = allocate_kv_caches(
        model, batch_size, max_cache_len, mdtype
    )
    kc0 = kcs2[0]
    kc0_entry = kc0.to("cpu").clone()

    print("\n=== N2: one K projection + cache write, shared body ===", flush=True)

    def two_calls(h, key_caches):
        k1 = _shared_k_projection_cache(attn, h, key_caches[0])
        k2 = _shared_k_projection_cache(
            attn, h * 1.0, key_caches[1]
        )
        return k1, k2

    n2_fn = torch.compile(two_calls, dynamic=False)
    with torch.no_grad():
        n2_k, _n2_k2 = n2_fn(h_norm, kcs2)
    n2_k_host = n2_k.to("cpu").float()
    n2_cache_host = kc0.to("cpu").float()

    with torch.no_grad():
        kc0.copy_(kc0_entry.to(DEVICE))

    print("\n=== N1: one K projection + cache write, inline body ===", flush=True)

    def one_call(h, key_cache):
        return _shared_k_projection_cache(attn, h, key_cache)

    n1_fn = torch.compile(one_call, dynamic=False)
    with torch.no_grad():
        n1_k = n1_fn(h_norm, kc0)
    n1_k_host = n1_k.to("cpu").float()
    n1_cache_host = kc0.to("cpu").float()

    k_delta = (n2_k_host - n1_k_host).abs().max().item()
    cache_delta = (n2_cache_host - n1_cache_host).abs().max().item()
    print(f"max|K _1(call1) - _0(call1)| = {k_delta:.6f}")
    print(
        "max|K cache _1(call1) - _0(call1)| "
        f"= {cache_delta:.6f}"
    )

    if max(k_delta, cache_delta) > 0.5:
        print(">>> One BMM feeding one cache write reproduces Blocker 6.")
    else:
        print(
            ">>> Single BMM + cache write is clean; the failure requires "
            "interaction among multiple projections/copies."
        )


if __name__ == "__main__":
    main()
