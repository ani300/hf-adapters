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
"""Blocker 6 attention-pipeline discriminator.

``standalone_sdpa_discriminator.py`` proves that SDPA alone is bit-exact when
compiled as a two-call shared HOP body versus a one-call inline body.  This is
the next bisection: measure the whole ``StandardGQAAttention`` pipeline while
excluding the decoder block's layer norms, residuals, and MLP.

The measured region contains Q/K/V projections, RoPE, KV-cache writes, SDPA,
and the output projection.  The two builds consume the same normalized hidden
state, frequencies, mask, and restored cache tensors.  A large delta localizes
Blocker 6 to attention/projection fusion; zero proves that the defect requires
the rest of the decoder block.

Read-only diagnostic; no RNG seed.  Run from the hf-adapters worktree:

    PYTHONPATH="$PWD" HF_HOME=/mnt/models/hf_cache/ \
        python repros/hier/standalone_attention_discriminator.py
"""

import math

import torch
from torch.compiler import nested_compile_region

from hf_adapters.auto_spyre_model import torch_dtype_for_model_path
from hf_adapters.hf_common import (
    DEVICE,
    allocate_kv_caches,
    build_prefill_mask,
    generation_cache_len,
    get_backbone,
    get_model_dtype,
    move_model_to_spyre,
)
from tests.conftest import load_ref_model, resolve_adapter_module_for_test

MODEL = "ibm-granite/granite-3.3-2b-instruct"
BLOCK_SIZE = 64


@nested_compile_region
def _shared_attention(attn, *args):
    out, _key_cache, _value_cache = attn.forward(*args)
    return out


def main():
    from transformers import AutoTokenizer

    adapter = resolve_adapter_module_for_test(MODEL)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = load_ref_model(model_path=MODEL, adapter_mod=adapter)

    prompt = "The capital of France is"
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
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

    kcs2, vcs2 = allocate_kv_caches(model, batch_size, max_cache_len, mdtype)
    kc0, vc0 = kcs2[0], vcs2[0]
    kc0_entry = kc0.to("cpu").clone()
    vc0_entry = vc0.to("cpu").clone()

    print("\n===N2=== two attention region calls -> shared HOP body", flush=True)

    def two_calls(h, selected_freqs, attn_mask, kcs, vcs):
        out1 = _shared_attention(
            attn, h, selected_freqs, attn_mask, kcs[0], vcs[0], False, 0, 0
        )
        out2 = _shared_attention(
            attn,
            h * 1.0,
            selected_freqs,
            attn_mask,
            kcs[1],
            vcs[1],
            False,
            0,
            0,
        )
        return out1, out2

    n2_fn = torch.compile(two_calls, dynamic=False)
    with torch.no_grad():
        n2_out1, _n2_out2 = n2_fn(h_norm, freqs, mask, kcs2, vcs2)
    n2_host = n2_out1.to("cpu").float()

    with torch.no_grad():
        kc0.copy_(kc0_entry.to(DEVICE))
        vc0.copy_(vc0_entry.to(DEVICE))

    print("\n===N1=== one attention region call -> inline body", flush=True)

    def one_call(h, selected_freqs, attn_mask, kc, vc):
        return _shared_attention(
            attn, h, selected_freqs, attn_mask, kc, vc, False, 0, 0
        )

    n1_fn = torch.compile(one_call, dynamic=False)
    with torch.no_grad():
        n1_out = n1_fn(h_norm, freqs, mask, kc0, vc0)
    n1_host = n1_out.to("cpu").float()

    delta = (n2_host - n1_host).abs().max().item()
    print(
        f"\n=== max|attention _1(call1) - _0(call1)| = {delta:.6f} ===",
        flush=True,
    )
    if delta > 0.5:
        print(
            ">>> Attention pipeline reproduces Blocker 6: bisect projection/RoPE/"
            "KV/SDPA/output-projection fusion.",
            flush=True,
        )
    else:
        print(
            ">>> Attention pipeline is clean: Blocker 6 requires decoder-block "
            "norm/residual/MLP fusion.",
            flush=True,
        )


if __name__ == "__main__":
    main()
