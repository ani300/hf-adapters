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
"""Blocker 6 STANDALONE-BODY discriminator: is the shared-HOP-body kernel (`_1`)
WRONG PER SE, or only wrong as a 2-call effect?

The static SDSC diff (memory: project_invoke_subgraph_layout_prop) showed the
shared-HOP body compiles to a kernel `_1` whose attention batchmatmul tiles the
x-axis 8x4 where the inlined N=1 kernel `_0` tiles it flat-32. Both use 32 cores
on `mb`; coverage is provably identical (8x4==32). So the fold difference is a
TILING RESHAPE, not lost compute. Open question the JSON cannot settle: does the
`_1` build compute the WRONG answer on a SINGLE correct invocation (artifact
wrong per se -> shared-body BUILD is the root cause), or is it bit-correct alone
and only the >=2-call replay corrupts it (runtime scheduling effect)?

DISCRIMINATOR (no eager reference; isolates build-of-`_1` from call-count):
  1. Compile a 2-call region graph (N=2 -> forces the shared-HOP `_1` lowering).
     Capture call-1's EXACT device inputs (h0, freqs, mask, kc, vc) and its
     output tap. This tap is `_1` evaluated on call-1's inputs.
  2. Compile the SAME block as a 1-call region graph (N=1 -> forces the inlined
     `_0` lowering) and run it on the IDENTICAL captured device tensors.
  3. Compare. Because inputs are byte-identical device tensors, the ONLY variable
     is which compiled body (`_1` vs `_0`) evaluated them.

     |_1(call1 inputs) - _0(call1 inputs)| ~ 0     -> `_1` artifact is CORRECT
        on a single invocation; the 14.61 bug is a >=2-call runtime replay effect
        (reopen the runtime-replay lead; fold reshape is benign).
     |_1(call1 inputs) - _0(call1 inputs)| ~ 14.61 -> `_1` artifact is WRONG PER
        SE; the shared-body BUILD mis-plans call-1's own computation. Fold
        planner / subgraph-body lowering is the root cause.

To remove the KV-allocation asymmetry that two_layer_call_isolation left open,
the N=1 re-run consumes the SAME kc/vc device tensors call-1 used in the N=2 run
(freshly re-allocated caches are NOT used). Since call-1 wrote into kc/vc during
the N=2 run, we snapshot kc/vc to host BEFORE the N=2 run and restore them onto
the SAME device tensors right before the N=1 re-run, so `_0` sees exactly the
KV state `_1` saw at entry.

Read-only diagnostic; NO fix, NO RNG seed. Run (worktree, card free):

    PYTHONPATH="$PWD" HF_HOME=/mnt/models/hf_cache/ \
        python repros/hier/standalone_body_discriminator.py 2>&1 | tee /tmp/standalone_body.log
"""
import math

import torch

from hf_adapters.auto_spyre_model import torch_dtype_for_model_path
from hf_adapters.hf_common import (
    DEVICE,
    allocate_kv_caches,
    build_prefill_mask,
    generation_cache_len,
    get_backbone,
    get_model_dtype,
    move_model_to_spyre,
    nested_region_block,
)
from tests.conftest import load_ref_model, resolve_adapter_module_for_test

MODEL = "ibm-granite/granite-3.3-2b-instruct"
BLOCK_SIZE = 64


def main():
    from transformers import AutoTokenizer

    adapter = resolve_adapter_module_for_test(MODEL)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = load_ref_model(model_path=MODEL, adapter_mod=adapter)

    prompt = "The capital of France is"
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"]
    seq_len = input_ids.shape[1]
    batch_size = 1

    dtype = torch_dtype_for_model_path(MODEL)
    move_model_to_spyre(model=model, module=adapter, dtype=dtype)
    mdtype = get_model_dtype(model)

    padded_len = math.ceil(seq_len / BLOCK_SIZE) * BLOCK_SIZE
    prompt_offset = padded_len - seq_len
    if prompt_offset > 0:
        pad = input_ids.new_zeros((batch_size, prompt_offset))
        padded_ids = torch.cat([pad, input_ids], dim=1)
    else:
        padded_ids = input_ids
    position_ids = torch.zeros((batch_size, padded_len), dtype=torch.long)
    position_ids[:, prompt_offset:] = torch.arange(seq_len)
    max_cache_len = generation_cache_len(seq_len, 5)
    prefill_mask = build_prefill_mask(
        batch_size, padded_len, max_cache_len, prompt_offset, dtype=mdtype
    )

    dev_ids = padded_ids.to(DEVICE)
    dev_pos = position_ids.to(DEVICE)
    dev_mask = prefill_mask.to(DEVICE)
    selected_freqs = model._spyre_rope(dev_ids, dev_pos)

    backbone = get_backbone(model)
    blk0 = list(backbone.layers)[0]  # SAME block instance in both graphs

    with torch.no_grad():
        h0 = backbone.embed_tokens(dev_ids) * backbone.embedding_multiplier

    rc0 = nested_region_block(blk0)

    # ============ (N=2) forces the shared-HOP `_1` lowering ============
    # call-1 inputs = (h0, freqs, mask, kc, vc). Snapshot kc/vc host state at
    # entry so we can restore it for the N=1 re-run (call-1 mutates kc/vc).
    kcs2, vcs2 = allocate_kv_caches(model, batch_size, max_cache_len, mdtype)
    kc0, vc0 = kcs2[0], vcs2[0]
    kc0_entry = kc0.to("cpu").clone()
    vc0_entry = vc0.to("cpu").clone()

    print("\n===N2=== two region calls -> `_1` shared-HOP body", flush=True)

    def two_layer(h, freqs, mask, kcs, vcs):
        h1 = rc0(h, freqs, mask, kcs[0], vcs[0], False, 0, 0)
        h2 = rc0(h1, freqs, mask, kcs[1], vcs[1], False, 0, 0)
        return [h1, h2]

    n2_fn = torch.compile(two_layer, dynamic=False)
    with torch.no_grad():
        n2_taps = n2_fn(h0, selected_freqs, dev_mask, kcs2, vcs2)
    n2_tap0_host = n2_taps[0].to("cpu").float()
    print(f"===N2=== done ({len(n2_taps)} taps); tap0 = `_1`(call-1 inputs)",
          flush=True)

    # ============ (N=1) forces the inlined `_0` lowering ============
    # Restore kc0/vc0 to their ENTRY state so `_0` sees exactly the KV `_1` saw.
    with torch.no_grad():
        kc0.copy_(kc0_entry.to(DEVICE))
        vc0.copy_(vc0_entry.to(DEVICE))

    print("\n===N1=== one region call -> inlined `_0` body", flush=True)

    def one_layer(h, freqs, mask, kc, vc):
        return rc0(h, freqs, mask, kc, vc, False, 0, 0)

    n1_fn = torch.compile(one_layer, dynamic=False)
    with torch.no_grad():
        # IDENTICAL device tensors call-1 used: h0, selected_freqs, dev_mask,
        # and the restored kc0/vc0.
        n1_out = n1_fn(h0, selected_freqs, dev_mask, kc0, vc0)
    n1_host = n1_out.to("cpu").float()
    print(f"===N1=== done, shape={tuple(n1_out.shape)} = `_0`(call-1 inputs)",
          flush=True)

    # =================== discriminator ===================
    d = (n2_tap0_host - n1_host).abs().max().item()
    print(f"\n=== max| _1(call1 inputs) - _0(call1 inputs) | = {d:.6f} ===",
          flush=True)
    if d > 0.5:
        print(">>> `_1` ARTIFACT IS WRONG PER SE on a single invocation: the "
              "shared-HOP-body BUILD mis-plans call-1's own computation. "
              "Fold-planner / subgraph-body lowering is the root cause.",
              flush=True)
    else:
        print(">>> `_1` artifact is CORRECT alone: the 14.61 bug is a >=2-call "
              "runtime replay effect, NOT the shared-body build. The 8x4 fold "
              "reshape is benign; reopen the runtime-replay lead.", flush=True)


if __name__ == "__main__":
    main()
