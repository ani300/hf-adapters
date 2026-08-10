# Copyright 2025 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Profile a decoupled prefill→decode loop on Spyre.

Runs BATCH_SIZE sequential batch=1 prefills at SEQ_LEN, stitches the per-
sequence KV caches into one batch=BATCH_SIZE cache, then runs a single batched
decode step against it. See
docs/superpowers/specs/2026-08-10-profile-prefill-decode-spyre-design.md.
"""

import argparse
import contextlib
import os
import sys
import time


def _apply_hf_home(argv):
    argv = sys.argv[1:] if argv is None else argv
    hf_home = None
    for i, tok in enumerate(argv):
        if tok == "--hf-home" and i + 1 < len(argv):
            hf_home = argv[i + 1]
        elif tok.startswith("--hf-home="):
            hf_home = tok.split("=", 1)[1]
    if hf_home:
        os.environ["HF_HOME"] = hf_home


_apply_hf_home(None)

import torch  # noqa: E402
from torch.profiler import ProfilerActivity, profile  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from hf_adapters.hf_common import (
    BLOCK_SIZE,
    DEVICE,
    allocate_kv_caches,
    build_expansion_mask,
    build_prefill_mask,
    generation_cache_len,
    get_model_dtype,
    pad_and_position,
    select_next_token,
)

MODELS = {
    "ministral3": ("mistralai/Ministral-3-14B-Instruct-2512", torch.bfloat16),
    "granite8b": ("ibm-granite/granite-3.3-8b-instruct", torch.float16),
}


@contextlib.contextmanager
def _noop_region(_name):
    yield


def build_prompts(base: str, batch_size: int) -> list[str]:
    """Return batch_size throwaway prompts distinct enough to be separate seqs."""
    return [f"{base} (variant {i})" for i in range(batch_size)]


def _encode_to_seq_len(tokenizer, prompt: str, seq_len: int) -> torch.Tensor:
    """Tokenize prompt and repeat/truncate to exactly seq_len real tokens -> [1, seq_len]."""
    ids = tokenizer(prompt, return_tensors="pt")["input_ids"][0]
    if ids.numel() == 0:
        ids = torch.tensor([tokenizer.eos_token_id or 0])
    reps = (seq_len + ids.numel() - 1) // ids.numel()
    ids = ids.repeat(reps)[:seq_len]
    return ids.unsqueeze(0)  # [1, seq_len]


def run_batched_prefill_then_decode(
    model, module, tokenizer, seq_len=512, batch_size=4, record=None
):
    """4x sequential bs=1 prefill -> stitch -> 1 batched decode. Returns a dict."""
    import time

    region = record or _noop_region
    dtype = get_model_dtype(model)
    max_cache_len = generation_cache_len(seq_len, 1)

    prompts = build_prompts("The capital of France is", batch_size)

    per_seq_keys = []      # list over seqs; each is list-over-layers of [1,n_kv,L,hd]
    per_seq_values = []
    first_tokens = []      # [1] each
    prompt_offsets_list = []
    decode_pos_list = []   # [1, BLOCK_SIZE] each
    padded_lens = []
    prefill_times = []

    for b in range(batch_size):
        ids = _encode_to_seq_len(tokenizer, prompts[b], seq_len)  # [1, seq_len]
        actual_lengths = torch.tensor([ids.shape[1]])
        padded_ids, padded_len, prompt_offsets, position_ids = pad_and_position(
            ids, actual_lengths
        )
        padded_lens.append(padded_len)
        key_caches, value_caches = allocate_kv_caches(model, 1, max_cache_len, dtype)
        prefill_mask = build_prefill_mask(1, padded_len, max_cache_len, prompt_offsets, dtype=dtype)

        t0 = time.time()
        with region(f"prefill_seq_{b}"):
            logits = module._run_forward(
                model,
                padded_ids.to(DEVICE),
                position_ids.to(DEVICE),
                prefill_mask.to(DEVICE),
                key_caches,
                value_caches,
                is_filling=False,
                token_index=0,
                cache_position=0,
            )
        prefill_times.append(time.time() - t0)

        next_logits = logits.to("cpu")[:, -1, :]                       # [1, vocab]
        first_tokens.append(select_next_token(next_logits, False, None, None, None))  # [1]
        per_seq_keys.append([kc.to("cpu") for kc in key_caches])
        per_seq_values.append([vc.to("cpu") for vc in value_caches])
        prompt_offsets_list.append(prompt_offsets)                     # [1]

        # Seed decode_pos exactly as hf_common.generate does at i==0:
        # decode_pos[0, j] = actual_len + j - BLOCK_SIZE
        actual_len = actual_lengths[0].item()
        dp = torch.zeros((1, BLOCK_SIZE), dtype=torch.long)
        for j in range(BLOCK_SIZE):
            dp[0, j] = actual_len + j - BLOCK_SIZE
        decode_pos_list.append(dp)

    assert len(set(padded_lens)) == 1, f"padded_len mismatch: {padded_lens}"
    padded_len = padded_lens[0]

    # --- Stitch: cat per-seq bs=1 caches along batch dim -> bs=batch_size ---
    t0 = time.time()
    with region("stitch"):
        n_layers = len(per_seq_keys[0])
        key_caches = [
            torch.cat([per_seq_keys[b][l] for b in range(batch_size)], dim=0).to(DEVICE)
            for l in range(n_layers)
        ]
        value_caches = [
            torch.cat([per_seq_values[b][l] for b in range(batch_size)], dim=0).to(DEVICE)
            for l in range(n_layers)
        ]
        prompt_offsets = torch.cat(prompt_offsets_list, dim=0)         # [batch_size]
        decode_pos = torch.cat(decode_pos_list, dim=0)                 # [batch_size, BLOCK_SIZE]
        # Decode input: slot 0 = each seq's first next-token, rest zero pad.
        next_input = torch.zeros((batch_size, BLOCK_SIZE), dtype=torch.long)
        next_input[:, 0] = torch.cat(first_tokens, dim=0)
    stitch_time = time.time() - t0

    # --- One batched decode (expansion) step ---
    current_cache_len = padded_len + BLOCK_SIZE
    decode_pos = decode_pos + BLOCK_SIZE
    exp_mask = build_expansion_mask(
        batch_size, BLOCK_SIZE, max_cache_len, current_cache_len, prompt_offsets, dtype=dtype
    )
    t0 = time.time()
    with region("decode_bs4"):
        logits = module._run_forward(
            model,
            next_input.to(DEVICE),
            decode_pos.to(DEVICE),
            exp_mask.to(DEVICE),
            key_caches,
            value_caches,
            is_filling=False,
            token_index=0,
            cache_position=padded_len,
        )
    decode_time = time.time() - t0

    next_logits = logits.to("cpu")[:, -BLOCK_SIZE, :]                  # [batch_size, vocab]
    next_tokens = select_next_token(next_logits, False, None, None, None)  # [batch_size]

    return {
        "next_tokens": next_tokens,
        "key_caches": key_caches,
        "value_caches": value_caches,
        "max_cache_len": max_cache_len,
        "timings": {
            "prefill": prefill_times,
            "stitch": stitch_time,
            "decode": decode_time,
        },
    }


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="granite8b", choices=sorted(MODELS))
    p.add_argument("--out", default=None,
                   help="Chrome-trace output path (default: <model>_prefill_decode_trace.json).")
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--hf-home", default=os.environ.get("HF_HOME"))
    p.add_argument("--with-stack", action="store_true",
                   help="Record per-event Python stack + module hierarchy (slower, larger trace).")
    return p


def run_profile(model_path, dtype, seq_len, batch_size, out_path, with_stack=False):
    from hf_adapters import AutoSpyreModelForCausalLM
    from hf_adapters.auto_spyre_model import resolve_adapter_module

    print("=" * 70)
    print(f"  profiling prefill→decode  {model_path}  (dtype={dtype})")
    print(f"  seq_len={seq_len} batch_size={batch_size}")
    print("=" * 70)

    t0 = time.time()
    model = AutoSpyreModelForCausalLM.from_pretrained(model_path, dtype=dtype)
    module = resolve_adapter_module(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print(f"  Load time: {time.time() - t0:.1f}s")

    # Warmup (not profiled): triggers torch.compile so the profiled pass
    # measures execution, not compile. All prefills share cache_position=0 →
    # one binary; the decode step is a distinct shape/position → its own binary.
    print("\n[warmup] compiling (this run is not profiled)...")
    run_batched_prefill_then_decode(model, module, tokenizer, seq_len, batch_size)

    print(f"[profile] capturing trace... (with_stack={with_stack})")
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1],
        record_shapes=True,
        with_stack=with_stack,
        with_modules=with_stack,
    ) as prof:
        out = run_batched_prefill_then_decode(
            model, module, tokenizer, seq_len, batch_size,
            record=torch.profiler.record_function,
        )
    prof.export_chrome_trace(out_path)  # do NOT call prof.events()/key_averages() (#114)

    t = out["timings"]
    print(f"\n  next_tokens: {out['next_tokens'].tolist()}")
    print(f"  prefill times (ms): {[f'{x*1000:.1f}' for x in t['prefill']]}")
    print(f"  stitch time (ms):   {t['stitch']*1000:.1f}")
    print(f"  decode time (ms):   {t['decode']*1000:.1f}")
    print(f"  trace → {out_path}")
    print("  open in https://ui.perfetto.dev/ or chrome://tracing")


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.hf_home:
        os.environ["HF_HOME"] = args.hf_home
    from huggingface_hub import constants
    print(f"  HF hub cache: {constants.HF_HUB_CACHE}")
    path, dtype = MODELS[args.model]
    out = args.out or f"{args.model}_prefill_decode_trace.json"
    run_profile(path, dtype, args.seq_len, args.batch_size, out, args.with_stack)


if __name__ == "__main__":
    main()
