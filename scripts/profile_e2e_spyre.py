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

"""Profile the Ministral 3 14B end-to-end generation loop on Spyre.

Runs the same load + generate path as the e2e smoke test twice:

  1. A warmup run (no profiler) that triggers torch.compile so the Spyre
     Inductor backend codegens and caches every kernel the workload hits.
  2. A profiled run under torch.profiler with CPU + PrivateUse1 (Spyre-side)
     activities. Because the cache is warm, this measures execution, not
     compile time. The trace is written to a single Chrome-trace JSON.

Usage (on the Spyre pod, with the project root on PYTHONPATH)::

    python scripts/profile_e2e_spyre.py --model ministral3

Open the resulting trace in https://ui.perfetto.dev/ or chrome://tracing.
"""

import argparse
import time

import torch
from torch.profiler import ProfilerActivity, profile
from transformers import AutoTokenizer

from hf_adapters import AutoSpyreModelForCausalLM

try:
    import torch_spyre  # noqa: F401  (registers the "spyre" device)
except ImportError:
    # Allowed only off-pod for --help / arg parsing; the profiled run
    # (Task 2) needs the real device and will fail loudly without it.
    torch_spyre = None

# Registry key -> (HF path, Spyre-safe dtype). Kept inline so this script has
# no dependency on the tests/ package. Ministral 3 uses bfloat16 (see
# hf_adapters.auto_spyre_model.MODEL_PATH_TO_TORCH_DTYPE).
MODELS: dict[str, tuple[str, "torch.dtype"]] = {
    "ministral3": ("mistralai/Ministral-3-14B-Instruct-2512", torch.bfloat16),
}

DEFAULT_PROMPT = "The capital of France is"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="ministral3",
        choices=sorted(MODELS),
        help="Registry key to profile (default: ministral3).",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Chrome-trace output path (default: <model>_trace.json).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=5,
        help="New-token budget (default: 5, matching the smoke test).",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help=f"Prompt to generate from (default: {DEFAULT_PROMPT!r}).",
    )
    return parser


def run_profile(
    model_path: str,
    dtype: "torch.dtype",
    prompt: str,
    max_new_tokens: int,
    out_path: str,
) -> None:
    """Load *model_path*, warm the compile cache, then profile one generate."""
    print(f"{'=' * 70}")
    print(f"  profiling {model_path}  (dtype={dtype})")
    print(f"{'=' * 70}")

    t0 = time.time()
    model = AutoSpyreModelForCausalLM.from_pretrained(model_path, dtype=dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print(f"  Load time: {time.time() - t0:.1f}s")
    print(f"  Prompt: {prompt!r}")

    gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=False, timing=True)

    # --- Warmup: first generate triggers torch.compile; NOT profiled. This
    #     is what "warms the compiler cache" so the profiled run below measures
    #     steady-state execution rather than compile time.
    print("\n[warmup] compiling (this run is not profiled)...")
    model.generate(tokenizer, [prompt], **gen_kwargs)

    # --- Profiled run: cache is warm. CPU + PrivateUse1 (Spyre device) activity.
    #     NOTE(#114): do NOT call prof.events() / prof.key_averages().table() —
    #     reading the event buffer hits the libaiupti/kineto ABI decode crash.
    #     export_chrome_trace() writes the trace without walking that buffer.
    print("[profile] capturing trace...")
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1],
        record_shapes=True,
    ) as prof:
        outputs = model.generate(tokenizer, [prompt], **gen_kwargs)

    prof.export_chrome_trace(out_path)

    output_text = outputs[0] if outputs else ""
    print(f"\n  Output: {output_text!r}")
    print(f"  trace → {out_path}")
    print("  open in https://ui.perfetto.dev/ or chrome://tracing")


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    path, dtype = MODELS[args.model]
    out = args.out or f"{args.model}_trace.json"
    run_profile(
        model_path=path,
        dtype=dtype,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        out_path=out,
    )


if __name__ == "__main__":
    main()
