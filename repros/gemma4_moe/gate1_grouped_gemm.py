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
Gate 1: Gemma 4 MoE grouped/batched matmul on Spyre.

Tests that both Option 4A row-batched and dense expert-batched matmul geometries
compile on Spyre and produce numerically correct results matching CPU.

Outcome (2026-07-31):
---------------------
FAILED: Row-batched matmul [256, 1, 2816] x [256, 2816, 704] produces massive
numeric divergence (~12% mismatched elements, max diff 1.0, greatest relative
diff >1000x). Compilation completes without abort, but results are wrong.
Pattern: consistent offsets relative to CPU reference (got min/max slightly
higher), suggesting a systematic layout or computation issue rather than
overflow/rounding.

Root cause assessment:
  - Not a compile abort (no out_reuse_dim.size()==1 or layout error raised)
  - Not fp16 rounding (12% failures >> 1-2% fp16 noise threshold)
  - Likely a batched-gemm layout/tiling interaction with how Spyre maps
    batch dimensions to the accelerator's work distribution or memory layout

Recommendation: This gate BLOCKS the MoE design using Option 4A row-batched
geometry. Fall back to dense expert-compute-all + broadcast mask (less efficient,
but known to work). Alternatively, investigate whether spyre_hint() on batch
and output dims can fix the layout, or whether a reshape to [T*K*1, H] x [T*K*1, F]
(eliminate batch dim) + reshape-back would work.
"""
import os

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import torch_spyre
torch_spyre._autoload()

import torch


H, F, E, T, K = 2816, 704, 8, 32, 8  # small E/T to iterate fast; F=moe_intermediate_size


def row_batched(a, w):           # 4A geometry: one weight per row
    return torch.bmm(a, w)       # [N,1,H] x [N,H,F] -> [N,1,F]


def expert_batched(a, w):        # dense geometry: all experts
    return torch.bmm(a, w)       # [E,T,H] x [E,H,F] -> [E,T,F]


def check(fn, a_shape, w_shape):
    print(f"\nTesting {a_shape} x {w_shape}")
    a = torch.randn(*a_shape, dtype=torch.float16)
    w = torch.randn(*w_shape, dtype=torch.float16)

    # Reference on CPU
    ref = fn(a, w)
    print(f"  Ref shape: {ref.shape}, dtype: {ref.dtype}")

    # Compile and run on Spyre
    cfn = torch.compile(fn, dynamic=False)
    a_spyre = a.to("spyre")
    w_spyre = w.to("spyre")
    print(f"  Moved to spyre: a={a_spyre.device}, w={w_spyre.device}")
    got = cfn(a_spyre, w_spyre).cpu()
    print(f"  Got shape: {got.shape}, dtype: {got.dtype}")

    # Detailed comparison
    print(f"  ref min/max: {ref.min():.6f} / {ref.max():.6f}")
    print(f"  got min/max: {got.min():.6f} / {got.max():.6f}")

    try:
        torch.testing.assert_close(got, ref, atol=1e-2, rtol=1e-2)
        print(f"  OK {a_shape} x {w_shape}")
    except AssertionError as e:
        print(f"  FAILED: {e}")
        raise


if __name__ == "__main__":
    check(row_batched, (T * K, 1, H), (T * K, H, F))
    check(expert_batched, (E, T, H), (E, H, F))
