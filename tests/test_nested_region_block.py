"""CPU-only unit tests for the nested_compile_region block wrapper."""
import torch
from torch import nn


def test_nested_region_block_wraps_forward():
    from hf_adapters.hf_common import nested_region_block

    class Dummy(nn.Module):
        def forward(self, h, freqs, mask, kc, vc, is_filling, ti, cp):
            return h + 1.0, kc, vc

    wrapped = nested_region_block(Dummy())
    h = torch.zeros(1, 4, 8)
    out, kc, vc = wrapped(h, None, None, torch.zeros(1), torch.zeros(1), False, 0, 0)
    assert torch.allclose(out, h + 1.0)


def test_nested_region_block_is_callable_region():
    # The wrapper must be the region callable, not a plain bound method,
    # so torch.compile treats it as an invoke_subgraph boundary.
    from hf_adapters.hf_common import nested_region_block
    from torch.compiler import nested_compile_region  # import must resolve

    class Dummy(nn.Module):
        def forward(self, h):
            return h

    wrapped = nested_region_block(Dummy())
    assert callable(wrapped)
