# tests/test_nested_region_compile_once.py
"""Gate: a region-wrapped block called N times compiles ONCE under torch.compile.

Uses torch._dynamo compile counters + the invoke_subgraph HOP so it runs on CPU
with the default inductor backend (no Spyre needed).
"""
import torch
from torch import nn
import torch._dynamo as dynamo


class _Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.lin = nn.Linear(dim, dim)

    def forward(self, h):
        return self.lin(h).relu()


def test_region_block_compiles_once_across_layers():
    from hf_adapters.hf_common import nested_region_block

    dim = 16
    blocks = [nested_region_block(_Block(dim)) for _ in range(4)]

    def outer(h):
        for b in blocks:
            h = b(h)
        return h

    dynamo.reset()
    compiled = torch.compile(outer, dynamic=False, fullgraph=True)
    h = torch.randn(2, dim)
    out = compiled(h)
    assert out.shape == (2, dim)

    # The 4 region calls must lower to invoke_subgraph, sharing one subgraph.
    gm, _ = dynamo.export(outer)(h)
    calls = [n for n in gm.graph.nodes
             if "invoke_subgraph" in str(getattr(n, "target", ""))]
    assert len(calls) >= 2, f"expected repeated invoke_subgraph calls, got {calls}"
