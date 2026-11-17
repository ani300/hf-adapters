# RFC 48 sparse MoE experiment

Set `SPYRE_MOE_IMPL=sparse_prefill` before preparing/loading the model to select
the experimental chunked prefill in the Gemma4 and OLMoE adapters. Set it to
`sparse` to also select whole-batch decode over the union of selected experts.
The default, `current`, selects the existing implementation.

The implementation is in `hf_adapters/sparse_moe.py`. It follows
[RFC 48](https://github.com/torch-spyre/RFCs/pull/48), using
`torch.ops.spyre.non_zero_pos_to_idx(x, dim)` and
`torch.ops.spyre.count_non_zero_round_up(x, dim, g)` directly. It does not register
replacement kernels. These ops and device-value loop counts still need support
in torch-spyre and the backend; this path is for compiler bring-up.

Prefill uses a maximum chunk size of `512 * ceil(E/K)`, which is 8192 at E=128,
K=8. Each chunk is padded to 512-token granularity, packs weights and int32 token
ids per expert, and appends a zero input row and a sink output row. Chunk,
expert, and token-tile loops use `for_each_tile`. The inner trip count comes
from the FP32 count tensor, exposed as a symbolic size through `item`,
`torch.sym_int`, and bounded `narrow` operations. Output accumulation is
functional gather/add/scatter to satisfy the loop's no-alias contract.

The adapters retain their activation, router precision, normalization, and
expert-scale semantics. Decode selection uses score magnitude because Gemma4
scales may be signed. Dummy stick dimensions are explicitly expanded to 32
elements; layout conversion, unstaggering, and index-to-address conversion are
left to the backend. `aten.scatter` requires logical int64 indices, which Spyre
stores as int32, so the FP32 scan positions are cast at that boundary.

TP continues to slice each expert's intermediate dimension and reduce routed
partials outside these functions. Gemma4's dense and routed branches have
separate output RMS normalizations, so their reductions cannot be replaced by
one reduction of their combined partials before those normalizations. The
existing model-specific collective placement is retained. OLMoE does not gain
a new TP implementation from this experiment.

`tests/cpu/test_sparse_moe_rfc48.py` checks packing, empty/skewed experts, tails,
multiple chunks, 512-token granularity, both activations, signed/zero scales,
and 4-/8-way intermediate-dimension partial sums. These tests explicitly use
diagnostic CPU implementations of the proposed ops and a CPU executor for
tile semantics from `tests/_sparse_moe_reference.py`; they do not validate
Inductor or device execution. No test helpers are imported by model code.

On the development workspace, the detailed failure report and standalone
stage probe are under `/mnt/devel/inductor_src/analysis/moe-rfc48/`. The lower
stack proposal is [Deeptools PR 4986](https://github.ibm.com/ai-chip-toolchain/deeptools/pull/4986).
