# Gemma-4 MoE — single shared expert-weight layout (design)

**Supersedes:** the weight-materialization half of
`docs/superpowers/specs/2026-08-20-gemma4-moe-phase-dispatch-design.md`
(the seq_len phase-dispatch design). That spec's *dispatch* decision stands
(prefill → persistent, decode → loop, chosen per-forward by `seq_len`); this
spec replaces its *weight-materialization* decision, which materialized two
independent device weight sets and OOM'd the card.

## Problem

The phase-dispatch design has `prepare_for_spyre` build BOTH expert-weight
layouts on every one of the 30 MoE layers:

- persistent (prefill): `_spyre_persistent_gate/up/down` — a `[K,E,N]`
  contiguous backing exposing logical `[E,K,N]` views;
- loop-on-topk (decode): `_spyre_gate_up_dev` `[E,H,2M]` + `_spyre_down_dev`
  `[E,M,H]` — E-outermost HBM-resident stacks.

For `google/gemma-4-26B-A4B-it` (H=2816, E=128, M=704, 30 layers, fp16) one
weight set is ~42.5 GiB; both together are ~85 GiB. On a ~96 GiB card that
leaves no room for attention weights, embeddings, KV cache, and the router —
the generic `model.to("spyre")` sweep OOMs (`0x340f FlexAllocator
OutOfMemory`) before the first forward runs. This is deterministic, not
transient.

The two sets hold the *same* logical weights; only their device layout
differs. The fix is to store **one** physical device layout per expert weight
that BOTH paths read.

## Constraints (hard, from the user)

1. **Expert dim outermost on device.** The decode path gathers per
   `(token, expert)` (`gate_dev[idx_addr]`) and the persistent path streams one
   expert bank per tile; both require E at device position 0.
2. **Matmul stick rule** (torch-spyre `tensors_and_layouts.md`, `C[m,n] =
   A[m,k] @ B[k,n]`): the **activation** `A` sticks on the **contraction** dim
   `k`; the **weight** `B` sticks on its **free/output** dim `n`, with `k`
   padded to whole sticks. So each expert weight sticks on its *free* dim.
3. **Logical shape unchanged.** PyTorch `size()`/`stride()` stay the logical
   `[E,·,·]`; only the *device* layout (`device_tensor_layout()`) carries the
   E-outer + free-on-stick arrangement. Use the device-dims
   `SpyreTensorLayout(device_size, stride_map, device_dtype)` overload, never a
   host reshape.
4. **Apply at model load time.** The layout is set when weights move to the
   device, in torch-spyre `model_utils.py`, invoked from the adapter's
   `prepare_for_spyre`.
5. **Split the fused `gate_up`** into separate `gate` and `up` weights, so the
   decode path drops `[E,H,2M]` + `chunk(2)` and gathers `gate`/`up`
   independently — making decode's operands identical to prefill's.

## The single layout

fp16 ⇒ `eps` (elems_per_stick) = 64. M = 704 → 11 stick-tiles; H = 2816 → 44
stick-tiles. Both divide cleanly.

| Weight | Logical | Matmul it feeds | contract / free | Device layout (E outer, free on stick) |
|---|---|---|---|---|
| `gate` | `[E,H,M]` | `x[·,T,H] @ gate → [E,T,M]` | H / **M** | `device_size=[E, H, M//eps, eps]`, `stride_map=[H*M, M, eps, 1]` |
| `up`   | `[E,H,M]` | `x[·,T,H] @ up   → [E,T,M]` | H / **M** | same as `gate` |
| `down` | `[E,M,H]` | `act[E,T,M] @ down → [E,T,H]` | M / **H** | `device_size=[E, M, H//eps, eps]`, `stride_map=[M*H, H, eps, 1]` |

Template: the existing `_dma_to_spyre_indirect_access` (embedding
`device_size=[rows, d//eps, eps]`, `stride_map=[d, eps, 1]`), generalized to a
leading E dim and a 3-D logical tensor: the free dim is the one split into
`(free//eps, eps)` sticks, the contraction dim sits between E and the split
free dim, and E is the outermost device dim.

Footprint: one copy ≈ 42.5 GiB / 30 layers. Fits, with headroom for the rest
of the model. Solves the OOM by construction.

### Why free-on-stick + E-outer is consistent for both paths

- **Prefill (persistent, hint-body):** `torch.matmul(x[1,T,H], gate[E,H,M]) →
  [E,T,M]` under `spyre_hint(num_tiles_per_dim={"E":128})`. E outermost = one
  expert bank streamed per loop trip; `gate` sticked on its free dim M is the
  correct weight-operand layout for that matmul. `down[E,M,H]` sticked on free
  H likewise.
- **Decode (loop-on-topk):** `gate_dev[idx_addr] → [T,K,H,M]` is a gather with
  the indexed (E) dim outermost — the indirect-access requirement — and the
  gathered `[H,M]` slab is already the correct weight layout (sticked on free
  M) for the per-row bmm `x[T,K,1,H] @ W[T,K,H,M] → [T,K,1,M]`. `down` gather
  → `[T,K,M,H]`, bmm contracts M, free H. No restickify.

The same physical bytes serve both because "E outermost + free dim on stick"
is simultaneously the gather-source layout (constraint 1) and the matmul
weight layout (constraint 2).

## Design

### A. torch-spyre — load-time layout applier

**File:** `torch_spyre/torch_spyre/model_utils.py`.

Add a public helper that stickifies a rank-3 `[E, C, F]` expert weight with E
outermost and the free dim `F` on the stick, returning the device tensor whose
logical shape/stride is unchanged:

```python
def dma_moe_expert_weight_to_spyre(weight, target_dtype=None):
    """Transfer a rank-3 [E, C, F] MoE expert weight to Spyre, E-outermost
    with the free/output dim F split into sticks (matmul weight-operand layout
    that is also gather-optimal along E).

    device_size = [E, C, F // eps, eps]; stride_map = [C*F, F, eps, 1].
    Requires F % eps == 0; warns + returns None (caller falls back) otherwise.
    """
```

Mirror `_dma_to_spyre_indirect_access`: `contiguous()` the host tensor, query
`eps` via `SpyreTensorLayout(...).elems_per_stick()`, guard `F % eps == 0`
(warn + `None` on failure), build the 3-arg device-dims layout with
`get_device_dtype(dev_dtype)`, `spyre_empty_with_layout(weight.size(),
weight.stride(), dev_dtype, layout)`, `copy_tensor`. Export it (module `__all__`
if present) so the adapter can import it.

The generic `_transfer_module` sweep is **not** changed — MoE expert weights
are plain attrs set explicitly in `prepare_for_spyre`, not `nn.Linear`/
`nn.Embedding` params, so they never reach the sweep. (Confirmed:
`_transfer_module` applies indirect-access only to `nn.Embedding`.)

Python-only; no C++ rebuild.

### B. hf-adapters — one weight set, applied at load

**File:** `hf_adapters/hf_gemma4_moe.py`, `prepare_for_spyre` (weight block
~678-712).

De-fuse and materialize exactly three device tensors per block via the new
applier:

```python
gate_up_t = experts.gate_up_proj.data.transpose(1, 2).contiguous()  # [E,H,2M]
down_t   = experts.down_proj.data.transpose(1, 2).contiguous()      # [E,M,H]
M = gate_up_t.shape[2] // 2
gate_l = gate_up_t[:, :, :M].contiguous()  # [E,H,M]
up_l   = gate_up_t[:, :, M:].contiguous()  # [E,H,M]
block._spyre_gate = dma_moe_expert_weight_to_spyre(gate_l)   # [E,H,M], E-outer, M on stick
block._spyre_up   = dma_moe_expert_weight_to_spyre(up_l)     # [E,H,M]
block._spyre_down = dma_moe_expert_weight_to_spyre(down_t)   # [E,M,H], E-outer, H on stick
```

Delete `_spyre_persistent_gate/up/down`, `_spyre_persistent_route_identity`,
`_spyre_gate_up_dev`, `_spyre_down_dev`, and the CPU-resident
`_spyre_gate_up_t`/`_spyre_down_t` intermediates. Keep the one-time router
device-move.

### C. hf-adapters — persistent path reads the shared set (hint-body)

**File:** `_moe_expert_persistent` (~346), `_moe_ffn_persistent` (~359),
`forward` (~541).

Restore the hint-body matmul persistent from `git stash@{0}` — declare
dims, `name_tensor_dims`, `spyre_hint(num_tiles_per_dim={"E":experts},
work_div={"T":32})`, three matmuls + gelu-tanh SwiGLU + route-weighted sum —
but **drop** the stash's `_pack_persistent_expert_weight` `[K,E,N]` packing;
the body now takes the three already-laid-out device tensors `gate`/`up`/`down`
directly (they are already E-outermost + free-on-stick from the applier). Drop
the `torch.ops.spyre.moe_ffn.default` call entirely (op absent on
`pr3892-moe`). `forward`'s persistent call passes `block._spyre_gate`,
`block._spyre_up`, `block._spyre_down`.

The persistent route weight: keep the stash's dense `[E,T,1]` route
materialization (`.permute(1,0,2).contiguous().clone()`), which the coarse-tile
read-copy needs; that is orthogonal to the expert-weight layout.

### D. hf-adapters — decode path reads the shared set (split gather)

**File:** `_compiled_moe_loop_region` (~113), `_moe_ffn_loop` (~200), `forward`
(~554).

Replace the fused gather + chunk:

```python
W_gu = gate_up_dev[idx_addr]        # [T,K,H,2M]
gu = torch.matmul(gathered.unsqueeze(-2), W_gu)  # [T,K,1,2M]
g, u = gu.chunk(2, dim=-1)          # [T,K,1,M]
```

with the split gather (two separate index_selects on the shared tensors):

```python
W_g = gate_dev[idx_addr]            # [T,K,H,M]
W_u = up_dev[idx_addr]              # [T,K,H,M]
g = torch.matmul(gathered.unsqueeze(-2), W_g)  # [T,K,1,M]
u = torch.matmul(gathered.unsqueeze(-2), W_u)  # [T,K,1,M]
```

`act = F.gelu(g, approximate="tanh") * u`; `W_dn = down_dev[idx_addr]  #
[T,K,M,H]`; `row_out = matmul(act, W_dn).squeeze(-2)` unchanged. Rename the
region/`_moe_ffn_loop` params `gate_up_dev → gate_dev, up_dev` (two tensors
now). `forward`'s decode call passes `block._spyre_gate`, `block._spyre_up`,
`block._spyre_down`.

## Verification

Card-gated (needs the Spyre card); no host-only unit test possible for the
device layout. Import-parse + on-card prefill A/B is the gate, consistent with
the prior plan's inherent card-gating.

1. **Import/parse:** `python -c "import torch, torch_spyre; import
   hf_adapters.hf_gemma4_moe"` → `IMPORT_OK`; the new applier imports from
   `model_utils`.
2. **Load fits (the OOM gate):** model loads to the card without `0x340f`.
   With one ~42.5 GiB set + the rest of the model, the load completes.
3. **On-card prefill (persistent):** `repros/gemma4_moe/ab_persistent_vs_chunked.py
   persistent` (env `HF_HUB_CACHE=/mnt/models/hf_cache/hub HF_HUB_OFFLINE=1`,
   PREFILL_TOKENS=512, NEW_TOKENS=1) compiles and emits `' the'` at ≈7.3 s warm
   (the pre-refactor persistent number). `_untracked_*` named-dim warnings are
   expected and non-fatal.
4. **On-card device-layout sanity:** after load, `device_tensor_layout()` on
   `block._spyre_gate` reports `device_size` with E outermost and a trailing
   `eps` stick dim (not the logical `[E,H,M]`), confirming the layout applied
   rather than falling back to default.

## Notes / risks

- Decode (loop-on-topk) at E=128 depends on the incoming torch-spyre compiler
  batch (index consumption in `[T,K]`, fp32→int32 topk-index cast op,
  `index_mask`); those land per the phase-dispatch plan's Tasks 5-6 and the
  `temporal-stargazing-taco` plan. This spec does not require them for the
  prefill (persistent) gate — it makes decode *read the right tensors* so that
  when the compiler batch lands, no weight-layout rework is needed.
- torch-spyre change is Python-only (`model_utils.py`) → no C++ rebuild.
- The adapter-only Global Constraint from the phase-dispatch plan is relaxed by
  explicit user directive ("apply this SpyreTensorLayout at model load time")
  to allow the `model_utils.py` applier. Still no GitHub push; commits are
  local working-tree checkpoints.
- Clear `/tmp/torchinductor_aviros` with `mv` (never `rm -rf`) after the
  torch-spyre change before the on-card run.
