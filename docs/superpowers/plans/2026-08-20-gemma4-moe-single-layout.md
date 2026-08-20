# Gemma-4 MoE single shared expert-weight layout — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the OOM-inducing dual expert-weight materialization with ONE shared E-outermost, free-dim-on-stick device layout per expert weight that both the prefill (persistent hint-body) and decode (loop-on-topk) MoE paths read.

**Architecture:** A new torch-spyre `model_utils.py` applier stickifies rank-3 `[E,C,F]` MoE expert weights with E outermost and the free dim `F` split into sticks (device-dims `SpyreTensorLayout`, logical shape unchanged). The gemma4 adapter de-fuses `gate_up` into separate `gate`/`up`, materializes exactly three device tensors via that applier in `prepare_for_spyre`, and both FFN paths read those three: the persistent path via restored hint-body matmuls (dropping the absent `moe_ffn` op), the decode path via a split gather (dropping the fused `[E,H,2M]` + `chunk(2)`).

**Tech Stack:** PyTorch OOT PrivateUse1 backend (torch-spyre), HuggingFace Transformers runtime monkey-patch adapter (hf-adapters), IBM Spyre accelerator, fp16.

**Spec:** `docs/superpowers/specs/2026-08-20-gemma4-moe-single-layout-design.md`

## Global Constraints

- **Two repos:** torch-spyre (branch `pr3892-moe`, Python-only change → no C++ rebuild) and hf-adapters (branch `gemma4-moe-persistent-moe-ffn`). Each commits in its own repo.
- **Single physical weight set:** exactly three device tensors per block — `block._spyre_gate` `[E,H,M]`, `block._spyre_up` `[E,H,M]`, `block._spyre_down` `[E,M,H]` — read by BOTH paths. No `_spyre_persistent_*`, no `_spyre_gate_up_dev`/`_spyre_down_dev`, no CPU `_spyre_gate_up_t`/`_spyre_down_t`.
- **Layout invariants:** E outermost on device (constraint 1); weight sticked on its free/output dim (constraint 2: `gate`/`up` free = M, `down` free = H); logical PyTorch `size()`/`stride()` unchanged — layout lives only in `device_tensor_layout()` (constraint 3, device-dims `SpyreTensorLayout(device_size, stride_map, get_device_dtype(dtype))` overload).
- **fp16:** `eps` (elems_per_stick) = 64. M=704→11 stick-tiles, H=2816→44 stick-tiles; both divide cleanly (guard `F % eps == 0` anyway, warn+return None on failure).
- **Persistent body = hint-body matmuls** (from `git stash@{0}`): `declare_tensor_dim`/`name_tensor_dims` + `spyre_hint(num_tiles_per_dim={"E":E}, work_div={"T":32})` + three matmuls + gelu-tanh SwiGLU + route-weighted sum. NO `torch.ops.spyre.moe_ffn.default` (absent on `pr3892-moe`).
- **KEEP already-correct working-tree edits:** the seq_len phase-dispatch in `forward` (`if seq_len > 1:` @534) and the collapsed mode-flag globals/assert stay. This plan changes only the weight materialization, the persistent body, and the decode gather — plus the call-site argument lists in `forward`.
- **import torch before import torch_spyre.**
- **Commits:** DCO sign-off (`git commit -s`); LOCAL working-tree checkpoints, NOT pushed to GitHub. No force-push.
- **Recursive-delete guard:** never `rm -rf`; clear `/tmp/torchinductor_aviros` with `mv`.
- **On-card A/B:** host `aviros-spyre-test`; env `HF_HUB_CACHE=/mnt/models/hf_cache/hub HF_HUB_OFFLINE=1`; model `google/gemma-4-26B-A4B-it`; PREFILL_TOKENS=512, NEW_TOKENS=1; harness `repros/gemma4_moe/ab_persistent_vs_chunked.py persistent`. `_untracked_*` named-dim warnings are expected and non-fatal.
- **Verification is card-gated:** no host-only unit test can exercise the device layout. Per-task verification is import-parse + grep + (final task) on-card prefill. This is inherent card-gating, NOT a test-hygiene defect — reviewers must not flag "missing tests".

---

## File Structure

- `torch-spyre/torch_spyre/model_utils.py` — add `dma_moe_expert_weight_to_spyre` applier (Task 1). One clear responsibility: transfer a rank-3 MoE expert weight to the device with the shared layout.
- `hf-adapters/hf_adapters/hf_gemma4_moe.py` — de-fuse + materialize one set in `prepare_for_spyre` (Task 3); restore hint-body persistent (Task 2); split decode gather (Task 4); rewire `forward` call sites (folded into Tasks 2 & 4). Single adapter file, already the project's pattern.

Task order rationale: Task 1 (torch-spyre applier) is the dependency both adapter paths consume, so it goes first and is independently importable. Task 2 (persistent body) and Task 4 (decode gather) each rewrite one FFN path + its `forward` call site. Task 3 (materialization) produces the three tensors both consume — but it is written against the applier (Task 1) and the tensor names are fixed by this plan, so it can land before or after 2/4; sequenced after Task 1 and before the on-card gate so the names exist when 2/4 wire to them. The on-card prefill gate lives in the LAST adapter task (Task 4) so it runs against the fully-wired tree.

---

### Task 1: torch-spyre — `dma_moe_expert_weight_to_spyre` applier

**Files:**
- Modify: `torch-spyre/torch_spyre/model_utils.py` (add function after `_dma_to_spyre_indirect_access`, ~line 200)

**Interfaces:**
- Consumes: existing module-level `SpyreTensorLayout`, `spyre_empty_with_layout`, `copy_tensor`, `get_device_dtype`, `warnings` (all already imported in this file — confirm by grep before editing).
- Produces: `dma_moe_expert_weight_to_spyre(weight: torch.Tensor, target_dtype: torch.dtype | None = None) -> torch.Tensor | None` — a public helper (no leading underscore) the gemma4 adapter imports as `from torch_spyre.model_utils import dma_moe_expert_weight_to_spyre`.

- [ ] **Step 1: Add the applier function**

Insert after `_dma_to_spyre_indirect_access` (which ends ~line 199, before the `# --- Model loading ---` banner). It generalizes that helper from rank-2 `[rows, d]` to rank-3 `[E, C, F]`: E outermost, contraction `C` un-tiled in the middle, free `F` split into `(F//eps, eps)` sticks.

```python
def dma_moe_expert_weight_to_spyre(
    weight: torch.Tensor,
    target_dtype: torch.dtype | None = None,
) -> torch.Tensor | None:
    """Transfer a rank-3 ``[E, C, F]`` MoE expert weight to Spyre.

    The expert dim ``E`` is placed outermost on the device and the free/output
    dim ``F`` is split into stick-sized blocks, giving device dims
    ``[E, C, F // eps, eps]`` where ``eps`` is the elements-per-stick for the
    device dtype. This one layout serves both MoE paths:

      * the decode gather ``weight[idx]`` needs the indexed (expert) dim
        outermost -- the "indirect access" requirement; and
      * the matmul weight operand ``A[m,k] @ B[k,n]`` needs the weight sticked
        on its free dim ``n`` (here ``F``), with the contraction dim ``C``
        un-split in the middle.

    ``gate``/``up`` are logical ``[E, H, M]`` (contract H, free M); ``down`` is
    logical ``[E, M, H]`` (contract M, free H). The logical PyTorch
    shape/stride is unchanged; only the device layout carries this arrangement.

    Uses the 3-arg device-dims ``SpyreTensorLayout`` overload with the *device*
    dtype (``get_device_dtype``).

    Requires ``F % eps == 0``; otherwise the sticks can't tile the free dim, so
    we warn and return ``None`` to signal the caller to fall back to a default
    move (which still loads and runs, just without the shared-layout
    optimization).

    Caller must ensure ``weight.ndim == 3``.
    """
    assert weight.ndim == 3, "MoE expert-weight path is for rank-3 [E,C,F] only"

    if not weight.is_contiguous():
        weight = weight.contiguous()
    dev_dtype = target_dtype if target_dtype is not None else weight.dtype

    experts, contract, free = weight.shape
    # elems_per_stick is dtype-aware (64 at fp16/bf16, 32 at fp32), so query it
    # rather than hardcoding a stick size.
    eps = SpyreTensorLayout(list(weight.shape), dev_dtype).elems_per_stick()
    if free % eps != 0:
        warnings.warn(
            f"MoE expert-weight free dim {free} is not a multiple of the Spyre "
            f"stick size {eps} for dtype {dev_dtype}; falling back to the "
            "default layout (no shared-layout optimization) for this weight.",
            stacklevel=2,
        )
        return None

    layout = SpyreTensorLayout(
        [experts, contract, free // eps, eps],  # device_size: E outermost
        [contract * free, free, eps, 1],  # stride_map
        get_device_dtype(dev_dtype),
    )
    dst = spyre_empty_with_layout(weight.size(), weight.stride(), dev_dtype, layout)
    copy_tensor(weight, dst, non_blocking=False)
    return dst
```

- [ ] **Step 2: Export it if the module has an `__all__`**

Run: `grep -n "__all__" torch-spyre/torch_spyre/model_utils.py`
- If `__all__` exists, add `"dma_moe_expert_weight_to_spyre"` to it.
- If it does not exist, do nothing (public-by-default; the name has no leading underscore).

- [ ] **Step 3: Confirm the imports the function needs are present**

Run:
```bash
cd /mnt/devel/inductor_src/torch-spyre
grep -n "^import warnings\|spyre_empty_with_layout\|^from.*copy_tensor\|def get_device_dtype\|SpyreTensorLayout" torch_spyre/model_utils.py | head
```
Expected: `warnings`, `spyre_empty_with_layout`, `copy_tensor`, `get_device_dtype`, and `SpyreTensorLayout` all already referenced in this file (the sibling `_dma_to_spyre_*` helpers use every one). If `warnings` is somehow not imported at module top, add `import warnings` at the top with the other stdlib imports.

- [ ] **Step 4: Verify the module imports**

Run:
```bash
cd /mnt/devel/inductor_src/torch-spyre
python -c "import torch, torch_spyre; from torch_spyre.model_utils import dma_moe_expert_weight_to_spyre; print('APPLIER_OK')"
```
Expected: `APPLIER_OK`. (Import-only; no device needed. torch-spyre is enabled by `import torch` then `import torch_spyre`.)

- [ ] **Step 5: Commit (torch-spyre)**

```bash
cd /mnt/devel/inductor_src/torch-spyre
git add torch_spyre/model_utils.py
git commit -s -m "feat(model_utils): dma_moe_expert_weight_to_spyre shared MoE layout

Rank-3 [E,C,F] expert-weight applier: expert dim outermost on device, free
dim F split into sticks ([E,C,F//eps,eps]). One layout serves both the decode
gather (indexed expert dim outermost) and the matmul weight operand (sticked
on free dim). Logical shape/stride unchanged. Python-only.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: hf-adapters — restore hint-body persistent path (drop `moe_ffn` op)

**Files:**
- Modify: `hf_adapters/hf_gemma4_moe.py` — `_moe_expert_persistent` (~346-356), `_moe_ffn_persistent` (~359-404 the `spyre_config.patch` block), `forward` persistent call site (~535-547)

**Interfaces:**
- Consumes: `spyre_hint` from `torch_spyre._inductor.propagate_hints`; `declare_tensor_dim`, `name_tensor_dims` from `torch_spyre._inductor.wsr.propagate_named_dims`; `F` (torch.nn.functional, already imported). The three device tensors `block._spyre_gate`/`_up`/`_down` produced by Task 3 (logical `[E,H,M]`/`[E,H,M]`/`[E,M,H]`).
- Produces: `_moe_expert_persistent(x_expert, routing_weight, gate, up, down, K)` returns `[T,H]` via hinted matmuls, no custom op.

- [ ] **Step 1: Replace `_moe_expert_persistent` with the hint-body**

Replace the entire current body (@346-356, the `torch.ops.spyre.moe_ffn.default(...)` call) with the hint-body from `git stash@{0}` (the `_moe_expert_persistent` there), WITHOUT the stash's separate `_pack_persistent_expert_weight` helper (the weights arrive already laid out by Task 3):

```python
def _moe_expert_persistent(x_expert, routing_weight, gate, up, down, K):
    """Run the dense expert body as one hinted, ordinary PyTorch program.

    The dims are declared and each operand is named, then the matmuls run under
    a coarse-tile ``spyre_hint(num_tiles_per_dim={"E": E})`` (one tile per
    expert): coarse-tiling emits one expert body inside one counted device loop
    -- stage ``[T,H]`` once, advance gate/up/down/route per expert, keep the
    running sum in the LX accumulator, drain once. No custom op.

    Shapes: ``x_expert`` [T,H]; ``gate``/``up`` logical [E,H,M]; ``down`` logical
    [E,M,H]; ``routing_weight`` [T,E,1]. gate/up/down arrive already in the
    shared device layout (E outermost, free dim on stick) from prepare_for_spyre.
    """
    from torch_spyre._inductor.propagate_hints import spyre_hint
    from torch_spyre._inductor.wsr.propagate_named_dims import (
        declare_tensor_dim,
        name_tensor_dims,
    )

    experts, hidden, inter = gate.shape
    tokens = x_expert.shape[0]
    for name, extent in (
        ("E", experts),
        ("T", tokens),
        ("H", hidden),
        ("M", inter),
        ("ONE", 1),
    ):
        declare_tensor_dim(name, extent)

    x_singleton = x_expert.unsqueeze(0)
    # routing_weight arrives as routing_sticks[..., :1]: a size-1 NARROWING slice
    # over a physically 64-wide broadcast stick, so its logical [T,E,1] view still
    # carries the 64-lane strides. Fed into the coarse-tile read-copy that way, the
    # outermost-dim size derivation (total_elems // stride) under-counts T by the
    # dropped 64 factor and rejects the T=512 tile. Materialize a genuinely dense,
    # owned [E,T,1] buffer so the derivation recovers E / T correctly. .contiguous()
    # alone on the permuted slice can be elided upstream; .clone() forces an owned
    # contiguous copy.
    route = routing_weight.permute(1, 0, 2).contiguous().clone()  # [T,E,1]->[E,T,ONE]
    name_tensor_dims(x_expert, ["T", "H"])
    name_tensor_dims(gate, ["E", "H", "M"])
    name_tensor_dims(up, ["E", "H", "M"])
    name_tensor_dims(down, ["E", "M", "H"])
    name_tensor_dims(route, ["E", "T", "ONE"])
    with spyre_hint(num_tiles_per_dim={"E": experts}, work_div={"T": 32}):
        gate_out = torch.matmul(x_singleton, gate)  # [1,T,H]@[E,H,M]->[E,T,M]
        up_out = torch.matmul(x_singleton, up)
        activated = F.gelu(gate_out, approximate="tanh") * up_out
        down_out = torch.matmul(activated, down)  # [E,T,M]@[E,M,H]->[E,T,H]
        return (down_out * route).sum(dim=0)  # [T,H]
```

- [ ] **Step 2: Update the `spyre_config.patch` in `_moe_ffn_persistent`**

In `_moe_ffn_persistent` (~390-396), the `spyre_config.patch({...})` currently sets `"enable_dense_expert_persistent": True` (a flag for the removed `moe_ffn` op path). Per `git stash@{0}`, replace that key with `"allow_all_ops_in_lx_planning": True`. The block becomes:

```python
    with spyre_config.patch(
        {
            "sencores": 32,
            "lx_planning": True,
            "allow_all_ops_in_lx_planning": True,
        }
    ):
        return compiled_persistent(
            x_expert,
            routing_weight,
            gate,
            up,
            down,
            K,
        )
```
Leave the rest of `_moe_ffn_persistent` (the `compiled_route(...)` call, `routing_weight = routing_sticks[..., :1]`) unchanged. Its signature `(x_router, x_expert, router, compiled_route, compiled_persistent, gate, up, down, route_identity, K, eps)` stays — `route_identity` is still consumed by `compiled_route`.

- [ ] **Step 3: Rewire the `forward` persistent call site to the shared tensors**

In `forward` (~535-547), change the three expert-weight args from the deleted persistent names to the shared names. `route_identity` stays (routing still uses it). Replace:
```python
                self._spyre_persistent_gate,
                self._spyre_persistent_up,
                self._spyre_persistent_down,
                self._spyre_persistent_route_identity,
```
with:
```python
                self._spyre_gate,
                self._spyre_up,
                self._spyre_down,
                self._spyre_persistent_route_identity,
```
(`_spyre_persistent_route_identity` is still produced by Task 3 — the routing identity is unrelated to the expert-weight layout.)

- [ ] **Step 4: Verify import/parse**

Run:
```bash
cd /mnt/devel/inductor_src/hf-adapters
python -c "import torch, torch_spyre; import hf_adapters.hf_gemma4_moe; print('IMPORT_OK')"
grep -n "torch.ops.spyre.moe_ffn\|enable_dense_expert_persistent\|_spyre_persistent_gate\|_spyre_persistent_up\|_spyre_persistent_down" hf_adapters/hf_gemma4_moe.py
```
Expected: `IMPORT_OK`. The grep must show NO live references to `torch.ops.spyre.moe_ffn`, `enable_dense_expert_persistent`, or the three `_spyre_persistent_gate/up/down` names in `forward`/`_moe_expert_persistent` bodies (docstring mentions are acceptable but prefer none). `_spyre_persistent_route_identity` MAY still appear (kept).

- [ ] **Step 5: Do NOT commit yet**

`prepare_for_spyre` still produces the old tensor names until Task 3; committing here would leave `forward` referencing `self._spyre_gate` before it is set. Proceed to Task 3; Tasks 2+3+4 commit together at Task 4 (they are one coherent rewrite — the tree is only runnable once all three land). Record this staging in the ledger.

---

### Task 3: hf-adapters — materialize ONE shared weight set in `prepare_for_spyre`

**Files:**
- Modify: `hf_adapters/hf_gemma4_moe.py` — `prepare_for_spyre` weight block (~678-712)

**Interfaces:**
- Consumes: `dma_moe_expert_weight_to_spyre` (Task 1); `experts.gate_up_proj`, `experts.down_proj` (the stock HF MoE expert weights); `router` params.
- Produces: on every block — `block._spyre_gate` `[E,H,M]`, `block._spyre_up` `[E,H,M]`, `block._spyre_down` `[E,M,H]` (device, shared layout); `block._spyre_persistent_route_identity` (unchanged); router `proj.weight`/`scale`/`per_expert_scale` moved to device once.

- [ ] **Step 1: Add the applier import**

At the top of `hf_gemma4_moe.py`, with the other `torch_spyre` imports, add:
```python
from torch_spyre.model_utils import dma_moe_expert_weight_to_spyre
```
(Confirm placement AFTER `import torch` / `import torch_spyre` per the import-order rule. Grep the existing top-of-file imports first and match their style.)

- [ ] **Step 2: Replace the weight-materialization block**

Read the current block first (`prepare_for_spyre`, ~678-712) to get exact surrounding lines, then replace the fused/dual materialization (the `gate_up_t`/`down_t` derivation through the `_spyre_gate_up_dev`/`_spyre_down_dev` assignments and any `_spyre_persistent_*` assignments) with the single-set block. `gate_up_t`/`down_t` are derived at ~678-679; keep those two derivation lines and replace everything from the `M = gate_up_t.shape[2] // 2` through the last expert-weight assignment:

```python
        # ONE shared expert-weight set, read by BOTH the prefill persistent
        # hint-body and the decode loop-on-topk gather. De-fuse gate_up into
        # separate gate/up, then lay each out E-outermost with the free dim on
        # the stick (dma_moe_expert_weight_to_spyre): simultaneously the
        # gather-source layout (expert dim outermost) and the matmul
        # weight-operand layout (sticked on free dim). ~42.5 GiB / 30 layers
        # (one set, not two) -- fits the card.
        M = gate_up_t.shape[2] // 2
        gate_l = gate_up_t[:, :, :M].contiguous()  # [E,H,M]
        up_l = gate_up_t[:, :, M:].contiguous()  # [E,H,M]
        block._spyre_gate = dma_moe_expert_weight_to_spyre(gate_l)  # [E,H,M]
        block._spyre_up = dma_moe_expert_weight_to_spyre(up_l)  # [E,H,M]
        block._spyre_down = dma_moe_expert_weight_to_spyre(down_t)  # [E,M,H]
        # Fall back to a plain device move if the free dim doesn't tile into
        # sticks (should not happen for gemma-4: M=704, H=2816 both divide 64).
        if block._spyre_gate is None:
            block._spyre_gate = gate_l.to("spyre")
        if block._spyre_up is None:
            block._spyre_up = up_l.to("spyre")
        if block._spyre_down is None:
            block._spyre_down = down_t.to("spyre")

        # Persistent routing identity (unrelated to expert-weight layout; the
        # packed router expands its one-hot rows against this eye).
        block._spyre_persistent_route_identity = torch.eye(
            64, dtype=gate_up_t.dtype
        ).to("spyre")

        # The router runs on-device in BOTH paths, so its weights must be
        # device-resident. Reassign the Parameter object (a cross-backend
        # param.data set_data raises on the type change). Done once here.
        router.proj.weight = torch.nn.Parameter(
            router.proj.weight.data.to("spyre"), requires_grad=False
        )
        router.scale = torch.nn.Parameter(
            router.scale.data.to("spyre"), requires_grad=False
        )
        router.per_expert_scale = torch.nn.Parameter(
            router.per_expert_scale.data.to("spyre"), requires_grad=False
        )
```

Notes for the implementer:
- `router` is bound just above in the loop (grep `router = block.router` — currently ~line before the old branch). Confirm it is in scope; if the old code bound it inside a deleted `if`, add `router = block.router` before the router moves.
- Delete any CPU-resident `_spyre_gate_up_t`/`_spyre_down_t` assignments if present (they were staged for a generic `model.to` sweep that no longer applies). Grep for them; if the block sets them, remove those lines.
- Do NOT keep `_spyre_gate_up_dev`/`_spyre_down_dev` (decode now reads `_spyre_gate`/`_up`/`_down` — Task 4).

- [ ] **Step 3: Verify import/parse + no stale weight names**

Run:
```bash
cd /mnt/devel/inductor_src/hf-adapters
python -c "import torch, torch_spyre; import hf_adapters.hf_gemma4_moe; print('IMPORT_OK')"
grep -n "_spyre_gate_up_dev\|_spyre_down_dev\|_spyre_persistent_gate\|_spyre_persistent_up\|_spyre_persistent_down\|_spyre_gate_up_t\|_spyre_down_t" hf_adapters/hf_gemma4_moe.py
```
Expected: `IMPORT_OK`. The grep must show NO live assignments or reads of the six deleted names (docstring mentions tolerable but prefer clean). `block._spyre_gate`/`_up`/`_down` and `_spyre_persistent_route_identity` are the only expert/identity attrs set.

- [ ] **Step 4: Do NOT commit yet**

Decode (`forward` else-branch + `_moe_ffn_loop` + `_compiled_moe_loop_region`) still reads `self._spyre_gate_up_dev`/`_spyre_down_dev`, now unset → runtime AttributeError on a decode forward. Task 4 rewires decode. Commit at Task 4.

---

### Task 4: hf-adapters — split decode gather + rewire decode call site + on-card gate + commit

**Files:**
- Modify: `hf_adapters/hf_gemma4_moe.py` — `_compiled_moe_loop_region` (~113-197), `_moe_ffn_loop` (~200-244), `forward` decode call site (~549-559)

**Interfaces:**
- Consumes: `block._spyre_gate`/`_up`/`_down` (Task 3); the B5-B8 `idx_addr` prep (unchanged); `spyre_hint` (already imported in the region).
- Produces: `_compiled_moe_loop_region(..., gate_dev, up_dev, down_dev, token_ids, K, tile, eps)` gathering `gate`/`up`/`down` separately; `_moe_ffn_loop(x_router, x_expert, router, compiled_loop, gate_dev, up_dev, down_dev, K, tile, eps)`; `forward` decode call passing the three shared tensors.

- [ ] **Step 1: Split the gather in `_compiled_moe_loop_region`**

The region signature currently is `(x_router, x_expert, router_proj_w, router_scale, router_scalar_root_size, per_expert_scale, gate_up_dev, down_dev, token_ids, K, tile, eps)`. Change `gate_up_dev` → `gate_dev, up_dev` (down_dev stays). Update the signature and the docstring line `gate_up_dev: [E,H,2M]...` to describe `gate_dev`/`up_dev` `[E,H,M]` and `down_dev` `[E,M,H]`, E outermost, free dim on stick.

Inside the `with spyre_hint(tiles={"row": tile}):` block (~185-193), replace the fused gather + bmm + chunk:
```python
        gathered = x_expert[:, None, :].expand(T, K, H)  # [T,K,H]
        W_gu = gate_up_dev[idx_addr]  # [T,K,H,2M] on-device index_select
        W_dn = down_dev[idx_addr]  # [T,K,M,H]
        gu = torch.matmul(gathered.unsqueeze(-2), W_gu)  # [T,K,1,2M] batched
        g, u = gu.chunk(2, dim=-1)  # [T,K,1,M]
        act = F.gelu(g, approximate="tanh") * u  # [T,K,1,M]
        row_out = torch.matmul(act, W_dn).squeeze(-2)  # [T,K,H]
        row_out = row_out * w[..., None]  # [T,K,1] broadcast
```
with the split gather (two separate index_selects on the shared tensors, one bmm each):
```python
        gathered = x_expert[:, None, :].expand(T, K, H)  # [T,K,H]
        W_g = gate_dev[idx_addr]  # [T,K,H,M] on-device index_select
        W_u = up_dev[idx_addr]  # [T,K,H,M]
        W_dn = down_dev[idx_addr]  # [T,K,M,H]
        g = torch.matmul(gathered.unsqueeze(-2), W_g)  # [T,K,1,M] batched
        u = torch.matmul(gathered.unsqueeze(-2), W_u)  # [T,K,1,M]
        act = F.gelu(g, approximate="tanh") * u  # [T,K,1,M]
        row_out = torch.matmul(act, W_dn).squeeze(-2)  # [T,K,H]
        row_out = row_out * w[..., None]  # [T,K,1] broadcast
```

- [ ] **Step 2: Update `_moe_ffn_loop` signature + region call**

`_moe_ffn_loop` currently is `(x_router, x_expert, router, compiled_loop, gate_up_dev, down_dev, K, tile, eps)`. Change `gate_up_dev` → `gate_dev, up_dev`. Update the docstring `gate_up_dev`/`down_dev` mention. In the `compiled_loop(...)` call (~222-234), change the weight args from:
```python
        gate_up_dev,
        down_dev,
```
to:
```python
        gate_dev,
        up_dev,
        down_dev,
```
(matching the region's new signature order: `..., per_expert_scale, gate_dev, up_dev, down_dev, token_ids, ...`). Leave the host `index_add` combine unchanged.

- [ ] **Step 3: Rewire the `forward` decode call site**

In `forward` (~549-559), change the decode call's weight args from:
```python
                self._spyre_gate_up_dev,
                self._spyre_down_dev,
```
to:
```python
                self._spyre_gate,
                self._spyre_up,
                self._spyre_down,
```

- [ ] **Step 4: Verify import/parse + argument-count coherence**

Run:
```bash
cd /mnt/devel/inductor_src/hf-adapters
python -c "import torch, torch_spyre; import hf_adapters.hf_gemma4_moe; print('IMPORT_OK')"
grep -n "gate_up_dev\|_spyre_gate_up_dev\|_spyre_down_dev\|torch.ops.spyre.moe_ffn" hf_adapters/hf_gemma4_moe.py
```
Expected: `IMPORT_OK`. The grep returns NOTHING (all fused/dual names gone from live code and docstrings). Manually confirm the three call sites agree on argument count: `forward` decode call → `_moe_ffn_loop` params → `compiled_loop`/`_compiled_moe_loop_region` params.

- [ ] **Step 5: On-card prefill gate (the real verification)**

The router is still on `torch.where` (index_mask not landed), so the persistent prefill path traces. Clear the inductor cache (guard-safe `mv`) and run the persistent prefill leg:
```bash
cd /mnt/devel/inductor_src/hf-adapters
mv /tmp/torchinductor_aviros /tmp/torchinductor_aviros.old.$$ 2>/dev/null || true
HF_HUB_CACHE=/mnt/models/hf_cache/hub HF_HUB_OFFLINE=1 \
  python -u repros/gemma4_moe/ab_persistent_vs_chunked.py persistent 2>&1 | tee /tmp/single_layout_prefill.log
```
Expected:
1. **No `0x340f FlexAllocator OutOfMemory`** — the model loads with ONE ~42.5 GiB weight set (the whole point of this plan).
2. Compiles and runs end-to-end; emits `' the'`; warm generation ≈ 7.3 s (the pre-refactor persistent number).
3. `_untracked_*` named-dim warnings are expected and do NOT abort.

If it OOMs again: STOP and diagnose (the footprint math says one set fits ~42.5 GiB on ~96 GiB; a second OOM means something else is doubly-resident — check for a lingering `_spyre_gate_up_t`/`_spyre_down_t` or a stale `_spyre_persistent_*`). If it aborts in compile or the token changes: diagnose before committing (do not commit red). If the harness cannot select the mode (the mode global is gone — dispatch is by seq_len), run a minimal prefill via `generate` with PREFILL_TOKENS=512, NEW_TOKENS=1 instead; report the token and warm time either way.

- [ ] **Step 6: On-card device-layout sanity (confirms the layout applied)**

After a successful load in the same run (or a tiny standalone probe), confirm one expert weight's device layout is the shared arrangement, not the default. Add a one-off probe or inspect in the harness:
```python
lay = model.model.layers[<first MoE layer idx>]._spyre_gate.device_tensor_layout()
print("device_size", lay.device_size)  # expect [E, H, M//64, 64], E outermost + trailing stick
```
Expected: `device_size` has 4 entries with E (128) outermost and a trailing 64 stick dim — NOT the logical 3-entry `[E,H,M]`. This proves `dma_moe_expert_weight_to_spyre` applied rather than falling back. (If a standalone probe is impractical on the loaded 26B model, note that the successful compile + correct token in Step 5 is itself strong evidence the layout is consistent, and record the `device_size` from a small synthetic `[128, 2816, 704]` fp16 tensor run through the applier instead.)

- [ ] **Step 7: Commit (hf-adapters — Tasks 2+3+4 together)**

```bash
cd /mnt/devel/inductor_src/hf-adapters
git add hf_adapters/hf_gemma4_moe.py
git commit -s -m "refactor(gemma4-moe): single shared expert-weight layout for both FFN paths

Replace the dual expert-weight materialization (which OOM'd the card at
~85 GiB) with ONE shared device layout per weight, read by both the prefill
persistent hint-body and the decode loop-on-topk gather. De-fuse gate_up into
separate gate/up. Lay each expert weight out E-outermost with the free dim on
the stick via torch-spyre dma_moe_expert_weight_to_spyre -- simultaneously the
gather-source layout (expert dim outermost) and the matmul weight-operand
layout (sticked on free dim). Restore the hint-body persistent matmuls,
dropping the absent spyre::moe_ffn op. One weight set is ~42.5 GiB/30 layers,
fits the card. Persistent prefill re-verified on card: emits ' the' at ~7.3s
warm. Working-tree checkpoint; not pushed.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:** Spec §A (applier) → Task 1. §B (one weight set at load) → Task 3. §C (hint-body persistent reads shared set) → Task 2. §D (split decode gather) → Task 4. Verification 1-2 (import, OOM gate) → Task 4 Step 4-5. Verification 3 (on-card prefill) → Task 4 Step 5. Verification 4 (device-layout sanity) → Task 4 Step 6. All covered.

**Placeholder scan:** No TBDs; every code step has the actual code. The one judgment point (device-layout probe on a 26B model) has an explicit fallback (synthetic-tensor probe).

**Type/name consistency:** The three shared tensor names `_spyre_gate`/`_spyre_up`/`_spyre_down` are identical across Tasks 2 (persistent read), 3 (production), 4 (decode read). `dma_moe_expert_weight_to_spyre` signature identical in Task 1 (def) and Task 3 (call). `_moe_ffn_loop`/`_compiled_moe_loop_region` param rename `gate_up_dev → gate_dev, up_dev` applied consistently across region def, orchestrator call, and forward call (Task 4 Steps 1-3). `_spyre_persistent_route_identity` kept in both Task 2 (read) and Task 3 (production).

**Staging note:** Tasks 2+3+4 are one coherent rewrite of the adapter — the tree is only runnable after all three (each leaves a dangling name until the next). They commit together at Task 4 Step 7; Tasks 2 and 3 explicitly do NOT commit. Task 1 (torch-spyre, different repo) commits independently at its Step 5. The executor must NOT treat Task 2's or Task 3's "no commit" as a failure — it is the designed staging, matching the prior plan's Task 3+4 back-to-back pattern.
