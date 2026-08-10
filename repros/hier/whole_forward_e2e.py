"""Task 5 (HOST) — validate the WHOLE-FORWARD compiled path on Spyre.

The stock tests/spyre/test_e2e_token_compare_spyre.py drives the Spyre side via
``adapter._run_forward`` (the eager module fn). That exercises the region-wrapped
decoder blocks and the _run_backbone_forward loop (R1 KV mutation, R2 head break
inside the block), but NOT Task 3's ``model._spyre_run_forward`` (the torch.compile
of the whole forward: embed/mul/rope prologue + block loop + norm/head/scaling
epilogue) nor Task 4's generate() routing shim.

This harness closes that gap: it reuses the SAME greedy-decode driver, KV cache
setup, and masks as the stock test (imported directly, so the decode logic is
byte-identical), but feeds the run_forward_fn produced by Task 4's
``_resolve_run_forward_fn(model, adapter._run_forward)`` — which prefers
``model._spyre_run_forward`` (Task 3) when present. It compares greedy tokens
against the stock HF CPU reference, exactly like the token-compare test.

Any mismatch or error here is a STOP-AND-REPORT (R1/R2), not a fallback.

Run (on the Spyre host):
    HF_TOKEN=... HF_HOME=/mnt/models/hf_cache/ \
        python -m pytest -s -vvv repros/hier/whole_forward_e2e.py -k granite
"""

import pytest

from hf_adapters.auto_spyre_model import (
    _resolve_run_forward_fn,
    torch_dtype_for_model_path,
)
from hf_adapters.hf_common import move_model_to_spyre
from tests.conftest import load_ref_model, resolve_adapter_module_for_test
from tests.model_registry import (
    CAUSAL_PATHS,
    NON_BLOCKING_CAUSAL_MODELS,
    xfail_non_blocking,
)

# Reuse the stock test's helpers verbatim so the reference path, decode driver,
# KV allocation, masks, and comparison are identical — only the Spyre-side
# run_forward_fn differs (whole-forward compiled vs eager _run_forward).
from tests.spyre.test_e2e_token_compare_spyre import (
    adapter_greedy_steps,
    hf_greedy_steps,
    _compare_results,
)


def _run_whole_forward_test(model_path: str, num_decode: int = 4):
    from transformers import AutoTokenizer

    adapter = resolve_adapter_module_for_test(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = load_ref_model(model_path=model_path, adapter_mod=adapter)

    prompt = "The capital of France is"
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"]
    print(f"  Prompt: {prompt!r} ({input_ids.shape[1]} tokens)")

    print("  Running HF reference on CPU ...")
    hf_results = hf_greedy_steps(model, input_ids, num_decode=num_decode)

    spyre_dtype = torch_dtype_for_model_path(model_path)
    move_model_to_spyre(model=model, module=adapter, dtype=spyre_dtype)

    # THE POINT: route through Task 4's resolver, which prefers Task 3's
    # compiled model._spyre_run_forward over the eager adapter._run_forward.
    run_forward_fn = _resolve_run_forward_fn(model, adapter._run_forward)
    assert getattr(model, "_spyre_run_forward", None) is not None, (
        "prepare_for_spyre did not attach _spyre_run_forward — whole-forward "
        "path is not being exercised"
    )

    print("  Running WHOLE-FORWARD compiled path on Spyre ...")
    adapter_results = adapter_greedy_steps(
        run_forward_fn,
        model,
        input_ids,
        num_decode=num_decode,
    )

    return _compare_results(hf_results, adapter_results, tokenizer, model_path)


@pytest.mark.parametrize(
    "model_path", xfail_non_blocking(CAUSAL_PATHS, table=NON_BLOCKING_CAUSAL_MODELS)
)
def test_whole_forward_token_compare_spyre(model_path: str) -> None:
    rows = _run_whole_forward_test(model_path)
    mismatches = [r for r in rows if not r["top1_match"]]
    n_match = sum(1 for r in rows if r["top1_match"])
    print(f"\nWhole-forward top-1 agreement: {n_match}/{len(rows)} steps")
    assert not mismatches, mismatches
