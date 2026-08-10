"""CPU shape/behavior test for the decoupled prefill→stitch→decode routine.

DEVICE='cpu' patching of hf_common happens in tests/conftest.py; this file is
plain pytest and must be run via the CPU lane (pytest tests/cpu/...).
"""
import gc
import importlib.util
import sys
from pathlib import Path

import torch

from tests.conftest import (
    get_dtype_for_cpu,
    load_ref_model,
    resolve_adapter_module_for_test,
)
from tests.cpu.conftest import _set_rope_dtype, _unwrap_compiled_blocks
from tests.model_registry import CAUSAL_LM_MODELS

MODEL_KEY = "granite2b"
SEQ_LEN = 128           # BLOCK_SIZE multiple; keeps the CPU test light
BATCH_SIZE = 4


def _load_script_module():
    path = Path(__file__).resolve().parents[2] / "scripts" / "profile_prefill_decode_spyre.py"
    spec = importlib.util.spec_from_file_location("profile_prefill_decode_spyre", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_prefill_decode_shapes_cpu():
    script = _load_script_module()
    model_path = CAUSAL_LM_MODELS[MODEL_KEY]["path"]
    hf_common_mod = sys.modules["hf_adapters.hf_common"]
    adapter_mod = resolve_adapter_module_for_test(model_path)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model = load_ref_model(model_path, adapter_mod)
    adapter_mod.prepare_for_spyre(model)
    _unwrap_compiled_blocks(model)
    _set_rope_dtype(model, get_dtype_for_cpu(model_path))

    out = script.run_batched_prefill_then_decode(
        model, adapter_mod, tokenizer, seq_len=SEQ_LEN, batch_size=BATCH_SIZE
    )

    # padded_len = 128, max_cache_len = 128 + 64 = 192
    expected_cache_len = hf_common_mod.generation_cache_len(SEQ_LEN, 1)
    assert out["max_cache_len"] == expected_cache_len == 192

    n_layers = len(out["key_caches"])
    assert n_layers > 0
    for kc in out["key_caches"]:
        assert kc.shape[0] == BATCH_SIZE
        assert kc.shape[2] == expected_cache_len
    assert out["next_tokens"].shape == (BATCH_SIZE,)
    assert out["next_tokens"].dtype == torch.long

    del model
    gc.collect()
