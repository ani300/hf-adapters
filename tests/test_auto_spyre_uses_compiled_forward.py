"""The causal-LM generate path must prefer self._spyre_run_forward when present.

Tests the free helper _resolve_run_forward_fn(model, module_run_forward) that the
model_generate closure will use to choose the forward callable.
"""
import types


def test_resolve_prefers_compiled_forward():
    import hf_adapters.auto_spyre_model as asm

    called = {}

    def compiled_fw(input_ids, position_ids, attn_mask, kc, vc,
                    is_filling, token_index, cache_position):
        called["hit"] = True
        return "logits"

    model = types.SimpleNamespace(_spyre_run_forward=compiled_fw)

    def eager(*a, **k):
        raise AssertionError("eager path used despite compiled forward present")

    fn = asm._resolve_run_forward_fn(model, eager)
    # generate() calls fn(model, ...args...): the shim must drop the model arg.
    out = fn(model, "ids", "pos", "mask", [], [],
             is_filling=False, token_index=0, cache_position=0)
    assert called.get("hit") is True
    assert out == "logits"


def test_resolve_falls_back_to_eager_when_no_compiled():
    import hf_adapters.auto_spyre_model as asm

    model = types.SimpleNamespace()  # no _spyre_run_forward

    def eager(m, *a, **k):
        return ("eager", m)

    fn = asm._resolve_run_forward_fn(model, eager)
    assert fn is eager
