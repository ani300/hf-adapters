"""prepare_for_spyre must attach a compiled whole-forward callable and region blocks."""
import types


def test_prepare_attaches_compiled_run_forward(monkeypatch):
    import hf_adapters.hf_granite as g

    created = {}

    def fake_region_blocks(layers, is_res_mul=None):
        created["region_blocks"] = True
        return ["blk0", "blk1"]

    def fake_compile(fn, **kw):
        created["compiled"] = True
        created["dynamic"] = kw.get("dynamic")
        return fn

    # Stub the heavy prep pieces so this stays a CPU wiring test.
    monkeypatch.setattr(g, "prepare_standard_gqa_region_blocks", fake_region_blocks, raising=False)
    monkeypatch.setattr(g, "prepare_rope_and_heads", lambda m: None, raising=False)
    monkeypatch.setattr(g, "patch_rmsnorm", lambda c: None, raising=False)
    monkeypatch.setattr(g, "pad_lm_head", lambda m: None, raising=False)
    monkeypatch.setattr(g.torch, "compile", fake_compile, raising=False)

    model = types.SimpleNamespace()
    model.config = types.SimpleNamespace()
    # get_backbone(model).layers must exist; stub via monkeypatch.
    backbone = types.SimpleNamespace(layers=[object(), object()])
    monkeypatch.setattr(g, "get_backbone", lambda m: backbone, raising=False)

    g.prepare_for_spyre(model)

    assert created.get("region_blocks") is True
    assert created.get("compiled") is True
    assert created.get("dynamic") is False
    assert model._spyre_compiled_blocks == ["blk0", "blk1"]
    assert callable(model._spyre_run_forward)
