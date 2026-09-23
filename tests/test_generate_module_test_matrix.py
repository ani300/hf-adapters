import importlib.util
from collections import Counter
from pathlib import Path

import pytest

_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / ".github"
    / "scripts"
    / "generate_module_test_matrix.py"
)
_SPEC = importlib.util.spec_from_file_location("generate_module_test_matrix", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
module_test_matrix = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(module_test_matrix)


def _add_configs(config_dir: Path, *names: str) -> None:
    config_dir.mkdir()
    for name in names:
        (config_dir / name).touch()


def test_only_measured_slow_configs_are_sharded(tmp_path: Path) -> None:
    config_dir = tmp_path / "module_tests"
    _add_configs(
        config_dir,
        "ordinary.yaml",
        "Ministral-3-14B-Instruct-2512.yaml",
    )

    entries = module_test_matrix.generate_matrix(config_dir)["include"]

    assert entries[0] == {
        "config": "Ministral-3-14B-Instruct-2512.yaml",
        "shard": "forward",
        "pytest_filter": "test_forward",
    }
    assert [entry["shard"] for entry in entries[:5]] == [
        "forward",
        "with-cpu",
        "eager-prefill",
        "eager-other",
        "remaining",
    ]
    assert entries[5] == {
        "config": "ordinary.yaml",
        "shard": "all",
        "pytest_filter": "",
    }


def test_cli_writes_compact_github_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_dir = tmp_path / "module_tests"
    _add_configs(config_dir, "ordinary.yaml")
    github_output = tmp_path / "github-output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(github_output))
    monkeypatch.setattr("sys.argv", ["generate_module_test_matrix.py", str(config_dir)])

    module_test_matrix.main()

    assert github_output.read_text() == (
        'module_config_matrix={"include":[{"config":"ordinary.yaml",'
        '"shard":"all","pytest_filter":""}]}\n'
    )


def test_repository_matrix_shards_each_timing_outlier_five_ways() -> None:
    config_dir = Path(__file__).resolve().parent / "configs" / "module_tests"

    entries = module_test_matrix.generate_matrix(config_dir)["include"]
    counts = Counter(entry["config"] for entry in entries)

    assert {
        config for config, count in counts.items() if count > 1
    } == module_test_matrix.SHARDED_CONFIGS
    assert all(counts[config] == 5 for config in module_test_matrix.SHARDED_CONFIGS)


def test_empty_config_directory_is_rejected(tmp_path: Path) -> None:
    config_dir = tmp_path / "module_tests"
    config_dir.mkdir()

    with pytest.raises(ValueError, match="No module-test YAML configs found"):
        module_test_matrix.generate_matrix(config_dir)
