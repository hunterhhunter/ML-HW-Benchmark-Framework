import importlib.util
from pathlib import Path

import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parent.parent / "models" / "prepare_mobilint_llm.py"
)


def _load_script_module():
    spec = importlib.util.spec_from_file_location(
        "prepare_mobilint_llm",
        SCRIPT_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("model", "capacity", "repo_id", "relative_path"),
    [
        (
            "llama-3.1-8b",
            1,
            "mobilint/Llama-3.1-8B-Instruct",
            Path("llama-3.1-8b/standard"),
        ),
        (
            "llama-3.1-8b",
            16,
            "mobilint/Llama-3.1-8B-Instruct-Batch16",
            Path("llama-3.1-8b/batch16"),
        ),
        (
            "llama-3.2-3b",
            32,
            "mobilint/Llama-3.2-3B-Instruct-Batch32",
            Path("llama-3.2-3b/batch32"),
        ),
    ],
)
def test_download_selects_official_repo_and_stable_local_directory(
    tmp_path,
    model,
    capacity,
    repo_id,
    relative_path,
):
    module = _load_script_module()
    calls = []

    def fake_snapshot_download(**kwargs):
        calls.append(kwargs)
        return kwargs["local_dir"]

    downloaded = module.download_model(
        model=model,
        batch_capacity=capacity,
        output_root=tmp_path,
        revision="pinned-revision",
        snapshot_download=fake_snapshot_download,
    )

    assert downloaded == (tmp_path / relative_path).resolve()
    assert calls == [
        {
            "repo_id": repo_id,
            "revision": "pinned-revision",
            "local_dir": str((tmp_path / relative_path).resolve()),
        }
    ]


@pytest.mark.parametrize(
    ("model", "capacity"),
    [
        ("llama-3.3-70b", 1),
        ("llama-3.1-8b", 2),
        ("llama-3.2-3b", True),
    ],
)
def test_download_rejects_unknown_model_or_capacity_before_hub_call(
    tmp_path,
    model,
    capacity,
):
    module = _load_script_module()
    calls = []

    with pytest.raises(ValueError, match="model|batch_capacity"):
        module.download_model(
            model=model,
            batch_capacity=capacity,
            output_root=tmp_path,
            snapshot_download=lambda **kwargs: calls.append(kwargs),
        )

    assert calls == []
