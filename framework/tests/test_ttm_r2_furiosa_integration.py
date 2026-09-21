import hashlib
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from core.model_spec import Task
import main as benchmark_main
from runtimes.furiosa_torch_models import get_torch_model_adapter
from ttm_r2 import profile
from ttm_r2.core import TTMR2Core


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


@pytest.fixture
def checkpoint(monkeypatch, tmp_path):
    source = tmp_path / "ibm-granite_granite-timeseries-ttm-r2"
    source.mkdir()
    config = b"config"
    weights = b"weights"
    (source / "config.json").write_bytes(config)
    (source / "model.safetensors").write_bytes(weights)
    monkeypatch.setattr(profile, "TTM_R2_CONFIG_SHA256", _sha256(config))
    monkeypatch.setattr(profile, "TTM_R2_MODEL_SHA256", _sha256(weights))
    return source


def test_ttm_r2_furiosa_adapter_uses_fixed_semantic_contract(checkpoint):
    adapter = get_torch_model_adapter("ttm-r2")

    assert adapter.task is Task.TIME_SERIES_FORECASTING
    assert adapter.input_names == ("past_values",)
    assert adapter.input_shapes == {"past_values": (1, 512, 1)}
    assert adapter.input_dtypes == {"past_values": "float32"}
    assert adapter.output_names == ("forecast",)
    assert adapter.output_shapes == {"forecast": (1, 96, 1)}
    adapter.validate_source(checkpoint)


def test_ttm_r2_furiosa_loader_wraps_the_local_model_in_fixed_core(
    monkeypatch, checkpoint
):
    from ttm_r2 import core as ttm_core

    class _Model(torch.nn.Module):
        def forward(self, *, past_values, return_dict):
            return SimpleNamespace(prediction_outputs=past_values[:, :96, :])

    model = _Model()
    monkeypatch.setattr(ttm_core, "load_ttm_r2_model", lambda path: model)

    loaded = get_torch_model_adapter("ttm-r2").loader(checkpoint)

    assert isinstance(loaded, TTMR2Core)
    assert loaded.training is False


def test_furiosa_torch_cli_accepts_ttm_r2_time_series(checkpoint):
    args = Namespace(
        model="ttm-r2",
        model_path=str(checkpoint),
        batch_size=1,
        worker_count=None,
        compile=True,
    )

    assert benchmark_main._validate_furiosa_torch_cli(
        args, Task.TIME_SERIES_FORECASTING
    ) == checkpoint.resolve()


def test_ttm_r2_source_hash_failure_happens_before_model_loading(
    monkeypatch, checkpoint
):
    from ttm_r2 import core as ttm_core

    called = False

    def fail_if_called(path):
        nonlocal called
        called = True
        raise AssertionError("model loader must not run after source hash failure")

    monkeypatch.setattr(ttm_core, "load_ttm_r2_model", fail_if_called)
    (checkpoint / "model.safetensors").write_bytes(b"changed")

    with pytest.raises(ValueError, match="SHA-256"):
        get_torch_model_adapter("ttm-r2").loader(checkpoint)
    assert called is False
