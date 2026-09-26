import hashlib

import numpy as np
import pandas as pd
import pytest

from core.inference_pipeline import InferencePipeline
from core.model_spec import Model_Spec, Task
from dataloader import create_dataloader
from ttm_r2 import profile


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def canonical_csv(monkeypatch, tmp_path):
    csv_path = tmp_path / "ETTh1.csv"
    values = np.arange(8640 + 2880 + 2880, dtype=np.float32)
    pd.DataFrame({"date": np.arange(values.size), "OT": values}).to_csv(
        csv_path, index=False
    )
    monkeypatch.setattr(profile, "TTM_R2_DATASET_SHA256", _sha256(csv_path))
    return csv_path


@pytest.fixture
def model_spec():
    return Model_Spec(
        name="ttm-r2",
        task=Task.TIME_SERIES_FORECASTING,
        input_shapes={"past_values": (1, 512, 1)},
        input_dtype={"past_values": "float32"},
        output_shapes={"forecast": (1, 96, 1)},
    )


@pytest.fixture
def loader(canonical_csv, model_spec):
    from dataloader.ttm_r2_etth1_loader import TTMR2ETTh1Loader

    return TTMR2ETTh1Loader(model_spec, csv_path=str(canonical_csv))


def test_loader_emits_unbatched_prepared_input_and_raw_target(loader):
    sample = loader.load_by_index(0)

    assert sample["input"]["past_values"].shape == (512, 1)
    assert sample["input"]["past_values"].dtype == np.float32
    assert sample["label"]["future_values"].shape == (96, 1)
    assert set(sample["label"]) == {
        "future_values",
        "loc",
        "scale",
        "model_loc",
        "model_scale",
    }


def test_framework_collation_creates_exact_device_shape(loader):
    class _Runtime:
        pass

    collated = InferencePipeline(loader, _Runtime()).collate_batch(
        [loader.load_by_index(0)]
    )

    assert collated["input"]["past_values"].shape == (1, 512, 1)


def test_loader_uses_canonical_test_origins(loader):
    first = loader.load_by_index(0)
    last = loader.load_by_index(239)

    def restore_context(sample):
        label = sample["label"]
        reference = (
            sample["input"]["past_values"] * label["model_scale"]
            + label["model_loc"]
        )
        return reference * label["scale"] + label["loc"]

    assert restore_context(first)[-1, 0] == pytest.approx(11519.0, abs=1e-3)
    assert first["label"]["future_values"][0, 0] == 11520.0
    assert last["label"]["future_values"][0, 0] == 11759.0
    assert last["label"]["future_values"][-1, 0] == 11854.0
    assert loader.total_samples == 240
    assert loader.get_metadata()["total_samples"] == 240


def test_loader_reset_returns_to_window_zero(loader):
    pipeline = InferencePipeline(loader, object())
    assert loader.load_single()["window_idx"] == 0
    assert loader.load_single()["window_idx"] == 1

    pipeline.reset_dataloader_cursor()

    assert loader.load_single()["window_idx"] == 0


def test_loader_rejects_wrong_dataset_hash_before_loading_windows(
    monkeypatch, tmp_path, model_spec
):
    from dataloader import ttm_r2_etth1_loader as loader_module

    csv_path = tmp_path / "ETTh1.csv"
    csv_path.write_text("date,OT\n0,1\n", encoding="utf-8")
    called = False

    def fail_if_called(config):
        nonlocal called
        called = True
        raise AssertionError("window loader must not run after hash failure")

    monkeypatch.setattr(loader_module, "load_etth1_windows", fail_if_called)

    with pytest.raises(ValueError, match="SHA-256"):
        loader_module.TTMR2ETTh1Loader(model_spec, csv_path=str(csv_path))
    assert called is False


def test_factory_routes_only_ttm_r2_to_the_canonical_loader(
    canonical_csv, model_spec
):
    from dataloader.ttm_r2_etth1_loader import TTMR2ETTh1Loader

    loader = create_dataloader(model_spec, csv_path=str(canonical_csv))

    assert isinstance(loader, TTMR2ETTh1Loader)
