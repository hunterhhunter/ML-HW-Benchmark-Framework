import pandas as pd
import pytest
import torch

from ttm_r1.etth1_quality import (
    ETTh1QualityConfig,
    evaluate_prepared_windows,
    forecast_metrics,
    load_train_calibration_contexts,
    load_etth1_windows,
    percentage_degradation,
    prediction_delta_metrics,
    write_calibration_inputs,
)
from ttm_r1.host_adapter import TTMR1HostAdapter


class _Last96Core(torch.nn.Module):
    """Small deterministic core that preserves the last 96 normalized values."""

    def forward(self, past_values):
        return past_values[:, -96:, :]


def test_etth1_windows_start_at_the_test_boundary_and_use_only_past_context(tmp_path):
    """Catches a quality run that leaks targets or starts before the test split."""
    csv_path = tmp_path / "ETTh1.csv"
    values = list(range(8640 + 2880 + 2880))
    pd.DataFrame({"date": range(len(values)), "OT": values}).to_csv(csv_path, index=False)

    contexts, targets, split = load_etth1_windows(
        ETTh1QualityConfig(dataset_path=csv_path, windows=2)
    )

    assert split == {
        "train": 8640,
        "validation": 2880,
        "test": 2880,
        "test_start": 11520,
        "windows": 2,
    }
    assert contexts.shape == (2, 512, 1)
    assert targets.shape == (2, 96, 1)
    assert contexts[0, -1, 0].item() == 11519
    assert targets[0, 0, 0].item() == 11520
    assert targets[1, 0, 0].item() == 11521


def test_quality_metrics_are_calculated_from_real_prediction_tensors():
    """Catches MAE, RMSE, delta, or zero-baseline degradation miscalculation."""
    cpu = torch.tensor([[[1.0], [3.0]]])
    rngd = torch.tensor([[[2.0], [5.0]]])
    target = torch.tensor([[[0.0], [1.0]]])

    assert forecast_metrics(cpu, target) == pytest.approx(
        {"mae": 1.5, "rmse": 1.5811388300841898}
    )
    assert prediction_delta_metrics(cpu, rngd) == pytest.approx(
        {"mae": 1.5, "rmse": 1.5811388300841898, "max_abs_error": 2.0}
    )
    assert percentage_degradation(2.0, 2.5) == 25.0
    assert percentage_degradation(0.0, 1.0) is None


def test_evaluator_restores_cpu_and_device_forecasts_from_identical_prepared_inputs():
    """Catches the device path receiving raw context rather than CPU-prepared input."""
    seen = []

    def device_runner(inputs):
        seen.append(inputs[0].clone())
        return inputs[0][:, -96:, :]

    result = evaluate_prepared_windows(
        cpu_core=_Last96Core(),
        adapter=TTMR1HostAdapter(),
        contexts=torch.arange(2 * 512, dtype=torch.float32).reshape(2, 512, 1),
        targets=torch.zeros((2, 96, 1), dtype=torch.float32),
        device_runner=device_runner,
    )

    assert len(seen) == 2
    assert result["cpu_predictions"].shape == (2, 96, 1)
    assert torch.equal(result["cpu_predictions"], result["rngd_predictions"])
    assert result["prediction_delta"] == {
        "mae": 0.0,
        "rmse": 0.0,
        "max_abs_error": 0.0,
    }


def test_train_calibration_contexts_are_sorted_and_never_use_validation_values(tmp_path):
    """Catches calibration origins crossing the fixed ETTh1 train boundary."""
    csv_path = tmp_path / "ETTh1.csv"
    values = list(range(8640 + 2880 + 2880))
    pd.DataFrame({"date": range(len(values)), "OT": values}).to_csv(csv_path, index=False)

    contexts, metadata = load_train_calibration_contexts(
        ETTh1QualityConfig(dataset_path=csv_path), samples=3
    )

    assert metadata["origins"] == [512, 4576, 8640]
    assert contexts.shape == (3, 512, 1)
    assert contexts[-1, -1, 0].item() == 8639


def test_calibration_writer_saves_each_host_prepared_core_input(tmp_path):
    """Catches feeding raw, unscaled ETTh1 data into ARIES PTQ calibration."""
    contexts = torch.arange(2 * 512, dtype=torch.float32).reshape(2, 512, 1)
    result = write_calibration_inputs(TTMR1HostAdapter(), contexts, tmp_path)

    assert result["samples"] == 2
    assert (tmp_path / "calibration-000.npy").is_file()
    assert (tmp_path / "calibration-001.npy").is_file()
    assert (tmp_path / "calibration-manifest.json").is_file()
