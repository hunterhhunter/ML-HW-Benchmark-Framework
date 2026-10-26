"""Pure ETTh1 windowing and metric primitives for TTM-R1 quality evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import numpy as np
import torch


_TRAIN_LENGTH = 8640
_VALIDATION_LENGTH = 2880
_TEST_LENGTH = 2880


@dataclass(frozen=True)
class ETTh1QualityConfig:
    """Fixed univariate ETTh1 contract for zero-shot quality measurement."""

    dataset_path: Path
    column: str = "OT"
    context_length: int = 512
    prediction_length: int = 96
    windows: int = 240

    def __post_init__(self) -> None:
        if self.column != "OT":
            raise ValueError("TTM-R1 ETTh1 quality evaluation requires the OT column")
        if self.context_length != 512:
            raise ValueError("TTM-R1 ETTh1 quality evaluation requires context_length=512")
        if self.prediction_length != 96:
            raise ValueError("TTM-R1 ETTh1 quality evaluation requires prediction_length=96")
        if not 1 <= self.windows <= _TEST_LENGTH - self.prediction_length + 1:
            raise ValueError("ETTh1 quality windows must fit entirely within the test split")


def load_etth1_windows(
    config: ETTh1QualityConfig,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
    """Load chronological OT contexts and their future test targets without leakage."""
    if not config.dataset_path.is_file():
        raise ValueError(f"ETTh1 CSV is missing: {config.dataset_path}")

    frame = pd.read_csv(config.dataset_path)
    if config.column not in frame.columns:
        raise ValueError(f"ETTh1 CSV has no {config.column!r} column")
    values = torch.tensor(frame[config.column].to_numpy(), dtype=torch.float32)

    test_start = _TRAIN_LENGTH + _VALIDATION_LENGTH
    required_length = test_start + config.windows + config.prediction_length - 1
    if values.numel() < required_length:
        raise ValueError("ETTh1 CSV does not contain enough requested test windows")
    if not bool(torch.isfinite(values).all()):
        raise ValueError("ETTh1 OT values must be finite")

    starts = range(test_start, test_start + config.windows)
    contexts = torch.stack(
        [values[start - config.context_length : start] for start in starts]
    ).unsqueeze(-1)
    targets = torch.stack(
        [values[start : start + config.prediction_length] for start in starts]
    ).unsqueeze(-1)
    expected_context_shape = (config.windows, config.context_length, 1)
    expected_target_shape = (config.windows, config.prediction_length, 1)
    if tuple(contexts.shape) != expected_context_shape:
        raise ValueError(f"ETTh1 context shape mismatch: {tuple(contexts.shape)}")
    if tuple(targets.shape) != expected_target_shape:
        raise ValueError(f"ETTh1 target shape mismatch: {tuple(targets.shape)}")

    return contexts, targets, {
        "train": _TRAIN_LENGTH,
        "validation": _VALIDATION_LENGTH,
        "test": _TEST_LENGTH,
        "test_start": test_start,
        "windows": config.windows,
    }


def load_train_calibration_contexts(
    config: ETTh1QualityConfig, samples: int = 256
) -> tuple[torch.Tensor, dict[str, object]]:
    """Select sorted, train-only raw contexts for ARIES PTQ calibration."""
    if not 1 <= samples <= _TRAIN_LENGTH - config.context_length + 1:
        raise ValueError("calibration samples must fit the fixed ETTh1 train split")
    if not config.dataset_path.is_file():
        raise ValueError(f"ETTh1 CSV is missing: {config.dataset_path}")
    frame = pd.read_csv(config.dataset_path)
    if config.column not in frame.columns:
        raise ValueError(f"ETTh1 CSV has no {config.column!r} column")
    values = torch.tensor(frame[config.column].to_numpy(), dtype=torch.float32)
    if values.numel() < _TRAIN_LENGTH or not bool(torch.isfinite(values).all()):
        raise ValueError("ETTh1 OT calibration values must be finite through the train split")

    origins = np.linspace(config.context_length, _TRAIN_LENGTH, num=samples, dtype=int)
    if len(set(origins.tolist())) != samples:
        raise ValueError("calibration origin selection contains duplicates")
    contexts = torch.stack(
        [values[origin - config.context_length : origin] for origin in origins]
    ).unsqueeze(-1)
    expected = (samples, config.context_length, 1)
    if tuple(contexts.shape) != expected:
        raise ValueError(f"ETTh1 calibration context shape mismatch: {tuple(contexts.shape)}")
    return contexts, {"split": "train", "samples": samples, "origins": origins.tolist()}


def write_calibration_inputs(
    adapter: Any, contexts: torch.Tensor, directory: Path
) -> dict[str, object]:
    """Persist CPU-prepared core inputs for an MXQ compiler calibration directory."""
    _validate_window_tensor(
        contexts, "calibration contexts", (None, adapter.contract.external_input.shape[1], 1)
    )
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    files = []
    for index, context in enumerate(contexts):
        destination = directory / f"calibration-{index:03d}.npy"
        if destination.exists():
            raise FileExistsError(f"calibration input already exists: {destination}")
        prepared = adapter.prepare(context.unsqueeze(0)).past_values.detach().cpu().numpy()
        np.save(destination, prepared)
        digest = hashlib.sha256(destination.read_bytes()).hexdigest()
        files.append({"path": destination.name, "sha256": digest})
    manifest = directory / "calibration-manifest.json"
    if manifest.exists():
        raise FileExistsError(f"calibration manifest already exists: {manifest}")
    manifest.write_text(
        json.dumps(
            {"samples": len(files), "shape": [1, 512, 1], "dtype": "float32", "files": files},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return {"samples": len(files), "directory": str(directory.resolve()), "manifest": str(manifest.resolve())}


def forecast_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    """Calculate MAE and RMSE against ETTh1 ground truth."""
    delta = _validated_delta(prediction, target, "prediction", "target")
    return {"mae": float(delta.mean()), "rmse": float(delta.square().mean().sqrt())}


def prediction_delta_metrics(cpu: torch.Tensor, rngd: torch.Tensor) -> dict[str, float]:
    """Measure the output difference between CPU and RNGD predictions."""
    delta = _validated_delta(cpu, rngd, "cpu", "rngd")
    return {
        "mae": float(delta.mean()),
        "rmse": float(delta.square().mean().sqrt()),
        "max_abs_error": float(delta.max()),
    }


def percentage_degradation(cpu_metric: float, rngd_metric: float) -> float | None:
    """Return positive percentage when a device metric is worse than CPU's."""
    if cpu_metric == 0.0:
        return None
    return (rngd_metric / cpu_metric - 1.0) * 100.0


def evaluate_prepared_windows(
    cpu_core: torch.nn.Module,
    adapter: Any,
    contexts: torch.Tensor,
    targets: torch.Tensor,
    device_runner: Callable[[tuple[torch.Tensor]], torch.Tensor],
) -> dict[str, object]:
    """Evaluate CPU and device cores from the same host-prepared windows."""
    if not isinstance(cpu_core, torch.nn.Module):
        raise ValueError("cpu_core must be a torch Module")
    _validate_window_tensor(
        contexts, "contexts", (None, adapter.contract.external_input.shape[1], 1)
    )
    _validate_window_tensor(
        targets, "targets", (contexts.shape[0], adapter.contract.core_output.shape[1], 1)
    )

    cpu_predictions: list[torch.Tensor] = []
    rngd_predictions: list[torch.Tensor] = []
    for context in contexts:
        prepared = adapter.prepare(context.unsqueeze(0))
        with torch.inference_mode():
            cpu_output = cpu_core(prepared.past_values)
            rngd_output = device_runner((prepared.past_values,))
        _validate_core_forecast(cpu_output, adapter.contract.core_output.shape, "CPU")
        _validate_core_forecast(rngd_output, adapter.contract.core_output.shape, "RNGD")
        cpu_predictions.append(prepared.restore(cpu_output).squeeze(0).detach().cpu())
        rngd_predictions.append(
            prepared.restore(rngd_output.detach().cpu()).squeeze(0).detach().cpu()
        )

    cpu = torch.stack(cpu_predictions)
    rngd = torch.stack(rngd_predictions)
    return {
        "cpu_predictions": cpu,
        "rngd_predictions": rngd,
        "cpu_task": forecast_metrics(cpu, targets),
        "rngd_task": forecast_metrics(rngd, targets),
        "prediction_delta": prediction_delta_metrics(cpu, rngd),
    }


def _validated_delta(
    left: torch.Tensor, right: torch.Tensor, left_name: str, right_name: str
) -> torch.Tensor:
    if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
        raise ValueError(f"{left_name} and {right_name} must be torch Tensors")
    if tuple(left.shape) != tuple(right.shape):
        raise ValueError(f"{left_name} and {right_name} shapes must match")
    if left.dtype != torch.float32 or right.dtype != torch.float32:
        raise ValueError(f"{left_name} and {right_name} must use float32")
    if not bool(torch.isfinite(left).all()) or not bool(torch.isfinite(right).all()):
        raise ValueError(f"{left_name} and {right_name} must be finite")
    return (left - right).abs()


def _validate_window_tensor(
    tensor: torch.Tensor, name: str, shape: tuple[int | None, int, int]
) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(f"{name} must be a torch Tensor")
    if tensor.ndim != 3 or tuple(tensor.shape[1:]) != shape[1:]:
        raise ValueError(f"{name} shape does not match the fixed ETTh1 contract")
    if shape[0] is not None and tensor.shape[0] != shape[0]:
        raise ValueError(f"{name} window count does not match")
    if tensor.dtype != torch.float32 or not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} must be finite float32")


def _validate_core_forecast(
    forecast: torch.Tensor, expected_shape: tuple[int, ...], source: str
) -> None:
    if not isinstance(forecast, torch.Tensor):
        raise ValueError(f"{source} forecast must be a torch Tensor")
    if tuple(forecast.shape) != expected_shape:
        raise ValueError(f"{source} forecast shape mismatch: {tuple(forecast.shape)}")
    if forecast.dtype != torch.float32 or not bool(torch.isfinite(forecast).all()):
        raise ValueError(f"{source} forecast must be finite float32")
