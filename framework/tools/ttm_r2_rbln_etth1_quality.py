#!/usr/bin/env python3
"""Measure TTM-R2 ETTh1 quality on CPU and an existing RBLN-CA22 artifact."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

if __package__ in {None, ""}:
    root = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(root), str(root / "src")]

import numpy as np
import torch

from chronos_bolt.evidence import write_result
from ttm_r1.etth1_quality import (
    ETTh1QualityConfig,
    evaluate_prepared_windows,
    load_etth1_windows,
    percentage_degradation,
)
from ttm_r2.core import TTMR2Core, load_ttm_r2_model
from ttm_r2.host_adapter import TTMR2HostAdapter


_RESULT_NAME = "rbln-etth1-quality-result.json"
_PREDICTIONS_NAME = "rbln-etth1-quality-predictions.npz"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--dataset-path", required=True, type=Path)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--windows", type=int, default=240)
    parser.add_argument("--strict-parity-result", type=Path)
    return parser


def _rbln_runner(artifact: Path, contract: Any) -> Callable[[tuple[torch.Tensor]], torch.Tensor]:
    if not artifact.is_file():
        raise ValueError(f"RBLN artifact is missing: {artifact}")
    try:
        rebel = importlib.import_module("rebel")
    except (ImportError, ModuleNotFoundError) as error:
        raise ImportError("RBLN ETTh1 quality requires the rebel-compiler SDK") from error
    runtime = rebel.Runtime(str(artifact.resolve()), device=0, tensor_type="np", timeout=60)

    def run(inputs: tuple[torch.Tensor]) -> torch.Tensor:
        if len(inputs) != 1:
            raise ValueError("RBLN TTM-R2 quality runner expects one core input")
        value = inputs[0]
        expected = contract.core_inputs[0]
        if tuple(value.shape) != expected.shape or value.dtype != torch.float32:
            raise ValueError("RBLN TTM-R2 quality input does not match fixed ABI")
        raw = runtime(np.ascontiguousarray(value.detach().cpu().numpy()))
        if isinstance(raw, (tuple, list)):
            if len(raw) != 1:
                raise ValueError("RBLN TTM-R2 quality runner returned multiple outputs")
            raw = raw[0]
        output = np.asarray(raw)
        if output.shape != contract.core_output.shape:
            raise ValueError(f"RBLN TTM-R2 quality output shape mismatch: {output.shape}")
        if output.dtype != np.float32 or not np.isfinite(output).all():
            raise ValueError("RBLN TTM-R2 quality output must be finite float32")
        return torch.from_numpy(output.copy())

    return run


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _strict_parity_status(path: Path | None) -> str:
    if path is None:
        return "unknown"
    if not path.is_file():
        raise ValueError(f"strict parity result is missing: {path}")
    status = json.loads(path.read_text(encoding="utf-8")).get("status")
    if not isinstance(status, str):
        raise ValueError("strict parity result has no string status")
    return status


def _describe_contract(contract: Any) -> dict[str, object]:
    def describe(tensor: Any) -> dict[str, object]:
        return {"name": tensor.name, "shape": list(tensor.shape), "dtype": tensor.dtype}

    return {
        "external_input": describe(contract.external_input),
        "external_output": describe(contract.external_output),
        "core_inputs": [describe(item) for item in contract.core_inputs],
        "core_output": describe(contract.core_output),
    }


def _checkpoint_evidence(model_path: Path) -> dict[str, object]:
    if not model_path.is_dir():
        raise ValueError(f"TTM-R2 checkpoint is missing: {model_path}")
    return {
        "path": str(model_path.resolve()),
        "files": {path.name: _sha256(path) for path in sorted(model_path.iterdir()) if path.is_file()},
    }


def run_quality(
    model_path: Path,
    dataset_path: Path,
    artifact: Path,
    windows: int,
    strict_result: Path | None,
) -> tuple[dict[str, object], dict[str, torch.Tensor]]:
    model = load_ttm_r2_model(str(model_path))
    cpu_core = TTMR2Core(model).eval()
    adapter = TTMR2HostAdapter(cpu_core.contract, split_ttm_scaler=True)
    contexts, targets, split = load_etth1_windows(
        ETTh1QualityConfig(dataset_path=dataset_path, windows=windows)
    )
    metrics = evaluate_prepared_windows(
        cpu_core, adapter, contexts, targets, _rbln_runner(artifact, cpu_core.contract)
    )
    cpu_task, ca22_task = metrics["cpu_task"], metrics["rngd_task"]
    if not isinstance(cpu_task, dict) or not isinstance(ca22_task, dict):
        raise ValueError("TTM-R2 ETTh1 evaluator did not return task metrics")
    result: dict[str, object] = {
        "status": "measured",
        "vendor": "rbln",
        "runtime_success": True,
        "task_quality_status": "measured",
        "strict_parity_status": _strict_parity_status(strict_result),
        "contract": _describe_contract(cpu_core.contract),
        "checkpoint": _checkpoint_evidence(model_path),
        "artifact": {
            "path": str(artifact.resolve()),
            "sha256": _sha256(artifact),
            "size_bytes": artifact.stat().st_size,
        },
        "dataset": {
            "path": str(dataset_path.resolve()),
            "sha256": _sha256(dataset_path),
            "column": "OT",
            "split": split,
        },
        "cpu_task": cpu_task,
        "rbln_task": ca22_task,
        "prediction_delta": metrics["prediction_delta"],
        "degradation_percent": {
            name: percentage_degradation(float(cpu_task[name]), float(ca22_task[name]))
            for name in ("mae", "rmse")
        },
        "execution": {"ca22": ["ttm-r2-core"], "cpu": ["host_adapter_scaler", "host_adapter_restore"]},
    }
    predictions = {
        "cpu_predictions": metrics["cpu_predictions"].detach().cpu(),
        "rbln_predictions": metrics["rngd_predictions"].detach().cpu(),
        "targets": targets.detach().cpu(),
    }
    return result, predictions


def _write_predictions(destination: Path, predictions: dict[str, torch.Tensor]) -> None:
    if destination.exists():
        raise FileExistsError(f"predictions already exist: {destination}")
    np.savez_compressed(destination, **{name: value.numpy() for name, value in predictions.items()})


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    destination = args.output_dir / _RESULT_NAME
    try:
        result, predictions = run_quality(
            args.model_path,
            args.dataset_path,
            args.artifact,
            args.windows,
            args.strict_parity_result,
        )
        _write_predictions(args.output_dir / _PREDICTIONS_NAME, predictions)
        print(write_result(destination, result))
    except Exception as error:
        if not destination.exists():
            write_result(destination, {
                "status": "failed",
                "vendor": "rbln",
                "runtime_success": False,
                "task_quality_status": "not_measured",
                "error": {"type": type(error).__name__, "message": str(error)},
            })
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
