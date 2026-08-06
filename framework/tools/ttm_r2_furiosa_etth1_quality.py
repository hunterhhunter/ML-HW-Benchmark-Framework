#!/usr/bin/env python3
"""Measure TTM-R2 ETTh1 quality on CPU and strict Furiosa RNGD."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
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


_RESULT_NAME = "furiosa-etth1-quality-result.json"
_PREDICTIONS_NAME = "furiosa-etth1-quality-predictions.npz"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--dataset-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--windows", type=int, default=240)
    parser.add_argument("--strict-parity-result", type=Path)
    return parser


def compile_furiosa_runner(
    cpu_core: torch.nn.Module, contract: Any
) -> Callable[[tuple[torch.Tensor]], torch.Tensor]:
    """Compile one deepcopy once; all window calls reuse the strict artifact."""
    import furiosa.torch as furiosa_torch
    from furiosa.torch.config import CompilerConfig, TacticHintConfig

    device = torch.device("furiosa:0")
    device_core = deepcopy(cpu_core).eval().to(device)
    backend = furiosa_torch.backend.with_config(
        CompilerConfig(tactic_hint=TacticHintConfig.Default), eager_fallback=False
    )
    compiled = torch.compile(device_core, backend=backend, fullgraph=True, dynamic=False)

    def run(inputs: tuple[torch.Tensor]) -> torch.Tensor:
        if len(inputs) != 1:
            raise ValueError("Furiosa TTM-R2 quality runner expects one core input")
        value = inputs[0]
        expected = contract.core_inputs[0]
        if tuple(value.shape) != expected.shape or value.dtype != torch.float32:
            raise ValueError("Furiosa TTM-R2 quality input does not match fixed ABI")
        with torch.inference_mode():
            output = compiled(value.to(device))
        if isinstance(output, (tuple, list)):
            if len(output) != 1:
                raise ValueError("Furiosa TTM-R2 quality runner returned multiple outputs")
            output = output[0]
        if tuple(output.shape) != contract.core_output.shape:
            raise ValueError(f"Furiosa TTM-R2 quality output shape mismatch: {tuple(output.shape)}")
        if output.dtype != torch.float32 or not bool(torch.isfinite(output).all()):
            raise ValueError("Furiosa TTM-R2 quality output must be finite float32")
        return output

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
    payload = json.loads(path.read_text(encoding="utf-8"))
    status = payload.get("status")
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
    model_path: Path, dataset_path: Path, windows: int, strict_result: Path | None
) -> tuple[dict[str, object], dict[str, torch.Tensor]]:
    model = load_ttm_r2_model(str(model_path))
    cpu_core = TTMR2Core(model).eval()
    adapter = TTMR2HostAdapter(cpu_core.contract, split_ttm_scaler=True)
    contexts, targets, split = load_etth1_windows(ETTh1QualityConfig(dataset_path=dataset_path, windows=windows))
    metrics = evaluate_prepared_windows(
        cpu_core, adapter, contexts, targets, compile_furiosa_runner(cpu_core, cpu_core.contract)
    )
    cpu_task, rngd_task = metrics["cpu_task"], metrics["rngd_task"]
    if not isinstance(cpu_task, dict) or not isinstance(rngd_task, dict):
        raise ValueError("TTM-R2 ETTh1 evaluator did not return task metrics")
    result: dict[str, object] = {
        "status": "measured", "vendor": "furiosa", "runtime_success": True,
        "task_quality_status": "measured", "strict_parity_status": _strict_parity_status(strict_result),
        "compile_mode": {"fullgraph": True, "dynamic": False, "eager_fallback": False},
        "dataset": {"path": str(dataset_path.resolve()), "sha256": _sha256(dataset_path), "column": "OT", "split": split},
        "contract": _describe_contract(cpu_core.contract),
        "checkpoint": _checkpoint_evidence(model_path),
        "cpu_task": cpu_task, "rngd_task": rngd_task, "prediction_delta": metrics["prediction_delta"],
        "degradation_percent": {name: percentage_degradation(float(cpu_task[name]), float(rngd_task[name])) for name in ("mae", "rmse")},
    }
    predictions = {
        name: metrics[name].detach().cpu()
        for name in ("cpu_predictions", "rngd_predictions")
    }
    predictions["targets"] = targets.detach().cpu()
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
            args.model_path, args.dataset_path, args.windows, args.strict_parity_result
        )
        _write_predictions(args.output_dir / _PREDICTIONS_NAME, predictions)
        print(write_result(destination, result))
    except Exception as error:
        if not destination.exists():
            write_result(destination, {
                "status": "failed", "vendor": "furiosa", "runtime_success": False,
                "task_quality_status": "not_measured",
                "error": {"type": type(error).__name__, "message": str(error)},
            })
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
