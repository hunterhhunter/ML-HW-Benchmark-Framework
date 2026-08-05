#!/usr/bin/env python3
"""Measure fixed TTM-R1 ETTh1 quality on CPU and strict Furiosa RNGD."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

if __package__ in {None, ""}:
    _FRAMEWORK_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_FRAMEWORK_ROOT))
    sys.path.insert(0, str(_FRAMEWORK_ROOT / "src"))

import numpy as np
import torch

from chronos_bolt.evidence import write_result
from ttm_r1.contracts import TTMR1Contract
from ttm_r1.core import TTMR1Core, load_ttm_r1_model
from ttm_r1.etth1_quality import (
    ETTh1QualityConfig,
    evaluate_prepared_windows,
    load_etth1_windows,
    percentage_degradation,
)
from ttm_r1.host_adapter import TTMR1HostAdapter


_RESULT_NAME = "furiosa-etth1-quality-result.json"
_PREDICTIONS_NAME = "furiosa-etth1-quality-predictions.npz"


def build_parser() -> argparse.ArgumentParser:
    """Build the explicit local-only ETTh1 quality command contract."""
    parser = argparse.ArgumentParser(
        description="Measure fixed TTM-R1 ETTh1 OT quality on CPU and Furiosa RNGD."
    )
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--dataset-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--windows", type=int, default=240)
    parser.add_argument("--strict-parity-result", type=Path)
    return parser


def compile_furiosa_runner(
    cpu_core: torch.nn.Module, contract: TTMR1Contract
) -> Callable[[tuple[torch.Tensor]], torch.Tensor]:
    """Compile a deepcopy once, preserving the supplied CPU reference core."""
    import furiosa.torch as furiosa_torch
    from furiosa.torch.config import CompilerConfig, TacticHintConfig

    device = torch.device("furiosa:0")
    device_core = deepcopy(cpu_core).eval().to(device)
    compiler_config = CompilerConfig(tactic_hint=TacticHintConfig.Default)
    backend = furiosa_torch.backend.with_config(compiler_config, eager_fallback=False)
    compiled = torch.compile(device_core, backend=backend, fullgraph=True, dynamic=False)

    def run(inputs: tuple[torch.Tensor]) -> torch.Tensor:
        if len(inputs) != 1:
            raise ValueError("Furiosa TTM-R1 quality runner expects one core input")
        value = inputs[0]
        expected = contract.core_inputs[0]
        if tuple(value.shape) != expected.shape or value.dtype != torch.float32:
            raise ValueError("Furiosa TTM-R1 quality input does not match fixed ABI")
        with torch.inference_mode():
            output = compiled(value.to(device))
        if isinstance(output, (tuple, list)):
            if len(output) != 1:
                raise ValueError("Furiosa TTM-R1 quality runner returned multiple outputs")
            output = output[0]
        if tuple(output.shape) != contract.core_output.shape:
            raise ValueError(f"Furiosa TTM-R1 quality output shape mismatch: {tuple(output.shape)}")
        if output.dtype != torch.float32 or not bool(torch.isfinite(output).all()):
            raise ValueError("Furiosa TTM-R1 quality output must be finite float32")
        return output

    return run


def run_quality(
    model_path: Path,
    dataset_path: Path,
    windows: int,
    strict_parity_result: Path | None,
) -> tuple[dict[str, object], dict[str, torch.Tensor]]:
    """Run CPU and strict RNGD predictions against the same ETTh1 targets."""
    model = load_ttm_r1_model(str(model_path))
    cpu_core = TTMR1Core(model).eval()
    adapter = TTMR1HostAdapter(cpu_core.contract, split_ttm_scaler=True)
    contexts, targets, split = load_etth1_windows(
        ETTh1QualityConfig(dataset_path=dataset_path, windows=windows)
    )
    runner = compile_furiosa_runner(cpu_core, cpu_core.contract)
    quality = evaluate_prepared_windows(cpu_core, adapter, contexts, targets, runner)
    cpu_task = quality["cpu_task"]
    rngd_task = quality["rngd_task"]
    if not isinstance(cpu_task, dict) or not isinstance(rngd_task, dict):
        raise ValueError("TTM-R1 ETTh1 evaluator did not return task metrics")

    result: dict[str, object] = {
        "status": "measured",
        "vendor": "furiosa",
        "runtime_success": True,
        "task_quality_status": "measured",
        "strict_parity_status": _strict_parity_status(strict_parity_result),
        "compile_mode": {"fullgraph": True, "dynamic": False, "eager_fallback": False},
        "contract": _describe_contract(cpu_core.contract),
        "dataset": {
            "path": str(dataset_path.resolve()),
            "sha256": _sha256(dataset_path),
            "column": "OT",
            "split": split,
        },
        "checkpoint": _checkpoint_evidence(model_path),
        "cpu_task": cpu_task,
        "rngd_task": rngd_task,
        "prediction_delta": quality["prediction_delta"],
        "degradation_percent": {
            name: percentage_degradation(float(cpu_task[name]), float(rngd_task[name]))
            for name in ("mae", "rmse")
        },
    }
    predictions = {
        "cpu_predictions": _as_cpu_tensor(quality["cpu_predictions"], "cpu_predictions"),
        "rngd_predictions": _as_cpu_tensor(quality["rngd_predictions"], "rngd_predictions"),
        "targets": targets.detach().cpu(),
    }
    return result, predictions


def _as_cpu_tensor(value: object, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"TTM-R1 ETTh1 evaluator did not return {name}")
    return value.detach().cpu()


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


def _describe_contract(contract: TTMR1Contract) -> dict[str, object]:
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
        raise ValueError(f"TTM-R1 checkpoint is missing: {model_path}")
    files = [path for path in sorted(model_path.iterdir()) if path.is_file()]
    return {
        "path": str(model_path.resolve()),
        "files": {path.name: _sha256(path) for path in files},
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _write_predictions(destination: Path, predictions: dict[str, torch.Tensor]) -> None:
    if destination.exists():
        raise FileExistsError(f"predictions already exist: {destination}")
    np.savez_compressed(
        destination,
        cpu_predictions=predictions["cpu_predictions"].numpy(),
        rngd_predictions=predictions["rngd_predictions"].numpy(),
        targets=predictions["targets"].numpy(),
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Write immutable runtime evidence and propagate any Furiosa failure."""
    args = build_parser().parse_args(argv)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / _RESULT_NAME
    try:
        result, predictions = run_quality(
            args.model_path, args.dataset_path, args.windows, args.strict_parity_result
        )
        _write_predictions(output_dir / _PREDICTIONS_NAME, predictions)
        print(write_result(result_path, result))
    except Exception as error:
        if not result_path.exists():
            write_result(
                result_path,
                {
                    "status": "failed",
                    "vendor": "furiosa",
                    "runtime_success": False,
                    "task_quality_status": "not_measured",
                    "error": {"type": type(error).__name__, "message": str(error)},
                },
            )
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
