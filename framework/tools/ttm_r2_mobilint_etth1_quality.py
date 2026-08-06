#!/usr/bin/env python3
"""Run a calibrated TTM-R2 MXQ ETTh1 quality evaluation on remote ARIES."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Sequence

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
from ttm_r1.mobilint_aries import quantize_core_input, restore_artifact_output
from ttm_r2.core import TTMR2Core, load_ttm_r2_model
from ttm_r2.host_adapter import TTMR2HostAdapter


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Measure calibrated TTM-R2 MXQ ETTh1 quality on ARIES")
    for name in ("model-path", "dataset-path", "artifact", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--compile-result", type=Path)
    parser.add_argument("--windows", type=int, default=240)
    return parser


def build_aries_runner(model: Any):
    input_shape = tuple(model.get_model_input_shape()[0])
    output_shape = tuple(model.get_model_output_shape()[0])
    scale = model.get_input_scale()[0]

    def run(core_input: np.ndarray) -> tuple[np.ndarray, int]:
        value, saturated = quantize_core_input(core_input, input_shape, scale)
        raw = model.infer_to_float([value])[0]
        return restore_artifact_output(raw, output_shape), saturated

    return run


def describe_scale(scale: Any) -> dict[str, object]:
    return {
        "scale": float(scale.scale),
        "is_uniform": bool(scale.is_uniform),
        "scale_list": [float(value) for value in scale.scale_list],
        "zero_point": int(scale.zero_point),
        "is_asymmetric": bool(scale.is_asymmetric),
        "zero_points": [int(value) for value in scale.zero_points],
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(args: argparse.Namespace) -> Path:
    import qbruntime

    if 0 not in qbruntime.get_available_device_numbers():
        raise RuntimeError("ARIES device 0 is unavailable")
    if not args.artifact.is_file():
        raise ValueError(f"MXQ artifact is missing: {args.artifact}")
    model = load_ttm_r2_model(str(args.model_path))
    core = TTMR2Core(model).eval()
    adapter = TTMR2HostAdapter(core.contract, split_ttm_scaler=True)
    contexts, targets, split = load_etth1_windows(
        ETTh1QualityConfig(args.dataset_path, windows=args.windows)
    )
    runtime = qbruntime.load(str(args.artifact))
    runner = build_aries_runner(runtime)
    saturation = 0
    runtime_abi = {
        "input_shape": list(runtime.get_model_input_shape()[0]),
        "output_shape": list(runtime.get_model_output_shape()[0]),
        "input_scale": describe_scale(runtime.get_input_scale()[0]),
    }

    def device(inputs: tuple[torch.Tensor]) -> torch.Tensor:
        nonlocal saturation
        output, count = runner(inputs[0].detach().cpu().numpy())
        saturation += count
        return torch.from_numpy(output)

    try:
        quality = evaluate_prepared_windows(core, adapter, contexts, targets, device)
    finally:
        runtime.dispose()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "mobilint-etth1-quality-predictions.npz",
        cpu_predictions=quality["cpu_predictions"].numpy(),
        aries_predictions=quality["rngd_predictions"].numpy(),
        targets=targets.numpy(),
    )
    result: dict[str, object] = {
        "status": "measured",
        "vendor": "mobilint",
        "runtime_success": True,
        "task_quality_status": "measured",
        "quantization_status": "saturated" if saturation else "unsaturated",
        "saturation": {"elements": saturation, "total": args.windows * 512},
        "artifact": {
            "path": str(args.artifact.resolve()),
            "sha256": _sha256(args.artifact),
            "size_bytes": args.artifact.stat().st_size,
        },
        "runtime_abi": runtime_abi,
        "dataset": {"path": str(args.dataset_path.resolve()), "sha256": _sha256(args.dataset_path), "column": "OT", "split": split},
        "cpu_task": quality["cpu_task"],
        "aries_task": quality["rngd_task"],
        "prediction_delta": quality["prediction_delta"],
        "degradation_percent": {
            name: percentage_degradation(quality["cpu_task"][name], quality["rngd_task"][name])
            for name in ("mae", "rmse")
        },
    }
    if args.compile_result is not None:
        if not args.compile_result.is_file():
            raise ValueError(f"compile result is missing: {args.compile_result}")
        result["compile_result"] = json.loads(args.compile_result.read_text(encoding="utf-8"))
    return write_result(output_dir / "mobilint-etth1-quality-result.json", result)


def main(argv: Sequence[str] | None = None) -> int:
    print(run(build_parser().parse_args(argv)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
