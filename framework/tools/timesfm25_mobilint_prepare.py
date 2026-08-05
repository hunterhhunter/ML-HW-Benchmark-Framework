#!/usr/bin/env python3
"""Prepare fixed ONNX, fixture, and calibration files for local ARIES MXQ compilation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

if __package__ in {None, ""}:
    _ROOT = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(_ROOT), str(_ROOT / "src")]

from chronos_bolt.evidence import write_result
from timesfm25.reference import run_preflight
from tools.timesfm25_compile import compare_device_output, describe_contract
from tools.timesfm25_vendors.mobilint import export_core_onnx, run_onnx_reference


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--calibration-samples", type=int, default=64)
    args = parser.parse_args()
    if args.calibration_samples < 2:
        raise ValueError("--calibration-samples must be at least two")
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"choose a new output directory: {output}")
    output.mkdir(parents=True)
    preflight = run_preflight(args.model_path)
    onnx = export_core_onnx(preflight.core, preflight.core_inputs["normal"], preflight.contract, output / "timesfm25-point-core.onnx")
    onnx_parity = {
        name: compare_device_output(preflight.core_outputs[name], torch.from_numpy(run_onnx_reference(onnx, values[0].numpy(), preflight.contract)))
        for name, values in preflight.core_inputs.items()
    }
    if not all(metric["within_tolerance"] for metric in onnx_parity.values()):
        raise RuntimeError("ONNX CPU parity failed; do not compile this graph for ARIES")
    np.savez_compressed(
        output / "timesfm25-inference-fixture.npz",
        context=preflight.context.numpy(),
        normal_input=preflight.core_inputs["normal"][0].numpy(),
        flipped_input=preflight.core_inputs["flipped"][0].numpy(),
        expected_normal=preflight.core_outputs["normal"].numpy(),
        expected_flipped=preflight.core_outputs["flipped"].numpy(),
        expected_public=preflight.public_output.numpy(),
    )
    calibration = output / "calibration"
    calibration.mkdir()
    rng = np.random.default_rng(20260805)
    for index in range(args.calibration_samples):
        base = rng.standard_normal((1, 1024), dtype=np.float32)
        value = base if index % 2 == 0 else -base
        np.save(calibration / f"calibration-{index:03d}.npy", value)
    result = write_result(output / "prepare-result.json", {
        "status": "prepared", "model_path": str(args.model_path.resolve()),
        "contract": describe_contract(preflight.contract), "host_parity": preflight.host_parity,
        "onnx": {"path": str(onnx), "size_bytes": onnx.stat().st_size}, "onnx_core_parity": onnx_parity,
        "fixture": str(output / "timesfm25-inference-fixture.npz"), "calibration_samples": args.calibration_samples,
    })
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
