#!/usr/bin/env python3
"""Build an ETTh1-train-calibrated ARIES MXQ for the fixed TTM-R1 core."""

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

from chronos_bolt.evidence import write_result
from ttm_r1.core import TTMR1Core, load_ttm_r1_model
from ttm_r1.etth1_quality import (
    ETTh1QualityConfig,
    load_train_calibration_contexts,
    write_calibration_inputs,
)
from ttm_r1.host_adapter import TTMR1HostAdapter
from tools.ttm_r1_vendors.mobilint import export_core_onnx


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build train-calibrated TTM-R1 ARIES MXQ")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--calibration-samples", type=int, default=256)
    return parser


def compile_mxq(
    onnx_path: Path, calibration_dir: Path, output: Path, feed: np.ndarray, *, qbcompiler_module: Any | None = None
) -> None:
    if not onnx_path.is_file() or not calibration_dir.is_dir() or output.exists():
        raise ValueError("ONNX, calibration directory, and new MXQ output are required")
    if feed.shape != (1, 512, 1) or feed.dtype != np.float32:
        raise ValueError("MXQ feed must be float32 [1,512,1]")
    compiler = qbcompiler_module
    if compiler is None:
        import qbcompiler as compiler
    compiler.mxq_compile_V2(
        str(onnx_path), target_device="aries-rb", calib_data_path=str(calibration_dir),
        save_path=str(output), backend="onnx", feed_dict={"past_values": feed},
        device="cpu", cpu_offload=False, use_random_calib=False,
    )
    if not output.is_file() or output.stat().st_size == 0:
        raise ValueError("qbcompiler did not create a nonempty MXQ")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(args: argparse.Namespace) -> Path:
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    onnx, calibration, mxq = output / "ttm-r1-core.onnx", output / "calibration", output / "ttm-r1-core.mxq"
    result = output / "local-aries-compile-result.json"
    if result.exists() or onnx.exists() or mxq.exists():
        raise FileExistsError("output directory already contains TTM-R1 ARIES compile evidence")
    model = load_ttm_r1_model(str(args.model_path))
    core = TTMR1Core(model).eval()
    adapter = TTMR1HostAdapter(core.contract, split_ttm_scaler=True)
    contexts, selection = load_train_calibration_contexts(
        ETTh1QualityConfig(args.dataset_path), args.calibration_samples
    )
    calibration_info = write_calibration_inputs(adapter, contexts, calibration)
    feed = adapter.prepare(contexts[0:1]).past_values.detach().cpu().numpy().astype(np.float32)
    export_core_onnx(core, (feed_to_torch(feed),), core.contract, onnx)
    compile_mxq(onnx, calibration, mxq, feed)
    return write_result(result, {
        "status": "compiled_unvalidated", "vendor": "mobilint", "target_device": "aries-rb",
        "compile_options": {"device": "cpu", "cpu_offload": False, "use_random_calib": False},
        "calibration": {**selection, **calibration_info, "manifest_sha256": _sha256(calibration / "calibration-manifest.json")},
        "dataset": {"path": str(args.dataset_path.resolve()), "sha256": _sha256(args.dataset_path)},
        "artifact": {"path": str(mxq.resolve()), "sha256": _sha256(mxq), "size_bytes": mxq.stat().st_size},
        "onnx": {"path": str(onnx.resolve()), "sha256": _sha256(onnx), "size_bytes": onnx.stat().st_size},
    })


def feed_to_torch(value: np.ndarray):
    import torch
    return torch.from_numpy(value)


def main(argv: Sequence[str] | None = None) -> int:
    print(run(build_parser().parse_args(argv)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
