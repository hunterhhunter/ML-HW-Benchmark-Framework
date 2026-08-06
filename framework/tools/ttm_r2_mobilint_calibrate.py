#!/usr/bin/env python3
"""Build an ETTh1-train-calibrated ARIES MXQ for the fixed TTM-R2 core."""

from __future__ import annotations

import argparse
import hashlib
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
    load_train_calibration_contexts,
    write_calibration_inputs,
)
from tools.ttm_r1_vendors.mobilint import export_core_onnx, run_onnx_reference
from ttm_r2.core import TTMR2Core, load_ttm_r2_model
from ttm_r2.host_adapter import TTMR2HostAdapter


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build train-calibrated TTM-R2 ARIES MXQ")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--calibration-samples", type=int, default=256)
    parser.add_argument("--prepare-only", action="store_true")
    return parser


def compile_mxq(
    onnx_path: Path,
    calibration_dir: Path,
    output: Path,
    feed: np.ndarray,
    *,
    qbcompiler_module: Any | None = None,
) -> None:
    if not onnx_path.is_file() or not calibration_dir.is_dir() or output.exists():
        raise ValueError("ONNX, calibration directory, and new MXQ output are required")
    if feed.shape != (1, 512, 1) or feed.dtype != np.float32:
        raise ValueError("MXQ feed must be float32 [1,512,1]")
    compiler = qbcompiler_module
    if compiler is None:
        import qbcompiler as compiler
    compiler.mxq_compile_V2(
        str(onnx_path),
        target_device="aries-rb",
        calib_data_path=str(calibration_dir),
        save_path=str(output),
        backend="onnx",
        feed_dict={"past_values": feed},
        device="cpu",
        cpu_offload=False,
        use_random_calib=False,
    )
    if not output.is_file() or output.stat().st_size == 0:
        raise ValueError("qbcompiler did not create a nonempty MXQ")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_aries_abi_calibration(source: Path, destination: Path) -> dict[str, object]:
    """Convert semantic `[1,512,1]` samples to QBC's lowered `[1,8,64]` ABI."""
    if not source.is_dir() or destination.exists():
        raise ValueError("source calibration directory and a new ABI directory are required")
    destination.mkdir(parents=True)
    files = []
    for path in sorted(source.glob("calibration-*.npy")):
        value = np.load(path)
        if value.shape != (1, 512, 1) or value.dtype != np.float32:
            raise ValueError(f"semantic calibration tensor has unexpected ABI: {path}")
        target = destination / path.name
        np.save(target, value.reshape(1, 8, 64))
        files.append({"path": target.name, "sha256": _sha256(target)})
    if not files:
        raise ValueError("semantic calibration directory contains no samples")
    return {"samples": len(files), "shape": [1, 8, 64], "files": files}


def run(args: argparse.Namespace) -> Path:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx = output_dir / "ttm-r2-core.onnx"
    calibration = output_dir / "calibration"
    compiler_calibration = output_dir / "calibration-aries-abi"
    mxq = output_dir / "ttm-r2-core.mxq"
    result = output_dir / "local-aries-compile-result.json"
    if result.exists() or onnx.exists() or mxq.exists():
        raise FileExistsError("output directory already contains TTM-R2 ARIES compile evidence")

    model = load_ttm_r2_model(str(args.model_path))
    core = TTMR2Core(model).eval()
    adapter = TTMR2HostAdapter(core.contract, split_ttm_scaler=True)
    contexts, selection = load_train_calibration_contexts(
        ETTh1QualityConfig(args.dataset_path), args.calibration_samples
    )
    calibration_info = write_calibration_inputs(adapter, contexts, calibration)
    compiler_calibration_info = write_aries_abi_calibration(calibration, compiler_calibration)
    feed = adapter.prepare(contexts[0:1]).past_values.detach().cpu().numpy().astype(np.float32)
    feed_tensor = torch.from_numpy(feed)
    with torch.inference_mode():
        expected = core(feed_tensor).detach().cpu().numpy()
    export_core_onnx(core, (feed_tensor,), core.contract, onnx)
    onnx_output = run_onnx_reference(onnx, (feed,), core.contract)
    np.testing.assert_allclose(onnx_output, expected, rtol=1e-5, atol=1e-6)
    payload: dict[str, object] = {
        "status": "prepared",
        "vendor": "mobilint",
        "target_device": "aries-rb",
        "compile_options": {"device": "cpu", "cpu_offload": False, "use_random_calib": False},
        "onnx_cpu_parity": {
            "max_abs_error": float(np.abs(onnx_output - expected).max()),
            "mean_abs_error": float(np.abs(onnx_output - expected).mean()),
        },
        "calibration": {
            **selection,
            **calibration_info,
            "manifest_sha256": _sha256(calibration / "calibration-manifest.json"),
        },
        "compiler_calibration": compiler_calibration_info,
        "dataset": {"path": str(args.dataset_path.resolve()), "sha256": _sha256(args.dataset_path)},
        "onnx": {"path": str(onnx.resolve()), "sha256": _sha256(onnx), "size_bytes": onnx.stat().st_size},
    }
    if args.prepare_only:
        return write_result(result, payload)
    compile_mxq(onnx, compiler_calibration, mxq, feed)
    payload["status"] = "compiled_unvalidated"
    payload["artifact"] = {
        "path": str(mxq.resolve()), "sha256": _sha256(mxq), "size_bytes": mxq.stat().st_size
    }
    return write_result(result, payload)


def main(argv: Sequence[str] | None = None) -> int:
    print(run(build_parser().parse_args(argv)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
