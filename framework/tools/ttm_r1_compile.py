#!/usr/bin/env python3
"""Reference parity and strict vendor dispatch for fixed-shape TTM-R1."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

if __package__ in {None, ""}:
    _FRAMEWORK_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_FRAMEWORK_ROOT))
    sys.path.insert(0, str(_FRAMEWORK_ROOT / "src"))

from chronos_bolt.contracts import CompileStatus
from chronos_bolt.evidence import write_result
from ttm_r1.contracts import TTMR1Contract
from ttm_r1.reference import run_preflight


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compile or validate the fixed IBM Granite TTM-R1 core."
    )
    parser.add_argument(
        "--vendor", required=True, choices=("reference", "rbln", "furiosa", "mobilint")
    )
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--describe", action="store_true")
    return parser


def describe_contract(contract: TTMR1Contract | None = None) -> dict[str, object]:
    """Serialize the static ABI without loading a model or vendor SDK."""
    contract = contract or TTMR1Contract.fixed()
    return {
        "external_input": _tensor_description(contract.external_input),
        "external_output": _tensor_description(contract.external_output),
        "core_input_names": [item.name for item in contract.core_inputs],
        "core_inputs": [_tensor_description(item) for item in contract.core_inputs],
        "core_output": _tensor_description(contract.core_output),
    }


def _tensor_description(tensor) -> dict[str, object]:
    return {"name": tensor.name, "shape": list(tensor.shape), "dtype": tensor.dtype}


class DeviceParityError(RuntimeError):
    """An artifact ran on hardware but did not meet the shared numeric gate."""


def compare_device_output(expected, actual) -> dict[str, float | int | bool]:
    """Record deterministic FP32 parity evidence without hiding mismatches."""
    import torch

    expected = expected.detach().cpu().to(dtype=torch.float32)
    actual = actual.detach().cpu().to(dtype=torch.float32)
    delta = (actual - expected).abs()
    close = torch.isclose(actual, expected, rtol=1e-3, atol=1e-3)
    return {
        "max_abs_error": float(delta.max()),
        "mean_abs_error": float(delta.mean()),
        "rmse": float(delta.square().mean().sqrt()),
        "mismatched_elements": int((~close).sum()),
        "within_tolerance": bool(close.all()),
    }


def _device_failure(
    output_dir: Path,
    vendor: str,
    model_path: Path,
    contract: TTMR1Contract,
    error: Exception,
) -> None:
    """Persist a compiler/runtime exception before propagating it to the caller."""
    destination = output_dir / f"{vendor}-result.json"
    if not destination.exists():
        write_result(
            destination,
            {
                "status": CompileStatus.COMPILE_FAILED.value,
                "vendor": vendor,
                "model_path": str(model_path.resolve()),
                "contract": describe_contract(contract),
                "error": {"type": type(error).__name__, "message": str(error)},
            },
        )


def run_reference(model_path: Path, output_dir: Path) -> Path:
    """Write strict CPU preflight evidence before invoking a device compiler."""
    preflight = run_preflight(model_path)
    return write_result(
        output_dir / "reference-result.json",
        {
            "status": CompileStatus.COMPILED.value,
            "vendor": "reference",
            "model_path": str(model_path.resolve()),
            "contract": describe_contract(preflight.contract),
            "host_parity": preflight.host_parity,
        },
    )


def run_rbln(model_path: Path, output_dir: Path) -> Path:
    """Compile, execute, and record fixed-shape CA22 evidence."""
    import numpy as np
    import torch

    from tools.ttm_r1_vendors.rbln import compile_rbln, run_rbln_artifact

    preflight = run_preflight(model_path)
    artifact = output_dir / "ttm-r1-core.rbln"
    try:
        compiled = compile_rbln(preflight.core, preflight.contract, artifact)
        parity = {}
        for name, values in preflight.core_inputs.items():
            output = run_rbln_artifact(
                artifact,
                tuple(value.detach().cpu().numpy().astype(np.float32, copy=False) for value in values),
                preflight.contract,
            )
            parity[name] = compare_device_output(
                preflight.core_outputs[name], torch.from_numpy(output)
            )
    except Exception as error:
        _device_failure(output_dir, "rbln", model_path, preflight.contract, error)
        raise
    verified = all(result["within_tolerance"] for result in parity.values())
    result = write_result(
        output_dir / "rbln-result.json",
        {
            "status": CompileStatus.DEVICE_VERIFIED.value if verified else CompileStatus.PARITY_FAILED.value,
            "vendor": "rbln",
            "model_path": str(model_path.resolve()),
            "contract": describe_contract(preflight.contract),
            "host_parity": preflight.host_parity,
            "artifact": compiled["artifact"],
            "inspection": compiled["inspection"],
            "device_core_parity": parity,
        },
    )
    if not verified:
        raise DeviceParityError(
            f"RBLN artifact compiled and ran on CA22, but numeric parity failed; inspect {result}"
        )
    return result


def run_furiosa(model_path: Path, output_dir: Path) -> Path:
    """Strict-compile and execute the TTM core once on RNGD."""
    from tools.ttm_r1_vendors.furiosa import run_furiosa_core

    preflight = run_preflight(model_path)
    try:
        output = run_furiosa_core(
            preflight.core, preflight.core_inputs["finite"], preflight.contract
        )
        parity = compare_device_output(preflight.core_outputs["finite"], output)
    except Exception as error:
        _device_failure(output_dir, "furiosa", model_path, preflight.contract, error)
        raise
    result = write_result(
        output_dir / "furiosa-result.json",
        {
            "status": CompileStatus.DEVICE_VERIFIED.value
            if parity["within_tolerance"]
            else CompileStatus.PARITY_FAILED.value,
            "vendor": "furiosa",
            "model_path": str(model_path.resolve()),
            "contract": describe_contract(preflight.contract),
            "host_parity": preflight.host_parity,
            "compile_mode": {"fullgraph": True, "dynamic": False, "eager_fallback": False},
            "device_core_parity": {"finite": parity},
        },
    )
    if not parity["within_tolerance"]:
        raise DeviceParityError(
            f"Furiosa strict graph ran on RNGD, but numeric parity failed; inspect {result}"
        )
    return result


def run_mobilint(model_path: Path, output_dir: Path) -> Path:
    """Export, compile, execute, and record fixed-shape ARIES evidence."""
    import numpy as np
    import torch

    from tools.ttm_r1_vendors.mobilint import (
        compile_mblt,
        export_core_onnx,
        run_mblt,
        run_onnx_reference,
    )

    preflight = run_preflight(model_path)
    onnx_path = output_dir / "ttm-r1-core.onnx"
    artifact = output_dir / "ttm-r1-core.mblt"
    try:
        export_core_onnx(
            preflight.core, preflight.core_inputs["finite"], preflight.contract, onnx_path
        )
        onnx_parity = {}
        for name, values in preflight.core_inputs.items():
            onnx_output = run_onnx_reference(
                onnx_path,
                tuple(value.detach().cpu().numpy().astype(np.float32, copy=False) for value in values),
                preflight.contract,
            )
            onnx_parity[name] = compare_device_output(
                preflight.core_outputs[name], torch.from_numpy(onnx_output)
            )
        if not all(result["within_tolerance"] for result in onnx_parity.values()):
            raise DeviceParityError("ONNX Runtime CPU output did not meet the shared TTM-R1 parity gate")
        compiled = compile_mblt(onnx_path, artifact)
        parity = {}
        for name, values in preflight.core_inputs.items():
            output = run_mblt(
                artifact,
                tuple(value.detach().cpu().numpy().astype(np.float32, copy=False) for value in values),
                preflight.contract,
            )
            parity[name] = compare_device_output(
                preflight.core_outputs[name], torch.from_numpy(output)
            )
    except Exception as error:
        _device_failure(output_dir, "mobilint", model_path, preflight.contract, error)
        raise
    verified = all(result["within_tolerance"] for result in parity.values())
    result = write_result(
        output_dir / "mobilint-result.json",
        {
            "status": CompileStatus.DEVICE_VERIFIED.value if verified else CompileStatus.PARITY_FAILED.value,
            "vendor": "mobilint",
            "model_path": str(model_path.resolve()),
            "contract": describe_contract(preflight.contract),
            "host_parity": preflight.host_parity,
            "onnx": {"path": str(onnx_path.resolve()), "size_bytes": onnx_path.stat().st_size},
            "onnx_core_parity": onnx_parity,
            "artifact": compiled["artifact"],
            "target_device": compiled["target_device"],
            "device_core_parity": parity,
        },
    )
    if not verified:
        raise DeviceParityError(
            f"Mobilint artifact compiled and ran on ARIES, but numeric parity failed; inspect {result}"
        )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.describe:
        print(json.dumps(describe_contract(), sort_keys=True))
        return 0
    if args.model_path is None:
        raise ValueError("--model-path is required for execution")
    if args.output_dir is None:
        raise ValueError("--output-dir is required for execution")
    dispatch = {
        "reference": run_reference,
        "rbln": run_rbln,
        "furiosa": run_furiosa,
        "mobilint": run_mobilint,
    }
    print(dispatch[args.vendor](args.model_path, args.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
