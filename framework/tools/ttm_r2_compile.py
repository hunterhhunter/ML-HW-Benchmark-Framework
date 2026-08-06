#!/usr/bin/env python3
"""Reference and strict CA22/RNGD validation for fixed TTM-R2."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

if __package__ in {None, ""}:
    root = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(root), str(root / "src")]

from chronos_bolt.contracts import CompileStatus
from chronos_bolt.evidence import write_result
from ttm_r2.contracts import TTMR2Contract
from ttm_r2.reference import run_preflight


class DeviceParityError(RuntimeError):
    """A strict device artifact executed but missed the numeric gate."""


def _tensor(value) -> dict[str, object]:
    return {"name": value.name, "shape": list(value.shape), "dtype": value.dtype}


def describe_contract(contract: TTMR2Contract | None = None) -> dict[str, object]:
    contract = contract or TTMR2Contract.fixed()
    return {
        "external_input": _tensor(contract.external_input),
        "external_output": _tensor(contract.external_output),
        "core_input_names": [item.name for item in contract.core_inputs],
        "core_inputs": [_tensor(item) for item in contract.core_inputs],
        "core_output": _tensor(contract.core_output),
    }


def compare(expected, actual) -> dict[str, float | int | bool]:
    import torch

    expected = expected.detach().cpu().float()
    actual = actual.detach().cpu().float()
    delta = (actual - expected).abs()
    close = torch.isclose(actual, expected, rtol=1e-3, atol=1e-3)
    return {
        "max_abs_error": float(delta.max()), "mean_abs_error": float(delta.mean()),
        "rmse": float(delta.square().mean().sqrt()), "mismatched_elements": int((~close).sum()),
        "within_tolerance": bool(close.all()),
    }


def _failure(output_dir: Path, vendor: str, model_path: Path, contract, error: Exception) -> None:
    destination = output_dir / f"{vendor}-result.json"
    if not destination.exists():
        write_result(destination, {
            "status": CompileStatus.COMPILE_FAILED.value, "vendor": vendor,
            "model_path": str(model_path.resolve()), "contract": describe_contract(contract),
            "error": {"type": type(error).__name__, "message": str(error)},
        })


def run_reference(model_path: Path, output_dir: Path) -> Path:
    preflight = run_preflight(model_path)
    return write_result(output_dir / "reference-result.json", {
        "status": CompileStatus.COMPILED.value, "vendor": "reference",
        "model_path": str(model_path.resolve()), "contract": describe_contract(preflight.contract),
        "host_parity": preflight.host_parity,
    })


def run_rbln(model_path: Path, output_dir: Path) -> Path:
    import numpy as np
    import torch
    from tools.ttm_r2_vendors.rbln import compile_rbln, run_rbln_artifact

    preflight = run_preflight(model_path)
    try:
        compiled = compile_rbln(preflight.core, preflight.contract, output_dir / "ttm-r2-core.rbln")
        parity = {}
        for name, inputs in preflight.core_inputs.items():
            output = run_rbln_artifact(
                compiled["artifact"]["path"],
                tuple(item.detach().cpu().numpy().astype(np.float32, copy=False) for item in inputs),
                preflight.contract,
            )
            parity[name] = compare(preflight.core_outputs[name], torch.from_numpy(output))
    except Exception as error:
        _failure(output_dir, "rbln", model_path, preflight.contract, error)
        raise
    verified = all(item["within_tolerance"] for item in parity.values())
    result = write_result(output_dir / "rbln-result.json", {
        "status": CompileStatus.DEVICE_VERIFIED.value if verified else CompileStatus.PARITY_FAILED.value,
        "vendor": "rbln", "model_path": str(model_path.resolve()),
        "contract": describe_contract(preflight.contract), "host_parity": preflight.host_parity,
        "artifact": compiled["artifact"], "inspection": compiled["inspection"], "device_core_parity": parity,
    })
    if not verified:
        raise DeviceParityError(f"R2 artifact ran on CA22 but parity failed; inspect {result}")
    return result


def run_furiosa(model_path: Path, output_dir: Path) -> Path:
    from tools.ttm_r2_vendors.furiosa import run_furiosa_core

    preflight = run_preflight(model_path)
    try:
        output = run_furiosa_core(preflight.core, preflight.core_inputs["finite"], preflight.contract)
        parity = compare(preflight.core_outputs["finite"], output)
    except Exception as error:
        _failure(output_dir, "furiosa", model_path, preflight.contract, error)
        raise
    result = write_result(output_dir / "furiosa-result.json", {
        "status": CompileStatus.DEVICE_VERIFIED.value if parity["within_tolerance"] else CompileStatus.PARITY_FAILED.value,
        "vendor": "furiosa", "model_path": str(model_path.resolve()),
        "contract": describe_contract(preflight.contract), "host_parity": preflight.host_parity,
        "compile_mode": {"fullgraph": True, "dynamic": False, "eager_fallback": False},
        "device_core_parity": {"finite": parity},
    })
    if not parity["within_tolerance"]:
        raise DeviceParityError(f"R2 strict graph ran on RNGD but parity failed; inspect {result}")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vendor", choices=("reference", "rbln", "furiosa"), required=True)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--describe", action="store_true")
    args = parser.parse_args(argv)
    if args.describe:
        print(json.dumps(describe_contract(), sort_keys=True))
        return 0
    if args.model_path is None or args.output_dir is None:
        raise ValueError("--model-path and --output-dir are required")
    print({"reference": run_reference, "rbln": run_rbln, "furiosa": run_furiosa}[args.vendor](args.model_path, args.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
