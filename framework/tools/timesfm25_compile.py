#!/usr/bin/env python3
"""Strict fixed-shape CPU, RBLN, and Furiosa validation for TimesFM 2.5."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

if __package__ in {None, ""}:
    _ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_ROOT))
    sys.path.insert(0, str(_ROOT / "src"))

from chronos_bolt.contracts import CompileStatus
from chronos_bolt.evidence import write_result
from timesfm25.contracts import TimesFM25Contract
from timesfm25.reference import run_preflight


class DeviceParityError(RuntimeError):
    """A device graph executed but did not meet the explicit numeric gate."""


def _tensor(tensor) -> dict[str, object]:
    return {"name": tensor.name, "shape": list(tensor.shape), "dtype": tensor.dtype}


def describe_contract(contract: TimesFM25Contract | None = None) -> dict[str, object]:
    """Serialize the ABI without loading a checkpoint or vendor SDK."""
    contract = contract or TimesFM25Contract.fixed()
    return {
        "external_input": _tensor(contract.external_input),
        "external_output": _tensor(contract.external_output),
        "core_input_names": [item.name for item in contract.core_inputs],
        "core_inputs": [_tensor(item) for item in contract.core_inputs],
        "core_output": _tensor(contract.core_output),
    }


def compare_device_output(expected, actual) -> dict[str, float | int | bool]:
    """Calculate a fixed FP32 numeric gate without masking any mismatch."""
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


def _host_output(preflight, device_outputs):
    return preflight.adapter.prepare(preflight.context).restore(
        device_outputs["normal"], device_outputs["flipped"]
    )


def _failure(output_dir: Path, vendor: str, model_path: Path, contract, error: Exception) -> None:
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


def _complete_device_result(
    output_dir: Path, vendor: str, model_path: Path, preflight, device_outputs, **extra
) -> Path:
    import torch

    core_parity = {
        name: compare_device_output(preflight.core_outputs[name], value)
        for name, value in device_outputs.items()
    }
    device_public = _host_output(preflight, device_outputs)
    public_parity = compare_device_output(preflight.public_output, device_public)
    verified = all(value["within_tolerance"] for value in core_parity.values()) and public_parity[
        "within_tolerance"
    ]
    payload = {
        "status": CompileStatus.DEVICE_VERIFIED.value
        if verified
        else CompileStatus.PARITY_FAILED.value,
        "vendor": vendor,
        "model_path": str(model_path.resolve()),
        "contract": describe_contract(preflight.contract),
        "host_parity": preflight.host_parity,
        "device_core_parity": core_parity,
        "device_public_parity": public_parity,
    }
    payload.update(extra)
    result = write_result(output_dir / f"{vendor}-result.json", payload)
    if not verified:
        raise DeviceParityError(
            f"{vendor} graph compiled and ran, but numeric parity failed; inspect {result}"
        )
    return result


def run_rbln(model_path: Path, output_dir: Path) -> Path:
    import numpy as np
    import torch

    from tools.timesfm25_vendors.rbln import compile_rbln, run_rbln_artifact

    preflight = run_preflight(model_path)
    try:
        compiled = compile_rbln(preflight.core, preflight.contract, output_dir / "timesfm25-point-core.rbln")
        outputs = {
            name: torch.from_numpy(
                run_rbln_artifact(
                    compiled["artifact"]["path"],
                    (values[0].detach().cpu().numpy().astype(np.float32, copy=False),),
                    preflight.contract,
                )
            )
            for name, values in preflight.core_inputs.items()
        }
    except Exception as error:
        _failure(output_dir, "rbln", model_path, preflight.contract, error)
        raise
    return _complete_device_result(
        output_dir, "rbln", model_path, preflight, outputs,
        artifact=compiled["artifact"], inspection=compiled["inspection"],
    )


def run_furiosa(model_path: Path, output_dir: Path) -> Path:
    from tools.timesfm25_vendors.furiosa import compile_furiosa_runner

    preflight = run_preflight(model_path)
    try:
        runner = compile_furiosa_runner(preflight.core, preflight.contract)
        outputs = {name: runner(values).detach().cpu() for name, values in preflight.core_inputs.items()}
    except Exception as error:
        _failure(output_dir, "furiosa", model_path, preflight.contract, error)
        raise
    return _complete_device_result(
        output_dir,
        "furiosa",
        model_path,
        preflight,
        outputs,
        compile_mode={"fullgraph": True, "dynamic": False, "eager_fallback": False},
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vendor", choices=("reference", "rbln", "furiosa"), required=True)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--describe", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.describe:
        print(json.dumps(describe_contract(), sort_keys=True))
        return 0
    if args.model_path is None or args.output_dir is None:
        raise ValueError("--model-path and --output-dir are required for execution")
    dispatch = {"reference": run_reference, "rbln": run_rbln, "furiosa": run_furiosa}
    print(dispatch[args.vendor](args.model_path, args.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
