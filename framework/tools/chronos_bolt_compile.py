#!/usr/bin/env python3
"""Reference parity and vendor dispatch for a fixed Chronos-Bolt core."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

if __package__ in {None, ""}:
    _FRAMEWORK_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_FRAMEWORK_ROOT))
    sys.path.insert(0, str(_FRAMEWORK_ROOT / "src"))

from chronos_bolt.contracts import ChronosBoltContract
from chronos_bolt.core import ChronosBoltTransformerCore, load_chronos_bolt_model
from chronos_bolt.evidence import write_result
from chronos_bolt.host_adapter import ChronosBoltHostAdapter


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compile or validate the fixed Chronos-Bolt Tiny Transformer core."
    )
    parser.add_argument(
        "--vendor", required=True, choices=("reference", "rbln", "furiosa", "mobilint")
    )
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--describe", action="store_true")
    return parser


def describe_contract(contract: ChronosBoltContract | None = None) -> dict[str, object]:
    """Serialize the concrete Tiny ABI without importing a vendor SDK."""
    contract = contract or ChronosBoltContract.tiny(d_model=256, use_reg_token=True)
    return {
        "external_input": {
            "name": contract.external_input.name,
            "shape": list(contract.external_input.shape),
            "dtype": contract.external_input.dtype,
        },
        "external_output": {
            "name": contract.external_output.name,
            "shape": list(contract.external_output.shape),
            "dtype": contract.external_output.dtype,
        },
        "core_input_names": [item.name for item in contract.core_inputs],
        "core_inputs": [
            {"name": item.name, "shape": list(item.shape), "dtype": item.dtype}
            for item in contract.core_inputs
        ],
        "core_output": {
            "name": contract.core_output.name,
            "shape": list(contract.core_output.shape),
            "dtype": contract.core_output.dtype,
        },
        "checkpoint_uses_reg_token": contract.use_reg_token,
    }


def _reference_contexts(torch_module):
    finite = torch_module.linspace(-1.0, 1.0, 512, dtype=torch_module.float32).reshape(1, 512)
    nan_context = finite.clone()
    nan_context[:, 9::31] = torch_module.nan
    return {"finite": finite, "nan": nan_context}


@dataclass(frozen=True)
class ReferencePreflight:
    """Reference parity evidence and the finite static tensors for a device run."""

    core: ChronosBoltTransformerCore
    contract: ChronosBoltContract
    core_inputs: dict[str, tuple[object, object, object]]
    core_outputs: dict[str, object]
    parity: dict[str, dict[str, object]]


def run_preflight(model_path: Path) -> ReferencePreflight:
    """Prove the split matches the original model before compiling any vendor path."""
    if not model_path.is_dir():
        raise ValueError(f"--model-path must be a local checkpoint directory: {model_path}")
    import torch

    model = load_chronos_bolt_model(str(model_path))
    adapter = ChronosBoltHostAdapter(model)
    core = ChronosBoltTransformerCore(model).eval()
    parity: dict[str, dict[str, object]] = {}
    core_inputs: dict[str, tuple[object, object, object]] = {}
    core_outputs: dict[str, object] = {}
    with torch.no_grad():
        for name, context in _reference_contexts(torch).items():
            full = model(context=context).quantile_preds.to(dtype=torch.float32)
            prepared = adapter.prepare(context)
            normalized = core(
                prepared.input_embeds,
                prepared.attention_mask,
                prepared.decoder_input_embeds,
            )
            split = prepared.restore(normalized)
            if not bool(torch.isfinite(full).all()):
                raise ValueError(f"full-model output contains non-finite values for {name}")
            if not bool(torch.isfinite(split).all()):
                raise ValueError(f"split-core output contains non-finite values for {name}")
            torch.testing.assert_close(split, full, rtol=1e-5, atol=1e-6)
            delta = (split - full).abs()
            parity[name] = {
                "shape": list(split.shape),
                "dtype": str(split.dtype).removeprefix("torch."),
                "max_abs_error": float(delta.max()),
                "mean_abs_error": float(delta.mean()),
            }
            core_inputs[name] = (
                prepared.input_embeds,
                prepared.attention_mask,
                prepared.decoder_input_embeds,
            )
            core_outputs[name] = normalized
    if set(core_inputs) != {"finite", "nan"} or set(core_outputs) != {"finite", "nan"}:
        raise RuntimeError("reference preflight did not produce both required examples")
    return ReferencePreflight(
        core=core,
        contract=core.contract,
        core_inputs=core_inputs,
        core_outputs=core_outputs,
        parity=parity,
    )


def run_reference(model_path: Path, output_dir: Path) -> Path:
    """Write CPU reference parity evidence without requiring a vendor SDK."""
    preflight = run_preflight(model_path)
    return write_result(
        output_dir / "reference-result.json",
        {
            "status": "compiled",
            "vendor": "reference",
            "model_path": str(model_path.resolve()),
            "contract": describe_contract(preflight.contract),
            "parity": preflight.parity,
        },
    )


def _compare_device_output(expected, actual, *, vendor: str) -> dict[str, float | int | bool]:
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


class DeviceParityError(RuntimeError):
    """The artifact ran on-device, but its numeric gate remains unverified."""


def run_rbln(model_path: Path, output_dir: Path) -> Path:
    """Compile, inspect, and execute one RBLN core artifact after CPU parity."""
    import numpy as np

    from tools.chronos_bolt_vendors.rbln import compile_rbln, run_rbln as execute_rbln

    preflight = run_preflight(model_path)
    artifact = output_dir / "chronos-bolt-tiny-core.rbln"
    compiled = compile_rbln(preflight.core, preflight.contract, artifact)
    import torch

    device_parity = {}
    for name, core_inputs in preflight.core_inputs.items():
        inputs = tuple(
            value.detach().cpu().numpy().astype(np.float32, copy=False)
            for value in core_inputs
        )
        device_output = execute_rbln(artifact, inputs, preflight.contract)
        device_parity[name] = _compare_device_output(
            preflight.core_outputs[name],
            torch.from_numpy(device_output),
            vendor="rbln",
        )
    verified = all(result["within_tolerance"] for result in device_parity.values())
    result = write_result(
        output_dir / "rbln-result.json",
        {
            "status": "device_verified" if verified else "parity_failed",
            "vendor": "rbln",
            "model_path": str(model_path.resolve()),
            "contract": describe_contract(preflight.contract),
            "reference_parity": preflight.parity,
            "artifact": compiled["artifact"],
            "inspection": compiled["inspection"],
            "device_core_parity": device_parity,
        },
    )
    if not verified:
        raise DeviceParityError(
            "RBLN artifact compiled and ran on CA22, but numeric parity failed; "
            f"inspect {result}"
        )
    return result


def run_furiosa(model_path: Path, output_dir: Path) -> Path:
    """Strict-compile and execute the core on RNGD after CPU parity."""
    from tools.chronos_bolt_vendors.furiosa import run_furiosa as execute_furiosa

    preflight = run_preflight(model_path)
    device_output = execute_furiosa(
        preflight.core,
        preflight.core_inputs["finite"],
        preflight.contract,
    )
    comparison = _compare_device_output(
        preflight.core_outputs["finite"],
        device_output,
        vendor="furiosa",
    )
    result = write_result(
        output_dir / "furiosa-result.json",
        {
            "status": "device_verified",
            "vendor": "furiosa",
            "model_path": str(model_path.resolve()),
            "contract": describe_contract(preflight.contract),
            "reference_parity": preflight.parity,
            "compile_mode": {
                "fullgraph": True,
                "dynamic": False,
                "eager_fallback": False,
            },
            "device_core_parity": comparison,
        },
    )
    if not comparison["within_tolerance"]:
        raise DeviceParityError(
            "Furiosa strict graph ran on RNGD, but numeric parity failed; "
            f"inspect {result}"
        )
    return result


def run_mobilint(model_path: Path, output_dir: Path) -> Path:
    """Export, compile, and run the same static core on local ARIES hardware."""
    import numpy as np
    import torch

    from tools.chronos_bolt_vendors.mobilint import (
        compile_mblt,
        export_core_onnx,
        run_mblt,
    )

    preflight = run_preflight(model_path)
    onnx_path = output_dir / "chronos-bolt-tiny-core.onnx"
    artifact = output_dir / "chronos-bolt-tiny-core.mblt"
    export_core_onnx(
        preflight.core,
        preflight.core_inputs["finite"],
        preflight.contract,
        onnx_path,
    )
    compiled = compile_mblt(onnx_path, artifact)
    device_parity = {}
    for name, core_inputs in preflight.core_inputs.items():
        inputs = tuple(
            value.detach().cpu().numpy().astype(np.float32, copy=False)
            for value in core_inputs
        )
        device_output = run_mblt(artifact, inputs, preflight.contract)
        device_parity[name] = _compare_device_output(
            preflight.core_outputs[name],
            torch.from_numpy(device_output),
            vendor="mobilint",
        )
    verified = all(result["within_tolerance"] for result in device_parity.values())
    result = write_result(
        output_dir / "mobilint-result.json",
        {
            "status": "device_verified" if verified else "parity_failed",
            "vendor": "mobilint",
            "model_path": str(model_path.resolve()),
            "contract": describe_contract(preflight.contract),
            "reference_parity": preflight.parity,
            "onnx": {
                "path": str(onnx_path.resolve()),
                "size_bytes": onnx_path.stat().st_size,
            },
            "artifact": compiled["artifact"],
            "target_device": compiled["target_device"],
            "device_core_parity": device_parity,
        },
    )
    if not verified:
        raise DeviceParityError(
            "Mobilint artifact compiled and ran on ARIES, but numeric parity failed; "
            f"inspect {result}"
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
    if args.vendor == "reference":
        print(run_reference(args.model_path, args.output_dir))
        return 0
    if args.vendor == "rbln":
        print(run_rbln(args.model_path, args.output_dir))
        return 0
    if args.vendor == "furiosa":
        print(run_furiosa(args.model_path, args.output_dir))
        return 0
    if args.vendor == "mobilint":
        print(run_mobilint(args.model_path, args.output_dir))
        return 0
    raise RuntimeError(f"unsupported vendor dispatch: {args.vendor}")


if __name__ == "__main__":
    raise SystemExit(main())
