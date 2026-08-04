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
    """Delay importing the proprietary Rebellions SDK until its path is selected."""
    from tools.ttm_r1_vendors.rbln import run_rbln as execute

    return execute(model_path, output_dir)


def run_furiosa(model_path: Path, output_dir: Path) -> Path:
    """Delay importing Furiosa tooling until its path is selected."""
    from tools.ttm_r1_vendors.furiosa import run_furiosa as execute

    return execute(model_path, output_dir)


def run_mobilint(model_path: Path, output_dir: Path) -> Path:
    """Delay importing Mobilint tooling until its path is selected."""
    from tools.ttm_r1_vendors.mobilint import run_mobilint as execute

    return execute(model_path, output_dir)


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
