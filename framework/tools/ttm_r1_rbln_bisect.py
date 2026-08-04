#!/usr/bin/env python3
"""Compile fixed TTM-R1 stages independently to localize a CA22 failure."""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import Sequence

if __package__ in {None, ""}:
    _FRAMEWORK_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_FRAMEWORK_ROOT))
    sys.path.insert(0, str(_FRAMEWORK_ROOT / "src"))

import torch

from chronos_bolt.evidence import write_result
from ttm_r1.core import TTMR1Core, load_ttm_r1_model
from ttm_r1.host_adapter import TTMR1HostAdapter
from ttm_r1.rbln_bisect import build_probe_stages, compile_probe_stages


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Locate a fixed TTM-R1 stage that cannot compile for RBLN-CA22."
    )
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def run_bisection(model_path: Path, output_dir: Path) -> Path:
    """Build CPU boundary tensors and compile all stage probes independently."""
    if not model_path.is_dir():
        raise ValueError(f"TTM-R1 requires a local checkpoint directory: {model_path}")
    if output_dir.exists():
        raise FileExistsError(f"Choose a new output directory: {output_dir}")

    model = load_ttm_r1_model(str(model_path))
    core = TTMR1Core(model).eval()
    adapter = TTMR1HostAdapter(core.contract)
    context = torch.linspace(-4.0, 7.0, steps=512, dtype=torch.float32).reshape(1, 512, 1)
    past_values = adapter.prepare(context).past_values
    stages = build_probe_stages(core.model, past_values)
    try:
        rebel = importlib.import_module("rebel")
    except (ImportError, ModuleNotFoundError) as exc:
        raise ImportError("RBLN bisection requires the rebel-compiler SDK") from exc

    output_dir.mkdir(parents=True)
    report = compile_probe_stages(rebel, stages, output_dir / "stages")
    return write_result(
        output_dir / "rbln-bisect-result.json",
        {
            "status": "diagnostic_complete",
            "vendor": "rbln",
            "model_path": str(model_path.resolve()),
            "stages": report,
        },
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(run_bisection(args.model_path, args.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
