#!/usr/bin/env python3
"""Run a transferred TimesFM 2.5 MXQ on ARIES and retain strict parity evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

if __package__ in {None, ""}:
    _ROOT = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(_ROOT), str(_ROOT / "src")]

from chronos_bolt.evidence import write_result
from timesfm25.contracts import TimesFM25Contract
from tools.timesfm25_compile import compare_device_output, describe_contract
from tools.timesfm25_vendors.mobilint import run_mxq


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _restore(context: np.ndarray, normal: np.ndarray, flipped: np.ndarray) -> np.ndarray:
    loc = context.mean(axis=1, keepdims=True)
    scale = context.std(axis=1, keepdims=True, ddof=1)
    result = (normal - flipped) / 2 * scale + loc
    return np.maximum(result, 0) if float(context.min()) >= 0 else result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    fixture = np.load(args.fixture)
    contract = TimesFM25Contract.fixed()
    normal = run_mxq(args.artifact, fixture["normal_input"], contract)
    flipped = run_mxq(args.artifact, fixture["flipped_input"], contract)
    public = _restore(fixture["context"], normal.output, flipped.output)
    core_parity = {
        "normal": compare_device_output(_as_torch(fixture["expected_normal"]), _as_torch(normal.output)),
        "flipped": compare_device_output(_as_torch(fixture["expected_flipped"]), _as_torch(flipped.output)),
    }
    public_parity = compare_device_output(_as_torch(fixture["expected_public"]), _as_torch(public))
    verified = all(item["within_tolerance"] for item in core_parity.values()) and public_parity["within_tolerance"]
    result = write_result(output / "mobilint-result.json", {
        "status": "device_verified" if verified else "parity_failed", "vendor": "mobilint", "target_device": "aries-rb",
        "contract": describe_contract(contract), "artifact": {"path": str(args.artifact.resolve()), "sha256": _sha256(args.artifact), "size_bytes": args.artifact.stat().st_size},
        "device_core_parity": core_parity, "device_public_parity": public_parity,
        "normal": {"input_abi": list(normal.input_abi), "output_abi": list(normal.output_abi), "saturated_elements": normal.saturated_elements},
        "flipped": {"input_abi": list(flipped.input_abi), "output_abi": list(flipped.output_abi), "saturated_elements": flipped.saturated_elements},
    })
    print(result)
    if not verified:
        raise SystemExit(f"ARIES executed but strict parity failed; inspect {result}")
    return 0


def _as_torch(value):
    import torch
    return torch.from_numpy(np.asarray(value, dtype=np.float32))


if __name__ == "__main__":
    raise SystemExit(main())
