#!/usr/bin/env python3
"""Compile prepared TimesFM 2.5 ONNX files into a fresh ARIES MXQ artifact."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from timesfm25_vendors.mobilint import compile_mxq


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", required=True, type=Path)
    args = parser.parse_args()
    root = args.prepared_dir.resolve()
    fixture = np.load(root / "timesfm25-inference-fixture.npz")
    report = compile_mxq(
        root / "timesfm25-point-core.onnx", root / "timesfm25-point-core.mxq",
        root / "calibration", fixture["normal_input"],
    )
    print(report["artifact"]["path"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
