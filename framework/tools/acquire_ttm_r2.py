#!/usr/bin/env python3
"""Acquire the exact IBM Granite TTM-R2 512-96 checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    root = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(root), str(root / "src")]

from ttm_r2.download import download_checkpoint


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    print(download_checkpoint(parser.parse_args().output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
