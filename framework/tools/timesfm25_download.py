#!/usr/bin/env python3
"""Download and write immutable local evidence for TimesFM 2.5."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

if __package__ in {None, ""}:
    root = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(root), str(root / "src")]

from timesfm25.download import download_checkpoint


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download the official TimesFM 2.5 checkpoint")
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    print(download_checkpoint(build_parser().parse_args(argv).output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
