"""Download the canonical local TTM-R2 checkpoint."""

from __future__ import annotations

import sys
from pathlib import Path


FRAMEWORK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FRAMEWORK_ROOT / "src"))

from ttm_r2.download import download_checkpoint


def main() -> int:
    destination = Path(__file__).resolve().parent / (
        "ibm-granite_granite-timeseries-ttm-r2"
    )
    print(download_checkpoint(destination))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
