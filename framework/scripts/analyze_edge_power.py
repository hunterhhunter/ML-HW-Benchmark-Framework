"""Print system and baseline-subtracted energy for one edge benchmark run."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from core.edge_power_analysis import analyze_run


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_csv", type=Path)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    try:
        report = analyze_run(args.results_csv, args.run_id)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"edge power analysis failed: {exc}\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
