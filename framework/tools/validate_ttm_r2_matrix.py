#!/usr/bin/env python3
"""Validate one full TTM-R2 result row for each supported NPU target."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Sequence


EXPECTED = {
    "furiosa-rngd-torch": {
        "MAE": 1.7680668830871582,
        "RMSE": 2.1058883666992188,
    },
    "rbln-static": {
        "MAE": 1.7690539360046387,
        "RMSE": 2.10709547996521,
    },
    "mobilint-aries": {
        "MAE": 1.8834694623947144,
        "RMSE": 2.2656056880950928,
    },
}

TTM_R2_CONTRACT_ID = "ttm-r2-etth1-ot-512-96-v1"
TTM_R2_EXPECTED_WINDOWS = 240
TTM_R2_DATASET_SHA256 = (
    "f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066"
)
RBLN_ARTIFACT_SHA256 = (
    "4159ce147a9d91524117b39eba67a706df1c5c987e3317a999e3992c2d3bf172"
)
ARIES_ARTIFACT_SHA256 = (
    "208958c81f1c62ad47557158b788f7f758ac8ef634032286e7f304dac752e217"
)
ARIES_EXPECTED_QUANTIZED_ELEMENTS = TTM_R2_EXPECTED_WINDOWS * 512
QUALITY_RTOL = 1e-3
QUALITY_ATOL = 1e-4

_ARTIFACT_HASHES = {
    "rbln-static": ("RBLN", RBLN_ARTIFACT_SHA256),
    "mobilint-aries": ("ARIES", ARIES_ARTIFACT_SHA256),
}


def _read_latest_row(path: Path, target_id: str) -> dict[str, str]:
    csv_path = Path(path)
    if not csv_path.is_file():
        raise ValueError(
            f"{target_id}: result CSV must be an existing regular file: "
            f"{csv_path}"
        )
    try:
        with csv_path.open(newline="", encoding="utf-8") as source:
            rows = list(csv.DictReader(source))
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ValueError(f"{target_id}: result CSV could not be read") from exc

    matching_model = [row for row in rows if row.get("model_name") == "ttm-r2"]
    matches = [
        (index, row)
        for index, row in enumerate(rows)
        if row.get("model_name") == "ttm-r2"
        and row.get("target_id") == target_id
    ]
    if not matches:
        observed = sorted(
            {
                str(row.get("target_id", ""))
                for row in matching_model
                if row.get("target_id")
            }
        )
        raise ValueError(
            f"target_id gate failed: expected {target_id!r} in {csv_path}; "
            f"observed {observed}"
        )
    _, latest = max(
        matches,
        key=lambda item: (item[1].get("timestamp", ""), item[0]),
    )
    return latest


def _integer(row: dict[str, str], field: str, target_id: str) -> int:
    raw = row.get(field, "")
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{target_id}: {field} must be an integer, got {raw!r}"
        ) from exc
    return value


def _finite_metric(row: dict[str, str], field: str, target_id: str) -> float:
    raw = row.get(field, "")
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{target_id}: {field} must be a finite number, got {raw!r}"
        ) from exc
    if not math.isfinite(value):
        raise ValueError(
            f"{target_id}: {field} must be a finite number, got {raw!r}"
        )
    return value


def _validate_row(row: dict[str, str], target_id: str) -> dict[str, Any]:
    errors: list[str] = []

    if row.get("ttm_contract_id") != TTM_R2_CONTRACT_ID:
        errors.append(
            f"contract gate expected {TTM_R2_CONTRACT_ID!r}"
        )
    if row.get("ttm_validation_scope") != "full":
        errors.append("validation scope gate expected 'full'")

    try:
        expected_windows = _integer(row, "ttm_expected_windows", target_id)
        if expected_windows != TTM_R2_EXPECTED_WINDOWS:
            errors.append(
                "expected-window gate requires "
                f"{TTM_R2_EXPECTED_WINDOWS}, got {expected_windows}"
            )
    except ValueError as exc:
        errors.append(str(exc))

    try:
        total_samples = _integer(row, "Total Samples", target_id)
        if total_samples != TTM_R2_EXPECTED_WINDOWS:
            errors.append(
                f"sample gate requires 240 processed windows, got {total_samples}"
            )
    except ValueError as exc:
        total_samples = None
        errors.append(str(exc))

    if row.get("ttm_dataset_sha256") != TTM_R2_DATASET_SHA256:
        errors.append(
            "dataset SHA-256 gate failed: expected canonical ETTh1 bytes"
        )

    artifact_sha256 = row.get("ttm_artifact_sha256", "")
    if target_id in _ARTIFACT_HASHES:
        vendor, expected_artifact = _ARTIFACT_HASHES[target_id]
        if artifact_sha256 != expected_artifact:
            errors.append(
                f"{vendor} artifact SHA-256 gate failed: "
                f"expected {expected_artifact}, got {artifact_sha256!r}"
            )

    metrics: dict[str, float] = {}
    for field, expected in EXPECTED[target_id].items():
        try:
            actual = _finite_metric(row, field, target_id)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        metrics[field] = actual
        if not math.isclose(
            actual,
            expected,
            rel_tol=QUALITY_RTOL,
            abs_tol=QUALITY_ATOL,
        ):
            errors.append(
                f"{field} tolerance gate failed: expected {expected} "
                f"with rtol={QUALITY_RTOL}, atol={QUALITY_ATOL}; got {actual}"
            )

    saturation_elements = None
    saturation_total = None
    if target_id == "mobilint-aries":
        try:
            saturation_elements = _integer(
                row, "mobilint_saturation_elements", target_id
            )
            if saturation_elements != 0:
                errors.append(
                    "ARIES saturation gate requires zero saturated elements, "
                    f"got {saturation_elements}"
                )
        except ValueError as exc:
            errors.append(str(exc))
        try:
            saturation_total = _integer(
                row, "mobilint_saturation_total", target_id
            )
            if saturation_total != ARIES_EXPECTED_QUANTIZED_ELEMENTS:
                errors.append(
                    "ARIES saturation coverage gate requires "
                    f"{ARIES_EXPECTED_QUANTIZED_ELEMENTS} measured elements, "
                    f"got {saturation_total}"
                )
        except ValueError as exc:
            errors.append(str(exc))

    if errors:
        raise ValueError(f"{target_id}: " + "; ".join(errors))

    summary: dict[str, Any] = {
        "run_id": row.get("run_id", ""),
        "timestamp": row.get("timestamp", ""),
        "samples": total_samples,
        "MAE": metrics["MAE"],
        "RMSE": metrics["RMSE"],
        "dataset_sha256": row["ttm_dataset_sha256"],
    }
    if target_id in _ARTIFACT_HASHES:
        summary["artifact_sha256"] = artifact_sha256
    if target_id == "mobilint-aries":
        summary["saturation_elements"] = saturation_elements
        summary["saturation_total"] = saturation_total
    return summary


def validate_matrix(
    furiosa_csv: Path,
    rbln_csv: Path,
    mobilint_csv: Path,
) -> dict[str, Any]:
    """Return a JSON-compatible summary when all operational gates pass."""
    paths = {
        "furiosa-rngd-torch": Path(furiosa_csv),
        "rbln-static": Path(rbln_csv),
        "mobilint-aries": Path(mobilint_csv),
    }
    targets = {
        target_id: _validate_row(
            _read_latest_row(path, target_id), target_id
        )
        for target_id, path in paths.items()
    }
    return {
        "status": "pass",
        "contract_id": TTM_R2_CONTRACT_ID,
        "expected_windows": TTM_R2_EXPECTED_WINDOWS,
        "targets": targets,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate full TTM-R2 ETTh1 runs on Furiosa RNGD, "
            "Rebellions CA22, and Mobilint ARIES."
        )
    )
    parser.add_argument("--furiosa-csv", type=Path, required=True)
    parser.add_argument("--rbln-csv", type=Path, required=True)
    parser.add_argument("--mobilint-csv", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = validate_matrix(
            args.furiosa_csv, args.rbln_csv, args.mobilint_csv
        )
        exit_code = 0
    except ValueError as exc:
        summary = {"status": "fail", "error": str(exc)}
        exit_code = 1
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
