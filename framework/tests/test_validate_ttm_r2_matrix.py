import csv
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.validate_ttm_r2_matrix import main, validate_matrix


DATASET_SHA = "f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066"
RBLN_SHA = "4159ce147a9d91524117b39eba67a706df1c5c987e3317a999e3992c2d3bf172"
ARIES_SHA = "208958c81f1c62ad47557158b788f7f758ac8ef634032286e7f304dac752e217"


def _row(target_id, mae, rmse, **overrides):
    values = {
        "run_id": f"{target_id}-run",
        "timestamp": "2026-08-06 02:35:00",
        "model_name": "ttm-r2",
        "target_id": target_id,
        "ttm_contract_id": "ttm-r2-etth1-ot-512-96-v1",
        "ttm_validation_scope": "full",
        "ttm_expected_windows": "240",
        "ttm_dataset_sha256": DATASET_SHA,
        "ttm_artifact_sha256": "",
        "mobilint_saturation_elements": "",
        "mobilint_saturation_total": "",
        "MAE": str(mae),
        "RMSE": str(rmse),
        "Total Samples": "240",
    }
    values.update(overrides)
    return values


@pytest.fixture
def matrix_rows():
    return {
        "furiosa": _row(
            "furiosa-rngd-torch",
            1.7680668830871582,
            2.1058883666992188,
        ),
        "rbln": _row(
            "rbln-static",
            1.7690539360046387,
            2.10709547996521,
            ttm_artifact_sha256=RBLN_SHA,
        ),
        "mobilint": _row(
            "mobilint-aries",
            1.8834694623947144,
            2.2656056880950928,
            ttm_artifact_sha256=ARIES_SHA,
            mobilint_saturation_elements="0",
            mobilint_saturation_total="122880",
        ),
    }


def _write_csv(path, rows):
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_matrix(tmp_path, rows):
    paths = {}
    for name, row in rows.items():
        path = tmp_path / f"{name}.csv"
        _write_csv(path, [row])
        paths[name] = path
    return paths


def test_matrix_passes_three_full_finite_device_runs(tmp_path, matrix_rows):
    paths = _write_matrix(tmp_path, matrix_rows)

    result = validate_matrix(paths["furiosa"], paths["rbln"], paths["mobilint"])

    assert result["status"] == "pass"
    assert set(result["targets"]) == {
        "furiosa-rngd-torch",
        "rbln-static",
        "mobilint-aries",
    }
    assert result["targets"]["mobilint-aries"]["saturation_elements"] == 0


def test_matrix_selects_latest_matching_ttm_row(tmp_path, matrix_rows):
    paths = _write_matrix(tmp_path, matrix_rows)
    older = dict(matrix_rows["furiosa"])
    older.update(
        timestamp="2026-08-05 00:00:00",
        run_id="older",
        **{"Total Samples": "1"},
    )
    unrelated = dict(matrix_rows["furiosa"])
    unrelated.update(
        timestamp="2026-08-07 00:00:00",
        run_id="unrelated",
        model_name="patchtst-etth1",
    )
    _write_csv(paths["furiosa"], [older, matrix_rows["furiosa"], unrelated])

    result = validate_matrix(paths["furiosa"], paths["rbln"], paths["mobilint"])

    assert result["targets"]["furiosa-rngd-torch"]["run_id"] == (
        "furiosa-rngd-torch-run"
    )


def test_matrix_rejects_smoke_or_saturated_aries(tmp_path, matrix_rows):
    matrix_rows["mobilint"]["ttm_validation_scope"] = "smoke"
    matrix_rows["mobilint"]["Total Samples"] = "239"
    matrix_rows["mobilint"]["mobilint_saturation_elements"] = "1"
    paths = _write_matrix(tmp_path, matrix_rows)

    with pytest.raises(ValueError, match=r"240.*saturation"):
        validate_matrix(paths["furiosa"], paths["rbln"], paths["mobilint"])


@pytest.mark.parametrize("target_name", ["furiosa", "rbln", "mobilint"])
def test_matrix_rejects_dataset_hash_mismatch(
    tmp_path, matrix_rows, target_name
):
    matrix_rows[target_name]["ttm_dataset_sha256"] = "wrong"
    paths = _write_matrix(tmp_path, matrix_rows)

    with pytest.raises(ValueError, match="dataset SHA-256"):
        validate_matrix(paths["furiosa"], paths["rbln"], paths["mobilint"])


@pytest.mark.parametrize(
    ("target_name", "message"),
    [("rbln", "RBLN artifact SHA-256"), ("mobilint", "ARIES artifact SHA-256")],
)
def test_matrix_rejects_artifact_hash_mismatch(
    tmp_path, matrix_rows, target_name, message
):
    matrix_rows[target_name]["ttm_artifact_sha256"] = "wrong"
    paths = _write_matrix(tmp_path, matrix_rows)

    with pytest.raises(ValueError, match=message):
        validate_matrix(paths["furiosa"], paths["rbln"], paths["mobilint"])


@pytest.mark.parametrize(
    ("field", "value"),
    [("MAE", ""), ("RMSE", "nan")],
)
def test_matrix_rejects_missing_or_nonfinite_quality_metric(
    tmp_path, matrix_rows, field, value
):
    matrix_rows["furiosa"][field] = value
    paths = _write_matrix(tmp_path, matrix_rows)

    with pytest.raises(ValueError, match=field):
        validate_matrix(paths["furiosa"], paths["rbln"], paths["mobilint"])


def test_matrix_rejects_wrong_target_id(tmp_path, matrix_rows):
    matrix_rows["rbln"]["target_id"] = "cpu"
    paths = _write_matrix(tmp_path, matrix_rows)

    with pytest.raises(ValueError, match="target_id.*rbln-static"):
        validate_matrix(paths["furiosa"], paths["rbln"], paths["mobilint"])


@pytest.mark.parametrize("field", ["MAE", "RMSE"])
def test_matrix_rejects_quality_outside_historical_tolerance(
    tmp_path, matrix_rows, field
):
    matrix_rows["rbln"][field] = "99.0"
    paths = _write_matrix(tmp_path, matrix_rows)

    with pytest.raises(ValueError, match=rf"{field}.*tolerance"):
        validate_matrix(paths["furiosa"], paths["rbln"], paths["mobilint"])


def test_cli_prints_one_json_summary_and_returns_zero(
    tmp_path, matrix_rows, capsys
):
    paths = _write_matrix(tmp_path, matrix_rows)

    exit_code = main(
        [
            "--furiosa-csv",
            str(paths["furiosa"]),
            "--rbln-csv",
            str(paths["rbln"]),
            "--mobilint-csv",
            str(paths["mobilint"]),
        ]
    )

    assert exit_code == 0
    output = capsys.readouterr()
    assert output.err == ""
    assert json.loads(output.out)["status"] == "pass"
    assert output.out.count("\n") == 1


def test_cli_prints_one_failure_json_and_returns_nonzero(
    tmp_path, matrix_rows, capsys
):
    matrix_rows["mobilint"]["mobilint_saturation_elements"] = "1"
    paths = _write_matrix(tmp_path, matrix_rows)

    exit_code = main(
        [
            "--furiosa-csv",
            str(paths["furiosa"]),
            "--rbln-csv",
            str(paths["rbln"]),
            "--mobilint-csv",
            str(paths["mobilint"]),
        ]
    )

    assert exit_code == 1
    output = capsys.readouterr()
    assert output.err == ""
    assert json.loads(output.out)["status"] == "fail"
    assert output.out.count("\n") == 1
