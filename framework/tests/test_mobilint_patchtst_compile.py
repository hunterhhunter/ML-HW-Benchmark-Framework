import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from tools.mobilint_compile_recipes.patchtst_etth1 import (
    build_patchtst_wrapper,
    compile_stage,
    prepare_calibration,
    resolve_model_revision,
    source_smoke,
    static_patchify,
    write_multi_input_calibration,
)


FRAMEWORK_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ID = "ibm-granite/granite-timeseries-patchtst"
RESOLVED_REVISION = "a" * 40
FEATURES = ("HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT")


class _UnfoldPatchifier(torch.nn.Module):
    def forward(self, past_values):
        return (
            past_values[:, 8:, :]
            .unfold(1, 12, 12)
            .permute(0, 2, 1, 3)
            .contiguous()
        )


class _FakePatchTST(torch.nn.Module):
    def __init__(self, *, prediction_length=96):
        super().__init__()
        self.config = SimpleNamespace(
            context_length=512,
            prediction_length=prediction_length,
            num_input_channels=7,
            patch_length=12,
            patch_stride=12,
        )
        self.model = torch.nn.Module()
        self.model.patchifier = _UnfoldPatchifier()
        self.last_mask_dtype = None

    def forward(self, *, past_values, past_observed_mask, return_dict):
        assert return_dict is True
        self.last_mask_dtype = past_observed_mask.dtype
        patches = self.model.patchifier(past_values)
        channel_summary = patches.mean(dim=(2, 3))
        mask_summary = past_observed_mask.to(past_values.dtype).mean(dim=1)
        prediction = (channel_summary + mask_summary).unsqueeze(1).expand(-1, 96, -1)
        return SimpleNamespace(prediction_outputs=prediction.contiguous())


class _FakeRevisionApi:
    def model_info(self, repo_id, *, revision):
        assert repo_id == SOURCE_ID
        assert revision == "main"
        return SimpleNamespace(sha=RESOLVED_REVISION)


class _FakeCalibrationConfig:
    class MaxPercentile:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    def __init__(self, **kwargs):
        self.kwargs = kwargs


def _sample_inputs():
    values = torch.linspace(-2.0, 3.0, 512 * 7).reshape(1, 512, 7)
    positions = torch.arange(512 * 7).reshape(1, 512, 7)
    mask = positions.remainder(5).ne(0)
    return values, mask


def _write_etth1(path):
    rows = 11_520
    positions = np.arange(rows, dtype=np.float32)[:, None]
    data = np.concatenate(
        [np.sin(positions / (index + 3)) + index for index in range(7)],
        axis=1,
    ).astype(np.float32)
    frame = pd.DataFrame(data, columns=FEATURES)
    frame.insert(0, "date", pd.date_range("2016-07-01", periods=rows, freq="h"))
    frame.to_csv(path, index=False)


def _prepare(tmp_path, *, variant="stock"):
    dataset = tmp_path / "ETTh1.csv"
    _write_etth1(dataset)
    attempt_root = tmp_path / "attempt"
    attempt_root.mkdir()
    manifest = prepare_calibration(
        dataset,
        attempt_root,
        variant=variant,
        requested_revision="main",
        revision_api=_FakeRevisionApi(),
    )
    return attempt_root, manifest


def test_static_patchifier_matches_checkpoint_unfold_layout():
    values = torch.arange(1 * 512 * 7, dtype=torch.float32).reshape(1, 512, 7)
    expected = (
        values[:, 8:, :]
        .unfold(1, 12, 12)
        .permute(0, 2, 1, 3)
        .contiguous()
    )

    actual = static_patchify(values)

    assert actual.shape == (1, 7, 42, 12)
    torch.testing.assert_close(actual, expected)


def test_calibration_json_orders_equal_shape_inputs_by_contract(tmp_path):
    samples = [
        {
            "past_values": np.ones((1, 512, 7), dtype=np.float32),
            "past_observed_mask": np.ones((1, 512, 7), dtype=np.bool_),
        }
    ]
    original_values = samples[0]["past_values"].copy()
    original_mask = samples[0]["past_observed_mask"].copy()

    path = write_multi_input_calibration(samples, tmp_path / "calibration")

    payload = json.loads(path.read_text())
    assert payload["info"]["input names"] == [
        "past_values",
        "past_observed_mask",
    ]
    assert payload["calib paths"][0][0].endswith("past_values.npy")
    assert payload["calib paths"][0][1].endswith("past_observed_mask.npy")
    values = np.load(payload["calib paths"][0][0], allow_pickle=False)
    mask = np.load(payload["calib paths"][0][1], allow_pickle=False)
    assert values.dtype == np.float32
    assert mask.dtype == np.bool_
    np.testing.assert_array_equal(samples[0]["past_values"], original_values)
    np.testing.assert_array_equal(samples[0]["past_observed_mask"], original_mask)


def test_calibration_writer_rejects_invalid_or_existing_output(tmp_path):
    root = tmp_path / "calibration"
    invalid = [{
        "past_values": np.zeros((1, 512, 7), dtype=np.float64),
        "past_observed_mask": np.ones((1, 512, 7), dtype=np.bool_),
    }]
    with pytest.raises(ValueError, match="past_values dtype"):
        write_multi_input_calibration(invalid, root)
    assert not root.exists()

    valid = [{
        "past_values": np.zeros((1, 512, 7), dtype=np.float32),
        "past_observed_mask": np.ones((1, 512, 7), dtype=np.bool_),
    }]
    write_multi_input_calibration(valid, root)
    with pytest.raises(FileExistsError, match="already exists"):
        write_multi_input_calibration(valid, root)


def test_stock_and_compat_wrappers_preserve_output_and_external_abi():
    stock_model = _FakePatchTST().eval()
    compat_model = copy.deepcopy(stock_model).eval()
    stock_patchifier = stock_model.model.patchifier
    compat_patchifier = compat_model.model.patchifier
    values, mask = _sample_inputs()

    stock = build_patchtst_wrapper(stock_model, "stock")
    compat = build_patchtst_wrapper(compat_model, "compat-static-patchifier")
    with torch.no_grad():
        stock_output = stock(values, mask)
        compat_output = compat(values, mask)

    assert stock_model.model.patchifier is stock_patchifier
    assert compat_model.model.patchifier is not compat_patchifier
    assert stock_model.last_mask_dtype == torch.bool
    assert compat_model.last_mask_dtype == torch.float32
    assert stock_output.shape == compat_output.shape == (1, 96, 7)
    torch.testing.assert_close(
        compat_output,
        stock_output,
        rtol=1e-5,
        atol=1e-6,
    )


def test_wrapper_rejects_checkpoint_contract_drift():
    with pytest.raises(ValueError, match="prediction_length"):
        build_patchtst_wrapper(_FakePatchTST(prediction_length=48), "stock")


def test_resolve_revision_requires_an_exact_commit_sha():
    assert resolve_model_revision(SOURCE_ID, RESOLVED_REVISION) == RESOLVED_REVISION

    api = SimpleNamespace(
        model_info=lambda repo_id, revision: SimpleNamespace(sha="main")
    )
    with pytest.raises(ValueError, match="exact commit SHA"):
        resolve_model_revision(SOURCE_ID, "main", api=api)


def test_prepare_uses_real_ett_loader_and_writes_complete_provenance(tmp_path):
    attempt_root, manifest = _prepare(tmp_path)

    expected_indices = [index * 189 // 31 for index in range(32)]
    assert manifest["source_id"] == SOURCE_ID
    assert manifest["requested_revision"] == "main"
    assert manifest["resolved_revision"] == RESOLVED_REVISION
    assert manifest["etth1_sha256"] == hashlib.sha256(
        (tmp_path / "ETTh1.csv").read_bytes()
    ).hexdigest()
    assert manifest["calibration_indices"] == expected_indices
    assert manifest["loader"] == {
        "split": "val",
        "split_boundaries": [8640, 11520],
        "context_length": 512,
        "prediction_length": 96,
        "stride": 12,
        "window_count": 190,
    }
    assert manifest["input_order"] == ["past_values", "past_observed_mask"]
    assert manifest["inputs"] == [
        {"name": "past_values", "shape": [1, 512, 7], "dtype": "float32"},
        {"name": "past_observed_mask", "shape": [1, 512, 7], "dtype": "bool"},
    ]
    assert manifest["normalization"]["enabled"] is True
    assert manifest["normalization"]["method"] == "per-window RevIN"
    assert manifest["normalization"]["epsilon"] == 1e-8
    assert len(manifest["normalization"]["samples"]) == 32
    assert json.loads((attempt_root / "source-manifest.json").read_text()) == manifest
    report = json.loads((attempt_root / "compile-report.json").read_text())
    assert report["resolved_revision"] == RESOLVED_REVISION
    assert report["compiler_options"]["mxq"]["inference_scheme"] == "global8"
    calibration = json.loads((attempt_root / "calibration" / "calibration.json").read_text())
    assert len(calibration["calib paths"]) == 32
    first_values = np.load(calibration["calib paths"][0][0], allow_pickle=False)
    first_mask = np.load(calibration["calib paths"][0][1], allow_pickle=False)
    assert first_values.shape == first_mask.shape == (1, 512, 7)
    assert first_values.dtype == np.float32
    assert first_mask.dtype == np.bool_
    assert np.isfinite(first_values).all()


def test_prepare_refuses_to_mutate_an_existing_calibration_root(tmp_path):
    attempt_root, manifest = _prepare(tmp_path)
    before = (attempt_root / "source-manifest.json").read_bytes()

    with pytest.raises(FileExistsError, match="already exists"):
        prepare_calibration(
            tmp_path / "ETTh1.csv",
            attempt_root,
            variant="stock",
            requested_revision=RESOLVED_REVISION,
        )

    assert (attempt_root / "source-manifest.json").read_bytes() == before
    assert json.loads(before) == manifest


def test_source_smoke_records_cpu_output_without_vendor_import(tmp_path):
    attempt_root, _ = _prepare(tmp_path, variant="compat-static-patchifier")

    output = source_smoke(
        attempt_root,
        "compat-static-patchifier",
        model_loader=lambda source_id, revision: _FakePatchTST().eval(),
    )

    assert output.shape == (1, 96, 7)
    assert output.dtype == torch.float32
    assert torch.isfinite(output).all()
    assert "qbcompiler" not in sys.modules
    report = json.loads((attempt_root / "compile-report.json").read_text())
    assert report["source_smoke"] == {
        "output_shape": [1, 96, 7],
        "output_dtype": "float32",
        "finite": True,
        "equivalence_checked": True,
        "stock_compat_rtol": 1e-5,
        "stock_compat_atol": 1e-6,
    }


@pytest.mark.parametrize("stage", ["mblt", "mxq"])
def test_compile_stage_uses_manifest_input_and_records_artifact(tmp_path, stage):
    attempt_root, _ = _prepare(tmp_path, variant="stock")
    observed = {}

    def fake_mblt_compile(**kwargs):
        observed.update(kwargs)
        assert (attempt_root / "compile-report.json").is_file()
        Path(kwargs["mblt_save_path"]).write_bytes(b"mblt")

    def fake_mxq_compile(**kwargs):
        observed.update(kwargs)
        assert (attempt_root / "compile-report.json").is_file()
        Path(kwargs["save_path"]).write_bytes(b"mxq")

    compiler_api = SimpleNamespace(
        CalibrationConfig=_FakeCalibrationConfig,
        mxq_compile=fake_mxq_compile,
    )
    result = compile_stage(
        stage,
        attempt_root,
        "stock",
        model_loader=lambda source_id, revision: _FakePatchTST().eval(),
        mblt_compiler=fake_mblt_compile,
        mxq_compiler_api=compiler_api,
    )

    assert result.is_file() and result.read_bytes() == stage.encode()
    assert list(observed["feed_dict"]) == [
        "past_values",
        "past_observed_mask",
    ]
    assert observed["feed_dict"]["past_values"].dtype == torch.float32
    assert observed["feed_dict"]["past_observed_mask"].dtype == torch.bool
    if stage == "mxq":
        assert observed["calib_data_path"] == str(
            attempt_root / "calibration" / "calibration.json"
        )
        assert observed["inference_scheme"] == "global8"
    report = json.loads((attempt_root / "compile-report.json").read_text())
    assert report["source_smoke"] == {
        "output_shape": [1, 96, 7],
        "output_dtype": "float32",
        "finite": True,
        "equivalence_checked": False,
    }
    assert report["artifacts"][stage]["path"] == f"{stage}/patchtst-etth1-{stage}.{stage}"
    assert report["artifacts"][stage]["sha256"] == hashlib.sha256(
        stage.encode()
    ).hexdigest()


def test_repeated_compile_stage_does_not_mutate_successful_report(tmp_path):
    attempt_root, _ = _prepare(tmp_path, variant="stock")

    def fake_compile(**kwargs):
        Path(kwargs["mblt_save_path"]).write_bytes(b"mblt")

    compile_stage(
        "mblt",
        attempt_root,
        "stock",
        model_loader=lambda source_id, revision: _FakePatchTST().eval(),
        mblt_compiler=fake_compile,
    )
    report_path = attempt_root / "compile-report.json"
    before = report_path.read_bytes()

    with pytest.raises(FileExistsError, match="already exists"):
        compile_stage(
            "mblt",
            attempt_root,
            "stock",
            model_loader=lambda source_id, revision: _FakePatchTST().eval(),
            mblt_compiler=fake_compile,
        )

    assert report_path.read_bytes() == before


def test_describe_cli_needs_no_huggingface_or_vendor_runtime():
    environment = {
        "PATH": __import__("os").environ.get("PATH", ""),
        "PYTHONPATH": str(FRAMEWORK_ROOT),
    }
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.mobilint_compile_recipes.patchtst_etth1",
            "--stage",
            "describe",
            "--variant",
            "stock",
        ],
        check=True,
        cwd=FRAMEWORK_ROOT,
        env=environment,
        text=True,
        capture_output=True,
    )

    payload = json.loads(result.stdout)
    assert payload["source_id"] == SOURCE_ID
    assert payload["target_device"] == "aries-rb"
    assert payload["inference_scheme"] == "global8"
    assert [item["name"] for item in payload["runtime_inputs"]] == [
        "past_values",
        "past_observed_mask",
    ]
