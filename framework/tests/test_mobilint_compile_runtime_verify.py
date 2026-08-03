import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from tools.mobilint_compile_recipes.attempt import create_attempt, record_artifact
from tools.mobilint_compile_recipes.runtime_verify import verify_runtime


FRAMEWORK_ROOT = Path(__file__).resolve().parents[1]


def _read_result(root: Path) -> dict:
    return json.loads((root / "result.json").read_text(encoding="utf-8"))


def _write_result(root: Path, result: dict) -> None:
    (root / "result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )


def _mark_compiled(root: Path, artifact: Path) -> None:
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_bytes(b"compiled mxq")
    record_artifact(root, artifact)
    result = _read_result(root)
    for stage in ("MBLT_COMPILE", "MXQ_COMPILE"):
        result["stages"][stage].update(
            {
                "status": "pass",
                "started_at": "2026-08-03T00:00:00+00:00",
                "finished_at": "2026-08-03T00:00:01+00:00",
                "elapsed_seconds": 1.0,
                "exit_code": 0,
                "signal": None,
                "error": None,
            }
        )
    result["compile_status"] = "pass"
    _write_result(root, result)


def _prepared_resnet_attempt(tmp_path: Path) -> tuple[Path, Path]:
    root = create_attempt(tmp_path, "resnet", "resnet50", "default", {})
    artifact = root / "mxq" / "resnet50-mxq.mxq"
    _mark_compiled(root, artifact)
    calibration = root / "calibration" / "000.npy"
    calibration.parent.mkdir(parents=True, exist_ok=True)
    np.save(
        calibration,
        np.zeros((1, 224, 224, 3), dtype=np.uint8),
        allow_pickle=False,
    )
    manifest = {
        "model": "resnet50",
        "variant": "default",
        "source_id": "torchvision.models.resnet50:IMAGENET1K_V2",
        "samples": [
            {
                "calibration_path": "calibration/000.npy",
                "calibration_sha256": hashlib.sha256(
                    calibration.read_bytes()
                ).hexdigest(),
            }
        ],
    }
    (root / "source-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root, artifact


def _prepared_patchtst_attempt(tmp_path: Path) -> tuple[Path, Path]:
    root = create_attempt(tmp_path, "patchtst", "patchtst-etth1", "stock", {})
    artifact = root / "mxq" / "patchtst-etth1-mxq.mxq"
    _mark_compiled(root, artifact)
    sample = root / "calibration" / "000"
    sample.mkdir(parents=True)
    values = np.arange(1 * 512 * 7, dtype=np.float32).reshape(1, 512, 7)
    mask = np.ones((1, 512, 7), dtype=np.bool_)
    np.save(sample / "past_values.npy", values, allow_pickle=False)
    np.save(sample / "past_observed_mask.npy", mask, allow_pickle=False)
    manifest = {
        "model": "patchtst-etth1",
        "variant": "stock",
        "source_id": "ibm-granite/granite-timeseries-patchtst",
        "samples": [
            {
                "paths": {
                    "past_values": "calibration/000/past_values.npy",
                    "past_observed_mask": (
                        "calibration/000/past_observed_mask.npy"
                    ),
                }
            }
        ],
    }
    (root / "source-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root, artifact


def _prepared_yolo_attempt(tmp_path: Path) -> tuple[Path, Path]:
    root = create_attempt(tmp_path, "yolo", "yolov5m", "default", {})
    artifact = root / "mxq" / "yolov5m-mxq.mxq"
    _mark_compiled(root, artifact)
    calibration = root / "calibration" / "000.npy"
    calibration.parent.mkdir(parents=True)
    np.save(
        calibration,
        np.full((1, 640, 640, 3), 114, dtype=np.uint8),
        allow_pickle=False,
    )
    manifest = {
        "model": "yolov5m",
        "variant": "default",
        "source_id": (
            "ultralytics/yolov5@86fd1ab270cb2f7e53ee7412cd4a0650bf4bcc51"
        ),
        "samples": [
            {
                "calibration_path": "calibration/000.npy",
                "calibration_sha256": hashlib.sha256(
                    calibration.read_bytes()
                ).hexdigest(),
            }
        ],
    }
    (root / "source-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root, artifact


def _prepared_bert_attempt(tmp_path: Path, *, task: str = "sst2") -> tuple[Path, Path]:
    model = "bert-sst2" if task == "sst2" else "bert-squad1"
    root = create_attempt(tmp_path, f"bert-{task}", model, "default", {})
    task_root = root / task
    artifact = task_root / "mxq" / f"{task}.mxq"
    _mark_compiled(root, artifact)
    calibration_root = task_root / "calibration_data"
    calibration_root.mkdir(parents=True)
    sequence_length = 9
    for index in range(32):
        np.save(
            calibration_root / f"{index:03d}.npy",
            np.zeros((1, sequence_length, 768), dtype=np.float32),
            allow_pickle=False,
        )
    model_id = (
        "textattack/bert-base-uncased-SST-2"
        if task == "sst2"
        else "csarron/bert-base-uncased-squad-v1"
    )
    manifest = {
        "task": task,
        "model_id": model_id,
        "target_device": "aries-rb",
        "dataset_size": 64,
        "calibration_indices": [index * 63 // 31 for index in range(32)],
        "sequence_lengths": [sequence_length] * 32,
        "calibration_files": 32,
        "mxq_inputs": [
            {"name": "embeddings", "shape": [1, -1, 768], "dtype": "float32"}
        ],
        "verified_runtime_outputs": (
            ["logits"] if task == "sst2" else ["end_logits", "start_logits"]
        ),
    }
    (task_root / "calibration_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    result = _read_result(root)
    result["bert_provenance"] = {
        "task_root": str(task_root.resolve()),
        "calibration_manifest": "calibration_manifest.json",
    }
    _write_result(root, result)
    return root, artifact


class FakeQbRuntime:
    __version__ = "v1.3.2"

    class Cluster:
        Cluster0 = "cluster-0"

    class Core:
        Core0 = "core-0"

    class CoreId:
        def __init__(self, cluster, core):
            self.cluster = cluster
            self.core = core

    class Accelerator:
        def __init__(self, device_id):
            self.device_id = device_id

    def __init__(
        self,
        *,
        input_dtypes,
        input_shapes,
        output_shapes,
        outputs,
        metadata_api="current",
        setter_result=True,
        construction_error=None,
        launch_error=None,
        infer_error=None,
        dispose_error=None,
    ):
        self.input_dtypes = input_dtypes
        self.input_shapes = input_shapes
        self.output_shapes = output_shapes
        self.outputs = outputs
        self.metadata_api = metadata_api
        self.setter_result = setter_result
        self.construction_error = construction_error
        self.launch_error = launch_error
        self.infer_error = infer_error
        self.dispose_error = dispose_error
        self.configs = []
        self.models = []
        owner = self

        class ModelConfig:
            def __init__(self):
                self.calls = []
                owner.configs.append(self)

            def set_single_core_mode(self, num_cores, core_ids):
                self.calls.append(("single", num_cores, core_ids))
                return owner.setter_result

            def set_global8_core_mode(self):
                self.calls.append(("global8",))
                return owner.setter_result

        class Model:
            def __init__(self, path, config):
                if owner.construction_error is not None:
                    raise owner.construction_error
                self.path = path
                self.config = config
                self.launch_calls = []
                self.infer_calls = []
                self.dispose_calls = 0
                owner.models.append(self)

            def launch(self):
                self.launch_calls.append(())
                if owner.launch_error is not None:
                    raise owner.launch_error

            def infer(self, inputs):
                self.infer_calls.append(inputs)
                if owner.infer_error is not None:
                    raise owner.infer_error
                return owner.outputs

            def get_model_input_shape(self):
                return owner.input_shapes

            def get_model_output_shape(self):
                return owner.output_shapes

            def dispose(self):
                self.dispose_calls += 1
                if owner.dispose_error is not None:
                    raise owner.dispose_error

        if metadata_api == "current":
            Model.get_input_dtypes = lambda self: owner.input_dtypes
        elif metadata_api == "legacy":
            Model.get_model_input_data_type = lambda self: owner.input_dtypes
        else:
            raise ValueError(metadata_api)
        self.ModelConfig = ModelConfig
        self.Model = Model


def _resnet_sdk(**overrides) -> FakeQbRuntime:
    values = {
        "input_dtypes": ["Uint8"],
        "input_shapes": [(224, 224, 3)],
        "output_shapes": [(1, 1, 1000)],
        "outputs": [np.zeros((1, 1, 1, 1000), dtype=np.float32)],
    }
    values.update(overrides)
    return FakeQbRuntime(**values)


def test_runtime_verify_updates_hardware_stages_and_preserves_quality(tmp_path):
    root, artifact = _prepared_resnet_attempt(tmp_path)
    before = _read_result(root)
    before["quality_status"] = "pass"
    before["quality"] = {"sample_count": "64", "metrics": {"Top-1": "0.75"}}
    _write_result(root, before)
    sdk = _resnet_sdk()

    result = verify_runtime(root, artifact, sdk)

    assert result["runtime_status"] == "pass"
    assert result["contract_status"] == "pass"
    assert result["quality_status"] == "pass"
    assert result["quality"] == before["quality"]
    assert [
        result["stages"][stage]["status"]
        for stage in ("ARIES_LOAD", "CONTRACT_CHECK", "TASK_SMOKE")
    ] == ["pass", "pass", "pass"]
    assert result["stages"]["MXQ_COMPILE"] == before["stages"]["MXQ_COMPILE"]
    assert sdk.configs[0].calls == [("global8",)]
    assert len(sdk.models) == 1
    model = sdk.models[0]
    assert len(model.launch_calls) == len(model.infer_calls) == 1
    assert isinstance(model.infer_calls[0], np.ndarray)
    assert model.infer_calls[0].shape == (1, 224, 224, 3)
    assert model.infer_calls[0].dtype == np.uint8
    assert model.dispose_calls == 1


def test_patchtst_uses_saved_inputs_in_contract_order_with_legacy_metadata(tmp_path):
    root, artifact = _prepared_patchtst_attempt(tmp_path)
    sdk = FakeQbRuntime(
        input_dtypes=["Float32", "Bool"],
        input_shapes=[(1, 512, 7), (1, 512, 7)],
        output_shapes=[(1, 1, 96, 7)],
        outputs=[np.zeros((1, 1, 96, 7), dtype=np.float32)],
        metadata_api="legacy",
    )

    result = verify_runtime(root, artifact, sdk)

    assert result["runtime_status"] == result["contract_status"] == "pass"
    payload = sdk.models[0].infer_calls[0]
    assert isinstance(payload, list) and len(payload) == 2
    assert payload[0].dtype == np.float32
    assert payload[1].dtype == np.bool_
    assert float(payload[0][0, 0, 0]) == 0.0
    assert bool(payload[1][0, 0, 0]) is True


@pytest.mark.parametrize("reverse_outputs", [False, True])
def test_yolov5m_validates_three_raw_heads_in_recorded_order(
    tmp_path, reverse_outputs
):
    root, artifact = _prepared_yolo_attempt(tmp_path)
    outputs = [
        np.zeros((1, 20, 20, 255), dtype=np.float32),
        np.zeros((1, 40, 40, 255), dtype=np.float32),
        np.zeros((1, 80, 80, 255), dtype=np.float32),
    ]
    if reverse_outputs:
        outputs.reverse()
    sdk = FakeQbRuntime(
        input_dtypes=["Uint8"],
        input_shapes=[(640, 640, 3)],
        output_shapes=[(20, 20, 255), (40, 40, 255), (80, 80, 255)],
        outputs=outputs,
    )

    if reverse_outputs:
        with pytest.raises(ValueError, match="output shape"):
            verify_runtime(root, artifact, sdk)
        assert _read_result(root)["contract_status"] == "fail"
    else:
        result = verify_runtime(root, artifact, sdk)
        assert result["runtime_status"] == result["contract_status"] == "pass"
        assert [
            output["name"] for output in result["runtime_verification"]["outputs"]
        ] == [
            "mobilint_yolov5_stride32",
            "mobilint_yolov5_stride16",
            "mobilint_yolov5_stride8",
        ]


@pytest.mark.parametrize("task", ["sst2", "squad1"])
def test_bert_uses_explicit_single_core_and_concrete_dynamic_sequence(tmp_path, task):
    root, artifact = _prepared_bert_attempt(tmp_path, task=task)
    if task == "sst2":
        output_shapes = [(1, 1, 2)]
        outputs = [np.zeros((1, 1, 2), dtype=np.float32)]
    else:
        output_shapes = [(1, -1, 1), (1, -1, 1)]
        outputs = [
            np.zeros((1, 9, 1), dtype=np.float32),
            np.ones((1, 9, 1), dtype=np.float32),
        ]
    sdk = FakeQbRuntime(
        input_dtypes="Float32",
        input_shapes=[(1, -1, 768)],
        output_shapes=output_shapes,
        outputs=outputs,
        metadata_api="legacy",
    )

    result = verify_runtime(root, artifact, sdk)

    assert result["runtime_status"] == result["contract_status"] == "pass"
    call = sdk.configs[0].calls[0]
    assert call[0:2] == ("single", None)
    assert len(call[2]) == 1
    assert (call[2][0].cluster, call[2][0].core) == ("cluster-0", "core-0")
    assert sdk.models[0].infer_calls[0].shape == (1, 9, 768)


@pytest.mark.parametrize(
    ("input_dtypes", "input_shapes", "output_shapes", "message"),
    [
        (["Float32"], [(224, 224, 3)], [(1, 1, 1000)], "input dtype"),
        (["Uint8", "Uint8"], [(224, 224, 3)], [(1, 1, 1000)], "input dtype count"),
        (["Uint8"], [(3, 224, 224)], [(1, 1, 1000)], "input shape"),
        (["Uint8"], [(224, 224, 3)], [(1, 1, 1000), (1, 1)], "output count"),
        (["Uint8"], [(224, 224, 3)], [(1, 1000, 1, 1)], "output shape"),
    ],
)
def test_metadata_contract_mismatch_is_recorded_independently(
    tmp_path, input_dtypes, input_shapes, output_shapes, message
):
    root, artifact = _prepared_resnet_attempt(tmp_path)
    sdk = _resnet_sdk(
        input_dtypes=input_dtypes,
        input_shapes=input_shapes,
        output_shapes=output_shapes,
    )

    with pytest.raises(ValueError, match=message):
        verify_runtime(root, artifact, sdk)

    result = _read_result(root)
    assert result["stages"]["ARIES_LOAD"]["status"] == "pass"
    assert result["stages"]["CONTRACT_CHECK"]["status"] == "fail"
    assert result["stages"]["TASK_SMOKE"]["status"] == "not_run"
    assert result["runtime_status"] == "not_run"
    assert result["contract_status"] == "fail"
    assert result["failed_at"] == "CONTRACT_CHECK"
    assert sdk.models[0].dispose_calls == 1


def test_permitted_boundary_singletons_normalize_but_axis_reorder_does_not(tmp_path):
    root, artifact = _prepared_resnet_attempt(tmp_path)
    passing = _resnet_sdk(
        input_shapes=[(1, 224, 224, 3, 1)],
        output_shapes=[(1, 1, 1000, 1)],
        outputs=[np.zeros((1, 1, 1, 1000, 1), dtype=np.float32)],
    )
    assert verify_runtime(root, artifact, passing)["contract_status"] == "pass"

    other_root, other_artifact = _prepared_resnet_attempt(tmp_path / "other")
    failing = _resnet_sdk(
        output_shapes=[(1, 1000, 1, 1)],
        outputs=[np.zeros((1, 1000, 1, 1), dtype=np.float32)],
    )
    with pytest.raises(ValueError, match="output shape"):
        verify_runtime(other_root, other_artifact, failing)


@pytest.mark.parametrize(
    ("outputs", "message"),
    [
        ([], "output count"),
        ([np.zeros((1, 1000, 1, 1), dtype=np.float32)], "output shape"),
        ([np.zeros((1, 1, 1000), dtype=np.float64)], "output dtype"),
        ([np.full((1, 1, 1000), np.nan, dtype=np.float32)], "finite"),
    ],
)
def test_inference_output_contract_failures_are_recorded(tmp_path, outputs, message):
    root, artifact = _prepared_resnet_attempt(tmp_path)
    sdk = _resnet_sdk(outputs=outputs)

    with pytest.raises(ValueError, match=message):
        verify_runtime(root, artifact, sdk)

    result = _read_result(root)
    assert result["stages"]["ARIES_LOAD"]["status"] == "pass"
    assert result["stages"]["CONTRACT_CHECK"]["status"] == "fail"
    assert result["stages"]["TASK_SMOKE"]["status"] == "not_run"
    assert result["contract_status"] == "fail"
    assert result["runtime_status"] == "not_run"
    assert sdk.models[0].dispose_calls == 1


@pytest.mark.parametrize(
    ("failure", "stage", "constructed", "disposed"),
    [
        ("setter", "ARIES_LOAD", False, False),
        ("construction", "ARIES_LOAD", False, False),
        ("launch", "ARIES_LOAD", True, True),
        ("infer", "TASK_SMOKE", True, True),
        ("dispose", "TASK_SMOKE", True, True),
    ],
)
def test_runtime_lifecycle_failures_record_exact_stage_and_dispose(
    tmp_path, failure, stage, constructed, disposed
):
    root, artifact = _prepared_resnet_attempt(tmp_path)
    overrides = {}
    if failure == "setter":
        overrides["setter_result"] = False
    else:
        overrides[f"{failure}_error"] = RuntimeError(f"{failure} failed")
    sdk = _resnet_sdk(**overrides)

    with pytest.raises(RuntimeError, match=failure):
        verify_runtime(root, artifact, sdk)

    result = _read_result(root)
    assert result["stages"][stage]["status"] == "fail"
    assert result["failed_at"] == stage
    assert (len(sdk.models) == 1) is constructed
    if constructed:
        assert (sdk.models[0].dispose_calls == 1) is disposed


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("unrecorded", "recorded artifact"),
        ("hash", "SHA256"),
        ("escape", "inside the attempt root"),
    ],
)
def test_artifact_provenance_rejects_untrusted_mxq_before_sdk(
    tmp_path, mutation, message
):
    root, artifact = _prepared_resnet_attempt(tmp_path)
    if mutation == "unrecorded":
        artifact = root / "mxq" / "other.mxq"
        artifact.write_bytes(b"other mxq")
    elif mutation == "hash":
        artifact.write_bytes(b"tampered mxq")
    else:
        artifact = tmp_path / "escaped.mxq"
        artifact.write_bytes(b"escaped mxq")
    sdk = _resnet_sdk()

    with pytest.raises(ValueError, match=message):
        verify_runtime(root, artifact, sdk)

    assert sdk.models == []
    result = _read_result(root)
    assert result["stages"]["ARIES_LOAD"]["status"] == "fail"
    assert result["failed_at"] == "ARIES_LOAD"


@pytest.mark.parametrize("mutation", ["dtype", "shape", "nonfinite", "escape"])
def test_saved_smoke_input_contract_is_validated_before_hardware(tmp_path, mutation):
    root, artifact = _prepared_resnet_attempt(tmp_path)
    manifest_path = root / "source-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    path = root / manifest["samples"][0]["calibration_path"]
    if mutation == "dtype":
        np.save(path, np.zeros((1, 224, 224, 3), dtype=np.float32), allow_pickle=False)
    elif mutation == "shape":
        np.save(path, np.zeros((1, 224, 223, 3), dtype=np.uint8), allow_pickle=False)
    elif mutation == "nonfinite":
        manifest["samples"][0].pop("calibration_sha256")
        np.save(path, np.full((1, 224, 224, 3), np.nan, dtype=np.float32), allow_pickle=False)
    else:
        escaped = tmp_path / "outside.npy"
        np.save(escaped, np.zeros((1, 224, 224, 3), dtype=np.uint8), allow_pickle=False)
        manifest["samples"][0]["calibration_path"] = str(escaped)
        manifest["samples"][0].pop("calibration_sha256")
    if mutation in {"dtype", "shape"}:
        manifest["samples"][0]["calibration_sha256"] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    sdk = _resnet_sdk()

    with pytest.raises(ValueError):
        verify_runtime(root, artifact, sdk)

    assert sdk.models == []
    result = _read_result(root)
    assert result["stages"]["CONTRACT_CHECK"]["status"] == "fail"
    assert result["failed_at"] == "CONTRACT_CHECK"


@pytest.mark.parametrize("mutation", ["missing-input", "nonfinite"])
def test_patchtst_saved_multi_input_contract_rejects_bad_samples(tmp_path, mutation):
    root, artifact = _prepared_patchtst_attempt(tmp_path)
    manifest_path = root / "source-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if mutation == "missing-input":
        del manifest["samples"][0]["paths"]["past_observed_mask"]
    else:
        values_path = root / manifest["samples"][0]["paths"]["past_values"]
        value = np.load(values_path, allow_pickle=False)
        value[0, 0, 0] = np.nan
        np.save(values_path, value, allow_pickle=False)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    sdk = FakeQbRuntime(
        input_dtypes=["Float32", "Bool"],
        input_shapes=[(1, 512, 7), (1, 512, 7)],
        output_shapes=[(1, 1, 96, 7)],
        outputs=[np.zeros((1, 1, 96, 7), dtype=np.float32)],
    )

    with pytest.raises(ValueError, match="input|finite"):
        verify_runtime(root, artifact, sdk)

    assert sdk.models == []
    assert _read_result(root)["stages"]["CONTRACT_CHECK"]["status"] == "fail"


def test_runtime_rejects_mxq_stage_that_is_not_pass_without_mutation(tmp_path):
    root, artifact = _prepared_resnet_attempt(tmp_path)
    result = _read_result(root)
    result["compile_status"] = "not_run"
    result["stages"]["MXQ_COMPILE"]["status"] = "not_run"
    _write_result(root, result)
    before = (root / "result.json").read_bytes()
    sdk = _resnet_sdk()

    with pytest.raises(ValueError, match="MXQ_COMPILE.*pass"):
        verify_runtime(root, artifact, sdk)

    assert sdk.models == []
    assert (root / "result.json").read_bytes() == before


def test_runtime_rejects_malformed_artifact_schema_without_mutation(tmp_path):
    root, artifact = _prepared_resnet_attempt(tmp_path)
    result = _read_result(root)
    result["artifacts"][0].pop("sha256")
    _write_result(root, result)
    before = (root / "result.json").read_bytes()
    sdk = _resnet_sdk()

    with pytest.raises(ValueError, match="artifact record schema"):
        verify_runtime(root, artifact, sdk)

    assert sdk.models == []
    assert (root / "result.json").read_bytes() == before


def test_runtime_rejects_reverification_without_mutating_evidence(tmp_path):
    root, artifact = _prepared_resnet_attempt(tmp_path)
    verify_runtime(root, artifact, _resnet_sdk())
    before = (root / "result.json").read_bytes()

    with pytest.raises(ValueError, match="already recorded"):
        verify_runtime(root, artifact, _resnet_sdk())

    assert (root / "result.json").read_bytes() == before


def test_runtime_verify_help_does_not_import_qbruntime(tmp_path):
    poison = tmp_path / "qbruntime.py"
    poison.write_text("raise AssertionError('qbruntime imported during help')\n")
    environment = {
        "PATH": os.environ["PATH"],
        "PYTHONPATH": os.pathsep.join((str(tmp_path), str(FRAMEWORK_ROOT))),
    }

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.mobilint_compile_recipes.runtime_verify",
            "--help",
        ],
        cwd=FRAMEWORK_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--attempt-root" in completed.stdout
    assert "--artifact" in completed.stdout
    assert "--core-mode" not in completed.stdout
