"""Asset preparation respects explicit paths and reports failed exports."""

from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

import main as benchmark_main
from core.model_spec import Task
from utils.dataset_resolver import resolve_dataset_paths


def _args(tmp_path, *, model=None, dataset=None):
    return Namespace(
        backend="onnxruntime", hef=None, artifact=None, compile=False,
        onnx=str(model or tmp_path / "model.onnx"), model_path=None,
        dataset=str(dataset or tmp_path / "dataset"),
    )


def _target():
    return SimpleNamespace(target_id="cpu", uses_compiler=False, artifact_format="onnx")


def test_explicit_missing_model_is_not_replaced_by_profile_download(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(benchmark_main, "_run_prepare_script", calls.append)
    profile = {
        "default_model_path": str(tmp_path / "default.onnx"),
        "prepare_model_script": "models/prepare_yolov5m.py",
    }
    with pytest.raises(FileNotFoundError, match="model|artifact"):
        benchmark_main.run_auto_prepare(
            profile, _args(tmp_path, model=tmp_path / "custom.onnx"), _target()
        )
    assert calls == []


def test_explicit_missing_dataset_is_not_replaced_by_profile_download(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(benchmark_main, "_run_prepare_script", calls.append)
    model = tmp_path / "model.onnx"
    model.write_bytes(b"model")
    profile = {
        "default_dataset_path": str(tmp_path / "default"),
        "prepare_dataset_script": "datasets/prepare_coco128.py",
    }
    with pytest.raises(FileNotFoundError, match="dataset"):
        benchmark_main.run_auto_prepare(
            profile, _args(tmp_path, model=model, dataset=tmp_path / "custom"), _target()
        )
    assert calls == []


def test_default_model_script_must_create_requested_file(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(benchmark_main, "_run_prepare_script", calls.append)
    expected = tmp_path / "default.onnx"
    profile = {
        "default_model_path": str(expected),
        "prepare_model_script": "models/prepare_yolov5m.py",
    }
    with pytest.raises(FileNotFoundError, match="model|artifact"):
        benchmark_main.run_auto_prepare(profile, _args(tmp_path, model=expected), _target())
    assert calls == ["models/prepare_yolov5m.py"]


def test_default_hf_model_directory_needs_config_and_weights(tmp_path, monkeypatch):
    model_dir = tmp_path / "llama"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    calls = []
    monkeypatch.setattr(benchmark_main, "_run_prepare_script", calls.append)
    profile = {
        "default_model_path": str(model_dir),
        "prepare_model_script": "models/prepare_llama_3_1_8b.py",
    }
    args = _args(tmp_path)
    args.backend = "vllm"
    args.model_path = str(model_dir)
    with pytest.raises(FileNotFoundError, match="model|artifact"):
        benchmark_main.run_auto_prepare(profile, args, _target())
    assert calls == ["models/prepare_llama_3_1_8b.py"]


def test_explicit_detection_image_dir_survives_label_autodiscovery(tmp_path):
    root = tmp_path / "dataset"
    (root / "images" / "val2017").mkdir(parents=True)
    (root / "custom_images").mkdir()
    image_dir, label_dir = resolve_dataset_paths(
        Task.OBJECT_DETECTION, str(root), "custom_images", ""
    )
    assert Path(image_dir) == root / "custom_images"
    assert Path(label_dir) == root / "labels" / "val2017"


def test_yolov8_default_exports_only_active_pose_profile(monkeypatch):
    import sys
    import models.prepare_yolov8_vision as prepare

    calls = []
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=object()))
    monkeypatch.setattr(prepare, "_export_model", lambda yolo, root, name, info: calls.append(name))
    prepare.main([])
    assert calls == ["yolov8s-pose"]


def test_yolov8_export_failure_propagates(monkeypatch):
    import sys
    import models.prepare_yolov8_vision as prepare

    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=object()))
    def fail(*args):
        raise RuntimeError("export failed")
    monkeypatch.setattr(prepare, "_export_model", fail)
    with pytest.raises(RuntimeError, match="export failed"):
        prepare.main([])


def test_yolov5_export_failure_propagates(monkeypatch, tmp_path):
    import sys
    import models.prepare_yolov5m as prepare

    monkeypatch.setattr(prepare, "__file__", str(tmp_path / "prepare_yolov5m.py"))
    monkeypatch.setitem(
        sys.modules, "ultralytics",
        SimpleNamespace(YOLO=lambda _: SimpleNamespace(export=lambda **_: (_ for _ in ()).throw(RuntimeError("export failed")))),
    )
    with pytest.raises(RuntimeError, match="export failed"):
        prepare.main()


def test_hf_snapshot_must_contain_requested_model_file(monkeypatch, tmp_path):
    import models.prepare_hf_model as prepare

    target = tmp_path / "Kalray_resnet50"
    target.mkdir()
    (target / "config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(prepare, "snapshot_download", lambda **kwargs: str(target))
    with pytest.raises(FileNotFoundError, match="resnet50-v1-7s.onnx"):
        prepare.download_model_repository(
            "Kalray/resnet50", output_root=str(tmp_path),
            required_files=("resnet50-v1-7s.onnx",),
        )


def test_hf_snapshot_does_not_accept_config_only(monkeypatch, tmp_path):
    import models.prepare_hf_model as prepare

    target = tmp_path / "meta-llama_Llama-3.1-8B"
    target.mkdir()
    (target / "config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(prepare, "snapshot_download", lambda **kwargs: str(target))
    with pytest.raises(FileNotFoundError, match="weight|model"):
        prepare.download_model_repository(
            "meta-llama/Llama-3.1-8B", output_root=str(tmp_path)
        )
