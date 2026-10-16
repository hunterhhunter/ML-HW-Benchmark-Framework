import sys
from types import SimpleNamespace

import pytest

import main as benchmark_main
from core.model_spec import Model_Spec, Task
from dataloader.regulus_vision_profiles import REGULUS_VG_E2E_YOLOV8S_POSE_RAW_V1 as PROFILE


def _spec():
    return Model_Spec(name="yolov8s-pose", task=Task.POSE_ESTIMATION,
        input_shapes={"images": (1, 3, 640, 640)}, input_dtype={"images": "float32"},
        output_shapes={"output": (1, 56, 8400)}, model_paths={"mobilint": "model.mxq"})


def _run(monkeypatch, tmp_path, *, mode="e2e", target="mobilint-regulus"):
    import utils.dataset_resolver as dataset_resolver
    artifact = tmp_path / "model.mxq"
    artifact.touch()
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    captured = {"quality": []}

    class Loader:
        total_samples = 5
        def get_metadata(self):
            return {"runtime_options": PROFILE.runtime_contract()}

    class Runtime:
        def load(self, model):
            captured["loaded"] = model

    monkeypatch.setattr(dataset_resolver, "resolve_dataset_paths", lambda *a: (None, None))
    monkeypatch.setattr(benchmark_main, "create_model_spec", lambda *a, **k: _spec())
    monkeypatch.setattr(benchmark_main, "create_dataloader", lambda **k: captured.setdefault("loader", k) and Loader())
    monkeypatch.setattr(benchmark_main, "create_runtime", lambda *a, **k: captured.setdefault("runtime", k) and Runtime())
    monkeypatch.setattr(benchmark_main, "create_evaluator", lambda *a, **k: captured["quality"].append("evaluator") or object())
    monkeypatch.setattr(benchmark_main, "create_decoder", lambda *a, **k: captured.setdefault("decoder", k) and object())
    monkeypatch.setattr(benchmark_main, "execute_benchmark", lambda *a, **k: captured.setdefault("execution", k) and 0)
    argv = ["main.py", "--model", "yolov8s-pose", "--target", target,
            "--artifact", str(artifact), "--dataset", str(dataset),
            "--image-preprocess-profile", PROFILE.profile_id]
    if mode == "async_queue":
        argv += ["--inference-mode", "async_queue", "--async-pass", "runtime-only"]
    monkeypatch.setattr(sys, "argv", argv)
    return captured, benchmark_main.main()


def test_regulus_pose_main_routes_profile_loader_decoder_and_evaluator(monkeypatch, tmp_path):
    captured, status = _run(monkeypatch, tmp_path)
    assert status == 0
    assert captured["loader"]["regulus_pose_profile"] is PROFILE
    assert captured["loader"]["target_id"] == "mobilint-regulus"
    assert captured["decoder"]["regulus_pose_profile"] is PROFILE
    assert captured["decoder"]["target_id"] == "mobilint-regulus"
    assert captured["quality"] == ["evaluator"]
    assert captured["loaded"].spec.input_dtype["images"] == "uint8"
    assert captured["execution"]["result_metadata"]["mobilint_vision_profile_id"] == PROFILE.profile_id


def test_regulus_pose_runtime_only_skips_quality_components(monkeypatch, tmp_path):
    captured, status = _run(monkeypatch, tmp_path, mode="async_queue")
    assert status == 0
    assert captured["quality"] == []
    assert "decoder" not in captured
    assert captured["execution"]["decoder"] is None
    assert captured["execution"]["evaluator"] is None


def test_regulus_pose_profile_rejected_for_aries_before_runtime_load(monkeypatch, tmp_path):
    with pytest.raises((ValueError, SystemExit)):
        captured, _ = _run(monkeypatch, tmp_path, target="mobilint-aries")
