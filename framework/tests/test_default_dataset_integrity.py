"""A path alone does not establish that a default benchmark dataset is ready."""

import json
from argparse import Namespace
from types import SimpleNamespace

import numpy as np
import pytest

import main as benchmark_main


@pytest.mark.parametrize("script,path_suffix", [
    ("datasets/prepare_imagenet_1k.py", "imagenet_1k"),
    ("datasets/prepare_cifar10.py", "cifar10"),
    ("datasets/prepare_coco128.py", "coco128"),
    ("datasets/prepare_coco_vision.py", "coco"),
])
def test_empty_default_dataset_directory_is_rejected(tmp_path, script, path_suffix):
    root = tmp_path / path_suffix
    root.mkdir()
    with pytest.raises((FileNotFoundError, ValueError)):
        benchmark_main._validate_prepared_dataset_path(str(root), script)


def test_squad_json_requires_a_question(tmp_path):
    path = tmp_path / "val.json"
    path.write_text(json.dumps({"data": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="SQuAD|question|QA"):
        benchmark_main._validate_prepared_dataset_path(str(path), "datasets/prepare_squad2.py")


def test_default_coco128_rejects_all_missing_label_files(tmp_path):
    images = tmp_path / "images" / "train2017"
    labels = tmp_path / "labels" / "train2017"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    for index in range(128):
        (images / f"{index:012d}.jpg").write_bytes(b"image")
    with pytest.raises(ValueError, match="label"):
        benchmark_main._validate_prepared_dataset_path(
            str(tmp_path), "datasets/prepare_coco128.py"
        )


@pytest.mark.parametrize("script,names", [
    ("datasets/prepare_text_numpy.py", ("input_ids", "attention_mask", "labels")),
    ("datasets/prepare_squad_numpy.py", ("input_ids", "attention_mask", "token_type_ids", "start_positions", "end_positions")),
])
def test_default_numpy_arrays_must_have_same_nonzero_sample_count(tmp_path, script, names):
    for index, name in enumerate(names):
        shape = (2 if index == 0 else 1, 4) if name in {"input_ids", "attention_mask", "token_type_ids"} else (1,)
        np.save(tmp_path / f"{name}.npy", np.zeros(shape, dtype=np.int64))
    with pytest.raises(ValueError, match="sample|length|shape"):
        benchmark_main._validate_prepared_dataset_path(str(tmp_path), script)


def test_default_numpy_scalar_input_is_reported_as_invalid(tmp_path):
    np.save(tmp_path / "input_ids.npy", np.array(1, dtype=np.int64))
    np.save(tmp_path / "attention_mask.npy", np.ones((1, 4), dtype=np.int64))
    np.save(tmp_path / "labels.npy", np.zeros((1,), dtype=np.int64))
    with pytest.raises(ValueError, match="sample|shape"):
        benchmark_main._validate_prepared_dataset_path(
            str(tmp_path), "datasets/prepare_text_numpy.py"
        )


def test_existing_incomplete_default_imagenet_triggers_prepare(tmp_path, monkeypatch):
    root = tmp_path / "imagenet_1k"
    (root / "val").mkdir(parents=True)
    calls = []
    monkeypatch.setattr(benchmark_main, "_run_prepare_script", calls.append)
    args = Namespace(
        backend="onnxruntime", hef=None, artifact=None, compile=False,
        onnx=None, model_path=str(tmp_path / "model.onnx"), dataset=str(root),
    )
    profile = {
        "prepare_dataset_script": "datasets/prepare_imagenet_1k.py",
        "default_dataset_path": str(root),
    }
    target = SimpleNamespace(target_id="cuda", uses_compiler=False, artifact_format="onnx")

    with pytest.raises((FileNotFoundError, ValueError)):
        benchmark_main.run_auto_prepare(profile, args, target)
    assert calls == ["datasets/prepare_imagenet_1k.py"]
