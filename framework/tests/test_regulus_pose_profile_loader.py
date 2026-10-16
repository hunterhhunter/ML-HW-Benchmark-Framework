import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from core.model_spec import Model_Spec, Task
from dataloader.regulus_vision_profiles import (
    REGULUS_VG_E2E_YOLOV8S_POSE_RAW_V1,
    apply_regulus_pose_profile,
    resolve_regulus_pose_profile,
)
from dataloader.regulus_pose_estimation_loader import RegulusPoseEstimationLoader
from preprocessor.regulus_pose import RegulusPosePreprocessor
from coco_test_utils import write_coco_fixture, make_pose_spec
from decoders.regulus_yolov8_pose import RegulusYoloV8PoseRawHeadDecoder
from evaluators.pose_estimation_evaluator import PoseEstimationEvaluator


def _resolve(tmp_path, **changes):
    args = dict(target_id="mobilint-regulus", model_name="yolov8s-pose",
                task=Task.POSE_ESTIMATION, artifact_path=tmp_path / "model.mxq",
                requested_profile="regulus-vg-e2e-yolov8s-pose-raw-v1",
                requested_mode="auto", requested_layout="NCHW", layout_was_default=True)
    args.update(changes)
    return resolve_regulus_pose_profile(**args)


def test_regulus_pose_profile_accepts_only_explicit_regulus_contract(tmp_path):
    profile = _resolve(tmp_path)
    assert profile is REGULUS_VG_E2E_YOLOV8S_POSE_RAW_V1
    assert profile.runtime_contract() == {
        "vision_profile_id": "regulus-vg-e2e-yolov8s-pose-raw-v1",
        "expected_input_dtype": "uint8", "expected_input_layout": "NHWC",
        "expected_unbatched_input_shape": [640, 640, 3],
        "max_input_batch_size": 1,
        "expected_unbatched_output_shapes": [
            [80, 80, 51], [80, 80, 64], [80, 80, 1],
            [40, 40, 51], [40, 40, 64], [40, 40, 1],
            [20, 20, 51], [20, 20, 64], [20, 20, 1],
        ],
    }
    spec = Model_Spec(name="yolov8s-pose", task=Task.POSE_ESTIMATION,
                      input_shapes={"image": (1, 3, 640, 640)},
                      input_dtype={"image": "float32"},
                      output_shapes={"output": (1, 56, 8400)},
                      model_paths={"mobilint": "model.mxq"})
    updated = apply_regulus_pose_profile(spec, profile)
    assert updated.input_shapes["image"] == (1, 640, 640, 3)
    assert updated.input_dtype["image"] == "uint8"
    assert len(updated.output_shapes) == 9
    assert spec.input_shapes["image"] == (1, 3, 640, 640)
    for changes in (
        {"target_id": "mobilint-aries"}, {"requested_profile": "auto"},
        {"model_name": "yolov5m"}, {"task": Task.OBJECT_DETECTION},
        {"requested_layout": "NCHW", "layout_was_default": False},
        {"requested_mode": "normalized"},
    ):
        with pytest.raises(ValueError):
            _resolve(tmp_path, **changes)


def test_regulus_pose_loader_preserves_coco_identity_and_raw_input(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    Image.new("RGB", (100, 50), (255, 0, 0)).save(images / "one.png")
    annotation = tmp_path / "pose.json"
    annotation.write_text(json.dumps({
        "images": [{"id": 17, "file_name": "one.png", "width": 100, "height": 50}],
        "annotations": [{"id": 1, "image_id": 17, "category_id": 1, "keypoints": [0] * 51}],
        "categories": [{"id": 1, "name": "person", "keypoints": [str(i) for i in range(17)]}],
    }))
    spec = Model_Spec(name="yolov8s-pose", task=Task.POSE_ESTIMATION,
                      input_shapes={"image": (1, 640, 640, 3)},
                      input_dtype={"image": "uint8"}, output_shapes={"output": (1, 56, 8400)},
                      model_paths={"mobilint": "model.mxq"})
    loader = RegulusPoseEstimationLoader(spec, target_id="mobilint-regulus",
        regulus_pose_profile=_resolve(tmp_path), image_dir=images,
        label_path=annotation, dataset_path=tmp_path, cache_dir=tmp_path / "cache")
    sample = loader.load_by_index(0)
    cached_sample = loader.load_by_index(0)
    np.testing.assert_array_equal(cached_sample["input"], sample["input"])
    assert cached_sample["preprocess_context"] == sample["preprocess_context"]
    assert sample["input"].shape == (640, 640, 3)
    assert sample["input"].dtype == np.uint8
    assert sample["input"][0, 0].tolist() == [114, 114, 114]
    assert sample["input"][320, 320].tolist() == [255, 0, 0]
    assert sample["label"]["image_id"] == 17
    assert sample["preprocess_context"]["pad_y"] == 160
    assert loader.get_metadata()["image_preprocess_mode"] == "raw"
    assert loader.get_metadata()["runtime_options"]["vision_profile_id"].startswith("regulus-")


def test_regulus_pose_cache_separates_source_and_recipe(tmp_path):
    pre = RegulusPosePreprocessor(_resolve(tmp_path))
    first = tmp_path / "a" / "same.png"
    second = tmp_path / "b" / "same.png"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_bytes(b"one")
    second.write_bytes(b"two")
    key1 = pre.get_cache_path(tmp_path / "cache", first)
    assert key1 != pre.get_cache_path(tmp_path / "cache", second)
    first.write_bytes(b"changed")
    assert key1 != pre.get_cache_path(tmp_path / "cache", first)


def test_regulus_pose_decoder_to_coco_evaluator_restores_original_coordinates(tmp_path):
    fixture = write_coco_fixture(tmp_path)
    spec = apply_regulus_pose_profile(make_pose_spec(), _resolve(tmp_path))
    loader = RegulusPoseEstimationLoader(spec, target_id="mobilint-regulus",
        regulus_pose_profile=_resolve(tmp_path), image_dir=fixture["images"],
        label_path=fixture["pose"], dataset_path=tmp_path, cache_dir=tmp_path / "cache")
    sample = loader.load_by_index(0)
    assert sample["preprocess_context"]["scale"] == 80
    assert sample["preprocess_context"]["pad_y"] == 80
    heads = {}
    for size in (80, 40, 20):
        heads[f"k{size}"] = np.zeros((1, size, size, 51), dtype=np.float32)
        heads[f"d{size}"] = np.zeros((1, size, size, 64), dtype=np.float32)
        heads[f"c{size}"] = np.full((1, size, size, 1), -20, dtype=np.float32)
    heads["c80"][0, 40, 40, 0] = 10
    prediction = RegulusYoloV8PoseRawHeadDecoder(_resolve(tmp_path)).decode(heads)
    evaluator = PoseEstimationEvaluator(annotation_file=str(fixture["pose"]))
    evaluator.add_batch(prediction, [sample], 1.0)
    assert len(evaluator._records) == 1
    assert evaluator._records[0]["image_id"] == 1
    assert evaluator._records[0]["keypoints"][:3] == pytest.approx([4, 3, 0.5])
