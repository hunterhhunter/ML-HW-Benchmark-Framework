import json

import numpy as np
import pytest

from coco_test_utils import write_coco_fixture
from evaluators.pose_estimation_evaluator import PoseEstimationEvaluator


def _pose_labels(
    *,
    original_height=6,
    original_width=8,
    input_height=6,
    input_width=8,
    scale=1.0,
    pad_x=0.0,
    pad_y=0.0,
):
    return [
        {
            "label": {
                "image_id": 1,
                "file_name": "000000000001.jpg",
            },
            "preprocess_context": {
                "original_height": original_height,
                "original_width": original_width,
                "input_height": input_height,
                "input_width": input_width,
                "scale": scale,
                "pad_x": pad_x,
                "pad_y": pad_y,
            },
        }
    ]


def _perfect_outputs():
    return {
        "detections": np.array(
            [[0, 0, 0.99, 2, 1, 6, 5]], dtype=np.float32
        ),
        "keypoints": np.tile([4, 3, 1.0], (1, 17, 1)).astype(
            np.float32
        ),
    }


def test_perfect_keypoints_have_near_one_oks_map(tmp_path):
    paths = write_coco_fixture(tmp_path)
    evaluator = PoseEstimationEvaluator(annotation_file=str(paths["pose"]))

    evaluator.add_batch(_perfect_outputs(), _pose_labels(), 2.0)
    metrics = evaluator.compute()

    assert metrics["OKS mAP"] > 0.99
    assert metrics["OKS AP50"] > 0.99
    assert metrics["Total Samples"] == 1
    assert metrics["Average Detections"] == 1.0
    assert len(evaluator._records[0]["keypoints"]) == 51


def test_shifted_and_empty_predictions_reduce_or_zero_oks(tmp_path):
    paths = write_coco_fixture(tmp_path)
    shifted = _perfect_outputs()
    shifted["keypoints"][:, :, :2] += 2.0
    evaluator = PoseEstimationEvaluator(annotation_file=str(paths["pose"]))
    evaluator.add_batch(shifted, _pose_labels(), 1.0)
    assert evaluator.compute()["OKS mAP"] < 0.99

    empty = PoseEstimationEvaluator(annotation_file=str(paths["pose"]))
    empty.add_batch(
        {
            "detections": np.empty((0, 7), dtype=np.float32),
            "keypoints": np.empty((0, 17, 3), dtype=np.float32),
        },
        _pose_labels(),
        {"total_ms": 1.0},
    )
    metrics = empty.compute()
    assert metrics["OKS mAP"] == 0.0
    assert metrics["OKS AP50"] == 0.0
    assert metrics["Total Samples"] == 1


def test_restore_keypoints_removes_padding_scale_and_preserves_confidence(
    tmp_path,
):
    paths = write_coco_fixture(tmp_path)
    evaluator = PoseEstimationEvaluator(annotation_file=str(paths["pose"]))
    context = _pose_labels(
        original_height=3,
        original_width=4,
        input_height=8,
        input_width=8,
        scale=2.0,
        pad_y=1.0,
    )[0]["preprocess_context"]
    keypoints = np.tile([6, 5, 0.7], (17, 1)).astype(np.float32)

    restored = evaluator._restore_keypoints(keypoints, context)

    np.testing.assert_allclose(restored[0], [3, 2, 0.7])


def test_pose_evaluator_rejects_row_and_keypoint_shape_mismatches(tmp_path):
    paths = write_coco_fixture(tmp_path)
    evaluator = PoseEstimationEvaluator(annotation_file=str(paths["pose"]))
    with pytest.raises(ValueError, match="row count"):
        evaluator.add_batch(
            {
                "detections": np.zeros((1, 7), dtype=np.float32),
                "keypoints": np.empty((0, 17, 3), dtype=np.float32),
            },
            _pose_labels(),
            1.0,
        )
    with pytest.raises(ValueError, match=r"\(N, 17, 3\)"):
        evaluator.add_batch(
            {
                "detections": np.zeros((1, 7), dtype=np.float32),
                "keypoints": np.zeros((1, 16, 3), dtype=np.float32),
            },
            _pose_labels(),
            1.0,
        )


def test_pose_evaluator_rejects_non_person_category_file(tmp_path):
    paths = write_coco_fixture(tmp_path)
    payload = json.loads(paths["pose"].read_text(encoding="utf-8"))
    payload["categories"][0]["id"] = 2
    paths["pose"].write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="category id 1"):
        PoseEstimationEvaluator(annotation_file=str(paths["pose"]))


def test_pose_evaluator_rejects_nonzero_local_class(tmp_path):
    paths = write_coco_fixture(tmp_path)
    evaluator = PoseEstimationEvaluator(annotation_file=str(paths["pose"]))
    outputs = _perfect_outputs()
    outputs["detections"][0, 1] = 1

    with pytest.raises(ValueError, match="local person class 0"):
        evaluator.add_batch(outputs, _pose_labels(), 1.0)
