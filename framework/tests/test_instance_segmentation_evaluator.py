import json

import numpy as np
import pytest

from coco_test_utils import write_coco_fixture
from evaluators.instance_segmentation_evaluator import (
    InstanceSegmentationEvaluator,
)


def _seg_labels(
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
    mask = np.zeros((1, 6, 8), dtype=np.uint8)
    mask[0, 1:5, 2:6] = 1
    return {
        "detections": np.array(
            [[0, 0, 0.99, 2, 1, 6, 5]], dtype=np.float32
        ),
        "masks": mask,
    }


def test_perfect_mask_prediction_has_near_one_coco_map(tmp_path):
    paths = write_coco_fixture(tmp_path)
    evaluator = InstanceSegmentationEvaluator(
        annotation_file=str(paths["instances"])
    )

    evaluator.add_batch(_perfect_outputs(), _seg_labels(), 2.0)
    metrics = evaluator.compute()

    assert metrics["Mask mAP"] > 0.99
    assert metrics["Mask AP50"] > 0.99
    assert metrics["Total Samples"] == 1
    assert metrics["Average Detections"] == 1.0
    assert isinstance(evaluator._records[0]["segmentation"]["counts"], str)


def test_shifted_and_empty_predictions_reduce_or_zero_metrics(tmp_path):
    paths = write_coco_fixture(tmp_path)
    shifted = _perfect_outputs()
    shifted["masks"] = np.roll(shifted["masks"], 2, axis=2)
    evaluator = InstanceSegmentationEvaluator(
        annotation_file=str(paths["instances"])
    )
    evaluator.add_batch(shifted, _seg_labels(), 1.0)
    assert evaluator.compute()["Mask mAP"] < 0.99

    empty = InstanceSegmentationEvaluator(
        annotation_file=str(paths["instances"])
    )
    empty.add_batch(
        {
            "detections": np.empty((0, 7), dtype=np.float32),
            "masks": np.empty((0, 6, 8), dtype=np.uint8),
        },
        _seg_labels(),
        {"total_ms": 1.0},
    )
    metrics = empty.compute()
    assert metrics["Mask mAP"] == 0.0
    assert metrics["Mask AP50"] == 0.0
    assert metrics["Total Samples"] == 1


def test_coco_eval_is_limited_to_seen_image_ids(tmp_path):
    paths = write_coco_fixture(tmp_path)
    payload = json.loads(paths["instances"].read_text(encoding="utf-8"))
    payload["images"].append(
        {"id": 2, "file_name": "unseen.jpg", "width": 8, "height": 6}
    )
    second = dict(payload["annotations"][0])
    second.update({"id": 2, "image_id": 2})
    payload["annotations"].append(second)
    paths["instances"].write_text(json.dumps(payload), encoding="utf-8")
    evaluator = InstanceSegmentationEvaluator(
        annotation_file=str(paths["instances"])
    )

    evaluator.add_batch(_perfect_outputs(), _seg_labels(), 1.0)

    assert evaluator.compute()["Mask mAP"] > 0.99


def test_restore_mask_removes_letterbox_padding(tmp_path):
    paths = write_coco_fixture(tmp_path)
    evaluator = InstanceSegmentationEvaluator(
        annotation_file=str(paths["instances"])
    )
    padded = np.zeros((8, 8), dtype=np.uint8)
    padded[2:6] = 1
    context = _seg_labels(
        original_height=2,
        original_width=4,
        input_height=8,
        input_width=8,
        scale=2.0,
        pad_y=2.0,
    )[0]["preprocess_context"]

    restored = evaluator._restore_mask(padded, context)

    assert restored.shape == (2, 4)
    assert restored.dtype == np.uint8
    assert restored.all()


def test_mask_count_must_match_detection_count(tmp_path):
    paths = write_coco_fixture(tmp_path)
    evaluator = InstanceSegmentationEvaluator(
        annotation_file=str(paths["instances"])
    )

    with pytest.raises(ValueError, match="row count"):
        evaluator.add_batch(
            {
                "detections": np.zeros((1, 7), dtype=np.float32),
                "masks": np.zeros((0, 6, 8), dtype=np.uint8),
            },
            _seg_labels(),
            1.0,
        )


def test_masks_must_match_input_geometry_and_be_binary(tmp_path):
    paths = write_coco_fixture(tmp_path)
    evaluator = InstanceSegmentationEvaluator(
        annotation_file=str(paths["instances"])
    )
    outputs = _perfect_outputs()
    outputs["masks"] = np.full((1, 6, 8), 2, dtype=np.uint8)
    with pytest.raises(ValueError, match="binary"):
        evaluator.add_batch(outputs, _seg_labels(), 1.0)

    outputs = _perfect_outputs()
    outputs["masks"] = np.zeros((1, 5, 8), dtype=np.uint8)
    with pytest.raises(ValueError, match="input geometry"):
        evaluator.add_batch(outputs, _seg_labels(), 1.0)
