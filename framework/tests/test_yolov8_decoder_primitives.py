import numpy as np
import pytest

from decoders.yolo_vision import (
    as_bcn,
    class_aware_nms,
    resolve_output,
    xywh_to_xyxy,
)


def test_as_bcn_accepts_bcn_and_transposes_bnc():
    bcn = np.zeros((2, 56, 10), dtype=np.float32)

    assert as_bcn(bcn, 56).shape == (2, 56, 10)
    assert as_bcn(bcn.transpose(0, 2, 1), 56).shape == (2, 56, 10)


def test_as_bcn_rejects_ambiguous_and_non_finite_values():
    with pytest.raises(ValueError, match="unique 56-feature axis"):
        as_bcn(np.zeros((1, 56, 56), dtype=np.float32), 56)
    values = np.zeros((1, 56, 1), dtype=np.float32)
    values[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        as_bcn(values, 56)


def test_xywh_to_xyxy_converts_without_mutating_input():
    boxes = np.array([[5, 6, 4, 2]], dtype=np.float32)

    converted = xywh_to_xyxy(boxes)

    np.testing.assert_array_equal(converted, [[3, 5, 7, 7]])
    np.testing.assert_array_equal(boxes, [[5, 6, 4, 2]])


def test_class_aware_nms_keeps_overlapping_different_classes():
    boxes = np.array(
        [[0, 0, 10, 10], [1, 1, 9, 9], [1, 1, 9, 9]],
        dtype=np.float32,
    )
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    classes = np.array([0, 0, 1], dtype=np.int64)

    kept = class_aware_nms(boxes, scores, classes, 0.5, 10)

    np.testing.assert_array_equal(kept, [0, 2])


def test_class_aware_nms_has_stable_ties_and_global_limit():
    boxes = np.array(
        [[0, 0, 1, 1], [3, 3, 4, 4], [6, 6, 7, 7]],
        dtype=np.float32,
    )
    scores = np.array([0.5, 0.5, 0.9], dtype=np.float32)
    classes = np.array([0, 1, 1], dtype=np.int64)

    kept = class_aware_nms(boxes, scores, classes, 0.5, 2)

    np.testing.assert_array_equal(kept, [2, 0])


def test_class_aware_nms_rejects_misaligned_rows():
    with pytest.raises(ValueError, match="same row count"):
        class_aware_nms(
            np.zeros((2, 4), dtype=np.float32),
            np.zeros((1,), dtype=np.float32),
            np.zeros((2,), dtype=np.int64),
            0.5,
            10,
        )


def test_resolve_output_selects_unique_match_and_rejects_ambiguity():
    outputs = {
        "other": np.zeros((1, 5), dtype=np.float32),
        "pose": np.zeros((1, 56, 2), dtype=np.float32),
    }
    selected = resolve_output(
        outputs,
        lambda value: value.ndim == 3 and 56 in value.shape,
        "pose prediction",
    )
    assert selected is outputs["pose"]

    outputs["also_pose"] = np.zeros((1, 2, 56), dtype=np.float32)
    with pytest.raises(ValueError, match="ambiguous pose prediction"):
        resolve_output(
            outputs,
            lambda value: value.ndim == 3 and 56 in value.shape,
            "pose prediction",
        )
