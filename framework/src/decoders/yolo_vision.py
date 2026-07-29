"""Validated NumPy primitives shared by YOLOv8 vision decoders."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np


DETECTIONS_KEY = "detections"
MASKS_KEY = "masks"
KEYPOINTS_KEY = "keypoints"


def as_bcn(array: np.ndarray, feature_count: int) -> np.ndarray:
    """Return a finite rank-three prediction in batch-channel-candidate form."""
    if feature_count <= 0:
        raise ValueError("feature_count must be positive")
    value = np.asarray(array)
    if value.ndim != 3:
        raise ValueError(f"expected rank-3 prediction, got {value.shape}")
    if not np.issubdtype(value.dtype, np.number):
        raise ValueError("prediction values must be numeric")
    if not np.isfinite(value).all():
        raise ValueError("prediction values must be finite")
    if value.shape[1] == feature_count and value.shape[2] != feature_count:
        normalized = value
    elif value.shape[2] == feature_count and value.shape[1] != feature_count:
        normalized = value.transpose(0, 2, 1)
    else:
        raise ValueError(
            "prediction does not contain a unique "
            f"{feature_count}-feature axis: {value.shape}"
        )
    return np.ascontiguousarray(normalized, dtype=np.float32)


def resolve_output(
    outputs: Mapping[str, Any],
    predicate: Callable[[np.ndarray], bool],
    description: str,
) -> np.ndarray:
    """Resolve exactly one runtime output matching a shape predicate."""
    matches = []
    for name, output in outputs.items():
        value = np.asarray(output)
        if predicate(value):
            matches.append((name, value))
    if not matches:
        raise ValueError(
            f"missing {description}; available outputs: {list(outputs)}"
        )
    if len(matches) > 1:
        raise ValueError(
            f"ambiguous {description}; matched {[name for name, _ in matches]}"
        )
    return matches[0][1]


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """Convert center-x/center-y/width/height boxes to corner coordinates."""
    value = np.asarray(boxes)
    if value.ndim < 1 or value.shape[-1] != 4:
        raise ValueError(f"boxes must end with four coordinates: {value.shape}")
    if not np.issubdtype(value.dtype, np.number) or not np.isfinite(value).all():
        raise ValueError("box coordinates must be finite numeric values")
    converted = value.astype(np.float32, copy=True)
    center_x = value[..., 0]
    center_y = value[..., 1]
    width = value[..., 2]
    height = value[..., 3]
    converted[..., 0] = center_x - width / 2.0
    converted[..., 1] = center_y - height / 2.0
    converted[..., 2] = center_x + width / 2.0
    converted[..., 3] = center_y + height / 2.0
    return converted


def class_aware_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    iou_threshold: float,
    max_detections: int,
) -> np.ndarray:
    """Apply stable per-class NMS and return global score-sorted row indices."""
    box_array = np.asarray(boxes)
    score_array = np.asarray(scores)
    class_array = np.asarray(class_ids)
    if box_array.ndim != 2 or box_array.shape[1] != 4:
        raise ValueError(f"boxes must have shape (N, 4), got {box_array.shape}")
    if score_array.ndim != 1 or class_array.ndim != 1:
        raise ValueError("scores and class_ids must be rank-one arrays")
    if not (
        len(box_array) == len(score_array) == len(class_array)
    ):
        raise ValueError("boxes, scores, and class_ids must have the same row count")
    if not np.issubdtype(box_array.dtype, np.number) or not np.isfinite(
        box_array
    ).all():
        raise ValueError("box coordinates must be finite numeric values")
    if not np.issubdtype(score_array.dtype, np.number) or not np.isfinite(
        score_array
    ).all():
        raise ValueError("scores must be finite numeric values")
    if not np.issubdtype(class_array.dtype, np.number) or not np.isfinite(
        class_array
    ).all():
        raise ValueError("class_ids must be finite numeric values")
    if not 0.0 <= float(iou_threshold) <= 1.0:
        raise ValueError("iou_threshold must be in [0, 1]")
    if isinstance(max_detections, bool) or int(max_detections) < 0:
        raise ValueError("max_detections cannot be negative")

    row_count = len(box_array)
    limit = int(max_detections)
    if row_count == 0 or limit == 0:
        return np.empty((0,), dtype=np.int64)

    original_indices = np.arange(row_count, dtype=np.int64)
    global_order = np.lexsort((original_indices, -score_array))
    kept: list[int] = []
    for class_id in np.unique(class_array):
        class_order = global_order[class_array[global_order] == class_id]
        while len(class_order):
            selected = int(class_order[0])
            kept.append(selected)
            if len(class_order) == 1:
                break
            remaining = class_order[1:]
            overlaps = _iou_one_to_many(
                box_array[selected], box_array[remaining]
            )
            class_order = remaining[overlaps <= iou_threshold]

    kept_array = np.asarray(kept, dtype=np.int64)
    final_order = np.lexsort(
        (kept_array, -score_array[kept_array])
    )
    return kept_array[final_order][:limit]


def _iou_one_to_many(box: np.ndarray, others: np.ndarray) -> np.ndarray:
    left = np.maximum(box[0], others[:, 0])
    top = np.maximum(box[1], others[:, 1])
    right = np.minimum(box[2], others[:, 2])
    bottom = np.minimum(box[3], others[:, 3])
    intersection = np.maximum(0.0, right - left) * np.maximum(
        0.0, bottom - top
    )
    box_area = max(0.0, float(box[2] - box[0])) * max(
        0.0, float(box[3] - box[1])
    )
    other_areas = np.maximum(0.0, others[:, 2] - others[:, 0]) * np.maximum(
        0.0, others[:, 3] - others[:, 1]
    )
    union = box_area + other_areas - intersection
    return np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection, dtype=np.float32),
        where=union > 0,
    )
