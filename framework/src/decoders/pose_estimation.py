"""NumPy postprocessing for Ultralytics YOLOv8 pose estimation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from .yolo_vision import (
    DETECTIONS_KEY,
    KEYPOINTS_KEY,
    as_bcn,
    class_aware_nms,
    resolve_output,
    xywh_to_xyxy,
)


class YoloV8PoseDecoder:
    """Decode YOLOv8s-pose predictions into boxes and 17 COCO keypoints."""

    FEATURE_COUNT = 56
    KEYPOINT_SHAPE = (17, 3)

    def __init__(
        self,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        max_detections: int = 300,
    ) -> None:
        if not 0.0 <= float(conf_threshold) <= 1.0:
            raise ValueError("conf_threshold must be in [0, 1]")
        if not 0.0 <= float(iou_threshold) <= 1.0:
            raise ValueError("iou_threshold must be in [0, 1]")
        if isinstance(max_detections, bool) or int(max_detections) < 0:
            raise ValueError("max_detections cannot be negative")
        self.conf_threshold = float(conf_threshold)
        self.iou_threshold = float(iou_threshold)
        self.max_detections = int(max_detections)

    def decode(
        self, outputs: Mapping[str, Any]
    ) -> dict[str, np.ndarray]:
        raw_prediction = resolve_output(
            outputs,
            lambda value: value.ndim == 3
            and self.FEATURE_COUNT in value.shape,
            "pose prediction",
        )
        prediction = as_bcn(raw_prediction, self.FEATURE_COUNT)
        detection_rows: list[list[float]] = []
        keypoint_rows: list[np.ndarray] = []

        for local_index, features in enumerate(
            prediction.transpose(0, 2, 1)
        ):
            scores = features[:, 4]
            candidates = np.flatnonzero(scores > self.conf_threshold)
            if len(candidates) == 0:
                continue
            candidate_boxes = xywh_to_xyxy(features[candidates, :4])
            candidate_scores = scores[candidates]
            candidate_keypoints = features[candidates, 5:].reshape(
                -1, *self.KEYPOINT_SHAPE
            )
            keep = class_aware_nms(
                candidate_boxes,
                candidate_scores,
                np.zeros(len(candidates), dtype=np.int64),
                self.iou_threshold,
                self.max_detections,
            )
            for selected_row in keep:
                box = candidate_boxes[selected_row]
                detection_rows.append(
                    [
                        float(local_index),
                        0.0,
                        float(candidate_scores[selected_row]),
                        *box.astype(float).tolist(),
                    ]
                )
                keypoint_rows.append(candidate_keypoints[selected_row])

        detections = np.asarray(
            detection_rows, dtype=np.float32
        ).reshape(-1, 7)
        keypoints = np.asarray(
            keypoint_rows, dtype=np.float32
        ).reshape(-1, *self.KEYPOINT_SHAPE)
        return {DETECTIONS_KEY: detections, KEYPOINTS_KEY: keypoints}
