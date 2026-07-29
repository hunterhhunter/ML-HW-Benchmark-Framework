"""NumPy postprocessing for Ultralytics YOLOv8 instance segmentation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import cv2
import numpy as np

from .yolo_vision import (
    DETECTIONS_KEY,
    MASKS_KEY,
    as_bcn,
    class_aware_nms,
    resolve_output,
    xywh_to_xyxy,
)


class YoloV8SegmentationDecoder:
    """Decode raw YOLOv8s-seg predictions into boxes and binary masks."""

    FEATURE_COUNT = 116
    CLASS_COUNT = 80
    MASK_COUNT = 32

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
            "segmentation prediction",
        )
        raw_prototypes = resolve_output(
            outputs,
            lambda value: value.ndim == 4
            and value.shape[1] == self.MASK_COUNT,
            "segmentation prototypes",
        )
        prediction = as_bcn(raw_prediction, self.FEATURE_COUNT)
        prototypes = np.asarray(raw_prototypes)
        if not np.issubdtype(prototypes.dtype, np.number):
            raise ValueError("segmentation prototypes must be numeric")
        if not np.isfinite(prototypes).all():
            raise ValueError("segmentation prototypes must be finite")
        prototypes = prototypes.astype(np.float32, copy=False)
        if prototypes.shape[0] != prediction.shape[0]:
            raise ValueError("segmentation prediction/prototype batch mismatch")
        if prototypes.shape[2] <= 0 or prototypes.shape[3] <= 0:
            raise ValueError("segmentation prototype dimensions must be positive")
        return self._decode_batch(prediction, prototypes)

    def _decode_batch(
        self, prediction: np.ndarray, prototypes: np.ndarray
    ) -> dict[str, np.ndarray]:
        prototype_height, prototype_width = prototypes.shape[2:]
        input_height = prototype_height * 4
        input_width = prototype_width * 4
        detection_rows: list[list[float]] = []
        mask_rows: list[np.ndarray] = []

        for local_index, features in enumerate(
            prediction.transpose(0, 2, 1)
        ):
            class_probabilities = features[
                :, 4 : 4 + self.CLASS_COUNT
            ]
            class_ids = np.argmax(class_probabilities, axis=1).astype(
                np.int64, copy=False
            )
            scores = np.max(class_probabilities, axis=1)
            candidates = np.flatnonzero(scores > self.conf_threshold)
            if len(candidates) == 0:
                continue

            candidate_boxes = xywh_to_xyxy(features[candidates, :4])
            candidate_scores = scores[candidates]
            candidate_classes = class_ids[candidates]
            candidate_coefficients = features[
                candidates, 4 + self.CLASS_COUNT :
            ]
            keep = class_aware_nms(
                candidate_boxes,
                candidate_scores,
                candidate_classes,
                self.iou_threshold,
                self.max_detections,
            )
            if len(keep) == 0:
                continue

            selected_coefficients = candidate_coefficients[keep]
            prototype = prototypes[local_index].reshape(
                self.MASK_COUNT, -1
            )
            mask_logits = selected_coefficients @ prototype
            mask_logits = mask_logits.reshape(
                len(keep), prototype_height, prototype_width
            )

            for selected_row, logits in zip(keep, mask_logits):
                box = candidate_boxes[selected_row]
                score = float(candidate_scores[selected_row])
                class_id = int(candidate_classes[selected_row])
                detection_rows.append(
                    [
                        float(local_index),
                        float(class_id),
                        score,
                        *box.astype(float).tolist(),
                    ]
                )
                upsampled = cv2.resize(
                    logits,
                    (input_width, input_height),
                    interpolation=cv2.INTER_LINEAR,
                )
                mask_rows.append(
                    self._crop_and_threshold(
                        upsampled, box, input_height, input_width
                    )
                )

        detections = np.asarray(
            detection_rows, dtype=np.float32
        ).reshape(-1, 7)
        if mask_rows:
            masks = np.ascontiguousarray(
                np.stack(mask_rows, axis=0), dtype=np.uint8
            )
        else:
            masks = np.empty(
                (0, input_height, input_width), dtype=np.uint8
            )
        return {DETECTIONS_KEY: detections, MASKS_KEY: masks}

    @staticmethod
    def _crop_and_threshold(
        logits: np.ndarray,
        box: np.ndarray,
        input_height: int,
        input_width: int,
    ) -> np.ndarray:
        left = int(np.clip(np.floor(box[0]), 0, input_width))
        top = int(np.clip(np.floor(box[1]), 0, input_height))
        right = int(np.clip(np.ceil(box[2]), 0, input_width))
        bottom = int(np.clip(np.ceil(box[3]), 0, input_height))
        cropped = np.zeros((input_height, input_width), dtype=np.uint8)
        if right > left and bottom > top:
            cropped[top:bottom, left:right] = (
                logits[top:bottom, left:right] > 0.0
            ).astype(np.uint8)
        return cropped
