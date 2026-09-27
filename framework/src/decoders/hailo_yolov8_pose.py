"""Decode the nine raw YOLOv8 pose heads emitted by a Hailo HEF."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from .pose_estimation import YoloV8PoseDecoder


class HailoYoloV8PoseRawHeadDecoder:
    _SPATIAL = {(80, 80), (40, 40), (20, 20)}
    _CHANNELS = {1, 51, 64}

    def __init__(
        self,
        *,
        conf_threshold: float = 0.001,
        iou_threshold: float = 0.70,
        max_detections: int = 300,
        class_scores_are_probabilities: bool = False,
    ) -> None:
        self._postprocess = YoloV8PoseDecoder(
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            max_detections=max_detections,
        )
        self.class_scores_are_probabilities = bool(
            class_scores_are_probabilities
        )

    def decode(self, outputs: Mapping[str, Any]) -> dict[str, np.ndarray]:
        return self._postprocess.decode(
            {"output0": self._decode_heads(outputs)}
        )

    def result_metadata(self) -> dict[str, Any]:
        return {
            "hailo_yolo_raw_head_decode": "yolov8-pose-dfl-nhwc-v1",
            "hailo_yolov8_pose_class_activation": (
                "npu_sigmoid"
                if self.class_scores_are_probabilities
                else "host_sigmoid"
            ),
            "hailo_yolo_confidence_threshold": (
                self._postprocess.conf_threshold
            ),
            "hailo_yolo_iou_threshold": self._postprocess.iou_threshold,
            "hailo_yolo_max_detections": self._postprocess.max_detections,
        }

    def _decode_heads(self, outputs: Mapping[str, Any]) -> np.ndarray:
        heads = self._normalize_heads(outputs)
        levels = []
        for spatial in sorted(heads, reverse=True):
            keypoints, dfl, classes = heads[spatial]
            height, width = spatial
            locations = height * width
            grid = self._grid(height, width)
            distances = self._decode_dfl(dfl, locations)
            anchors = grid + 0.5
            top_left = anchors - distances[:, :2]
            bottom_right = anchors + distances[:, 2:]
            boxes = np.concatenate(
                (
                    (top_left + bottom_right) / 2.0,
                    bottom_right - top_left,
                ),
                axis=1,
            ) * float(640 // height)
            batch = dfl.shape[0]
            scores = classes.reshape(
                batch, locations, 1
            ).transpose(0, 2, 1)
            if not self.class_scores_are_probabilities:
                scores = _sigmoid(scores)
            pose = keypoints.reshape(
                batch, locations, 17, 3
            ).transpose(0, 2, 3, 1).copy()
            pose[:, :, :2] = (
                pose[:, :, :2] * 2.0
                + grid.reshape(1, 1, 2, locations)
            ) * float(640 // height)
            pose[:, :, 2:] = _sigmoid(pose[:, :, 2:])
            levels.append(
                np.concatenate(
                    (boxes, scores, pose.reshape(batch, 51, locations)),
                    axis=1,
                )
            )
        return np.ascontiguousarray(
            np.concatenate(levels, axis=2), dtype=np.float32
        )

    @staticmethod
    def _decode_dfl(values: np.ndarray, locations: int) -> np.ndarray:
        batch, height, width, channels = values.shape
        if channels != 64:
            raise ValueError(
                "Hailo pose DFL head requires 64 channels, "
                f"got {channels}."
            )
        logits = values.reshape(
            batch, height, width, 4, 16
        ).transpose(0, 3, 4, 1, 2)
        logits = logits.reshape(batch, 4, 16, locations)
        weights = np.arange(16, dtype=np.float32).reshape(1, 1, 16, 1)
        return (_softmax(logits, axis=2) * weights).sum(axis=2)

    def _normalize_heads(
        self, outputs: Mapping[str, Any]
    ) -> dict[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray]]:
        if len(outputs) != 9:
            raise ValueError(
                f"Hailo pose decoder expects nine raw heads, got {len(outputs)}."
            )
        grouped: dict[tuple[int, int], dict[int, np.ndarray]] = {}
        for name, value in outputs.items():
            array = np.asarray(value, dtype=np.float32)
            if array.ndim == 3:
                array = array[np.newaxis, ...]
            if array.ndim != 4:
                raise ValueError(
                    f"Hailo pose head {name!r} must be HWC or NHWC, "
                    f"got {array.shape}."
                )
            spatial = tuple(map(int, array.shape[1:3]))
            channels = int(array.shape[3])
            if spatial not in self._SPATIAL or channels not in self._CHANNELS:
                raise ValueError(
                    f"Hailo pose head {name!r} has unsupported shape "
                    f"{array.shape}."
                )
            if channels in grouped.setdefault(spatial, {}):
                raise ValueError(
                    "Hailo pose heads duplicate "
                    f"{channels}-channel tensor at {spatial}."
                )
            grouped[spatial][channels] = array
        if set(grouped) != self._SPATIAL or any(
            set(value) != self._CHANNELS for value in grouped.values()
        ):
            raise ValueError(
                "Hailo pose raw heads do not satisfy the 3-scale DFL contract."
            )
        return {
            spatial: (value[51], value[64], value[1])
            for spatial, value in grouped.items()
        }

    @staticmethod
    def _grid(height: int, width: int) -> np.ndarray:
        y, x = np.meshgrid(
            np.arange(height, dtype=np.float32),
            np.arange(width, dtype=np.float32),
            indexing="ij",
        )
        return np.stack((x, y), axis=0).reshape(
            1, 2, height * width
        )


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return 1.0 / (1.0 + np.exp(-values))


def _softmax(values: np.ndarray, *, axis: int) -> np.ndarray:
    shifted = values - np.max(values, axis=axis, keepdims=True)
    numerator = np.exp(shifted)
    return numerator / np.sum(numerator, axis=axis, keepdims=True)
