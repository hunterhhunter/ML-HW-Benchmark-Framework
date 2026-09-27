"""Host decoder for DeepX YOLOv8 pose artifacts cut at nine raw heads."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from .hailo_yolov8_pose import HailoYoloV8PoseRawHeadDecoder


def _as_nhwc_head(
    name: str,
    value: Any,
    *,
    expected_channels: set[int],
) -> tuple[np.ndarray, str]:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 3:
        array = array[np.newaxis, :, :, :]
    if array.ndim != 4:
        raise ValueError(
            f"DeepX raw head {name!r} must be HWC, CHW, NHWC, or NCHW, "
            f"got {array.shape}."
        )

    first_candidate = int(array.shape[1]) in expected_channels
    last_candidate = int(array.shape[-1]) in expected_channels
    if first_candidate and not last_candidate:
        return np.ascontiguousarray(array.transpose(0, 2, 3, 1)), "NCHW"
    if last_candidate and not first_candidate:
        return np.ascontiguousarray(array), "NHWC"
    if first_candidate and last_candidate:
        raise ValueError(
            f"DeepX raw head {name!r} has ambiguous channel axis in "
            f"shape {array.shape}."
        )
    raise ValueError(
        f"DeepX raw head {name!r} has unsupported channels in "
        f"shape {array.shape}; expected one of {sorted(expected_channels)}."
    )


class DeepXYoloV8PoseRawHeadDecoder:
    """Normalize DeepX NCHW heads and apply the shared host Pose decoder."""

    _EXPECTED_CHANNELS = {1, 51, 64}

    def __init__(
        self,
        *,
        conf_threshold: float = 0.001,
        iou_threshold: float = 0.70,
        max_detections: int = 300,
        class_scores_are_probabilities: bool = False,
        raw_head_abi: str = "yolov8-pose-dfl-nchw-v1",
    ) -> None:
        self._decoder = HailoYoloV8PoseRawHeadDecoder(
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            max_detections=max_detections,
            class_scores_are_probabilities=class_scores_are_probabilities,
        )
        self.class_scores_are_probabilities = bool(
            class_scores_are_probabilities
        )
        self.raw_head_abi = str(raw_head_abi)
        self._source_layout = "unobserved"

    def decode(self, outputs: Mapping[str, Any]) -> dict[str, np.ndarray]:
        normalized, source_layout = self._normalize_heads(outputs)
        self._source_layout = source_layout
        bounded_activations = {
            name: (
                np.clip(value, -80.0, 80.0)
                if int(value.shape[-1]) in {1, 51}
                else value
            )
            for name, value in normalized.items()
        }
        return self._decoder.decode(bounded_activations)

    def result_metadata(self) -> dict[str, Any]:
        return {
            "deepx_raw_head_abi": self.raw_head_abi,
            "deepx_raw_head_layout_source": self._source_layout,
            "deepx_yolov8_pose_dfl": "softmax_host",
            "deepx_yolov8_pose_score": (
                "sigmoid_npu"
                if self.class_scores_are_probabilities
                else "sigmoid_host"
            ),
            "deepx_yolov8_pose_visibility": "sigmoid_host",
        }

    def _normalize_heads(
        self,
        outputs: Mapping[str, Any],
    ) -> tuple[dict[str, np.ndarray], str]:
        normalized: dict[str, np.ndarray] = {}
        layouts: set[str] = set()
        grouped: dict[tuple[int, int], set[int]] = {}
        for name, value in outputs.items():
            head, layout = _as_nhwc_head(
                str(name),
                value,
                expected_channels=self._EXPECTED_CHANNELS,
            )
            spatial = (int(head.shape[1]), int(head.shape[2]))
            channels = int(head.shape[3])
            if spatial not in {(80, 80), (40, 40), (20, 20)}:
                raise ValueError(
                    f"DeepX pose raw head {name!r} has unsupported spatial "
                    f"size {spatial}."
                )
            level = grouped.setdefault(spatial, set())
            if channels in level:
                raise ValueError(
                    "DeepX pose raw heads duplicate "
                    f"{channels}-channel tensor at {spatial}."
                )
            level.add(channels)
            normalized[str(name)] = head
            layouts.add(layout)
        if len(layouts) != 1:
            raise ValueError(
                "DeepX YOLOv8 pose raw heads must use one common source "
                f"layout, got {sorted(layouts)}."
            )
        if set(grouped) != {(80, 80), (40, 40), (20, 20)} or any(
            channels != self._EXPECTED_CHANNELS
            for channels in grouped.values()
        ):
            raise ValueError(
                "DeepX pose raw heads are missing one or more expected heads."
            )
        return normalized, next(iter(layouts))
