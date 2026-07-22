"""NumPy decoder for Mobilint YOLOv5 raw output heads."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from .object_detection import DETECTIONS_KEY, DetectionDecoder, nms_pure_numpy


class MobilintYoloV5HeadDecoder(DetectionDecoder):
    """Decode profile-described YOLOv5 heads with multi-label, class-aware NMS."""

    def __init__(
        self,
        profile: Any,
        conf_threshold: float = 0.001,
        iou_threshold: float = 0.65,
        max_nms: int = 30_000,
        max_det: int = 300,
        max_class_offset: float = 7_680,
    ):
        if not 0.0 < conf_threshold < 1.0:
            raise ValueError("conf_threshold must be strictly between 0 and 1.")
        if max_nms <= 0:
            raise ValueError("max_nms must be positive.")
        if max_det <= 0:
            raise ValueError("max_det must be positive.")
        if max_class_offset <= 0:
            raise ValueError("max_class_offset must be positive.")

        recipe = getattr(profile, "output_recipe", None)
        if type(recipe).__name__ != "YoloV5RawHeadRecipe":
            raise ValueError(
                "MobilintYoloV5HeadDecoder requires a YoloV5RawHeadRecipe."
            )

        self.profile = profile
        self.recipe = recipe
        self.conf_threshold = float(conf_threshold)
        self.iou_threshold = float(iou_threshold)
        self.max_nms = int(max_nms)
        self.max_det = int(max_det)
        self.max_class_offset = float(max_class_offset)
        self._feature_count = 5 + int(recipe.class_count)
        self._heads_by_spatial = self._build_spatial_contract(profile, recipe)

    def decode(self, outputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        decoded, raw_objectness = self._decode_heads(outputs)
        rows = []
        inverse_conf = np.log(
            self.conf_threshold / (1.0 - self.conf_threshold)
        )

        for batch_index in range(decoded.shape[0]):
            candidates = decoded[batch_index][
                raw_objectness[batch_index] > inverse_conf
            ]
            if candidates.size == 0:
                continue

            scores = candidates[:, 4:5] * candidates[:, 5:]
            anchor_indices, class_indices = np.nonzero(scores > self.conf_threshold)
            if anchor_indices.size == 0:
                continue

            boxes = _xywh_to_xyxy(candidates[anchor_indices, :4])
            candidate_scores = scores[anchor_indices, class_indices]
            order = np.argsort(candidate_scores)[::-1][: self.max_nms]
            offset_boxes = (
                boxes[order]
                + class_indices[order, np.newaxis] * self.max_class_offset
            )
            keep = nms_pure_numpy(
                offset_boxes,
                candidate_scores[order],
                self.iou_threshold,
            )[: self.max_det]
            selected = order[np.asarray(keep, dtype=np.intp)]

            batch_column = np.full(selected.size, batch_index, dtype=np.float32)
            rows.append(
                np.column_stack(
                    (
                        batch_column,
                        class_indices[selected],
                        candidate_scores[selected],
                        boxes[selected],
                    )
                )
            )

        if not rows:
            return {DETECTIONS_KEY: np.empty((0, 7), dtype=np.float32)}
        return {
            DETECTIONS_KEY: np.concatenate(rows, axis=0).astype(np.float32, copy=False)
        }

    def _decode_heads(
        self, outputs: Dict[str, np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Normalize, decode, and concatenate heads for parity/debug tests."""
        normalized = self._normalize_heads(outputs)
        decoded_heads = []
        raw_objectness_heads = []

        for spatial, (stride, anchors) in self._heads_by_spatial.items():
            head = normalized[spatial]
            batch, height, width, _ = head.shape
            anchor_count = len(anchors)
            raw = head.reshape(
                batch,
                height,
                width,
                anchor_count,
                self._feature_count,
            )
            probabilities = _stable_sigmoid(raw)
            grid_y, grid_x = np.meshgrid(
                np.arange(height, dtype=np.float32),
                np.arange(width, dtype=np.float32),
                indexing="ij",
            )
            grid = np.stack((grid_x, grid_y), axis=-1).reshape(
                1, height, width, 1, 2
            )
            anchor_array = np.asarray(anchors, dtype=np.float32).reshape(
                1, 1, 1, anchor_count, 2
            )

            decoded = probabilities.copy()
            decoded[..., :2] = (
                probabilities[..., :2] * 2.0 - 0.5 + grid
            ) * float(stride)
            decoded[..., 2:4] = (
                probabilities[..., 2:4] * 2.0
            ) ** 2 * anchor_array
            decoded_heads.append(decoded.reshape(batch, -1, self._feature_count))
            raw_objectness_heads.append(raw[..., 4].reshape(batch, -1))

        return (
            np.concatenate(decoded_heads, axis=1).astype(np.float32, copy=False),
            np.concatenate(raw_objectness_heads, axis=1).astype(
                np.float32, copy=False
            ),
        )

    def _normalize_heads(
        self, outputs: Dict[str, np.ndarray]
    ) -> dict[tuple[int, int], np.ndarray]:
        if len(outputs) != int(self.recipe.expected_heads):
            raise ValueError(
                "MobilintYoloV5HeadDecoder expects exactly "
                f"{self.recipe.expected_heads} raw heads, got {len(outputs)}."
            )

        normalized: dict[tuple[int, int], np.ndarray] = {}
        expected_spatial = set(self._heads_by_spatial)
        expected_channels = {
            len(anchors) * self._feature_count
            for _, anchors in self._heads_by_spatial.values()
        }
        batch_size = None

        for name, value in outputs.items():
            head = np.asarray(value)
            if _looks_like_nchw(head.shape, expected_spatial, expected_channels):
                raise ValueError(
                    f"Raw head {name!r} must use NHWC layout, got {head.shape}."
                )
            if head.ndim == 3:
                head = head[np.newaxis, ...]
            elif head.ndim != 4:
                raise ValueError(
                    f"Raw head {name!r} must have (H,W,C) or (B,H,W,C) NHWC "
                    f"shape, got {head.shape}."
                )

            spatial = (int(head.shape[1]), int(head.shape[2]))
            if spatial in normalized:
                raise ValueError(f"Raw heads contain duplicate spatial size {spatial}.")
            if spatial not in self._heads_by_spatial:
                raise ValueError(
                    f"Raw head {name!r} has unexpected spatial size {spatial}; "
                    f"expected {sorted(expected_spatial)} in NHWC layout."
                )

            _, anchors = self._heads_by_spatial[spatial]
            expected_channel_count = len(anchors) * self._feature_count
            if head.shape[3] != expected_channel_count:
                raise ValueError(
                    f"Raw head {name!r} channel count must be "
                    f"{expected_channel_count}, got {head.shape[3]}."
                )
            if batch_size is None:
                batch_size = int(head.shape[0])
            elif head.shape[0] != batch_size:
                raise ValueError(
                    "Raw head batch sizes must match; "
                    f"expected {batch_size}, got {head.shape[0]} for {name!r}."
                )
            normalized[spatial] = np.asarray(head, dtype=np.float32)

        missing = expected_spatial.difference(normalized)
        if missing:
            raise ValueError(f"Raw heads are missing spatial sizes {sorted(missing)}.")
        return normalized

    def _build_spatial_contract(self, profile: Any, recipe: Any):
        input_hw = getattr(getattr(profile, "input_recipe", None), "input_hw", None)
        if input_hw is None:
            input_shape = tuple(profile.unbatched_input_shape)
            input_hw = input_shape[:2]
        input_height, input_width = (int(input_hw[0]), int(input_hw[1]))

        heads_by_spatial = {}
        for stride_value, anchor_values in recipe.anchors_by_stride:
            stride = int(stride_value)
            if stride <= 0 or input_height % stride or input_width % stride:
                raise ValueError(f"Invalid YOLOv5 stride {stride} for input {input_hw}.")
            spatial = (input_height // stride, input_width // stride)
            if spatial in heads_by_spatial:
                raise ValueError(f"Recipe has duplicate spatial size {spatial}.")
            anchors = tuple(tuple(float(axis) for axis in anchor) for anchor in anchor_values)
            heads_by_spatial[spatial] = (stride, anchors)

        if len(heads_by_spatial) != int(recipe.expected_heads):
            raise ValueError(
                "YoloV5RawHeadRecipe expected_heads does not match anchors_by_stride."
            )
        return heads_by_spatial


def _stable_sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    result = np.empty_like(values)
    positive = values >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


def _xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    converted = np.empty_like(boxes, dtype=np.float32)
    converted[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
    converted[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
    converted[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
    converted[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
    return converted


def _looks_like_nchw(
    shape: tuple[int, ...],
    expected_spatial: set[tuple[int, int]],
    expected_channels: set[int],
) -> bool:
    if len(shape) == 4:
        return shape[1] in expected_channels and (shape[2], shape[3]) in expected_spatial
    if len(shape) == 3:
        return shape[0] in expected_channels and (shape[1], shape[2]) in expected_spatial
    return False
