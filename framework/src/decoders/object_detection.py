"""Object detection output decoders.

Canonical detection rows are:
    [local_image_index, class_id, confidence, x1, y1, x2, y2]

Coordinates are in model-input pixels, matching the evaluator's ground-truth
coordinate system after preprocessing.
"""

from __future__ import annotations

import abc
from collections.abc import Iterable
from typing import Any, Dict, List, Tuple

import numpy as np


DETECTIONS_KEY = "detections"


class DetectionDecoder(abc.ABC):
    """Base interface for object detection output decoders."""

    @abc.abstractmethod
    def decode(self, outputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Convert runtime outputs into canonical detections."""


class RawYoloDetectionDecoder(DetectionDecoder):
    """Decode raw YOLOv5/YOLOv8 tensors and apply numpy NMS."""

    def __init__(self, conf_threshold: float = 0.25, iou_threshold: float = 0.45):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

    def decode(self, outputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        pred_key = next(iter(outputs))
        preds = np.asarray(outputs[pred_key])
        if preds.ndim == 2:
            preds = preds[np.newaxis, ...]
        if preds.ndim != 3:
            raise ValueError(
                "RawYoloDetectionDecoder expects (B, anchors, features) or "
                f"(B, features, anchors), got shape {preds.shape}"
            )

        detections = self._decode_raw_yolo(preds)
        return {DETECTIONS_KEY: _as_detection_array(detections)}

    def _decode_raw_yolo(self, preds: np.ndarray) -> List[List[float]]:
        detections: List[List[float]] = []

        if preds.shape[1] < preds.shape[2] and preds.shape[2] > 256:
            preds = np.transpose(preds, (0, 2, 1))
            is_yolov8 = True
        else:
            is_yolov8 = False

        for local_idx in range(preds.shape[0]):
            img_preds = preds[local_idx]
            if img_preds.ndim != 2 or img_preds.shape[1] < 5:
                continue

            p_cx, p_cy, p_w, p_h = (
                img_preds[:, 0],
                img_preds[:, 1],
                img_preds[:, 2],
                img_preds[:, 3],
            )
            boxes = np.stack(
                [p_cx - p_w / 2, p_cy - p_h / 2, p_cx + p_w / 2, p_cy + p_h / 2],
                axis=1,
            )

            if is_yolov8:
                class_probs = img_preds[:, 4:]
                if class_probs.shape[1] == 0:
                    continue
                class_confs = np.max(class_probs, axis=1)
                mask = class_confs > self.conf_threshold
                filtered_boxes = boxes[mask]
                filtered_conf = class_confs[mask]
                filtered_class_probs = class_probs[mask]
                final_confs = filtered_conf
            else:
                class_probs = img_preds[:, 5:]
                if class_probs.shape[1] == 0:
                    continue
                obj_conf = img_preds[:, 4]
                mask = obj_conf > self.conf_threshold
                filtered_boxes = boxes[mask]
                filtered_conf = obj_conf[mask]
                filtered_class_probs = class_probs[mask]
                final_confs = filtered_conf * np.max(filtered_class_probs, axis=1)

            if len(filtered_boxes) == 0:
                continue

            class_ids = np.argmax(filtered_class_probs, axis=1)
            keep_indices = _nms_pure_numpy(filtered_boxes, final_confs, self.iou_threshold)
            for keep_idx in keep_indices:
                detections.append(
                    [float(local_idx), float(class_ids[keep_idx]), float(final_confs[keep_idx])]
                    + filtered_boxes[keep_idx].astype(float).tolist()
                )

        return detections


class HailoYoloNMSDecoder(DetectionDecoder):
    """Decode Hailo YOLO NMS postprocess tensors.

    Expected Hailo layout is class-major, most commonly (B, classes, 5, max_boxes)
    or (classes, 5, max_boxes). The 5-value axis is interpreted as box coords
    plus confidence. The default box order is YOLO-style xyxy; set box_order
    when a compiled HEF exposes a different NMS coordinate convention.
    """

    def __init__(
        self,
        conf_threshold: float = 0.25,
        image_size: int | Tuple[int, int] = 640,
        box_order: str = "xyxy",
        clip_boxes: bool = True,
        debug: bool = False,
        debug_samples: int = 1,
    ):
        self.conf_threshold = conf_threshold
        self.target_h, self.target_w = _coerce_hw(image_size)
        self.box_order = box_order.lower()
        self.clip_boxes = clip_boxes
        self.debug = debug
        self.debug_samples = debug_samples
        self._debug_seen = 0

    def decode(self, outputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        pred_key = next(iter(outputs))
        ragged_detections = self._try_decode_ragged_nms(pred_key, outputs[pred_key])
        if ragged_detections is not None:
            return {DETECTIONS_KEY: _as_detection_array(ragged_detections)}

        pred = np.asarray(outputs[pred_key], dtype=np.float32)
        class_major = self._to_batched_class_major(pred)
        self._debug_tensor(pred_key, pred, class_major)

        detections: List[List[float]] = []
        for batch_idx in range(class_major.shape[0]):
            per_batch = class_major[batch_idx]
            for class_id in range(per_batch.shape[0]):
                values = per_batch[class_id]
                if values.shape[0] != 5:
                    continue

                coords = values[:4, :].T
                scores = values[4, :]
                mask = np.isfinite(scores) & (scores > self.conf_threshold)
                if not np.any(mask):
                    continue

                boxes = self._boxes_to_xyxy(coords[mask])
                boxes = self._scale_normalized_boxes(boxes)
                if self.clip_boxes:
                    boxes = self._clip_boxes(boxes)

                scores = scores[mask]
                valid = (
                    np.isfinite(boxes).all(axis=1)
                    & ((boxes[:, 2] - boxes[:, 0]) > 0)
                    & ((boxes[:, 3] - boxes[:, 1]) > 0)
                )
                for box, score in zip(boxes[valid], scores[valid]):
                    detections.append(
                        [float(batch_idx), float(class_id), float(score)]
                        + box.astype(float).tolist()
                    )

        return {DETECTIONS_KEY: _as_detection_array(detections)}

    def _try_decode_ragged_nms(
        self, pred_key: str, raw_value: Any
    ) -> List[List[float]] | None:
        """Decode HailoRT NMS outputs returned as per-class ragged arrays.

        Some HailoRT versions expose NMS postprocess output as a dense tensor,
        while others return an object/list container of per-class detections.
        Each class entry is expected to be empty or shaped like (N, 5)/(5, N):
        box coordinates plus score.
        """
        try:
            dense = np.asarray(raw_value)
            if dense.dtype != object:
                return None
        except ValueError:
            pass

        batches = self._normalize_ragged_batches(raw_value)
        if batches is None:
            return None

        detections: List[List[float]] = []
        debug_scores: List[float] = []
        debug_fields: List[np.ndarray] = []

        for batch_idx, per_batch in enumerate(batches):
            for class_id, class_item in enumerate(per_batch):
                rows = self._coerce_ragged_class_rows(class_item)
                if rows is None or rows.size == 0:
                    continue

                coords = rows[:, :4]
                scores = rows[:, 4]
                debug_scores.extend(scores[np.isfinite(scores)].astype(float).tolist())
                debug_fields.append(rows[:, :5])

                mask = np.isfinite(scores) & (scores > self.conf_threshold)
                if not np.any(mask):
                    continue

                boxes = self._boxes_to_xyxy(coords[mask])
                boxes = self._scale_normalized_boxes(boxes)
                if self.clip_boxes:
                    boxes = self._clip_boxes(boxes)

                scores = scores[mask]
                valid = (
                    np.isfinite(boxes).all(axis=1)
                    & ((boxes[:, 2] - boxes[:, 0]) > 0)
                    & ((boxes[:, 3] - boxes[:, 1]) > 0)
                )
                for box, score in zip(boxes[valid], scores[valid]):
                    detections.append(
                        [float(batch_idx), float(class_id), float(score)]
                        + box.astype(float).tolist()
                    )

        self._debug_ragged_tensor(pred_key, batches, debug_scores, debug_fields)
        return detections

    def _normalize_ragged_batches(self, raw_value: Any) -> List[List[Any]] | None:
        seq = _to_sequence(raw_value)
        if seq is None:
            return None
        if _is_class_collection(seq):
            return [seq]

        batches: List[List[Any]] = []
        for item in seq:
            item_seq = _to_sequence(item)
            if item_seq is None or not _is_class_collection(item_seq):
                return None
            batches.append(item_seq)
        return batches

    def _coerce_ragged_class_rows(self, class_item: Any) -> np.ndarray | None:
        arr = np.asarray(class_item, dtype=np.float32)
        if arr.size == 0:
            return np.empty((0, 5), dtype=np.float32)
        if arr.ndim == 1:
            if arr.shape[0] < 5:
                return None
            arr = arr.reshape(1, -1)
        elif arr.ndim == 2:
            if arr.shape[1] < 5 and arr.shape[0] >= 5:
                arr = arr.T
        else:
            return None

        if arr.ndim != 2 or arr.shape[1] < 5:
            return None
        return arr[:, :5].astype(np.float32, copy=False)

    def _debug_tensor(
        self, pred_key: str, pred: np.ndarray, class_major: np.ndarray
    ) -> None:
        if not self.debug or self._debug_seen >= self.debug_samples:
            return

        finite = pred[np.isfinite(pred)]
        if finite.size == 0:
            print(
                f"[HailoYoloNMSDecoder][debug] output={pred_key} "
                f"shape={pred.shape} dtype={pred.dtype} all_non_finite"
            )
            self._debug_seen += 1
            return

        field_values = class_major[:, :, :5, :]
        field_mins = np.min(field_values, axis=(0, 1, 3))
        field_maxs = np.max(field_values, axis=(0, 1, 3))
        scores = class_major[:, :, 4, :]
        score_finite = scores[np.isfinite(scores)]
        if score_finite.size:
            top_scores = np.sort(score_finite.reshape(-1))[-5:][::-1].astype(float).tolist()
            score_min = float(np.min(score_finite))
            score_max = float(np.max(score_finite))
            above = int(np.sum(score_finite > self.conf_threshold))
        else:
            top_scores = []
            score_min = float("nan")
            score_max = float("nan")
            above = 0

        print(
            "[HailoYoloNMSDecoder][debug] "
            f"output={pred_key} raw_shape={pred.shape} class_major_shape={class_major.shape} "
            f"dtype={pred.dtype} min={float(np.min(finite)):.6g} max={float(np.max(finite)):.6g} "
            f"field_mins={field_mins.astype(float).tolist()} "
            f"field_maxs={field_maxs.astype(float).tolist()} "
            f"score_min={score_min:.6g} score_max={score_max:.6g} "
            f"scores_above_threshold={above} threshold={self.conf_threshold} "
            f"top_scores={top_scores}"
        )
        self._debug_seen += 1

    def _debug_ragged_tensor(
        self,
        pred_key: str,
        batches: List[List[Any]],
        scores: List[float],
        fields: List[np.ndarray],
    ) -> None:
        if not self.debug or self._debug_seen >= self.debug_samples:
            return

        score_arr = np.asarray(scores, dtype=np.float32)
        non_empty_classes = 0
        for per_batch in batches:
            for class_item in per_batch:
                rows = self._coerce_ragged_class_rows(class_item)
                if rows is not None and rows.size > 0:
                    non_empty_classes += 1
        if score_arr.size:
            top_scores = np.sort(score_arr.reshape(-1))[-5:][::-1].astype(float).tolist()
            score_min = float(np.min(score_arr))
            score_max = float(np.max(score_arr))
            above = int(np.sum(score_arr > self.conf_threshold))
        else:
            top_scores = []
            score_min = float("nan")
            score_max = float("nan")
            above = 0

        if fields:
            field_arr = np.concatenate(fields, axis=0)
            field_mins = np.min(field_arr, axis=0).astype(float).tolist()
            field_maxs = np.max(field_arr, axis=0).astype(float).tolist()
        else:
            field_mins = []
            field_maxs = []

        print(
            "[HailoYoloNMSDecoder][debug] "
            f"output={pred_key} ragged_batches={len(batches)} "
            f"classes_per_batch={[len(batch) for batch in batches]} "
            f"non_empty_classes={non_empty_classes} "
            f"field_mins={field_mins} field_maxs={field_maxs} "
            f"score_min={score_min:.6g} score_max={score_max:.6g} "
            f"scores_above_threshold={above} threshold={self.conf_threshold} "
            f"top_scores={top_scores}"
        )
        self._debug_seen += 1

    def _to_batched_class_major(self, pred: np.ndarray) -> np.ndarray:
        arr = pred
        if arr.ndim == 3:
            arr = arr[np.newaxis, ...]
        if arr.ndim != 4:
            raise ValueError(
                "HailoYoloNMSDecoder expects class-major NMS output with 3 or 4 dims, "
                f"got shape {pred.shape}"
            )

        if arr.shape[2] == 5:
            return arr
        if arr.shape[3] == 5:
            return np.transpose(arr, (0, 1, 3, 2))
        if arr.shape[1] == 5:
            return np.transpose(arr, (0, 2, 1, 3))

        raise ValueError(
            "HailoYoloNMSDecoder could not find the 5-value bbox axis in "
            f"shape {pred.shape}"
        )

    def _boxes_to_xyxy(self, coords: np.ndarray) -> np.ndarray:
        if self.box_order == "yxyx":
            y1, x1, y2, x2 = coords.T
            return np.stack([x1, y1, x2, y2], axis=1)
        if self.box_order == "xyxy":
            return coords.astype(np.float32, copy=False)
        if self.box_order == "xywh":
            cx, cy, w, h = coords.T
            return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
        raise ValueError(f"Unsupported Hailo NMS box_order: {self.box_order}")

    def _scale_normalized_boxes(self, boxes: np.ndarray) -> np.ndarray:
        if boxes.size == 0:
            return boxes
        finite = boxes[np.isfinite(boxes)]
        if finite.size == 0:
            return boxes
        if float(np.max(np.abs(finite))) <= 2.0:
            boxes = boxes.copy()
            boxes[:, [0, 2]] *= self.target_w
            boxes[:, [1, 3]] *= self.target_h
        return boxes

    def _clip_boxes(self, boxes: np.ndarray) -> np.ndarray:
        boxes = boxes.copy()
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0.0, float(self.target_w))
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0.0, float(self.target_h))
        return boxes


def _nms_pure_numpy(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> List[int]:
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = np.maximum(x2 - x1, 0.0) * np.maximum(y2 - y1, 0.0)

    order = scores.argsort()[::-1]
    keep: List[int] = []

    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / np.maximum(areas[i] + areas[order[1:]] - inter, 1e-6)
        order = order[np.where(iou <= iou_threshold)[0] + 1]

    return keep


def _as_detection_array(rows: List[List[float]]) -> np.ndarray:
    if not rows:
        return np.empty((0, 7), dtype=np.float32)
    return np.asarray(rows, dtype=np.float32).reshape(-1, 7)


def _coerce_hw(image_size: int | Tuple[int, int]) -> Tuple[int, int]:
    if isinstance(image_size, tuple):
        return int(image_size[0]), int(image_size[1])
    return int(image_size), int(image_size)


def _to_sequence(value: Any) -> List[Any] | None:
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            return value.tolist()
        if value.ndim == 0:
            return None
        return list(value)
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, dict)):
        return list(value)
    return None


def _is_class_collection(seq: List[Any]) -> bool:
    if not isinstance(seq, list):
        return False
    if not seq:
        return True
    for item in seq:
        try:
            arr = np.asarray(item, dtype=np.float32)
        except (TypeError, ValueError):
            return False
        if arr.size == 0:
            continue
        if arr.ndim == 1 and arr.shape[0] >= 5:
            continue
        if arr.ndim == 2 and (arr.shape[0] >= 5 or arr.shape[1] >= 5):
            continue
        return False
    return True
