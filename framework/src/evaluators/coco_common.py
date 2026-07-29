"""Shared state, validation, and official COCOeval boundary."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import numpy as np

from .base import Evaluator


_PYCOCO_IMPORT_ERROR: ImportError | None = None
try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
except ImportError as exc:  # pragma: no cover - exercised in dependency-free envs
    COCO = None
    COCOeval = None
    _PYCOCO_IMPORT_ERROR = exc


class CocoEvaluatorBase(Evaluator):
    """Provide validated streaming state for task-specific COCO evaluators."""

    CONTEXT_KEYS = (
        "original_height",
        "original_width",
        "input_height",
        "input_width",
        "scale",
        "pad_x",
        "pad_y",
    )

    def __init__(
        self,
        annotation_file: str,
        iou_type: str,
        **options: Any,
    ) -> None:
        del options
        if COCO is None or COCOeval is None:
            raise ImportError(
                "COCO segmentation and pose evaluation requires "
                "pycocotools==2.0.11; install framework/requirements.txt"
            ) from _PYCOCO_IMPORT_ERROR
        if not annotation_file:
            raise ValueError("annotation_file is required for COCO evaluation")
        annotation_path = Path(annotation_file)
        if not annotation_path.is_file():
            raise FileNotFoundError(
                f"COCO annotation file does not exist: {annotation_path}"
            )
        if iou_type not in {"segm", "keypoints"}:
            raise ValueError("COCO iou_type must be 'segm' or 'keypoints'")
        self.annotation_file = str(annotation_path)
        self.iou_type = iou_type
        self._coco_gt = COCO(self.annotation_file)
        self.category_ids = sorted(int(item) for item in self._coco_gt.getCatIds())
        self._valid_image_ids = {
            int(item) for item in self._coco_gt.getImgIds()
        }
        if not self.category_ids:
            raise ValueError("COCO ground truth contains no categories")
        if not self._valid_image_ids:
            raise ValueError("COCO ground truth contains no images")
        self._reset()

    def _reset(self) -> None:
        self._records: list[dict[str, Any]] = []
        self._image_ids: list[int] = []
        self._seen_image_ids: set[int] = set()
        self._timing_records: list[float] = []
        self._total_samples = 0
        self._total_detections = 0

    def _normalize_batch_labels(
        self, labels: Any
    ) -> list[dict[str, Any]]:
        if isinstance(labels, Mapping):
            items = [labels]
        elif isinstance(labels, Sequence) and not isinstance(
            labels, (str, bytes)
        ):
            items = list(labels)
        else:
            raise ValueError("COCO batch labels must be a list of mappings")

        normalized = []
        for local_index, item in enumerate(items):
            if not isinstance(item, Mapping):
                raise ValueError(
                    f"COCO batch label {local_index} must be a mapping"
                )
            label = item.get("label")
            if not isinstance(label, Mapping) or "image_id" not in label:
                raise ValueError(
                    f"COCO batch label {local_index} requires image_id"
                )
            image_id_value = label["image_id"]
            if isinstance(image_id_value, bool) or not isinstance(
                image_id_value, Integral
            ):
                raise ValueError("COCO image_id must be an integer")
            image_id = int(image_id_value)
            if image_id not in self._valid_image_ids:
                raise ValueError(f"unknown COCO image_id {image_id}")

            context = item.get("preprocess_context")
            normalized_context = self._normalize_context(context)
            normalized.append(
                {
                    "image_id": image_id,
                    "file_name": label.get("file_name"),
                    "preprocess_context": normalized_context,
                }
            )
        return normalized

    def _normalize_context(self, context: Any) -> dict[str, int | float]:
        if not isinstance(context, Mapping):
            raise ValueError("COCO label requires preprocess_context")
        for key in self.CONTEXT_KEYS:
            if key not in context:
                raise ValueError(f"preprocess_context requires {key}")
        normalized: dict[str, int | float] = {}
        dimension_keys = {
            "original_height",
            "original_width",
            "input_height",
            "input_width",
        }
        for key in self.CONTEXT_KEYS:
            value = context[key]
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError(f"preprocess_context {key} must be numeric")
            numeric = float(value)
            if not np.isfinite(numeric):
                raise ValueError(f"preprocess_context {key} must be finite")
            if key in dimension_keys:
                if not numeric.is_integer() or numeric <= 0:
                    raise ValueError(
                        f"preprocess_context {key} must be a positive integer"
                    )
                normalized[key] = int(numeric)
            else:
                normalized[key] = numeric
        if normalized["scale"] <= 0:
            raise ValueError("preprocess_context scale must be positive")
        if normalized["pad_x"] < 0 or normalized["pad_y"] < 0:
            raise ValueError("preprocess_context padding cannot be negative")
        return normalized

    def _restore_boxes(
        self, boxes: np.ndarray, context: Mapping[str, Any]
    ) -> np.ndarray:
        value = np.asarray(boxes)
        if value.ndim != 2 or value.shape[1] != 4:
            raise ValueError(f"boxes must have shape (N, 4), got {value.shape}")
        if not np.issubdtype(value.dtype, np.number) or not np.isfinite(
            value
        ).all():
            raise ValueError("boxes must contain finite numeric values")
        restored = value.astype(np.float32, copy=True)
        normalized = self._normalize_context(context)
        scale = float(normalized["scale"])
        restored[:, [0, 2]] = (
            restored[:, [0, 2]] - float(normalized["pad_x"])
        ) / scale
        restored[:, [1, 3]] = (
            restored[:, [1, 3]] - float(normalized["pad_y"])
        ) / scale
        restored[:, [0, 2]] = np.clip(
            restored[:, [0, 2]],
            0.0,
            float(normalized["original_width"]),
        )
        restored[:, [1, 3]] = np.clip(
            restored[:, [1, 3]],
            0.0,
            float(normalized["original_height"]),
        )
        return restored

    def _validate_local_indices(
        self, detections: np.ndarray, batch_size: int
    ) -> np.ndarray:
        value = np.asarray(detections)
        if value.ndim != 2 or value.shape[1] != 7:
            raise ValueError(
                f"detections must have shape (N, 7), got {value.shape}"
            )
        if not np.issubdtype(value.dtype, np.number) or not np.isfinite(
            value
        ).all():
            raise ValueError("detections must contain finite numeric values")
        local_indices = value[:, 0]
        if not np.equal(local_indices, np.floor(local_indices)).all():
            raise ValueError("detections require integer local image indices")
        if (
            np.any(local_indices < 0)
            or np.any(local_indices >= int(batch_size))
        ):
            raise ValueError("detection local image index is outside batch")
        return value.astype(np.float32, copy=False)

    def _category_id(self, local_class: Any) -> int:
        if isinstance(local_class, bool) or not isinstance(
            local_class, Real
        ):
            raise ValueError("local class index must be numeric")
        numeric = float(local_class)
        if not np.isfinite(numeric) or not numeric.is_integer():
            raise ValueError("local class index must be a finite integer")
        index = int(numeric)
        if index < 0 or index >= len(self.category_ids):
            raise ValueError(f"local class index {index} is out of range")
        return self.category_ids[index]

    def _record_batch(
        self,
        labels: list[dict[str, Any]],
        detection_count: int,
        timing_ms: Any,
    ) -> None:
        if isinstance(detection_count, bool) or not isinstance(
            detection_count, Integral
        ):
            raise ValueError("detection_count must be an integer")
        if detection_count < 0:
            raise ValueError("detection_count cannot be negative")
        image_ids = [int(item["image_id"]) for item in labels]
        duplicate_ids = self._seen_image_ids.intersection(image_ids)
        if len(set(image_ids)) != len(image_ids) or duplicate_ids:
            duplicates = sorted(
                duplicate_ids
                or {
                    image_id
                    for image_id in image_ids
                    if image_ids.count(image_id) > 1
                }
            )
            raise ValueError(f"duplicate evaluated COCO image_id: {duplicates}")
        timing_value = self._normalize_timing(timing_ms)
        self._seen_image_ids.update(image_ids)
        self._image_ids.extend(image_ids)
        self._total_samples += len(labels)
        self._total_detections += int(detection_count)
        self._timing_records.append(timing_value)

    @staticmethod
    def _normalize_timing(timing_ms: Any) -> float:
        if isinstance(timing_ms, Mapping):
            if "total_ms" not in timing_ms:
                raise ValueError("timing dictionary requires total_ms")
            timing_ms = timing_ms["total_ms"]
        if isinstance(timing_ms, bool) or not isinstance(timing_ms, Real):
            raise ValueError("timing_ms must be numeric")
        value = float(timing_ms)
        if not np.isfinite(value) or value < 0:
            raise ValueError("timing_ms must be finite and non-negative")
        return value

    def _latency_metrics(self) -> dict[str, int | float]:
        if self._timing_records:
            average_latency = float(np.mean(self._timing_records))
            p99_latency = float(np.percentile(self._timing_records, 99))
            total_latency = float(np.sum(self._timing_records))
        else:
            average_latency = 0.0
            p99_latency = 0.0
            total_latency = 0.0
        fps = (
            self._total_samples * 1000.0 / total_latency
            if total_latency > 0
            else 0.0
        )
        return {
            "Total Samples": self._total_samples,
            "Average Detections": self._total_detections
            / max(1, self._total_samples),
            "Average Latency (ms)": average_latency,
            "P99 Latency (ms)": p99_latency,
            "FPS": fps,
        }

    def _run_coco_eval(
        self,
        records: list[dict[str, Any]],
        image_ids: list[int],
    ) -> np.ndarray:
        normalized_image_ids = sorted({int(item) for item in image_ids})
        unknown = set(normalized_image_ids).difference(self._valid_image_ids)
        if unknown:
            raise ValueError(
                f"unknown COCO evaluation image IDs: {sorted(unknown)}"
            )
        stat_count = 10 if self.iou_type == "keypoints" else 12
        if not records:
            return np.zeros(stat_count, dtype=np.float64)
        if not normalized_image_ids:
            raise ValueError("COCO evaluation requires at least one image_id")
        coco_results = self._coco_gt.loadRes(records)
        evaluator = COCOeval(self._coco_gt, coco_results, self.iou_type)
        evaluator.params.imgIds = normalized_image_ids
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
        return np.asarray(evaluator.stats, dtype=np.float64)
