"""Official COCO instance-segmentation evaluator with streaming RLE records."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import cv2
import numpy as np

from core.inference_result import InferenceResult
from core.model_spec import Model_Spec, Task
from decoders.yolo_vision import DETECTIONS_KEY, MASKS_KEY

from .coco_common import CocoEvaluatorBase


_MASK_IMPORT_ERROR: ImportError | None = None
try:
    from pycocotools import mask as mask_utils
except ImportError as exc:  # pragma: no cover - dependency-free environments
    mask_utils = None
    _MASK_IMPORT_ERROR = exc


class InstanceSegmentationEvaluator(CocoEvaluatorBase):
    """Accumulate compact COCO RLE records and compute official mask AP."""

    METRIC_INDEX = {
        "Mask mAP": 0,
        "Mask AP50": 1,
        "Mask AP75": 2,
        "Mask AP Small": 3,
        "Mask AP Medium": 4,
        "Mask AP Large": 5,
    }

    def __init__(self, annotation_file: str, **options: Any) -> None:
        super().__init__(annotation_file, "segm", **options)
        if mask_utils is None:
            raise ImportError(
                "COCO mask encoding requires pycocotools==2.0.11"
            ) from _MASK_IMPORT_ERROR

    def add_batch(
        self,
        outputs: Mapping[str, Any],
        labels: Any,
        timing_ms: Any,
    ) -> None:
        if DETECTIONS_KEY not in outputs or MASKS_KEY not in outputs:
            raise ValueError(
                "segmentation outputs require detections and masks"
            )
        batch = self._normalize_batch_labels(labels)
        detections = self._validate_local_indices(
            np.asarray(outputs[DETECTIONS_KEY]), len(batch)
        )
        masks = np.asarray(outputs[MASKS_KEY])
        if masks.ndim != 3:
            raise ValueError(
                f"segmentation masks must have shape (N, H, W), got {masks.shape}"
            )
        if len(detections) != len(masks):
            raise ValueError("segmentation detection/mask row count mismatch")
        if not np.issubdtype(masks.dtype, np.number) or not np.isfinite(
            masks
        ).all():
            raise ValueError("segmentation masks must be finite numeric values")
        if not np.logical_or(masks == 0, masks == 1).all():
            raise ValueError("segmentation masks must be binary")

        batch_records = []
        for detection, binary_mask in zip(detections, masks):
            local_index = int(detection[0])
            item = batch[local_index]
            context = item["preprocess_context"]
            restored_mask = self._restore_mask(binary_mask, context)
            rle = mask_utils.encode(
                np.asfortranarray(restored_mask, dtype=np.uint8)
            )
            counts = rle.get("counts")
            if isinstance(counts, bytes):
                rle["counts"] = counts.decode("ascii")

            restored_box = self._restore_boxes(
                detection[None, 3:7], context
            )[0]
            width = float(restored_box[2] - restored_box[0])
            height = float(restored_box[3] - restored_box[1])
            if width <= 0 or height <= 0:
                raise ValueError("segmentation detection box must have positive area")
            batch_records.append(
                {
                    "image_id": item["image_id"],
                    "category_id": self._category_id(detection[1]),
                    "segmentation": rle,
                    "bbox": [
                        float(restored_box[0]),
                        float(restored_box[1]),
                        width,
                        height,
                    ],
                    "score": float(detection[2]),
                }
            )

        self._record_batch(batch, len(detections), timing_ms)
        self._records.extend(batch_records)

    def _restore_mask(
        self, mask: np.ndarray, context: Mapping[str, Any]
    ) -> np.ndarray:
        normalized = self._normalize_context(context)
        value = np.asarray(mask)
        input_height = int(normalized["input_height"])
        input_width = int(normalized["input_width"])
        if value.ndim != 2 or value.shape != (input_height, input_width):
            raise ValueError(
                "segmentation mask must match input geometry "
                f"{(input_height, input_width)}, got {value.shape}"
            )
        if not np.issubdtype(value.dtype, np.number) or not np.isfinite(
            value
        ).all():
            raise ValueError("segmentation mask must contain finite values")

        original_height = int(normalized["original_height"])
        original_width = int(normalized["original_width"])
        scale = float(normalized["scale"])
        resized_height = round(original_height * scale)
        resized_width = round(original_width * scale)
        top = int(round(float(normalized["pad_y"])))
        left = int(round(float(normalized["pad_x"])))
        bottom = min(input_height, top + resized_height)
        right = min(input_width, left + resized_width)
        top = max(0, top)
        left = max(0, left)
        if bottom <= top or right <= left:
            raise ValueError("letterbox content area is empty")
        content = (value[top:bottom, left:right] > 0).astype(np.uint8)
        restored = cv2.resize(
            content,
            (original_width, original_height),
            interpolation=cv2.INTER_NEAREST,
        )
        return np.ascontiguousarray(restored, dtype=np.uint8)

    def compute(self) -> dict[str, int | float]:
        stats = self._run_coco_eval(self._records, self._image_ids)
        metrics: dict[str, int | float] = {
            name: max(0.0, float(stats[index]))
            for name, index in self.METRIC_INDEX.items()
        }
        metrics.update(self._latency_metrics())
        return metrics

    def evaluate(self, result: InferenceResult) -> dict[str, int | float]:
        self._reset()
        if result.timing_records:
            timing = float(
                np.mean(
                    [
                        self._normalize_timing(item)
                        for item in result.timing_records
                    ]
                )
            )
        else:
            timing = 0.0
        self.add_batch(result.outputs, result.labels, timing)
        return self.compute()

    def is_applicable(
        self,
        device_spec: dict[str, Any],
        model_spec: Model_Spec,
    ) -> bool:
        del device_spec
        return model_spec.task is Task.INSTANCE_SEGMENTATION

    def get_metric_names(self) -> list[str]:
        return [
            *self.METRIC_INDEX,
            "Total Samples",
            "Average Detections",
            "Average Latency (ms)",
            "P99 Latency (ms)",
            "FPS",
        ]
