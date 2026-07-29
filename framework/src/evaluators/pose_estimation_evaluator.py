"""Official COCO keypoint evaluator for YOLOv8 pose predictions."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from core.inference_result import InferenceResult
from core.model_spec import Model_Spec, Task
from decoders.yolo_vision import DETECTIONS_KEY, KEYPOINTS_KEY

from .coco_common import CocoEvaluatorBase


class PoseEstimationEvaluator(CocoEvaluatorBase):
    """Accumulate restored keypoint records and compute official COCO OKS AP."""

    METRIC_INDEX = {
        "OKS mAP": 0,
        "OKS AP50": 1,
        "OKS AP75": 2,
        "OKS AP Medium": 3,
        "OKS AP Large": 4,
    }

    def __init__(self, annotation_file: str, **options: Any) -> None:
        super().__init__(annotation_file, "keypoints", **options)
        person_category = self._coco_gt.cats.get(1)
        if not isinstance(person_category, dict):
            raise ValueError("COCO pose ground truth requires category id 1")
        keypoint_names = person_category.get("keypoints")
        if not isinstance(keypoint_names, list) or len(keypoint_names) != 17:
            raise ValueError(
                "COCO pose category id 1 requires 17 keypoint names"
            )

    def add_batch(
        self,
        outputs: Mapping[str, Any],
        labels: Any,
        timing_ms: Any,
    ) -> None:
        if DETECTIONS_KEY not in outputs or KEYPOINTS_KEY not in outputs:
            raise ValueError("pose outputs require detections and keypoints")
        batch = self._normalize_batch_labels(labels)
        detections = self._validate_local_indices(
            np.asarray(outputs[DETECTIONS_KEY]), len(batch)
        )
        keypoints = np.asarray(outputs[KEYPOINTS_KEY])
        if keypoints.ndim != 3 or keypoints.shape[1:] != (17, 3):
            raise ValueError(
                "pose keypoints must have shape (N, 17, 3), "
                f"got {keypoints.shape}"
            )
        if len(detections) != len(keypoints):
            raise ValueError("pose detection/keypoint row count mismatch")
        if not np.issubdtype(keypoints.dtype, np.number) or not np.isfinite(
            keypoints
        ).all():
            raise ValueError("pose keypoints must contain finite numeric values")

        batch_records = []
        for detection, keypoint_row in zip(detections, keypoints):
            if detection[1] != 0:
                raise ValueError("COCO pose detections require local person class 0")
            local_index = int(detection[0])
            item = batch[local_index]
            context = item["preprocess_context"]
            restored_keypoints = self._restore_keypoints(
                keypoint_row, context
            )
            restored_box = self._restore_boxes(
                detection[None, 3:7], context
            )[0]
            width = float(restored_box[2] - restored_box[0])
            height = float(restored_box[3] - restored_box[1])
            if width <= 0 or height <= 0:
                raise ValueError("pose detection box must have positive area")
            batch_records.append(
                {
                    "image_id": item["image_id"],
                    "category_id": 1,
                    "keypoints": restored_keypoints.astype(
                        float, copy=False
                    ).reshape(-1).tolist(),
                    "score": float(detection[2]),
                    "bbox": [
                        float(restored_box[0]),
                        float(restored_box[1]),
                        width,
                        height,
                    ],
                }
            )

        self._record_batch(batch, len(detections), timing_ms)
        self._records.extend(batch_records)

    def _restore_keypoints(
        self,
        keypoints: np.ndarray,
        context: Mapping[str, Any],
    ) -> np.ndarray:
        value = np.asarray(keypoints)
        if value.shape != (17, 3):
            raise ValueError(
                f"keypoints must have shape (17, 3), got {value.shape}"
            )
        if not np.issubdtype(value.dtype, np.number) or not np.isfinite(
            value
        ).all():
            raise ValueError("keypoints must contain finite numeric values")
        normalized = self._normalize_context(context)
        restored = value.astype(np.float32, copy=True)
        restored[:, 0] = np.clip(
            (
                restored[:, 0]
                - float(normalized["pad_x"])
            )
            / float(normalized["scale"]),
            0.0,
            float(normalized["original_width"]),
        )
        restored[:, 1] = np.clip(
            (
                restored[:, 1]
                - float(normalized["pad_y"])
            )
            / float(normalized["scale"]),
            0.0,
            float(normalized["original_height"]),
        )
        return restored

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
        return model_spec.task is Task.POSE_ESTIMATION

    def get_metric_names(self) -> list[str]:
        return [
            *self.METRIC_INDEX,
            "Total Samples",
            "Average Detections",
            "Average Latency (ms)",
            "P99 Latency (ms)",
            "FPS",
        ]
