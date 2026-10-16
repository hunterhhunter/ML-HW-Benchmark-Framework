"""Regulus-specific decoder for the VG-E2E nine-head Pose MXQ."""

from collections.abc import Mapping
from typing import Any

import numpy as np

from core.regulus_vision_contracts import RegulusPoseArtifactProfile
from dataloader.regulus_vision_profiles import REGULUS_VG_E2E_YOLOV8S_POSE_RAW_V1
from .pose_estimation import YoloV8PoseDecoder


class RegulusYoloV8PoseRawHeadDecoder:
    def __init__(self, profile: RegulusPoseArtifactProfile, *,
                 conf_threshold: float = 0.25, iou_threshold: float = 0.70,
                 max_detections: int = 300, max_nms_candidates: int = 30000):
        if profile is not REGULUS_VG_E2E_YOLOV8S_POSE_RAW_V1:
            raise ValueError("Regulus Pose decoder requires its profile")
        if max_nms_candidates < 1:
            raise ValueError("max_nms_candidates must be positive")
        self.profile = profile
        self.max_nms_candidates = max_nms_candidates
        self._postprocess = YoloV8PoseDecoder(conf_threshold=conf_threshold,
            iou_threshold=iou_threshold, max_detections=max_detections)

    def decode(self, outputs: Mapping[str, Any]) -> dict[str, np.ndarray]:
        return self._postprocess.decode({"output0": self._decode_heads(outputs)})

    def result_metadata(self) -> dict[str, object]:
        return {
            "mobilint_vision_profile_id": self.profile.profile_id,
            "regulus_pose_confidence_threshold": self._postprocess.conf_threshold,
            "regulus_pose_iou_threshold": self._postprocess.iou_threshold,
            "regulus_pose_max_detections": self._postprocess.max_detections,
            "regulus_pose_raw_head_decode": "yolov8-pose-dfl-nhwc-v1",
            "regulus_pose_class_activation": "host_sigmoid",
        }

    def _decode_heads(self, outputs: Mapping[str, Any]) -> np.ndarray:
        recipe = self.profile.input_recipe
        expected = {(recipe.input_hw[0] // stride, recipe.input_hw[1] // stride): stride
                    for stride in recipe.strides}
        if len(outputs) != len(expected) * len(recipe.channels):
            raise ValueError("Regulus Pose requires exactly nine raw heads")
        grouped: dict[int, dict[int, np.ndarray]] = {}
        batch = None
        for name, output in outputs.items():
            value = np.asarray(output)
            if value.ndim == 3:
                value = value[None, ...]
            if (value.ndim != 4 or tuple(value.shape[1:3]) not in expected
                    or value.shape[3] not in recipe.channels):
                raise ValueError(f"Regulus Pose head {name!r} has unsupported NHWC shape {value.shape}")
            if not np.issubdtype(value.dtype, np.number) or not np.isfinite(value).all():
                raise ValueError(f"Regulus Pose head {name!r} must be finite numeric")
            if batch is None:
                batch = value.shape[0]
            elif batch != value.shape[0]:
                raise ValueError("Regulus Pose head batch sizes differ")
            stride = expected[tuple(value.shape[1:3])]
            level = grouped.setdefault(stride, {})
            if value.shape[3] in level:
                raise ValueError("Regulus Pose duplicate head role")
            level[value.shape[3]] = value.astype(np.float32, copy=False)
        if set(grouped) != set(recipe.strides) or any(set(v) != set(recipe.channels) for v in grouped.values()):
            raise ValueError("Regulus Pose heads do not satisfy the 9-head ABI")
        decoded = []
        for stride in recipe.strides:
            keypoints, dfl, classes = (grouped[stride][ch] for ch in recipe.channels)
            batch, height, width, _ = dfl.shape
            locations = height * width
            gy, gx = np.meshgrid(np.arange(height, dtype=np.float32),
                                 np.arange(width, dtype=np.float32), indexing="ij")
            grid = np.stack((gx, gy), axis=0).reshape(1, 2, locations)
            logits = dfl.reshape(batch, height, width, 4, recipe.dfl_bins)
            logits = logits.transpose(0, 3, 4, 1, 2).reshape(batch, 4, recipe.dfl_bins, locations)
            shifted = logits - logits.max(axis=2, keepdims=True)
            probabilities = np.exp(shifted)
            probabilities /= probabilities.sum(axis=2, keepdims=True)
            weights = np.arange(recipe.dfl_bins, dtype=np.float32).reshape(1, 1, -1, 1)
            distances = (probabilities * weights).sum(axis=2)
            anchors = grid + 0.5
            tl, br = anchors - distances[:, :2], anchors + distances[:, 2:]
            boxes = np.concatenate(((tl + br) / 2, br - tl), axis=1) * stride
            scores = classes.reshape(batch, locations, 1).transpose(0, 2, 1)
            if recipe.class_scores_are_probabilities:
                if np.min(scores) < 0 or np.max(scores) > 1:
                    raise ValueError("Regulus Pose class probabilities outside [0,1]")
            else:
                scores = _sigmoid(scores)
            pose = keypoints.reshape(batch, locations, recipe.keypoint_count, 3)
            pose = pose.transpose(0, 2, 3, 1).copy()
            pose[:, :, :2] = (pose[:, :, :2] * 2 + grid.reshape(1, 1, 2, locations)) * stride
            pose[:, :, 2:] = _sigmoid(pose[:, :, 2:])
            decoded.append(np.concatenate((boxes, scores, pose.reshape(batch, -1, locations)), axis=1))
        prediction = np.ascontiguousarray(np.concatenate(decoded, axis=2), dtype=np.float32)
        if prediction.shape[2] > self.max_nms_candidates:
            raise ValueError("Regulus Pose candidates exceed max_nms_candidates")
        return prediction


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -80, 80)))
