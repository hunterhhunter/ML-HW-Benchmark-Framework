"""COCO pose loader for the Hailo YOLOv8s-pose nine-head ABI."""

from __future__ import annotations

from typing import Any

from core.model_spec import Model_Spec
from preprocessor.hailo_yolov8_pose import HailoYoloV8PosePreprocessor

from .coco_pose_loader import CocoPoseLoader


class HailoPoseEstimationLoader(CocoPoseLoader):
    """Preserve COCO identity while producing uint8 NHWC Hailo input."""

    def __init__(self, model_spec: Model_Spec, **kwargs: Any) -> None:
        options = dict(kwargs)
        options["preprocessor"] = HailoYoloV8PosePreprocessor()
        options["target_hw"] = (640, 640)
        options["layout"] = "NHWC"
        options["image_preprocess_mode"] = "normalized"
        options["image_resize_mode"] = "letterbox"
        super().__init__(model_spec, **options)

    def get_metadata(self) -> dict[str, Any]:
        metadata = super().get_metadata()
        metadata["hailo_pose_input"] = {
            "preprocess": "pil-bilinear-letterbox-pad114",
            "layout": "NHWC",
            "dtype": "uint8",
        }
        metadata["runtime_options"] = {
            "input_format_type": "uint8",
            "input_layout": "NHWC",
            "output_format_type": "float32",
            "hailo_yolov8_pose_raw_heads": True,
            # The vendor YOLOv8s-pose HEFs expose class heads quantized to
            # [0, 1], so the class sigmoid already belongs to the HEF.
            "yolov8_pose_class_scores_are_probabilities": True,
        }
        return metadata
