"""COCO pose loader for the Hailo YOLOv8s-pose nine-head ABI."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any

from core.model_spec import Model_Spec
from preprocessor.hailo_yolov8_pose import HailoYoloV8PosePreprocessor

from .coco_pose_loader import CocoPoseLoader


_POSE_HEAD_CONTRACT = {
    (height, height, channels)
    for height in (80, 40, 20)
    for channels in (64, 1, 51)
}


def resolve_hailo_pose_output_abi(output_infos) -> dict[str, Any] | None:
    """Validate the nine-head pose ABI from HEF vstream metadata."""
    try:
        infos = list(output_infos)
    except TypeError:
        return None
    if len(infos) != 9:
        return None

    by_shape = {}
    for info in infos:
        try:
            shape = tuple(int(value) for value in info.shape)
        except (AttributeError, TypeError, ValueError):
            return None
        if shape not in _POSE_HEAD_CONTRACT or shape in by_shape:
            return None
        by_shape[shape] = info
    if set(by_shape) != _POSE_HEAD_CONTRACT:
        return None

    for height in (80, 40, 20):
        try:
            quant_info = by_shape[(height, height, 1)].quant_info
            lower = float(quant_info.limvals_min)
            upper = float(quant_info.limvals_max)
        except (AttributeError, TypeError, ValueError):
            return None
        if abs(lower) > 1e-6 or abs(upper - 1.0) > 1e-6:
            return None

    return {
        "id": "yolov8-pose-dfl-nhwc-v1",
        "class_scores_are_probabilities": True,
    }


def resolve_hailo_pose_raw_head_abi(
    model_id: str,
    artifact_path: str | Path | None,
) -> dict[str, Any] | None:
    """Inspect a HEF without acquiring a device and identify its pose ABI."""
    if str(model_id) != "yolov8s-pose" or not artifact_path:
        return None
    path = Path(artifact_path)
    if not path.is_file() or path.suffix.lower() != ".hef":
        return None
    try:
        hailo_platform = import_module("hailo_platform")
        hef = hailo_platform.HEF(str(path))
        output_infos = hef.get_output_vstream_infos()
    except Exception:
        return None
    return resolve_hailo_pose_output_abi(output_infos)


class HailoPoseEstimationLoader(CocoPoseLoader):
    """Preserve COCO identity while producing uint8 NHWC Hailo input."""

    def __init__(self, model_spec: Model_Spec, **kwargs: Any) -> None:
        options = dict(kwargs)
        self.hailo_pose_output_abi = resolve_hailo_pose_raw_head_abi(
            model_spec.name,
            options.get("artifact_path"),
        )
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
        runtime_options = {
            "input_format_type": "uint8",
            "input_layout": "NHWC",
            "output_format_type": "float32",
        }
        if self.hailo_pose_output_abi is not None:
            runtime_options.update(
                {
                    "hailo_yolov8_pose_raw_heads": True,
                    "hailo_pose_output_abi": self.hailo_pose_output_abi["id"],
                    "yolov8_pose_class_scores_are_probabilities": (
                        self.hailo_pose_output_abi[
                            "class_scores_are_probabilities"
                        ]
                    ),
                }
            )
        metadata["runtime_options"] = runtime_options
        return metadata
