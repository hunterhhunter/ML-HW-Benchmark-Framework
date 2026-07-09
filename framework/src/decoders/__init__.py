"""Output decoders convert runtime tensors into evaluator-ready payloads."""

from core.model_spec import Model_Spec, Task

from .object_detection import (
    DETECTIONS_KEY,
    DetectionDecoder,
    HailoYoloNMSDecoder,
    RawYoloDetectionDecoder,
)


def create_decoder(model_spec: Model_Spec, **kwargs):
    """Return a task/backend specific decoder, or None when no decoder is needed."""
    if model_spec.task == Task.OBJECT_DETECTION:
        return create_object_detection_decoder(model_spec, **kwargs)
    return None


def create_object_detection_decoder(model_spec: Model_Spec, **kwargs) -> DetectionDecoder:
    backend = str(kwargs.get("backend", "")).lower()
    runtime_options = kwargs.get("runtime_options") or {}
    conf_threshold = kwargs.get(
        "conf_threshold",
        runtime_options.get("hailo_nms_conf_threshold", runtime_options.get("conf_threshold", 0.25)),
    )
    iou_threshold = kwargs.get(
        "iou_threshold",
        runtime_options.get("iou_threshold", 0.45),
    )
    image_size = kwargs.get("image_size", 640)
    debug = bool(kwargs.get("debug", runtime_options.get("debug_tensors", False)))

    if backend in {"hailort", "hailo", "hailo8"}:
        return HailoYoloNMSDecoder(
            conf_threshold=conf_threshold,
            image_size=runtime_options.get("hailo_nms_image_size", image_size),
            box_order=runtime_options.get("hailo_nms_box_order", "xyxy"),
            debug=debug,
        )

    return RawYoloDetectionDecoder(
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
    )


__all__ = [
    "DETECTIONS_KEY",
    "DetectionDecoder",
    "HailoYoloNMSDecoder",
    "RawYoloDetectionDecoder",
    "create_decoder",
    "create_object_detection_decoder",
]
