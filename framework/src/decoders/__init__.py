"""Output decoders convert runtime tensors into evaluator-ready payloads."""

from core.model_spec import Model_Spec, Task
from core.mobilint_vision_contracts import YoloV5RawHeadRecipe

from .mobilint_yolov5 import MobilintYoloV5HeadDecoder
from .object_detection import (
    DETECTIONS_KEY,
    DetectionDecoder,
    HailoYoloNMSDecoder,
    RawYoloDetectionDecoder,
    nms_pure_numpy,
)


def create_decoder(model_spec: Model_Spec, **kwargs):
    """Return a task/backend specific decoder, or None when no decoder is needed."""
    if model_spec.task == Task.OBJECT_DETECTION:
        return create_object_detection_decoder(model_spec, **kwargs)
    return None


def create_object_detection_decoder(model_spec: Model_Spec, **kwargs) -> DetectionDecoder:
    backend = str(kwargs.get("backend", "")).lower()
    runtime_options = kwargs.get("runtime_options") or {}

    if backend == "mobilint":
        profile = kwargs.get("mobilint_vision_profile")
        if profile is None:
            raise ValueError(
                "Mobilint object detection requires mobilint_vision_profile."
            )
        recipe = getattr(profile, "output_recipe", None)
        if not isinstance(recipe, YoloV5RawHeadRecipe):
            raise ValueError(
                "Mobilint object detection requires a YoloV5RawHeadRecipe."
            )

        options = dict(profile.decoder_defaults)
        aliases = {
            "conf_threshold": "confidence_threshold",
            "iou_threshold": "iou_threshold",
            "max_nms": "max_nms_candidates",
            "max_det": "max_detections",
            "max_class_offset": "max_class_offset",
        }
        for constructor_name, profile_name in aliases.items():
            if profile_name in runtime_options:
                options[profile_name] = runtime_options[profile_name]
            if constructor_name in runtime_options:
                options[profile_name] = runtime_options[constructor_name]
            if profile_name in kwargs:
                options[profile_name] = kwargs[profile_name]
            if constructor_name in kwargs:
                options[profile_name] = kwargs[constructor_name]

        return MobilintYoloV5HeadDecoder(
            profile,
            conf_threshold=options["confidence_threshold"],
            iou_threshold=options["iou_threshold"],
            max_nms=options["max_nms_candidates"],
            max_det=options["max_detections"],
            max_class_offset=options["max_class_offset"],
        )

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
            box_order=runtime_options.get("hailo_nms_box_order", "yxyx"),
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
    "MobilintYoloV5HeadDecoder",
    "RawYoloDetectionDecoder",
    "create_decoder",
    "create_object_detection_decoder",
    "nms_pure_numpy",
]
