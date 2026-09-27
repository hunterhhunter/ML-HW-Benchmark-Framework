"""Output decoders convert runtime tensors into evaluator-ready payloads."""

from __future__ import annotations

from importlib import import_module

from core.model_spec import Model_Spec, Task
from core.mobilint_vision_contracts import YoloV5RawHeadRecipe


_LAZY_EXPORTS = {
    "MobilintYoloV5HeadDecoder": ".mobilint_yolov5",
    "DETECTIONS_KEY": ".object_detection",
    "DetectionDecoder": ".object_detection",
    "HailoYoloNMSDecoder": ".object_detection",
    "RawYoloDetectionDecoder": ".object_detection",
    "nms_pure_numpy": ".object_detection",
    "YoloV8SegmentationDecoder": ".instance_segmentation",
    "YoloV8PoseDecoder": ".pose_estimation",
    "HailoYoloV8PoseRawHeadDecoder": ".hailo_yolov8_pose",
    "DeepXYoloV8PoseRawHeadDecoder": ".deepx_yolov8_pose",
}


def _load_export(name: str):
    if name in globals():
        return globals()[name]
    try:
        module_name = _LAZY_EXPORTS[name]
    except KeyError:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        ) from None
    module = import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __getattr__(name: str):
    """Keep vision-only OpenCV imports out of non-vision processes."""
    return _load_export(name)


def create_decoder(model_spec: Model_Spec, **kwargs):
    """Return a task/backend specific decoder, or None when no decoder is needed."""
    if model_spec.task == Task.OBJECT_DETECTION:
        return create_object_detection_decoder(model_spec, **kwargs)
    if model_spec.task in {
        Task.INSTANCE_SEGMENTATION,
        Task.POSE_ESTIMATION,
    }:
        backend = str(kwargs.get("backend", "")).lower()
        runtime_options = kwargs.get("runtime_options") or {}
        decoder_options = {
            "conf_threshold": kwargs.get(
                "conf_threshold", runtime_options.get("conf_threshold", 0.25)
            ),
            "iou_threshold": kwargs.get(
                "iou_threshold", runtime_options.get("iou_threshold", 0.45)
            ),
            "max_detections": kwargs.get(
                "max_detections", runtime_options.get("max_detections", 300)
            ),
        }
        if model_spec.task == Task.INSTANCE_SEGMENTATION:
            if backend == "deepx":
                return None
            return _load_export("YoloV8SegmentationDecoder")(
                **decoder_options
            )
        annotation_file = kwargs.get("annotation_file")
        if backend in {"hailort", "hailo", "hailo8"}:
            if not annotation_file:
                return None
            if (
                runtime_options.get("hailo_yolov8_pose_raw_heads") is not True
                or runtime_options.get("hailo_pose_output_abi")
                != "yolov8-pose-dfl-nhwc-v1"
            ):
                raise ValueError(
                    "Hailo pose accuracy requires a verified pose output ABI."
                )
            return _load_export("HailoYoloV8PoseRawHeadDecoder")(
                conf_threshold=kwargs.get(
                    "conf_threshold",
                    runtime_options.get("conf_threshold", 0.001),
                ),
                iou_threshold=kwargs.get(
                    "iou_threshold",
                    runtime_options.get("iou_threshold", 0.70),
                ),
                max_detections=kwargs.get(
                    "max_detections",
                    runtime_options.get("max_detections", 300),
                ),
                class_scores_are_probabilities=runtime_options.get(
                    "yolov8_pose_class_scores_are_probabilities", False
                ),
            )
        if backend == "deepx":
            if not annotation_file:
                return None
            raw_head_abi = runtime_options.get("deepx_raw_head_abi")
            if raw_head_abi in {
                "yolov8-pose-dfl-nchw-v1",
                "yolov8-pose-dfl-class-probability-nchw-v1",
            }:
                return _load_export("DeepXYoloV8PoseRawHeadDecoder")(
                    conf_threshold=kwargs.get(
                        "conf_threshold",
                        runtime_options.get("conf_threshold", 0.001),
                    ),
                    iou_threshold=kwargs.get(
                        "iou_threshold",
                        runtime_options.get("iou_threshold", 0.70),
                    ),
                    max_detections=kwargs.get(
                        "max_detections",
                        runtime_options.get("max_detections", 300),
                    ),
                    class_scores_are_probabilities=runtime_options.get(
                        "yolov8_pose_class_scores_are_probabilities", False
                    ),
                    raw_head_abi=raw_head_abi,
                )
            if runtime_options.get("deepx_packed_pose_abi") == (
                "yolov8-pose-packed-b56n-v1"
            ):
                return _load_export("YoloV8PoseDecoder")(
                    conf_threshold=kwargs.get(
                        "conf_threshold",
                        runtime_options.get("conf_threshold", 0.001),
                    ),
                    iou_threshold=kwargs.get(
                        "iou_threshold",
                        runtime_options.get("iou_threshold", 0.70),
                    ),
                    max_detections=kwargs.get(
                        "max_detections",
                        runtime_options.get("max_detections", 300),
                    ),
                )
            raise ValueError(
                "DeepX pose accuracy requires a verified pose output ABI."
            )
        return _load_export("YoloV8PoseDecoder")(**decoder_options)
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

        return _load_export("MobilintYoloV5HeadDecoder")(
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
        return _load_export("HailoYoloNMSDecoder")(
            conf_threshold=conf_threshold,
            image_size=runtime_options.get("hailo_nms_image_size", image_size),
            box_order=runtime_options.get("hailo_nms_box_order", "yxyx"),
            debug=debug,
        )

    return _load_export("RawYoloDetectionDecoder")(
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
    )


__all__ = [
    "DETECTIONS_KEY",
    "DetectionDecoder",
    "HailoYoloNMSDecoder",
    "MobilintYoloV5HeadDecoder",
    "RawYoloDetectionDecoder",
    "YoloV8SegmentationDecoder",
    "YoloV8PoseDecoder",
    "HailoYoloV8PoseRawHeadDecoder",
    "DeepXYoloV8PoseRawHeadDecoder",
    "create_decoder",
    "create_object_detection_decoder",
    "nms_pure_numpy",
]
