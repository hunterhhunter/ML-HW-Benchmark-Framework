from dataclasses import dataclass, replace
from pathlib import Path

from core.model_spec import Model_Spec, Task


@dataclass(frozen=True)
class ResNetCenterCropRecipe:
    resize_short_side: int = 232
    crop_hw: tuple[int, int] = (224, 224)
    interpolation: str = "pil_bilinear"
    resize_rounding: str = "integer_truncation"
    crop_rounding: str = "python_round"
    version: str = "1"


@dataclass(frozen=True)
class YoloV5LetterboxRecipe:
    input_hw: tuple[int, int] = (640, 640)
    interpolation: str = "opencv_linear"
    resize_rounding: str = "python_round"
    padding_rounding: str = "ultralytics_minus_plus_0_1"
    pad_color: tuple[int, int, int] = (114, 114, 114)
    version: str = "1"


@dataclass(frozen=True)
class YoloV5RawHeadRecipe:
    class_count: int
    anchors_by_stride: tuple[tuple[int, tuple[tuple[int, int], ...]], ...]
    expected_heads: int = 3
    version: str = "1"


@dataclass(frozen=True)
class MobilintVisionArtifactProfile:
    profile_id: str
    model_name: str
    task: Task
    artifact_basenames: tuple[str, ...]
    preprocess_mode: str
    color_order: str
    input_layout: str
    input_dtype: str
    unbatched_input_shape: tuple[int, ...]
    max_batch_size: int
    input_recipe: ResNetCenterCropRecipe | YoloV5LetterboxRecipe
    expected_output_shapes: tuple[tuple[int, ...], ...] = ()
    output_recipe: YoloV5RawHeadRecipe | None = None
    decoder_defaults: tuple[tuple[str, float | int], ...] = ()

    def runtime_contract(self) -> dict[str, object]:
        contract: dict[str, object] = {
            "vision_profile_id": self.profile_id,
            "expected_input_dtype": self.input_dtype,
            "expected_input_layout": self.input_layout,
            "expected_unbatched_input_shape": list(self.unbatched_input_shape),
            "max_input_batch_size": self.max_batch_size,
        }
        if self.expected_output_shapes:
            contract["expected_unbatched_output_shapes"] = [
                list(shape) for shape in self.expected_output_shapes
            ]
        return contract


MOBILINT_RESNET50_IMAGENET1K_V2 = MobilintVisionArtifactProfile(
    profile_id="mobilint-resnet50-imagenet1k-v2",
    model_name="resnet50",
    task=Task.IMAGE_CLASSIFICATION,
    artifact_basenames=("resnet50_IMAGENET1K_V2.mxq",),
    preprocess_mode="raw",
    color_order="RGB",
    input_layout="NHWC",
    input_dtype="uint8",
    unbatched_input_shape=(224, 224, 3),
    max_batch_size=1,
    input_recipe=ResNetCenterCropRecipe(),
)

MOBILINT_YOLOV5M_DEFAULT = MobilintVisionArtifactProfile(
    profile_id="mobilint-yolov5m-default",
    model_name="yolov5m",
    task=Task.OBJECT_DETECTION,
    artifact_basenames=("yolov5m.mxq",),
    preprocess_mode="raw",
    color_order="RGB",
    input_layout="NHWC",
    input_dtype="uint8",
    unbatched_input_shape=(640, 640, 3),
    max_batch_size=1,
    input_recipe=YoloV5LetterboxRecipe(),
    expected_output_shapes=((20, 20, 255), (40, 40, 255), (80, 80, 255)),
    output_recipe=YoloV5RawHeadRecipe(
        class_count=80,
        anchors_by_stride=(
            (8, ((10, 13), (16, 30), (33, 23))),
            (16, ((30, 61), (62, 45), (59, 119))),
            (32, ((116, 90), (156, 198), (373, 326))),
        ),
    ),
    decoder_defaults=(
        ("confidence_threshold", 0.001),
        ("iou_threshold", 0.65),
        ("max_detections", 300),
        ("max_nms_candidates", 30000),
        ("max_class_offset", 7680),
    ),
)


_PROFILES = (
    MOBILINT_RESNET50_IMAGENET1K_V2,
    MOBILINT_YOLOV5M_DEFAULT,
)


def _normalize_model_name(value: str) -> str:
    return "".join(ch for ch in value.casefold() if ch.isalnum())


def _available_profiles() -> str:
    return ", ".join(profile.profile_id for profile in _PROFILES)


def _resolution_error(message: str) -> ValueError:
    return ValueError(f"{message} Available profiles: {_available_profiles()}.")


def resolve_mobilint_vision_profile(
    *,
    model_name: str,
    task: Task,
    artifact_path: Path,
    requested_profile: str,
    requested_mode: str,
    requested_layout: str,
    layout_was_default: bool,
) -> MobilintVisionArtifactProfile:
    normalized_model_name = _normalize_model_name(model_name)

    if requested_profile == "auto":
        artifact_basename = Path(artifact_path).name
        profile = next(
            (
                candidate
                for candidate in _PROFILES
                if _normalize_model_name(candidate.model_name) == normalized_model_name
                and candidate.task is task
                and artifact_basename in candidate.artifact_basenames
            ),
            None,
        )
        if profile is None:
            raise _resolution_error(
                "No Mobilint vision profile matches "
                f"model {model_name!r}, task {task.name!r}, and artifact "
                f"{artifact_basename!r}."
            )
    else:
        profile = next(
            (
                candidate
                for candidate in _PROFILES
                if candidate.profile_id == requested_profile
            ),
            None,
        )
        if profile is None:
            raise _resolution_error(
                f"Unknown Mobilint vision profile {requested_profile!r}."
            )
        if _normalize_model_name(profile.model_name) != normalized_model_name:
            raise _resolution_error(
                f"Profile {profile.profile_id!r} model mismatch for {model_name!r}."
            )
        if profile.task is not task:
            raise _resolution_error(
                f"Profile {profile.profile_id!r} task mismatch for {task.name!r}."
            )

    if (
        requested_mode.casefold() != "auto"
        and requested_mode.casefold() != profile.preprocess_mode.casefold()
    ):
        raise _resolution_error(
            f"Profile {profile.profile_id!r} requires preprocess mode "
            f"{profile.preprocess_mode!r}, not {requested_mode!r}."
        )

    if not layout_was_default and requested_layout.upper() != profile.input_layout:
        raise _resolution_error(
            f"Profile {profile.profile_id!r} requires input layout "
            f"{profile.input_layout!r}, not {requested_layout!r}."
        )

    return profile


def apply_mobilint_vision_profile(
    spec: Model_Spec,
    profile: MobilintVisionArtifactProfile,
) -> Model_Spec:
    if spec.task is not profile.task:
        raise ValueError(f"Profile {profile.profile_id!r} task mismatch.")

    input_name = next(iter(spec.input_shapes))
    input_shapes = dict(spec.input_shapes)
    input_dtype = dict(spec.input_dtype)
    input_shapes[input_name] = (1, *profile.unbatched_input_shape)
    input_dtype[input_name] = profile.input_dtype

    output_shapes = dict(spec.output_shapes)
    if profile.expected_output_shapes:
        output_shapes = {
            f"mobilint_yolov5_stride{640 // shape[0]}": (1, *shape)
            for shape in profile.expected_output_shapes
        }

    return replace(
        spec,
        input_shapes=input_shapes,
        input_dtype=input_dtype,
        output_shapes=output_shapes,
    )
