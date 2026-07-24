"""Lightweight shared contracts for supported Mobilint vision artifacts."""

from dataclasses import dataclass

from .model_spec import Task


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
