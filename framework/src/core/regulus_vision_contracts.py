"""Regulus-specific VG-E2E Pose artifact ABI."""

from dataclasses import dataclass

from .model_spec import Task


@dataclass(frozen=True)
class RegulusPoseRawHeadRecipe:
    input_hw: tuple[int, int] = (640, 640)
    strides: tuple[int, ...] = (8, 16, 32)
    channels: tuple[int, ...] = (51, 64, 1)
    dfl_bins: int = 16
    keypoint_count: int = 17
    class_scores_are_probabilities: bool = False
    version: str = "1"


@dataclass(frozen=True)
class RegulusPoseArtifactProfile:
    profile_id: str
    model_name: str
    task: Task
    input_recipe: RegulusPoseRawHeadRecipe
    preprocess_mode: str = "raw"
    color_order: str = "RGB"
    input_layout: str = "NHWC"
    input_dtype: str = "uint8"
    max_batch_size: int = 1

    @property
    def unbatched_input_shape(self) -> tuple[int, ...]:
        return (*self.input_recipe.input_hw, 3)

    @property
    def expected_output_shapes(self) -> tuple[tuple[int, ...], ...]:
        h, w = self.input_recipe.input_hw
        return tuple(
            (h // stride, w // stride, channels)
            for stride in self.input_recipe.strides
            for channels in self.input_recipe.channels
        )

    def runtime_contract(self) -> dict[str, object]:
        return {
            "vision_profile_id": self.profile_id,
            "expected_input_dtype": self.input_dtype,
            "expected_input_layout": self.input_layout,
            "expected_unbatched_input_shape": list(self.unbatched_input_shape),
            "max_input_batch_size": self.max_batch_size,
            "expected_unbatched_output_shapes": [
                list(shape) for shape in self.expected_output_shapes
            ],
        }
