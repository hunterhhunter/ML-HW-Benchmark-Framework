"""Hailo-specific image classification dataloader.

Hailo HEFs commonly expose quantized image inputs. Keep the generic ONNX path
normalized, but default Hailo image inputs to resized/cropped raw RGB pixels so
the runtime can send UINT8 tensors to HailoRT.
"""

from dataclasses import dataclass, field
from typing import Any, Dict

from .image_classification_loader import ImageClassificationLoader
from .preprocess_strategies import (
    MLPerfResNet50Preprocess,
    MLPerfResNet50RawPreprocess,
    PreprocessStrategy,
)
from core.model_spec import Model_Spec


@dataclass
class HailoImageInputConfig:
    preprocess_mode: str
    input_layout: str
    input_format_type: str
    runtime_options: dict[str, Any] = field(default_factory=dict)


def resolve_hailo_image_input_config(
    *,
    requested_mode: str = "auto",
    layout: str = "NHWC",
) -> HailoImageInputConfig:
    normalized_mode = str(requested_mode or "auto").lower()
    if normalized_mode == "auto":
        normalized_mode = "raw"
    if normalized_mode not in ("raw", "normalized"):
        raise ValueError(f"Unsupported Hailo image preprocess mode: {requested_mode}")

    input_format_type = "uint8" if normalized_mode == "raw" else "float32"
    input_layout = str(layout or "NHWC").upper()
    return HailoImageInputConfig(
        preprocess_mode=normalized_mode,
        input_layout=input_layout,
        input_format_type=input_format_type,
        runtime_options={
            "input_format_type": input_format_type,
            "input_layout": input_layout,
        },
    )


class HailoImageClassificationLoader(ImageClassificationLoader):
    """Image classification loader that follows the HailoRT image input contract."""

    def __init__(self, model_spec: Model_Spec, **kwargs):
        kwargs = dict(kwargs)
        self.hailo_input_config = resolve_hailo_image_input_config(
            requested_mode=kwargs.get("image_preprocess_mode", "auto"),
            layout=kwargs.get("layout", "NHWC"),
        )
        kwargs["preprocess_strategy"] = self._create_preprocess_strategy()
        super().__init__(model_spec, **kwargs)

    def _create_preprocess_strategy(self) -> PreprocessStrategy:
        if self.hailo_input_config.preprocess_mode == "raw":
            print(
                "[DataLoader] Hailo image preprocess: raw "
                f"(resize/crop only; runtime input={self.hailo_input_config.input_layout}/"
                f"{self.hailo_input_config.input_format_type})"
            )
            return MLPerfResNet50RawPreprocess()

        print(
            "[DataLoader] Hailo image preprocess: normalized "
            f"(ImageNet mean/std; runtime input={self.hailo_input_config.input_layout}/"
            f"{self.hailo_input_config.input_format_type})"
        )
        return MLPerfResNet50Preprocess()

    def get_metadata(self) -> Dict[str, Any]:
        metadata = super().get_metadata()
        metadata["hailo_input"] = {
            "preprocess_mode": self.hailo_input_config.preprocess_mode,
            "input_layout": self.hailo_input_config.input_layout,
            "input_format_type": self.hailo_input_config.input_format_type,
        }
        metadata["runtime_options"] = dict(self.hailo_input_config.runtime_options)
        return metadata
