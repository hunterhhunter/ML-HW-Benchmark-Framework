"""Task-specific object detection loader for resolved Mobilint profiles."""

from typing import Any

from core.mobilint_vision_contracts import MobilintVisionArtifactProfile
from core.model_spec import Model_Spec, Task
from preprocessor.mobilint_vision import MobilintYoloV5Preprocessor

from .object_detection_loader import ObjectDetectionLoader


class MobilintObjectDetectionLoader(ObjectDetectionLoader):
    def __init__(self, model_spec: Model_Spec, **kwargs):
        options = dict(kwargs)
        profile = options.pop("mobilint_vision_profile", None)
        if not isinstance(profile, MobilintVisionArtifactProfile):
            raise ValueError(
                "Mobilint detection loader requires a resolved vision profile."
            )
        if profile.task is not Task.OBJECT_DETECTION:
            raise ValueError(
                "Mobilint detection loader received a non-detection profile."
            )
        self.mobilint_vision_profile = profile
        options["preprocessor"] = MobilintYoloV5Preprocessor(profile)
        options["backend"] = "mobilint"
        options["layout"] = "NHWC"
        options["image_preprocess_mode"] = "raw"
        options["image_resize_mode"] = "letterbox"
        super().__init__(model_spec, **options)

    def get_metadata(self) -> dict[str, Any]:
        metadata = super().get_metadata()
        metadata["mobilint_vision_profile"] = self.mobilint_vision_profile.profile_id
        metadata["runtime_options"] = self.mobilint_vision_profile.runtime_contract()
        return metadata
