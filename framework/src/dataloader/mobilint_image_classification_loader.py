"""Task-specific classification loader for resolved Mobilint profiles."""

from typing import Any

from core.mobilint_vision_contracts import MobilintVisionArtifactProfile
from core.model_spec import Model_Spec, Task
from preprocessor.mobilint_vision import MobilintResNetCenterCropPreprocess

from .image_classification_loader import ImageClassificationLoader


class MobilintImageClassificationLoader(ImageClassificationLoader):
    def __init__(self, model_spec: Model_Spec, **kwargs):
        options = dict(kwargs)
        profile = options.pop("mobilint_vision_profile", None)
        if not isinstance(profile, MobilintVisionArtifactProfile):
            raise ValueError(
                "Mobilint classification loader requires a resolved vision profile."
            )
        if profile.task is not Task.IMAGE_CLASSIFICATION:
            raise ValueError(
                "Mobilint classification loader received a non-classification profile."
            )
        self.mobilint_vision_profile = profile
        options["layout"] = profile.input_layout
        options["preprocess_strategy"] = MobilintResNetCenterCropPreprocess(profile)
        super().__init__(model_spec, **options)

    def get_metadata(self) -> dict[str, Any]:
        metadata = super().get_metadata()
        metadata["mobilint_vision_profile"] = self.mobilint_vision_profile.profile_id
        metadata["runtime_options"] = self.mobilint_vision_profile.runtime_contract()
        return metadata
