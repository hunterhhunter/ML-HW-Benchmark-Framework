"""COCO Pose loader for the Regulus raw-head artifact."""

from typing import Any

from core.model_spec import Model_Spec, Task
from core.regulus_vision_contracts import RegulusPoseArtifactProfile
from preprocessor.regulus_pose import RegulusPosePreprocessor
from .coco_pose_loader import CocoPoseLoader


class RegulusPoseEstimationLoader(CocoPoseLoader):
    def __init__(self, model_spec: Model_Spec, **kwargs: Any) -> None:
        options = dict(kwargs)
        profile = options.pop("regulus_pose_profile", None)
        if options.pop("target_id", None) != "mobilint-regulus":
            raise ValueError("Regulus Pose loader requires mobilint-regulus target")
        if not isinstance(profile, RegulusPoseArtifactProfile) or profile.task is not Task.POSE_ESTIMATION:
            raise ValueError("Regulus Pose loader requires a Pose profile")
        self.regulus_pose_profile = profile
        options.update(preprocessor=RegulusPosePreprocessor(profile),
                       target_hw=profile.input_recipe.input_hw, layout="NHWC",
                       image_preprocess_mode="raw", image_resize_mode="letterbox")
        super().__init__(model_spec, **options)

    def get_metadata(self) -> dict[str, Any]:
        metadata = super().get_metadata()
        metadata["image_preprocess_mode"] = "raw"
        metadata["regulus_pose_profile"] = self.regulus_pose_profile.profile_id
        metadata["runtime_options"] = self.regulus_pose_profile.runtime_contract()
        return metadata
