"""Explicit Regulus Pose profile; generic MXQ basenames never select it."""

from dataclasses import replace
from pathlib import Path

from core.model_spec import Model_Spec, Task
from core.regulus_vision_contracts import RegulusPoseArtifactProfile, RegulusPoseRawHeadRecipe


REGULUS_VG_E2E_YOLOV8S_POSE_RAW_V1 = RegulusPoseArtifactProfile(
    profile_id="regulus-vg-e2e-yolov8s-pose-raw-v1",
    model_name="yolov8s-pose",
    task=Task.POSE_ESTIMATION,
    input_recipe=RegulusPoseRawHeadRecipe(),
)


def resolve_regulus_pose_profile(
    *, target_id: str, model_name: str, task: Task, artifact_path: Path,
    requested_profile: str, requested_mode: str, requested_layout: str,
    layout_was_default: bool,
) -> RegulusPoseArtifactProfile:
    profile = REGULUS_VG_E2E_YOLOV8S_POSE_RAW_V1
    if target_id != "mobilint-regulus":
        raise ValueError("Regulus Pose profile requires mobilint-regulus target")
    if requested_profile != profile.profile_id:
        raise ValueError(f"Regulus Pose requires explicit profile {profile.profile_id}")
    if "".join(filter(str.isalnum, model_name.casefold())) != "yolov8spose":
        raise ValueError("Regulus Pose profile model mismatch")
    if task is not Task.POSE_ESTIMATION:
        raise ValueError("Regulus Pose profile task mismatch")
    if Path(artifact_path).suffix.casefold() != ".mxq":
        raise ValueError("Regulus Pose requires an MXQ artifact")
    if requested_mode.casefold() not in {"auto", "raw"}:
        raise ValueError("Regulus Pose requires raw preprocess mode")
    if not layout_was_default and requested_layout.upper() != "NHWC":
        raise ValueError("Regulus Pose requires NHWC input layout")
    return profile


def apply_regulus_pose_profile(
    spec: Model_Spec, profile: RegulusPoseArtifactProfile,
) -> Model_Spec:
    if spec.task is not Task.POSE_ESTIMATION or profile is not REGULUS_VG_E2E_YOLOV8S_POSE_RAW_V1:
        raise ValueError("Regulus Pose profile/task mismatch")
    input_name = next(iter(spec.input_shapes))
    shapes = dict(spec.input_shapes)
    dtypes = dict(spec.input_dtype)
    shapes[input_name] = (1, *profile.unbatched_input_shape)
    dtypes[input_name] = profile.input_dtype
    outputs = {
        f"regulus_pose_{role}_stride{stride}": (1, h, w, channels)
        for stride in profile.input_recipe.strides
        for role, channels in zip(("keypoints", "dfl", "class"), profile.input_recipe.channels)
        for h, w in [(profile.input_recipe.input_hw[0] // stride,
                      profile.input_recipe.input_hw[1] // stride)]
    }
    return replace(spec, input_shapes=shapes, input_dtype=dtypes, output_shapes=outputs)
