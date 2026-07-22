from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from core.model_spec import Model_Spec, Task
from dataloader.mobilint_vision_profiles import (
    MOBILINT_RESNET50_IMAGENET1K_V2,
    MOBILINT_YOLOV5M_DEFAULT,
    ResNetCenterCropRecipe,
    YoloV5LetterboxRecipe,
    YoloV5RawHeadRecipe,
    apply_mobilint_vision_profile,
    resolve_mobilint_vision_profile,
)


AVAILABLE_PROFILE_IDS = (
    "mobilint-resnet50-imagenet1k-v2",
    "mobilint-yolov5m-default",
)


def _resolve(
    artifact: Path,
    *,
    model_name: str = "resnet50",
    task: Task = Task.IMAGE_CLASSIFICATION,
    requested_profile: str = "auto",
    requested_mode: str = "auto",
    requested_layout: str = "NCHW",
    layout_was_default: bool = True,
):
    return resolve_mobilint_vision_profile(
        model_name=model_name,
        task=task,
        artifact_path=artifact,
        requested_profile=requested_profile,
        requested_mode=requested_mode,
        requested_layout=requested_layout,
        layout_was_default=layout_was_default,
    )


@pytest.mark.parametrize(
    "task,model,basename,expected",
    [
        (
            Task.IMAGE_CLASSIFICATION,
            "resnet50",
            "resnet50_IMAGENET1K_V2.mxq",
            MOBILINT_RESNET50_IMAGENET1K_V2,
        ),
        (
            Task.OBJECT_DETECTION,
            "YOLOv5m",
            "yolov5m.mxq",
            MOBILINT_YOLOV5M_DEFAULT,
        ),
    ],
)
def test_auto_resolves_official_artifacts(task, model, basename, expected, tmp_path):
    artifact = tmp_path / basename
    artifact.touch()
    actual = resolve_mobilint_vision_profile(
        model_name=model,
        task=task,
        artifact_path=artifact,
        requested_profile="auto",
        requested_mode="auto",
        requested_layout="NCHW",
        layout_was_default=True,
    )
    assert actual is expected


def test_explicit_profile_allows_renamed_artifact_but_not_wrong_task(tmp_path):
    renamed = tmp_path / "renamed.mxq"
    renamed.touch()
    actual = resolve_mobilint_vision_profile(
        model_name="yolov5m",
        task=Task.OBJECT_DETECTION,
        artifact_path=renamed,
        requested_profile="mobilint-yolov5m-default",
        requested_mode="raw",
        requested_layout="NHWC",
        layout_was_default=False,
    )
    assert actual is MOBILINT_YOLOV5M_DEFAULT
    with pytest.raises(ValueError, match="task") as exc_info:
        resolve_mobilint_vision_profile(
            model_name="yolov5m",
            task=Task.IMAGE_CLASSIFICATION,
            artifact_path=renamed,
            requested_profile="mobilint-yolov5m-default",
            requested_mode="raw",
            requested_layout="NHWC",
            layout_was_default=False,
        )
    assert all(profile_id in str(exc_info.value) for profile_id in AVAILABLE_PROFILE_IDS)


def test_auto_rejects_unknown_artifact_basename_and_lists_profiles(tmp_path):
    artifact = tmp_path / "renamed.mxq"
    artifact.touch()
    with pytest.raises(ValueError, match="artifact") as exc_info:
        _resolve(artifact)
    assert all(profile_id in str(exc_info.value) for profile_id in AVAILABLE_PROFILE_IDS)


def test_auto_rejects_unknown_normalized_model_and_lists_profiles(tmp_path):
    artifact = tmp_path / "resnet50_IMAGENET1K_V2.mxq"
    artifact.touch()
    with pytest.raises(ValueError, match="model") as exc_info:
        _resolve(artifact, model_name="resnet51")
    assert all(profile_id in str(exc_info.value) for profile_id in AVAILABLE_PROFILE_IDS)


def test_explicit_profile_still_requires_matching_normalized_model(tmp_path):
    artifact = tmp_path / "renamed.mxq"
    artifact.touch()
    with pytest.raises(ValueError, match="model") as exc_info:
        _resolve(
            artifact,
            model_name="resnet51",
            requested_profile="mobilint-resnet50-imagenet1k-v2",
        )
    assert all(profile_id in str(exc_info.value) for profile_id in AVAILABLE_PROFILE_IDS)


def test_normalized_mode_conflicts_with_raw_profile(tmp_path):
    artifact = tmp_path / "resnet50_IMAGENET1K_V2.mxq"
    artifact.touch()
    with pytest.raises(ValueError, match="mode") as exc_info:
        _resolve(artifact, requested_mode="normalized")
    assert all(profile_id in str(exc_info.value) for profile_id in AVAILABLE_PROFILE_IDS)


def test_explicit_nchw_conflicts_with_nhwc_profile(tmp_path):
    artifact = tmp_path / "resnet50_IMAGENET1K_V2.mxq"
    artifact.touch()
    with pytest.raises(ValueError, match="layout") as exc_info:
        _resolve(
            artifact,
            requested_layout="NCHW",
            layout_was_default=False,
        )
    assert all(profile_id in str(exc_info.value) for profile_id in AVAILABLE_PROFILE_IDS)


def test_unknown_explicit_profile_lists_available_profile_ids(tmp_path):
    artifact = tmp_path / "renamed.mxq"
    artifact.touch()
    with pytest.raises(ValueError, match="unknown-profile") as exc_info:
        _resolve(artifact, requested_profile="unknown-profile")
    assert all(profile_id in str(exc_info.value) for profile_id in AVAILABLE_PROFILE_IDS)


def test_profiles_have_exact_immutable_recipes_and_contracts():
    assert MOBILINT_RESNET50_IMAGENET1K_V2.input_recipe == ResNetCenterCropRecipe()
    assert MOBILINT_RESNET50_IMAGENET1K_V2.runtime_contract() == {
        "vision_profile_id": "mobilint-resnet50-imagenet1k-v2",
        "expected_input_dtype": "uint8",
        "expected_input_layout": "NHWC",
        "expected_unbatched_input_shape": [224, 224, 3],
        "max_input_batch_size": 1,
    }

    assert MOBILINT_YOLOV5M_DEFAULT.input_recipe == YoloV5LetterboxRecipe()
    assert MOBILINT_YOLOV5M_DEFAULT.output_recipe == YoloV5RawHeadRecipe(
        class_count=80,
        anchors_by_stride=(
            (8, ((10, 13), (16, 30), (33, 23))),
            (16, ((30, 61), (62, 45), (59, 119))),
            (32, ((116, 90), (156, 198), (373, 326))),
        ),
    )
    assert MOBILINT_YOLOV5M_DEFAULT.decoder_defaults == (
        ("confidence_threshold", 0.001),
        ("iou_threshold", 0.65),
        ("max_detections", 300),
        ("max_nms_candidates", 30000),
        ("max_class_offset", 7680),
    )
    assert MOBILINT_YOLOV5M_DEFAULT.runtime_contract() == {
        "vision_profile_id": "mobilint-yolov5m-default",
        "expected_input_dtype": "uint8",
        "expected_input_layout": "NHWC",
        "expected_unbatched_input_shape": [640, 640, 3],
        "max_input_batch_size": 1,
        "expected_unbatched_output_shapes": [
            [20, 20, 255],
            [40, 40, 255],
            [80, 80, 255],
        ],
    }
    with pytest.raises(FrozenInstanceError):
        MOBILINT_YOLOV5M_DEFAULT.profile_id = "changed"


def test_apply_profile_replaces_input_without_mutating_original_spec():
    spec = Model_Spec(
        name="resnet50",
        task=Task.IMAGE_CLASSIFICATION,
        input_shapes={"image": (1, 3, 224, 224), "aux": (1, 2)},
        input_dtype={"image": "float32", "aux": "float32"},
        output_shapes={"logits": (1, 1000)},
        model_paths={"mobilint": "model.mxq"},
    )

    updated = apply_mobilint_vision_profile(
        spec,
        MOBILINT_RESNET50_IMAGENET1K_V2,
    )

    assert updated is not spec
    assert updated.input_shapes == {"image": (1, 224, 224, 3), "aux": (1, 2)}
    assert updated.input_dtype == {"image": "uint8", "aux": "float32"}
    assert updated.output_shapes == {"logits": (1, 1000)}
    assert spec.input_shapes == {"image": (1, 3, 224, 224), "aux": (1, 2)}
    assert spec.input_dtype == {"image": "float32", "aux": "float32"}
    assert spec.output_shapes == {"logits": (1, 1000)}


def test_apply_yolo_profile_replaces_outputs_in_runtime_metadata_order():
    spec = Model_Spec(
        name="yolov5m",
        task=Task.OBJECT_DETECTION,
        input_shapes={"images": (1, 3, 640, 640)},
        input_dtype={"images": "float32"},
        output_shapes={"old_output": (1, 25200, 85)},
    )

    updated = apply_mobilint_vision_profile(spec, MOBILINT_YOLOV5M_DEFAULT)

    assert list(updated.output_shapes.items()) == [
        ("mobilint_yolov5_stride32", (1, 20, 20, 255)),
        ("mobilint_yolov5_stride16", (1, 40, 40, 255)),
        ("mobilint_yolov5_stride8", (1, 80, 80, 255)),
    ]
    assert spec.output_shapes == {"old_output": (1, 25200, 85)}


def test_apply_profile_rejects_task_mismatch():
    spec = Model_Spec(
        name="resnet50",
        task=Task.OBJECT_DETECTION,
        input_shapes={"image": (1, 3, 224, 224)},
        input_dtype={"image": "float32"},
        output_shapes={"output": (1, 1000)},
    )

    with pytest.raises(ValueError, match="task mismatch"):
        apply_mobilint_vision_profile(spec, MOBILINT_RESNET50_IMAGENET1K_V2)
