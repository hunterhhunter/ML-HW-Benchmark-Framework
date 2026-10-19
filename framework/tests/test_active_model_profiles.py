import pytest

from core.model_profiles import SUPPORTED_PROFILES, create_model_spec


REMOVED_PROFILES = {
    "yolov8m",
    "yolov8m-seg",
    "yolov8m-pose",
    "yolov8m_seg",
    "yolov8-seg-m",
    "yolov8m_pose",
    "yolov8-pose-m",
    "yolov8s-seg",
    "patchtst-fm-r1",
    "patchtst-etth1",
}


def test_removed_model_profiles_and_aliases_are_unavailable_to_cli():
    assert REMOVED_PROFILES.isdisjoint(SUPPORTED_PROFILES)
    for model_name in REMOVED_PROFILES:
        with pytest.raises(ValueError, match="지원되지 않는 모델 프로필"):
            create_model_spec(model_name, "unused.onnx", sniff_onnx=False)


def test_active_model_profiles_remain_available():
    assert {"resnet50", "yolov5m", "yolov8s-pose", "ttm-r2"} <= SUPPORTED_PROFILES.keys()
