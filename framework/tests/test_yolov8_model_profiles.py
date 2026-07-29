from pathlib import Path

import onnx
from onnx import TensorProto, helper

from core.model_profiles import SUPPORTED_PROFILES, create_model_spec
from core.model_spec import Task
import models.prepare_yolov8_vision as vision_export


def _write_two_output_model(path: Path) -> None:
    graph = helper.make_graph(
        [
            helper.make_node("Identity", ["images"], ["predictions"]),
            helper.make_node("Identity", ["images"], ["prototypes"]),
        ],
        "two-output",
        [
            helper.make_tensor_value_info(
                "images", TensorProto.FLOAT, [1, 3, 640, 640]
            )
        ],
        [
            helper.make_tensor_value_info(
                "predictions", TensorProto.FLOAT, [1, 116, 8400]
            ),
            helper.make_tensor_value_info(
                "prototypes", TensorProto.FLOAT, [1, 32, 160, 160]
            ),
        ],
    )
    onnx.save(helper.make_model(graph), path)


def test_small_yolov8_profiles_drive_exact_tasks_and_asset_paths():
    seg = SUPPORTED_PROFILES["yolov8s-seg"]
    pose = SUPPORTED_PROFILES["yolov8s-pose"]

    assert seg["task"] is Task.INSTANCE_SEGMENTATION
    assert pose["task"] is Task.POSE_ESTIMATION
    assert seg["default_model_path"] == (
        "models/yolov8s-seg/yolov8s-seg.onnx"
    )
    assert pose["default_model_path"] == (
        "models/yolov8s-pose/yolov8s-pose.onnx"
    )
    assert vision_export.MODELS["yolov8s-seg"] == {
        "weights": "yolov8s-seg.pt",
        "output_dir": "yolov8s-seg",
        "onnx_name": "yolov8s-seg.onnx",
        "output_count": 2,
    }
    assert vision_export.MODELS["yolov8s-pose"] == {
        "weights": "yolov8s-pose.pt",
        "output_dir": "yolov8s-pose",
        "onnx_name": "yolov8s-pose.onnx",
        "output_count": 1,
    }


def test_segmentation_spec_binds_every_graph_output_name_by_ordinal(tmp_path):
    model_path = tmp_path / "seg.onnx"
    _write_two_output_model(model_path)

    spec = create_model_spec("yolov8s-seg", str(model_path))

    assert spec.input_shapes == {"images": (1, 3, 640, 640)}
    assert spec.output_shapes == {
        "predictions": (1, 116, 8400),
        "prototypes": (1, 32, 160, 160),
    }


def test_export_validation_rejects_corruption_and_wrong_output_count(tmp_path):
    invalid = tmp_path / "invalid.onnx"
    invalid.write_bytes(b"broken")
    model_path = tmp_path / "seg.onnx"
    _write_two_output_model(model_path)

    assert vision_export._valid_onnx_export(
        invalid, expected_output_count=2
    ) is False
    assert vision_export._valid_onnx_export(
        model_path, expected_output_count=2
    ) is True
    assert vision_export._valid_onnx_export(
        model_path, expected_output_count=1
    ) is False
