import json
from types import SimpleNamespace

import numpy as np
import pytest

from coco_test_utils import make_pose_spec, make_seg_spec, write_coco_fixture
from dataloader import create_dataloader
from dataloader.deepx_image_classification_loader import (
    read_dxnn_compile_config,
)
from dataloader.hailo_pose_estimation_loader import (
    resolve_hailo_pose_output_abi,
)
import dataloader.hailo_pose_estimation_loader as hailo_pose_loader
from decoders import create_decoder
from evaluators import create_evaluator


def _pose_loader(tmp_path, *, backend, artifact_path=None):
    fixture = write_coco_fixture(tmp_path)
    loader = create_dataloader(
        make_pose_spec(),
        backend=backend,
        artifact_path=artifact_path,
        dataset_path=str(fixture["images"]),
        image_dir=str(fixture["images"]),
        label_path=str(fixture["pose"]),
        layout="NCHW",
        image_preprocess_mode="normalized",
        image_resize_mode="letterbox",
    )
    return fixture, loader


def _nine_hailo_pose_heads(class_logit=-20.0):
    outputs = {}
    for height in (80, 40, 20):
        outputs[f"pose_{height}_keypoints"] = np.zeros(
            (1, height, height, 51), dtype=np.float32
        )
        outputs[f"pose_{height}_dfl"] = np.zeros(
            (1, height, height, 64), dtype=np.float32
        )
        outputs[f"pose_{height}_class"] = np.full(
            (1, height, height, 1), class_logit, dtype=np.float32
        )
    return outputs


_DEEPX_POSE_RAW_OUTPUTS = (
    "/model.22/cv2.0/cv2.0.2/Conv_output_0",
    "/model.22/cv2.1/cv2.1.2/Conv_output_0",
    "/model.22/cv2.2/cv2.2.2/Conv_output_0",
    "/model.22/cv3.0/cv3.0.2/Conv_output_0",
    "/model.22/cv3.1/cv3.1.2/Conv_output_0",
    "/model.22/cv3.2/cv3.2.2/Conv_output_0",
    "/model.22/cv4.0/cv4.0.2/Conv_output_0",
    "/model.22/cv4.1/cv4.1.2/Conv_output_0",
    "/model.22/cv4.2/cv4.2.2/Conv_output_0",
)

_DEEPX_POSE_CLASS_PROBABILITY_OUTPUTS = (
    "p3_bbox_raw",
    "p3_class_probability",
    "p3_keypoint_raw",
    "p4_bbox_raw",
    "p4_class_probability",
    "p4_keypoint_raw",
    "p5_bbox_raw",
    "p5_class_probability",
    "p5_keypoint_raw",
)

_DEEPX_POSE_PACKED_DFL_OUTPUTS = (
    "/model.22/dfl/conv/Conv_output_0",
    "/model.22/cv3.0/cv3.0.2/Conv_output_0",
    "/model.22/cv3.1/cv3.1.2/Conv_output_0",
    "/model.22/cv3.2/cv3.2.2/Conv_output_0",
    "/model.22/cv4.0/cv4.0.2/Conv_output_0",
    "/model.22/cv4.1/cv4.1.2/Conv_output_0",
    "/model.22/cv4.2/cv4.2.2/Conv_output_0",
)


def _write_pose_dxnn(path, graph_info):
    rmap_info = json.dumps(
        {
            "inputs": [
                {
                    "name": "images",
                    "dtype": "UINT8",
                    "shape": [1, 640, 640, 3],
                }
            ]
        },
        sort_keys=True,
    ).encode("utf-8")
    graph_payload = json.dumps(graph_info, sort_keys=True).encode("utf-8")
    header = {
        "size": 8192,
        "data": {
            "graph_info": {
                "type": "str",
                "offset": 0,
                "size": len(graph_payload),
            },
            "compiled_data": {
                "M1A_4K": {
                    "npu_0": {
                        "rmap_info": {
                            "type": "str",
                            "offset": len(graph_payload),
                            "size": len(rmap_info),
                        }
                    }
                }
            },
        },
    }
    encoded_header = json.dumps(header, sort_keys=True).encode("utf-8")
    assert len(encoded_header) < 8184
    path.write_bytes(
        b"DXNN"
        + (8).to_bytes(4, "little")
        + encoded_header
        + b"\0" * (8184 - len(encoded_header))
        + graph_payload
        + rmap_info
    )
    return path


def _deepx_nchw_pose_heads(class_logit=-20.0):
    outputs = {}
    for height in (80, 40, 20):
        outputs[f"pose_{height}_keypoints"] = np.zeros(
            (1, 51, height, height), dtype=np.float32
        )
        outputs[f"pose_{height}_dfl"] = np.zeros(
            (1, 64, height, height), dtype=np.float32
        )
        outputs[f"pose_{height}_class"] = np.full(
            (1, 1, height, height), class_logit, dtype=np.float32
        )
    return outputs


def test_pose_metadata_extension_preserves_dxnn_compile_config_reader(tmp_path):
    compile_config = {"default_loader": {"preprocessings": [{"div": 255.0}]}}
    payload = json.dumps(compile_config, sort_keys=True).encode("utf-8")
    header = {
        "size": 8192,
        "data": {
            "compile_config": {
                "type": "str",
                "offset": 0,
                "size": len(payload),
            }
        },
    }
    encoded_header = json.dumps(header, sort_keys=True).encode("utf-8")
    artifact = tmp_path / "compile-config.dxnn"
    artifact.write_bytes(
        b"DXNN"
        + (8).to_bytes(4, "little")
        + encoded_header
        + b"\0" * (8184 - len(encoded_header))
        + payload
    )

    assert read_dxnn_compile_config(artifact) == compile_config


def test_hailo_pose_factory_routes_loader_and_decoder(tmp_path, monkeypatch):
    artifact = tmp_path / "pose.hef"
    artifact.write_bytes(b"verified by fake HEF metadata")
    monkeypatch.setattr(
        hailo_pose_loader,
        "resolve_hailo_pose_raw_head_abi",
        lambda model_id, artifact_path: {
            "id": "yolov8-pose-dfl-nhwc-v1",
            "class_scores_are_probabilities": True,
        },
    )
    fixture, loader = _pose_loader(
        tmp_path / "dataset",
        backend="hailort",
        artifact_path=artifact,
    )

    sample = loader.load_single()
    runtime_options = loader.get_metadata()["runtime_options"]
    decoder = create_decoder(
        make_pose_spec(),
        backend="hailort",
        runtime_options=runtime_options,
        annotation_file=str(fixture["pose"]),
    )

    assert type(loader).__name__ == "HailoPoseEstimationLoader"
    assert sample["input"].shape == (640, 640, 3)
    assert sample["input"].dtype == np.uint8
    assert sample["label"] == {
        "image_id": 1,
        "file_name": "000000000001.jpg",
    }
    assert runtime_options == {
        "input_format_type": "uint8",
        "input_layout": "NHWC",
        "output_format_type": "float32",
        "hailo_yolov8_pose_raw_heads": True,
        "hailo_pose_output_abi": "yolov8-pose-dfl-nhwc-v1",
        "yolov8_pose_class_scores_are_probabilities": True,
    }
    assert type(decoder).__name__ == "HailoYoloV8PoseRawHeadDecoder"


def test_hailo_pose_output_abi_requires_nine_verified_heads_and_probability_range():
    infos = []
    for height in (80, 40, 20):
        for channels in (64, 1, 51):
            infos.append(
                SimpleNamespace(
                    shape=(height, height, channels),
                    quant_info=SimpleNamespace(
                        limvals_min=0.0 if channels == 1 else -8.0,
                        limvals_max=1.0 if channels == 1 else 8.0,
                    ),
                )
            )

    assert resolve_hailo_pose_output_abi(infos) == {
        "id": "yolov8-pose-dfl-nhwc-v1",
        "class_scores_are_probabilities": True,
    }

    infos[-2].quant_info.limvals_max = 8.0
    assert resolve_hailo_pose_output_abi(infos) is None


def test_hailo_unknown_pose_hef_does_not_claim_verified_abi(
    tmp_path,
    monkeypatch,
):
    artifact = tmp_path / "unknown.hef"
    artifact.write_bytes(b"unknown")
    monkeypatch.setattr(
        hailo_pose_loader,
        "resolve_hailo_pose_raw_head_abi",
        lambda model_id, artifact_path: None,
    )
    fixture, loader = _pose_loader(
        tmp_path / "dataset",
        backend="hailort",
        artifact_path=artifact,
    )
    runtime_options = loader.get_metadata()["runtime_options"]

    assert "hailo_yolov8_pose_raw_heads" not in runtime_options
    with pytest.raises(ValueError, match="verified pose output ABI"):
        create_decoder(
            make_pose_spec(),
            backend="hailort",
            runtime_options=runtime_options,
            annotation_file=str(fixture["pose"]),
        )


def test_hailo_pose_factory_decoder_accepts_nine_raw_heads():
    decoder = create_decoder(
        make_pose_spec(),
        backend="hailort",
        runtime_options={
            "hailo_yolov8_pose_raw_heads": True,
            "hailo_pose_output_abi": "yolov8-pose-dfl-nhwc-v1",
        },
        annotation_file="instances_val2017.json",
    )

    result = decoder.decode(_nine_hailo_pose_heads())

    assert result["detections"].shape == (0, 7)
    assert result["keypoints"].shape == (0, 17, 3)


def test_hailo_vendor_pose_factory_does_not_apply_class_sigmoid_twice():
    decoder = create_decoder(
        make_pose_spec(),
        backend="hailort",
        runtime_options={
            "hailo_yolov8_pose_raw_heads": True,
            "hailo_pose_output_abi": "yolov8-pose-dfl-nhwc-v1",
            "yolov8_pose_class_scores_are_probabilities": True,
        },
        annotation_file="instances_val2017.json",
    )

    packed = decoder._decode_heads(_nine_hailo_pose_heads(class_logit=0.2))

    assert np.allclose(packed[:, 4], 0.2)


def test_deepx_packed_pose_loader_preserves_coco_identity(tmp_path):
    _, loader = _pose_loader(tmp_path, backend="deepx")

    sample = loader.load_single()

    assert sample["label"] == {
        "image_id": 1,
        "file_name": "000000000001.jpg",
    }


def test_deepx_packed_pose_factory_routes_decoder_and_accuracy_evaluator(tmp_path):
    artifact = _write_pose_dxnn(
        tmp_path / "cpu-tail-pose.dxnn",
        {
            "offloading": False,
            "outputs": ["output0"],
            "graphs": [
                {
                    "name": "npu_0",
                    "device": "NPU",
                    "outputs": [
                        {"name": name}
                        for name in _DEEPX_POSE_PACKED_DFL_OUTPUTS
                    ],
                },
                {
                    "name": "cpu_0",
                    "device": "CPU",
                    "inputs": [
                        {"name": name}
                        for name in _DEEPX_POSE_PACKED_DFL_OUTPUTS
                    ],
                    "outputs": [{"name": "output0", "tail": True}],
                },
            ],
        },
    )
    fixture, loader = _pose_loader(
        tmp_path / "dataset", backend="deepx", artifact_path=artifact
    )
    runtime_options = loader.get_metadata()["runtime_options"]

    decoder = create_decoder(
        make_pose_spec(),
        backend="deepx",
        runtime_options=runtime_options,
        annotation_file=str(fixture["pose"]),
    )
    evaluator = create_evaluator(
        make_pose_spec(),
        backend="deepx",
        annotation_file=str(fixture["pose"]),
    )

    assert type(decoder).__name__ == "YoloV8PoseDecoder"
    assert decoder.conf_threshold == 0.001
    assert decoder.iou_threshold == 0.70
    assert type(evaluator).__name__ == "PoseEstimationEvaluator"


def test_deepx_pose_enablement_preserves_segmentation_latency_only_contract():
    decoder = create_decoder(make_seg_spec(), backend="deepx")
    evaluator = create_evaluator(
        make_seg_spec(),
        backend="deepx",
        annotation_file="unused-for-latency-only.json",
    )

    assert decoder is None
    assert type(evaluator).__name__ == "LatencyOnlyEvaluator"


def test_deepx_raw_pose_artifact_routes_nchw_head_decoder(tmp_path):
    artifact = _write_pose_dxnn(
        tmp_path / "raw-pose.dxnn",
        {
            "offloading": False,
            "outputs": list(_DEEPX_POSE_RAW_OUTPUTS),
            "graphs": [
                {
                    "name": "npu_0",
                    "device": "NPU",
                    "outputs": [
                        {"name": name} for name in _DEEPX_POSE_RAW_OUTPUTS
                    ],
                }
            ],
        },
    )
    fixture, loader = _pose_loader(
        tmp_path / "dataset", backend="deepx", artifact_path=artifact
    )
    runtime_options = loader.get_metadata()["runtime_options"]

    decoder = create_decoder(
        make_pose_spec(),
        backend="deepx",
        runtime_options=runtime_options,
        annotation_file=str(fixture["pose"]),
    )
    result = decoder.decode(_deepx_nchw_pose_heads())

    assert runtime_options["deepx_raw_head_abi"] == (
        "yolov8-pose-dfl-nchw-v1"
    )
    assert type(decoder).__name__ == "DeepXYoloV8PoseRawHeadDecoder"
    assert result["detections"].shape == (0, 7)
    assert result["keypoints"].shape == (0, 17, 3)


def test_deepx_cpu_tail_pose_artifact_enables_ort_for_packed_output(tmp_path):
    artifact = _write_pose_dxnn(
        tmp_path / "cpu-tail-pose.dxnn",
        {
            "offloading": False,
            "outputs": ["output0"],
            "graphs": [
                {
                    "name": "npu_0",
                    "device": "NPU",
                    "outputs": [
                        {"name": name} for name in _DEEPX_POSE_RAW_OUTPUTS
                    ],
                },
                {
                    "name": "cpu_0",
                    "device": "CPU",
                    "inputs": [
                        {"name": name} for name in _DEEPX_POSE_RAW_OUTPUTS
                    ],
                    "outputs": [{"name": "output0", "tail": True}],
                },
            ],
        },
    )
    fixture, loader = _pose_loader(
        tmp_path / "dataset", backend="deepx", artifact_path=artifact
    )
    runtime_options = loader.get_metadata()["runtime_options"]

    decoder = create_decoder(
        make_pose_spec(),
        backend="deepx",
        runtime_options=runtime_options,
        annotation_file=str(fixture["pose"]),
    )

    assert runtime_options["use_ort"] is True
    assert runtime_options["deepx_packed_pose_abi"] == (
        "yolov8-pose-packed-b56n-v1"
    )
    assert "deepx_raw_head_abi" not in runtime_options
    assert type(decoder).__name__ == "YoloV8PoseDecoder"


def test_deepx_unknown_cpu_tail_does_not_claim_packed_pose_abi(tmp_path):
    artifact = _write_pose_dxnn(
        tmp_path / "unknown-cpu-tail.dxnn",
        {
            "offloading": False,
            "outputs": ["output0"],
            "graphs": [
                {
                    "name": "npu_0",
                    "device": "NPU",
                    "outputs": [{"name": "unrelated_intermediate"}],
                },
                {
                    "name": "cpu_0",
                    "device": "CPU",
                    "inputs": [{"name": "unrelated_intermediate"}],
                    "outputs": [{"name": "output0", "tail": True}],
                },
            ],
        },
    )
    fixture, loader = _pose_loader(
        tmp_path / "dataset", backend="deepx", artifact_path=artifact
    )
    runtime_options = loader.get_metadata()["runtime_options"]

    assert runtime_options["use_ort"] is True
    assert "deepx_packed_pose_abi" not in runtime_options
    with pytest.raises(ValueError, match="verified pose output ABI"):
        create_decoder(
            make_pose_spec(),
            backend="deepx",
            runtime_options=runtime_options,
            annotation_file=str(fixture["pose"]),
        )


def test_deepx_class_probability_raw_pose_artifact_preserves_activation(tmp_path):
    artifact = _write_pose_dxnn(
        tmp_path / "class-probability-raw-pose.dxnn",
        {
            "offloading": False,
            "outputs": list(_DEEPX_POSE_CLASS_PROBABILITY_OUTPUTS),
            "graphs": [
                {
                    "name": "npu_0",
                    "device": "NPU",
                    "outputs": [
                        {"name": name}
                        for name in _DEEPX_POSE_CLASS_PROBABILITY_OUTPUTS
                    ],
                }
            ],
        },
    )
    fixture, loader = _pose_loader(
        tmp_path / "dataset", backend="deepx", artifact_path=artifact
    )
    runtime_options = loader.get_metadata()["runtime_options"]

    decoder = create_decoder(
        make_pose_spec(),
        backend="deepx",
        runtime_options=runtime_options,
        annotation_file=str(fixture["pose"]),
    )
    normalized, _ = decoder._normalize_heads(
        _deepx_nchw_pose_heads(class_logit=0.2)
    )
    packed = decoder._decoder._decode_heads(normalized)

    assert runtime_options["deepx_raw_head_abi"] == (
        "yolov8-pose-dfl-class-probability-nchw-v1"
    )
    assert runtime_options["yolov8_pose_class_scores_are_probabilities"] is True
    assert type(decoder).__name__ == "DeepXYoloV8PoseRawHeadDecoder"
    assert decoder.class_scores_are_probabilities is True
    assert np.allclose(packed[:, 4], 0.2)
    assert decoder.result_metadata()["deepx_raw_head_abi"] == (
        "yolov8-pose-dfl-class-probability-nchw-v1"
    )


def test_deepx_latency_only_pose_without_verified_abi_has_no_decoder():
    decoder = create_decoder(
        make_pose_spec(),
        backend="deepx",
        runtime_options={},
        annotation_file=None,
    )

    assert decoder is None


def test_deepx_accuracy_pose_unknown_abi_fails_closed():
    with pytest.raises(ValueError, match="verified pose output ABI"):
        create_decoder(
            make_pose_spec(),
            backend="deepx",
            runtime_options={},
            annotation_file="person_keypoints_val2017.json",
        )


def test_hailo_accuracy_pose_unknown_abi_fails_closed():
    with pytest.raises(ValueError, match="verified pose output ABI"):
        create_decoder(
            make_pose_spec(),
            backend="hailort",
            runtime_options={},
            annotation_file="person_keypoints_val2017.json",
        )


def test_hailo_accuracy_pose_boolean_without_abi_marker_fails_closed():
    with pytest.raises(ValueError, match="verified pose output ABI"):
        create_decoder(
            make_pose_spec(),
            backend="hailort",
            runtime_options={"hailo_yolov8_pose_raw_heads": True},
            annotation_file="person_keypoints_val2017.json",
        )
