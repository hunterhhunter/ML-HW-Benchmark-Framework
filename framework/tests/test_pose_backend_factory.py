import json

import numpy as np

from coco_test_utils import make_pose_spec, make_seg_spec, write_coco_fixture
from dataloader import create_dataloader
from dataloader.deepx_image_classification_loader import (
    read_dxnn_compile_config,
)
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


def test_hailo_pose_factory_routes_loader_and_decoder(tmp_path):
    _, loader = _pose_loader(tmp_path, backend="hailort")

    sample = loader.load_single()
    runtime_options = loader.get_metadata()["runtime_options"]
    decoder = create_decoder(
        make_pose_spec(),
        backend="hailort",
        runtime_options=runtime_options,
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
        "yolov8_pose_class_scores_are_probabilities": True,
    }
    assert type(decoder).__name__ == "HailoYoloV8PoseRawHeadDecoder"


def test_hailo_pose_factory_decoder_accepts_nine_raw_heads():
    decoder = create_decoder(
        make_pose_spec(),
        backend="hailort",
        runtime_options={"hailo_yolov8_pose_raw_heads": True},
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
            "yolov8_pose_class_scores_are_probabilities": True,
        },
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
    fixture, loader = _pose_loader(tmp_path, backend="deepx")
    runtime_options = loader.get_metadata()["runtime_options"]

    decoder = create_decoder(
        make_pose_spec(),
        backend="deepx",
        runtime_options=runtime_options,
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
    _, loader = _pose_loader(
        tmp_path / "dataset", backend="deepx", artifact_path=artifact
    )
    runtime_options = loader.get_metadata()["runtime_options"]

    decoder = create_decoder(
        make_pose_spec(),
        backend="deepx",
        runtime_options=runtime_options,
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
                    "outputs": [{"name": "intermediate"}],
                },
                {
                    "name": "cpu_0",
                    "device": "CPU",
                    "outputs": [{"name": "output0"}],
                },
            ],
        },
    )
    _, loader = _pose_loader(
        tmp_path / "dataset", backend="deepx", artifact_path=artifact
    )
    runtime_options = loader.get_metadata()["runtime_options"]

    decoder = create_decoder(
        make_pose_spec(),
        backend="deepx",
        runtime_options=runtime_options,
    )

    assert runtime_options["use_ort"] is True
    assert "deepx_raw_head_abi" not in runtime_options
    assert type(decoder).__name__ == "YoloV8PoseDecoder"
