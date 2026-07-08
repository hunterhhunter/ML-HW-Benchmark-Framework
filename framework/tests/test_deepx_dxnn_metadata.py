import json
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
from PIL import Image

from core.benchmarkrunner import BenchmarkRunner
from core.compiled_model import CompiledModel
from core.model_spec import Model_Spec, Task
from core.model_profiles import create_model_spec
from dataloader import create_dataloader
from dataloader.deepx_loader import DeepXDataLoader
from dataloader.deepx_image_classification_loader import (
    DeepXDirectResizeRawPreprocess,
    DeepXImageClassificationLoader,
    deepx_rmap_image_input_layout,
    deepx_rmap_input_dtype,
    read_dxnn_rmap_input_info,
    resolve_deepx_image_input_config,
)
from dataloader.deepx_vision_loader import (
    DeepXInstanceSegmentationLoader,
    DeepXObjectDetectionLoader,
    DeepXPoseEstimationLoader,
)
from evaluators import create_evaluator
from evaluators.latency_evaluator import LatencyOnlyEvaluator
from runtimes import create_runtime


def _write_fake_dxnn(path: Path, rmap_info: dict) -> None:
    payload = json.dumps(rmap_info).encode("utf-8")
    header = {
        "size": 8192,
        "data": {
            "compiled_data": {
                "M1A_4K": {
                    "npu_0": {
                        "rmap_info": {
                            "type": "str",
                            "offset": 0,
                            "size": len(payload),
                        },
                    },
                },
            },
        },
    }
    header_bytes = json.dumps(header).encode("utf-8")
    assert len(header_bytes) < 8184
    path.write_bytes(b"DXNN" + (1).to_bytes(4, "little") + header_bytes + b"\0" * (8184 - len(header_bytes)) + payload)


def _install_fake_dx_engine(monkeypatch, output_shape=(1, 84, 8400)):
    state = {"calls": []}

    class FakeBoundOption:
        NPU_ALL = "NPU_ALL"

    class FakeInferenceOption:
        BOUND_OPTION = FakeBoundOption

        def set_devices(self, devices):
            self.devices = devices

        def set_bound_option(self, bound_option):
            self.bound_option = bound_option

    class FakeInferenceEngine:
        def __init__(self, model_path, option=None):
            self.model_path = model_path
            self.option = option

        def get_input_tensor_names(self):
            return ["images"]

        def get_output_tensor_names(self):
            return ["output"]

        def get_input_tensors_info(self):
            return [{"name": "images", "shape": [1, 640, 640, 3], "dtype": np.dtype("uint8")}]

        def get_output_tensors_info(self):
            return [{"name": "output", "shape": list(output_shape), "dtype": np.dtype("float32")}]

        def run(self, input_data):
            state["calls"].append(input_data)
            return [np.ones(output_shape, dtype=np.float32)]

        def dispose(self):
            pass

    module = types.SimpleNamespace(
        InferenceEngine=FakeInferenceEngine,
        InferenceOption=FakeInferenceOption,
        __version__="fake",
    )
    monkeypatch.setitem(sys.modules, "dx_engine", module)
    return state


def test_read_dxnn_rmap_input_info_reads_uint8_nhwc_image_input(tmp_path):
    artifact = tmp_path / "model.dxnn"
    _write_fake_dxnn(
        artifact,
        {
            "inputs": [
                {
                    "name": "input.1",
                    "dtype": "UINT8",
                    "shape": [1, 224, 224, 3],
                },
            ],
        },
    )

    info = read_dxnn_rmap_input_info(artifact)

    assert info["name"] == "input.1"
    assert deepx_rmap_input_dtype(info) == "UINT8"
    assert deepx_rmap_image_input_layout(info) == "NHWC"


def test_deepx_uint8_rmap_input_resolves_raw_dxapp_runtime_options(tmp_path):
    artifact = tmp_path / "model.dxnn"
    _write_fake_dxnn(
        artifact,
        {
            "inputs": [
                {
                    "name": "input.1",
                    "dtype": "UINT8",
                    "shape": [1, 224, 224, 3],
                },
            ],
        },
    )

    config = resolve_deepx_image_input_config(
        artifact_path=artifact,
        compile_options={},
        requested_mode="auto",
        compile_enabled=False,
    )

    assert config.preprocess_mode == "raw"
    assert config.expects_uint8_image is True
    assert config.runtime_options == {
        "input_layout": "NHWC",
        "input_dtype": "uint8",
        "input_batch_axis": "squeeze",
        "single_input_run_style": "list",
    }


def test_deepx_direct_resize_raw_preprocess_matches_dxapp_simple_resize_contract():
    strategy = DeepXDirectResizeRawPreprocess()
    img = Image.new("RGB", (320, 240), color=(10, 20, 30))

    tensor = strategy(
        img,
        target_hw=(224, 224),
        mean=np.array([0.485, 0.456, 0.406], dtype=np.float32),
        std=np.array([0.229, 0.224, 0.225], dtype=np.float32),
    )

    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == np.float32
    assert tensor[0, 0, 0] == 10.0
    assert tensor[1, 0, 0] == 20.0
    assert tensor[2, 0, 0] == 30.0


def test_create_dataloader_routes_deepx_image_classification_to_deepx_loader(tmp_path):
    artifact = tmp_path / "model.dxnn"
    _write_fake_dxnn(
        artifact,
        {
            "inputs": [
                {
                    "name": "input.1",
                    "dtype": "UINT8",
                    "shape": [1, 224, 224, 3],
                },
            ],
        },
    )
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    Image.new("RGB", (320, 240), color=(10, 20, 30)).save(image_dir / "sample.jpg")
    label_path = tmp_path / "labels.json"
    label_path.write_text(json.dumps({"sample.jpg": 7}), encoding="utf-8")
    spec = Model_Spec(
        name="resnet50",
        task=Task.IMAGE_CLASSIFICATION,
        input_shapes={"input.1": (1, 3, 224, 224)},
        input_dtype={"input.1": "float32"},
        output_shapes={"495": (1, 1000)},
        model_paths={},
    )

    loader = create_dataloader(
        spec,
        dataset_path=str(tmp_path),
        image_dir=str(image_dir),
        label_path=str(label_path),
        backend="deepx",
        artifact_path=str(artifact),
        compile_options={},
        compile_enabled=False,
        image_preprocess_mode="auto",
    )
    sample = loader.load_single()

    assert isinstance(loader, DeepXDataLoader)
    assert isinstance(loader.delegate, DeepXImageClassificationLoader)
    assert sample["input"].shape == (224, 224, 3)
    assert sample["input"].dtype == np.float32
    assert sample["label"] == 7
    metadata = loader.get_metadata()
    assert metadata["runtime_options"] == {
        "input_layout": "NHWC",
        "input_dtype": "uint8",
        "input_batch_axis": "squeeze",
        "single_input_run_style": "list",
    }
    assert metadata["deepx"] == {
        "task": "IMAGE_CLASSIFICATION",
        "delegate_loader": "DeepXImageClassificationLoader",
    }


def test_create_dataloader_routes_deepx_object_detection_through_deepx_loader(tmp_path):
    artifact = tmp_path / "model.dxnn"
    _write_fake_dxnn(
        artifact,
        {
            "inputs": [
                {
                    "name": "images",
                    "dtype": "UINT8",
                    "shape": [1, 640, 640, 3],
                },
            ],
        },
    )
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    Image.new("RGB", (320, 240), color=(10, 20, 30)).save(image_dir / "sample.jpg")
    label_dir = tmp_path / "labels"
    label_dir.mkdir()
    (label_dir / "sample.txt").write_text("0 0.5 0.5 0.25 0.25\n", encoding="utf-8")

    spec = Model_Spec(
        name="yolov5m",
        task=Task.OBJECT_DETECTION,
        input_shapes={"images": (1, 3, 640, 640)},
        input_dtype={"images": "float32"},
        output_shapes={"output": (1, 25200, 85)},
        model_paths={},
    )

    loader = create_dataloader(
        spec,
        dataset_path=str(tmp_path),
        image_dir=str(image_dir),
        label_path=str(label_dir),
        backend="deepx",
        artifact_path=str(artifact),
        compile_options={},
        compile_enabled=False,
    )
    sample = loader.load_single()
    metadata = loader.get_metadata()

    assert isinstance(loader, DeepXDataLoader)
    assert isinstance(loader.delegate, DeepXObjectDetectionLoader)
    assert sample["input"].shape == (640, 640, 3)
    assert sample["input"].dtype == np.uint8
    assert sample["label"].shape == (1, 5)
    assert metadata["runtime_options"] == {
        "input_layout": "NHWC",
        "input_batch_axis": "squeeze",
        "single_input_run_style": "list",
        "input_dtype": "uint8",
    }
    assert metadata["deepx"] == {
        "task": "OBJECT_DETECTION",
        "delegate_loader": "DeepXObjectDetectionLoader",
    }


def test_deepx_object_detection_runner_reaches_dxrt_uint8_nhwc_contract(monkeypatch, tmp_path):
    state = _install_fake_dx_engine(monkeypatch)
    artifact = tmp_path / "model.dxnn"
    _write_fake_dxnn(
        artifact,
        {
            "inputs": [
                {
                    "name": "images",
                    "dtype": "UINT8",
                    "shape": [1, 640, 640, 3],
                },
            ],
        },
    )
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    Image.new("RGB", (320, 240), color=(10, 20, 30)).save(image_dir / "sample.jpg")
    label_dir = tmp_path / "labels"
    label_dir.mkdir()
    (label_dir / "sample.txt").write_text("0 0.5 0.5 0.25 0.25\n", encoding="utf-8")

    spec = create_model_spec(
        "yolov8m",
        "models/yolov8m/yolov8m.onnx",
        sniff_onnx=False,
    )
    loader = create_dataloader(
        spec,
        dataset_path=str(tmp_path),
        image_dir=str(image_dir),
        label_path=str(label_dir),
        backend="deepx",
        artifact_path=str(artifact),
        compile_options={},
        compile_enabled=False,
    )
    runtime = create_runtime("deepx", device="npu0", **loader.get_metadata()["runtime_options"])
    runtime.load(CompiledModel(spec=spec, backend_name="deepx", artifact_path=artifact))
    evaluator = LatencyOnlyEvaluator(task_name=spec.task.name)

    metrics = BenchmarkRunner(loader, runtime, evaluator).run(warmup_runs=1, batch_size=1, max_steps=1)
    runtime.unload()

    assert len(state["calls"]) == 2
    for sdk_call in state["calls"]:
        assert isinstance(sdk_call, list)
        assert len(sdk_call) == 1
        assert sdk_call[0].shape == (640, 640, 3)
        assert sdk_call[0].dtype == np.uint8
    assert metrics["Task"] == "OBJECT_DETECTION"
    assert metrics["Total Samples"] == 1
    assert metrics["Output Shapes"] == {"output": (1, 84, 8400)}


def test_deepx_segmentation_float_nchw_uses_letterbox_and_preserves_ragged_labels(tmp_path):
    artifact = tmp_path / "model.dxnn"
    _write_fake_dxnn(
        artifact,
        {
            "inputs": [
                {
                    "name": "images",
                    "dtype": "FLOAT",
                    "shape": [1, 3, 640, 640],
                },
            ],
        },
    )
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    Image.new("RGB", (320, 240), color=(10, 20, 30)).save(image_dir / "sample.jpg")
    label_dir = tmp_path / "labels"
    label_dir.mkdir()
    (label_dir / "sample.txt").write_text(
        "0 0.10 0.20 0.30 0.40 0.50 0.60\n"
        "1 0.20 0.30 0.40 0.50 0.60 0.70 0.80 0.90\n",
        encoding="utf-8",
    )

    spec = create_model_spec(
        "yolov8-seg-m",
        "models/yolov8m-seg/yolov8m-seg.onnx",
        sniff_onnx=False,
    )
    loader = create_dataloader(
        spec,
        dataset_path=str(tmp_path),
        image_dir=str(image_dir),
        label_path=str(label_dir),
        backend="deepx",
        artifact_path=str(artifact),
        layout="NHWC",
        compile_options={},
        compile_enabled=False,
    )
    sample = loader.load_single()
    metadata = loader.get_metadata()

    assert isinstance(loader.delegate, DeepXInstanceSegmentationLoader)
    assert sample["input"].shape == (3, 640, 640)
    assert sample["input"].dtype == np.float32
    assert np.isclose(sample["input"][0, 0, 0], 114.0 / 255.0)
    assert sample["preprocess_context"]["scale"] == 2.0
    assert sample["preprocess_context"]["pad_x"] == 0
    assert sample["preprocess_context"]["pad_y"] == 80
    assert isinstance(sample["label"], list)
    assert [row.shape[0] for row in sample["label"]] == [7, 9]
    assert metadata["runtime_options"] == {
        "input_layout": "NCHW",
        "input_batch_axis": "squeeze",
        "single_input_run_style": "list",
        "input_dtype": "float32",
    }


def test_deepx_pose_uint8_nhwc_preserves_keypoint_label_columns(tmp_path):
    artifact = tmp_path / "model.dxnn"
    _write_fake_dxnn(
        artifact,
        {
            "inputs": [
                {
                    "name": "images",
                    "dtype": "UINT8",
                    "shape": [1, 640, 640, 3],
                },
            ],
        },
    )
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    Image.new("RGB", (640, 480), color=(50, 60, 70)).save(image_dir / "sample.jpg")
    label_dir = tmp_path / "labels"
    label_dir.mkdir()
    pose_values = [0, 0.5, 0.5, 0.25, 0.25] + [0.1] * 51
    (label_dir / "sample.txt").write_text(
        " ".join(str(value) for value in pose_values) + "\n",
        encoding="utf-8",
    )

    spec = create_model_spec(
        "yolov8-pose-m",
        "models/yolov8m-pose/yolov8m-pose.onnx",
        sniff_onnx=False,
    )
    loader = create_dataloader(
        spec,
        dataset_path=str(tmp_path),
        image_dir=str(image_dir),
        label_path=str(label_dir),
        backend="deepx",
        artifact_path=str(artifact),
        compile_options={},
        compile_enabled=False,
    )
    sample = loader.load_single()

    assert isinstance(loader.delegate, DeepXPoseEstimationLoader)
    assert sample["input"].shape == (640, 640, 3)
    assert sample["input"].dtype == np.uint8
    assert sample["label"].shape == (1, 56)
    assert np.allclose(sample["label"][0, 5:], np.array([0.1] * 51, dtype=np.float32))


def test_create_dataloader_routes_deepx_segmentation_and_pose_to_deepx_vision_loader(tmp_path):
    artifact = tmp_path / "model.dxnn"
    _write_fake_dxnn(
        artifact,
        {
            "inputs": [
                {
                    "name": "images",
                    "dtype": "UINT8",
                    "shape": [1, 640, 640, 3],
                },
            ],
        },
    )
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    Image.new("RGB", (320, 240), color=(10, 20, 30)).save(image_dir / "sample.jpg")

    seg_spec = Model_Spec(
        name="yolov8m-seg",
        task=Task.INSTANCE_SEGMENTATION,
        input_shapes={"images": (1, 3, 640, 640)},
        input_dtype={"images": "float32"},
        output_shapes={"output0": (1, 116, 8400), "output1": (1, 32, 160, 160)},
        model_paths={},
    )
    pose_spec = Model_Spec(
        name="yolov8m-pose",
        task=Task.POSE_ESTIMATION,
        input_shapes={"images": (1, 3, 640, 640)},
        input_dtype={"images": "float32"},
        output_shapes={"output": (1, 56, 8400)},
        model_paths={},
    )

    seg_loader = create_dataloader(
        seg_spec,
        dataset_path=str(tmp_path),
        image_dir=str(image_dir),
        backend="deepx",
        artifact_path=str(artifact),
        compile_options={},
        compile_enabled=False,
    )
    pose_loader = create_dataloader(
        pose_spec,
        dataset_path=str(tmp_path),
        image_dir=str(image_dir),
        backend="deepx",
        artifact_path=str(artifact),
        compile_options={},
        compile_enabled=False,
    )

    assert isinstance(seg_loader, DeepXDataLoader)
    assert isinstance(seg_loader.delegate, DeepXInstanceSegmentationLoader)
    assert seg_loader.load_single()["input"].shape == (640, 640, 3)
    assert seg_loader.get_metadata()["deepx"] == {
        "task": "INSTANCE_SEGMENTATION",
        "delegate_loader": "DeepXInstanceSegmentationLoader",
    }

    assert isinstance(pose_loader, DeepXDataLoader)
    assert isinstance(pose_loader.delegate, DeepXPoseEstimationLoader)
    assert pose_loader.load_single()["input"].shape == (640, 640, 3)
    assert pose_loader.get_metadata()["deepx"] == {
        "task": "POSE_ESTIMATION",
        "delegate_loader": "DeepXPoseEstimationLoader",
    }


def test_yolov8_vision_profiles_create_specs_and_latency_evaluators():
    detection_spec = create_model_spec(
        "yolov8m",
        "models/yolov8m/yolov8m.onnx",
        sniff_onnx=False,
    )
    seg_spec = create_model_spec(
        "yolov8-seg-m",
        "models/yolov8m-seg/yolov8m-seg.onnx",
        sniff_onnx=False,
    )
    pose_spec = create_model_spec(
        "yolov8-pose-m",
        "models/yolov8m-pose/yolov8m-pose.onnx",
        sniff_onnx=False,
    )

    assert detection_spec.task == Task.OBJECT_DETECTION
    assert seg_spec.task == Task.INSTANCE_SEGMENTATION
    assert pose_spec.task == Task.POSE_ESTIMATION
    assert isinstance(create_evaluator(seg_spec), LatencyOnlyEvaluator)
    assert isinstance(create_evaluator(pose_spec), LatencyOnlyEvaluator)
