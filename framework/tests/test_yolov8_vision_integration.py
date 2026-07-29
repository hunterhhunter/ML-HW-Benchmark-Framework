from types import SimpleNamespace

import numpy as np
import pytest

import main as benchmark_main
from coco_test_utils import make_pose_spec, make_seg_spec, write_coco_fixture
from core.async_inference.completion import CompletionCoordinator
from core.async_inference.metrics import AsyncMetricsCollector
from core.async_inference.types import BatchCompletion, InferenceRequest
from core.inference_pipeline import InferencePipeline
from core.model_spec import Task
from dataloader.coco_instance_segmentation_loader import (
    CocoInstanceSegmentationLoader,
)
from decoders import create_decoder
from decoders.instance_segmentation import YoloV8SegmentationDecoder
from decoders.pose_estimation import YoloV8PoseDecoder
from evaluators import create_evaluator
from evaluators.instance_segmentation_evaluator import (
    InstanceSegmentationEvaluator,
)
from evaluators.pose_estimation_evaluator import PoseEstimationEvaluator
from run_all_onnx_benchmarks import BENCHMARK_CONFIGS


def test_factories_route_both_vision_tasks(tmp_path):
    paths = write_coco_fixture(tmp_path)
    seg_spec = make_seg_spec()
    pose_spec = make_pose_spec()

    seg_decoder = create_decoder(seg_spec, backend="onnxruntime")
    pose_decoder = create_decoder(pose_spec, backend="onnxruntime")

    assert isinstance(seg_decoder, YoloV8SegmentationDecoder)
    assert isinstance(pose_decoder, YoloV8PoseDecoder)
    assert isinstance(
        create_evaluator(
            seg_spec,
            backend="onnxruntime",
            annotation_file=str(paths["instances"]),
        ),
        InstanceSegmentationEvaluator,
    )
    assert isinstance(
        create_evaluator(
            pose_spec,
            backend="onnxruntime",
            annotation_file=str(paths["pose"]),
        ),
        PoseEstimationEvaluator,
    )


def test_pipeline_preserves_coco_identity_and_letterbox_context(tmp_path):
    paths = write_coco_fixture(tmp_path)
    spec = make_seg_spec()
    loader = CocoInstanceSegmentationLoader(
        spec,
        dataset_path=str(tmp_path),
        image_dir=str(paths["images"]),
        label_path=str(paths["instances"]),
        target_hw=(8, 8),
    )
    runtime = SimpleNamespace(
        compiled_model=SimpleNamespace(spec=spec),
        supports_generate=lambda: False,
    )
    pipeline = InferencePipeline(loader, runtime)

    collated = pipeline.collate_batch(loader.load_batch(1))
    labels = pipeline.prepare_eval_labels(collated)

    assert labels[0]["label"]["image_id"] == 1
    assert labels[0]["preprocess_context"]["scale"] == 1.0
    assert labels[0]["preprocess_context"]["pad_y"] == 1.0


@pytest.mark.parametrize(
    "task", [Task.INSTANCE_SEGMENTATION, Task.POSE_ESTIMATION]
)
def test_main_builds_accuracy_vision_component_kwargs(task):
    args = SimpleNamespace(
        image_preprocess_mode="auto",
        image_resize_mode="auto",
    )

    loader_kwargs, evaluator_kwargs = benchmark_main._build_vision_task_kwargs(
        task, args, "/dataset/annotations/task.json"
    )

    assert loader_kwargs == {
        "image_preprocess_mode": "normalized",
        "image_resize_mode": "letterbox",
    }
    assert evaluator_kwargs == {
        "annotation_file": "/dataset/annotations/task.json"
    }


@pytest.mark.parametrize("queue_capacity", [None, 1])
def test_inline_and_queued_completion_deliver_same_canonical_payload(
    queue_capacity,
):
    class Pipeline:
        def prepare_eval_labels(self, collated):
            return collated["label"]

    class Decoder:
        def decode(self, outputs):
            assert outputs["raw"].shape == (1, 1)
            return {
                "detections": np.array(
                    [[0, 0, 0.9, 1, 1, 2, 2]], dtype=np.float32
                ),
                "keypoints": np.ones((1, 17, 3), dtype=np.float32),
            }

    class Evaluator:
        def __init__(self):
            self.calls = []

        def add_batch(self, outputs, labels, timing_ms):
            self.calls.append((outputs, labels, timing_ms))

    evaluator = Evaluator()
    coordinator = CompletionCoordinator(
        pipeline=Pipeline(),
        evaluator=evaluator,
        decoder=Decoder(),
        metrics=AsyncMetricsCollector(0, 1),
        queue_capacity=queue_capacity,
        raise_callback_errors=True,
    )
    request = InferenceRequest(
        request_id=0,
        sample_index=0,
        sample={},
        scheduled_ns=0,
        issued_ns=0,
        enqueued_ns=1,
        sample_count=1,
    )
    completion = BatchCompletion(
        requests=[request],
        collated={"label": [{"image_id": 1}]},
        outputs={"raw": np.ones((1, 1), dtype=np.float32)},
        timing_ms=1.0,
        runtime_started_ns=2,
        runtime_finished_ns=3,
        worker_id=0,
        batch_size=1,
    )

    coordinator.start()
    coordinator.register(request)
    coordinator.submit(completion)
    if queue_capacity is not None:
        assert coordinator.wait_for_all(timeout=1.0) is True
    assert coordinator.stop(timeout=1.0) is True

    assert len(evaluator.calls) == 1
    assert evaluator.calls[0][0]["keypoints"].shape == (1, 17, 3)
    assert evaluator.calls[0][1] == [{"image_id": 1}]


def test_benchmark_registry_requires_exact_yolov8_vision_assets():
    configs = {item["model"]: item for item in BENCHMARK_CONFIGS}

    assert configs["yolov8s-seg"]["required_files"] == [
        "models/yolov8s-seg/yolov8s-seg.onnx",
        "datasets/coco/images/val2017",
        "datasets/coco/annotations/instances_val2017.json",
    ]
    assert configs["yolov8s-pose"]["required_files"] == [
        "models/yolov8s-pose/yolov8s-pose.onnx",
        "datasets/coco/images/val2017",
        "datasets/coco/annotations/person_keypoints_val2017.json",
    ]
