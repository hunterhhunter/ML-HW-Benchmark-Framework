from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest

from core.compiled_model import CompiledModel
from core.model_profiles import create_model_spec
from dataloader import create_dataloader
from decoders import create_decoder
from evaluators import create_evaluator
from runtimes.onnx_rt import OnnxRuntime
from utils.dataset_resolver import resolve_dataset_paths


CASES = [
    (
        "yolov8s-seg",
        "models/yolov8s-seg/yolov8s-seg.onnx",
        "instances_val2017.json",
        "Mask mAP",
        "masks",
        2,
    ),
    (
        "yolov8s-pose",
        "models/yolov8s-pose/yolov8s-pose.onnx",
        "person_keypoints_val2017.json",
        "OKS mAP",
        "keypoints",
        1,
    ),
]


@pytest.mark.parametrize(
    (
        "model_name",
        "model_path",
        "annotation_name",
        "metric_key",
        "canonical_key",
        "output_count",
    ),
    CASES,
)
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_real_yolov8_model_runs_requested_provider(
    model_name,
    model_path,
    annotation_name,
    metric_key,
    canonical_key,
    output_count,
    device,
):
    dataset_root = Path("datasets/coco")
    required = [
        Path(model_path),
        dataset_root / "images" / "val2017",
        dataset_root / "annotations" / annotation_name,
    ]
    if not all(path.exists() for path in required):
        pytest.skip("real YOLOv8/COCO assets are not prepared")
    if (
        device == "cuda"
        and "CUDAExecutionProvider" not in ort.get_available_providers()
    ):
        pytest.skip("CUDAExecutionProvider unavailable")

    spec = create_model_spec(model_name, model_path)
    image_dir, annotation_file = resolve_dataset_paths(
        spec.task, str(dataset_root), "", ""
    )
    loader = create_dataloader(
        spec,
        dataset_path=str(dataset_root),
        image_dir=image_dir,
        label_path=annotation_file,
        cache_dir=None,
    )
    runtime = OnnxRuntime(device=device)
    runtime.load(
        CompiledModel(
            spec,
            "onnxruntime",
            Path(model_path),
        )
    )
    try:
        sample = loader.load_by_index(0)
        input_name = next(iter(spec.input_shapes))
        outputs = runtime.run({input_name: sample["input"][None]})

        assert len(outputs) == output_count
        assert all(isinstance(value, np.ndarray) for value in outputs.values())
        decoded = create_decoder(spec, backend="onnxruntime").decode(outputs)
        assert decoded["detections"].ndim == 2
        assert decoded["detections"].shape[1] == 7
        assert canonical_key in decoded
        assert len(decoded[canonical_key]) == len(decoded["detections"])

        evaluator = create_evaluator(
            spec,
            backend="onnxruntime",
            annotation_file=annotation_file,
        )
        evaluator.add_batch(
            decoded,
            [
                {
                    "label": sample["label"],
                    "preprocess_context": sample["preprocess_context"],
                }
            ],
            1.0,
        )
        metrics = evaluator.compute()
        assert metric_key in metrics
        assert metrics["Total Samples"] == 1

        active = runtime.get_device_spec()["active_providers"]
        requested = (
            "CUDAExecutionProvider"
            if device == "cuda"
            else "CPUExecutionProvider"
        )
        assert requested in active
    finally:
        runtime.unload()
