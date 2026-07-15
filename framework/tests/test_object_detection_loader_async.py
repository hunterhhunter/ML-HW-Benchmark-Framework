from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from core.async_inference.runner import AsyncBenchmarkRunner
from core.async_inference.types import AsyncInferenceConfig, RunStatus
from core.model_spec import Model_Spec, Task
from dataloader.object_detection_loader import ObjectDetectionLoader


class DeterministicDetectionPreprocessor:
    def get_cache_path(self, cache_dir, image_filename):
        del cache_dir, image_filename
        return None

    def load_or_preprocess_with_context(self, cache_path, image_path):
        del cache_path
        sample_number = 1 if Path(image_path).stem == "a" else 2
        return (
            np.full((3, 2, 2), sample_number, dtype=np.float32),
            {
                "image_name": Path(image_path).name,
                "scale": float(sample_number),
            },
        )


class DetectionRuntime:
    def __init__(self, spec):
        self.compiled_model = SimpleNamespace(spec=spec)

    def supports_generate(self):
        return False

    def max_concurrent_workers(self):
        return 1

    def supports_dynamic_batching(self):
        return False

    def max_dynamic_batch_size(self):
        return 1

    def supports_batch_generation(self):
        return False

    def run(self, inputs):
        batch_size = inputs["images"].shape[0]
        return {
            "detections": np.zeros((batch_size, 1), dtype=np.float32),
        }


class RecordingDetectionEvaluator:
    def __init__(self):
        self.labels = []

    def add_batch(self, outputs, labels, timing_ms):
        del outputs, timing_ms
        self.labels.extend(labels)

    def compute(self):
        return {"Total Samples": len(self.labels)}


def _detection_spec():
    return Model_Spec(
        name="tiny-detection",
        task=Task.OBJECT_DETECTION,
        input_shapes={"images": (None, 3, 2, 2)},
        input_dtype={"images": "float32"},
        output_shapes={"detections": (None, 1)},
        model_paths={"onnx": Path("tiny-detection.onnx")},
    )


def _make_loader(tmp_path):
    image_dir = tmp_path / "images"
    label_dir = tmp_path / "labels"
    image_dir.mkdir()
    label_dir.mkdir()
    (image_dir / "a.jpg").write_bytes(b"synthetic-a")
    (image_dir / "b.png").write_bytes(b"synthetic-b")
    (label_dir / "a.txt").write_text(
        "1 0.5 0.5 0.25 0.75\n",
        encoding="utf-8",
    )
    (label_dir / "b.txt").write_text(
        "2 0.25 0.75 0.5 0.25\n",
        encoding="utf-8",
    )
    spec = _detection_spec()
    return (
        ObjectDetectionLoader(
            spec,
            image_dir=str(image_dir),
            label_path=str(label_dir),
            preprocessor=DeterministicDetectionPreprocessor(),
        ),
        spec,
    )


def test_object_detection_loader_random_access_drives_async_completion(
    tmp_path,
):
    loader, spec = _make_loader(tmp_path)

    second = loader.load_by_index(1)
    assert loader.current_idx == 0
    first = loader.load_single()
    assert loader.current_idx == 1
    first_again = loader.load_by_index(0)
    assert loader.current_idx == 1
    with pytest.raises(IndexError, match="out of range"):
        loader.load_by_index(-1)
    with pytest.raises(IndexError, match="out of range"):
        loader.load_by_index(2)
    assert loader.current_idx == 1

    expected_keys = {"input", "label", "img_path", "preprocess_context"}
    assert set(first) == set(first_again) == set(second) == expected_keys
    np.testing.assert_array_equal(first["input"], first_again["input"])
    np.testing.assert_array_equal(first["label"], first_again["label"])
    assert first["img_path"] == first_again["img_path"]
    assert first["preprocess_context"] == first_again["preprocess_context"]

    evaluator = RecordingDetectionEvaluator()
    result = AsyncBenchmarkRunner(
        loader,
        DetectionRuntime(spec),
        evaluator,
    ).run(
        AsyncInferenceConfig(
            queue_capacity=2,
            max_batch_size=1,
            batch_timeout_ms=0,
            min_samples=1,
        ),
        warmup_runs=0,
    )

    assert result.status is RunStatus.VALID
    assert result.metrics["async_completed_requests"] == 2
    assert result.metrics["async_outstanding_requests"] == 0
    assert result.metrics["Total Samples"] == 2
    assert loader.current_idx == 1
    assert [item["preprocess_context"] for item in evaluator.labels] == [
        {"image_name": "a.jpg", "scale": 1.0},
        {"image_name": "b.png", "scale": 2.0},
    ]
    np.testing.assert_array_equal(
        evaluator.labels[0]["label"],
        np.array([[1, 0.5, 0.5, 0.25, 0.75]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        evaluator.labels[1]["label"],
        np.array([[2, 0.25, 0.75, 0.5, 0.25]], dtype=np.float32),
    )
