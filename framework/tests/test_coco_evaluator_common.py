import numpy as np
import pytest

from coco_test_utils import write_coco_fixture
from core.inference_result import InferenceResult
from evaluators.coco_common import CocoEvaluatorBase


class _ConcreteTestEvaluator(CocoEvaluatorBase):
    def add_batch(self, outputs, labels, timing_ms):
        normalized = self._normalize_batch_labels(labels)
        self._record_batch(normalized, 0, timing_ms)

    def compute(self):
        return self._latency_metrics()

    def evaluate(self, result):
        self._reset()
        timing = result.timing_records[0] if result.timing_records else 0.0
        self.add_batch(result.outputs, result.labels, timing)
        return self.compute()

    def is_applicable(self, device_spec, model_spec):
        return True

    def get_metric_names(self):
        return ["Total Samples", "Average Latency (ms)"]


def _labels(image_id=1):
    return [
        {
            "label": {
                "image_id": image_id,
                "file_name": "000000000001.jpg",
            },
            "preprocess_context": {
                "original_height": 3,
                "original_width": 4,
                "input_height": 8,
                "input_width": 8,
                "scale": 2.0,
                "pad_x": 0.0,
                "pad_y": 1.0,
            },
        }
    ]


def test_normalize_labels_requires_image_id_and_complete_context(tmp_path):
    paths = write_coco_fixture(tmp_path)
    evaluator = _ConcreteTestEvaluator(str(paths["instances"]), "segm")

    with pytest.raises(ValueError, match="image_id"):
        evaluator._normalize_batch_labels(
            [{"label": {}, "preprocess_context": {}}]
        )
    with pytest.raises(ValueError, match="pad_y"):
        labels = _labels()
        labels[0]["preprocess_context"].pop("pad_y")
        evaluator._normalize_batch_labels(labels)
    with pytest.raises(ValueError, match="unknown COCO image_id 2"):
        evaluator._normalize_batch_labels(_labels(image_id=2))


def test_restore_boxes_removes_padding_scale_and_clips(tmp_path):
    paths = write_coco_fixture(tmp_path)
    evaluator = _ConcreteTestEvaluator(str(paths["instances"]), "segm")

    restored = evaluator._restore_boxes(
        np.array([[2, 2, 6, 6], [-5, -5, 20, 20]], dtype=np.float32),
        _labels()[0]["preprocess_context"],
    )

    np.testing.assert_allclose(
        restored,
        [[1, 0.5, 3, 2.5], [0, 0, 4, 3]],
    )


def test_validate_local_indices_rejects_fractional_or_out_of_batch_rows(
    tmp_path,
):
    paths = write_coco_fixture(tmp_path)
    evaluator = _ConcreteTestEvaluator(str(paths["instances"]), "segm")

    with pytest.raises(ValueError, match="integer local image indices"):
        evaluator._validate_local_indices(
            np.array([[0.5, 0, 0.9, 0, 0, 1, 1]], dtype=np.float32),
            batch_size=1,
        )
    with pytest.raises(ValueError, match="outside batch"):
        evaluator._validate_local_indices(
            np.array([[1, 0, 0.9, 0, 0, 1, 1]], dtype=np.float32),
            batch_size=1,
        )


def test_record_batch_normalizes_timing_and_reports_throughput(tmp_path):
    paths = write_coco_fixture(tmp_path)
    evaluator = _ConcreteTestEvaluator(str(paths["instances"]), "segm")
    normalized = evaluator._normalize_batch_labels(_labels())

    evaluator._record_batch(normalized, 3, {"total_ms": 4.0})

    assert evaluator._latency_metrics() == {
        "Total Samples": 1,
        "Average Detections": 3.0,
        "Average Latency (ms)": 4.0,
        "P99 Latency (ms)": 4.0,
        "FPS": 250.0,
    }


def test_evaluate_resets_streaming_state(tmp_path):
    paths = write_coco_fixture(tmp_path)
    evaluator = _ConcreteTestEvaluator(str(paths["instances"]), "segm")
    result = InferenceResult(
        outputs={}, timing_records=[2.0], labels=_labels()
    )

    first = evaluator.evaluate(result)
    second = evaluator.evaluate(result)

    assert first == second
    assert second["Total Samples"] == 1


def test_no_prediction_coco_eval_returns_zero_stats(tmp_path):
    paths = write_coco_fixture(tmp_path)
    evaluator = _ConcreteTestEvaluator(str(paths["instances"]), "segm")

    stats = evaluator._run_coco_eval([], [1])

    np.testing.assert_array_equal(stats, np.zeros(12, dtype=np.float64))
