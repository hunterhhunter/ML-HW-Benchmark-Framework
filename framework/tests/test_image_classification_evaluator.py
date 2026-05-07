import os
import sys
import types
from importlib.machinery import ModuleSpec

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
EVALUATORS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src', 'evaluators'))

import numpy as np
import pytest

# CI/minimal dev shells may not have sklearn installed; this test only needs
# the evaluator's latency/throughput path, so provide a tiny metrics stub.
try:
    from sklearn.metrics import precision_recall_fscore_support  # noqa: F401
except ModuleNotFoundError:
    sklearn_stub = types.ModuleType("sklearn")
    metrics_stub = types.ModuleType("sklearn.metrics")
    sklearn_stub.__spec__ = ModuleSpec("sklearn", loader=None, is_package=True)
    metrics_stub.__spec__ = ModuleSpec("sklearn.metrics", loader=None, is_package=False)

    def precision_recall_fscore_support(*args, **kwargs):
        return 0.0, 0.0, 0.0, None

    metrics_stub.precision_recall_fscore_support = precision_recall_fscore_support
    sklearn_stub.metrics = metrics_stub
    sys.modules["sklearn"] = sklearn_stub
    sys.modules["sklearn.metrics"] = metrics_stub

if "evaluators" not in sys.modules:
    evaluators_stub = types.ModuleType("evaluators")
    evaluators_stub.__path__ = [EVALUATORS_DIR]
    evaluators_stub.__spec__ = ModuleSpec("evaluators", loader=None, is_package=True)
    sys.modules["evaluators"] = evaluators_stub

from evaluators.image_classification_evaluator import ImageClassificationEvaluator


def test_image_classification_reports_samples_per_second():
    evaluator = ImageClassificationEvaluator(top_k=(1, 2))

    evaluator.add_batch(
        outputs={"logits": np.array([[4.0, 1.0, 0.0], [0.1, 3.0, 0.2]])},
        labels=np.array([0, 1]),
        timing_ms=10.0,
    )
    evaluator.add_batch(
        outputs={"logits": np.array([[0.2, 0.5, 4.0], [2.0, 1.0, 0.0]])},
        labels=np.array([2, 0]),
        timing_ms=30.0,
    )

    metrics = evaluator.compute()

    assert metrics["Total Samples"] == 4
    assert metrics["Average Latency (ms)"] == pytest.approx(20.0)
    assert metrics["Samples/s"] == pytest.approx(100.0)


def test_image_classification_metric_names_include_throughput():
    evaluator = ImageClassificationEvaluator(top_k=(1, 5))

    assert "Samples/s" in evaluator.get_metric_names()
