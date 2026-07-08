import numpy as np
from typing import Any, Dict, List

from .base import Evaluator
from core.inference_result import InferenceResult
from core.model_spec import Model_Spec


class LatencyOnlyEvaluator(Evaluator):
    """Evaluator for tasks whose accuracy postprocess is not wired yet."""

    def __init__(self, **eval_options):
        self.task_name = eval_options.get("task_name", "UNKNOWN")
        self._timing_records: List[float] = []
        self._total_samples = 0
        self._output_shapes: Dict[str, tuple] = {}

    def add_batch(self, outputs: Dict[str, np.ndarray], labels: Any, timing_ms: float) -> None:
        if isinstance(timing_ms, dict):
            timing_value = float(timing_ms.get("total_ms", 0.0))
        else:
            timing_value = float(timing_ms)
        self._timing_records.append(timing_value)
        self._total_samples += self._infer_batch_size(outputs, labels)
        for name, value in outputs.items():
            self._output_shapes[str(name)] = tuple(np.asarray(value).shape)

    def compute(self) -> Dict[str, Any]:
        metrics: Dict[str, Any] = {
            "Task": self.task_name,
            "Total Samples": self._total_samples,
            "Accuracy Metric": "not_configured",
            "Output Shapes": dict(self._output_shapes),
        }
        metrics.update(self._latency_metrics())
        return metrics

    def evaluate(self, result: InferenceResult) -> Dict[str, Any]:
        self._timing_records = []
        self._total_samples = 0
        self._output_shapes = {}
        timing = float(np.mean(result.timing_records)) if result.timing_records else 0.0
        self.add_batch(result.outputs, result.labels, timing)
        return self.compute()

    def is_applicable(self, device_spec: Dict[str, Any], model_spec: Model_Spec) -> bool:
        return True

    def get_metric_names(self) -> List[str]:
        return [
            "Task",
            "Total Samples",
            "Accuracy Metric",
            "Average Latency (ms)",
            "P99 Latency (ms)",
            "FPS",
            "Output Shapes",
        ]

    def _infer_batch_size(self, outputs: Dict[str, np.ndarray], labels: Any) -> int:
        if isinstance(labels, list):
            return len(labels)
        if outputs:
            first = np.asarray(next(iter(outputs.values())))
            if first.ndim > 0:
                return int(first.shape[0])
        return 1

    def _latency_metrics(self) -> Dict[str, float]:
        if not self._timing_records:
            return {}
        avg_latency = float(np.mean(self._timing_records))
        p99_latency = float(np.percentile(self._timing_records, 99))
        fps = 1000.0 / avg_latency if avg_latency > 0 else 0.0
        return {
            "Average Latency (ms)": avg_latency,
            "P99 Latency (ms)": p99_latency,
            "FPS": fps,
        }
