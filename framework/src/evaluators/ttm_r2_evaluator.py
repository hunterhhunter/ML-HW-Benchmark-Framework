"""Streaming raw-scale metrics for the canonical TTM-R2 workload."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from core.inference_result import InferenceResult
from core.model_spec import Model_Spec

from .base import Evaluator


class TTMR2Evaluator(Evaluator):
    """Restore TTM-R2 forecasts and accumulate raw ETTh1 errors."""

    def __init__(self, output_key: str = "forecast", **kwargs):
        del kwargs
        self._output_key = output_key
        self._reset()

    def _reset(self) -> None:
        self._absolute_error_sum = 0.0
        self._squared_error_sum = 0.0
        self._element_count = 0
        self._sample_count = 0
        self._timing_records: List[float] = []

    def add_batch(
        self,
        outputs: Dict[str, np.ndarray],
        labels: Any,
        timing_ms: float,
    ) -> None:
        prediction = self._validated_prediction(outputs)
        flat_labels = self._flatten_labels(labels)
        if prediction.shape[0] != len(flat_labels):
            raise ValueError(
                "TTMR2Evaluator batch size and label count must match: "
                f"{prediction.shape[0]} != {len(flat_labels)}"
            )
        self._accumulate(prediction, flat_labels)
        timing = float(timing_ms)
        if not np.isfinite(timing) or timing < 0.0:
            raise ValueError("TTMR2Evaluator timing_ms must be finite and nonnegative")
        self._timing_records.append(timing)

    def compute(self) -> Dict[str, Any]:
        if self._sample_count == 0 or self._element_count == 0:
            raise ValueError("TTMR2Evaluator has no samples to evaluate")

        mae = self._absolute_error_sum / self._element_count
        mse = self._squared_error_sum / self._element_count
        rmse = float(np.sqrt(mse))
        latencies = np.asarray(self._timing_records, dtype=np.float64)
        average_latency = float(np.mean(latencies)) if latencies.size else 0.0
        p99_latency = (
            float(np.percentile(latencies, 99)) if latencies.size else 0.0
        )
        total_seconds = float(np.sum(latencies)) / 1000.0
        samples_per_second = (
            self._sample_count / total_seconds if total_seconds > 0.0 else 0.0
        )
        return {
            "MAE": float(mae),
            "MSE": float(mse),
            "RMSE": rmse,
            "Average Latency (ms)": average_latency,
            "P99 Latency (ms)": p99_latency,
            "Samples/s": float(samples_per_second),
            "Total Samples": self._sample_count,
        }

    def evaluate(self, result: InferenceResult) -> Dict[str, Any]:
        self._reset()
        prediction = self._validated_prediction(result.outputs)
        flat_labels = self._flatten_labels(result.labels)
        if prediction.shape[0] != len(flat_labels):
            raise ValueError(
                "TTMR2Evaluator batch size and label count must match: "
                f"{prediction.shape[0]} != {len(flat_labels)}"
            )
        self._accumulate(prediction, flat_labels)
        self._timing_records = [float(value) for value in result.timing_records]
        return self.compute()

    def is_applicable(
        self, device_spec: Dict[str, Any], model_spec: Model_Spec
    ) -> bool:
        del device_spec
        return model_spec.name == "ttm-r2"

    def get_metric_names(self) -> List[str]:
        return [
            "MAE",
            "MSE",
            "RMSE",
            "Average Latency (ms)",
            "P99 Latency (ms)",
            "Samples/s",
            "Total Samples",
        ]

    def _validated_prediction(self, outputs: Dict[str, np.ndarray]) -> np.ndarray:
        if self._output_key not in outputs:
            raise ValueError(
                f"TTMR2Evaluator outputs must contain '{self._output_key}'"
            )
        prediction = np.asarray(outputs[self._output_key])
        if prediction.ndim != 3 or tuple(prediction.shape[1:]) != (96, 1):
            raise ValueError(
                "TTMR2Evaluator forecast shape must be [B,96,1], got "
                f"{tuple(prediction.shape)}"
            )
        if prediction.shape[0] < 1:
            raise ValueError("TTMR2Evaluator forecast batch must be nonempty")
        if prediction.dtype != np.float32:
            raise ValueError("TTMR2Evaluator forecast must use float32")
        if not bool(np.isfinite(prediction).all()):
            raise ValueError("TTMR2Evaluator forecast must be finite")
        return prediction

    @staticmethod
    def _flatten_labels(labels: Any) -> List[Dict[str, Any]]:
        if isinstance(labels, list):
            if labels and isinstance(labels[0], list):
                return [label for batch in labels for label in batch]
            return labels
        return [labels]

    def _accumulate(
        self,
        prediction: np.ndarray,
        labels: List[Dict[str, Any]],
    ) -> None:
        for index, label in enumerate(labels):
            required = {
                "future_values",
                "loc",
                "scale",
                "model_loc",
                "model_scale",
            }
            missing = required.difference(label)
            if missing:
                raise ValueError(
                    "TTMR2Evaluator label is missing restoration fields: "
                    f"{sorted(missing)}"
                )
            target = np.asarray(label["future_values"])
            if target.shape != (96, 1):
                raise ValueError(
                    "TTMR2Evaluator future_values shape must be [96,1], got "
                    f"{target.shape}"
                )
            if target.dtype != np.float32 or not bool(np.isfinite(target).all()):
                raise ValueError(
                    "TTMR2Evaluator future_values must be finite float32"
                )

            restoration = {}
            for name in ("loc", "scale", "model_loc", "model_scale"):
                value = np.asarray(label[name])
                if value.shape != (1, 1):
                    raise ValueError(
                        f"TTMR2Evaluator {name} shape must be [1,1], got "
                        f"{value.shape}"
                    )
                if value.dtype != np.float32 or not bool(np.isfinite(value).all()):
                    raise ValueError(
                        f"TTMR2Evaluator {name} must be finite float32"
                    )
                restoration[name] = value

            restored = (
                prediction[index] * restoration["model_scale"]
                + restoration["model_loc"]
            ) * restoration["scale"] + restoration["loc"]
            delta = restored - target
            self._absolute_error_sum += float(np.abs(delta).sum())
            self._squared_error_sum += float(np.square(delta).sum())
            self._element_count += int(delta.size)
            self._sample_count += 1
