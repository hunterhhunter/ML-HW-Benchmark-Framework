import json
import sys
import threading
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from core.async_inference.runner import AsyncBenchmarkRunner
from core.async_inference.types import AsyncInferenceConfig, RunStatus


class Loader:
    current_idx = 0

    def get_metadata(self):
        return {"total_samples": 1, "is_static_batched": False}

    def load_by_index(self, index):
        return {
            "input": np.asarray([1.0]),
            "label": np.asarray([2.0]),
        }


class Runtime:
    compiled_model = None

    def supports_generate(self):
        return False

    def max_concurrent_workers(self):
        return 1

    def supports_dynamic_batching(self):
        return True

    def max_dynamic_batch_size(self):
        return None

    def run(self, inputs):
        return {"output": inputs["input"] * 2}


class BlockingInt(int):
    @staticmethod
    def _block(*args, **kwargs):
        threading.Event().wait()

    __float__ = _block
    __int__ = _block
    __str__ = _block
    __repr__ = _block


class BlockingIterator:
    def __iter__(self):
        return self

    def __next__(self):
        threading.Event().wait()


class BlockingError(RuntimeError):
    @staticmethod
    def _block(*args, **kwargs):
        threading.Event().wait()

    __str__ = _block
    __repr__ = _block


class Evaluator:
    def __init__(self, mode):
        self.mode = mode
        self.total = 0

    def add_batch(self, outputs, labels, timing_ms):
        self.total += len(labels)

    def compute(self):
        if self.mode == "int":
            return {"Total Samples": BlockingInt(self.total)}
        if self.mode == "exception":
            raise BlockingError("blocked exception formatting")
        return {
            "Total Samples": self.total,
            "blocked": BlockingIterator(),
        }


def main():
    mode = sys.argv[1]
    result = AsyncBenchmarkRunner(
        Loader(),
        Runtime(),
        Evaluator(mode),
    ).run(
        AsyncInferenceConfig(
            batch_timeout_ms=0,
            min_samples=1,
            flush_timeout_sec=0.05,
        ),
        warmup_runs=0,
    )

    assert result.status is RunStatus.INVALID
    if mode == "exception":
        assert "request_failed" in result.invalid_reasons
        assert result.details["callback_errors"]
    else:
        assert "result_serialization_failed" in result.invalid_reasons
        assert result.details["serialization_errors"]
    assert result.metrics["async_completed_samples"] == 1
    payload = json.dumps(
        {"metrics": result.metrics, "details": result.details},
        allow_nan=False,
        sort_keys=True,
    )
    print(f"HOSTILE_RESULT={payload}")


if __name__ == "__main__":
    main()
