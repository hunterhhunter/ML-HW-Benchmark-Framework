import gc
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


class BlockingDestructor:
    def __del__(self):
        print(
            f"DESTRUCTOR_THREAD={threading.current_thread().name}",
            flush=True,
        )
        threading.Event().wait()


class BlockingDestructorError(RuntimeError):
    def __del__(self):
        print(
            f"DESTRUCTOR_THREAD={threading.current_thread().name}",
            flush=True,
        )
        threading.Event().wait()


class CyclicBlockingDestructor:
    def __init__(self):
        self.cycle = self

    def __del__(self):
        print(
            f"CYCLIC_DESTRUCTOR_THREAD={threading.current_thread().name}",
            flush=True,
        )
        threading.Event().wait()


class CyclicBlockingDestructorError(RuntimeError):
    def __init__(self, message):
        super().__init__(message)
        self.cycle = self

    def __del__(self):
        print(
            f"CYCLIC_DESTRUCTOR_THREAD={threading.current_thread().name}",
            flush=True,
        )
        threading.Event().wait()


class Evaluator:
    def __init__(self, mode):
        self.mode = mode
        self.total = 0

    def add_batch(self, outputs, labels, timing_ms):
        self.total += len(labels)

    def compute(self):
        if self.mode.startswith("monitor_"):
            return {"Total Samples": self.total}
        if self.mode == "evaluator_cycle_del":
            return {
                "Total Samples": self.total,
                "blocked": CyclicBlockingDestructor(),
            }
        if self.mode == "exception_cycle_del":
            raise CyclicBlockingDestructorError(
                "blocked cyclic exception destruction"
            )
        if self.mode == "evaluator_del":
            return {
                "Total Samples": self.total,
                "blocked": BlockingDestructor(),
            }
        if self.mode == "exception_del":
            raise BlockingDestructorError("blocked exception destruction")
        if self.mode == "int":
            return {"Total Samples": BlockingInt(self.total)}
        if self.mode == "exception":
            raise BlockingError("blocked exception formatting")
        return {
            "Total Samples": self.total,
            "blocked": BlockingIterator(),
        }


class Monitor:
    def __init__(self, mode):
        self.mode = mode

    def start(self):
        return None

    def stop(self):
        return None

    def summary(self):
        if self.mode == "monitor_exception_cycle_del":
            raise CyclicBlockingDestructorError(
                "blocked cyclic monitor exception destruction"
            )
        if self.mode == "monitor_cycle_del":
            return {
                "hw_blocked": CyclicBlockingDestructor(),
            }
        if self.mode == "monitor_exception_del":
            raise BlockingDestructorError(
                "blocked monitor exception destruction"
            )
        return {
            "hw_blocked": BlockingDestructor(),
        }


def main():
    mode = sys.argv[1]
    cyclic_modes = {
        "evaluator_cycle_del",
        "exception_cycle_del",
        "monitor_cycle_del",
        "monitor_exception_cycle_del",
    }
    if mode in cyclic_modes:
        gc.disable()
    result = AsyncBenchmarkRunner(
        Loader(),
        Runtime(),
        Evaluator(mode),
        monitor=(
            Monitor(mode)
            if mode
            in {
                "monitor_del",
                "monitor_exception_del",
                "monitor_cycle_del",
                "monitor_exception_cycle_del",
            }
            else None
        ),
    ).run(
        AsyncInferenceConfig(
            batch_timeout_ms=0,
            min_samples=1,
            flush_timeout_sec=0.05,
        ),
        warmup_runs=0,
    )

    payload = json.dumps(
        {"metrics": result.metrics, "details": result.details},
        allow_nan=False,
        sort_keys=True,
    )
    print(f"HOSTILE_RESULT={payload}", flush=True)

    if mode in cyclic_modes:
        assert not gc.isenabled()
        print("MAIN_COLLECT_START", flush=True)
        gc.collect()
        print("MAIN_COLLECT_DONE", flush=True)
        assert result.status is RunStatus.INVALID
        assert "callback_timeout" in result.invalid_reasons
        timeout = next(
            item
            for item in result.details["callback_errors"]
            if item["error_type"] == "TimeoutError"
        )
        assert timeout["callback_alive"] is True
        assert timeout["callback_state"] == "collecting"
        assert timeout["callback_value_ready"] is True
        assert timeout["callback_disposal_finished"] is False
        assert result.details["outstanding_callbacks"]
        return

    assert result.status is RunStatus.INVALID
    if mode in {"exception", "exception_del"}:
        assert "request_failed" in result.invalid_reasons
        assert result.details["callback_errors"]
    elif mode in {
        "evaluator_del",
        "monitor_del",
        "monitor_exception_del",
    }:
        assert "callback_timeout" in result.invalid_reasons
        assert result.details["callback_errors"]
        assert result.details["outstanding_callbacks"]
        timeout = next(
            item
            for item in result.details["callback_errors"]
            if item["error_type"] == "TimeoutError"
        )
        assert timeout["callback_alive"] is True
        assert timeout["callback_state"] == "disposing"
        assert timeout["callback_value_ready"] is True
        assert timeout["callback_disposal_finished"] is False
    else:
        assert "result_serialization_failed" in result.invalid_reasons
        assert result.details["serialization_errors"]
    assert result.metrics["async_completed_samples"] == 1


if __name__ == "__main__":
    main()
