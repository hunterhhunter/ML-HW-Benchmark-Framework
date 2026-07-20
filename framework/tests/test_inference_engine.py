import threading

import numpy as np

from core.inference_engine import InferenceEngine


class FakeLoader:
    def __init__(self):
        self.current_idx = 0
        self.samples = [
            {"input": np.array([1.0, 2.0], dtype=np.float32), "label": 3},
            {"input": np.array([3.0, 4.0], dtype=np.float32), "label": 7},
        ]

    def get_metadata(self):
        return {
            "is_static_batched": False,
            "total_samples": len(self.samples),
        }

    def load_batch(self, batch_size):
        batch = self.samples[self.current_idx:self.current_idx + batch_size]
        self.current_idx += len(batch)
        return batch

    def load_by_index(self, index):
        return self.samples[index]


class FakeRuntime:
    compiled_model = None

    def __init__(self):
        self.warmup_calls = 0

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
        values = inputs["input"]
        return {"output": values.sum(axis=1, keepdims=True)}

    def warmup(self, inputs, num_runs=1):
        del inputs
        self.warmup_calls += num_runs


class FakeEvaluator:
    def __init__(self):
        self.rows = []

    def add_batch(self, outputs, labels, timing_ms):
        del timing_ms
        predictions = outputs["output"].reshape(-1).tolist()
        self.rows.extend(zip(predictions, labels))

    def compute(self):
        return {
            "pairs": list(self.rows),
            "Total Samples": len(self.rows),
        }


class FailingEvaluator(FakeEvaluator):
    def __init__(self, primary):
        super().__init__()
        self.primary = primary

    def add_batch(self, outputs, labels, timing_ms):
        del outputs, labels, timing_ms
        raise self.primary


class RecordingMonitor:
    def __init__(self, events, summary):
        self.events = events
        self._summary = summary

    def start(self):
        self.events.append("start")

    def stop(self):
        self.events.append("stop")

    def summary(self):
        return dict(self._summary)


def test_e2e_engine_uses_inline_completion_and_no_async_threads():
    before = {thread.ident for thread in threading.enumerate()}
    engine = InferenceEngine(FakeLoader(), FakeRuntime(), FakeEvaluator())
    result = engine.run_e2e(batch_size=2)
    assert result == {"pairs": [(3.0, 3), (7.0, 7)], "Total Samples": 2}
    assert engine.completion.queue is None
    assert engine.completion.thread is None
    created = [t for t in threading.enumerate() if t.ident not in before]
    assert not [t for t in created if t.name.startswith("async-")]


def test_e2e_engine_warmup_resets_loader_before_measurement():
    loader, runtime, evaluator = FakeLoader(), FakeRuntime(), FakeEvaluator()
    engine = InferenceEngine(loader, runtime, evaluator)
    engine.warmup(runs=1, batch_size=1)
    result = engine.run_e2e(batch_size=2)
    assert runtime.warmup_calls == 1
    assert result["Total Samples"] == 2
