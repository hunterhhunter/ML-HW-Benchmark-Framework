from pathlib import Path
from types import SimpleNamespace

import numpy as np

from core.benchmarkrunner import BenchmarkRunner
from core.inference_pipeline import InferencePipeline
from core.model_spec import Model_Spec, Task


class FakeLoader:
    def __init__(self):
        self.current_idx = 0
        self.samples = [
            {"input": np.array([1.0, 2.0], dtype=np.float32), "label": 3},
            {"input": np.array([3.0, 4.0], dtype=np.float32), "label": 7},
        ]

    def get_metadata(self):
        return {"is_static_batched": False, "total_samples": len(self.samples)}

    def load_batch(self, batch_size):
        batch = self.samples[self.current_idx:self.current_idx + batch_size]
        self.current_idx += len(batch)
        return batch

    def load_by_index(self, index):
        return self.samples[index]


class FakeRuntime:
    def __init__(self):
        spec = Model_Spec(
            name="sum",
            task=Task.IMAGE_CLASSIFICATION,
            input_shapes={"input": (None, 2)},
            input_dtype={"input": "float32"},
            output_shapes={"output": (None, 1)},
            model_paths={"onnx": Path("sum.onnx")},
        )
        self.compiled_model = SimpleNamespace(spec=spec)
        self.warmup_calls = 0

    def supports_generate(self):
        return False

    def run(self, inputs):
        values = inputs["input"]
        return {"output": values.sum(axis=1, keepdims=True)}

    def warmup(self, inputs, num_runs=1):
        self.warmup_calls += num_runs


class FakeEvaluator:
    def __init__(self):
        self.rows = []

    def add_batch(self, outputs, labels, timing_ms):
        self.rows.extend(zip(outputs["output"].reshape(-1).tolist(), labels))

    def compute(self):
        return {"pairs": list(self.rows), "Total Samples": len(self.rows)}


def test_pipeline_collates_inputs_and_preserves_labels():
    loader = FakeLoader()
    pipeline = InferencePipeline(loader, FakeRuntime())

    collated = pipeline.collate_batch(loader.samples)

    np.testing.assert_array_equal(
        collated["input"],
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
    )
    assert collated["label"] == [3, 7]


def test_pipeline_preserves_preprocess_context_in_eval_labels():
    loader = FakeLoader()
    pipeline = InferencePipeline(loader, FakeRuntime())
    collated = {
        "input": np.zeros((2, 2), dtype=np.float32),
        "label": [1, 2],
        "preprocess_context": [{"scale": 1.0}, {"scale": 2.0}],
    }

    assert pipeline.prepare_eval_labels(collated) == [
        {"label": 1, "preprocess_context": {"scale": 1.0}},
        {"label": 2, "preprocess_context": {"scale": 2.0}},
    ]


def test_benchmark_runner_keeps_existing_result_contract():
    loader = FakeLoader()
    runtime = FakeRuntime()
    evaluator = FakeEvaluator()

    result = BenchmarkRunner(loader, runtime, evaluator).run(
        warmup_runs=1,
        batch_size=2,
    )

    assert runtime.warmup_calls == 1
    assert result == {
        "pairs": [(3.0, 3), (7.0, 7)],
        "Total Samples": 2,
    }
