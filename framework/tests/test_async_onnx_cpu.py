from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

from core.async_inference.runner import AsyncBenchmarkRunner
from core.async_inference.types import AsyncInferenceConfig, RunStatus
from core.benchmarkrunner import BenchmarkRunner
from core.compiled_model import CompiledModel
from core.model_spec import Model_Spec, Task
from runtimes.onnx_rt import OnnxRuntime


class TinySumLoader:
    def __init__(self):
        self.current_idx = 0
        self.samples = [
            {
                "input": np.array([1.0, 2.0], dtype=np.float32),
                "label": 3.0,
            },
            {
                "input": np.array([3.0, 4.0], dtype=np.float32),
                "label": 7.0,
            },
            {
                "input": np.array([5.0, 6.0], dtype=np.float32),
                "label": 11.0,
            },
            {
                "input": np.array([7.0, 8.0], dtype=np.float32),
                "label": 15.0,
            },
        ]

    def get_metadata(self):
        return {
            "total_samples": len(self.samples),
            "is_static_batched": False,
        }

    def load_by_index(self, index):
        return self.samples[index]

    def load_batch(self, batch_size):
        start = self.current_idx
        end = min(start + batch_size, len(self.samples))
        self.current_idx = end
        return self.samples[start:end]


class SumEvaluator:
    def __init__(self):
        self.correct = 0
        self.total = 0

    def add_batch(self, outputs, labels, timing_ms):
        del timing_ms
        predicted = outputs["output"].reshape(-1)
        expected = np.asarray(labels)
        self.correct += int(np.sum(predicted == expected))
        self.total += len(expected)

    def compute(self):
        return {
            "accuracy": self.correct / self.total,
            "Total Samples": self.total,
        }


def _create_sum_model(path: Path, *, batch_dimension=None) -> None:
    model_name = (
        "tiny-dynamic-batch-sum"
        if batch_dimension is None
        else "tiny-fixed-batch-sum"
    )
    input_info = helper.make_tensor_value_info(
        "input",
        TensorProto.FLOAT,
        [batch_dimension, 2],
    )
    output_info = helper.make_tensor_value_info(
        "output",
        TensorProto.FLOAT,
        [batch_dimension, 1],
    )
    node = helper.make_node(
        "ReduceSum",
        inputs=["input"],
        outputs=["output"],
        axes=[1],
        keepdims=1,
    )
    graph = helper.make_graph(
        [node],
        model_name,
        [input_info],
        [output_info],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 11)],
    )
    model.ir_version = 8
    onnx.save(model, path)


def _load_cpu_runtime(path: Path, *, batch_dimension=None) -> OnnxRuntime:
    model_name = (
        "tiny-dynamic-batch-sum"
        if batch_dimension is None
        else "tiny-fixed-batch-sum"
    )
    spec = Model_Spec(
        name=model_name,
        task=Task.IMAGE_CLASSIFICATION,
        input_shapes={"input": (batch_dimension, 2)},
        input_dtype={"input": "float32"},
        output_shapes={"output": (batch_dimension, 1)},
        model_paths={"onnx": str(path)},
    )
    compiled = CompiledModel(
        spec=spec,
        backend_name="onnxruntime",
        artifact_path=path,
    )
    runtime = OnnxRuntime(device="cpu")
    try:
        runtime.load(compiled)
    except BaseException:
        runtime.unload()
        raise
    return runtime


def test_loaded_onnx_device_spec_reports_actual_cpu_provider(tmp_path):
    model_path = tmp_path / "tiny-sum.onnx"
    _create_sum_model(model_path)
    runtime = _load_cpu_runtime(model_path)
    try:
        assert runtime.get_device_spec()["active_providers"] == [
            "CPUExecutionProvider"
        ]
    finally:
        runtime.unload()


def test_async_onnx_cpu_matches_e2e_and_uses_dynamic_batches(tmp_path):
    model_path = tmp_path / "tiny-sum.onnx"
    _create_sum_model(model_path)

    e2e_runtime = _load_cpu_runtime(model_path)
    try:
        e2e_result = BenchmarkRunner(
            TinySumLoader(),
            e2e_runtime,
            SumEvaluator(),
        ).run(warmup_runs=0, batch_size=2)
    finally:
        e2e_runtime.unload()

    async_runtime = _load_cpu_runtime(model_path)
    try:
        active_providers = async_runtime.session.get_providers()
        assert active_providers == ["CPUExecutionProvider"]
        async_result = AsyncBenchmarkRunner(
            TinySumLoader(),
            async_runtime,
            SumEvaluator(),
        ).run(
            AsyncInferenceConfig(
                queue_capacity=4,
                max_batch_size=2,
                batch_timeout_ms=10,
                min_samples=1,
            ),
            warmup_runs=0,
        )
    finally:
        async_runtime.unload()

    assert async_result.status is RunStatus.VALID
    assert async_result.metrics["accuracy"] == e2e_result["accuracy"] == 1.0
    assert async_result.metrics["Total Samples"] == (
        e2e_result["Total Samples"]
    ) == 4
    assert async_result.details["batch_size"]["max"] == 2.0
    assert async_result.metrics["async_outstanding_requests"] == 0


def test_fixed_batch_onnx_reports_limit_and_rejects_larger_dynamic_batch(
    tmp_path,
):
    model_path = tmp_path / "tiny-fixed-batch-sum.onnx"
    _create_sum_model(model_path, batch_dimension=1)
    runtime = _load_cpu_runtime(model_path, batch_dimension=1)

    try:
        assert runtime.compiled_model.spec.input_shapes["input"] == (1, 2)
        assert runtime.input_shapes["input"] == (1, 2)
        assert runtime.max_dynamic_batch_size() == 1

        with pytest.raises(
            ValueError,
            match=(
                "max_batch_size=2 exceeds runtime capability 1"
            ),
        ):
            AsyncBenchmarkRunner(
                TinySumLoader(),
                runtime,
                SumEvaluator(),
            ).run(
                AsyncInferenceConfig(
                    queue_capacity=2,
                    max_batch_size=2,
                    min_samples=1,
                ),
                warmup_runs=0,
            )
    finally:
        runtime.unload()
