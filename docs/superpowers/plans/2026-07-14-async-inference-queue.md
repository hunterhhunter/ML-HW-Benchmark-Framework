# Async Inference Queue Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve the existing sequential e2e path while adding a trustworthy, bounded, instrumented async inference queue with Offline and Server-like producers, CLI selection, and durable result artifacts.

**Architecture:** The current `BenchmarkRunner` continues to own the default e2e path. A new `AsyncBenchmarkRunner` composes a bounded `AsyncInferenceEngine`, a single serialized completion coordinator, deterministic workload producers, and a metrics collector; MLPerf LoadGen remains a design reference only and is neither imported nor reimplemented.

**Tech Stack:** Python 3.10+, `threading`, `queue.Queue`, dataclasses, NumPy, ONNX/ONNX Runtime CPU, pytest, CSV/JSON/JSONL.

**Approved Specification:** [`../specs/2026-07-14-async-inference-queue-design.md`](../specs/2026-07-14-async-inference-queue-design.md), including the LoadGen metric inventory, applicability decisions, expected gains, and risk mitigations.

**Acceptance note (2026-07-15):** The inline code in this plan records the
initial TDD increments, not the final trust boundary. Review hardening replaced
the Task 8 overwrite-style artifact examples with reservation-bound,
no-overwrite publication and bounded cleanup, as specified in section 41 of the
approved design. The implemented module remains framework-owned: it does not
import, embed, reimplement, or claim API/log compatibility with MLPerf LoadGen.

## Global Constraints

- Scope is limited to `framework` core, CLI, result storage, tests, and framework documentation.
- Do not modify the FastAPI backend or React frontend.
- Do not add `mlperf_loadgen`, SUT/QSL compatibility, MLPerf log compatibility, submission, audit, or compliance code.
- `--inference-mode` defaults to `e2e`, and every existing CLI command must keep its current behavior.
- CI acceptance uses Mock Runtime and ONNX Runtime CPU with one worker.
- First-token lifecycle is a mock-tested extension contract only; wiring a real streaming vLLM engine is a follow-up and estimated TTFT must never be relabeled as an event measurement.
- `worker_count` defaults to 1; values above 1 require an explicit runtime capability.
- Async `batch_size` defaults to 1; dynamic batching is opt-in by choosing a larger existing `--batch-size`.
- New latency fields use the `async_` prefix and never replace the existing runtime-only evaluator latency fields.
- Every accepted request reaches exactly one terminal state, and normal flush ends with zero outstanding requests.
- Tests must not assert that async mode is faster; they assert correctness, boundedness, lifecycle safety, and metric invariants.
- Preserve unrelated dirty-worktree changes and stage only files listed by the current task.

---

### Task 1: Extract the shared inference pipeline without changing e2e behavior

**Files:**
- Create: `framework/src/core/inference_pipeline.py`
- Create: `framework/tests/test_inference_pipeline.py`
- Modify: `framework/src/core/benchmarkrunner.py:1-178`

**Interfaces:**
- Consumes: existing `DataLoader.get_metadata()`, `Runtime.run()`, optional `Runtime.generate()`, and `GenerationResult`.
- Produces: `InferencePipeline.collate_batch(batch)`, `prepare_runtime_input(collated_input)`, `prepare_eval_labels(collated)`, and `invoke(runtime_input) -> RuntimeInvocation`.

- [ ] **Step 1: Write failing pipeline and e2e-regression tests**

Create `framework/tests/test_inference_pipeline.py`:

```python
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
```

- [ ] **Step 2: Run the tests and verify the missing module failure**

Run:

```bash
cd framework
python -m pytest tests/test_inference_pipeline.py -q
```

Expected: collection fails with `ModuleNotFoundError: No module named 'core.inference_pipeline'`.

- [ ] **Step 3: Implement the shared pipeline**

Create `framework/src/core/inference_pipeline.py`:

```python
import time
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from .model_spec import Task


@dataclass(frozen=True)
class RuntimeInvocation:
    outputs: Dict[str, Any]
    timing_ms: float | Dict[str, Any]
    generated_tokens: int = 0


class InferencePipeline:
    def __init__(self, dataloader, runtime, max_new_tokens: int = 256):
        self.dataloader = dataloader
        self.runtime = runtime
        self.max_new_tokens = max_new_tokens
        metadata = dataloader.get_metadata()
        self.is_static_batched = bool(metadata.get("is_static_batched", False))
        self.stop_token_ids = metadata.get("stop_token_ids")

        self.input_name = "input"
        compiled_model = getattr(runtime, "compiled_model", None)
        if compiled_model is not None:
            self.input_name = next(iter(compiled_model.spec.input_shapes))

        spec = getattr(compiled_model, "spec", None)
        self.is_llm = bool(
            spec is not None
            and spec.task == Task.NLP_GENERATION
            and runtime.supports_generate()
        )

    def collate_batch(self, batch_list: Any) -> Dict[str, Any]:
        if self.is_static_batched:
            return batch_list
        collated: Dict[str, Any] = {}
        for key in batch_list[0]:
            if key != "input":
                collated[key] = [item[key] for item in batch_list]
                continue
            first_input = batch_list[0][key]
            if isinstance(first_input, dict):
                collated[key] = {
                    name: np.stack(
                        [item[key][name] for item in batch_list],
                        axis=0,
                    )
                    for name in first_input
                }
            else:
                collated[key] = np.stack(
                    [item[key] for item in batch_list],
                    axis=0,
                )
        return collated

    def prepare_runtime_input(self, collated_input: Any) -> Dict[str, Any]:
        if isinstance(collated_input, dict):
            return collated_input
        return {self.input_name: collated_input}

    def prepare_eval_labels(self, collated: Dict[str, Any]) -> Any:
        labels = collated["label"]
        contexts = collated.get("preprocess_context")
        if not isinstance(labels, list) or not isinstance(contexts, list):
            return labels
        if len(labels) != len(contexts):
            return labels
        return [
            {"label": label, "preprocess_context": context}
            for label, context in zip(labels, contexts)
        ]

    def invoke(self, runtime_input: Dict[str, Any]) -> RuntimeInvocation:
        if self.is_llm:
            result = self.runtime.generate(
                runtime_input,
                max_new_tokens=self.max_new_tokens,
                stop_token_ids=self.stop_token_ids,
            )
            outputs = {"generated_ids": result.generated_ids}
            if result.generated_lengths is not None:
                outputs["generated_lengths"] = result.generated_lengths
            timing = {
                "total_ms": result.total_ms,
                "ttft_ms": result.ttft_ms,
                "tpot_ms": result.tpot_ms,
                "timing_mode": result.timing_mode,
                "uses_kv_cache": result.uses_kv_cache,
                "timing_source": result.timing_source,
            }
            return RuntimeInvocation(
                outputs=outputs,
                timing_ms=timing,
                generated_tokens=result.num_tokens,
            )

        started = time.perf_counter()
        outputs = self.runtime.run(runtime_input)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return RuntimeInvocation(outputs=outputs, timing_ms=elapsed_ms)
```

- [ ] **Step 4: Refactor `BenchmarkRunner` to delegate without changing its API**

In `framework/src/core/benchmarkrunner.py`:

```python
from .inference_pipeline import InferencePipeline
```

At the end of `__init__` add:

```python
        self._pipeline = InferencePipeline(
            dataloader=dataloader,
            runtime=runtime,
            max_new_tokens=max_new_tokens,
        )
```

Replace the three private helper bodies with compatibility delegates:

```python
    def _collate_batch(self, batch_list: Any) -> Dict[str, Any]:
        return self._pipeline.collate_batch(batch_list)

    def _prepare_runtime_input(
        self,
        collated_input: Any,
        fallback_name: str,
    ) -> Dict[str, Any]:
        del fallback_name
        return self._pipeline.prepare_runtime_input(collated_input)

    def _prepare_eval_labels(self, collated: Dict[str, Any]) -> Any:
        return self._pipeline.prepare_eval_labels(collated)
```

Replace the LLM detection and invocation branches in `run()` with:

```python
        is_llm = self._pipeline.is_llm
```

and:

```python
            invocation = self._pipeline.invoke(runtime_input)
            outputs = invocation.outputs
            latency_ms = invocation.timing_ms
```

- [ ] **Step 5: Verify e2e behavior and focused regression tests**

Run:

```bash
cd framework
python -m pytest tests/test_inference_pipeline.py tests/test_factory_api.py tests/test_main_paths.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit the extraction**

```bash
git add framework/src/core/inference_pipeline.py framework/src/core/benchmarkrunner.py framework/tests/test_inference_pipeline.py
git commit -m "refactor(framework): share inference pipeline"
```

### Task 2: Define async contracts, configuration, and runtime capabilities

**Files:**
- Create: `framework/src/core/async_inference/__init__.py`
- Create: `framework/src/core/async_inference/types.py`
- Create: `framework/tests/test_async_types.py`
- Modify: `framework/src/runtimes/base.py:54-61`
- Modify: `framework/src/runtimes/onnx_rt.py` (declare CPU-tested dynamic batch capability)

**Interfaces:**
- Consumes: `Runtime` and normalized pipeline outputs from Task 1.
- Produces: `AsyncInferenceConfig`, `InferenceRequest`, `BatchCompletion`, `FirstTokenEvent`, `RequestTrace`, `AsyncBenchmarkResult`, and conservative runtime capability methods.

- [ ] **Step 1: Write configuration and capability tests**

Create `framework/tests/test_async_types.py`:

```python
from dataclasses import replace

import numpy as np
import pytest

from core.async_inference.types import (
    AsyncInferenceConfig,
    AsyncScenario,
    FirstTokenEvent,
    InferenceRequest,
)
from runtimes.base import Runtime


def test_default_config_is_bounded_and_single_worker():
    config = AsyncInferenceConfig()
    config.validate()
    assert config.scenario is AsyncScenario.OFFLINE
    assert config.queue_capacity == 256
    assert config.worker_count == 1
    assert config.max_batch_size == 1


def test_server_like_requires_positive_target_qps():
    with pytest.raises(ValueError, match="target_qps"):
        AsyncInferenceConfig(scenario=AsyncScenario.SERVER_LIKE).validate()


def test_queue_capacity_must_cover_max_batch_size():
    with pytest.raises(ValueError, match="queue_capacity"):
        AsyncInferenceConfig(
            queue_capacity=2,
            max_batch_size=4,
        ).validate()


def test_max_samples_may_intentionally_end_before_minimum_for_invalid_run():
    AsyncInferenceConfig(min_samples=100, max_samples=10).validate()


def test_request_is_immutable():
    request = InferenceRequest(
        request_id=0,
        sample_index=3,
        sample={"input": np.array([1.0]), "label": 1},
        scheduled_ns=10,
        issued_ns=20,
        enqueued_ns=0,
    )
    queued = replace(request, enqueued_ns=30)
    assert request.enqueued_ns == 0
    assert queued.enqueued_ns == 30


def test_runtime_capabilities_are_conservative_by_default():
    assert Runtime.max_concurrent_workers(None) == 1
    assert Runtime.supports_dynamic_batching(None) is False
    assert Runtime.max_dynamic_batch_size(None) == 1
    assert Runtime.supports_batch_generation(None) is False
    assert Runtime.supports_streaming_generate(None) is False
    assert callable(Runtime.generate_stream)


def test_first_token_event_is_an_explicit_runtime_contract():
    event = FirstTokenEvent(
        request_id=7,
        first_token_ns=123,
        token_count=1,
    )
    assert event.request_id == 7
    assert event.first_token_ns == 123
    assert event.token_count == 1
```

- [ ] **Step 2: Run the tests and verify the missing package failure**

Run:

```bash
cd framework
python -m pytest tests/test_async_types.py -q
```

Expected: collection fails because `core.async_inference` does not exist.

- [ ] **Step 3: Implement the async data contracts**

Create `framework/src/core/async_inference/types.py`:

```python
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Sequence


class AsyncScenario(str, Enum):
    OFFLINE = "offline"
    SERVER_LIKE = "server_like"


class EngineState(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    DRAINING = "draining"
    FAILED = "failed"
    STOPPED = "stopped"


class TerminalStatus(str, Enum):
    COMPLETED = "completed"
    FAILED = "failed"
    REJECTED = "rejected"


class RunStatus(str, Enum):
    VALID = "valid"
    INVALID = "invalid"


@dataclass(frozen=True)
class AsyncInferenceConfig:
    scenario: AsyncScenario = AsyncScenario.OFFLINE
    queue_capacity: int = 256
    worker_count: int = 1
    max_batch_size: int = 1
    batch_timeout_ms: float = 1.0
    submit_timeout_sec: float = 30.0
    flush_timeout_sec: float = 300.0
    request_timeout_ms: float = 0.0
    min_samples: int = 100
    min_duration_sec: float = 0.0
    max_samples: Optional[int] = None
    target_qps: Optional[float] = None
    schedule_seed: int = 0
    latency_slo_ms: Optional[float] = None

    def validate(self) -> None:
        if self.queue_capacity < 1:
            raise ValueError("queue_capacity must be >= 1")
        if self.worker_count < 1:
            raise ValueError("worker_count must be >= 1")
        if self.max_batch_size < 1:
            raise ValueError("max_batch_size must be >= 1")
        if self.queue_capacity < self.max_batch_size:
            raise ValueError("queue_capacity must be >= max_batch_size")
        if self.batch_timeout_ms < 0:
            raise ValueError("batch_timeout_ms must be >= 0")
        if self.submit_timeout_sec <= 0 or self.flush_timeout_sec <= 0:
            raise ValueError("submit_timeout_sec and flush_timeout_sec must be > 0")
        if self.request_timeout_ms < 0:
            raise ValueError("request_timeout_ms must be >= 0")
        if self.min_samples < 1 or self.min_duration_sec < 0:
            raise ValueError("minimum run constraints are invalid")
        if self.max_samples is not None and self.max_samples < 1:
            raise ValueError("max_samples must be >= 1")
        if self.scenario is AsyncScenario.SERVER_LIKE:
            if self.target_qps is None or self.target_qps <= 0:
                raise ValueError("server_like requires target_qps > 0")
        elif self.target_qps is not None:
            raise ValueError("target_qps is only valid for server_like")
        if self.latency_slo_ms is not None and self.latency_slo_ms <= 0:
            raise ValueError("latency_slo_ms must be > 0")


@dataclass(frozen=True)
class InferenceRequest:
    request_id: int
    sample_index: int
    sample: Dict[str, Any]
    scheduled_ns: int
    issued_ns: int
    enqueued_ns: int
    sample_count: int = 1


@dataclass(frozen=True)
class FirstTokenEvent:
    request_id: int
    first_token_ns: int
    token_count: int = 1


@dataclass(frozen=True)
class BatchCompletion:
    requests: Sequence[InferenceRequest]
    collated: Dict[str, Any]
    outputs: Optional[Dict[str, Any]]
    timing_ms: float | Dict[str, Any] | None
    runtime_started_ns: int
    runtime_finished_ns: int
    worker_id: int
    batch_size: int
    generated_tokens: int = 0
    error_type: Optional[str] = None
    error_message: Optional[str] = None


@dataclass(frozen=True)
class RequestTrace:
    request_id: int
    sample_index: int
    status: TerminalStatus
    scheduled_ns: int
    issued_ns: int
    enqueued_ns: int
    runtime_started_ns: int
    runtime_finished_ns: int
    completed_ns: int
    worker_id: int
    batch_size: int
    timed_out: bool
    sample_count: int = 1
    error_type: Optional[str] = None
    error_message: Optional[str] = None


@dataclass(frozen=True)
class AsyncBenchmarkResult:
    metrics: Dict[str, Any]
    details: Dict[str, Any]
    status: RunStatus
    invalid_reasons: Sequence[str] = field(default_factory=tuple)
    warnings: Sequence[str] = field(default_factory=tuple)
```

Create `framework/src/core/async_inference/__init__.py`:

```python
from .types import (
    AsyncBenchmarkResult,
    AsyncInferenceConfig,
    AsyncScenario,
    BatchCompletion,
    EngineState,
    FirstTokenEvent,
    InferenceRequest,
    RequestTrace,
    RunStatus,
    TerminalStatus,
)

__all__ = [
    "AsyncBenchmarkResult",
    "AsyncInferenceConfig",
    "AsyncScenario",
    "BatchCompletion",
    "EngineState",
    "FirstTokenEvent",
    "InferenceRequest",
    "RequestTrace",
    "RunStatus",
    "TerminalStatus",
]
```

- [ ] **Step 4: Add conservative capability methods to `Runtime`**

Append to `framework/src/runtimes/base.py`:

```python
    def max_concurrent_workers(self) -> int:
        """Safe number of simultaneous run/generate calls for this instance."""
        return 1

    def supports_dynamic_batching(self) -> bool:
        """Whether run() accepts a batch assembled from independent requests."""
        return False

    def max_dynamic_batch_size(self):
        """Maximum assembled batch size, or None for a model-dynamic axis."""
        return 1

    def supports_batch_generation(self) -> bool:
        """Whether generate() accepts more than one independent request."""
        return False

    def supports_streaming_generate(self) -> bool:
        """Whether the runtime can emit a real first-token event."""
        return False

    def generate_stream(
        self,
        inputs,
        on_first_token,
        max_new_tokens=256,
        stop_token_ids=None,
    ):
        """Future streaming contract; unsupported runtimes never enter this path."""
        raise NotImplementedError(
            "this runtime does not support streaming generation"
        )
```

Add the explicitly verified capability to `framework/src/runtimes/onnx_rt.py`:

```python
    def supports_dynamic_batching(self) -> bool:
        return True

    def max_dynamic_batch_size(self):
        if not self.input_shapes:
            return 1
        fixed_batch_dims = [
            shape[0]
            for shape in self.input_shapes.values()
            if shape and isinstance(shape[0], int) and shape[0] > 0
        ]
        return min(fixed_batch_dims) if fixed_batch_dims else None
```

- [ ] **Step 5: Run the focused tests**

Run:

```bash
cd framework
python -m pytest tests/test_async_types.py tests/test_factory_api.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit the contracts**

```bash
git add framework/src/core/async_inference framework/src/runtimes/base.py framework/src/runtimes/onnx_rt.py framework/tests/test_async_types.py
git commit -m "feat(framework): define async inference contracts"
```

### Task 3: Implement compact timing, queue, worker, and validity metrics

**Files:**
- Create: `framework/src/core/async_inference/metrics.py`
- Create: `framework/tests/test_async_metrics.py`

**Interfaces:**
- Consumes: `RequestTrace` from Task 2 and explicit monotonic timestamps.
- Produces: thread-safe `AsyncMetricsCollector` with `finalize(end_ns) -> Dict[str, Dict[str, Any]]` containing exact `summary` and `details` keys.

- [ ] **Step 1: Write deterministic metric tests with explicit nanosecond timestamps**

Create `framework/tests/test_async_metrics.py`:

```python
import pytest

from core.async_inference.metrics import AsyncMetricsCollector
from core.async_inference.types import RequestTrace, TerminalStatus


def make_trace(request_id, issued_ns, started_ns, finished_ns, completed_ns):
    return RequestTrace(
        request_id=request_id,
        sample_index=request_id,
        status=TerminalStatus.COMPLETED,
        scheduled_ns=issued_ns,
        issued_ns=issued_ns,
        enqueued_ns=issued_ns,
        runtime_started_ns=started_ns,
        runtime_finished_ns=finished_ns,
        completed_ns=completed_ns,
        worker_id=0,
        batch_size=1,
        timed_out=False,
    )


def test_metrics_compute_exact_latency_decomposition_and_percentiles():
    metrics = AsyncMetricsCollector(
        started_ns=0,
        worker_count=1,
        latency_slo_ms=5.0,
    )
    metrics.record_submitted()
    metrics.record_accepted(now_ns=0, queue_depth=1)
    metrics.record_queue_depth(depth=0, now_ns=1_000_000)
    metrics.record_worker_busy(worker_id=0, started_ns=2_000_000, finished_ns=5_000_000)
    metrics.record_terminal(
        make_trace(0, 0, 2_000_000, 5_000_000, 6_000_000)
    )

    result = metrics.finalize(end_ns=10_000_000)

    assert result["summary"]["async_completed_requests"] == 1
    assert result["details"]["timing_ms"]["queue_wait"]["p50"] == pytest.approx(2.0)
    assert result["details"]["timing_ms"]["service_time"]["mean"] == pytest.approx(3.0)
    assert result["details"]["timing_ms"]["e2e_latency"]["max"] == pytest.approx(6.0)
    assert result["details"]["queue"]["depth_mean"] == pytest.approx(0.1)
    assert result["details"]["workers"]["utilization"] == pytest.approx(0.3)
    assert result["summary"]["async_issued_requests_per_sec"] == pytest.approx(100.0)
    assert result["summary"]["async_over_latency_slo_requests"] == 1
    assert result["details"]["queue"]["submit_block_total_ms"] == pytest.approx(0.0)


def test_timeout_is_diagnostic_subset_not_extra_terminal_count():
    metrics = AsyncMetricsCollector(started_ns=0, worker_count=1)
    metrics.record_submitted()
    metrics.record_accepted(now_ns=0, queue_depth=1)
    trace = make_trace(0, 0, 1_000_000, 2_000_000, 3_000_000)
    metrics.record_terminal(
        RequestTrace(**{**trace.__dict__, "timed_out": True})
    )

    result = metrics.finalize(end_ns=4_000_000)

    summary = result["summary"]
    assert summary["async_completed_requests"] == 1
    assert summary["async_timed_out_requests"] == 1
    assert summary["async_outstanding_requests"] == 0
    assert "request_timeout" in result["details"]["invalid_reasons"]


def test_rejected_request_preserves_counter_invariants():
    metrics = AsyncMetricsCollector(started_ns=0, worker_count=1)
    metrics.record_submitted()
    metrics.record_rejected("queue_full")

    result = metrics.finalize(end_ns=1_000_000)

    assert result["details"]["counter_invariants"]["valid"] is True
    assert result["summary"]["async_rejected_requests"] == 1
    assert "request_rejected" in result["details"]["invalid_reasons"]


def test_begin_measurement_excludes_engine_startup_time():
    metrics = AsyncMetricsCollector(started_ns=1, worker_count=1)
    metrics.begin_measurement(started_ns=1_000_000_000)
    metrics.record_submitted()
    metrics.record_rejected("queue_full")

    result = metrics.finalize(end_ns=2_000_000_000)

    assert result["summary"]["async_issued_requests_per_sec"] == pytest.approx(1.0)
```

- [ ] **Step 2: Run the tests and verify the missing collector failure**

Run:

```bash
cd framework
python -m pytest tests/test_async_metrics.py -q
```

Expected: collection fails because `core.async_inference.metrics` does not exist.

- [ ] **Step 3: Implement compact distributions and time-weighted gauges**

Create `framework/src/core/async_inference/metrics.py`:

```python
from array import array
from collections import Counter
from threading import Lock
from typing import Dict

import numpy as np

from .types import RequestTrace, TerminalStatus


PERCENTILES = (50.0, 90.0, 95.0, 97.0, 99.0, 99.9)


class TimingDistribution:
    def __init__(self):
        self.values = array("d")

    def add(self, value_ms: float) -> None:
        self.values.append(float(value_ms))

    def summary(self) -> Dict[str, float | int | None]:
        if not self.values:
            return {
                "count": 0,
                "min": None,
                "max": None,
                "mean": None,
                "sum": 0.0,
                "p50": None,
                "p90": None,
                "p95": None,
                "p97": None,
                "p99": None,
                "p99_9": None,
            }
        values = np.frombuffer(self.values, dtype=np.float64)
        percentiles = np.percentile(values, PERCENTILES)
        return {
            "count": int(values.size),
            "min": float(values.min()),
            "max": float(values.max()),
            "mean": float(values.mean()),
            "sum": float(values.sum()),
            "p50": float(percentiles[0]),
            "p90": float(percentiles[1]),
            "p95": float(percentiles[2]),
            "p97": float(percentiles[3]),
            "p99": float(percentiles[4]),
            "p99_9": float(percentiles[5]),
        }


class TimeWeightedGauge:
    def __init__(self, started_ns: int, initial: int = 0):
        self.last_ns = started_ns
        self.value = initial
        self.area = 0
        self.minimum = initial
        self.maximum = initial

    def update(self, value: int, now_ns: int) -> None:
        effective_ns = max(now_ns, self.last_ns)
        self.area += self.value * (effective_ns - self.last_ns)
        self.value = value
        self.last_ns = effective_ns
        self.minimum = min(self.minimum, value)
        self.maximum = max(self.maximum, value)

    def summary(self, end_ns: int, started_ns: int) -> Dict[str, float | int]:
        self.update(self.value, end_ns)
        duration = max(1, end_ns - started_ns)
        return {
            "min": self.minimum,
            "max": self.maximum,
            "mean": self.area / duration,
        }


class AsyncMetricsCollector:
    def __init__(
        self,
        started_ns: int,
        worker_count: int,
        latency_slo_ms: float | None = None,
    ):
        self.started_ns = started_ns
        self.worker_count = worker_count
        self.latency_slo_ms = latency_slo_ms
        self.lock = Lock()
        self.counters = Counter()
        self.invalid_reasons = set()
        self.warnings = set()
        self.error_types = Counter()
        self.error_request_examples = {}
        self.queue_depth = TimeWeightedGauge(started_ns)
        self.inflight = TimeWeightedGauge(started_ns)
        self.worker_busy_ns = Counter()
        self.worker_batches = Counter()
        self.worker_samples = Counter()
        self.batch_sizes = TimingDistribution()
        self.generation_timing_sources = Counter()
        self.timings = {
            "scheduler_delay": TimingDistribution(),
            "submit_wait": TimingDistribution(),
            "queue_wait": TimingDistribution(),
            "service_time": TimingDistribution(),
            "completion_overhead": TimingDistribution(),
            "e2e_latency": TimingDistribution(),
            "ttft_event": TimingDistribution(),
            "reported_ttft": TimingDistribution(),
            "reported_tpot": TimingDistribution(),
        }

    def begin_measurement(self, started_ns: int) -> None:
        with self.lock:
            if self.counters:
                raise RuntimeError("measurement already contains events")
            self.started_ns = started_ns
            self.queue_depth = TimeWeightedGauge(started_ns)
            self.inflight = TimeWeightedGauge(started_ns)

    def record_submitted(self) -> None:
        with self.lock:
            self.counters["submitted"] += 1

    def record_accepted(self, now_ns: int, queue_depth: int) -> None:
        with self.lock:
            self.counters["accepted"] += 1
            self.inflight.update(self.counters["accepted"] - self.counters["terminal"], now_ns)
            self.queue_depth.update(queue_depth, now_ns)

    def record_rejected(self, reason: str) -> None:
        with self.lock:
            self.counters["rejected"] += 1
            self.counters[f"rejected:{reason}"] += 1
            self.invalid_reasons.add("request_rejected")

    def record_queue_depth(self, depth: int, now_ns: int) -> None:
        with self.lock:
            self.queue_depth.update(depth, now_ns)

    def record_queue_full(self) -> None:
        with self.lock:
            self.counters["queue_full_events"] += 1

    def record_worker_busy(
        self,
        worker_id: int,
        started_ns: int,
        finished_ns: int,
        batch_size: int = 1,
        sample_count: int | None = None,
    ) -> None:
        with self.lock:
            self.worker_busy_ns[worker_id] += max(0, finished_ns - started_ns)
            self.worker_batches[worker_id] += 1
            self.worker_samples[worker_id] += (
                batch_size if sample_count is None else sample_count
            )
            self.batch_sizes.add(batch_size)

    def add_invalid_reason(self, reason: str) -> None:
        with self.lock:
            self.invalid_reasons.add(reason)

    def add_warning(self, warning: str) -> None:
        with self.lock:
            self.warnings.add(warning)

    def record_first_token(self, request, event) -> None:
        with self.lock:
            if event.first_token_ns < request.issued_ns:
                self.invalid_reasons.add("timing_invariant_failed")
                return
            self.counters["first_token_events"] += 1
            self.timings["ttft_event"].add(
                (event.first_token_ns - request.issued_ns) / 1_000_000.0
            )

    def record_generation(self, generated_tokens: int, timing_ms) -> None:
        if generated_tokens <= 0:
            return
        with self.lock:
            self.counters["completed_tokens"] += generated_tokens
            if not isinstance(timing_ms, dict):
                return
            reported_ttft = timing_ms.get("ttft_ms")
            reported_tpot = timing_ms.get("tpot_ms")
            if reported_ttft is not None:
                self.timings["reported_ttft"].add(reported_ttft)
            if reported_tpot is not None:
                self.timings["reported_tpot"].add(reported_tpot)
            self.generation_timing_sources[
                timing_ms.get("timing_source", "unknown")
            ] += 1

    def record_terminal(self, trace: RequestTrace) -> None:
        with self.lock:
            status = trace.status.value
            self.counters[status] += 1
            self.counters[f"{status}_samples"] += trace.sample_count
            self.counters["terminal"] += 1
            if trace.status is TerminalStatus.FAILED:
                self.invalid_reasons.add("request_failed")
            if trace.timed_out:
                self.counters["timed_out"] += 1
                self.invalid_reasons.add("request_timeout")
            if trace.error_type:
                self.error_types[trace.error_type] += 1
                examples = self.error_request_examples.setdefault(
                    trace.error_type,
                    [],
                )
                if len(examples) < 5:
                    examples.append(trace.request_id)
            self.inflight.update(
                self.counters["accepted"] - self.counters["terminal"],
                trace.completed_ns,
            )

            ns_to_ms = 1.0 / 1_000_000.0
            values = {
                "scheduler_delay": trace.issued_ns - trace.scheduled_ns,
                "submit_wait": trace.enqueued_ns - trace.issued_ns,
                "queue_wait": trace.runtime_started_ns - trace.enqueued_ns,
                "service_time": trace.runtime_finished_ns - trace.runtime_started_ns,
                "completion_overhead": trace.completed_ns - trace.runtime_finished_ns,
                "e2e_latency": trace.completed_ns - trace.issued_ns,
            }
            for name, value_ns in values.items():
                self.timings[name].add(value_ns * ns_to_ms)
            if (
                self.latency_slo_ms is not None
                and values["e2e_latency"] * ns_to_ms > self.latency_slo_ms
            ):
                self.counters["over_latency_slo"] += 1
            timing_sum = (
                values["submit_wait"]
                + values["queue_wait"]
                + values["service_time"]
                + values["completion_overhead"]
            )
            if abs(values["e2e_latency"] - timing_sum) > 50_000:
                self.invalid_reasons.add("timing_invariant_failed")

    def finalize(self, end_ns: int) -> Dict[str, Dict]:
        with self.lock:
            submitted = self.counters["submitted"]
            accepted = self.counters["accepted"]
            rejected = self.counters["rejected"]
            completed = self.counters["completed"]
            completed_samples = self.counters["completed_samples"]
            failed = self.counters["failed"]
            outstanding = accepted - completed - failed
            invariant_valid = (
                submitted == accepted + rejected
                and accepted == completed + failed + outstanding
                and outstanding >= 0
            )
            if not invariant_valid:
                self.invalid_reasons.add("counter_invariant_failed")
            if outstanding:
                self.invalid_reasons.add("flush_timeout")

            duration_sec = max(1, end_ns - self.started_ns) / 1_000_000_000.0
            queue = self.queue_depth.summary(end_ns, self.started_ns)
            inflight = self.inflight.summary(end_ns, self.started_ns)
            timing = {name: dist.summary() for name, dist in self.timings.items()}
            total_busy = sum(self.worker_busy_ns.values())
            utilization = total_busy / (
                max(1, self.worker_count) * duration_sec * 1_000_000_000.0
            )
            summary = {
                "async_submitted_requests": submitted,
                "async_accepted_requests": accepted,
                "async_completed_requests": completed,
                "async_completed_samples": completed_samples,
                "async_failed_requests": failed,
                "async_rejected_requests": rejected,
                "async_timed_out_requests": self.counters["timed_out"],
                "async_over_latency_slo_requests": self.counters[
                    "over_latency_slo"
                ],
                "async_outstanding_requests": outstanding,
                "async_issued_requests_per_sec": submitted / duration_sec,
                "async_completed_samples_per_sec": completed_samples / duration_sec,
                "async_completed_tokens_per_sec": (
                    self.counters["completed_tokens"] / duration_sec
                ),
                "async_queue_depth_max": queue["max"],
                "async_worker_utilization": utilization,
                "async_e2e_latency_p50_ms": timing["e2e_latency"]["p50"],
                "async_e2e_latency_p95_ms": timing["e2e_latency"]["p95"],
                "async_e2e_latency_p99_ms": timing["e2e_latency"]["p99"],
                "async_queue_wait_p99_ms": timing["queue_wait"]["p99"],
                "async_service_time_p99_ms": timing["service_time"]["p99"],
            }
            details = {
                "measurement_duration_sec": duration_sec,
                "measurement": {
                    "started_monotonic_ns": self.started_ns,
                    "ended_monotonic_ns": end_ns,
                    "duration_sec": duration_sec,
                },
                "invalid_reasons": sorted(self.invalid_reasons),
                "warnings": sorted(self.warnings),
                "counter_invariants": {
                    "valid": invariant_valid,
                    "submitted_equals_accepted_plus_rejected": submitted == accepted + rejected,
                    "accepted_equals_terminal_plus_outstanding": (
                        accepted == completed + failed + outstanding
                    ),
                },
                "counts": dict(self.counters),
                "timing_ms": timing,
                "queue": {
                    "depth_min": queue["min"],
                    "depth_max": queue["max"],
                    "depth_mean": queue["mean"],
                    "full_events": self.counters["queue_full_events"],
                    "submit_block_total_ms": timing["submit_wait"]["sum"],
                    "inflight_min": inflight["min"],
                    "inflight_max": inflight["max"],
                    "inflight_mean": inflight["mean"],
                },
                "workers": {
                    "utilization": utilization,
                    "busy_ns": dict(self.worker_busy_ns),
                    "batches": dict(self.worker_batches),
                    "samples": dict(self.worker_samples),
                },
                "batch_size": self.batch_sizes.summary(),
                "failure_types": dict(self.error_types),
                "failure_request_examples": self.error_request_examples,
                "generation": {
                    "completed_tokens": self.counters["completed_tokens"],
                    "timing_sources": dict(self.generation_timing_sources),
                    "event_ttft_ms": timing["ttft_event"],
                    "reported_ttft_ms": timing["reported_ttft"],
                    "reported_tpot_ms": timing["reported_tpot"],
                },
            }
            return {"summary": summary, "details": details}
```

- [ ] **Step 4: Run the metrics tests**

Run:

```bash
cd framework
python -m pytest tests/test_async_metrics.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit the metrics collector**

```bash
git add framework/src/core/async_inference/metrics.py framework/tests/test_async_metrics.py
git commit -m "feat(framework): collect async queue metrics"
```

### Task 4: Serialize completion, evaluation, and exact-once terminal handling

**Files:**
- Create: `framework/src/core/async_inference/completion.py`
- Create: `framework/tests/test_async_completion.py`

**Interfaces:**
- Consumes: `BatchCompletion`, `InferencePipeline.prepare_eval_labels()`, decoder, evaluator, and `AsyncMetricsCollector`.
- Produces: `FirstTokenTracker` plus `CompletionCoordinator.start()`, `register(request)`, `submit(completion)`, `wait_for_all(timeout)`, and `stop()`.

- [ ] **Step 1: Write exact-once and serialized evaluator tests**

Create `framework/tests/test_async_completion.py`:

```python
from dataclasses import replace
import threading
import time

import numpy as np

from core.async_inference.completion import (
    CompletionCoordinator,
    FirstTokenTracker,
)
from core.async_inference.metrics import AsyncMetricsCollector
from core.async_inference.types import (
    BatchCompletion,
    FirstTokenEvent,
    InferenceRequest,
)


class FakePipeline:
    def prepare_eval_labels(self, collated):
        return collated["label"]


class RecordingEvaluator:
    def __init__(self):
        self.calls = []

    def add_batch(self, outputs, labels, timing_ms):
        self.calls.append(
            (threading.get_ident(), outputs["output"].tolist(), labels, timing_ms)
        )


def request(request_id):
    return InferenceRequest(
        request_id=request_id,
        sample_index=request_id,
        sample={"input": np.array([request_id]), "label": request_id},
        scheduled_ns=0,
        issued_ns=0,
        enqueued_ns=1,
    )


def completion(req):
    return BatchCompletion(
        requests=[req],
        collated={"label": [req.sample_index]},
        outputs={"output": np.array([[req.sample_index]])},
        timing_ms=1.0,
        runtime_started_ns=2,
        runtime_finished_ns=3,
        worker_id=0,
        batch_size=1,
    )


def test_completion_runs_evaluator_on_coordinator_thread_and_drains():
    evaluator = RecordingEvaluator()
    metrics = AsyncMetricsCollector(started_ns=0, worker_count=1)
    coordinator = CompletionCoordinator(
        pipeline=FakePipeline(),
        evaluator=evaluator,
        decoder=None,
        metrics=metrics,
        queue_capacity=1,
    )
    coordinator.start()
    req = request(0)
    coordinator.register(req)
    coordinator.submit(completion(req))

    assert coordinator.wait_for_all(timeout=1.0) is True
    coordinator.stop(timeout=1.0)

    assert len(evaluator.calls) == 1
    assert evaluator.calls[0][0] != threading.get_ident()


def test_duplicate_completion_marks_run_invalid_without_double_evaluation():
    evaluator = RecordingEvaluator()
    metrics = AsyncMetricsCollector(started_ns=0, worker_count=1)
    coordinator = CompletionCoordinator(
        pipeline=FakePipeline(),
        evaluator=evaluator,
        decoder=None,
        metrics=metrics,
        queue_capacity=2,
    )
    coordinator.start()
    req = request(0)
    coordinator.register(req)
    item = completion(req)
    coordinator.submit(item)
    assert coordinator.wait_for_all(timeout=1.0) is True
    coordinator.submit(item)
    time.sleep(0.02)
    coordinator.stop(timeout=1.0)

    details = metrics.finalize(end_ns=time.monotonic_ns())["details"]
    assert len(evaluator.calls) == 1
    assert "duplicate_completion" in details["invalid_reasons"]


def test_mixed_duplicate_batch_fails_new_member_without_double_evaluation():
    evaluator = RecordingEvaluator()
    metrics = AsyncMetricsCollector(started_ns=0, worker_count=1)
    coordinator = CompletionCoordinator(
        pipeline=FakePipeline(),
        evaluator=evaluator,
        decoder=None,
        metrics=metrics,
        queue_capacity=2,
    )
    coordinator.start()
    first = request(0)
    second = request(1)
    coordinator.register(first)
    coordinator.register(second)
    coordinator.submit(completion(first))
    with coordinator.condition:
        assert coordinator.condition.wait_for(
            lambda: bool(coordinator.terminal[0]),
            timeout=1.0,
        )
    coordinator.submit(
        replace(completion(second), requests=[first, second])
    )

    assert coordinator.wait_for_all(timeout=1.0) is True
    coordinator.stop(timeout=1.0)
    result = metrics.finalize(end_ns=time.monotonic_ns())
    assert len(evaluator.calls) == 1
    assert result["summary"]["async_failed_requests"] == 1
    assert "duplicate_completion" in result["details"]["invalid_reasons"]


def test_first_token_contract_rejects_duplicate_and_final_before_event():
    metrics = AsyncMetricsCollector(started_ns=0, worker_count=1)
    tracker = FirstTokenTracker(metrics)
    first = request(0)
    tracker.register(first)
    event = FirstTokenEvent(request_id=0, first_token_ns=2)
    assert tracker.record(event) is True
    assert tracker.record(event) is False
    assert tracker.finalize(request_id=0, generated_tokens=1) is True

    second = request(1)
    tracker.register(second)
    assert tracker.finalize(request_id=1, generated_tokens=1) is False
    details = metrics.finalize(end_ns=time.monotonic_ns())["details"]
    assert "timing_invariant_failed" in details["invalid_reasons"]
    assert details["generation"]["event_ttft_ms"]["count"] == 1


def test_completion_thread_failure_unblocks_waiter_immediately():
    metrics = AsyncMetricsCollector(started_ns=0, worker_count=1)
    coordinator = CompletionCoordinator(
        pipeline=FakePipeline(),
        evaluator=RecordingEvaluator(),
        decoder=None,
        metrics=metrics,
        queue_capacity=1,
    )

    def crash(_completion):
        raise RuntimeError("planned coordinator crash")

    coordinator._handle = crash
    coordinator.start()
    req = request(0)
    coordinator.register(req)
    coordinator.submit(completion(req))

    assert coordinator.wait_for_all(timeout=1.0) is False
    assert coordinator.stop(timeout=1.0) is False
    details = metrics.finalize(end_ns=time.monotonic_ns())["details"]
    assert "completion_thread_failed" in details["invalid_reasons"]
```

- [ ] **Step 2: Run the tests and verify the missing coordinator failure**

Run:

```bash
cd framework
python -m pytest tests/test_async_completion.py -q
```

Expected: collection fails because `core.async_inference.completion` does not exist.

- [ ] **Step 3: Implement the coordinator**

Create `framework/src/core/async_inference/completion.py`:

```python
import logging
import queue
import threading
import time
from typing import Callable, Optional

from .types import (
    BatchCompletion,
    InferenceRequest,
    RequestTrace,
    TerminalStatus,
)


_STOP = object()
LOGGER = logging.getLogger(__name__)


def _safe_error_message(message) -> str:
    return " ".join(str(message).split())[:512]


class FirstTokenTracker:
    """Lifecycle contract for future real streaming runtime integrations."""

    def __init__(self, metrics):
        self.metrics = metrics
        self.pending = {}
        self.events = {}
        self.lock = threading.Lock()

    def register(self, request: InferenceRequest) -> None:
        with self.lock:
            self.pending[request.request_id] = request

    def record(self, event) -> bool:
        with self.lock:
            request = self.pending.get(event.request_id)
            if request is None or event.request_id in self.events:
                self.metrics.add_invalid_reason("timing_invariant_failed")
                return False
            self.events[event.request_id] = event
        self.metrics.record_first_token(request, event)
        return True

    def finalize(self, request_id: int, generated_tokens: int) -> bool:
        with self.lock:
            request = self.pending.pop(request_id, None)
            event = self.events.pop(request_id, None)
        valid = request is not None and (generated_tokens == 0 or event is not None)
        if not valid:
            self.metrics.add_invalid_reason("timing_invariant_failed")
        return valid


class CompletionCoordinator:
    def __init__(
        self,
        pipeline,
        evaluator,
        decoder,
        metrics,
        queue_capacity: int,
        request_timeout_ms: float = 0.0,
        trace_callback: Optional[Callable[[RequestTrace], None]] = None,
        clock_ns: Callable[[], int] = time.monotonic_ns,
    ):
        self.pipeline = pipeline
        self.evaluator = evaluator
        self.decoder = decoder
        self.metrics = metrics
        self.request_timeout_ns = int(request_timeout_ms * 1_000_000)
        self.trace_callback = trace_callback
        self.clock_ns = clock_ns
        self.queue = queue.Queue(maxsize=queue_capacity)
        self.condition = threading.Condition()
        self.outstanding = {}
        self.terminal = bytearray()
        self.thread_error = None
        self.thread = threading.Thread(
            target=self._run,
            name="async-completion",
            daemon=True,
        )

    def start(self) -> None:
        self.thread.start()

    def register(self, request: InferenceRequest) -> None:
        with self.condition:
            if request.request_id in self.outstanding:
                raise ValueError(f"duplicate request_id: {request.request_id}")
            self.outstanding[request.request_id] = request
            required = request.request_id + 1 - len(self.terminal)
            if required > 0:
                self.terminal.extend(b"\x00" * required)

    def unregister_rejected(self, request_id: int) -> None:
        with self.condition:
            self.outstanding.pop(request_id, None)
            self.condition.notify_all()

    def submit(self, completion: BatchCompletion) -> None:
        self.queue.put(completion)

    def wait_for_all(self, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        with self.condition:
            while self.outstanding:
                if self.thread_error is not None:
                    self.metrics.add_invalid_reason("completion_thread_failed")
                    return False
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self.metrics.add_invalid_reason("flush_timeout")
                    return False
                self.condition.wait(timeout=remaining)
            return True

    def stop(self, timeout: float) -> bool:
        try:
            self.queue.put(_STOP, timeout=timeout)
        except queue.Full:
            self.metrics.add_invalid_reason("completion_thread_failed")
            return False
        self.thread.join(timeout=timeout)
        if self.thread.is_alive() or self.thread_error is not None:
            self.metrics.add_invalid_reason("completion_thread_failed")
            return False
        return True

    def _run(self) -> None:
        try:
            while True:
                item = self.queue.get()
                try:
                    if item is _STOP:
                        return
                    self._handle(item)
                finally:
                    self.queue.task_done()
        except BaseException as exc:
            LOGGER.exception("async completion coordinator failed")
            with self.condition:
                self.thread_error = f"{type(exc).__name__}: {_safe_error_message(exc)}"
                self.metrics.add_invalid_reason("completion_thread_failed")
                self.condition.notify_all()

    def _handle(self, completion: BatchCompletion) -> None:
        known = []
        membership_error = False
        with self.condition:
            for request in completion.requests:
                request_id = request.request_id
                if request_id < len(self.terminal) and self.terminal[request_id]:
                    self.metrics.add_invalid_reason("duplicate_completion")
                    membership_error = True
                    continue
                if request_id not in self.outstanding:
                    self.metrics.add_invalid_reason("unknown_completion")
                    membership_error = True
                    continue
                known.append(request)

        if not known:
            return

        error_type = completion.error_type
        error_message = (
            _safe_error_message(completion.error_message)
            if completion.error_message is not None
            else None
        )
        if membership_error:
            error_type = "InvalidCompletionMembership"
            error_message = "batch contained duplicate or unknown request IDs"
        if error_type is None:
            try:
                outputs = completion.outputs
                if self.decoder is not None:
                    outputs = self.decoder.decode(outputs)
                labels = self.pipeline.prepare_eval_labels(completion.collated)
                self.evaluator.add_batch(outputs, labels, completion.timing_ms)
            except Exception as exc:
                LOGGER.exception("async decoder or evaluator failed")
                error_type = type(exc).__name__
                error_message = _safe_error_message(exc)

        if error_type is None:
            self.metrics.record_generation(
                completion.generated_tokens,
                completion.timing_ms,
            )

        completed_ns = self.clock_ns()
        for request in known:
            elapsed_ns = completed_ns - request.issued_ns
            timed_out = bool(
                self.request_timeout_ns
                and elapsed_ns > self.request_timeout_ns
            )
            status = (
                TerminalStatus.COMPLETED
                if error_type is None
                else TerminalStatus.FAILED
            )
            trace = RequestTrace(
                request_id=request.request_id,
                sample_index=request.sample_index,
                status=status,
                scheduled_ns=request.scheduled_ns,
                issued_ns=request.issued_ns,
                enqueued_ns=request.enqueued_ns,
                runtime_started_ns=completion.runtime_started_ns,
                runtime_finished_ns=completion.runtime_finished_ns,
                completed_ns=completed_ns,
                worker_id=completion.worker_id,
                batch_size=completion.batch_size,
                timed_out=timed_out,
                sample_count=request.sample_count,
                error_type=error_type,
                error_message=error_message,
            )
            self.metrics.record_terminal(trace)
            if self.trace_callback is not None:
                try:
                    self.trace_callback(trace)
                except Exception:
                    self.metrics.add_warning("request_trace_write_failed")
            with self.condition:
                self.terminal[request.request_id] = 1
                self.outstanding.pop(request.request_id, None)
                self.condition.notify_all()
```

- [ ] **Step 4: Run the coordinator tests**

Run:

```bash
cd framework
python -m pytest tests/test_async_completion.py tests/test_async_metrics.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit completion serialization**

```bash
git add framework/src/core/async_inference/completion.py framework/tests/test_async_completion.py
git commit -m "feat(framework): serialize async completions"
```

### Task 5: Implement the bounded engine, workers, and dynamic batching

**Files:**
- Create: `framework/src/core/async_inference/engine.py`
- Create: `framework/tests/test_async_engine.py`

**Interfaces:**
- Consumes: `InferencePipeline`, `CompletionCoordinator`, `AsyncMetricsCollector`, `AsyncInferenceConfig`.
- Produces: `AsyncInferenceEngine.start()`, `submit(request, block) -> bool`, `cancel_queued(reason)`, `close_submission()`, `flush()`, and `shutdown()`.

- [ ] **Step 1: Write boundedness, batching, and failure-drain tests**

Create `framework/tests/test_async_engine.py`:

```python
import threading
import time
from dataclasses import replace

import numpy as np
import pytest

from core.async_inference.completion import CompletionCoordinator
from core.async_inference.engine import AsyncInferenceEngine
from core.async_inference.metrics import AsyncMetricsCollector
from core.async_inference.types import AsyncInferenceConfig, InferenceRequest
from core.inference_pipeline import InferencePipeline


class Loader:
    def get_metadata(self):
        return {"is_static_batched": False, "total_samples": 8}


class Runtime:
    compiled_model = None

    def __init__(self, fail=False, max_batch_size=None):
        self.fail = fail
        self.max_batch_size = max_batch_size
        self.batch_sizes = []

    def supports_generate(self):
        return False

    def max_concurrent_workers(self):
        return 1

    def supports_dynamic_batching(self):
        return True

    def max_dynamic_batch_size(self):
        return self.max_batch_size

    def run(self, inputs):
        if self.fail:
            raise RuntimeError("planned failure")
        self.batch_sizes.append(len(inputs["input"]))
        return {"output": np.asarray(inputs["input"])}


class Evaluator:
    def __init__(self):
        self.samples = 0

    def add_batch(self, outputs, labels, timing_ms):
        self.samples += len(labels)


def build(config, runtime=None):
    runtime = runtime or Runtime()
    pipeline = InferencePipeline(Loader(), runtime)
    metrics = AsyncMetricsCollector(time.monotonic_ns(), config.worker_count)
    evaluator = Evaluator()
    coordinator = CompletionCoordinator(
        pipeline,
        evaluator,
        None,
        metrics,
        queue_capacity=config.worker_count,
    )
    engine = AsyncInferenceEngine(
        runtime,
        pipeline,
        config,
        coordinator,
        metrics,
    )
    return engine, runtime, evaluator, metrics


def make_request(request_id):
    now = time.monotonic_ns()
    return InferenceRequest(
        request_id=request_id,
        sample_index=request_id,
        sample={"input": np.array([request_id], dtype=np.float32), "label": request_id},
        scheduled_ns=now,
        issued_ns=now,
        enqueued_ns=0,
    )


def test_engine_dynamically_batches_and_drains_every_request():
    config = AsyncInferenceConfig(
        queue_capacity=8,
        max_batch_size=4,
        batch_timeout_ms=20,
        min_samples=1,
    )
    engine, runtime, evaluator, metrics = build(config)
    engine.start()
    for request_id in range(8):
        assert engine.submit(make_request(request_id), block=True)
    engine.close_submission()
    assert engine.flush() is True
    assert engine.shutdown() is True

    result = metrics.finalize(time.monotonic_ns())
    assert evaluator.samples == 8
    assert sum(runtime.batch_sizes) == 8
    assert max(runtime.batch_sizes) == 4
    assert result["summary"]["async_outstanding_requests"] == 0


def test_runtime_failure_does_not_deadlock_flush():
    config = AsyncInferenceConfig(queue_capacity=2, min_samples=1)
    engine, runtime, evaluator, metrics = build(config, Runtime(fail=True))
    engine.start()
    assert engine.submit(make_request(0), block=True)
    engine.close_submission()
    assert engine.flush() is True
    assert engine.shutdown() is True

    result = metrics.finalize(time.monotonic_ns())
    assert result["summary"]["async_failed_requests"] == 1
    assert result["details"]["failure_types"] == {"RuntimeError": 1}
    assert result["details"]["failure_request_examples"] == {"RuntimeError": [0]}
    assert evaluator.samples == 0


def test_engine_rejects_batch_size_above_runtime_capability():
    config = AsyncInferenceConfig(
        queue_capacity=2,
        max_batch_size=2,
        min_samples=1,
    )
    with pytest.raises(ValueError, match="exceeds runtime capability"):
        build(config, Runtime(max_batch_size=1))


def test_cancel_queued_terminalizes_requests_before_flush():
    config = AsyncInferenceConfig(queue_capacity=4, min_samples=1)
    engine, runtime, evaluator, metrics = build(config)
    engine.coordinator.start()

    queued = make_request(0)
    now = time.monotonic_ns()
    queued = replace(queued, enqueued_ns=now)
    engine.coordinator.register(queued)
    metrics.record_submitted()
    metrics.record_accepted(now_ns=now, queue_depth=1)
    engine.requests.put_nowait(queued)
    engine.slots.acquire(blocking=False)

    assert engine.cancel_queued("KeyboardInterrupt") == 1
    assert engine.flush() is True
    assert engine.coordinator.stop(timeout=1.0) is True
    result = metrics.finalize(time.monotonic_ns())
    assert result["summary"]["async_failed_requests"] == 1
    assert result["summary"]["async_outstanding_requests"] == 0
```

- [ ] **Step 2: Run the tests and verify the missing engine failure**

Run:

```bash
cd framework
python -m pytest tests/test_async_engine.py -q
```

Expected: collection fails because `core.async_inference.engine` does not exist.

- [ ] **Step 3: Implement the engine**

Create `framework/src/core/async_inference/engine.py`:

```python
import logging
import queue
import threading
import time
from dataclasses import replace

from .types import BatchCompletion, EngineState


_STOP = object()
LOGGER = logging.getLogger(__name__)


class AsyncInferenceEngine:
    def __init__(self, runtime, pipeline, config, coordinator, metrics):
        config.validate()
        if config.worker_count > runtime.max_concurrent_workers():
            raise ValueError(
                f"worker_count={config.worker_count} exceeds runtime capability "
                f"{runtime.max_concurrent_workers()}"
            )
        if config.max_batch_size > 1 and not pipeline.is_static_batched:
            if not runtime.supports_dynamic_batching():
                raise ValueError("runtime does not support dynamic batching")
            max_dynamic_batch_size = runtime.max_dynamic_batch_size()
            if (
                max_dynamic_batch_size is not None
                and config.max_batch_size > max_dynamic_batch_size
            ):
                raise ValueError(
                    f"max_batch_size={config.max_batch_size} exceeds runtime "
                    f"capability {max_dynamic_batch_size}"
                )
        if pipeline.is_llm and config.max_batch_size > 1:
            if not runtime.supports_batch_generation():
                raise ValueError("runtime does not support batch generation")

        self.runtime = runtime
        self.pipeline = pipeline
        self.config = config
        self.coordinator = coordinator
        self.metrics = metrics
        self.state = EngineState.CREATED
        self.state_lock = threading.Lock()
        self.requests = queue.Queue(maxsize=config.queue_capacity)
        self.slots = threading.BoundedSemaphore(config.queue_capacity)
        self.workers = [
            threading.Thread(
                target=self._worker,
                args=(worker_id,),
                name=f"async-worker-{worker_id}",
                daemon=True,
            )
            for worker_id in range(config.worker_count)
        ]

    def start(self) -> None:
        with self.state_lock:
            if self.state is not EngineState.CREATED:
                raise RuntimeError(f"cannot start engine in {self.state.value}")
            self.state = EngineState.RUNNING
        self.coordinator.start()
        for worker in self.workers:
            worker.start()

    def submit(self, request, block: bool) -> bool:
        with self.state_lock:
            if self.state is not EngineState.RUNNING:
                raise RuntimeError(f"cannot submit in {self.state.value}")
        self.metrics.record_submitted()
        if block:
            acquired = self.slots.acquire(
                blocking=True,
                timeout=self.config.submit_timeout_sec,
            )
        else:
            acquired = self.slots.acquire(blocking=False)
        if not acquired:
            self.metrics.record_queue_full()
            self.metrics.record_rejected("queue_full")
            if block:
                self.metrics.add_invalid_reason("queue_submit_timeout")
            return False

        queued = replace(request, enqueued_ns=time.monotonic_ns())
        self.coordinator.register(queued)
        self.metrics.record_accepted(
            now_ns=queued.enqueued_ns,
            queue_depth=self.requests.qsize() + 1,
        )
        self.requests.put_nowait(queued)
        return True

    def close_submission(self) -> None:
        with self.state_lock:
            if self.state is EngineState.RUNNING:
                self.state = EngineState.DRAINING

    def cancel_queued(self, reason: str) -> int:
        cancelled = 0
        while True:
            try:
                request = self.requests.get_nowait()
            except queue.Empty:
                break
            if request is _STOP:
                self.requests.task_done()
                continue
            self.slots.release()
            now_ns = time.monotonic_ns()
            self.metrics.record_queue_depth(self.requests.qsize(), now_ns)
            self.coordinator.submit(
                BatchCompletion(
                    requests=[request],
                    collated={},
                    outputs=None,
                    timing_ms=None,
                    runtime_started_ns=now_ns,
                    runtime_finished_ns=now_ns,
                    worker_id=-1,
                    batch_size=1,
                    error_type="CancelledError",
                    error_message=reason,
                )
            )
            self.requests.task_done()
            cancelled += 1
        return cancelled

    def flush(self) -> bool:
        return self.coordinator.wait_for_all(self.config.flush_timeout_sec)

    def shutdown(self) -> bool:
        deadline = time.monotonic() + self.config.flush_timeout_sec
        ok = True
        for _ in self.workers:
            remaining = max(0.0, deadline - time.monotonic())
            try:
                self.requests.put(_STOP, timeout=remaining)
            except queue.Full:
                ok = False
                self.metrics.add_invalid_reason("worker_shutdown_failed")
        for worker in self.workers:
            remaining = max(0.0, deadline - time.monotonic())
            worker.join(timeout=remaining)
            if worker.is_alive():
                ok = False
                self.metrics.add_invalid_reason("worker_shutdown_failed")
        ok = self.coordinator.stop(self.config.flush_timeout_sec) and ok
        with self.state_lock:
            self.state = EngineState.STOPPED if ok else EngineState.FAILED
        return ok

    def _worker(self, worker_id: int) -> None:
        pending = None
        consecutive_failures = 0
        while True:
            first = pending
            pending = None
            if first is None:
                first = self.requests.get()
                if first is _STOP:
                    self.requests.task_done()
                    return
                self.slots.release()
                self.metrics.record_queue_depth(
                    self.requests.qsize(),
                    time.monotonic_ns(),
                )

            batch = [first]
            if self.config.max_batch_size > 1 and not self.pipeline.is_static_batched:
                deadline_ns = time.monotonic_ns() + int(
                    self.config.batch_timeout_ms * 1_000_000
                )
                while len(batch) < self.config.max_batch_size:
                    remaining_sec = (deadline_ns - time.monotonic_ns()) / 1_000_000_000
                    if remaining_sec <= 0:
                        break
                    try:
                        candidate = self.requests.get(timeout=remaining_sec)
                    except queue.Empty:
                        break
                    if candidate is _STOP:
                        self.requests.put(_STOP)
                        self.requests.task_done()
                        break
                    self.slots.release()
                    self.metrics.record_queue_depth(
                        self.requests.qsize(),
                        time.monotonic_ns(),
                    )
                    if self._batch_key(candidate) != self._batch_key(first):
                        pending = candidate
                        break
                    batch.append(candidate)

            collated = {}
            started_ns = None
            try:
                collated = self.pipeline.collate_batch(
                    [item.sample for item in batch]
                )
                runtime_input = self.pipeline.prepare_runtime_input(
                    collated["input"]
                )
                started_ns = time.monotonic_ns()
                invocation = self.pipeline.invoke(runtime_input)
                finished_ns = time.monotonic_ns()
                completion = BatchCompletion(
                    requests=batch,
                    collated=collated,
                    outputs=invocation.outputs,
                    timing_ms=invocation.timing_ms,
                    runtime_started_ns=started_ns,
                    runtime_finished_ns=finished_ns,
                    worker_id=worker_id,
                    batch_size=len(batch),
                    generated_tokens=invocation.generated_tokens,
                )
                consecutive_failures = 0
            except Exception as exc:
                finished_ns = time.monotonic_ns()
                if started_ns is None:
                    started_ns = finished_ns
                LOGGER.exception(
                    "async runtime batch failed on worker %s",
                    worker_id,
                )
                consecutive_failures += 1
                completion = BatchCompletion(
                    requests=batch,
                    collated=collated,
                    outputs=None,
                    timing_ms=None,
                    runtime_started_ns=started_ns,
                    runtime_finished_ns=finished_ns,
                    worker_id=worker_id,
                    batch_size=len(batch),
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )
                if consecutive_failures >= 3:
                    self.metrics.add_invalid_reason("request_failed")
                    with self.state_lock:
                        self.state = EngineState.FAILED

            self.metrics.record_worker_busy(
                worker_id,
                started_ns,
                finished_ns,
                len(batch),
                sum(request.sample_count for request in batch),
            )
            self.coordinator.submit(completion)
            for _ in batch:
                self.requests.task_done()

    @staticmethod
    def _batch_key(request):
        value = request.sample["input"]
        if isinstance(value, dict):
            return tuple(
                (name, array.dtype.str, tuple(array.shape))
                for name, array in sorted(value.items())
            )
        return (value.dtype.str, tuple(value.shape))
```

- [ ] **Step 4: Assert the submit/completion race never makes inflight negative**

Add this assertion to `test_engine_dynamically_batches_and_drains_every_request` after finalization:

```python
    assert result["details"]["queue"]["inflight_min"] >= 0
```

The implementation registers and counts a request before putting it on the worker-visible queue, so even a zero-latency runtime cannot complete before accepted accounting.

- [ ] **Step 5: Run engine, completion, and metrics tests**

Run:

```bash
cd framework
python -m pytest tests/test_async_engine.py tests/test_async_completion.py tests/test_async_metrics.py -q
```

Expected: all tests pass; neither failure test waits for the flush timeout.

- [ ] **Step 6: Commit the engine**

```bash
git add framework/src/core/async_inference/engine.py framework/tests/test_async_engine.py
git commit -m "feat(framework): add bounded async inference engine"
```

### Task 6: Add deterministic Offline and Server-like producers

**Files:**
- Create: `framework/src/core/async_inference/producers.py`
- Create: `framework/tests/test_async_producers.py`

**Interfaces:**
- Consumes: `DataLoader.load_by_index(index)`, `AsyncInferenceEngine.submit()`, and `AsyncInferenceConfig`.
- Produces: `OfflineProducer.run() -> ProducerResult` and `ServerLikeProducer.run() -> ProducerResult`.

- [ ] **Step 1: Write producer tests with a fake clock and submitter**

Create `framework/tests/test_async_producers.py`:

```python
import numpy as np

from core.async_inference.producers import (
    FakeableClock,
    OfflineProducer,
    ServerLikeProducer,
)
from core.async_inference.types import AsyncInferenceConfig, AsyncScenario


class Loader:
    def get_metadata(self):
        return {"total_samples": 3}

    def load_by_index(self, index):
        return {"input": np.array([index]), "label": index}


class Submitter:
    def __init__(self):
        self.requests = []

    def submit(self, request, block):
        self.requests.append((request, block))
        return True


def test_offline_submits_each_sample_once_with_blocking_backpressure():
    submitter = Submitter()
    clock = FakeableClock()
    producer = OfflineProducer(
        Loader(),
        submitter,
        AsyncInferenceConfig(min_samples=1),
        clock=clock,
    )

    result = producer.run()

    assert result.attempted == 3
    assert [request.sample_index for request, _ in submitter.requests] == [0, 1, 2]
    assert all(block is True for _, block in submitter.requests)


def test_server_like_schedule_is_reproducible_and_non_blocking():
    config = AsyncInferenceConfig(
        scenario=AsyncScenario.SERVER_LIKE,
        target_qps=10,
        min_samples=4,
        min_duration_sec=0,
        max_samples=4,
        schedule_seed=7,
    )
    first_submitter = Submitter()
    second_submitter = Submitter()
    ServerLikeProducer(
        Loader(),
        first_submitter,
        config,
        clock=FakeableClock(),
    ).run()
    ServerLikeProducer(
        Loader(),
        second_submitter,
        config,
        clock=FakeableClock(),
    ).run()

    first_schedule = [item[0].scheduled_ns for item in first_submitter.requests]
    second_schedule = [item[0].scheduled_ns for item in second_submitter.requests]
    assert first_schedule == second_schedule
    assert all(block is False for _, block in first_submitter.requests)
```

- [ ] **Step 2: Run tests and verify the missing producer failure**

Run:

```bash
cd framework
python -m pytest tests/test_async_producers.py -q
```

Expected: collection fails because `core.async_inference.producers` does not exist.

- [ ] **Step 3: Implement producer clocks and request generation**

Create `framework/src/core/async_inference/producers.py`:

```python
import random
import time
from dataclasses import dataclass

from .types import InferenceRequest


class SystemClock:
    def monotonic_ns(self):
        return time.monotonic_ns()

    def sleep(self, seconds):
        time.sleep(seconds)


class FakeableClock:
    def __init__(self):
        self.now_ns = 0

    def monotonic_ns(self):
        return self.now_ns

    def sleep(self, seconds):
        self.now_ns += max(0, int(seconds * 1_000_000_000))


@dataclass(frozen=True)
class ProducerResult:
    attempted: int
    accepted: int
    rejected: int
    producer_load_ms: float


class BaseProducer:
    def __init__(self, dataloader, submitter, config, clock=None):
        self.dataloader = dataloader
        self.submitter = submitter
        self.config = config
        self.clock = clock or SystemClock()
        metadata = dataloader.get_metadata()
        self.total_samples = int(metadata["total_samples"])
        self.is_static_batched = bool(metadata.get("is_static_batched", False))
        if self.total_samples < 1:
            raise ValueError("dataloader total_samples must be >= 1")

    def _sample(self, request_id):
        index = request_id % self.total_samples
        load_started = self.clock.monotonic_ns()
        sample = self.dataloader.load_by_index(index)
        load_finished = self.clock.monotonic_ns()
        input_value = sample["input"]
        if self.is_static_batched:
            if isinstance(input_value, dict):
                input_value = next(iter(input_value.values()))
            sample_count = int(input_value.shape[0])
        else:
            sample_count = 1
        return index, sample, sample_count, load_finished - load_started


class OfflineProducer(BaseProducer):
    def run(self):
        limit = (
            self.config.max_samples
            if self.config.max_samples is not None
            else self.total_samples
        )
        accepted = 0
        rejected = 0
        load_ns = 0
        for request_id in range(limit):
            index, sample, sample_count, elapsed = self._sample(request_id)
            load_ns += elapsed
            issued_ns = self.clock.monotonic_ns()
            request = InferenceRequest(
                request_id=request_id,
                sample_index=index,
                sample=sample,
                scheduled_ns=issued_ns,
                issued_ns=issued_ns,
                enqueued_ns=0,
                sample_count=sample_count,
            )
            if self.submitter.submit(request, block=True):
                accepted += 1
            else:
                rejected += 1
        return ProducerResult(limit, accepted, rejected, load_ns / 1_000_000.0)


class ServerLikeProducer(BaseProducer):
    def run(self):
        rng = random.Random(self.config.schedule_seed)
        started_ns = self.clock.monotonic_ns()
        scheduled_ns = started_ns
        accepted = 0
        rejected = 0
        attempted = 0
        load_ns = 0
        while True:
            elapsed_sec = (
                self.clock.monotonic_ns() - started_ns
            ) / 1_000_000_000.0
            minimum_met = (
                attempted >= self.config.min_samples
                and elapsed_sec >= self.config.min_duration_sec
            )
            maximum_met = (
                self.config.max_samples is not None
                and attempted >= self.config.max_samples
            )
            if minimum_met or maximum_met:
                break

            if attempted:
                interval_sec = rng.expovariate(self.config.target_qps)
                scheduled_ns += int(interval_sec * 1_000_000_000)
            remaining_sec = (
                scheduled_ns - self.clock.monotonic_ns()
            ) / 1_000_000_000.0
            if remaining_sec > 0:
                self.clock.sleep(remaining_sec)

            index, sample, sample_count, elapsed = self._sample(attempted)
            load_ns += elapsed
            issued_ns = self.clock.monotonic_ns()
            request = InferenceRequest(
                request_id=attempted,
                sample_index=index,
                sample=sample,
                scheduled_ns=scheduled_ns,
                issued_ns=issued_ns,
                enqueued_ns=0,
                sample_count=sample_count,
            )
            if self.submitter.submit(request, block=False):
                accepted += 1
            else:
                rejected += 1
            attempted += 1
        return ProducerResult(
            attempted,
            accepted,
            rejected,
            load_ns / 1_000_000.0,
        )
```

- [ ] **Step 4: Run producer tests**

Run:

```bash
cd framework
python -m pytest tests/test_async_producers.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit the producers**

```bash
git add framework/src/core/async_inference/producers.py framework/tests/test_async_producers.py
git commit -m "feat(framework): add async workload producers"
```

### Task 7: Orchestrate warmup, monitoring, validity, and quality evaluation

**Files:**
- Create: `framework/src/core/async_inference/runner.py`
- Create: `framework/tests/test_async_runner.py`
- Modify: `framework/src/core/async_inference/__init__.py`

**Interfaces:**
- Consumes: all Tasks 1-6, existing evaluator/decoder/monitor contracts.
- Produces: `AsyncBenchmarkRunner.run() -> AsyncBenchmarkResult`.

- [ ] **Step 1: Write end-to-end Mock Runtime runner tests**

Create `framework/tests/test_async_runner.py`:

```python
from types import SimpleNamespace

import numpy as np

from core.async_inference.runner import AsyncBenchmarkRunner
from core.async_inference.types import (
    AsyncInferenceConfig,
    AsyncScenario,
    RunStatus,
)


class Loader:
    def __init__(self):
        self.samples = [
            {"input": np.array([1.0]), "label": 2.0},
            {"input": np.array([2.0]), "label": 4.0},
            {"input": np.array([3.0]), "label": 6.0},
        ]

    def get_metadata(self):
        return {"total_samples": len(self.samples), "is_static_batched": False}

    def load_by_index(self, index):
        return self.samples[index]

    def load_batch(self, batch_size):
        return self.samples[:batch_size]


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

    def warmup(self, inputs, num_runs=1):
        return None


class Evaluator:
    def __init__(self):
        self.correct = 0
        self.total = 0

    def add_batch(self, outputs, labels, timing_ms):
        predicted = outputs["output"].reshape(-1)
        expected = np.asarray(labels)
        self.correct += int(np.sum(predicted == expected))
        self.total += len(expected)

    def compute(self):
        return {
            "accuracy": self.correct / self.total,
            "Total Samples": self.total,
        }


class Monitor:
    def __init__(self):
        self.events = []

    def start(self):
        self.events.append("start")

    def stop(self):
        self.events.append("stop")

    def summary(self):
        return {"hw_test_samples": 1}


def test_runner_returns_quality_async_and_hardware_metrics():
    monitor = Monitor()
    runner = AsyncBenchmarkRunner(
        dataloader=Loader(),
        runtime=Runtime(),
        evaluator=Evaluator(),
        monitor=monitor,
    )
    result = runner.run(
        AsyncInferenceConfig(
            queue_capacity=4,
            max_batch_size=2,
            batch_timeout_ms=10,
            min_samples=1,
        ),
        warmup_runs=1,
    )

    assert result.status is RunStatus.VALID
    assert result.metrics["accuracy"] == 1.0
    assert result.metrics["Total Samples"] == 3
    assert result.metrics["async_completed_requests"] == 3
    assert result.metrics["async_evaluator_samples"] == 3
    assert result.details["flush_duration_ms"] >= 0
    assert result.metrics["hw_test_samples"] == 1
    assert monitor.events == ["start", "stop"]


def test_server_like_reports_target_achieved_and_gap():
    result = AsyncBenchmarkRunner(
        dataloader=Loader(),
        runtime=Runtime(),
        evaluator=Evaluator(),
    ).run(
        AsyncInferenceConfig(
            scenario=AsyncScenario.SERVER_LIKE,
            queue_capacity=4,
            target_qps=1_000,
            min_samples=3,
            max_samples=3,
        ),
        warmup_runs=0,
    )

    assert result.metrics["async_target_qps"] == 1_000
    assert result.metrics["async_achieved_qps"] > 0
    assert result.metrics["async_target_qps_gap"] == (
        result.metrics["async_achieved_qps"] - 1_000
    )
```

- [ ] **Step 2: Run the test and verify the missing runner failure**

Run:

```bash
cd framework
python -m pytest tests/test_async_runner.py -q
```

Expected: collection fails because `core.async_inference.runner` does not exist.

- [ ] **Step 3: Implement `AsyncBenchmarkRunner`**

Create `framework/src/core/async_inference/runner.py`:

```python
import time
from numbers import Number

from core.inference_pipeline import InferencePipeline

from .completion import CompletionCoordinator
from .engine import AsyncInferenceEngine
from .metrics import AsyncMetricsCollector
from .producers import OfflineProducer, ServerLikeProducer
from .types import (
    AsyncBenchmarkResult,
    AsyncScenario,
    RunStatus,
)


class AsyncBenchmarkRunner:
    def __init__(
        self,
        dataloader,
        runtime,
        evaluator,
        max_new_tokens=256,
        monitor=None,
        decoder=None,
        trace_callback=None,
    ):
        self.dataloader = dataloader
        self.runtime = runtime
        self.evaluator = evaluator
        self.monitor = monitor
        self.decoder = decoder
        self.trace_callback = trace_callback
        self.pipeline = InferencePipeline(
            dataloader,
            runtime,
            max_new_tokens=max_new_tokens,
        )

    def run(self, config, warmup_runs=1):
        config.validate()
        if warmup_runs > 0:
            warmup_batch = self.dataloader.load_batch(config.max_batch_size)
            if warmup_batch:
                collated = self.pipeline.collate_batch(warmup_batch)
                runtime_input = self.pipeline.prepare_runtime_input(
                    collated["input"]
                )
                self.runtime.warmup(runtime_input, num_runs=warmup_runs)

        metrics = AsyncMetricsCollector(
            time.monotonic_ns(),
            config.worker_count,
            latency_slo_ms=config.latency_slo_ms,
        )
        coordinator = CompletionCoordinator(
            pipeline=self.pipeline,
            evaluator=self.evaluator,
            decoder=self.decoder,
            metrics=metrics,
            queue_capacity=config.worker_count,
            request_timeout_ms=config.request_timeout_ms,
            trace_callback=self.trace_callback,
        )
        engine = AsyncInferenceEngine(
            runtime=self.runtime,
            pipeline=self.pipeline,
            config=config,
            coordinator=coordinator,
            metrics=metrics,
        )
        producer_class = (
            OfflineProducer
            if config.scenario is AsyncScenario.OFFLINE
            else ServerLikeProducer
        )
        producer = producer_class(self.dataloader, engine, config)

        engine.start()
        started_ns = time.monotonic_ns()
        metrics.begin_measurement(started_ns)
        if self.monitor is not None:
            self.monitor.start()
        try:
            producer_result = producer.run()
        except KeyboardInterrupt:
            metrics.add_invalid_reason("producer_error")
            engine.cancel_queued("KeyboardInterrupt")
            producer_result = None
        except Exception:
            metrics.add_invalid_reason("producer_error")
            producer_result = None
        finally:
            engine.close_submission()

        flush_started_ns = time.monotonic_ns()
        flushed = engine.flush()
        flush_finished_ns = time.monotonic_ns()
        if self.monitor is not None:
            self.monitor.stop()
        shutdown = engine.shutdown()
        ended_ns = (
            flush_finished_ns if flushed else time.monotonic_ns()
        )

        collected = metrics.finalize(ended_ns)
        details = collected["details"]
        details["config"] = {
            "scenario": config.scenario.value,
            "queue_capacity": config.queue_capacity,
            "worker_count": config.worker_count,
            "max_batch_size": config.max_batch_size,
            "batch_timeout_ms": config.batch_timeout_ms,
            "submit_timeout_sec": config.submit_timeout_sec,
            "flush_timeout_sec": config.flush_timeout_sec,
            "request_timeout_ms": config.request_timeout_ms,
            "min_samples": config.min_samples,
            "min_duration_sec": config.min_duration_sec,
            "max_samples": config.max_samples,
            "target_qps": config.target_qps,
            "schedule_seed": config.schedule_seed,
            "latency_slo_ms": config.latency_slo_ms,
        }
        if producer_result is not None:
            details["producer"] = {
                "attempted": producer_result.attempted,
                "accepted": producer_result.accepted,
                "rejected": producer_result.rejected,
                "producer_load_ms": producer_result.producer_load_ms,
            }

        completed_samples = collected["summary"]["async_completed_samples"]
        duration = details["measurement_duration_sec"]
        invalid_reasons = set(details["invalid_reasons"])
        if completed_samples == 0:
            invalid_reasons.add("no_samples")
        if completed_samples < config.min_samples:
            invalid_reasons.add("min_samples_not_met")
        if duration < config.min_duration_sec:
            invalid_reasons.add("min_duration_not_met")
        p99 = collected["summary"]["async_e2e_latency_p99_ms"]
        if (
            config.latency_slo_ms is not None
            and p99 is not None
            and p99 > config.latency_slo_ms
        ):
            invalid_reasons.add("latency_slo_not_met")
        if not flushed:
            invalid_reasons.add("flush_timeout")
        if not shutdown:
            invalid_reasons.add("worker_shutdown_failed")
        if completed_samples < 1000:
            details["warnings"] = sorted(
                set(details["warnings"]) | {"tail_percentile_low_sample_count"}
            )

        quality_metrics = self.evaluator.compute()
        final_metrics = dict(quality_metrics)
        final_metrics.update(collected["summary"])
        final_metrics["async_achieved_qps"] = final_metrics[
            "async_completed_samples_per_sec"
        ]
        if config.target_qps is not None:
            final_metrics["async_target_qps"] = config.target_qps
            final_metrics["async_target_qps_gap"] = (
                final_metrics["async_achieved_qps"] - config.target_qps
            )
        evaluator_samples = quality_metrics.get("Total Samples")
        if isinstance(evaluator_samples, Number):
            final_metrics["async_evaluator_samples"] = evaluator_samples
            if evaluator_samples != completed_samples:
                invalid_reasons.add("counter_invariant_failed")
        if self.monitor is not None:
            final_metrics.update(self.monitor.summary())
        status = RunStatus.INVALID if invalid_reasons else RunStatus.VALID
        final_metrics["async_run_status"] = status.value
        final_metrics["async_invalid_reasons"] = ",".join(sorted(invalid_reasons))
        details["invalid_reasons"] = sorted(invalid_reasons)
        details["quality_metrics"] = quality_metrics
        details["evaluator_samples"] = evaluator_samples
        details["status"] = status.value
        details["flush_duration_ms"] = (
            flush_finished_ns - flush_started_ns
        ) / 1_000_000.0

        return AsyncBenchmarkResult(
            metrics=final_metrics,
            details=details,
            status=status,
            invalid_reasons=tuple(sorted(invalid_reasons)),
            warnings=tuple(details["warnings"]),
        )
```

- [ ] **Step 4: Export the runner**

Append to `framework/src/core/async_inference/__init__.py`:

```python
from .runner import AsyncBenchmarkRunner

__all__.append("AsyncBenchmarkRunner")
```

- [ ] **Step 5: Run all async unit and integration tests**

Run:

```bash
cd framework
python -m pytest tests/test_async_types.py tests/test_async_metrics.py tests/test_async_completion.py tests/test_async_engine.py tests/test_async_producers.py tests/test_async_runner.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit the runner**

```bash
git add framework/src/core/async_inference/runner.py framework/src/core/async_inference/__init__.py framework/tests/test_async_runner.py
git commit -m "feat(framework): orchestrate async benchmarks"
```

### Task 8: Persist run IDs, JSON sidecars, and optional bounded JSONL traces

**Files:**
- Create: `framework/src/core/async_inference/trace.py`
- Create: `framework/tests/test_async_result_artifacts.py`
- Modify: `framework/src/core/result_store.py:34-145`

**Interfaces:**
- Consumes: `AsyncBenchmarkResult.details` and `RequestTrace`.
- Produces: `create_run_id()`, backward-compatible `save_result()` async metadata parameters, `save_async_details()`, and `RequestTraceWriter`.

- [ ] **Step 1: Write result artifact tests**

Create `framework/tests/test_async_result_artifacts.py`:

```python
import json

import numpy as np

from core.async_inference.trace import RequestTraceWriter
from core.async_inference.types import RequestTrace, TerminalStatus
from core.result_store import (
    load_results,
    save_async_details,
    save_result,
)


def test_save_result_accepts_preallocated_run_id_and_async_metadata(tmp_path):
    csv_path = tmp_path / "results.csv"
    run_id = save_result(
        run_id="fixed123",
        metrics={"async_completed_requests": 2},
        model_name="tiny",
        task="IMAGE_CLASSIFICATION",
        backend="onnxruntime",
        device="cpu",
        batch_size=1,
        warmup_runs=0,
        inference_mode="async_queue",
        scenario="offline",
        async_run_status="valid",
        results_path=csv_path,
    )

    assert run_id == "fixed123"
    row = load_results(results_path=csv_path)[0]
    assert row["inference_mode"] == "async_queue"
    assert row["scenario"] == "offline"
    assert row["async_run_status"] == "valid"


def test_save_result_migrates_legacy_csv_header_for_async_metadata(tmp_path):
    csv_path = tmp_path / "results.csv"
    csv_path.write_text(
        "run_id,model_name,accuracy\nold00001,tiny,1.0\n",
        encoding="utf-8",
    )

    save_result(
        metrics={"accuracy": 1.0},
        model_name="tiny",
        task="IMAGE_CLASSIFICATION",
        backend="onnxruntime",
        device="cpu",
        batch_size=1,
        warmup_runs=0,
        inference_mode="async_queue",
        scenario="offline",
        results_path=csv_path,
    )

    header = csv_path.read_text(encoding="utf-8").splitlines()[0].split(",")
    assert "inference_mode" in header
    assert "scenario" in header


def test_save_async_details_is_atomic_json(tmp_path):
    path = save_async_details(
        "fixed123",
        {"counts": {"completed": 2}},
        results_dir=tmp_path,
    )
    assert json.loads(path.read_text())["schema_version"] == "1.0"
    assert not path.with_suffix(".json.tmp").exists()


def test_save_async_details_normalizes_numpy_scalars(tmp_path):
    path = save_async_details(
        "fixed123",
        {"quality_metrics": {"accuracy": np.float32(1.0)}},
        results_dir=tmp_path,
    )
    assert json.loads(path.read_text())["quality_metrics"]["accuracy"] == 1.0


def test_trace_writer_never_serializes_payloads(tmp_path):
    writer = RequestTraceWriter(tmp_path / "trace.jsonl", capacity=2)
    writer.start()
    writer.write(
        RequestTrace(
            request_id=1,
            sample_index=2,
            status=TerminalStatus.COMPLETED,
            scheduled_ns=1,
            issued_ns=2,
            enqueued_ns=3,
            runtime_started_ns=4,
            runtime_finished_ns=5,
            completed_ns=6,
            worker_id=0,
            batch_size=1,
            timed_out=False,
        )
    )
    assert writer.close(timeout=1.0) is True

    text = (tmp_path / "trace.jsonl").read_text()
    assert "request_id" in text
    assert "input" not in text
    assert "output" not in text
```

- [ ] **Step 2: Run tests and verify missing APIs**

Run:

```bash
cd framework
python -m pytest tests/test_async_result_artifacts.py -q
```

Expected: import errors for `RequestTraceWriter` and `save_async_details`.

- [ ] **Step 3: Implement the bounded trace writer**

Create `framework/src/core/async_inference/trace.py`:

```python
import json
import queue
import threading
from dataclasses import asdict
from pathlib import Path


_STOP = object()


class RequestTraceWriter:
    def __init__(self, path, capacity=1024):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.queue = queue.Queue(maxsize=capacity)
        self.dropped = 0
        self.thread = threading.Thread(
            target=self._run,
            name="async-trace-writer",
            daemon=True,
        )

    def start(self):
        self.thread.start()

    def write(self, trace):
        row = asdict(trace)
        row["status"] = trace.status.value
        try:
            self.queue.put_nowait(row)
        except queue.Full:
            self.dropped += 1

    def close(self, timeout):
        try:
            self.queue.put(_STOP, timeout=timeout)
        except queue.Full:
            return False
        self.thread.join(timeout=timeout)
        return not self.thread.is_alive()

    def _run(self):
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        with open(tmp_path, "w", encoding="utf-8") as handle:
            while True:
                row = self.queue.get()
                try:
                    if row is _STOP:
                        break
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                finally:
                    self.queue.task_done()
            handle.flush()
        tmp_path.replace(self.path)
```

- [ ] **Step 4: Extend `result_store` with stable async metadata and sidecars**

Add imports:

```python
import json
```

Extend `META_COLUMNS` after `artifact_format`:

```python
    "inference_mode",
    "scenario",
    "queue_capacity",
    "worker_count",
    "batch_timeout_ms",
    "target_qps",
    "schedule_seed",
    "async_run_status",
    "async_invalid_reasons",
    "details_path",
    "request_trace_path",
```

Add these helpers:

```python
def create_run_id() -> str:
    return str(uuid.uuid4())[:8]


def _json_default(value):
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def save_async_details(
    run_id: str,
    details: Dict[str, Any],
    results_dir: Optional[Path] = None,
) -> Path:
    root = Path(results_dir) if results_dir is not None else DEFAULT_RESULTS_DIR
    path = root / "details" / f"{run_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".json.tmp")
    payload = {
        **details,
        "schema_version": "1.0",
        "run_id": run_id,
    }
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(
            payload,
            handle,
            ensure_ascii=False,
            indent=2,
            default=_json_default,
        )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, path)
    return path
```

Add optional arguments to `save_result()`:

```python
    run_id: Optional[str] = None,
    inference_mode: str = "e2e",
    scenario: str = "",
    queue_capacity: Optional[int] = None,
    worker_count: Optional[int] = None,
    batch_timeout_ms: Optional[float] = None,
    target_qps: Optional[float] = None,
    schedule_seed: Optional[int] = None,
    async_run_status: str = "",
    async_invalid_reasons: str = "",
    details_path: str = "",
    request_trace_path: str = "",
```

Replace run ID creation with:

```python
    run_id = run_id or create_run_id()
```

Add these values to the metadata row:

```python
        "inference_mode": inference_mode,
        "scenario": scenario,
        "queue_capacity": "" if queue_capacity is None else queue_capacity,
        "worker_count": "" if worker_count is None else worker_count,
        "batch_timeout_ms": "" if batch_timeout_ms is None else batch_timeout_ms,
        "target_qps": "" if target_qps is None else target_qps,
        "schedule_seed": "" if schedule_seed is None else schedule_seed,
        "async_run_status": async_run_status,
        "async_invalid_reasons": async_invalid_reasons,
        "details_path": details_path,
        "request_trace_path": request_trace_path,
```

Replace the existing header merge block so newly introduced metadata columns are also migrated into an already-existing CSV:

```python
        metric_keys = [key for key in row if key not in META_COLUMNS]
        if existing_columns:
            new_meta_keys = [
                key for key in META_COLUMNS if key not in existing_columns
            ]
            new_metric_keys = [
                key for key in metric_keys if key not in existing_columns
            ]
            all_columns = existing_columns + new_meta_keys + new_metric_keys
        else:
            all_columns = META_COLUMNS + metric_keys
```

- [ ] **Step 5: Run artifact and existing result-store tests**

Run:

```bash
cd framework
python -m pytest tests/test_async_result_artifacts.py tests/test_result_store.py -q
```

Expected: all tests pass, including old callers that do not provide a run ID.

- [ ] **Step 6: Commit result artifacts**

```bash
git add framework/src/core/async_inference/trace.py framework/src/core/result_store.py framework/tests/test_async_result_artifacts.py
git commit -m "feat(framework): persist async benchmark artifacts"
```

### Task 9: Add CLI selection and conditional async validation

**Files:**
- Create: `framework/tests/test_async_cli.py`
- Modify: `framework/src/main.py:1-527`

**Interfaces:**
- Consumes: `AsyncInferenceConfig`, `AsyncBenchmarkRunner`, `create_run_id()`, `save_async_details()`.
- Produces: `build_parser()`, `validate_async_args(args)`, preserved `RUN_ID=<run-id>` output, and branch selection by `--inference-mode`.

- [ ] **Step 1: Write CLI parsing and validation tests**

Create `framework/tests/test_async_cli.py`:

```python
import pytest

import main as benchmark_main


def parse(extra):
    parser = benchmark_main.build_parser()
    return parser.parse_args(["--model", "resnet50", *extra])


def test_default_inference_mode_is_e2e():
    args = parse([])
    assert args.inference_mode == "e2e"


def test_server_like_requires_target_qps():
    args = parse([
        "--inference-mode", "async_queue",
        "--scenario", "server_like",
    ])
    with pytest.raises(ValueError, match="target-qps"):
        benchmark_main.validate_async_args(args)


def test_e2e_rejects_async_only_options():
    args = parse(["--queue-capacity", "16"])
    with pytest.raises(ValueError, match="async_queue"):
        benchmark_main.validate_async_args(args)


def test_e2e_rejects_explicit_async_scenario():
    args = parse(["--scenario", "offline"])
    with pytest.raises(ValueError, match="async_queue"):
        benchmark_main.validate_async_args(args)


def test_async_rejects_max_steps_and_points_to_max_samples():
    args = parse([
        "--inference-mode", "async_queue",
        "--max-steps", "2",
    ])
    with pytest.raises(ValueError, match="max-samples"):
        benchmark_main.validate_async_args(args)
```

- [ ] **Step 2: Run tests and verify `build_parser` is missing**

Run:

```bash
cd framework
python -m pytest tests/test_async_cli.py -q
```

Expected: tests fail with `AttributeError: module 'main' has no attribute 'build_parser'`.

- [ ] **Step 3: Extract parser construction and add async arguments**

Move the existing parser declarations from `main()` into:

```python
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified BenchmarkRunner CLI Orchestrator"
    )
```

Keep every existing `add_argument` call, then add:

```python
    parser.add_argument(
        "--inference-mode",
        choices=["e2e", "async_queue"],
        default="e2e",
        help="추론 실행 방식 (기본: e2e)",
    )
    parser.add_argument(
        "--scenario",
        choices=["offline", "server_like"],
        default=None,
        help="async_queue 부하 시나리오 (미지정 시 offline)",
    )
    parser.add_argument("--target-qps", type=float, default=None)
    parser.add_argument("--queue-capacity", type=int, default=None)
    parser.add_argument("--worker-count", type=int, default=None)
    parser.add_argument("--batch-timeout-ms", type=float, default=None)
    parser.add_argument("--submit-timeout-sec", type=float, default=None)
    parser.add_argument("--flush-timeout-sec", type=float, default=None)
    parser.add_argument("--request-timeout-ms", type=float, default=None)
    parser.add_argument("--min-samples", type=int, default=None)
    parser.add_argument("--min-duration-sec", type=float, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--schedule-seed", type=int, default=None)
    parser.add_argument("--latency-slo-ms", type=float, default=None)
    parser.add_argument("--save-request-trace", action="store_true")
    return parser
```

Add conditional validation:

```python
ASYNC_ONLY_ARGUMENTS = {
    "scenario",
    "target_qps",
    "queue_capacity",
    "worker_count",
    "batch_timeout_ms",
    "submit_timeout_sec",
    "flush_timeout_sec",
    "request_timeout_ms",
    "min_samples",
    "min_duration_sec",
    "max_samples",
    "schedule_seed",
    "latency_slo_ms",
}


def validate_async_args(args: argparse.Namespace) -> None:
    if args.inference_mode == "e2e":
        supplied = [
            name
            for name in ASYNC_ONLY_ARGUMENTS
            if getattr(args, name) is not None
        ]
        if args.save_request_trace:
            supplied.append("save_request_trace")
        if supplied:
            raise ValueError(
                f"async_queue 전용 옵션입니다: {', '.join(sorted(supplied))}"
            )
        return
    if args.max_steps is not None:
        raise ValueError(
            "async_queue에서는 --max-steps 대신 --max-samples를 사용하세요"
        )
    if (args.scenario or "offline") == "server_like" and args.target_qps is None:
        raise ValueError("server_like에는 --target-qps가 필요합니다")
```

Start `main()` with:

```python
    parser = build_parser()
    args = parser.parse_args()
    try:
        validate_async_args(args)
    except ValueError as exc:
        parser.error(str(exc))
```

- [ ] **Step 4: Build async config and branch runner creation**

Add imports:

```python
from core.async_inference import (
    AsyncInferenceConfig,
    AsyncScenario,
    AsyncBenchmarkRunner,
)
from core.async_inference.trace import RequestTraceWriter
from core.result_store import create_run_id, save_async_details
```

Replace the current runner block with:

```python
    run_id = create_run_id()
    async_result = None
    trace_writer = None
    trace_path = ""
    if args.inference_mode == "e2e":
        runner = BenchmarkRunner(
            dataloader=loader,
            runtime=runtime,
            evaluator=evaluator,
            max_new_tokens=args.max_new_tokens,
            monitor=hw_monitor,
            decoder=decoder,
        )
        results = runner.run(
            warmup_runs=args.warmup,
            batch_size=args.batch_size,
            max_steps=args.max_steps,
        )
    else:
        scenario_name = args.scenario or "offline"
        min_samples = args.min_samples if args.min_samples is not None else 100
        min_duration = args.min_duration_sec
        if min_duration is None:
            min_duration = 10.0 if scenario_name == "server_like" else 0.0
        config = AsyncInferenceConfig(
            scenario=AsyncScenario(scenario_name),
            queue_capacity=(
                args.queue_capacity if args.queue_capacity is not None else 256
            ),
            worker_count=(
                args.worker_count if args.worker_count is not None else 1
            ),
            max_batch_size=args.batch_size,
            batch_timeout_ms=(
                args.batch_timeout_ms
                if args.batch_timeout_ms is not None
                else 1.0
            ),
            submit_timeout_sec=(
                args.submit_timeout_sec
                if args.submit_timeout_sec is not None
                else 30.0
            ),
            flush_timeout_sec=(
                args.flush_timeout_sec
                if args.flush_timeout_sec is not None
                else 300.0
            ),
            request_timeout_ms=(
                args.request_timeout_ms
                if args.request_timeout_ms is not None
                else 0.0
            ),
            min_samples=min_samples,
            min_duration_sec=min_duration,
            max_samples=args.max_samples,
            target_qps=args.target_qps,
            schedule_seed=(
                args.schedule_seed if args.schedule_seed is not None else 0
            ),
            latency_slo_ms=args.latency_slo_ms,
        )
        config.validate()
        if args.save_request_trace:
            trace_file = FRAMEWORK_ROOT / "results" / "traces" / f"{run_id}.jsonl"
            trace_writer = RequestTraceWriter(trace_file)
            trace_writer.start()
            trace_path = str(trace_file.relative_to(FRAMEWORK_ROOT))
        runner = AsyncBenchmarkRunner(
            dataloader=loader,
            runtime=runtime,
            evaluator=evaluator,
            max_new_tokens=args.max_new_tokens,
            monitor=hw_monitor,
            decoder=decoder,
            trace_callback=trace_writer.write if trace_writer else None,
        )
        trace_closed = True
        try:
            async_result = runner.run(config, warmup_runs=args.warmup)
            results = async_result.metrics
        finally:
            if trace_writer is not None:
                trace_closed = trace_writer.close(
                    timeout=config.flush_timeout_sec
                )
        if trace_writer is not None:
            if not trace_closed:
                async_result.details["warnings"].append(
                    "request_trace_close_timeout"
                )
                trace_path = ""
            if trace_writer.dropped:
                async_result.details["warnings"].append(
                    f"request_trace_dropped:{trace_writer.dropped}"
                )
```

- [ ] **Step 5: Persist the sidecar and expanded metadata**

Move the existing `target_meta = target_metadata(...)` assignment above the sidecar block so it is computed once, then place this immediately before `save_result()`:

```python
    target_meta = target_metadata(target, compile_metadata)
    details_path = ""
    if async_result is not None:
        async_result.details["run"] = {
            "model_name": args.model,
            "task": task_enum.name,
            "backend": args.backend,
            "device": args.device,
            "batch_size": args.batch_size,
            "warmup_runs": args.warmup,
            "target_id": target_meta["target_id"],
        }
        async_result.details["hardware_metrics"] = {
            key: value
            for key, value in results.items()
            if key.startswith("hw_")
        }
        details_file = save_async_details(run_id, async_result.details)
        details_path = str(details_file.relative_to(FRAMEWORK_ROOT))
```

Pass the following additional arguments to `save_result()`:

```python
        run_id=run_id,
        inference_mode=args.inference_mode,
        scenario=scenario_name if args.inference_mode == "async_queue" else "",
        queue_capacity=(
            config.queue_capacity if args.inference_mode == "async_queue" else None
        ),
        worker_count=(
            config.worker_count if args.inference_mode == "async_queue" else None
        ),
        batch_timeout_ms=(
            config.batch_timeout_ms if args.inference_mode == "async_queue" else None
        ),
        target_qps=(
            config.target_qps if args.inference_mode == "async_queue" else None
        ),
        schedule_seed=(
            config.schedule_seed if args.inference_mode == "async_queue" else None
        ),
        async_run_status=(
            async_result.status.value if async_result is not None else ""
        ),
        async_invalid_reasons=(
            ",".join(async_result.invalid_reasons)
            if async_result is not None
            else ""
        ),
        details_path=details_path,
        request_trace_path=trace_path,
```

Retain the existing `RUN_ID={run_id}` line. If async flush timed out and outstanding work remains, skip `runtime.unload()` and exit non-zero after artifacts are saved. Otherwise unload normally, then return a non-zero exit status for any invalid async run:

```python
    outstanding = results.get("async_outstanding_requests", 0)
    if outstanding:
        print(
            f"[Error] runtime unload skipped: {outstanding} requests are still active",
            file=sys.stderr,
        )
        sys.exit(1)
    runtime.unload()
    if async_result is not None and async_result.status.value == "invalid":
        sys.exit(1)
```

- [ ] **Step 6: Run CLI, path, and result tests**

Run:

```bash
cd framework
python -m pytest tests/test_async_cli.py tests/test_main_paths.py tests/test_result_store.py tests/test_async_result_artifacts.py -q
```

Expected: all tests pass.

- [ ] **Step 7: Commit CLI integration**

```bash
git add framework/src/main.py framework/tests/test_async_cli.py
git commit -m "feat(framework): expose async inference CLI"
```

### Task 10: Verify real ONNX Runtime CPU execution and document measurement semantics

**Files:**
- Create: `framework/tests/test_async_onnx_cpu.py`
- Create: `docs/async-inference-queue.md`
- Modify: `framework/README.md`

**Interfaces:**
- Consumes: complete async runner and `OnnxRuntime(device="cpu")`.
- Produces: a real CPU integration proof, usage documentation, and explicit non-MLPerf wording.

- [ ] **Step 1: Write a self-contained ONNX CPU integration test**

Create `framework/tests/test_async_onnx_cpu.py`:

```python
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper

from core.async_inference.runner import AsyncBenchmarkRunner
from core.async_inference.types import AsyncInferenceConfig, RunStatus
from core.benchmarkrunner import BenchmarkRunner
from core.compiled_model import CompiledModel
from core.model_spec import Model_Spec, Task
from runtimes.onnx_rt import OnnxRuntime


class Loader:
    def __init__(self):
        self.current_idx = 0
        self.samples = [
            {"input": np.array([1.0, 2.0], dtype=np.float32), "label": 3.0},
            {"input": np.array([3.0, 4.0], dtype=np.float32), "label": 7.0},
            {"input": np.array([5.0, 6.0], dtype=np.float32), "label": 11.0},
            {"input": np.array([7.0, 8.0], dtype=np.float32), "label": 15.0},
        ]

    def get_metadata(self):
        return {"total_samples": len(self.samples), "is_static_batched": False}

    def load_by_index(self, index):
        return self.samples[index]

    def load_batch(self, batch_size):
        rows = self.samples[self.current_idx:self.current_idx + batch_size]
        self.current_idx += len(rows)
        return rows


class SumEvaluator:
    def __init__(self):
        self.correct = 0
        self.total = 0

    def add_batch(self, outputs, labels, timing_ms):
        predicted = outputs["output"].reshape(-1)
        expected = np.asarray(labels)
        self.correct += int(np.sum(predicted == expected))
        self.total += len(expected)

    def compute(self):
        return {
            "accuracy": self.correct / self.total,
            "Total Samples": self.total,
        }


def create_sum_model(path):
    input_info = helper.make_tensor_value_info(
        "input",
        TensorProto.FLOAT,
        [None, 2],
    )
    output_info = helper.make_tensor_value_info(
        "output",
        TensorProto.FLOAT,
        [None, 1],
    )
    node = helper.make_node(
        "ReduceSum",
        inputs=["input"],
        outputs=["output"],
        axes=[1],
        keepdims=1,
    )
    graph = helper.make_graph([node], "sum", [input_info], [output_info])
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 11)],
    )
    model.ir_version = 8
    onnx.save(model, path)


def runtime_for(path):
    spec = Model_Spec(
        name="tiny-sum",
        task=Task.IMAGE_CLASSIFICATION,
        input_shapes={"input": (None, 2)},
        input_dtype={"input": "float32"},
        output_shapes={"output": (None, 1)},
        model_paths={"onnx": str(path)},
    )
    compiled = CompiledModel(
        spec=spec,
        backend_name="onnxruntime",
        artifact_path=Path(path),
    )
    runtime = OnnxRuntime(device="cpu")
    runtime.load(compiled)
    return runtime


def test_async_onnx_cpu_matches_e2e_quality_and_uses_dynamic_batches(tmp_path):
    model_path = tmp_path / "sum.onnx"
    create_sum_model(model_path)

    e2e_runtime = runtime_for(model_path)
    e2e = BenchmarkRunner(
        Loader(),
        e2e_runtime,
        SumEvaluator(),
    ).run(warmup_runs=0, batch_size=2)
    e2e_runtime.unload()

    async_runtime = runtime_for(model_path)
    assert async_runtime.session.get_providers()[0] == "CPUExecutionProvider"
    result = AsyncBenchmarkRunner(
        Loader(),
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
    async_runtime.unload()

    assert result.status is RunStatus.VALID
    assert result.metrics["accuracy"] == e2e["accuracy"] == 1.0
    assert result.metrics["Total Samples"] == e2e["Total Samples"] == 4
    assert result.details["batch_size"]["max"] == 2.0
    assert result.metrics["async_outstanding_requests"] == 0
```

- [ ] **Step 2: Run the ONNX CPU test**

Run:

```bash
cd framework
python -m pytest tests/test_async_onnx_cpu.py -q
```

Expected: one passing test using `CPUExecutionProvider`; no CUDA requirement and no performance threshold.

- [ ] **Step 3: Document CLI examples and interpretation**

Add to `framework/README.md`:

````markdown
## 비동기 추론 큐

기존 순차 실행은 기본값이며 그대로 사용할 수 있습니다.

```bash
python src/main.py --model resnet50 --inference-mode e2e
```

Offline형 비동기 큐는 가능한 한 빠르게 요청을 공급합니다.

```bash
python src/main.py \
  --model resnet50 \
  --inference-mode async_queue \
  --scenario offline \
  --batch-size 4 \
  --queue-capacity 256 \
  --worker-count 1 \
  --batch-timeout-ms 1
```

Server-like 부하는 seed 기반 요청 간격으로 target QPS를 재현합니다.

```bash
python src/main.py \
  --model resnet50 \
  --inference-mode async_queue \
  --scenario server_like \
  --target-qps 100 \
  --min-duration-sec 10 \
  --min-samples 100
```

`async_queue` 결과는 MLPerf 결과가 아닙니다. MLPerf LoadGen은 신뢰성
설계를 위한 레퍼런스로만 사용했으며, 프레임워크는 LoadGen을 import하거나
SUT/QSL 및 공식 validity 규칙을 구현하지 않습니다.

지표 정의와 결과 해석은 [비동기 추론 큐 측정 가이드](../docs/async-inference-queue.md)를 참고하세요.
````

- [ ] **Step 4: Create the dedicated async measurement guide**

Create `docs/async-inference-queue.md`:

````markdown
# 비동기 추론 큐 측정 가이드

이 모드는 MLPerf 결과나 LoadGen 호환 구현이 아니다. LoadGen에서 검증된 요청
발행/완료 분리, exact-once completion, monotonic timing, validity 원칙을
설계 레퍼런스로 사용한 프레임워크 자체 측정 모듈이다.

## 측정 경계

비동기 모드의 `async_e2e_latency`는 request submit 호출부터 completion
coordinator가 terminal 상태를 기록할 때까지다.

```text
async_e2e_latency
  = submit_wait
  + queue_wait
  + service_time
  + completion_overhead
```

- `submit_wait`: bounded queue 공간을 기다린 시간
- `queue_wait`: queue 진입 후 runtime 시작까지의 시간
- `service_time`: runtime 호출 시간
- `completion_overhead`: decoder, evaluator, terminal 처리 시간

Server-like의 `async_target_qps_gap`은 `achieved_qps - target_qps`다. 음수 폭이
커지고 queue wait/P99가 함께 증가하면 해당 설정에서 포화가 시작된 것으로
해석한다. 표본이 1,000개보다 적은 P99.9에는
`tail_percentile_low_sample_count` 경고가 함께 저장된다.

기존 `Average Latency (ms)`는 runtime 중심의 batch latency이므로
`async_e2e_latency`와 같은 값으로 해석하지 않는다. 처리량이 증가해도 queue
wait과 P99가 증가할 수 있으며, 이 변화는 최적화 실패가 아니라 포화 상태를
드러낸 결과일 수 있다.
````

- [ ] **Step 5: Run focused and full verification**

Run:

```bash
cd framework
python -m pytest tests/test_async_types.py tests/test_async_metrics.py tests/test_async_completion.py tests/test_async_engine.py tests/test_async_producers.py tests/test_async_runner.py tests/test_async_result_artifacts.py tests/test_async_cli.py tests/test_async_onnx_cpu.py -q
```

Expected: all async tests pass.

Run:

```bash
cd framework
python -m pytest tests -q
```

Expected: all environment-independent tests pass; hardware/model-dependent tests retain their existing skip behavior.

Run:

```bash
git diff --check
```

Expected: no whitespace errors.

- [ ] **Step 6: Commit integration verification and docs**

```bash
git add framework/tests/test_async_onnx_cpu.py framework/README.md docs/async-inference-queue.md
git commit -m "test(framework): verify async inference on CPU"
```

### Task 11: Final regression, artifact inspection, and plan acceptance

**Files:**
- Verify: `framework/src/core/async_inference/`
- Verify: `framework/src/main.py`
- Verify: `framework/src/core/result_store.py`
- Verify: `framework/results/`

**Interfaces:**
- Consumes: the complete implementation.
- Produces: evidence that the implementation satisfies the approved design and has not introduced MLPerf coupling.

- [ ] **Step 1: Prove there is no MLPerf runtime dependency**

Run:

```bash
rg -n "import mlperf|from mlperf|mlperf_loadgen|ConstructSUT|ConstructQSL" framework/src/core/async_inference framework/src/main.py framework/src/core/result_store.py
```

Expected: no matches.

- [ ] **Step 2: Prove the default remains e2e**

Run:

```bash
cd framework
python src/main.py --help
```

Expected: help lists `--inference-mode {e2e,async_queue}` and identifies `e2e` as the default.

- [ ] **Step 3: Run the final framework test suite**

Run:

```bash
cd framework
python -m pytest tests -q
```

Expected: exit code 0, with only pre-existing environment-dependent skips.

- [ ] **Step 4: Inspect one async artifact set**

First confirm the zero-config assets are already local, so this acceptance step cannot trigger a download:

```bash
cd framework
test -f models/Kalray_resnet50/resnet50-v1-7s.onnx
test -d datasets/imagenet_1k
```

If either check fails, record the manual CLI inspection as skipped and rely on the self-contained ONNX CPU integration test from Task 10. Otherwise run:

```bash
cd framework
python src/main.py --model resnet50 --inference-mode async_queue --scenario offline --max-samples 4 --min-samples 1 --batch-size 2 --warmup 0
```

Expected:

- stdout ends with `RUN_ID=<8-character id>`;
- CSV row has `inference_mode=async_queue` and `async_run_status=valid`;
- `results/details/<run-id>.json` exists;
- sidecar `counts` satisfy the counter invariant;
- no `results/traces/<run-id>.jsonl` exists because trace was not requested.

- [ ] **Step 5: Inspect repository status and commit the final review fixes only if needed**

Run:

```bash
git status --short
git log --oneline -10
```

Expected: implementation commits are visible as separate reviewable units. The user's pre-existing changes and the deliberately generated local benchmark row/sidecar may remain uncommitted; do not stage, overwrite, or clean them as part of this plan.
