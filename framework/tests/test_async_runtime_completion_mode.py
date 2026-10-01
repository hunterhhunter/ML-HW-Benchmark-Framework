"""The performance completion path must not execute quality callbacks."""

import numpy as np

from core.async_inference.completion import CompletionCoordinator
from core.async_inference.metrics import AsyncMetricsCollector
from core.async_inference.types import (
    AsyncPassKind,
    BatchCompletion,
    InferenceRequest,
)


class ExplodingPipeline:
    def prepare_eval_labels(self, collated):
        raise AssertionError("quality labels must not be prepared")


class ExplodingDecoder:
    def decode(self, outputs):
        raise AssertionError("decoder must not run in runtime-only pass")


class ExplodingEvaluator:
    def add_batch(self, outputs, labels, timing_ms):
        raise AssertionError("evaluator must not run in runtime-only pass")


def test_runtime_only_completion_retires_request_without_quality_callbacks():
    metrics = AsyncMetricsCollector(0, 1, pass_kind=AsyncPassKind.RUNTIME_ONLY)
    coordinator = CompletionCoordinator(
        pipeline=ExplodingPipeline(),
        evaluator=ExplodingEvaluator(),
        decoder=ExplodingDecoder(),
        metrics=metrics,
        queue_capacity=None,
        pass_kind=AsyncPassKind.RUNTIME_ONLY,
    )
    request = InferenceRequest(
        request_id=0,
        sample_index=0,
        sample={},
        scheduled_ns=0,
        issued_ns=0,
        enqueued_ns=1,
    )
    coordinator.register(request)
    metrics.record_runtime_completion(
        request_count=1,
        sample_count=1,
        generated_tokens=0,
        finished_ns=3,
    )
    coordinator.submit(BatchCompletion(
        requests=[request],
        collated={},
        outputs={"output": np.asarray([[1]])},
        timing_ms=1.0,
        runtime_started_ns=2,
        runtime_finished_ns=3,
        worker_id=0,
        batch_size=1,
    ))

    assert coordinator.snapshot_outstanding() == ()
    assert metrics.finalize(10, producer_finished_ns=3)["summary"][
        "async_completed_requests"
    ] == 1


def test_empty_output_dict_is_failed_terminal_not_success():
    metrics = AsyncMetricsCollector(0, 1, pass_kind=AsyncPassKind.RUNTIME_ONLY)
    coordinator = CompletionCoordinator(
        pipeline=ExplodingPipeline(),
        evaluator=ExplodingEvaluator(),
        decoder=ExplodingDecoder(),
        metrics=metrics,
        queue_capacity=None,
        pass_kind=AsyncPassKind.RUNTIME_ONLY,
    )
    request = InferenceRequest(
        request_id=1,
        sample_index=0,
        sample={},
        scheduled_ns=0,
        issued_ns=0,
        enqueued_ns=1,
    )
    coordinator.register(request)
    coordinator.submit(BatchCompletion(
        requests=[request],
        collated={},
        outputs={},
        timing_ms=1.0,
        runtime_started_ns=2,
        runtime_finished_ns=3,
        worker_id=0,
        batch_size=1,
    ))

    summary = metrics.finalize(10, producer_finished_ns=3)["summary"]
    assert summary["async_completed_requests"] == 0
    assert summary["async_failed_requests"] == 1
    assert summary["async_runtime_completed_samples"] == 0
