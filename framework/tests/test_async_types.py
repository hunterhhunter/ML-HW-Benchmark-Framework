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


@pytest.mark.parametrize(
    "value",
    [
        True,
        False,
        0,
        -1,
        1.5,
        np.float64(1.0),
        float("nan"),
        float("inf"),
        float("-inf"),
        "1",
        None,
    ],
)
def test_min_samples_must_be_a_positive_integral_count(value):
    with pytest.raises(ValueError, match="min_samples"):
        AsyncInferenceConfig(min_samples=value).validate()


@pytest.mark.parametrize(
    "value",
    [
        True,
        False,
        0,
        -1,
        1.5,
        np.float64(1.0),
        float("nan"),
        float("inf"),
        float("-inf"),
        "1",
    ],
)
def test_non_none_max_samples_must_be_a_positive_integral_count(value):
    with pytest.raises(ValueError, match="max_samples"):
        AsyncInferenceConfig(max_samples=value).validate()


@pytest.mark.parametrize(
    "value",
    [1, 7, np.int32(2), np.int64(3), np.uint64(4)],
)
def test_sample_constraints_accept_python_and_numpy_integral_counts(value):
    AsyncInferenceConfig(
        min_samples=value,
        max_samples=value,
    ).validate()


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
