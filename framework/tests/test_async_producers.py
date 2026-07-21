import numpy as np
import pytest

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
    def __init__(self, decisions=None, clock=None, submit_delay_sec=0):
        self.requests = []
        self.decisions = iter(decisions) if decisions is not None else None
        self.clock = clock
        self.submit_delay_sec = submit_delay_sec

    def submit(self, request, block):
        self.requests.append((request, block))
        if self.clock is not None:
            self.clock.sleep(self.submit_delay_sec)
        if self.decisions is None:
            return True
        return next(self.decisions)


class TimedLoader(Loader):
    def __init__(self, clock, load_delay_sec):
        self.clock = clock
        self.load_delay_sec = load_delay_sec

    def load_by_index(self, index):
        self.clock.sleep(self.load_delay_sec)
        return super().load_by_index(index)


class InputLoader:
    def __init__(self, input_value, *, is_static_batched, label=9):
        self.sample = {"input": input_value, "label": label}
        self.is_static_batched = is_static_batched

    def get_metadata(self):
        return {
            "total_samples": 1,
            "is_static_batched": self.is_static_batched,
        }

    def load_by_index(self, index):
        assert index == 0
        return self.sample


class MetadataLoader(Loader):
    def __init__(self, metadata):
        self.metadata = metadata

    def get_metadata(self):
        return self.metadata


class PreparationTimedArray:
    def __init__(self, values, clock, delay_sec):
        self.values = np.asarray(values)
        self.clock = clock
        self.delay_sec = delay_sec

    def __array__(self, dtype=None, copy=None):
        self.clock.sleep(self.delay_sec)
        result = np.asarray(self.values, dtype=dtype)
        if copy:
            result = result.copy()
        return result


def test_offline_submits_each_sample_once_with_blocking_backpressure():
    submitter = Submitter()
    producer = OfflineProducer(
        Loader(),
        submitter,
        AsyncInferenceConfig(min_samples=1),
        clock=FakeableClock(),
    )

    result = producer.run()

    assert result.attempted == 3
    assert result.accepted == 3
    assert result.rejected == 0
    assert [request.sample_index for request, _ in submitter.requests] == [
        0,
        1,
        2,
    ]
    assert all(block is True for _, block in submitter.requests)


@pytest.mark.parametrize(
    ("max_samples", "expected_indexes"),
    [
        (None, [0, 1, 2]),
        (3, [0, 1, 2]),
        (2, [0, 1]),
        (5, [0, 1, 2]),
    ],
)
def test_offline_stops_at_dataset_or_max_samples_without_repeating(
    max_samples,
    expected_indexes,
):
    submitter = Submitter()
    result = OfflineProducer(
        Loader(),
        submitter,
        AsyncInferenceConfig(min_samples=1, max_samples=max_samples),
        clock=FakeableClock(),
    ).run()

    assert result.attempted == len(expected_indexes)
    assert [
        request.sample_index for request, _ in submitter.requests
    ] == expected_indexes


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

    first_schedule = [
        request.scheduled_ns for request, _ in first_submitter.requests
    ]
    second_schedule = [
        request.scheduled_ns for request, _ in second_submitter.requests
    ]
    assert first_schedule == second_schedule
    assert first_schedule[0] == 0
    assert first_schedule == sorted(first_schedule)
    assert all(block is False for _, block in first_submitter.requests)


def test_offline_records_outcomes_and_only_dataloader_time_as_producer_load():
    clock = FakeableClock()
    submitter = Submitter(
        decisions=[True, False, True],
        clock=clock,
        submit_delay_sec=0.007,
    )
    producer = OfflineProducer(
        TimedLoader(clock, load_delay_sec=0.002),
        submitter,
        AsyncInferenceConfig(min_samples=1),
        clock=clock,
    )

    result = producer.run()

    assert result.attempted == 3
    assert result.accepted == 2
    assert result.rejected == 1
    assert result.producer_load_ms == 6.0


def test_offline_timestamps_are_captured_after_loading_and_before_submit():
    clock = FakeableClock()
    submitter = Submitter(
        clock=clock,
        submit_delay_sec=0.007,
    )
    OfflineProducer(
        TimedLoader(clock, load_delay_sec=0.002),
        submitter,
        AsyncInferenceConfig(min_samples=1),
        clock=clock,
    ).run()

    timestamps = [
        (request.scheduled_ns, request.issued_ns, request.enqueued_ns)
        for request, _ in submitter.requests
    ]
    assert timestamps == [
        (2_000_000, 2_000_000, 0),
        (11_000_000, 11_000_000, 0),
        (20_000_000, 20_000_000, 0),
    ]


def test_server_like_waits_for_min_samples_and_min_duration():
    clock = FakeableClock()
    submitter = Submitter(decisions=[True, False, True])
    result = ServerLikeProducer(
        TimedLoader(clock, load_delay_sec=0.002),
        submitter,
        AsyncInferenceConfig(
            scenario=AsyncScenario.SERVER_LIKE,
            target_qps=1,
            min_samples=2,
            min_duration_sec=1,
            schedule_seed=1,
        ),
        clock=clock,
    ).run()

    assert result.attempted == 3
    assert result.accepted == 2
    assert result.rejected == 1
    assert result.producer_load_ms == 6.0
    assert submitter.requests[-1][0].issued_ns >= 1_000_000_000


def test_server_like_min_duration_starts_at_first_request_issue():
    clock = FakeableClock()
    submitter = Submitter()

    result = ServerLikeProducer(
        TimedLoader(clock, load_delay_sec=2),
        submitter,
        AsyncInferenceConfig(
            scenario=AsyncScenario.SERVER_LIKE,
            target_qps=1,
            min_samples=1,
            min_duration_sec=1,
            max_samples=2,
            schedule_seed=1,
        ),
        clock=clock,
    ).run()

    issued_ns = [
        request.issued_ns for request, _ in submitter.requests
    ]
    assert result.attempted == 2
    assert issued_ns == [2_000_000_000, 4_000_000_000]
    assert issued_ns[-1] - issued_ns[0] >= 1_000_000_000


def test_server_like_max_samples_bounds_run_and_cycles_sample_indexes():
    clock = FakeableClock()
    submitter = Submitter()
    result = ServerLikeProducer(
        TimedLoader(clock, load_delay_sec=0.001),
        submitter,
        AsyncInferenceConfig(
            scenario=AsyncScenario.SERVER_LIKE,
            target_qps=10,
            min_samples=100,
            min_duration_sec=100,
            max_samples=5,
            schedule_seed=3,
        ),
        clock=clock,
    ).run()

    assert result.attempted == 5
    assert result.accepted == 5
    assert result.rejected == 0
    assert result.producer_load_ms == 5.0
    assert [
        request.sample_index for request, _ in submitter.requests
    ] == [0, 1, 2, 0, 1]


def test_server_like_never_issues_before_its_nanosecond_deadline():
    submitter = Submitter()
    ServerLikeProducer(
        Loader(),
        submitter,
        AsyncInferenceConfig(
            scenario=AsyncScenario.SERVER_LIKE,
            target_qps=14_650_450,
            min_samples=100,
            min_duration_sec=100,
            max_samples=2,
            schedule_seed=0,
        ),
        clock=FakeableClock(),
    ).run()

    assert [
        request.scheduled_ns for request, _ in submitter.requests
    ] == [0, 126]
    assert all(
        request.issued_ns >= request.scheduled_ns
        for request, _ in submitter.requests
    )


@pytest.mark.parametrize(
    ("input_value", "expected_shapes"),
    [
        (np.ones((4,), dtype=np.float32), (1, 4)),
        (
            {
                "tokens": np.ones((4,), dtype=np.int64),
                "mask": np.ones((4,), dtype=np.int64),
            },
            {"tokens": (1, 4), "mask": (1, 4)},
        ),
    ],
)
def test_static_loader_indexed_sample_gets_one_leading_batch_dimension(
    input_value,
    expected_shapes,
):
    loader = InputLoader(input_value, is_static_batched=True)
    submitter = Submitter()

    OfflineProducer(
        loader,
        submitter,
        AsyncInferenceConfig(min_samples=1),
        clock=FakeableClock(),
    ).run()

    request = submitter.requests[0][0]
    actual_input = request.sample["input"]
    if isinstance(actual_input, dict):
        assert {
            name: value.shape for name, value in actual_input.items()
        } == expected_shapes
    else:
        assert actual_input.shape == expected_shapes
    assert request.sample_count == 1
    assert loader.sample["input"] is input_value


def test_non_static_loader_input_shape_is_not_changed():
    input_value = np.ones((4,), dtype=np.float32)
    submitter = Submitter()

    OfflineProducer(
        InputLoader(input_value, is_static_batched=False),
        submitter,
        AsyncInferenceConfig(min_samples=1),
        clock=FakeableClock(),
    ).run()

    request = submitter.requests[0][0]
    assert request.sample["input"] is input_value
    assert request.sample_count == 1


@pytest.mark.parametrize(
    ("label", "expected_shapes"),
    [
        (np.array(9), (1,)),
        (
            {
                "start_positions": np.array(1),
                "end_positions": np.array(3),
            },
            {"start_positions": (1,), "end_positions": (1,)},
        ),
    ],
)
def test_static_loader_labels_are_normalized_as_a_one_item_batch(
    label,
    expected_shapes,
):
    loader = InputLoader(
        np.ones((4,), dtype=np.float32),
        is_static_batched=True,
        label=label,
    )
    submitter = Submitter()

    OfflineProducer(
        loader,
        submitter,
        AsyncInferenceConfig(min_samples=1),
        clock=FakeableClock(),
    ).run()

    actual_label = submitter.requests[0][0].sample["label"]
    if isinstance(actual_label, dict):
        assert {
            name: value.shape for name, value in actual_label.items()
        } == expected_shapes
    else:
        assert actual_label.shape == expected_shapes
    assert loader.sample["label"] is label


@pytest.mark.parametrize(
    "input_value",
    [
        np.asarray([1.0, 2.0], dtype=np.float32),
        {
            "tokens": np.asarray([1, 2], dtype=np.int64),
            "mask": np.asarray([1, 1], dtype=np.int64),
        },
    ],
)
def test_static_request_owns_input_array_storage(input_value):
    loader = InputLoader(input_value, is_static_batched=True)
    submitter = Submitter()
    OfflineProducer(
        loader,
        submitter,
        AsyncInferenceConfig(min_samples=1),
        clock=FakeableClock(),
    ).run()
    request_input = submitter.requests[0][0].sample["input"]

    if isinstance(input_value, dict):
        input_value["tokens"][:] = 99
        input_value["mask"][:] = 0
        np.testing.assert_array_equal(
            request_input["tokens"],
            np.asarray([[1, 2]]),
        )
        np.testing.assert_array_equal(
            request_input["mask"],
            np.asarray([[1, 1]]),
        )
    else:
        input_value[:] = 99
        np.testing.assert_array_equal(
            request_input,
            np.asarray([[1.0, 2.0]], dtype=np.float32),
        )


def test_static_preparation_is_included_in_load_time_before_issue():
    clock = FakeableClock()
    loader = InputLoader(
        PreparationTimedArray([1.0, 2.0], clock, 0.002),
        is_static_batched=True,
        label=PreparationTimedArray(9, clock, 0.003),
    )
    submitter = Submitter()

    result = OfflineProducer(
        loader,
        submitter,
        AsyncInferenceConfig(min_samples=1),
        clock=clock,
    ).run()

    request = submitter.requests[0][0]
    assert result.producer_load_ms == 5.0
    assert request.scheduled_ns == 5_000_000
    assert request.issued_ns == 5_000_000


@pytest.mark.parametrize(
    "metadata",
    [
        {},
        {"total_samples": 0},
        {"total_samples": -1},
        {"total_samples": True},
        {"total_samples": 1.5},
    ],
)
def test_producer_rejects_metadata_without_positive_integer_sample_count(
    metadata,
):
    with pytest.raises(ValueError, match="total_samples"):
        OfflineProducer(
            MetadataLoader(metadata),
            Submitter(),
            AsyncInferenceConfig(min_samples=1),
            clock=FakeableClock(),
        )


@pytest.mark.parametrize(
    ("producer_type", "config"),
    [
        (
            OfflineProducer,
            AsyncInferenceConfig(min_samples=1, max_samples=0),
        ),
        (
            OfflineProducer,
            AsyncInferenceConfig(min_samples=True),
        ),
        (
            OfflineProducer,
            AsyncInferenceConfig(min_samples=1, max_samples=1.5),
        ),
        (
            OfflineProducer,
            AsyncInferenceConfig(
                min_samples=1,
                max_samples=float("nan"),
            ),
        ),
        (
            ServerLikeProducer,
            AsyncInferenceConfig(
                scenario=AsyncScenario.SERVER_LIKE,
                target_qps=0,
                min_samples=1,
            ),
        ),
    ],
)
def test_producer_validates_async_config_before_loading_metadata(
    producer_type,
    config,
):
    with pytest.raises(ValueError):
        producer_type(
            MetadataLoader(None),
            Submitter(),
            config,
            clock=FakeableClock(),
        )


@pytest.mark.parametrize(
    ("producer_type", "config", "scenario_name"),
    [
        (
            OfflineProducer,
            AsyncInferenceConfig(
                scenario=AsyncScenario.SERVER_LIKE,
                target_qps=1,
                min_samples=1,
            ),
            "offline",
        ),
        (
            ServerLikeProducer,
            AsyncInferenceConfig(min_samples=1),
            "server_like",
        ),
    ],
)
def test_producer_requires_matching_scenario(
    producer_type,
    config,
    scenario_name,
):
    with pytest.raises(ValueError, match=scenario_name):
        producer_type(
            Loader(),
            Submitter(),
            config,
            clock=FakeableClock(),
        )


@pytest.mark.parametrize(
    "config",
    [
        AsyncInferenceConfig(
            scenario=AsyncScenario.SERVER_LIKE,
            target_qps=float("nan"),
            min_samples=1,
        ),
        AsyncInferenceConfig(
            scenario=AsyncScenario.SERVER_LIKE,
            target_qps=float("inf"),
            min_samples=1,
        ),
        AsyncInferenceConfig(
            scenario=AsyncScenario.SERVER_LIKE,
            target_qps=1,
            min_samples=1,
            min_duration_sec=float("nan"),
        ),
        AsyncInferenceConfig(
            scenario=AsyncScenario.SERVER_LIKE,
            target_qps=1,
            min_samples=1,
            min_duration_sec=float("inf"),
        ),
    ],
)
def test_server_like_rejects_non_finite_scheduling_config(config):
    with pytest.raises(ValueError, match="finite"):
        ServerLikeProducer(
            MetadataLoader(None),
            Submitter(),
            config,
            clock=FakeableClock(),
        )
