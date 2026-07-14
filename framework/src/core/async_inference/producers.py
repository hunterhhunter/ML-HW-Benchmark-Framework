import math
import random
import time
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral

import numpy as np

from .types import AsyncScenario, InferenceRequest


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
        self.now_ns += max(0, math.ceil(seconds * 1_000_000_000))


@dataclass(frozen=True)
class ProducerResult:
    attempted: int
    accepted: int
    rejected: int
    producer_load_ms: float


class BaseProducer:
    scenario = None

    def __init__(self, dataloader, submitter, config, clock=None):
        config.validate()
        if config.scenario is not self.scenario:
            raise ValueError(
                f"{type(self).__name__} requires "
                f"scenario={self.scenario.value}"
            )
        self._validate_scenario_config(config)
        self.dataloader = dataloader
        self.submitter = submitter
        self.config = config
        self.clock = clock if clock is not None else SystemClock()
        metadata = dataloader.get_metadata()
        if not isinstance(metadata, Mapping):
            raise ValueError("dataloader metadata must contain total_samples")
        total_samples = metadata.get("total_samples")
        if (
            isinstance(total_samples, bool)
            or not isinstance(total_samples, Integral)
            or total_samples < 1
        ):
            raise ValueError(
                "dataloader metadata total_samples must be a positive integer"
            )
        self.total_samples = int(total_samples)
        self.is_static_batched = bool(
            metadata.get("is_static_batched", False)
        )

    def _validate_scenario_config(self, config):
        pass

    def _load_sample(self, index):
        load_started_ns = self.clock.monotonic_ns()
        sample = self.dataloader.load_by_index(index)
        elapsed_ns = self.clock.monotonic_ns() - load_started_ns
        if self.is_static_batched:
            sample = dict(sample)
            sample["input"] = self._as_single_item_batch(sample["input"])
            if "label" in sample:
                sample["label"] = self._as_single_item_batch(
                    sample["label"]
                )
        return sample, elapsed_ns

    @staticmethod
    def _as_single_item_batch(value):
        if isinstance(value, Mapping):
            return {
                name: np.expand_dims(np.asarray(item), axis=0)
                for name, item in value.items()
            }
        return np.expand_dims(np.asarray(value), axis=0)


class OfflineProducer(BaseProducer):
    scenario = AsyncScenario.OFFLINE

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
            index = request_id % self.total_samples
            sample, elapsed_ns = self._load_sample(index)
            load_ns += elapsed_ns
            issued_ns = self.clock.monotonic_ns()
            request = InferenceRequest(
                request_id=request_id,
                sample_index=index,
                sample=sample,
                scheduled_ns=issued_ns,
                issued_ns=issued_ns,
                enqueued_ns=0,
                sample_count=1,
            )
            if self.submitter.submit(request, block=True):
                accepted += 1
            else:
                rejected += 1
        return ProducerResult(
            attempted=limit,
            accepted=accepted,
            rejected=rejected,
            producer_load_ms=load_ns / 1_000_000,
        )


class ServerLikeProducer(BaseProducer):
    scenario = AsyncScenario.SERVER_LIKE

    def _validate_scenario_config(self, config):
        if not math.isfinite(config.target_qps) or not math.isfinite(
            config.min_duration_sec
        ):
            raise ValueError(
                "server_like scheduling values must be finite"
            )

    def run(self):
        rng = random.Random(self.config.schedule_seed)
        started_ns = self.clock.monotonic_ns()
        scheduled_ns = started_ns
        attempted = 0
        accepted = 0
        rejected = 0
        load_ns = 0

        while True:
            elapsed_sec = (
                self.clock.monotonic_ns() - started_ns
            ) / 1_000_000_000
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
                scheduled_ns += int(
                    rng.expovariate(self.config.target_qps) * 1_000_000_000
                )
            remaining_sec = (
                scheduled_ns - self.clock.monotonic_ns()
            ) / 1_000_000_000
            if remaining_sec > 0:
                self.clock.sleep(remaining_sec)

            index = attempted % self.total_samples
            sample, elapsed_ns = self._load_sample(index)
            load_ns += elapsed_ns
            issued_ns = self.clock.monotonic_ns()
            request = InferenceRequest(
                request_id=attempted,
                sample_index=index,
                sample=sample,
                scheduled_ns=scheduled_ns,
                issued_ns=issued_ns,
                enqueued_ns=0,
                sample_count=1,
            )
            if self.submitter.submit(request, block=False):
                accepted += 1
            else:
                rejected += 1
            attempted += 1

        return ProducerResult(
            attempted=attempted,
            accepted=accepted,
            rejected=rejected,
            producer_load_ms=load_ns / 1_000_000,
        )
