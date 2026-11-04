"""Strict Prometheus classic histogram parsing and delta quantiles."""

import math
from dataclasses import dataclass
from typing import Mapping

from prometheus_client.parser import text_string_to_metric_families


@dataclass(frozen=True)
class HistogramSnapshot:
    name: str
    labels: tuple[tuple[str, str], ...]
    bounds: tuple[float, ...]
    cumulative_counts: tuple[float, ...]
    total_sum: float
    count: float


def _labels_match(labels: dict, required: Mapping[str, str]) -> bool:
    return all(labels.get(key) == value for key, value in required.items())


def _validate_snapshot(snapshot: HistogramSnapshot) -> None:
    if type(snapshot) is not HistogramSnapshot:
        raise TypeError("histogram must be an exact HistogramSnapshot")
    if (
        not snapshot.bounds
        or len(snapshot.bounds) != len(snapshot.cumulative_counts)
        or not math.isinf(snapshot.bounds[-1])
        or snapshot.bounds[-1] < 0
    ):
        raise ValueError("histogram bucket schema is invalid")
    if any(
        not math.isfinite(boundary)
        for boundary in snapshot.bounds[:-1]
    ) or any(
        left >= right
        for left, right in zip(snapshot.bounds, snapshot.bounds[1:])
    ):
        raise ValueError("histogram boundaries must be strictly increasing")
    if (
        not math.isfinite(snapshot.total_sum)
        or snapshot.total_sum < 0
        or not math.isfinite(snapshot.count)
        or snapshot.count < 0
        or any(
            not math.isfinite(value) or value < 0
            for value in snapshot.cumulative_counts
        )
        or any(
            left > right
            for left, right in zip(
                snapshot.cumulative_counts,
                snapshot.cumulative_counts[1:],
            )
        )
        or snapshot.cumulative_counts[-1] != snapshot.count
    ):
        raise ValueError("histogram count or sum invariant failed")


def parse_histogram(
    text: str,
    name: str,
    labels: Mapping[str, str],
) -> HistogramSnapshot:
    if type(text) is not str or type(name) is not str or type(labels) is not dict:
        raise TypeError("histogram text, name, and labels require exact builtins")
    candidates = {}
    for metric in text_string_to_metric_families(text):
        if metric.name != name or metric.type != "histogram":
            continue
        for sample in metric.samples:
            sample_labels = dict(sample.labels)
            sample_labels.pop("le", None)
            if not _labels_match(sample_labels, labels):
                continue
            identity = tuple(sorted(sample_labels.items()))
            candidate = candidates.setdefault(
                identity,
                {"buckets": [], "sum": None, "count": None},
            )
            value = float(sample.value)
            if sample.name == f"{name}_bucket":
                raw_boundary = sample.labels.get("le")
                boundary = (
                    math.inf
                    if raw_boundary == "+Inf"
                    else float(raw_boundary)
                )
                candidate["buckets"].append((boundary, value))
            elif sample.name == f"{name}_sum":
                candidate["sum"] = value
            elif sample.name == f"{name}_count":
                candidate["count"] = value

    if not candidates:
        raise ValueError(f"histogram not found: {name}")
    if len(candidates) != 1:
        raise ValueError(f"ambiguous histogram label scope: {name}")
    identity, candidate = next(iter(candidates.items()))
    if candidate["sum"] is None or candidate["count"] is None:
        raise ValueError(f"histogram sum/count missing: {name}")
    buckets = sorted(candidate["buckets"], key=lambda item: item[0])
    snapshot = HistogramSnapshot(
        name=name,
        labels=identity,
        bounds=tuple(boundary for boundary, _ in buckets),
        cumulative_counts=tuple(value for _, value in buckets),
        total_sum=candidate["sum"],
        count=candidate["count"],
    )
    _validate_snapshot(snapshot)
    return snapshot


def histogram_delta(
    before: HistogramSnapshot,
    after: HistogramSnapshot,
) -> HistogramSnapshot:
    _validate_snapshot(before)
    _validate_snapshot(after)
    if (
        before.name != after.name
        or before.labels != after.labels
        or before.bounds != after.bounds
    ):
        raise ValueError("histogram scope or bucket schema changed")
    if (
        after.total_sum < before.total_sum
        or after.count < before.count
        or any(
            after_value < before_value
            for before_value, after_value in zip(
                before.cumulative_counts,
                after.cumulative_counts,
            )
        )
    ):
        raise ValueError("histogram counter reset detected")
    delta = HistogramSnapshot(
        name=before.name,
        labels=before.labels,
        bounds=before.bounds,
        cumulative_counts=tuple(
            after_value - before_value
            for before_value, after_value in zip(
                before.cumulative_counts,
                after.cumulative_counts,
            )
        ),
        total_sum=after.total_sum - before.total_sum,
        count=after.count - before.count,
    )
    _validate_snapshot(delta)
    return delta


def histogram_quantile(
    histogram: HistogramSnapshot,
    quantile: float,
) -> float | None:
    _validate_snapshot(histogram)
    if type(quantile) not in (int, float) or type(quantile) is bool:
        raise TypeError("quantile must be numeric")
    quantile = float(quantile)
    if not math.isfinite(quantile) or not 0 <= quantile <= 1:
        raise ValueError("quantile must be between zero and one")
    if histogram.count == 0:
        return None

    rank = quantile * histogram.count
    for index, cumulative_count in enumerate(histogram.cumulative_counts):
        if cumulative_count < rank:
            continue
        upper = histogram.bounds[index]
        if math.isinf(upper):
            return 0.0 if index == 0 else histogram.bounds[index - 1]
        previous_count = (
            0.0 if index == 0 else histogram.cumulative_counts[index - 1]
        )
        lower = 0.0 if index == 0 else histogram.bounds[index - 1]
        bucket_count = cumulative_count - previous_count
        if bucket_count == 0:
            return upper
        return lower + (upper - lower) * (
            (rank - previous_count) / bucket_count
        )
    return histogram.bounds[-2] if len(histogram.bounds) > 1 else 0.0
