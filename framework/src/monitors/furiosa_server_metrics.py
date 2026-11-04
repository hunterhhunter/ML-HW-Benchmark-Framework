"""Furiosa server Prometheus snapshot collection and delta normalization."""

import math
from typing import Mapping

import httpx
from prometheus_client.parser import text_string_to_metric_families

from .prometheus_histogram import (
    HistogramSnapshot,
    histogram_delta,
    histogram_quantile,
    parse_histogram,
)


PERCENTILES = (50, 85, 90, 95, 99)
SUCCESS_COUNTER = "furiosa_llm_request_success_total"
GENERATION_TOKENS = "furiosa_llm_request_generation_tokens"
LATENCY_HISTOGRAMS = {
    "ttft": "furiosa_llm_time_to_first_token_seconds",
    "itl": "furiosa_llm_inter_token_latency_seconds",
    "e2el": "furiosa_llm_e2e_request_latency_seconds",
}


def _labels_match(labels: dict, required: Mapping[str, str]) -> bool:
    return all(labels.get(key) == value for key, value in required.items())


def _counter_value(text: str, name: str, labels: Mapping[str, str]) -> float:
    values = []
    matched_scopes = set()
    for metric in text_string_to_metric_families(text):
        for sample in metric.samples:
            if sample.name != name or not _labels_match(sample.labels, labels):
                continue
            value = float(sample.value)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"invalid counter value: {name}")
            base_scope = tuple(
                sorted(
                    (key, value)
                    for key, value in sample.labels.items()
                    if key in ("model_name", "engine")
                )
            )
            matched_scopes.add(base_scope)
            values.append(value)
    if not values:
        raise ValueError(f"counter not found: {name}")
    if len(matched_scopes) != 1:
        raise ValueError(f"ambiguous counter label scope: {name}")
    return sum(values)


def _raw_delta_buckets(histogram: HistogramSnapshot) -> list[dict]:
    return [
        {
            "le": "+Inf" if math.isinf(boundary) else boundary,
            "count": count,
        }
        for boundary, count in zip(
            histogram.bounds,
            histogram.cumulative_counts,
        )
    ]


def normalize_furiosa_server_metrics(
    before_text: str,
    after_text: str,
    *,
    labels: Mapping[str, str],
    endpoint: str = "/metrics",
) -> dict:
    if type(labels) is not dict or any(
        type(key) is not str or type(value) is not str
        for key, value in labels.items()
    ):
        raise TypeError("labels must be an exact str-to-str dict")
    normalized_labels = dict(sorted(labels.items()))

    before_success = _counter_value(before_text, SUCCESS_COUNTER, labels)
    after_success = _counter_value(after_text, SUCCESS_COUNTER, labels)
    if after_success < before_success:
        raise ValueError("request success counter reset detected")
    success_delta = after_success - before_success

    generation_delta = histogram_delta(
        parse_histogram(before_text, GENERATION_TOKENS, labels),
        parse_histogram(after_text, GENERATION_TOKENS, labels),
    )
    metrics = {
        "server_vendor_successful_requests": success_delta,
        "server_vendor_generation_tokens": generation_delta.total_sum,
    }
    histogram_details = {
        "generation_tokens": {
            "count": generation_delta.count,
            "sum": generation_delta.total_sum,
            "raw_delta_buckets": _raw_delta_buckets(generation_delta),
        }
    }

    for short_name, metric_name in LATENCY_HISTOGRAMS.items():
        delta = histogram_delta(
            parse_histogram(before_text, metric_name, labels),
            parse_histogram(after_text, metric_name, labels),
        )
        quantiles = {}
        for percentile in PERCENTILES:
            seconds = histogram_quantile(delta, percentile / 100.0)
            milliseconds = None if seconds is None else seconds * 1000.0
            metrics[
                f"server_vendor_{short_name}_p{percentile}_ms"
            ] = milliseconds
            quantiles[f"p{percentile}_ms"] = milliseconds
        histogram_details[short_name] = {
            "count": delta.count,
            "sum_seconds": delta.total_sum,
            "quantiles_ms": quantiles,
            "raw_delta_buckets": _raw_delta_buckets(delta),
        }
        if short_name == "itl":
            metrics["server_vendor_itl_observations"] = delta.count

    return {
        "metrics": metrics,
        "details": {
            "source": "furiosa_server_prometheus_histogram_delta",
            "endpoint": endpoint,
            "labels": normalized_labels,
            "unit_in": "seconds",
            "unit_out": "milliseconds",
            "quantile_algorithm": (
                "prometheus_histogram_linear_interpolation"
            ),
            "histograms": histogram_details,
        },
        "invalid_reasons": [],
    }


class FuriosaServerMetricsCollector:
    def __init__(
        self,
        base_url: str,
        *,
        labels: Mapping[str, str],
        client=None,
        timeout_sec: float = 5.0,
    ):
        if type(base_url) is not str or not base_url:
            raise ValueError("base_url is required")
        if type(timeout_sec) not in (int, float) or type(timeout_sec) is bool:
            raise TypeError("timeout_sec must be numeric")
        timeout_sec = float(timeout_sec)
        if not math.isfinite(timeout_sec) or timeout_sec <= 0:
            raise ValueError("timeout_sec must be finite and positive")
        self.base_url = base_url.rstrip("/")
        self.labels = dict(labels)
        self.client = httpx.Client() if client is None else client
        self._owns_client = client is None
        self.timeout_sec = timeout_sec

    def snapshot_text(self) -> str:
        response = self.client.get(
            f"{self.base_url}/metrics",
            follow_redirects=False,
            timeout=self.timeout_sec,
        )
        response.raise_for_status()
        if type(response.text) is not str:
            raise TypeError("metrics response text must be str")
        return response.text

    def close(self) -> None:
        if self._owns_client:
            self.client.close()
