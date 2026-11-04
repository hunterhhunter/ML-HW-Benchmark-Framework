import math

import pytest

from monitors.prometheus_histogram import (
    HistogramSnapshot,
    histogram_delta,
    histogram_quantile,
    parse_histogram,
)


BEFORE = """
# TYPE latency_seconds histogram
latency_seconds_bucket{engine="npu:0",model_name="llama",le="0.005"} 5
latency_seconds_bucket{engine="npu:0",model_name="llama",le="0.01"} 8
latency_seconds_bucket{engine="npu:0",model_name="llama",le="+Inf"} 10
latency_seconds_sum{engine="npu:0",model_name="llama"} 0.07
latency_seconds_count{engine="npu:0",model_name="llama"} 10
"""

AFTER = """
# TYPE latency_seconds histogram
latency_seconds_bucket{engine="npu:0",model_name="llama",le="0.005"} 7
latency_seconds_bucket{engine="npu:0",model_name="llama",le="0.01"} 13
latency_seconds_bucket{engine="npu:0",model_name="llama",le="+Inf"} 16
latency_seconds_sum{engine="npu:0",model_name="llama"} 0.115
latency_seconds_count{engine="npu:0",model_name="llama"} 16
"""


def test_histogram_delta_and_prometheus_interpolation():
    labels = {"model_name": "llama", "engine": "npu:0"}
    delta = histogram_delta(
        parse_histogram(BEFORE, "latency_seconds", labels),
        parse_histogram(AFTER, "latency_seconds", labels),
    )

    assert delta.bounds == (0.005, 0.01, math.inf)
    assert delta.cumulative_counts == pytest.approx((2.0, 5.0, 6.0))
    assert delta.count == pytest.approx(6.0)
    assert delta.total_sum == pytest.approx(0.045)
    assert histogram_quantile(delta, 0.50) == pytest.approx(0.0066666667)
    assert histogram_quantile(delta, 0.85) == pytest.approx(0.01)


def test_empty_histogram_delta_has_no_quantile():
    snapshot = parse_histogram(BEFORE, "latency_seconds", {"model_name": "llama", "engine": "npu:0"})
    delta = histogram_delta(snapshot, snapshot)

    assert delta.count == 0
    assert histogram_quantile(delta, 0.99) is None


@pytest.mark.parametrize(
    "after",
    [
        HistogramSnapshot(
            name="latency_seconds",
            labels=(("engine", "npu:0"), ("model_name", "llama")),
            bounds=(0.005, 0.01, math.inf),
            cumulative_counts=(4.0, 7.0, 9.0),
            total_sum=0.06,
            count=9.0,
        ),
        HistogramSnapshot(
            name="latency_seconds",
            labels=(("engine", "npu:1"), ("model_name", "llama")),
            bounds=(0.005, 0.01, math.inf),
            cumulative_counts=(7.0, 13.0, 16.0),
            total_sum=0.115,
            count=16.0,
        ),
        HistogramSnapshot(
            name="latency_seconds",
            labels=(("engine", "npu:0"), ("model_name", "llama")),
            bounds=(0.005, 0.02, math.inf),
            cumulative_counts=(7.0, 13.0, 16.0),
            total_sum=0.115,
            count=16.0,
        ),
        HistogramSnapshot(
            name="latency_seconds",
            labels=(("engine", "npu:0"), ("model_name", "llama")),
            bounds=(0.005, 0.01, math.inf),
            cumulative_counts=(7.0, 13.0, 15.0),
            total_sum=0.115,
            count=16.0,
        ),
    ],
)
def test_histogram_delta_rejects_reset_scope_or_schema_change(after):
    before = parse_histogram(
        BEFORE,
        "latency_seconds",
        {"model_name": "llama", "engine": "npu:0"},
    )

    with pytest.raises(ValueError):
        histogram_delta(before, after)


def test_parser_rejects_ambiguous_label_scope():
    text = BEFORE + BEFORE.replace('engine="npu:0"', 'engine="npu:1"')

    with pytest.raises(ValueError, match="ambiguous"):
        parse_histogram(text, "latency_seconds", {"model_name": "llama"})
