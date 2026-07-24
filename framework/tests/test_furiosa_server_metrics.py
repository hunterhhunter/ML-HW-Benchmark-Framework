import pytest

from monitors.furiosa_server_metrics import (
    FuriosaServerMetricsCollector,
    normalize_furiosa_server_metrics,
)


LABELS = 'model_name="llama",engine="npu:0"'


def histogram(name, buckets, total_sum, count):
    lines = [f"# TYPE {name} histogram"]
    for boundary, value in buckets:
        lines.append(f'{name}_bucket{{{LABELS},le="{boundary}"}} {value}')
    lines.extend(
        [
            f"{name}_sum{{{LABELS}}} {total_sum}",
            f"{name}_count{{{LABELS}}} {count}",
        ]
    )
    return "\n".join(lines)


def exposition(multiplier):
    success = 10 + 2 * multiplier
    token_count = 10 + 2 * multiplier
    token_sum = 100 + 5 * multiplier
    parts = [
        "# TYPE furiosa_llm_request_success_total counter",
        (
            "furiosa_llm_request_success_total"
            f'{{{LABELS},finished_reason="stop"}} {success}'
        ),
        histogram(
            "furiosa_llm_request_generation_tokens",
            (("4", 4 + multiplier), ("8", token_count), ("+Inf", token_count)),
            token_sum,
            token_count,
        ),
    ]
    for name in (
        "furiosa_llm_time_to_first_token_seconds",
        "furiosa_llm_inter_token_latency_seconds",
        "furiosa_llm_e2e_request_latency_seconds",
    ):
        parts.append(
            histogram(
                name,
                (("0.005", 5 + multiplier), ("0.01", 8 + 2 * multiplier), ("+Inf", 10 + 2 * multiplier)),
                0.07 + 0.025 * multiplier,
                10 + 2 * multiplier,
            )
        )
    return "\n".join(parts) + "\n"


def test_furiosa_metrics_normalizer_namespaces_vendor_histogram_delta():
    result = normalize_furiosa_server_metrics(
        exposition(0),
        exposition(1),
        labels={"model_name": "llama", "engine": "npu:0"},
    )

    assert result["invalid_reasons"] == []
    metrics = result["metrics"]
    assert metrics["server_vendor_successful_requests"] == pytest.approx(2.0)
    assert metrics["server_vendor_generation_tokens"] == pytest.approx(5.0)
    for name in ("ttft", "itl", "e2el"):
        assert metrics[f"server_vendor_{name}_p50_ms"] == pytest.approx(5.0)
        assert metrics[f"server_vendor_{name}_p85_ms"] == pytest.approx(8.5)

    details = result["details"]
    assert details["source"] == "furiosa_server_prometheus_histogram_delta"
    assert details["endpoint"] == "/metrics"
    assert details["labels"] == {"engine": "npu:0", "model_name": "llama"}
    assert details["unit_in"] == "seconds"
    assert details["unit_out"] == "milliseconds"
    assert details["quantile_algorithm"] == (
        "prometheus_histogram_linear_interpolation"
    )
    assert details["histograms"]["ttft"]["raw_delta_buckets"][-1] == {
        "le": "+Inf",
        "count": 2.0,
    }


def test_furiosa_metrics_rejects_counter_reset():
    with pytest.raises(ValueError, match="reset"):
        normalize_furiosa_server_metrics(
            exposition(1),
            exposition(0),
            labels={"model_name": "llama", "engine": "npu:0"},
        )


class Response:
    text = "metrics"

    def raise_for_status(self):
        return None


class Client:
    def __init__(self):
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return Response()


def test_collector_uses_bounded_get_without_redirects():
    client = Client()
    collector = FuriosaServerMetricsCollector(
        "http://127.0.0.1:8000",
        labels={"model_name": "llama", "engine": "npu:0"},
        client=client,
        timeout_sec=2.5,
    )

    assert collector.snapshot_text() == "metrics"
    assert client.calls == [
        (
            "http://127.0.0.1:8000/metrics",
            {"follow_redirects": False, "timeout": 2.5},
        )
    ]
