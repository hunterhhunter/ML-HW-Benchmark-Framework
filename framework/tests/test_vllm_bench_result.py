import json
from pathlib import Path

import pytest

from adapters.vllm_bench_result import normalize_vllm_bench_result


FIXTURE = Path(__file__).parent / "fixtures" / "vllm_bench_serve_0_16_0.json"
PERCENTILES = (50, 85, 90, 95, 99)


def payload():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_normalizer_recomputes_all_client_percentiles_from_detailed_arrays():
    result = normalize_vllm_bench_result(payload())

    assert result["invalid_reasons"] == []
    metrics = result["metrics"]
    assert metrics["server_successful_requests"] == 2
    assert metrics["server_output_tokens"] == 5
    assert metrics["server_output_tokens_per_sec"] == pytest.approx(5.0)
    expected = {
        "ttft": {50: 150.0, 85: 185.0, 90: 190.0, 95: 195.0, 99: 199.0},
        "tpot": {50: 32.5, 85: 37.75, 90: 38.5, 95: 39.25, 99: 39.85},
        "itl": {50: 30.0, 85: 37.0, 90: 38.0, 95: 39.0, 99: 39.8},
        "e2el": {50: 195.0, 85: 226.5, 90: 231.0, 95: 235.5, 99: 239.1},
    }
    for metric_name, values in expected.items():
        for percentile, value in values.items():
            assert metrics[
                f"server_client_{metric_name}_p{percentile}_ms"
            ] == pytest.approx(value)

    details = result["details"]
    assert details["source"] == "vllm_bench_serve_detailed_json"
    assert details["schema_version"] == "vllm-0.16.0"
    assert details["percentile_method"] == "numpy.percentile(method=linear)"
    assert details["raw_client_itl_samples"] == 3
    assert details["successful_request_rows"] == 2


def test_normalizer_does_not_copy_prompt_or_response_text():
    source = payload()
    source["prompt"] = "SECRET PROMPT"
    source["generated_texts"] = ["SECRET RESPONSE"] * 3

    result = normalize_vllm_bench_result(source)
    serialized = json.dumps(result, sort_keys=True)

    assert "SECRET" not in serialized
    assert "prompt" not in result["details"]
    assert "generated_texts" not in result["details"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("duration", float("nan")),
        ("ttfts", [0.1, -0.2, 0.0]),
        ("output_lens", [3, "2", 0]),
        ("completed", None),
    ],
)
def test_normalizer_rejects_invalid_numeric_contract(field, value):
    source = payload()
    source[field] = value

    with pytest.raises((TypeError, ValueError)):
        normalize_vllm_bench_result(source)


def test_normalizer_requires_every_requested_aggregate_percentile():
    source = payload()
    del source["p85_itl_ms"]

    with pytest.raises(ValueError, match="p85_itl_ms"):
        normalize_vllm_bench_result(source)


def test_normalizer_marks_detailed_success_count_mismatch_invalid():
    source = payload()
    source["errors"] = ["", "timeout", "timeout"]

    result = normalize_vllm_bench_result(source)

    assert "vllm_detailed_success_count_mismatch" in result["invalid_reasons"]


def test_normalizer_marks_aggregate_percentile_mismatch_invalid():
    source = payload()
    source["p99_ttft_ms"] = 999.0

    result = normalize_vllm_bench_result(source)

    assert "vllm_aggregate_percentile_mismatch:ttft:p99" in result[
        "invalid_reasons"
    ]
