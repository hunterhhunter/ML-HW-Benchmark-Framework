"""Normalize sanitized vLLM 0.16 serving benchmark evidence."""

import math
from collections.abc import Mapping

import numpy as np


PERCENTILES = (50, 85, 90, 95, 99)
PERCENTILE_METHOD = "linear"


def _exact_nonnegative_int(value, name: str) -> int:
    if type(value) is not int or value < 0:
        raise TypeError(f"{name} must be an exact non-negative int")
    return value


def _finite_nonnegative(value, name: str) -> float:
    if type(value) not in (int, float) or type(value) is bool:
        raise TypeError(f"{name} must be an exact numeric value")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return normalized


def _numeric_list(value, name: str) -> tuple[float, ...]:
    if type(value) is not list:
        raise TypeError(f"{name} must be a list")
    return tuple(
        _finite_nonnegative(item, f"{name}[{index}]")
        for index, item in enumerate(value)
    )


def _integer_list(value, name: str) -> tuple[int, ...]:
    if type(value) is not list:
        raise TypeError(f"{name} must be a list")
    return tuple(
        _exact_nonnegative_int(item, f"{name}[{index}]")
        for index, item in enumerate(value)
    )


def _percentiles(values: tuple[float, ...]) -> dict[int, float]:
    if not values:
        return {percentile: None for percentile in PERCENTILES}
    calculated = np.percentile(
        np.asarray(values, dtype=np.float64),
        PERCENTILES,
        method=PERCENTILE_METHOD,
    )
    return {
        percentile: float(value)
        for percentile, value in zip(PERCENTILES, calculated)
    }


def normalize_vllm_bench_result(payload: Mapping) -> dict:
    """Return namespaced client metrics from a vLLM 0.16 detailed result."""
    if type(payload) is not dict:
        raise TypeError("vLLM benchmark result must be an exact dict")

    duration = _finite_nonnegative(payload.get("duration"), "duration")
    if duration <= 0:
        raise ValueError("duration must be greater than zero")
    completed = _exact_nonnegative_int(payload.get("completed"), "completed")
    failed = _exact_nonnegative_int(payload.get("failed"), "failed")
    total_output = _exact_nonnegative_int(
        payload.get("total_output_tokens"),
        "total_output_tokens",
    )
    output_throughput = _finite_nonnegative(
        payload.get("output_throughput"),
        "output_throughput",
    )
    output_lens = _integer_list(payload.get("output_lens"), "output_lens")
    ttfts = _numeric_list(payload.get("ttfts"), "ttfts")

    raw_itls = payload.get("itls")
    if type(raw_itls) is not list:
        raise TypeError("itls must be a list")
    itls = tuple(
        _numeric_list(value, f"itls[{index}]")
        for index, value in enumerate(raw_itls)
    )
    errors = payload.get("errors")
    if type(errors) is not list or any(
        error is not None and type(error) is not str for error in errors
    ):
        raise TypeError("errors must be a list containing str or None values")

    request_count = len(output_lens)
    if not (
        request_count == len(ttfts) == len(itls) == len(errors)
        and request_count == completed + failed
    ):
        raise ValueError("vLLM detailed request arrays have inconsistent lengths")

    successful_indexes = tuple(
        index for index, error in enumerate(errors) if not error
    )
    invalid_reasons = []
    if failed:
        invalid_reasons.append("vllm_failed_requests")
    if len(successful_indexes) != completed:
        invalid_reasons.append("vllm_detailed_success_count_mismatch")

    successful_ttft_ms = tuple(ttfts[index] * 1000.0 for index in successful_indexes)
    successful_output_lens = tuple(output_lens[index] for index in successful_indexes)
    successful_itl_ms = tuple(
        value * 1000.0
        for index in successful_indexes
        for value in itls[index]
    )
    request_tpot_ms = []
    request_e2el_ms = []
    for index in successful_indexes:
        output_len = output_lens[index]
        generation_seconds = sum(itls[index])
        request_e2el_ms.append((ttfts[index] + generation_seconds) * 1000.0)
        if output_len > 1:
            request_tpot_ms.append(
                generation_seconds / (output_len - 1) * 1000.0
            )

    distributions = {
        "ttft": successful_ttft_ms,
        "tpot": tuple(request_tpot_ms),
        "itl": successful_itl_ms,
        "e2el": tuple(request_e2el_ms),
    }
    calculated = {
        name: _percentiles(values)
        for name, values in distributions.items()
    }

    metrics = {
        "server_successful_requests": completed,
        "server_failed_requests": failed,
        "server_output_tokens": total_output,
        "server_output_tokens_per_sec": total_output / duration,
    }
    for metric_name, percentile_values in calculated.items():
        for percentile, value in percentile_values.items():
            aggregate_name = f"p{percentile}_{metric_name}_ms"
            if aggregate_name not in payload:
                raise ValueError(
                    f"missing required aggregate percentile: {aggregate_name}"
                )
            aggregate_value = _finite_nonnegative(
                payload.get(aggregate_name),
                aggregate_name,
            )
            if value is None or not math.isclose(
                aggregate_value,
                value,
                rel_tol=1e-6,
                abs_tol=1e-6,
            ):
                invalid_reasons.append(
                    "vllm_aggregate_percentile_mismatch:"
                    f"{metric_name}:p{percentile}"
                )
            metrics[
                f"server_client_{metric_name}_p{percentile}_ms"
            ] = value

    calculated_output = sum(successful_output_lens)
    if calculated_output != total_output:
        invalid_reasons.append("vllm_total_output_tokens_mismatch")
    if not math.isclose(
        output_throughput,
        total_output / duration,
        rel_tol=1e-6,
        abs_tol=1e-6,
    ):
        invalid_reasons.append("vllm_output_throughput_mismatch")

    provenance = payload.get("fixture_provenance")
    vllm_version = None
    if type(provenance) is dict:
        candidate = provenance.get("vllm_version")
        if type(candidate) is str and len(candidate) <= 64:
            vllm_version = candidate
    return {
        "metrics": metrics,
        "details": {
            "source": "vllm_bench_serve_detailed_json",
            "schema_version": "vllm-0.16.0",
            "vllm_version": vllm_version,
            "percentile_method": "numpy.percentile(method=linear)",
            "successful_request_rows": len(successful_indexes),
            "raw_client_itl_samples": len(successful_itl_ms),
            "raw_seconds": {
                "ttfts": [ttfts[index] for index in successful_indexes],
                "itls": [
                    list(itls[index]) for index in successful_indexes
                ],
                "output_lens": list(successful_output_lens),
            },
        },
        "invalid_reasons": sorted(set(invalid_reasons)),
    }
