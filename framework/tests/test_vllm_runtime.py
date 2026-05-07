import sys
import types
from pathlib import Path

import pytest

from core.compiled_model import CompiledModel
from core.model_spec import Model_Spec, Task
from runtimes.vllm_rt import VllmRuntime


def _compiled_model(tmp_path: Path) -> CompiledModel:
    spec = Model_Spec(
        name="llama-test",
        task=Task.NLP_GENERATION,
        input_shapes={"input_ids": (1, 8)},
        input_dtype={"input_ids": "int64"},
        output_shapes={"generated_ids": (1, 4)},
        model_paths={"hf": str(tmp_path)},
    )
    return CompiledModel(spec=spec, backend_name="vllm", artifact_path=tmp_path)


def _install_fake_vllm(
    monkeypatch: pytest.MonkeyPatch,
    *,
    engine_accepts_device: bool,
    platform_is_cpu: bool,
) -> dict:
    captured: dict = {}

    vllm_mod = types.ModuleType("vllm")
    vllm_mod.__path__ = []

    class LLM:
        def __init__(self, model: str, **kwargs):
            captured["model"] = model
            captured["kwargs"] = kwargs

    vllm_mod.LLM = LLM

    engine_mod = types.ModuleType("vllm.engine")
    engine_mod.__path__ = []
    arg_utils_mod = types.ModuleType("vllm.engine.arg_utils")

    if engine_accepts_device:
        class EngineArgs:
            def __init__(self, model: str = "", device: str = "cuda"):
                pass
    else:
        class EngineArgs:
            def __init__(self, model: str = ""):
                pass

    arg_utils_mod.EngineArgs = EngineArgs

    platforms_mod = types.ModuleType("vllm.platforms")

    class CurrentPlatform:
        @staticmethod
        def is_cpu() -> bool:
            return platform_is_cpu

    platforms_mod.current_platform = CurrentPlatform()

    monkeypatch.setitem(sys.modules, "vllm", vllm_mod)
    monkeypatch.setitem(sys.modules, "vllm.engine", engine_mod)
    monkeypatch.setitem(sys.modules, "vllm.engine.arg_utils", arg_utils_mod)
    monkeypatch.setitem(sys.modules, "vllm.platforms", platforms_mod)
    return captured


def test_vllm_cpu_target_raises_clear_error_without_cpu_backend(monkeypatch, tmp_path):
    _install_fake_vllm(
        monkeypatch,
        engine_accepts_device=False,
        platform_is_cpu=False,
    )

    runtime = VllmRuntime(device="cpu")

    with pytest.raises(RuntimeError, match="CPU backend"):
        runtime.load(_compiled_model(tmp_path))


def test_vllm_cpu_target_omits_device_when_cpu_platform_detected(monkeypatch, tmp_path):
    captured = _install_fake_vllm(
        monkeypatch,
        engine_accepts_device=False,
        platform_is_cpu=True,
    )

    runtime = VllmRuntime(device="cpu")
    runtime.load(_compiled_model(tmp_path))

    assert "device" not in captured["kwargs"]
    assert "gpu_memory_utilization" not in captured["kwargs"]


def test_vllm_cpu_target_passes_device_when_engine_accepts_it(monkeypatch, tmp_path):
    captured = _install_fake_vllm(
        monkeypatch,
        engine_accepts_device=True,
        platform_is_cpu=False,
    )

    runtime = VllmRuntime(device="cpu")
    runtime.load(_compiled_model(tmp_path))

    assert captured["kwargs"]["device"] == "cpu"
    assert "gpu_memory_utilization" not in captured["kwargs"]


def test_vllm_timing_uses_v1_request_metrics():
    metrics = types.SimpleNamespace(
        first_token_latency=0.123,
        first_token_ts=10.0,
        last_token_ts=10.456,
        num_generation_tokens=4,
    )
    outputs = [types.SimpleNamespace(metrics=metrics)]

    ttft_ms, tpot_ms, timing_source = VllmRuntime._extract_timing_from_vllm_metrics(
        outputs,
        total_ms=999.0,
        num_tokens=4,
    )

    assert ttft_ms == pytest.approx(123.0)
    assert tpot_ms == pytest.approx(152.0)
    assert timing_source == "vllm_request_metrics"


def test_vllm_timing_uses_legacy_request_metrics():
    metrics = types.SimpleNamespace(
        arrival_time=100.0,
        first_token_time=100.2,
        finished_time=100.8,
    )
    outputs = [types.SimpleNamespace(metrics=metrics)]

    ttft_ms, tpot_ms, timing_source = VllmRuntime._extract_timing_from_vllm_metrics(
        outputs,
        total_ms=999.0,
        num_tokens=4,
    )

    assert ttft_ms == pytest.approx(200.0)
    assert tpot_ms == pytest.approx(200.0)
    assert timing_source == "vllm_request_metrics"


def test_vllm_timing_falls_back_when_request_metrics_missing():
    outputs = [types.SimpleNamespace(metrics=None)]

    ttft_ms, tpot_ms, timing_source = VllmRuntime._extract_timing_from_vllm_metrics(
        outputs,
        total_ms=80.0,
        num_tokens=4,
    )

    assert ttft_ms == pytest.approx(20.0)
    assert tpot_ms == pytest.approx(20.0)
    assert timing_source == "estimated_from_total"


def test_vllm_timing_falls_back_when_request_metrics_are_unset():
    metrics = types.SimpleNamespace(
        first_token_latency=0.0,
        first_token_ts=0.0,
        last_token_ts=0.0,
        num_generation_tokens=4,
    )
    outputs = [types.SimpleNamespace(metrics=metrics)]

    ttft_ms, tpot_ms, timing_source = VllmRuntime._extract_timing_from_vllm_metrics(
        outputs,
        total_ms=80.0,
        num_tokens=4,
    )

    assert ttft_ms == pytest.approx(20.0)
    assert tpot_ms == pytest.approx(20.0)
    assert timing_source == "estimated_from_total"
