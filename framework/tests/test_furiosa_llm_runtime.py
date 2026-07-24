import sys
import types
from pathlib import Path

import numpy as np
import pytest

from core.compiled_model import CompiledModel
from core.model_spec import Model_Spec, Task
from runtimes.furiosa_llm_rt import FuriosaLlmRuntime


class _Completion:
    def __init__(self, token_ids):
        self.token_ids = token_ids


class _RequestOutput:
    def __init__(self, token_ids):
        self.outputs = [_Completion(token_ids)]


def _compiled_model(tmp_path: Path) -> CompiledModel:
    model_dir = tmp_path / "llama"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    fxb_path = tmp_path / "llama.fxb"
    fxb_path.write_bytes(b"fake-fxb")
    spec = Model_Spec(
        name="llama",
        task=Task.NLP_GENERATION,
        input_shapes={"input_ids": (1, 8)},
        input_dtype={"input_ids": "int64"},
        output_shapes={"generated_ids": (1, 4)},
        model_paths={"hf_model": str(model_dir)},
    )
    return CompiledModel(spec, "furiosa_llm", fxb_path)


def _install_fake_sdk(monkeypatch, generated_tokens=((31, 32), (41,))):
    state = {"llm_init": [], "generate": [], "shutdowns": 0}

    class SamplingParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class SchedulerConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class LLM:
        def __init__(self, model_id_or_path, **kwargs):
            state["llm_init"].append((model_id_or_path, kwargs))

        def generate(self, prompts, sampling_params=None, prompt_token_ids=None):
            state["generate"].append(
                {
                    "prompts": prompts,
                    "sampling_params": sampling_params,
                    "prompt_token_ids": prompt_token_ids,
                }
            )
            outputs = [_RequestOutput(list(tokens)) for tokens in generated_tokens]
            return outputs[0] if len(outputs) == 1 else outputs

        def shutdown(self):
            state["shutdowns"] += 1

    sdk = types.ModuleType("furiosa_llm")
    sdk.LLM = LLM
    sdk.SamplingParams = SamplingParams
    api = types.ModuleType("furiosa_llm.api")
    api.SchedulerConfig = SchedulerConfig
    monkeypatch.setitem(sys.modules, "furiosa_llm", sdk)
    monkeypatch.setitem(sys.modules, "furiosa_llm.api", api)
    return state


def test_furiosa_sdk_is_imported_only_when_loading(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "furiosa_llm", None)
    runtime = FuriosaLlmRuntime()

    with pytest.raises(ImportError, match="furiosa-llm"):
        runtime.load(_compiled_model(tmp_path))


def test_load_passes_hf_fxb_device_and_scheduler_options(monkeypatch, tmp_path):
    state = _install_fake_sdk(monkeypatch)
    compiled = _compiled_model(tmp_path)
    runtime = FuriosaLlmRuntime(
        device="npu:0",
        devices="npu:0:*",
        data_parallel_size=2,
        pipeline_parallel_size=1,
        max_io_memory_mb=4096,
        seed=17,
        cache_dir=tmp_path / "cache",
        npu_queue_limit=3,
        max_processing_samples=128,
        spare_blocks_ratio=0.15,
    )

    runtime.load(compiled)

    model_path, kwargs = state["llm_init"][0]
    assert model_path == compiled.spec.model_paths["hf_model"]
    assert kwargs["fxb"] == str(compiled.artifact_path)
    assert kwargs["devices"] == "npu:0:*"
    assert kwargs["data_parallel_size"] == 2
    assert kwargs["pipeline_parallel_size"] == 1
    assert kwargs["max_io_memory_mb"] == 4096
    assert kwargs["seed"] == 17
    assert kwargs["cache_dir"] == tmp_path / "cache"
    assert kwargs["scheduler_config"].kwargs == {
        "npu_queue_limit": 3,
        "max_processing_samples": 128,
        "spare_blocks_ratio": 0.15,
    }


def test_load_omits_fxb_for_hub_artifact_resolution(monkeypatch):
    state = _install_fake_sdk(monkeypatch)
    spec = Model_Spec(
        name="llama",
        task=Task.NLP_GENERATION,
        input_shapes={"input_ids": (1, 8)},
        input_dtype={"input_ids": "int64"},
        output_shapes={"generated_ids": (1, 4)},
        model_paths={"hf_model": "furiosa-ai/Llama-3.1-8B-Instruct"},
    )
    compiled = CompiledModel(spec, "furiosa_llm", None)
    runtime = FuriosaLlmRuntime(device="npu:0")

    runtime.load(compiled)

    model_reference, kwargs = state["llm_init"][0]
    assert model_reference == "furiosa-ai/Llama-3.1-8B-Instruct"
    assert "fxb" not in kwargs
    assert kwargs["devices"] == "npu:0"
    assert runtime.is_compatible(compiled) is True


def test_generate_trims_padding_and_normalizes_batch_outputs(monkeypatch, tmp_path):
    state = _install_fake_sdk(monkeypatch)
    runtime = FuriosaLlmRuntime(device="npu:0")
    runtime.load(_compiled_model(tmp_path))

    result = runtime.generate(
        {
            "input_ids": np.array([[0, 11, 12], [21, 22, 0]], dtype=np.int64),
            "attention_mask": np.array([[0, 1, 1], [1, 1, 0]], dtype=np.int64),
            "position_ids": np.array([[0, 0, 1], [0, 1, 0]], dtype=np.int64),
        },
        max_new_tokens=7,
        stop_token_ids=(2, 3),
    )

    call = state["generate"][0]
    assert call["prompts"] == ["", ""]
    assert call["prompt_token_ids"]["input_ids"] == [[11, 12], [21, 22]]
    assert call["prompt_token_ids"]["attention_mask"] == [[1, 1], [1, 1]]
    assert "position_ids" not in call["prompt_token_ids"]
    assert call["sampling_params"].kwargs == {
        "max_tokens": 7,
        "temperature": 0.0,
        "stop_token_ids": [2, 3],
    }
    np.testing.assert_array_equal(result.generated_ids, [[31, 32], [41, 0]])
    np.testing.assert_array_equal(result.generated_lengths, [2, 1])
    assert result.num_tokens == 3
    assert result.total_ms >= 0.0
    assert result.ttft_ms is None
    assert result.tpot_ms is None
    assert result.timing_source == "wall_clock_total_only"


def test_generate_normalizes_single_output_and_shutdown_is_idempotent(
    monkeypatch, tmp_path
):
    state = _install_fake_sdk(monkeypatch, generated_tokens=((9, 10),))
    runtime = FuriosaLlmRuntime()
    runtime.load(_compiled_model(tmp_path))

    result = runtime.generate(
        {"input_ids": np.array([1, 2, 3], dtype=np.int64)},
        max_new_tokens=2,
    )
    runtime.unload()
    runtime.unload()

    np.testing.assert_array_equal(result.generated_ids, [[9, 10]])
    np.testing.assert_array_equal(result.generated_lengths, [2])
    assert state["shutdowns"] == 1


def test_unload_does_not_shutdown_llm_until_native_backend_stops(
    monkeypatch, tmp_path
):
    state = _install_fake_sdk(monkeypatch, generated_tokens=((9,),))
    runtime = FuriosaLlmRuntime()
    runtime.load(_compiled_model(tmp_path))

    class Backend:
        def __init__(self):
            self.results = [False, True]
            self.calls = []

        def shutdown(self, timeout):
            self.calls.append(timeout)
            return self.results.pop(0)

    backend = Backend()
    runtime._native_backend = backend

    with pytest.raises(RuntimeError, match="native async backend"):
        runtime.unload()

    assert state["shutdowns"] == 0
    assert runtime._native_backend is backend
    assert runtime._llm is not None

    runtime.unload()
    runtime.unload()

    assert backend.calls == [5.0, 5.0]
    assert state["shutdowns"] == 1
