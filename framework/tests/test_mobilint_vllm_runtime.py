import asyncio
import json
import sys
import threading
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from core.compiled_model import CompiledModel
from core.model_spec import Model_Spec, Task
from core.runtime_executor import NativeAsyncRuntimeExecutor
from core.targets import resolve_target
from runtimes import create_runtime
from runtimes import mobilint_vllm

_PROMPT = {
    "input_ids": np.array([[0, 11, 12, 0]]),
    "attention_mask": np.array([[0, 1, 1, 0]]),
}


@pytest.fixture
def artifact(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "mobilint-llama"}), encoding="utf-8"
    )
    (tmp_path / "model.mxq").touch()
    spec = Model_Spec(
        name="llama-3.1-8b",
        task=Task.NLP_GENERATION,
        input_shapes={"input_ids": (1, 4)},
        input_dtype={"input_ids": "int64"},
        output_shapes={"generated_ids": (1, 3)},
    )
    return CompiledModel(spec, "mobilint_vllm", tmp_path)


@pytest.fixture
def sdk(monkeypatch):
    state = SimpleNamespace(
        starts=[], calls=[], shutdowns=0, active=0, max_active=0, aborted=0,
        response=[[21], [21, 22, 23]], hang=False, finish=True, finish_reason="length",
    )

    class Platform:
        pass

    class AsyncLLM:
        vllm_config = SimpleNamespace(
            scheduler_config=SimpleNamespace(
                max_num_seqs=16, max_num_batched_tokens=2048, enable_chunked_prefill=True,
            ),
            model_config=SimpleNamespace(max_model_len=4096),
            cache_config=SimpleNamespace(enable_prefix_caching=False),
        )

        @classmethod
        def from_engine_args(cls, args):
            assert asyncio.get_running_loop()
            state.starts.append(args)
            return cls()

        async def generate(self, prompt, sampling, request_id):
            state.calls.append((prompt, sampling, request_id))
            state.active += 1
            state.max_active = max(state.max_active, state.active)
            try:
                await asyncio.sleep(60 if state.hang else 0.02)
                for i, ids in enumerate(state.response):
                    yield SimpleNamespace(
                        outputs=[SimpleNamespace(token_ids=ids, finish_reason=state.finish_reason)],
                        finished=state.finish and i == len(state.response) - 1,
                    )
            except asyncio.CancelledError:
                state.aborted += 1
                raise
            finally:
                state.active -= 1

        def shutdown(self, *, timeout):
            assert asyncio.get_running_loop()
            assert state.active == 0
            state.shutdowns += 1

    modules = {
        "vllm": {"SamplingParams": SimpleNamespace},
        "vllm.engine.arg_utils": {"AsyncEngineArgs": SimpleNamespace},
        "vllm.platforms": {"current_platform": Platform()},
        "vllm.sampling_params": {"RequestOutputKind": SimpleNamespace(CUMULATIVE="cumulative")},
        "vllm.v1.engine.async_llm": {"AsyncLLM": AsyncLLM},
        "vllm_mblt.mblt_platform": {"MbltPlatform": Platform},
    }
    for name, attrs in modules.items():
        module = ModuleType(name)
        vars(module).update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(
        mobilint_vllm, "version", {"vllm": "0.32.0", "vllm-mblt": "0.10.0"}.__getitem__
    )
    monkeypatch.delenv("QBRUNTIME_VISIBLE_DEVICES", raising=False)
    return state


def test_sync_and_native_requests_share_one_engine(artifact, sdk):
    target = resolve_target("mobilint-aries-vllm", "onnxruntime", "cpu")
    assert resolve_target(None, "mobilint_vllm", "cpu") is target
    runtime = create_runtime(
        target.runtime_name, device=target.device, **target.runtime_options,
        max_num_seqs=16, max_model_len=4096, enable_prefix_caching=False,
    )
    try:
        runtime.load(artifact)
        runtime.warmup(_PROMPT)
        result = runtime.generate(_PROMPT, max_new_tokens=3, stop_token_ids=[23])
        np.testing.assert_array_equal(result.generated_ids, [[21, 22, 23]])
        assert result.generated_lengths.tolist() == [3]
        assert result.ttft_ms is not None and result.tpot_ms is not None
        assert [e.cumulative_tokens for e in result.generation_observation.events] == [1, 3]

        backend = runtime.create_native_backend(max_new_tokens=3, stop_token_ids=[23])
        executor = NativeAsyncRuntimeExecutor(backend, max_inflight=2, completion_timeout_sec=2)
        results = []

        def execute():
            execution = executor.execute(_PROMPT)
            executor.acknowledge(execution)
            results.append(execution)

        workers = [threading.Thread(target=execute) for _ in range(2)]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(2)

        assert len(results) == 2
        assert all(row.error_type is None and row.generated_tokens == 3 for row in results)
        assert all(row.outputs["generated_lengths"].tolist() == [3] for row in results)
        assert executor.shutdown(timeout=0)
        assert sdk.max_active == 2
        assert len(sdk.starts) == 1
        assert sdk.starts[0].model_loader_extra_config == {"dev_no": 0}
        assert sdk.starts[0].trust_remote_code is True
        prompt, sampling, _ = sdk.calls[-1]
        assert prompt == {"prompt_token_ids": [11, 12]}
        assert sampling.stop_token_ids == [23]
        assert sampling.temperature == 0 and sampling.output_kind == "cumulative"
        assert runtime.get_device_spec()["max_num_batched_tokens"] == 2048
    finally:
        runtime.unload()
    assert sdk.shutdowns == 1


def test_request_timeout_delivers_one_error_after_stream_exits(artifact, sdk):
    sdk.hang = True
    runtime = mobilint_vllm.MobilintVllmRuntime(request_timeout_sec=0.05)
    try:
        runtime.load(artifact)
        backend = runtime.create_native_backend(max_new_tokens=3)
        executor = NativeAsyncRuntimeExecutor(backend, max_inflight=1, completion_timeout_sec=2)
        execution = executor.execute(_PROMPT)
        assert execution.error_type == "TimeoutError"
        assert sdk.aborted == 1 and sdk.active == 0
        executor.acknowledge(execution)
        assert executor.shutdown(timeout=0)
        assert executor.snapshot().duplicate_callbacks == 0
    finally:
        runtime.unload()


def test_unload_cancels_inflight_request_and_releases_engine(artifact, sdk):
    sdk.hang = True
    runtime = mobilint_vllm.MobilintVllmRuntime()
    runtime.load(artifact)
    backend = runtime.create_native_backend(max_new_tokens=3)
    outcome = []
    backend.submit_async(_PROMPT, outcome.append)

    runtime.unload()
    assert sdk.aborted == 1
    assert sdk.shutdowns == 1
    assert len(outcome) == 1
    assert outcome[0].error_type == "CancelledError"
    with pytest.raises(RuntimeError, match="not accepting requests"):
        backend.submit_async(_PROMPT, outcome.append)
    runtime.unload()


def test_rejects_wrong_platform_then_can_reload(artifact, sdk, monkeypatch):
    platforms = sys.modules["vllm.platforms"]
    with monkeypatch.context() as patch:
        patch.setattr(platforms, "current_platform", object())
        runtime = mobilint_vllm.MobilintVllmRuntime()
        with pytest.raises(RuntimeError, match="did not activate"):
            runtime.load(artifact)
    assert not sdk.starts
    try:
        runtime.load(artifact)
        sdk.finish = False
        with pytest.raises(RuntimeError, match="without a final completion"):
            runtime.generate(_PROMPT)
        sdk.finish = True
        sdk.finish_reason = "abort"
        with pytest.raises(RuntimeError, match="reason: abort"):
            runtime.generate(_PROMPT)
        with pytest.raises(ValueError, match="one integer input_ids"):
            runtime.generate({"input_ids": np.array([[1], [2]])})
    finally:
        runtime.unload()


def test_cli_loads_registered_target_and_forwards_context(artifact, sdk, monkeypatch):
    import main

    model_path = artifact.artifact_path
    dataset = model_path / "dataset.json"
    dataset.write_text("{}", encoding="utf-8")
    loader = SimpleNamespace(total_samples=1, get_metadata=dict)

    def create_loader(**kwargs):
        assert kwargs["tokenizer_path"] == str(model_path)
        assert kwargs["max_length"] == 509
        return loader

    def benchmark(args, *, runtime, target, **kwargs):
        try:
            config = main.build_async_config(args)
            executor = main._build_async_runtime_executor(args, target, runtime, loader, config)
            execution = executor.execute(_PROMPT)
            executor.acknowledge(execution)
            assert execution.generated_tokens == 3
            assert executor.shutdown(timeout=0)
            diagnostics = main._safe_runtime_diagnostics(runtime)
            assert diagnostics["backend"] == "mobilint_vllm"
            assert diagnostics["max_num_seqs"] == 16
            assert diagnostics["max_num_batched_tokens"] == 2048
            assert diagnostics["enable_prefix_caching"] is False
            assert main._runtime_result_metadata(diagnostics) == {"runtime_version": "0.10.0"}
            return 0
        finally:
            runtime.unload()

    monkeypatch.setattr(main, "create_dataloader", create_loader)
    monkeypatch.setattr(main, "create_evaluator", lambda *args, **kwargs: None)
    monkeypatch.setattr(main, "create_decoder", lambda *args, **kwargs: None)
    monkeypatch.setattr(main, "execute_benchmark", benchmark)
    monkeypatch.setattr(sys, "argv", [
        "main.py", "--model", "llama-3.1-8b", "--target", "mobilint-aries-vllm",
        "--model-path", str(model_path), "--dataset", str(dataset),
        "--inference-mode", "async_queue", "--batch-size", "1", "--worker-count", "2",
        "--max-model-len", "512", "--max-new-tokens", "3",
        "--runtime-option", "max_num_seqs=16",
    ])
    assert main.main() == 0
    assert sdk.starts[0].max_model_len == 512
    assert sdk.starts[0].max_num_seqs == 16
    assert sdk.starts[0].model == str(model_path)
