import builtins
import importlib
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

import mobilint_device
from core.compiled_model import CompiledModel
from core.model_spec import Model_Spec, Task
from core.runtime_executor import GenerationOutputEvent


def _spec(task: Task = Task.NLP_GENERATION) -> Model_Spec:
    return Model_Spec(
        name="llama",
        task=task,
        input_shapes={"input_ids": (1, 8)},
        input_dtype={"input_ids": "int64"},
        output_shapes={"generated_ids": (1, 4)},
    )


@pytest.fixture
def compiled_model(tmp_path: Path) -> CompiledModel:
    model_dir = tmp_path / "llama-mxq"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({"max_batch_size": 1}),
        encoding="utf-8",
    )
    return CompiledModel(_spec(), "mobilint_llm", model_dir)


@pytest.fixture
def fake_sdk(monkeypatch):
    runtime_module = importlib.import_module("runtimes.mobilint_llm_rt")
    state = types.SimpleNamespace(
        sessions=[],
        model=None,
        model_load_calls=[],
        model_load_error=None,
        acquire_error=None,
        acquire_retains_owner=False,
        release_errors=[],
        dispose_errors=[],
        tensor_calls=[],
        no_grad_entries=0,
        no_grad_exits=0,
    )

    class FakeDeviceSession:
        def __init__(self, device_id, expected_family):
            self.device_id = device_id
            self.expected_family = expected_family
            self.acquire_calls = 0
            self.release_calls = 0
            self.info = None
            self.module = None
            state.sessions.append(self)

        def acquire(self):
            self.acquire_calls += 1
            if state.acquire_error is not None:
                if state.acquire_retains_owner:
                    self.module = object()
                raise state.acquire_error
            self.module = object()
            self.info = types.SimpleNamespace(
                device_id=self.device_id,
                device_type=1,
                family="aries",
            )
            return self.info

        def release(self):
            self.release_calls += 1
            if state.release_errors:
                raise state.release_errors.pop(0)
            self.info = None
            self.module = None

    class FakeModel:
        def __init__(self):
            self.dispose_calls = 0
            self.generate_calls = 0
            self.generate_kwargs = None
            self.callbacks = [[21], [22], [23]]
            self.returned_tokens = None

        def dispose(self):
            self.dispose_calls += 1
            if state.dispose_errors:
                raise state.dispose_errors.pop(0)

        def generate(self, **kwargs):
            self.generate_calls += 1
            self.generate_kwargs = kwargs
            streamer = kwargs["streamer"]
            prompt = _to_numpy(kwargs["input_ids"])
            streamer.put(kwargs["input_ids"])
            for tokens in self.callbacks:
                streamer.put(np.asarray(tokens, dtype=np.int64))
            batch_size = prompt.shape[0]
            if batch_size == 1:
                continuation = (
                    np.concatenate(
                        [
                            np.asarray(tokens, dtype=np.int64).reshape(-1)
                            for tokens in self.callbacks
                        ]
                    ).reshape(1, -1)
                    if self.callbacks
                    else np.empty((1, 0), dtype=np.int64)
                )
            else:
                continuation = (
                    np.stack(
                        [
                            np.asarray(tokens, dtype=np.int64).reshape(batch_size)
                            for tokens in self.callbacks
                        ],
                        axis=1,
                    )
                    if self.callbacks
                    else np.empty((batch_size, 0), dtype=np.int64)
                )
            if self.returned_tokens is not None:
                continuation = np.asarray(self.returned_tokens, dtype=np.int64)
                if continuation.ndim == 1:
                    continuation = continuation.reshape(1, -1)
            return np.concatenate([prompt, continuation], axis=1)

    state.model = FakeModel()

    class FakeAutoModelForCausalLM:
        @classmethod
        def from_pretrained(cls, model_path, **kwargs):
            state.model_load_calls.append((model_path, kwargs))
            if state.model_load_error is not None:
                raise state.model_load_error
            return state.model

    monkeypatch.setattr(runtime_module, "MobilintDeviceSession", FakeDeviceSession)

    package_names = (
        "mblt_model_zoo",
        "mblt_model_zoo.hf_transformers",
        "mblt_model_zoo.hf_transformers.models",
        "mblt_model_zoo.hf_transformers.models.llama",
    )
    for name in package_names:
        package = types.ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)
    registration_name = (
        "mblt_model_zoo.hf_transformers.models.llama.modeling_llama"
    )
    monkeypatch.setitem(
        sys.modules,
        registration_name,
        types.ModuleType(registration_name),
    )

    transformers = types.ModuleType("transformers")
    transformers.AutoModelForCausalLM = FakeAutoModelForCausalLM
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    class FakeTensor:
        def __init__(self, value):
            self._value = np.asarray(value)

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self._value

    class NoGrad:
        def __enter__(self):
            state.no_grad_entries += 1

        def __exit__(self, exc_type, exc_value, traceback):
            state.no_grad_exits += 1

    torch = types.ModuleType("torch")
    torch.long = object()

    def as_tensor(value, dtype=None, device=None):
        state.tensor_calls.append(
            {
                "value": np.asarray(value).copy(),
                "dtype": dtype,
                "device": device,
            }
        )
        return FakeTensor(value)

    torch.as_tensor = as_tensor
    torch.tensor = as_tensor
    torch.no_grad = NoGrad
    monkeypatch.setitem(sys.modules, "torch", torch)
    return runtime_module, state


def _to_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _fail_vendor_import(monkeypatch, blocked_name, error):
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == blocked_name or name.startswith(f"{blocked_name}."):
            raise error
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    return error


def _block_vendor_import(monkeypatch, blocked_name):
    missing = ModuleNotFoundError(
        f"No module named '{blocked_name}'",
        name=blocked_name,
    )
    _fail_vendor_import(monkeypatch, blocked_name, missing)
    return missing


def test_import_does_not_load_vendor_packages(monkeypatch):
    module_name = "runtimes.mobilint_llm_rt"
    previous = sys.modules.pop(module_name, None)
    attempted = []
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if (
            name in {"mbltml", "torch", "transformers"}
            or name.startswith("mblt_model_zoo")
        ):
            attempted.append(name)
            raise AssertionError(f"eager vendor import: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    try:
        importlib.import_module(module_name)
    finally:
        sys.modules.pop(module_name, None)
        if previous is not None:
            sys.modules[module_name] = previous

    assert attempted == []


@pytest.mark.parametrize("device_id", [True, 0.0, "0", 1, -1])
def test_constructor_requires_exact_device_zero(device_id):
    from runtimes.mobilint_llm_rt import MobilintLlmRuntime

    with pytest.raises(ValueError, match="ARIES device 0 only"):
        MobilintLlmRuntime(device_id=device_id)


def test_constructor_requires_exact_aries_family():
    from runtimes.mobilint_llm_rt import MobilintLlmRuntime

    with pytest.raises(ValueError, match="ARIES device 0 only"):
        MobilintLlmRuntime(expected_family="regulus")


def test_load_requires_local_directory_and_passes_explicit_dev_no(
    fake_sdk, compiled_model
):
    runtime_module, state = fake_sdk
    runtime = runtime_module.MobilintLlmRuntime()

    runtime.load(compiled_model)

    assert state.model_load_calls == [
        (str(compiled_model.artifact_path), {"dev_no": 0})
    ]
    assert state.sessions[0].device_id == 0
    assert state.sessions[0].expected_family == "aries"
    assert runtime.compiled_model is compiled_model
    assert runtime.max_dynamic_batch_size() == 1
    assert runtime.supports_dynamic_batching() is False
    assert runtime.supports_batch_generation() is False


@pytest.mark.parametrize(
    ("config_text", "message"),
    [
        ("{}", "max_batch_size"),
        ('{"max_batch_size": true}', "positive integer"),
        ('{"max_batch_size": 0}', "positive integer"),
        ("not-json", "valid JSON"),
    ],
)
def test_load_rejects_invalid_artifact_capacity_before_device_acquisition(
    fake_sdk,
    compiled_model,
    config_text,
    message,
):
    runtime_module, state = fake_sdk
    (compiled_model.artifact_path / "config.json").write_text(
        config_text,
        encoding="utf-8",
    )
    runtime = runtime_module.MobilintLlmRuntime()

    with pytest.raises(ValueError, match=message):
        runtime.load(compiled_model)

    assert state.sessions == []
    assert state.model_load_calls == []


@pytest.mark.parametrize("capacity", [16, 32])
def test_load_exposes_grouped_generation_capacity(
    fake_sdk,
    compiled_model,
    capacity,
):
    runtime_module, _ = fake_sdk
    (compiled_model.artifact_path / "config.json").write_text(
        json.dumps({"max_batch_size": capacity}),
        encoding="utf-8",
    )
    runtime = runtime_module.MobilintLlmRuntime()

    runtime.load(compiled_model)

    assert runtime.max_dynamic_batch_size() == capacity
    assert runtime.supports_dynamic_batching() is True
    assert runtime.supports_batch_generation() is True
    assert runtime.max_concurrent_workers() == 1
    assert runtime.get_device_spec()["max_batch_size"] == capacity


def test_incompatible_task_and_non_directory_are_rejected(
    fake_sdk, compiled_model, tmp_path
):
    runtime_module, state = fake_sdk
    runtime = runtime_module.MobilintLlmRuntime()
    wrong_task = CompiledModel(
        _spec(Task.NLP_CLASSIFICATION),
        "mobilint_llm",
        compiled_model.artifact_path,
    )
    model_file = tmp_path / "llama.mxq"
    model_file.write_bytes(b"mxq")
    wrong_artifact = CompiledModel(_spec(), "mobilint_llm", model_file)

    assert runtime.is_compatible(compiled_model)
    assert not runtime.is_compatible(wrong_task)
    assert not runtime.is_compatible(wrong_artifact)
    with pytest.raises(ValueError, match="local NLP generation model directory"):
        runtime.load(wrong_task)
    with pytest.raises(ValueError, match="local NLP generation model directory"):
        runtime.load(wrong_artifact)
    assert state.sessions == []


def test_load_failure_releases_device_session(fake_sdk, compiled_model):
    runtime_module, state = fake_sdk
    load_error = RuntimeError("load failed")
    state.model_load_error = load_error
    runtime = runtime_module.MobilintLlmRuntime()

    with pytest.raises(
        RuntimeError,
        match="Mobilint Model Zoo model load failed",
    ) as exc:
        runtime.load(compiled_model)

    assert exc.value.__cause__ is load_error
    assert str(exc.value) == "Mobilint Model Zoo model load failed."
    assert state.sessions[0].release_calls == 1
    assert runtime.compiled_model is None
    assert runtime._device_session is None


def test_missing_model_zoo_is_actionable_and_releases_session(
    fake_sdk, compiled_model, monkeypatch
):
    runtime_module, state = fake_sdk
    missing = _block_vendor_import(monkeypatch, "mblt_model_zoo")
    runtime = runtime_module.MobilintLlmRuntime()

    with pytest.raises(
        ImportError,
        match=r"mblt-model-zoo\[transformers\].*Transformers",
    ) as caught:
        runtime.load(compiled_model)

    assert caught.value.__cause__ is missing
    assert state.sessions[0].release_calls == 1
    assert runtime.compiled_model is None
    assert runtime._model is None
    assert runtime._device_session is None
    assert runtime._device_info is None
    assert runtime._cleanup_pending is False


def test_missing_transformers_is_actionable_and_releases_session(
    fake_sdk, compiled_model, monkeypatch
):
    runtime_module, state = fake_sdk
    missing = _block_vendor_import(monkeypatch, "transformers")
    runtime = runtime_module.MobilintLlmRuntime()

    with pytest.raises(
        ImportError,
        match=r"mblt-model-zoo\[transformers\].*Transformers",
    ) as caught:
        runtime.load(compiled_model)

    assert caught.value.__cause__ is missing
    assert state.sessions[0].release_calls == 1
    assert runtime.compiled_model is None
    assert runtime._model is None
    assert runtime._device_session is None
    assert runtime._device_info is None
    assert runtime._cleanup_pending is False


def test_model_zoo_import_execution_failure_is_sanitized_after_cleanup(
    fake_sdk, compiled_model, monkeypatch
):
    runtime_module, state = fake_sdk
    import_error = RuntimeError("registration failed")
    _fail_vendor_import(monkeypatch, "mblt_model_zoo", import_error)
    runtime = runtime_module.MobilintLlmRuntime()

    with pytest.raises(RuntimeError) as caught:
        runtime.load(compiled_model)

    assert str(caught.value) == "Mobilint Model Zoo model load failed."
    assert caught.value.__cause__ is import_error
    assert state.sessions[0].release_calls == 1
    assert runtime.compiled_model is None
    assert runtime._model is None
    assert runtime._device_session is None
    assert runtime._device_info is None
    assert runtime._cleanup_pending is False


def test_model_zoo_import_execution_and_release_failure_is_retryable(
    fake_sdk, compiled_model, monkeypatch
):
    runtime_module, state = fake_sdk
    import_error = OSError("registration library failed")
    _fail_vendor_import(monkeypatch, "mblt_model_zoo", import_error)
    state.release_errors.append(RuntimeError("shutdown failed"))
    runtime = runtime_module.MobilintLlmRuntime()

    with pytest.raises(RuntimeError, match="rollback cleanup is incomplete") as caught:
        runtime.load(compiled_model)

    assert caught.value.__cause__ is import_error
    assert "shutdown failed" in str(caught.value)
    assert state.sessions[0].release_calls == 1
    assert runtime.compiled_model is None
    assert runtime._model is None
    assert runtime._device_session is state.sessions[0]
    assert runtime._cleanup_pending is True

    runtime.unload()

    assert state.sessions[0].release_calls == 2
    assert runtime._device_session is None
    assert runtime._cleanup_pending is False


def test_missing_model_zoo_and_release_failure_retains_cleanup_owner(
    fake_sdk, compiled_model, monkeypatch
):
    runtime_module, state = fake_sdk
    state.release_errors.append(RuntimeError("shutdown failed"))
    missing = _block_vendor_import(monkeypatch, "mblt_model_zoo")
    runtime = runtime_module.MobilintLlmRuntime()

    with pytest.raises(RuntimeError, match="rollback cleanup is incomplete") as caught:
        runtime.load(compiled_model)

    assert caught.value.__cause__ is missing
    assert "shutdown failed" in str(caught.value)
    assert state.sessions[0].release_calls == 1
    assert runtime.compiled_model is None
    assert runtime._model is None
    assert runtime._device_session is state.sessions[0]
    assert runtime._cleanup_pending is True

    runtime.unload()

    assert state.sessions[0].release_calls == 2
    assert runtime._device_session is None
    assert runtime._cleanup_pending is False


def test_second_load_is_rejected(fake_sdk, compiled_model):
    runtime_module, state = fake_sdk
    runtime = runtime_module.MobilintLlmRuntime()
    runtime.load(compiled_model)

    with pytest.raises(RuntimeError, match="already loaded"):
        runtime.load(compiled_model)

    assert len(state.sessions) == 1
    assert len(state.model_load_calls) == 1


def test_unload_disposes_once_and_releases_session(fake_sdk, compiled_model):
    runtime_module, state = fake_sdk
    runtime = runtime_module.MobilintLlmRuntime()
    runtime.load(compiled_model)

    runtime.unload()
    runtime.unload()

    assert state.model.dispose_calls == 1
    assert state.sessions[0].release_calls == 1
    assert runtime.compiled_model is None


def test_dispose_failure_retains_cleanup_owner_and_blocks_use(
    fake_sdk, compiled_model
):
    runtime_module, state = fake_sdk
    state.dispose_errors.append(RuntimeError("dispose failed"))
    runtime = runtime_module.MobilintLlmRuntime()
    runtime.load(compiled_model)

    with pytest.raises(RuntimeError, match="dispose failed"):
        runtime.unload()

    assert runtime._model is state.model
    assert runtime._device_session is state.sessions[0]
    assert state.sessions[0].release_calls == 0
    for operation in (
        lambda: runtime.load(compiled_model),
        lambda: runtime.generate({"input_ids": np.array([[1]])}),
        lambda: runtime.run({"input_ids": np.array([[1]])}),
    ):
        with pytest.raises(RuntimeError, match="cleanup is incomplete"):
            operation()

    runtime.unload()
    assert state.model.dispose_calls == 2
    assert state.sessions[0].release_calls == 1


def test_release_failure_does_not_double_dispose_on_retry(fake_sdk, compiled_model):
    runtime_module, state = fake_sdk
    state.release_errors.append(RuntimeError("shutdown failed"))
    runtime = runtime_module.MobilintLlmRuntime()
    runtime.load(compiled_model)

    with pytest.raises(RuntimeError, match="shutdown failed"):
        runtime.unload()

    assert runtime._model is None
    assert runtime._device_session is state.sessions[0]
    assert state.model.dispose_calls == 1

    runtime.unload()
    assert state.model.dispose_calls == 1
    assert state.sessions[0].release_calls == 2


def test_load_and_rollback_cleanup_failure_is_actionable_and_retryable(
    fake_sdk, compiled_model
):
    runtime_module, state = fake_sdk
    load_error = RuntimeError("load failed")
    state.model_load_error = load_error
    state.release_errors.append(RuntimeError("shutdown failed"))
    runtime = runtime_module.MobilintLlmRuntime()

    with pytest.raises(RuntimeError, match="rollback cleanup is incomplete") as exc:
        runtime.load(compiled_model)

    assert exc.value.__cause__ is load_error
    assert "shutdown failed" in str(exc.value)
    assert runtime._device_session is state.sessions[0]
    assert runtime.compiled_model is None
    with pytest.raises(RuntimeError, match="cleanup is incomplete"):
        runtime.load(compiled_model)

    runtime.unload()
    assert state.sessions[0].release_calls == 2


def test_acquire_failure_publishes_cleanup_owner_before_rollback(
    fake_sdk, compiled_model
):
    runtime_module, state = fake_sdk
    acquire_error = RuntimeError("shutdown failed")
    state.acquire_error = acquire_error
    state.acquire_retains_owner = True
    runtime = runtime_module.MobilintLlmRuntime()

    with pytest.raises(RuntimeError, match="rollback cleanup is incomplete") as exc:
        runtime.load(compiled_model)

    assert exc.value.__cause__ is acquire_error
    assert runtime._device_session is state.sessions[0]
    assert runtime.compiled_model is None
    assert state.sessions[0].release_calls == 0

    state.acquire_error = None
    runtime.unload()
    assert state.sessions[0].release_calls == 1


class _ValidationFakeMbltml:
    MBLTML_DEVICE_ARIES = 1
    MBLTML_DEVICE_REGULUS = 2
    MBLTML_DEVICE_REGULUS_USB = 4

    def __init__(self):
        self.shutdown_calls = 0
        self.shutdown_error = None

    def mbltmlInitDevices(self, selected):
        self.selected = set(selected)

    def mbltmlGetDeviceCount(self):
        return 1

    def mbltmlGetDeviceType(self, device_id):
        return self.MBLTML_DEVICE_REGULUS

    def mbltmlShutdown(self):
        self.shutdown_calls += 1
        if self.shutdown_error is not None:
            raise self.shutdown_error


def test_real_acquire_rollback_failure_retains_owner_without_second_release(
    monkeypatch, compiled_model
):
    fake = _ValidationFakeMbltml()
    fake.shutdown_error = RuntimeError("shutdown failed")
    monkeypatch.setattr(mobilint_device, "_STATE", mobilint_device._MbltmlState())
    monkeypatch.setattr(mobilint_device, "import_module", lambda name: fake)
    runtime_module = importlib.import_module("runtimes.mobilint_llm_rt")
    runtime = runtime_module.MobilintLlmRuntime()

    with pytest.raises(
        RuntimeError,
        match=(
            "model load failed and rollback cleanup is incomplete"
            ".*shutdown failed.*call unload\\(\\) to retry cleanup"
        ),
    ) as caught:
        runtime.load(compiled_model)

    assert "shutdown failed" in str(caught.value.__cause__)
    assert "expected ARIES" in str(caught.value.__cause__.__context__)
    assert fake.shutdown_calls == 1
    assert mobilint_device._STATE.cleanup_pending is True
    assert runtime._device_session is not None
    assert runtime._cleanup_pending is True
    assert runtime.compiled_model is None
    for operation in (
        lambda: runtime.load(compiled_model),
        lambda: runtime.generate({"input_ids": np.array([[1]])}),
        lambda: runtime.run({"input_ids": np.array([[1]])}),
    ):
        with pytest.raises(RuntimeError, match="cleanup is incomplete"):
            operation()

    fake.shutdown_error = None
    runtime.unload()
    runtime.unload()

    assert fake.shutdown_calls == 2
    assert mobilint_device._STATE.cleanup_pending is False
    assert runtime._device_session is None
    assert runtime._cleanup_pending is False


def test_real_acquire_validation_failure_clears_fully_rolled_back_owner(
    monkeypatch, compiled_model
):
    fake = _ValidationFakeMbltml()
    monkeypatch.setattr(mobilint_device, "_STATE", mobilint_device._MbltmlState())
    monkeypatch.setattr(mobilint_device, "import_module", lambda name: fake)
    runtime_module = importlib.import_module("runtimes.mobilint_llm_rt")
    runtime = runtime_module.MobilintLlmRuntime()

    with pytest.raises(
        RuntimeError,
        match="Mobilint Model Zoo model load failed",
    ) as caught:
        runtime.load(compiled_model)

    assert "expected ARIES" in str(caught.value.__cause__)
    assert fake.shutdown_calls == 1
    assert runtime._device_session is None
    assert runtime._cleanup_pending is False
    assert mobilint_device._STATE.cleanup_pending is False


def test_real_acquire_missing_mbltml_preserves_actionable_import_error(
    monkeypatch, compiled_model
):
    missing = ModuleNotFoundError("No module named 'mbltml'", name="mbltml")

    def missing_mbltml(name):
        assert name == "mbltml"
        raise missing

    monkeypatch.setattr(mobilint_device, "_STATE", mobilint_device._MbltmlState())
    monkeypatch.setattr(mobilint_device, "import_module", missing_mbltml)
    runtime_module = importlib.import_module("runtimes.mobilint_llm_rt")
    runtime = runtime_module.MobilintLlmRuntime()

    with pytest.raises(ImportError, match="optional 'mbltml' package") as caught:
        runtime.load(compiled_model)

    assert caught.value.__cause__ is missing
    assert runtime.compiled_model is None
    assert runtime._model is None
    assert runtime._device_session is None
    assert runtime._device_info is None
    assert runtime._cleanup_pending is False
    assert mobilint_device._STATE.module is None
    assert mobilint_device._STATE.ref_count == 0
    assert mobilint_device._STATE.cleanup_pending is False


def test_get_device_spec_contains_only_safe_device_diagnostics(
    fake_sdk, compiled_model
):
    runtime_module, _ = fake_sdk
    runtime = runtime_module.MobilintLlmRuntime(device="npu:0")
    runtime.load(compiled_model)

    spec = runtime.get_device_spec()

    assert spec == {
        "backend": "mobilint_llm",
        "device": "npu:0",
        "device_id": 0,
        "expected_family": "aries",
        "detected_family": "aries",
        "device_type": 1,
        "accelerator_vendor": "Mobilint",
        "accelerator_name": "ARIES",
        "max_batch_size": 1,
    }
    assert str(compiled_model.artifact_path) not in repr(spec)
    assert "prompt" not in repr(spec).lower()


def _loaded_runtime(runtime_module, compiled_model, clock_values):
    runtime = runtime_module.MobilintLlmRuntime(
        clock_ns=iter(clock_values).__next__,
    )
    runtime.load(compiled_model)
    return runtime


def test_generate_records_prompt_free_token_events(fake_sdk, compiled_model):
    runtime_module, state = fake_sdk
    runtime = _loaded_runtime(
        runtime_module,
        compiled_model,
        [1_000_000, 6_000_000, 8_000_000, 12_000_000, 14_000_000],
    )

    result = runtime.generate(
        {
            "input_ids": np.array([[11, 12]], dtype=np.int64),
            "attention_mask": np.array([[1, 1]], dtype=np.int64),
        },
        max_new_tokens=3,
        stop_token_ids=[2, 3],
    )

    np.testing.assert_array_equal(result.generated_ids, [21, 22, 23])
    np.testing.assert_array_equal(result.generated_lengths, [3])
    assert result.num_tokens == 3
    assert result.ttft_ms == pytest.approx(5.0)
    assert result.tpot_ms == pytest.approx(3.0)
    assert result.total_ms == pytest.approx(13.0)
    assert result.timing_mode == "kv_cache"
    assert result.uses_kv_cache is True
    assert result.timing_source == "mobilint_transformers_streamer"
    assert result.generation_observation.backend_submitted_ns == 1_000_000
    assert result.generation_observation.source == "mobilint_transformers_streamer"
    assert result.generation_observation.events == (
        GenerationOutputEvent(observed_ns=6_000_000, cumulative_tokens=1),
        GenerationOutputEvent(observed_ns=8_000_000, cumulative_tokens=2),
        GenerationOutputEvent(observed_ns=12_000_000, cumulative_tokens=3),
    )
    kwargs = state.model.generate_kwargs
    np.testing.assert_array_equal(_to_numpy(kwargs["input_ids"]), [[11, 12]])
    np.testing.assert_array_equal(_to_numpy(kwargs["attention_mask"]), [[1, 1]])
    assert kwargs["max_new_tokens"] == 3
    assert kwargs["do_sample"] is False
    assert kwargs["eos_token_id"] == [2, 3]
    assert state.no_grad_entries == 1
    assert state.no_grad_exits == 1


def test_grouped_generate_uses_capacity_as_maximum_and_returns_row_lengths(
    fake_sdk,
    compiled_model,
):
    runtime_module, state = fake_sdk
    (compiled_model.artifact_path / "config.json").write_text(
        json.dumps({"max_batch_size": 16}),
        encoding="utf-8",
    )
    state.model.callbacks = [[21, 41], [2, 42], [0, 2]]
    runtime = _loaded_runtime(
        runtime_module,
        compiled_model,
        [1_000_000, 6_000_000, 8_000_000, 12_000_000, 14_000_000],
    )
    input_ids = np.array(
        [
            [0, 11, 12],
            [31, 32, 33],
        ],
        dtype=np.int64,
    )
    attention_mask = np.array(
        [
            [0, 1, 1],
            [1, 1, 1],
        ],
        dtype=np.int64,
    )

    result = runtime.generate(
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        },
        max_new_tokens=3,
        stop_token_ids=[2],
    )

    np.testing.assert_array_equal(
        result.generated_ids,
        [[21, 2, 0], [41, 42, 2]],
    )
    np.testing.assert_array_equal(result.generated_lengths, [2, 3])
    assert result.num_tokens == 5
    assert result.ttft_ms == pytest.approx(5.0)
    assert result.tpot_ms == pytest.approx(3.0)
    assert result.generation_observation.events == (
        GenerationOutputEvent(observed_ns=6_000_000, cumulative_tokens=2),
        GenerationOutputEvent(observed_ns=8_000_000, cumulative_tokens=4),
        GenerationOutputEvent(observed_ns=12_000_000, cumulative_tokens=6),
    )
    kwargs = state.model.generate_kwargs
    np.testing.assert_array_equal(_to_numpy(kwargs["input_ids"]), input_ids)
    np.testing.assert_array_equal(
        _to_numpy(kwargs["attention_mask"]),
        attention_mask,
    )


def test_grouped_generate_accepts_actual_batch_equal_to_capacity(
    fake_sdk,
    compiled_model,
):
    runtime_module, state = fake_sdk
    capacity = 16
    (compiled_model.artifact_path / "config.json").write_text(
        json.dumps({"max_batch_size": capacity}),
        encoding="utf-8",
    )
    state.model.callbacks = [list(range(100, 100 + capacity))]
    runtime = _loaded_runtime(
        runtime_module,
        compiled_model,
        [1_000_000, 2_000_000, 3_000_000],
    )

    result = runtime.generate(
        {
            "input_ids": np.ones((capacity, 2), dtype=np.int64),
            "attention_mask": np.ones((capacity, 2), dtype=np.int64),
        },
        max_new_tokens=1,
    )

    assert result.generated_ids.shape == (capacity, 1)
    np.testing.assert_array_equal(
        result.generated_lengths,
        np.ones(capacity, dtype=np.int64),
    )
    assert result.num_tokens == capacity


def test_grouped_generate_repacks_right_padding_as_compact_left_padding(
    fake_sdk,
    compiled_model,
):
    runtime_module, state = fake_sdk
    (compiled_model.artifact_path / "config.json").write_text(
        json.dumps({"max_batch_size": 16}),
        encoding="utf-8",
    )
    state.model.callbacks = [[21, 41]]
    runtime = _loaded_runtime(
        runtime_module,
        compiled_model,
        [1_000_000, 2_000_000, 3_000_000],
    )

    runtime.generate(
        {
            "input_ids": np.array(
                [
                    [11, 12, 2, 2],
                    [31, 32, 33, 2],
                ],
                dtype=np.int64,
            ),
            "attention_mask": np.array(
                [
                    [1, 1, 0, 0],
                    [1, 1, 1, 0],
                ],
                dtype=np.int64,
            ),
        },
        max_new_tokens=1,
    )

    kwargs = state.model.generate_kwargs
    np.testing.assert_array_equal(
        _to_numpy(kwargs["input_ids"]),
        [[2, 11, 12], [31, 32, 33]],
    )
    np.testing.assert_array_equal(
        _to_numpy(kwargs["attention_mask"]),
        [[0, 1, 1], [1, 1, 1]],
    )


def test_generate_rejects_actual_batch_above_artifact_capacity_before_sdk_call(
    fake_sdk,
    compiled_model,
):
    runtime_module, state = fake_sdk
    (compiled_model.artifact_path / "config.json").write_text(
        json.dumps({"max_batch_size": 16}),
        encoding="utf-8",
    )
    runtime = runtime_module.MobilintLlmRuntime()
    runtime.load(compiled_model)

    with pytest.raises(ValueError, match="actual batch size 17.*capacity 16"):
        runtime.generate(
            {
                "input_ids": np.ones((17, 2), dtype=np.int64),
                "attention_mask": np.ones((17, 2), dtype=np.int64),
            }
        )

    assert state.model.generate_calls == 0


def test_generate_missing_torch_is_actionable_and_lazy(
    fake_sdk, compiled_model, monkeypatch
):
    runtime_module, state = fake_sdk
    runtime = runtime_module.MobilintLlmRuntime()
    runtime.load(compiled_model)
    missing = _block_vendor_import(monkeypatch, "torch")

    with pytest.raises(ImportError, match="optional torch package") as caught:
        runtime.generate({"input_ids": np.array([[11, 12]])})

    assert caught.value.__cause__ is missing
    assert state.model.generate_calls == 0


def test_generate_one_token_has_no_tpot(fake_sdk, compiled_model):
    runtime_module, state = fake_sdk
    state.model.callbacks = [[21]]
    runtime = _loaded_runtime(
        runtime_module,
        compiled_model,
        [1_000_000, 6_000_000, 7_000_000],
    )

    result = runtime.generate({"input_ids": np.array([[11, 12]])})

    assert result.num_tokens == 1
    assert result.ttft_ms == pytest.approx(5.0)
    assert result.tpot_ms is None


def test_generate_zero_tokens_has_no_token_timings(fake_sdk, compiled_model):
    runtime_module, state = fake_sdk
    state.model.callbacks = []
    runtime = _loaded_runtime(
        runtime_module,
        compiled_model,
        [1_000_000, 4_000_000],
    )

    result = runtime.generate({"input_ids": np.array([[11, 12]])})

    np.testing.assert_array_equal(result.generated_ids, [])
    np.testing.assert_array_equal(result.generated_lengths, [0])
    assert result.num_tokens == 0
    assert result.ttft_ms is None
    assert result.tpot_ms is None
    assert result.total_ms == pytest.approx(3.0)
    assert result.generation_observation.events == ()


@pytest.mark.parametrize(
    ("input_ids", "attention_mask"),
    [
        ([0, 0, 11, 12], [0, 0, 1, 1]),
        ([11, 12, 0, 0], [1, 1, 0, 0]),
        ([0, 11, 0, 12], [0, 1, 0, 1]),
    ],
)
def test_generate_removes_left_right_and_internal_padding(
    fake_sdk, compiled_model, input_ids, attention_mask
):
    runtime_module, state = fake_sdk
    runtime = _loaded_runtime(
        runtime_module,
        compiled_model,
        [1_000_000, 2_000_000, 3_000_000, 4_000_000, 5_000_000],
    )
    original_ids = np.array([input_ids], dtype=np.int64)
    original_mask = np.array([attention_mask], dtype=np.int64)

    runtime.generate(
        {"input_ids": original_ids, "attention_mask": original_mask},
        max_new_tokens=3,
    )

    kwargs = state.model.generate_kwargs
    np.testing.assert_array_equal(_to_numpy(kwargs["input_ids"]), [[11, 12]])
    np.testing.assert_array_equal(_to_numpy(kwargs["attention_mask"]), [[1, 1]])
    np.testing.assert_array_equal(original_ids, [input_ids])
    np.testing.assert_array_equal(original_mask, [attention_mask])


def test_generate_rejects_batch_larger_than_one(fake_sdk, compiled_model):
    runtime_module, state = fake_sdk
    runtime = _loaded_runtime(runtime_module, compiled_model, [1_000_000])

    with pytest.raises(ValueError, match="batch size 1"):
        runtime.generate(
            {
                "input_ids": np.array([[11, 12], [21, 22]]),
                "attention_mask": np.ones((2, 2), dtype=np.int64),
            }
        )

    assert state.model.generate_calls == 0


def test_grouped_streamer_callback_preserves_one_cumulative_event(
    fake_sdk, compiled_model
):
    runtime_module, state = fake_sdk
    state.model.callbacks = [[21, 22], [23]]
    runtime = _loaded_runtime(
        runtime_module,
        compiled_model,
        [1_000_000, 6_000_000, 12_000_000, 14_000_000],
    )

    result = runtime.generate({"input_ids": np.array([[11, 12]])})

    assert result.generation_observation.events == (
        GenerationOutputEvent(observed_ns=6_000_000, cumulative_tokens=2),
        GenerationOutputEvent(observed_ns=12_000_000, cumulative_tokens=3),
    )
    assert result.tpot_ms == pytest.approx(3.0)


def test_streamer_output_count_mismatch_is_rejected(fake_sdk, compiled_model):
    runtime_module, state = fake_sdk
    state.model.callbacks = [[21], [22]]
    state.model.returned_tokens = [21]
    runtime = _loaded_runtime(
        runtime_module,
        compiled_model,
        [1_000_000, 2_000_000, 3_000_000, 4_000_000],
    )

    with pytest.raises(RuntimeError, match="streamed 2 tokens.*returned 1"):
        runtime.generate({"input_ids": np.array([[11, 12]])})


def test_returned_tokens_without_stream_events_are_rejected(fake_sdk, compiled_model):
    runtime_module, state = fake_sdk
    state.model.callbacks = []
    state.model.returned_tokens = [21]
    runtime = _loaded_runtime(
        runtime_module,
        compiled_model,
        [1_000_000, 2_000_000],
    )

    with pytest.raises(RuntimeError, match="streamed 0 tokens.*returned 1"):
        runtime.generate({"input_ids": np.array([[11, 12]])})


def test_run_and_warmup_delegate_to_greedy_generation(fake_sdk, compiled_model):
    runtime_module, state = fake_sdk
    state.model.callbacks = [[21]]
    runtime = _loaded_runtime(
        runtime_module,
        compiled_model,
        [
            1_000_000,
            2_000_000,
            3_000_000,
            4_000_000,
            5_000_000,
            6_000_000,
            7_000_000,
            8_000_000,
            9_000_000,
        ],
    )
    inputs = {"input_ids": np.array([[11, 12]])}

    outputs = runtime.run(inputs)
    runtime.warmup(inputs, num_runs=2)

    np.testing.assert_array_equal(outputs["generated_ids"], [21])
    assert state.model.generate_calls == 3
    assert state.model.generate_kwargs["max_new_tokens"] == 1
