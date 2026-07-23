import asyncio
import threading
from copy import deepcopy
from importlib.metadata import PackageNotFoundError
from pathlib import Path

import numpy as np
import pytest

from core.compiled_model import CompiledModel
from core.model_spec import Model_Spec, Task
from runtimes.rbln_rt import RblnRuntime


class FakeTensor:
    def __init__(self, name, shape, dtype):
        self.name = name
        self.shape = shape
        self.dtype = dtype


class FakeInspect:
    compiler_version = "0.11.0"
    npu = "RBLN-CA22"
    tensor_parallel_size = 1
    uuid = "artifact-uuid"
    alloc_per_node = (4096,)
    inputs = (
        FakeTensor("input_ids", (1, 8), "int64"),
        FakeTensor("attention_mask", (1, 8), "int64"),
    )
    outputs = (FakeTensor("logits", (1, 2), "float32"),)


class FakeRBLNCompiledModel:
    owner = None

    @classmethod
    def inspect(cls, path):
        owner = cls.owner
        owner.inspect_calls.append(path)
        if owner.inspect_error is not None:
            raise owner.inspect_error
        return deepcopy(owner.inspected)


class FakeSyncRuntime:
    def __init__(self, owner):
        self.owner = owner
        self.calls = []
        self.outputs = owner.runtime_outputs

    def __call__(self, *inputs):
        self.calls.append(inputs)
        if self.owner.runtime_call_error is not None:
            raise self.owner.runtime_call_error
        return self.outputs

    def __del__(self):
        try:
            self.owner.destruction_threads.append(threading.get_ident())
        except BaseException:
            pass


class FakeAsyncRuntime:
    def __init__(self, owner):
        self.owner = owner

    async def async_run(self, *inputs):
        owner = self.owner
        loop = asyncio.get_running_loop()
        with owner.async_condition:
            call_number = len(owner.async_run_threads) + 1
            call_id = f"async-{call_number}"
            gate = asyncio.Event()
            owner.async_run_threads.append(threading.get_ident())
            owner.async_input_refs.append(tuple(inputs))
            owner.async_pending[call_id] = (loop, gate)
            owner.async_condition.notify_all()
        await gate.wait()
        with owner.async_condition:
            owner.async_pending.pop(call_id, None)
            owner.async_input_refs[call_number - 1] = ()
            error = owner.async_errors.pop(call_number, None)
            output = owner.async_outputs.pop(
                call_number,
                np.array([[0.25, 0.75]], dtype=np.float32),
            )
            owner.async_condition.notify_all()
        if error is not None:
            raise error
        return output

    def __del__(self):
        try:
            gate = self.owner.async_destruction_gate
            if gate is not None:
                self.owner.async_destruction_entered.set()
                gate.wait()
            self.owner.destruction_threads.append(threading.get_ident())
        except BaseException:
            pass


class FakeRebel:
    __version__ = "0.11.0"
    RBLNCompiledModel = FakeRBLNCompiledModel

    def __init__(self):
        self.available = True
        self.detected_npu = "RBLN-CA22"
        self.inspected = FakeInspect()
        self.inspect_error = None
        self.runtime_error = None
        self.runtime_call_error = None
        self.async_runtime_error = None
        self.async_constructor_gate = None
        self.async_constructor_entered = threading.Event()
        self.async_destruction_gate = None
        self.async_destruction_entered = threading.Event()
        self.runtime_outputs = np.array([[0.25, 0.75]], dtype=np.float32)
        self.availability_calls = []
        self.name_calls = []
        self.inspect_calls = []
        self.runtime_calls = []
        self.async_runtime_calls = []
        self.sync_instances = []
        self.async_calls = []
        self.destruction_threads = []
        self.async_constructor_thread = None
        self.async_run_threads = []
        self.async_input_refs = []
        self.async_pending = {}
        self.async_outputs = {}
        self.async_errors = {}
        self.async_condition = threading.Condition()

    def npu_is_available(self, device_id):
        self.availability_calls.append(device_id)
        return self.available

    def get_npu_name(self, device_id):
        self.name_calls.append(device_id)
        return self.detected_npu

    def Runtime(self, path, **kwargs):
        self.runtime_calls.append((path, kwargs))
        if self.runtime_error is not None:
            raise self.runtime_error
        instance = FakeSyncRuntime(self)
        self.sync_instances.append(instance)
        return instance

    def AsyncRuntime(self, path, **kwargs):
        self.async_constructor_thread = threading.get_ident()
        self.async_runtime_calls.append((path, kwargs))
        self.async_constructor_entered.set()
        if self.async_constructor_gate is not None:
            self.async_constructor_gate.wait()
        if self.async_runtime_error is not None:
            raise self.async_runtime_error
        loop = asyncio.get_event_loop()

        def keep_sandboxed_loop_responsive():
            if not loop.is_closed():
                loop.call_later(0.005, keep_sandboxed_loop_responsive)

        loop.call_later(0.005, keep_sandboxed_loop_responsive)
        return FakeAsyncRuntime(self)

    def wait_for_async_calls(self, count, timeout=1.0):
        with self.async_condition:
            return self.async_condition.wait_for(
                lambda: len(self.async_run_threads) >= count,
                timeout=timeout,
            )

    def release_call(self, call_number):
        call_id = f"async-{call_number}"
        with self.async_condition:
            pending = self.async_pending.get(call_id)
        if pending is None:
            return False
        loop, gate = pending
        loop.call_soon_threadsafe(gate.set)
        return True


@pytest.fixture
def fake_rebel(monkeypatch):
    def missing_distribution_metadata(name):
        assert name == "rebel-compiler"
        raise PackageNotFoundError(name)

    monkeypatch.setattr(
        "runtimes.rbln_rt.importlib_metadata.version",
        missing_distribution_metadata,
    )
    fake = FakeRebel()
    FakeRBLNCompiledModel.owner = fake
    yield fake
    FakeRBLNCompiledModel.owner = None


def compiled_model(
    path,
    *,
    backend="rbln",
    input_shapes=None,
    input_dtypes=None,
    output_shapes=None,
):
    path = Path(path)
    path.touch(exist_ok=True)
    spec = Model_Spec(
        name="bert",
        task=Task.NLP_CLASSIFICATION,
        input_shapes=input_shapes
        or {"input_ids": (1, 8), "attention_mask": (1, 8)},
        input_dtype=input_dtypes
        or {"input_ids": "int64", "attention_mask": "int64"},
        output_shapes=output_shapes or {"logits": (1, 2)},
        model_paths={"rbln": str(path)},
    )
    return CompiledModel(spec, backend, path)


def load_with_fake(monkeypatch, fake_rebel, model, **runtime_options):
    monkeypatch.setattr(
        "runtimes.rbln_rt.import_module", lambda name: fake_rebel
    )
    runtime = RblnRuntime(**runtime_options)
    runtime.load(model)
    return runtime


@pytest.fixture
def loaded_runtime(tmp_path, monkeypatch, fake_rebel):
    return load_with_fake(
        monkeypatch,
        fake_rebel,
        compiled_model(tmp_path / "loaded.rbln"),
        runtime_timeout_sec=17,
        max_async_inflight=4,
    )


def valid_inputs():
    return {
        "attention_mask": np.ones((1, 8), dtype=np.int64),
        "input_ids": np.arange(8, dtype=np.int64).reshape(1, 8),
    }
