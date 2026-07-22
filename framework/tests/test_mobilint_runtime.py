import sys
import types
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from core.compiled_model import CompiledModel
from core.model_spec import Model_Spec, Task
from runtimes.mobilint_rt import MobilintRuntime


def _compiled_model(tmp_path, *, suffix=".mxq", backend="mobilint"):
    artifact = tmp_path / f"model{suffix}"
    artifact.write_bytes(b"fake")
    spec = Model_Spec(
        name="two-input",
        task=Task.NLP_CLASSIFICATION,
        input_shapes={"input_ids": (1, 4), "attention_mask": (1, 4)},
        input_dtype={"input_ids": "int64", "attention_mask": "int64"},
        output_shapes={"logits": (1, 2), "hidden": (1, 4)},
        model_paths={"mxq": str(artifact)},
    )
    return CompiledModel(spec, backend, artifact)


def _install_fake_qbruntime(monkeypatch):
    state = {
        "accelerators": [],
        "configs": [],
        "models": [],
        "setter_results": {},
        "launch_error": None,
        "dispose_error": None,
    }

    class Accelerator:
        def __init__(self, device_id):
            self.device_id = device_id
            state["accelerators"].append(self)

    class ModelConfig:
        def __init__(self):
            self.calls = []
            state["configs"].append(self)

        def _record(self, name, *args):
            self.calls.append((name, *args))
            default = None if name in {"activation_slots", "async"} else True
            return state["setter_results"].get(name, default)

        def set_auto_core_mode(self):
            return self._record("auto")

        def set_single_core_mode(self, num_cores=None):
            return self._record("single", num_cores)

        def set_multi_core_mode(self):
            return self._record("multi")

        def set_global4_core_mode(self):
            return self._record("global4")

        def set_global8_core_mode(self):
            return self._record("global8")

        def set_activation_slots(self, count):
            return self._record("activation_slots", count)

        def set_async_pipeline_enabled(self, enabled):
            return self._record("async", enabled)

    class Model:
        def __init__(self, path, config):
            self.path = path
            self.config = config
            self.launches = []
            self.infer_calls = []
            self.dispose_calls = 0
            self.outputs = [
                np.array([[0.25, 0.75]], dtype=np.float32),
                np.arange(4, dtype=np.float32).reshape(1, 4),
            ]
            state["models"].append(self)

        def launch(self, accelerator):
            self.launches.append(accelerator)
            if state["launch_error"] is not None:
                raise state["launch_error"]

        def infer(self, inputs):
            self.infer_calls.append(inputs)
            return self.outputs

        def dispose(self):
            self.dispose_calls += 1
            if state["dispose_error"] is not None:
                raise state["dispose_error"]

    module = types.ModuleType("qbruntime")
    module.Accelerator = Accelerator
    module.ModelConfig = ModelConfig
    module.Model = Model
    monkeypatch.setitem(sys.modules, "qbruntime", module)
    return state


class FakeDeviceSession:
    instances = []
    release_error = None
    cleanup_pending = False

    def __init__(self, device_id, expected_family):
        self.device_id = device_id
        self.expected_family = expected_family
        self.release_calls = 0
        self.info = None
        self.__class__.instances.append(self)

    def acquire(self):
        if self.__class__.cleanup_pending:
            raise RuntimeError("device cleanup is incomplete")
        self.info = types.SimpleNamespace(
            device_id=self.device_id,
            device_type=1 if self.expected_family == "aries" else 2,
            family=self.expected_family,
        )
        return self.info

    def release(self):
        self.release_calls += 1
        if self.__class__.release_error is not None:
            self.__class__.cleanup_pending = True
            raise self.__class__.release_error
        self.info = None
        self.__class__.cleanup_pending = False


@pytest.fixture(autouse=True)
def fake_device_session(monkeypatch):
    FakeDeviceSession.instances = []
    FakeDeviceSession.release_error = None
    FakeDeviceSession.cleanup_pending = False
    monkeypatch.setattr(
        "runtimes.mobilint_rt.MobilintDeviceSession",
        FakeDeviceSession,
    )


def test_qbruntime_is_imported_only_during_load(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "qbruntime", None)
    runtime = MobilintRuntime(expected_family="aries")

    with pytest.raises(ImportError, match="mobilint-qb-runtime"):
        runtime.load(_compiled_model(tmp_path))

    assert FakeDeviceSession.instances[0].release_calls == 1


@pytest.mark.parametrize(
    ("core_mode", "num_cores", "expected_call"),
    [
        (None, None, None),
        ("auto", None, ("auto",)),
        ("single", None, ("single", None)),
        ("single", 3, ("single", 3)),
        ("multi", None, ("multi",)),
        ("global4", None, ("global4",)),
        ("global8", None, ("global8",)),
    ],
)
def test_load_configures_and_launches_exact_device(
    monkeypatch, tmp_path, core_mode, num_cores, expected_call
):
    state = _install_fake_qbruntime(monkeypatch)
    runtime = MobilintRuntime(
        device_id=2,
        expected_family="regulus",
        core_mode=core_mode,
        num_cores=num_cores,
        activation_slots=4,
        async_pipeline_enabled=True,
    )

    runtime.load(_compiled_model(tmp_path))

    config = state["configs"][0]
    if expected_call is None:
        assert not any(
            call[0] in {"auto", "single", "multi", "global4", "global8"}
            for call in config.calls
        )
    else:
        assert expected_call in config.calls
    assert config.calls[-2:] == [("activation_slots", 4), ("async", True)]
    assert state["accelerators"][0].device_id == 2
    assert state["models"][0].path.endswith("model.mxq")
    assert state["models"][0].launches == [state["accelerators"][0]]


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"core_mode": "invalid"}, "core_mode must be one of"),
        ({"core_mode": "multi", "num_cores": 2}, "only with core_mode='single'"),
        ({"num_cores": 2}, "only with core_mode='single'"),
        ({"core_mode": "single", "num_cores": 0}, "positive integer"),
        ({"activation_slots": 0}, "positive integer"),
        ({"async_pipeline_enabled": 1}, "must be a boolean"),
    ],
)
def test_init_rejects_invalid_core_and_pipeline_combinations(options, message):
    with pytest.raises(ValueError, match=message):
        MobilintRuntime(expected_family="aries", **options)


@pytest.mark.parametrize(
    ("setter_name", "options", "message"),
    [
        ("auto", {"core_mode": "auto"}, "core_mode=auto"),
        ("single", {"core_mode": "single"}, "core_mode=single"),
        ("multi", {"core_mode": "multi"}, "core_mode=multi"),
        ("global4", {"core_mode": "global4"}, "core_mode=global4"),
        ("global8", {"core_mode": "global8"}, "core_mode=global8"),
        ("activation_slots", {"activation_slots": 2}, "activation_slots"),
        ("async", {"async_pipeline_enabled": True}, "async pipeline"),
    ],
)
def test_load_rolls_back_when_sdk_rejects_configuration(
    monkeypatch, tmp_path, setter_name, options, message
):
    state = _install_fake_qbruntime(monkeypatch)
    state["setter_results"][setter_name] = False
    runtime = MobilintRuntime(expected_family="aries", **options)

    with pytest.raises(RuntimeError, match=message):
        runtime.load(_compiled_model(tmp_path))

    assert state["models"] == []
    assert FakeDeviceSession.instances[0].release_calls == 1
    assert runtime.compiled_model is None


def test_load_disposes_constructed_model_and_releases_when_launch_fails(
    monkeypatch, tmp_path
):
    state = _install_fake_qbruntime(monkeypatch)
    state["launch_error"] = RuntimeError("launch failed")
    runtime = MobilintRuntime(expected_family="aries")

    with pytest.raises(RuntimeError, match="launch failed"):
        runtime.load(_compiled_model(tmp_path))

    assert state["models"][0].dispose_calls == 1
    assert FakeDeviceSession.instances[0].release_calls == 1
    assert runtime.compiled_model is None
    assert runtime._model is None


def test_launch_failure_retains_model_when_rollback_disposal_fails(
    monkeypatch, tmp_path
):
    state = _install_fake_qbruntime(monkeypatch)
    launch_error = RuntimeError("launch failed")
    dispose_error = RuntimeError("dispose failed")
    state["launch_error"] = launch_error
    state["dispose_error"] = dispose_error
    runtime = MobilintRuntime(expected_family="aries")

    with pytest.raises(
        RuntimeError,
        match=(
            "load failed and rollback cleanup is incomplete.*dispose failed"
            r".*call unload\(\) to retry cleanup"
        ),
    ) as caught:
        runtime.load(_compiled_model(tmp_path))

    assert caught.value.__cause__ is launch_error
    assert runtime._model is state["models"][0]
    assert runtime._device_session is FakeDeviceSession.instances[0]
    assert FakeDeviceSession.instances[0].release_calls == 0
    with pytest.raises(RuntimeError, match="cleanup is incomplete"):
        runtime.load(_compiled_model(tmp_path))
    with pytest.raises(RuntimeError, match="cleanup is incomplete"):
        runtime.run({})

    state["dispose_error"] = None
    runtime.unload()
    assert state["models"][0].dispose_calls == 2
    assert FakeDeviceSession.instances[0].release_calls == 1


def test_launch_failure_retains_session_when_rollback_release_fails(
    monkeypatch, tmp_path
):
    state = _install_fake_qbruntime(monkeypatch)
    launch_error = RuntimeError("launch failed")
    release_error = RuntimeError("release failed")
    state["launch_error"] = launch_error
    FakeDeviceSession.release_error = release_error
    runtime = MobilintRuntime(expected_family="aries")

    with pytest.raises(
        RuntimeError,
        match=(
            "load failed and rollback cleanup is incomplete.*release failed"
            r".*call unload\(\) to retry cleanup"
        ),
    ) as caught:
        runtime.load(_compiled_model(tmp_path))

    assert caught.value.__cause__ is launch_error
    assert runtime._model is None
    assert runtime._device_session is FakeDeviceSession.instances[0]
    with pytest.raises(RuntimeError, match="cleanup is incomplete"):
        runtime.load(_compiled_model(tmp_path))
    with pytest.raises(RuntimeError, match="cleanup is incomplete"):
        runtime.run({})

    FakeDeviceSession.release_error = None
    runtime.unload()
    assert state["models"][0].dispose_calls == 1
    assert FakeDeviceSession.instances[0].release_calls == 2


def test_run_preserves_spec_input_and_output_order(monkeypatch, tmp_path):
    state = _install_fake_qbruntime(monkeypatch)
    runtime = MobilintRuntime(expected_family="aries")
    runtime.load(_compiled_model(tmp_path))
    input_ids = np.array([[1, 2, 3, 4]], dtype=np.int64)
    attention_mask = np.array([[1, 1, 1, 0]], dtype=np.int64)

    outputs = runtime.run(
        {"attention_mask": attention_mask, "input_ids": input_ids}
    )

    submitted = state["models"][0].infer_calls[0]
    assert isinstance(submitted, list)
    np.testing.assert_array_equal(submitted[0], input_ids)
    np.testing.assert_array_equal(submitted[1], attention_mask)
    assert list(outputs) == ["logits", "hidden"]
    np.testing.assert_array_equal(outputs["logits"], [[0.25, 0.75]])


def test_run_submits_contiguous_inputs(monkeypatch, tmp_path):
    state = _install_fake_qbruntime(monkeypatch)
    runtime = MobilintRuntime(expected_family="aries")
    runtime.load(_compiled_model(tmp_path))
    input_ids = np.arange(8, dtype=np.int64).reshape(1, 8)[:, ::2]
    attention_mask = np.arange(8, dtype=np.int64).reshape(1, 8)[:, 1::2]
    assert not input_ids.flags.c_contiguous
    assert not attention_mask.flags.c_contiguous

    runtime.run({"input_ids": input_ids, "attention_mask": attention_mask})

    submitted = state["models"][0].infer_calls[0]
    assert all(value.flags.c_contiguous for value in submitted)
    np.testing.assert_array_equal(submitted[0], input_ids)
    np.testing.assert_array_equal(submitted[1], attention_mask)


def test_run_rejects_missing_input_and_bad_sdk_outputs(monkeypatch, tmp_path):
    state = _install_fake_qbruntime(monkeypatch)
    runtime = MobilintRuntime(expected_family="aries")
    runtime.load(_compiled_model(tmp_path))

    with pytest.raises(ValueError, match="missing required inputs: attention_mask"):
        runtime.run({"input_ids": np.ones((1, 4), dtype=np.int64)})
    assert state["models"][0].infer_calls == []

    state["models"][0].outputs = None
    with pytest.raises(RuntimeError, match="returned no outputs"):
        runtime.run(
            {
                "input_ids": np.ones((1, 4), dtype=np.int64),
                "attention_mask": np.ones((1, 4), dtype=np.int64),
            }
        )

    state["models"][0].outputs = [np.ones((1, 2), dtype=np.float32)]
    with pytest.raises(RuntimeError, match="expected 2 outputs, received 1"):
        runtime.run(
            {
                "input_ids": np.ones((1, 4), dtype=np.int64),
                "attention_mask": np.ones((1, 4), dtype=np.int64),
            }
        )


def test_unload_disposes_and_releases_exactly_once(monkeypatch, tmp_path):
    state = _install_fake_qbruntime(monkeypatch)
    runtime = MobilintRuntime(expected_family="aries")
    runtime.load(_compiled_model(tmp_path))

    runtime.unload()
    runtime.unload()

    assert state["models"][0].dispose_calls == 1
    assert FakeDeviceSession.instances[0].release_calls == 1


def test_unload_retains_state_when_model_disposal_fails(monkeypatch, tmp_path):
    state = _install_fake_qbruntime(monkeypatch)
    runtime = MobilintRuntime(expected_family="aries")
    runtime.load(_compiled_model(tmp_path))
    state["dispose_error"] = RuntimeError("dispose failed")

    with pytest.raises(RuntimeError, match="dispose failed"):
        runtime.unload()

    assert runtime._model is state["models"][0]
    assert runtime._device_session is FakeDeviceSession.instances[0]
    assert FakeDeviceSession.instances[0].release_calls == 0
    with pytest.raises(RuntimeError, match="cleanup is incomplete"):
        runtime.load(_compiled_model(tmp_path))
    with pytest.raises(RuntimeError, match="cleanup is incomplete"):
        runtime.run({})

    state["dispose_error"] = None
    runtime.unload()
    runtime.unload()
    assert state["models"][0].dispose_calls == 2
    assert FakeDeviceSession.instances[0].release_calls == 1


def test_unload_retains_state_when_device_release_fails(monkeypatch, tmp_path):
    state = _install_fake_qbruntime(monkeypatch)
    runtime = MobilintRuntime(expected_family="aries")
    runtime.load(_compiled_model(tmp_path))
    FakeDeviceSession.release_error = RuntimeError("release failed")

    with pytest.raises(RuntimeError, match="release failed"):
        runtime.unload()

    assert runtime._model is None
    assert runtime._device_session is FakeDeviceSession.instances[0]
    with pytest.raises(RuntimeError, match="cleanup is incomplete"):
        runtime.load(_compiled_model(tmp_path))
    with pytest.raises(RuntimeError, match="cleanup is incomplete"):
        runtime.run({})

    FakeDeviceSession.release_error = None
    runtime.unload()
    runtime.unload()
    assert state["models"][0].dispose_calls == 1
    assert FakeDeviceSession.instances[0].release_calls == 2

    runtime.load(_compiled_model(tmp_path))
    runtime.unload()
    assert len(state["models"]) == 2
    assert state["models"][1].dispose_calls == 1
    assert FakeDeviceSession.instances[1].release_calls == 1


def test_compatibility_requires_mxq_and_mobilint_backend(tmp_path):
    runtime = MobilintRuntime(expected_family="aries")

    assert runtime.is_compatible(_compiled_model(tmp_path)) is True
    assert runtime.is_compatible(_compiled_model(tmp_path, suffix=".bin")) is False
    assert (
        runtime.is_compatible(_compiled_model(tmp_path, backend="onnxruntime"))
        is False
    )
