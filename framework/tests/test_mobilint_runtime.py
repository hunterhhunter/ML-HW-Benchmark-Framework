import sys
import traceback
import types
from enum import Enum
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from core.compiled_model import CompiledModel
from core.mobilint_tensor_contracts import build_mobilint_tensor_contract
from core.model_spec import Model_Spec, Task
import mobilint_device
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


def _dynamic_bert_compiled_model(tmp_path):
    artifact = tmp_path / "sst2.mxq"
    artifact.write_bytes(b"fake")
    spec = Model_Spec(
        name="bert-base-uncased",
        task=Task.NLP_CLASSIFICATION,
        input_shapes={"embeddings": (1, -1, 768)},
        input_dtype={"embeddings": "float32"},
        output_shapes={"logits": (1, 2)},
        model_paths={"mxq": str(artifact)},
    )
    return CompiledModel(spec, "mobilint", artifact)


def _vision_compiled_model(tmp_path, profile):
    artifact = tmp_path / profile["artifact_basename"]
    artifact.write_bytes(b"fake")
    spec = Model_Spec(
        name=profile["model_name"],
        task=profile["task"],
        input_shapes={
            profile["input_name"]: (
                profile["max_input_batch_size"],
                *profile["expected_unbatched_input_shape"],
            )
        },
        input_dtype={
            profile["input_name"]: profile["expected_input_dtype"]
        },
        output_shapes={
            name: (profile["max_input_batch_size"], *shape)
            for name, shape in zip(
                profile["output_names"],
                profile.get("expected_unbatched_output_shapes", ()),
            )
        }
        or {"output": (profile["max_input_batch_size"], 1000)},
    )
    return CompiledModel(spec, "mobilint", artifact), _vision_contract(profile)


RESNET_PROFILE = {
    "vision_profile_id": "mobilint-resnet50-imagenet1k-v2",
    "model_name": "resnet50",
    "task": Task.IMAGE_CLASSIFICATION,
    "artifact_basename": "resnet50_IMAGENET1K_V2.mxq",
    "input_name": "input",
    "expected_input_dtype": "uint8",
    "expected_input_layout": "NHWC",
    "expected_unbatched_input_shape": (224, 224, 3),
    "max_input_batch_size": 1,
    "output_names": ("output",),
}

YOLO_PROFILE = {
    "vision_profile_id": "mobilint-yolov5m-default",
    "model_name": "yolov5m",
    "task": Task.OBJECT_DETECTION,
    "artifact_basename": "yolov5m.mxq",
    "input_name": "images",
    "expected_input_dtype": "uint8",
    "expected_input_layout": "NHWC",
    "expected_unbatched_input_shape": (640, 640, 3),
    "max_input_batch_size": 1,
    "output_names": ("stride32", "stride16", "stride8"),
    "expected_unbatched_output_shapes": (
        (20, 20, 255),
        (40, 40, 255),
        (80, 80, 255),
    ),
}


_VISION_CONTRACT_KEYS = {
    "vision_profile_id",
    "expected_input_dtype",
    "expected_input_layout",
    "expected_unbatched_input_shape",
    "max_input_batch_size",
    "expected_unbatched_output_shapes",
}


class DataType(Enum):
    Int64 = "Int64"
    Uint8 = "Uint8"
    Float32 = "Float32"
    Bool = "Bool"


class Cluster(Enum):
    Cluster0 = 0


class Core(Enum):
    Core0 = 0


class CoreId:
    def __init__(self, cluster, core):
        self.cluster = cluster
        self.core = core


def _vision_contract(profile, **overrides):
    contract = {
        key: value for key, value in profile.items() if key in _VISION_CONTRACT_KEYS
    }
    contract.update(overrides)
    return contract


def _set_matching_sdk_contract(state, profile):
    state["input_shapes"] = [profile["expected_unbatched_input_shape"]]
    state["input_dtypes"] = DataType.Uint8
    state["output_shapes"] = [
        shape
        for shape in profile.get("expected_unbatched_output_shapes", ())
    ] or [(1000,)]


def _install_fake_qbruntime(monkeypatch, *, missing_getters=()):
    state = {
        "accelerators": [],
        "configs": [],
        "models": [],
        "setter_results": {},
        "launch_error": None,
        "dispose_error": None,
        "getter_errors": {},
        "input_shapes": [(1, 4), (1, 4)],
        "input_dtypes": DataType.Int64,
        "output_shapes": [(1, 2), (1, 4)],
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

        def set_single_core_mode(self, num_cores=None, core_ids=None):
            return self._record("single", num_cores, core_ids)

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

        def get_model_input_shape(self):
            if "get_model_input_shape" in state["getter_errors"]:
                raise state["getter_errors"]["get_model_input_shape"]
            return state["input_shapes"]

        def get_model_input_data_type(self):
            if "get_model_input_data_type" in state["getter_errors"]:
                raise state["getter_errors"]["get_model_input_data_type"]
            return state["input_dtypes"]

        def get_model_output_shape(self):
            if "get_model_output_shape" in state["getter_errors"]:
                raise state["getter_errors"]["get_model_output_shape"]
            return state["output_shapes"]

        def dispose(self):
            self.dispose_calls += 1
            if state["dispose_error"] is not None:
                raise state["dispose_error"]

    for getter_name in missing_getters:
        delattr(Model, getter_name)

    module = types.ModuleType("qbruntime")
    module.Accelerator = Accelerator
    module.ModelConfig = ModelConfig
    module.Model = Model
    module.DataType = DataType
    module.Cluster = Cluster
    module.Core = Core
    module.CoreId = CoreId
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


class FailingValidationMbltml:
    MBLTML_DEVICE_ARIES = 1
    MBLTML_DEVICE_REGULUS = 2
    MBLTML_DEVICE_REGULUS_USB = 4

    def __init__(self):
        self.device_type = self.MBLTML_DEVICE_REGULUS
        self.shutdown_error = RuntimeError("shutdown failed")
        self.init_devices_calls = []
        self.shutdown_calls = 0

    def mbltmlInitDevices(self, device_types):
        self.init_devices_calls.append(set(device_types))

    def mbltmlGetDeviceCount(self):
        return 1

    def mbltmlGetDeviceType(self, device_id):
        assert device_id == 0
        return self.device_type

    def mbltmlShutdown(self):
        self.shutdown_calls += 1
        if self.shutdown_error is not None:
            raise RuntimeError(str(self.shutdown_error))


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


@pytest.mark.parametrize("profile", [RESNET_PROFILE, YOLO_PROFILE])
def test_load_accepts_matching_vision_sdk_contract_and_exposes_diagnostics(
    monkeypatch, tmp_path, profile
):
    state = _install_fake_qbruntime(monkeypatch)
    _set_matching_sdk_contract(state, profile)
    if profile is YOLO_PROFILE:
        state["output_shapes"].reverse()
    compiled_model, contract = _vision_compiled_model(tmp_path, profile)
    runtime = MobilintRuntime(expected_family="aries", **contract)

    runtime.load(compiled_model)

    spec = runtime.get_device_spec()
    assert spec["vision_profile_id"] == profile["vision_profile_id"]
    assert spec["expected_input_dtype"] == "uint8"
    assert spec["actual_input_dtype"] == "uint8"
    assert spec["expected_input_layout"] == "NHWC"
    assert spec["expected_unbatched_input_shape"] == tuple(
        profile["expected_unbatched_input_shape"]
    )
    assert spec["max_input_batch_size"] == 1
    assert spec["actual_input_shape"] == tuple(
        profile["expected_unbatched_input_shape"]
    )
    assert spec["expected_unbatched_output_shapes"] == tuple(
        profile.get("expected_unbatched_output_shapes", ())
    )
    if profile is YOLO_PROFILE:
        assert spec["actual_output_shapes"] == tuple(state["output_shapes"])
    else:
        assert spec["actual_output_shapes"] == ()


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"vision_profile_id": "profile"}, "all be provided together"),
        (
            _vision_contract(
                RESNET_PROFILE, expected_input_dtype="not-a-dtype"
            ),
            "expected_input_dtype",
        ),
        (
            _vision_contract(RESNET_PROFILE, expected_input_layout="HWCN"),
            "expected_input_layout",
        ),
        (
            _vision_contract(
                RESNET_PROFILE,
                expected_unbatched_input_shape=[224, True, 3],
            ),
            "expected_unbatched_input_shape",
        ),
        (
            _vision_contract(RESNET_PROFILE, max_input_batch_size=True),
            "max_input_batch_size",
        ),
        (
            _vision_contract(
                RESNET_PROFILE, expected_unbatched_output_shapes=[[]]
            ),
            "expected_unbatched_output_shapes",
        ),
    ],
)
def test_init_rejects_invalid_optional_vision_contract(options, message):
    with pytest.raises(ValueError, match=message):
        MobilintRuntime(expected_family="aries", **options)


@pytest.mark.parametrize(
    ("failure", "missing_getters"),
    [
        ("missing input shape getter", ("get_model_input_shape",)),
        ("missing input dtype getter", ("get_model_input_data_type",)),
        ("missing output shape getter", ("get_model_output_shape",)),
        ("input dtype mismatch", ()),
        ("input shape mismatch", ()),
        ("output count mismatch", ()),
        ("output shape mismatch", ()),
    ],
)
def test_load_contract_failure_rolls_back_model_and_session(
    monkeypatch, tmp_path, failure, missing_getters
):
    profile = (
        YOLO_PROFILE if "output" in failure else RESNET_PROFILE
    )
    state = _install_fake_qbruntime(
        monkeypatch, missing_getters=missing_getters
    )
    _set_matching_sdk_contract(state, profile)
    if failure == "input dtype mismatch":
        state["input_dtypes"] = DataType.Float32
    elif failure == "input shape mismatch":
        state["input_shapes"] = [(3, 224, 224)]
    elif failure == "output count mismatch":
        state["output_shapes"] = state["output_shapes"][:-1]
    elif failure == "output shape mismatch":
        state["output_shapes"][-1] = (81, 80, 255)
    compiled_model, contract = _vision_compiled_model(tmp_path, profile)
    runtime = MobilintRuntime(expected_family="aries", **contract)

    with pytest.raises((RuntimeError, ValueError)) as caught:
        runtime.load(compiled_model)

    message = str(caught.value)
    assert profile["vision_profile_id"] in message
    assert profile["artifact_basename"] in message
    assert "expected" in message
    assert "actual" in message
    assert state["models"][0].dispose_calls == 1
    assert FakeDeviceSession.instances[0].release_calls == 1
    assert runtime.compiled_model is None
    assert runtime._model is None


@pytest.mark.parametrize("wrapper_type", [list, tuple])
def test_load_rejects_multi_element_sdk_dtype_wrapper(
    monkeypatch, tmp_path, wrapper_type
):
    state = _install_fake_qbruntime(monkeypatch)
    _set_matching_sdk_contract(state, RESNET_PROFILE)
    state["input_dtypes"] = wrapper_type(
        (
            DataType.Uint8,
            DataType.Uint8,
        )
    )
    compiled_model, contract = _vision_compiled_model(tmp_path, RESNET_PROFILE)
    runtime = MobilintRuntime(expected_family="aries", **contract)

    with pytest.raises(RuntimeError, match="SDK input dtype count"):
        runtime.load(compiled_model)

    assert state["models"][0].dispose_calls == 1
    assert FakeDeviceSession.instances[0].release_calls == 1


def test_sdk_metadata_getter_failure_redacts_complete_traceback(
    monkeypatch, tmp_path
):
    state = _install_fake_qbruntime(monkeypatch)
    _set_matching_sdk_contract(state, RESNET_PROFILE)
    secret = "api-token=top-secret-value"
    vendor_error = RuntimeError(secret)
    state["getter_errors"]["get_model_input_shape"] = vendor_error
    compiled_model, contract = _vision_compiled_model(tmp_path, RESNET_PROFILE)
    runtime = MobilintRuntime(expected_family="aries", **contract)

    with pytest.raises(RuntimeError) as caught:
        runtime.load(compiled_model)

    assert "get_model_input_shape" in str(caught.value)
    assert "RuntimeError" in str(caught.value)
    assert secret not in str(caught.value)
    rendered = "".join(
        traceback.format_exception(
            type(caught.value),
            caught.value,
            caught.value.__traceback__,
        )
    )
    assert "get_model_input_shape" in rendered
    assert "RuntimeError" in rendered
    assert secret not in rendered
    assert caught.value.__cause__ is not vendor_error


def test_contract_free_nlp_load_does_not_require_metadata_getters(
    monkeypatch, tmp_path
):
    state = _install_fake_qbruntime(
        monkeypatch,
        missing_getters=(
            "get_model_input_shape",
            "get_model_input_data_type",
            "get_model_output_shape",
        ),
    )
    runtime = MobilintRuntime(expected_family="aries")

    runtime.load(_compiled_model(tmp_path))
    outputs = runtime.run(
        {
            "input_ids": np.ones((1, 4), dtype=np.int64),
            "attention_mask": np.ones((1, 4), dtype=np.int64),
        }
    )

    assert list(outputs) == ["logits", "hidden"]
    assert len(state["models"][0].infer_calls) == 1


def test_tensor_contract_validates_multi_input_sdk_metadata_and_runtime_arrays(
    monkeypatch, tmp_path
):
    state = _install_fake_qbruntime(monkeypatch)
    state["input_shapes"] = [(1, 1, 4), (4,)]
    state["input_dtypes"] = [DataType.Int64, DataType.Int64]
    state["output_shapes"] = [(1, 1, 2), (1, 4)]
    runtime = MobilintRuntime(
        expected_family="aries",
        artifact_profile_id="mobilint-test-bert-tensor-v1",
        expected_input_names=["input_ids", "attention_mask"],
        expected_input_dtypes=["int64", "int64"],
        expected_unbatched_input_shapes=[[4], [4]],
        expected_output_names=["logits", "hidden"],
        expected_unbatched_output_shapes=[[2], [4]],
        max_input_batch_size=1,
        native_async_supported=False,
    )

    runtime.load(_compiled_model(tmp_path))
    outputs = runtime.run(
        {
            "attention_mask": np.ones((1, 4), dtype=np.int64),
            "input_ids": np.ones((1, 4), dtype=np.int64),
        }
    )

    assert list(outputs) == ["logits", "hidden"]
    assert runtime.native_async_max_batch_size() is None
    diagnostics = runtime.get_device_spec()
    assert diagnostics["artifact_profile_id"] == "mobilint-test-bert-tensor-v1"
    assert diagnostics["expected_input_names"] == (
        "input_ids",
        "attention_mask",
    )
    assert diagnostics["actual_input_shapes"] == ((4,), (4,))
    assert diagnostics["actual_output_shapes"] == ((2,), (4,))


def test_tensor_contract_rejects_wrong_named_input_dtype_before_sdk_infer(
    monkeypatch, tmp_path
):
    state = _install_fake_qbruntime(monkeypatch)
    state["input_shapes"] = [(4,), (4,)]
    state["input_dtypes"] = [DataType.Int64, DataType.Int64]
    state["output_shapes"] = [(2,), (4,)]
    runtime = MobilintRuntime(
        expected_family="aries",
        artifact_profile_id="mobilint-test-bert-tensor-v1",
        expected_input_names=["input_ids", "attention_mask"],
        expected_input_dtypes=["int64", "int64"],
        expected_unbatched_input_shapes=[[4], [4]],
        expected_output_names=["logits", "hidden"],
        expected_unbatched_output_shapes=[[2], [4]],
        max_input_batch_size=1,
        native_async_supported=False,
    )
    runtime.load(_compiled_model(tmp_path))

    with pytest.raises(ValueError, match="attention_mask.*dtype mismatch"):
        runtime.run(
            {
                "input_ids": np.ones((1, 4), dtype=np.int64),
                "attention_mask": np.ones((1, 4), dtype=np.float32),
            }
        )

    assert state["models"][0].infer_calls == []


def test_tensor_contract_rejects_mismatched_input_batch_dimensions(
    monkeypatch, tmp_path
):
    state = _install_fake_qbruntime(monkeypatch)
    state["input_shapes"] = [(4,), (4,)]
    state["input_dtypes"] = [DataType.Int64, DataType.Int64]
    state["output_shapes"] = [(2,), (4,)]
    runtime = MobilintRuntime(
        expected_family="aries",
        artifact_profile_id="mobilint-test-bert-tensor-v1",
        expected_input_names=["input_ids", "attention_mask"],
        expected_input_dtypes=["int64", "int64"],
        expected_unbatched_input_shapes=[[4], [4]],
        expected_output_names=["logits", "hidden"],
        expected_unbatched_output_shapes=[[2], [4]],
        max_input_batch_size=4,
        native_async_supported=False,
    )
    runtime.load(_compiled_model(tmp_path))

    with pytest.raises(ValueError, match="input batch dimensions must match"):
        runtime.run(
            {
                "input_ids": np.ones((4, 4), dtype=np.int64),
                "attention_mask": np.ones((1, 4), dtype=np.int64),
            }
        )

    assert state["models"][0].infer_calls == []


def test_tensor_contract_rejects_unbatched_output_for_multi_sample_input(
    monkeypatch, tmp_path
):
    state = _install_fake_qbruntime(monkeypatch)
    state["input_shapes"] = [(4,), (4,)]
    state["input_dtypes"] = [DataType.Int64, DataType.Int64]
    state["output_shapes"] = [(2,), (4,)]
    runtime = MobilintRuntime(
        expected_family="aries",
        artifact_profile_id="mobilint-test-bert-tensor-v1",
        expected_input_names=["input_ids", "attention_mask"],
        expected_input_dtypes=["int64", "int64"],
        expected_unbatched_input_shapes=[[4], [4]],
        expected_output_names=["logits", "hidden"],
        expected_unbatched_output_shapes=[[2], [4]],
        max_input_batch_size=4,
        native_async_supported=False,
    )
    runtime.load(_compiled_model(tmp_path))
    state["models"][0].outputs = [
        np.zeros(2, dtype=np.float32),
        np.zeros(4, dtype=np.float32),
    ]

    with pytest.raises(RuntimeError, match="output shape mismatch"):
        runtime.run(
            {
                "input_ids": np.ones((4, 4), dtype=np.int64),
                "attention_mask": np.ones((4, 4), dtype=np.int64),
            }
        )

    assert len(state["models"][0].infer_calls) == 1


def test_tensor_contract_disables_sdk_native_async_even_when_pipeline_enabled(
    monkeypatch, tmp_path
):
    state = _install_fake_qbruntime(monkeypatch)
    state["input_shapes"] = [(4,), (4,)]
    state["input_dtypes"] = [DataType.Int64, DataType.Int64]
    state["output_shapes"] = [(2,), (4,)]
    runtime = MobilintRuntime(
        expected_family="aries",
        async_pipeline_enabled=True,
        artifact_profile_id="mobilint-test-bert-tensor-v1",
        expected_input_names=["input_ids", "attention_mask"],
        expected_input_dtypes=["int64", "int64"],
        expected_unbatched_input_shapes=[[4], [4]],
        expected_output_names=["logits", "hidden"],
        expected_unbatched_output_shapes=[[2], [4]],
        max_input_batch_size=1,
        native_async_supported=False,
    )
    runtime.load(_compiled_model(tmp_path))

    with pytest.raises(RuntimeError, match="does not support SDK native async"):
        runtime.create_native_backend()

    assert runtime.native_async_max_batch_size() is None


@pytest.mark.parametrize(
    "sdk_input_shape",
    [(1, -1, 768), (1, 9, 768)],
)
def test_dynamic_tensor_contract_accepts_sdk_shape_and_concrete_runtime_input(
    monkeypatch, tmp_path, sdk_input_shape
):
    state = _install_fake_qbruntime(monkeypatch)
    state["input_shapes"] = [sdk_input_shape]
    state["input_dtypes"] = DataType.Float32
    state["output_shapes"] = [(1, 2)]
    compiled_model = _dynamic_bert_compiled_model(tmp_path)
    contract = build_mobilint_tensor_contract(
        compiled_model.spec,
        max_batch_size=1,
        profile_id="mobilint-bert-sst2-embedding-v1",
    )
    runtime = MobilintRuntime(
        expected_family="aries", **contract.runtime_contract()
    )

    runtime.load(compiled_model)
    state["models"][0].outputs = [
        np.array([[0.25, 0.75]], dtype=np.float32)
    ]
    outputs = runtime.run(
        {"embeddings": np.zeros((1, 9, 768), dtype=np.float32)}
    )

    assert list(outputs) == ["logits"]
    assert outputs["logits"].shape == (1, 2)
    assert state["models"][0].infer_calls[0].shape == (1, 9, 768)


@pytest.mark.parametrize(
    ("shape", "message"),
    [
        ((1, 9, 767), "shape mismatch"),
        ((1, 768), "rank mismatch"),
        ((1, 0, 768), "shape mismatch"),
    ],
)
def test_dynamic_tensor_contract_rejects_invalid_concrete_runtime_shape(
    monkeypatch, tmp_path, shape, message
):
    state = _install_fake_qbruntime(monkeypatch)
    state["input_shapes"] = [(1, -1, 768)]
    state["input_dtypes"] = DataType.Float32
    state["output_shapes"] = [(1, 2)]
    compiled_model = _dynamic_bert_compiled_model(tmp_path)
    contract = build_mobilint_tensor_contract(
        compiled_model.spec,
        max_batch_size=1,
        profile_id="mobilint-bert-sst2-embedding-v1",
    )
    runtime = MobilintRuntime(
        expected_family="aries", **contract.runtime_contract()
    )
    runtime.load(compiled_model)

    with pytest.raises(ValueError, match=message):
        runtime.run({"embeddings": np.zeros(shape, dtype=np.float32)})

    assert state["models"][0].infer_calls == []


@pytest.mark.parametrize("dimension", [0, -2])
def test_tensor_runtime_contract_rejects_invalid_dynamic_declaration(
    dimension,
):
    with pytest.raises(ValueError, match="expected_unbatched_input_shapes"):
        MobilintRuntime(
            expected_family="aries",
            artifact_profile_id="dynamic-profile",
            expected_input_names=["embeddings"],
            expected_input_dtypes=["float32"],
            expected_unbatched_input_shapes=[[dimension, 768]],
            expected_output_names=["logits"],
            expected_unbatched_output_shapes=[[2]],
            max_input_batch_size=1,
            native_async_supported=False,
        )


def test_acquire_validation_and_shutdown_failure_retains_retry_owner(
    monkeypatch, tmp_path
):
    qbruntime_state = _install_fake_qbruntime(monkeypatch)
    device_sdk = FailingValidationMbltml()
    monkeypatch.setattr(
        mobilint_device,
        "_STATE",
        mobilint_device._MbltmlState(),
    )
    monkeypatch.setattr(
        mobilint_device,
        "import_module",
        lambda name: device_sdk,
    )
    monkeypatch.setattr(
        "runtimes.mobilint_rt.MobilintDeviceSession",
        mobilint_device.MobilintDeviceSession,
    )
    runtime = MobilintRuntime(expected_family="aries")
    compiled_model = _compiled_model(tmp_path)

    with pytest.raises(
        RuntimeError,
        match=(
            "load failed and rollback cleanup is incomplete.*shutdown failed"
            r".*call unload\(\) to retry cleanup"
        ),
    ) as caught:
        runtime.load(compiled_model)

    assert "shutdown failed" in str(caught.value.__cause__)
    assert "expected ARIES" in str(caught.value.__cause__.__context__)
    assert runtime._device_session is not None
    assert mobilint_device._STATE.cleanup_pending is True
    assert device_sdk.shutdown_calls == 2
    assert qbruntime_state["models"] == []
    with pytest.raises(RuntimeError, match="cleanup is incomplete"):
        runtime.load(compiled_model)
    with pytest.raises(RuntimeError, match="cleanup is incomplete"):
        runtime.run({})

    device_sdk.shutdown_error = None
    runtime.unload()
    assert device_sdk.shutdown_calls == 3
    assert mobilint_device._STATE.cleanup_pending is False

    device_sdk.device_type = device_sdk.MBLTML_DEVICE_ARIES
    runtime.load(compiled_model)
    runtime.unload()
    assert len(qbruntime_state["models"]) == 1
    assert qbruntime_state["models"][0].dispose_calls == 1
    assert device_sdk.init_devices_calls == [{1}, {1}]
    assert device_sdk.shutdown_calls == 4


@pytest.mark.parametrize(
    ("core_mode", "num_cores", "expected_call"),
    [
        (None, None, None),
        ("auto", None, ("auto",)),
        ("single", 3, ("single", 3, None)),
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


def test_single_core_mode_targets_cluster0_core0_with_qbruntime_v13(
    monkeypatch, tmp_path
):
    state = _install_fake_qbruntime(monkeypatch)
    runtime = MobilintRuntime(
        expected_family="aries",
        core_mode="single",
    )

    runtime.load(_compiled_model(tmp_path))

    call = state["configs"][0].calls[0]
    assert call[0] == "single"
    assert call[1] is None
    assert len(call[2]) == 1
    assert call[2][0].cluster is Cluster.Cluster0
    assert call[2][0].core is Core.Core0


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


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("dtype", "dtype mismatch"),
        ("layout", "shape mismatch"),
        ("batch", "batch mismatch"),
        ("name", "missing required inputs: input"),
    ],
)
def test_vision_input_contract_rejects_before_sdk_infer(
    monkeypatch, tmp_path, case, message
):
    state = _install_fake_qbruntime(monkeypatch)
    _set_matching_sdk_contract(state, RESNET_PROFILE)
    compiled_model, contract = _vision_compiled_model(tmp_path, RESNET_PROFILE)
    runtime = MobilintRuntime(expected_family="aries", **contract)
    runtime.load(compiled_model)
    array = np.ones((1, 224, 224, 3), dtype=np.uint8)
    name = "input"
    if case == "dtype":
        array = array.astype(np.float32)
    elif case == "layout":
        array = np.ones((1, 3, 224, 224), dtype=np.uint8)
    elif case == "batch":
        array = np.ones((2, 224, 224, 3), dtype=np.uint8)
    elif case == "name":
        name = "wrong"

    with pytest.raises(ValueError, match=message):
        runtime.run({name: array})

    assert state["models"][0].infer_calls == []


def test_ordered_inputs_rejects_unexpected_name_before_sdk_infer(
    monkeypatch, tmp_path
):
    state = _install_fake_qbruntime(monkeypatch)
    _set_matching_sdk_contract(state, RESNET_PROFILE)
    compiled_model, contract = _vision_compiled_model(tmp_path, RESNET_PROFILE)
    runtime = MobilintRuntime(expected_family="aries", **contract)
    runtime.load(compiled_model)
    state["models"][0].outputs = [
        np.zeros((1, 1000), dtype=np.float32)
    ]

    with pytest.raises(ValueError, match="unexpected inputs: typo"):
        runtime.run(
            {
                "input": np.zeros((1, 224, 224, 3), dtype=np.uint8),
                "typo": np.zeros((1,), dtype=np.uint8),
            }
        )

    assert state["models"][0].infer_calls == []


def test_sync_vision_accepts_batch_at_advertised_contract_max(
    monkeypatch, tmp_path
):
    profile = {
        **RESNET_PROFILE,
        "max_input_batch_size": 2,
        "expected_unbatched_output_shapes": ((1000,),),
    }
    state = _install_fake_qbruntime(monkeypatch)
    _set_matching_sdk_contract(state, profile)
    compiled_model, contract = _vision_compiled_model(tmp_path, profile)
    runtime = MobilintRuntime(expected_family="aries", **contract)
    runtime.load(compiled_model)
    state["models"][0].outputs = [
        np.zeros((2, 1000), dtype=np.float32)
    ]

    outputs = runtime.run(
        {"input": np.zeros((2, 224, 224, 3), dtype=np.uint8)}
    )

    assert state["models"][0].infer_calls[0].shape == (2, 224, 224, 3)
    assert outputs["output"].shape == (2, 1000)


@pytest.mark.parametrize("batch_size", [0, 3])
def test_sync_vision_rejects_batch_outside_advertised_contract(
    monkeypatch, tmp_path, batch_size
):
    profile = {**RESNET_PROFILE, "max_input_batch_size": 2}
    state = _install_fake_qbruntime(monkeypatch)
    _set_matching_sdk_contract(state, profile)
    compiled_model, contract = _vision_compiled_model(tmp_path, profile)
    runtime = MobilintRuntime(expected_family="aries", **contract)
    runtime.load(compiled_model)

    with pytest.raises(
        ValueError,
        match=r"expected 1 <= batch size <= 2",
    ):
        runtime.run(
            {
                "input": np.zeros(
                    (batch_size, 224, 224, 3),
                    dtype=np.uint8,
                )
            }
        )

    assert state["models"][0].infer_calls == []


def test_output_validation_accepts_reordered_heads_at_contract_max():
    runtime = MobilintRuntime(
        expected_family="aries",
        vision_profile_id="custom-max-two",
        expected_input_dtype="uint8",
        expected_input_layout="NHWC",
        expected_unbatched_input_shape=[224, 224, 3],
        max_input_batch_size=2,
        expected_unbatched_output_shapes=[[10], [20]],
    )
    runtime._output_names = ("large", "small")

    outputs = runtime._normalize_outputs(
        [
            np.zeros((2, 20), dtype=np.float32),
            np.zeros((2, 10), dtype=np.float32),
        ],
        expected_batch_size=2,
    )

    assert outputs["large"].shape == (2, 20)
    assert outputs["small"].shape == (2, 10)


@pytest.mark.parametrize(
    ("max_batch_size", "output_batch_size"),
    [(1, 2), (2, 0), (2, 3)],
)
def test_output_validation_rejects_batch_outside_advertised_contract(
    max_batch_size,
    output_batch_size,
):
    runtime = MobilintRuntime(
        expected_family="aries",
        vision_profile_id="custom-batch-bounds",
        expected_input_dtype="uint8",
        expected_input_layout="NHWC",
        expected_unbatched_input_shape=[224, 224, 3],
        max_input_batch_size=max_batch_size,
        expected_unbatched_output_shapes=[[1000]],
    )
    runtime._output_names = ("output",)

    with pytest.raises(RuntimeError, match="output shape mismatch"):
        runtime._normalize_outputs(
            [
                np.zeros(
                    (output_batch_size, 1000),
                    dtype=np.float32,
                )
            ],
            expected_batch_size=min(max_batch_size, 1),
        )


def test_output_validation_keeps_unbatched_sdk_outputs_allowed():
    runtime = MobilintRuntime(
        expected_family="aries",
        vision_profile_id="custom-max-two",
        expected_input_dtype="uint8",
        expected_input_layout="NHWC",
        expected_unbatched_input_shape=[224, 224, 3],
        max_input_batch_size=2,
        expected_unbatched_output_shapes=[[1000]],
    )
    runtime._output_names = ("output",)

    outputs = runtime._normalize_outputs(
        [np.zeros((1000,), dtype=np.float32)],
        expected_batch_size=2,
    )

    assert outputs["output"].shape == (1000,)


@pytest.mark.parametrize(
    "output_shapes",
    [
        ((2, 20), (1, 10)),
        ((1, 20), (2, 10)),
    ],
)
def test_output_validation_rejects_mixed_batches_for_batch_two_input(
    output_shapes,
):
    runtime = MobilintRuntime(
        expected_family="aries",
        vision_profile_id="custom-max-two",
        expected_input_dtype="uint8",
        expected_input_layout="NHWC",
        expected_unbatched_input_shape=[224, 224, 3],
        max_input_batch_size=2,
        expected_unbatched_output_shapes=[[10], [20]],
    )
    runtime._output_names = ("large", "small")

    with pytest.raises(RuntimeError, match="output shape mismatch"):
        runtime._normalize_outputs(
            [
                np.zeros(output_shapes[0], dtype=np.float32),
                np.zeros(output_shapes[1], dtype=np.float32),
            ],
            expected_batch_size=2,
        )


def test_vision_input_contract_makes_noncontiguous_array_contiguous(
    monkeypatch, tmp_path
):
    state = _install_fake_qbruntime(monkeypatch)
    _set_matching_sdk_contract(state, RESNET_PROFILE)
    compiled_model, contract = _vision_compiled_model(tmp_path, RESNET_PROFILE)
    runtime = MobilintRuntime(expected_family="aries", **contract)
    runtime.load(compiled_model)
    state["models"][0].outputs = [
        np.zeros((1, 1000), dtype=np.float32)
    ]
    array = np.zeros((1, 224, 224, 6), dtype=np.uint8)[..., ::2]
    assert array.shape == (1, 224, 224, 3)
    assert not array.flags.c_contiguous

    runtime.run({"input": array})

    submitted = state["models"][0].infer_calls[0]
    assert isinstance(submitted, np.ndarray)
    assert submitted.flags.c_contiguous
    np.testing.assert_array_equal(submitted, array)


def test_sync_normalized_outputs_reject_yolo_shape_contract(
    monkeypatch, tmp_path
):
    state = _install_fake_qbruntime(monkeypatch)
    _set_matching_sdk_contract(state, YOLO_PROFILE)
    compiled_model, contract = _vision_compiled_model(tmp_path, YOLO_PROFILE)
    runtime = MobilintRuntime(expected_family="aries", **contract)
    runtime.load(compiled_model)
    state["models"][0].outputs = [
        np.empty((1, 20, 20, 255), dtype=np.float32),
        np.empty((1, 40, 40, 255), dtype=np.float32),
        np.empty((1, 81, 80, 255), dtype=np.float32),
    ]

    with pytest.raises(RuntimeError, match="output shape mismatch"):
        runtime.run(
            {"images": np.zeros((1, 640, 640, 3), dtype=np.uint8)}
        )

    assert len(state["models"][0].infer_calls) == 1


def test_output_shape_multiset_rejects_mixed_unbatched_and_batched_heads():
    runtime = MobilintRuntime(
        expected_family="aries",
        vision_profile_id="mixed-output-representations",
        expected_input_dtype="uint8",
        expected_input_layout="NHWC",
        expected_unbatched_input_shape=[2, 2, 3],
        max_input_batch_size=1,
        expected_unbatched_output_shapes=[[10], [20]],
    )
    runtime._output_names = ("matrix", "vector")

    with pytest.raises(RuntimeError, match="output shape mismatch"):
        runtime._normalize_outputs(
            [np.empty((20,)), np.empty((1, 10))],
            expected_batch_size=1,
        )


def test_output_shape_multiset_accepts_all_unbatched_reordered_heads():
    runtime = MobilintRuntime(
        expected_family="aries",
        vision_profile_id="prefix-ambiguous-output-shapes",
        expected_input_dtype="uint8",
        expected_input_layout="NHWC",
        expected_unbatched_input_shape=[2, 2, 3],
        max_input_batch_size=1,
        expected_unbatched_output_shapes=[[1, 2], [2]],
    )
    runtime._output_names = ("vector", "matrix")

    outputs = runtime._normalize_outputs(
        [np.empty((2,)), np.empty((1, 2))],
        expected_batch_size=1,
    )

    assert tuple(outputs) == ("vector", "matrix")


def test_tensor_outputs_are_reshaped_to_logical_batched_shapes():
    runtime = MobilintRuntime(
        expected_family="aries",
        artifact_profile_id="mobilint-bert-sst2-embedding-v1",
        expected_input_names=["embeddings"],
        expected_input_dtypes=["float32"],
        expected_unbatched_input_shapes=[[-1, 768]],
        expected_output_names=["logits"],
        expected_unbatched_output_shapes=[[2]],
        max_input_batch_size=1,
        native_async_supported=False,
    )
    runtime._output_names = ("logits",)

    outputs = runtime._normalize_outputs(
        [np.zeros((1, 1, 1, 2), dtype=np.float32)],
        expected_batch_size=1,
    )

    assert outputs["logits"].shape == (1, 2)


def test_squad_outputs_bind_reversed_sdk_order_and_flatten():
    runtime = MobilintRuntime(
        expected_family="aries",
        artifact_profile_id="mobilint-bert-squad1-embedding-v1",
        expected_input_names=["embeddings"],
        expected_input_dtypes=["float32"],
        expected_unbatched_input_shapes=[[-1, 768]],
        expected_output_names=["end_logits", "start_logits"],
        expected_unbatched_output_shapes=[[-1], [-1]],
        max_input_batch_size=1,
        native_async_supported=False,
    )
    runtime._output_names = ("end_logits", "start_logits")

    outputs = runtime._normalize_outputs(
        [
            np.full((1, 1, 9, 1), 2.0, dtype=np.float32),
            np.full((1, 1, 9, 1), 1.0, dtype=np.float32),
        ],
        expected_batch_size=1,
    )

    assert tuple(outputs) == ("end_logits", "start_logits")
    assert outputs["end_logits"].shape == (1, 9)
    assert outputs["start_logits"].shape == (1, 9)
    assert np.all(outputs["end_logits"] == 2.0)
    assert np.all(outputs["start_logits"] == 1.0)


def test_squad_tensor_contract_accepts_singleton_heavy_sdk_metadata(
    monkeypatch, tmp_path
):
    state = _install_fake_qbruntime(monkeypatch)
    state["input_shapes"] = [(1, -1, 768)]
    state["input_dtypes"] = DataType.Float32
    state["output_shapes"] = [(1, -1, 1), (1, -1, 1)]
    artifact = tmp_path / "squad1.mxq"
    artifact.write_bytes(b"fake")
    spec = Model_Spec(
        name="bert-base-uncased-squad-v1",
        task=Task.QUESTION_ANSWERING,
        input_shapes={"embeddings": (1, -1, 768)},
        input_dtype={"embeddings": "float32"},
        output_shapes={
            "end_logits": (1, -1),
            "start_logits": (1, -1),
        },
    )
    contract = build_mobilint_tensor_contract(
        spec,
        max_batch_size=1,
        profile_id="mobilint-bert-squad1-embedding-v1",
    )
    runtime = MobilintRuntime(
        expected_family="aries", **contract.runtime_contract()
    )

    runtime.load(CompiledModel(spec, "mobilint", artifact))

    assert runtime.get_device_spec()["actual_output_shapes"] == (
        (-1,),
        (-1,),
    )


def test_tensor_output_normalization_rejects_incompatible_element_count():
    runtime = MobilintRuntime(
        expected_family="aries",
        artifact_profile_id="fixed-output",
        expected_input_names=["input"],
        expected_input_dtypes=["float32"],
        expected_unbatched_input_shapes=[[4]],
        expected_output_names=["output"],
        expected_unbatched_output_shapes=[[2]],
        max_input_batch_size=1,
        native_async_supported=False,
    )
    runtime._output_names = ("output",)

    with pytest.raises(RuntimeError, match="element count"):
        runtime._normalize_outputs(
            [np.zeros((1, 3), dtype=np.float32)],
            expected_batch_size=1,
        )


def test_tensor_output_normalization_rejects_multiple_wildcards():
    runtime = MobilintRuntime(
        expected_family="aries",
        artifact_profile_id="ambiguous-output",
        expected_input_names=["input"],
        expected_input_dtypes=["float32"],
        expected_unbatched_input_shapes=[[4]],
        expected_output_names=["output"],
        expected_unbatched_output_shapes=[[-1, -1]],
        max_input_batch_size=1,
        native_async_supported=False,
    )
    runtime._output_names = ("output",)

    with pytest.raises(RuntimeError, match="multiple dynamic dimensions"):
        runtime._normalize_outputs(
            [np.zeros((1, 2, 3), dtype=np.float32)],
            expected_batch_size=1,
        )


@pytest.mark.parametrize("expected_batch_size", [None, 0, 2, "1"])
def test_tensor_output_normalization_rejects_invalid_requested_batch(
    expected_batch_size,
):
    runtime = MobilintRuntime(
        expected_family="aries",
        artifact_profile_id="batch-one-output",
        expected_input_names=["input"],
        expected_input_dtypes=["float32"],
        expected_unbatched_input_shapes=[[4]],
        expected_output_names=["output"],
        expected_unbatched_output_shapes=[[2]],
        max_input_batch_size=1,
        native_async_supported=False,
    )
    runtime._output_names = ("output",)

    with pytest.raises(RuntimeError, match="requested batch size"):
        runtime._normalize_outputs(
            [np.zeros((1, 2), dtype=np.float32)],
            expected_batch_size=expected_batch_size,
        )


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


def test_native_backend_factory_requires_loaded_async_runtime(monkeypatch, tmp_path):
    state = _install_fake_qbruntime(monkeypatch)
    _set_matching_sdk_contract(state, RESNET_PROFILE)
    compiled_model, contract = _vision_compiled_model(tmp_path, RESNET_PROFILE)
    unloaded = MobilintRuntime(expected_family="aries", **contract)
    with pytest.raises(RuntimeError, match="not loaded"):
        unloaded.create_native_backend()

    synchronous = MobilintRuntime(expected_family="aries", **contract)
    synchronous.load(compiled_model)
    with pytest.raises(RuntimeError, match="async_pipeline_enabled=True"):
        synchronous.create_native_backend()
    synchronous.unload()

    runtime = MobilintRuntime(
        expected_family="aries",
        async_pipeline_enabled=True,
        activation_slots=3,
        **contract,
    )
    runtime.load(compiled_model)
    first = runtime.create_native_backend()

    assert runtime.native_async_max_batch_size() == 1
    assert runtime.create_native_backend() is first
    runtime.unload()
    assert len(state["models"]) == 2


def test_unload_skips_dispose_until_native_backend_quiesces(
    monkeypatch, tmp_path
):
    state = _install_fake_qbruntime(monkeypatch)
    runtime = MobilintRuntime(
        expected_family="aries",
        async_pipeline_enabled=True,
    )
    runtime.load(_compiled_model(tmp_path))

    class Backend:
        def __init__(self):
            self.results = [False, True]

        def shutdown(self, timeout):
            assert timeout == 5.0
            return self.results.pop(0)

    backend = Backend()
    runtime._native_backend = backend

    with pytest.raises(RuntimeError, match="did not quiesce"):
        runtime.unload()
    assert state["models"][0].dispose_calls == 0
    assert runtime._native_backend is backend
    assert runtime._cleanup_pending is True
    with pytest.raises(RuntimeError, match="cleanup is incomplete"):
        runtime.run({})

    runtime.unload()
    assert state["models"][0].dispose_calls == 1
    assert runtime._native_backend is None


def test_cleanup_retry_after_native_backend_quiesces_does_not_reshutdown_or_redispose(
    monkeypatch, tmp_path
):
    state = _install_fake_qbruntime(monkeypatch)
    runtime = MobilintRuntime(
        expected_family="aries",
        async_pipeline_enabled=True,
    )
    runtime.load(_compiled_model(tmp_path))

    class Backend:
        def __init__(self):
            self.shutdown_calls = 0

        def shutdown(self, timeout):
            self.shutdown_calls += 1
            return True

    backend = Backend()
    runtime._native_backend = backend
    FakeDeviceSession.release_error = RuntimeError("release failed")

    with pytest.raises(RuntimeError, match="release failed"):
        runtime.unload()

    assert backend.shutdown_calls == 1
    assert runtime._native_backend is None
    assert state["models"][0].dispose_calls == 1
    assert runtime._device_session is FakeDeviceSession.instances[0]

    FakeDeviceSession.release_error = None
    runtime.unload()
    assert backend.shutdown_calls == 1
    assert state["models"][0].dispose_calls == 1
    assert FakeDeviceSession.instances[0].release_calls == 2


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
