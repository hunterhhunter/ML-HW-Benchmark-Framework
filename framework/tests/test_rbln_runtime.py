import json
import subprocess
import sys
import threading
import weakref
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from runtimes.rbln_rt import RblnRuntime
from rbln_test_utils import (
    FakeRBLNCompiledModel,
    FakeRebel,
    FakeTensor,
    compiled_model as _compiled_model,
    fake_rebel,
    load_with_fake as _load_with_fake,
    loaded_runtime,
    valid_inputs,
)


@pytest.mark.parametrize("backend", ["rbln", "rebel", "rbln-static"])
def test_compatible_accepts_rbln_artifact_and_backend_aliases(
    tmp_path, backend
):
    runtime = RblnRuntime()

    assert runtime.is_compatible(
        _compiled_model(tmp_path / "model.rbln", backend=backend)
    )


@pytest.mark.parametrize(
    ("suffix", "backend"),
    [(".onnx", "rbln"), (".rbln", "onnx"), (".rbln", "rbln-vllm")],
)
def test_compatible_rejects_wrong_suffix_or_backend(tmp_path, suffix, backend):
    runtime = RblnRuntime()

    assert not runtime.is_compatible(
        _compiled_model(tmp_path / f"model{suffix}", backend=backend)
    )


@pytest.mark.parametrize(
    ("option", "value", "message"),
    [
        ("device_id", True, "device_id"),
        ("device_id", 0.0, "device_id"),
        ("device_id", 1, "device_id"),
        ("async_parallel", False, "async_parallel"),
        ("async_parallel", 3, "async_parallel"),
        ("max_async_inflight", 0, "max_async_inflight"),
        ("max_async_inflight", 1.0, "max_async_inflight"),
        ("runtime_timeout_sec", True, "runtime_timeout_sec"),
        ("runtime_timeout_sec", float("nan"), "runtime_timeout_sec"),
        ("runtime_timeout_sec", float("inf"), "runtime_timeout_sec"),
        ("runtime_timeout_sec", 0, "runtime_timeout_sec"),
        ("shutdown_timeout_sec", -1, "shutdown_timeout_sec"),
    ],
)
def test_compatible_constructor_options_reject_unsafe_numeric_values(
    option, value, message
):
    with pytest.raises(ValueError, match=message):
        RblnRuntime(**{option: value})


def test_compatible_constructor_options_reject_conversion_objects():
    class ConvertsToInt:
        def __int__(self):
            return 0

    class ConvertsToFloat:
        def __float__(self):
            return 1.0

    with pytest.raises(ValueError, match="device_id"):
        RblnRuntime(device_id=ConvertsToInt())
    with pytest.raises(ValueError, match="runtime_timeout_sec"):
        RblnRuntime(runtime_timeout_sec=ConvertsToFloat())
    with pytest.raises(ValueError, match="runtime_timeout_sec"):
        RblnRuntime(runtime_timeout_sec=10**10000)


def test_module_import_and_construction_do_not_load_rebel():
    code = """
import builtins
import sys
sys.path.insert(0, 'src')
real_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name == 'rebel' or name.startswith('rebel.'):
        raise AssertionError('rebel imported eagerly')
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
from runtimes.rbln_rt import RblnRuntime
RblnRuntime()
"""

    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parent.parent,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_load_inspects_contract_without_allocating_runtime(
    tmp_path, monkeypatch, fake_rebel
):
    artifact = tmp_path / "bert.rbln"
    compiled_model = _compiled_model(artifact, backend="rbln")

    runtime = _load_with_fake(
        monkeypatch,
        fake_rebel,
        compiled_model,
        device="0",
        async_parallel=2,
        max_async_inflight=4,
    )

    assert fake_rebel.availability_calls == [0]
    assert fake_rebel.name_calls == [0]
    assert fake_rebel.inspect_calls == [str(artifact)]
    assert fake_rebel.runtime_calls == []
    assert fake_rebel.async_runtime_calls == []
    device_spec = runtime.get_device_spec()
    assert device_spec == {
        "backend": "rbln",
        "device": "0",
        "device_id": 0,
        "accelerator_vendor": "Rebellions",
        "accelerator_name": "RBLN-CA22",
        "execution_mode": "loaded",
        "detected_npu": "RBLN-CA22",
        "sdk_version": "0.11.0",
        "artifact_compiler_version": "0.11.0",
        "artifact_npu": "RBLN-CA22",
        "tensor_parallel_size": 1,
        "artifact_uuid": "artifact-uuid",
        "artifact_alloc_per_node": [4096],
        "input_names": ["input_ids", "attention_mask"],
        "input_shapes": [[1, 8], [1, 8]],
        "input_dtypes": ["int64", "int64"],
        "output_names": ["logits"],
        "output_shapes": [[1, 2]],
        "output_dtypes": ["float32"],
        "async_parallel": 2,
        "max_async_inflight": 4,
    }
    assert json.loads(json.dumps(device_spec)) == device_spec


def test_load_rechecks_artifact_file_before_import(
    tmp_path, monkeypatch, fake_rebel
):
    compiled_model = _compiled_model(tmp_path / "removed.rbln")
    compiled_model.artifact_path.unlink()
    imported = []
    monkeypatch.setattr(
        "runtimes.rbln_rt.import_module", lambda name: imported.append(name)
    )

    with pytest.raises(FileNotFoundError, match="RBLN artifact"):
        RblnRuntime().load(compiled_model)

    assert imported == []


def test_load_reports_missing_optional_sdk(tmp_path, monkeypatch):
    compiled_model = _compiled_model(tmp_path / "model.rbln")

    def missing_sdk(name):
        raise ModuleNotFoundError("No module named 'rebel'")

    monkeypatch.setattr("runtimes.rbln_rt.import_module", missing_sdk)

    with pytest.raises(ImportError, match="rebel-compiler"):
        RblnRuntime().load(compiled_model)


def test_load_rejects_unavailable_device_before_inspection(
    tmp_path, monkeypatch, fake_rebel
):
    fake_rebel.available = False

    with pytest.raises(RuntimeError, match="device 0 is not available"):
        _load_with_fake(
            monkeypatch, fake_rebel, _compiled_model(tmp_path / "model.rbln")
        )

    assert fake_rebel.name_calls == []
    assert fake_rebel.inspect_calls == []
    assert fake_rebel.runtime_calls == []
    assert fake_rebel.async_runtime_calls == []


def test_load_rejects_actual_device_name_mismatch_before_inspection(
    tmp_path, monkeypatch, fake_rebel
):
    fake_rebel.detected_npu = "RBLN-CA25"

    with pytest.raises(RuntimeError, match="requires detected NPU RBLN-CA22"):
        _load_with_fake(
            monkeypatch, fake_rebel, _compiled_model(tmp_path / "model.rbln")
        )

    assert fake_rebel.inspect_calls == []
    assert fake_rebel.runtime_calls == []


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("npu", "RBLN-CA25", "artifact target NPU"),
        ("tensor_parallel_size", 2, "tensor_parallel_size"),
        ("inputs", None, "input descriptors"),
        ("outputs", None, "output descriptors"),
        ("outputs", (), "at least one output"),
    ],
)
def test_load_rejects_incompatible_artifact_contract_before_allocation(
    tmp_path, monkeypatch, fake_rebel, field, value, message
):
    setattr(fake_rebel.inspected, field, value)

    with pytest.raises(ValueError, match=message):
        _load_with_fake(
            monkeypatch, fake_rebel, _compiled_model(tmp_path / "model.rbln")
        )

    assert fake_rebel.runtime_calls == []
    assert fake_rebel.async_runtime_calls == []


@pytest.mark.parametrize("dimension", [None, -1, 0, True, 1.5, "8"])
def test_load_rejects_dynamic_or_invalid_dimensions_before_allocation(
    tmp_path, monkeypatch, fake_rebel, dimension
):
    fake_rebel.inspected.inputs = (
        FakeTensor("input_ids", (1, dimension), "int64"),
        FakeTensor("attention_mask", (1, 8), "int64"),
    )

    with pytest.raises(ValueError, match="static positive integer dimensions"):
        _load_with_fake(
            monkeypatch, fake_rebel, _compiled_model(tmp_path / "model.rbln")
        )

    assert fake_rebel.runtime_calls == []


def test_load_bounds_hostile_shape_iterator_failure(
    tmp_path, monkeypatch, fake_rebel
):
    class HostileShape:
        def __iter__(self):
            raise RuntimeError("sensitive-shape-detail-" * 100)

    fake_rebel.inspected.inputs = (
        FakeTensor("input_ids", HostileShape(), "int64"),
        FakeTensor("attention_mask", (1, 8), "int64"),
    )

    with pytest.raises(
        ValueError, match="static positive integer dimensions"
    ) as caught:
        _load_with_fake(
            monkeypatch, fake_rebel, _compiled_model(tmp_path / "model.rbln")
        )

    assert "sensitive-shape-detail" not in str(caught.value)
    assert len(str(caught.value)) <= 256
    assert fake_rebel.runtime_calls == []


def test_load_bounds_hostile_descriptor_iterator_failure(
    tmp_path, monkeypatch, fake_rebel
):
    class HostileDescriptors:
        def __iter__(self):
            raise RuntimeError("sensitive-descriptor-detail-" * 100)

    fake_rebel.inspected.inputs = HostileDescriptors()

    with pytest.raises(ValueError, match="descriptors must be a sequence") as caught:
        _load_with_fake(
            monkeypatch, fake_rebel, _compiled_model(tmp_path / "model.rbln")
        )

    assert "sensitive-descriptor-detail" not in str(caught.value)
    assert len(str(caught.value)) <= 256
    assert fake_rebel.runtime_calls == []


@pytest.mark.parametrize(
    ("descriptors", "message"),
    [
        (
            (
                FakeTensor("input_ids", (1, 8), "int64"),
                FakeTensor("input_ids", (1, 8), "int64"),
            ),
            "duplicate input descriptor name",
        ),
        (
            (
                FakeTensor("", (1, 8), "int64"),
                FakeTensor("attention_mask", (1, 8), "int64"),
            ),
            "missing input descriptor name",
        ),
    ],
)
def test_load_rejects_missing_or_duplicate_input_names(
    tmp_path, monkeypatch, fake_rebel, descriptors, message
):
    fake_rebel.inspected.inputs = descriptors

    with pytest.raises(ValueError, match=message):
        _load_with_fake(
            monkeypatch, fake_rebel, _compiled_model(tmp_path / "model.rbln")
        )

    assert fake_rebel.runtime_calls == []


@pytest.mark.parametrize(
    ("descriptors", "message"),
    [
        (
            (
                FakeTensor("logits", (1, 2), "float32"),
                FakeTensor("logits", (1, 2), "float32"),
            ),
            "duplicate output descriptor name",
        ),
        (
            (
                FakeTensor(None, (1, 2), "float32"),
                FakeTensor("hidden", (1, 4), "float32"),
            ),
            "missing output descriptor name",
        ),
    ],
)
def test_load_rejects_missing_or_duplicate_output_names(
    tmp_path, monkeypatch, fake_rebel, descriptors, message
):
    fake_rebel.inspected.outputs = descriptors

    with pytest.raises(ValueError, match=message):
        _load_with_fake(
            monkeypatch, fake_rebel, _compiled_model(tmp_path / "model.rbln")
        )

    assert fake_rebel.runtime_calls == []


def test_load_normalizes_mapping_and_attribute_metadata_to_same_contract(
    tmp_path, monkeypatch, fake_rebel
):
    attribute_runtime = _load_with_fake(
        monkeypatch,
        fake_rebel,
        _compiled_model(tmp_path / "attribute.rbln"),
    )
    expected = attribute_runtime.get_device_spec()
    attribute_runtime.unload()
    fake_rebel.inspected = {
        "compiler_version": "0.11.0",
        "npu": "RBLN-CA22",
        "tensor_parallel_size": 1,
        "uuid": "artifact-uuid",
        "alloc_per_node": [4096],
        "inputs": [
            {"name": "input_ids", "shape": [1, 8], "dtype": "int64"},
            {
                "name": "attention_mask",
                "shape": [1, 8],
                "dtype": "int64",
            },
        ],
        "outputs": [
            {"name": "logits", "shape": [1, 2], "dtype": "float32"}
        ],
    }

    mapping_runtime = _load_with_fake(
        monkeypatch,
        fake_rebel,
        _compiled_model(tmp_path / "mapping.rbln"),
    )

    assert mapping_runtime.get_device_spec() == expected


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "inputs",
            (
                FakeTensor("input_ids", (1, 4), "int64"),
                FakeTensor("attention_mask", (1, 8), "int64"),
            ),
            "input shape",
        ),
        (
            "inputs",
            (
                FakeTensor("input_ids", (1, 8), "int32"),
                FakeTensor("attention_mask", (1, 8), "int64"),
            ),
            "input dtype",
        ),
        (
            "outputs",
            (FakeTensor("logits", (1, 4), "float32"),),
            "output shape",
        ),
        (
            "outputs",
            (FakeTensor("scores", (1, 2), "float32"),),
            "output descriptor names",
        ),
    ],
)
def test_load_requires_artifact_descriptors_to_match_model_spec(
    tmp_path, monkeypatch, fake_rebel, field, value, message
):
    setattr(fake_rebel.inspected, field, value)

    with pytest.raises(ValueError, match=message):
        _load_with_fake(
            monkeypatch, fake_rebel, _compiled_model(tmp_path / "model.rbln")
        )

    assert fake_rebel.runtime_calls == []


def test_load_allows_single_input_positional_name_fallback(
    tmp_path, monkeypatch, fake_rebel
):
    fake_rebel.inspected.inputs = (
        FakeTensor("artifact_input", (1, 8), "float32"),
    )
    compiled_model = _compiled_model(
        tmp_path / "single.rbln",
        input_shapes={"profile_input": (1, 8)},
        input_dtypes={"profile_input": "float32"},
    )

    runtime = _load_with_fake(
        monkeypatch, fake_rebel, compiled_model
    )

    assert runtime.get_device_spec()["input_names"] == ["artifact_input"]


def test_load_allows_single_unnamed_output_positional_name_fallback(
    tmp_path, monkeypatch, fake_rebel
):
    fake_rebel.inspected.outputs = (
        FakeTensor(None, (1, 2), "float32"),
    )
    compiled_model = _compiled_model(
        tmp_path / "single-output.rbln",
        output_shapes={"output": (1, 2)},
    )

    runtime = _load_with_fake(monkeypatch, fake_rebel, compiled_model)
    outputs = runtime.run(valid_inputs())

    assert runtime.get_device_spec()["output_names"] == ["output"]
    assert list(outputs) == ["output"]


def test_load_omits_absent_tensor_parallel_size(
    tmp_path, monkeypatch, fake_rebel
):
    fake_rebel.inspected.tensor_parallel_size = None

    runtime = _load_with_fake(
        monkeypatch,
        fake_rebel,
        _compiled_model(tmp_path / "single-device.rbln"),
    )

    assert "tensor_parallel_size" not in runtime.get_device_spec()


def test_load_rejects_multi_input_name_mismatch_instead_of_guessing(
    tmp_path, monkeypatch, fake_rebel
):
    fake_rebel.inspected.inputs = (
        FakeTensor("first", (1, 8), "int64"),
        FakeTensor("second", (1, 8), "int64"),
    )

    with pytest.raises(ValueError, match="input descriptor names"):
        _load_with_fake(
            monkeypatch, fake_rebel, _compiled_model(tmp_path / "model.rbln")
        )

    assert fake_rebel.runtime_calls == []


def test_load_omits_absent_optional_provenance_instead_of_fabricating_it(
    tmp_path, monkeypatch, fake_rebel
):
    fake_rebel.inspected = {
        "npu": "RBLN-CA22",
        "tensor_parallel_size": 1,
        "inputs": [
            {"name": "input_ids", "shape": [1, 8], "dtype": "int64"},
            {
                "name": "attention_mask",
                "shape": [1, 8],
                "dtype": "int64",
            },
        ],
        "outputs": [
            {"name": "logits", "shape": [1, 2], "dtype": "float32"}
        ],
    }

    runtime = _load_with_fake(
        monkeypatch, fake_rebel, _compiled_model(tmp_path / "model.rbln")
    )

    device_spec = runtime.get_device_spec()
    assert "artifact_compiler_version" not in device_spec
    assert "artifact_uuid" not in device_spec
    assert "artifact_alloc_per_node" not in device_spec


def test_load_omits_nonprimitive_module_version_fallback(
    tmp_path, monkeypatch, fake_rebel
):
    fake_rebel.__version__ = object()

    runtime = _load_with_fake(
        monkeypatch, fake_rebel, _compiled_model(tmp_path / "model.rbln")
    )

    assert "sdk_version" not in runtime.get_device_spec()


def test_load_omits_sdk_version_when_module_version_access_raises(
    tmp_path, monkeypatch, fake_rebel
):
    class HostileVersionRebel(FakeRebel):
        @property
        def __version__(self):
            raise RuntimeError("sensitive-version-detail-" * 100)

    hostile_rebel = HostileVersionRebel()
    FakeRBLNCompiledModel.owner = hostile_rebel

    runtime = _load_with_fake(
        monkeypatch,
        hostile_rebel,
        _compiled_model(tmp_path / "model.rbln"),
    )

    assert "sdk_version" not in runtime.get_device_spec()


def test_load_failure_is_atomic_and_retryable(
    tmp_path, monkeypatch, fake_rebel
):
    compiled_model = _compiled_model(tmp_path / "model.rbln")
    monkeypatch.setattr(
        "runtimes.rbln_rt.import_module", lambda name: fake_rebel
    )
    runtime = RblnRuntime()
    fake_rebel.inspect_error = RuntimeError("hostile inspect details")

    with pytest.raises(RuntimeError, match="(?i)could not inspect RBLN artifact"):
        runtime.load(compiled_model)

    assert fake_rebel.runtime_calls == []
    assert runtime.compiled_model is None
    fake_rebel.inspect_error = None
    runtime.load(compiled_model)
    assert runtime.compiled_model is compiled_model


def test_load_rejects_second_load_until_unloaded(
    tmp_path, monkeypatch, fake_rebel
):
    compiled_model = _compiled_model(tmp_path / "model.rbln")
    runtime = _load_with_fake(monkeypatch, fake_rebel, compiled_model)

    with pytest.raises(RuntimeError, match="already loaded"):
        runtime.load(compiled_model)

    assert fake_rebel.inspect_calls == [str(compiled_model.artifact_path)]


def test_run_creates_one_sync_runtime_with_static_sdk_options(
    loaded_runtime, fake_rebel
):
    inputs = valid_inputs()

    first_outputs = loaded_runtime.run(inputs)
    second_outputs = loaded_runtime.run(inputs)

    assert fake_rebel.runtime_calls == [
        (
            str(loaded_runtime.compiled_model.artifact_path),
            {"device": 0, "tensor_type": "np", "timeout": 17.5},
        )
    ]
    assert fake_rebel.async_runtime_calls == []
    assert len(fake_rebel.sync_instances[0].calls) == 2
    assert list(first_outputs) == ["logits"]
    assert list(second_outputs) == ["logits"]


def test_sync_run_orders_inputs_and_normalizes_named_outputs(
    loaded_runtime, fake_rebel
):
    inputs = valid_inputs()

    outputs = loaded_runtime.run(inputs)

    assert list(outputs) == ["logits"]
    assert fake_rebel.sync_instances[0].calls[0][0] is inputs["input_ids"]
    assert (
        fake_rebel.sync_instances[0].calls[0][1]
        is inputs["attention_mask"]
    )


def test_warmup_reuses_sync_runtime_and_inspected_input_order(
    loaded_runtime, fake_rebel
):
    inputs = valid_inputs()

    loaded_runtime.warmup(inputs, num_runs=3)

    assert len(fake_rebel.runtime_calls) == 1
    assert len(fake_rebel.sync_instances[0].calls) == 3
    for call in fake_rebel.sync_instances[0].calls:
        assert call[0] is inputs["input_ids"]
        assert call[1] is inputs["attention_mask"]


@pytest.mark.parametrize(
    ("inputs", "message"),
    [
        (
            {"input_ids": np.ones((1, 8), dtype=np.int64)},
            "missing required inputs",
        ),
        (
            {
                **valid_inputs(),
                "unexpected": np.ones((1, 8), dtype=np.int64),
            },
            "unexpected inputs",
        ),
        (
            {
                **valid_inputs(),
                "input_ids": np.ones((1, 8), dtype=np.int32),
            },
            "input dtype",
        ),
        (
            {
                **valid_inputs(),
                "input_ids": np.ones((1, 4), dtype=np.int64),
            },
            "input shape",
        ),
        (
            {
                **valid_inputs(),
                "input_ids": np.array(1, dtype=np.int64),
            },
            "scalar input",
        ),
        (
            {
                **valid_inputs(),
                "input_ids": np.ones((2, 8), dtype=np.int64),
            },
            "batch dimension N=1",
        ),
        (
            {
                **valid_inputs(),
                "input_ids": [[1] * 8],
            },
            "NumPy array",
        ),
    ],
)
def test_run_rejects_invalid_inputs_before_sdk_allocation(
    loaded_runtime, fake_rebel, inputs, message
):
    with pytest.raises((TypeError, ValueError), match=message):
        loaded_runtime.run(inputs)

    assert fake_rebel.runtime_calls == []


def test_run_requires_an_input_dictionary_before_sdk_allocation(
    loaded_runtime, fake_rebel
):
    with pytest.raises(TypeError, match="inputs must be a dictionary"):
        loaded_runtime.run([np.ones((1, 8), dtype=np.int64)])

    assert fake_rebel.runtime_calls == []


def test_run_copies_noncontiguous_inputs_without_casting(
    loaded_runtime, fake_rebel
):
    source = np.arange(16, dtype=np.int64).reshape(1, 16)
    input_ids = source[:, ::2]
    attention_mask = np.ones((1, 16), dtype=np.int64)[:, ::2]
    assert not input_ids.flags.c_contiguous
    assert not attention_mask.flags.c_contiguous

    loaded_runtime.run(
        {"input_ids": input_ids, "attention_mask": attention_mask}
    )

    call = fake_rebel.sync_instances[0].calls[0]
    assert call[0].flags.c_contiguous
    assert call[1].flags.c_contiguous
    assert call[0].dtype == input_ids.dtype
    np.testing.assert_array_equal(call[0], input_ids)


def test_run_uses_profile_name_for_single_input_fallback(
    tmp_path, monkeypatch, fake_rebel
):
    fake_rebel.inspected.inputs = (
        FakeTensor("artifact_input", (1, 8), "float32"),
    )
    compiled_model = _compiled_model(
        tmp_path / "single.rbln",
        input_shapes={"profile_input": (1, 8)},
        input_dtypes={"profile_input": "float32"},
    )
    runtime = _load_with_fake(monkeypatch, fake_rebel, compiled_model)
    value = np.ones((1, 8), dtype=np.float32)

    runtime.run({"profile_input": value})

    assert fake_rebel.sync_instances[0].calls[0] == (value,)


@pytest.mark.parametrize(
    "raw_outputs",
    [
        np.array([[0.25, 0.75]], dtype=np.float32),
        [np.array([[0.25, 0.75]], dtype=np.float32)],
        (np.array([[0.25, 0.75]], dtype=np.float32),),
    ],
)
def test_run_maps_bare_or_wrapped_single_output_to_inspected_name(
    loaded_runtime, fake_rebel, raw_outputs
):
    loaded_runtime.run(valid_inputs())
    fake_rebel.sync_instances[0].outputs = raw_outputs

    outputs = loaded_runtime.run(valid_inputs())

    assert list(outputs) == ["logits"]
    expected = raw_outputs if isinstance(raw_outputs, np.ndarray) else raw_outputs[0]
    assert outputs["logits"] is expected


def test_run_maps_multiple_outputs_in_inspected_order(
    tmp_path, monkeypatch, fake_rebel
):
    fake_rebel.inspected.outputs = (
        FakeTensor("logits", (1, 2), "float32"),
        FakeTensor("hidden", (1, 4), "float32"),
    )
    compiled_model = _compiled_model(
        tmp_path / "multiple.rbln",
        output_shapes={"hidden": (1, 4), "logits": (1, 2)},
    )
    runtime = _load_with_fake(monkeypatch, fake_rebel, compiled_model)
    logits = np.array([[0.25, 0.75]], dtype=np.float32)
    hidden = np.arange(4, dtype=np.float32).reshape(1, 4)
    fake_rebel.runtime_outputs = [logits, hidden]

    outputs = runtime.run(valid_inputs())

    assert list(outputs) == ["logits", "hidden"]
    assert outputs["logits"] is logits
    assert outputs["hidden"] is hidden


def test_run_accepts_exact_named_output_mapping_defensively(
    loaded_runtime, fake_rebel
):
    loaded_runtime.run(valid_inputs())
    logits = np.array([[0.25, 0.75]], dtype=np.float32)
    fake_rebel.sync_instances[0].outputs = {"logits": logits}

    outputs = loaded_runtime.run(valid_inputs())

    assert outputs == {"logits": logits}


@pytest.mark.parametrize(
    ("raw_outputs", "message"),
    [
        ([], "output count"),
        (
            [
                np.ones((1, 2), dtype=np.float32),
                np.ones((1, 2), dtype=np.float32),
            ],
            "output count",
        ),
        (np.ones((1, 2), dtype=np.float64), "output dtype"),
        (np.ones((1, 3), dtype=np.float32), "output shape"),
        ([[0.25, 0.75]], "NumPy arrays"),
        ({"scores": np.ones((1, 2), dtype=np.float32)}, "output names"),
    ],
)
def test_run_rejects_invalid_sdk_outputs_with_bounded_error(
    loaded_runtime, fake_rebel, raw_outputs, message
):
    loaded_runtime.run(valid_inputs())
    fake_rebel.sync_instances[0].outputs = raw_outputs

    with pytest.raises(RuntimeError, match=message) as caught:
        loaded_runtime.run(valid_inputs())

    assert len(str(caught.value)) <= 256


def test_run_before_load_is_rejected():
    with pytest.raises(RuntimeError, match="not loaded"):
        RblnRuntime().run(valid_inputs())


def test_run_runtime_constructor_failure_leaves_loaded_runtime_retryable(
    loaded_runtime, fake_rebel
):
    constructor_error = RuntimeError("constructor details")
    fake_rebel.runtime_error = constructor_error

    with pytest.raises(RuntimeError, match="could not create sync runtime") as caught:
        loaded_runtime.run(valid_inputs())

    assert caught.value.__cause__ is constructor_error
    assert loaded_runtime.compiled_model is not None
    assert loaded_runtime._sync_runtime is None
    fake_rebel.runtime_error = None
    assert list(loaded_runtime.run(valid_inputs())) == ["logits"]
    assert len(fake_rebel.runtime_calls) == 2


def test_run_sdk_failure_is_bounded_and_preserves_sync_runtime(
    loaded_runtime, fake_rebel
):
    sdk_error = RuntimeError("unbounded vendor detail")
    fake_rebel.runtime_call_error = sdk_error

    with pytest.raises(RuntimeError, match="synchronous inference failed") as caught:
        loaded_runtime.run(valid_inputs())

    assert caught.value.__cause__ is sdk_error
    assert "unbounded vendor detail" not in str(caught.value)
    assert loaded_runtime._sync_runtime is fake_rebel.sync_instances[0]


def test_batch_capability_is_exact_builtin_one():
    value = RblnRuntime().native_async_max_batch_size()

    assert type(value) is int
    assert value == 1


def test_run_sync_mode_reports_one_concurrent_worker(
    loaded_runtime, fake_rebel
):
    assert loaded_runtime.max_concurrent_workers() == 1

    loaded_runtime.run(valid_inputs())

    assert loaded_runtime.max_concurrent_workers() == 1


def test_execution_mode_transitions_from_loaded_to_sync(
    loaded_runtime, fake_rebel
):
    assert loaded_runtime.get_device_spec()["execution_mode"] == "loaded"

    loaded_runtime.run(valid_inputs())

    assert loaded_runtime.get_device_spec()["execution_mode"] == "sync"


def test_run_and_warmup_reject_native_async_ownership(
    loaded_runtime, fake_rebel
):
    loaded_runtime._native_backend = object()
    loaded_runtime._execution_mode = "native_async"

    with pytest.raises(RuntimeError, match="native async mode"):
        loaded_runtime.run(valid_inputs())
    with pytest.raises(RuntimeError, match="native async mode"):
        loaded_runtime.warmup(valid_inputs())

    assert fake_rebel.runtime_calls == []
    assert loaded_runtime.max_concurrent_workers() == 4


def test_unload_clears_runtime_is_idempotent_and_allows_reload(
    loaded_runtime, fake_rebel
):
    compiled_model = loaded_runtime.compiled_model
    loaded_runtime.run(valid_inputs())

    loaded_runtime.unload()
    loaded_runtime.unload()

    assert loaded_runtime.compiled_model is None
    assert loaded_runtime._sync_runtime is None
    with pytest.raises(RuntimeError, match="not loaded"):
        loaded_runtime.run(valid_inputs())

    loaded_runtime.load(compiled_model)
    loaded_runtime.run(valid_inputs())
    assert len(fake_rebel.runtime_calls) == 2
    assert len(fake_rebel.inspect_calls) == 2


def test_unload_releases_sync_runtime_on_calling_thread(
    loaded_runtime, fake_rebel
):
    loaded_runtime.run(valid_inputs())
    instance_reference = weakref.ref(fake_rebel.sync_instances[0])
    fake_rebel.sync_instances.clear()

    loaded_runtime.unload()

    assert instance_reference() is None
    assert fake_rebel.destruction_threads == [threading.get_ident()]


def test_unload_preserves_loaded_state_until_async_shutdown_is_proven(
    loaded_runtime, fake_rebel
):
    class Backend:
        def __init__(self):
            self.results = [False, True]
            self.calls = []

        def shutdown(self, timeout):
            self.calls.append(timeout)
            return self.results.pop(0)

    backend = Backend()
    loaded_runtime._native_backend = backend
    loaded_runtime._execution_mode = "native_async"

    with pytest.raises(RuntimeError, match="did not quiesce"):
        loaded_runtime.unload()

    assert loaded_runtime.compiled_model is not None
    assert loaded_runtime._native_backend is backend
    with pytest.raises(RuntimeError, match="cleanup is incomplete"):
        loaded_runtime.load(loaded_runtime.compiled_model)

    loaded_runtime.unload()
    assert backend.calls == [300.0, 300.0]
    assert loaded_runtime.compiled_model is None
    assert loaded_runtime._native_backend is None
