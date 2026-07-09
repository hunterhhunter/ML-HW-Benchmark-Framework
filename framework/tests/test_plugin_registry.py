import sys
import time
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pytest

import compilers as compiler_registry
import monitors as monitor_registry
import runtimes as runtime_registry
from compilers import (
    CompilerEntry,
    get_compiler,
    get_compiler_entry,
    list_compilers,
    normalize_compile_result,
    register_compiler,
)
from core.compiled_model import CompiledModel
from core.model_spec import Model_Spec, Task
from core.targets import (
    TargetSpec,
    get_target,
    list_targets,
    resolve_target,
    validate_registry_graph,
)
from monitors import (
    CollectorEntry,
    create_hw_monitor,
    get_collector_entry,
    list_collectors,
    register_collector,
)
from runtimes import (
    RuntimeEntry,
    create_runtime,
    get_runtime_entry,
    list_runtimes,
    register_runtime,
)


def _make_spec(
    source_path: Path,
    input_shapes: dict[str, tuple[int, ...]] | None = None,
    output_shapes: dict[str, tuple[int, ...]] | None = None,
) -> Model_Spec:
    source_path.write_text("dummy onnx", encoding="utf-8")
    input_shapes = input_shapes or {"input": (1, 3, 4, 4)}
    output_shapes = output_shapes or {"logits": (1, 10)}
    return Model_Spec(
        name="dummy",
        task=Task.IMAGE_CLASSIFICATION,
        input_shapes=input_shapes,
        input_dtype={name: "float32" for name in input_shapes},
        output_shapes=output_shapes,
        model_paths={"onnx": str(source_path)},
    )


def _install_fake_dx_engine(
    monkeypatch,
    input_names: tuple[str, ...] = ("input",),
    output_names: tuple[str, ...] = ("logits",),
    output_shape: tuple[int, ...] = (1, 10),
):
    state = {"engines": [], "async_calls": []}

    class FakeBoundOption:
        NPU_ALL = "NPU_ALL"
        NPU_0 = "NPU_0"
        NPU_1 = "NPU_1"
        NPU_2 = "NPU_2"
        NPU_01 = "NPU_01"
        NPU_12 = "NPU_12"
        NPU_02 = "NPU_02"

    class FakeInferenceOption:
        BOUND_OPTION = FakeBoundOption

        def __init__(self):
            self.devices = None
            self.bound_option = None
            self.use_ort = None
            self.buffer_count = None

        def set_devices(self, devices):
            self.devices = devices

        def set_bound_option(self, bound_option):
            self.bound_option = bound_option

        def set_use_ort(self, use_ort):
            self.use_ort = use_ort

        def set_buffer_count(self, buffer_count):
            self.buffer_count = buffer_count

    class FakeInferenceEngine:
        def __init__(self, model_path, option=None):
            self.model_path = model_path
            self.option = option
            self.calls = []
            self.disposed = False
            state["engines"].append(self)

        def get_input_tensor_names(self):
            return list(input_names)

        def get_input_tensors_info(self):
            return [
                {"name": name, "shape": [1, 3, 4, 4], "dtype": np.dtype("float32"), "elem_size": 4}
                for name in input_names
            ]

        def get_output_tensor_names(self):
            return list(output_names)

        def get_output_tensors_info(self):
            return [
                {"name": name, "shape": list(output_shape), "dtype": np.dtype("float32"), "elem_size": 4}
                for name in output_names
            ]

        def run(self, input_data):
            self.calls.append(("run", input_data))
            if (
                len(input_names) == 1
                and isinstance(input_data, list)
                and len(input_data) > 1
                and all(isinstance(item, np.ndarray) for item in input_data)
            ):
                return [
                    [np.full(output_shape, float(idx + 1), dtype=np.float32)]
                    for idx in range(len(input_data))
                ]
            if isinstance(input_data, np.ndarray):
                return [np.full(output_shape, 5.0, dtype=np.float32)]
            if input_data and isinstance(input_data[0], list):
                return [
                    [np.full(output_shape, float(idx + 1), dtype=np.float32)]
                    for idx in range(len(input_data))
                ]
            return [np.full(output_shape, 3.0, dtype=np.float32)]

        def run_multi_input(self, input_tensors):
            self.calls.append(("run_multi_input", input_tensors))
            return [np.full(output_shape, 7.0, dtype=np.float32)]

        def run_async(self, input_data, user_arg=None, output_buffer=None):
            state["async_calls"].append(("run_async", input_data, user_arg, output_buffer))
            return 1

        def run_async_multi_input(self, input_tensors, user_arg=None, output_buffer=None):
            state["async_calls"].append(("run_async_multi_input", input_tensors, user_arg, output_buffer))
            return 2

        def wait(self, job_id):
            state["async_calls"].append(("wait", job_id))
            return [np.full(output_shape, 9.0, dtype=np.float32)]

        def dispose(self):
            self.disposed = True

    fake_module = types.ModuleType("dx_engine")
    fake_module.__version__ = "fake-1.1.4"
    fake_module.InferenceOption = FakeInferenceOption
    fake_module.InferenceEngine = FakeInferenceEngine
    monkeypatch.setitem(sys.modules, "dx_engine", fake_module)
    return state


def _write_fake_dxcom(path: Path, artifact_names: tuple[str, ...] = ("compiled.dxnn",)) -> Path:
    artifact_lines = "\n".join(
        f"(out / {name!r}).write_text('fake dxnn', encoding='utf-8')"
        for name in artifact_names
    )
    script = f"""#!{sys.executable}
import json
import sys
from pathlib import Path

if "--version" in sys.argv:
    print("dxcom fake 2.3.0")
    sys.exit(0)

args = sys.argv[1:]
out = Path(args[args.index("-o") + 1])
out.mkdir(parents=True, exist_ok=True)
with (out / "invocations.jsonl").open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(args) + "\\n")
{artifact_lines}
"""
    path.write_text(script, encoding="utf-8")
    path.chmod(0o755)
    return path


def test_builtin_registries_expose_mock_npu():
    assert any(item["name"] == "mock_npu" for item in list_runtimes())
    assert any(item["name"] == "mock_npu" for item in list_compilers())
    assert any(item["name"] == "mock_npu" for item in list_collectors())
    assert any(target.target_id == "vendor_mock_npu" for target in list_targets())


def test_builtin_registries_expose_hailo8():
    assert any(item["name"] == "hailort" for item in list_runtimes())
    assert any(item["name"] == "hailo" for item in list_collectors())
    target = get_target("hailo8")
    assert target.runtime_name == "hailort"
    assert target.artifact_format == "hef"
    assert target.runtime_options["input_format_type"] == "uint8"
    assert target.runtime_options["output_format_type"] == "auto"
    assert target.runtime_options["accelerator_name"] == "Hailo-8 M.2"
    assert "hailo" in target.monitor_names


def test_builtin_registries_expose_hailo10h():
    target = get_target("hailo10h")
    assert target.runtime_name == "hailort"
    assert target.artifact_format == "hef"
    assert target.accelerator_vendor == "Hailo"
    assert target.accelerator_name == "Hailo-10H"
    assert target.runtime_options["output_format_type"] == "auto"
    assert target.runtime_options["accelerator_name"] == "Hailo-10H"
    assert target.monitor_options["hailo"]["accelerator_name"] == "Hailo-10H"
    assert "hailo" in target.monitor_names


def test_builtin_registries_expose_deepx():
    assert any(item["name"] == "deepx" for item in list_runtimes())
    assert any(item["name"] == "deepx" for item in list_compilers())
    assert any(item["name"] == "deepx" for item in list_collectors())
    target = get_target("deepx")
    assert target.runtime_name == "deepx"
    assert target.compiler_name == "deepx"
    assert target.artifact_format == "dxnn"
    assert target.accelerator_vendor == "DEEPX"
    assert "compile" in target.capabilities
    assert "monitor" in target.capabilities
    assert "deepx" in target.monitor_names
    assert target.runtime_options["sdk_module"] == "dx_engine"
    assert target.runtime_options["bound_option"] == "NPU_ALL"


def test_registry_entry_lookup_helpers_normalize_aliases():
    assert get_runtime_entry(" ONNX ").name == "onnxruntime"
    assert get_compiler_entry(" DXCOM ").name == "deepx"
    assert get_collector_entry(" HAILORT ").name == "hailo"


def test_component_registries_reject_alias_collisions(monkeypatch):
    monkeypatch.setattr(
        runtime_registry,
        "_RUNTIME_REGISTRY",
        dict(runtime_registry._RUNTIME_REGISTRY),
    )
    monkeypatch.setattr(
        compiler_registry,
        "_COMPILER_REGISTRY",
        dict(compiler_registry._COMPILER_REGISTRY),
    )
    monkeypatch.setattr(
        monitor_registry,
        "_COLLECTOR_REGISTRY",
        dict(monitor_registry._COLLECTOR_REGISTRY),
    )

    with pytest.raises(ValueError, match="runtime registry key 'onnx'"):
        register_runtime(RuntimeEntry(
            name="other_runtime",
            module="runtimes.mock_npu_rt",
            class_name="MockNpuRuntime",
            aliases=("onnx",),
        ))
    assert get_runtime_entry("onnx").name == "onnxruntime"

    with pytest.raises(ValueError, match="compiler registry key 'dxcom'"):
        register_compiler(CompilerEntry(
            name="other_compiler",
            module="compilers.mock_npu_compiler",
            class_name="MockNpuCompiler",
            aliases=("dxcom",),
        ))
    assert get_compiler_entry("dxcom").name == "deepx"

    with pytest.raises(ValueError, match="collector registry key 'hailort'"):
        register_collector(CollectorEntry(
            name="other_collector",
            module="monitors.mock_npu_collector",
            class_name="MockNpuCollector",
            aliases=("hailort",),
        ))
    assert get_collector_entry("hailort").name == "hailo"


def test_builtin_targets_pass_registry_graph_validation():
    report = validate_registry_graph()

    assert report["ok"], report
    assert report["errors"] == []
    assert report["warnings"] == []
    assert {item["target_id"] for item in report["targets"]} == {
        target.target_id for target in list_targets()
    }


def test_registry_graph_reports_missing_component_references():
    broken_target = TargetSpec(
        target_id="broken_npu",
        label="Broken NPU",
        runtime_name="missing_runtime",
        device="npu0",
        compiler_name="missing_compiler",
        monitor_names=("missing_collector",),
        artifact_format="",
        capabilities=("onnx",),
    )

    report = validate_registry_graph([broken_target])

    assert report["ok"] is False
    assert {(error["field"], error["value"]) for error in report["errors"]} == {
        ("runtime_name", "missing_runtime"),
        ("compiler_name", "missing_compiler"),
        ("monitor_names", "missing_collector"),
        ("artifact_format", ""),
    }


def test_registry_graph_warns_on_intentional_unsupported_runtime():
    iree_target = TargetSpec(
        target_id="iree-cuda",
        label="IREE CUDA",
        runtime_name="iree",
        device="cuda",
        monitor_names=("system",),
        artifact_format="vmfb",
        capabilities=("local",),
    )

    report = validate_registry_graph([iree_target])
    strict_report = validate_registry_graph([iree_target], strict=True)

    assert report["ok"] is True
    assert report["warnings"][0]["field"] == "runtime_name"
    assert "marked unsupported" in report["warnings"][0]["message"]
    assert strict_report["ok"] is False


def test_registry_graph_rejects_known_compiler_artifact_mismatch():
    mismatched_target = TargetSpec(
        target_id="mock_mismatch",
        label="Mock Mismatch",
        runtime_name="mock_npu",
        device="npu0",
        compiler_name="mock_npu",
        monitor_names=("mock_npu", "system"),
        artifact_format="wrongbin",
        capabilities=("onnx", "compile", "monitor"),
        compiler_options={"artifact_format": "mockbin"},
    )

    report = validate_registry_graph([mismatched_target])

    assert report["ok"] is False
    assert report["errors"][0]["field"] == "artifact_format"
    assert "compiler_options" in report["errors"][0]["message"]


def test_resolve_target_preserves_legacy_cpu():
    target = resolve_target(None, "onnxruntime", "cpu")
    assert target.target_id == "cpu"
    assert target.runtime_name == "onnxruntime"
    assert target.device == "cpu"


def test_resolve_target_maps_hailort_backend():
    target = resolve_target(None, "hailort", "device0")
    assert target.target_id == "hailo8"
    assert target.device == "device0"


def test_resolve_target_accepts_explicit_hailo10h_target():
    target = resolve_target("hailo10h", "hailort", "device0")
    assert target.target_id == "hailo10h"
    assert target.runtime_name == "hailort"
    assert target.accelerator_name == "Hailo-10H"


def test_resolve_target_maps_deepx_backend():
    target = resolve_target(None, "deepx", "npu0")
    assert target.target_id == "deepx"
    assert target.device == "npu0"


def test_deepx_runtime_sets_inference_option_and_disposes(monkeypatch, tmp_path):
    state = _install_fake_dx_engine(monkeypatch)
    spec = _make_spec(tmp_path / "source.onnx")
    artifact = tmp_path / "model.dxnn"
    artifact.write_text("fake deepx artifact", encoding="utf-8")

    runtime = create_runtime(
        "deepx",
        device="npu2",
        device_ids="1,2",
        bound_option="NPU_01",
        use_ort=True,
        buffer_count=8,
    )
    compiled_model = CompiledModel(
        spec=spec,
        backend_name="deepx",
        artifact_path=artifact,
    )
    runtime.load(compiled_model)
    engine = state["engines"][0]
    runtime.unload()

    assert engine.option.devices == [1, 2]
    assert engine.option.bound_option == "NPU_01"
    assert engine.option.use_ort is True
    assert engine.option.buffer_count == 8
    assert engine.disposed is True


def test_deepx_runtime_rejects_invalid_bound_option(monkeypatch, tmp_path):
    state = _install_fake_dx_engine(monkeypatch)
    spec = _make_spec(tmp_path / "source.onnx")
    artifact = tmp_path / "model.dxnn"
    artifact.write_text("fake deepx artifact", encoding="utf-8")
    runtime = create_runtime("deepx", device="npu0", bound_option="NPU_99")

    with pytest.raises(ValueError, match="Unsupported DeepX bound_option"):
        runtime.load(CompiledModel(spec=spec, backend_name="deepx", artifact_path=artifact))

    assert state["engines"] == []


@pytest.mark.parametrize("buffer_count", [0, 101])
def test_deepx_runtime_rejects_invalid_buffer_count(monkeypatch, tmp_path, buffer_count):
    state = _install_fake_dx_engine(monkeypatch)
    spec = _make_spec(tmp_path / "source.onnx")
    artifact = tmp_path / "model.dxnn"
    artifact.write_text("fake deepx artifact", encoding="utf-8")
    runtime = create_runtime("deepx", device="npu0", buffer_count=buffer_count)

    with pytest.raises(ValueError, match="DeepX buffer_count"):
        runtime.load(CompiledModel(spec=spec, backend_name="deepx", artifact_path=artifact))

    assert state["engines"] == []


def test_deepx_runtime_single_input_uses_dxrt_run_formats(monkeypatch, tmp_path):
    state = _install_fake_dx_engine(monkeypatch)
    spec = _make_spec(tmp_path / "source.onnx")
    artifact = tmp_path / "model.dxnn"
    artifact.write_text("fake deepx artifact", encoding="utf-8")

    runtime = create_runtime("deepx", device="npu0")
    runtime.load(CompiledModel(spec=spec, backend_name="deepx", artifact_path=artifact))

    single_outputs = runtime.run({"input": np.ones((1, 3, 4, 4), dtype=np.float32)})
    batch_outputs = runtime.run({"input": np.ones((2, 3, 4, 4), dtype=np.float32)})
    runtime.unload()

    engine = state["engines"][0]
    single_call = engine.calls[0]
    batch_call = engine.calls[1]

    assert single_call[0] == "run"
    assert len(single_call[1]) == 1
    assert single_call[1][0].shape == (1, 3, 4, 4)
    assert single_outputs["logits"].shape == (1, 10)
    assert np.all(single_outputs["logits"] == 3.0)

    assert batch_call[0] == "run"
    assert len(batch_call[1]) == 2
    assert batch_call[1][0].shape == (1, 3, 4, 4)
    assert batch_outputs["logits"].shape == (2, 10)
    assert np.all(batch_outputs["logits"][0] == 1.0)
    assert np.all(batch_outputs["logits"][1] == 2.0)


def test_deepx_runtime_single_input_can_run_array_style(monkeypatch, tmp_path):
    state = _install_fake_dx_engine(monkeypatch)
    spec = _make_spec(tmp_path / "source.onnx")
    artifact = tmp_path / "model.dxnn"
    artifact.write_text("fake deepx artifact", encoding="utf-8")

    runtime = create_runtime("deepx", device="npu0", single_input_run_style="array")
    runtime.load(CompiledModel(spec=spec, backend_name="deepx", artifact_path=artifact))

    outputs = runtime.run({"input": np.ones((1, 3, 4, 4), dtype=np.float32)})
    runtime.unload()

    call = state["engines"][0].calls[0]
    assert call[0] == "run"
    assert isinstance(call[1], np.ndarray)
    assert call[1].shape == (1, 3, 4, 4)
    assert outputs["logits"].shape == (1, 10)
    assert np.all(outputs["logits"] == 5.0)


def test_deepx_runtime_can_squeeze_single_input_batch_axis(monkeypatch, tmp_path):
    state = _install_fake_dx_engine(monkeypatch)
    spec = _make_spec(tmp_path / "source.onnx")
    artifact = tmp_path / "model.dxnn"
    artifact.write_text("fake deepx artifact", encoding="utf-8")

    runtime = create_runtime("deepx", device="npu0", input_batch_axis="squeeze")
    runtime.load(CompiledModel(spec=spec, backend_name="deepx", artifact_path=artifact))

    runtime.run({"input": np.ones((1, 3, 4, 4), dtype=np.float32)})
    runtime.unload()

    call = state["engines"][0].calls[0]
    assert call[0] == "run"
    assert len(call[1]) == 1
    assert call[1][0].shape == (3, 4, 4)


def test_deepx_runtime_can_cast_input_to_uint8_nhwc(monkeypatch, tmp_path):
    state = _install_fake_dx_engine(monkeypatch)
    spec = _make_spec(tmp_path / "source.onnx")
    artifact = tmp_path / "model.dxnn"
    artifact.write_text("fake deepx artifact", encoding="utf-8")

    runtime = create_runtime(
        "deepx",
        device="npu0",
        input_layout="NHWC",
        input_dtype="uint8",
    )
    runtime.load(CompiledModel(spec=spec, backend_name="deepx", artifact_path=artifact))

    sample = np.array([[[[-1, 0, 128, 300]] * 4] * 3], dtype=np.float32)
    runtime.run({"input": sample})
    runtime.unload()

    call = state["engines"][0].calls[0]
    assert call[0] == "run"
    assert len(call[1]) == 1
    sdk_input = call[1][0]
    assert sdk_input.shape == (1, 4, 4, 3)
    assert sdk_input.dtype == np.uint8
    assert sdk_input.min() == 0
    assert sdk_input.max() == 255


def test_deepx_runtime_can_match_dxapp_uint8_nhwc_single_sample(monkeypatch, tmp_path):
    state = _install_fake_dx_engine(monkeypatch)
    spec = _make_spec(tmp_path / "source.onnx")
    artifact = tmp_path / "model.dxnn"
    artifact.write_text("fake deepx artifact", encoding="utf-8")

    runtime = create_runtime(
        "deepx",
        device="npu0",
        input_layout="NHWC",
        input_dtype="uint8",
        input_batch_axis="squeeze",
        single_input_run_style="list",
    )
    runtime.load(CompiledModel(spec=spec, backend_name="deepx", artifact_path=artifact))

    runtime.run({"input": np.ones((1, 3, 4, 4), dtype=np.float32) * 128})
    runtime.unload()

    call = state["engines"][0].calls[0]
    assert call[0] == "run"
    assert isinstance(call[1], list)
    assert len(call[1]) == 1
    sdk_input = call[1][0]
    assert sdk_input.shape == (4, 4, 3)
    assert sdk_input.dtype == np.uint8
    assert np.all(sdk_input == 128)


def test_deepx_runtime_run_and_warmup_never_use_async_api(monkeypatch, tmp_path):
    state = _install_fake_dx_engine(monkeypatch)
    spec = _make_spec(tmp_path / "source.onnx")
    artifact = tmp_path / "model.dxnn"
    artifact.write_text("fake deepx artifact", encoding="utf-8")

    runtime = create_runtime("deepx", device="npu0")
    runtime.load(CompiledModel(spec=spec, backend_name="deepx", artifact_path=artifact))
    sample = {"input": np.ones((1, 3, 4, 4), dtype=np.float32)}

    runtime.warmup(sample, num_runs=2)
    runtime.run(sample)
    runtime.unload()

    engine = state["engines"][0]
    assert [call[0] for call in engine.calls] == ["run", "run", "run"]
    assert state["async_calls"] == []


def test_deepx_runtime_microbatch_mode_runs_one_sample_at_a_time(monkeypatch, tmp_path):
    state = _install_fake_dx_engine(monkeypatch)
    spec = _make_spec(tmp_path / "source.onnx")
    artifact = tmp_path / "model.dxnn"
    artifact.write_text("fake deepx artifact", encoding="utf-8")

    runtime = create_runtime("deepx", device="npu0", batch_mode="microbatch")
    runtime.load(CompiledModel(spec=spec, backend_name="deepx", artifact_path=artifact))
    outputs = runtime.run({"input": np.ones((2, 3, 4, 4), dtype=np.float32)})
    runtime.unload()

    engine = state["engines"][0]
    assert [call[0] for call in engine.calls] == ["run", "run"]
    assert all(len(call[1]) == 1 for call in engine.calls)
    assert all(call[1][0].shape == (1, 3, 4, 4) for call in engine.calls)
    assert outputs["logits"].shape == (2, 10)
    assert np.all(outputs["logits"] == 3.0)


def test_deepx_runtime_multi_input_uses_named_and_explicit_batch_formats(monkeypatch, tmp_path):
    state = _install_fake_dx_engine(
        monkeypatch,
        input_names=("left", "right"),
        output_names=("scores",),
        output_shape=(1, 5),
    )
    spec = _make_spec(
        tmp_path / "source.onnx",
        input_shapes={"left": (1, 3, 4, 4), "right": (1, 3, 4, 4)},
        output_shapes={"scores": (1, 5)},
    )
    artifact = tmp_path / "model.dxnn"
    artifact.write_text("fake deepx artifact", encoding="utf-8")

    runtime = create_runtime("deepx", device="npu0")
    runtime.load(CompiledModel(spec=spec, backend_name="deepx", artifact_path=artifact))

    single_outputs = runtime.run({
        "left": np.ones((1, 3, 4, 4), dtype=np.float32),
        "right": np.ones((1, 3, 4, 4), dtype=np.float32),
    })
    batch_outputs = runtime.run({
        "left": np.ones((2, 3, 4, 4), dtype=np.float32),
        "right": np.ones((2, 3, 4, 4), dtype=np.float32),
    })
    runtime.unload()

    engine = state["engines"][0]
    single_call = engine.calls[0]
    batch_call = engine.calls[1]

    assert single_call[0] == "run_multi_input"
    assert set(single_call[1]) == {"left", "right"}
    assert single_outputs["scores"].shape == (1, 5)
    assert np.all(single_outputs["scores"] == 7.0)

    assert batch_call[0] == "run"
    assert len(batch_call[1]) == 2
    assert len(batch_call[1][0]) == 2
    assert batch_call[1][0][0].shape == (1, 3, 4, 4)
    assert batch_outputs["scores"].shape == (2, 5)
    assert np.all(batch_outputs["scores"][0] == 1.0)
    assert np.all(batch_outputs["scores"][1] == 2.0)


def test_deepx_runtime_missing_dx_engine_error_mentions_package(monkeypatch, tmp_path):
    monkeypatch.delitem(sys.modules, "dx_engine", raising=False)
    spec = _make_spec(tmp_path / "source.onnx")
    artifact = tmp_path / "model.dxnn"
    artifact.write_text("fake deepx artifact", encoding="utf-8")
    runtime = create_runtime("deepx", device="npu0")

    with pytest.raises(ImportError, match="dx_engine"):
        runtime.load(CompiledModel(spec=spec, backend_name="deepx", artifact_path=artifact))


def test_deepx_compiler_builds_dxcom_command_and_caches(tmp_path):
    fake_dxcom = _write_fake_dxcom(tmp_path / "dxcom")
    config_path = tmp_path / "config.json"
    config_path.write_text('{"inputs": {"input": [1, 3, 4, 4]}}', encoding="utf-8")
    spec = _make_spec(tmp_path / "source.onnx")

    compiler = get_compiler(
        "deepx",
        config_path=str(config_path),
        dxcom_bin=str(fake_dxcom),
        opt_level="1",
        aggressive_partitioning="true",
        gen_log="true",
        float64_calibration="true",
        compile_input_nodes="Conv1,Conv2",
        compile_output_nodes="Out1",
    )
    first = normalize_compile_result(compiler.compile(spec, str(tmp_path / "artifacts")))
    second = normalize_compile_result(compiler.compile(spec, str(tmp_path / "artifacts")))

    artifact_path = Path(first.artifact_path)
    assert artifact_path.exists()
    assert artifact_path.name.startswith("dummy_deepx_")
    assert artifact_path.suffix == ".dxnn"
    assert first.metadata["compiler_name"] == "deepx"
    assert first.metadata["compiler_version"] == "dxcom fake 2.3.0"
    assert first.metadata["artifact_format"] == "dxnn"
    assert first.metadata["cache_hit"] is False
    assert second.metadata["cache_hit"] is True
    assert second.artifact_path == first.artifact_path

    command = first.metadata["compiler_command"]
    assert command[0] == str(fake_dxcom)
    assert command[command.index("-m") + 1] == str(tmp_path / "source.onnx")
    assert command[command.index("-c") + 1] == str(config_path)
    assert "--opt_level" in command
    assert command[command.index("--opt_level") + 1] == "1"
    assert "--aggressive_partitioning" in command
    assert "--gen_log" in command
    assert "--float64_calibration" in command
    assert "--compile_input_nodes" in command
    assert command[command.index("--compile_input_nodes") + 1] == "Conv1,Conv2"
    assert "--compile_output_nodes" in command
    assert command[command.index("--compile_output_nodes") + 1] == "Out1"

    invocations = (Path(first.metadata["output_dir"]) / "invocations.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(invocations) == 1


def test_deepx_compiler_requires_config_path():
    with pytest.raises(ValueError, match="config_path"):
        get_compiler("deepx")


def test_deepx_compiler_rejects_unknown_options(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported DeepX compiler option"):
        get_compiler("deepx", config_path=str(config_path), unsupported="1")


def test_deepx_compiler_missing_dxcom_error(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text("{}", encoding="utf-8")
    spec = _make_spec(tmp_path / "source.onnx")
    compiler = get_compiler("deepx", config_path=str(config_path), dxcom_bin=str(tmp_path / "missing_dxcom"))

    with pytest.raises(FileNotFoundError, match="DX-COM executable"):
        compiler.compile(spec, str(tmp_path / "artifacts"))


def test_deepx_compiler_errors_when_dxcom_outputs_no_dxnn(tmp_path):
    fake_dxcom = _write_fake_dxcom(tmp_path / "dxcom", artifact_names=())
    config_path = tmp_path / "config.json"
    config_path.write_text("{}", encoding="utf-8")
    spec = _make_spec(tmp_path / "source.onnx")
    compiler = get_compiler("deepx", config_path=str(config_path), dxcom_bin=str(fake_dxcom))

    with pytest.raises(RuntimeError, match="produced no .dxnn"):
        compiler.compile(spec, str(tmp_path / "artifacts"))


def test_deepx_compiler_errors_on_ambiguous_dxnn_outputs(tmp_path):
    fake_dxcom = _write_fake_dxcom(tmp_path / "dxcom", artifact_names=("a.dxnn", "b.dxnn"))
    config_path = tmp_path / "config.json"
    config_path.write_text("{}", encoding="utf-8")
    spec = _make_spec(tmp_path / "source.onnx")
    compiler = get_compiler("deepx", config_path=str(config_path), dxcom_bin=str(fake_dxcom))

    with pytest.raises(RuntimeError, match="multiple .dxnn"):
        compiler.compile(spec, str(tmp_path / "artifacts"))


def test_mock_npu_compile_cache_and_runtime(tmp_path):
    spec = _make_spec(tmp_path / "model.onnx")
    target = get_target("vendor_mock_npu")

    compiler = get_compiler(target.compiler_name, **target.compiler_options)
    first = normalize_compile_result(compiler.compile(spec, str(tmp_path / "artifacts")))
    second = normalize_compile_result(compiler.compile(spec, str(tmp_path / "artifacts")))

    assert Path(first.artifact_path).exists()
    assert first.metadata["cache_hit"] is False
    assert second.metadata["cache_hit"] is True

    runtime = create_runtime(target.runtime_name, device=target.device, **target.runtime_options)
    compiled_model = CompiledModel(
        spec=spec,
        backend_name=target.runtime_name,
        artifact_path=Path(first.artifact_path),
    )
    runtime.load(compiled_model)
    outputs = runtime.run({"input": np.ones((2, 3, 4, 4), dtype=np.float32)})
    runtime.unload()

    assert outputs["logits"].shape == (2, 10)
    assert np.all(outputs["logits"] == 0)


def test_mock_npu_monitor_summary_contains_accel_metrics():
    target = get_target("vendor_mock_npu")
    monitor = create_hw_monitor(
        interval=0.01,
        collector_names=list(target.monitor_names),
        collector_options=target.monitor_options,
    )
    monitor.start()
    time.sleep(0.03)
    monitor.stop()
    summary = monitor.summary()

    assert summary["hw_accel_vendor"] == "MockNPU"
    assert summary["hw_accel_name"] == "Mock NPU PCIe Adapter"
    assert "hw_accel_util_avg" in summary
