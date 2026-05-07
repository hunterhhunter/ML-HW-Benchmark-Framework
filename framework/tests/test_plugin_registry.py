import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from compilers import get_compiler, list_compilers, normalize_compile_result
from core.compiled_model import CompiledModel
from core.model_spec import Model_Spec, Task
from core.targets import get_target, list_targets, resolve_target
from monitors import create_hw_monitor, list_collectors
from runtimes import create_runtime, list_runtimes


def _make_spec(source_path: Path) -> Model_Spec:
    source_path.write_text("dummy onnx", encoding="utf-8")
    return Model_Spec(
        name="dummy",
        task=Task.IMAGE_CLASSIFICATION,
        input_shapes={"input": (1, 3, 4, 4)},
        input_dtype={"input": "float32"},
        output_shapes={"logits": (1, 10)},
        model_paths={"onnx": str(source_path)},
    )


def test_builtin_registries_expose_mock_npu():
    assert any(item["name"] == "mock_npu" for item in list_runtimes())
    assert any(item["name"] == "mock_npu" for item in list_compilers())
    assert any(item["name"] == "mock_npu" for item in list_collectors())
    assert any(target.target_id == "vendor_mock_npu" for target in list_targets())


def test_resolve_target_preserves_legacy_cpu():
    target = resolve_target(None, "onnxruntime", "cpu")
    assert target.target_id == "cpu"
    assert target.runtime_name == "onnxruntime"
    assert target.device == "cpu"


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
