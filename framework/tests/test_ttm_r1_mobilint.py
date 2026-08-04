from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ttm_r1.contracts import TTMR1Contract
from tools.ttm_r1_vendors.mobilint import (
    compile_mblt,
    export_core_onnx,
    run_mblt,
    run_onnx_reference,
)


class _Core(torch.nn.Module):
    def forward(self, past_values):
        del past_values
        return torch.zeros((1, 96, 1), dtype=torch.float32)


def test_mobilint_export_uses_legacy_static_onnx_exporter(monkeypatch, tmp_path):
    """Catches dynamic/dynamo ONNX export leaking into ARIES setup."""
    captured = {}

    def fake_export(model, inputs, path, **kwargs):
        captured.update(model=model, inputs=inputs, **kwargs)
        Path(path).write_bytes(b"onnx")

    monkeypatch.setattr(torch.onnx, "export", fake_export)
    output = export_core_onnx(
        _Core(),
        (torch.ones((1, 512, 1), dtype=torch.float32),),
        TTMR1Contract.fixed(),
        tmp_path / "core.onnx",
    )

    assert output.is_file()
    assert captured["dynamo"] is False
    assert captured["dynamic_axes"] is None
    assert captured["input_names"] == ["past_values"]
    assert captured["output_names"] == ["forecast"]


def test_mobilint_compile_uses_aries_rb_and_a_new_mblt_path(tmp_path):
    """Catches compiling an artifact for a different Mobilint family."""
    onnx_path = tmp_path / "core.onnx"
    onnx_path.write_bytes(b"onnx")
    artifact = tmp_path / "core.mblt"
    captured = {}

    def compile_v2(model, *, target_device, mblt_save_path, backend, device, cpu_offload):
        captured.update(
            model=model,
            target_device=target_device,
            mblt_save_path=mblt_save_path,
            backend=backend,
            device=device,
            cpu_offload=cpu_offload,
        )
        Path(mblt_save_path).write_bytes(b"mblt")

    report = compile_mblt(
        onnx_path,
        artifact,
        qbcompiler_module=SimpleNamespace(mblt_compile_V2=compile_v2),
    )

    assert captured["target_device"] == "aries-rb"
    assert report["artifact"]["size_bytes"] == 4


def test_mobilint_onnx_reference_uses_cpu_provider_and_fixed_tensor_names(tmp_path):
    """Catches an exported ONNX graph being trusted without a CPU execution check."""
    onnx_path = tmp_path / "core.onnx"
    onnx_path.write_bytes(b"onnx")
    captured = {}

    class _Session:
        def run(self, outputs, inputs):
            captured["outputs"] = outputs
            captured["inputs"] = inputs
            return [np.zeros((1, 96, 1), dtype=np.float32)]

    runtime = SimpleNamespace(
        InferenceSession=lambda path, providers: (
            captured.update(path=path, providers=providers) or _Session()
        )
    )

    output = run_onnx_reference(
        onnx_path,
        (np.ones((1, 512, 1), dtype=np.float32),),
        TTMR1Contract.fixed(),
        onnxruntime_module=runtime,
    )

    assert output.shape == (1, 96, 1)
    assert captured["providers"] == ["CPUExecutionProvider"]
    assert captured["outputs"] == ["forecast"]
    assert set(captured["inputs"]) == {"past_values"}


def test_mobilint_first_run_uses_float_output_and_validates_every_tensor(tmp_path):
    """Catches a compile-only MBLT being counted as an ARIES inference."""
    artifact = tmp_path / "core.mblt"
    artifact.write_bytes(b"mblt")
    captured = {}

    class _Model:
        def get_model_input_shape(self):
            return [(1, 512, 1)]

        def get_model_output_shape(self):
            return [(1, 96, 1)]

        def infer_to_float(self, inputs):
            captured["inputs"] = inputs
            return [np.zeros((1, 96, 1), dtype=np.float32)]

    runtime = SimpleNamespace(
        get_available_device_numbers=lambda: [0],
        load=lambda _path: _Model(),
    )
    output = run_mblt(
        artifact,
        (np.ones((1, 512, 1), dtype=np.float32),),
        TTMR1Contract.fixed(),
        qbruntime_module=runtime,
    )

    assert output.shape == (1, 96, 1)
    assert captured["inputs"][0].dtype == np.float32


def test_mobilint_run_rejects_a_host_without_aries_zero(tmp_path):
    """Catches host inference being mistaken for ARIES device execution."""
    artifact = tmp_path / "core.mblt"
    artifact.write_bytes(b"mblt")
    runtime = SimpleNamespace(get_available_device_numbers=lambda: [])

    with pytest.raises(RuntimeError, match="device 0"):
        run_mblt(
            artifact,
            (np.ones((1, 512, 1), dtype=np.float32),),
            TTMR1Contract.fixed(),
            qbruntime_module=runtime,
        )
