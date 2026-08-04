from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from chronos_bolt.contracts import ChronosBoltContract
from tools.chronos_bolt_vendors.mobilint import compile_mblt, export_core_onnx, run_mblt


def _contract():
    return ChronosBoltContract.tiny(d_model=4, use_reg_token=True)


def test_mobilint_export_uses_legacy_static_onnx_exporter(monkeypatch, tmp_path):
    """Catches a PyTorch default dynamo exporter dependency leaking into ARIES setup."""
    contract = _contract()
    captured = {}

    class _Core(torch.nn.Module):
        def forward(self, input_embeds, attention_mask, decoder_input_embeds):
            del input_embeds, attention_mask, decoder_input_embeds
            return torch.zeros(contract.core_output.shape, dtype=torch.float32)

    def fake_export(model, inputs, path, **kwargs):
        captured["model"] = model
        captured["inputs"] = inputs
        captured.update(kwargs)
        Path(path).write_bytes(b"onnx")

    monkeypatch.setattr(torch.onnx, "export", fake_export)
    inputs = tuple(torch.ones(item.shape, dtype=torch.float32) for item in contract.core_inputs)

    output = export_core_onnx(_Core(), inputs, contract, tmp_path / "core.onnx")

    assert output.is_file()
    assert captured["dynamo"] is False
    assert captured["dynamic_axes"] is None
    assert captured["input_names"] == [item.name for item in contract.core_inputs]


def test_mobilint_compile_uses_aries_rb_and_a_new_mblt_path(tmp_path):
    """Catches compiling the ARIES artifact for another Mobilint family."""
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

    compiler = SimpleNamespace(mblt_compile_V2=compile_v2)

    report = compile_mblt(onnx_path, artifact, qbcompiler_module=compiler)

    assert captured == {
        "model": str(onnx_path),
        "target_device": "aries-rb",
        "mblt_save_path": str(artifact),
        "backend": "onnx",
        "device": "cpu",
        "cpu_offload": False,
    }
    assert report["target_device"] == "aries-rb"
    assert report["artifact"]["size_bytes"] == 4


def test_mobilint_first_run_uses_float_output_and_validates_every_tensor(tmp_path):
    """Catches treating a compile-only MBLT as an ARIES device execution."""
    contract = _contract()
    artifact = tmp_path / "core.mblt"
    artifact.write_bytes(b"mblt")
    captured = {}

    class _Model:
        def get_model_input_shape(self):
            return [item.shape for item in contract.core_inputs]

        def get_model_output_shape(self):
            return [contract.core_output.shape]

        def infer_to_float(self, inputs):
            captured["inputs"] = inputs
            return [np.zeros(contract.core_output.shape, dtype=np.float32)]

    runtime = SimpleNamespace(
        get_available_device_numbers=lambda: [0],
        load=lambda path: _Model(),
    )
    inputs = tuple(
        np.ones(item.shape, dtype=np.float32) for item in contract.core_inputs
    )

    output = run_mblt(artifact, inputs, contract, qbruntime_module=runtime)

    assert output.shape == (1, 9, 64)
    assert output.dtype == np.float32
    assert len(captured["inputs"]) == 3
    assert all(value.dtype == np.float32 for value in captured["inputs"])


def test_mobilint_run_rejects_a_host_without_aries_zero(tmp_path):
    """Catches host execution being mistaken for an ARIES inference."""
    contract = _contract()
    artifact = tmp_path / "core.mblt"
    artifact.write_bytes(b"mblt")
    runtime = SimpleNamespace(get_available_device_numbers=lambda: [])
    inputs = tuple(
        np.ones(item.shape, dtype=np.float32) for item in contract.core_inputs
    )

    with pytest.raises(RuntimeError, match="device 0"):
        run_mblt(artifact, inputs, contract, qbruntime_module=runtime)
