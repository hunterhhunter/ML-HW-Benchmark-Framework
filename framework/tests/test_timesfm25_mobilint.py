from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from timesfm25.contracts import TimesFM25Contract
from tools.timesfm25_vendors.mobilint import compile_mxq, export_core_onnx, run_mxq


class _Core(torch.nn.Module):
    def forward(self, normalized_context):
        del normalized_context
        return torch.zeros((1, 128), dtype=torch.float32)


def test_mobilint_exports_one_static_timesfm_tensor(monkeypatch, tmp_path):
    captured = {}

    def fake_export(model, inputs, path, **kwargs):
        captured.update(inputs=inputs, **kwargs)
        Path(path).write_bytes(b"onnx")

    monkeypatch.setattr(torch.onnx, "export", fake_export)
    export_core_onnx(
        _Core(), (torch.ones((1, 1024), dtype=torch.float32),),
        TimesFM25Contract.fixed(), tmp_path / "core.onnx"
    )

    assert captured["dynamo"] is False
    assert captured["dynamic_axes"] is None
    assert captured["input_names"] == ["normalized_context"]
    assert captured["output_names"] == ["point_forecast"]


def test_mobilint_compiles_mxq_for_aries_with_explicit_calibration(tmp_path):
    onnx, calibration, artifact = tmp_path / "core.onnx", tmp_path / "calibration", tmp_path / "core.mxq"
    onnx.write_bytes(b"onnx")
    calibration.mkdir()
    np.save(calibration / "calibration-000.npy", np.zeros((1, 1024), dtype=np.float32))
    captured = {}

    def compile_v2(model, **kwargs):
        captured.update(model=model, **kwargs)
        Path(kwargs["save_path"]).write_bytes(b"mxq")

    report = compile_mxq(
        onnx, artifact, calibration, np.zeros((1, 1024), dtype=np.float32),
        qbcompiler_module=SimpleNamespace(mxq_compile_V2=compile_v2),
    )

    assert report["target_device"] == "aries-rb"
    assert captured["use_random_calib"] is False
    assert captured["feed_dict"]["normalized_context"].shape == (1, 1024)


def test_mobilint_runtime_reshapes_fixed_input_and_uses_int8_to_float(tmp_path):
    artifact = tmp_path / "core.mxq"
    artifact.write_bytes(b"mxq")
    captured = {}

    class _Scale:
        is_uniform, is_asymmetric, scale, scale_list = True, False, 10.0, []

    class _Model:
        def get_model_input_shape(self): return [(1, 32, 32)]
        def get_model_output_shape(self): return [(1, 1, 128)]
        def get_input_scale(self): return [_Scale()]
        def infer_to_float(self, inputs):
            captured["input"] = inputs[0]
            return [np.zeros((1, 1, 128), dtype=np.float32)]
        def dispose(self): captured["disposed"] = True

    runtime = SimpleNamespace(get_available_device_numbers=lambda: [0], load=lambda _: _Model())
    report = run_mxq(
        artifact, np.ones((1, 1024), dtype=np.float32), TimesFM25Contract.fixed(),
        qbruntime_module=runtime,
    )

    assert report.output.shape == (1, 128)
    assert captured["input"].shape == (1, 32, 32)
    assert captured["input"].dtype == np.int8
    assert captured["disposed"] is True
