import runpy
from pathlib import Path

import numpy as np


def test_calibration_cli_defaults_to_256_samples():
    """Catches a local calibration command that silently changes sample count."""
    module = runpy.run_path(
        "framework/tools/ttm_r1_mobilint_calibrate.py", run_name="not_main"
    )
    args = module["build_parser"]().parse_args(
        ["--model-path", "/model", "--dataset-path", "/data", "--output-dir", "/out"]
    )
    assert args.calibration_samples == 256


def test_mxq_compile_disables_random_calibration_and_cpu_offload(tmp_path):
    """Catches producing another random-calibrated or CPU-offloaded ARIES artifact."""
    module = runpy.run_path(
        "framework/tools/ttm_r1_mobilint_calibrate.py", run_name="not_main"
    )
    onnx = tmp_path / "core.onnx"
    onnx.write_bytes(b"onnx")
    calibration = tmp_path / "calibration"
    calibration.mkdir()
    output = tmp_path / "core.mxq"
    captured = {}

    class _QBC:
        @staticmethod
        def mxq_compile_V2(model, **kwargs):
            captured["model"] = model
            captured.update(kwargs)
            Path(kwargs["save_path"]).write_bytes(b"mxq")

    module["compile_mxq"](
        onnx,
        calibration,
        output,
        np.zeros((1, 512, 1), dtype=np.float32),
        qbcompiler_module=_QBC(),
    )

    assert captured["target_device"] == "aries-rb"
    assert captured["device"] == "cpu"
    assert captured["cpu_offload"] is False
    assert captured["use_random_calib"] is False
