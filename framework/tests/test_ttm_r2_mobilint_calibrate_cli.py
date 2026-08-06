import runpy
from pathlib import Path

import numpy as np


def test_r2_calibration_cli_defaults_to_256_train_samples():
    module = runpy.run_path("framework/tools/ttm_r2_mobilint_calibrate.py", run_name="not_main")
    args = module["build_parser"]().parse_args([
        "--model-path", "/model", "--dataset-path", "/data", "--output-dir", "/out",
    ])

    assert args.calibration_samples == 256


def test_r2_mxq_compile_uses_static_aries_configuration(tmp_path):
    module = runpy.run_path("framework/tools/ttm_r2_mobilint_calibrate.py", run_name="not_main")
    onnx = tmp_path / "core.onnx"
    onnx.write_bytes(b"onnx")
    calibration = tmp_path / "calibration"
    calibration.mkdir()
    output = tmp_path / "core.mxq"
    captured = {}

    class Compiler:
        @staticmethod
        def mxq_compile_V2(model, **kwargs):
            captured["model"] = model
            captured.update(kwargs)
            Path(kwargs["save_path"]).write_bytes(b"mxq")

    module["compile_mxq"](
        onnx, calibration, output, np.zeros((1, 512, 1), dtype=np.float32), qbcompiler_module=Compiler(),
    )

    assert captured["target_device"] == "aries-rb"
    assert captured["device"] == "cpu"
    assert captured["cpu_offload"] is False
    assert captured["use_random_calib"] is False
