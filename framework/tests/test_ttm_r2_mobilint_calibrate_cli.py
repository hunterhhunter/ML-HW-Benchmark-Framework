import runpy
from pathlib import Path

import numpy as np


def test_r2_calibration_cli_defaults_to_256_train_samples():
    module = runpy.run_path("framework/tools/ttm_r2_mobilint_calibrate.py", run_name="not_main")
    args = module["build_parser"]().parse_args([
        "--model-path", "/model", "--dataset-path", "/data", "--output-dir", "/out",
    ])

    assert args.calibration_samples == 256
    assert args.prepare_only is False


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


def test_r2_compiler_calibration_uses_lowered_patch_abi_without_manifest(tmp_path):
    module = runpy.run_path("framework/tools/ttm_r2_mobilint_calibrate.py", run_name="not_main")
    source = tmp_path / "semantic-calibration"
    source.mkdir()
    np.save(source / "calibration-000.npy", np.arange(512, dtype=np.float32).reshape(1, 512, 1))
    (source / "calibration-manifest.json").write_text("{}", encoding="utf-8")
    target = tmp_path / "aries-abi-calibration"

    report = module["write_aries_abi_calibration"](source, target)

    files = sorted(target.iterdir())
    assert report["samples"] == 1
    assert [path.name for path in files] == ["calibration-000.npy"]
    assert np.load(files[0]).shape == (1, 8, 64)
