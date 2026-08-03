import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.mobilint_compile_recipes.compiler import run_mblt_compile, run_mxq_compile
from tools.mobilint_compile_recipes.contracts import (
    contract_to_dict,
    get_recipe,
    select_even_indices,
    sha256_file,
)


def test_contract_modules_do_not_eagerly_import_qbcompiler():
    assert "qbcompiler" not in sys.modules


def test_resnet_recipe_preserves_existing_runtime_abi():
    recipe = get_recipe("resnet50", "default")

    assert recipe.target_device == "aries-rb"
    assert recipe.inference_scheme == "global8"
    assert [(x.name, x.shape, x.dtype) for x in recipe.runtime_inputs] == [
        ("input_np", (1, 224, 224, 3), "uint8")
    ]
    assert recipe.outputs[0].shape == (1, 1000)


def test_yolov5m_recipe_preserves_three_raw_heads():
    recipe = get_recipe("yolov5m", "default")

    assert [x.shape for x in recipe.outputs] == [
        (1, 20, 20, 255),
        (1, 40, 40, 255),
        (1, 80, 80, 255),
    ]


def test_patchtst_variants_share_external_contract():
    stock = contract_to_dict(get_recipe("patchtst-etth1", "stock"))
    compat = contract_to_dict(
        get_recipe("patchtst-etth1", "compat-static-patchifier")
    )

    for key in ("target_device", "inference_scheme", "runtime_inputs", "outputs"):
        assert stock[key] == compat[key]


def test_even_indices_are_deterministic_and_include_endpoints():
    assert select_even_indices(100, 4) == (0, 33, 66, 99)


def test_even_indices_reject_invalid_requested_count():
    with pytest.raises(ValueError, match="count"):
        select_even_indices(4, 5)


def test_sha256_file_returns_the_artifact_digest(tmp_path):
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"mobilint")

    assert sha256_file(artifact) == hashlib.sha256(b"mobilint").hexdigest()


def test_mblt_compile_uses_recipe_boundary_and_copies_feed_dict(tmp_path):
    captured = {}

    def fake_compile(**kwargs):
        captured.update(kwargs)
        Path(kwargs["mblt_save_path"]).write_bytes(b"mblt")

    feed_dict = {"input_np": object()}
    output = tmp_path / "resnet50.mblt"

    result = run_mblt_compile(
        recipe=get_recipe("resnet50", "default"),
        model="model",
        feed_dict=feed_dict,
        output=output,
        compiler=fake_compile,
    )

    assert result == output
    assert captured == {
        "model": "model",
        "mblt_save_path": str(output),
        "target_device": "aries-rb",
        "backend": "torch",
        "feed_dict": feed_dict,
        "cpu_offload": True,
    }
    assert captured["feed_dict"] is not feed_dict


class _FakeCalibrationConfig:
    class MaxPercentile:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _FakeUint8InputConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def test_mxq_compile_uses_exact_vision_recipe_options(tmp_path):
    captured = {}

    def fake_compile(**kwargs):
        captured.update(kwargs)
        Path(kwargs["save_path"]).write_bytes(b"mxq")

    calibration = tmp_path / "calibration.json"
    calibration.write_text("{}")
    feed_dict = {"input_np": object()}
    output = tmp_path / "yolov5m.mxq"
    api = SimpleNamespace(
        CalibrationConfig=_FakeCalibrationConfig,
        Uint8InputConfig=_FakeUint8InputConfig,
        mxq_compile=fake_compile,
    )

    result = run_mxq_compile(
        recipe=get_recipe("yolov5m", "default"),
        model="model",
        feed_dict=feed_dict,
        calibration_path=calibration,
        output=output,
        compiler_api=api,
    )

    assert result == output
    assert captured["model"] == "model"
    assert captured["target_device"] == "aries-rb"
    assert captured["save_path"] == str(output)
    assert captured["calib_data_path"] == str(calibration)
    assert captured["backend"] == "torch"
    assert captured["inference_scheme"] == "global8"
    assert captured["config_preset"] == "yolo_640"
    assert captured["yolo_decode_include"] is False
    assert captured["feed_dict"] == feed_dict
    assert captured["feed_dict"] is not feed_dict
    assert captured["uint8_input_config"].kwargs == {
        "apply": True,
        "inputs": ["input_np"],
        "division_factor": 255.0,
    }
    config = captured["calibration_config"]
    assert config.kwargs["method"] == 1
    assert config.kwargs["output"] == 0
    assert config.kwargs["mode"] == 1
    assert config.kwargs["max_percentile"].kwargs == {
        "percentile": 0.999,
        "topk_ratio": 0.01,
    }


def test_mxq_compile_omits_vision_only_options_for_patchtst(tmp_path):
    captured = {}

    def fake_compile(**kwargs):
        captured.update(kwargs)
        Path(kwargs["save_path"]).write_bytes(b"mxq")

    calibration = tmp_path / "calibration.json"
    calibration.write_text("{}")
    api = SimpleNamespace(
        CalibrationConfig=_FakeCalibrationConfig,
        Uint8InputConfig=_FakeUint8InputConfig,
        mxq_compile=fake_compile,
    )

    run_mxq_compile(
        recipe=get_recipe("patchtst-etth1", "stock"),
        model="model",
        feed_dict={},
        calibration_path=calibration,
        output=tmp_path / "patchtst.mxq",
        compiler_api=api,
    )

    assert captured["inference_scheme"] == "global8"
    assert "config_preset" not in captured
    assert "yolo_decode_include" not in captured
    assert "uint8_input_config" not in captured


@pytest.mark.parametrize(
    ("function", "output"),
    [
        ("mblt", "invalid.mxq"),
        ("mxq", "invalid.mblt"),
    ],
)
def test_compiler_helpers_require_the_matching_artifact_suffix(
    tmp_path, function, output
):
    recipe = get_recipe("resnet50", "default")
    if function == "mblt":
        call = lambda: run_mblt_compile(
            recipe=recipe,
            model="model",
            feed_dict={},
            output=tmp_path / output,
            compiler=lambda **_: None,
        )
    else:
        calibration = tmp_path / "calibration.json"
        calibration.write_text("{}")
        api = SimpleNamespace(
            CalibrationConfig=_FakeCalibrationConfig,
            Uint8InputConfig=_FakeUint8InputConfig,
            mxq_compile=lambda **_: None,
        )
        call = lambda: run_mxq_compile(
            recipe=recipe,
            model="model",
            feed_dict={},
            calibration_path=calibration,
            output=tmp_path / output,
            compiler_api=api,
        )

    with pytest.raises(ValueError, match="suffix"):
        call()


def test_compiler_helpers_refuse_existing_artifacts(tmp_path):
    output = tmp_path / "resnet50.mblt"
    output.write_bytes(b"prior artifact")

    with pytest.raises(FileExistsError, match="already exists"):
        run_mblt_compile(
            recipe=get_recipe("resnet50", "default"),
            model="model",
            feed_dict={},
            output=output,
            compiler=lambda **_: None,
        )


@pytest.mark.parametrize("kind", ["mblt", "mxq"])
def test_compiler_helpers_reject_zero_byte_artifacts(tmp_path, kind):
    recipe = get_recipe("resnet50", "default")
    output = tmp_path / f"resnet50.{kind}"
    if kind == "mblt":
        def empty_compile(**kwargs):
            Path(kwargs["mblt_save_path"]).touch()

        call = lambda: run_mblt_compile(
            recipe=recipe,
            model="model",
            feed_dict={},
            output=output,
            compiler=empty_compile,
        )
    else:
        calibration = tmp_path / "calibration.json"
        calibration.write_text("{}")

        def empty_compile(**kwargs):
            Path(kwargs["save_path"]).touch()

        call = lambda: run_mxq_compile(
            recipe=recipe,
            model="model",
            feed_dict={},
            calibration_path=calibration,
            output=output,
            compiler_api=SimpleNamespace(
                CalibrationConfig=_FakeCalibrationConfig,
                Uint8InputConfig=_FakeUint8InputConfig,
                mxq_compile=empty_compile,
            ),
        )

    with pytest.raises(RuntimeError, match="empty artifact"):
        call()
