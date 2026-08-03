import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from tools.mobilint_compile_recipes.resnet50 import (
    ResNet50SourceWrapper,
    compile_stage,
    prepare_calibration,
    preprocess_calibration_image,
    source_smoke,
    validate_compiler_input,
)


class _TinyResNet(torch.nn.Module):
    def forward(self, value):
        assert value.shape == (1, 3, 224, 224)
        return value.mean(dim=(2, 3)).sum(dim=1, keepdim=True).expand(-1, 1000)


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    Image.new("RGB", (320, 240), color).save(path)


def _prepare(tmp_path: Path):
    dataset = tmp_path / "imagenet-val"
    dataset.mkdir()
    for index in range(32):
        _write_image(dataset / f"{31 - index:03d}.png", (index, 20, 30))
    attempt_root = tmp_path / "attempt"
    attempt_root.mkdir()
    manifest = prepare_calibration(dataset, attempt_root)
    return dataset, attempt_root, manifest


def test_resnet_wrapper_normalizes_unit_nhwc_to_nchw():
    wrapper = ResNet50SourceWrapper(torch.nn.Identity())

    output = wrapper(torch.zeros((1, 224, 224, 3), dtype=torch.float32))

    assert output.shape == (1, 3, 224, 224)
    expected = torch.tensor([-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225])
    torch.testing.assert_close(output[0, :, 0, 0], expected)


def test_resnet_calibration_is_raw_rgb_uint8_nhwc():
    value = preprocess_calibration_image(
        Image.new("RGB", (320, 240), (10, 20, 30))
    )

    assert value.shape == (1, 224, 224, 3)
    assert value.dtype == np.uint8
    np.testing.assert_array_equal(value[0, 0, 0], np.array([10, 20, 30], dtype=np.uint8))


def test_resnet_calibration_uses_232_short_side_center_crop_and_rgb_nhwc():
    width, height = 300, 500
    pixels = np.empty((height, width, 3), dtype=np.uint8)
    pixels[..., 0] = np.arange(width, dtype=np.uint8)[None, :]
    pixels[..., 1] = np.arange(height, dtype=np.uint8)[:, None]
    pixels[..., 2] = (
        pixels[..., 0].astype(np.uint16) + pixels[..., 1].astype(np.uint16)
    ).astype(np.uint8)
    image = Image.fromarray(pixels, mode="RGB")

    actual = preprocess_calibration_image(image)
    expected = np.asarray(
        image.resize((232, 387), Image.Resampling.BILINEAR).crop((4, 81, 228, 305)),
        dtype=np.uint8,
    )

    assert actual.shape == (1, 224, 224, 3)
    np.testing.assert_array_equal(actual[0], expected)


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (torch.zeros((1, 224, 224, 3), dtype=torch.float64), "float32"),
        (torch.zeros((1, 3, 224, 224), dtype=torch.float32), "shape"),
        (torch.full((1, 224, 224, 3), float("nan")), "finite"),
        (torch.full((1, 224, 224, 3), 1.01), "unit range"),
    ],
)
def test_eager_preflight_rejects_invalid_compiler_input(value, message):
    with pytest.raises(ValueError, match=message):
        validate_compiler_input(value)


def test_resnet_wrapper_is_torchscript_trace_compatible_without_eager_branches():
    wrapper = ResNet50SourceWrapper(torch.nn.Identity())
    value = torch.full((1, 224, 224, 3), 0.25)

    traced = torch.jit.trace(wrapper, value)

    torch.testing.assert_close(traced(value), wrapper(value))


def test_recipe_import_and_describe_do_not_import_torch():
    framework_root = Path(__file__).resolve().parents[1]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(framework_root), str(framework_root / "src"))
    )
    program = """
import builtins
import sys

original_import = builtins.__import__
def blocked_import(name, *args, **kwargs):
    if name == 'torch' or name.startswith('torch.'):
        raise AssertionError('torch import is forbidden')
    return original_import(name, *args, **kwargs)

builtins.__import__ = blocked_import
from tools.mobilint_compile_recipes import resnet50
assert 'torch' not in sys.modules
assert resnet50.main(['--stage', 'describe']) == 0
assert 'torch' not in sys.modules
"""

    result = subprocess.run(
        [sys.executable, "-c", program],
        cwd=framework_root,
        env=environment,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["model"] == "resnet50"


def test_prepare_selects_sorted_endpoint_inclusive_images_and_records_hashes(tmp_path):
    dataset, attempt_root, manifest = _prepare(tmp_path)

    expected_paths = sorted(dataset.glob("*.png"))
    assert manifest["calibration_indices"] == list(range(32))
    assert [record["source_path"] for record in manifest["samples"]] == [
        str(path.resolve()) for path in expected_paths
    ]
    assert [record["source_sha256"] for record in manifest["samples"]] == [
        hashlib.sha256(path.read_bytes()).hexdigest() for path in expected_paths
    ]
    assert manifest["torchvision"]["weight_enum"] == "IMAGENET1K_V2"
    assert manifest["torchvision"]["weight_url"]
    assert manifest["runtime_input"] == {
        "name": "input_np", "shape": [1, 224, 224, 3], "dtype": "uint8"
    }
    assert manifest["compiler_input"] == {
        "name": "input_np", "shape": [1, 224, 224, 3], "dtype": "float32"
    }
    first = np.load(attempt_root / manifest["samples"][0]["calibration_path"], allow_pickle=False)
    assert first.shape == (1, 224, 224, 3)
    assert first.dtype == np.uint8
    assert [record["calibration_size_bytes"] for record in manifest["samples"]] == [
        (attempt_root / record["calibration_path"]).stat().st_size
        for record in manifest["samples"]
    ]
    assert json.loads((attempt_root / "source-manifest.json").read_text()) == manifest


def test_prepare_rejects_non_image_without_creating_recipe_outputs(tmp_path):
    dataset = tmp_path / "imagenet-val"
    dataset.mkdir()
    for index in range(32):
        _write_image(dataset / f"{index:03d}.png", (10, 20, 30))
    (dataset / "README.txt").write_text("not an image")
    attempt_root = tmp_path / "attempt"
    attempt_root.mkdir()

    with pytest.raises(ValueError, match="not a readable RGB image"):
        prepare_calibration(dataset, attempt_root)

    assert list(attempt_root.iterdir()) == []


def test_prepare_does_not_mutate_existing_recipe_outputs(tmp_path):
    dataset, attempt_root, manifest = _prepare(tmp_path)
    before = (attempt_root / "source-manifest.json").read_bytes()

    with pytest.raises(FileExistsError, match="already exists"):
        prepare_calibration(dataset, attempt_root)

    assert (attempt_root / "source-manifest.json").read_bytes() == before
    assert json.loads(before) == manifest


def test_source_smoke_checks_official_transform_equivalence_and_finite_logits(tmp_path):
    _, attempt_root, _ = _prepare(tmp_path)
    official_transform = lambda image: torch.full((3, 224, 224), 0.25)

    output = source_smoke(
        attempt_root,
        model_loader=lambda: _TinyResNet().eval(),
        official_transform=official_transform,
    )

    assert output.shape == (1, 1000)
    assert output.dtype == torch.float32
    assert torch.isfinite(output).all()
    report = json.loads((attempt_root / "compile-report.json").read_text())
    assert report["source_smoke"] == {
        "output_shape": [1, 1000],
        "output_dtype": "float32",
        "finite": True,
        "official_transform_equivalence": {"rtol": 1e-5, "atol": 1e-6},
    }


@pytest.mark.parametrize("stage", ["mblt", "mxq"])
def test_compile_stage_records_artifact_and_resnet_mxq_evidence(tmp_path, stage):
    _, attempt_root, _ = _prepare(tmp_path)

    def fake_mblt(**kwargs):
        Path(kwargs["mblt_save_path"]).write_bytes(b"mblt")

    def fake_mxq(**kwargs):
        Path(kwargs["save_path"]).write_bytes(b"mxq")

    api = SimpleNamespace(
        CalibrationConfig=type(
            "CalibrationConfig", (),
            {"MaxPercentile": lambda **kwargs: SimpleNamespace(**kwargs),
             "__init__": lambda self, **kwargs: self.__dict__.update(kwargs)},
        ),
        Uint8InputConfig=lambda **kwargs: SimpleNamespace(**kwargs),
        mxq_compile=fake_mxq,
    )
    class FakePreset:
        def model_dump(self, *, by_alias, exclude_none):
            assert by_alias is True
            assert exclude_none is True
            return {
                "compiler": {"passes": ["base", "classification"], "nested": {"x": 1}},
                "quantization": {"activation": {"scheme": "per_tensor"}},
            }

    artifact = compile_stage(
        stage,
        attempt_root,
        model_loader=lambda: _TinyResNet().eval(),
        official_transform=lambda image: torch.full((3, 224, 224), 0.25),
        mblt_compiler=fake_mblt,
        mxq_compiler_api=api,
        preset_loader=lambda name: FakePreset(),
    )

    assert artifact.is_file() and artifact.stat().st_size > 0
    report = json.loads((attempt_root / "compile-report.json").read_text())
    assert report["active_compiler_stage"] is None
    assert report["artifacts"][stage]["sha256"] == hashlib.sha256(artifact.read_bytes()).hexdigest()
    if stage == "mxq":
        assert report["resolved_mxq_preset"] == {
            "name": "classification_torchvision",
            "resolved": {
                "compiler": {
                    "passes": ["base", "classification"], "nested": {"x": 1}
                },
                "quantization": {"activation": {"scheme": "per_tensor"}},
            },
            "overrides": {
                "target_device": "aries-rb",
                "inference_scheme": "global8",
                "calibration_config": {
                    "method": 1, "output": 0, "mode": 1,
                    "max_percentile": {"percentile": 0.999, "topk_ratio": 0.01},
                },
                "uint8_input_config": {
                    "apply": True, "inputs": ["input_np"], "division_factor": 255.0
                },
            },
        }


def test_mxq_preset_dump_is_recorded_before_compiler_failure(tmp_path):
    _, attempt_root, _ = _prepare(tmp_path)
    expected_dump = {"inherited": {"from": "base"}, "backend": {"torch": True}}

    class FakePreset:
        def model_dump(self, *, by_alias, exclude_none):
            return expected_dump

    def failing_mxq(**kwargs):
        report = json.loads((attempt_root / "compile-report.json").read_text())
        assert report["active_compiler_stage"] == "mxq"
        assert report["resolved_mxq_preset"]["resolved"] == expected_dump
        raise RuntimeError("compiler failed")

    api = SimpleNamespace(
        CalibrationConfig=type(
            "CalibrationConfig", (),
            {"MaxPercentile": lambda **kwargs: SimpleNamespace(**kwargs),
             "__init__": lambda self, **kwargs: self.__dict__.update(kwargs)},
        ),
        Uint8InputConfig=lambda **kwargs: SimpleNamespace(**kwargs),
        mxq_compile=failing_mxq,
    )

    with pytest.raises(RuntimeError, match="compiler failed"):
        compile_stage(
            "mxq",
            attempt_root,
            model_loader=lambda: _TinyResNet().eval(),
            official_transform=lambda image: torch.full((3, 224, 224), 0.25),
            mxq_compiler_api=api,
            preset_loader=lambda name: FakePreset(),
        )

    report = json.loads((attempt_root / "compile-report.json").read_text())
    assert report["active_compiler_stage"] == "mxq"
    assert report["resolved_mxq_preset"]["resolved"] == expected_dump


def test_mblt_does_not_resolve_the_mxq_preset(tmp_path):
    _, attempt_root, _ = _prepare(tmp_path)

    def fake_mblt(**kwargs):
        Path(kwargs["mblt_save_path"]).write_bytes(b"mblt")

    compile_stage(
        "mblt",
        attempt_root,
        model_loader=lambda: _TinyResNet().eval(),
        official_transform=lambda image: torch.full((3, 224, 224), 0.25),
        mblt_compiler=fake_mblt,
        preset_loader=lambda name: (_ for _ in ()).throw(AssertionError("preset used")),
    )


def test_compile_stage_never_retries_a_failed_active_compiler_stage(tmp_path):
    _, attempt_root, _ = _prepare(tmp_path)
    report_path = attempt_root / "compile-report.json"
    report = json.loads(report_path.read_text())
    report["active_compiler_stage"] = "mxq"
    report_path.write_text(json.dumps(report))
    before = report_path.read_bytes()

    with pytest.raises(RuntimeError, match="fresh attempt root"):
        compile_stage(
            "mblt",
            attempt_root,
            model_loader=lambda: _TinyResNet().eval(),
            official_transform=lambda image: torch.full((3, 224, 224), 0.25),
            mblt_compiler=lambda **kwargs: None,
        )

    assert report_path.read_bytes() == before
