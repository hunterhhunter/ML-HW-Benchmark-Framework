import hashlib
import json
from pathlib import Path
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
    artifact = compile_stage(
        stage,
        attempt_root,
        model_loader=lambda: _TinyResNet().eval(),
        official_transform=lambda image: torch.full((3, 224, 224), 0.25),
        mblt_compiler=fake_mblt,
        mxq_compiler_api=api,
    )

    assert artifact.is_file() and artifact.stat().st_size > 0
    report = json.loads((attempt_root / "compile-report.json").read_text())
    assert report["active_compiler_stage"] is None
    assert report["artifacts"][stage]["sha256"] == hashlib.sha256(artifact.read_bytes()).hexdigest()
    if stage == "mxq":
        assert report["resolved_mxq_preset"] == {
            "name": "classification_torchvision",
            "config_preset_argument": "classification_torchvision",
            "uint8_input_config": {
                "apply": True, "inputs": ["input_np"], "division_factor": 255.0
            },
        }


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
