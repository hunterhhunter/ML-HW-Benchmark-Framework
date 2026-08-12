import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from tools.mobilint_compile_recipes.yolov5m import (
    EXPECTED_YOLOV5_REVISION,
    YoloV5RawHeadWrapper,
    compile_stage,
    load_source_model,
    prepare_calibration,
    preprocess_calibration_image,
    source_smoke,
    validate_compiler_input,
    validate_raw_source_output,
    validate_sources,
)


@pytest.fixture(autouse=True)
def _isolate_yolov5_module_namespaces():
    relevant = {
        name: module
        for name, module in sys.modules.items()
        if name in {"models", "utils"}
        or name.startswith("models.")
        or name.startswith("utils.")
    }
    for name in relevant:
        sys.modules.pop(name, None)
    try:
        yield
    finally:
        for name in tuple(sys.modules):
            if name in {"models", "utils"} or name.startswith(("models.", "utils.")):
                sys.modules.pop(name, None)
        sys.modules.update(relevant)


class FakeYolo(torch.nn.Module):
    def __init__(
        self,
        *,
        dtype=torch.float32,
        malformed=False,
        decoded_only=False,
        depth_multiple=0.67,
        width_multiple=0.75,
        yaml_file="yolov5m.yaml",
    ):
        super().__init__()
        self.output_dtype = dtype
        self.malformed = malformed
        self.decoded_only = decoded_only
        self.fused = False
        self.yaml = {
            "depth_multiple": depth_multiple,
            "width_multiple": width_multiple,
        }
        if yaml_file is not None:
            self.yaml_file = yaml_file
        self.detect = SimpleNamespace(
            anchors=torch.tensor(
                [
                    [[10 / 8, 13 / 8], [16 / 8, 30 / 8], [33 / 8, 23 / 8]],
                    [[30 / 16, 61 / 16], [62 / 16, 45 / 16], [59 / 16, 119 / 16]],
                    [[116 / 32, 90 / 32], [156 / 32, 198 / 32], [373 / 32, 326 / 32]],
                ],
                dtype=torch.float32,
            ),
            stride=torch.tensor([8.0, 16.0, 32.0]),
        )
        self.model = [torch.nn.Identity(), self.detect]

    def fuse(self):
        self.fused = True
        return self

    def forward(self, value):
        decoded = torch.zeros((1, 25200, 85), dtype=self.output_dtype)
        if self.decoded_only:
            return decoded
        heads = [
            torch.full((1, 3, 80, 80, 85), 8.0, dtype=self.output_dtype),
            torch.full((1, 3, 40, 40, 85), 16.0, dtype=self.output_dtype),
            torch.full(
                (1, 2 if self.malformed else 3, 20, 20, 85),
                32.0,
                dtype=self.output_dtype,
            ),
        ]
        return decoded, heads


def _write_image(path: Path, width=7, height=5, color=(10, 20, 30)) -> None:
    Image.new("RGB", (width, height), color).save(path)


def _source_tree(tmp_path: Path):
    root = tmp_path / "yolov5"
    (root / "models").mkdir(parents=True)
    (root / "models" / "experimental.py").write_text("# experimental\n")
    (root / "models" / "yolo.py").write_text("# yolo\n")
    weights = tmp_path / "yolov5m.pt"
    weights.write_bytes(b"pinned-yolov5m-weights")
    return root, weights


def _git_blob(value: bytes) -> str:
    header = f"blob {len(value)}\0".encode()
    return hashlib.sha1(header + value).hexdigest()


def _pin_revision(monkeypatch, root: Path, *, index_overrides=None):
    pinned_blobs = {
        name: _git_blob((root / name).read_bytes())
        for name in ("models/experimental.py", "models/yolo.py")
    }

    def fake_run(argv, **kwargs):
        command = argv[3:]
        if command == ["rev-parse", "HEAD"]:
            value = EXPECTED_YOLOV5_REVISION
        elif command[0:1] == ["rev-parse"] and command[1].startswith("HEAD:"):
            value = pinned_blobs[command[1].removeprefix("HEAD:")]
        elif command[0:1] == ["rev-parse"] and command[1].startswith(":"):
            name = command[1].removeprefix(":")
            value = (index_overrides or {}).get(name, pinned_blobs[name])
        elif command[0:1] == ["hash-object"]:
            value = _git_blob((root / command[1]).read_bytes())
        else:
            raise AssertionError(f"unexpected git command: {argv}")
        return SimpleNamespace(stdout=value + "\n")

    module = sys.modules["tools.mobilint_compile_recipes.yolov5m"]
    monkeypatch.setattr(module.subprocess, "run", fake_run)
    return pinned_blobs


def _lightweight_attempt(tmp_path: Path):
    source_root, weights = _source_tree(tmp_path)
    dataset = tmp_path / "coco128"
    dataset.mkdir()
    source_image = dataset / "000.jpg"
    _write_image(source_image)
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    calibration = attempt / "calibration"
    calibration.mkdir()
    array_path = calibration / "000.npy"
    np.save(
        array_path,
        np.full((1, 640, 640, 3), 114, dtype=np.uint8),
        allow_pickle=False,
    )
    calibration_json = calibration / "calibration.json"
    calibration_json.write_text(
        json.dumps(
            {
                "info": {"input names": ["input_np"]},
                "calib paths": [[str(array_path)]],
            }
        )
    )
    manifest = {
        "model": "yolov5m",
        "variant": "default",
        "source_id": "ultralytics/yolov5@" + EXPECTED_YOLOV5_REVISION,
        "yolov5": {
            "root": str(source_root.resolve()),
            "revision": EXPECTED_YOLOV5_REVISION,
            "required_files": {
                "models/experimental.py": {
                    "sha256": hashlib.sha256(b"# experimental\n").hexdigest(),
                    "git_blob": _git_blob(b"# experimental\n"),
                },
                "models/yolo.py": {
                    "sha256": hashlib.sha256(b"# yolo\n").hexdigest(),
                    "git_blob": _git_blob(b"# yolo\n"),
                },
            },
        },
        "weights": {
            "path": str(weights.resolve()),
            "size_bytes": weights.stat().st_size,
            "sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
        },
        "samples": [
            {
                "source_path": str(source_image.resolve()),
                "calibration_path": "calibration/000.npy",
            }
        ],
    }
    (attempt / "source-manifest.json").write_text(json.dumps(manifest))
    report = {
        "model": "yolov5m",
        "variant": "default",
        "source_smoke": None,
        "active_compiler_stage": None,
        "resolved_mxq_preset": None,
        "artifacts": {},
    }
    (attempt / "compile-report.json").write_text(json.dumps(report))
    return attempt


def _fake_compiler_api(callback):
    class CalibrationConfig:
        class MaxPercentile:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        def __init__(self, **kwargs):
            self.kwargs = kwargs

    return SimpleNamespace(
        CalibrationConfig=CalibrationConfig,
        Uint8InputConfig=lambda **kwargs: SimpleNamespace(**kwargs),
        mxq_compile=callback,
    )


def _record_source_smoke(attempt: Path) -> None:
    report_path = attempt / "compile-report.json"
    report = json.loads(report_path.read_text())
    report["source_smoke"] = {"validated": True}
    report_path.write_text(json.dumps(report))


def test_validate_sources_rejects_wrong_revision(monkeypatch, tmp_path):
    root, weights = _source_tree(tmp_path)
    module = sys.modules["tools.mobilint_compile_recipes.yolov5m"]
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout="deadbeef\n"),
    )

    with pytest.raises(RuntimeError, match=EXPECTED_YOLOV5_REVISION):
        validate_sources(root, weights)


@pytest.mark.parametrize("required", ["models/experimental.py", "models/yolo.py"])
def test_validate_sources_rejects_missing_required_source(required, tmp_path):
    root, weights = _source_tree(tmp_path)
    (root / required).unlink()

    with pytest.raises(FileNotFoundError, match=required):
        validate_sources(root, weights)


def test_validate_sources_rejects_empty_weights_before_git(monkeypatch, tmp_path):
    root, weights = _source_tree(tmp_path)
    weights.write_bytes(b"")
    module = sys.modules["tools.mobilint_compile_recipes.yolov5m"]
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("git called")),
    )

    with pytest.raises(ValueError, match="weight file is empty"):
        validate_sources(root, weights)


def test_validate_sources_requires_exact_yolov5m_weight_basename(monkeypatch, tmp_path):
    root, weights = _source_tree(tmp_path)
    renamed = weights.with_name("renamed.pt")
    weights.rename(renamed)
    module = sys.modules["tools.mobilint_compile_recipes.yolov5m"]
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("git called")),
    )

    with pytest.raises(ValueError, match="basename.*yolov5m.pt"):
        validate_sources(root, renamed)


def test_prepare_rejects_dirty_tracked_source_before_creating_outputs(monkeypatch, tmp_path):
    root, weights = _source_tree(tmp_path)
    _pin_revision(monkeypatch, root)
    (root / "models" / "yolo.py").write_text("# locally modified yolo\n")
    dataset = tmp_path / "coco128-images"
    dataset.mkdir()
    attempt = tmp_path / "attempt"
    attempt.mkdir()

    with pytest.raises(ValueError, match="differs from pinned HEAD.*models/yolo.py"):
        prepare_calibration(dataset, attempt, root, weights)

    assert list(attempt.iterdir()) == []


def test_validate_sources_rejects_staged_required_source_change(monkeypatch, tmp_path):
    root, weights = _source_tree(tmp_path)
    _pin_revision(
        monkeypatch,
        root,
        index_overrides={"models/yolo.py": "1" * 40},
    )

    with pytest.raises(ValueError, match="staged source differs.*models/yolo.py"):
        validate_sources(root, weights)


def test_load_source_model_uses_attempt_load_cpu_then_fuse_and_eval(monkeypatch, tmp_path):
    root, weights = _source_tree(tmp_path)
    _pin_revision(monkeypatch, root)
    observed = {}
    model = FakeYolo().train()

    def attempt_load(path, *, map_location):
        observed.update(path=path, map_location=map_location)
        return model

    loaded = load_source_model(root, weights, attempt_loader=attempt_load)

    assert loaded is model
    assert observed == {"path": str(weights.resolve()), "map_location": "cpu"}
    assert loaded.fused is True
    assert loaded.training is False


@pytest.mark.parametrize(
    "model",
    [
        FakeYolo(depth_multiple=1.0, width_multiple=1.0),
        FakeYolo(yaml_file="yolov5l.yaml"),
    ],
    ids=("wrong_multipliers", "wrong_yaml_identity"),
)
def test_load_source_model_rejects_non_yolov5m_architecture(monkeypatch, tmp_path, model):
    root, weights = _source_tree(tmp_path)
    _pin_revision(monkeypatch, root)

    with pytest.raises(ValueError, match="YOLOv5m architecture"):
        load_source_model(root, weights, attempt_loader=lambda *args, **kwargs: model)


@pytest.mark.parametrize("cached_name", ["models.yolo", "utils.general"])
def test_default_source_load_requires_clean_yolo_module_namespaces(
    monkeypatch, tmp_path, cached_name
):
    root, weights = _source_tree(tmp_path)
    _pin_revision(monkeypatch, root)
    cached = ModuleType(cached_name)
    cached.__file__ = str(tmp_path / "other-checkout" / "cached.py")
    monkeypatch.setitem(sys.modules, cached_name, cached)

    with pytest.raises(RuntimeError, match="fresh process.*namespace"):
        load_source_model(root, weights)


def test_default_source_load_rejects_modules_imported_outside_pinned_checkout(
    monkeypatch, tmp_path
):
    root, weights = _source_tree(tmp_path)
    _pin_revision(monkeypatch, root)
    module = sys.modules["tools.mobilint_compile_recipes.yolov5m"]

    def fake_import(name):
        assert name == "models.experimental"
        models_package = ModuleType("models")
        models_package.__file__ = str(root / "models" / "__init__.py")
        experimental = ModuleType("models.experimental")
        experimental.__file__ = str(root / "models" / "experimental.py")
        experimental.attempt_load = lambda *args, **kwargs: FakeYolo()
        rogue_utils = ModuleType("utils.general")
        rogue_utils.__file__ = str(tmp_path / "other-checkout" / "utils" / "general.py")
        monkeypatch.setitem(sys.modules, "models", models_package)
        monkeypatch.setitem(sys.modules, "models.experimental", experimental)
        monkeypatch.setitem(sys.modules, "utils.general", rogue_utils)
        return experimental

    monkeypatch.setattr(module.importlib, "import_module", fake_import)

    with pytest.raises(RuntimeError, match="outside the pinned checkout.*utils.general"):
        load_source_model(root, weights)


def test_yolo_wrapper_reverses_pinned_source_heads_to_nhwc255_output_order():
    outputs = YoloV5RawHeadWrapper(FakeYolo())(
        torch.zeros((1, 640, 640, 3), dtype=torch.float32)
    )

    assert [tuple(value.shape) for value in outputs] == [
        (1, 20, 20, 255),
        (1, 40, 40, 255),
        (1, 80, 80, 255),
    ]
    assert [float(value[0, 0, 0, 0]) for value in outputs] == [32.0, 16.0, 8.0]
    assert all(value.dtype == torch.float32 for value in outputs)


def test_yolo_wrapper_forward_is_strict_torch_export_graph():
    wrapper = YoloV5RawHeadWrapper(FakeYolo())
    value = torch.zeros((1, 640, 640, 3), dtype=torch.float32)

    exported = torch.export.export(wrapper, (value,), strict=True)
    outputs = exported.module()(value)

    assert [tuple(output.shape) for output in outputs] == [
        (1, 20, 20, 255),
        (1, 40, 40, 255),
        (1, 80, 80, 255),
    ]
    assert "isfinite" not in exported.graph_module.code


def test_eager_raw_source_validation_rejects_decoded_only_output():
    source_output = FakeYolo(decoded_only=True)(
        torch.zeros((1, 3, 640, 640), dtype=torch.float32)
    )

    with pytest.raises(ValueError, match="undecoded raw heads"):
        validate_raw_source_output(source_output)


@pytest.mark.parametrize(
    ("model", "message"),
    [
        (FakeYolo(malformed=True), "raw head shape"),
        (FakeYolo(dtype=torch.float64), "float32"),
    ],
)
def test_eager_raw_source_validation_rejects_invalid_heads(model, message):
    source_output = model(torch.zeros((1, 3, 640, 640), dtype=torch.float32))

    with pytest.raises(ValueError, match=message):
        validate_raw_source_output(source_output)


def test_eager_raw_source_validation_rejects_nonfinite_heads():
    class NonFiniteYolo(FakeYolo):
        def forward(self, value):
            decoded, heads = super().forward(value)
            heads[0][0, 0, 0, 0, 0] = float("nan")
            return decoded, heads

    source_output = NonFiniteYolo()(
        torch.zeros((1, 3, 640, 640), dtype=torch.float32)
    )

    with pytest.raises(ValueError, match="finite"):
        validate_raw_source_output(source_output)


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (torch.zeros((1, 640, 640, 3), dtype=torch.float64), "float32"),
        (torch.zeros((1, 3, 640, 640), dtype=torch.float32), "shape"),
        (torch.full((1, 640, 640, 3), float("nan")), "finite"),
        (torch.full((1, 640, 640, 3), 1.01), "unit range"),
    ],
)
def test_eager_preflight_rejects_invalid_compiler_input(value, message):
    with pytest.raises(ValueError, match=message):
        validate_compiler_input(value)


def test_preprocess_non_square_image_matches_mobilint_yolo_rgb_letterbox():
    width, height = 5, 9
    pixels = np.zeros((height, width, 3), dtype=np.uint8)
    pixels[..., 0] = 11
    pixels[..., 1] = 22
    pixels[..., 2] = 33

    actual = preprocess_calibration_image(Image.fromarray(pixels, mode="RGB"))

    assert actual.shape == (1, 640, 640, 3)
    assert actual.dtype == np.uint8
    assert actual.flags.c_contiguous
    np.testing.assert_array_equal(actual[0, 0, 0], [114, 114, 114])
    np.testing.assert_array_equal(actual[0, 320, 320], [11, 22, 33])


def test_prepare_selects_32_sorted_endpoint_inclusive_coco_images_and_records_hashes(
    monkeypatch, tmp_path
):
    root, weights = _source_tree(tmp_path)
    pinned_blobs = _pin_revision(monkeypatch, root)
    dataset = tmp_path / "coco128-images"
    dataset.mkdir()
    for index in range(64):
        _write_image(dataset / f"{63 - index:03d}.png", color=(index, 20, 30))
    attempt = tmp_path / "attempt"
    attempt.mkdir()

    manifest = prepare_calibration(dataset, attempt, root, weights)

    expected_indices = [index * 63 // 31 for index in range(32)]
    expected_sources = [sorted(dataset.glob("*.png"))[index] for index in expected_indices]
    assert manifest["calibration_indices"] == expected_indices
    assert [sample["source_path"] for sample in manifest["samples"]] == [
        str(path.resolve()) for path in expected_sources
    ]
    assert [sample["source_sha256"] for sample in manifest["samples"]] == [
        hashlib.sha256(path.read_bytes()).hexdigest() for path in expected_sources
    ]
    assert manifest["yolov5"]["revision"] == EXPECTED_YOLOV5_REVISION
    assert manifest["yolov5"]["required_files"] == {
        name: {
            "sha256": hashlib.sha256((root / name).read_bytes()).hexdigest(),
            "git_blob": pinned_blobs[name],
        }
        for name in ("models/experimental.py", "models/yolo.py")
    }
    assert manifest["weights"] == {
        "path": str(weights.resolve()),
        "size_bytes": weights.stat().st_size,
        "sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
    }
    report = json.loads((attempt / "compile-report.json").read_text())
    assert report["compiler_options"]["mxq"] == {
        "target_device": "aries-rb",
        "backend": "torch",
        "inference_scheme": "global8",
        "config_preset": "yolo_640",
        "yolo_decode_include": False,
        "uint8_input_config": {
            "apply": True,
            "inputs": ["input_np"],
            "division_factor": 255.0,
        },
        "calibration": {
            "method": 1,
            "output": 0,
            "mode": 1,
            "max_percentile": 0.999,
            "topk_ratio": 0.01,
        },
    }
    first = np.load(attempt / manifest["samples"][0]["calibration_path"], allow_pickle=False)
    assert first.shape == (1, 640, 640, 3)
    assert first.dtype == np.uint8
    assert [record["calibration_size_bytes"] for record in manifest["samples"]] == [
        (attempt / record["calibration_path"]).stat().st_size
        for record in manifest["samples"]
    ]
    assert manifest["samples"][0]["calibration_sha256"] == hashlib.sha256(
        (attempt / manifest["samples"][0]["calibration_path"]).read_bytes()
    ).hexdigest()
    assert json.loads((attempt / "source-manifest.json").read_text()) == manifest


def test_prepare_rejects_non_image_without_creating_recipe_outputs(monkeypatch, tmp_path):
    root, weights = _source_tree(tmp_path)
    _pin_revision(monkeypatch, root)
    dataset = tmp_path / "coco128-images"
    dataset.mkdir()
    for index in range(32):
        _write_image(dataset / f"{index:03d}.png")
    (dataset / "README.txt").write_text("not an image")
    attempt = tmp_path / "attempt"
    attempt.mkdir()

    with pytest.raises(ValueError, match="not a readable RGB image"):
        prepare_calibration(dataset, attempt, root, weights)

    assert list(attempt.iterdir()) == []


def test_prepare_does_not_mutate_existing_recipe_outputs(monkeypatch, tmp_path):
    root, weights = _source_tree(tmp_path)
    _pin_revision(monkeypatch, root)
    dataset = tmp_path / "coco128-images"
    dataset.mkdir()
    for index in range(32):
        _write_image(dataset / f"{index:03d}.png")
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    (attempt / "source-manifest.json").write_bytes(b"existing")

    with pytest.raises(FileExistsError, match="already exists"):
        prepare_calibration(dataset, attempt, root, weights)

    assert (attempt / "source-manifest.json").read_bytes() == b"existing"
    assert list(attempt.iterdir()) == [attempt / "source-manifest.json"]


def test_source_smoke_records_keyed_head_metadata_in_emitted_output_order(tmp_path):
    attempt = _lightweight_attempt(tmp_path)

    outputs = source_smoke(attempt, model_loader=lambda: FakeYolo().eval())

    assert [tuple(value.shape) for value in outputs] == [
        (1, 20, 20, 255),
        (1, 40, 40, 255),
        (1, 80, 80, 255),
    ]
    report = json.loads((attempt / "compile-report.json").read_text())
    assert report["source_smoke"] == {
        "output_shapes": [[1, 20, 20, 255], [1, 40, 40, 255], [1, 80, 80, 255]],
        "output_dtypes": ["float32", "float32", "float32"],
        "finite": True,
        "heads": [
            {
                "output_name": "mobilint_yolov5_stride32",
                "spatial_hw": [20, 20],
                "stride": 32.0,
                "pixel_anchors": [[116.0, 90.0], [156.0, 198.0], [373.0, 326.0]],
            },
            {
                "output_name": "mobilint_yolov5_stride16",
                "spatial_hw": [40, 40],
                "stride": 16.0,
                "pixel_anchors": [[30.0, 61.0], [62.0, 45.0], [59.0, 119.0]],
            },
            {
                "output_name": "mobilint_yolov5_stride8",
                "spatial_hw": [80, 80],
                "stride": 8.0,
                "pixel_anchors": [[10.0, 13.0], [16.0, 30.0], [33.0, 23.0]],
            },
        ],
    }


def test_source_smoke_rejects_weight_drift_from_prepared_manifest(monkeypatch, tmp_path):
    attempt = _lightweight_attempt(tmp_path)
    manifest = json.loads((attempt / "source-manifest.json").read_text())
    Path(manifest["weights"]["path"]).write_bytes(b"mutated-after-prepare")
    module = sys.modules["tools.mobilint_compile_recipes.yolov5m"]
    monkeypatch.setattr(
        module,
        "load_source_model",
        lambda *args, **kwargs: FakeYolo().eval(),
    )

    with pytest.raises(ValueError, match="weight SHA256 mismatch"):
        source_smoke(attempt)

    report = json.loads((attempt / "compile-report.json").read_text())
    assert report["source_smoke"] is None


def test_source_smoke_rejects_git_blob_drift_in_prepared_manifest(monkeypatch, tmp_path):
    attempt = _lightweight_attempt(tmp_path)
    manifest_path = attempt / "source-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["yolov5"]["required_files"]["models/yolo.py"]["git_blob"] = "0" * 40
    manifest_path.write_text(json.dumps(manifest))
    module = sys.modules["tools.mobilint_compile_recipes.yolov5m"]
    monkeypatch.setattr(
        module,
        "load_source_model",
        lambda *args, **kwargs: FakeYolo().eval(),
    )

    with pytest.raises(ValueError, match="Git blob mismatch.*models/yolo.py"):
        source_smoke(attempt)


def test_compile_stage_requires_prior_source_smoke_without_loading_model(tmp_path):
    attempt = _lightweight_attempt(tmp_path)
    report_path = attempt / "compile-report.json"
    before = report_path.read_bytes()

    with pytest.raises(RuntimeError, match="source-smoke stage.*before compiler"):
        compile_stage(
            "mblt",
            attempt,
            model_loader=lambda: (_ for _ in ()).throw(AssertionError("model loaded")),
            mblt_compiler=lambda **kwargs: None,
        )

    assert report_path.read_bytes() == before


@pytest.mark.parametrize("stage", ["mblt", "mxq"])
def test_compile_stage_records_artifact_and_exact_compiler_evidence(tmp_path, stage):
    attempt = _lightweight_attempt(tmp_path)
    _record_source_smoke(attempt)
    observed = {}

    def fake_mblt(**kwargs):
        observed.update(kwargs)
        Path(kwargs["mblt_save_path"]).write_bytes(b"mblt")

    def fake_mxq(**kwargs):
        observed.update(kwargs)
        Path(kwargs["save_path"]).write_bytes(b"mxq")

    class FakePreset:
        def model_dump(self, *, by_alias, exclude_none):
            assert by_alias is True and exclude_none is True
            return {
                "compiler": {"passes": ["base", "yolo"], "nested": {"x": 1}},
                "quantization": {"activation": {"scheme": "per_tensor"}},
            }

    artifact = compile_stage(
        stage,
        attempt,
        model_loader=lambda: FakeYolo().eval(),
        mblt_compiler=fake_mblt,
        mxq_compiler_api=_fake_compiler_api(fake_mxq),
        preset_loader=lambda name: FakePreset(),
    )

    assert artifact.is_file() and artifact.stat().st_size > 0
    report = json.loads((attempt / "compile-report.json").read_text())
    assert report["active_compiler_stage"] is None
    assert report["artifacts"][stage]["sha256"] == hashlib.sha256(
        artifact.read_bytes()
    ).hexdigest()
    if stage == "mxq":
        assert observed["target_device"] == "aries-rb"
        assert observed["inference_scheme"] == "global8"
        assert observed["config_preset"] == "yolo_640"
        assert observed["yolo_decode_include"] is False
        assert observed["uint8_input_config"].division_factor == 255.0
        assert report["resolved_mxq_preset"] == {
            "name": "yolo_640",
            "resolved": {
                "compiler": {"passes": ["base", "yolo"], "nested": {"x": 1}},
                "quantization": {"activation": {"scheme": "per_tensor"}},
            },
            "overrides": {
                "target_device": "aries-rb",
                "inference_scheme": "global8",
                "yolo_decode_include": False,
                "calibration_config": {
                    "method": 1,
                    "output": 0,
                    "mode": 1,
                    "max_percentile": {"percentile": 0.999, "topk_ratio": 0.01},
                },
                "uint8_input_config": {
                    "apply": True,
                    "inputs": ["input_np"],
                    "division_factor": 255.0,
                },
            },
        }


def test_mxq_preset_and_active_stage_survive_failure_and_block_cross_stage_retry(tmp_path):
    attempt = _lightweight_attempt(tmp_path)
    _record_source_smoke(attempt)
    expected_dump = {"inherited": {"from": "base"}, "backend": {"torch": True}}

    class FakePreset:
        def model_dump(self, *, by_alias, exclude_none):
            return expected_dump

    def failing_mxq(**kwargs):
        report = json.loads((attempt / "compile-report.json").read_text())
        assert report["active_compiler_stage"] == "mxq"
        assert report["resolved_mxq_preset"]["resolved"] == expected_dump
        raise RuntimeError("compiler failed")

    with pytest.raises(RuntimeError, match="compiler failed"):
        compile_stage(
            "mxq",
            attempt,
            model_loader=lambda: FakeYolo().eval(),
            mxq_compiler_api=_fake_compiler_api(failing_mxq),
            preset_loader=lambda name: FakePreset(),
        )

    report_path = attempt / "compile-report.json"
    before = report_path.read_bytes()
    with pytest.raises(RuntimeError, match="fresh attempt root"):
        compile_stage(
            "mblt",
            attempt,
            model_loader=lambda: FakeYolo().eval(),
            mblt_compiler=lambda **kwargs: None,
        )
    assert report_path.read_bytes() == before


def test_compile_stage_rejects_existing_output_without_mutating_report(tmp_path):
    attempt = _lightweight_attempt(tmp_path)
    output = attempt / "mblt" / "yolov5m-mblt.mblt"
    output.parent.mkdir()
    output.write_bytes(b"existing")
    report_path = attempt / "compile-report.json"
    before = report_path.read_bytes()

    with pytest.raises(FileExistsError, match="already exists"):
        compile_stage(
            "mblt",
            attempt,
            model_loader=lambda: FakeYolo().eval(),
            mblt_compiler=lambda **kwargs: None,
        )

    assert output.read_bytes() == b"existing"
    assert report_path.read_bytes() == before


def test_recipe_import_and_describe_are_lazy_without_torch_yolo_or_qbcompiler():
    framework_root = Path(__file__).resolve().parents[1]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(framework_root), str(framework_root / "src"))
    )
    program = r'''
import builtins
import json
import sys

original_import = builtins.__import__
def blocked_import(name, *args, **kwargs):
    if (name == "torch" or name.startswith("torch.") or
            name == "qbcompiler" or name.startswith("qbcompiler.") or
            name == "models" or name.startswith("models.")):
        raise AssertionError(f"forbidden eager import: {name}")
    return original_import(name, *args, **kwargs)

builtins.__import__ = blocked_import
from tools.mobilint_compile_recipes import yolov5m
assert yolov5m.main(["--stage", "describe"]) == 0
assert not any(name == "torch" or name.startswith("torch.") for name in sys.modules)
assert not any(name == "qbcompiler" or name.startswith("qbcompiler.") for name in sys.modules)
'''

    result = subprocess.run(
        [sys.executable, "-c", program],
        cwd=framework_root,
        env=environment,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["model"] == "yolov5m"
    assert payload["source_id"].endswith(EXPECTED_YOLOV5_REVISION)
    assert [output["shape"] for output in payload["outputs"]] == [
        [1, 20, 20, 255],
        [1, 40, 40, 255],
        [1, 80, 80, 255],
    ]
