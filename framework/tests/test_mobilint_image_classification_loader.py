import os
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from core.model_spec import Model_Spec, Task
from dataloader.mobilint_image_classification_loader import (
    MobilintImageClassificationLoader,
)
from dataloader.mobilint_vision_profiles import (
    MOBILINT_RESNET50_IMAGENET1K_V2,
    MOBILINT_YOLOV5M_DEFAULT,
)
from preprocessor import MobilintResNetCenterCropPreprocess
from preprocessor.image_preprocessor import ImagePreprocessor


def model_zoo_resnet_reference(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB")
    width, height = image.size
    if width < height:
        resized = (232, int(232 * height / width))
    else:
        resized = (int(232 * width / height), 232)
    image = image.resize(resized, Image.Resampling.BILINEAR)
    left = round((resized[0] - 224) / 2)
    top = round((resized[1] - 224) / 2)
    cropped = image.crop((left, top, left + 224, top + 224))
    return np.transpose(np.asarray(cropped, dtype=np.uint8), (2, 0, 1))


def _classification_spec() -> Model_Spec:
    return Model_Spec(
        name="resnet50",
        task=Task.IMAGE_CLASSIFICATION,
        input_shapes={"input": (1, 224, 224, 3)},
        input_dtype={"input": "uint8"},
        output_shapes={"output": (1, 1000)},
    )


def _gradient(kind: str, *, width: int, height: int) -> np.ndarray:
    x = np.arange(width, dtype=np.uint16)[None, :]
    y = np.arange(height, dtype=np.uint16)[:, None]
    if kind == "horizontal":
        red = np.broadcast_to(x % 256, (height, width))
        green = np.broadcast_to((3 * x + 17) % 256, (height, width))
        blue = np.broadcast_to((7 * x + 31) % 256, (height, width))
    elif kind == "vertical":
        red = np.broadcast_to(y % 256, (height, width))
        green = np.broadcast_to((5 * y + 19) % 256, (height, width))
        blue = np.broadcast_to((11 * y + 37) % 256, (height, width))
    else:
        red = (x + 3 * y) % 256
        green = (5 * x + 7 * y + 23) % 256
        blue = (11 * x + 13 * y + 41) % 256
    return np.stack((red, green, blue), axis=-1).astype(np.uint8)


@pytest.mark.parametrize(
    ("kind", "width", "height"),
    (
        ("horizontal", 320, 240),
        ("vertical", 240, 320),
        ("odd_offset", 301, 233),
    ),
)
def test_mobilint_resnet_loader_matches_model_zoo_pixels_and_cache(
    tmp_path: Path,
    kind: str,
    width: int,
    height: int,
):
    image_dir = tmp_path / "images"
    cache_dir = tmp_path / "cache"
    image_dir.mkdir()
    image_path = image_dir / "sample.png"
    Image.fromarray(_gradient(kind, width=width, height=height), mode="RGB").save(
        image_path
    )
    label_path = tmp_path / "labels.txt"
    label_path.write_text("sample.png 17\n")

    with Image.open(image_path) as source:
        expected_chw = model_zoo_resnet_reference(source)

    loader = MobilintImageClassificationLoader(
        _classification_spec(),
        dataset_path=str(tmp_path),
        image_dir=str(image_dir),
        label_path=str(label_path),
        cache_dir=str(cache_dir),
        mobilint_vision_profile=MOBILINT_RESNET50_IMAGENET1K_V2,
    )
    sample = loader.load_single()

    cache_paths = list(cache_dir.glob("sample_*.npy"))
    assert len(cache_paths) == 1
    cached_chw = np.load(cache_paths[0])
    assert cached_chw.dtype == np.uint8
    assert cached_chw.flags.c_contiguous
    np.testing.assert_array_equal(cached_chw, expected_chw)

    assert sample["input"].shape == (224, 224, 3)
    assert sample["input"].dtype == np.uint8
    assert sample["input"].flags.c_contiguous
    np.testing.assert_array_equal(sample["input"], np.transpose(expected_chw, (1, 2, 0)))


def test_mobilint_resnet_strategy_cache_config_contains_profile_and_full_recipe():
    profile = MOBILINT_RESNET50_IMAGENET1K_V2
    strategy = MobilintResNetCenterCropPreprocess(profile)

    assert strategy.cache_config() == {
        "profile_id": profile.profile_id,
        **asdict(profile.input_recipe),
    }


def test_mobilint_resnet_cache_key_changes_with_profile_id(tmp_path: Path):
    first_profile = MOBILINT_RESNET50_IMAGENET1K_V2
    second_profile = replace(first_profile, profile_id="mobilint-resnet50-cache-variant")
    first = ImagePreprocessor(
        target_hw=(224, 224),
        strategy=MobilintResNetCenterCropPreprocess(first_profile),
    )
    second = ImagePreprocessor(
        target_hw=(224, 224),
        strategy=MobilintResNetCenterCropPreprocess(second_profile),
    )

    assert first.get_cache_path(str(tmp_path), "sample.png") != second.get_cache_path(
        str(tmp_path), "sample.png"
    )


def test_mobilint_classification_loader_exposes_runtime_contract(tmp_path: Path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    Image.new("RGB", (224, 224), color=(1, 2, 3)).save(image_dir / "sample.png")
    label_path = tmp_path / "labels.txt"
    label_path.write_text("sample.png 1\n")

    loader = MobilintImageClassificationLoader(
        _classification_spec(),
        image_dir=str(image_dir),
        label_path=str(label_path),
        mobilint_vision_profile=MOBILINT_RESNET50_IMAGENET1K_V2,
    )

    metadata = loader.get_metadata()
    assert metadata["mobilint_vision_profile"] == (
        MOBILINT_RESNET50_IMAGENET1K_V2.profile_id
    )
    assert metadata["runtime_options"] == (
        MOBILINT_RESNET50_IMAGENET1K_V2.runtime_contract()
    )


@pytest.mark.parametrize(
    ("options", "message"),
    (
        ({}, "requires a resolved vision profile"),
        (
            {"mobilint_vision_profile": MOBILINT_YOLOV5M_DEFAULT},
            "received a non-classification profile",
        ),
    ),
)
def test_mobilint_classification_loader_rejects_invalid_profile_before_parent(
    options: dict,
    message: str,
):
    with pytest.raises(ValueError, match=message):
        MobilintImageClassificationLoader(_classification_spec(), **options)
