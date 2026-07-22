import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from core.model_spec import Model_Spec, Task
from dataloader.mobilint_object_detection_loader import MobilintObjectDetectionLoader
from dataloader.mobilint_vision_profiles import (
    MOBILINT_RESNET50_IMAGENET1K_V2,
    MOBILINT_YOLOV5M_DEFAULT,
)
from preprocessor import MobilintYoloV5Preprocessor
from preprocessor.object_detection_preprocessor import ObjectDetectionPreprocessor


def model_zoo_letterbox_reference(rgb: np.ndarray):
    h0, w0 = rgb.shape[:2]
    ratio = min(640 / h0, 640 / w0)
    new_w, new_h = int(round(w0 * ratio)), int(round(h0 * ratio))
    resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    dw, dh = (640 - new_w) / 2, (640 - new_h) / 2
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    result = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )
    return result, ratio, left, top


def _detection_spec() -> Model_Spec:
    return Model_Spec(
        name="yolov5m",
        task=Task.OBJECT_DETECTION,
        input_shapes={"images": (1, 640, 640, 3)},
        input_dtype={"images": "uint8"},
        output_shapes={
            "stride32": (1, 20, 20, 255),
            "stride16": (1, 40, 40, 255),
            "stride8": (1, 80, 80, 255),
        },
    )


def _rgb_gradient(width: int, height: int) -> np.ndarray:
    x = np.arange(width, dtype=np.uint16)[None, :]
    y = np.arange(height, dtype=np.uint16)[:, None]
    return np.stack(
        (
            (x + 3 * y) % 256,
            (5 * x + 7 * y + 29) % 256,
            (11 * x + 13 * y + 43) % 256,
        ),
        axis=-1,
    ).astype(np.uint8)


def _make_loader(tmp_path: Path, *, width: int, height: int):
    image_dir = tmp_path / "images"
    label_dir = tmp_path / "labels"
    cache_dir = tmp_path / "cache"
    image_dir.mkdir()
    label_dir.mkdir()
    rgb = _rgb_gradient(width, height)
    Image.fromarray(rgb, mode="RGB").save(image_dir / "sample.png")
    (label_dir / "sample.txt").write_text("3 0.5 0.5 0.25 0.125\n")
    loader = MobilintObjectDetectionLoader(
        _detection_spec(),
        dataset_path=str(tmp_path),
        image_dir=str(image_dir),
        label_path=str(label_dir),
        cache_dir=str(cache_dir),
        mobilint_vision_profile=MOBILINT_YOLOV5M_DEFAULT,
    )
    return loader, rgb, cache_dir


def _cache_payload(cache_path: Path) -> dict[str, np.ndarray]:
    with np.load(cache_path, allow_pickle=False) as cached:
        return {name: np.array(cached[name], copy=True) for name in cached.files}


def test_mobilint_yolov5_loader_matches_model_zoo_pixels_context_and_cache(
    tmp_path: Path,
):
    loader, rgb, cache_dir = _make_loader(tmp_path, width=500, height=375)
    expected, ratio, left, top = model_zoo_letterbox_reference(rgb)

    sample = loader.load_single()

    assert sample["input"].shape == (640, 640, 3)
    assert sample["input"].dtype == np.uint8
    assert sample["input"].flags.c_contiguous
    np.testing.assert_array_equal(sample["input"], expected)
    assert sample["preprocess_context"] == {
        "original_width": 500,
        "original_height": 375,
        "input_width": 640,
        "input_height": 640,
        "scale": ratio,
        "pad_x": left,
        "pad_y": top,
        "layout": "NHWC",
        "resize_mode": "letterbox",
        "ratio_pad": ((1.28, 1.28), (0, 80)),
        "profile_id": MOBILINT_YOLOV5M_DEFAULT.profile_id,
    }

    cache_path = Path(loader.preprocessor.get_cache_path(str(cache_dir), "sample.png"))
    generic_path = ObjectDetectionPreprocessor(
        target_hw=(640, 640),
        layout="NHWC",
        preprocess_mode="raw",
        resize_mode="letterbox",
    ).get_cache_path(str(cache_dir), "sample.png")
    assert generic_path is not None
    assert Path(generic_path).name == "sample_letterbox_raw_NHWC_640x640.npz"
    assert cache_path != Path(generic_path)
    assert re.fullmatch(r"sample_mobilint_yolov5m_[0-9a-f]{10}\.npz", cache_path.name)

    cached_input, cached_context = loader.preprocessor.load_or_preprocess_with_context(
        str(cache_path), str(tmp_path / "does-not-exist.png")
    )
    assert cached_input.flags.c_contiguous
    np.testing.assert_array_equal(cached_input, expected)
    assert cached_context == sample["preprocess_context"]
    assert isinstance(cached_context["ratio_pad"], tuple)
    assert all(isinstance(item, tuple) for item in cached_context["ratio_pad"])


@pytest.mark.parametrize(
    "field",
    (
        "profile_id",
        "dtype",
        "shape",
        "layout",
        "input_width",
        "input_height",
        "original_width",
        "original_height",
        "scale",
        "pad_x",
        "pad_y",
        "resize_mode",
        "ratio_pad",
    ),
)
def test_mobilint_yolov5_cache_rejects_every_stale_contract_field(
    tmp_path: Path,
    field: str,
):
    loader, _, cache_dir = _make_loader(tmp_path, width=500, height=375)
    loader.load_single()
    cache_path = Path(
        loader.preprocessor.get_cache_path(str(cache_dir), "sample.png")
    )
    payload = _cache_payload(cache_path)
    corruptions = {
        "profile_id": "stale-profile",
        "layout": "NCHW",
        "input_width": 641,
        "input_height": 639,
        "original_width": 501,
        "original_height": 376,
        "scale": 1.27,
        "pad_x": 1,
        "pad_y": 81,
        "resize_mode": "stretch",
        "ratio_pad": np.array(((1.27, 1.28), (1, 80))),
    }
    if field == "dtype":
        payload["input"] = payload["input"].astype(np.float32)
    elif field == "shape":
        payload["input"] = payload["input"][..., :2]
    else:
        payload[field] = np.asarray(corruptions[field])
    np.savez_compressed(cache_path, **payload)

    with pytest.raises(ValueError, match="Mobilint YOLO cache"):
        loader.preprocessor.load_or_preprocess_with_context(
            str(cache_path),
            str(tmp_path / "does-not-exist.png"),
        )


def test_mobilint_yolov5_cache_rejects_missing_context_field(tmp_path: Path):
    loader, _, cache_dir = _make_loader(tmp_path, width=500, height=375)
    loader.load_single()
    cache_path = Path(
        loader.preprocessor.get_cache_path(str(cache_dir), "sample.png")
    )
    payload = _cache_payload(cache_path)
    payload.pop("ratio_pad")
    np.savez_compressed(cache_path, **payload)

    with pytest.raises(ValueError, match="Mobilint YOLO cache"):
        loader.preprocessor.load_or_preprocess_with_context(
            str(cache_path),
            str(tmp_path / "does-not-exist.png"),
        )


@pytest.mark.parametrize(
    ("width", "height"),
    (
        (500, 500),
        (374, 500),
    ),
    ids=("square", "portrait_odd_padding"),
)
def test_mobilint_yolov5_letterbox_matches_square_and_portrait_rounding(
    tmp_path: Path,
    width: int,
    height: int,
):
    loader, rgb, _ = _make_loader(tmp_path, width=width, height=height)
    expected, ratio, left, top = model_zoo_letterbox_reference(rgb)

    sample = loader.load_single()

    np.testing.assert_array_equal(sample["input"], expected)
    assert sample["preprocess_context"]["ratio_pad"] == (
        (ratio, ratio),
        (left, top),
    )
    if width < height:
        assert left == 80
        assert sample["input"].shape == (640, 640, 3)


def test_mobilint_detection_loader_exposes_runtime_contract(tmp_path: Path):
    loader, _, _ = _make_loader(tmp_path, width=32, height=24)

    metadata = loader.get_metadata()

    assert loader.backend == "mobilint"
    assert loader.preprocess_mode == "raw"
    assert loader.resize_mode == "letterbox"
    assert loader.layout == "NHWC"
    assert metadata["mobilint_vision_profile"] == MOBILINT_YOLOV5M_DEFAULT.profile_id
    assert metadata["runtime_options"] == MOBILINT_YOLOV5M_DEFAULT.runtime_contract()


def test_mobilint_yolov5_preprocessor_accepts_only_yolo_profile():
    with pytest.raises(ValueError, match="YOLO vision profile"):
        MobilintYoloV5Preprocessor(MOBILINT_RESNET50_IMAGENET1K_V2)


@pytest.mark.parametrize(
    ("options", "message"),
    (
        ({}, "requires a resolved vision profile"),
        (
            {"mobilint_vision_profile": MOBILINT_RESNET50_IMAGENET1K_V2},
            "received a non-detection profile",
        ),
    ),
)
def test_mobilint_detection_loader_rejects_invalid_profile_before_parent(
    options: dict,
    message: str,
):
    with pytest.raises(ValueError, match=message):
        MobilintObjectDetectionLoader(_detection_spec(), **options)
