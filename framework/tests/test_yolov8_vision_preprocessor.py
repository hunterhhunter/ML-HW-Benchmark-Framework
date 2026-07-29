from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from preprocessor.yolo_vision_preprocessor import YoloVisionPreprocessor


def test_letterbox_returns_exact_chw_tensor_and_context():
    source = Image.fromarray(
        np.full((2, 4, 3), [255, 0, 0], dtype=np.uint8), "RGB"
    )
    processor = YoloVisionPreprocessor(target_hw=(8, 8))

    tensor, context = processor.preprocess_with_context(source)

    assert tensor.shape == (3, 8, 8)
    assert tensor.dtype == np.float32
    assert tensor.flags.c_contiguous
    assert context == {
        "original_height": 2,
        "original_width": 4,
        "input_height": 8,
        "input_width": 8,
        "scale": 2.0,
        "pad_x": 0.0,
        "pad_y": 2.0,
    }
    content = tensor[:, 2:6, :]
    expected_content = np.broadcast_to(
        np.array([1.0, 0.0, 0.0], dtype=np.float32)[:, None, None],
        content.shape,
    )
    np.testing.assert_allclose(content, expected_content)
    np.testing.assert_allclose(tensor[:, :2, :], 114.0 / 255.0)


def test_corrupt_cache_is_rebuilt(tmp_path):
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (4, 2), (1, 2, 3)).save(image_path)
    processor = YoloVisionPreprocessor(target_hw=(8, 8))
    cache = processor.get_cache_path(tmp_path / "cache", image_path.name)
    assert cache is not None
    Path(cache).parent.mkdir()
    Path(cache).write_bytes(b"not-an-npz")

    tensor, context = processor.load_or_preprocess_with_context(
        cache, str(image_path)
    )

    assert tensor.shape == (3, 8, 8)
    assert context["original_width"] == 4
    with np.load(cache, allow_pickle=False) as rebuilt:
        assert int(rebuilt["cache_version"]) == processor.CACHE_VERSION


def test_schema_incompatible_cache_is_rebuilt(tmp_path):
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (4, 2), (9, 8, 7)).save(image_path)
    processor = YoloVisionPreprocessor(target_hw=(8, 8))
    cache = processor.get_cache_path(tmp_path / "cache", image_path.name)
    assert cache is not None
    Path(cache).parent.mkdir()
    np.savez_compressed(
        cache,
        cache_version=processor.CACHE_VERSION - 1,
        input=np.zeros((3, 8, 8), dtype=np.float32),
    )

    tensor, context = processor.load_or_preprocess_with_context(
        cache, image_path
    )

    assert context["original_height"] == 2
    content = tensor[:, 2:6, :]
    expected_content = np.broadcast_to(
        np.array([9.0, 8.0, 7.0], dtype=np.float32)[:, None, None]
        / 255.0,
        content.shape,
    )
    np.testing.assert_allclose(content, expected_content)


def test_cache_path_separates_preprocessor_version_and_geometry(tmp_path):
    first = YoloVisionPreprocessor(target_hw=(8, 8))
    second = YoloVisionPreprocessor(target_hw=(16, 8))

    first_path = first.get_cache_path(tmp_path, "sample.jpg")
    second_path = second.get_cache_path(tmp_path, "sample.jpg")

    assert first_path != second_path
    assert f"v{first.CACHE_VERSION}" in Path(first_path).name


def test_non_nchw_layout_is_rejected():
    with pytest.raises(ValueError, match="NCHW"):
        YoloVisionPreprocessor(layout="NHWC")
