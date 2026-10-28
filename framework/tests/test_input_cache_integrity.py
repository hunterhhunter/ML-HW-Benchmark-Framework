"""A cached input must belong to the exact source used by this sample."""

from pathlib import Path

import numpy as np
from PIL import Image

from core.model_spec import Model_Spec, Task
from dataloader.image_classification_loader import ImageClassificationLoader
from dataloader.object_detection_loader import ObjectDetectionLoader
from dataloader.deepx_vision_loader import DeepXObjectDetectionLoader
from preprocessor.ettm_preprocessor import ETTmPreprocessor
from preprocessor.yolo_vision_preprocessor import YoloVisionPreprocessor


def _spec(task):
    return Model_Spec(
        name="fixture", task=task,
        input_shapes={"input": (1, 3, 8, 8)},
        input_dtype={"input": "float32"},
        output_shapes={"output": (1, 10)},
    )


def test_classification_cache_separates_images_with_same_stem(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    Image.new("RGB", (256, 256), (255, 0, 0)).save(images / "sample.jpg")
    Image.new("RGB", (256, 256), (0, 0, 255)).save(images / "sample.png")
    labels = tmp_path / "labels.txt"
    labels.write_text("sample.jpg 0\nsample.png 1\n")
    loader = ImageClassificationLoader(
        _spec(Task.IMAGE_CLASSIFICATION), image_dir=str(images),
        label_path=str(labels), cache_dir=str(tmp_path / "cache"),
    )
    first, second = loader.load_batch(2)
    assert not np.array_equal(first["input"], second["input"])


def test_classification_cache_rebuilds_when_source_changes(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    source = images / "sample.jpg"
    Image.new("RGB", (256, 256), (255, 0, 0)).save(source)
    labels = tmp_path / "labels.txt"
    labels.write_text("sample.jpg 0\n")
    kwargs = dict(image_dir=str(images), label_path=str(labels), cache_dir=str(tmp_path / "cache"))
    first = ImageClassificationLoader(_spec(Task.IMAGE_CLASSIFICATION), **kwargs).load_single()["input"]
    Image.new("RGB", (256, 256), (0, 0, 255)).save(source)
    second = ImageClassificationLoader(_spec(Task.IMAGE_CLASSIFICATION), **kwargs).load_single()["input"]
    assert not np.array_equal(first, second)


def test_detection_cache_rebuilds_when_source_changes(tmp_path):
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    source = images / "sample.jpg"
    Image.new("RGB", (8, 8), (255, 0, 0)).save(source)
    kwargs = dict(image_dir=str(images), label_path=str(labels), cache_dir=str(tmp_path / "cache"), image_preprocess_mode="raw")
    first = ObjectDetectionLoader(_spec(Task.OBJECT_DETECTION), **kwargs).load_single()["input"]
    Image.new("RGB", (8, 8), (0, 0, 255)).save(source)
    second = ObjectDetectionLoader(_spec(Task.OBJECT_DETECTION), **kwargs).load_single()["input"]
    assert not np.array_equal(first, second)


def test_yolo_vision_cache_rebuilds_when_source_changes(tmp_path):
    source = tmp_path / "sample.jpg"
    processor = YoloVisionPreprocessor(target_hw=(8, 8))
    Image.new("RGB", (8, 8), (255, 0, 0)).save(source)
    cache = processor.get_cache_path(tmp_path / "cache", source.name)
    first, _ = processor.load_or_preprocess_with_context(cache, str(source))
    Image.new("RGB", (8, 8), (0, 0, 255)).save(source)
    second, _ = processor.load_or_preprocess_with_context(cache, str(source))
    assert not np.array_equal(first, second)


def test_ettm_cache_rebuilds_when_window_changes(tmp_path):
    processor = ETTmPreprocessor()
    cache = processor.get_cache_path(str(tmp_path), "val", 4, 2, 1, 0)
    first = processor.load_or_preprocess_window(cache, np.array([[1.], [2.], [3.], [4.]], dtype=np.float32))
    second = processor.load_or_preprocess_window(cache, np.array([[40.], [30.], [20.], [10.]], dtype=np.float32))
    assert first["norm_stats"]["mean"].item() != second["norm_stats"]["mean"].item()


def test_deepx_vision_cache_rebuilds_when_source_changes(tmp_path):
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    source = images / "sample.jpg"
    Image.new("RGB", (8, 8), (255, 0, 0)).save(source)
    loader = DeepXObjectDetectionLoader(
        _spec(Task.OBJECT_DETECTION), image_dir=str(images), label_path=str(labels),
        target_hw=(8, 8), cache_dir=str(tmp_path / "cache"),
    )
    first = loader.load_by_index(0)["input"]
    Image.new("RGB", (8, 8), (0, 0, 255)).save(source)
    second = loader.load_by_index(0)["input"]
    assert not np.array_equal(first, second)
