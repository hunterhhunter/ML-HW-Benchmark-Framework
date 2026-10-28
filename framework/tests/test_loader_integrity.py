"""Reject incomplete samples at loader construction, before inference starts."""

import json

import numpy as np
import pytest
from PIL import Image

from core.model_spec import Model_Spec, Task
from dataloader.bert_classification_loader import BertClassificationLoader
from dataloader.bert_qa_loader import BertQALoader
from dataloader.deepx_vision_loader import DeepXObjectDetectionLoader
from dataloader.image_classification_loader import ImageClassificationLoader
from dataloader.llama_loader import LlamaLoader
from dataloader.object_detection_loader import ObjectDetectionLoader


def _spec(task):
    return Model_Spec(
        name="fixture", task=task,
        input_shapes={"input": (1, 3, 8, 8)},
        input_dtype={"input": "float32"},
        output_shapes={"output": (1, 10)},
    )


def test_classification_rejects_missing_label_file(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    Image.new("RGB", (8, 8)).save(images / "sample.jpg")
    with pytest.raises(FileNotFoundError, match="label"):
        ImageClassificationLoader(
            _spec(Task.IMAGE_CLASSIFICATION), image_dir=str(images),
            label_path=str(tmp_path / "missing.txt"),
        )


def test_classification_rejects_non_image_entry_and_unlabeled_image(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    Image.new("RGB", (8, 8)).save(images / "sample.jpg")
    (images / "README").write_text("not an image")
    labels = tmp_path / "labels.txt"
    labels.write_text("sample.jpg 0\n")
    with pytest.raises(ValueError, match="image|entry"):
        ImageClassificationLoader(
            _spec(Task.IMAGE_CLASSIFICATION), image_dir=str(images),
            label_path=str(labels),
        )

    (images / "README").unlink()
    Image.new("RGB", (8, 8)).save(images / "second.png")
    with pytest.raises(ValueError, match="label"):
        ImageClassificationLoader(
            _spec(Task.IMAGE_CLASSIFICATION), image_dir=str(images),
            label_path=str(labels),
        )


def test_classification_rejects_noninteger_json_label(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    Image.new("RGB", (8, 8)).save(images / "sample.jpg")
    labels = tmp_path / "labels.json"
    labels.write_text(json.dumps({"sample.jpg": "0"}), encoding="utf-8")
    with pytest.raises(ValueError, match="label"):
        ImageClassificationLoader(
            _spec(Task.IMAGE_CLASSIFICATION), image_dir=str(images),
            label_path=str(labels),
        )


def test_detection_requires_label_directory_but_allows_negative_image(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    Image.new("RGB", (8, 8)).save(images / "negative.jpg")
    labels = tmp_path / "labels"
    with pytest.raises(FileNotFoundError, match="label"):
        ObjectDetectionLoader(
            _spec(Task.OBJECT_DETECTION), image_dir=str(images),
            label_path=str(labels),
        )
    labels.mkdir()
    loader = ObjectDetectionLoader(
        _spec(Task.OBJECT_DETECTION), image_dir=str(images),
        label_path=str(labels),
    )
    assert loader.load_single()["label"].shape == (0, 5)


@pytest.mark.parametrize("label", ["class cx cy w h\n", "0 0.5 0.5\n", "1.5 0.5 0.5 0.2 0.2\n", "0 nan 0.5 0.2 0.2\n"])
def test_detection_rejects_malformed_nonempty_label(tmp_path, label):
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    Image.new("RGB", (8, 8)).save(images / "sample.jpg")
    (labels / "sample.txt").write_text(label)
    loader = ObjectDetectionLoader(
        _spec(Task.OBJECT_DETECTION), image_dir=str(images), label_path=str(labels),
    )
    with pytest.raises(ValueError, match="label"):
        loader.load_single()


def test_deepx_detection_rejects_missing_label_directory_and_malformed_label(tmp_path):
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    images.mkdir()
    Image.new("RGB", (8, 8)).save(images / "sample.jpg")
    kwargs = dict(image_dir=str(images), label_path=str(labels), target_hw=(8, 8))
    with pytest.raises(FileNotFoundError, match="label"):
        DeepXObjectDetectionLoader(_spec(Task.OBJECT_DETECTION), **kwargs)

    labels.mkdir()
    (labels / "sample.txt").write_text("class cx cy w h\n")
    loader = DeepXObjectDetectionLoader(_spec(Task.OBJECT_DETECTION), **kwargs)
    with pytest.raises(ValueError, match="label"):
        loader.load_single()


def test_bert_classification_rejects_misaligned_arrays(tmp_path):
    np.save(tmp_path / "input_ids.npy", np.ones((1, 4), dtype=np.int64))
    np.save(tmp_path / "attention_mask.npy", np.ones((1, 4), dtype=np.int64))
    np.save(tmp_path / "labels.npy", np.ones((2,), dtype=np.int64))
    with pytest.raises(ValueError, match="sample|shape|length"):
        BertClassificationLoader(_spec(Task.NLP_CLASSIFICATION), str(tmp_path))


def test_bert_qa_rejects_misaligned_arrays(tmp_path):
    for name in ("input_ids", "attention_mask", "token_type_ids"):
        np.save(tmp_path / f"{name}.npy", np.ones((1, 4), dtype=np.int64))
    np.save(tmp_path / "start_positions.npy", np.zeros((2,), dtype=np.int64))
    np.save(tmp_path / "end_positions.npy", np.zeros((2,), dtype=np.int64))
    with pytest.raises(ValueError, match="sample|shape|length"):
        BertQALoader(_spec(Task.QUESTION_ANSWERING), str(tmp_path))


def test_llama_requires_nonempty_squad_file(tmp_path):
    path = tmp_path / "missing.json"
    with pytest.raises(FileNotFoundError, match="SQuAD|missing"):
        LlamaLoader(_spec(Task.NLP_GENERATION), squad_json=str(path), preprocessor=object())

    path.write_text(json.dumps({"data": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="sample|question|QA"):
        LlamaLoader(_spec(Task.NLP_GENERATION), squad_json=str(path), preprocessor=object())
