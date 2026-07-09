from pathlib import Path

import numpy as np
from PIL import Image

from core.model_spec import Model_Spec, Task
from dataloader import HailoImageClassificationLoader, create_dataloader


def _make_spec() -> Model_Spec:
    return Model_Spec(
        name="resnet50_hailo",
        task=Task.IMAGE_CLASSIFICATION,
        input_shapes={"input": (1, 224, 224, 3)},
        input_dtype={"input": "uint8"},
        output_shapes={"logits": (1, 1000)},
        model_paths={"hef": "model.hef"},
    )


def _make_dataset(tmp_path: Path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    label_path = tmp_path / "labels.txt"
    image = Image.new("RGB", (300, 300), color=(255, 128, 0))
    image.save(image_dir / "sample.jpg")
    label_path.write_text("sample.jpg 0\n", encoding="utf-8")
    return image_dir, label_path


def test_hailo_loader_defaults_to_raw_nhwc_uint8_runtime_contract(tmp_path):
    image_dir, label_path = _make_dataset(tmp_path)

    loader = create_dataloader(
        _make_spec(),
        backend="hailort",
        dataset_path=str(tmp_path),
        image_dir=str(image_dir),
        label_path=str(label_path),
        layout="NHWC",
    )

    assert isinstance(loader, HailoImageClassificationLoader)
    metadata = loader.get_metadata()
    assert metadata["hailo_input"] == {
        "preprocess_mode": "raw",
        "input_layout": "NHWC",
        "input_format_type": "uint8",
    }
    assert metadata["runtime_options"] == {
        "input_format_type": "uint8",
        "input_layout": "NHWC",
    }

    batch = loader.load_batch(1)
    tensor = batch[0]["input"]
    assert tensor.shape == (224, 224, 3)
    assert tensor.dtype == np.float32
    assert float(tensor.max()) == 255.0
    assert float(tensor.min()) >= 0.0


def test_hailo_loader_allows_explicit_normalized_float_input(tmp_path):
    image_dir, label_path = _make_dataset(tmp_path)

    loader = create_dataloader(
        _make_spec(),
        backend="hailort",
        dataset_path=str(tmp_path),
        image_dir=str(image_dir),
        label_path=str(label_path),
        layout="NCHW",
        image_preprocess_mode="normalized",
    )

    metadata = loader.get_metadata()
    assert metadata["hailo_input"] == {
        "preprocess_mode": "normalized",
        "input_layout": "NCHW",
        "input_format_type": "float32",
    }
    assert metadata["runtime_options"] == {
        "input_format_type": "float32",
        "input_layout": "NCHW",
    }

    batch = loader.load_batch(1)
    tensor = batch[0]["input"]
    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == np.float32
    assert float(tensor.min()) < 0.0
