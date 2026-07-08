import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
from PIL import Image

from core.model_spec import Model_Spec, Task
from dataloader import create_dataloader
from dataloader.deepx_image_classification_loader import (
    DeepXDirectResizeRawPreprocess,
    DeepXImageClassificationLoader,
    deepx_rmap_image_input_layout,
    deepx_rmap_input_dtype,
    read_dxnn_rmap_input_info,
    resolve_deepx_image_input_config,
)


def _write_fake_dxnn(path: Path, rmap_info: dict) -> None:
    payload = json.dumps(rmap_info).encode("utf-8")
    header = {
        "size": 8192,
        "data": {
            "compiled_data": {
                "M1A_4K": {
                    "npu_0": {
                        "rmap_info": {
                            "type": "str",
                            "offset": 0,
                            "size": len(payload),
                        },
                    },
                },
            },
        },
    }
    header_bytes = json.dumps(header).encode("utf-8")
    assert len(header_bytes) < 8184
    path.write_bytes(b"DXNN" + (1).to_bytes(4, "little") + header_bytes + b"\0" * (8184 - len(header_bytes)) + payload)


def test_read_dxnn_rmap_input_info_reads_uint8_nhwc_image_input(tmp_path):
    artifact = tmp_path / "model.dxnn"
    _write_fake_dxnn(
        artifact,
        {
            "inputs": [
                {
                    "name": "input.1",
                    "dtype": "UINT8",
                    "shape": [1, 224, 224, 3],
                },
            ],
        },
    )

    info = read_dxnn_rmap_input_info(artifact)

    assert info["name"] == "input.1"
    assert deepx_rmap_input_dtype(info) == "UINT8"
    assert deepx_rmap_image_input_layout(info) == "NHWC"


def test_deepx_uint8_rmap_input_resolves_raw_dxapp_runtime_options(tmp_path):
    artifact = tmp_path / "model.dxnn"
    _write_fake_dxnn(
        artifact,
        {
            "inputs": [
                {
                    "name": "input.1",
                    "dtype": "UINT8",
                    "shape": [1, 224, 224, 3],
                },
            ],
        },
    )

    config = resolve_deepx_image_input_config(
        artifact_path=artifact,
        compile_options={},
        requested_mode="auto",
        compile_enabled=False,
    )

    assert config.preprocess_mode == "raw"
    assert config.expects_uint8_image is True
    assert config.runtime_options == {
        "input_layout": "NHWC",
        "input_dtype": "uint8",
        "input_batch_axis": "squeeze",
        "single_input_run_style": "list",
    }


def test_deepx_direct_resize_raw_preprocess_matches_dxapp_simple_resize_contract():
    strategy = DeepXDirectResizeRawPreprocess()
    img = Image.new("RGB", (320, 240), color=(10, 20, 30))

    tensor = strategy(
        img,
        target_hw=(224, 224),
        mean=np.array([0.485, 0.456, 0.406], dtype=np.float32),
        std=np.array([0.229, 0.224, 0.225], dtype=np.float32),
    )

    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == np.float32
    assert tensor[0, 0, 0] == 10.0
    assert tensor[1, 0, 0] == 20.0
    assert tensor[2, 0, 0] == 30.0


def test_create_dataloader_routes_deepx_image_classification_to_deepx_loader(tmp_path):
    artifact = tmp_path / "model.dxnn"
    _write_fake_dxnn(
        artifact,
        {
            "inputs": [
                {
                    "name": "input.1",
                    "dtype": "UINT8",
                    "shape": [1, 224, 224, 3],
                },
            ],
        },
    )
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    Image.new("RGB", (320, 240), color=(10, 20, 30)).save(image_dir / "sample.jpg")
    label_path = tmp_path / "labels.json"
    label_path.write_text(json.dumps({"sample.jpg": 7}), encoding="utf-8")
    spec = Model_Spec(
        name="resnet50",
        task=Task.IMAGE_CLASSIFICATION,
        input_shapes={"input.1": (1, 3, 224, 224)},
        input_dtype={"input.1": "float32"},
        output_shapes={"495": (1, 1000)},
        model_paths={},
    )

    loader = create_dataloader(
        spec,
        dataset_path=str(tmp_path),
        image_dir=str(image_dir),
        label_path=str(label_path),
        backend="deepx",
        artifact_path=str(artifact),
        compile_options={},
        compile_enabled=False,
        image_preprocess_mode="auto",
    )
    sample = loader.load_single()

    assert isinstance(loader, DeepXImageClassificationLoader)
    assert sample["input"].shape == (224, 224, 3)
    assert sample["input"].dtype == np.float32
    assert sample["label"] == 7
    assert loader.get_metadata()["runtime_options"] == {
        "input_layout": "NHWC",
        "input_dtype": "uint8",
        "input_batch_axis": "squeeze",
        "single_input_run_style": "list",
    }
