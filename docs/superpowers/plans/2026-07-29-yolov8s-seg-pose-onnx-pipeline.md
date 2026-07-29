# YOLOv8s Segmentation and Pose ONNX Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add complete framework-native `yolov8s-seg` and `yolov8s-pose` pipelines from official COCO input through ONNX Runtime inference, NumPy postprocessing, and official COCO accuracy evaluation on CPU and CUDA.

**Architecture:** Both tasks share exact Ultralytics-compatible letterbox preprocessing, lazy COCO image loading, validated NumPy decoder primitives, and a streaming COCO evaluator base. Task-specific decoders convert raw ONNX tensors to canonical NumPy payloads; task-specific evaluators restore original coordinates and delegate mask AP or OKS AP calculation to `pycocotools==2.0.11`.

**Tech Stack:** Python 3.12, NumPy, Pillow, OpenCV, ONNX, ONNX Runtime GPU 1.24.4, Ultralytics 8.4.24 for export only, pycocotools 2.0.11, pytest 9.

## Global Constraints

- Canonical model names are exactly `yolov8s-seg` and `yolov8s-pose`; existing ResNet50, YOLOv5m, Hailo, DeepX, and medium YOLOv8 profiles remain compatible.
- Runtime component boundaries contain only NumPy arrays and native Python metadata; Torch and Ultralytics are prohibited from production inference, postprocessing, and evaluator modules.
- Input is RGB, contiguous NCHW `float32`, shape `(3, 640, 640)` per sample, normalized to `[0, 1]`, with centered `(114, 114, 114)` letterbox padding.
- Accuracy uses official COCO 2017 `instances_val2017.json` mask metrics and `person_keypoints_val2017.json` 17-keypoint OKS metrics through `pycocotools==2.0.11`.
- Unit and integration tests use deterministic local fixtures and never download external assets.
- Every production behavior must have an observed failing test before implementation, then a focused passing run, then the affected suite.
- Real runtime verification must execute both exact ONNX models on active `CPUExecutionProvider` and active `CUDAExecutionProvider`; CUDA fallback is failure.
- Model weights, ONNX files, COCO archives, images, annotations, caches, and benchmark CSV outputs are not committed.
- Preserve unrelated dirty-worktree changes; every commit stages only files listed in its task.

---

## File Map

- `framework/src/core/model_profiles.py`: exact profiles and ordinal binding of every ONNX output name.
- `framework/models/prepare_yolov8_vision.py`: idempotent export of the two small YOLOv8 variants.
- `framework/datasets/prepare_coco_vision.py`: safe, validated preparation of official COCO validation assets.
- `framework/src/utils/dataset_resolver.py`: task-specific image and annotation conventions.
- `framework/src/preprocessor/yolo_vision_preprocessor.py`: shared letterbox, tensor formatting, and versioned atomic cache.
- `framework/src/dataloader/coco_vision_loader.py`: validated lazy COCO index and shared DataLoader mechanics.
- `framework/src/dataloader/coco_instance_segmentation_loader.py`: segmentation task guard and category contract.
- `framework/src/dataloader/coco_pose_loader.py`: pose task guard and person-keypoint contract.
- `framework/src/decoders/yolo_vision.py`: shape resolution, box conversion, and deterministic class-aware NumPy NMS.
- `framework/src/decoders/instance_segmentation.py`: YOLOv8 segmentation tensor and prototype decoding.
- `framework/src/decoders/pose_estimation.py`: YOLOv8 pose box and 17-keypoint decoding.
- `framework/src/evaluators/coco_common.py`: label/context validation, coordinate restoration, timing, and COCOeval execution.
- `framework/src/evaluators/instance_segmentation_evaluator.py`: mask restoration, RLE encoding, and mask metric naming.
- `framework/src/evaluators/pose_estimation_evaluator.py`: keypoint restoration and OKS metric naming.
- Existing package factories, CLI, benchmark registry, requirements, and README files: public integration only.
- `framework/tests/coco_test_utils.py`: deterministic valid COCO fixture builders shared by loader and evaluator tests.
- Focused `framework/tests/test_yolov8_*` modules: one responsibility per subsystem.

---

### Task 1: Exact Profiles and Multi-output ONNX Binding

**Files:**
- Modify: `framework/src/core/model_profiles.py`
- Modify: `framework/models/prepare_yolov8_vision.py`
- Create: `framework/tests/test_yolov8_model_profiles.py`

**Interfaces:**
- Consumes: `create_model_spec(model_name: str, onnx_path: str, task: Task, sniff_onnx: bool, source_format: str) -> Model_Spec`.
- Produces: `_parse_onnx_io_names(onnx_path: str) -> tuple[list[str], list[str]]`; exact small-model profiles whose `Model_Spec.output_shapes` keys match every ONNX graph output by ordinal.

- [ ] **Step 1: Write failing profile and ordinal-binding tests**

```python
# framework/tests/test_yolov8_model_profiles.py
from pathlib import Path

import onnx
from onnx import TensorProto, helper

from core.model_profiles import SUPPORTED_PROFILES, create_model_spec
from core.model_spec import Task


def _write_two_output_model(path: Path) -> None:
    graph = helper.make_graph(
        [
            helper.make_node("Identity", ["images"], ["predictions"]),
            helper.make_node("Identity", ["images"], ["prototypes"]),
        ],
        "two-output",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 640, 640])],
        [
            helper.make_tensor_value_info("predictions", TensorProto.FLOAT, [1, 116, 8400]),
            helper.make_tensor_value_info("prototypes", TensorProto.FLOAT, [1, 32, 160, 160]),
        ],
    )
    onnx.save(helper.make_model(graph), path)


def test_small_yolov8_profiles_have_exact_tasks_and_paths():
    seg = SUPPORTED_PROFILES["yolov8s-seg"]
    pose = SUPPORTED_PROFILES["yolov8s-pose"]
    assert seg["task"] is Task.INSTANCE_SEGMENTATION
    assert pose["task"] is Task.POSE_ESTIMATION
    assert seg["default_model_path"] == "models/yolov8s-seg/yolov8s-seg.onnx"
    assert pose["default_model_path"] == "models/yolov8s-pose/yolov8s-pose.onnx"


def test_segmentation_spec_binds_both_graph_output_names(tmp_path):
    model_path = tmp_path / "seg.onnx"
    _write_two_output_model(model_path)
    spec = create_model_spec("yolov8s-seg", str(model_path))
    assert spec.input_shapes == {"images": (1, 3, 640, 640)}
    assert spec.output_shapes == {
        "predictions": (1, 116, 8400),
        "prototypes": (1, 32, 160, 160),
    }


def test_export_registry_uses_exact_small_weights_and_output_counts():
    assert MODELS["yolov8s-seg"] == {
        "weights": "yolov8s-seg.pt", "output_dir": "yolov8s-seg",
        "onnx_name": "yolov8s-seg.onnx", "output_count": 2,
    }
    assert MODELS["yolov8s-pose"] == {
        "weights": "yolov8s-pose.pt", "output_dir": "yolov8s-pose",
        "onnx_name": "yolov8s-pose.onnx", "output_count": 1,
    }


def test_export_validation_rejects_invalid_or_wrong_output_count(tmp_path):
    invalid = tmp_path / "invalid.onnx"
    invalid.write_bytes(b"broken")
    assert _valid_onnx_export(invalid, expected_output_count=2) is False
    model_path = tmp_path / "seg.onnx"
    _write_two_output_model(model_path)
    assert _valid_onnx_export(model_path, expected_output_count=2) is True
    assert _valid_onnx_export(model_path, expected_output_count=1) is False
```

- [ ] **Step 2: Run the tests and confirm RED**

Run from `framework/`:

```bash
.venv/bin/python -m pytest tests/test_yolov8_model_profiles.py -v
```

Expected: the exact `yolov8s-*` keys are missing, and the two-output expectation cannot be satisfied by the existing single-output parser.

- [ ] **Step 3: Implement exact profiles and ordinal output binding**

```python
# framework/src/core/model_profiles.py
def _parse_onnx_io_names(onnx_path: str) -> tuple[list[str], list[str]]:
    model = onnx.load(onnx_path)
    return (
        [value.name for value in model.graph.input],
        [value.name for value in model.graph.output],
    )


def _bind_auto_output_shapes(profile_shapes, output_names):
    auto_items = [
        (key, shape)
        for key, shape in profile_shapes.items()
        if key == "__auto__" or key.startswith("__auto_")
    ]
    if not auto_items:
        return dict(profile_shapes)
    if len(auto_items) != len(output_names):
        raise ValueError(
            f"profile expects {len(auto_items)} ONNX outputs, graph exposes {len(output_names)}"
        )
    return {
        output_name: auto_items[index][1]
        for index, output_name in enumerate(output_names)
    }
```

Add profiles using `__auto_0__` and `__auto_1__` for segmentation and `__auto__` for pose. Update `create_model_spec()` to use the returned input/output name lists and `_bind_auto_output_shapes()`, retaining existing single-input and single-output behavior.

Update the exporter registry with exact entries:

```python
MODELS.update({
    "yolov8s-seg": {
        "weights": "yolov8s-seg.pt",
        "output_dir": "yolov8s-seg",
        "onnx_name": "yolov8s-seg.onnx",
        "output_count": 2,
    },
    "yolov8s-pose": {
        "weights": "yolov8s-pose.pt",
        "output_dir": "yolov8s-pose",
        "onnx_name": "yolov8s-pose.onnx",
        "output_count": 1,
    },
})


def _valid_onnx_export(path: Path, expected_output_count: int) -> bool:
    try:
        model = onnx.load(path)
        onnx.checker.check_model(model)
    except (OSError, ValueError, onnx.checker.ValidationError):
        return False
    return len(model.graph.input) == 1 and len(model.graph.output) == expected_output_count
```

Have `_export_model()` return early only when `_valid_onnx_export(final_onnx_path, info["output_count"])` is true. An invalid existing file is replaced by the newly exported model and the replacement is validated before success is printed.

- [ ] **Step 4: Verify GREEN and existing profile compatibility**

```bash
.venv/bin/python -m pytest tests/test_yolov8_model_profiles.py tests/test_deepx_dxnn_metadata.py tests/test_main_paths.py -q
```

Expected: all selected tests pass; medium DeepX profile construction remains unchanged.

- [ ] **Step 5: Commit only Task 1 files**

```bash
git add framework/src/core/model_profiles.py framework/models/prepare_yolov8_vision.py framework/tests/test_yolov8_model_profiles.py
git commit -m "feat: register yolov8s vision models"
```

---

### Task 2: Official COCO Asset Preparation and Dataset Resolution

**Files:**
- Create: `framework/datasets/prepare_coco_vision.py`
- Modify: `framework/src/utils/dataset_resolver.py`
- Create: `framework/tests/test_coco_vision_assets.py`

**Interfaces:**
- Produces: `prepare_coco_vision(dataset_root: Path, download: Callable = urlretrieve) -> None`; `resolve_dataset_paths(task: Task, dataset_path: str, image_dir_arg: str, label_dir_arg: str) -> tuple[str, str]` returns the official annotation JSON as its second value for segmentation and pose.
- Consumes: `Task.INSTANCE_SEGMENTATION`, `Task.POSE_ESTIMATION`, explicit CLI image/label overrides.

- [ ] **Step 1: Write failing safe-preparation and resolver tests**

```python
# framework/tests/test_coco_vision_assets.py
from pathlib import Path
from zipfile import ZipFile

import pytest

from core.model_spec import Task
from datasets.prepare_coco_vision import _safe_extract, validate_coco_vision_assets
from utils.dataset_resolver import resolve_dataset_paths


def test_resolver_selects_task_specific_official_annotations(tmp_path):
    image_dir = tmp_path / "images" / "val2017"
    ann_dir = tmp_path / "annotations"
    image_dir.mkdir(parents=True)
    ann_dir.mkdir()
    instances = ann_dir / "instances_val2017.json"
    keypoints = ann_dir / "person_keypoints_val2017.json"
    instances.write_text("{}")
    keypoints.write_text("{}")
    assert resolve_dataset_paths(Task.INSTANCE_SEGMENTATION, str(tmp_path), "", "") == (
        str(image_dir), str(instances)
    )
    assert resolve_dataset_paths(Task.POSE_ESTIMATION, str(tmp_path), "", "") == (
        str(image_dir), str(keypoints)
    )


def test_safe_extract_rejects_archive_path_escape(tmp_path):
    archive = tmp_path / "bad.zip"
    with ZipFile(archive, "w") as stream:
        stream.writestr("../escape.txt", "bad")
    with ZipFile(archive) as stream, pytest.raises(ValueError, match="outside"):
        _safe_extract(stream, tmp_path / "dataset")


def test_asset_validation_names_every_missing_path(tmp_path):
    with pytest.raises(FileNotFoundError, match="instances_val2017.json"):
        validate_coco_vision_assets(tmp_path)
```

- [ ] **Step 2: Run the tests and confirm RED**

```bash
.venv/bin/python -m pytest tests/test_coco_vision_assets.py -v
```

Expected: import failure for `datasets.prepare_coco_vision` and incorrect label-directory resolution for the two tasks.

- [ ] **Step 3: Implement safe preparation and exact conventions**

```python
# framework/datasets/prepare_coco_vision.py
COCO_DOWNLOADS = {
    "val2017.zip": "https://images.cocodataset.org/zips/val2017.zip",
    "annotations_trainval2017.zip": (
        "https://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    ),
}


def _safe_extract(archive: zipfile.ZipFile, destination: Path) -> None:
    root = destination.resolve()
    for member in archive.infolist():
        target = (destination / member.filename).resolve()
        if target != root and root not in target.parents:
            raise ValueError(f"archive member extracts outside dataset root: {member.filename}")
    archive.extractall(destination)


def validate_coco_vision_assets(dataset_root: Path) -> None:
    required = [
        dataset_root / "images" / "val2017",
        dataset_root / "annotations" / "instances_val2017.json",
        dataset_root / "annotations" / "person_keypoints_val2017.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("missing COCO vision assets: " + ", ".join(missing))


def prepare_coco_vision(
    dataset_root: Path,
    download: Callable[[str, str], object] = urllib.request.urlretrieve,
) -> None:
    dataset_root = Path(dataset_root)
    dataset_root.mkdir(parents=True, exist_ok=True)
    try:
        validate_coco_vision_assets(dataset_root)
        return
    except FileNotFoundError:
        pass
    downloads = [
        ("val2017.zip", COCO_DOWNLOADS["val2017.zip"], dataset_root / "images"),
        (
            "annotations_trainval2017.zip",
            COCO_DOWNLOADS["annotations_trainval2017.zip"],
            dataset_root,
        ),
    ]
    for filename, url, extract_root in downloads:
        archive_path = dataset_root / filename
        if not archive_path.exists():
            partial = archive_path.with_suffix(archive_path.suffix + ".part")
            download(url, str(partial))
            os.replace(partial, archive_path)
        extract_root.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive_path) as archive:
            _safe_extract(archive, extract_root)
    validate_coco_vision_assets(dataset_root)
```

Wrap the default CLI entrypoint around `prepare_coco_vision(Path("datasets/coco"))`. Delete a partial path on a caught download or extraction error, do not delete validated archives, and print success only after validation. In `dataset_resolver.py`, keep explicit arguments authoritative and select the two exact annotation filenames when no label override is supplied.

- [ ] **Step 4: Verify GREEN without network access**

```bash
.venv/bin/python -m pytest tests/test_coco_vision_assets.py tests/test_main_paths.py tests/test_object_detection_loader.py -q
```

Expected: all selected tests pass and object-detection COCO128 conventions remain intact.

- [ ] **Step 5: Commit only Task 2 files**

```bash
git add framework/datasets/prepare_coco_vision.py framework/src/utils/dataset_resolver.py framework/tests/test_coco_vision_assets.py
git commit -m "feat: prepare official coco vision data"
```

---

### Task 3: Shared YOLO Vision Letterbox Preprocessor

**Files:**
- Create: `framework/src/preprocessor/yolo_vision_preprocessor.py`
- Modify: `framework/src/preprocessor/__init__.py`
- Create: `framework/tests/test_yolov8_vision_preprocessor.py`

**Interfaces:**
- Produces: `YoloVisionPreprocessor(target_hw=(640, 640), layout="NCHW")`; `preprocess_with_context(raw_input) -> tuple[np.ndarray, dict]`; `load_or_preprocess_with_context(cache_path, raw_input) -> tuple[np.ndarray, dict]`; `get_cache_path(cache_dir, image_filename) -> str | None`.
- Consumes: file paths or `PIL.Image.Image`; context keys exactly match the design spec.

- [ ] **Step 1: Write failing pixel, geometry, and cache tests**

```python
# framework/tests/test_yolov8_vision_preprocessor.py
from pathlib import Path

import numpy as np
from PIL import Image

from preprocessor.yolo_vision_preprocessor import YoloVisionPreprocessor


def test_letterbox_returns_exact_chw_tensor_and_context():
    source = Image.fromarray(np.full((2, 4, 3), [255, 0, 0], dtype=np.uint8), "RGB")
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
    np.testing.assert_allclose(tensor[:, 2:6, :], np.array([1.0, 0.0, 0.0])[:, None, None])
    np.testing.assert_allclose(tensor[:, :2, :], 114.0 / 255.0)


def test_corrupt_cache_is_rebuilt(tmp_path):
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (4, 2), (1, 2, 3)).save(image_path)
    processor = YoloVisionPreprocessor(target_hw=(8, 8))
    cache = processor.get_cache_path(tmp_path / "cache", image_path.name)
    Path(cache).parent.mkdir()
    Path(cache).write_bytes(b"not-an-npz")
    tensor, context = processor.load_or_preprocess_with_context(cache, str(image_path))
    assert tensor.shape == (3, 8, 8)
    assert context["original_width"] == 4
```

- [ ] **Step 2: Run the tests and confirm RED**

```bash
.venv/bin/python -m pytest tests/test_yolov8_vision_preprocessor.py -v
```

Expected: module import failure because the shared preprocessor does not exist.

- [ ] **Step 3: Implement deterministic letterbox and atomic cache**

```python
# framework/src/preprocessor/yolo_vision_preprocessor.py
class YoloVisionPreprocessor(BasePreprocessor):
    CACHE_VERSION = 1

    def __init__(self, target_hw=(640, 640), layout="NCHW"):
        self.target_hw = tuple(int(value) for value in target_hw)
        self.layout = str(layout).upper()
        if len(self.target_hw) != 2 or any(value <= 0 for value in self.target_hw):
            raise ValueError("target_hw must contain two positive dimensions")
        if self.layout != "NCHW":
            raise ValueError("YOLOv8 segmentation and pose require NCHW layout")

    def preprocess_with_context(self, raw_input):
        image = Image.open(raw_input) if isinstance(raw_input, (str, os.PathLike)) else raw_input
        image = image.convert("RGB")
        original_width, original_height = image.size
        if original_width <= 0 or original_height <= 0:
            raise ValueError("image dimensions must be positive")
        input_height, input_width = self.target_hw
        scale = min(input_width / original_width, input_height / original_height)
        resized_width = round(original_width * scale)
        resized_height = round(original_height * scale)
        image = image.resize((resized_width, resized_height), Image.Resampling.BILINEAR)
        pad_x = (input_width - resized_width) // 2
        pad_y = (input_height - resized_height) // 2
        canvas = Image.new("RGB", (input_width, input_height), (114, 114, 114))
        canvas.paste(image, (pad_x, pad_y))
        array = np.asarray(canvas, dtype=np.float32) / 255.0
        tensor = np.ascontiguousarray(array.transpose(2, 0, 1))
        return tensor, {
            "original_height": original_height,
            "original_width": original_width,
            "input_height": input_height,
            "input_width": input_width,
            "scale": float(scale),
            "pad_x": float(pad_x),
            "pad_y": float(pad_y),
        }
```

Implement `.npz` cache serialization with `cache_version`, validate every context key and tensor dtype/shape on read, catch `OSError`, `ValueError`, and `EOFError`, and replace a same-directory temporary file with `os.replace()` only after `np.savez_compressed()` succeeds. Export the class in `preprocessor/__init__.py`.

- [ ] **Step 4: Verify GREEN and no regression in existing image preprocessing**

```bash
.venv/bin/python -m pytest tests/test_yolov8_vision_preprocessor.py tests/test_object_detection_loader.py tests/test_image_classification_loader.py -q
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit only Task 3 files**

```bash
git add framework/src/preprocessor/yolo_vision_preprocessor.py framework/src/preprocessor/__init__.py framework/tests/test_yolov8_vision_preprocessor.py
git commit -m "feat: add yolov8 vision preprocessing"
```

---

### Task 4: Validated Lazy COCO Vision Loaders

**Files:**
- Create: `framework/src/dataloader/coco_vision_loader.py`
- Create: `framework/src/dataloader/coco_instance_segmentation_loader.py`
- Create: `framework/src/dataloader/coco_pose_loader.py`
- Modify: `framework/src/dataloader/__init__.py`
- Create: `framework/tests/coco_test_utils.py`
- Create: `framework/tests/test_coco_vision_loader.py`

**Interfaces:**
- Consumes: `Model_Spec`, `dataset_path`, `image_dir`, `label_path` pointing to COCO JSON, optional `cache_dir`, shared preprocessor.
- Produces: `CocoInstanceSegmentationLoader` and `CocoPoseLoader` implementing all abstract DataLoader methods and the exact sample schema in the design.

- [ ] **Step 1: Create valid tiny COCO fixtures and failing loader tests**

```python
# framework/tests/coco_test_utils.py
def write_coco_fixture(root: Path, *, include_keypoints: bool = True) -> dict[str, Path]:
    image_dir = root / "images" / "val2017"
    annotation_dir = root / "annotations"
    image_dir.mkdir(parents=True)
    annotation_dir.mkdir()
    Image.new("RGB", (8, 6), (255, 255, 255)).save(image_dir / "000000000001.jpg")
    images = [{"id": 1, "file_name": "000000000001.jpg", "width": 8, "height": 6}]
    instances = {
        "images": images,
        "annotations": [{
            "id": 1, "image_id": 1, "category_id": 1, "bbox": [2, 1, 4, 4],
            "area": 16, "iscrowd": 0, "segmentation": [[2, 1, 6, 1, 6, 5, 2, 5]],
        }],
        "categories": [{"id": 1, "name": "person", "supercategory": "person"}],
    }
    keypoints = [4, 3, 2] * 17
    pose = {
        "images": images,
        "annotations": [{
            "id": 1, "image_id": 1, "category_id": 1, "bbox": [2, 1, 4, 4],
            "area": 16, "iscrowd": 0, "num_keypoints": 17, "keypoints": keypoints,
        }],
        "categories": [{
            "id": 1, "name": "person", "supercategory": "person",
            "keypoints": [f"kpt_{index}" for index in range(17)], "skeleton": [],
        }],
    }
    instances_path = annotation_dir / "instances_val2017.json"
    pose_path = annotation_dir / "person_keypoints_val2017.json"
    instances_path.write_text(json.dumps(instances))
    pose_path.write_text(json.dumps(pose))
    return {"images": image_dir, "instances": instances_path, "pose": pose_path}


def make_seg_spec() -> Model_Spec:
    return Model_Spec(
        name="yolov8s-seg", task=Task.INSTANCE_SEGMENTATION,
        input_shapes={"images": (1, 3, 640, 640)}, input_dtype={"images": "float32"},
        output_shapes={"output0": (1, 116, 8400), "output1": (1, 32, 160, 160)},
    )


def make_pose_spec() -> Model_Spec:
    return Model_Spec(
        name="yolov8s-pose", task=Task.POSE_ESTIMATION,
        input_shapes={"images": (1, 3, 640, 640)}, input_dtype={"images": "float32"},
        output_shapes={"output0": (1, 56, 8400)},
    )
```

```python
# framework/tests/test_coco_vision_loader.py
def test_segmentation_loader_returns_tensor_identity_and_context(tmp_path):
    paths = write_coco_fixture(tmp_path)
    loader = CocoInstanceSegmentationLoader(
        make_seg_spec(), dataset_path=str(tmp_path), image_dir=str(paths["images"]),
        label_path=str(paths["instances"]), target_hw=(8, 8),
    )
    sample = loader.load_by_index(0)
    assert sample["input"].shape == (3, 8, 8)
    assert sample["label"] == {"image_id": 1, "file_name": "000000000001.jpg"}
    assert sample["preprocess_context"]["original_width"] == 8
    assert loader.get_metadata()["category_ids"] == [1]


def test_loader_rejects_duplicate_image_ids(tmp_path):
    paths = write_coco_fixture(tmp_path)
    payload = json.loads(paths["instances"].read_text())
    payload["images"].append(dict(payload["images"][0]))
    paths["instances"].write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="duplicate image id 1"):
        CocoInstanceSegmentationLoader(
            make_seg_spec(), dataset_path=str(tmp_path), image_dir=str(paths["images"]),
            label_path=str(paths["instances"]), target_hw=(8, 8),
        )
```

- [ ] **Step 2: Run the tests and confirm RED**

```bash
.venv/bin/python -m pytest tests/test_coco_vision_loader.py -v
```

Expected: loader module imports fail because none of the three loader modules exist.

- [ ] **Step 3: Implement the shared index and thin task-specific loaders**

```python
# framework/src/dataloader/coco_vision_loader.py
class CocoVisionLoader(DataLoader):
    expected_task: Task

    def __init__(self, model_spec: Model_Spec, **kwargs):
        if model_spec.task is not self.expected_task:
            raise ValueError(f"{type(self).__name__} requires {self.expected_task.name}")
        self.model_spec = model_spec
        self.image_dir = Path(kwargs["image_dir"])
        self.annotation_file = Path(kwargs["label_path"])
        self.cache_dir = kwargs.get("cache_dir")
        preprocess_mode = str(kwargs.get("image_preprocess_mode", "auto")).lower()
        resize_mode = str(kwargs.get("image_resize_mode", "auto")).lower()
        if preprocess_mode not in {"auto", "normalized"}:
            raise ValueError("COCO YOLOv8 loaders require normalized float input")
        if resize_mode not in {"auto", "letterbox"}:
            raise ValueError("COCO YOLOv8 loaders require letterbox resize")
        self.preprocessor = kwargs.get("preprocessor") or YoloVisionPreprocessor(
            target_hw=kwargs.get("target_hw", (640, 640)), layout=kwargs.get("layout", "NCHW")
        )
        payload = json.loads(self.annotation_file.read_text())
        self.images = self._validate_and_index(payload)
        self.category_ids = sorted(int(item["id"]) for item in payload["categories"])
        self.total_samples = len(self.images)
        self.current_idx = 0

    def _sample_at(self, index: int) -> dict:
        image = self.images[index]
        image_path = self.image_dir / image["file_name"]
        cache_path = self.preprocessor.get_cache_path(self.cache_dir, image["file_name"])
        tensor, context = self.preprocessor.load_or_preprocess_with_context(cache_path, image_path)
        return {
            "input": tensor,
            "label": {"image_id": int(image["id"]), "file_name": image["file_name"]},
            "preprocess_context": context,
            "img_path": str(image_path),
        }
```

Implement deterministic image-ID ordering, duplicate detection, required JSON keys, every referenced image file check, sequential `load_single()`/`load_batch()`, cursor-free `load_by_index()`, `get_labels()`, `get_metadata()`, and `preprocess()`. The pose subclass additionally requires the person category with 17 keypoint names; the segmentation subclass requires at least one category and annotations with segmentation data. Route both tasks in `dataloader/__init__.py`.

- [ ] **Step 4: Verify GREEN and DataLoader API compatibility**

```bash
.venv/bin/python -m pytest tests/test_coco_vision_loader.py tests/test_factory_api.py tests/test_inference_pipeline.py -q
```

Expected: all selected tests pass, including sequential and random-access assertions.

- [ ] **Step 5: Commit only Task 4 files**

```bash
git add framework/src/dataloader/coco_vision_loader.py framework/src/dataloader/coco_instance_segmentation_loader.py framework/src/dataloader/coco_pose_loader.py framework/src/dataloader/__init__.py framework/tests/coco_test_utils.py framework/tests/test_coco_vision_loader.py
git commit -m "feat: load coco segmentation and pose data"
```

---

### Task 5: Shared Validated NumPy Decoder Primitives

**Files:**
- Create: `framework/src/decoders/yolo_vision.py`
- Create: `framework/tests/test_yolov8_decoder_primitives.py`

**Interfaces:**
- Produces: `DETECTIONS_KEY`, `MASKS_KEY`, `KEYPOINTS_KEY`; `as_bcn(array, feature_count) -> np.ndarray`; `resolve_output(outputs, predicate, description) -> np.ndarray`; `xywh_to_xyxy(boxes) -> np.ndarray`; `class_aware_nms(boxes, scores, class_ids, iou_threshold, max_detections) -> np.ndarray`.
- Consumes: finite NumPy arrays only; returns stable score-descending integer indices.

- [ ] **Step 1: Write failing shape, NMS, and validation tests**

```python
# framework/tests/test_yolov8_decoder_primitives.py
import numpy as np
import pytest

from decoders.yolo_vision import as_bcn, class_aware_nms, resolve_output


def test_as_bcn_accepts_bcn_and_transposes_bnc():
    bcn = np.zeros((2, 56, 10), dtype=np.float32)
    assert as_bcn(bcn, 56).shape == (2, 56, 10)
    assert as_bcn(bcn.transpose(0, 2, 1), 56).shape == (2, 56, 10)


def test_class_aware_nms_keeps_overlapping_different_classes():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 9, 9], [1, 1, 9, 9]], dtype=np.float32)
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    classes = np.array([0, 0, 1], dtype=np.int64)
    np.testing.assert_array_equal(class_aware_nms(boxes, scores, classes, 0.5, 10), [0, 2])


def test_resolve_output_rejects_ambiguous_matches():
    outputs = {"a": np.zeros((1, 56, 1)), "b": np.zeros((1, 56, 2))}
    with pytest.raises(ValueError, match="ambiguous pose prediction"):
        resolve_output(outputs, lambda value: value.ndim == 3 and 56 in value.shape, "pose prediction")


def test_as_bcn_rejects_non_finite_values():
    values = np.zeros((1, 56, 1), dtype=np.float32)
    values[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        as_bcn(values, 56)
```

- [ ] **Step 2: Run the tests and confirm RED**

```bash
.venv/bin/python -m pytest tests/test_yolov8_decoder_primitives.py -v
```

Expected: module import failure for `decoders.yolo_vision`.

- [ ] **Step 3: Implement pure NumPy primitives**

```python
# framework/src/decoders/yolo_vision.py
DETECTIONS_KEY = "detections"
MASKS_KEY = "masks"
KEYPOINTS_KEY = "keypoints"


def as_bcn(array: np.ndarray, feature_count: int) -> np.ndarray:
    value = np.asarray(array)
    if value.ndim != 3:
        raise ValueError(f"expected rank-3 prediction, got {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError("prediction values must be finite")
    if value.shape[1] == feature_count and value.shape[2] != feature_count:
        return value.astype(np.float32, copy=False)
    if value.shape[2] == feature_count and value.shape[1] != feature_count:
        return value.transpose(0, 2, 1).astype(np.float32, copy=False)
    raise ValueError(f"prediction does not contain a unique {feature_count}-feature axis: {value.shape}")


def resolve_output(outputs, predicate, description):
    matches = [(name, np.asarray(value)) for name, value in outputs.items() if predicate(np.asarray(value))]
    if len(matches) != 1:
        names = [name for name, _ in matches]
        raise ValueError(f"expected exactly one {description}; matched {names}")
    return matches[0][1]
```

Implement vectorized `xywh_to_xyxy()`, IoU, and per-class NMS. Sort candidates by descending score with original index as the tie-breaker, concatenate kept indices, sort the final set by the same rule, and truncate to `max_detections`.

- [ ] **Step 4: Verify GREEN and existing object decoder behavior**

```bash
.venv/bin/python -m pytest tests/test_yolov8_decoder_primitives.py tests/test_object_detection_decoders.py -q
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit only Task 5 files**

```bash
git add framework/src/decoders/yolo_vision.py framework/tests/test_yolov8_decoder_primitives.py
git commit -m "feat: add numpy yolov8 decoder primitives"
```

---

### Task 6: YOLOv8s Instance Segmentation Decoder

**Files:**
- Create: `framework/src/decoders/instance_segmentation.py`
- Create: `framework/tests/test_yolov8_segmentation_decoder.py`

**Interfaces:**
- Consumes: one finite prediction tensor with 116 features, one finite NCHW prototype tensor with 32 channels, confidence/NMS/max-detection options.
- Produces: `YoloV8SegmentationDecoder.decode(outputs) -> {"detections": float32[N,7], "masks": uint8[N,H,W]}`.

- [ ] **Step 1: Write failing synthetic decoding tests**

```python
# framework/tests/test_yolov8_segmentation_decoder.py
def _seg_outputs():
    prediction = np.zeros((1, 116, 2), dtype=np.float32)
    prediction[0, :4, 0] = [4, 4, 4, 4]
    prediction[0, 4, 0] = 0.9
    prediction[0, 84, 0] = 10.0
    prediction[0, :4, 1] = [4.2, 4.2, 4, 4]
    prediction[0, 4, 1] = 0.8
    prediction[0, 84, 1] = 10.0
    prototype = np.zeros((1, 32, 2, 2), dtype=np.float32)
    prototype[0, 0] = 1.0
    return {"prediction": prediction, "proto": prototype}


def test_decoder_applies_nms_and_builds_aligned_binary_masks():
    result = YoloV8SegmentationDecoder(
        conf_threshold=0.25, iou_threshold=0.5, max_detections=10
    ).decode(_seg_outputs())
    assert result["detections"].shape == (1, 7)
    assert result["detections"][0, :3].tolist() == pytest.approx([0, 0, 0.9])
    assert result["masks"].shape == (1, 8, 8)
    assert result["masks"].dtype == np.uint8
    assert result["masks"][0, 2:6, 2:6].all()
    assert not result["masks"][0, :2].any()


def test_decoder_returns_typed_empty_arrays():
    outputs = _seg_outputs()
    outputs["prediction"][:, 4:84, :] = 0.0
    result = YoloV8SegmentationDecoder().decode(outputs)
    assert result["detections"].shape == (0, 7)
    assert result["masks"].shape == (0, 8, 8)
```

- [ ] **Step 2: Run the tests and confirm RED**

```bash
.venv/bin/python -m pytest tests/test_yolov8_segmentation_decoder.py -v
```

Expected: module/class import failure.

- [ ] **Step 3: Implement segmentation decode, mask projection, crop, and threshold**

```python
# framework/src/decoders/instance_segmentation.py
class YoloV8SegmentationDecoder:
    FEATURE_COUNT = 116
    CLASS_COUNT = 80
    MASK_COUNT = 32

    def decode(self, outputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        raw_prediction = resolve_output(
            outputs, lambda value: value.ndim == 3 and self.FEATURE_COUNT in value.shape,
            "segmentation prediction",
        )
        prototypes = resolve_output(
            outputs, lambda value: value.ndim == 4 and value.shape[1] == self.MASK_COUNT,
            "segmentation prototypes",
        ).astype(np.float32, copy=False)
        prediction = as_bcn(raw_prediction, self.FEATURE_COUNT)
        if prediction.shape[0] != prototypes.shape[0]:
            raise ValueError("segmentation prediction/prototype batch mismatch")
        return self._decode_batch(prediction, prototypes)
```

For each image, transpose to anchors by features, compute class ID/score from channels `4:84`, filter `score > conf_threshold`, convert boxes, call `class_aware_nms()`, project `coefficients @ prototype.reshape(32, -1)`, resize logits with `cv2.INTER_LINEAR` to four times prototype dimensions, zero pixels outside each clipped integer box, threshold `> 0`, and append local image index to each canonical detection row.

- [ ] **Step 4: Verify GREEN and malformed-output cases**

```bash
.venv/bin/python -m pytest tests/test_yolov8_segmentation_decoder.py tests/test_yolov8_decoder_primitives.py -q
```

Expected: all tests pass, including ambiguous output, batch mismatch, non-finite, and empty-result cases.

- [ ] **Step 5: Commit only Task 6 files**

```bash
git add framework/src/decoders/instance_segmentation.py framework/tests/test_yolov8_segmentation_decoder.py
git commit -m "feat: decode yolov8 instance masks"
```

---

### Task 7: YOLOv8s Pose Decoder

**Files:**
- Create: `framework/src/decoders/pose_estimation.py`
- Create: `framework/tests/test_yolov8_pose_decoder.py`

**Interfaces:**
- Consumes: one finite rank-3 prediction with 56 features: 4 box, 1 person score, and 51 keypoint values.
- Produces: `YoloV8PoseDecoder.decode(outputs) -> {"detections": float32[N,7], "keypoints": float32[N,17,3]}`.

- [ ] **Step 1: Write failing pose reshape, NMS, and empty tests**

```python
# framework/tests/test_yolov8_pose_decoder.py
def test_pose_decoder_keeps_keypoints_aligned_after_nms():
    prediction = np.zeros((1, 56, 2), dtype=np.float32)
    prediction[0, :5, 0] = [4, 4, 4, 4, 0.9]
    prediction[0, 5:, 0] = np.tile([4, 3, 0.8], 17)
    prediction[0, :5, 1] = [4.1, 4.1, 4, 4, 0.7]
    prediction[0, 5:, 1] = np.tile([1, 1, 0.2], 17)
    result = YoloV8PoseDecoder(iou_threshold=0.5).decode({"pose": prediction})
    assert result["detections"].shape == (1, 7)
    assert result["keypoints"].shape == (1, 17, 3)
    np.testing.assert_allclose(result["keypoints"][0, 0], [4, 3, 0.8])


def test_pose_decoder_rejects_wrong_feature_count():
    with pytest.raises(ValueError, match="pose prediction"):
        YoloV8PoseDecoder().decode({"pose": np.zeros((1, 55, 10), dtype=np.float32)})
```

- [ ] **Step 2: Run the tests and confirm RED**

```bash
.venv/bin/python -m pytest tests/test_yolov8_pose_decoder.py -v
```

Expected: module/class import failure.

- [ ] **Step 3: Implement pose decoding with row-aligned keypoints**

```python
# framework/src/decoders/pose_estimation.py
class YoloV8PoseDecoder:
    FEATURE_COUNT = 56
    KEYPOINT_SHAPE = (17, 3)

    def decode(self, outputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        raw = resolve_output(
            outputs, lambda value: value.ndim == 3 and self.FEATURE_COUNT in value.shape,
            "pose prediction",
        )
        prediction = as_bcn(raw, self.FEATURE_COUNT)
        detections, keypoints = [], []
        for local_index, features in enumerate(prediction.transpose(0, 2, 1)):
            scores = features[:, 4]
            candidates = np.flatnonzero(scores > self.conf_threshold)
            boxes = xywh_to_xyxy(features[candidates, :4])
            keep = class_aware_nms(
                boxes, scores[candidates], np.zeros(len(candidates), dtype=np.int64),
                self.iou_threshold, self.max_detections,
            )
            for kept in keep:
                anchor = candidates[kept]
                detections.append([local_index, 0, scores[anchor], *boxes[kept]])
                keypoints.append(features[anchor, 5:].reshape(self.KEYPOINT_SHAPE))
        return {
            DETECTIONS_KEY: np.asarray(detections, dtype=np.float32).reshape(-1, 7),
            KEYPOINTS_KEY: np.asarray(keypoints, dtype=np.float32).reshape(-1, 17, 3),
        }
```

- [ ] **Step 4: Verify GREEN and common primitive regressions**

```bash
.venv/bin/python -m pytest tests/test_yolov8_pose_decoder.py tests/test_yolov8_decoder_primitives.py -q
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit only Task 7 files**

```bash
git add framework/src/decoders/pose_estimation.py framework/tests/test_yolov8_pose_decoder.py
git commit -m "feat: decode yolov8 pose keypoints"
```

---

### Task 8: Streaming COCO Evaluator Base and Official Fixture Dependency

**Files:**
- Modify: `framework/requirements.txt`
- Create: `framework/src/evaluators/coco_common.py`
- Create: `framework/tests/test_coco_evaluator_common.py`

**Interfaces:**
- Consumes: official annotation JSON, decoder payload, labels from `InferencePipeline.prepare_eval_labels()`, scalar or timing dictionary.
- Produces: `CocoEvaluatorBase(annotation_file: str, iou_type: str)` helpers `_normalize_batch_labels`, `_restore_boxes`, `_latency_metrics`, `_run_coco_eval`; category IDs and image IDs validated against ground truth.

- [ ] **Step 1: Install the evaluator dependency and write failing common-helper tests**

Install `pycocotools==2.0.11` into the existing venv without changing production files yet, then write:

```bash
.venv/bin/python -m pip install pycocotools==2.0.11
```

```python
# framework/tests/test_coco_evaluator_common.py
class _ConcreteTestEvaluator(CocoEvaluatorBase):
    def add_batch(self, outputs, labels, timing_ms):
        self._record_batch(self._normalize_batch_labels(labels), 0, timing_ms)

    def compute(self):
        return self._latency_metrics()

    def evaluate(self, result):
        self._reset()
        self.add_batch(result.outputs, result.labels, result.timing_records[0])
        return self.compute()

    def is_applicable(self, device_spec, model_spec):
        return True

    def get_metric_names(self):
        return ["Total Samples", "Average Latency (ms)"]


def test_normalize_labels_requires_image_id_and_complete_context(tmp_path):
    paths = write_coco_fixture(tmp_path)
    evaluator = _ConcreteTestEvaluator(str(paths["instances"]), "segm")
    with pytest.raises(ValueError, match="image_id"):
        evaluator._normalize_batch_labels([{"label": {}, "preprocess_context": {}}])


def test_restore_boxes_removes_padding_and_scale(tmp_path):
    paths = write_coco_fixture(tmp_path)
    evaluator = _ConcreteTestEvaluator(str(paths["instances"]), "segm")
    restored = evaluator._restore_boxes(
        np.array([[2, 2, 6, 6]], dtype=np.float32),
        {"original_height": 3, "original_width": 4, "input_height": 8,
         "input_width": 8, "scale": 2.0, "pad_x": 0.0, "pad_y": 1.0},
    )
    np.testing.assert_allclose(restored, [[1, 0.5, 3, 2.5]])


def test_no_prediction_coco_eval_returns_zero_stats(tmp_path):
    paths = write_coco_fixture(tmp_path)
    evaluator = _ConcreteTestEvaluator(str(paths["instances"]), "segm")
    stats = evaluator._run_coco_eval([], [1])
    np.testing.assert_array_equal(stats, np.zeros(12, dtype=np.float64))
```

- [ ] **Step 2: Run the tests and confirm RED**

```bash
.venv/bin/python -m pytest tests/test_coco_evaluator_common.py -v
```

Expected: import failure for `evaluators.coco_common` after the dependency itself imports successfully.

- [ ] **Step 3: Implement the shared evaluator state and COCOeval boundary**

Add `pycocotools==2.0.11` to `framework/requirements.txt`, then implement:

```python
# framework/src/evaluators/coco_common.py
class CocoEvaluatorBase(Evaluator):
    def __init__(self, annotation_file: str, iou_type: str, **options):
        if not annotation_file:
            raise ValueError("annotation_file is required for COCO evaluation")
        self.annotation_file = str(annotation_file)
        self.iou_type = iou_type
        self._coco_gt = COCO(self.annotation_file)
        self.category_ids = sorted(self._coco_gt.getCatIds())
        self._valid_image_ids = set(self._coco_gt.getImgIds())
        self._reset()

    def _run_coco_eval(self, records, image_ids):
        if not records:
            length = 10 if self.iou_type == "keypoints" else 12
            return np.zeros(length, dtype=np.float64)
        coco_results = self._coco_gt.loadRes(records)
        evaluator = COCOeval(self._coco_gt, coco_results, self.iou_type)
        evaluator.params.imgIds = sorted(set(image_ids))
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
        return np.asarray(evaluator.stats, dtype=np.float64)
```

Implement exact context-key validation, local-index validation, image-ID membership, box clipping, timing dictionary normalization via `total_ms`, state reset, average detections, average latency, P99, and FPS. Keep task record conversion abstract in subclasses.

- [ ] **Step 4: Verify GREEN and dependency import**

```bash
.venv/bin/python -c "import pycocotools; print('pycocotools import OK')"
.venv/bin/python -m pytest tests/test_coco_evaluator_common.py tests/test_object_detection_evaluator.py -q
```

Expected: import succeeds and all selected tests pass.

- [ ] **Step 5: Commit only Task 8 files**

```bash
git add framework/requirements.txt framework/src/evaluators/coco_common.py framework/tests/test_coco_evaluator_common.py
git commit -m "feat: add official coco evaluator base"
```

---

### Task 9: Official Instance Segmentation Evaluator

**Files:**
- Create: `framework/src/evaluators/instance_segmentation_evaluator.py`
- Create: `framework/tests/test_instance_segmentation_evaluator.py`

**Interfaces:**
- Consumes: canonical `detections` and row-aligned binary `masks`, batch labels/context, annotation path.
- Produces: streaming COCO RLE records and metrics `Mask mAP`, `Mask AP50`, `Mask AP75`, `Mask AP Small`, `Mask AP Medium`, `Mask AP Large`, sample/detection/latency metrics.

- [ ] **Step 1: Write failing perfect, empty, and alignment tests**

```python
# framework/tests/test_instance_segmentation_evaluator.py
def _seg_labels():
    return [{
        "label": {"image_id": 1, "file_name": "000000000001.jpg"},
        "preprocess_context": {
            "original_height": 6, "original_width": 8,
            "input_height": 6, "input_width": 8,
            "scale": 1.0, "pad_x": 0.0, "pad_y": 0.0,
        },
    }]


def test_perfect_mask_prediction_has_near_one_coco_map(tmp_path):
    paths = write_coco_fixture(tmp_path)
    mask = np.zeros((1, 6, 8), dtype=np.uint8)
    mask[0, 1:5, 2:6] = 1
    outputs = {
        "detections": np.array([[0, 0, 0.99, 2, 1, 6, 5]], dtype=np.float32),
        "masks": mask,
    }
    evaluator = InstanceSegmentationEvaluator(annotation_file=str(paths["instances"]))
    evaluator.add_batch(outputs, _seg_labels(), 2.0)
    metrics = evaluator.compute()
    assert metrics["Mask mAP"] > 0.99
    assert metrics["Mask AP50"] > 0.99
    assert metrics["Total Samples"] == 1


def test_mask_count_must_match_detection_count(tmp_path):
    paths = write_coco_fixture(tmp_path)
    evaluator = InstanceSegmentationEvaluator(annotation_file=str(paths["instances"]))
    with pytest.raises(ValueError, match="row count"):
        evaluator.add_batch(
            {"detections": np.zeros((1, 7)), "masks": np.zeros((0, 6, 8), dtype=np.uint8)},
            _seg_labels(), 1.0,
        )
```

- [ ] **Step 2: Run the tests and confirm RED**

```bash
.venv/bin/python -m pytest tests/test_instance_segmentation_evaluator.py -v
```

Expected: evaluator module/class import failure.

- [ ] **Step 3: Implement mask restoration, RLE, streaming records, and metric names**

```python
# framework/src/evaluators/instance_segmentation_evaluator.py
class InstanceSegmentationEvaluator(CocoEvaluatorBase):
    METRIC_INDEX = {
        "Mask mAP": 0, "Mask AP50": 1, "Mask AP75": 2,
        "Mask AP Small": 3, "Mask AP Medium": 4, "Mask AP Large": 5,
    }

    def add_batch(self, outputs, labels, timing_ms):
        detections = np.asarray(outputs[DETECTIONS_KEY], dtype=np.float32).reshape(-1, 7)
        masks = np.asarray(outputs[MASKS_KEY], dtype=np.uint8)
        if len(detections) != len(masks):
            raise ValueError("segmentation detection/mask row count mismatch")
        batch = self._normalize_batch_labels(labels)
        for row, binary_mask in zip(detections, masks):
            local_index = int(row[0])
            label, context = batch[local_index]
            restored = self._restore_mask(binary_mask, context)
            rle = mask_utils.encode(np.asfortranarray(restored, dtype=np.uint8))
            rle["counts"] = rle["counts"].decode("ascii")
            box = self._restore_boxes(row[None, 3:7], context)[0]
            self._records.append({
                "image_id": label["image_id"],
                "category_id": self.category_ids[int(row[1])],
                "segmentation": rle,
                "bbox": [float(box[0]), float(box[1]), float(box[2]-box[0]), float(box[3]-box[1])],
                "score": float(row[2]),
            })
        self._record_batch(batch, len(detections), timing_ms)
```

Implement `_restore_mask()` by clipping integer padding boundaries, cropping, resizing to `(original_width, original_height)` with `cv2.INTER_NEAREST`, and returning binary `uint8`. `compute()` calls `_run_coco_eval`, maps non-negative stats by `METRIC_INDEX`, merges common metrics, and treats absent predictions as zero. `evaluate()` resets, adds one result, and computes.

- [ ] **Step 4: Verify GREEN with official COCOeval fixture**

```bash
.venv/bin/python -m pytest tests/test_instance_segmentation_evaluator.py tests/test_coco_evaluator_common.py -q
```

Expected: perfect prediction exceeds 0.99, shifted/empty predictions are lower or zero, subset IDs exclude unseen images, and all tests pass.

- [ ] **Step 5: Commit only Task 9 files**

```bash
git add framework/src/evaluators/instance_segmentation_evaluator.py framework/tests/test_instance_segmentation_evaluator.py
git commit -m "feat: evaluate coco instance masks"
```

---

### Task 10: Official Pose OKS Evaluator

**Files:**
- Create: `framework/src/evaluators/pose_estimation_evaluator.py`
- Create: `framework/tests/test_pose_estimation_evaluator.py`

**Interfaces:**
- Consumes: canonical `detections` and row-aligned `(N,17,3)` keypoints, labels/context, person-keypoint annotation path.
- Produces: COCO keypoint result records and `OKS mAP`, `OKS AP50`, `OKS AP75`, `OKS AP Medium`, `OKS AP Large`, sample/detection/latency metrics.

- [ ] **Step 1: Write failing perfect-keypoint, coordinate restoration, and empty tests**

```python
# framework/tests/test_pose_estimation_evaluator.py
def test_perfect_keypoints_have_near_one_oks_map(tmp_path):
    paths = write_coco_fixture(tmp_path)
    keypoints = np.tile([4, 3, 1.0], (1, 17, 1)).astype(np.float32)
    outputs = {
        "detections": np.array([[0, 0, 0.99, 2, 1, 6, 5]], dtype=np.float32),
        "keypoints": keypoints,
    }
    labels = [{
        "label": {"image_id": 1, "file_name": "000000000001.jpg"},
        "preprocess_context": {
            "original_height": 6, "original_width": 8, "input_height": 6,
            "input_width": 8, "scale": 1.0, "pad_x": 0.0, "pad_y": 0.0,
        },
    }]
    evaluator = PoseEstimationEvaluator(annotation_file=str(paths["pose"]))
    evaluator.add_batch(outputs, labels, 2.0)
    metrics = evaluator.compute()
    assert metrics["OKS mAP"] > 0.99
    assert metrics["OKS AP50"] > 0.99


def test_pose_evaluator_rejects_non_person_category_file(tmp_path):
    paths = write_coco_fixture(tmp_path)
    payload = json.loads(paths["pose"].read_text())
    payload["categories"][0]["id"] = 2
    paths["pose"].write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="category id 1"):
        PoseEstimationEvaluator(annotation_file=str(paths["pose"]))
```

- [ ] **Step 2: Run the tests and confirm RED**

```bash
.venv/bin/python -m pytest tests/test_pose_estimation_evaluator.py -v
```

Expected: evaluator module/class import failure.

- [ ] **Step 3: Implement keypoint restoration and official OKS records**

```python
# framework/src/evaluators/pose_estimation_evaluator.py
class PoseEstimationEvaluator(CocoEvaluatorBase):
    METRIC_INDEX = {
        "OKS mAP": 0, "OKS AP50": 1, "OKS AP75": 2,
        "OKS AP Medium": 3, "OKS AP Large": 4,
    }

    def _restore_keypoints(self, keypoints, context):
        restored = np.asarray(keypoints, dtype=np.float32).copy()
        restored[:, 0] = np.clip(
            (restored[:, 0] - context["pad_x"]) / context["scale"],
            0, context["original_width"],
        )
        restored[:, 1] = np.clip(
            (restored[:, 1] - context["pad_y"]) / context["scale"],
            0, context["original_height"],
        )
        return restored
```

Validate keypoint shape and row alignment, require category ID 1 with 17 keypoint names, restore each box/keypoint array, flatten to 51 floats, accumulate `{image_id, category_id: 1, keypoints, score, bbox}` records, and map `COCOeval(self._coco_gt, coco_results, "keypoints")` stats. Preserve keypoint confidence values.

- [ ] **Step 4: Verify GREEN with official OKS calculation**

```bash
.venv/bin/python -m pytest tests/test_pose_estimation_evaluator.py tests/test_coco_evaluator_common.py -q
```

Expected: perfect prediction exceeds 0.99, shifted predictions reduce OKS, empty predictions return zeros, and all selected tests pass.

- [ ] **Step 5: Commit only Task 10 files**

```bash
git add framework/src/evaluators/pose_estimation_evaluator.py framework/tests/test_pose_estimation_evaluator.py
git commit -m "feat: evaluate coco pose keypoints"
```

---

### Task 11: Factory, CLI, Pipeline, Benchmark Registry, and Documentation Integration

**Files:**
- Modify: `framework/src/decoders/__init__.py`
- Modify: `framework/src/evaluators/__init__.py`
- Modify: `framework/src/main.py`
- Modify: `framework/tests/run_all_onnx_benchmarks.py`
- Modify: `framework/README.md`
- Modify: `framework/src/dataloader/README.md`
- Modify: `framework/src/evaluators/README.md`
- Create: `framework/tests/test_yolov8_vision_integration.py`

**Interfaces:**
- Consumes: exact model task, backend name, resolved annotation file, existing CLI options, loaders/decoders/evaluators from Tasks 4, 6, 7, 9, and 10.
- Produces: zero-config and explicit CLI component assembly for both tasks; both e2e and async completion reuse the same contracts.

- [ ] **Step 1: Write failing factory and metadata-flow integration tests**

```python
# framework/tests/test_yolov8_vision_integration.py
from argparse import Namespace
from types import SimpleNamespace

import numpy as np
import pytest

from core.async_inference.completion import CompletionCoordinator
from core.async_inference.metrics import AsyncMetricsCollector
from core.async_inference.types import BatchCompletion, InferenceRequest


def test_factories_route_both_vision_tasks(tmp_path):
    paths = write_coco_fixture(tmp_path)
    seg_spec = make_seg_spec()
    pose_spec = make_pose_spec()
    seg_decoder = create_decoder(seg_spec, backend="onnxruntime")
    pose_decoder = create_decoder(pose_spec, backend="onnxruntime")
    assert isinstance(seg_decoder, YoloV8SegmentationDecoder)
    assert isinstance(pose_decoder, YoloV8PoseDecoder)
    assert isinstance(
        create_evaluator(seg_spec, annotation_file=str(paths["instances"])),
        InstanceSegmentationEvaluator,
    )
    assert isinstance(
        create_evaluator(pose_spec, annotation_file=str(paths["pose"])),
        PoseEstimationEvaluator,
    )


def test_pipeline_preserves_coco_identity_and_letterbox_context(tmp_path):
    paths = write_coco_fixture(tmp_path)
    spec = make_seg_spec()
    loader = CocoInstanceSegmentationLoader(
        spec, dataset_path=str(tmp_path), image_dir=str(paths["images"]),
        label_path=str(paths["instances"]), target_hw=(8, 8),
    )
    runtime = SimpleNamespace(
        compiled_model=SimpleNamespace(spec=spec),
        supports_generate=lambda: False,
    )
    pipeline = InferencePipeline(loader, runtime)
    collated = pipeline.collate_batch(loader.load_batch(1))
    labels = pipeline.prepare_eval_labels(collated)
    assert labels[0]["label"]["image_id"] == 1
    assert labels[0]["preprocess_context"]["scale"] == 1.0


@pytest.mark.parametrize("task", [Task.INSTANCE_SEGMENTATION, Task.POSE_ESTIMATION])
def test_main_builds_accuracy_vision_component_kwargs(task):
    args = Namespace(image_preprocess_mode="auto", image_resize_mode="auto")
    loader_kwargs, evaluator_kwargs = benchmark_main._build_vision_task_kwargs(
        task, args, "/dataset/annotations/task.json"
    )
    assert loader_kwargs == {
        "image_preprocess_mode": "normalized",
        "image_resize_mode": "letterbox",
    }
    assert evaluator_kwargs == {"annotation_file": "/dataset/annotations/task.json"}


@pytest.mark.parametrize("queue_capacity", [None, 1])
def test_inline_and_queued_completion_deliver_same_canonical_payload(queue_capacity):
    class Pipeline:
        def prepare_eval_labels(self, collated):
            return collated["label"]

    class Decoder:
        def decode(self, outputs):
            assert outputs["raw"].shape == (1, 1)
            return {
                "detections": np.array([[0, 0, 0.9, 1, 1, 2, 2]], dtype=np.float32),
                "keypoints": np.ones((1, 17, 3), dtype=np.float32),
            }

    class Evaluator:
        def __init__(self):
            self.calls = []

        def add_batch(self, outputs, labels, timing_ms):
            self.calls.append((outputs, labels, timing_ms))

    evaluator = Evaluator()
    coordinator = CompletionCoordinator(
        pipeline=Pipeline(), evaluator=evaluator, decoder=Decoder(),
        metrics=AsyncMetricsCollector(0, 1), queue_capacity=queue_capacity,
        raise_callback_errors=True,
    )
    request = InferenceRequest(
        request_id=0, sample_index=0, sample={}, scheduled_ns=0,
        issued_ns=0, enqueued_ns=1, sample_count=1,
    )
    completion = BatchCompletion(
        requests=[request], collated={"label": [{"image_id": 1}]},
        outputs={"raw": np.ones((1, 1), dtype=np.float32)}, timing_ms=1.0,
        runtime_started_ns=2, runtime_finished_ns=3, worker_id=0, batch_size=1,
    )
    coordinator.start()
    coordinator.register(request)
    coordinator.submit(completion)
    if queue_capacity is not None:
        assert coordinator.wait_for_all(timeout=1.0) is True
    assert coordinator.stop(timeout=1.0) is True
    assert len(evaluator.calls) == 1
    assert evaluator.calls[0][0]["keypoints"].shape == (1, 17, 3)
    assert evaluator.calls[0][1] == [{"image_id": 1}]
```

- [ ] **Step 2: Run the tests and confirm RED**

```bash
.venv/bin/python -m pytest tests/test_yolov8_vision_integration.py -v
```

Expected: decoder/evaluator factories return `None` or latency-only evaluators, and CLI component kwargs omit the annotation file.

- [ ] **Step 3: Wire factories and CLI without changing completion orchestration**

```python
# framework/src/decoders/__init__.py
def create_decoder(model_spec: Model_Spec, **kwargs):
    if model_spec.task == Task.OBJECT_DETECTION:
        return create_object_detection_decoder(model_spec, **kwargs)
    if model_spec.task == Task.INSTANCE_SEGMENTATION:
        return YoloV8SegmentationDecoder(
            conf_threshold=kwargs.get("conf_threshold", 0.25),
            iou_threshold=kwargs.get("iou_threshold", 0.45),
            max_detections=kwargs.get("max_detections", 300),
        )
    if model_spec.task == Task.POSE_ESTIMATION:
        return YoloV8PoseDecoder(
            conf_threshold=kwargs.get("conf_threshold", 0.25),
            iou_threshold=kwargs.get("iou_threshold", 0.45),
            max_detections=kwargs.get("max_detections", 300),
        )
    return None
```

Route the two evaluator classes lazily. In `main.py`, apply letterbox preprocessing options to `OBJECT_DETECTION`, `INSTANCE_SEGMENTATION`, and `POSE_ESTIMATION`; pass the resolved JSON path as `annotation_file` for the two new evaluator tasks. Do not modify `CompletionCoordinator`: it already calls decoder before evaluator and preserves collated labels.

Use this testable assembly helper and merge its two dictionaries into the existing loader/evaluator kwargs:

```python
# framework/src/main.py
def _build_vision_task_kwargs(task_enum: Task, args, label_path: str) -> tuple[dict, dict]:
    loader_kwargs: dict = {}
    evaluator_kwargs: dict = {}
    if task_enum == Task.OBJECT_DETECTION:
        loader_kwargs = {
            "image_preprocess_mode": args.image_preprocess_mode,
            "image_resize_mode": args.image_resize_mode,
        }
    elif task_enum in (Task.INSTANCE_SEGMENTATION, Task.POSE_ESTIMATION):
        loader_kwargs = {
            "image_preprocess_mode": (
                "normalized" if args.image_preprocess_mode == "auto" else args.image_preprocess_mode
            ),
            "image_resize_mode": (
                "letterbox" if args.image_resize_mode == "auto" else args.image_resize_mode
            ),
        }
        evaluator_kwargs = {"annotation_file": label_path}
    return loader_kwargs, evaluator_kwargs
```

Add benchmark entries whose required files are the exact ONNX path, `images/val2017`, and task annotation JSON. Update README preparation, smoke, full evaluation, and CPU/CUDA commands, explicitly stating that `--max-steps 1` is smoke validation and omitting it is full COCO evaluation.

- [ ] **Step 4: Verify GREEN across integration and existing model paths**

```bash
.venv/bin/python -m pytest tests/test_yolov8_vision_integration.py tests/test_factory_api.py tests/test_inference_pipeline.py tests/test_async_completion.py tests/test_object_detection_e2e.py tests/test_resnet50.py -q
```

Expected: all selected tests pass. Existing ResNet50 and YOLOv5m paths remain green.

- [ ] **Step 5: Commit only Task 11 files**

```bash
git add framework/src/decoders/__init__.py framework/src/evaluators/__init__.py framework/src/main.py framework/tests/run_all_onnx_benchmarks.py framework/README.md framework/src/dataloader/README.md framework/src/evaluators/README.md framework/tests/test_yolov8_vision_integration.py
git commit -m "feat: integrate yolov8 vision benchmarks"
```

---

### Task 12: Real ONNX CPU/CUDA Execution and Completion Audit

**Files:**
- Create: `framework/tests/test_yolov8_onnx_runtime.py`
- Create: `docs/yolov8s-vision-verification.md`

**Interfaces:**
- Consumes: exported ONNX files, official COCO validation data, `OnnxRuntime`, actual provider lists, CLI pipeline.
- Produces: provider-gated real inference tests and recorded verification evidence for both models on CPU and CUDA.

- [ ] **Step 1: Prepare exact real assets and inspect them**

From `framework/`, install the new pin and prepare assets:

```bash
.venv/bin/python -m pip install pycocotools==2.0.11
.venv/bin/python models/prepare_yolov8_vision.py
.venv/bin/python datasets/prepare_coco_vision.py
.venv/bin/python -c "import onnx; from pathlib import Path; [print(p, [i.name for i in onnx.load(p).graph.input], [o.name for o in onnx.load(p).graph.output]) for p in map(str, [Path('models/yolov8s-seg/yolov8s-seg.onnx'), Path('models/yolov8s-pose/yolov8s-pose.onnx')])]"
```

Expected: both ONNX files exist; segmentation exposes one input and two outputs, pose exposes one input and one output; official image and both annotation paths validate.

- [ ] **Step 2: Write provider-gated real runtime tests before relying on manual CLI output**

```python
# framework/tests/test_yolov8_onnx_runtime.py
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest

from core.compiled_model import CompiledModel
from core.model_profiles import create_model_spec
from dataloader import create_dataloader
from decoders import create_decoder
from evaluators import create_evaluator
from runtimes.onnx_rt import OnnxRuntime
from utils.dataset_resolver import resolve_dataset_paths


@pytest.mark.parametrize(
    ("model_name", "model_path", "annotation_name", "metric_key", "canonical_key"),
    [
        (
            "yolov8s-seg", "models/yolov8s-seg/yolov8s-seg.onnx",
            "instances_val2017.json", "Mask mAP", "masks",
        ),
        (
            "yolov8s-pose", "models/yolov8s-pose/yolov8s-pose.onnx",
            "person_keypoints_val2017.json", "OKS mAP", "keypoints",
        ),
    ],
)
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_real_yolov8_model_runs_requested_provider(
    model_name, model_path, annotation_name, metric_key, canonical_key, device
):
    dataset_root = Path("datasets/coco")
    required = [
        Path(model_path), dataset_root / "images" / "val2017",
        dataset_root / "annotations" / annotation_name,
    ]
    if not all(path.exists() for path in required):
        pytest.skip("real YOLOv8/COCO assets are not prepared")
    if device == "cuda" and "CUDAExecutionProvider" not in ort.get_available_providers():
        pytest.skip("CUDAExecutionProvider unavailable")
    spec = create_model_spec(model_name, model_path)
    image_dir, annotation_file = resolve_dataset_paths(
        spec.task, str(dataset_root), "", ""
    )
    loader = create_dataloader(
        spec, dataset_path=str(dataset_root), image_dir=image_dir,
        label_path=annotation_file, cache_dir=None,
    )
    runtime = OnnxRuntime(device=device)
    runtime.load(CompiledModel(spec, "onnxruntime", Path(model_path)))
    sample = loader.load_by_index(0)
    outputs = runtime.run({next(iter(spec.input_shapes)): sample["input"][None]})
    assert all(isinstance(value, np.ndarray) for value in outputs.values())
    assert outputs
    decoded = create_decoder(spec, backend="onnxruntime").decode(outputs)
    assert decoded["detections"].ndim == 2
    assert canonical_key in decoded
    evaluator = create_evaluator(spec, annotation_file=annotation_file)
    evaluator.add_batch(
        decoded,
        [{"label": sample["label"], "preprocess_context": sample["preprocess_context"]}],
        1.0,
    )
    assert metric_key in evaluator.compute()
    active = runtime.get_device_spec()["active_providers"]
    requested = "CUDAExecutionProvider" if device == "cuda" else "CPUExecutionProvider"
    assert requested in active
    runtime.unload()
```

- [ ] **Step 3: Run real CPU tests and fix only observed failures with RED-GREEN evidence**

```bash
.venv/bin/python -m pytest tests/test_yolov8_onnx_runtime.py -v -k "cpu"
.venv/bin/python src/main.py --model yolov8s-seg --target cpu --dataset datasets/coco --max-steps 1 --warmup 1
.venv/bin/python src/main.py --model yolov8s-pose --target cpu --dataset datasets/coco --max-steps 1 --warmup 1
```

Expected: both tests and both CLI runs exit zero, sessions report active `CPUExecutionProvider`, decoder payloads have aligned rows, and final mask/OKS metric keys are printed.

- [ ] **Step 4: Run real CUDA tests with fallback rejection**

```bash
.venv/bin/python -m pytest tests/test_yolov8_onnx_runtime.py -v -k "cuda"
.venv/bin/python src/main.py --model yolov8s-seg --target cuda --dataset datasets/coco --max-steps 1 --warmup 1
.venv/bin/python src/main.py --model yolov8s-pose --target cuda --dataset datasets/coco --max-steps 1 --warmup 1
```

Expected: both tests and CLI runs exit zero; each loaded session includes active `CUDAExecutionProvider`. A provider initialization warning followed by CPU-only active providers is failure and must not be recorded as CUDA verification.

- [ ] **Step 5: Run the full regression and static checks**

```bash
.venv/bin/python -m pytest tests/ -q
.venv/bin/python -m compileall -q src models datasets tests
git diff --check
```

Expected: complete test suite exits zero, compileall exits zero, and `git diff --check` prints nothing.

- [ ] **Step 6: Record authoritative evidence and audit every acceptance criterion**

Write `docs/yolov8s-vision-verification.md` with:

```markdown
# YOLOv8s Vision Verification

- Date and environment: Python, ONNX Runtime, CUDA provider, GPU name.
- Asset hashes: SHA-256 for both ONNX files and annotation JSON files.
- CPU commands: exact commands and exit status for segmentation and pose.
- CUDA commands: exact commands, exit status, and active provider lists.
- Tensor evidence: input/output names, shapes, and dtypes for both models.
- Pipeline evidence: canonical decoder shapes and final evaluator metric keys.
- Regression evidence: full pytest pass count and duration.
- Acceptance audit: one evidence reference for every design acceptance criterion.
```

Do not paste model binaries, full logs, or downloaded dataset content. Record actual values from the fresh commands; do not infer or prefill them.

- [ ] **Step 7: Commit tests and verification evidence**

```bash
git add framework/tests/test_yolov8_onnx_runtime.py docs/yolov8s-vision-verification.md
git commit -m "test: verify yolov8s onnx cpu and cuda"
```

- [ ] **Step 8: Perform final fresh verification before completion claim**

```bash
cd framework
.venv/bin/python -m pytest tests/ -q
.venv/bin/python -m pytest tests/test_yolov8_onnx_runtime.py -v
cd ..
git status --short
git log --oneline --decorate -15
```

Expected: all framework and real-runtime tests pass, CPU and CUDA cases are not skipped, only pre-existing unrelated user changes remain unstaged, and the task commits are visible.

---

## Plan Completion Conditions

Implementation is complete only when all task checkboxes are backed by command output and commits, every design acceptance criterion has direct evidence in `docs/yolov8s-vision-verification.md`, the full suite passes freshly, and all four real model/provider combinations execute without skip or CPU fallback.
