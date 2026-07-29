# YOLOv8s Segmentation and Pose ONNX Pipeline Design

**Date:** 2026-07-29
**Status:** High-level design approved; written specification pending review
**Scope:** `yolov8s-seg` instance segmentation and `yolov8s-pose` pose estimation on ONNX Runtime CPU and CUDA

## Context

The framework already defines `INSTANCE_SEGMENTATION` and `POSE_ESTIMATION` tasks, but its general ONNX path does not implement either task end to end. The current model registry contains medium-sized YOLOv8 placeholders added for a DeepX path, the generic DataLoader factory rejects both tasks, the decoder factory only handles object detection, and the evaluator factory returns latency-only metrics. The existing COCO directory also contains bounding-box-only annotations and cannot prove segmentation or keypoint accuracy.

This change adds exact `yolov8s-seg` and `yolov8s-pose` profiles and completes the framework-native path from COCO image loading through preprocessing, ONNX inference, NumPy postprocessing, and official COCO evaluation. ResNet50 and YOLOv5m behavior remains unchanged.

## Goals

- Run `yolov8s-seg` and `yolov8s-pose` through the existing `BenchmarkRunner` and CLI without task-specific orchestration outside the framework.
- Provide DataLoader, preprocessing, decoder/postprocessor, evaluator, profile, model preparation, dataset preparation, and test coverage for both tasks.
- Preserve framework boundaries: runtime inputs and outputs are NumPy arrays or native Python metadata; Torch tensors do not cross component interfaces.
- Measure standard COCO instance-segmentation mask metrics and pose OKS metrics with `pycocotools`.
- Verify real exported ONNX models on both `CPUExecutionProvider` and `CUDAExecutionProvider`, rejecting CUDA-to-CPU fallback.
- Develop every behavior using a recorded RED-GREEN-REFACTOR TDD cycle.

## Non-goals

- Training or fine-tuning YOLO models.
- Reimplementing COCO metric mathematics.
- Supporting arbitrary segmentation or pose model output schemas in this change.
- Changing the existing ResNet50, YOLOv5m, Hailo, or DeepX contracts except where shared factory routing must recognize the two new profiles.
- Committing model weights, ONNX binaries, COCO images, or COCO annotations to Git.

## Accepted Assumptions

- The authoritative evaluation dataset is official COCO 2017 validation data.
- Instance segmentation uses `annotations/instances_val2017.json`.
- Pose estimation uses `annotations/person_keypoints_val2017.json` and COCO's 17-keypoint person schema.
- Both models use a 640 by 640 RGB input and Ultralytics-style letterbox preprocessing.
- Automated unit and integration tests use tiny deterministic COCO fixtures. Full COCO evaluation is available as a normal benchmark command but is not part of the fast test suite.
- Real runtime smoke verification processes a small number of genuine COCO images; it validates execution and result structure, while the official evaluator tests validate metric correctness independently.

## Chosen Approach

The implementation will use framework-native NumPy preprocessing and postprocessing, then convert compact predictions to official COCO result records for `pycocotools` evaluation.

Wrapping the Ultralytics Validator was rejected because it would couple normal benchmark execution to Torch, Ultralytics internal batch dictionaries, and version-sensitive validator behavior. A custom AP implementation was rejected because it would not provide authoritative equivalence with COCO mask AP and OKS AP. Ultralytics remains an asset-preparation dependency for downloading weights and exporting ONNX; it is not part of inference or evaluation component contracts.

## Architecture

### Profiles and Assets

`src/core/model_profiles.py` will add exact canonical profiles:

- `yolov8s-seg`: task `INSTANCE_SEGMENTATION`, input `(1, 3, 640, 640)`, prediction output `(1, 116, 8400)`, prototype output `(1, 32, 160, 160)`.
- `yolov8s-pose`: task `POSE_ESTIMATION`, input `(1, 3, 640, 640)`, output `(1, 56, 8400)`.

Model-spec creation will bind all ONNX output names by graph order instead of assuming literal names such as `output0` and `output1`. It will validate the number of outputs expected by each profile. Existing model profiles and aliases remain available.

`models/prepare_yolov8_vision.py` will export `yolov8s-seg.pt` and `yolov8s-pose.pt` as dynamic-batch ONNX models at the profile paths. Re-running the script is idempotent when valid outputs already exist.

The COCO preparation script will download official `val2017.zip` and `annotations_trainval2017.zip`, extract only into the configured dataset root, and verify that the image directory and both required validation annotation files exist. Downloaded artifacts remain ignored by Git.

### Dataset Resolution and Loader Contracts

Dataset resolution will recognize both new tasks and select:

- images: `<dataset>/images/val2017`
- segmentation labels: `<dataset>/annotations/instances_val2017.json`
- pose labels: `<dataset>/annotations/person_keypoints_val2017.json`

Explicit CLI image and label paths continue to take precedence. A missing directory, missing annotation file, malformed JSON structure, duplicate image ID, or image referenced by the annotation file but absent on disk raises a targeted error before benchmarking begins.

The two concrete loaders share a focused COCO vision base and expose the existing DataLoader API. Each sample has this shape:

```python
{
    "input": np.ndarray,  # contiguous float32 CHW, values in [0, 1]
    "label": {
        "image_id": int,
        "file_name": str,
    },
    "preprocess_context": {
        "original_height": int,
        "original_width": int,
        "input_height": 640,
        "input_width": 640,
        "scale": float,
        "pad_x": float,
        "pad_y": float,
    },
    "img_path": str,
}
```

The loader is lazy, supports sequential batches and `load_by_index()`, and does not retain decoded images. Metadata includes total samples, task, annotation file, input geometry, category mapping, and `is_static_batched=False`. Cache keys include preprocessing version, resize mode, layout, dtype, and target size so stale object-detection caches cannot be reused accidentally.

### Preprocessing

A shared YOLO vision preprocessor will:

1. Decode the source image as RGB.
2. Calculate `scale = min(640 / original_width, 640 / original_height)`.
3. Resize with bilinear interpolation using Ultralytics-compatible rounded dimensions.
4. Apply centered padding with RGB value `(114, 114, 114)` and preserve the exact padding offsets.
5. Convert HWC RGB bytes to contiguous CHW `float32` divided by 255.

It returns both the model tensor and the coordinate restoration context. Cached tensors and context are written together in `.npz` files. Corrupt or schema-incompatible cache entries are ignored and rebuilt atomically.

### Segmentation Postprocessing

The segmentation decoder consumes the raw prediction tensor and mask prototypes, resolving them by validated shape rather than dictionary order. For each batch item it will:

1. Interpret 116 channels as 4 box coordinates, 80 COCO class scores, and 32 mask coefficients.
2. Filter by confidence, convert boxes from center-width-height to `xyxy`, and run class-aware NumPy NMS.
3. Limit results to the configured maximum detections.
4. Multiply retained coefficients by the 32 mask prototypes.
5. Upsample mask logits to the 640 by 640 input space, crop them to their retained boxes, and threshold logits at zero.

The canonical decoder result is:

```python
{
    "detections": np.ndarray,  # (N, 7): local image, class, score, x1, y1, x2, y2
    "masks": np.ndarray,       # (N, 640, 640), uint8, aligned row-for-row
}
```

Malformed output ranks, ambiguous prediction/prototype outputs, non-finite values, invalid feature counts, or row-count mismatches raise clear `ValueError`s. An image with no retained instances returns correctly shaped empty arrays.

### Pose Postprocessing

The pose decoder interprets 56 channels as 4 box coordinates, one person score, and 17 `(x, y, visibility/confidence)` keypoints. It applies confidence filtering, box conversion, class-aware NumPy NMS, and maximum-detection limiting. Keypoint coordinates remain in model-input pixels and keypoint confidence remains unchanged.

The canonical result is:

```python
{
    "detections": np.ndarray,  # (N, 7), class is the local COCO person class
    "keypoints": np.ndarray,   # (N, 17, 3), aligned row-for-row
}
```

Invalid ranks, feature counts, non-finite coordinate data, and row mismatches fail explicitly. Empty predictions retain shapes `(0, 7)` and `(0, 17, 3)`.

### Coordinate Restoration and Evaluation

Evaluators receive decoder output plus per-sample `image_id` and preprocessing context through the existing `InferencePipeline.prepare_eval_labels()` path.

The segmentation evaluator removes letterbox padding, resizes each binary mask to the exact original image size with nearest-neighbor interpolation, converts it to Fortran-contiguous COCO RLE, maps the contiguous YOLO class to the non-contiguous official COCO category ID, and accumulates lightweight result dictionaries. It also restores boxes to original coordinates for valid COCO result records.

The pose evaluator removes padding and divides keypoint and box coordinates by the recorded scale. It maps the single local person class to COCO category ID 1 and flattens keypoints to the official 51-value representation.

At `compute()`, each evaluator invokes `COCOeval` only for image IDs actually seen in the benchmark. This makes `--max-steps` produce a meaningful subset evaluation while a full run evaluates all `val2017` images. Returned metrics include:

- segmentation: `Mask mAP`, `Mask AP50`, `Mask AP75`, `Mask AP Small`, `Mask AP Medium`, `Mask AP Large`.
- pose: `OKS mAP`, `OKS AP50`, `OKS AP75`, `OKS AP Medium`, `OKS AP Large`.
- both: total samples, average detections, and the framework's existing latency/throughput metrics.

No-prediction runs return zero AP values and valid sample/timing counts rather than failing inside `pycocotools`. Evaluator state is reset between `evaluate()` calls and remains streaming: dense masks are encoded and released per batch.

### Framework Integration

- `create_dataloader()` routes `INSTANCE_SEGMENTATION` and `POSE_ESTIMATION` to their concrete loaders for ONNX Runtime.
- `create_decoder()` returns the corresponding task decoder.
- `create_evaluator()` returns the corresponding COCO evaluator instead of `LatencyOnlyEvaluator`.
- `main.py` forwards image preprocessing options for all three YOLO vision tasks and passes the resolved annotation file to the evaluator.
- Existing `BenchmarkRunner`, synchronous completion path, and asynchronous completion path remain unchanged. Both modes use the same decoder and evaluator contracts.
- `run_all_onnx_benchmarks.py` adds entries for both profiles with exact asset checks and preparation hints.

## Dependency Policy

`pycocotools==2.0.11` is added as a pinned framework dependency for authoritative RLE encoding and COCO evaluation. NumPy, Pillow, and OpenCV already exist in the environment and cover tensor manipulation and resizing. Torch and Ultralytics are used only by the model export script, not by production inference, postprocessing, or evaluation modules.

If `pycocotools` is unavailable, creating a segmentation or pose evaluator raises an actionable dependency error. It must not silently fall back to a custom metric.

## Error Handling

- Asset preparation validates archive paths and expected outputs and never reports success for a partial extraction.
- Profile creation validates ONNX input/output counts and static feature dimensions when those dimensions are available.
- Loaders validate dataset structure before the first sample and identify the offending path or image ID.
- Preprocessing rejects non-positive image dimensions and unsupported layouts.
- Decoders reject ambiguous or incompatible tensors instead of choosing the first output silently.
- Evaluators validate image IDs against the loaded COCO ground truth and validate detection-to-mask/keypoint row alignment.
- CUDA verification requires `CUDAExecutionProvider` in both the available provider list and the loaded session's active provider list; CPU fallback is an error.

## TDD Strategy

Every production behavior follows the same cycle: add one focused failing test, run it and confirm the expected feature-related failure, implement the minimum code, run the focused test to green, then run the affected suite before refactoring.

Test layers are:

1. Profile and factory tests for exact model names, tasks, multi-output ONNX binding, dataset routing, loader routing, decoder routing, and evaluator routing.
2. Preprocessor tests with synthetic non-square RGB images to verify pixel layout, padding, scale, offsets, dtype, contiguity, and cache invalidation.
3. Loader tests with tiny COCO JSON fixtures for ordering, image IDs, missing assets, sequential batching, random access, and metadata.
4. Decoder tests with constructed tensors for confidence filtering, class-aware NMS, mask coefficient/prototype composition, keypoint reshaping, empty results, and malformed shapes.
5. Evaluator tests with tiny valid COCO ground truth where perfect predictions produce the expected official metrics, shifted predictions reduce metrics, empty predictions produce zeros, and subset image IDs exclude unseen images.
6. Pipeline tests proving metadata survives collation and that synchronous and asynchronous completion feed identical canonical outputs to evaluators.
7. Real-asset ONNX smoke tests for both models on CPU and CUDA.

Tests assert observable behavior rather than mocked call counts. External downloads are not performed by the normal test suite.

## Real Runtime Verification

After unit and integration suites pass, the exact exported assets will be checked with these equivalent CLI runs for each model:

```bash
python src/main.py --model yolov8s-seg --target cpu \
  --dataset datasets/coco --max-steps 1 --warmup 1
python src/main.py --model yolov8s-seg --target cuda \
  --dataset datasets/coco --max-steps 1 --warmup 1
python src/main.py --model yolov8s-pose --target cpu \
  --dataset datasets/coco --max-steps 1 --warmup 1
python src/main.py --model yolov8s-pose --target cuda \
  --dataset datasets/coco --max-steps 1 --warmup 1
```

Verification evidence records the model path, real image ID, input/output tensor shapes and dtypes, active provider list, non-crashing decoder/evaluator result, and final metrics. CUDA verification passes only when the loaded session reports `CUDAExecutionProvider`; merely having the provider installed is insufficient.

## Acceptance Criteria

- Exact CLI model names `yolov8s-seg` and `yolov8s-pose` resolve without custom code outside registries and factories.
- Official COCO validation assets can be prepared by repository scripts and are validated before use.
- Both DataLoaders produce contiguous `(3, 640, 640)` `float32` inputs plus image and coordinate metadata.
- Segmentation decoder produces aligned boxes, scores, classes, and binary instance masks from real ONNX outputs.
- Pose decoder produces aligned boxes, scores, and 17-keypoint predictions from real ONNX outputs.
- Official `pycocotools` evaluators return the specified mask and OKS metrics on deterministic fixtures and real-data smoke runs.
- Focused tests contain recorded failing and passing executions for every implementation task, and the complete framework test suite passes afterward.
- Each model completes a real ONNX Runtime CPU run and a real CUDA run with the requested provider active.
- Existing ResNet50 and YOLOv5m test coverage remains green.
- Documentation contains preparation, benchmark, and verification commands and clearly distinguishes smoke validation from full COCO evaluation.
