# Mobilint MXQ Vision Pre/Post-processing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Mobilint Model Zoo의 ResNet50 ImageNet V2와 YOLOv5m DEFAULT MXQ를 ARIES/REGULUS에서 동일한 artifact profile, 정확한 `NHWC uint8` 전처리, YOLOv5 raw-head 후처리로 실행한다.

**Architecture:** 불변 `MobilintVisionArtifactProfile` registry가 model/task/artifact를 typed input/output recipe와 tensor contract로 연결한다. task별 loader가 기존 dataset/cursor/evaluator context 규약을 재사용하고, `MobilintRuntime`은 SDK metadata와 배열 계약만 검증하며, Mobilint YOLO decoder가 raw heads를 Model Zoo 방식으로 decode/NMS한다.

**Tech Stack:** Python 3.12, NumPy, Pillow, OpenCV 4.13, pytest 9, optional Mobilint qb Runtime 1.3.2

## Global Constraints

- Production에서 `mblt_model_zoo`, `MBLT_Engine`, Torch를 import하지 않는다.
- ARIES와 REGULUS는 같은 profile, preprocessor, loader, decoder를 공유한다.
- runtime은 cast, normalize, resize, letterbox, YOLO decode, NMS를 수행하지 않는다.
- ResNet50 입력 계약은 `(1,224,224,3)` `uint8`; YOLOv5m 입력 계약은 `(1,640,640,3)` `uint8`이다.
- YOLOv5m output 계약은 unbatched `(20,20,255)`, `(40,40,255)`, `(80,80,255)` 세 head다.
- YOLOv5m anchors는 stride 8/16/32에 각각 `[(10,13),(16,30),(33,23)]`, `[(30,61),(62,45),(59,119)]`, `[(116,90),(156,198),(373,326)]`다.
- YOLOv5m 기본값은 confidence `0.001`, IoU `0.65`, `max_nms=30000`, `max_det=300`, class offset `7680`이다.
- unknown Mobilint vision artifact, profile/layout/mode 충돌, malformed input/output은 generic fallback 없이 실패한다.
- 기존 generic, Hailo, DeepX, Mobilint NLP/LLM, monitor, native-async queue ownership 동작을 변경하지 않는다.
- 실제 장치 batch 지원 범위는 1이며 sync/native async가 같은 validation과 raw outputs를 사용한다.
- Mobilint SDK는 선택 의존성으로 유지하고 unit test에서는 fake `qbruntime`을 사용한다.
- 모든 production 변경은 해당 RED test가 의도한 이유로 실패한 것을 확인한 뒤 작성한다.

---

### Task 1: Artifact profile registry and immutable Model_Spec contract

**Files:**
- Create: `framework/src/dataloader/mobilint_vision_profiles.py`
- Create: `framework/tests/test_mobilint_vision_profiles.py`

**Interfaces:**
- Produces: `MobilintVisionArtifactProfile`, `ResNetCenterCropRecipe`, `YoloV5LetterboxRecipe`, `YoloV5RawHeadRecipe`
- Produces: `resolve_mobilint_vision_profile(...) -> MobilintVisionArtifactProfile`
- Produces: `apply_mobilint_vision_profile(spec, profile) -> Model_Spec`
- Produces: `profile.runtime_contract() -> dict[str, object]`

- [ ] **Step 1: Write profile resolution RED tests**

  Add tests with this public API and exact cases:

  ```python
  from pathlib import Path

  import pytest

  from core.model_spec import Model_Spec, Task
  from dataloader.mobilint_vision_profiles import (
      MOBILINT_RESNET50_IMAGENET1K_V2,
      MOBILINT_YOLOV5M_DEFAULT,
      apply_mobilint_vision_profile,
      resolve_mobilint_vision_profile,
  )


  @pytest.mark.parametrize("task,model,basename,expected", [
      (Task.IMAGE_CLASSIFICATION, "resnet50", "resnet50_IMAGENET1K_V2.mxq", MOBILINT_RESNET50_IMAGENET1K_V2),
      (Task.OBJECT_DETECTION, "YOLOv5m", "yolov5m.mxq", MOBILINT_YOLOV5M_DEFAULT),
  ])
  def test_auto_resolves_official_artifacts(task, model, basename, expected, tmp_path):
      artifact = tmp_path / basename
      artifact.touch()
      actual = resolve_mobilint_vision_profile(
          model_name=model,
          task=task,
          artifact_path=artifact,
          requested_profile="auto",
          requested_mode="auto",
          requested_layout="NCHW",
          layout_was_default=True,
      )
      assert actual is expected


  def test_explicit_profile_allows_renamed_artifact_but_not_wrong_task(tmp_path):
      renamed = tmp_path / "renamed.mxq"
      renamed.touch()
      actual = resolve_mobilint_vision_profile(
          model_name="yolov5m",
          task=Task.OBJECT_DETECTION,
          artifact_path=renamed,
          requested_profile="mobilint-yolov5m-default",
          requested_mode="raw",
          requested_layout="NHWC",
          layout_was_default=False,
      )
      assert actual is MOBILINT_YOLOV5M_DEFAULT
      with pytest.raises(ValueError, match="task"):
          resolve_mobilint_vision_profile(
              model_name="yolov5m",
              task=Task.IMAGE_CLASSIFICATION,
              artifact_path=renamed,
              requested_profile="mobilint-yolov5m-default",
              requested_mode="raw",
              requested_layout="NHWC",
              layout_was_default=False,
          )
  ```

  Add separate tests for unknown basename/model, normalized mode, explicit NCHW, and listed available profile IDs. Parameterize `mobilint-aries`/`mobilint-regulus` only at integration level because target family is intentionally absent from this resolver.

- [ ] **Step 2: Run the tests and verify RED**

  Run:

  ```bash
  cd framework
  uv run --with pytest==9.0.2 --with numpy --with pillow python -m pytest -q tests/test_mobilint_vision_profiles.py
  ```

  Expected: collection fails because `dataloader.mobilint_vision_profiles` does not exist.

- [ ] **Step 3: Implement typed frozen profiles and exact registry values**

  Implement these types and constants; store nested mappings as tuples so the frozen profile has no mutable payload.

  ```python
  @dataclass(frozen=True)
  class ResNetCenterCropRecipe:
      resize_short_side: int = 232
      crop_hw: tuple[int, int] = (224, 224)
      interpolation: str = "pil_bilinear"
      resize_rounding: str = "integer_truncation"
      crop_rounding: str = "python_round"
      version: str = "1"


  @dataclass(frozen=True)
  class YoloV5LetterboxRecipe:
      input_hw: tuple[int, int] = (640, 640)
      interpolation: str = "opencv_linear"
      resize_rounding: str = "python_round"
      padding_rounding: str = "ultralytics_minus_plus_0_1"
      pad_color: tuple[int, int, int] = (114, 114, 114)
      version: str = "1"


  @dataclass(frozen=True)
  class YoloV5RawHeadRecipe:
      class_count: int
      anchors_by_stride: tuple[tuple[int, tuple[tuple[int, int], ...]], ...]
      expected_heads: int = 3
      version: str = "1"


  @dataclass(frozen=True)
  class MobilintVisionArtifactProfile:
      profile_id: str
      model_name: str
      task: Task
      artifact_basenames: tuple[str, ...]
      preprocess_mode: str
      color_order: str
      input_layout: str
      input_dtype: str
      unbatched_input_shape: tuple[int, ...]
      max_batch_size: int
      input_recipe: ResNetCenterCropRecipe | YoloV5LetterboxRecipe
      expected_output_shapes: tuple[tuple[int, ...], ...] = ()
      output_recipe: YoloV5RawHeadRecipe | None = None
      decoder_defaults: tuple[tuple[str, float | int], ...] = ()

      def runtime_contract(self) -> dict[str, object]:
          contract: dict[str, object] = {
              "vision_profile_id": self.profile_id,
              "expected_input_dtype": self.input_dtype,
              "expected_input_layout": self.input_layout,
              "expected_unbatched_input_shape": list(self.unbatched_input_shape),
              "max_input_batch_size": self.max_batch_size,
          }
          if self.expected_output_shapes:
              contract["expected_unbatched_output_shapes"] = [
                  list(shape) for shape in self.expected_output_shapes
              ]
          return contract
  ```

  Instantiate exactly two profiles with the values from Global Constraints. Normalize model names using `"".join(ch for ch in value.casefold() if ch.isalnum())`. Auto resolution requires exact normalized model, exact `Task`, and exact case-sensitive artifact basename; explicit resolution skips only basename matching. Validate mode/layout after selecting the profile and include available IDs in every resolution error.

- [ ] **Step 4: Implement immutable Model_Spec replacement and test it**

  Preserve the first input name and replace the full output mapping only when the profile declares output heads.

  ```python
  def apply_mobilint_vision_profile(
      spec: Model_Spec,
      profile: MobilintVisionArtifactProfile,
  ) -> Model_Spec:
      if spec.task is not profile.task:
          raise ValueError(f"Profile {profile.profile_id!r} task mismatch.")
      input_name = next(iter(spec.input_shapes))
      input_shapes = dict(spec.input_shapes)
      input_dtype = dict(spec.input_dtype)
      input_shapes[input_name] = (1, *profile.unbatched_input_shape)
      input_dtype[input_name] = profile.input_dtype
      output_shapes = dict(spec.output_shapes)
      if profile.expected_output_shapes:
          output_shapes = {
              f"mobilint_yolov5_stride{640 // shape[0]}": (1, *shape)
              for shape in profile.expected_output_shapes
          }
      return replace(
          spec,
          input_shapes=input_shapes,
          input_dtype=input_dtype,
          output_shapes=output_shapes,
      )
  ```

  Assert the original frozen spec remains unchanged and the YOLO result has three outputs in qb Runtime metadata order: 20, 40, 80.

- [ ] **Step 5: Run focused tests and commit**

  ```bash
  cd framework
  uv run --with pytest==9.0.2 --with numpy --with pillow python -m pytest -q tests/test_mobilint_vision_profiles.py
  git add src/dataloader/mobilint_vision_profiles.py tests/test_mobilint_vision_profiles.py
  git commit -m "feat: add Mobilint vision artifact profiles"
  ```

  Expected: all focused tests pass.

---

### Task 2: Exact ResNet50 and YOLOv5m preprocessors with task loaders

**Files:**
- Create: `framework/src/preprocessor/mobilint_vision.py`
- Create: `framework/src/dataloader/mobilint_image_classification_loader.py`
- Create: `framework/src/dataloader/mobilint_object_detection_loader.py`
- Modify: `framework/src/preprocessor/__init__.py`
- Create: `framework/tests/test_mobilint_image_classification_loader.py`
- Create: `framework/tests/test_mobilint_object_detection_loader.py`

**Interfaces:**
- Consumes: Task 1 profiles and `runtime_contract()`
- Produces: `MobilintResNetCenterCropPreprocess(PreprocessStrategy)`
- Produces: `MobilintYoloV5Preprocessor(BasePreprocessor)`
- Produces: `MobilintImageClassificationLoader`, `MobilintObjectDetectionLoader`

- [ ] **Step 1: Write independent pixel-parity RED tests**

  The ResNet reference must not call production helpers:

  ```python
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
  ```

  Use horizontal, vertical, and odd-offset RGB gradients. Assert cached CHW output is pixel-identical, `uint8`, and contiguous; loader output is `(224,224,3)` NHWC.

  The YOLO reference uses OpenCV only in the test:

  ```python
  def model_zoo_letterbox_reference(rgb: np.ndarray):
      h0, w0 = rgb.shape[:2]
      ratio = min(640 / h0, 640 / w0)
      new_w, new_h = int(round(w0 * ratio)), int(round(h0 * ratio))
      resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
      dw, dh = (640 - new_w) / 2, (640 - new_h) / 2
      left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
      top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
      result = cv2.copyMakeBorder(
          resized, top, bottom, left, right, cv2.BORDER_CONSTANT,
          value=(114, 114, 114),
      )
      return result, ratio, left, top
  ```

  Assert `500x375` produces `(640,640,3)` `uint8`, `ratio_pad=((1.28,1.28),(0,80))`, flat evaluator context keys, and a cache path different from generic `letterbox_raw_NHWC_640x640.npz`. Add square and portrait cases that exercise `round(d±0.1)`.

- [ ] **Step 2: Run both new test files and verify RED**

  ```bash
  cd framework
  uv run --with pytest==9.0.2 --with numpy --with pillow --with opencv-python==4.13.0.92 python -m pytest -q tests/test_mobilint_image_classification_loader.py tests/test_mobilint_object_detection_loader.py
  ```

  Expected: collection fails because the Mobilint preprocessor and loader modules do not exist.

- [ ] **Step 3: Implement exact preprocessors and profile-aware cache keys**

  `MobilintResNetCenterCropPreprocess` validates the recipe and returns contiguous CHW `uint8`; its `cache_config()` includes every recipe field and `profile_id`.

  `MobilintYoloV5Preprocessor` accepts only the YOLO profile, uses `cv2.INTER_LINEAR`, returns contiguous HWC `uint8`, and persists this context in `.npz`:

  ```python
  context = {
      "original_width": int(w0),
      "original_height": int(h0),
      "input_width": 640,
      "input_height": 640,
      "scale": float(ratio),
      "pad_x": int(left),
      "pad_y": int(top),
      "layout": "NHWC",
      "resize_mode": "letterbox",
      "ratio_pad": ((float(ratio), float(ratio)), (int(left), int(top))),
      "profile_id": profile.profile_id,
  }
  ```

  Build its cache signature from a canonical JSON payload containing profile ID, recipe class/version, input size, interpolation, both rounding policies, pad color, layout, and dtype; hash with SHA-1 and use the first 10 hex characters. On cache load reconstruct `ratio_pad` as nested tuples and return a contiguous array.

- [ ] **Step 4: Implement thin task-specific loaders**

  Require a resolved profile rather than resolving twice:

  ```python
  class MobilintImageClassificationLoader(ImageClassificationLoader):
      def __init__(self, model_spec: Model_Spec, **kwargs):
          options = dict(kwargs)
          profile = options.pop("mobilint_vision_profile", None)
          if not isinstance(profile, MobilintVisionArtifactProfile):
              raise ValueError("Mobilint classification loader requires a resolved vision profile.")
          if profile.task is not Task.IMAGE_CLASSIFICATION:
              raise ValueError("Mobilint classification loader received a non-classification profile.")
          self.mobilint_vision_profile = profile
          options["layout"] = profile.input_layout
          options["preprocess_strategy"] = MobilintResNetCenterCropPreprocess(profile)
          super().__init__(model_spec, **options)

      def get_metadata(self) -> dict[str, Any]:
          metadata = super().get_metadata()
          metadata["mobilint_vision_profile"] = self.mobilint_vision_profile.profile_id
          metadata["runtime_options"] = self.mobilint_vision_profile.runtime_contract()
          return metadata
  ```

  The detection loader injects `MobilintYoloV5Preprocessor(profile)`, forces `backend="mobilint"`, raw/letterbox/NHWC, reuses parent label/cursor methods, and returns the same metadata keys/runtime contract. Reject profile/task mismatches before calling the parent.

- [ ] **Step 5: Run loader tests and existing context regressions**

  ```bash
  cd framework
  uv run --with pytest==9.0.2 --with numpy --with pillow --with opencv-python==4.13.0.92 python -m pytest -q tests/test_mobilint_image_classification_loader.py tests/test_mobilint_object_detection_loader.py tests/test_object_detection_loader.py tests/test_object_detection_loader_async.py tests/test_object_detection_evaluator.py tests/test_inference_pipeline.py
  ```

  Expected: all tests pass; generic/Hailo context remains unchanged.

- [ ] **Step 6: Commit preprocessors and loaders**

  ```bash
  git add framework/src/preprocessor/mobilint_vision.py framework/src/preprocessor/__init__.py framework/src/dataloader/mobilint_image_classification_loader.py framework/src/dataloader/mobilint_object_detection_loader.py framework/tests/test_mobilint_image_classification_loader.py framework/tests/test_mobilint_object_detection_loader.py
  git commit -m "feat: add Mobilint vision preprocessors"
  ```

---

### Task 3: Model Zoo-compatible YOLOv5 raw-head decoder

**Files:**
- Create: `framework/src/decoders/mobilint_yolov5.py`
- Modify: `framework/src/decoders/object_detection.py`
- Modify: `framework/src/decoders/__init__.py`
- Create: `framework/tests/test_mobilint_yolov5_decoder.py`
- Modify: `framework/tests/test_object_detection_decoders.py`

**Interfaces:**
- Consumes: `YoloV5RawHeadRecipe` and profile decoder defaults
- Produces: `MobilintYoloV5HeadDecoder(DetectionDecoder)`
- Produces: public `nms_pure_numpy(boxes, scores, iou_threshold) -> list[int]`

- [ ] **Step 1: Write raw-head decode and NMS RED tests**

  Create fixtures with three zero-logit heads in arbitrary dictionary order. Assert spatial matching ignores names/order, both `(H,W,255)` and `(1,H,W,255)` normalize, and the concatenated debug helper/result has 25,200 rows. For a selected stride-8 cell/anchor, set logits and assert:

  ```python
  expected_xy = (sigmoid(raw_xy) * 2.0 - 0.5 + grid_xy) * 8.0
  expected_wh = (sigmoid(raw_wh) * 2.0) ** 2 * np.array([10.0, 13.0])
  expected_score = sigmoid(raw_obj) * sigmoid(raw_class)
  ```

  Add independent behavior tests where objectness exceeds the threshold but the product does not, one anchor yields two class candidates, identical boxes of different classes survive class-aware NMS, identical boxes of the same class suppress, and `max_det` truncates. Add malformed head count, duplicate spatial size, channel, NCHW, and batch mismatch tests.

- [ ] **Step 2: Run decoder tests and verify RED**

  ```bash
  cd framework
  uv run --with pytest==9.0.2 --with numpy python -m pytest -q tests/test_mobilint_yolov5_decoder.py tests/test_object_detection_decoders.py
  ```

  Expected: Mobilint decoder import fails; existing decoder tests continue to pass when run alone.

- [ ] **Step 3: Expose the shared low-level NMS without changing legacy semantics**

  Rename `_nms_pure_numpy` to `nms_pure_numpy`, update current internal callers, and retain an alias for internal compatibility:

  ```python
  def nms_pure_numpy(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> List[int]:
      x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
      areas = np.maximum(x2 - x1, 0.0) * np.maximum(y2 - y1, 0.0)
      order = scores.argsort()[::-1]
      keep: List[int] = []
      while order.size > 0:
          index = int(order[0])
          keep.append(index)
          if order.size == 1:
              break
          xx1 = np.maximum(x1[index], x1[order[1:]])
          yy1 = np.maximum(y1[index], y1[order[1:]])
          xx2 = np.minimum(x2[index], x2[order[1:]])
          yy2 = np.minimum(y2[index], y2[order[1:]])
          width = np.maximum(0.0, xx2 - xx1)
          height = np.maximum(0.0, yy2 - yy1)
          intersection = width * height
          union = np.maximum(areas[index] + areas[order[1:]] - intersection, 1e-6)
          iou = intersection / union
          order = order[np.where(iou <= iou_threshold)[0] + 1]
      return keep


  _nms_pure_numpy = nms_pure_numpy
  ```

  The RED regression must prove `RawYoloDetectionDecoder` still performs its existing objectness-first, single-class, class-agnostic behavior.

- [ ] **Step 4: Implement complete Mobilint decode/filter/NMS**

  Constructor inputs are `profile`, `conf_threshold`, `iou_threshold`, `max_nms`, `max_det`, `max_class_offset`. Normalize heads by actual spatial dimensions, reshape `(B,H,W,3,85)`, apply stable sigmoid, decode anchors/grid, and concatenate.

  For every batch:

  ```python
  inverse_conf = np.log(conf_threshold / (1.0 - conf_threshold))
  candidates = decoded[raw_objectness_logits > inverse_conf]
  scores = candidates[:, 4:5] * candidates[:, 5:]
  anchor_indices, class_indices = np.nonzero(scores > conf_threshold)
  boxes = xywh_to_xyxy(candidates[anchor_indices, :4])
  candidate_scores = scores[anchor_indices, class_indices]
  order = np.argsort(candidate_scores)[::-1][:max_nms]
  offset_boxes = boxes[order] + class_indices[order, None] * max_class_offset
  keep = nms_pure_numpy(offset_boxes, candidate_scores[order], iou_threshold)[:max_det]
  ```

  Preserve un-offset boxes in canonical rows. Return `{DETECTIONS_KEY: float32_array_shape_N_by_7}`. Reject `conf_threshold` outside `(0,1)` and non-positive limits.

- [ ] **Step 5: Wire decoder factory to the selected Mobilint profile**

  In `create_object_detection_decoder`, when backend is `mobilint`, require `mobilint_vision_profile`; require its output recipe to be `YoloV5RawHeadRecipe`; merge explicit decoder kwargs over `dict(profile.decoder_defaults)`; instantiate `MobilintYoloV5HeadDecoder`. Leave Hailo and generic branches unchanged.

- [ ] **Step 6: Run focused tests and commit**

  ```bash
  cd framework
  uv run --with pytest==9.0.2 --with numpy python -m pytest -q tests/test_mobilint_yolov5_decoder.py tests/test_object_detection_decoders.py
  git add src/decoders/mobilint_yolov5.py src/decoders/object_detection.py src/decoders/__init__.py tests/test_mobilint_yolov5_decoder.py tests/test_object_detection_decoders.py
  git commit -m "feat: decode Mobilint YOLOv5 raw heads"
  ```

  Expected: all tests pass and existing decoder behavior is unchanged.

---

### Task 4: Factory and CLI pipeline integration

**Files:**
- Modify: `framework/src/dataloader/__init__.py`
- Modify: `framework/src/main.py`
- Modify: `framework/tests/test_main_paths.py`

**Interfaces:**
- Consumes: Tasks 1–3 public profile/loader/decoder APIs
- Produces: one selected profile object shared by spec, loader, runtime contract, and decoder

- [ ] **Step 1: Write factory/CLI RED tests**

  Add tests that `create_dataloader()` selects each Mobilint loader by task and keeps generic/Hailo/DeepX routing unchanged. Add parser tests for default `--image-preprocess-profile auto`. Use existing `StopAfterLoader`/monkeypatch main-path patterns to assert official ResNet and YOLO artifacts resolve on both `mobilint-aries` and `mobilint-regulus`, effective layout becomes NHWC, artifact-local spec is passed to `CompiledModel`, and the same profile object reaches loader and decoder.

  Add failures for unknown artifact, non-Mobilint explicit profile, explicit NCHW, normalized mode, and CLI attempts to override any protected key.

- [ ] **Step 2: Run main/factory tests and verify RED**

  ```bash
  cd framework
  uv run --with pytest==9.0.2 --with numpy --with pillow --with opencv-python==4.13.0.92 python -m pytest -q tests/test_main_paths.py tests/test_mobilint_image_classification_loader.py tests/test_mobilint_object_detection_loader.py
  ```

  Expected: parser/factory assertions fail because the profile is not wired.

- [ ] **Step 3: Export loaders and route Mobilint tasks**

  Add public imports/`__all__` entries. Before generic task routing:

  ```python
  if backend == "mobilint":
      if task is Task.IMAGE_CLASSIFICATION:
          return MobilintImageClassificationLoader(model_spec, **kwargs)
      if task is Task.OBJECT_DETECTION:
          return MobilintObjectDetectionLoader(model_spec, **kwargs)
      if task in {
          Task.SEMANTIC_SEGMENTATION,
          Task.INSTANCE_SEGMENTATION,
          Task.POSE_ESTIMATION,
      }:
          raise ValueError(f"Mobilint vision task {task.name} is not supported.")
  ```

- [ ] **Step 4: Resolve once after artifact selection and apply the spec**

  Add parser option `--image-preprocess-profile` with default `auto`. After final `artifact_path` is known and before `CompiledModel`, resolve only for Mobilint classification/detection, apply the profile, and set effective layout:

  ```python
  def _validate_image_preprocess_profile_scope(
      requested_profile: str,
      *,
      backend: str,
      task: Task,
  ) -> None:
      if str(requested_profile or "auto").strip().casefold() == "auto":
          return
      if backend != "mobilint" or task not in {
          Task.IMAGE_CLASSIFICATION,
          Task.OBJECT_DETECTION,
      }:
          raise ValueError(
              "--image-preprocess-profile is supported only for Mobilint raw vision targets."
          )
  ```

  Call this helper immediately after `task_enum` is known.

  ```python
  mobilint_vision_profile = None
  if args.backend == "mobilint" and task_enum in {
      Task.IMAGE_CLASSIFICATION,
      Task.OBJECT_DETECTION,
  }:
      mobilint_vision_profile = resolve_mobilint_vision_profile(
          model_name=args.model,
          task=task_enum,
          artifact_path=artifact_path,
          requested_profile=args.image_preprocess_profile,
          requested_mode=args.image_preprocess_mode,
          requested_layout=args.layout,
          layout_was_default=layout_was_default,
      )
      spec = apply_mobilint_vision_profile(spec, mobilint_vision_profile)
      args.layout = mobilint_vision_profile.input_layout
  ```

  Move the CLI banner after this resolution. Pass the exact object as `mobilint_vision_profile` in loader kwargs and decoder kwargs.

- [ ] **Step 5: Protect artifact contract runtime options**

  Add this set:

  ```python
  _MOBILINT_VISION_CONTRACT_OPTIONS = frozenset({
      "vision_profile_id",
      "expected_input_dtype",
      "expected_input_layout",
      "expected_unbatched_input_shape",
      "max_input_batch_size",
      "expected_unbatched_output_shapes",
  })
  ```

  In `_merge_runtime_option_layers`, for Mobilint vision reject CLI keys intersecting the set before merging, while permitting `core_mode`, `activation_slots`, and async tuning. Loader-owned values must merge normally and remain protected by existing locked device/family logic.

- [ ] **Step 6: Run integration regressions and commit**

  ```bash
  cd framework
  uv run --with pytest==9.0.2 --with numpy --with pillow --with opencv-python==4.13.0.92 python -m pytest -q tests/test_main_paths.py tests/test_mobilint_vision_profiles.py tests/test_mobilint_image_classification_loader.py tests/test_mobilint_object_detection_loader.py tests/test_mobilint_yolov5_decoder.py tests/test_hailo_image_loader.py tests/test_deepx_dxnn_metadata.py
  git add src/dataloader/__init__.py src/main.py tests/test_main_paths.py
  git commit -m "feat: wire Mobilint vision profiles"
  ```

---

### Task 5: MobilintRuntime input/output contract validation for sync and native async

**Files:**
- Modify: `framework/src/runtimes/mobilint_rt.py`
- Modify: `framework/tests/test_mobilint_runtime.py`
- Modify: `framework/tests/test_mobilint_native_backend.py`

**Interfaces:**
- Consumes: loader `runtime_contract()` keys from Task 1
- Produces: pre-SDK array validation and diagnostic expected/actual contract fields

- [ ] **Step 1: Extend fake qbruntime and write RED tests**

  Fake Model implements:

  ```python
  def get_model_input_shape(self):
      return state["input_shapes"]

  def get_model_input_data_type(self):
      return state["input_dtypes"]

  def get_model_output_shape(self):
      return state["output_shapes"]
  ```

  Test ResNet and YOLO matching contracts, each getter missing, dtype/shape/output-count/output-shape mismatch with dispose/session rollback, and unchanged NLP behavior when contract options are absent. Before `infer`, reject float32, NCHW, batch 2, and wrong input name; accept a non-contiguous view by making a contiguous copy. Test `_normalize_outputs` rejects raw output shapes that differ from the YOLO contract for sync and future-based native async.

- [ ] **Step 2: Run runtime tests and verify RED**

  ```bash
  cd framework
  uv run --with pytest==9.0.2 --with numpy python -m pytest -q tests/test_mobilint_runtime.py tests/test_mobilint_native_backend.py
  ```

  Expected: new metadata and array validation assertions fail.

- [ ] **Step 3: Parse the optional all-or-none contract**

  Normalize dtype through `np.dtype(value).name`; layout accepts only NCHW/NHWC; shapes accept non-empty positive integer lists/tuples but reject booleans. Require profile ID, dtype, layout, input shape, and positive max batch together. Output shapes may be empty, otherwise normalize a non-empty sequence of shapes. Store `_actual_input_dtype`, `_actual_input_shape`, `_actual_output_shapes`, and clear them in `_clear_model_state()`.

- [ ] **Step 4: Validate SDK metadata inside the existing rollback boundary**

  Add getter helpers that unwrap the SDK's one-input list and enum `.name`, require SDK v1.3-compatible getters when a contract exists, and compare output shapes as a multiset because SDK order is not semantically trusted. Call `_validate_model_contract()` after Model construction/launch but before exiting the current `try`; any mismatch therefore uses `_cleanup_resources()`.

  Error messages include profile ID, artifact basename, expected, and actual values.

- [ ] **Step 5: Validate actual arrays and normalized outputs**

  `_ordered_inputs()` first converts to contiguous arrays and then calls:

  ```python
  def _validate_runtime_input_array(self, name: str, array: np.ndarray) -> None:
      if array.dtype.name != self.expected_input_dtype:
          raise ValueError(
              f"Mobilint input {name!r} dtype mismatch for {self.vision_profile_id}: "
              f"expected {self.expected_input_dtype}, received {array.dtype.name}."
          )
      if array.ndim != len(self.expected_unbatched_input_shape) + 1:
          raise ValueError(
              f"Mobilint input {name!r} rank mismatch for {self.vision_profile_id}: "
              f"expected batch plus {self.expected_unbatched_input_shape}, received {array.shape}."
          )
      if not 1 <= array.shape[0] <= self.max_input_batch_size:
          raise ValueError(
              f"Mobilint input {name!r} batch mismatch for {self.vision_profile_id}: "
              f"expected 1..{self.max_input_batch_size}, received {array.shape[0]}."
          )
      if tuple(array.shape[1:]) != self.expected_unbatched_input_shape:
          raise ValueError(
              f"Mobilint input {name!r} shape mismatch for {self.vision_profile_id}: "
              f"expected {self.expected_unbatched_input_shape}, received {array.shape[1:]}."
          )
  ```

  `_normalize_outputs()` validates each returned array with an optional leading batch dimension and compares the unbatched shape multiset. Both `run()` and `MobilintNativeBackend` already use `_ordered_inputs()`/`_normalize_outputs()`, so do not duplicate validation in async code.

- [ ] **Step 6: Expose diagnostics, run tests, and commit**

  Add profile ID and expected/actual input/output contracts to `get_device_spec()`. Then run:

  ```bash
  cd framework
  uv run --with pytest==9.0.2 --with numpy python -m pytest -q tests/test_mobilint_runtime.py tests/test_mobilint_native_backend.py tests/test_mobilint_llm_runtime.py
  git add src/runtimes/mobilint_rt.py tests/test_mobilint_runtime.py tests/test_mobilint_native_backend.py
  git commit -m "feat: validate Mobilint vision tensor contracts"
  ```

  Expected: all tests pass and Mobilint LLM remains contract-free.

---

### Task 6: Documentation, full SDK-free verification, and ARIES2 acceptance commands

**Files:**
- Modify: `framework/src/runtimes/README.md`
- Modify: `framework/README.md`
- Modify: `docs/superpowers/specs/2026-07-22-mobilint-image-preprocessing-design.md`

**Interfaces:**
- Consumes: all prior tasks
- Produces: exact user-facing ResNet50/YOLOv5m commands and verified branch state

- [ ] **Step 1: Document exact supported profiles and boundaries**

  Add a table containing profile ID, official basename, task, input, outputs, default thresholds, and ARIES/REGULUS sharing. Document that Model Zoo is a parity oracle only, unknown artifacts fail, SDK metadata is required, compiler integration and other YOLO variants are out of scope.

  Add commands using:

  ```bash
  --target mobilint-aries
  --artifact framework/models/mobilint/resnet50/aries/resnet50_IMAGENET1K_V2.mxq
  --image-preprocess-profile auto
  --layout NHWC
  --no-compile
  ```

  and the YOLO artifact `framework/models/mobilint/yolov5m/aries/yolov5m.mxq`, COCO dataset, `--runtime-option core_mode=global8`, `--runtime-option conf_threshold=0.001`, and `--runtime-option iou_threshold=0.65`. Include sync, `--monitor`, and this native-async suffix:

  ```bash
  --inference-mode async_queue \
  --scenario offline \
  --queue-capacity 16 \
  --worker-count 1 \
  --max-samples 10
  ```

- [ ] **Step 2: Run static checks and the full relevant regression suite**

  ```bash
  git diff --check
  cd framework
  uv run --with-requirements requirements.txt python -m pytest -q tests/test_mobilint_vision_profiles.py tests/test_mobilint_image_classification_loader.py tests/test_mobilint_object_detection_loader.py tests/test_mobilint_yolov5_decoder.py tests/test_mobilint_runtime.py tests/test_mobilint_native_backend.py tests/test_mobilint_llm_runtime.py tests/test_main_paths.py tests/test_object_detection_decoders.py tests/test_object_detection_loader.py tests/test_object_detection_loader_async.py tests/test_object_detection_evaluator.py tests/test_inference_pipeline.py tests/test_hailo_image_loader.py tests/test_deepx_dxnn_metadata.py
  ```

  Expected: `git diff --check` has no output and every listed test passes. If a dependency installation is unavailable, record the exact command/error and still run every subset supported by the existing environment; do not claim the unavailable suite passed.

- [ ] **Step 3: Commit documentation**

  ```bash
  git add framework/src/runtimes/README.md framework/README.md docs/superpowers/specs/2026-07-22-mobilint-image-preprocessing-design.md
  git commit -m "docs: add Mobilint vision runtime guide"
  ```

- [ ] **Step 4: Prepare, but do not claim, hardware acceptance**

  Provide the ARIES2 commands for ResNet warmup 2/10 steps and YOLO sync/monitor/native-async. Ask the user to return SDK/driver versions, artifact hashes, Model Zoo/framework top-k or detection comparison, monitor power/utilization/energy, and native-async shutdown counts. Actual hardware success remains pending until those logs are supplied.
