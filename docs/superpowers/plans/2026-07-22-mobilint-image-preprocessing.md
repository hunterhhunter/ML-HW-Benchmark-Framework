# Mobilint MXQ Image Preprocessing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Mobilint Model Zoo의 `resnet50_IMAGENET1K_V2.mxq` 입력 계약을 프레임워크 전처리 계층에 구현해 ARIES와 REGULUS에서 동일한 `NHWC uint8` 입력으로 동기 및 native-async 추론을 실행한다.

**Architecture:** artifact basename과 model 이름으로 불변 `MobilintImageInputConfig`를 해석하고, 전용 이미지 분류 loader가 Model Zoo와 픽셀 단위로 같은 전처리를 수행한다. `main.py`는 같은 config로 artifact-local `Model_Spec`과 runtime options를 만들며, `MobilintRuntime`은 SDK가 보고한 MXQ 계약과 실제 추론 배열을 검증만 하고 cast나 normalize는 하지 않는다.

**Tech Stack:** Python 3.12, NumPy, Pillow, pytest, Mobilint qb Runtime 1.3.2(fake SDK unit tests + ARIES2 hardware acceptance)

## Global Constraints

- Production에서 Model Zoo의 `MBLT_Engine` 또는 `model.preprocess()`를 생성하지 않는다. 이 API는 별도 MXQ model lifecycle과 NPU launch를 소유한다.
- ARIES/REGULUS 분기는 전처리에 추가하지 않는다. 프로파일은 artifact 입력 계약에 귀속한다.
- runtime은 `float32`를 `uint8`로 cast하거나 입력을 normalize하지 않는다.
- 등록되지 않은 Mobilint image-classification MXQ는 generic ImageNet 전처리로 fallback하지 않는다.
- 기존 generic, Hailo, DeepX, Mobilint NLP/다중 입력 경로는 입력 계약 옵션이 없을 때 현재 동작을 유지한다.
- 첫 프로파일과 실제 장치 인수 범위의 batch size는 1이다.
- Mobilint SDK는 선택 의존성으로 유지하며 unit test에서는 fake `qbruntime`을 사용한다.
- 각 RED 단계는 해당 assertion이 의도한 이유로 실패하는지 확인한 뒤 production 코드를 작성한다.

---

## Task 1: Model Zoo-compatible Mobilint profile and preprocessing strategy

**Files:**

- Create: `framework/src/dataloader/mobilint_image_classification_loader.py`
- Create: `framework/tests/test_mobilint_image_classification_loader.py`

- [ ] **Step 1: Write resolver and preprocessing RED tests**

  `framework/tests/test_mobilint_image_classification_loader.py`에 임시 이미지와 `Model_Spec` fixture를 만들고 다음 계약을 먼저 고정한다.

  ```python
  import numpy as np
  import pytest
  from PIL import Image

  from core.model_spec import Model_Spec, Task
  from dataloader.mobilint_image_classification_loader import (
      MOBILINT_RESNET50_IMAGENET1K_V2_PROFILE,
      MobilintImageClassificationLoader,
      MobilintResNet50V2Preprocess,
      apply_mobilint_image_input_config,
      resolve_mobilint_image_input_config,
  )
  ```

  반드시 포함할 test case:

  - `auto` + model `resnet50` + basename `resnet50_IMAGENET1K_V2.mxq`가 profile을 선택한다.
  - 명시적 `mobilint-resnet50-imagenet1k-v2`는 rename된 `.mxq`에도 적용된다.
  - `auto`에서 다른 basename 또는 다른 model 이름은 사용 가능한 profile ID가 포함된 `ValueError`를 낸다.
  - `requested_mode="normalized"`는 raw profile과 충돌해 `ValueError`를 낸다.
  - 명시적 `requested_layout="NCHW"`, `layout_was_default=False`는 NHWC profile과 충돌한다.
  - parser 기본 layout에서 전달된 `NCHW`, `layout_was_default=True`는 config의 `NHWC`로 해석된다.
  - `apply_mobilint_image_input_config()`가 첫 input 이름을 보존하면서 shape을 `(1, 224, 224, 3)`, dtype을 `uint8`로 바꾼 새 frozen spec을 반환하고 원본 spec은 변경하지 않는다.

  Model Zoo와 독립적인 reference helper를 test 안에 둔다. 구현 코드의 helper를 호출해 expected 값을 만들지 않는다.

  ```python
  def _model_zoo_reference(image: Image.Image) -> np.ndarray:
      image = image.convert("RGB")
      width, height = image.size
      if width < height:
          resized = (232, int(232 * height / width))
      else:
          resized = (int(232 * width / height), 232)
      image = image.resize(resized, Image.Resampling.BILINEAR)
      left = round((resized[0] - 224) / 2)
      top = round((resized[1] - 224) / 2)
      image = image.crop((left, top, left + 224, top + 224))
      return np.transpose(np.asarray(image, dtype=np.uint8), (2, 0, 1))
  ```

  `301x500`과 `500x301` RGB gradient 이미지를 사용해 strategy 결과가 reference와 `np.array_equal`이고, 결과가 `(3, 224, 224)`, `np.uint8`, C-contiguous인지 검사한다. 홀수 차이에서 `round`와 floor가 달라지는 크기를 써 crop 규칙 회귀도 잡는다.

- [ ] **Step 2: Run the new tests and confirm RED**

  Run:

  ```bash
  cd framework
  python -m pytest -q tests/test_mobilint_image_classification_loader.py
  ```

  Expected: collection이 `ModuleNotFoundError: No module named 'dataloader.mobilint_image_classification_loader'`로 실패한다. pytest가 없는 shell이면 작업용 venv에서 pytest를 준비한 뒤 같은 명령을 실행하고, 이 import 실패가 아닌 환경 오류를 RED 증거로 사용하지 않는다.

- [ ] **Step 3: Implement the immutable profile registry and resolver**

  `framework/src/dataloader/mobilint_image_classification_loader.py`에 다음 public API를 만든다.

  ```python
  from dataclasses import dataclass, replace
  from pathlib import Path
  from typing import Any

  import numpy as np
  from PIL import Image

  from core.model_spec import Model_Spec
  from .image_classification_loader import ImageClassificationLoader
  from .preprocess_strategies import PreprocessStrategy


  MOBILINT_RESNET50_IMAGENET1K_V2_PROFILE = (
      "mobilint-resnet50-imagenet1k-v2"
  )


  @dataclass(frozen=True)
  class MobilintImageInputConfig:
      profile_id: str
      model_name: str
      preprocess_mode: str
      resize_short_side: int
      crop_hw: tuple[int, int]
      color_order: str
      input_layout: str
      input_dtype: str
      unbatched_input_shape: tuple[int, ...]

      @property
      def runtime_options(self) -> dict[str, Any]:
          return {
              "expected_input_dtype": self.input_dtype,
              "expected_input_layout": self.input_layout,
              "expected_unbatched_input_shape": self.unbatched_input_shape,
              "max_input_batch_size": 1,
          }
  ```

  Registry에는 정확히 첫 profile 하나를 등록한다.

  ```python
  _PROFILES = {
      MOBILINT_RESNET50_IMAGENET1K_V2_PROFILE: MobilintImageInputConfig(
          profile_id=MOBILINT_RESNET50_IMAGENET1K_V2_PROFILE,
          model_name="resnet50",
          preprocess_mode="raw",
          resize_short_side=232,
          crop_hw=(224, 224),
          color_order="RGB",
          input_layout="NHWC",
          input_dtype="uint8",
          unbatched_input_shape=(224, 224, 3),
      )
  }
  _AUTO_ARTIFACTS = {
      ("resnet50", "resnet50_IMAGENET1K_V2.mxq"):
          MOBILINT_RESNET50_IMAGENET1K_V2_PROFILE,
  }
  ```

  Resolver signature는 다음으로 고정한다.

  ```python
  def resolve_mobilint_image_input_config(
      *,
      model_name: str,
      artifact_path: str | Path,
      requested_profile: str = "auto",
      requested_mode: str = "auto",
      requested_layout: str = "NCHW",
      layout_was_default: bool = False,
  ) -> MobilintImageInputConfig:
      normalized_model = str(model_name).strip().lower()
      normalized_profile = str(requested_profile or "auto").strip().lower()
      available = ", ".join(sorted(_PROFILES))
      if normalized_profile == "auto":
          key = (normalized_model, Path(artifact_path).name)
          profile_id = _AUTO_ARTIFACTS.get(key)
          if profile_id is None:
              raise ValueError(
                  "No automatic Mobilint image profile for "
                  f"model={normalized_model!r}, "
                  f"artifact={Path(artifact_path).name!r}; "
                  f"available profiles: {available}. Select one with "
                  "--image-preprocess-profile."
              )
      else:
          profile_id = normalized_profile
          if profile_id not in _PROFILES:
              raise ValueError(
                  f"Unknown Mobilint image profile {profile_id!r}; "
                  f"available profiles: {available}."
              )

      config = _PROFILES[profile_id]
      if normalized_model != config.model_name:
          raise ValueError(
              f"Profile {profile_id!r} requires model "
              f"{config.model_name!r}, received {normalized_model!r}."
          )

      normalized_mode = str(requested_mode or "auto").strip().lower()
      if normalized_mode not in {"auto", config.preprocess_mode}:
          raise ValueError(
              f"Profile {profile_id!r} requires preprocess mode "
              f"{config.preprocess_mode!r}, received {normalized_mode!r}."
          )

      normalized_layout = str(requested_layout or "NCHW").strip().upper()
      if normalized_layout not in {"NCHW", "NHWC"}:
          raise ValueError(
              f"Unsupported image layout {normalized_layout!r}."
          )
      if not layout_was_default and normalized_layout != config.input_layout:
          raise ValueError(
              f"Profile {profile_id!r} requires layout "
              f"{config.input_layout}, received explicit "
              f"{normalized_layout}."
          )
      return config
  ```

  입력 문자열은 profile/model은 trim, mode/profile ID는 lowercase, layout은 uppercase로 정규화한다. `auto`는 `_AUTO_ARTIFACTS[(model_name, Path(artifact_path).name)]`만 인정한다. 명시 profile은 registry lookup만 한다. 선택 후 model 이름, mode(`auto` 또는 `raw`), explicit layout을 검증하고 config 자체를 반환한다. 오류에는 artifact basename, 요청값, `available profiles: mobilint-resnet50-imagenet1k-v2`를 포함한다.

  `apply_mobilint_image_input_config()`는 첫 input name을 찾아 `dataclasses.replace()`로 dict도 새로 생성한다.

  ```python
  def apply_mobilint_image_input_config(
      model_spec: Model_Spec,
      config: MobilintImageInputConfig,
  ) -> Model_Spec:
      input_name = next(iter(model_spec.input_shapes))
      input_shapes = dict(model_spec.input_shapes)
      input_dtype = dict(model_spec.input_dtype)
      input_shapes[input_name] = (1, *config.unbatched_input_shape)
      input_dtype[input_name] = config.input_dtype
      return replace(
          model_spec,
          input_shapes=input_shapes,
          input_dtype=input_dtype,
      )
  ```

- [ ] **Step 4: Implement exact Model Zoo pixel preprocessing**

  같은 파일에 `MobilintResNet50V2Preprocess(PreprocessStrategy)`를 구현한다. generic `_resize_short_side_center_crop()`은 resize에서 `round`, crop에서 floor를 사용하므로 재사용하지 않는다.

  구현 순서는 `RGB -> integer-truncating short-side resize -> PIL bilinear -> round-based center crop -> uint8 HWC -> contiguous CHW`이다. `target_hw`가 config의 `crop_hw`와 다르면 silently adapt하지 말고 `ValueError`로 실패한다.

  ```python
  class MobilintResNet50V2Preprocess(PreprocessStrategy):
      CACHE_VERSION = 1

      def __init__(self, config: MobilintImageInputConfig):
          self.config = config

      def cache_config(self) -> dict[str, Any]:
          return {
              "version": self.CACHE_VERSION,
              "profile_id": self.config.profile_id,
              "resize_short_side": self.config.resize_short_side,
              "crop_hw": list(self.config.crop_hw),
              "color_order": self.config.color_order,
              "input_dtype": self.config.input_dtype,
              "input_layout": self.config.input_layout,
          }

      def __call__(self, img, target_hw, mean, std) -> np.ndarray:
          del mean, std
          if tuple(target_hw) != self.config.crop_hw:
              raise ValueError(
                  f"Profile {self.config.profile_id!r} requires crop "
                  f"{self.config.crop_hw}, received {tuple(target_hw)}."
              )
          image = img.convert(self.config.color_order)
          width, height = image.size
          short_side = self.config.resize_short_side
          if width < height:
              resized = (short_side, int(short_side * height / width))
          else:
              resized = (int(short_side * width / height), short_side)
          image = image.resize(resized, Image.Resampling.BILINEAR)
          crop_h, crop_w = self.config.crop_hw
          left = round((resized[0] - crop_w) / 2)
          top = round((resized[1] - crop_h) / 2)
          image = image.crop((left, top, left + crop_w, top + crop_h))
          hwc = np.asarray(image, dtype=np.uint8)
          return np.ascontiguousarray(np.transpose(hwc, (2, 0, 1)))
  ```

  `ImagePreprocessor` cache key는 strategy class와 `cache_config()`를 이미 포함하므로 core cache 코드는 수정하지 않는다. 새 test에서 generic `MLPerfResNet50Preprocess`의 cache path와 Mobilint strategy cache path가 다름을 확인한다.

- [ ] **Step 5: Implement the Mobilint loader and metadata contract**

  `MobilintImageClassificationLoader`는 resolver를 내부에서 다시 실행하지 않고 확정된 `mobilint_input_config`를 필수로 받는다. 이는 main, spec, loader가 서로 다른 결정을 내리지 않게 한다.

  ```python
  class MobilintImageClassificationLoader(ImageClassificationLoader):
      def __init__(self, model_spec: Model_Spec, **kwargs):
          options = dict(kwargs)
          config = options.pop("mobilint_input_config", None)
          if not isinstance(config, MobilintImageInputConfig):
              raise ValueError(
                  "MobilintImageClassificationLoader requires a resolved "
                  "MobilintImageInputConfig."
              )
          self.mobilint_input_config = config
          options["layout"] = config.input_layout
          options["preprocess_strategy"] = (
              MobilintResNet50V2Preprocess(config)
          )
          super().__init__(model_spec, **options)

      def get_metadata(self) -> dict[str, Any]:
          metadata = super().get_metadata()
          config = self.mobilint_input_config
          metadata["mobilint_image_input"] = {
              "profile_id": config.profile_id,
              "preprocess_mode": config.preprocess_mode,
              "input_dtype": config.input_dtype,
              "input_layout": config.input_layout,
              "unbatched_input_shape": list(config.unbatched_input_shape),
          }
          metadata["runtime_options"] = dict(config.runtime_options)
          return metadata
  ```

  Loader integration test는 `load_single()` 결과가 `(224, 224, 3)` `uint8` contiguous이고 metadata가 위 계약과 같은지 확인한다.

- [ ] **Step 6: Run focused tests and commit**

  Run:

  ```bash
  cd framework
  python -m pytest -q tests/test_mobilint_image_classification_loader.py
  ```

  Expected: all tests pass.

  Commit:

  ```bash
  git add framework/src/dataloader/mobilint_image_classification_loader.py framework/tests/test_mobilint_image_classification_loader.py
  git commit -m "feat: add Mobilint image input profiles"
  ```

---

## Task 2: Wire the profile through factory, CLI, and artifact-local Model_Spec

**Files:**

- Modify: `framework/src/dataloader/__init__.py`
- Modify: `framework/src/main.py`
- Modify: `framework/tests/test_mobilint_image_classification_loader.py`
- Modify: `framework/tests/test_main_paths.py`

- [ ] **Step 1: Write factory and CLI RED tests**

  Factory test:

  - test fixture의 `spec`, `dataset_root`, `image_dir`, `label_path`를 넣어 `create_dataloader(spec, backend="mobilint", mobilint_input_config=config, dataset_path=str(dataset_root), image_dir=str(image_dir), label_path=str(label_path))`를 호출하면 `MobilintImageClassificationLoader`를 반환한다.
  - 같은 spec의 `backend="onnxruntime"`은 `ImageClassificationLoader`를 유지한다.

  `framework/tests/test_main_paths.py`에는 다음 test를 추가한다.

  - `build_parser()`가 `--image-preprocess-profile`을 받고 default가 `auto`다.
  - Mobilint image target에 explicit profile을 주고 `--layout NCHW`를 명시하면 resolver 오류가 CLI 실행 전에 드러난다.
  - non-Mobilint backend에 non-auto profile을 지정하면 무시하지 않고 scope 오류가 난다.
  - 공식 basename의 임시 MXQ, `resnet50` fake spec, default layout으로 main path를 실행하면 loader factory가 받은 값은 `layout="NHWC"`, input shape `(1, 224, 224, 3)`, dtype `uint8`, 동일 config 객체다. 이 test는 `mobilint-aries`와 `mobilint-regulus`로 parameterize해 두 target이 같은 profile resolver와 loader를 사용하는지 확인한다.

  마지막 test는 기존 `StopAfterLoader` 패턴과 `monkeypatch`를 사용한다. `create_model_spec`, `resolve_dataset_paths`, `create_dataloader`를 fake로 바꾸고, runtime 생성 전에 capture가 끝나게 한다. artifact는 `tmp_path / "resnet50_IMAGENET1K_V2.mxq"`에 실제 빈 파일을 만들어 `CompiledModel` 존재 검사를 통과시킨다.

- [ ] **Step 2: Run focused tests and confirm RED**

  Run:

  ```bash
  cd framework
  python -m pytest -q \
    tests/test_mobilint_image_classification_loader.py \
    tests/test_main_paths.py
  ```

  Expected: Mobilint factory가 generic loader를 반환하고 parser에 새 옵션이 없어 새 assertions가 실패한다.

- [ ] **Step 3: Export the loader and add the factory branch**

  `framework/src/dataloader/__init__.py`에서 새 public symbols를 import/export한다.

  ```python
  from .mobilint_image_classification_loader import (
      MobilintImageClassificationLoader,
      MobilintImageInputConfig,
      apply_mobilint_image_input_config,
      resolve_mobilint_image_input_config,
  )
  ```

  DeepX와 Hailo 분기 뒤, generic image-classification 분기 앞에 다음을 추가한다.

  ```python
  if str(kwargs.get("backend", "")).lower() == "mobilint":
      if task == Task.IMAGE_CLASSIFICATION:
          return MobilintImageClassificationLoader(model_spec, **kwargs)
  ```

  `__all__`에도 네 symbols를 추가한다. object detection과 NLP task routing은 변경하지 않는다.

- [ ] **Step 4: Add CLI profile scope and protected contract options**

  `build_parser()`의 기존 이미지 전처리 옵션 옆에 추가한다.

  ```python
  parser.add_argument(
      "--image-preprocess-profile",
      default="auto",
      help=(
          "Artifact-specific image preprocessing profile. "
          "Mobilint MXQ defaults to exact model/artifact auto-detection."
      ),
  )
  ```

  `main.py`의 top-level import를 다음처럼 확장한다.

  ```python
  from dataloader import (
      apply_mobilint_image_input_config,
      create_dataloader,
      resolve_mobilint_image_input_config,
  )
  ```

  scope validator를 작은 pure helper로 추가하고 `task_enum` 확정 직후 호출한다.

  ```python
  def _validate_image_preprocess_profile_scope(
      profile_id: str,
      *,
      backend: str,
      task: Task,
  ) -> None:
      if str(profile_id or "auto").strip().lower() == "auto":
          return
      if backend != "mobilint" or task != Task.IMAGE_CLASSIFICATION:
          raise ValueError(
              "--image-preprocess-profile is supported only for Mobilint "
              "raw image-classification targets."
          )
  ```

  Loader가 선언한 입력 계약은 사용자가 우회할 수 있는 튜닝 옵션이 아니다. 다음 keys를 상수로 만들고 `_merge_runtime_option_layers()`에서 Mobilint일 때 CLI override에 하나라도 나타나면 `ValueError`를 낸다.

  ```python
  _MOBILINT_INPUT_CONTRACT_OPTIONS = frozenset({
      "expected_input_dtype",
      "expected_input_layout",
      "expected_unbatched_input_shape",
      "max_input_batch_size",
  })
  ```

  CLI에는 `core_mode`, `activation_slots`, async 설정 같은 실행 튜닝만 계속 허용한다. `test_main_paths.py`에서 `expected_input_dtype=float32` override가 거부되고 `core_mode=global8`은 합쳐지는 것을 확인한다.

- [ ] **Step 5: Resolve after artifact selection and before CompiledModel**

  artifact compile/resolution이 끝난 직후, 현재 `CompiledModel` 생성 전에 다음 흐름을 추가한다.

  ```python
  mobilint_input_config = None
  if args.backend == "mobilint" and task_enum == Task.IMAGE_CLASSIFICATION:
      try:
          mobilint_input_config = resolve_mobilint_image_input_config(
              model_name=args.model,
              artifact_path=artifact_path,
              requested_profile=args.image_preprocess_profile,
              requested_mode=args.image_preprocess_mode,
              requested_layout=args.layout,
              layout_was_default=layout_was_default,
          )
          spec = apply_mobilint_image_input_config(
              spec,
              mobilint_input_config,
          )
          args.layout = mobilint_input_config.input_layout
      except ValueError as exc:
          print(f"[Error] Mobilint image input profile: {exc}")
          sys.exit(1)

  compiled_model = CompiledModel(
      spec=spec,
      backend_name=args.backend,
      artifact_path=artifact_path,
  )
  ```

  CLI banner는 effective layout을 보여야 하므로 현재 artifact 해석보다 앞에 있는 banner를 config 해석 뒤로 옮긴다. dataset path resolution은 그대로 유지하되, 이동으로 생기는 변수 사용 순서를 test로 확인한다.

  Mobilint image loader kwargs를 명시적으로 추가한다.

  ```python
  elif args.backend == "mobilint" and task_enum == Task.IMAGE_CLASSIFICATION:
      loader_kwargs.update({
          "backend": "mobilint",
          "artifact_path": str(artifact_path),
          "image_preprocess_mode": args.image_preprocess_mode,
          "mobilint_input_config": mobilint_input_config,
      })
  ```

  `CompiledModel.spec`과 `create_dataloader(model_spec=spec)`에는 반드시 같은 교체된 `spec`을 전달한다.

- [ ] **Step 6: Run focused and neighboring regressions**

  Run:

  ```bash
  cd framework
  python -m pytest -q \
    tests/test_mobilint_image_classification_loader.py \
    tests/test_main_paths.py \
    tests/test_hailo_image_loader.py \
    tests/test_deepx_dxnn_metadata.py
  ```

  Expected: all tests pass; Hailo raw remains its existing dtype/resize behavior and generic/DeepX routing is unchanged.

- [ ] **Step 7: Commit CLI and factory wiring**

  ```bash
  git add \
    framework/src/dataloader/__init__.py \
    framework/src/main.py \
    framework/tests/test_mobilint_image_classification_loader.py \
    framework/tests/test_main_paths.py
  git commit -m "feat: wire Mobilint image profiles into CLI"
  ```

---

## Task 3: Validate MXQ metadata and actual arrays in MobilintRuntime

**Files:**

- Modify: `framework/src/runtimes/mobilint_rt.py`
- Modify: `framework/tests/test_mobilint_runtime.py`

- [ ] **Step 1: Extend fake qbruntime and write contract RED tests**

  기존 `_install_fake_qbruntime()` state에 기본 단일 이미지 metadata를 추가한다.

  ```python
  from enum import Enum

  class FakeDataType(Enum):
      Uint8 = 1
      Float32 = 2

  state["input_shapes"] = [(224, 224, 3)]
  state["input_dtypes"] = [FakeDataType.Uint8]
  ```

  fake `Model`에는 다음 getters를 제공한다.

  ```python
  def get_model_input_shape(self):
      return state["input_shapes"]

  def get_model_input_data_type(self):
      return state["input_dtypes"]
  ```

  기존 NLP two-input `_compiled_model()`은 입력 계약 runtime options가 없으므로 그대로 통과해야 한다. 별도 `_compiled_image_model()`은 input `(1, 224, 224, 3)`, dtype `uint8`, output `(1, 1, 1000)`인 한 input MXQ를 만든다.

  새 tests:

  - 기대 계약과 SDK metadata가 일치하면 `load()`가 성공하고 `get_device_spec()`에 expected/actual 계약이 기록된다.
  - SDK dtype이 `Float32`이면 `load()`가 명확한 `dtype` mismatch로 실패하고 fake model dispose/device release가 각각 한 번 실행된다.
  - SDK shape이 `(3, 224, 224)`이면 `load()`가 명확한 `shape` mismatch로 rollback한다.
  - 계약 options가 있는데 getter가 없으면 지원하지 않는 SDK 계약 오류로 rollback한다.
  - 계약 options가 없는 기존 Mobilint NLP test double은 getters를 호출하지 않는다.
  - `run()`에 `(1, 224, 224, 3)` `float32`, `(1, 3, 224, 224)` `uint8`, batch 2 `uint8`을 주면 fake `infer()` 호출 전 각각 실패한다.
  - 올바른 non-contiguous `uint8` view를 주면 runtime이 contiguous copy를 만든 뒤 검증·전달해 기존 contiguous-input 계약을 보존한다.

- [ ] **Step 2: Run runtime tests and confirm RED**

  Run:

  ```bash
  cd framework
  python -m pytest -q tests/test_mobilint_runtime.py
  ```

  Expected: new metadata/array validation assertions fail while existing runtime tests still identify the current behavior.

- [ ] **Step 3: Parse optional expected input contract in `__init__`**

  module-level normalizer를 먼저 추가한다. `bool`을 정수 dimension으로 받지 않고, dtype은 NumPy canonical name으로 저장한다.

  ```python
  def _normalize_expected_dtype(value: Any) -> str | None:
      if value is None:
          return None
      if type(value) is not str:
          raise ValueError("expected_input_dtype must be a dtype name.")
      try:
          return np.dtype(value.strip().lower()).name
      except (TypeError, ValueError) as exc:
          raise ValueError(
              f"Unsupported expected_input_dtype: {value!r}."
          ) from exc


  def _normalize_expected_layout(value: Any) -> str | None:
      if value is None:
          return None
      if type(value) is not str:
          raise ValueError("expected_input_layout must be NCHW or NHWC.")
      normalized = value.strip().upper()
      if normalized not in {"NCHW", "NHWC"}:
          raise ValueError("expected_input_layout must be NCHW or NHWC.")
      return normalized


  def _normalize_expected_shape(value: Any) -> tuple[int, ...] | None:
      if value is None:
          return None
      if not isinstance(value, (list, tuple)) or not value:
          raise ValueError(
              "expected_unbatched_input_shape must be a non-empty sequence."
          )
      if any(
          isinstance(dim, (bool, np.bool_))
          or not isinstance(dim, (int, np.integer))
          or int(dim) <= 0
          for dim in value
      ):
          raise ValueError(
              "expected_unbatched_input_shape dimensions must be positive "
              "integers."
          )
      return tuple(int(dim) for dim in value)
  ```

  `MobilintRuntime.__init__()`에서 네 options를 읽는다.

  ```python
  self.expected_input_dtype = _normalize_expected_dtype(
      runtime_options.get("expected_input_dtype")
  )
  self.expected_input_layout = _normalize_expected_layout(
      runtime_options.get("expected_input_layout")
  )
  self.expected_unbatched_input_shape = _normalize_expected_shape(
      runtime_options.get("expected_unbatched_input_shape")
  )
  max_batch = runtime_options.get("max_input_batch_size")
  if max_batch is not None and (
      type(max_batch) is not int or max_batch <= 0
  ):
      raise ValueError("max_input_batch_size must be a positive integer.")
  self.max_input_batch_size = max_batch
  ```

  네 값을 tuple로 모아 `sum(value is not None for value in contract)`가 0 또는 4가 아니면 `ValueError("Mobilint expected input contract must define dtype, layout, shape, and max batch together.")`로 실패시킨다. 현재 profile은 `uint8`, `NHWC`, `(224, 224, 3)`, `1`이다.

  다음 상태를 추가하고 `_clear_model_state()`에서 지운다.

  ```python
  self._actual_input_dtype: str | None = None
  self._actual_unbatched_input_shape: tuple[int, ...] | None = None
  ```

- [ ] **Step 4: Validate SDK metadata inside the existing rollback boundary**

  다음 private helpers를 만든다: `_has_expected_input_contract()`는 네 계약 필드가 모두 설정됐는지 반환하고, `_read_single_model_input_shape()`와 `_read_single_model_input_dtype()`는 아래 정규화 규칙으로 SDK 값을 읽으며, `_validate_model_input_contract()`와 `_validate_runtime_input_array()`가 각각 model metadata와 batched array를 비교한다.

  ```python
  def _has_expected_input_contract(self) -> bool:
      return self.expected_input_dtype is not None

  def _read_single_model_input_shape(self) -> tuple[int, ...]:
      getter = getattr(self._model, "get_model_input_shape", None)
      if not callable(getter):
          raise RuntimeError(
              "qbruntime.Model does not expose get_model_input_shape(); "
              "Mobilint SDK v1.3.2-compatible input metadata is required."
          )
      reported = getter()
      if (
          isinstance(reported, (list, tuple))
          and len(reported) == 1
          and isinstance(reported[0], (list, tuple))
      ):
          reported = reported[0]
      if not isinstance(reported, (list, tuple)):
          raise RuntimeError(
              f"Unexpected qbruntime input shape metadata: {reported!r}."
          )
      if not reported or any(
          isinstance(dim, (bool, np.bool_))
          or not isinstance(dim, (int, np.integer))
          or int(dim) <= 0
          for dim in reported
      ):
          raise RuntimeError(
              f"Unexpected qbruntime input shape metadata: {reported!r}."
          )
      return tuple(int(dim) for dim in reported)

  def _read_single_model_input_dtype(self) -> str:
      getter = getattr(self._model, "get_model_input_data_type", None)
      if not callable(getter):
          raise RuntimeError(
              "qbruntime.Model does not expose "
              "get_model_input_data_type(); Mobilint SDK v1.3.2-compatible "
              "input metadata is required."
          )
      reported = getter()
      if isinstance(reported, (list, tuple)):
          if len(reported) != 1:
              raise RuntimeError(
                  f"Expected one qbruntime input dtype, received {reported!r}."
              )
          reported = reported[0]
      name = getattr(reported, "name", None)
      token = str(name if name is not None else reported).split(".")[-1]
      normalized = token.strip(" <>:").lower()
      aliases = {"uint8": "uint8", "u8": "uint8", "float32": "float32"}
      if normalized not in aliases:
          raise RuntimeError(
              f"Unexpected qbruntime input dtype metadata: {reported!r}."
          )
      return aliases[normalized]

  def _validate_model_input_contract(self) -> None:
      if not self._has_expected_input_contract():
          return
      actual_shape = self._read_single_model_input_shape()
      actual_dtype = self._read_single_model_input_dtype()
      self._actual_unbatched_input_shape = actual_shape
      self._actual_input_dtype = actual_dtype
      if actual_shape != self.expected_unbatched_input_shape:
          raise ValueError(
              "Mobilint MXQ input shape mismatch: expected "
              f"{self.expected_unbatched_input_shape}, received "
              f"{actual_shape}."
          )
      if actual_dtype != self.expected_input_dtype:
          raise ValueError(
              "Mobilint MXQ input dtype mismatch: expected "
              f"{self.expected_input_dtype}, received {actual_dtype}."
          )

  def _validate_runtime_input_array(
      self,
      input_name: str,
      array: np.ndarray,
  ) -> None:
      expected_shape = (
          self.max_input_batch_size,
          *self.expected_unbatched_input_shape,
      )
      if array.dtype != np.dtype(self.expected_input_dtype):
          raise ValueError(
              f"Mobilint input {input_name!r} requires dtype "
              f"{self.expected_input_dtype}, received {array.dtype}."
          )
      if (
          array.ndim != len(self.expected_unbatched_input_shape) + 1
          or array.shape[0] < 1
          or array.shape[0] > self.max_input_batch_size
          or tuple(array.shape[1:]) != self.expected_unbatched_input_shape
      ):
          raise ValueError(
              f"Mobilint input {input_name!r} requires layout "
              f"{self.expected_input_layout} and batched shape up to "
              f"{expected_shape}, received {array.shape}."
          )
  ```

  SDK shape는 `[(224, 224, 3)]`와 `(224, 224, 3)`를 모두 정규화하되 여러 inputs나 비정수/비양수 dim은 명시적으로 거부한다. SDK dtype은 단일 값 또는 길이 1 sequence를 받고, enum의 `.name`을 우선 사용한 뒤 `DataType.Uint8` 문자열도 `uint8`로 정규화한다. 예상할 수 없는 반환 형식은 검증 생략이 아니라 SDK API contract 오류다.

  `self._model.launch(self._accelerator)` 바로 뒤, 기존 `try` 블록 안에서 `_validate_model_input_contract()`를 호출한다. 따라서 metadata mismatch도 현행 rollback cleanup 경로를 그대로 탄다. expected contract가 전혀 없으면 getters를 조회하지 않아 기존 Mobilint NLP/다중 입력 동작을 유지한다.

- [ ] **Step 5: Validate batched arrays before sync and async SDK calls**

  `_ordered_inputs()`는 먼저 기존처럼 `np.asarray()` 후 `np.ascontiguousarray()`를 적용한다. expected contract가 있으면 단일 input만 허용하고 각 array에 다음을 검사한다.

  ```text
  dtype == np.dtype(expected_input_dtype)
  ndim == len(expected_unbatched_input_shape) + 1
  shape[0] <= max_input_batch_size
  shape[0] > 0
  shape[1:] == expected_unbatched_input_shape
  C-contiguous after normalization
  ```

  오류에는 input name, expected dtype/shape/layout, actual dtype/shape를 포함한다. `_ordered_inputs()`는 sync `run()`과 `MobilintNativeBackend.submit()` 양쪽에서 이미 사용되므로 별도 async 전처리 분기를 만들지 않는다.

  `get_device_spec()`에 다음 diagnostic fields를 추가한다.

  ```python
  "expected_input_dtype": self.expected_input_dtype,
  "expected_input_layout": self.expected_input_layout,
  "expected_unbatched_input_shape": self.expected_unbatched_input_shape,
  "actual_input_dtype": self._actual_input_dtype,
  "actual_unbatched_input_shape": self._actual_unbatched_input_shape,
  "max_input_batch_size": self.max_input_batch_size,
  ```

- [ ] **Step 6: Run runtime and native-async regressions**

  Run:

  ```bash
  cd framework
  python -m pytest -q \
    tests/test_mobilint_runtime.py \
    tests/test_mobilint_native_backend.py \
    tests/test_async_cli.py
  ```

  Expected: all tests pass. 기존 input-order와 contiguous-copy test, cleanup retry test, native Future completion test도 계속 통과한다.

- [ ] **Step 7: Commit runtime validation**

  ```bash
  git add framework/src/runtimes/mobilint_rt.py framework/tests/test_mobilint_runtime.py
  git commit -m "feat: validate Mobilint MXQ image inputs"
  ```

---

## Task 4: Document the supported profile and run complete SDK-free verification

**Files:**

- Modify: `framework/src/runtimes/README.md`

- [ ] **Step 1: Add the user-facing Mobilint image example**

  Mobilint runtime section에 다음 내용을 기록한다.

  - 지원 profile: `mobilint-resnet50-imagenet1k-v2`
  - `auto`가 요구하는 정확한 조합: model `resnet50`, basename `resnet50_IMAGENET1K_V2.mxq`
  - 입력 계약: raw RGB, short side 232, center crop 224, NHWC, `uint8`, batch 1
  - unknown/renamed MXQ에는 explicit `--image-preprocess-profile`이 필요함
  - runtime이 입력을 cast/normalize하지 않음
  - ARIES와 REGULUS가 같은 artifact profile을 공유함

  실제 실행 예시는 다음 형태로 둔다. default layout은 profile이 NHWC로 해석하므로 `--layout`은 생략한다.

  ```bash
  MXQ=framework/models/mobilint/resnet50/aries/resnet50_IMAGENET1K_V2.mxq

  .venv-mobilint/bin/python framework/src/main.py \
    --model resnet50 \
    --target mobilint-aries \
    --artifact "$MXQ" \
    --dataset datasets/imagenet_1k \
    --inference-mode e2e \
    --batch-size 1 \
    --warmup 2 \
    --max-steps 10 \
    --runtime-option core_mode=global8 \
    --no-compile
  ```

  rename된 artifact 예시에는 다음을 추가한다.

  ```bash
  --image-preprocess-profile mobilint-resnet50-imagenet1k-v2
  ```

- [ ] **Step 2: Run focused Mobilint verification**

  Run:

  ```bash
  cd framework
  python -m pytest -q \
    tests/test_mobilint_image_classification_loader.py \
    tests/test_mobilint_runtime.py \
    tests/test_mobilint_native_backend.py \
    tests/test_mobilint_collector.py \
    tests/test_hw_monitor.py \
    tests/test_main_paths.py \
    tests/test_async_cli.py
  ```

  Expected: all tests pass.

- [ ] **Step 3: Run neighboring backend regressions**

  Run:

  ```bash
  cd framework
  python -m pytest -q \
    tests/test_hailo_image_loader.py \
    tests/test_deepx_dxnn_metadata.py \
    tests/test_llama_evaluator_regression_1.py
  ```

  Expected: all tests pass; cache signature, Hailo loader, DeepX metadata, Mobilint-independent NLP paths show no regression.

- [ ] **Step 4: Run the full SDK-free suite**

  Run:

  ```bash
  cd framework
  python -m pytest -q
  ```

  Expected: all tests pass. 환경에 종속된 integration tests가 기존 marker 정책으로 skip되면 skip count와 reason을 최종 보고에 기록한다.

- [ ] **Step 5: Commit documentation**

  ```bash
  git add framework/src/runtimes/README.md
  git commit -m "docs: add Mobilint image profile guide"
  ```

---

## Task 5: Verify on the ARIES2 host

**Files:**

- No production file changes expected
- Record command output in the PR description or review comment

- [ ] **Step 1: Confirm the installed SDK and device before inference**

  Run on the NPU server:

  ```bash
  dpkg-query -W \
    -f='${Package}\t${Status}\t${Version}\n' \
    mobilint-aries-driver mobilint-qb-runtime mobilint-cli
  lsmod | grep '^aries'
  ls -l /dev/aries0
  /usr/bin/mobilint-cli status
  .venv-mobilint/bin/python -c 'import qbruntime; print(qbruntime.__version__)'
  ```

  Expected: driver/runtime/CLI installed, `aries` module loaded, `/dev/aries0` present, CLI detects ARIES2, qbruntime reports v1.3.2-compatible SDK.

- [ ] **Step 2: Run e2e acceptance with the known image dataset**

  Run from the repository root on the NPU server:

  ```bash
  MXQ=framework/models/mobilint/resnet50/aries/resnet50_IMAGENET1K_V2.mxq

  .venv-mobilint/bin/python framework/src/main.py \
    --model resnet50 \
    --target mobilint-aries \
    --artifact "$MXQ" \
    --dataset /home/etri_ecas/ML-HW-Benchmark-Framework/datasets/imagenet_1k \
    --inference-mode e2e \
    --batch-size 1 \
    --warmup 2 \
    --max-steps 10 \
    --runtime-option core_mode=global8 \
    --no-compile
  ```

  Expected:

  - banner의 effective layout이 `NHWC`다.
  - loader metadata/log가 profile과 `uint8`을 보고한다.
  - `Model_DtypeMismatched`가 발생하지 않는다.
  - 10 steps가 완료되고 첫 이미지의 top-1/top-5가 기존 Model Zoo 결과와 일치한다. 기준 top-1은 `Italian greyhound`, probability 약 `51.45%`다.

- [ ] **Step 3: Run monitored acceptance**

  Run:

  ```bash
  .venv-mobilint/bin/python framework/src/main.py \
    --model resnet50 \
    --target mobilint-aries \
    --artifact "$MXQ" \
    --dataset /home/etri_ecas/ML-HW-Benchmark-Framework/datasets/imagenet_1k \
    --inference-mode e2e \
    --batch-size 1 \
    --warmup 2 \
    --max-steps 10 \
    --runtime-option core_mode=global8 \
    --monitor \
    --no-compile
  ```

  Expected: utilization, memory, temperature, power samples와 energy 집계가 기록되고 inference 결과는 Step 2와 같다. 모니터 adapter나 metric schema는 이번 변경에서 수정하지 않는다.

- [ ] **Step 4: Run native async acceptance**

  Run:

  ```bash
  .venv-mobilint/bin/python framework/src/main.py \
    --model resnet50 \
    --target mobilint-aries \
    --artifact "$MXQ" \
    --dataset /home/etri_ecas/ML-HW-Benchmark-Framework/datasets/imagenet_1k \
    --inference-mode async_queue \
    --batch-size 1 \
    --warmup 2 \
    --min-samples 10 \
    --max-samples 10 \
    --worker-count 1 \
    --runtime-option core_mode=global8 \
    --runtime-option activation_slots=1 \
    --no-compile
  ```

  Expected:

  - Mobilint native backend가 SDK `infer_async()`를 사용한다.
  - sync와 동일한 `(1, 224, 224, 3)` `uint8` 입력을 받는다.
  - 완료 시 pending/outstanding request가 0이고 shutdown timeout이나 dispose 오류가 없다.

- [ ] **Step 5: Final review before PR update**

  Run:

  ```bash
  git status --short
  git diff origin/main...HEAD --check
  git log --oneline origin/main..HEAD
  ```

  Expected: 의도한 source/test/docs만 변경됐고 whitespace error가 없으며 Task별 commits가 보인다. Hardware log에서 SDK version, e2e 결과, monitor 결과, native-async 결과를 PR에 첨부하고, SDK-free CI와 hardware acceptance를 구분해 보고한다.
