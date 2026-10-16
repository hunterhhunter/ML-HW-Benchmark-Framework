# Regulus Pose Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 기존 `mobilint-regulus` 타깃에서 VG-E2E YOLOv8s-pose 9-head MXQ를 raw RGB 입력으로 실행하고, 별도 E2E 패스에서 COCO OKS AP를 산출한다.

**Architecture:** 기존 `MobilintRuntime`은 SDK 어댑터로 유지한다. 새 artifact 계약·전처리·로더·디코더는 `Regulus*` 타입과 `regulus-*` 프로필 ID로 분리하고 `target_id == "mobilint-regulus"`일 때만 팩토리에서 선택한다. 기존 Pose evaluator와 결과 저장 구조를 재사용한다.

**Tech Stack:** Python 3.12, NumPy, OpenCV, PIL, pytest, COCO/pycocotools, Regulus 보드의 qbruntime.

**Spec:** `docs/regulus-pose-async-design.md`

## Global Constraints

- 기존 CLI 타깃 ID `mobilint-regulus`와 `MobilintRuntime`을 유지한다. 새 Pose 기능의 이름은 Regulus 전용이다.
- Pose 입력은 RGB `uint8` NHWC `(640, 640, 3)`, batch 1, letterbox이며 출력은 stride 8/16/32 각각 keypoint 51, DFL 64, class 1채널의 9-head다.
- 명시적 프로필 ID는 `regulus-vg-e2e-yolov8s-pose-raw-v1`이다. 일반적인 `model.mxq` 파일명만으로 자동 선택하지 않는다.
- `runtime-only`는 decoder/evaluator 없이 실행한다. Pose 정확도는 별도 E2E에서 COCO OKS AP로 검증한다.
- 모델·데이터셋·SDK·원시 측정 결과는 커밋하지 않는다. PR 제목·본문은 한글로 작성한다.
- 로컬 테스트 인터프리터는 `/home/swlab-youngjin/ML-HW-Benchmark-Framework/framework/.venv/bin/python`이다.

## Review Focus

- 같은 `yolov8s-pose` 이름으로 Aries를 선택하면 Regulus 프로필·로더·디코더가 선택되지 않아야 한다. Task 1·3의 타깃 분리 테스트.
- `model.mxq`라는 공통 파일명이나 잘못된 출력 head 계약은 Pose 프로필을 자동 승인하지 않아야 한다. Task 1의 음성 테스트.
- head 순서가 섞이거나 51/64/1채널이 중복·누락되면 디코더가 추측하지 않고 실패해야 한다. Task 2의 ABI 테스트.
- 서로 다른 COCO 이미지가 같은 파일명을 갖거나 원본이 변경되면 이전 전처리 텐서를 재사용하지 않아야 한다. Task 1의 캐시 식별 테스트.
- raw `uint8` 입력을 상위 COCO 로더의 `normalized` 모드로 위장하지 않고 실제 dtype/layout을 보고해야 한다. Task 1·3의 입력 테스트.

---

### Task 1: Regulus Pose artifact 프로필과 입력 로더

**Files:**
- Create: `framework/src/core/regulus_vision_contracts.py` — `RegulusPoseRawHeadRecipe`, `RegulusPoseArtifactProfile` 및 `runtime_contract()`.
- Create: `framework/src/dataloader/regulus_vision_profiles.py` — 프로필 상수, 해석, 모델 스펙 적용.
- Create: `framework/src/preprocessor/regulus_pose.py` — raw RGB letterbox와 원본 좌표 문맥, source-aware 캐시.
- Create: `framework/src/dataloader/regulus_pose_estimation_loader.py` — COCO Pose 정체성 보존, Regulus runtime 옵션 제공.
- Modify: `framework/src/dataloader/coco_vision_loader.py:22-66,201-214` — 주입된 전처리기의 명시적 raw 계약 허용 및 캐시 원본 경로 전달.
- Test: `framework/tests/test_regulus_pose_profile_loader.py`.

**Interfaces:**
- Consumes: `Model_Spec`, `Task.POSE_ESTIMATION`, 기존 `CocoPoseLoader`, 런타임의 `vision_profile_id`/`expected_*` 계약.
- Produces: `resolve_regulus_pose_profile(*, target_id: str, model_name: str, task: Task, artifact_path: Path, requested_profile: str, requested_mode: str, requested_layout: str, layout_was_default: bool) -> RegulusPoseArtifactProfile`; `apply_regulus_pose_profile(spec: Model_Spec, profile: RegulusPoseArtifactProfile) -> Model_Spec`; `RegulusPoseEstimationLoader(model_spec, **kwargs)`; `profile.runtime_contract() -> dict[str, object]`.

- [ ] **Step 1: 실패 테스트 작성.** `test_regulus_pose_profile_accepts_only_explicit_regulus_contract`는 올바른 ID·타깃에서만 위 9개 shape와 raw `uint8`/NHWC를 반환하고, Aries·`auto`·잘못된 모델/Task/레이아웃을 거부한다고 단언한다. `test_regulus_pose_loader_preserves_coco_identity_and_raw_input`는 1개 합성 COCO 이미지를 사용해 입력 `(640,640,3)` `uint8`, letterbox context, image_id를 확인한다. `test_regulus_pose_cache_separates_source_and_recipe`는 같은 이름의 다른 원본과 변경된 원본에 다른 키를 요구한다.
- [ ] **Step 2: 실패 확인.** `cd framework && /home/swlab-youngjin/ML-HW-Benchmark-Framework/framework/.venv/bin/python -m pytest -q tests/test_regulus_pose_profile_loader.py`; 새 모듈 미존재로 실패해야 한다.
- [ ] **Step 3: 최소 구현.** `RegulusPoseArtifactProfile.runtime_contract()`가 기존 `MobilintRuntime` 검증 키를 제공하도록 구현한다. `RegulusPosePreprocessor`는 `image_preprocess_mode = "raw"`, `target_hw = (640,640)`, `preprocess_with_context(raw_input) -> tuple[np.ndarray, dict]`를 제공하며 PIL RGB 읽기·OpenCV linear resize·pad 114·Ultralytics `round(±0.1)`을 사용한다. 캐시 키는 프로필 ID·전처리 버전·원본 절대 경로·원본 파일 SHA-256으로 만든다. COCO 로더는 선언된 raw 전처리기만 허용하고 일반 normalized 기본값을 유지한다.
- [ ] **Step 4: 통과 확인.** 위 테스트와 `tests/test_pose_backend_factory.py`, `tests/test_mobilint_vision_profiles.py` 실행; 모두 PASS.
- [ ] **Step 5: 이 Task의 파일만 커밋.** 메시지 `feat: Regulus Pose 입력 계약과 COCO 로더 추가`.

### Task 2: Regulus 9-head Pose 디코더

**Files:**
- Create: `framework/src/decoders/regulus_yolov8_pose.py` — 9-head DFL·class·keypoint decode와 NMS.
- Modify: `framework/src/decoders/__init__.py:11-25,45-150` — Regulus 프로필과 타깃이 함께 맞을 때만 선택.
- Test: `framework/tests/test_regulus_pose_decoder.py`.

**Interfaces:**
- Consumes: Task 1의 `RegulusPoseArtifactProfile`과 `RegulusPoseRawHeadRecipe`, `decoders.yolo_vision`의 detection/keypoint 키와 NMS 보조 함수.
- Produces: `RegulusYoloV8PoseRawHeadDecoder(profile: RegulusPoseArtifactProfile, *, conf_threshold: float = 0.25, iou_threshold: float = 0.70, max_detections: int = 300, max_nms_candidates: int = 30000)`; `.decode(outputs: Mapping[str, Any]) -> dict[str, np.ndarray]`; `.result_metadata() -> dict[str, object]`.

- [ ] **Step 1: 실패 테스트 작성.** `test_regulus_pose_decoder_decodes_nine_heads_to_coco_payload`는 합성 9-head의 DFL 거리·class sigmoid·keypoint 좌표/visibility를 확인한다. `test_regulus_pose_decoder_accepts_head_permutation`과 `test_regulus_pose_decoder_rejects_duplicate_missing_nonfinite_heads`는 shape/role 결합과 fail-closed를 확인한다. `test_regulus_pose_factory_does_not_route_aries_or_unknown_abi`는 기존 generic·Hailo·DeepX 분기를 보존한다.
- [ ] **Step 2: 실패 확인.** `cd framework && /home/swlab-youngjin/ML-HW-Benchmark-Framework/framework/.venv/bin/python -m pytest -q tests/test_regulus_pose_decoder.py`; 새 디코더 미존재로 실패해야 한다.
- [ ] **Step 3: 최소 구현.** 기존 검증된 Regulus 9-head 디코더를 참조하되 새 파일에서 Regulus 전용으로 구현한다. 80/40/20 격자의 51/64/1 head를 shape로 결합하고, DFL 16-bin softmax, stride 8/16/32, class 및 visibility sigmoid, 최대 후보 30,000개 제한과 NMS를 적용한다. 입력 계약이 다르면 하드웨어 호출 전 또는 decode 시 명시적 오류를 낸다.
- [ ] **Step 4: 통과 확인.** 위 테스트와 `tests/test_yolov8_pose_decoder.py`, `tests/test_pose_estimation_evaluator.py`, `tests/test_pose_backend_factory.py` 실행; 모두 PASS.
- [ ] **Step 5: 이 Task의 파일만 커밋.** 메시지 `feat: Regulus Pose 9-head 디코더 연결`.

### Task 3: main.py 통합과 보드 정확도 검증

**Files:**
- Modify: `framework/src/main.py:151-167,3536-3543,3644-3755,3820-3850` — Pose 프로필 해석, 로더·decoder 주입, runtime-only 분기 유지.
- Modify: `framework/src/dataloader/__init__.py:14-39,60-125` — Regulus 전용 Pose 분기.
- Modify: `framework/README.md` — 명시적 프로필 실행법, 성능/정확도 경계.
- Test: `framework/tests/test_main_paths.py`, `framework/tests/test_regulus_pose_main.py`.

**Interfaces:**
- Consumes: Task 1의 프로필·로더, Task 2의 디코더, 기존 `create_evaluator()` 및 `MobilintRuntime`.
- Produces: `main.py`에서 `--target mobilint-regulus --model yolov8s-pose --image-preprocess-profile regulus-vg-e2e-yolov8s-pose-raw-v1` 실행 경로; 기존 CSV의 `mobilint_vision_profile_id` 필드에는 Regulus-prefixed 프로필 ID를 보존하여 스키마 호환.

- [ ] **Step 1: 실패 테스트 작성.** `test_regulus_pose_main_routes_profile_loader_decoder_and_evaluator`는 프로필·로더·디코더가 동일 계약을 받는지 확인한다. `test_regulus_pose_runtime_only_skips_quality_components`는 decoder/evaluator 미생성과 `quality_status=skipped`를 확인한다. `test_regulus_pose_profile_rejected_for_aries_before_runtime_load`는 타깃 격리를 확인한다.
- [ ] **Step 2: 실패 확인.** `cd framework && /home/swlab-youngjin/ML-HW-Benchmark-Framework/framework/.venv/bin/python -m pytest -q tests/test_regulus_pose_main.py`; Regulus 분기 미존재로 실패해야 한다.
- [ ] **Step 3: 최소 구현.** `target.target_id`로 Regulus Pose 경로를 게이트하고, `loader_kwargs["target_id"]`와 `create_decoder(..., target_id=target.target_id, regulus_pose_profile=profile)`를 명시한다. 새 프로필의 `runtime_contract()`를 기존 runtime 옵션 병합으로 전달한다. raw 입력을 normalized로 표시하지 않는다. README에 동일 artifact·데이터·전처리의 성능/정확도 실행을 분리해 설명한다.
- [ ] **Step 4: 통과 확인.** Task 1~3의 신규 테스트와 기존 `tests/test_main_paths.py tests/test_pose_backend_factory.py tests/test_mobilint_runtime.py` 실행; 모두 PASS.
- [ ] **Step 5: 이 Task의 파일만 커밋.** 메시지 `feat: Regulus Pose main 진입과 정확도 경로 연결`.
- [ ] **Step 6: 실기기 검증.** `root@192.168.0.79`에서 artifact SHA·9-head shape·NPU binding을 확인한 뒤 10건 Pose E2E smoke, COCO val2017 5,000건 Pose E2E 정확도를 실행한다. decoder 임계값 `confidence=0.25`, `IoU=0.70`과 OKS AP, 예측 샘플, 결과 원본을 기록한다. 참조와 동일 설정일 때만 수치 비교한다.

## 통합 인계

세 계획이 모두 완료된 뒤 세 모델에 대해 별도 5,000건 async/offline/runtime-only를 실행한다. schema v2, `framework_runtime_call_return`, 5,000 submitted/accepted/completed/runtime-completed, 실패·거절·타임아웃·미완료 0, trace 5,000행, `quality_status=skipped`를 확인한다. 전체 테스트 및 diff 검토 후 한글 제목·본문으로 `main` 대상 PR을 생성한다.
