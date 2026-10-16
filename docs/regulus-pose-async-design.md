# Regulus Pose·async 기능의 main 통합 설계

## 목적과 완료 기준

`origin/main`의 기존 `mobilint-regulus` 타깃과 `MobilintRuntime`을 유지하면서, 세 VG-E2E MXQ 모델(ResNet-50, YOLOv5m, YOLOv8s-pose)을 프레임워크 `src/main.py`로 실행한다. 세 모델의 `async_queue`/`offline`/`runtime-only` 5,000건 측정은 입력 준비를 측정 전에 끝낸 상태에서 재현할 수 있어야 한다. YOLOv8s-pose는 별도의 E2E 정확도 실행에서 기존 COCO Pose evaluator로 OKS AP를 산출할 수 있어야 한다.

이번 변경은 벤더 네이티브 실행기, 새로운 Regulus 런타임, SDK의 native async 동작 변경, 벤치마크 결과 수치 자체를 main에 편입하지 않는다. 모델·데이터셋·SDK·측정 결과 원본은 PR에 추가하지 않는다. PR 브랜치는 `feat/regulus-pose-async-support`이고 PR 제목과 본문은 한글로 작성한다.

## 현재 상태와 접근

`main`에는 Regulus 타깃, qb 기반 Mobilint 런타임, ResNet-50·YOLOv5m 비전 프로필, `--async-pass runtime-only`, COCO Pose evaluator가 이미 있다. 빠진 부분은 VG-E2E Pose의 입력 및 9-head 출력 계약, 그 출력을 평가 가능한 예측으로 바꾸는 디코더, async 실행 전 입력 사전 적재, Regulus Yocto Python에서 `fcntl`을 사용할 수 없을 때의 파일 잠금 호환성이다.

기존 측정용 worktree의 Pose 프로필·mmap 로더·파일 잠금 대체 구현을 출발점으로 삼되 그대로 복사해 완료로 간주하지 않는다. 특히 Pose 데이터로더가 raw `uint8` 텐서를 만들면서 상위 COCO 로더에는 `normalized` 모드라고 전달하는 우회를 없애고, 실제 입력 계약을 코드와 결과 메타데이터에서 일치시킨다. 과거 아카이브 브랜치의 Regulus 전용 런타임과 벤더 네이티브 코드는 가져오지 않는다.

## 구성 요소와 데이터 흐름

### 1. VG-E2E Pose artifact 계약과 입력

`core/mobilint_vision_contracts.py`와 `dataloader/mobilint_vision_profiles.py`에 Regulus VG-E2E YOLOv8s-pose 프로필을 정의한다. 입력은 RGB, `uint8`, NHWC, 640×640, batch 1이며, 기존 Mobilint YOLO letterbox 전처리와 COCO 원본 좌표 복원을 위한 `preprocess_context`를 사용한다. 출력은 stride 8·16·32마다 keypoint 51채널, DFL box 64채널, class 1채널의 총 9개 head다. 런타임이 검증할 모양과 decoder가 사용할 head 역할을 명시한다.

`main.py`의 Mobilint 비전 프로필 해석에 Pose를 포함하고, 데이터로더 팩토리에서 Pose를 `MobilintPoseEstimationLoader`로 연결한다. 상위 COCO 로더는 주입된 전처리기가 raw 입력을 제공하는 경우를 명시적으로 허용하되, 일반 Pose 로더의 normalized 기본 동작은 바꾸지 않는다. 캐시에는 프로필·전처리 방식·원본 이미지 식별자가 반영되어 서로 다른 입력을 같은 텐서로 재사용하지 않아야 한다.

배포 MXQ가 모두 `model.mxq`처럼 같은 이름일 수 있으므로 파일명만으로 Pose 프로필을 추정하지 않는다. 실행 시 `--image-preprocess-profile mobilint-vg-e2e-yolov8s-pose-raw-v1`을 명시하고, 모델 이름·태스크·입력·출력 계약이 맞지 않으면 장치 로드 전에 오류를 낸다.

### 2. Pose 정확도 경로

`decoders` 팩토리는 Mobilint Pose 프로필의 9-head recipe가 확인된 경우에만 Regulus raw-head 디코더를 선택한다. 이 디코더는 head를 공간 크기·채널 수로 묶고 DFL box, class 활성화, 17개 keypoint 좌표·가시성, 후보 필터, NMS를 수행한다. 출력은 기존 `PoseEstimationEvaluator`가 요구하는 `detections`와 `keypoints` 형식이다. 다른 백엔드의 packed 56-channel Pose 디코더나 Hailo·DeepX 경로는 변경하지 않는다.

class head가 logits인지 probability인지, keypoint 변환식과 head 순서는 실제 MXQ 출력 및 기존 검증된 Regulus 참조 결과로 확인하고 프로필에 고정한다. 알 수 없는 출력 ABI는 조용히 추정하지 않고 실패시킨다. 임계값은 프로필 기본값과 명시적 CLI override를 기록하며, 서로 다른 confidence/IoU 설정의 AP를 같은 결과로 취급하지 않는다. Pose 정확도는 COCO OKS AP로 보고, 객체 검출 AP50과 섞지 않는다.

`runtime-only` 패스는 현재 설계대로 decoder/evaluator를 생성하지 않는다. 정확도 검증은 별도의 E2E 실행에서 수행한다. 기존 벤더 네이티브 정확도 결과를 이 프레임워크 실행의 검증으로 대신하지 않는다.

### 3. async 입력 사전 적재

`--async-preload-inputs`는 명시적으로 켠 `async_queue` 실행에서만 사용한다. 선택된 샘플의 전처리 결과를 타이머 시작 전에 하나의 mmap 파일로 materialize하고, 측정 중에는 같은 샘플 순서·라벨·원본 좌표 문맥을 유지하며 텐서를 공급한다. `e2e`에서 이 옵션을 주면 즉시 거부한다. 입력이 정적인 단일 NumPy 텐서가 아니거나 shape/dtype이 서로 다르면 명확히 실패한다.

기본값은 끔이다. 사전 적재 소요 시간과 임시 파일 크기는 처리량에서 제외되며 결과에 사용 여부·샘플 수·shape·dtype을 남긴다. 실행 종료와 예외 발생 시 임시 파일을 정리하고, 디스크 용량 부족 및 잘못된 샘플 수를 측정 전에 확인한다. mmap 공급에서도 page fault나 producer 부하가 남을 수 있으므로 `async_runtime_completed_samples_per_sec`를 순수 NPU 처리량이라고 부르지 않는다.

### 4. 보드 파일 잠금 호환성

`core/artifact_reservation.py`와 `core/result_store.py`가 공통 파일 잠금 어댑터를 사용한다. 표준 Python에 `fcntl`이 있으면 이를 그대로 사용하고, Linux에서 모듈이 없는 경우에만 libc `flock`으로 대체한다. `EINTR` 재시도와 잠금 해제를 포함하며, 다른 플랫폼의 예기치 않은 `fcntl` 부재를 묵살하지 않는다. 파일 잠금의 대상과 저장 형식은 바꾸지 않는다.

## 검증과 병합 조건

1. SDK 없이 실행되는 단위 테스트: Pose 프로필 해석·입력 dtype/layout/context·캐시 분리·9-head 계약·팩토리 분기·decoder의 정상/오류 입력·NMS·좌표 복원, mmap 샘플 순서·라벨 보존·예외/종료 정리, `fcntl` 존재/부재 경로, E2E에서 preload 옵션 거부.
2. 기존 Mobilint ResNet-50·YOLOv5m, Hailo·DeepX Pose, async runtime-only 회귀 테스트를 통과시킨다.
3. Regulus 보드에서 실제 MXQ의 장치 계열·입출력 ABI·NPU bundle 0/Cluster0-Core0 바인딩을 확인하고, 세 모델 각각 소량 smoke 후 5,000건 async/offline runtime-only를 실행한다. 각 run은 schema v2, `framework_runtime_call_return`, `valid`, 제출/수락/완료/runtime 완료 5,000, 오류·거절·타임아웃·미완료 0, trace 5,000행, `quality_status=skipped`를 확인한다.
4. 별도 Pose E2E 정확도 실행에서 COCO val2017 이미지·annotation 5,000개와 명시한 decoder 임계값으로 OKS AP를 산출한다. 참조 출력과 예측·좌표 변환을 대조하고 artifact SHA, 전처리 프로필, 코드 커밋, 실행 명령 및 결과 원본을 보관한다. 참조 실행과 설정이 다르면 수치의 일치를 합격 조건으로 쓰지 않는다.
5. PR에는 실행법·측정 경계·preload 사용 여부·정확도 상태를 한글로 문서화한다. 전력 측정이나 벤더 네이티브 경로는 이번 완료 조건이 아니다.
