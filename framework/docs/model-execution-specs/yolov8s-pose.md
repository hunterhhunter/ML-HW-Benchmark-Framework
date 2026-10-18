# YOLOv8s-pose 실행 명세: 에지 NPU

`main@30c4db75`의 실행 경로를 기준으로 확인된 조합과 재현 명령을 정리한다.
측정 구간과 지표는 [측정 계약](../measurement-contract.md)을 따른다.
실행 자산의 checksum은 [별도 기록](yolov8s-pose-sha256.md)에 있다.

## 모델·데이터

- 모델: `yolov8s-pose` (`POSE_ESTIMATION`, `PoseEstimationEvaluator`)
- 입력: COCO 2017 validation 이미지 5,000장과
  `annotations/person_keypoints_val2017.json`.

## 확인된 조합

| target / Runtime | 사용 artifact | 확인된 실행 모드 |
|---|---|---|
| `hailo8` / HailoRT 4.24.0 | 벤더 제공 `yolov8s_pose_8.hef` | 동기 `e2e`, `async_queue` 품질·런타임 패스 |
| `hailo10h` / HailoRT 5.3.0 | 벤더 제공 `yolov8s_pose_10.hef` | 동기 `e2e`, `async_queue` 품질·런타임 패스 |
| `deepx` / DX-RT 3.4.0 | 벤더 제공 `yolov8-s-pose_640x640.dxnn` | 동기 `e2e`, `async_queue` 품질·런타임 패스 |
| `mobilint-regulus` / qbruntime 1.2.0 | 직접 컴파일 모델 MXQ `model.mxq` | 동기 `e2e`, `async_queue` 품질·런타임 패스 |

## 입력 계약

| target | 전처리·Runtime 입력 |
|---|---|
| Hailo-8/10H | RGB → 640×640 letterbox, padding 114 → NHWC `uint8`. |
| DEEPX DX-M1 | RGB → 640×640 letterbox, padding 114. 입력 layout·dtype은 DXNN metadata에 맞춘다. |
| Mobilint Regulus | `regulus-vg-e2e-yolov8s-pose-raw-v1` profile: RGB → 640×640 letterbox → NHWC `uint8`. `qbruntime.Model.infer()` 사용. |

## 출력 계약

Pose 평가기에 전달하는 decoder 출력은 `detections` `(N, 7)`과
`keypoints` `(N, 17, 3)`이다. 박스 행은
`[배치 내 이미지 인덱스, person class 0, confidence, x1, y1, x2, y2]`,
키포인트 한 행은 17개 `(x, y, visibility)`다. 좌표는 decoder 단계에서
640×640 모델 입력 기준이고, 평가기가 원본 이미지 좌표로 복원한다.

| target | Runtime 출력 → 평가기 입력 변환 |
|---|---|
| Hailo-8/10H | HEF 출력 VStream 9개: 80×80, 40×40, 20×20 각 scale에 NHWC `51`개 키포인트·`64`개 DFL 박스·`1`개 class 채널이 있다. HEF metadata에서 이 ABI와 class probability 범위를 확인한 뒤 host decoder가 DFL·키포인트를 풀고 NMS를 적용한다. class score는 이미 probability여서 host sigmoid를 다시 적용하지 않는다. |
| DEEPX DX-M1 | DXNN graph metadata로 **9개 NCHW raw head** (`51`/`64`/`1` 채널 × 3 scale) 또는 검증된 CPU tail의 **packed `(B, 56, N)`** 출력 중 하나를 판별한다. raw head는 `DeepXYoloV8PoseRawHeadDecoder`, packed 출력은 `YoloV8PoseDecoder`가 처리한다. raw class 값의 sigmoid 여부도 ABI로 구분한다. 이 명세에 기록된 DXNN의 실제 선택 ABI·텐서 이름은 artifact metadata로 확인해야 한다. |
| Mobilint Regulus | MXQ 출력이 프레임워크에서 batch 1 NHWC raw head 9개로 정규화된다. 80×80, 40×40, 20×20 각 scale에 `51`/`64`/`1` 채널이 있다. `RegulusYoloV8PoseRawHeadDecoder`가 DFL·키포인트와 class logit의 sigmoid를 host에서 계산하고 NMS를 적용한다. |

## 재현 명령

각 target의 SDK가 설치된 장비에서 실행한다. 아래 경로를 실제 데이터·artifact
위치에 맞춘다. 동기 smoke는 `--max-steps 5000`을 `--max-steps 20`으로 바꾼다.

```bash
cd /path/to/ML-HW-Benchmark-Framework/framework
export COCO_ROOT=/absolute/path/to/coco
export IMAGES="$COCO_ROOT/images/val2017"
export LABELS="$COCO_ROOT/annotations/person_keypoints_val2017.json"
test -d "$IMAGES" && test -f "$LABELS"
sha256sum "$LABELS"  # 별도 SHA-256 기록과 대조
```

```bash
# Hailo-8
export ARTIFACT=/absolute/path/to/yolov8s_pose_8.hef
sha256sum "$ARTIFACT"
python3 src/main.py --model yolov8s-pose --target hailo8 \
  --artifact "$ARTIFACT" --no-compile --dataset "$COCO_ROOT" \
  --image-dir "$IMAGES" --label-dir "$LABELS" \
  --inference-mode e2e --batch-size 1 --warmup 10 --max-steps 5000 \
  --runtime-option timeout_ms=10000 --results-path results/yolov8s-pose-hailo8-e2e.csv

# Hailo-10H
export ARTIFACT=/absolute/path/to/yolov8s_pose_10.hef
sha256sum "$ARTIFACT"
python3 src/main.py --model yolov8s-pose --target hailo10h \
  --artifact "$ARTIFACT" --no-compile --dataset "$COCO_ROOT" \
  --image-dir "$IMAGES" --label-dir "$LABELS" \
  --inference-mode e2e --batch-size 1 --warmup 10 --max-steps 5000 \
  --results-path results/yolov8s-pose-hailo10h-e2e.csv

# DEEPX DX-M1
export ARTIFACT=/absolute/path/to/yolov8-s-pose_640x640.dxnn
sha256sum "$ARTIFACT"
python3 src/main.py --model yolov8s-pose --target deepx \
  --artifact "$ARTIFACT" --no-compile --dataset "$COCO_ROOT" \
  --image-dir "$IMAGES" --label-dir "$LABELS" \
  --inference-mode e2e --batch-size 1 --warmup 10 --max-steps 5000 \
  --runtime-option device_ids=0 --runtime-option bound_option=NPU_ALL \
  --runtime-option buffer_count=6 --results-path results/yolov8s-pose-deepx-e2e.csv
```

Hailo-8·Hailo-10H·DEEPX의 async 조합은 batch 1, worker 4, offline
5,000장이다. `TARGET`과 `ARTIFACT`를 같은 행의 값으로 지정한다.

```bash
export TARGET=hailo8  # hailo8, hailo10h, deepx 중 하나
export ARTIFACT=/absolute/path/to/target/yolov8s-pose/artifact
sha256sum "$ARTIFACT"
python3 src/main.py --model yolov8s-pose --target "$TARGET" \
  --artifact "$ARTIFACT" --no-compile --dataset "$COCO_ROOT" \
  --image-dir "$IMAGES" --label-dir "$LABELS" \
  --inference-mode async_queue --async-pass quality-and-runtime \
  --scenario offline --batch-size 1 --worker-count 4 --queue-capacity 256 \
  --max-samples 5000 --min-samples 5000 --min-duration-sec 0 \
  --schedule-seed 0 --warmup 10 \
  --results-path "results/yolov8s-pose-${TARGET}-async.csv"
```

Regulus 보드에는 직접 컴파일한 모델 MXQ와 NPU device 0이 필요하다. 현재 `main`의
Pose profile 이름을 명시한다.

```bash
export ARTIFACT=/absolute/path/to/regulus/yolov8s-pose/model.mxq
sha256sum "$ARTIFACT"
# 동기 E2E
python3 src/main.py --model yolov8s-pose --target mobilint-regulus \
  --artifact "$ARTIFACT" --no-compile --dataset "$COCO_ROOT" \
  --image-dir "$IMAGES" --label-dir "$LABELS" \
  --image-preprocess-profile regulus-vg-e2e-yolov8s-pose-raw-v1 \
  --image-preprocess-mode raw --layout NHWC \
  --inference-mode e2e --batch-size 1 --warmup 10 --max-steps 5000 \
  --results-path results/yolov8s-pose-regulus-e2e.csv

# async 품질·런타임
python3 src/main.py --model yolov8s-pose --target mobilint-regulus \
  --artifact "$ARTIFACT" --no-compile --dataset "$COCO_ROOT" \
  --image-dir "$IMAGES" --label-dir "$LABELS" \
  --image-preprocess-profile regulus-vg-e2e-yolov8s-pose-raw-v1 \
  --image-preprocess-mode raw --layout NHWC \
  --inference-mode async_queue --scenario offline --async-pass quality-and-runtime \
  --batch-size 1 --worker-count 4 --queue-capacity 256 \
  --runtime-option activation_slots=4 --warmup 10 \
  --max-samples 5000 --min-samples 5000 --min-duration-sec 0 \
  --schedule-seed 0 --flush-timeout-sec 600 --async-preload-inputs \
  --save-request-trace --results-path results/yolov8s-pose-regulus-async.csv
```

SDK 점검과 실행 절차는 [Hailo async 가이드](../../../docs/hailo-async-runtime.md),
[DeepX 가이드](../../../docs/deepx-setup.md),
[Regulus 가이드](../../../docs/mobilint-regulus-runtime.md)를 참조한다.
