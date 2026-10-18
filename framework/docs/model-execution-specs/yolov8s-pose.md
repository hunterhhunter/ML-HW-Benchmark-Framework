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
| `mobilint-regulus` / qbruntime 1.2.0 | 연구팀 컴파일 `model.mxq` | `async_queue runtime-only` |

## 입력 계약

| target | 전처리·Runtime 입력 |
|---|---|
| Hailo-8/10H | RGB → 640×640 letterbox, padding 114 → NHWC `uint8`. |
| DEEPX DX-M1 | RGB → 640×640 letterbox, padding 114. 입력 layout·dtype은 DXNN metadata에 맞춘다. |
| Mobilint Regulus | `regulus-vg-e2e-yolov8s-pose-raw-v1` profile: RGB → 640×640 letterbox → NHWC `uint8`. `qbruntime.Model.infer()` 사용. |

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

Regulus 보드에는 해당 MXQ와 NPU device 0이 필요하다. 보존된 5,000장 실행은
이전 profile 이름 `mobilint-vg-e2e-yolov8s-pose-raw-v1`을 사용했다. 아래는 현재
`main`이 요구하는 profile 이름이다.

```bash
export ARTIFACT=/absolute/path/to/regulus/yolov8s-pose/model.mxq
sha256sum "$ARTIFACT"
python3 src/main.py --model yolov8s-pose --target mobilint-regulus \
  --artifact "$ARTIFACT" --no-compile --dataset "$COCO_ROOT" \
  --image-dir "$IMAGES" --label-dir "$LABELS" \
  --image-preprocess-profile regulus-vg-e2e-yolov8s-pose-raw-v1 \
  --image-preprocess-mode raw --layout NHWC \
  --inference-mode async_queue --scenario offline --async-pass runtime-only \
  --batch-size 1 --worker-count 4 --queue-capacity 256 \
  --runtime-option activation_slots=4 --warmup 10 \
  --max-samples 5000 --min-samples 5000 --min-duration-sec 0 \
  --schedule-seed 0 --flush-timeout-sec 600 --async-preload-inputs \
  --save-request-trace --results-path results/yolov8s-pose-regulus-runtime-only.csv
```

SDK 점검과 실행 절차는 [Hailo async 가이드](../../../docs/hailo-async-runtime.md),
[DeepX 가이드](../../../docs/deepx-setup.md),
[Regulus 가이드](../../../docs/mobilint-regulus-runtime.md)를 참조한다.
