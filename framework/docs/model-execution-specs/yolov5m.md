# YOLOv5m 실행 명세: 에지 NPU

`main@30c4db75`의 실행 경로를 기준으로 확인된 조합과 재현 명령을 정리한다.
측정 구간과 지표는 [측정 계약](../measurement-contract.md)을 따른다.
실행 자산의 checksum은 [별도 기록](yolov5m-sha256.md)에 있다.

## 모델·데이터

- 모델: `yolov5m` (`OBJECT_DETECTION`, `ObjectDetectionEvaluator`)
- 입력: COCO 2017 validation 이미지 5,000장과 이미지별 YOLO 형식 라벨
  `labels/val2017_yolo/*.txt`. `--label-dir`에는 이 디렉터리를 지정한다.

## 확인된 조합

| target / Runtime | 사용 artifact | 확인된 실행 모드 |
|---|---|---|
| `hailo8` / HailoRT 4.24.0 | 벤더 제공 `yolov5m_8.hef` | 동기 `e2e`, `async_queue` 품질·런타임 패스 |
| `hailo10h` / HailoRT 5.3.0 | 벤더 제공 `yolov5m_10.hef` | 동기 `e2e`, `async_queue` 품질·런타임 패스 |
| `deepx` / DX-RT 3.4.0 | 벤더 제공 `yolov5-m_640x640.dxnn` | 동기 `e2e`, `async_queue` 품질·런타임 패스 |
| `mobilint-regulus` / qbruntime 1.2.0 | 직접 컴파일 모델 MXQ `model.mxq` | 동기 `e2e`, `async_queue` 품질·런타임 패스 |

## 입력 계약

| target | 전처리·Runtime 입력 |
|---|---|
| Hailo-8/10H | RGB → 640×640 letterbox, padding 114 → raw `uint8`. |
| DEEPX DX-M1 | RGB → 640×640 letterbox, padding 114. 입력 layout·dtype은 DXNN metadata에 맞춘다. |
| Mobilint Regulus | `mobilint-yolov5m-default` profile: RGB → 640×640 letterbox → NHWC `uint8`. `qbruntime.Model.infer()` 사용. |

## 출력 계약

평가기에는 decoder가 만든 `detections` 배열 `(N, 7)`을 전달한다. 한 행은
`[배치 내 이미지 인덱스, class ID, confidence, x1, y1, x2, y2]`이며, 좌표는
640×640 모델 입력 기준이다. 탐지가 없으면 `N=0`이다. 이 공통 형식에 이르는
Runtime 출력과 NMS 위치는 target별로 다르다.

| target | Runtime 출력 → `detections` 변환 |
|---|---|
| Hailo-8/10H | YOLOv5m HEF의 NMS 적용 출력. HailoRT가 class-major dense 텐서 `(B, class, 5, 최대 박스 수)` 또는 class별 가변 길이 목록으로 줄 수 있다. `HailoYoloNMSDecoder`가 기본 `ymin, xmin, ymax, xmax, score` 순서를 `xyxy`로 바꾸고 공통 행을 만든다. HEF 내부 NMS를 host 공통 NMS로 간주하지 않는다. |
| DEEPX DX-M1 | 현재 `RawYoloDetectionDecoder` 경로는 후보별 `[cx, cy, w, h, objectness, 80개 class score]`를 담은 단일 `(B, 후보, 85)` 텐서로 해석하고 host에서 박스 변환·NMS를 수행한다. 기본 ONNX 프로필의 `(1, 25200, 85)`는 DXNN의 물리 출력 shape 보증이 아니므로 실제 DX-RT 출력 이름·shape와 특징 축 순서를 확인한다. |
| Mobilint Regulus | MXQ 출력이 프레임워크에서 batch 1 NHWC raw head 3개 `(1, 20, 20, 255)`, `(1, 40, 40, 255)`, `(1, 80, 80, 255)`로 정규화된다. `255 = 3 × (5 + 80)`이며, `MobilintYoloV5HeadDecoder`가 공간 크기로 head를 구분해 anchor decode와 host NMS를 수행한다. |

## 재현 명령

각 target의 SDK가 설치된 장비에서 실행한다. 아래 경로를 실제 데이터·artifact
위치에 맞춘다. 동기 smoke는 `--max-steps 5000`을 `--max-steps 20`으로 바꾼다.

```bash
cd /path/to/ML-HW-Benchmark-Framework/framework
export COCO_ROOT=/absolute/path/to/coco
export IMAGES="$COCO_ROOT/images/val2017"
export LABELS="$COCO_ROOT/labels/val2017_yolo"
test -d "$IMAGES" && test -d "$LABELS"
```

```bash
# Hailo-8
export ARTIFACT=/absolute/path/to/yolov5m_8.hef
sha256sum "$ARTIFACT"
python3 src/main.py --model yolov5m --target hailo8 \
  --artifact "$ARTIFACT" --no-compile --dataset "$COCO_ROOT" \
  --image-dir "$IMAGES" --label-dir "$LABELS" \
  --inference-mode e2e --batch-size 1 --warmup 10 --max-steps 5000 \
  --runtime-option timeout_ms=10000 --results-path results/yolov5m-hailo8-e2e.csv

# Hailo-10H
export ARTIFACT=/absolute/path/to/yolov5m_10.hef
sha256sum "$ARTIFACT"
python3 src/main.py --model yolov5m --target hailo10h \
  --artifact "$ARTIFACT" --no-compile --dataset "$COCO_ROOT" \
  --image-dir "$IMAGES" --label-dir "$LABELS" \
  --inference-mode e2e --batch-size 1 --warmup 10 --max-steps 5000 \
  --results-path results/yolov5m-hailo10h-e2e.csv

# DEEPX DX-M1
export ARTIFACT=/absolute/path/to/yolov5-m_640x640.dxnn
sha256sum "$ARTIFACT"
python3 src/main.py --model yolov5m --target deepx \
  --artifact "$ARTIFACT" --no-compile --dataset "$COCO_ROOT" \
  --image-dir "$IMAGES" --label-dir "$LABELS" \
  --inference-mode e2e --batch-size 1 --warmup 10 --max-steps 5000 \
  --runtime-option device_ids=0 --runtime-option bound_option=NPU_ALL \
  --runtime-option buffer_count=6 --results-path results/yolov5m-deepx-e2e.csv
```

Hailo-8·Hailo-10H·DEEPX의 async 조합은 batch 1, worker 4, offline
5,000장이다. `TARGET`과 `ARTIFACT`를 같은 행의 값으로 지정한다.

```bash
export TARGET=hailo8  # hailo8, hailo10h, deepx 중 하나
export ARTIFACT=/absolute/path/to/target/yolov5m/artifact
sha256sum "$ARTIFACT"
python3 src/main.py --model yolov5m --target "$TARGET" \
  --artifact "$ARTIFACT" --no-compile --dataset "$COCO_ROOT" \
  --image-dir "$IMAGES" --label-dir "$LABELS" \
  --inference-mode async_queue --async-pass quality-and-runtime \
  --scenario offline --batch-size 1 --worker-count 4 --queue-capacity 256 \
  --max-samples 5000 --min-samples 5000 --min-duration-sec 0 \
  --schedule-seed 0 --warmup 10 \
  --results-path "results/yolov5m-${TARGET}-async.csv"
```

Regulus 보드에는 직접 컴파일한 모델 MXQ와 NPU device 0이 필요하다.

```bash
export ARTIFACT=/absolute/path/to/regulus/yolov5m/model.mxq
sha256sum "$ARTIFACT"
# 동기 E2E
python3 src/main.py --model yolov5m --target mobilint-regulus \
  --artifact "$ARTIFACT" --no-compile --dataset "$COCO_ROOT" \
  --image-dir "$IMAGES" --label-dir "$LABELS" \
  --image-preprocess-profile mobilint-yolov5m-default \
  --image-preprocess-mode raw --layout NHWC \
  --inference-mode e2e --batch-size 1 --warmup 10 --max-steps 5000 \
  --results-path results/yolov5m-regulus-e2e.csv

# async 품질·런타임
python3 src/main.py --model yolov5m --target mobilint-regulus \
  --artifact "$ARTIFACT" --no-compile --dataset "$COCO_ROOT" \
  --image-dir "$IMAGES" --label-dir "$LABELS" \
  --image-preprocess-profile mobilint-yolov5m-default \
  --image-preprocess-mode raw --layout NHWC \
  --inference-mode async_queue --scenario offline --async-pass quality-and-runtime \
  --batch-size 1 --worker-count 4 --queue-capacity 256 \
  --runtime-option activation_slots=4 --warmup 10 \
  --max-samples 5000 --min-samples 5000 --min-duration-sec 0 \
  --schedule-seed 0 --flush-timeout-sec 600 --async-preload-inputs \
  --save-request-trace --results-path results/yolov5m-regulus-async.csv
```

SDK 점검과 실행 절차는 [Hailo async 가이드](../../../docs/hailo-async-runtime.md),
[DeepX 가이드](../../../docs/deepx-setup.md),
[Regulus 가이드](../../../docs/mobilint-regulus-runtime.md)를 참조한다.
