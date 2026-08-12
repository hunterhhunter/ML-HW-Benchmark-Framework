# Regulus qbruntime NPU-only 실행 가이드

`regulus` target은 qbcompiler 1.2로 만든 `.mxq`를 qbruntime 1.2에서 직접
실행한다. 검증한 Regulus RA 보드에서는 batch 1, device 0, bundle 0,
Cluster0/Core0으로 고정한다.

## NPU-only 및 E2E 조건

측정 전 다음 조건을 모두 확인한다.

- qbruntime 1.1.0 이상 및 `get_available_device_numbers()`의 device 0
- `ModelConfig.set_single_core_mode()`의 Cluster0/Core0 설정 성공
- `ModelConfig.force_single_npu_bundle(0)` 성공 및 getter 값 0
- `Model.launch()` 뒤 target core가 정확히 Cluster0/Core0
- MXQ와 입력의 shape, dtype, layout 일치
- 컴파일 manifest의 `cpu_offload=false`

측정은 실제 입력 전달과 출력 회수를 포함하는 `Model.infer()`를 사용한다.
전송을 생략하는 `infer_speedrun()`은 사용하지 않는다. CSV에는 다음 근거가
저장된다.

```text
runtime_name=qbruntime
runtime_version=v1.2.0
npu_only_verified=True
execution_binding=device=0,bundle=0,core=Cluster0/Core0
```

qbruntime v1.2.0에서는 동기 E2E만 허용한다. 이 버전의 `infer_async()`는 서로
다른 요청에 첫 출력 또는 zero 출력이 반복되는 문제가 실기기에서 재현되어,
프레임워크가 1.3.2 미만의 native async를 차단한다.

## Mobilint Model Zoo의 Ultralytics 모델 고정

비교 대상은 Mobilint가 배포한 Ultralytics 계열 ONNX를 리비전과 SHA-256으로
고정한다. 이름만 같은 다른 YOLO export로 바꾸지 않는다.

| 모델 | Mobilint 저장소 | 고정 리비전 | ONNX SHA-256 |
| --- | --- | --- | --- |
| YOLOv5m | `mobilint/YOLOv5m` | `4545117add8fb3bea343d944e3f6e91ee0b4910c` | `91500bbd772dd51d1cf1f36eefdf70ca4901991a65859070b99f2b07932bec19` |
| YOLOv8s-pose | `mobilint/YOLOv8s-pose` | `8d63aa2377c0cd180af965bd635e73ff0021af99` | `1c307373041c0a876c87c34b9e647ae39c81a5a9475b833cea8fecd6402eca8e` |

두 저장소에서 배포하는 MXQ는 `aries/` 아래에만 있다. Regulus용 공식 MXQ는
없으므로 Aries MXQ를 Regulus 결과로 사용하지 않는다. Model Zoo의 공식 NPU
정확도(YOLOv5m box mAP50-95 44.446%, YOLOv8s-pose pose mAP50-95 57.007%)도
Regulus 측정값으로 바꿔 적지 않는다.

원본은 다음처럼 정확히 내려받고 검증할 수 있다.

```bash
cd framework
mkdir -p models/mobilint_model_zoo/yolov5m \
  models/mobilint_model_zoo/yolov8s-pose

curl -L \
  https://huggingface.co/mobilint/YOLOv5m/resolve/4545117add8fb3bea343d944e3f6e91ee0b4910c/yolov5m.onnx \
  -o models/mobilint_model_zoo/yolov5m/yolov5m.onnx
curl -L \
  https://huggingface.co/mobilint/YOLOv8s-pose/resolve/8d63aa2377c0cd180af965bd635e73ff0021af99/yolov8s-pose.onnx \
  -o models/mobilint_model_zoo/yolov8s-pose/yolov8s-pose.onnx

sha256sum models/mobilint_model_zoo/yolov5m/yolov5m.onnx \
  models/mobilint_model_zoo/yolov8s-pose/yolov8s-pose.onnx
```

컴파일 도구는 위 두 해시와 동적 batch 입출력 계약이 모두 맞아야 공식
Model Zoo 모델로 승인한다. YOLOv5m은 compiler가 제거하는 Detect decode와 같은
세 raw head를 추출해도 framework decoder 결과가 같은지 별도로 검증한다.

## CPU 기준선

공식 ONNX와 프레임워크의 전처리·후처리를 먼저 CPU에서 검증했다.

| 모델/그래프 | 표본 | 정확도 | 평균 detection | 근거 CSV |
| --- | ---: | ---: | ---: | --- |
| ResNet50 동일 ONNX | ImageNet 3,000 | Top-1 `77.0333%`, Top-5 `92.9%` | - | `framework/results/resnet50-torchvision-cpu-3000.csv` |
| YOLOv5m 전체 ONNX | COCO128 10 | mAP@0.5 `0.9154473509` | 178.6 | `framework/results/yolov5m-ultralytics-val-native-nms-onnx-smoke-10.csv` |
| YOLOv5m 세 raw head | COCO128 10 | mAP@0.5 `0.7804388422` | 4.1 | `framework/results/yolov5m-mobilint-official-raw-heads-onnx-smoke-10.csv` |
| YOLOv8s-pose | COCO val 20 | OKS mAP `0.6567023820` | 24.75 | `framework/results/yolov8s-pose-ultralytics-val-onnx-smoke-20.csv` |

YOLOv5m 전체 graph와 raw-head graph는 기존 기본 threshold smoke에서 같은
결과를 냈다. 최종 Regulus 정확도 판정은 Ultralytics validation 조건인
`conf=0.001`, YOLOv5m `IoU=0.65`/multi-label, YOLOv8s-pose `IoU=0.7`,
`max_nms=30000`, `max_det=300`으로 다시 수행했다. 작은 smoke subset 수치는
공식 전체 COCO 수치와 직접 비교하지 않는다.

## Regulus MXQ 컴파일

컴파일은 x86-64 Ubuntu 22.04의 qbcompiler 1.2 환경에서 수행한다. 공식 Model
Zoo의 `best_result.json` 설정을 compiler API 값으로 변환해 적용한다.

| 모델 | percentile 인자 | top-k ratio | method/mode/output |
| --- | ---: | ---: | --- |
| YOLOv5m | `0.9872346388350748` | `0.004753684085368` | `1/1/1` |
| YOLOv8s-pose | `0.9880598600374023` | `0.0072453577151727145` | `1/1/1` |

```bash
cd framework

python tools/mobilint_regulus_compile.py \
  --model yolov5m \
  --onnx models/mobilint_model_zoo/yolov5m/yolov5m.onnx \
  --calibration-source datasets/coco128/images/train2017 \
  --calibration-count 100 --input-mode float32 \
  --calibration-method 1 --calibration-mode 1 \
  --calibration-output-mode 1 \
  --calibration-percentile 0.9872346388350748 \
  --calibration-topk-ratio 0.004753684085368 \
  --onnx-parser new --compiler-device cuda \
  --target-device regulus-ra \
  --output-dir artifacts/regulus-mobilint-model-zoo-recipe

python tools/mobilint_regulus_compile.py \
  --model yolov8s-pose \
  --onnx models/mobilint_model_zoo/yolov8s-pose/yolov8s-pose.onnx \
  --calibration-source datasets/coco/images/val2017 \
  --calibration-count 100 --input-mode float32 \
  --calibration-method 1 --calibration-mode 1 \
  --calibration-output-mode 1 \
  --calibration-percentile 0.9880598600374023 \
  --calibration-topk-ratio 0.0072453577151727145 \
  --onnx-parser new --compiler-device cuda \
  --target-device regulus-ra \
  --output-dir artifacts/regulus-mobilint-model-zoo-recipe
```

이 recipe의 MXQ 입력 계약은 normalized float32/NHWC다. 시험한 ResNet50 MXQ는
raw uint8/NHWC 계약이므로 모델별 MXQ dtype을 동일하다고 가정하면 안 된다.
프레임워크는 실행 전에 실제 MXQ의 shape와 dtype을 검사한다.

## 실기기 검증 결과

| 모델 | 상태 | 결과 |
| --- | --- | --- |
| ResNet50 | 정확도 미승인 | 동일 ONNX CPU Top-1 `77.0333%`/Top-5 `92.9%`; Regulus MXQ `55.3667%`/`78.7667%`, NPU-only |
| 공식 Model Zoo YOLOv5m | 정확도 미승인 | CPU mAP@0.5 `0.9154473509`; Regulus MXQ mAP `0.0`, 평균 detection `300.0`, NPU-only |
| 공식 Model Zoo YOLOv8s-pose | 정확도 미승인 | CPU OKS mAP `0.6567023820`; Regulus MXQ OKS mAP `0.0`, 평균 detection `290.95`, NPU-only |

세 MXQ 모두 `cpu_offload=false`, single scheme, Regulus RA target이며 보드의
device 0/bundle 0/Core0에서 실제 `Model.infer()`가 실행됐다. 초기 `conf=0.25`
실행에서는 후보가 0개였지만, Ultralytics의 `conf=0.001` 및 class-aware NMS를
적용하면 오히려 300개 상한까지 후보가 생성된다. 그럼에도 true positive가 없어
정확도 승인을 통과하지 못했다. YOLOv5m은 동일 이미지에서 ONNX와 compiler의
부동소수점 중간 그래프(MBLT) raw head가 일치했지만, MXQ의 head별 objectness
상관계수는 `0.2900`, `0.0524`, `0.2979`에 그쳤다. Pose는 입력 계약 및 max/layer/
channel/zero-point/sigmoid calibration 변형도 시험했으나 OKS mAP가 모두 0이었다.
이는 전처리·parser가 아니라 Regulus 양자화 산출물의 정확도 문제를 가리킨다.

YOLOv5m은 qbruntime `v1.2.0`에서 같은 normalized float32 NHWC 입력을 한 model
instance에 열 번 연속 `Model.infer()`했을 때도 raw P3 head가 안정화하지 않았다.
연속 호출 간 최대 절대차는 `4.80`~`13.63`이었다. 이 검사는 device 0, bundle 0,
Cluster0/Core0 고정과 실제 input/output 전송을 포함했으며 `infer_speedrun()`은
사용하지 않았다. 상세 절차와 원시 수치는
`docs/evidence/regulus-v1.2/yolov5m-qbruntime-v1.2-repeatability.json`에 남겼다.

ResNet50 역시 동일 source ONNX와 동일 3,000장에서 Top-1이 `21.6667%p`, Top-5가
`14.1333%p` 낮았다. 입력 계약과 normalization/calibration 변형을 100장에서
추가 점검했지만 최고 MXQ도 CPU `77%/94%` 대비 `60%/81%`였다. 따라서 세 모델
모두 latency와 throughput을 최종
NPU 비교표에 넣지 않는다. 근거는 다음 파일에 있다.

- `framework/results/resnet50-torchvision-cpu-3000.csv`
- `docs/evidence/regulus-v1.2/regulus-v1.2-resnet50-3000.csv`
- `docs/evidence/regulus-v1.2/yolov5m-mobilint-official-regulus-recipe.manifest.json`
- `docs/evidence/regulus-v1.2/yolov5m-ultralytics-val-regulus-smoke-10.csv`
- `docs/evidence/regulus-v1.2/yolov8s-pose-mobilint-official-regulus-recipe.manifest.json`
- `docs/evidence/regulus-v1.2/yolov8s-pose-ultralytics-val-regulus-smoke-20.csv`
- `docs/evidence/regulus-v1.2/yolov8s-pose-global-layer-regulus-smoke-20.csv`

즉, 공식 Ultralytics 계열 ONNX의 신원과 CPU 기준선, NPU-only 실행은 확인됐지만
qbcompiler/qbruntime 1.2로 직접 만든 Regulus MXQ의 정확도 동등성은 확인되지
않았다. 현재 YOLO recipe는 Mobilint Model Zoo의 Aries reference calibration
값을 Regulus RA에 적용한 것이므로, Mobilint가 승인한 Regulus용 MXQ 또는
Regulus 전용 compiler recipe가 나올 때까지 세 모델을 비교 대상에서 제외한다.

## 보드 실행 예

```bash
cd /root/ml-hw-benchmark

python3 src/main.py --model resnet50 --target regulus \
  --artifact /path/to/candidate-resnet50.mxq \
  --dataset datasets/imagenet_1k --layout NHWC --batch-size 1 \
  --warmup 10 --max-steps 3000 --no-compile \
  --results-path results/regulus-v1.2-resnet50-3000.csv
```

YOLO 계열은 새 MXQ를 만든 뒤 먼저 10~20장 smoke accuracy를 통과해야 full run을
허용한다. `npu_only_verified=True`만으로 정확도 동등성이 입증되는 것은 아니다.

## 비교 범위

Jetson+M.2 NPU와 Regulus SOM의 물리적 호스트 보드는 같게 만들 수 없다. 비교 시
동일 모델 weight, 데이터셋, batch, warmup, 전처리, E2E 측정 구간을 고정하고
각 벤더의 NPU-only binding을 기록한다. 따라서 결과 명칭은 “동일 보드의 NPU
비교”가 아니라 “조건을 표준화한 전체 시스템 비교”로 표기한다.
