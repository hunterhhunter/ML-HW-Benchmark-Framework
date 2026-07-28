# Mobilint MXQ Vision Pre/Post-processing Design

**상태:** 승인됨

**작성일:** 2026-07-22

**승인일:** 2026-07-22

**대상 범위:** Mobilint raw MXQ vision artifact profile, ResNet50 전처리,
YOLOv5m 전처리·후처리, runtime input/output contract 검증

## 1. 문제와 실제 장치 증거

Mobilint Model Zoo의 ARIES용 `resnet50_IMAGENET1K_V2.mxq`는 정상
실행됐지만, 같은 artifact를 프레임워크의 `mobilint-aries` target으로 실행하면 첫
warmup에서 다음 오류가 발생했다.

```text
Invalid input data type - Input: Float32, Supported: Uint8
QbRuntimeError: Model_DtypeMismatched
```

Model Zoo와 프레임워크 경로를 비교한 결과 ResNet50 V2 MXQ의 실제 입력 계약은 다음과
같다.

| 항목 | Mobilint Model Zoo | 기존 generic 분류 loader |
|---|---|---|
| resize | 짧은 변 232, PIL bilinear | 짧은 변 256, PIL bilinear |
| crop | 중앙 224×224, Python `round` | 중앙 224×224, floor |
| 색상·layout | RGB HWC | RGB, CLI에 따라 CHW/HWC |
| 값·dtype | 0..255 `uint8` | ImageNet 정규화 `float32` |
| runtime batch | `(1,224,224,3)` | NHWC 선택 시 같은 shape이나 dtype 불일치 |

추가로 실제 ARIES2 서버에서 Mobilint Model Zoo의 `mobilint/YOLOv5m` DEFAULT artifact를
다운로드해 qb Runtime metadata와 전처리 결과를 확인했다.

```text
Artifact: framework/models/mobilint/yolov5m/aries/yolov5m.mxq
Input dtype: DataType.Uint8
Input shapes: [(640, 640, 3)]
Output shapes: [(20, 20, 255), (40, 40, 255), (80, 80, 255)]

Original size: (500, 375) RGB
Preprocessed tensor: (640, 640, 3) uint8, contiguous
ratio_pad: ((1.28, 1.28), (0, 80))
Runtime batch: (1, 640, 640, 3) uint8
```

YOLOv5m의 세 output은 최종 `(B,25200,85)` prediction이 아니라 stride 32/16/8의 raw
detection heads다. 따라서 정확한 벤치마크를 위해 입력 letterbox뿐 아니라 anchor/grid
decode와 NMS도 필요하다.

## 2. 결정

Mobilint vision 지원을 ResNet 전용 loader로 구현하지 않는다. 공통
`MobilintVisionArtifactProfile` registry가 artifact별 입력·출력 계약과 typed recipe를
선택하고, task별 loader와 decoder가 기존 프레임워크 규약을 따른다.

책임은 다음처럼 나눈다.

- profile/resolver: model, task, artifact basename을 정확한 recipe와 tensor contract로
  연결한다.
- classification loader: dataset/cursor/cache를 재사용하고 ResNet recipe를 실행한다.
- detection loader: YOLO label/cursor/cache/context를 재사용하고 YOLOv5 recipe를
  실행한다.
- runtime: qb Runtime lifecycle, sync/native-async 호출, artifact metadata와 실제 배열
  계약을 검증한다.
- decoder: YOLOv5 raw heads를 decode한 후 Model Zoo와 같은 combined-confidence
  multi-label 후보 생성 및 class-aware NMS를 거쳐 canonical detection 형식으로 변환한다.
- evaluator: 기존 `preprocess_context`로 ground truth를 letterbox 좌표계에 맞춘다.

runtime은 입력을 cast/normalize하거나 YOLO 후처리를 수행하지 않는다. decoder가 NPU
실행 시간을 오염시키지 않도록 기존 inference pipeline의 runtime 호출 뒤에 유지한다.

ARIES와 REGULUS는 같은 artifact profile, preprocessor, decoder를 공유한다. 장치 family는
device selection, driver validation, monitor, runtime target에만 영향을 주며 vision recipe
key가 아니다.

## 3. 검토한 대안

### 3.1 선택: 공통 profile + task별 loader

입력·출력 계약 해석은 공유하면서 classification과 detection의 label/context 규약은
분리한다. 새 task는 profile recipe와 얇은 loader/decoder adapter만 추가하면 된다.

### 3.2 기각: 하나의 universal Mobilint vision loader

분류 label, YOLO box label, segmentation mask, pose keypoint, task별 cache/context가 한
클래스의 조건문으로 모인다. 현재 두 모델에는 가능하지만 다음 task에서 책임이
얽히므로 선택하지 않는다.

### 3.3 기각: generic/Hailo object preprocessor 변경

현재 generic YOLO, Hailo, DeepX의 interpolation, normalization, layout 계약을 바꿀
위험이 있다. Mobilint exact recipe는 별도 구현하고 기존 backend 기본값을 유지한다.

### 3.4 기각: production에서 Model Zoo `MBLT_Engine` 재사용

`MBLT_Engine` 생성은 전처리기만 만드는 것이 아니라 별도 qbruntime model 생성과 NPU
launch를 수행한다. raw `MobilintRuntime`과 함께 사용하면 model lifecycle이 중복된다.
Model Zoo YAML과 구현은 parity 기준으로만 사용한다.

## 4. 공통 artifact profile

새 `MobilintVisionArtifactProfile`은 frozen dataclass다.

```text
MobilintVisionArtifactProfile
  profile_id
  model_name
  task
  artifact_basenames
  preprocess_mode
  color_order
  input_layout
  input_dtype
  unbatched_input_shape
  max_batch_size
  input_recipe
  expected_output_shapes
  output_recipe
  decoder_defaults
```

`input_recipe`는 arbitrary dict가 아니라 다음 typed union이다.

```text
ResNetCenterCropRecipe
  resize_short_side
  crop_hw
  interpolation
  resize_rounding
  crop_rounding
  version

YoloV5LetterboxRecipe
  input_hw
  interpolation
  resize_rounding
  padding_rounding
  pad_color
  version
```

`output_recipe`는 없거나 `YoloV5RawHeadRecipe`다.

```text
YoloV5RawHeadRecipe
  class_count
  anchors_by_stride
  expected_heads
  version
```

runtime에 전달하는 계약은 task와 recipe를 포함하지 않는다.

```text
vision_profile_id
expected_input_dtype
expected_input_layout
expected_unbatched_input_shape
max_input_batch_size
expected_unbatched_output_shapes (optional)
```

profile과 runtime contract key는 사용자가 `--runtime-option`으로 바꿀 수 없다. core mode,
activation slots, timeout 같은 실행 tuning option만 CLI override를 허용한다.

## 5. 최초 등록 profile

### 5.1 ResNet50 ImageNet V2

```text
profile_id: mobilint-resnet50-imagenet1k-v2
model_name: resnet50
task: IMAGE_CLASSIFICATION
artifact_basenames: [resnet50_IMAGENET1K_V2.mxq]
preprocess_mode: raw
color_order: RGB
input_layout: NHWC
input_dtype: uint8
unbatched_input_shape: [224,224,3]
max_batch_size: 1
input_recipe:
  kind: resnet_center_crop
  resize_short_side: 232
  crop_hw: [224,224]
  interpolation: pil_bilinear
  resize_rounding: integer_truncation
  crop_rounding: python_round
```

### 5.2 YOLOv5m DEFAULT

```text
profile_id: mobilint-yolov5m-default
model_name: yolov5m
task: OBJECT_DETECTION
artifact_basenames: [yolov5m.mxq]
preprocess_mode: raw
color_order: RGB
input_layout: NHWC
input_dtype: uint8
unbatched_input_shape: [640,640,3]
max_batch_size: 1
input_recipe:
  kind: yolov5_letterbox
  input_hw: [640,640]
  interpolation: opencv_linear
  resize_rounding: python_round
  padding_rounding: ultralytics_minus_plus_0_1
  pad_color: [114,114,114]
expected_output_shapes:
  - [20,20,255]
  - [40,40,255]
  - [80,80,255]
output_recipe:
  kind: yolov5_raw_heads
  class_count: 80
  anchors_by_stride:
    8:  [[10,13], [16,30], [33,23]]
    16: [[30,61], [62,45], [59,119]]
    32: [[116,90], [156,198], [373,326]]
decoder_defaults:
  confidence_threshold: 0.001
  iou_threshold: 0.65
  max_detections: 300
  max_nms_candidates: 30000
  max_class_offset: 7680
```

`YOLOv5mu`, P6, segmentation, pose artifact는 이름이 비슷해도 이 profile과 자동 매칭하지
않는다.

## 6. Profile 선택과 CLI

`--image-preprocess-profile`의 기본값은 `auto`다.

- `auto`: `(normalized model name, Task, exact artifact basename)`이 registry entry와 모두
  일치할 때만 선택한다.
- explicit profile ID: rename된 artifact에도 profile을 적용하되 model과 task는 계속
  일치해야 한다.
- 알 수 없는 Mobilint vision MXQ: generic float preprocessing으로 fallback하지 않고
  사용 가능한 profile ID를 포함해 실패한다.
- `--image-preprocess-mode normalized`: 두 초기 raw profile과 충돌하므로 실패한다.
- default `--layout`: profile layout인 NHWC로 확정한다.
- explicit NCHW: profile과 충돌하므로 조용히 덮어쓰지 않고 실패한다.
- non-auto profile: Mobilint raw vision target에만 허용한다.

resolver는 profile 하나를 한 번만 선택한다. 같은 객체가 artifact-local `Model_Spec`,
loader, runtime contract, decoder에 전달된다.

## 7. ResNet50 전처리

ResNet strategy는 다음 순서를 정확히 따른다.

1. PIL로 읽고 RGB로 변환한다.
2. 짧은 변을 232로 맞추고 반대 변은 integer truncation으로 계산한다.
3. PIL bilinear interpolation으로 resize한다.
4. Python `round` 규칙으로 중앙 224×224 crop 위치를 계산한다.
5. normalize하지 않고 `numpy.uint8`을 유지한다.
6. 기존 image cache convention에 맞춰 contiguous CHW로 저장한다.
7. loader가 runtime 직전에 NHWC로 바꾼다.

Model Zoo와 parity test는 가로·세로 이미지 및 crop offset이 홀수인 이미지에서 픽셀
단위 `array_equal`을 요구한다.

## 8. YOLOv5m 전처리와 context

YOLO preprocessor는 Model Zoo의 `Reader(numpy) -> LetterBox -> SetOrder(HWC)`를
재현한다. MXQ가 `DataType.Uint8`이므로 `Normalize(cv)`는 적용하지 않는다.

원본 `(h0,w0,3)`에 대해 다음을 수행한다.

```text
r = min(640 / h0, 640 / w0)
new_w = int(round(w0 * r))
new_h = int(round(h0 * r))
resize = OpenCV INTER_LINEAR
dw = (640 - new_w) / 2
dh = (640 - new_h) / 2
left  = int(round(dw - 0.1))
right = int(round(dw + 0.1))
top   = int(round(dh - 0.1))
bottom= int(round(dh + 0.1))
padding = RGB (114,114,114)
```

출력은 contiguous `(640,640,3)` `uint8`이고 pipeline collate 뒤
`(1,640,640,3)`이 된다.

기존 evaluator와 호환되도록 sample에 다음 context를 넣는다.

```text
original_width
original_height
input_width
input_height
scale
pad_x = left
pad_y = top
layout = NHWC
resize_mode = letterbox
ratio_pad = ((r,r),(left,top))
profile_id
```

generic `ObjectDetectionPreprocessor`의 PIL bicubic, truncation, floor padding 구현은
Model Zoo와 다르므로 재사용하지 않는다. 기존 YOLO/Hailo/DeepX 동작도 변경하지 않는다.

cache key에는 profile ID, recipe kind/version, input size, OpenCV interpolation,
resize/padding rounding, pad color, layout, dtype를 모두 포함한다. 기존
`letterbox_raw_NHWC_640x640.npz`와 절대 공유하지 않는다.

## 9. YOLOv5m raw-head decoder

`MobilintYoloV5HeadDecoder`는 프레임워크 `DetectionDecoder`를 구현한다. production에서
Model Zoo postprocessor나 Torch를 import하지 않고 NumPy로 처리한다.

### 9.1 Head 정규화

- 정확히 세 output을 요구한다.
- `(H,W,255)`는 batch 1을 추가한다.
- `(B,H,W,255)`는 그대로 사용한다.
- head 순서와 output 이름은 신뢰하지 않는다.
- spatial size로 stride를 계산해 `(80,80)->8`, `(40,40)->16`, `(20,20)->32`에
  매칭한다.
- 중복 spatial size, batch 불일치, channel 255 불일치, NCHW/알 수 없는 layout은
  명시적으로 실패한다.

### 9.2 Anchor/grid decode

각 head를 `(B,H,W,3,85)`로 reshape하고 다음 수식을 적용한다.

```text
xy  = (sigmoid(raw_xy) * 2 - 0.5 + grid_xy) * stride
wh  = (sigmoid(raw_wh) * 2) ** 2 * anchor_wh
obj = sigmoid(raw_obj)
cls = sigmoid(raw_cls)
```

각 scale을 `(B,H*W*3,85)`로 flatten하고 concatenate한다.

```text
80*80*3 + 40*40*3 + 20*20*3 = 25200
decoded shape = (B,25200,85)
```

### 9.3 Confidence filtering과 NMS

기존 `RawYoloDetectionDecoder`에 그대로 전달하지 않는다. 해당 decoder는 YOLOv5에서
objectness만 먼저 threshold하고 anchor마다 최고 class 하나만 고르며 class-agnostic NMS를
수행한다. Mobilint Model Zoo 경로와 다음 세 가지가 다르므로 mAP parity를 보장할 수 없다.

`MobilintYoloV5HeadDecoder`는 Model Zoo의 anchor-based postprocess 순서를 NumPy로
재현한다.

1. raw objectness logit이 confidence threshold의 inverse-sigmoid보다 큰 anchor만 먼저
   남긴다. 이는 불필요한 sigmoid/decode 계산을 줄이는 동등한 prefilter다.
2. 각 anchor에서 `score[class] = sigmoid(objectness) * sigmoid(class_logit)`을 계산한다.
3. combined score가 threshold보다 큰 모든 `(anchor, class)` 조합을 후보로 만든다. 즉 한
   anchor가 여러 class 후보를 만들 수 있는 multi-label 동작을 보존한다.
4. score 내림차순으로 최대 `max_nms_candidates=30000`개를 남긴다.
5. box에 `class_id * max_class_offset` 좌표 offset을 더한 뒤 공통 NumPy NMS primitive를
   호출해 class-aware NMS를 수행한다.
6. 최대 `max_detections=300`개를 canonical row로 변환한다.

```text
[local_image_index, class_id, confidence, x1, y1, x2, y2]
```

공통 NumPy NMS primitive는 public decoder utility로 노출하되 기존
`RawYoloDetectionDecoder`의 filtering/NMS 의미는 변경하지 않는다. 따라서 Hailo, DeepX,
generic YOLO regression에 영향이 없다.

Model Zoo COCO mAP parity 기본값은 confidence `0.001`, IoU `0.65`, 최대 detection 300이다.
사용자가 명시한 decoder threshold는 override로 허용한다. 일반 시각화에서 더 높은
confidence를 쓰는 것은 허용하지만 benchmark 결과에는 effective threshold를 기록한다.

## 10. DataLoader와 decoder factory

새 task별 adapter는 기존 구현을 상속한다.

- `MobilintImageClassificationLoader(ImageClassificationLoader)`
  - ResNet strategy 주입
  - profile metadata와 runtime contract 제공
- `MobilintObjectDetectionLoader(ObjectDetectionLoader)`
  - Mobilint YOLO preprocessor 주입
  - 기존 YOLO label parsing, cursor, batch, context 전달 재사용
  - profile metadata, runtime contract, decoder defaults 제공

factory routing은 다음과 같다.

```text
backend=mobilint + IMAGE_CLASSIFICATION -> MobilintImageClassificationLoader
backend=mobilint + OBJECT_DETECTION     -> MobilintObjectDetectionLoader
backend=mobilint + unsupported vision task -> explicit unsupported error
other backends/tasks -> unchanged
```

decoder factory는 `backend=mobilint`, task `OBJECT_DETECTION`, selected output recipe
`yolov5_raw_heads`일 때 `MobilintYoloV5HeadDecoder`를 만든다. profile 없는 Mobilint
detection이나 다른 backend decoder는 기존 규약을 유지한다.

## 11. Artifact-local Model_Spec

profile 적용 helper는 frozen `Model_Spec`을 mutate하지 않고 새 instance를 만든다.

ResNet50은 첫 input을 `(1,224,224,3)` `uint8`로 바꾼다.

YOLOv5m은 첫 input과 output을 다음처럼 바꾼다.

```text
input: (1,640,640,3) uint8
outputs, qb Runtime 순서 기준:
  mobilint_yolov5_stride32: (1,20,20,255)
  mobilint_yolov5_stride16: (1,40,40,255)
  mobilint_yolov5_stride8:  (1,80,80,255)
```

decoder는 이름이나 순서 대신 실제 spatial size를 사용한다. runtime은 output 개수와
metadata shape multiset이 profile과 일치하는지 검증한다.

## 12. MobilintRuntime 계약 검증

runtime은 input contract가 제공된 vision profile에 한해 SDK v1.3.2 metadata getter를
필수로 사용한다.

- `get_model_input_shape()`
- `get_model_input_data_type()`
- `get_model_output_shape()` when expected output shapes exist

load 시 artifact metadata를 정규화해 profile과 비교한다. mismatch나 getter 부재는
검증 생략이 아니라 지원하지 않는 SDK/artifact 계약으로 실패한다. model 생성 또는
launch 뒤 실패하면 기존 rollback 경로가 dispose와 device session release를 수행한다.

`infer()`/`infer_async()` 직전에는 contiguous 변환 후 다음을 검사한다.

- 단일 vision input
- dtype exact match
- batch axis 존재 및 `1 <= N <= max_batch_size`
- batch를 제외한 shape exact match
- layout과 shape 일치

sync와 native async는 같은 `_ordered_inputs()`와 `_normalize_outputs()`를 사용한다.
native async의 N=1, worker 1, activation slot 1 초기 인수 범위는 유지한다.

`get_device_spec()`에는 expected/actual input dtype/shape/layout, expected/actual output
shapes, profile ID, SDK version을 진단 정보로 기록한다.

입력 cast, normalize, letterbox, YOLO decode, NMS는 runtime에 넣지 않는다.

## 13. 전체 데이터 흐름

```text
CLI model/target/artifact/profile
  -> Mobilint vision profile resolver
  -> artifact-local Model_Spec input/output contract
  -> task-specific Mobilint loader
     -> ResNet exact resize/crop OR YOLO exact letterbox/context
  -> batch collate: uint8 NHWC N=1
  -> MobilintRuntime metadata/array validation
  -> qbruntime infer() OR infer_async()
  -> named raw outputs
  -> task decoder
     -> ResNet existing path
     -> YOLOv5 raw-head anchor/grid decode
     -> Model Zoo-compatible combined-score multi-label candidates
     -> class-aware NumPy NMS
  -> evaluator with preprocessing context
  -> result/monitor persistence
```

monitor sampling과 async queue ownership은 변경하지 않는다. decoder는 runtime 호출이 끝난
뒤 실행되므로 NPU-only latency와 end-to-end latency 경계를 기존 방식대로 보존한다.

## 14. 오류 처리

SDK의 포괄적인 `Model_DtypeMismatched`보다 앞에서 다음 오류를 구체적으로 보고한다.

- 등록되지 않은 model/task/artifact 조합
- explicit profile과 model/task 불일치
- profile과 preprocess mode/layout 충돌
- MXQ input dtype/shape mismatch
- MXQ output count/shape mismatch
- runtime array dtype/shape/batch mismatch
- YOLO head 개수, spatial size, channel, layout, batch mismatch
- profile cache와 generic cache 혼용 시도
- 사용자가 runtime option으로 artifact contract를 override하려는 시도

오류에는 profile ID, expected 값, actual 값, artifact basename을 포함한다. raw tensor 값이나
vendor exception의 민감한 내용을 출력하지 않는다.

## 15. 테스트 전략

production 변경 전에 다음 RED tests를 추가한다.

### 15.1 Profile/resolver

- 두 official basename의 auto resolution
- rename artifact의 explicit profile resolution
- task/model/layout/mode conflict
- unknown MXQ fail-fast
- ARIES/REGULUS가 같은 profile을 선택
- frozen Model_Spec 원본 불변성

### 15.2 ResNet50

- 가로/세로/홀수 crop offset 이미지의 Model Zoo pixel parity
- `(224,224,3)` `uint8` contiguous loader output
- generic normalized cache와 분리

### 15.3 YOLOv5m input

- 가로/세로/정사각형 이미지의 OpenCV letterbox pixel parity
- resize round 및 `round(d±0.1)` padding parity
- pad color 114, RGB, NHWC, uint8, contiguous
- 실제 예제의 `ratio_pad=((1.28,1.28),(0,80))`
- framework flat context와 evaluator ground-truth transform
- Mobilint 전용 cache signature

### 15.4 YOLOv5m output

- 세 head 순서 permutation에 무관한 stride/anchor 매칭
- 3D unbatched와 4D batched head 정규화
- synthetic zero/logit fixtures로 xy/wh/objectness/class 수식 검증
- `(B,25200,85)` concatenate shape
- combined confidence threshold가 objectness-only threshold와 구분됨을 검증
- 한 anchor의 복수 class 후보와 class-aware NMS 검증
- `max_nms_candidates=30000`, `max_detections=300` 경계와 canonical output
- Model Zoo threshold defaults와 explicit override
- malformed head count/shape/channel/layout/batch fail-fast

### 15.5 Runtime/CLI/async

- fake qbruntime input/output metadata match/mismatch와 rollback
- float32/NCHW/batch2 rejection before SDK infer
- sync와 native async가 같은 uint8 input과 세 raw outputs를 보존
- factory/CLI가 같은 selected profile object를 모든 component에 전달
- CLI contract option override rejection

### 15.6 Regression

- generic image classification/detection
- Hailo image/detection
- DeepX vision
- Mobilint NLP/LLM
- Mobilint monitor/native async
- cache와 inference pipeline context 전달

## 16. ARIES2 hardware acceptance

이 절의 acceptance는 SDK-free 회귀 검증과 별개이며 실제 ARIES2 실행 로그를 받아야
완료된다. 아래 명령은 저장소 루트에서 실행한다.

### 16.1 ResNet50

- warmup 2, 10 steps가 dtype 오류 없이 완료
- Model Zoo와 첫 이미지 top-1/top-5 일치
- e2e, monitor, native async 종료 정상

```bash
python framework/src/main.py \
  --model resnet50 \
  --target mobilint-aries \
  --artifact framework/models/mobilint/resnet50/aries/resnet50_IMAGENET1K_V2.mxq \
  --dataset framework/datasets/imagenet_1k \
  --image-preprocess-profile auto \
  --layout NHWC \
  --no-compile \
  --warmup 2 \
  --max-steps 10 \
  --monitor
```

### 16.2 YOLOv5m

- artifact metadata가 확인된 입력/출력 계약과 일치
- Model Zoo와 같은 이미지에서 letterbox tensor와 ratio/pad 일치
- framework runtime이 세 raw heads를 손실 없이 decoder로 전달
- Model Zoo와 framework의 pre-NMS decoded boxes를 tolerance 내 비교
- 같은 confidence/IoU에서 최종 class/score/box 비교
- COCO val2017을 사용할 경우 Model Zoo 기본값 `0.001/0.65`로 mAP 비교
- sync와 native async 결과 동일
- monitor power/utilization/memory/temperature sample과 energy 기록

```bash
# sync
python framework/src/main.py \
  --model yolov5m \
  --target mobilint-aries \
  --artifact framework/models/mobilint/yolov5m/aries/yolov5m.mxq \
  --dataset /path/to/coco \
  --image-preprocess-profile auto \
  --layout NHWC \
  --no-compile \
  --runtime-option core_mode=global8 \
  --runtime-option conf_threshold=0.001 \
  --runtime-option iou_threshold=0.65 \
  --warmup 2 \
  --max-steps 10

# monitor
python framework/src/main.py \
  --model yolov5m \
  --target mobilint-aries \
  --artifact framework/models/mobilint/yolov5m/aries/yolov5m.mxq \
  --dataset /path/to/coco \
  --image-preprocess-profile auto \
  --layout NHWC \
  --no-compile \
  --runtime-option core_mode=global8 \
  --runtime-option conf_threshold=0.001 \
  --runtime-option iou_threshold=0.65 \
  --warmup 2 \
  --max-steps 10 \
  --monitor

# native async
python framework/src/main.py \
  --model yolov5m \
  --target mobilint-aries \
  --artifact framework/models/mobilint/yolov5m/aries/yolov5m.mxq \
  --dataset /path/to/coco \
  --image-preprocess-profile auto \
  --layout NHWC \
  --no-compile \
  --runtime-option core_mode=global8 \
  --runtime-option conf_threshold=0.001 \
  --runtime-option iou_threshold=0.65 \
  --monitor \
  --inference-mode async_queue \
  --scenario offline \
  --queue-capacity 16 \
  --worker-count 1 \
  --max-samples 10
```

`/path/to/coco`는 `images/val2017`과 `labels/val2017`을 포함하는 dataset root다.

hardware log에는 SDK/driver/runtime version, artifact hash, effective profile, thresholds를 함께
기록한다. ResNet Model Zoo/framework top-1/top-5 또는 YOLO letterbox, pre-NMS와 최종
detection/mAP 비교, monitor power/utilization/memory/temperature/energy와 sample coverage,
native-async submitted/completed/failed/outstanding shutdown count도 함께 수집한다. 실제
hardware success 상태는 이 로그가 제공될 때까지 pending이다.

## 17. 예상 파일 범위

production create:

- `framework/src/dataloader/mobilint_vision_profiles.py`
- `framework/src/dataloader/mobilint_image_classification_loader.py`
- `framework/src/dataloader/mobilint_object_detection_loader.py`
- `framework/src/preprocessor/mobilint_vision.py`
- `framework/src/decoders/mobilint_yolov5.py`

production modify:

- `framework/src/dataloader/__init__.py`
- `framework/src/preprocessor/__init__.py`
- `framework/src/decoders/__init__.py`
- `framework/src/main.py`
- `framework/src/decoders/object_detection.py`
- `framework/src/runtimes/mobilint_rt.py`
- `framework/src/runtimes/README.md`

tests create:

- `framework/tests/test_mobilint_vision_profiles.py`
- `framework/tests/test_mobilint_image_classification_loader.py`
- `framework/tests/test_mobilint_object_detection_loader.py`
- `framework/tests/test_mobilint_yolov5_decoder.py`

tests modify:

- `framework/tests/test_mobilint_runtime.py`
- `framework/tests/test_mobilint_native_backend.py`
- `framework/tests/test_main_paths.py`
- `framework/tests/test_object_detection_decoders.py`
- `framework/tests/test_object_detection_loader.py`
- `framework/tests/test_object_detection_loader_async.py`
- `framework/tests/test_object_detection_evaluator.py`
- `framework/tests/test_inference_pipeline.py`
- `framework/tests/test_hailo_image_loader.py`
- `framework/tests/test_deepx_dxnn_metadata.py`

## 18. 비범위

- Mobilint compiler adapter와 MXQ 자동 컴파일
- arbitrary custom preprocessing JSON/DSL
- YOLOv5mu, P6, segmentation, pose, OBB profile
- YOLOv8 이상 anchorless/DFL decoder
- Model Zoo `MBLT_Engine`을 production runtime으로 사용
- batch 1을 넘는 실제 장치 성능 검증
- monitor metric/result schema 변경
- async queue ownership 또는 scheduling 변경

## 19. 승인 기준

- ResNet50 V2와 YOLOv5m DEFAULT가 각각 official artifact로 정확히 auto resolve된다.
- ResNet 입력이 Model Zoo와 같은 `(1,224,224,3)` uint8이다.
- YOLOv5m 입력이 Model Zoo와 같은 `(1,640,640,3)` uint8이고 letterbox pixels/context가
  일치한다.
- YOLOv5m 세 raw heads가 `(B,25200,85)`로 정확히 decode되고 Model Zoo와 같은
  combined-confidence multi-label/class-aware NMS 및 기존 evaluator와 연결된다.
- runtime이 cast, normalize, resize, decode, NMS를 하지 않는다.
- unknown artifact와 malformed outputs가 조용히 fallback하지 않는다.
- ARIES/REGULUS가 같은 vision profile/loader/decoder를 공유한다.
- Hailo, DeepX, generic vision, Mobilint NLP/LLM 동작이 바뀌지 않는다.
- SDK-free tests가 통과하고 ARIES2의 ResNet/YOLO sync, monitor, native-async 인수 로그가
  확보된다.
