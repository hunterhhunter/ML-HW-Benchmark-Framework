# Mobilint MXQ 이미지 전처리 설계

**상태:** 승인됨

**승인일:** 2026-07-22

**대상 범위:** Mobilint raw MXQ 이미지 분류 입력, 전처리 factory, 런타임 입력 검증

## 1. 문제와 확인된 원인

공식 Mobilint Model Zoo의 ARIES용
`resnet50_IMAGENET1K_V2.mxq`는 정상 실행됐지만, 같은 artifact를 프레임워크의
`mobilint-aries` target으로 실행하면 첫 warmup에서 다음 오류가 발생했다.

```text
Invalid input data type - Input: Float32, Supported: Uint8
QbRuntimeError: Model_DtypeMismatched
```

두 경로의 입력을 역추적한 결과 차이는 다음과 같다.

| 항목 | Mobilint Model Zoo | 현재 프레임워크 일반 분류 로더 |
|---|---|---|
| resize | 짧은 변 232, bilinear | 짧은 변 256, bilinear |
| crop | 중앙 224×224 | 중앙 224×224 |
| 색상·layout | RGB, HWC | RGB, CLI에 따라 CHW/HWC |
| 값·dtype | 0..255 `uint8` | ImageNet 정규화 `float32` |
| batch 입력 | 검증 경로에서 BHWC | NHWC 선택 시 BHWC |

Model Zoo의 `MBLT_Engine`은 MXQ를 로드한 뒤
`get_model_input_data_type()`이 `DataType.Uint8`이면 YAML 전처리 목록에서
`Normalize`를 제거한다. V2 YAML은 짧은 변 232를 지정한다. 따라서 이번 실패는
Mobilint runtime의 추론 호출이나 장치 선택 문제가 아니라, artifact의 컴파일 입력
계약과 DataLoader가 만든 텐서 계약이 일치하지 않아 발생했다.

이 현상은 단순히 사용자가 직접 컴파일하지 않았기 때문에 생긴 것이 아니다.
컴파일 시 resize, normalize 같은 전처리를 그래프에 융합했는지에 따라 같은 모델도
runtime 입력이 `uint8` 또는 정규화된 `float32`가 될 수 있다. 사전 컴파일 MXQ를 사용할
때는 그 artifact를 만든 recipe와 입력 계약을 함께 선택해야 한다.

## 2. 결정

Mobilint 이미지 전처리는 `MobilintRuntime`이 아니라 전처리 계층이 담당한다. 기존
Hailo와 DeepX의 vendor-specific image loader 패턴을 따라
`MobilintImageClassificationLoader`와 Mobilint 입력 프로파일 resolver를 추가한다.

`MobilintRuntime`은 다음 책임만 갖는다.

- qb Runtime 모델 생성, launch, sync/native-async 추론과 dispose
- MXQ가 보고한 실제 입력 shape/dtype 확인
- 전처리 계층이 선언한 입력 계약과 실제 MXQ 계약 검증
- `infer()` 또는 `infer_async()` 호출 직전 실제 배열의 dtype/layout/shape 검증

runtime에서 정규화된 `float32`를 사후 `uint8`로 cast하지 않는다. 그 변환은 이미
정규화로 소실된 원본 픽셀을 복구할 수 없고, SDK 오류만 숨긴 채 정확도를 훼손한다.

ARIES와 REGULUS는 같은 MXQ 전처리 구현을 공유한다. 전처리는 장치 family가 아니라
artifact의 입력 계약으로 결정한다. 기존 target의 `expected_family`, device selector,
monitor와 native async 계약은 변경하지 않는다.

## 3. Model Zoo `model.preprocess()` 재사용 여부

`model.preprocess()`는 `qbruntime.Model` API가 아니라 Model Zoo의 `MBLT_Engine` API다.
`MBLT_Engine` 생성은 전처리기만 만드는 것이 아니라 MXQ 모델을 생성하고 NPU에
launch한다. 현재 raw `MobilintRuntime`과 함께 생성하면 같은 장치에 모델이 두 번
올라가고 리소스 lifecycle 소유권도 둘로 갈라진다.

따라서 production에서 `MBLT_Engine.preprocess()`를 직접 호출하지 않는다. Model Zoo의
공식 YAML과 구현은 입력 규약의 기준으로 사용하고, 프레임워크 전처리 전략이 동일한
픽셀을 만드는지 parity test로 검증한다. 이렇게 하면 다음을 모두 보존한다.

- raw qbruntime adapter의 선택적 SDK 의존성
- SDK `infer_async()` 기반 native async 경로
- 프레임워크가 소유하는 단일 model lifecycle
- Model Zoo 내부 API 변경과의 격리

향후 Model Zoo 전체 pipeline을 직접 실행해야 한다면 raw target에 섞지 않고 별도
high-level runtime target으로 설계한다.

## 4. 구성요소

### 4.1 MobilintImageInputConfig

Mobilint 입력 프로파일 resolver는 불변 configuration을 반환한다.

```text
MobilintImageInputConfig
  profile_id
  model_name
  preprocess_mode
  resize_short_side
  crop_hw
  color_order
  input_layout
  input_dtype
  unbatched_input_shape
```

최초 등록 프로파일은 다음 하나다.

```text
profile_id: mobilint-resnet50-imagenet1k-v2
model_name: resnet50
preprocess_mode: raw
resize_short_side: 232
crop_hw: [224, 224]
color_order: RGB
input_layout: NHWC
input_dtype: uint8
unbatched_input_shape: [224, 224, 3]
```

프로파일 registry는 장치 family를 key로 사용하지 않는다. 동일 입력 계약의 MXQ는
ARIES와 REGULUS에서 같은 프로파일을 선택한다.

### 4.2 프로파일 선택

새 CLI 옵션 `--image-preprocess-profile`을 추가하고 기본값은 `auto`로 한다.

- `auto`: model 이름과 공식 artifact basename이 등록 프로파일과 정확히 일치할 때만
  선택한다. 최초 자동 인식 대상은
  `resnet50_IMAGENET1K_V2.mxq`와 `resnet50`의 조합이다.
- 명시적 profile ID: artifact 파일을 rename했더라도 등록 프로파일을 선택한다.
- 알 수 없는 Mobilint MXQ: 일반 ImageNet `float32` 전처리로 fallback하지 않고 실행 전
  실패한다. 오류에는 사용 가능한 profile ID와 명시 방법을 포함한다.
- 알려진 `uint8` 프로파일에 `--image-preprocess-mode normalized`를 지정하면 충돌로
  실패한다. `auto` 또는 `raw`만 허용한다.
- non-`auto` profile은 Mobilint raw 이미지 분류 target에서만 허용한다. 다른 backend나
  task에 지정하면 무시하지 않고 CLI validation error로 처리한다.

CLI `--layout`이 기본값인 경우 resolver가 profile layout인 `NHWC`로 확정한다. 사용자가
`NCHW`를 명시했고 선택 profile이 `NHWC`를 요구하면 조용히 덮어쓰지 않고 충돌로
실패한다. 이를 위해 `main.py`가 이미 계산하는 `layout_was_default` 정보를 resolver에
전달한다.

현재 범위에서는 임의의 custom MXQ 전처리 JSON이나 compiler metadata 생성을
추가하지 않는다. 추후 compiler 연동 시 compiler가 생성한 artifact manifest를 같은
resolver 입력으로 연결한다.

### 4.3 MobilintResNet50V2Preprocess

Mobilint 전용 strategy는 Model Zoo V2와 같은 순서로 처리한다.

1. PIL로 열고 RGB로 변환한다.
2. 짧은 변을 232로 맞추고 aspect ratio를 유지한다.
3. PIL bilinear interpolation을 사용한다.
4. 중앙 224×224를 Model Zoo와 같은 반올림 규칙으로 crop한다.
5. normalize하지 않고 `numpy.uint8`을 유지한다.
6. 프레임워크 cache convention에 맞춰 CHW로 반환한다.
7. `ImageClassificationLoader._apply_layout()`이 runtime 직전에 NHWC로 변환한다.

strategy의 cache signature에는 profile ID, resize short side, crop 크기, dtype과
전처리 버전을 넣는다. 기존 정규화 `float32` cache와 V1/다른 variant cache를 절대
공유하지 않는다.

### 4.4 MobilintImageClassificationLoader

새 loader는 `ImageClassificationLoader`를 상속하고 다음만 추가한다.

- 확정된 `MobilintImageInputConfig`를 받는다.
- 해당 Mobilint preprocess strategy를 주입한다.
- profile이 요구하는 NHWC layout을 사용한다.
- metadata에 profile ID, dtype, layout, unbatched shape를 기록한다.
- runtime 검증용 expected input contract를 `runtime_options` metadata로 전달한다.

dataset 탐색, label mapping, batching, cache IO와 cursor는 기존 부모 구현을 그대로
사용한다. Mobilint 때문에 generic loader의 기본 전처리를 변경하지 않는다.

### 4.5 DataLoader factory와 CLI 조립

`main.py`는 Mobilint raw backend의 이미지 분류 task일 때 artifact와 model 이름으로
입력 config를 먼저 확정한다. 실제 artifact 계약을 반영하도록 artifact-local
`Model_Spec`의 입력 shape/dtype을 NHWC/`uint8`로 설정한 뒤 `CompiledModel`과 loader에
같은 config를 전달한다.

`create_dataloader()`는 `backend == "mobilint"`이고 task가 이미지 분류일 때
`MobilintImageClassificationLoader`를 선택한다. 다른 task와 다른 backend의 factory
분기는 바꾸지 않는다.

## 5. 데이터 흐름

```text
CLI model/target/artifact/profile
  -> Mobilint image profile resolver
  -> artifact-local Model_Spec input contract
  -> MobilintImageClassificationLoader
  -> exact resize/crop, cached CHW uint8
  -> layout 적용, batch collate: (1, 224, 224, 3) uint8
  -> MobilintRuntime input contract validation
  -> qbruntime.Model.infer() 또는 infer_async()
  -> 기존 decoder/evaluator/result 경로
```

sync와 async는 같은 DataLoader tensor를 사용한다. native async에서 유지하는 `N=1`
제한과 `activation_slots=1`, `worker_count=1` 초기 인수 조건은 그대로다. 전처리 변경은
framework bounded queue와 SDK Future 연결을 변경하지 않는다.

## 6. 오류 처리와 진단

다음 오류는 SDK의 포괄적인 `Model_DtypeMismatched`보다 앞에서 구체적으로 보고한다.

- 등록되지 않은 artifact/profile 조합
- 선택 profile과 `--image-preprocess-mode` 충돌
- MXQ가 보고한 dtype과 profile dtype 불일치
- MXQ가 보고한 unbatched input shape와 profile shape 불일치
- runtime 배열이 contiguous가 아니거나 dtype/layout/shape가 다른 경우
- batch size가 초기 지원 범위인 1을 벗어난 경우

runtime은 qb Runtime의 `get_model_input_shape()`와
`get_model_input_data_type()` 결과를 정규화해 profile과 비교한다. getter가 SDK 버전에서
없거나 예상 형식이 아니면 해당 metadata 검증을 건너뛰지 않고 지원하지 않는 SDK
계약으로 명확히 실패한다. SDK v1.3.2가 첫 실제 검증 기준이다.

모델 생성 또는 launch 뒤 계약 검증이 실패하면 기존 rollback 경로가 model dispose와
device session release를 수행한다. monitor가 이미 생성된 경우에도 기존 CLI cleanup
계약을 유지한다.

## 7. 테스트 설계

production 변경 전 다음 RED test를 추가한다.

1. Mobilint ResNet50 V2 factory가 현재 generic normalized loader를 선택해
   `float32`를 만드는 회귀 재현
2. `auto`가 공식 V2 artifact basename을 올바른 profile로 해석하는 계약
3. 명시적 profile이 rename된 artifact에도 적용되는 계약
4. 미등록 artifact와 normalized override를 fail-fast하는 계약
5. 전처리 출력이 `(224, 224, 3)`, `uint8`, RGB, contiguous가 되는 계약
6. 여러 종횡비 이미지에서 Model Zoo 기준 구현과 픽셀 단위로 같은 결과를 만드는 계약
7. profile별 cache signature가 기존 normalized cache와 충돌하지 않는 계약
8. DataLoader factory가 Mobilint에만 전용 loader를 선택하는 계약
9. fake qbruntime metadata와 실제 배열 dtype/shape/layout 검증 계약
10. e2e와 native async가 같은 `uint8` 입력을 SDK에 전달하는 계약

회귀 범위는 generic image loader, Hailo, DeepX, Mobilint runtime/native async, CLI path와
plugin registry test를 포함한다. SDK-free test에서는 fake qbruntime을 사용하며 Mobilint
패키지를 기본 requirements에 추가하지 않는다.

실제 ARIES2 인수 테스트는 SDK/driver/runtime `1.3.2` 계열 환경에서 다음 순서로 한다.

1. Model Zoo `predict`의 동일 이미지 top-5를 기준 결과로 보존한다.
2. 프레임워크 e2e, batch 1, warmup 2, 10 step이 dtype 오류 없이 완료되는지 확인한다.
3. 첫 이미지의 top-1/top-5가 Model Zoo 기준과 일치하는지 확인한다.
4. `--monitor` 실행에서 utilization, memory, temperature, power sample과 energy가
   기록되는지 확인한다.
5. `async_queue`, worker 1, activation slot 1에서 SDK `infer_async()`가 호출되고
   outstanding 0으로 종료되는지 확인한다.

현재 개발 host에는 Mobilint SDK/NPU와 pytest가 설치되어 있지 않으므로 실제 hardware
성공을 로컬에서 주장하지 않는다. hardware 출력은 사용자가 실행한 로그로 인수한다.

## 8. 예상 변경 범위

production:

- `framework/src/main.py`
- `framework/src/dataloader/__init__.py`
- `framework/src/dataloader/mobilint_image_classification_loader.py`
- `framework/src/runtimes/mobilint_rt.py`

Mobilint 전용 profile resolver와 strategy는
`mobilint_image_classification_loader.py`에 함께 둔다. generic strategy module에는
추가하지 않는다.

tests:

- `framework/tests/test_mobilint_image_classification_loader.py`
- `framework/tests/test_main_paths.py`
- `framework/tests/test_mobilint_runtime.py`
- 필요 시 cache와 DataLoader factory 기존 test

documentation:

- `framework/src/runtimes/README.md`
- 필요 시 DataLoader README와 CLI help example

상세 구현 계획에서 현재 import 경계와 기존 test fixture를 다시 확인해 정확한 파일
목록을 확정한다.

## 9. 비범위

- Mobilint compiler adapter 및 MXQ 자동 컴파일
- 임의 compiler recipe나 custom preprocessing JSON schema
- Model Zoo `MBLT_Engine`을 raw runtime으로 사용
- 다른 Mobilint vision model의 profile 등록
- batch 1을 넘는 실제 장치 성능 검증
- object detection, segmentation, pose 전처리 추가
- REGULUS 전력 계측 추가
- async queue, monitor metric 또는 result schema 변경

## 10. 승인 기준

이번 변경은 다음 조건을 모두 만족해야 완료다.

- Mobilint ResNet50 V2 MXQ에 Model Zoo와 같은 `uint8` 입력이 전달된다.
- runtime에서 임의 cast 또는 normalize를 수행하지 않는다.
- ARIES/REGULUS가 같은 artifact-specific preprocessor를 공유한다.
- 알 수 없는 MXQ는 generic `float32`로 조용히 실행하지 않는다.
- Hailo, DeepX와 generic image classification 동작이 바뀌지 않는다.
- SDK-free test가 통과하고 실제 ARIES2 e2e, monitor, native async 인수 로그가 확보된다.
