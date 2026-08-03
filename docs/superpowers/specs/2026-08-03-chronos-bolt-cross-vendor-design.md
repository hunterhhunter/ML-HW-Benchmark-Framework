# Chronos-Bolt 교차 벤더 컴파일 설계

## 목적

`amazon/chronos-bolt-tiny`의 문제가 되는 동적 전처리를 CPU 경계로 분리하고,
동일한 Transformer core를 Rebellions ATOM, Furiosa RNGD, Mobilint ARIES용으로
컴파일한다. Tiny가 성공하면 `mini`, `small`, `base`까지 같은 고정 계약으로
확장하여 벤더별 최대 성공 크기를 확인한다. 데이터 검증에는 ETTh1을 사용한다.

전처리 연산을 의미적으로 삭제하지 않는다. 원본 Chronos-Bolt의 정규화, 결측치
처리, 패치 구성, 임베딩 및 역정규화를 CPU adapter로 옮기고, 원본 full model과
분리 모델의 출력 동등성을 먼저 증명한다. 이 동등성 검사를 통과하지 않은 artifact는
컴파일에 성공해도 유효한 Chronos-Bolt 결과로 인정하지 않는다.

## 기준 모델과 고정 계약

첫 모델은 `amazon/chronos-bolt-tiny`이며 다음 계약을 변경하지 않는다.

- 외부 입력: FP32 `[1, 512]`
- 외부 출력: FP32 `[1, 9, 64]`
- quantile 순서: `[0.1, 0.2, ..., 0.9]`
- batch: 1
- context length: 512
- prediction length: 64
- input patch size/stride: 체크포인트 설정값 16/16
- patch count: 32
- training/eval 상태: `eval()` 및 inference-only
- CPU/eager fallback: 허용하지 않음

96 시점 예측은 이번 native compile 계약에 포함하지 않는다. Chronos-Bolt의 64 초과
예측은 첫 출력의 9개 quantile을 batch 방향으로 확장하고 두 번째 모델 호출 뒤 다시
quantile reduction을 수행하므로 `[1,512] -> [1,9,64]` 단일 artifact와 다른 계약이다.
64 시점 artifact가 검증된 뒤 host orchestration으로 별도 추가할 수 있다.

## 선택한 접근 방식

### 권장 경계: CPU adapter + NPU Transformer core

원본 구현의 `unfold`, `nanmean`, `nan_to_num`, NaN mask 및 동적 tensor 생성은
세 벤더의 공통 NPU graph에서 제외한다. 입력 patch embedding과 decoder start-token
embedding도 CPU adapter에서 계산한다. 이 경계가 세 compiler에 가장 단순하고
동일한 graph를 제공한다.

NPU core는 다음 학습된 모듈을 그대로 공유한다.

1. T5 encoder
2. T5 decoder
3. Chronos-Bolt output patch embedding/quantile head
4. `[1, 1, 9 * 64] -> [1, 9, 64]` 고정 reshape

입력 patch embedding은 학습된 가중치를 가진 모듈이지만 계산량이 작고, compiler
호환성을 위해 CPU 경계에 둔다. 이를 제거하거나 임의 값으로 대체하지 않는다. 이후
세 벤더에서 core가 모두 성공한 경우에만 patch embedding을 NPU graph 안으로 옮기는
확장 실험을 수행한다.

### 검토한 대안

1. **패치와 정규화만 CPU로 분리**: patch embedding까지 NPU에서 실행하므로 NPU
   연산 비중은 커지지만 residual MLP와 mask 경계가 추가되어 교차 벤더 성공 가능성이
   낮다.
2. **full model 정적 재작성**: `unfold`를 slice/stack으로 바꾸고 NaN 연산을 정적
   mask 연산으로 바꾼다. 원형에는 가장 가깝지만 각 compiler에 맞춘 graph 변형이
   생길 가능성이 커서 공정한 동일-core 비교가 어렵다.

두 대안은 권장 경계의 세 벤더 성공 이후 진단용 확장으로만 사용한다.

## 구성 요소와 책임

### `ChronosBoltHostAdapter`

CPU에서 다음을 수행한다.

1. 입력을 오른쪽 기준으로 512에 crop하거나 왼쪽을 NaN으로 pad한다.
2. explicit mask가 없으면 `~isnan(context)`로 observed mask를 만든다.
3. 원본 `InstanceNorm`과 동일하게 FP32 mean, variance, scale을 계산한다.
4. 16개씩 32개 patch를 정적 slice/reshape로 만든다.
5. patch mask를 0/1 FP32로 만들고 결측 입력을 0으로 치환한다.
6. context patch와 patch mask를 결합해 `[1, 32, 32]`을 만든다.
7. 원본 `input_patch_embedding`으로 `[1, 32, d_model]`을 계산한다.
8. 원본 `shared` embedding으로 decoder start embedding `[1, 1, d_model]`을
   계산한다.
9. `loc`과 `scale`을 보관하고 NPU 출력에 역정규화를 적용한다.

adapter가 NPU core로 넘기는 고정 ABI는 다음과 같다.

| 이름 | shape | dtype | 의미 |
|---|---:|---|---|
| `input_embeds` | `[1, 32, d_model]` | FP32 | 학습된 patch embedding 결과 |
| `attention_mask` | `[1, 32]` | FP32 | patch별 observed 여부 0/1 |
| `decoder_input_embeds` | `[1, 1, d_model]` | FP32 | decoder start token embedding |

### `ChronosBoltTransformerCore`

core는 Python container나 Hugging Face `ModelOutput`을 ABI로 노출하지 않는다.
encoder와 decoder는 `return_dict=False`로 호출하고 tensor 하나만 반환한다.

```text
input_embeds + attention_mask
        -> T5 encoder
encoder hidden + decoder_input_embeds + attention_mask
        -> T5 decoder
        -> output_patch_embedding
        -> fixed reshape [1, 9, 64]
```

core 출력은 정규화 공간의 FP32 `[1,9,64]`이다. host adapter가
`prediction * scale + loc`을 FP32로 수행해 외부 출력을 만든다.

### 모델 loader와 manifest

공통 loader는 Hugging Face revision을 고정해 로컬 snapshot으로 받은 뒤 다음을
manifest에 기록한다.

- repository ID와 resolved commit SHA
- model family size (`tiny`, `mini`, `small`, `base`)
- Chronos/Transformers/PyTorch 버전
- `d_model`, encoder/decoder layer 수, parameter 수
- weight dtype와 core ABI
- 원본 weight file SHA-256

모델 가중치와 vendor artifact는 Git에 커밋하지 않는다.

## ETTh1 데이터 흐름

ETTh1의 표준 경계 `(8640, 11520)`을 사용한다. train은 `[0,8640)`, validation은
`[8640,11520)`, test target은 `[11520,end)`이다. test의 첫 context는 target 이전
512개 관측치를 포함할 수 있다.

Chronos-Bolt는 단변량 모델이므로 ETTh1의 7개 채널을 결합된 다변량 입력으로
해석하지 않는다. 각 채널을 같은 `[1,512] -> [1,9,64]` artifact에 독립적으로
입력하고 결과를 합산한다.

- smoke: test 시작점의 7개 채널 1 window씩
- parity: seed로 고정한 test origin 32개 × 7채널
- dataset evaluation: 가능한 모든 64-step test origin × 7채널
- point forecast: 0.5 quantile
- point metrics: MAE, MSE, RMSE
- probabilistic metric: 9개 quantile의 weighted quantile loss

데이터의 원본 CSV SHA-256, split, channel 순서, window origin, stride를 결과
manifest에 기록한다. metric은 역정규화된 원래 ETTh1 scale에서 계산한다.

## 동등성 게이트

compiler를 호출하기 전에 원본 `ChronosBoltModelForForecasting.forward(context)`와
분리된 `host_preprocess -> core -> host_postprocess`를 CPU에서 비교한다.

1. finite synthetic input
2. NaN이 포함된 synthetic input
3. ETTh1 smoke 7개 window
4. ETTh1 parity 32 origin × 7개 채널

각 case에서 shape, dtype, finite 여부와 모든 quantile 값을 비교한다. FP32 CPU
기준은 `rtol=1e-5`, `atol=1e-6`을 시작 기준으로 사용한다. 통과하지 않으면 tolerance를
임의로 높이지 않고 첫 divergence 모듈을 기록한 뒤 adapter를 수정한다.

vendor compiler가 내부 정밀도를 낮추는 경우에는 vendor 출력과 CPU split-core
출력의 max/mean absolute error를 별도로 기록한다. 정확도 허용 기준은 compiler가
실제로 선택한 dtype을 확인한 뒤 명시하며, fallback이나 잘못된 shape를 tolerance로
가리지 않는다.

## 벤더별 환경과 컴파일

의존성 충돌을 막기 위해 다음 가상환경을 공유하지 않는다.

- `.venv-chronos-reference`: 원본 Chronos와 CPU parity
- `.venv-chronos-rbln`: Rebellions compiler
- `.venv-chronos-furiosa`: Furiosa Torch/compiler
- `.venv-chronos-mobilint`: Mobilint `qbcompiler`

설치 manifest에는 package version, wheel filename/URL, wheel SHA-256, Python version,
설치 시각을 남긴다. portal token, cookie, license text, credential 경로는 로그나 Git에
기록하지 않는다.

### Rebellions ATOM

- portal 인증 index에서 공식 `rebel-compiler`를 설치한다.
- 고정 example input 3개로 Torch core를 compile한다.
- 생성 artifact와 compiler metadata를 검사한다.
- ATOM 장치가 있는 서버에서는 load와 첫 추론까지 실행한다.

### Furiosa RNGD

- 공식 Furiosa repository/package에서 Furiosa Torch stack을 독립 환경에 설치한다.
- `fullgraph=True`, `dynamic=False`, `eager_fallback=False`를 고정한다.
- 공개 offline compile API가 있으면 exported core를 직접 컴파일한다.
- SDK가 compile을 첫 `furiosa:0` 호출에만 수행한다면, 장치 없는 현재 호스트에서는
  환경/export 검증까지만 분리해 기록하고 RNGD 서버에서 compile+load+first inference를
  완료해야 성공으로 판정한다. CPU fallback은 compile 성공으로 기록하지 않는다.

### Mobilint ARIES

- 제공된 `qbcompiler-1.2.0` wheel과 별도 Python 3.10 환경을 사용한다.
- target은 `aries-rb`, 입력 shape/dtype은 공통 core ABI로 고정한다.
- compile artifact, compiler report와 MXQ/입출력 metadata를 검사한다.
- ARIES 서버에서는 runtime load와 첫 추론을 추가 검증한다.

## 크기 확장 정책

각 벤더에서 Tiny가 유효하게 성공한 뒤 다음 순서로 같은 절차를 반복한다.

1. `amazon/chronos-bolt-mini`
2. `amazon/chronos-bolt-small`
3. `amazon/chronos-bolt-base`

큰 모델이 실패해도 이후 크기를 자동으로 성공 처리하지 않는다. compiler 오류,
timeout, host OOM, artifact 검사 실패를 각각 구분해 기록한다. 자원이 허용하면 남은
크기도 시도하되, 벤더별 "최대 성공 크기"는 실제 artifact 검사와 가능한 장치 첫
추론까지 통과한 가장 큰 모델로 정의한다.

## 결과와 실패 기록

각 compile은 별도 subprocess와 timeout 안에서 실행한다. 결과 디렉터리는 Git에서
제외하고 다음 파일을 생성한다.

- 공통 model/data/environment manifest JSON
- 벤더·모델별 compile stdout/stderr log
- 벤더·모델별 result JSON
- 비어 있지 않은 vendor artifact
- CPU parity report
- 장치가 있을 때 first-inference parity report
- 전체 결과 summary CSV/Markdown

result JSON의 상태는 `passed`, `failed`, `prerequisite_missing`, `not_runnable`을
구분한다. 장치가 없는 호스트에서 export만 성공한 결과를 `passed`로 승격하지 않는다.
오류 메시지는 안정적인 signature와 단계만 JSON에 넣고 전체 traceback은 로컬 log에
보관한다.

## 프레임워크 통합

컴파일 도구와 parity가 먼저 통과한 뒤 다음을 추가한다.

- Chronos-Bolt model profile과 고정 tensor contract
- ETTh1 7채널 단변량 window adapter
- CPU host pre/postprocessor
- 벤더별 artifact loader 연결
- 기존 `TimeSeriesForecastingEvaluator`의 MAE/MSE에 RMSE/WQL 확장
- 동기 E2E smoke와 async single-stream 검증 명령

runtime은 model-specific 전처리 구현을 소유하지 않는다. 공통 host adapter가 같은
tensor ABI를 만들고 vendor runtime은 compile artifact의 입력 순서와 dtype만 검증한다.

## 테스트와 완료 조건

### 단위 테스트

- 고정 patch 생성이 원본 `unfold`와 동일함
- NaN/explicit mask와 `loc/scale`이 원본 구현과 동일함
- host adapter의 shape/dtype/입력 순서
- core가 dictionary가 아닌 tensor를 반환함
- postprocess와 quantile 순서
- ETTh1 split/window/channel 집계
- manifest 및 compiler 결과 상태 분류

### 통합 테스트

- Tiny 원본 full model 대 split model CPU parity
- 세 vendor compiler adapter가 같은 exported core를 받음
- artifact가 존재하고 크기가 0보다 큼
- 잘못된 shape/dtype, fallback, timeout, 장치 없음이 명시적으로 실패 또는
  `not_runnable`로 분류됨
- 기존 time-series 및 vendor runtime 테스트 회귀 없음

### 목표 완료 조건

다음 증거가 모두 있어야 작업을 완료로 보고한다.

1. ETTh1 기반 Tiny CPU parity가 통과한다.
2. Rebellions, Furiosa, Mobilint 환경 설치와 버전 manifest가 존재한다.
3. 세 vendor 각각에 대해 Tiny의 compile 결과와 artifact 또는 명확한 재현 가능한
   compiler prerequisite/실패 증거가 있다.
4. Tiny 성공 vendor는 mini, small, base까지 확장 결과가 있다.
5. 장치가 있는 환경에서만 가능한 검증은 서버용 명령과 결과 schema가 있으며,
   실제 장치 실행 전에는 성공으로 표기하지 않는다.
6. 구현, 재현 명령, 로그 위치, 벤더별 최대 성공 크기와 남은 blocker가 문서화된다.

## 범위 밖

- Chronos-Bolt를 재학습하거나 체크포인트 weight를 변경하는 작업
- 기본 prediction length를 96으로 재구성하는 작업
- CPU/eager fallback을 NPU 성공으로 보고하는 작업
- 벤더 SDK 내부 compiler panic 또는 unsupported op 자체를 수정하는 작업
- portal credential, license 또는 모델 weight를 저장소에 커밋하는 작업
