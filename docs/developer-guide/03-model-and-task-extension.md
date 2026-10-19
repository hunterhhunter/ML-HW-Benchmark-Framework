# 03. 모델·태스크 확장

[목차](README.md) · [이전: 공용 계약](02-component-contracts.md) · [다음: NPU 어댑터 확장](04-npu-adapter-extension.md)

새 모델을 등록하기 전에 **원천 모델, 실제 실행 artifact, 데이터·정답,
입력 전처리, 출력 의미, 평가 지표**를 고정한다. 모델명이 같거나
`Model_Spec`의 shape가 같아도 HEF·DXNN·MXQ 등의 출력 계약은 다를 수 있다.
작업 범위는 아래처럼 결정한다.

| 변경 | 먼저 수정할 곳 | 추가로 확인할 곳 |
|---|---|---|
| 기존 task·입출력 계약과 같은 새 모델 | `core/model_profiles.py`, 모델·데이터 준비 스크립트 | 현재 Loader·Decoder·Evaluator가 그 artifact를 실제로 받는지 |
| 전처리·입력 ABI가 다른 모델 | 전처리기 또는 Loader, `dataloader/__init__.py` | `main.py`의 profile·옵션 전달, cache key |
| Runtime 출력 ABI가 다른 모델 | Decoder, `decoders/__init__.py` | evaluator 기대 형태, `main.py`의 ABI 선택 옵션 |
| 새로운 품질 지표가 필요한 모델 | Evaluator, `evaluators/__init__.py` | 정답 형식, 결과 metric 이름·단위 |
| 새로운 task | `Task` enum, Loader, Evaluator, 선택적 Decoder | factory 3곳, 데이터 경로 resolver, `main.py` 조립·검증 |

## 1. 모델 프로필과 자산

[`SUPPORTED_PROFILES`](../../framework/src/core/model_profiles.py)에 CLI 모델명,
`Task`, 입력 이름·shape·dtype, 출력 기대 shape, 기본 모델·데이터 경로,
필요한 준비 스크립트를 추가한다. `Model_Spec`은 이 profile과 실제 선택한
source 경로로 만들어진다. `__auto__`는 **ONNX를 파싱하는 경로에서만**
실제 I/O 이름으로 바뀐다. native artifact 실행에는 profile의 기본 이름이
남을 수 있으므로 vendor Runtime이 artifact metadata를 보고 입력·출력을
검증해야 한다.

자산 준비 스크립트는 `framework/models/`와 `framework/datasets/`에 둔다.
프레임워크가 자동 준비할 수 있는 경로인지,
벤더가 제공한 precompiled artifact를 사용자가 지정해야 하는 경로인지
구분한다. dataset split, 샘플 순서, 정답 형식과 license·인증 요구 사항도
모델별 실행 명세에 남긴다. 준비 스크립트의 성공이 장치 실행 검증은 아니다.

## 2. 입력 계약과 Loader 연결

기존 Loader가 같은 입력·정답 의미를 이미 제공하면 재사용한다. 새로운
resize/crop/letterbox, 정규화, 채널 순서, tokenization, tensor 이름·dtype,
batch 방식이 필요하면 [`DataLoader`](../../framework/src/dataloader/base.py)의
계약을 만족하는 구현을 추가하거나 전처리 전략을 분리한다.

1. `load_single()`, `load_batch()`, `load_by_index()`가 같은 sample 순서를
   사용하게 한다. `get_metadata()["total_samples"]`와 실제 개수를 대조한다.
2. `input`은 Runtime이 기대하는 배열 또는 입력명→배열 매핑이다. `label`은
   evaluator가 받아야 하는 정답이다. box·keypoint 좌표 복원 정보가 필요하면
   sample별 `preprocess_context`도 제공한다.
3. 캐시가 있다면 모델·artifact의 전처리 계약이 달라질 때 다른 파일을
   쓰도록 한다. 예를 들어 같은 원본 이미지라도 resize 방법과 색상 순서가
   바뀌면 캐시를 공유하면 안 된다.
4. [`create_dataloader()`](../../framework/src/dataloader/__init__.py)에
   task·backend·target별 선택 분기가 필요한지 확인한다. 벤더 전용 kwargs는
   [`main.py`](../../framework/src/main.py)의 `loader_kwargs` 조립부에서
   실제로 전달해야 한다.
5. 입력 sample을 artifact metadata의 이름·shape·dtype·layout·값 범위와
   비교하는 테스트를 만든다. ONNX profile의 기대 shape만 확인해서 끝내지 않는다.

`InferencePipeline`은 일반 sample 목록을 stack하며, 정적 batch loader는
`is_static_batched=True`로 이미 batch를 반환한다. 새 Loader가 어느 쪽인지
`get_metadata()`에 명시한다. 비동기 실행은 표본 경계와 반복 가능한 접근이
중요하므로 `load_by_index()`도 실제 데이터를 반환해야 한다.

## 3. 출력 계약과 Decoder 연결

Runtime의 첫 반환값을 관찰해 출력 **이름, shape, dtype, layout, 값의 의미,
그래프 내부 후처리**를 기록한다. 그 뒤 evaluator가 요구하는 형태까지
어느 계층이 변환할지 결정한다.

1. 이미 evaluator가 받는 형태라면 decoder는 `None`일 수 있다.
2. raw head, box, class score, mask, keypoint, NMS 결과처럼 의미 변환이
   필요하면 `framework/src/decoders/`에 `decode(outputs)`를 구현한다.
3. [`create_decoder()`](../../framework/src/decoders/__init__.py)에 선택 조건을
   연결한다. 선택 조건은 단순 모델명이 아니라 **검증한 출력 ABI**와 target·
   runtime option을 구분해야 한다. 지원하지 않는 ABI는 명시적으로 거부한다.
4. evaluator의 `add_batch()`가 받는 이름·shape·좌표계·class ID 범위를
   test fixture로 확인한다. 예를 들어 객체 탐지는 `detections (N, 7)`,
   pose는 `detections`와 `keypoints`를 사용한다.
5. 기존 evaluator를 쓸 수 있는지 확인한다. 정답 형식이나 metric 의미가
   달라진다면 새 evaluator가 필요하다.

구체적인 차이는 [ResNet50](../../framework/docs/model-execution-specs/resnet50.md),
[YOLOv5m](../../framework/docs/model-execution-specs/yolov5m.md),
[YOLOv8s-pose](../../framework/docs/model-execution-specs/yolov8s-pose.md)의
출력 계약을 예로 본다. Hailo NMS 출력, DEEPX raw/packed 출력,
Regulus raw head를 하나의 이름만으로 같은 decoder에 연결하면 안 된다.

## 4. Evaluator와 새 task

기존 task의 정답·출력 의미가 같으면
[`create_evaluator()`](../../framework/src/evaluators/__init__.py)의
기존 evaluator를 재사용한다. 품질 지표를 새로 정의해야 하면
`Evaluator`의 모든 추상 메서드를 구현한다. 실제 실행에서 호출되는
`add_batch()`는 batch별 경량 통계를 누산하고 `compute()`는 최종 지표를
반환한다. `evaluate()`는 호환 경로로 구현한다. `quality_status=passed`는
이 함수들이 오류 없이 집계됐다는 상태이지 품질 임계값 통과가 아니다.

새 task는 다음 순서로 연결한다.

1. [`Task`](../../framework/src/core/model_spec.py)를 추가하고 모델 profile을
   등록한다.
2. Loader와 선택적 전처리기, 정답 형식을 정의한다. 필요하면
   [dataset resolver](../../framework/src/utils/dataset_resolver.py)의 경로 규칙을
   확장한다.
3. Runtime 출력이 곧바로 evaluator 입력이 아니면 Decoder를 추가한다.
4. Evaluator에서 지표 이름·단위·분모를 정의한다.
5. `dataloader/__init__.py`, `decoders/__init__.py`,
   `evaluators/__init__.py`의 factory 분기를 확인한다.
6. `main.py`의 task별 검증, loader/evaluator kwargs, artifact·프로필 선택을
   연결한다. 현재 조립 로직에는 모델·벤더별 분기가 있으므로 **registry
   등록만으로 모든 새 task가 자동 연결되지는 않는다.**
7. 동일 sample로 Loader → Runtime 출력 → Decoder → Evaluator를 이어
   정상 출력·정답 정렬을 확인한다. 이후 소량 CLI와 전체 평가를 구분한다.

## 5. 문서와 결과 필드

새로 검증한 조합은 모델별 실행 명세에 출처, native artifact 취득·형식,
입출력 계약, 전후처리, 확인된 실행 모드, 실제 CLI 명령을 적는다.
SHA-256과 결과·정확도 수치는 각각 별도 기록·보고서와 연결한다.
새 metric을 저장한다면 [`result_store.py`](../../framework/src/core/result_store.py)
및 API 표시 경로를 확인하고, 단위와 측정 구간을
[측정 계약](../../framework/docs/measurement-contract.md)에 반영한다.
