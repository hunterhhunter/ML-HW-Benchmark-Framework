# 02. NPU·모델·데이터셋 설정

[목차](README.md) · [이전: 환경 구축 및 설치](01-installation.md) · [다음: 벤치마크 실행](03-execution.md)

## 실행 대상 선택

모델 이름과 `--target`을 [프레임워크 지원 목록](../../framework/README.md)에서
고른다. `--target`은 Runtime과 장치 등을 묶으며, 지정하면 `--backend`와
`--device`보다 우선한다. 목록에 등록된 것만으로 그 모델·장비 조합의 실행이
확인된 것은 아니다.

| 자산 | 확인할 내용 | 주요 CLI 옵션 |
|---|---|---|
| 모델 | 프로필 이름, 실제 모델 파일, 입력 shape·layout·dtype | `--model`, 필요 시 `--onnx` 또는 `--model-path` |
| 데이터 | 평가 cohort, 이미지·정답 파일, 파일 순서 | `--dataset`, 필요 시 `--image-dir`, `--label-dir` |
| NPU artifact | 대상 장비·Runtime과 맞는 파일, 입력 계약 | target에 따라 `--hef` 또는 `--artifact --no-compile` |

기본 프로필은 모델·데이터가 없으면 준비 스크립트를 실행할 수 있으므로 네트워크
접근이나 별도 인증이 필요할 수 있다. 준비 방법은 [모델](../../framework/models/README.md)과
[데이터셋](../../framework/datasets/README.md) 문서를 따른다. 사전 컴파일
artifact의 형식과 `--no-compile` 지원 여부는 target마다 다르다.

모델과 데이터의 기본 상대 경로는 `framework/`를 기준으로 해석된다. 새 체크아웃에는
대용량 모델·데이터 파일이 포함되지 않는다. 다른 경로를 지정했다면 그 위치에 자산을
직접 준비해야 하며, 기본 다운로드 스크립트는 다른 경로를 대신 채우지 않는다.
기본 데이터셋은 이미지 개수와 정답 파일까지 확인한다. NPU용 사전 컴파일 파일과
SDK·driver는 해당 벤더 절차로 별도 준비한다.

## 조합별 설정과 재현 명령

에지 NPU의 확인된 조합, 입력 형식, 실행 명령은
[ResNet50](../../framework/docs/model-execution-specs/resnet50.md),
[YOLOv5m](../../framework/docs/model-execution-specs/yolov5m.md),
[YOLOv8s-pose](../../framework/docs/model-execution-specs/yolov8s-pose.md) 명세에 있다.
서버 NPU 설정은 [Rebellions 정적 모델](../vendors/rebellions/rbln-setup.md),
[Rebellions LLM](../vendors/rebellions/rbln-vllm-setup.md),
[Furiosa](../vendors/furiosa/furiosa-rngd-setup.md),
[Mobilint ARIES](../vendors/mobilint/mobilint-aries-transformers.md) 절차를 따른다.
TTM-R2의 [3종 NPU 검증 기록](../../framework/docs/ttm-r2-framework.md)에 있는
경로는 당시 서버 구성에 해당한다.

웹 실행 폼은 모델·target, 데이터와 사전 컴파일 artifact 경로, 동기·비동기
옵션을 받는다. 각 경로는 NPU가 연결된 백엔드 호스트 기준이다. target 등록만으로
장치·SDK나 모델 조합이 검증된 것은 아니므로 위 실행 명세의 입력·출력 계약을
먼저 확인한다.
