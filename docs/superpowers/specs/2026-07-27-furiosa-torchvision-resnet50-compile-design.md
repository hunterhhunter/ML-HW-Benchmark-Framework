# Furiosa TorchVision ImageNet ResNet50 compile 검증 설계

## 목적

Kalray ResNet50 ONNX를 변환한 PyTorch 모델이 Furiosa Torch 2026.3 compiler에서
실패한 원인이 ONNX 변환 경로인지, RNGD compiler 자체인지 분리한다. ONNX 변환을
전혀 사용하지 않는 TorchVision ImageNet ResNet50을 strict Furiosa backend로
컴파일하고 첫 추론까지 수행하는 재현 가능한 서버 도구를 제공한다.

이 도구는 벤치마크 모델 프로필을 추가하는 기능이 아니다. 먼저 별도 프로세스에서
모델 전체가 RNGD에 컴파일되는지 판정하는 진단 도구다. 성공이 확인된 뒤에만 기존
`furiosa-rngd-torch` 공통 추론 파이프라인용 프로필을 추가한다.

## 모델과 입력 계약

- 모델: `torchvision.models.resnet50`
- 가중치: `ResNet50_Weights.IMAGENET1K_V2`
- 가중치 파일은 TorchVision/PyTorch의 표준 cache에 자동 다운로드한다.
- 입력: 고정 shape `(1, 3, 224, 224)`, `float32`, NCHW
- 입력값: 고정 seed로 생성한 동일한 CPU/NPU tensor
- 출력: `(1, 1000)` classification logits

실제 ImageNet 데이터셋이나 정확도 측정은 이 compile smoke test 범위에 포함하지
않는다. 여기서 ImageNet은 사전학습 가중치를 의미하며, compiler 지원 판정이 우선이다.

## 실행 흐름

새 도구 `framework/tools/compile_furiosa_resnet50.py`는 다음 순서로 실행한다.

1. CLI 인자와 실행 환경을 검증한다.
2. ImageNet V2 가중치를 내려받아 CPU 모델을 만든다.
3. 고정 입력으로 CPU reference logits와 Top-1 class를 계산한다.
4. 별도의 같은 모델에 같은 가중치를 적용하고 `furiosa:0`으로 이동한다.
5. 아래 strict backend 계약으로 `torch.compile`을 구성한다.
   - `CompilerConfig(tactic_hint=TacticHintConfig.Default)`
   - `eager_fallback=False`
   - `fullgraph=True`
   - `dynamic=False`
6. 첫 호출을 실행하여 compile, load, inference를 완료한다.
7. 두 번째 호출을 실행하여 warm 상태의 단일 추론 시간을 측정한다.
8. NPU 출력을 CPU로 옮겨 shape, finite 여부, CPU/NPU Top-1 일치 여부와
   최대 절대 오차를 출력한다.

단계 시작과 완료를 즉시 flush해서 출력한다. 따라서 장시간 compile 중인 경우에도
마지막으로 진입한 단계를 서버 터미널에서 확인할 수 있다.

## 성공과 실패 계약

성공은 다음 조건을 모두 만족해야 한다.

- strict Furiosa compile과 첫 호출이 예외 없이 완료된다.
- 두 번째 호출이 완료된다.
- 출력 shape가 `(1, 1000)`이다.
- 모든 NPU 출력이 finite다.
- CPU와 NPU의 Top-1 class가 일치한다.

컴파일 오류, Rust panic의 Python 예외 전파, 잘못된 출력, non-finite 출력 또는 Top-1
불일치는 실패다. 도구는 이를 숨기거나 CPU eager fallback으로 대체하지 않으며
non-zero exit code로 종료한다. 수치 오차는 진단용으로 출력하되 별도 임계값을
도입하지 않는다. BF16 등의 compiler 내부 정밀도 선택 때문에 logits 전체의 엄격한
FP32 equality는 이 smoke test의 지원 판정 기준이 아니기 때문이다.

## CLI와 운영 안전장치

기본 실행은 추가 인자 없이 가능하게 한다. 필요한 최소 옵션만 제공한다.

- `--device`: 기본값 `furiosa:0`
- `--seed`: 기본값 `0`
- `--torch-home`: 선택한 가중치 cache root

`--torch-home`은 프로세스 안에서 `TORCH_HOME`으로 적용한다. 기존 cache를 파괴하거나
덮어쓰지 않는다. Python 안에서 compiler thread를 강제 종료하지 않고, 운영자는
문서에 제공한 shell `timeout --signal=INT --kill-after=30s 45m`으로 프로세스 경계를
관리한다.

## 테스트 전략

개발 호스트에는 RNGD와 Furiosa SDK가 없을 수 있으므로 테스트는 vendor 모듈을
대체한 fake Torch/TorchVision/Furiosa 객체로 실행한다.

- 기본 CLI가 ImageNet V2 가중치와 `furiosa:0`을 선택하는지 검증
- `eager_fallback=False`, `fullgraph=True`, `dynamic=False`가 반드시 전달되는지 검증
- CPU와 NPU에 동일한 입력과 가중치를 사용하는지 검증
- shape, finite, Top-1 성공 시 결과 요약을 반환하는지 검증
- compile 오류와 Top-1 불일치를 성공으로 변환하지 않는지 검증
- 기존 Furiosa Torch runbook에 서버 명령과 결과 해석이 포함되는지 검증

구현은 vendor import를 실행 함수 내부에서 지연해 기존 테스트 수집과 다른 runtime을
오염시키지 않는다.

## 비범위

- 구형 Furiosa Model Zoo의 Warboy ENF 생성 및 실행
- ONNX 또는 onnx2torch 변환
- ImageNet validation dataset 다운로드와 Top-1 정확도 측정
- 재사용 가능한 FXB/ENF artifact 저장
- `resnet50` 기존 프로필의 모델 소스 변경
- compiler panic 우회 또는 eager CPU fallback

