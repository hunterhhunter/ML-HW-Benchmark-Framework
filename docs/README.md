# 문서 안내

이 문서들은 `origin/main@30c4db75`의 코드 경로를 기준으로 정리했다.
**사용·개발 가이드**는 현재 절차의 출발점이다. **검증·실험 기록**의 판정은
적힌 시점의 환경과 artifact에 한정된다. target 등록과 문서의 예시만으로
모든 모델 조합의 실장비 실행이 확인되는 것은 아니다.
`docs/`와 `framework/docs/`의 문서를 이 목차에서 함께 찾을 수 있다.

## 처음 읽을 문서

| 목적 | 문서 |
|---|---|
| 설치부터 실행·결과 확인까지 | [사용자 설명서](user-guide/README.md) |
| 구조 파악과 모델·NPU 확장 | [개발자 설명서](developer-guide/README.md) |
| 새 NPU 등록의 기존 예제 | [NPU 등록 방법](npu-plugin-registry.md). 현재 계약은 [개발자 NPU 확장 장](developer-guide/04-npu-adapter-extension.md)을 기준으로 한다. |

## 공통 측정·모델 계약

- [측정 계약](../framework/docs/measurement-contract.md): 동기·비동기 시간 경계와 결과 해석의 기준.
- [비동기 추론 큐 측정 가이드](async-inference-queue.md): 두 패스, 요청 수명, 세부 지표와 trace.
- 에지 모델 실행 명세: [ResNet50](../framework/docs/model-execution-specs/resnet50.md) ·
  [YOLOv5m](../framework/docs/model-execution-specs/yolov5m.md) ·
  [YOLOv8s-pose](../framework/docs/model-execution-specs/yolov8s-pose.md).
  자산 해시는 [ResNet50](../framework/docs/model-execution-specs/resnet50-sha256.md) ·
  [YOLOv5m](../framework/docs/model-execution-specs/yolov5m-sha256.md) ·
  [YOLOv8s-pose](../framework/docs/model-execution-specs/yolov8s-pose-sha256.md)에 분리했다.
- [TTM-R2 3종 NPU 실행](../framework/docs/ttm-r2-framework.md): 해당 모델의 고정 입력·평가 계약과 실행 절차.

## 벤더별 운영 문서

| 벤더 | 설치·실행·컴파일 | 문제 해결·추가 자료 |
|---|---|---|
| Hailo | [Hailo-8/10H Runtime](hailo-async-runtime.md) | 위의 에지 모델별 실행 명세 |
| DEEPX | [DX-COM·DX-RT 설치와 실행](deepx-setup.md) | 해당 문서의 설치 검증·오류 항목 |
| Mobilint | [Regulus Runtime](mobilint-regulus-runtime.md) · [ARIES Transformer·LLM](mobilint-aries-transformers.md) · [BERT 컴파일](mobilint-bert-compilation.md) | [ARIES 문제 해결 기록](mobilint-aries-troubleshooting.md) |
| Furiosa | [RNGD 설치·실행](furiosa-rngd-setup.md) | [문제 해결](furiosa-rngd-troubleshooting.md) |
| Rebellions | [RBLN 정적 운영](../framework/docs/rbln-setup.md) · [RBLN 컴파일](../framework/docs/rbln-compilation.md) · [RBLN vLLM](../framework/docs/rbln-vllm-setup.md) | [정적 모델 문제 해결](../framework/docs/rbln-troubleshooting.md) |

## 시점별 검증·실험 기록

다음 문서의 수치와 지원 판정은 작성 당시 자산·SDK·장치 조건에 묶여 있다.
현재 실행에는 위의 운영 가이드와 모델별 명세를 먼저 확인한다.

- [Cross-NPU Transformer·시계열 검증 현황](../framework/docs/cross-npu-model-validation.md): 2026-09-26까지의 중앙 검증 기록.
- [Regulus 정확도 검증 기록](regulus-runtime.md), [Rebellions ATOM Llama 검증 보고서](../framework/docs/rbln-vllm-atom-validation.md).
- [Mobilint 컴파일 실험](mobilint-compilation-experiments.md), [Furiosa 컴파일 실패 재현](furiosa-rngd-compilation-troubleshooting.md).
- [RNGD 논문용 serving 측정 프로토콜](rngd-paper-benchmark.md): 프레임워크 내부 실행과 측정 경계가 다른 별도 프로토콜.
- [이전 maccel 경로 안내](mobilint-accel-runtime.md): 과거 이름에서 현재 Regulus 가이드로 연결하는 문서.
- [구현 설계·계획 보관본](archive/implementation/README.md): 당시 작업 계획. 현재 사용법의 기준으로 쓰지 않는다.
