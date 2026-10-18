# 개발자 설명서

`origin/main@30c4db75`의 프레임워크 구현을 기준으로 작성한다. 새 모델·태스크·NPU를
추가하거나 실행 경로를 수정할 개발자가 코드를 찾고, 각 계층의 계약을 확인하고,
변경 사항을 검증하는 순서로 읽는다.

| 순서 | 문서 | 다루는 내용 |
|---|---|---|
| 01 | [전체 구조와 실행 흐름](01-architecture.md) | CLI·웹, 구성 요소 조립, 동기·비동기 실행, 저장 |
| 02 | [공용 구성 요소 계약](02-component-contracts.md) | 모델·target, Loader, Runtime, Decoder, Evaluator, Compiler, Monitor의 책임 |
| 03 | [모델·태스크 확장](03-model-and-task-extension.md) | 새 모델, 전처리·출력 형식, 평가기와 새 태스크 연결 |
| 04 | [NPU 어댑터 확장](04-npu-adapter-extension.md) | Runtime·Compiler·Collector·Target 등록과 native async |
| 05 | [변경 검증과 인수](05-verification.md) | 구조 테스트, 실행 확인, 결과 판독, 실장비 확인 |
| 06 | [함수별 구현 참조](06-implementation-reference.md) | 추상 메서드별 반환값·호출 위치와 조건부 hook, 등록 후 연결 지점 |
| 07 | [실행 설정과 조립](07-configuration-and-assembly.md) | CLI·target·자산 선택, 옵션 병합, factory·실행 진입점 |
| 08 | [비동기 실행 내부 구조](08-async-internals.md) | 요청·완료 handoff 흐름, native 자원 소유권, 수정 위치 |

사용자의 설치·실행 절차는 [사용자 설명서](../user-guide/README.md), 지표의 시간
경계는 [측정 계약](../../framework/docs/measurement-contract.md), 에지 모델의 실제
입출력과 실행 명령은 [모델별 실행 명세](../../framework/docs/model-execution-specs/resnet50.md)를
기준으로 한다. 이 설명서의 코드 예제는 인터페이스와 연결 방법을 보여준다.
특정 HEF·DXNN·MXQ·RBLN·FXB가 그 형태로 출력한다는 보증은 아니다.

## 먼저 읽을 코드

- [`main.py`](../../framework/src/main.py): CLI 옵션, target·모델·artifact 선택, 구성 요소 조립
- [`InferenceEngine`](../../framework/src/core/inference_engine.py): 동기·비동기 실행
- [`InferencePipeline`](../../framework/src/core/inference_pipeline.py): 입력 batch 구성
- [`CompletionCoordinator`](../../framework/src/core/async_inference/completion.py): decoder·evaluator 호출과 완료 처리
- [`result_store.py`](../../framework/src/core/result_store.py): CSV와 실행 산출물 연결
