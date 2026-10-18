# 프레임워크 소스 코드

`main.py`가 모델·target·artifact·데이터 설정을 해석하고 구성 요소를 조립한다.
전체 실행 흐름과 계층별 책임은 [개발자 설명서](../../docs/developer-guide/README.md)를
기준으로 읽는다. 설치·실행 명령은 [사용자 설명서](../../docs/user-guide/README.md)에 있다.

## 코드에서 시작할 곳

| 위치 | 현재 역할 |
|---|---|
| [`main.py`](main.py) | CLI 옵션, 기본 자산 준비, target·모델별 옵션 병합, 실행·결과 저장 |
| [`core/model_profiles.py`](core/model_profiles.py), [`core/targets.py`](core/targets.py) | 모델 프로필과 target 등록 정보 |
| [`core/inference_engine.py`](core/inference_engine.py) | 동기 `e2e` 루프와 `async_queue` 진입점 |
| [`core/benchmarkrunner.py`](core/benchmarkrunner.py) | 동기 실행에서 `InferenceEngine`을 호출하는 호환 façade |
| [`core/inference_pipeline.py`](core/inference_pipeline.py), [`core/runtime_executor.py`](core/runtime_executor.py) | batch 입력 구성과 Runtime 호출·완료 경계 |
| [`dataloader/`](dataloader/), [`preprocessor/`](preprocessor/) | 데이터·정답 공급과 모델 입력 준비 |
| [`runtimes/`](runtimes/), [`compilers/`](compilers/), [`monitors/`](monitors/) | SDK 추론 호출, 선택적 artifact 생성, 하드웨어 관측 |
| [`decoders/`](decoders/), [`evaluators/`](evaluators/) | 출력 의미 해석과 품질·시간 지표 계산 |
| [`core/result_store.py`](core/result_store.py) | CSV, 비동기 details·trace와 실행 ID 연결 |

Runtime·Compiler·Monitor·Target은 registry로 선택한다. Loader·Decoder·Evaluator는
현재 모델·task·target 조건에 따른 factory 분기를 사용한다. 새 모델이나 NPU의
수정 위치는 [모델·태스크 확장](../../docs/developer-guide/03-model-and-task-extension.md)과
[NPU 어댑터 확장](../../docs/developer-guide/04-npu-adapter-extension.md)을 따른다.
`iree` Runtime은 등록돼 있지만 `unsupported_reason`이 설정된 사용 불가 경로다.

## 실행 경로

동기 `e2e`에서는 `BenchmarkRunner`가 warmup과 monitor 수명을 관리하고
`InferenceEngine.run_e2e()`를 호출한다. `async_queue`에서는 `main.py`가
`InferenceEngine.run_async()`를 호출한다. 두 모드의 시간·완료 경계는
[측정 계약](../docs/measurement-contract.md), 저장 결과의 구조는
[결과 안내](../results/README.md)를 확인한다.

기본 모델 프로필에 준비 스크립트가 있어도 모든 target의 장치 artifact가
자동 생성되지는 않는다. 벤더 실행은 해당 SDK·artifact와
[모델별 실행 명세](../docs/model-execution-specs/resnet50.md)의 입출력 조건을
확인한다.
