# 12. 알려진 제약과 유지보수

[목차](README.md) · [이전: 웹과 CLI 연결](11-web-cli-integration.md)

`origin/main@30c4db75`와 `feat/web-cli-parity` 구현을 기준으로 정리한 제약이다. 웹 변경은 해당 브랜치 통합 후 적용된다. 아래의
상태는 구현·설정 계약을 설명하며, 모든 장치에서 실측 검증했다는 뜻은 아니다.
모델별 실행 조건은 [실행 명세](../../framework/docs/model-execution-specs/resnet50.md),
실행 결과와 증거는 별도 결과·보고서에서 관리한다.

| 범위 | 현재 제약 | 사용·유지보수 시 확인할 것 |
|---|---|---|
| IREE Runtime | [`iree` 등록 항목](../../framework/src/runtimes/__init__.py)은 `unsupported_reason`을 가진 placeholder다. 생성 시 사용 불가 오류가 난다. | 현재 실행에는 다른 지원 Runtime을 선택한다. IREE를 활성화할 때 공통 Runtime 계약과 registry 검증을 먼저 맞춘다. |
| target 목록 | [`TargetSpec` 등록](../../framework/src/core/targets.py)과 웹의 target 목록은 현재 host의 SDK·장치·artifact 준비 여부를 검사하지 않는다. | [설치·장치 확인](../user-guide/01-installation.md)과 모델별 artifact 입력 계약을 실행 전에 확인한다. 지원 범위를 늘릴 때 registry 표시와 실제 실행 가능 조건을 함께 갱신한다. |
| native async | target의 `native_async` capability만으로 모든 artifact의 SDK 비동기 실행이 보장되지는 않는다. [Mobilint Runtime](../../framework/src/runtimes/mobilint_rt.py)은 artifact profile의 지원 여부와 load 전 async pipeline 설정을 검사하고, [RBLN 정적 Runtime](../../framework/src/runtimes/rbln_rt.py)의 `async_parallel`은 1 또는 2만 허용한다. | 해당 모델의 [실행 명세](../../framework/docs/model-execution-specs/resnet50.md)와 Runtime 옵션을 확인한다. 새 artifact를 추가하면 native callback·timeout·자원 정리까지 [검증 절차](05-verification.md)로 확인한다. |
| RBLN vLLM | [준비된 모델 manifest](../../framework/src/runtimes/rbln_vllm_rt.py)의 batch·decoder 설정이 실행 옵션과 맞아야 한다. Llama 3.1 8B·3.2 3B의 단일 NPU 경로는 명시적 opt-in과 `max_num_seqs=1` 등을 요구하는 제한된 실험 경로다. | [RBLN vLLM 가이드](../../framework/docs/rbln-vllm-setup.md)의 준비·실행 조건을 맞춘다. 기본 target 옵션만 보고 다른 모델 디렉터리나 장치 수로 일반화하지 않는다. |
| 웹 실행 | CLI 옵션은 웹 schema·폼과 일대일로 연결되지만 등록 target·runner가 현재 host의 실제 SDK·장치 실행 가능성을 보증하지 않는다. | preview의 정적 검사 뒤에도 확인된 artifact·데이터·Python 환경에서 소량 실행한다. 새 CLI 옵션은 [요청 schema → 명령 조립 → 화면](11-web-cli-integration.md)을 함께 수정한다. |
| 웹 작업 상태 | SQLite job 상태는 재시작 후 조회할 수 있다. 이전 `running` 작업은 소유권을 확인할 수 없으면 `interrupted`로 남긴다. 작업 `completed`, CSV 존재, CSV 유효성은 별개다. | job의 두 run ID와 `result_present`, [CSV 상태 필드](../user-guide/04-results.md)를 대조한다. |
| 웹 결과 조회·삭제 | 기본·등록 CSV와 제한된 details·trace를 읽는다. [`delete_result()`](../../framework/src/core/result_store.py)는 CSV 행만 삭제하며 부속 파일은 남는다. CSV에 기록되지 않은 실행 조건은 웹이 추정하지 않는다. | 별도 CSV는 허용 root에서 등록하고, 누락 조건은 [부속 파일](09-result-persistence.md) 또는 실행 기록으로 확인한다. |
| 측정·전력 | 현재 동기 `e2e`의 Runtime 지연시간은 공통 System E2E 시간이 아니다. 전력 값도 수집 지점과 trace 상태가 다르면 같은 범위의 값이 아니다. | 수치에는 [측정 계약](../../framework/docs/measurement-contract.md)의 시간·전력 경계를 명시한다. 공통 System E2E나 전력 비교를 추가할 때는 계측 위치와 trace 검증을 별도로 구현한다. |

제약을 바꾼 뒤에는 이 표와 관련 운영 가이드, [변경 검증](05-verification.md)의
해당 단계를 함께 갱신한다. 실장비 확인 범위와 새로운 결과값은 실행에 사용한
artifact·환경·코드 revision과 연결해 별도 기록한다.
