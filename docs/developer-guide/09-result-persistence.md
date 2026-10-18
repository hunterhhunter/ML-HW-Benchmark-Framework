# 09. 결과 저장과 산출물 수명

[목차](README.md) · [이전: 비동기 실행 내부 구조](08-async-internals.md) · [다음: 벤더 구현 지도](10-vendor-implementation-map.md)

이 장은 결과를 **어떤 순서로 게시하고 실패 후 무엇을 확인하는지** 설명한다.
CSV 열과 비동기 JSON 필드의 의미는
[사용자 결과 가이드](../user-guide/04-results.md)와
[비동기 측정 가이드](../async-inference-queue.md)를 따른다. 저장 구현은
[`result_store.py`](../../framework/src/core/result_store.py), 예약 상태와
산출물 경로는 [`artifact_reservation.py`](../../framework/src/core/artifact_reservation.py)에 있다.

## 실행 모드별 저장 경로

| 경로 | 코드 흐름 | 완료 표시 |
|---|---|---|
| 동기 `e2e` | `main.py` → `save_result()` → CSV 한 행 | 저장 뒤 `RUN_ID=<id>` |
| `async_queue` | `reserve_run_artifacts()` → 실행·정리 → 선택적 trace, details JSON → `save_result()` → CSV 한 행 | 예약 직후 `RUN_ID_RESERVED=<id>`, CSV 반영 뒤 `RUN_ID=<id>` |

기본 CSV는 `framework/results/benchmark_results.csv`다. `--results-path`를
바꾸면 그 CSV의 상위 디렉터리가 해당 실행의 results root가 된다.
비동기 예약 객체는 같은 root 아래 `details/<run_id>.json`,
`details/<run_id>.failure.json`, `traces/<run_id>.jsonl` 경로를 정한다.
현재 CLI는 CSV의 `details_path`·`failure_details_path`·`request_trace_path`를
**results root의 상위 디렉터리 기준** 상대 경로로 기록한다. 예를 들어
`/work/results/run.csv`의 details 참조는 `results/details/<run_id>.json`이다.
웹 리더는 이전 `details/<run_id>.json` 형식도 허용하되, 두 형식 모두
최종 파일이 같은 results root 안에 있어야 한다. trace 파일은 옵션을
켠 경우에만 만든다.

`RUN_ID_RESERVED`는 식별자와 파일 소유권을 확보했다는 뜻이다. 정상·invalid
결과 모두 CSV 저장이 끝나면 `RUN_ID`를 출력할 수 있으므로, 이 문자열만으로
실행의 유효성이나 품질을 판정하지 않는다. `async_run_status`,
`quality_status`, 오류 수와 산출물 경로를 함께 확인한다.

## 예약부터 CSV 반영까지

1. [`main.py`](../../framework/src/main.py)의 `execute_benchmark()`가
   비동기 설정을 검증하고 `reserve_run_artifacts(results_path)`를 호출한다.
   예약은 같은 run ID의 CSV 행·예약 표식과 충돌하지 않는지 검사하고,
   owner token이 든 `RunArtifactReservation`을 반환한다.
2. 실행이 끝나면 요청과 native 작업의 정리 상태를 확인한다. 안전한 경우
   Runtime unload를 마친 뒤 정상 details를 게시한다. 선택적 request trace는
   별도 writer가 기록·닫는다. `save_async_details()`는 예약을 검증하고
   `schema_version`과 run ID를 포함한 JSON을 게시한다.
3. `save_result()`는 고정 메타데이터와 metrics를 합쳐 CSV 행을 만든다.
   `META_COLUMNS`의 이름과 겹치는 metric key는 메타데이터를 덮어쓰지 않는다.
   예약 모드에서는 예약 검증, CSV 잠금, 행 fingerprint와 pending/consumed
   상태를 확인한 뒤 같은 run ID의 행을 반영한다.
4. CSV 저장 중 예외가 나면 `main.py`는
   `get_reserved_result_state()`로 `active`·`pending`·`consumed`를 확인한다.
   재시도 가능한 pending 행은 **같은 내용**으로 다시 저장하며, 이미 반영된
   행을 새 행으로 중복 추가하지 않는다.

fatal 실패에서는 가능한 경우 invalid details와 CSV를 저장한다. 정상 details
또는 CSV가 이미 게시된 뒤에 실패하면 기존 파일을 고치지 않고
`save_async_failure_details()`로 별도 `.failure.json` 복구 기록을 게시한다.
CSV가 이미 반영됐다면 기존 행을 보존한다. 따라서 오류 조사 시 stdout의
예약 ID뿐 아니라 CSV 행, 정상 details, failure details와 stderr를 함께 본다.
복구 기록을 항상 만들 수 있다는 보장은 없으며 저장 실패 자체도 진단에 남긴다.

## 결과 형식을 확장할 때

| 바꾸는 것 | 수정·확인 위치 |
|---|---|
| 공통 메타데이터 열 | `result_store.py`의 `META_COLUMNS`, `save_result()` 인자와 행 구성, `main.py`의 `_result_save_kwargs()` |
| 지표 열 | evaluator 또는 async metrics 생성 위치와 `save_result(metrics=...)`; 이름·단위·측정 구간을 측정 문서에 기록 |
| details·trace | `main.py`의 details 조립, `async_inference/trace.py`, 예약 객체의 경로와 저장·복구 코드 |
| 웹 결과 표시 | `backend/app/services/result_sources.py`, `result_service.py`, `artifact_reader.py`, `frontend/src/webApi.ts`와 결과 화면 |

`load_results()`와 `get_result()`는 CSV를 읽는다. 현재 `delete_result()`는
CSV에서 해당 행만 제거하며 연결된 details·trace 파일을 삭제하지 않는다.
웹 결과 API도 이 삭제 함수를 사용한다. 저장·복구 변경은
[`test_result_store.py`](../../framework/tests/test_result_store.py)와
[`test_async_result_artifacts.py`](../../framework/tests/test_async_result_artifacts.py),
웹 변환은 [`test_results_api.py`](../../backend/tests/test_results_api.py)와
[`test_result_sources_api.py`](../../backend/tests/test_result_sources_api.py)에서
확인한다.
