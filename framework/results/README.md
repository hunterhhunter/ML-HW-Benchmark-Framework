# 벤치마크 결과 파일

기본 결과 CSV는 `benchmark_results.csv`다. 실행 시 생성되는 결과와
측정 자료는 로컬에만 저장하며 Git에서 추적하지 않는다. 웹 결과 화면은
로컬 기본 CSV가 있으면 이를 읽는다.
CLI에서 `--results-path`를 지정하면 해당 CSV의 상위 디렉터리가 그 실행의
결과 root가 된다. 기본 CSV를 포함한 실행 결과 CSV·details·trace·예약
파일은 Git에서 추적하지 않는다.

## 파일과 실행 ID

| 파일·출력 | 의미 |
|---|---|
| `benchmark_results.csv` 또는 지정한 CSV | 저장이 완료된 실행별 한 행. 고정 메타데이터 열과 모델·모드별 지표 열을 포함한다. |
| `RUN_ID_RESERVED=<id>` | 비동기 실행의 ID와 산출물 경로를 예약했다는 출력. 완료 판정이 아니다. |
| `RUN_ID=<id>` | CSV 저장 뒤 출력하는 ID. 비동기 실행에서는 이 값만으로 유효성·품질을 판정하지 않는다. |
| `details/<id>.json` | 비동기 실행의 상세 상태와 지표. CSV의 `details_path`로 찾는다. |
| `details/<id>.failure.json` | 게시 후 실패 등의 복구 기록. 생성된 경우 CSV의 `failure_details_path`로 찾는다. |
| `traces/<id>.jsonl` | `--save-request-trace`를 켠 경우의 요청별 기록. CSV의 `request_trace_path`로 찾는다. |
| `.run_artifacts/` | 비동기 예약·게시 상태를 관리하는 내부 파일. 결과 판독에는 CSV와 details를 사용한다. |

`details_path`·`failure_details_path`·`request_trace_path`는 결과 CSV의
상위 디렉터리를 기준으로 하는 상대 경로다. CSV 필드의 정의와 저장·복구
순서는 [`result_store.py`](../src/core/result_store.py)와
[개발자 결과 저장 장](../../docs/developer-guide/09-result-persistence.md)에 있다.
구조가 다른 과거 결과 파일은 현재 프레임워크의 기본 결과 파일로 읽거나
열을 임의로 보정하지 않는다.

## 결과를 읽을 때

비동기 실행은 CSV의 `async_run_status`, `quality_status`,
`comparison_eligible`, sample 수와 오류·미완료 수를 함께 확인한다.
`quality_status=passed`는 평가가 완료됐다는 상태이며 정확도 목표 통과를
뜻하지 않는다. 지표의 측정 구간은 [측정 계약](../docs/measurement-contract.md),
CSV에서 해당 실행을 찾는 명령은 [사용자 결과 안내](../../docs/user-guide/04-results.md)를
따른다. 웹 결과 목록은 기본 CSV와 등록한 다른 CSV를 읽는다. 현재 결과 삭제 API는 CSV 행만
삭제하며 연결된 details·trace 파일은 삭제하지 않는다.
