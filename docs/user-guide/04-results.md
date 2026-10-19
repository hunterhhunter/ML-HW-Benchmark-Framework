# 04. 결과 확인 및 해석

[목차](README.md) · [이전: 벤치마크 실행](03-execution.md) · [다음: 오류 대응](05-troubleshooting.md)

## 결과 파일 찾기

CLI 출력에서 `RUN_ID=<id>`와 결과 파일 경로를 확인한다. 기본 CSV는
`framework/results/benchmark_results.csv`이고 `--results-path`로 변경할 수
있다. CSV 한 행이 한 실행이다. `RUN_ID_RESERVED=<id>`만 있고 `RUN_ID=<id>`가
없으면 완료된 결과로 간주하지 않는다. `RUN_ID`가 출력되어도 실행 유효성은
CSV의 상태 필드로 따로 판정한다.

다음은 [소량 확인 예시](03-execution.md)의 CSV에서 해당 `RUN_ID` 행을 찾는다.
`RUN_ID=...`의 값과 CSV 경로를 실제 실행에 맞게 바꾼다.

```bash
cd /path/to/ML-HW-Benchmark-Framework/framework
RESULTS_CSV=results/user-guide-smoke.csv
RUN_ID=여기에_출력된_ID
python3 - "$RESULTS_CSV" "$RUN_ID" <<'PY'
import csv
import sys
from pathlib import Path

path = Path(sys.argv[1])
run_id = sys.argv[2]
with path.open(newline="") as file:
    rows = [row for row in csv.DictReader(file) if row["run_id"] == run_id]
if len(rows) != 1:
    raise SystemExit(f"run_id={run_id}: CSV에서 {len(rows)}개 행 발견 (예상: 1개)")
row = rows[0]
for key in (
    "run_id", "model_name", "target_id", "inference_mode",
    "async_run_status", "quality_status", "comparison_eligible",
    "async_metric_schema_version", "async_measurement_boundary",
    "quality_evaluator_samples", "async_runtime_completed_samples", "async_failed_requests",
    "async_rejected_requests", "async_timed_out_requests",
    "async_outstanding_requests", "async_runtime_completed_samples_per_sec",
):
    print(f"{key}: {row.get(key) or '-'}")
for key in ("details_path", "failure_details_path", "request_trace_path"):
    if row.get(key):
        relative = Path(row[key])
        artifact = (path.parent.parent / relative
                    if relative.parts and relative.parts[0] == path.parent.name
                    else path.parent / relative)
        print(f"{key}: {artifact} (exists={artifact.is_file()})")
PY
```

웹 **결과** 화면은 기본 CSV와 등록한 CSV source를 읽는다. 별도
`--results-path`로 실행했다면 서버에서 접근 가능한 파일을 결과 화면에
등록한다. 허용된 `WEB_RESULTS_ROOTS` 안의 기존 CSV만 등록할 수 있다.
결과의 식별자는 **source ID와 run ID의 쌍**이다. 같은 run ID가 다른 CSV에
있어도 한 결과로 합치지 않는다.

목록에는 실행 모드, 비동기 상태, 품질 상태와 원본 `comparison_eligible` 값이
표시된다. **상세**에서 전체 CSV 메타데이터와 저장된 측정값을 보고,
파일이 있으면 details·failure details·request trace를 제한된 크기로 읽는다.
빈 값이나 `NaN`은 0으로 채우지 않는다. **비교**에서는 결과 행과 원본 지표
key를 직접 선택한다. 선택한 key마다 별도의 가로 막대 그래프가 나타나며,
실행 조건 표를 펼쳐 함께 읽는다. 색상 강조는 동일 key의 수치상 최저·최고
표시일 뿐이다. 웹은 비교 가능 여부나 실험의 우열을 자동 판단하지 않는다.
삭제는 CSV 행만 제거하고 details·trace는 남긴다.

## 엣지 24개 조합을 웹에서 보기

기본 `framework/results/benchmark_results.csv`에는 모델 3종 × NPU 4종 ×
실행 모드 2종의 결과 24행이 들어 있다. 웹 **결과** 화면에서 기본 CSV source를
선택하면 바로 볼 수 있다. 같은 내용의 보관본은
[`edge-benchmark-results.csv`](../../result/final-result/edge-benchmark-results.csv)다.
기본 CSV를 새 실행이나 행 삭제로 변경한 뒤 이 24행으로 다시 맞추려면
저장소 루트에서 다음 명령을 실행한다. 현재 기본 CSV 내용은 덮어쓴다.

```bash
cp result/final-result/edge-benchmark-results.csv framework/results/benchmark_results.csv
```

통합본의 출처와 값별 측정 경계는
[`result/final-result/README.md`](../../result/final-result/README.md)에 기록했다.
원본 보관 자료가 없는 체크아웃에서는 표의 24행을 볼 수 있지만 원본 부속
파일까지 열 수 있는 것은 아니다.

## 상태와 지표 읽기

동기 `e2e`는 명령이 오류 없이 끝나고 해당 CSV 행이 저장됐는지 확인한다.
비동기 품질·런타임 패스는 `async_run_status=valid`, `quality_status=passed`,
`comparison_eligible=True`와 두 패스의 sample 수를 확인하고, 실패·거부·timeout·미완료
요청 수가 0인지 본다. `runtime-only`는 `quality_status=skipped`와
`comparison_eligible=False`가 예상 상태다. `quality_status=passed`는 평가기가 정상
집계한 상태이며 정확도 목표 통과를 뜻하지 않는다. schema v2의 공식 완료
처리량은 `async_runtime_completed_samples_per_sec`다. 현재 CLI가 기록한
`details_path`·`request_trace_path`는 `결과디렉터리명/details/...`,
`결과디렉터리명/traces/...` 형식이며 결과 디렉터리의 **상위**를 기준으로
해석한다. 이전 CSV의 `details/...`·`traces/...` 형식은 CSV가 있는
결과 디렉터리를 기준으로 해석한다. 웹은 어느 형식이든 최종 파일이 해당
CSV의 결과 디렉터리 안에 있는지 검사한다.

`e2e`는 동기 실행 모드 이름이다. 저장된 Runtime 지연시간을 System E2E
지연시간으로 해석하지 않는다. 지표의 시간 경계와 비교 조건은
[측정 계약](../../framework/docs/measurement-contract.md)을 따른다.
