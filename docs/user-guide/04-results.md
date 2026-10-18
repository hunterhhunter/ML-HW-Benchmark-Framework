# 04. 결과 확인 및 해석

[목차](README.md) · [이전: 벤치마크 실행](03-execution.md) · [다음: 오류 대응](05-troubleshooting.md)

## 결과 파일 찾기

CLI 출력에서 `RUN_ID=<id>`와 결과 파일 경로를 확인한다. 기본 CSV는
`framework/results/benchmark_results.csv`이고 `--results-path`로 변경할 수
있다. CSV 한 행이 한 실행이다. `RUN_ID_RESERVED=<id>`만 있고 `RUN_ID=<id>`가
없으면 완료된 결과로 간주하지 않는다.

다음은 [소량 확인 예시](03-execution.md)의 마지막 CSV 행을 읽는다. 다른 실행은
`path`를 해당 결과 파일로 바꾼다.

```bash
cd /path/to/ML-HW-Benchmark-Framework/framework
python3 - <<'PY'
import csv
from pathlib import Path

path = Path("results/user-guide-smoke.csv")
with path.open(newline="") as file:
    rows = list(csv.DictReader(file))
row = rows[-1]
for key in (
    "run_id", "model_name", "target_id", "inference_mode",
    "async_run_status", "quality_status", "comparison_eligible",
    "async_runtime_completed_samples_per_sec", "details_path",
):
    if row.get(key):
        print(f"{key}: {row[key]}")
PY
```

웹 결과 목록은 기본 `framework/results/benchmark_results.csv`를 읽는다.
CLI에서 별도 CSV 경로를 지정했다면 그 파일은 CLI로 확인한다.

## 상태와 지표 읽기

비동기 결과는 `async_run_status`, `quality_status`, `comparison_eligible`와
완료·실패 sample 수를 함께 본다. `quality_status=passed`는 평가기가 정상
집계한 상태이며 정확도 목표 통과를 뜻하지 않는다. schema v2의 공식 완료
처리량은 `async_runtime_completed_samples_per_sec`다. `details_path`와 선택적
`request_trace_path`는 결과 CSV의 상위 디렉터리를 기준으로 연다.

`e2e`는 동기 실행 모드 이름이다. 저장된 Runtime 지연시간을 System E2E
지연시간으로 해석하지 않는다. 지표의 시간 경계와 비교 조건은
[측정 계약](../../framework/docs/measurement-contract.md)을 따른다.
