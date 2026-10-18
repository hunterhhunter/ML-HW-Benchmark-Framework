# 05. 변경 검증과 인수

[목차](README.md) · [이전: NPU 어댑터 확장](04-npu-adapter-extension.md)

새 클래스가 import되고 registry에 보이는 단계부터 실제 장치에서 같은
artifact로 품질·측정 조건을 확인하는 단계까지 판정을 분리한다. 앞 단계의
성공을 뒷 단계의 성공으로 표시하지 않는다.

| 단계 | 확인할 것 | 확인하지 못하는 것 |
|---|---|---|
| 구조 | registry 이름·alias, factory 선택, SDK 없는 import | 실제 장치·artifact 실행 |
| 계약 | Loader sample·metadata, Runtime 반환, Decoder 출력, Evaluator 집계 | 실장비 SDK completion·성능 |
| CPU 또는 mock smoke | CLI 조립, `RUN_ID`와 결과 저장 | NPU 품질·성능 |
| 장치 소량 실행 | 실제 artifact load, 입력·출력 ABI, 완료·해제 | 전체 데이터 품질·부하 안정성 |
| 전체 검증 | 고정 cohort와 설정의 품질·성능·trace·전력 증거 | 다른 artifact/보드로의 자동 일반화 |

## 구조와 선택 경로

framework 디렉터리에서 **실행에 사용할 Python 환경**을 활성화한다.
다음 명령은 vendor SDK 없이 registry 참조를 검사한다.

```bash
cd /path/to/ML-HW-Benchmark-Framework/framework
export PYTHONPATH=src
python - <<'PY'
from core.targets import validate_registry_graph

report = validate_registry_graph()
assert report["ok"], report
print("registry graph: OK")
PY
python -m pytest -q tests/test_plugin_registry.py
```

`strict=True`는 등록된 unsupported placeholder의 warning도 실패로 보기
때문에, 현재 모든 target을 검사할 때의 기본 gate와 구분한다.
새 Runtime·Compiler·Collector를 등록했다면 해당 entry가 생성되거나
`unsupported_reason`대로 거부되는지, 다른 target import가 벤더 SDK 부재로
깨지지 않는지도 확인한다.

## 단위 계약 확인

변경한 계층의 실제 입력·출력을 fixture로 검사한다.

- **Loader/전처리:** 한 sample과 마지막 sample, `load_batch()` 소진,
  `load_by_index()` 순서, 입력 shape·dtype·layout·값 범위, label과
  `preprocess_context`, cache 조건 변경 시 다른 결과.
- **Runtime/Compiler:** 잘못된 artifact·SDK·장치에서 명시적 실패,
  `load → warmup → run → unload` 정상 경로, 실제 출력 이름·shape,
  buffer 재사용 여부, compile 결과 파일과 metadata.
- **Decoder/Evaluator:** 실제 artifact 출력과 같은 형태의 fixture,
  정상·빈 detection, 좌표·class ID 범위, 여러 batch의 `add_batch()`와
  `compute()` 결과, 잘못된 출력 ABI 거부.
- **Collector:** 없는 sensor의 `None` 처리, 시작·종료 중 예외가 나도
  자원 정리, 전력 측정 지점과 raw trace 연결.
- **native async:** submit 예외, callback 성공·오류·중복·지연,
  timeout 후 물리 완료, ACK, shutdown·unload 안전성.

기존 테스트 위치는 `framework/tests/test_plugin_registry.py`,
`test_runtime_executor.py`, `test_native_async_runtime_executor.py`,
`test_async_two_pass_cli.py`와 해당 Loader·Decoder·Evaluator·벤더 Runtime
테스트다. 변경한 기능에 맞는 테스트를 선택한다. 장치가 없는 환경에서
mock 테스트만 통과한 경우 실장비 확인으로 표기하지 않는다.

## 소량 CLI와 결과

먼저 `python src/main.py --help`가 같은 Python에서 열리는지 확인한다.
모델·데이터가 준비된 CPU 또는 mock 경로에서 `--warmup 0 --max-steps 1`
소량 실행을 수행하고 종료 코드, `RUN_ID`, CSV 행을 대조한다.
자동 준비 경로는 네트워크·인증·다운로드가 필요할 수 있다.

새 NPU에서는 모델 실행 명세의 **그 장치용 artifact와 입력 계약**으로 소량
동기 실행을 한다. CLI의 `--target`, `--artifact`/`--hef`, `--dataset`,
`--results-path`를 명시하고 로그에서 실제 장치·Runtime·artifact를 확인한다.
`RUN_ID`가 있어도 출력 품질이 맞는다는 뜻은 아니다.
[사용자 결과 확인 절차](../user-guide/04-results.md)로 해당 CSV 행과
`details_path`·`request_trace_path`를 연다.

비동기는 최소한 다음을 함께 본다.

1. `async_metric_schema_version=2`, 측정 경계와 사용한 pass·scenario.
2. `async_run_status`, `quality_status`, `comparison_eligible`.
3. 품질 패스 평가 sample 수와 성능 패스 Runtime 완료 sample 수,
   실패·거부·timeout·미완료 요청 수.
4. native executor의 callback·inflight 진단과 Runtime unload 완료.
5. producer load, queue wait, worker utilization, 실제 완료 처리량.

`quality_status=passed`는 지표 계산 완료 상태다. 품질 수치·정답 cohort·
실행 artifact가 기대한 조건을 만족하는지 별도로 확인한다.
`runtime-only` 결과는 `quality_status=skipped`가 정상이며 품질 근거가
없는 성능 진단으로 취급한다. 처리량과 latency의 정확한 분모는
[측정 계약](../../framework/docs/measurement-contract.md)을 따른다.

## 웹과 문서 갱신

CLI 옵션을 웹에도 제공하려면
[요청 schema](../../backend/app/schemas/benchmark.py),
[명령 조립](../../backend/app/services/benchmark_service.py),
[프론트엔드 API type](../../frontend/src/types.ts)과 실행 form을 함께
수정·확인한다. 현재 웹과 CLI의 옵션 범위는 같지 않다. 결과 필드를 바꾸면
[결과 API 변환](../../backend/app/services/result_service.py)과 UI 표시를
확인한다. 웹 연동을 변경했다면 backend API 테스트와 frontend build를
실행한다.

새 조합의 [모델별 실행 명세](../../framework/docs/model-execution-specs/resnet50.md)에
실제 artifact·입출력·전후처리·실행 명령을 적고, 결과값과 실행 증거는
별도 결과·보고서에 연결한다. 코드 revision, Runtime/SDK 버전, sample
cohort, 측정 구간, 실패·미완료 상태를 혼동하지 않는다.
