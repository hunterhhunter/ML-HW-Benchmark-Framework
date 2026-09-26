# 프레임워크 공통 원시 전력 추적 설계

## 목적

공통 동기·비동기 추론 실행기를 사용하는 모든 벤치마크 모델에 선택형
원시 전력 측정 기능을 추가한다. 첫 지원 대상은 Furiosa RNGD, Rebellions
RBLN, Mobilint ARIES다.

이 기능은 고정된 유휴 기준 구간과 실제 추론 측정 구간에서 순간 전력을
W 단위로 기록한다. 에너지, 유휴 전력 차감값, `J/inference`는 계산하지
않는다. 이러한 계산은 추후 별도 분석 단계에서 수행한다.

TTM-R2는 세 대상 모두에서 240개 윈도 실행 이력이 있으므로 첫 물리 장치
인수시험에 사용한다. 다만 전력 추적 구성요소는 TTM-R2 모듈을 가져오거나,
표본 수를 240개로 가정하거나, 모델 이름에 따라 분기해서는 안 된다.

## 범위

구현 범위는 다음과 같다.

- 명시적인 `--power-trace` CLI 플래그를 추가한다.
- `--monitor` 없이 단독으로 동작하고, 함께 사용해도 정상 동작한다.
- `e2e`와 `async_queue`의 실제 측정 추론 구간을 모두 지원한다.
- 워밍업 이후 3초 동안 유휴 기준 전력을 기록한다.
- 실제 추론 측정 구간 전체에서 원시 순간 전력을 기록한다.
- 실행마다 하나의 CSV 사이드카를 원자적으로 게시한다.
- 일반 결과 행과 전력 CSV를 실행 메타데이터로 연결한다.
- 전력 telemetry가 없거나 실패해도 추론 및 품질 결과는 보존한다.
- 벤더 collector에 중복 구현된 에너지 적분을 제거한다.

다음 항목은 구현하지 않는다.

- 총에너지 또는 그 밖의 J 단위 값 계산
- 유휴 전력 차감
- `J/inference` 계산
- 에너지 분석 명령이나 분석 CSV 생성
- 누락 전력을 0으로 채우거나 전압으로부터 전력을 추정하는 처리
- 범위가 다른 벤더 telemetry를 물리적으로 동일하다고 간주하는 처리
- 사람이 읽는 `furiosa-smi info` 표를 운영 환경의 데이터 원본으로 파싱하는
  처리
- `furiosa-smi info --format json` subprocess를 Python API의 묵시적 fallback으로
  사용하는 처리

첫 물리 장치 지원 범위는 Furiosa RNGD, Rebellions, Mobilint ARIES다. 공통
collector 인터페이스는 나중에 다른 대상을 추가할 수 있도록 설계하지만,
이번 변경에서는 Hailo, DeepX, Mobilint REGULUS, NVIDIA 등은 새 원시 전력
추적 계약에 포함하지 않는다.

## 기존 구조와 리팩터링 범위

현재 `HWMonitor`는 collector 생명주기와 백그라운드 폴링을 담당한다.
`BenchmarkRunner`는 워밍업 이후 모니터를 시작하고 동기 측정 루프가 끝난
뒤 정지한다. 비동기 실행기에도 이에 대응하는 start, stop, summary 보호
콜백이 있다. 각 target은 `monitor_names`와 `monitor_options`를 통해 벤더
collector를 선택한다.

현재 Rebellions와 Mobilint collector는 각자 전력을 에너지로 적분한다.
원시 전력 추적에서는 이 책임이 collector에 있으면 안 되며, 벤더마다 같은
수치 계산이 중복된다. 따라서 다음 collector 책임을 제거한다.

- 적분 목적으로만 보관하던 이전 전력 및 이전 시각 상태
- 사다리꼴 에너지 적분
- collector가 생성하던 `hw_accel_energy_j`
- 에너지 요약을 완성하기 위해서만 수행하던 종료 경계 전력 조회

collector는 장치 선택, SDK 생명주기, 원시 값 검증, 데이터 원본 메타데이터,
순간 telemetry를 계속 담당한다. `HWMonitor`는 생명주기의 유일한 소유자로
남고 선택형 원시 전력 추적 기능을 추가로 담당한다.

기존 `--monitor`의 CPU, RAM, 사용률, 온도 및 순간 전력 평균 요약은
유지한다. `--power-trace`만 사용한 실행에서는 평균·최대 전력을 일반 결과
행에 추가하지 않는다. 에너지 요약 필드는 모든 모드에서 제거한다.

## 사용자 CLI 계약

추가하는 플래그는 다음과 같다.

```text
--power-trace
```

플래그 조합별 동작은 다음과 같다.

| `--monitor` | `--power-trace` | 동작 |
|---|---|---|
| 없음 | 없음 | 기존 모니터링 없는 벤치마크 동작 |
| 있음 | 없음 | 에너지 필드를 제외한 기존 하드웨어 요약 |
| 없음 | 있음 | 원시 전력 CSV만 저장하고 하드웨어 요약은 병합하지 않음 |
| 있음 | 있음 | 기존 하드웨어 요약과 원시 전력 CSV를 함께 저장 |

초기 계약에서 유휴 기준 구간은 3초로 고정하며 별도 CLI 옵션으로 노출하지
않는다. 전력 측정 주기는 모델 profile이 아니라 target 및 collector 설정이
소유한다.

`--monitor-interval`은 기존 일반 모니터링 주기만 제어한다. 이 옵션이 전력
측정 주기를 암묵적으로 변경해서는 안 된다.

## 수집기 전력 계약

모니터 collector 추상화에 선택형 순간 전력 기능을 추가한다. 전력을 지원하는
collector는 다음 필드를 갖는 구조화된 값을 반환한다.

- `power_w`: 유한한 0 이상의 float. 실패한 경우 값 없음
- `status`: `ok`, `unavailable`, `read_error` 중 하나
- `source`: 안정적인 SDK 또는 도구 식별자
- `power_scope`: 해당 데이터 원본이 문서화한 측정 범위
- `device_id`: 실제로 선택한 가속기 식별자
- `error_code`: 값이 없을 때 사용하는 길이가 제한된 정제 진단값

쿼리 시작 및 종료 시각은 collector가 아니라 `HWMonitor`가 측정한다. 따라서
벤더와 무관하게 timestamp와 쿼리 지연시간의 의미가 동일하다.

기존 `collect()`는 일반 하드웨어 요약을 위해 유지한다. 전력 지원 collector는
좁은 범위의 `collect_power()`를 추가한다. 고주기 전력 측정 중 온도, 메모리,
사용률, 전류, 전압을 매번 함께 조회해서는 안 된다. collector는 같은 SDK
세션을 사용하는 일반 조회와 전력 조회를 직렬화해야 하며, 프레임워크는 하나의
벤더 collector에 동시 SDK 호출을 보내서는 안 된다.

전력 기능이 없다는 사실은 추론 예외가 아니라 측정 결과다. 사용할 수 있는
직접 전력 원본이 없는 target은 값을 만들지 않고
`power_trace_status=unavailable`을 기록한다.

## 벤더별 구현

### Mobilint ARIES

기존 `mbltml` 장치 세션과 `mbltmlGetTotalPower(device_id)`를 사용한다. 원본
식별자는 `mbltml`, 범위는 `device_total`로 기록한다. API가 반환하는 총전력
범위를 process 전력이나 NPU-core 전력으로 바꿔 표현하지 않는다.

`collect_power()`는 총전력 API만 호출한다. 기존 전류 및 전압 수집은 일반
`--monitor` 동작에만 남기며 W 값을 유도하는 데 사용하지 않는다. Mobilint
REGULUS는 해당 플랫폼에서 전력 API를 검증하기 전까지 원시 전력 추적을
지원하지 않는다.

2026-09-26 서버 조사에서 50ms 간격으로 100회 조회했을 때 호출 지연은
p99 0.085284ms, 최대 0.39403ms였지만 값은 약 1,000ms마다 한 번만 변경됐다.
기존 `HWMonitor`의 200ms 주기와 추론 경계 기록을 맞추기 위해 기본 측정 주기는
200ms로 둔다. 센서가 같은 값을 반환하더라도 중복 표본을 제거하지 않는다.
이 주기는 센서 갱신을 강제한다는 뜻이 아니며, 1초보다 짧은 추론에서는 시작과
종료 표본의 값이 같을 수 있다.

### Rebellions

현재 검증된 원본은 exact argv `rbln-smi -b -j -d <device_id>`의 JSON
출력이다. 기존 엄격한 파서로 `card_power`를 읽는다. 원본 식별자는
`rbln-smi-json`, 범위는 `whole_card`다.

collector의 에너지 상태는 제거하고 장치 및 JSON schema 검증은 유지한다.
2026-09-26 서버 조사에서 CA22의 `card_power`는 `uW` 문자열이었고 100회
1초 표본이 모두 성공했다. 호출 지연은 p99 4.567737ms, 최대 4.590526ms였으며
90회 값 변경이 관측됐다. 기본 측정 주기는 기존 `HWMonitor`와 같은 200ms로
시작한다. 이때 p99 기준 조회 duty는 약 2.3%이므로 서버 게이트에서 trace-off와
trace-on의 측정 루프 지연을 비교한다. 중앙 지연 증가가 2%를 넘으면 다른
collector는 그대로 두고 Rebellions 주기만 250ms 또는 500ms로 올린다.

### Furiosa RNGD

`FuriosaCollector`를 추가하고 두 Furiosa RNGD target에 등록한다. 운영용
collector는 설치된 공식 Python SMI binding을 사용하며 CLI 표 또는 JSON을
파싱하지 않는다.

2026-09-26 서버 조사에서 `furiosa-smi-py 2026.1.2`의
`init() -> list_devices() -> Device.power_consumption()` 경로가 확인됐다.
`Device.device_info().name()`은 `npu0`, 전력 callable은 W 단위 float를 반환했다.
50ms 간격 100회 조회가 모두 성공했고 호출 지연은 p99 1.557525ms, 최대
1.557729ms였다. 기본 측정 주기는 기존 `HWMonitor`와 같은 200ms로 둔다.
유휴 5초 동안 같은 값이 반복됐더라도 중복 표본은 원시 관측으로 보존한다.
원본 식별자는 `furiosa-smi-py`,
`power_scope`는 API docstring의 표현을 넘어서지 않는 `device`로 기록한다.

검증된 TTM-R2 실행 환경의 `furiosa-torch 2026.3.0` 가상환경에는 이 패키지가
설치되어 있지 않았다. 운영 가상환경에서 호환되는 Python SMI 패키지가 없으면
CLI로 대체하거나 다른 가상환경의 site-packages를 주입하지 않고
`unavailable`로 처리한다. `furiosa-smi info --format json`은 p50 약 92.28ms,
p95 약 96.52ms였으므로 추론 측정 중 polling 원본으로 사용하지 않는다.
Furiosa Torch requirements에는 서버에서 직접 검증한
`furiosa-smi-py==2026.1.2`를 명시하고, `furiosa-torch 2026.3.0`과의
`pip check`, 직접 API 100회 및 TTM-R2 부하 반응이 성공해야 이 pin을 확정한다.
패키지 부재를 `unavailable`로 기록하는 것은 올바른 실패 처리지만 세 target
지원 완료 조건을 충족한 것으로 보지 않는다.

## 측정 스케줄러

`HWMonitor`는 각 벤더 collector 세션의 스케줄러를 소유한다. 일반 모니터링과
전력 추적을 함께 사용하면 같은 SDK 세션에 두 스레드가 동시에 진입하지 않도록
실행 시점을 직렬화한다. 일반 조회와 전력 조회가 동시에 예정되면 전력 조회를
먼저 수행한다.

스케줄은 다음과 같이 monotonic deadline을 누적한다.

```text
next_deadline = previous_deadline + configured_period
```

각 쿼리 뒤에 전체 주기만큼 다시 sleep하면 쿼리 시간이 간격에 더해져 drift가
생기므로 이 방식을 사용하지 않는다. 쿼리가 하나 이상의 deadline을 넘기면
밀린 호출을 연속으로 실행하지 않는다. 대신 `overrun` 시도를 기록하고 다음
미래 deadline으로 이동한다.

각 조회 시도에는 예정 경과 시간을 기록한다. `HWMonitor`는 monotonic 쿼리
시작·종료 시각을 기록하고 두 시각의 중간값을 관측 시각으로 사용한다. 이를
통해 벤치마크 프로세스가 분석을 수행하지 않아도 실제 jitter를 보존한다.

10ms는 공통 기본값으로 사용하지 않는다. 최초 기본값은 세 target 모두 기존
`HWMonitor`와 같은 200ms다. 센서의 native 갱신 주기가 이보다 느릴 수 있으므로
같은 W가 반복된 표본도 원본 그대로 기록하며 deduplication하지 않는다.
trace-off와 trace-on을 번갈아 다섯 쌍 실행했을 때 trace-on 측정 루프 중앙
지연시간이 trace-off 대비 2% 이내여야 하고 telemetry 때문에 실패한 실행이
없어야 한다. 이를 넘으면 실측 비용이 가장 큰 Rebellions에 한해 250ms 또는
500ms로 조정한다. 이 검사는 설정 보정을 위한 것이며 매 벤치마크 실행 전에
자동 수행하지 않는다.

## 측정 생명주기

### 동기 `e2e`

1. runtime을 load하고 기존 load-time monitor 상태가 있으면 기록한다.
2. 전력 추적을 시작하지 않은 상태에서 지정된 워밍업을 수행한다.
3. collector와 trace writer를 `baseline` phase로 시작하고 기준 시작 표본을
   즉시 한 번 강제한다.
4. 3초 유휴 기준 전력을 200ms 주기로 기록한다.
5. `InferenceEngine.run_e2e()` 직전에 `inference` phase로 전환한 뒤 추론 시작
   표본을 즉시 한 번 강제하고 writer를 flush한다.
6. 기존 측정 루프 전체에서 전력 추적을 유지한다.
7. 루프가 반환하거나 예외를 발생시킨 직후 추론 종료 표본을 즉시 한 번
   강제한다.
8. 원래 추론 예외를 대체하지 않으면서 collector를 정리하고 trace를 종료한다.

### 비동기 `async_queue`

동일한 phase를 사용한다. 단, 비동기 워밍업이 끝난 뒤 측정 producer가 첫
logical request를 제출하도록 허용하기 직전에 `inference`로 전환한다. 제출이
끝났다는 이유만으로 추적을 종료하지 않는다. 기존 terminal completion 및
outstanding-request 종료 경계까지 모든 승인 작업이 완료된 뒤 종료한다.

trace는 가속기 호출 사이의 구간을 포함한 기존 측정 루프를 대상으로 한다.
따라서 순수 NPU kernel trace가 아니라 프레임워크 추론 측정 구간의 장치 전력
trace다.

추론이 200ms보다 짧아도 강제한 시작·종료 조회로 `inference` 표본 시도를 최소
두 번 남긴다. 다만 센서의 native 갱신이 추론보다 느리면 두 W 값이 같을 수
있으며, 이는 실패나 누락이 아니다. 이 기능은 workload 반복이나 최소 실행
시간을 추가하지 않고 실제 실행 구간에서 반환된 원시 전력만 기록한다.

## 원시 CSV 계약

성공 여부와 관계없이 예정된 전력 조회 시도 하나를 행 하나로 기록한다. schema는
다음과 같다.

```csv
schema_version,run_id,sample_index,phase,target_id,collector,monitor_source,device_id,power_scope,scheduled_elapsed_ms,observed_elapsed_ms,query_latency_ms,power_w,sample_status,error_code
```

필드 규칙은 다음과 같다.

- `schema_version`은 `1.0`이다.
- `run_id`는 미리 예약된 프레임워크 실행 식별자다.
- `sample_index`는 0부터 시작하며 조회 시도마다 1씩 증가한다.
- `phase`는 `baseline` 또는 `inference`다.
- `scheduled_elapsed_ms`는 trace 시작점 기준 예정 경과 시간이다.
- `observed_elapsed_ms`는 trace 시작점 기준 쿼리 중간 시각이다.
- `query_latency_ms`는 쿼리 종료 시각에서 시작 시각을 뺀 값이다.
- `power_w`는 `ok`일 때의 가공하지 않은 유한 W 값이며 그 외에는 비운다.
- `sample_status`는 `ok`, `unavailable`, `read_error`, `overrun` 중 하나다.
- `error_code`는 `ok`일 때 비우며, 그 외에는 제한된 정제 진단값을 기록한다.
  제한되지 않은 예외 문자열이나 SDK 출력 전체를 넣어서는 안 된다.

CSV에는 에너지, 평균값, 유휴 차감, 추론 횟수, 품질 지표를 기록하지 않는다.

## 실행 예약과 원자적 게시

기존 run artifact reservation을 일반화해 `--power-trace`를 사용한 `e2e`도
측정 전에 `run_id`를 예약할 수 있게 한다. 예약 경로는 다음과 같다.

```text
<results-root>/power/<run_id>.power.csv
```

writer는 같은 디렉터리에 숨김 임시 파일을 배타적으로 생성한다. header를 즉시
기록하고 측정 중에는 행을 buffering하며 phase 전환 시 flush한다. 종료 시에는
flush, fsync, close를 수행한 뒤 저장소의 기존 no-overwrite artifact 규칙으로
최종 파일명을 게시한다. 게시된 파일을 다시 읽어 SHA-256을 계산한다.

부분 기록 파일이 최종 경로에 나타나서는 안 된다. 실행이 중단되거나 게시가
실패하면 복구 가능한 부분 artifact를 `.partial.csv` 진단 이름으로 남길 수
있지만 complete trace로 보고하지 않는다. 전력 파일 게시 실패가 이미 게시된
artifact를 삭제하거나 덮어써서는 안 된다.

`--power-trace`를 사용하지 않은 일반 `e2e` 실행은 기존 결과 저장 및 run ID
동작을 유지한다. async reservation도 기존 details 및 request trace artifact와
호환되어야 한다.

## 결과 연결과 상태

일반 벤치마크 결과에는 다음 provenance만 저장한다.

- `power_trace_status`
- `power_trace_path`
- `power_trace_sha256`
- `power_trace_sample_count`
- `power_monitor_source`
- `power_scope`

`--power-trace`를 대신해 평균 전력이나 에너지 값을 일반 결과에 저장하지 않는다.
경로는 results root 기준 상대 경로이며 같은 `run_id`로 예약된 artifact를
가리켜야 한다. `power_trace_sample_count`는 실패한 시도를 포함한 게시 CSV의
data row 수다.

`disabled`, `unavailable`, `failed`일 때는 완성된 최종 artifact가 없으므로
최종 trace 경로, SHA-256, 표본 수를 비운다. 진단용 `.partial.csv`는
`power_trace_path`에 넣지 않는다.

수집 상태는 생명주기만 나타낸다.

| 상태 | 의미 |
|---|---|
| `disabled` | `--power-trace`를 요청하지 않음 |
| `complete` | 최종 CSV를 게시했고 모든 data row가 `sample_status=ok`임 |
| `partial` | 최종 CSV를 게시했지만 하나 이상의 조회 실패 또는 overrun이 있음 |
| `unavailable` | 선택 target에 검증된 전력 기능이 없음 |
| `failed` | trace 시작, 기록, 정리 또는 게시가 실패함 |

이 상태는 향후 에너지 분석의 통계적 유효성을 판정하지 않는다.

## 실패 및 정리 규약

전력 telemetry는 추론 정확성을 보조하는 정보다. telemetry 실패 때문에 성공한
추론이나 품질 결과를 버려서는 안 된다. `HWMonitor`는 일반 전력 조회 예외를
제한된 진단 행으로 변환하고 측정을 계속한다. writer 또는 collector 생명주기
실패는 trace 상태를 `failed`로 바꾸되 runner는 본래 벤치마크 결과를 보존한다.

추론 자체가 예외를 발생시켜도 trace stop과 collector 정리를 수행한다. 이때
발생한 보조 trace 예외는 진단으로 첨부하고 원래 추론 예외를 대체하지 않는다.
정리 소유권이 불확실하면 기존 transactional monitor 규칙에 따라 명시적으로
재시도할 수 있게 소유권을 유지하며 세션을 묵시적으로 버리지 않는다.

`--monitor`와 `--power-trace`를 함께 사용할 때 전력 이외의 기존 monitor 실패는
기존 동작을 유지한다. 새로운 비치명적 결과 보존 정책은 전력 추적 경로에만
적용한다.

## 검증 전략

### SDK 없는 테스트

- collector 전력 기능 기본값과 unavailable 동작
- 단위를 추측하지 않는 유한·0 이상 원시 W 검증
- fake clock을 이용한 쿼리 중간 timestamp 및 지연시간 계산
- monotonic deadline, jitter, 누락 slot 및 catch-up burst 방지
- baseline에서 inference로의 phase 전환과 경계 조회
- 기존 동기 측정 루프를 둘러싼 start/stop 위치
- 첫 비동기 측정 제출 전 start 및 terminal completion 후 stop
- `--power-trace` 단독 및 `--monitor`와의 조합
- 두 모드를 함께 사용할 때 collector 호출 직렬화
- 원시 CSV escaping, 실패 조회 행, schema 순서 및 sample index
- 원자적 no-overwrite 게시, fsync 실패, 부분 artifact, SHA-256 연결
- e2e와 async run reservation의 같은 artifact authority 사용
- collector 또는 writer 실패 이후 추론 성공 결과 보존
- Rebellions 및 Mobilint 에너지 적분과 에너지 요약 필드 제거
- monitor 모듈에 모델 이름 또는 TTM-R2 의존성이 없는지 확인

### 실제 장치 인수시험

Furiosa RNGD, Rebellions, Mobilint ARIES 각각에 대해 다음을 수행한다.

1. collector 구현 직후 전체 runner 연결을 시작하기 전에 작업 브랜치를 서버에
   반영하고 세 collector의 원시 W와 source/scope를 검증한다.
2. 도구 또는 패키지, driver, firmware, 장치 식별자, 원본, 범위를 기록한다.
3. 사전 조사에서 얻은 100회 유휴 전력 조회 결과와 생산 collector 결과가
   일치하는지 확인한다.
4. TTM-R2를 실행하는 동안 collector 값이 장치 부하에 반응하는지 확인한다.
   이 단계는 collector 게이트이며 compile/warmup/inference 경계의 최종 증거로
   사용하지 않는다.
5. 이 게이트를 통과한 뒤에만 결과 저장과 동기·비동기 runner 연결을 구현한다.
6. trace-off와 trace-on smoke를 번갈아 다섯 쌍 실행하고 trace-on 측정 루프
   중앙 지연시간이 trace-off 대비 2% 이내인지 확인한다.
7. canonical TTM-R2 240-window 벤치마크를 `--power-trace`로 실행한다.
8. 두 phase, monotonic timing, 원시 W, 최종 SHA-256, 결과 연결을 확인한다.
9. target이 지원하는 기존 비-TTM workload 하나를 실행해 전력 추적이 모델에
   종속되지 않았음을 확인한다.

실제 장치 인수시험은 수집과 provenance만 검증한다. 에너지 값을 계산하거나
승인하지 않는다.

## 문서 변경

프레임워크 CLI와 벤더 설정 문서에 다음 내용을 반영한다.

- `--power-trace`는 선택형이며 `--monitor`와 별개다.
- 워밍업은 제외하며 3초 유휴 기준 구간을 명시한다.
- CSV에는 원시 W와 쿼리 시각 정보만 저장한다.
- 벤더마다 `power_scope`가 다르며 향후 분석에서도 이를 유지해야 한다.
- 에너지와 `J/inference`는 벤치마크 출력이 아니다.
- unavailable 및 failed telemetry를 0으로 바꾸지 않는다.

`hw_accel_energy_j`를 설명하는 기존 Rebellions 및 Mobilint 문서는 해당 내용을
삭제하거나 원시 trace 계약으로 교체한다.

## 완료 조건

다음 조건을 모두 만족하면 구현이 완료된 것으로 본다.

1. SDK 없는 전력 추적 테스트가 모두 통과한다.
2. 명시적으로 제거한 에너지 필드 외의 기존 monitor 동작이 호환된다.
3. 두 추론 모드가 올바르게 연결된 원시 trace artifact를 게시한다.
4. trace 실패가 성공한 벤치마크를 추론 또는 품질 실패로 바꾸지 않는다.
5. 어떤 벤치마크 경로도 J 또는 `J/inference`를 계산하지 않는다.
6. 세 collector 구현 직후 서버 중간 게이트를 통과하고, 기본 200ms 주기와
   강제 경계 조회가 생산 collector에서도 유효함을 확인한다. Rebellions는
   trace-on 지연 증가가 2%를 넘을 때만 250ms 또는 500ms로 조정한다.
7. 사용 가능한 각 target에서 TTM-R2와 비-TTM workload 하나가 물리적으로
   수집한 원시 전력 trace를 생성한다.
