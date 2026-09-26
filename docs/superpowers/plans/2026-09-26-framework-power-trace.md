# 프레임워크 공통 원시 전력 추적 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**목표:** `--power-trace`를 지정한 모든 공통 추론 경로에서 Furiosa RNGD, Rebellions, Mobilint ARIES의 원시 순간 전력 W를 3초 유휴 기준 구간과 실제 추론 구간으로 나누어 안전한 CSV 사이드카에 기록한다.

**구조:** 기존 `HWMonitor`가 collector 생명주기, 전력 측정 deadline, phase 전환을 소유한다. 벤더 collector는 순간 전력만 반환하고, 새 `core.power_trace` 모듈이 공통 자료형과 예약된 CSV artifact 기록을 담당한다. 일반 결과 CSV에는 전력 수치가 아니라 trace 상태·경로·SHA-256·행 수·원본·측정 범위만 연결한다.

**기술 스택:** Python 3.10+, `threading`, `time.monotonic_ns`, 표준 `csv`/`hashlib`/`os`, 기존 POSIX artifact reservation, `pytest`와 fake SDK/clock

**설계 문서:** `docs/superpowers/specs/2026-09-26-framework-power-trace-design.md`

**작업 브랜치:** `feat/power_measurment`

**변경 전 기준선:** 기존 collector 집중 테스트
`test_hw_monitor.py`, `test_rbln_collector.py`, `test_mobilint_collector.py`는
`109 passed`다.

## 전역 제약

- 기능은 `--power-trace`를 명시한 실행에서만 활성화한다.
- 워밍업은 측정에서 제외하고, `HWMonitor.start()`가 반환하기 전에 3.0초 `baseline`을 기록한 뒤 `inference` phase로 전환한다.
- `--power-trace`는 `--monitor` 없이 동작해야 하며 두 플래그를 함께 사용해도 같은 벤더 SDK 세션을 동시에 호출하지 않는다.
- CSV에는 원시 W와 조회 시각·지연·상태만 저장한다. J, 에너지, 유휴 차감, `J/inference`, 분석 CSV를 구현하지 않는다.
- 누락 전력을 0으로 채우거나 전압으로부터 추정하지 않는다.
- 전력 수집 실패는 성공한 추론 및 품질 결과를 실패로 바꾸지 않는다.
- 최초 측정 주기는 기존 `HWMonitor`와 맞춘 Furiosa, ARIES, Rebellions 모두 200ms다. 10ms 또는 50ms 공통 주기는 사용하지 않는다. 시작·종료 경계에서는 주기와 무관하게 즉시 조회하고, 센서가 같은 값을 반환한 중복 표본도 제거하지 않는다.
- Furiosa 운영 수집기는 CLI 표/JSON을 파싱하지 않고 공식 `furiosa_smi_py`의 `init()`, `list_devices()`, `device.power_consumption()` 계약만 사용한다. 실행 가상환경에 패키지가 없거나 계약이 다르면 추측하지 않고 `unavailable`로 처리한다.
- Rebellions는 exact argv `rbln-smi -b -j -d <device_id>`와 `card_power`를 유지한다.
- ARIES는 `mbltmlGetTotalPower(device_id)`만으로 W를 읽는다.
- 벤더별 `power_scope`를 보존하며 서로 동일한 물리 경계라고 주장하지 않는다.
- 기존 Rebellions/Mobilint 에너지 적분 상태와 `hw_accel_energy_j`는 제거한다.
- SDK가 없는 로컬에서도 registry import와 전체 SDK-free 테스트가 동작해야 한다.
- Task 5의 collector 커밋과 서버 중간 게이트가 통과하기 전에는 Task 6 이후의 결과 저장 및 runner 연결을 시작하지 않는다.

## 검토 중점

- `--monitor --power-trace`에서 같은 collector의 일반 조회와 전력 조회가 겹쳐도 SDK 호출은 직렬화되고 표본은 사라지지 않아야 한다. Task 3과 Task 4의 동시성·캐시 테스트로 고정한다.
- writer가 중간에 실패해도 측정 추론과 품질 결과는 저장되고 `power_trace_status=failed`만 남아야 한다. Task 2와 Task 7의 실패 주입 테스트로 고정한다.
- async producer의 제출이 끝났지만 device 작업이 남아 있는 경우 마지막 물리 completion 전에는 전력 추적을 정지하지 않아야 한다. Task 8의 flush/outstanding 순서 테스트로 고정한다.
- Furiosa 패키지 또는 전력 API가 없을 때 CLI 표 파싱이나 0 대체 없이 `unavailable`이 되어야 한다. Task 5의 fake module 테스트로 고정한다.
- 쿼리가 주기를 초과할 때 밀린 호출을 몰아서 실행하지 않고 각 누락 deadline을 `overrun` 행으로 보존해야 한다. Task 3의 fake clock 테스트로 고정한다.

---

### Task 1: 공통 전력 자료형과 collector 계약

**Files:**
- Create: `framework/src/core/power_trace.py`
- Modify: `framework/src/monitors/base.py:15-40`
- Create: `framework/tests/test_power_trace.py`
- Modify: `framework/tests/test_hw_monitor.py`

**Interfaces:**
- Consumes: 기존 `Collector` ABC와 `RunArtifactReservation` 타입 이름
- Produces: `PowerTraceSource`, `PowerReading`, `PowerTraceSample`, `PowerTraceArtifact`, `PowerTraceConfig`, `Collector.power_trace_source()`, `Collector.collect_power()`

- [ ] **Step 1: 상태와 원시 값 검증 실패 테스트 작성**

`framework/tests/test_power_trace.py`에 다음 테스트를 작성한다.

```python
def test_ok_power_reading_requires_finite_non_negative_watts(): ...
def test_failed_power_reading_requires_empty_power_and_bounded_error_code(): ...
def test_power_trace_artifact_exposes_only_linkage_metadata(): ...
```

`ok`에 `None`, 음수, `nan`, `inf`를 주면 `ValueError`, 실패 상태에 W를 주면
`ValueError`, artifact metadata에는 여섯 개 `power_*` 연결 필드만 존재한다고
검증한다.

- [ ] **Step 2: 실패 확인**

Run:

```bash
/home/swlab-youngjin/ML-HW-Benchmark-Framework/.venv-mobilint-compile/bin/python -m pytest framework/tests/test_power_trace.py -q
```

Expected: `ModuleNotFoundError: No module named 'core.power_trace'`

- [ ] **Step 3: 공통 타입 정의**

`framework/src/core/power_trace.py`에 다음 정확한 타입을 정의한다.

```python
PowerSampleStatus = Literal["ok", "unavailable", "read_error", "overrun"]
PowerTraceStatus = Literal["disabled", "complete", "partial", "unavailable", "failed"]
PowerTracePhase = Literal["baseline", "inference"]

@dataclass(frozen=True)
class PowerTraceSource:
    collector: str
    monitor_source: str
    device_id: str
    power_scope: str
    sample_interval_sec: float

@dataclass(frozen=True)
class PowerReading:
    status: PowerSampleStatus
    power_w: float | None = None
    error_code: str = ""

@dataclass(frozen=True)
class PowerTraceConfig:
    run_id: str
    target_id: str
    baseline_duration_sec: float = 3.0

@dataclass(frozen=True)
class PowerTraceSample:
    phase: PowerTracePhase
    scheduled_elapsed_ms: float
    observed_elapsed_ms: float
    query_latency_ms: float
    reading: PowerReading

@dataclass(frozen=True)
class PowerTraceArtifact:
    status: PowerTraceStatus
    path: str = ""
    sha256: str = ""
    sample_count: int | None = None
    monitor_source: str = ""
    power_scope: str = ""

    def as_result_metadata(self) -> dict[str, object]: ...
```

`error_code`는 ASCII 영숫자·underscore·hyphen·colon만 허용하고 최대 128자로
제한한다. 이 모듈에는 에너지 관련 필드나 계산 함수를 추가하지 않는다.

- [ ] **Step 4: Collector의 선택형 전력 API 추가**

`Collector`에 기본 구현을 추가한다.

```python
def power_trace_source(self) -> PowerTraceSource | None:
    return None

def collect_power(self) -> PowerReading:
    raise RuntimeError("collector has no power trace source")
```

기존 collector는 수정하지 않아도 unavailable 대상이 되도록 한다.

- [ ] **Step 5: 단위 테스트 통과 확인**

Run:

```bash
/home/swlab-youngjin/ML-HW-Benchmark-Framework/.venv-mobilint-compile/bin/python -m pytest framework/tests/test_power_trace.py framework/tests/test_hw_monitor.py -q
```

Expected: PASS

- [ ] **Step 6: 커밋**

```bash
git add framework/src/core/power_trace.py framework/src/monitors/base.py framework/tests/test_power_trace.py framework/tests/test_hw_monitor.py
git commit -m "feat: define raw power trace contract"
```

### Task 2: 예약된 원시 CSV writer와 원자적 게시

**Files:**
- Modify: `framework/src/core/power_trace.py`
- Modify: `framework/src/core/artifact_reservation.py:120-174`
- Modify: `framework/src/core/result_store.py:202-268`
- Modify: `framework/tests/test_power_trace.py`
- Modify: `framework/tests/test_async_result_artifacts.py`

**Interfaces:**
- Consumes: Task 1의 `PowerTraceConfig`, `PowerTraceSource`, `PowerTraceSample`, `PowerTraceArtifact`; 기존 `RunArtifactReservation`
- Produces: `RunArtifactReservation.power_trace_path`, `PowerTraceWriter.start()`, `write_attempt()`, `flush()`, `finish()`, `fail()`

- [ ] **Step 1: CSV schema와 원자적 게시 실패 테스트 작성**

다음 테스트를 추가한다.

```python
def test_reservation_exposes_run_bound_power_trace_path(tmp_path): ...
def test_writer_publishes_exact_raw_schema_and_sha256(tmp_path): ...
def test_writer_records_failed_and_overrun_attempts_with_empty_power(tmp_path): ...
def test_writer_never_overwrites_existing_final_trace(tmp_path): ...
def test_writer_failure_leaves_no_final_artifact(tmp_path, monkeypatch): ...
```

최종 경로가 `<results-root>/power/<run_id>.power.csv`인지, header 순서가 설계와
일치하는지, `sample_index`가 모든 시도에 대해 0부터 증가하는지, `finish()`가
실제 파일 SHA-256과 data row 수를 반환하는지 검증한다.

- [ ] **Step 2: 실패 확인**

Run:

```bash
/home/swlab-youngjin/ML-HW-Benchmark-Framework/.venv-mobilint-compile/bin/python -m pytest framework/tests/test_power_trace.py framework/tests/test_async_result_artifacts.py -q
```

Expected: missing `power_trace_path` 또는 `PowerTraceWriter`로 FAIL

- [ ] **Step 3: reservation 경로 추가**

`RunArtifactReservation`에 다음 property를 추가하고 class docstring에서
async 전용 표현을 제거한다.

```python
@property
def power_trace_path(self) -> Path:
    return self.results_root / "power" / f"{self.run_id}.power.csv"
```

- [ ] **Step 4: PowerTraceWriter 구현**

`core.power_trace.PowerTraceWriter`의 public API를 다음으로 고정한다.

```python
class PowerTraceWriter:
    def __init__(
        self,
        reservation: RunArtifactReservation,
        config: PowerTraceConfig,
        source: PowerTraceSource,
    ): ...
    def start(self) -> None: ...
    def write_attempt(self, sample: PowerTraceSample) -> None: ...
    def flush(self) -> None: ...
    def finish(self) -> PowerTraceArtifact: ...
    def fail(self) -> PowerTraceArtifact: ...
```

표준 `csv.DictWriter`를 사용하고, 같은 `power/` 디렉터리의 배타적 숨김 임시
파일에 streaming write한다. phase 전환과 종료에 flush하고 종료 시 file 및
directory를 fsync한 뒤 기존 `link_no_overwrite`/reservation 검증 규칙으로
게시한다. `finish()` 후 published bytes를 읽어 SHA-256을 계산한다. `fail()`은
최종 경로를 만들지 않고 가능한 경우 `.partial.csv` 진단 파일을 보존한다.

- [ ] **Step 5: writer 테스트 통과 확인**

Run:

```bash
/home/swlab-youngjin/ML-HW-Benchmark-Framework/.venv-mobilint-compile/bin/python -m pytest framework/tests/test_power_trace.py framework/tests/test_async_result_artifacts.py -q
```

Expected: PASS

- [ ] **Step 6: 커밋**

```bash
git add framework/src/core/power_trace.py framework/src/core/artifact_reservation.py framework/src/core/result_store.py framework/tests/test_power_trace.py framework/tests/test_async_result_artifacts.py
git commit -m "feat: publish reserved raw power traces"
```

### Task 3: HWMonitor 전력 스케줄러와 3초 baseline

**Files:**
- Modify: `framework/src/monitors/base.py:42-285`
- Modify: `framework/src/monitors/__init__.py:45-105`
- Modify: `framework/tests/test_hw_monitor.py`

**Interfaces:**
- Consumes: Task 1·2의 전력 타입과 `PowerTraceWriter`
- Produces: 확장된 `HWMonitor.__init__()`, `configure_power_trace()`, `power_trace_metadata()`, `startup_timeout_hint_sec()`

- [ ] **Step 1: scheduler와 phase 테스트 작성**

다음 테스트를 fake clock, fake waiter, fake power collector로 작성한다.

```python
def test_power_trace_start_records_three_second_baseline_before_return(): ...
def test_short_inference_records_forced_start_and_stop_samples(): ...
def test_power_scheduler_uses_absolute_deadlines_without_drift(): ...
def test_power_scheduler_records_overrun_without_catchup_burst(): ...
def test_power_and_monitor_calls_are_serialized_per_collector(): ...
def test_power_stop_records_final_inference_boundary_and_publishes(): ...
def test_trace_only_monitor_returns_no_summary_metrics(): ...
def test_missing_power_collector_returns_unavailable_metadata(): ...
```

baseline 동안 `phase=baseline`, `start()` 반환 이후 `phase=inference`, stop 경계
행도 `inference`인지 검증한다. 200ms보다 짧은 추론에도 강제 시작·종료 조회가
최소 두 번 남고 같은 W가 중복돼도 그대로 보존하는지 검증한다. 50ms 주기
쿼리가 130ms 걸리는 경우 밀린 두
호출을 연속 수행하지 않고 `overrun` 행을 남기는지 검증한다.

- [ ] **Step 2: 실패 확인**

Run:

```bash
/home/swlab-youngjin/ML-HW-Benchmark-Framework/.venv-mobilint-compile/bin/python -m pytest framework/tests/test_hw_monitor.py -q
```

Expected: 새 constructor 인자와 메서드가 없어 FAIL

- [ ] **Step 3: HWMonitor 구성 API 구현**

constructor를 다음과 같이 확장한다.

```python
def __init__(
    self,
    interval: float = 0.2,
    *,
    summary_enabled: bool = True,
    power_trace_enabled: bool = False,
    power_trace_baseline_sec: float = 3.0,
    clock_ns: Callable[[], int] = time.monotonic_ns,
    wait_fn: Callable[[float], None] = time.sleep,
): ...

def configure_power_trace(
    self,
    *,
    reservation: RunArtifactReservation,
    target_id: str,
) -> None: ...

def power_trace_metadata(self) -> dict[str, object]: ...
def startup_timeout_hint_sec(self) -> float: ...
```

`startup_timeout_hint_sec()`은 trace가 켜지면 3.0, 아니면 0.0을 반환한다.

- [ ] **Step 4: deadline 기반 전력 루프 구현**

기존 summary poll과 같은 monitor 객체 안에서 power source 하나를 선택한다.
전력 조회 전후 `clock_ns()`를 읽고 중간값과 query latency를 계산한다. 같은
collector의 일반/전력 호출에는 하나의 lock을 사용하고, 동시에 due이면 전력
조회부터 수행한다. collector가 없으면 writer를 만들지 않고 상태를
`unavailable`로 유지한다.

- [ ] **Step 5: baseline과 실패 격리 구현**

trace가 켜진 `start()`는 writer 및 poll thread를 시작하고 baseline 시작 표본을
즉시 강제한 뒤 `wait_fn(3.0)`으로 baseline을 수집한다. 그 다음 phase를
`inference`로 바꾸고 추론 시작 표본을 즉시 강제한 뒤 writer를 flush하고
반환한다. `stop()`은 게시 전에 추론 종료 표본을 즉시 강제한다. 따라서 측정
루프가 200ms보다 짧아도 inference 조회 시도가 최소 두 번 남는다. power
read/write 실패는 내부 상태로 흡수하며
기존 일반 collector 시작 실패의 transactional cleanup 규칙은 유지한다.

- [ ] **Step 6: registry factory 확장**

`create_hw_monitor()`에 다음 keyword를 추가하여 `--power-trace`만 지정해도
monitor 객체가 생성되도록 한다.

```python
summary_enabled: bool = True
power_trace_enabled: bool = False
```

- [ ] **Step 7: 공통 모니터 회귀 테스트**

Run:

```bash
/home/swlab-youngjin/ML-HW-Benchmark-Framework/.venv-mobilint-compile/bin/python -m pytest framework/tests/test_hw_monitor.py framework/tests/test_plugin_registry.py -q
```

Expected: PASS

- [ ] **Step 8: 커밋**

```bash
git add framework/src/monitors/base.py framework/src/monitors/__init__.py framework/tests/test_hw_monitor.py framework/tests/test_plugin_registry.py
git commit -m "feat: trace raw power through hardware monitor"
```

### Task 4: Rebellions와 Mobilint의 순간 전력 수집 및 에너지 제거

**Files:**
- Modify: `framework/src/monitors/rbln_collector.py`
- Modify: `framework/src/monitors/mobilint_collector.py`
- Modify: `framework/src/core/targets.py:375-480`
- Modify: `framework/tests/test_rbln_collector.py`
- Modify: `framework/tests/test_mobilint_collector.py`
- Modify: `framework/tests/test_plugin_registry.py`

**Interfaces:**
- Consumes: Task 1의 `PowerTraceSource`, `PowerReading`; Task 3의 직렬화 계약
- Produces: 두 collector의 `power_trace_source()`와 `collect_power()`

- [ ] **Step 1: 기존 에너지 기대 테스트를 원시 전력 계약으로 교체**

`hw_accel_energy_j`, trapezoid, energy chain을 검증하는 테스트를 삭제하거나 다음
계약으로 교체한다.

```python
def test_rbln_power_source_is_whole_card_at_two_hundred_ms(): ...
def test_rbln_collect_power_uses_exact_json_command_and_card_power(): ...
def test_rbln_power_call_caches_full_snapshot_for_monitor_collect(): ...
def test_mobilint_aries_power_source_is_device_total_at_two_hundred_ms(): ...
def test_mobilint_collect_power_calls_only_total_power(): ...
def test_mobilint_regulus_has_no_power_trace_source(): ...
def test_vendor_summaries_never_emit_energy_j(): ...
```

- [ ] **Step 2: 실패 확인**

Run:

```bash
/home/swlab-youngjin/ML-HW-Benchmark-Framework/.venv-mobilint-compile/bin/python -m pytest framework/tests/test_rbln_collector.py framework/tests/test_mobilint_collector.py -q
```

Expected: 새 power API 부재와 기존 energy 출력 때문에 FAIL

- [ ] **Step 3: RblnCollector 수정**

기존 일반 telemetry의 `sample_interval_sec`와 별도로 constructor에
`power_sample_interval_sec: float = 0.2`를 추가한다.
`power_trace_source()`는 collector `rbln`, source `rbln-smi-json`, device ID,
scope `whole_card`, period `0.2`를 반환한다. `collect_power()`는 기존 strict
`_snapshot()`과 `card_power` parser를 사용한다. full snapshot을 한 번 cache하여
같은 deadline의 일반 `collect()`가 subprocess를 두 번 실행하지 않고 그 값을
한 번 소비하도록 한다. energy 필드와 `_record_success()`의 적분 부분은 제거한다.

주기 근거는 CA22의 `uW` 문자열 원본, 1초 간격 100회 무오류, p99
4.567737ms, 최대 4.590526ms 및 90회 값 변경이다. 200ms에서 p99 조회 duty는
약 2.3%이므로 서버 게이트에서 trace-on 지연 증가가 2%를 넘으면 Rebellions만
250ms 또는 500ms로 조정한다.

- [ ] **Step 4: MobilintCollector 수정**

constructor에 `power_sample_interval_sec: float = 0.2`를 추가한다.
ARIES에만 source `mbltml`, scope `device_total`, period `0.2`를 반환한다.
`collect_power()`는 `mbltmlGetTotalPower(device_id)`만 호출하여 `PowerReading`을
만들고, 전류·전압·온도 조회를 호출하지 않는다. REGULUS는 source `None`이다.
`_energy_j`, 이전 W/ns, stop energy boundary, energy summary를 제거한다.

주기 근거는 50ms 간격 100회 실측의 p99 0.085284ms, 최대 0.39403ms와
약 1,000ms마다 한 번인 값 변경 간격이다. 200ms polling은 추론 경계를 더
촘촘히 기록하기 위한 것이며 센서 갱신을 강제하지 않는다. 같은 W가 반복돼도
원시 표본을 제거하지 않는다.

- [ ] **Step 5: target 옵션 명시**

`rbln-static`, `rbln-vllm` monitor option에 `power_sample_interval_sec=0.2`,
`mobilint-aries`에도 `power_sample_interval_sec=0.2`를 넣는다. REGULUS에는
전력 주기를 넣지 않는다.

- [ ] **Step 6: 벤더·registry 테스트 통과 확인**

Run:

```bash
/home/swlab-youngjin/ML-HW-Benchmark-Framework/.venv-mobilint-compile/bin/python -m pytest framework/tests/test_rbln_collector.py framework/tests/test_mobilint_collector.py framework/tests/test_plugin_registry.py -q
```

Expected: PASS, 결과에서 `hw_accel_energy_j`가 없음

- [ ] **Step 7: 커밋**

```bash
git add framework/src/monitors/rbln_collector.py framework/src/monitors/mobilint_collector.py framework/src/core/targets.py framework/tests/test_rbln_collector.py framework/tests/test_mobilint_collector.py framework/tests/test_plugin_registry.py
git commit -m "refactor: expose vendor raw power readings"
```

### Task 5: Furiosa RNGD 공식 SMI collector

**Files:**
- Create: `framework/src/monitors/furiosa_collector.py`
- Modify: `framework/src/monitors/__init__.py`
- Modify: `framework/src/core/targets.py:350-375`
- Modify: `framework/requirements-furiosa-torch.txt`
- Create: `framework/tests/test_furiosa_collector.py`
- Modify: `framework/tests/test_furiosa_torch_environment_contract.py`
- Modify: `framework/tests/test_plugin_registry.py`

**Interfaces:**
- Consumes: Task 1의 power contract와 Task 3의 registry factory
- Produces: `FuriosaCollector`, registry key `furiosa`

- [ ] **Step 1: 공식 Python SMI 계약 테스트 작성**

fake `furiosa_smi_py`로 다음을 검증한다.

```python
def test_furiosa_selects_exact_npu_name_after_init_and_list_devices(): ...
def test_furiosa_collect_power_calls_device_power_consumption(): ...
def test_furiosa_rejects_non_finite_or_negative_power(): ...
def test_furiosa_missing_package_or_method_is_unavailable(): ...
def test_furiosa_source_is_device_at_two_hundred_ms(): ...
def test_furiosa_does_not_spawn_or_parse_cli_json_or_table(): ...
```

`npu0`만 선택하고 반환 float를 그대로 W로 보존하며, 패키지·메서드가 없으면
reflection fallback이나 subprocess 없이 unavailable인지 확인한다.

- [ ] **Step 2: 실패 확인**

Run:

```bash
/home/swlab-youngjin/ML-HW-Benchmark-Framework/.venv-mobilint-compile/bin/python -m pytest framework/tests/test_furiosa_collector.py framework/tests/test_plugin_registry.py -q
```

Expected: collector module 또는 registry key 부재로 FAIL

- [ ] **Step 3: FuriosaCollector 구현**

constructor와 source를 다음으로 고정한다.

```python
def __init__(
    self,
    device_name: str = "npu0",
    power_sample_interval_sec: float = 0.2,
): ...
```

`start()`에서 `furiosa_smi_py.init()`, `list_devices()`,
`device.device_info().name()`으로 정확한 장치를 하나 선택한다. `collect_power()`는
검증된 `float(device.power_consumption())`만 호출한다. source는
`furiosa-smi-py`, scope는 확인된 API docstring 범위를 넘어서지 않는 `device`로
기록한다. 설치 서버에서 이 계약이나 단위가 다르면 collector를 수정해 추측하지
말고 unavailable evidence를 남긴다.

실측 근거는 `furiosa-smi-py 2026.1.2`, `npu0`, W float, 50ms 간격 100회
무오류, p99 1.557525ms, 최대 1.557729ms다. 유휴 5초 동안 값이 변하지 않았어도
기본 주기는 기존 `HWMonitor`와 같은 0.2초로 두고 중복 원시 표본을 보존한다.
CLI JSON 호출은 p50 약 92.28ms였으므로 CLI fallback은 구현하지 않는다.

구현 근거는 Furiosa 공식 자료의 현재 Python SMI 경로로 한정한다.

- `furiosa-perf` monitor:
  <https://github.com/furiosa-ai/furiosa-perf/blob/main/furiosa_perf/runner/monitor.py>
- Furiosa metrics exporter:
  <https://github.com/furiosa-ai/furiosa-metrics-exporter>
- Furiosa SMI library 문서:
  <https://developer.furiosa.ai/latest/en/device_management/system_management_interface/furiosa_smi_lib.html>

- [ ] **Step 4: registry와 target 연결**

collector key `furiosa`를 lazy 등록하고 `furiosa-rngd`,
`furiosa-rngd-torch`의 `monitor_names`를 `("furiosa", "system")`으로 바꾼다.
두 target에 `device_name="npu0"`, `power_sample_interval_sec=0.2`를 명시한다.

- [ ] **Step 5: Furiosa Torch 환경 계약에 Python SMI pin 추가**

`framework/requirements-furiosa-torch.txt`에 서버에서 확인한
`furiosa-smi-py==2026.1.2`를 추가하고 environment contract 테스트가 정확한 pin을
요구하도록 수정한다. 이 pin은 서버 중간 게이트에서 `furiosa-torch 2026.3.0`과
함께 `pip check` 및 직접 전력 호출을 통과해야 확정된다.

- [ ] **Step 6: 테스트 통과 확인**

Run:

```bash
/home/swlab-youngjin/ML-HW-Benchmark-Framework/.venv-mobilint-compile/bin/python -m pytest framework/tests/test_furiosa_collector.py framework/tests/test_plugin_registry.py framework/tests/test_furiosa_torch_environment_contract.py framework/tests/test_furiosa_torch_bert_integration.py -q
```

Expected: PASS

- [ ] **Step 7: 커밋**

```bash
git add framework/src/monitors/furiosa_collector.py framework/src/monitors/__init__.py framework/src/core/targets.py framework/requirements-furiosa-torch.txt framework/tests/test_furiosa_collector.py framework/tests/test_furiosa_torch_environment_contract.py framework/tests/test_plugin_registry.py framework/tests/test_furiosa_torch_bert_integration.py
git commit -m "feat: collect Furiosa RNGD raw power"
```

- [ ] **Step 8: 세 collector 서버 중간 게이트**

이 커밋을 서버 작업 디렉터리에 반영한 뒤 Task 6을 시작하지 말고 중지한다.
RBLN, ARIES, Furiosa collector를 각각 해당 vendor Python 환경에서 직접
`start() -> collect_power() -> stop()`으로 실행해 source, scope, device ID와
원시 W를 확인한다. 유휴 100회 결과가 사전 조사와 같은 단위·대략적인 호출
지연을 유지하는지, 200ms polling과 baseline 시작·추론 시작·추론 종료 강제
조회가 동작하는지 확인한다. 같은 값이 반복돼도 원본 행이 보존돼야 한다.

그 다음 기존 TTM-R2 명령을 trace-off와 trace-on으로 번갈아 다섯 쌍 실행해
collector 값이 장치 부하에 반응하는지와 측정 루프 중앙 지연 증가를 확인한다.
증가가 2%를 넘으면 Rebellions만 250ms 또는 500ms로 조정하고 다시 측정한다.
이 단계는 collector/API 게이트이며 compile, warmup, inference 경계의
최종 증거로 사용하지 않는다. 하나라도 API/단위/장치 선택이 다르거나 추론에
간섭하면 Tasks 1-5의 해당 소유 task에서 수정하고 게이트를 다시 수행한다.

Furiosa의 검증된 `furiosa-torch 2026.3.0` 환경에는 조사 당시
`furiosa-smi-py`가 없었다. 다른 가상환경의 site-packages를 주입하거나 CLI로
우회하지 않는다. 호환 Python SMI 패키지가 실행 환경에 명시적으로 준비되지
않으면 예상 결과는 `unavailable`이며, 설치는 별도 사용자 승인 후 수행한다.
`unavailable` 기록은 정직한 실패 처리지만 세 target 지원 완료 게이트를
통과한 것은 아니다. Furiosa 게이트는 pin 설치, `pip check`, 직접 API 100회와
TTM-R2 부하 반응이 모두 성공해야 통과한다.

### Task 6: 결과 예약과 power trace provenance

**Files:**
- Modify: `framework/src/core/result_store.py:80-135, 928-1150`
- Modify: `framework/src/main.py:1070-1205`
- Modify: `framework/tests/test_result_store.py`
- Modify: `framework/tests/test_async_result_artifacts.py`

**Interfaces:**
- Consumes: Task 2의 reservation path와 `PowerTraceArtifact.as_result_metadata()`
- Produces: `save_result()`의 여섯 power trace keyword 및 e2e reservation 지원

- [ ] **Step 1: metadata와 reservation 테스트 작성**

```python
def test_result_store_persists_power_trace_linkage_as_metadata(tmp_path): ...
def test_e2e_result_accepts_matching_run_reservation(tmp_path): ...
def test_e2e_result_rejects_mismatched_power_reservation(tmp_path): ...
def test_disabled_or_unavailable_trace_has_empty_artifact_fields(tmp_path): ...
def test_power_trace_numeric_values_cannot_enter_metrics_namespace(tmp_path): ...
```

일반 결과가 W/J/평균값을 새 metadata로 추가하지 않고 여섯 linkage 필드만
추가하는지 확인한다.

- [ ] **Step 2: 실패 확인**

Run:

```bash
/home/swlab-youngjin/ML-HW-Benchmark-Framework/.venv-mobilint-compile/bin/python -m pytest framework/tests/test_result_store.py framework/tests/test_async_result_artifacts.py -q
```

Expected: `save_result()` keyword 또는 e2e reservation 거부로 FAIL

- [ ] **Step 3: META_COLUMNS와 save_result signature 확장**

다음 필드를 `request_trace_path` 뒤에 추가한다.

```text
power_trace_status
power_trace_path
power_trace_sha256
power_trace_sample_count
power_monitor_source
power_scope
```

`save_result()`에 동일 keyword와 빈 기본값을 추가하고 row에 명시적으로 넣는다.
metrics dict가 같은 이름을 덮어쓰지 못하게 기존 meta key 보호를 유지한다.

- [ ] **Step 4: e2e reservation 허용**

`reservation is not None`이면 mode와 무관하게 supplied `run_id`, reservation,
results path가 일치해야 하며 `_save_reserved_result()`를 사용한다. 기존
`async_queue`와 `external_server`는 계속 reservation을 필수로 요구한다.
reservation이 없는 보통 e2e 경로는 기존 `_save_unreserved_result()`를 유지한다.

- [ ] **Step 5: 결과 저장 테스트 통과 확인**

Run:

```bash
/home/swlab-youngjin/ML-HW-Benchmark-Framework/.venv-mobilint-compile/bin/python -m pytest framework/tests/test_result_store.py framework/tests/test_async_result_artifacts.py -q
```

Expected: PASS

- [ ] **Step 6: 커밋**

```bash
git add framework/src/core/result_store.py framework/src/main.py framework/tests/test_result_store.py framework/tests/test_async_result_artifacts.py
git commit -m "feat: link power traces to benchmark results"
```

### Task 7: CLI와 동기 e2e 실행 연결

**Files:**
- Modify: `framework/src/main.py:840-930, 2575-2640, 3501-3529`
- Modify: `framework/src/core/benchmarkrunner.py:125-163`
- Modify: `framework/tests/test_main_paths.py`
- Modify: `framework/tests/test_hw_monitor.py`

**Interfaces:**
- Consumes: Task 3의 monitor 구성/metadata API, Task 6의 예약 가능한 save_result
- Produces: `--power-trace`, trace-only monitor 생성, e2e run ID 사전 예약 및 저장

- [ ] **Step 1: CLI와 e2e 생명주기 테스트 작성**

```python
def test_power_trace_flag_is_explicit_and_defaults_false(): ...
def test_power_trace_alone_creates_monitor_without_summary(): ...
def test_e2e_power_trace_reserves_run_before_runner_starts(): ...
def test_e2e_warmup_precedes_baseline_and_inference_follows_start_return(): ...
def test_e2e_power_failure_preserves_metrics_and_saves_failed_status(): ...
def test_e2e_trace_path_and_sha_match_reserved_run_id(): ...
```

- [ ] **Step 2: 실패 확인**

Run:

```bash
/home/swlab-youngjin/ML-HW-Benchmark-Framework/.venv-mobilint-compile/bin/python -m pytest framework/tests/test_main_paths.py framework/tests/test_hw_monitor.py -q
```

Expected: parser와 wiring 부재로 FAIL

- [ ] **Step 3: CLI와 monitor 생성 조건 추가**

parser에 `--power-trace` `store_true`를 추가한다. main의 monitor 생성 조건을
`args.monitor or args.power_trace`로 바꾸고 factory에
`summary_enabled=args.monitor`, `power_trace_enabled=args.power_trace`를 전달한다.

- [ ] **Step 4: e2e reservation과 trace binding 추가**

`execute_benchmark()`의 e2e branch에서 `args.power_trace`일 때
`reserve_run_artifacts(actual_results_path)`를 먼저 호출하고
`hw_monitor.configure_power_trace(reservation=..., target_id=target.target_id)`를
호출한 뒤 runner를 시작한다. save kwargs에 같은 `run_id`, `reservation`,
`hw_monitor.power_trace_metadata()`를 넣는다. 플래그가 없으면 기존 경로를 그대로
사용하고 `power_trace_status=disabled`만 저장한다.

- [ ] **Step 5: BenchmarkRunner 경계 보존**

기존 warmup 뒤 `monitor.start()` 및 `finally: monitor.stop()` 구조를 유지한다.
trace-only에서는 `summary()`가 빈 dict를 반환하도록 Task 3 계약을 사용한다.
stop 또는 writer 실패는 monitor metadata로 변환하고 성공한 `metrics`를 버리지
않도록 예외 우선순위를 테스트와 일치시킨다.

- [ ] **Step 6: 동기 경로 테스트 통과 확인**

Run:

```bash
/home/swlab-youngjin/ML-HW-Benchmark-Framework/.venv-mobilint-compile/bin/python -m pytest framework/tests/test_main_paths.py framework/tests/test_hw_monitor.py -q
```

Expected: PASS

- [ ] **Step 7: 커밋**

```bash
git add framework/src/main.py framework/src/core/benchmarkrunner.py framework/tests/test_main_paths.py framework/tests/test_hw_monitor.py
git commit -m "feat: trace power in synchronous benchmarks"
```

### Task 8: async_queue 측정 경계와 결과 연결

**Files:**
- Modify: `framework/src/core/async_inference/runner.py:648-710, 848-990, 1041-1085, 1242-1278`
- Modify: `framework/src/main.py:2641-2895`
- Modify: `framework/tests/test_async_runner.py`
- Modify: `framework/tests/test_async_cli.py`
- Modify: `framework/tests/test_main_paths.py`

**Interfaces:**
- Consumes: Task 3의 blocking baseline start와 `startup_timeout_hint_sec()`; Task 6의 result metadata
- Produces: first submit 전 baseline 완료, final physical completion 뒤 stop, async power metadata 저장

- [ ] **Step 1: async 경계와 실패 테스트 작성**

```python
def test_async_power_baseline_finishes_before_first_measured_submit(): ...
def test_async_monitor_start_deadline_includes_baseline_timeout_hint(): ...
def test_async_power_stop_waits_until_flush_and_outstanding_zero(): ...
def test_async_power_failure_is_warning_not_run_invalid_reason(): ...
def test_async_result_links_reserved_power_trace(): ...
```

producer 제출 종료와 device completion을 분리한 fake runtime으로 stop이 flush 및
outstanding zero 이후 호출되는지 검증한다.

- [ ] **Step 2: 실패 확인**

Run:

```bash
/home/swlab-youngjin/ML-HW-Benchmark-Framework/.venv-mobilint-compile/bin/python -m pytest framework/tests/test_async_runner.py framework/tests/test_async_cli.py framework/tests/test_main_paths.py -q
```

Expected: timeout hint와 power metadata wiring 부재로 FAIL

- [ ] **Step 3: monitor start callback deadline 보정**

`_MeasuredSubmitter.start_monitor()`는 monitor에 callable
`startup_timeout_hint_sec()`가 있을 때 기존 callback timeout에 그 값을 더한다.
`monitor.start()`가 반환한 뒤에만 `producer.run()`을 호출하는 현재 순서는
유지하여 3초 baseline이 첫 measured submit보다 앞서도록 한다.

- [ ] **Step 4: stop 경계와 power 실패 분리**

기존 engine flush 및 outstanding shutdown 뒤 monitor stop 순서를 유지한다.
power trace 자체의 `partial/failed/unavailable`은 hardware callback 실패로 run을
INVALID 처리하지 않는다. 실제 monitor 메서드 예외와 callback timeout은 기존
async warning/invalid 규칙을 유지한다.

- [ ] **Step 5: async reservation에 writer 연결**

이미 생성되는 async reservation을 `hw_monitor.configure_power_trace()`에 전달하고,
성공/실패 persistence 경로에 `power_trace_metadata()`를 연결한다. request trace와
power trace는 같은 run ID를 사용하되 각자의 디렉터리와 SHA를 유지한다.

- [ ] **Step 6: async 테스트 통과 확인**

Run:

```bash
/home/swlab-youngjin/ML-HW-Benchmark-Framework/.venv-mobilint-compile/bin/python -m pytest framework/tests/test_async_runner.py framework/tests/test_async_cli.py framework/tests/test_main_paths.py -q
```

Expected: PASS

- [ ] **Step 7: 커밋**

```bash
git add framework/src/core/async_inference/runner.py framework/src/main.py framework/tests/test_async_runner.py framework/tests/test_async_cli.py framework/tests/test_main_paths.py
git commit -m "feat: trace power across async measurement lifecycle"
```

### Task 9: 문서, 전체 회귀검증, 서버 인수시험

**Files:**
- Modify: `framework/src/README.md`
- Modify: `framework/src/runtimes/README.md`
- Modify: `framework/docs/rbln-setup.md`
- Modify: `framework/docs/rbln-troubleshooting.md`
- Modify: `docs/furiosa-rngd-setup.md`
- Modify: `docs/furiosa-rngd-troubleshooting.md`
- Modify: `docs/mobilint-aries-transformers.md`
- Modify: `framework/docs/ttm-r2-framework.md`

**Interfaces:**
- Consumes: Tasks 1-8의 최종 CLI, CSV schema, 상태, 벤더 기본 주기
- Produces: 사용자 실행 명령, 서버 probe/인수 절차, 에너지 미계산 경계 문서

- [ ] **Step 1: 문서의 기존 energy 설명을 원시 trace 계약으로 교체**

`hw_accel_energy_j`와 collector 내부 적분 설명을 제거한다. 모든 관련 문서에
`--power-trace`가 `--monitor`와 별개이고, 3초 baseline과 inference W만 CSV에
저장하며 에너지는 계산하지 않는다고 명시한다.

- [ ] **Step 2: 서버 실측 근거와 Furiosa 의존성 문서화**

동일 서버의 세 장치에서 확인한 패키지/도구·driver·firmware, API/필드, 장치
식별자, 100회 latency와 값 변경 주기를 기록한다. 세 target의 최초 기본 주기가
모두 0.2초인 근거와 센서 갱신이 더 느릴 때 중복값을 보존한다는 제한을 남긴다.
Furiosa는 `furiosa-smi-py 2026.1.2` 직접 API와 검증된
TTM-R2 가상환경의 패키지 부재를 함께 기록하고, CLI JSON fallback이나 타
가상환경 site-packages 주입을 금지한다. 호환 패키지가 없으면
`unavailable`이라고 명시한다.

- [ ] **Step 3: 세 벤더 실행 예시 추가**

TTM-R2 및 한 개 비-TTM 모델 명령에 `--power-trace`를 추가하고 결과 CSV와
`power/<run_id>.power.csv` 연결 확인 명령을 작성한다. 에너지 분석 명령은
추가하지 않는다.

- [ ] **Step 4: 집중 회귀 테스트 실행**

Run:

```bash
/home/swlab-youngjin/ML-HW-Benchmark-Framework/.venv-mobilint-compile/bin/python -m pytest framework/tests/test_power_trace.py framework/tests/test_furiosa_collector.py framework/tests/test_hw_monitor.py framework/tests/test_rbln_collector.py framework/tests/test_mobilint_collector.py framework/tests/test_plugin_registry.py framework/tests/test_result_store.py framework/tests/test_async_result_artifacts.py framework/tests/test_main_paths.py framework/tests/test_async_runner.py framework/tests/test_async_cli.py -q
```

Expected: all selected tests PASS

- [ ] **Step 5: 기존 기준 테스트 재실행**

Run:

```bash
/home/swlab-youngjin/ML-HW-Benchmark-Framework/.venv-mobilint-compile/bin/python -m pytest framework/tests/test_hw_monitor.py framework/tests/test_rbln_collector.py framework/tests/test_mobilint_collector.py -q
```

Expected: PASS; 기존 기준은 변경 전 `109 passed`

- [ ] **Step 6: 정적 범위 검사**

Run:

```bash
rg -n "hw_accel_energy_j|_energy_j|_energy_joules|J/inference|joules_per" framework/src framework/tests
```

Expected: 전력 구현과 테스트에 에너지 계산 코드가 없고, 남은 일치는 명시적인
부재 검증 또는 역사 문서뿐임

- [ ] **Step 7: 동일 서버의 세 장치 최종 물리 인수시험**

Task 5 중간 게이트를 통과한 production collector로 100회 idle probe를 다시
실행해 사전 조사와 일치하는지 확인한다.
trace-off/trace-on을 번갈아 다섯 쌍 실행하여 trace-on 중앙 measured-loop 지연이
2% 이내인지 확인한다. 그 뒤 TTM-R2 240-window와 비-TTM 모델 하나를 실행해
`baseline`/`inference`, raw W, monotonic timing, SHA, run ID 연결을 확인한다.
전력 trace 성공과 모델 품질 성공은 별도 상태로 기록한다.

- [ ] **Step 8: 문서와 검증 변경 커밋**

```bash
git add framework/src/README.md framework/src/runtimes/README.md framework/docs/rbln-setup.md framework/docs/rbln-troubleshooting.md docs/furiosa-rngd-setup.md docs/furiosa-rngd-troubleshooting.md docs/mobilint-aries-transformers.md framework/docs/ttm-r2-framework.md
git commit -m "docs: document framework raw power tracing"
```

- [ ] **Step 9: 최종 브랜치 상태 확인**

Run:

```bash
git status --short --branch
git log --oneline --decorate -12
```

Expected: `feat/power_measurment`의 작업 트리가 clean이며 Tasks 1-9 커밋이 순서대로 존재함
