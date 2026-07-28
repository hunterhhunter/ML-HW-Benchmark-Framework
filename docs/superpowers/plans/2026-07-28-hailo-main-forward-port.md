# Hailo Native Async Main Forward-port Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 최신 `origin/main`의 terminal retirement lease 규약 위에서 Hailo-8/Hailo-10H ResNet50·YOLOv5m native async를 연속 요청 교착 없이 실행하고 기존 원격 기능 브랜치에 안전하게 배포한다.

**Architecture:** `origin/main`을 기존 Hailo 브랜치에 merge하고, main의 target capability → `Runtime.create_native_backend()` → `NativeAsyncRuntimeExecutor` 경로를 유일한 executor 선택 경로로 유지한다. Hailo adapter는 자기 자신을 native backend로 제공하되 SDK queue 크기와 Hailo별 completion budget을 공용 builder의 선택적 상한으로 전달하고, permit 반환은 main의 terminal retirement lease만 담당한다.

**Tech Stack:** Python 3.10, pytest, HailoRT/PyHailoRT InferModel API, Git merge/worktree

## Global Constraints

- 현재 `feat/hailo-native-async` 이력을 재작성하거나 force-push하지 않는다.
- 병합과 검증은 현재 사용자의 미커밋 DeepX·문서 변경과 분리된 clean worktree에서 수행한다.
- `origin/main`의 `NativeAsyncRuntimeExecutor`, completion handoff, retirement lease, terminal commit 규약을 기준 구현으로 유지한다.
- HailoRT `InferModel`의 bindings와 입출력 버퍼는 SDK callback 및 framework terminal 처리 전까지 유지한다.
- Hailo callback은 결과 복사를 별도 completion worker로 넘기고 짧게 반환한다.
- Hailo completion, submit, protocol, output-copy 오류는 fail-closed로 처리하고 ready 실패만 요청 단위 복구를 허용한다.
- 기존 Mobilint, Furiosa, RBLN 및 blocking runtime의 native executor 동작을 퇴행시키지 않는다.
- Hailo-8과 Hailo-10H는 같은 adapter를 사용하되 장치별 HailoRT와 HEF 호환성은 실장 검증에서 확인한다.

---

### Task 1: 격리된 forward-port 작업공간 생성

**Files:**
- Read: `.gitignore`
- Worktree: `/tmp/ml-hw-benchmark-hailo-forward-port`

**Interfaces:**
- Consumes: `feat/hailo-native-async`의 승인된 설계 커밋 `f41e8bb`, 최신 `origin/main`
- Produces: 사용자 미커밋 파일과 분리된 detached clean worktree

- [ ] **Step 1: 현재 저장소와 원격 기준점을 기록한다**

Run:

```bash
git status --short
git rev-parse feat/hailo-native-async
git rev-parse origin/feat/hailo-native-async
git rev-parse origin/main
```

Expected: 로컬 Hailo 브랜치는 `f41e8bb`을 포함하고, 사용자 변경은 기존 기본 worktree에만 표시된다.

- [ ] **Step 2: detached clean worktree를 만든다**

Run:

```bash
git worktree add --detach /tmp/ml-hw-benchmark-hailo-forward-port feat/hailo-native-async
git -C /tmp/ml-hw-benchmark-hailo-forward-port status --short
```

Expected: 두 번째 명령은 아무 파일도 출력하지 않는다.

- [ ] **Step 3: merge base와 예상 충돌 파일을 확인한다**

Run:

```bash
git -C /tmp/ml-hw-benchmark-hailo-forward-port merge-base HEAD origin/main
git -C /tmp/ml-hw-benchmark-hailo-forward-port merge-tree e712698a7f6f5d676df08bec605bd77431ba41ab HEAD origin/main
```

Expected: 텍스트 충돌은 `README.md`와 `framework/tests/test_runtime_executor.py`에 한정되고, 이후 자동 병합 파일은 논리 충돌 검토 대상이 된다.

### Task 2: 최신 main 병합과 현재 native backend 계약 회귀 테스트

**Files:**
- Modify: `README.md`
- Modify: `framework/tests/test_runtime_executor.py`
- Modify: `framework/tests/test_async_cli.py`
- Modify: `framework/tests/test_hailo_runtime.py`

**Interfaces:**
- Consumes: `main._build_async_runtime_executor(args, target, runtime, loader, config)`와 `HailoRuntime` fake InferModel fixture
- Produces: 현재 main executor 계약에서 Hailo queue/deadline 상한 및 backend factory를 요구하는 실패 테스트

- [ ] **Step 1: 최신 main을 commit 전 상태로 병합한다**

Run:

```bash
git -C /tmp/ml-hw-benchmark-hailo-forward-port merge --no-commit --no-ff origin/main
git -C /tmp/ml-hw-benchmark-hailo-forward-port status --short
```

Expected: merge는 충돌 상태로 멈추며 `README.md`와 `framework/tests/test_runtime_executor.py`가 unmerged로 표시된다.

- [ ] **Step 2: 텍스트 충돌은 양쪽 기능을 모두 유지한다**

`README.md`의 adapter 안내는 다음 두 줄을 모두 보존한다.

```markdown
Hailo-8/10H native async 규약과 ResNet50·YOLOv5m 실행 예시는 [docs/hailo-async-runtime.md](docs/hailo-async-runtime.md)를 참조하세요.
Furiosa-LLM 전용 환경과 RNGD 실행 절차는 [docs/furiosa-rngd-setup.md](docs/furiosa-rngd-setup.md)를 참조하세요.
```

`framework/tests/test_runtime_executor.py`의 import는 다음 합집합을 사용한다.

```python
from core.runtime_executor import (
    BlockingRuntimeExecutor,
    GenerationObservation,
    GenerationOutputEvent,
    NativeAsyncOutcome,
    NativeAsyncRuntimeExecutor,
)
```

Hailo 브랜치의 `OptInNativeRuntime` 및 `create_async_runtime_executor` 테스트는 현재 main 계약과 중복되므로 제거하고, main의 generation observation 테스트는 전부 보존한다.

- [ ] **Step 3: Hailo executor 상한을 검증하는 CLI 테스트를 먼저 작성한다**

`framework/tests/test_async_cli.py`에 현재 builder를 직접 검증하는 다음 형태의 테스트를 추가한다.

```python
def test_hailo_native_async_executor_uses_runtime_queue_and_timeout_limits():
    args = _async_args(
        "--target", "hailo10h",
        "--worker-count", "8",
        "--queue-capacity", "16",
        "--flush-timeout-sec", "300",
    )
    config = benchmark_main.build_async_config(args)
    backend = SimpleNamespace(submit_async=lambda inputs, callback: None)

    runtime = SimpleNamespace(
        supports_generate=lambda: False,
        native_async_max_batch_size=lambda: 1,
        native_async_max_inflight=lambda: 4,
        native_async_completion_timeout_sec=lambda: 12.5,
        create_native_backend=lambda: backend,
    )

    executor = benchmark_main._build_async_runtime_executor(
        args,
        get_target("hailo10h"),
        runtime,
        SimpleNamespace(get_metadata=lambda: {}),
        config,
    )

    assert isinstance(executor, NativeAsyncRuntimeExecutor)
    assert executor.backend is backend
    assert executor.max_inflight == 4
    assert executor.completion_timeout_sec == 12.5
```

같은 파일에 `native_async_max_inflight()`가 `0`, `native_async_completion_timeout_sec()`가 `float("nan")`이면 target 이름과 잘못된 hook 이름을 포함한 `RuntimeError`를 내는 두 검증 테스트를 추가한다.

- [ ] **Step 4: Hailo runtime factory 계약 테스트를 먼저 작성한다**

`framework/tests/test_hailo_runtime.py`의 InferModel fake load 테스트에 다음 assertion을 추가한다.

```python
assert runtime.native_async_max_batch_size() == 2
assert runtime.native_async_max_inflight() == 3
assert runtime.native_async_completion_timeout_sec() == pytest.approx(20.617)
assert runtime.create_native_backend() is runtime
```

legacy InferVStreams 테스트에는 다음 검증을 추가한다.

```python
with pytest.raises(RuntimeError, match="InferModel"):
    runtime.create_native_backend()
```

- [ ] **Step 5: 새 계약 테스트가 구현 부재로 실패하는지 확인한다**

Run:

```bash
cd /tmp/ml-hw-benchmark-hailo-forward-port/framework
python -m pytest tests/test_async_cli.py::test_hailo_native_async_executor_uses_runtime_queue_and_timeout_limits tests/test_hailo_runtime.py::test_hailo_runtime_native_async_resnet_preserves_batch_and_ready_contract -q
```

Expected: main builder가 runtime별 inflight/timeout hook을 아직 적용하지 않거나 Hailo에 `create_native_backend()`가 없어 FAIL한다.

### Task 3: Hailo를 main의 native executor 선택 경로로 전환

**Files:**
- Modify: `framework/src/runtimes/base.py`
- Modify: `framework/src/runtimes/hailo_rt.py`
- Modify: `framework/src/main.py`
- Modify: `framework/src/core/runtime_executor.py`
- Modify: `framework/src/core/targets.py`
- Modify: `framework/tests/test_async_cli.py`
- Modify: `framework/tests/test_hailo_runtime.py`
- Modify: `framework/tests/test_plugin_registry.py`
- Modify: `docs/hailo-async-runtime.md`
- Modify: `docs/async-inference-queue.md`
- Modify: `framework/src/runtimes/README.md`

**Interfaces:**
- Consumes: `Runtime.create_native_backend()`, `Runtime.native_async_max_batch_size()`, Hailo `submit_async(inputs, callback)`
- Produces: `Runtime.native_async_max_inflight() -> int | None`, `Runtime.native_async_completion_timeout_sec() -> float | None`, `HailoRuntime.create_native_backend() -> HailoRuntime`

- [ ] **Step 1: base runtime에 선택적 executor 상한 계약을 정의한다**

`framework/src/runtimes/base.py`의 `native_async_max_batch_size()` 옆에 다음 기본 메서드를 둔다. 다른 backend는 `None`을 받아 기존 main 동작을 유지한다.

```python
def native_async_max_inflight(self) -> int | None:
    """Optional runtime-specific cap for framework native dispatches."""
    return None

def native_async_completion_timeout_sec(self) -> float | None:
    """Optional runtime-specific logical completion deadline."""
    return None
```

예전 Hailo 브랜치가 추가한 base-level `supports_native_async()`와 `submit_async()` opt-in을 executor 선택에 사용하지 않는다. target capability와 `create_native_backend()`가 현재 main의 유일한 선택 규약이다.

- [ ] **Step 2: main builder가 선택적 상한을 검증하고 적용하게 한다**

`framework/src/main.py`에서 native backend 생성 뒤 다음 규칙을 적용한다.

```python
max_inflight = min(config.worker_count, config.queue_capacity)
runtime_max_inflight_getter = getattr(
    runtime, "native_async_max_inflight", None
)
runtime_max_inflight = (
    runtime_max_inflight_getter()
    if callable(runtime_max_inflight_getter)
    else None
)
if runtime_max_inflight is not None:
    if type(runtime_max_inflight) is not int or runtime_max_inflight <= 0:
        raise RuntimeError(
            f"target '{target.target_id}' native_async_max_inflight() "
            "must return None or a positive int."
        )
    max_inflight = min(max_inflight, runtime_max_inflight)

completion_timeout_sec = config.flush_timeout_sec
runtime_completion_timeout_getter = getattr(
    runtime, "native_async_completion_timeout_sec", None
)
runtime_completion_timeout = (
    runtime_completion_timeout_getter()
    if callable(runtime_completion_timeout_getter)
    else None
)
if runtime_completion_timeout is not None:
    if (
        isinstance(runtime_completion_timeout, bool)
        or not isinstance(runtime_completion_timeout, (int, float))
        or not math.isfinite(float(runtime_completion_timeout))
        or runtime_completion_timeout <= 0
    ):
        raise RuntimeError(
            f"target '{target.target_id}' native_async_completion_timeout_sec() "
            "must return None or a positive finite number."
        )
    completion_timeout_sec = min(
        completion_timeout_sec,
        float(runtime_completion_timeout),
    )
```

`NativeAsyncRuntimeExecutor`에는 계산한 `max_inflight`와 `completion_timeout_sec`을 넘긴다. hook이 없는 duck-typed runtime도 기존 테스트가 유지되도록 `getattr(..., None)`로 읽고, callable hook이 없으면 override 없이 진행한다.

- [ ] **Step 3: Hailo runtime을 현재 factory 계약에 연결한다**

`framework/src/runtimes/hailo_rt.py`에 다음 메서드를 추가하고 기존 queue/deadline 계산을 그대로 사용한다.

```python
def native_async_max_batch_size(self) -> int:
    return self._configured_batch_size()

def create_native_backend(self):
    if not self.supports_native_async():
        raise RuntimeError(
            "Hailo native async inference requires the InferModel API"
        )
    return self
```

`submit_async()`가 돌려주는 adapter job ID는 진단용으로만 사용하고, permit 수명은 `NativeAsyncRuntimeExecutor` dispatch token과 main retirement lease가 관리하게 유지한다.

- [ ] **Step 4: 예전 executor 선택 경로를 제거한다**

다음 자동 병합 잔여물을 제거한다.

- `framework/src/core/runtime_executor.py`의 `create_async_runtime_executor()`
- `framework/src/main.py`의 해당 import와 별도 executor 생성 블록
- `framework/tests/test_runtime_executor.py`의 옛 factory 테스트
- `framework/tests/test_async_cli.py`의 `create_async_runtime_executor` monkeypatch 주입 테스트

`framework/src/main.py`의 `_build_async_runtime_executor()` 호출과 `InferenceEngine(runtime_executor=runtime_executor)` 경로는 그대로 유지한다.

- [ ] **Step 5: Hailo target과 문서를 현재 계약에 맞춘다**

`hailo8`과 `hailo10h` target은 다음 capability를 포함한다.

```python
capabilities=(
    "hef", "sync", "async", "native_async", "latency",
    "throughput", "monitor", "npu", "local",
)
```

`docs/hailo-async-runtime.md`의 runtime 계약을 다음 흐름으로 고친다.

```text
target native_async capability
→ HailoRuntime.create_native_backend()
→ NativeAsyncRuntimeExecutor
→ terminal retirement lease ACK
```

InferModel API가 없는 legacy VStreams는 동기 `e2e`만 지원하며, `async_queue`에서는 명시적 오류를 반환한다고 기록한다. `async_ready_timeout_ms`, `async_completion_timeout_ms`, SDK queue 기반 inflight 상한 설명은 유지한다.

- [ ] **Step 6: 새 계약 테스트와 Hailo 전체 테스트를 통과시킨다**

Run:

```bash
cd /tmp/ml-hw-benchmark-hailo-forward-port/framework
python -m pytest tests/test_async_cli.py tests/test_hailo_runtime.py tests/test_plugin_registry.py tests/test_runtime_executor.py -q
```

Expected: 모든 테스트 PASS.

- [ ] **Step 7: conflict marker와 whitespace를 검사하고 merge commit을 만든다**

Run:

```bash
git -C /tmp/ml-hw-benchmark-hailo-forward-port grep -n '<<<<<<<\|=======\|>>>>>>>' -- README.md framework/src framework/tests docs
git -C /tmp/ml-hw-benchmark-hailo-forward-port diff --check
git -C /tmp/ml-hw-benchmark-hailo-forward-port status --short
git -C /tmp/ml-hw-benchmark-hailo-forward-port add -- README.md framework docs
git -C /tmp/ml-hw-benchmark-hailo-forward-port commit -am "merge: forward-port Hailo async onto current main"
```

Expected: marker 검색과 `diff --check`는 출력이 없고, merge commit은 부모로 Hailo 브랜치와 `origin/main`을 모두 가진다. 새 파일이 있다면 commit 전에 명시적 `git add -- <path>`로 포함한다.

### Task 4: permit 교착 및 backend 회귀 검증

**Files:**
- Test: `framework/tests/test_native_async_runtime_executor.py`
- Test: `framework/tests/test_async_completion.py`
- Test: `framework/tests/test_async_engine.py`
- Test: `framework/tests/test_mobilint_native_backend.py`
- Test: `framework/tests/test_furiosa_native_backend.py`
- Test: `framework/tests/test_rbln_native_backend.py`
- Test: `framework/tests/test_hailo_runtime.py`

**Interfaces:**
- Consumes: terminal retirement lease와 Hailo current-contract merge commit
- Produces: 1 permit/1 worker 연속 요청 및 다른 native backend 무퇴행 증거

- [ ] **Step 1: 사용자가 재현한 permit=1 연속 요청 회귀 테스트를 실행한다**

Run:

```bash
cd /tmp/ml-hw-benchmark-hailo-forward-port/framework
python -m pytest tests/test_native_async_runtime_executor.py::test_native_executor_releases_completed_handoff_before_next_request -q
```

Expected: PASS이며 첫 terminal commit 전에는 inflight가 1, commit 뒤 두 번째 job이 제출된다.

- [ ] **Step 2: completion lease와 native executor 묶음을 실행한다**

Run:

```bash
python -m pytest tests/test_native_async_runtime_executor.py tests/test_async_completion.py tests/test_async_engine.py -q
```

Expected: 모든 테스트 PASS.

- [ ] **Step 3: 모든 vendor native backend 회귀 테스트를 실행한다**

Run:

```bash
python -m pytest tests/test_mobilint_native_backend.py tests/test_furiosa_native_backend.py tests/test_rbln_native_backend.py tests/test_hailo_runtime.py -q
```

Expected: 모든 테스트 PASS 또는 SDK/장비 미설치로 명시적으로 skip하며 deadlock이나 unexpected failure가 없다.

- [ ] **Step 4: 전체 테스트 스위트를 실행한다**

Run:

```bash
python -m pytest -q
```

Expected: 전체 suite PASS. CUDA나 vendor 실장 환경 부재로 실패하는 테스트가 있으면 정확한 test node와 환경 원인을 기록하고, 이번 변경과 관련된 실패는 push 전에 해결한다.

### Task 5: 이력 검증, 정상 push, Jetson 실장 확인

**Files:**
- Read: Git commit graph
- Remote update: `origin/feat/hailo-native-async`

**Interfaces:**
- Consumes: 로컬 전체 검증을 통과한 detached merge commit
- Produces: force 없이 전진한 기존 원격 브랜치와 Jetson 재현 명령

- [ ] **Step 1: merge 부모와 변경 범위를 검증한다**

Run:

```bash
git -C /tmp/ml-hw-benchmark-hailo-forward-port show --no-patch --pretty=raw HEAD
git -C /tmp/ml-hw-benchmark-hailo-forward-port merge-base --is-ancestor origin/main HEAD
git -C /tmp/ml-hw-benchmark-hailo-forward-port merge-base --is-ancestor origin/feat/hailo-native-async HEAD
git -C /tmp/ml-hw-benchmark-hailo-forward-port status --short
```

Expected: 두 ancestry 검사는 exit 0이고 worktree는 clean이다.

- [ ] **Step 2: 기존 원격 Hailo 브랜치를 일반 push로 전진시킨다**

Run:

```bash
git -C /tmp/ml-hw-benchmark-hailo-forward-port push origin HEAD:feat/hailo-native-async
```

Expected: non-fast-forward나 force 없이 원격 브랜치가 새 merge commit으로 이동한다.

- [ ] **Step 3: Jetson에서 로컬 결과 파일을 보존하고 브랜치를 갱신한다**

Run on Jetson:

```bash
cd ~/ML-HW-Benchmark-Framework
git stash push -m "jetson benchmark results before hailo forward-port" -- framework/results/benchmark_results.csv
git fetch origin
git switch feat/hailo-native-async
git pull --ff-only origin feat/hailo-native-async
```

Expected: checkout 충돌 없이 새 merge commit을 가리킨다.

- [ ] **Step 4: Hailo10H YOLOv5m 20샘플을 재검증한다**

Run on Jetson:

```bash
cd ~/ML-HW-Benchmark-Framework/framework
python src/main.py \
  --model yolov5m \
  --target hailo10h \
  --hef models/hailo/10H/yolov5m.hef \
  --dataset datasets/coco128 \
  --inference-mode async_queue \
  --scenario offline \
  --batch-size 1 \
  --worker-count 1 \
  --warmup 0 \
  --max-samples 20 \
  --min-samples 20 \
  --runtime-option async_ready_timeout_ms=10000 \
  --runtime-option async_completion_timeout_ms=10000 \
  --flush-timeout-sec 300 \
  --debug
```

Expected: 20개 요청이 모두 완료되고 `async_completed_samples=20`, `async_failed_requests=0`, `async_timed_out_requests=0`, `async_outstanding_requests=0`, `async_native_inflight=0`이며 프로세스가 정상 종료한다.

- [ ] **Step 5: Hailo10H ResNet50 20샘플을 재검증한다**

Run on Jetson:

```bash
python src/main.py \
  --model resnet50 \
  --target hailo10h \
  --hef models/hailo/10H/resnet_v1_50.hef \
  --dataset datasets/imagenet_1k \
  --inference-mode async_queue \
  --scenario offline \
  --batch-size 1 \
  --worker-count 1 \
  --warmup 0 \
  --max-samples 20 \
  --min-samples 20 \
  --runtime-option async_ready_timeout_ms=10000 \
  --runtime-option async_completion_timeout_ms=10000 \
  --flush-timeout-sec 300 \
  --debug
```

Expected: YOLOv5m과 동일하게 20개 요청, outstanding 0, native inflight 0으로 정상 종료한다.
