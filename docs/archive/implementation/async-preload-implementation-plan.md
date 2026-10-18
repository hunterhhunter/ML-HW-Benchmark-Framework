# Async Input Preload Implementation Plan

> 당시 구현 계획의 보관본이다. 현재 비동기 실행 계약은
> [개발자 설명서](../../developer-guide/08-async-internals.md)와
> [비동기 측정 가이드](../../async-inference-queue.md)를 따른다.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 정적인 비전 입력을 async 측정 전에 mmap으로 준비하여 세 Regulus VG-E2E 모델의 runtime-only 처리량에서 이미지 전처리·압축 캐시 해제 비용을 제외한다.

**Architecture:** 타깃과 독립적인 `PreloadedMmapDataLoader`가 기존 로더를 감싼다. `main.py`는 명시적 `--async-preload-inputs`일 때 런타임 로드·측정 전에 래퍼를 만들고, 측정에는 기존 `async_queue` 엔진을 그대로 사용한다. 임시 파일과 입력 계약은 래퍼가 관리한다.

**Tech Stack:** Python 3.12, NumPy mmap, tempfile, pytest.

**Spec:** `docs/archive/implementation/regulus-pose-async-design.md`

## Global Constraints

- 옵션 이름은 `--async-preload-inputs`, 기본값은 꺼짐이며 `async_queue` 전용이다. `e2e`에서 지정하면 즉시 거부한다.
- 단일 정적 NumPy 텐서 입력만 허용한다. 샘플 순서·label·`preprocess_context`·image ID를 보존한다.
- 사전 적재는 측정 시작 전 완료한다. 보고 지표는 `async_runtime_completed_samples_per_sec`이며 순수 NPU 처리량이라고 부르지 않는다.
- 모델·데이터·임시 mmap·벤치마크 결과는 커밋하지 않는다.
- 로컬 테스트 인터프리터는 `python`이다.

## Review Focus

- `max_samples`가 데이터셋보다 크거나 0이면 부분 mmap을 만들지 않고 실패해야 한다. Task 1의 샘플 수 테스트.
- 샘플 중간에 shape/dtype이 바뀌거나 dict/객체 dtype 입력이 나타나면 실패 후 임시 파일을 정리해야 한다. Task 1의 실패 정리 테스트.
- mmap의 input view를 읽은 뒤 metadata·label·원본 좌표 문맥이 원본 로더와 같아야 한다. Task 1의 round-trip 테스트.
- 디스크 잔여 용량이 필요한 mmap 크기보다 작으면 측정 시작 전에 실패해야 한다. Task 1의 disk-space 테스트.
- 옵션을 `e2e`나 지원하지 않는 동적 입력에 주면 장치 로드 전에 오류가 나야 한다. Task 2의 CLI 테스트.

---

### Task 1: 정적 텐서 mmap 로더

**Files:**
- Create: `framework/src/dataloader/preloaded_mmap_loader.py`.
- Test: `framework/tests/test_preloaded_mmap_loader.py`.

**Interfaces:**
- Consumes: 기존 `DataLoader.load_by_index(index) -> dict`, `get_metadata()["total_samples"]`, NumPy 정적 배열.
- Produces: `PreloadedMmapDataLoader(inner: DataLoader, *, sample_count: int, directory: str | Path, progress_interval: int = 0)`; `load_by_index`, `load_single`, `load_batch`, `get_labels`, `get_metadata`, `close`.

- [ ] **Step 1: 실패 테스트 작성.** `test_preloaded_loader_preserves_inputs_and_labels`는 3개 서로 다른 샘플의 배열 값·순서·label/context·shape/dtype을 왕복 확인한다. `test_preloaded_loader_rejects_invalid_count_and_tensor_contract`는 0/초과/가변 shape/dtype/dict/object를 거부한다. `test_preloaded_loader_cleans_tempfile_on_error_and_close`는 실패와 명시적 종료 뒤 임시 파일이 없음을 확인한다. `test_preloaded_loader_rejects_insufficient_disk_space`는 디스크 용량 조회를 모킹한다.
- [ ] **Step 2: 실패 확인.** `cd framework && python -m pytest -q tests/test_preloaded_mmap_loader.py`; 새 클래스 미존재로 실패해야 한다.
- [ ] **Step 3: 최소 구현.** 첫 입력에서 전체 `(sample_count, *shape)` mmap 바이트 수를 산출하고 용량 검사 후 파일을 만든다. 모든 샘플의 텐서를 저장하되 label/context는 메모리에 보존한다. `close()`는 반복 호출에 안전하고 정상 종료·생성 실패·프로세스 종료 시 임시 파일을 정리한다.
- [ ] **Step 4: 통과 확인.** 위 테스트와 `tests/test_async_producers.py tests/test_async_engine.py` 실행; 모두 PASS.
- [ ] **Step 5: 이 Task의 파일만 커밋.** 메시지 `feat: async 입력 mmap 사전 적재 로더 추가`.

### Task 2: CLI·결과 경계에 사전 적재 연결

**Files:**
- Modify: `framework/src/main.py:871-1010,3750-3820` — 옵션, 모드 검증, 로더 래핑, sidecar 메타데이터.
- Modify: `framework/README.md` — 사전 적재와 Runtime-call 경계 설명.
- Test: `framework/tests/test_async_cli.py`, `framework/tests/test_async_two_pass_cli.py`, `framework/tests/test_main_paths.py`, `framework/tests/test_async_preload_cli.py`.

**Interfaces:**
- Consumes: Task 1의 `PreloadedMmapDataLoader`와 기존 `async_queue` 엔진.
- Produces: `--async-preload-inputs` CLI 옵션; details sidecar의 `run.async_preload_inputs`, `run.async_preload_sample_count`, `run.async_preload_input_shape`, `run.async_preload_input_dtype`.

- [ ] **Step 1: 실패 테스트 작성.** `test_async_preload_is_opt_in_and_e2e_rejects_it`는 기본값 false와 E2E 거부를 확인한다. `test_async_preload_finishes_before_runtime_load_and_measurement`는 mock 로더의 `load_by_index`가 런타임 로드/측정 전에 3번 호출됨을 확인한다. `test_async_preload_sidecar_records_tensor_contract`는 사용 여부·샘플 수·shape·dtype을 확인한다. `test_async_preload_rejects_dynamic_inputs_before_runtime_load`는 장치 미획득을 확인한다.
- [ ] **Step 2: 실패 확인.** `cd framework && python -m pytest -q tests/test_async_preload_cli.py`; 옵션 미존재로 실패해야 한다.
- [ ] **Step 3: 최소 구현.** `validate_async_args()`에 `e2e` 거부를 추가한다. 로더 검증 직후, 런타임 생성·`load()` 전에 `max_samples` 또는 전체 데이터셋 크기로 mmap을 완성한다. sidecar에 위 메타데이터를 남기고 종료 경로에서 `close()`를 호출한다. 기존 CSV 계측 필드와 `async_measurement_boundary=framework_runtime_call_return`은 바꾸지 않는다.
- [ ] **Step 4: 통과 확인.** 신규 테스트와 `tests/test_async_cli.py tests/test_async_two_pass_cli.py tests/test_async_runtime_boundary.py tests/test_main_paths.py` 실행; 모두 PASS.
- [ ] **Step 5: 이 Task의 파일만 커밋.** 메시지 `feat: async_queue 입력 사전 적재 옵션 연결`.

## 실기기 인계

Regulus에서 `--async-pass runtime-only --async-preload-inputs --max-samples 5000 --min-samples 5000 --worker-count 4 --queue-capacity 256`으로 세 모델을 각각 실행한다. 모델별 새 결과 디렉터리를 사용하고, 측정 전 적재 로그와 측정 중 producer load·queue depth를 함께 보존한다. 5,000 완료 및 오류 0을 확인하되 처리량 수치가 과거와 정확히 같아야 한다고 가정하지 않는다.
