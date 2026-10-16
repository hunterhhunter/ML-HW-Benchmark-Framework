# Yocto File Locking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Regulus Yocto Python에 `fcntl` 모듈이 없어도 기존 결과 저장·artifact 예약의 파일 잠금 의미를 유지한다.

**Architecture:** 공통 `core.file_locking` 어댑터는 표준 `fcntl`이 있으면 그대로 노출하고, Linux에서 모듈이 없는 경우에만 libc `flock`을 제공한다. 두 기존 호출자는 import 경로만 바꾼다. 잠금 대상·CSV/JSON 형식·예약 알고리즘은 변경하지 않는다.

**Tech Stack:** Python 3.12, ctypes/libc, Linux flock, pytest.

**Spec:** `docs/regulus-pose-async-design.md`

## Global Constraints

- 정상 CPython의 표준 `fcntl.flock`을 우선 사용한다. fallback은 Linux에서 `ModuleNotFoundError`로 `fcntl`이 없을 때만 사용한다.
- `LOCK_SH`, `LOCK_EX`, `LOCK_NB`, `LOCK_UN` 상수와 `flock(fd, operation)` 동작을 제공한다. `EINTR`은 재시도하고 다른 errno는 `OSError`로 전달한다.
- 결과 저장과 artifact 예약의 잠금·해제 범위와 저장 형식은 유지한다. 실행 결과 파일은 커밋하지 않는다.
- 로컬 테스트 인터프리터는 `/home/swlab-youngjin/ML-HW-Benchmark-Framework/framework/.venv/bin/python`이다.

## Review Focus

- 표준 `fcntl`이 있으면 fallback이 선택되지 않아야 한다. Task 1의 우선순위 테스트.
- `fcntl`이 없고 Linux일 때만 libc 경로가 동작해야 한다. Task 1의 플랫폼 테스트.
- libc가 `EINTR`을 반환하면 재시도하고 그 외 오류는 호출자에게 errno를 보존해 전달해야 한다. Task 1의 오류 테스트.
- 예외로 저장·예약이 중단되어도 이미 획득한 잠금은 해제되어야 한다. Task 1의 기존 호출자 회귀 테스트.
- 두 프로세스가 같은 결과 경로를 동시에 예약해도 중복 run ID/충돌이 없어야 한다. Task 1의 동시성 테스트.

---

### Task 1: 공통 flock 호환 어댑터와 호출자 변경

**Files:**
- Create: `framework/src/core/file_locking.py`.
- Modify: `framework/src/core/artifact_reservation.py:5,526-563` — import 경로만 변경.
- Modify: `framework/src/core/result_store.py:11,73-77,186-190` — import 경로만 변경.
- Test: `framework/tests/test_file_locking.py`, 기존 `framework/tests/test_async_result_artifacts.py`, `framework/tests/test_result_store.py`.

**Interfaces:**
- Consumes: 표준 `fcntl` 또는 libc `flock(int, int) -> int`.
- Produces: `from core.file_locking import fcntl`; `.flock(file_descriptor: int | file-like, operation: int) -> None` 및 표준 상수.

- [ ] **Step 1: 실패 테스트 작성.** `test_file_locking_prefers_stdlib_fcntl`는 정상 Linux Python에서 표준 모듈을 사용한다. `test_file_locking_falls_back_only_on_linux_missing_fcntl`는 import 차단을 모킹해 libc 호출 및 비-Linux 실패를 확인한다. `test_file_locking_retries_eintr_and_preserves_other_errno`는 libc 반환값/errno를 모킹한다. `test_result_reservation_releases_lock_after_exception`와 기존 동시 예약 테스트는 호출자 의미를 확인한다.
- [ ] **Step 2: 실패 확인.** `cd framework && /home/swlab-youngjin/ML-HW-Benchmark-Framework/framework/.venv/bin/python -m pytest -q tests/test_file_locking.py`; 새 모듈 미존재로 실패해야 한다.
- [ ] **Step 3: 최소 구현.** `core.file_locking`에서 `fcntl` import를 시도하고, Linux의 `ModuleNotFoundError`에만 `ctypes.CDLL(None, use_errno=True).flock`을 감싼 대체 객체를 만든다. 기존 두 파일은 `from .file_locking import fcntl`로 바꾸고 나머지 잠금 로직을 유지한다.
- [ ] **Step 4: 통과 확인.** 신규 테스트와 `tests/test_result_store.py tests/test_async_result_artifacts.py` 실행; 모두 PASS. Regulus 보드 Python에서 `python3 -c 'from core.file_locking import fcntl; print(fcntl.LOCK_EX)'`를 smoke 확인한다.
- [ ] **Step 5: 이 Task의 파일만 커밋.** 메시지 `fix: Regulus Yocto Python 파일 잠금 호환`.
