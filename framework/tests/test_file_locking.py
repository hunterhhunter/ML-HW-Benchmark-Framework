import builtins
import ctypes
import errno
import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import fcntl as stdlib_fcntl
import pytest

import core.file_locking as file_locking
from core.result_store import reserve_run_artifacts


def test_file_locking_prefers_stdlib_fcntl():
    assert file_locking.fcntl is stdlib_fcntl


def _load_without_fcntl(monkeypatch, *, platform="linux", flock=None):
    original_import = builtins.__import__
    def blocked_import(name, *args, **kwargs):
        if name == "fcntl":
            raise ModuleNotFoundError("No module named 'fcntl'", name="fcntl")
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", blocked_import)
    monkeypatch.setattr(sys, "platform", platform)
    if flock is not None:
        monkeypatch.setattr(ctypes, "CDLL", lambda *a, **k: SimpleNamespace(flock=flock))
    spec = importlib.util.spec_from_file_location("core.file_locking_fallback_test", Path(file_locking.__file__))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeFlock:
    def __init__(self, outcomes=(0,)):
        self.outcomes = iter(outcomes)
        self.calls = []
    def __call__(self, fd, operation):
        self.calls.append((fd, operation))
        result, error = next(self.outcomes)
        ctypes.set_errno(error)
        return result


def test_file_locking_falls_back_only_on_linux_missing_fcntl(monkeypatch, tmp_path):
    fake = FakeFlock(((0, 0),))
    fallback = _load_without_fcntl(monkeypatch, flock=fake)
    fd = os.open(tmp_path / "lock", os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fallback.fcntl.flock(fd, fallback.fcntl.LOCK_EX)
        assert fake.calls == [(fd, fallback.fcntl.LOCK_EX)]
    finally:
        os.close(fd)
    with pytest.raises(ModuleNotFoundError):
        _load_without_fcntl(monkeypatch, platform="darwin", flock=fake)


def test_file_locking_retries_eintr_and_preserves_other_errno(monkeypatch):
    fake = FakeFlock(((-1, errno.EINTR), (0, 0), (-1, errno.EACCES)))
    fallback = _load_without_fcntl(monkeypatch, flock=fake)
    fallback.fcntl.flock(7, fallback.fcntl.LOCK_EX)
    assert len(fake.calls) == 2
    with pytest.raises(OSError) as caught:
        fallback.fcntl.flock(7, fallback.fcntl.LOCK_EX)
    assert caught.value.errno == errno.EACCES


def test_result_reservation_releases_lock_after_exception(tmp_path):
    results = tmp_path / "results.csv"
    reserve_run_artifacts(results_path=results, run_id="unique001")
    with pytest.raises(Exception):
        reserve_run_artifacts(results_path=results, run_id="unique001")
    # The prior collision must not leave the index lock held.
    assert reserve_run_artifacts(results_path=results, run_id="unique002").run_id == "unique002"
