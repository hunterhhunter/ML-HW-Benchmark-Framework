import csv
import errno
import fcntl
import json
import multiprocessing
import os
import re
import stat
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import replace
from enum import Enum
from pathlib import Path

import numpy as np
import pytest

import core.artifact_reservation as artifact_reservation_module
import core.async_inference.trace as trace_module
import core.result_store as result_store_module
from core.async_inference.trace import RequestTraceWriter
from core.async_inference.types import RequestTrace, TerminalStatus
from core.result_store import (
    RunArtifactReservation,
    create_run_id,
    get_reserved_result_state,
    get_result,
    load_results,
    reserve_run_artifacts,
    save_async_details as _save_async_details,
    save_async_failure_details,
    save_result,
)
from core.result_store import delete_result


def make_trace(**changes):
    values = {
        "request_id": 1,
        "sample_index": 2,
        "status": TerminalStatus.COMPLETED,
        "scheduled_ns": 1,
        "issued_ns": 2,
        "enqueued_ns": 3,
        "runtime_started_ns": 4,
        "runtime_finished_ns": 5,
        "completed_ns": 6,
        "worker_id": 0,
        "batch_size": 1,
        "timed_out": False,
        "sample_count": 3,
        "error_type": None,
        "error_message": None,
    }
    values.update(changes)
    return RequestTrace(**values)


def make_trace_writer(tmp_path, capacity=1024, run_id="fixed123"):
    reservation = reserve_run_artifacts(
        results_path=tmp_path / "results.csv",
        run_id=run_id,
    )
    path = reservation.trace_path
    return (
        RequestTraceWriter(
            path,
            capacity=capacity,
            reservation=reservation,
        ),
        path,
        reservation,
    )


def save_minimal_result(csv_path, **changes):
    values = {
        "metrics": {"accuracy": 1.0},
        "model_name": "tiny",
        "task": "IMAGE_CLASSIFICATION",
        "backend": "onnxruntime",
        "device": "cpu",
        "batch_size": 1,
        "warmup_runs": 0,
        "results_path": csv_path,
    }
    values.update(changes)
    return save_result(**values)


_AUTO_RESERVATION = object()
_TEST_RESERVATIONS = {}


def save_async_details(
    run_id,
    details,
    results_dir=None,
    reservation=_AUTO_RESERVATION,
):
    """R1 test adapter: every successful sidecar call owns a reservation."""
    if reservation is _AUTO_RESERVATION:
        if type(run_id) is not str or re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}",
            run_id,
        ) is None:
            return _save_async_details(
                run_id,
                details,
                results_dir=results_dir,
            )
        root = (
            Path(results_dir)
            if results_dir is not None
            else result_store_module.DEFAULT_RESULTS_DIR
        )
        key = (os.getpid(), str(root.absolute()), run_id)
        reservation = _TEST_RESERVATIONS.get(key)
        if reservation is None:
            reservation = reserve_run_artifacts(
                results_path=root / "benchmark_results.csv",
                run_id=run_id,
            )
            _TEST_RESERVATIONS[key] = reservation
    return _save_async_details(
        run_id,
        details,
        results_dir=results_dir,
        reservation=reservation,
    )


def install_cleanup_swap(
    monkeypatch,
    module,
    target,
    *,
    active=lambda: True,
):
    """Swap a cleanup source immediately before its no-overwrite move."""
    target = Path(target)
    replacement = target.with_name(f".{target.name}.replacement")
    replacement.write_text("replacement", encoding="utf-8")
    real_replace = module.os.replace
    real_rename_noreplace = artifact_reservation_module._rename_noreplace
    state = {"swapped": False}

    def swap_target():
        real_replace(replacement, target)
        state["swapped"] = True

    def swap_then_quarantine(
        source_directory_fd,
        source,
        target_directory_fd,
        destination,
    ):
        if (
            not state["swapped"]
            and active()
            and source == target.name
            and str(destination).endswith(".quarantine")
        ):
            swap_target()
        return real_rename_noreplace(
            source_directory_fd,
            source,
            target_directory_fd,
            destination,
        )

    monkeypatch.setattr(
        artifact_reservation_module,
        "_rename_noreplace",
        swap_then_quarantine,
    )
    return state


def install_effective_close_failure(
    monkeypatch,
    module,
    target_descriptor,
    message,
):
    """Close one fd, reuse its number, then report the close as failed."""
    real_close = module.os.close
    state = {"failed": False}

    def close_then_raise(file_descriptor):
        if (
            not state["failed"]
            and file_descriptor == target_descriptor()
        ):
            state["failed"] = True
            real_close(file_descriptor)
            source = os.open("/dev/null", os.O_RDONLY)
            if source != file_descriptor:
                os.dup2(source, file_descriptor)
                real_close(source)
            state["sentinel_fd"] = file_descriptor
            raise OSError(message)
        return real_close(file_descriptor)

    monkeypatch.setattr(module.os, "close", close_then_raise)
    return state, real_close


def make_owned_cleanup_target(tmp_path):
    target = tmp_path / "owned.json"
    target.write_text("owned", encoding="utf-8")
    opened = target.stat()
    return target, (opened.st_dev, opened.st_ino)


def unlink_owned_target(target, expected_identity):
    directory = artifact_reservation_module.open_trusted_directory(
        target.parent,
        create=False,
    )
    try:
        return artifact_reservation_module._unlink_owned_entry(
            directory.file_descriptor,
            target.name,
            expected_identity,
            "owned artifact",
            directory_path=target.parent,
        )
    finally:
        directory.close()


def assert_cleanup_evidence(
    error,
    original_path,
    *,
    restored,
    recovery_path=None,
    unsupported=False,
):
    assert error.cleanup_original_path == str(original_path)
    assert error.cleanup_original_preserved is True
    assert error.cleanup_original_restored is restored
    if unsupported:
        assert error.cleanup_operation_unsupported is True
    if recovery_path is None:
        assert not hasattr(error, "cleanup_recovery_path")
    else:
        assert error.cleanup_recovery_path == str(recovery_path)
        assert recovery_path.exists()


def install_retained_quarantine_unlink_failure(
    monkeypatch,
    original_path,
    message,
):
    real_unlink = artifact_reservation_module.os.unlink

    def retain_quarantine(target, *args, **kwargs):
        if str(target).endswith(".quarantine"):
            original_path.write_text("restore collision", encoding="utf-8")
            raise OSError(message)
        return real_unlink(target, *args, **kwargs)

    monkeypatch.setattr(
        artifact_reservation_module.os,
        "unlink",
        retain_quarantine,
    )


def assert_retained_quarantine_secondary(error, phase, original_path):
    secondary = next(
        item
        for item in error.persistence_secondary_errors
        if item["phase"] == phase
    )
    quarantines = list(
        original_path.parent.glob(".artifact-cleanup-*.quarantine")
    )
    assert len(quarantines) == 1
    assert secondary["cleanup_recovery_path"] == str(quarantines[0])
    assert Path(secondary["cleanup_recovery_path"]).exists()
    assert secondary["cleanup_original_path"] == str(original_path)
    assert secondary["cleanup_original_preserved"] is True
    assert secondary["cleanup_original_restored"] is False


def assert_descriptor_close_secondary(error, phase, message):
    assert error.persistence_secondary_errors == [
        {
            "phase": phase,
            "error_type": "OSError",
            "error_message": message,
            "descriptor_close_state_uncertain": True,
        }
    ]


def install_cleanup_rename_interference(monkeypatch, target, scenario):
    replacement = target.with_name("replacement.json")
    replacement.write_text("replacement", encoding="utf-8")
    real_rename_noreplace = artifact_reservation_module._rename_noreplace
    state = {}

    def interfere(
        source_directory_fd,
        source_name,
        target_directory_fd,
        target_name,
    ):
        if scenario == "initial-destination-collision" and not state:
            collision = target.parent / target_name
            collision.write_text("collision", encoding="utf-8")
            state["collision"] = collision
        elif (
            scenario != "initial-destination-collision"
            and source_name == target.name
            and "quarantine" not in state
        ):
            os.replace(replacement, target)
            state["quarantine"] = target.parent / target_name
        elif (
            scenario == "restore-destination-collision"
            and str(source_name).endswith(".quarantine")
        ):
            target.write_text("restore collision", encoding="utf-8")
        return real_rename_noreplace(
            source_directory_fd,
            source_name,
            target_directory_fd,
            target_name,
        )

    monkeypatch.setattr(
        artifact_reservation_module,
        "_rename_noreplace",
        interfere,
    )
    return state


def save_result_process(csv_path, prefix, start, count):
    assert start.wait(5.0)
    for index in range(count):
        save_minimal_result(
            Path(csv_path),
            run_id=f"{prefix}{index:04d}",
            model_name=f"{prefix}-{index}",
        )


def save_duplicate_result_process(csv_path, start, outcomes):
    assert start.wait(5.0)
    try:
        save_minimal_result(Path(csv_path), run_id="shared123")
    except BaseException as exc:
        outcomes.put(("error", type(exc).__name__))
    else:
        outcomes.put(("ok", "shared123"))


def save_sidecar_process(results_dir, start, generation, outcomes):
    assert start.wait(5.0)
    try:
        save_async_details(
            "shared123",
            {"generation": generation},
            results_dir=Path(results_dir),
        )
    except BaseException as exc:
        outcomes.put(("error", type(exc).__name__))
    else:
        outcomes.put(("ok", generation))


def reserve_artifacts_process(results_path, start, outcomes):
    assert start.wait(5.0)
    try:
        reservation = reserve_run_artifacts(
            results_path=Path(results_path),
            run_id="shared123",
        )
    except BaseException as exc:
        outcomes.put(("error", type(exc).__name__))
    else:
        outcomes.put(("ok", reservation.owner_token))


def mixed_owner_bundle_process(reservation, use_forged_owner, start, outcomes):
    assert start.wait(5.0)
    selected = (
        replace(reservation, owner_token="0" * 64)
        if use_forged_owner
        else reservation
    )
    try:
        if use_forged_owner:
            save_minimal_result(
                reservation.results_path,
                run_id=reservation.run_id,
                inference_mode="async_queue",
                reservation=selected,
            )
        else:
            save_async_details(
                reservation.run_id,
                {"owner": "marker"},
                results_dir=reservation.results_root,
                reservation=selected,
            )
    except BaseException as exc:
        outcomes.put(("error", use_forged_owner, type(exc).__name__))
    else:
        outcomes.put(("ok", use_forged_owner, reservation.run_id))


def test_reserve_run_artifacts_creates_durable_owner_marker(tmp_path):
    results_path = tmp_path / "results" / "benchmark_results.csv"

    reservation = reserve_run_artifacts(
        results_path=results_path,
        run_id="fixed123",
    )

    assert type(reservation) is RunArtifactReservation
    assert reservation.run_id == "fixed123"
    assert reservation.results_root == results_path.parent.absolute()
    assert reservation.results_path == results_path.absolute()
    assert re.fullmatch(r"[0-9a-f]{64}", reservation.owner_token)
    assert reservation.details_path == (
        results_path.parent / "details" / "fixed123.json"
    ).absolute()
    assert reservation.trace_path == (
        results_path.parent / "traces" / "fixed123.jsonl"
    ).absolute()
    marker = reservation.marker_path
    assert marker.exists()
    assert stat.S_IMODE(marker.stat().st_mode) == 0o600
    assert json.loads(marker.read_text(encoding="utf-8")) == {
        "lease_device": reservation.lease_device,
        "lease_inode": reservation.lease_inode,
        "owner_token": reservation.owner_token,
        "results_path": str(reservation.results_path),
        "results_root": str(reservation.results_root),
        "run_id": "fixed123",
        "schema_version": "1.0",
    }


def test_reservation_creates_persistent_per_run_lock(tmp_path):
    reservation = reserve_run_artifacts(
        results_path=tmp_path / "results.csv",
        run_id="fixed123",
    )

    lock_path = reservation.marker_path.with_suffix(".lock")

    assert lock_path.is_file()
    assert stat.S_IMODE(lock_path.stat().st_mode) == 0o600


def test_replaced_run_lock_is_rejected_on_next_artifact_operation(tmp_path):
    reservation = reserve_run_artifacts(
        results_path=tmp_path / "results.csv",
        run_id="fixed123",
    )
    lock_path = reservation.marker_path.with_suffix(".lock")
    replacement = lock_path.with_suffix(".replacement")
    replacement.write_bytes(b"")
    os.replace(replacement, lock_path)

    with pytest.raises(ValueError, match="lease identity"):
        save_async_details(
            reservation.run_id,
            {"value": 1},
            results_dir=reservation.results_root,
            reservation=reservation,
        )

    assert not reservation.details_path.exists()


def test_legacy_lease_unbound_marker_fails_closed_without_migration(tmp_path):
    reservation = reserve_run_artifacts(
        results_path=tmp_path / "results.csv",
        run_id="fixed123",
    )
    legacy = replace(reservation)
    object.__setattr__(legacy, "lease_device", None)
    object.__setattr__(legacy, "lease_inode", None)
    original_marker = reservation.marker_path.read_bytes()

    with pytest.raises(ValueError, match="legacy.*lease identity"):
        save_async_details(
            reservation.run_id,
            {"value": 1},
            results_dir=reservation.results_root,
            reservation=legacy,
        )

    assert reservation.marker_path.read_bytes() == original_marker
    assert not reservation.details_path.exists()


def test_reservation_marker_file_fsync_failure_leaves_no_marker_or_temp(
    tmp_path,
    monkeypatch,
):
    results_path = tmp_path / "results.csv"
    marker_path = tmp_path / ".run_artifacts" / "fixed123.json"
    real_fsync = artifact_reservation_module.os.fsync

    def fail_marker_file_fsync(file_descriptor):
        target = os.readlink(f"/proc/self/fd/{file_descriptor}")
        if "fixed123.json" in Path(target).name:
            raise OSError("marker file fsync failed")
        return real_fsync(file_descriptor)

    monkeypatch.setattr(
        artifact_reservation_module.os,
        "fsync",
        fail_marker_file_fsync,
    )
    with pytest.raises(OSError, match="marker file fsync failed"):
        reserve_run_artifacts(results_path=results_path, run_id="fixed123")

    assert not marker_path.exists()
    assert not list(marker_path.parent.glob("*fixed123.json*.tmp"))


def test_reservation_marker_parent_relocation_rolls_back_pinned_final(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "root"
    relocated = tmp_path / "relocated-root"
    results_path = root / "results.csv"
    real_link = artifact_reservation_module.os.link
    swapped = False

    def link_then_relocate(source, target, *args, **kwargs):
        nonlocal swapped
        result = real_link(source, target, *args, **kwargs)
        if target == "fixed123.json" and not swapped:
            swapped = True
            root.rename(relocated)
            root.mkdir()
            (root / ".run_artifacts").mkdir()
        return result

    monkeypatch.setattr(
        artifact_reservation_module.os,
        "link",
        link_then_relocate,
    )
    with pytest.raises(ValueError, match="identity changed"):
        reserve_run_artifacts(results_path=results_path, run_id="fixed123")

    assert not (root / ".run_artifacts" / "fixed123.json").exists()
    assert not (relocated / ".run_artifacts" / "fixed123.json").exists()
    assert not list((relocated / ".run_artifacts").glob("*.tmp"))


def test_reservation_marker_directory_fsync_failure_rolls_back_final(
    tmp_path,
    monkeypatch,
):
    results_path = tmp_path / "results.csv"
    marker_path = tmp_path / ".run_artifacts" / "fixed123.json"
    real_fsync = artifact_reservation_module.os.fsync
    failed = False

    def fail_marker_directory_fsync(file_descriptor):
        nonlocal failed
        target = os.readlink(f"/proc/self/fd/{file_descriptor}")
        if (
            not failed
            and target.endswith("/.run_artifacts")
            and marker_path.exists()
        ):
            failed = True
            raise OSError("marker directory fsync failed")
        return real_fsync(file_descriptor)

    monkeypatch.setattr(
        artifact_reservation_module.os,
        "fsync",
        fail_marker_directory_fsync,
    )
    with pytest.raises(OSError, match="marker directory fsync failed"):
        reserve_run_artifacts(results_path=results_path, run_id="fixed123")

    assert not marker_path.exists()
    assert not list(marker_path.parent.glob("*.tmp"))


def test_reservation_marker_replacement_after_link_is_not_unlinked(
    tmp_path,
    monkeypatch,
):
    results_path = tmp_path / "results.csv"
    marker_path = tmp_path / ".run_artifacts" / "fixed123.json"
    real_link = artifact_reservation_module.os.link

    def link_then_replace_marker(source, target, *args, **kwargs):
        result = real_link(source, target, *args, **kwargs)
        if target == marker_path.name:
            replacement = marker_path.with_suffix(".replacement")
            replacement.write_text("replacement", encoding="utf-8")
            os.replace(replacement, marker_path)
        return result

    monkeypatch.setattr(
        artifact_reservation_module.os,
        "link",
        link_then_replace_marker,
    )

    with pytest.raises(ValueError, match="marker identity") as raised:
        reserve_run_artifacts(results_path=results_path, run_id="fixed123")

    assert marker_path.read_text(encoding="utf-8") == "replacement"
    assert raised.value.publication_state_uncertain is True
    assert raised.value.marker_file_may_remain is True
    assert type(raised.value.reservation_recovery) is RunArtifactReservation


def test_uncertain_reservation_marker_failure_exposes_explicit_recovery(
    tmp_path,
    monkeypatch,
):
    results_path = tmp_path / "results.csv"
    marker_path = tmp_path / ".run_artifacts" / "fixed123.json"
    real_fsync = artifact_reservation_module.os.fsync
    real_unlink = artifact_reservation_module.os.unlink
    failed_fsync = False

    def fail_marker_directory_fsync(file_descriptor):
        nonlocal failed_fsync
        target = os.readlink(f"/proc/self/fd/{file_descriptor}")
        if (
            not failed_fsync
            and target.endswith("/.run_artifacts")
            and marker_path.exists()
        ):
            failed_fsync = True
            raise OSError("marker directory fsync failed")
        return real_fsync(file_descriptor)

    def fail_marker_rollback(target, *args, **kwargs):
        if str(target).endswith(".quarantine"):
            raise OSError("marker rollback failed")
        return real_unlink(target, *args, **kwargs)

    monkeypatch.setattr(
        artifact_reservation_module.os,
        "fsync",
        fail_marker_directory_fsync,
    )
    monkeypatch.setattr(
        artifact_reservation_module.os,
        "unlink",
        fail_marker_rollback,
    )
    with pytest.raises(OSError, match="marker directory fsync failed") as raised:
        reserve_run_artifacts(results_path=results_path, run_id="fixed123")

    assert raised.value.publication_state_uncertain is True
    assert raised.value.marker_file_may_remain is True
    recovery = raised.value.reservation_recovery
    assert type(recovery) is RunArtifactReservation
    recovered = result_store_module.recover_run_artifact_reservation(recovery)
    assert recovered == recovery


@pytest.mark.parametrize("artifact", ["marker", "state"])
def test_generic_rollback_recovery_reports_existing_quarantine(
    tmp_path,
    monkeypatch,
    artifact,
):
    results_path = tmp_path / "results.csv"
    reservation = (
        reserve_run_artifacts(results_path=results_path, run_id="fixed123")
        if artifact == "state"
        else None
    )
    original_path = (
        reservation.pending_path
        if reservation is not None
        else tmp_path / ".run_artifacts" / "fixed123.json"
    )
    real_fsync = artifact_reservation_module.os.fsync
    failed_fsync = False

    def fail_directory_fsync(file_descriptor):
        nonlocal failed_fsync
        target = os.readlink(f"/proc/self/fd/{file_descriptor}")
        if (
            not failed_fsync
            and target.endswith("/.run_artifacts")
            and original_path.exists()
        ):
            failed_fsync = True
            raise OSError(f"{artifact} directory fsync failed")
        return real_fsync(file_descriptor)

    monkeypatch.setattr(
        artifact_reservation_module.os,
        "fsync",
        fail_directory_fsync,
    )
    install_retained_quarantine_unlink_failure(
        monkeypatch,
        original_path,
        f"{artifact} rollback failed",
    )
    with pytest.raises(
        OSError,
        match=f"{artifact} directory fsync failed",
    ) as raised:
        if reservation is None:
            reserve_run_artifacts(results_path=results_path, run_id="fixed123")
        else:
            with artifact_reservation_module.verify_reservation(
                reservation,
                reservation.run_id,
                results_path=reservation.results_path,
                require_active=False,
            ) as verified:
                artifact_reservation_module.publish_pending(
                    verified,
                    "a" * 64,
                    "transaction-time",
                )

    assert_retained_quarantine_secondary(
        raised.value,
        (
            "rollback_reservation_marker"
            if reservation is None
            else "rollback_state"
        ),
        original_path,
    )


def test_reservation_creation_lease_release_failure_exposes_recovery(
    tmp_path,
    monkeypatch,
):
    results_path = tmp_path / "results.csv"
    marker_path = tmp_path / ".run_artifacts" / "fixed123.json"
    real_flock = artifact_reservation_module.fcntl.flock
    failed = False

    def fail_release_after_marker(file_descriptor, operation):
        nonlocal failed
        if operation == fcntl.LOCK_UN and marker_path.exists() and not failed:
            failed = True
            raise OSError("reservation lease release failed")
        return real_flock(file_descriptor, operation)

    monkeypatch.setattr(
        artifact_reservation_module.fcntl,
        "flock",
        fail_release_after_marker,
    )

    with pytest.raises(OSError, match="lease release failed") as raised:
        reserve_run_artifacts(results_path=results_path, run_id="fixed123")

    assert marker_path.exists()
    recovery = raised.value.reservation_recovery
    assert recovery.marker_path == marker_path
    assert raised.value.publication_state_uncertain is True
    assert raised.value.marker_file_may_remain is True
    assert result_store_module.recover_run_artifact_reservation(recovery) is recovery
    assert marker_path.exists()


def test_reservation_run_id_is_never_allocated_twice(tmp_path):
    results_path = tmp_path / "results.csv"
    first = reserve_run_artifacts(results_path=results_path, run_id="fixed123")
    original = first.marker_path.read_bytes()

    with pytest.raises(FileExistsError, match="reserved"):
        reserve_run_artifacts(results_path=results_path, run_id="fixed123")

    assert first.marker_path.read_bytes() == original


def test_reservation_rejects_run_id_already_present_in_legacy_csv(tmp_path):
    results_path = tmp_path / "results.csv"
    results_path.write_text(
        "run_id,model_name\nfixed123,legacy\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="already exists"):
        reserve_run_artifacts(results_path=results_path, run_id="fixed123")

    assert not (tmp_path / ".run_artifacts" / "fixed123.json").exists()


def test_generated_reservation_id_retries_marker_and_csv_collisions(
    tmp_path,
    monkeypatch,
):
    results_path = tmp_path / "results.csv"
    results_path.write_text(
        "run_id,model_name\ncsv00001,legacy\n",
        encoding="utf-8",
    )
    reserve_run_artifacts(results_path=results_path, run_id="mark0001")
    candidates = iter(("csv00001", "mark0001", "fresh123"))
    monkeypatch.setattr(result_store_module, "create_run_id", lambda: next(candidates))

    reservation = reserve_run_artifacts(results_path=results_path)

    assert reservation.run_id == "fresh123"


def test_multiprocess_reservation_collision_has_one_owner(tmp_path):
    results_path = tmp_path / "results.csv"
    context = multiprocessing.get_context("spawn")
    start = context.Event()
    outcomes = context.Queue()
    processes = [
        context.Process(
            target=reserve_artifacts_process,
            args=(str(results_path), start, outcomes),
        )
        for _ in range(2)
    ]
    for process in processes:
        process.start()
    start.set()
    for process in processes:
        process.join(10.0)
        assert process.exitcode == 0

    results = [outcomes.get(timeout=1.0) for _ in processes]
    assert sorted(outcome for outcome, _ in results) == ["error", "ok"]
    assert next(value for outcome, value in results if outcome == "error") == (
        "FileExistsError"
    )


@pytest.mark.parametrize("relative", [False, True])
def test_reservation_rejects_intermediate_symlink_components(
    tmp_path,
    monkeypatch,
    relative,
):
    outside = tmp_path / "outside"
    outside.mkdir()
    link = tmp_path / "linked"
    link.symlink_to(outside, target_is_directory=True)
    if relative:
        monkeypatch.chdir(tmp_path)
        results_path = Path("linked") / "nested" / "results.csv"
    else:
        results_path = link / "nested" / "results.csv"

    with pytest.raises((OSError, ValueError), match="symlink"):
        reserve_run_artifacts(results_path=results_path, run_id="fixed123")

    assert list(outside.iterdir()) == []


def test_reservation_value_does_not_expose_owner_in_repr(tmp_path):
    reservation = reserve_run_artifacts(
        results_path=tmp_path / "results.csv",
        run_id="fixed123",
    )

    assert reservation.owner_token not in repr(reservation)
    forged = replace(reservation, owner_token="0" * 64)
    assert forged != reservation


def test_async_result_requires_reservation_before_filesystem_side_effects(tmp_path):
    results_path = tmp_path / "missing" / "results.csv"

    with pytest.raises(ValueError, match="reservation"):
        save_minimal_result(
            results_path,
            run_id="fixed123",
            inference_mode="async_queue",
        )

    assert not results_path.parent.exists()


def test_async_details_requires_reservation_before_artifact_side_effects(tmp_path):
    with pytest.raises(ValueError, match="reservation"):
        result_store_module.save_async_details(
            "fixed123",
            {"value": 1},
            results_dir=tmp_path / "missing",
        )

    assert not (tmp_path / "missing").exists()


@pytest.mark.parametrize("mismatch", ["token", "run_id", "root"])
def test_async_details_rejects_mismatched_reservation_without_artifacts(
    tmp_path,
    mismatch,
):
    reservation = reserve_run_artifacts(
        results_path=tmp_path / "owner" / "results.csv",
        run_id="fixed123",
    )
    selected = reservation
    run_id = reservation.run_id
    results_root = reservation.results_root
    if mismatch == "token":
        selected = replace(reservation, owner_token="0" * 64)
    elif mismatch == "run_id":
        run_id = "other123"
    else:
        results_root = tmp_path / "different" / "root"

    with pytest.raises(ValueError, match="reservation"):
        save_async_details(
            run_id,
            {"value": 1},
            results_dir=results_root,
            reservation=selected,
        )

    assert not (reservation.results_root / "details").exists()
    assert not (tmp_path / "different").exists()


def test_reserved_artifacts_reject_intermediate_symlink_swap(tmp_path):
    base = tmp_path / "base"
    results_path = base / "root" / "results.csv"
    reservation = reserve_run_artifacts(
        results_path=results_path,
        run_id="fixed123",
    )
    relocated = tmp_path / "relocated-base"
    base.rename(relocated)
    base.symlink_to(relocated, target_is_directory=True)

    with pytest.raises((OSError, ValueError), match="symlink"):
        save_async_details(
            reservation.run_id,
            {"value": 1},
            results_dir=reservation.results_root,
            reservation=reservation,
        )
    with pytest.raises((OSError, ValueError), match="symlink"):
        RequestTraceWriter(
            reservation.trace_path,
            reservation=reservation,
        )

    assert not (relocated / "root" / "details").exists()
    assert not (relocated / "root" / "traces").exists()


def test_one_reservation_persists_details_then_csv_and_is_consumed(tmp_path):
    results_path = tmp_path / "results.csv"
    reservation = reserve_run_artifacts(
        results_path=results_path,
        run_id="fixed123",
    )

    details_path = save_async_details(
        reservation.run_id,
        {"value": 1},
        results_dir=reservation.results_root,
        reservation=reservation,
    )
    saved_run_id = save_minimal_result(
        results_path,
        run_id=reservation.run_id,
        inference_mode="async_queue",
        reservation=reservation,
    )

    assert saved_run_id == reservation.run_id
    assert details_path == reservation.details_path
    assert reservation.consumed_path.exists()
    assert load_results(results_path=results_path)[0]["run_id"] == "fixed123"
    assert delete_result("fixed123", results_path=results_path) is True
    with pytest.raises(FileExistsError, match="reserved"):
        reserve_run_artifacts(results_path=results_path, run_id="fixed123")
    with pytest.raises(ValueError, match="consumed"):
        save_async_details(
            reservation.run_id,
            {"value": 2},
            results_dir=reservation.results_root,
            reservation=reservation,
        )


def test_csv_pre_replace_failure_keeps_pending_and_retry_resumes(
    tmp_path,
    monkeypatch,
):
    results_path = tmp_path / "results.csv"
    reservation = reserve_run_artifacts(
        results_path=results_path,
        run_id="fixed123",
    )
    pending_path = reservation.marker_path.with_suffix(".pending")
    real_replace = result_store_module.os.replace

    def fail_csv_replace(source, target, *args, **kwargs):
        if target == results_path.name:
            raise OSError("CSV replace failed")
        return real_replace(source, target, *args, **kwargs)

    monkeypatch.setattr(result_store_module.os, "replace", fail_csv_replace)
    with pytest.raises(OSError, match="CSV replace failed") as raised:
        save_minimal_result(
            results_path,
            run_id=reservation.run_id,
            inference_mode="async_queue",
            reservation=reservation,
        )

    assert raised.value.csv_commit_recovery_pending is True
    assert getattr(raised.value, "publication_state_uncertain", False) is False
    assert pending_path.exists()
    assert not reservation.consumed_path.exists()
    assert not results_path.exists()

    monkeypatch.setattr(result_store_module.os, "replace", real_replace)
    assert save_minimal_result(
        results_path,
        run_id=reservation.run_id,
        inference_mode="async_queue",
        reservation=reservation,
    ) == reservation.run_id
    assert [row["run_id"] for row in load_results(results_path=results_path)] == [
        reservation.run_id
    ]
    assert reservation.consumed_path.exists()
    assert not pending_path.exists()


def test_csv_pending_rejects_unrelated_same_id_row_without_consuming(
    tmp_path,
    monkeypatch,
):
    results_path = tmp_path / "results.csv"
    reservation = reserve_run_artifacts(
        results_path=results_path,
        run_id="fixed123",
    )
    real_replace = result_store_module.os.replace

    def fail_csv_replace(source, target, *args, **kwargs):
        if target == results_path.name:
            raise OSError("CSV replace failed")
        return real_replace(source, target, *args, **kwargs)

    monkeypatch.setattr(result_store_module.os, "replace", fail_csv_replace)
    with pytest.raises(OSError, match="CSV replace failed"):
        save_minimal_result(
            results_path,
            run_id=reservation.run_id,
            inference_mode="async_queue",
            reservation=reservation,
        )

    monkeypatch.setattr(result_store_module.os, "replace", real_replace)
    with open(results_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["run_id", "timestamp", "model_name"])
        writer.writerow([reservation.run_id, "forged-time", "intruder"])

    with pytest.raises(ValueError, match="transaction provenance|fingerprint"):
        save_minimal_result(
            results_path,
            run_id=reservation.run_id,
            inference_mode="async_queue",
            reservation=reservation,
        )

    assert reservation.pending_path.exists()
    assert not reservation.consumed_path.exists()
    assert load_results(results_path=results_path)[0]["model_name"] == "intruder"


def test_csv_pending_no_row_requires_same_canonical_retry_and_timestamp(
    tmp_path,
    monkeypatch,
):
    class LogicalDatetime:
        value = "first-time"

        @classmethod
        def now(cls):
            return cls()

        def strftime(self, _format):
            return self.value

    results_path = tmp_path / "results.csv"
    reservation = reserve_run_artifacts(
        results_path=results_path,
        run_id="fixed123",
    )
    real_replace = result_store_module.os.replace

    def fail_csv_replace(source, target, *args, **kwargs):
        if target == results_path.name:
            raise OSError("CSV replace failed")
        return real_replace(source, target, *args, **kwargs)

    monkeypatch.setattr(result_store_module, "datetime", LogicalDatetime)
    monkeypatch.setattr(result_store_module.os, "replace", fail_csv_replace)
    with pytest.raises(OSError, match="CSV replace failed"):
        save_minimal_result(
            results_path,
            run_id=reservation.run_id,
            inference_mode="async_queue",
            reservation=reservation,
            model_name="original",
        )

    monkeypatch.setattr(result_store_module.os, "replace", real_replace)
    LogicalDatetime.value = "later-time"
    with pytest.raises(ValueError, match="transaction provenance|fingerprint"):
        save_minimal_result(
            results_path,
            run_id=reservation.run_id,
            inference_mode="async_queue",
            reservation=reservation,
            model_name="changed",
        )

    assert save_minimal_result(
        results_path,
        run_id=reservation.run_id,
        inference_mode="async_queue",
        reservation=reservation,
        model_name="original",
    ) == reservation.run_id
    result = load_results(results_path=results_path)[0]
    assert result["timestamp"] == "first-time"
    assert result["model_name"] == "original"


def test_csv_post_replace_failure_is_uncertain_and_retry_does_not_append(
    tmp_path,
    monkeypatch,
):
    results_path = tmp_path / "results.csv"
    reservation = reserve_run_artifacts(
        results_path=results_path,
        run_id="fixed123",
    )
    pending_path = reservation.marker_path.with_suffix(".pending")
    real_fsync = result_store_module.os.fsync
    failed = False

    def fail_root_fsync_after_replace(file_descriptor):
        nonlocal failed
        opened = os.fstat(file_descriptor)
        if (
            not failed
            and stat.S_ISDIR(opened.st_mode)
            and (opened.st_dev, opened.st_ino)
            == (reservation.root_device, reservation.root_inode)
            and results_path.exists()
        ):
            failed = True
            raise OSError("CSV root fsync failed")
        return real_fsync(file_descriptor)

    monkeypatch.setattr(
        result_store_module.os,
        "fsync",
        fail_root_fsync_after_replace,
    )
    with pytest.raises(OSError, match="CSV root fsync failed") as raised:
        save_minimal_result(
            results_path,
            run_id=reservation.run_id,
            inference_mode="async_queue",
            reservation=reservation,
        )

    assert raised.value.csv_commit_recovery_pending is True
    assert raised.value.publication_state_uncertain is True
    assert pending_path.exists()
    assert not reservation.consumed_path.exists()
    assert [row["run_id"] for row in load_results(results_path=results_path)] == [
        reservation.run_id
    ]

    assert save_minimal_result(
        results_path,
        run_id=reservation.run_id,
        inference_mode="async_queue",
        reservation=reservation,
    ) == reservation.run_id
    assert [row["run_id"] for row in load_results(results_path=results_path)] == [
        reservation.run_id
    ]
    assert reservation.consumed_path.exists()
    assert not pending_path.exists()


def test_csv_consumed_publication_failure_retries_from_pending_row(
    tmp_path,
    monkeypatch,
):
    results_path = tmp_path / "results.csv"
    reservation = reserve_run_artifacts(
        results_path=results_path,
        run_id="fixed123",
    )
    pending_path = reservation.marker_path.with_suffix(".pending")
    real_link = artifact_reservation_module.os.link
    failed = False

    def fail_consumed_link(source, target, *args, **kwargs):
        nonlocal failed
        if target == reservation.consumed_path.name and not failed:
            failed = True
            raise OSError("consumed publication failed")
        return real_link(source, target, *args, **kwargs)

    monkeypatch.setattr(
        artifact_reservation_module.os,
        "link",
        fail_consumed_link,
    )
    with pytest.raises(OSError, match="consumed publication failed") as raised:
        save_minimal_result(
            results_path,
            run_id=reservation.run_id,
            inference_mode="async_queue",
            reservation=reservation,
        )

    assert raised.value.csv_commit_recovery_pending is True
    assert raised.value.publication_state_uncertain is True
    assert pending_path.exists()
    assert not reservation.consumed_path.exists()
    assert [row["run_id"] for row in load_results(results_path=results_path)] == [
        reservation.run_id
    ]

    assert save_minimal_result(
        results_path,
        run_id=reservation.run_id,
        inference_mode="async_queue",
        reservation=reservation,
    ) == reservation.run_id
    assert [row["run_id"] for row in load_results(results_path=results_path)] == [
        reservation.run_id
    ]
    assert reservation.consumed_path.exists()
    assert not pending_path.exists()


def test_csv_existing_pending_row_rejects_changed_retry_metrics(
    tmp_path,
    monkeypatch,
):
    results_path = tmp_path / "results.csv"
    reservation = reserve_run_artifacts(
        results_path=results_path,
        run_id="fixed123",
    )
    real_link = artifact_reservation_module.os.link
    failed = False

    def fail_consumed_link(source, target, *args, **kwargs):
        nonlocal failed
        if target == reservation.consumed_path.name and not failed:
            failed = True
            raise OSError("consumed publication failed")
        return real_link(source, target, *args, **kwargs)

    monkeypatch.setattr(
        artifact_reservation_module.os,
        "link",
        fail_consumed_link,
    )
    with pytest.raises(OSError, match="consumed publication failed"):
        save_minimal_result(
            results_path,
            run_id=reservation.run_id,
            inference_mode="async_queue",
            reservation=reservation,
            metrics={"accuracy": 1.0},
        )

    with pytest.raises(ValueError, match="transaction provenance|fingerprint"):
        save_minimal_result(
            results_path,
            run_id=reservation.run_id,
            inference_mode="async_queue",
            reservation=reservation,
            metrics={"accuracy": 0.5},
        )

    assert reservation.pending_path.exists()
    assert not reservation.consumed_path.exists()
    assert load_results(results_path=results_path)[0]["accuracy"] == "1.0"


def test_csv_pending_cleanup_failure_is_recoverable_without_duplicate(
    tmp_path,
    monkeypatch,
):
    results_path = tmp_path / "results.csv"
    reservation = reserve_run_artifacts(
        results_path=results_path,
        run_id="fixed123",
    )
    pending_path = reservation.marker_path.with_suffix(".pending")
    real_unlink = artifact_reservation_module.os.unlink
    failed = False

    def fail_pending_cleanup(target, *args, **kwargs):
        nonlocal failed
        if str(target).endswith(".quarantine") and not failed:
            failed = True
            raise OSError("pending cleanup failed")
        return real_unlink(target, *args, **kwargs)

    monkeypatch.setattr(
        artifact_reservation_module.os,
        "unlink",
        fail_pending_cleanup,
    )
    with pytest.raises(OSError, match="pending cleanup failed") as raised:
        save_minimal_result(
            results_path,
            run_id=reservation.run_id,
            inference_mode="async_queue",
            reservation=reservation,
        )

    assert raised.value.csv_commit_recovery_pending is True
    assert raised.value.publication_state_uncertain is True
    assert pending_path.exists()
    assert reservation.consumed_path.exists()

    assert save_minimal_result(
        results_path,
        run_id=reservation.run_id,
        inference_mode="async_queue",
        reservation=reservation,
    ) == reservation.run_id
    assert [row["run_id"] for row in load_results(results_path=results_path)] == [
        reservation.run_id
    ]
    assert not pending_path.exists()


def test_csv_pending_cleanup_fsync_failure_is_recoverable_without_duplicate(
    tmp_path,
    monkeypatch,
):
    results_path = tmp_path / "results.csv"
    reservation = reserve_run_artifacts(
        results_path=results_path,
        run_id="fixed123",
    )
    pending_path = reservation.marker_path.with_suffix(".pending")
    real_fsync = artifact_reservation_module.os.fsync
    failed = False

    def fail_pending_cleanup_fsync(file_descriptor):
        nonlocal failed
        opened = os.fstat(file_descriptor)
        if (
            not failed
            and stat.S_ISDIR(opened.st_mode)
            and (opened.st_dev, opened.st_ino)
            == (reservation.marker_device, reservation.marker_inode)
            and not pending_path.exists()
            and reservation.consumed_path.exists()
        ):
            failed = True
            raise OSError("pending cleanup fsync failed")
        return real_fsync(file_descriptor)

    monkeypatch.setattr(
        artifact_reservation_module.os,
        "fsync",
        fail_pending_cleanup_fsync,
    )
    with pytest.raises(OSError, match="pending cleanup fsync failed") as raised:
        save_minimal_result(
            results_path,
            run_id=reservation.run_id,
            inference_mode="async_queue",
            reservation=reservation,
        )

    assert raised.value.csv_commit_recovery_pending is True
    assert raised.value.publication_state_uncertain is True
    assert not pending_path.exists()
    assert reservation.consumed_path.exists()

    assert save_minimal_result(
        results_path,
        run_id=reservation.run_id,
        inference_mode="async_queue",
        reservation=reservation,
    ) == reservation.run_id
    assert [row["run_id"] for row in load_results(results_path=results_path)] == [
        reservation.run_id
    ]


@pytest.mark.parametrize("suffix", ["pending", "consumed"])
def test_reservation_state_replacement_after_link_is_not_unlinked(
    tmp_path,
    monkeypatch,
    suffix,
):
    reservation = reserve_run_artifacts(
        results_path=tmp_path / "results.csv",
        run_id="fixed123",
    )
    state_path = reservation.marker_path.with_suffix(f".{suffix}")
    real_link = artifact_reservation_module.os.link

    def link_then_replace_state(source, target, *args, **kwargs):
        result = real_link(source, target, *args, **kwargs)
        if target == state_path.name:
            replacement = state_path.with_suffix(".replacement")
            replacement.write_text("replacement", encoding="utf-8")
            os.replace(replacement, state_path)
        return result

    monkeypatch.setattr(
        artifact_reservation_module.os,
        "link",
        link_then_replace_state,
    )
    with artifact_reservation_module.verify_reservation(
        reservation,
        reservation.run_id,
        results_path=reservation.results_path,
        require_active=False,
    ) as verified:
        with pytest.raises(ValueError, match="state.*identity") as raised:
            if suffix == "pending":
                artifact_reservation_module.publish_pending(
                    verified,
                    "a" * 64,
                    "transaction-time",
                )
            else:
                artifact_reservation_module.publish_consumed(
                    verified,
                    "a" * 64,
                )

    assert state_path.read_text(encoding="utf-8") == "replacement"
    assert raised.value.publication_state_uncertain is True
    assert raised.value.state_file_may_remain is True


def test_clear_pending_does_not_unlink_replacement_after_read(
    tmp_path,
    monkeypatch,
):
    reservation = reserve_run_artifacts(
        results_path=tmp_path / "results.csv",
        run_id="fixed123",
    )
    fingerprint = "a" * 64
    with artifact_reservation_module.verify_reservation(
        reservation,
        reservation.run_id,
        results_path=reservation.results_path,
        require_active=False,
    ) as verified:
        artifact_reservation_module.publish_pending(
            verified,
            fingerprint,
            "transaction-time",
        )
    real_read_bytes = artifact_reservation_module._read_marker_bytes
    replaced = False

    def read_then_replace_pending(file_descriptor):
        nonlocal replaced
        value = real_read_bytes(file_descriptor)
        target = os.readlink(f"/proc/self/fd/{file_descriptor}")
        if target.endswith("fixed123.pending") and not replaced:
            replaced = True
            replacement = reservation.pending_path.with_suffix(".replacement")
            replacement.write_bytes(value)
            os.replace(replacement, reservation.pending_path)
        return value

    monkeypatch.setattr(
        artifact_reservation_module,
        "_read_marker_bytes",
        read_then_replace_pending,
    )
    with artifact_reservation_module.verify_reservation(
        reservation,
        reservation.run_id,
        results_path=reservation.results_path,
        require_active=False,
    ) as verified:
        with pytest.raises(ValueError, match="pending.*identity") as raised:
            artifact_reservation_module.clear_pending(verified, fingerprint)

    assert reservation.pending_path.exists()
    assert raised.value.publication_state_uncertain is True
    assert raised.value.state_file_may_remain is True


def test_clear_pending_cleanup_quarantines_stat_unlink_swap(
    tmp_path,
    monkeypatch,
):
    reservation = reserve_run_artifacts(
        results_path=tmp_path / "results.csv",
        run_id="fixed123",
    )
    fingerprint = "a" * 64
    with artifact_reservation_module.verify_reservation(
        reservation,
        reservation.run_id,
        results_path=reservation.results_path,
        require_active=False,
    ) as verified:
        artifact_reservation_module.publish_pending(
            verified,
            fingerprint,
            "transaction-time",
        )

    state = install_cleanup_swap(
        monkeypatch,
        artifact_reservation_module,
        reservation.pending_path,
    )
    with artifact_reservation_module.verify_reservation(
        reservation,
        reservation.run_id,
        results_path=reservation.results_path,
        require_active=False,
    ) as verified:
        with pytest.raises(
            artifact_reservation_module._ArtifactEntryIdentityError
        ) as raised:
            artifact_reservation_module.clear_pending(verified, fingerprint)

    assert state["swapped"] is True
    assert reservation.pending_path.read_text(encoding="utf-8") == "replacement"
    assert raised.value.cleanup_original_path == str(reservation.pending_path)
    assert raised.value.cleanup_original_preserved is True
    assert raised.value.cleanup_original_restored is True
    assert not hasattr(raised.value, "cleanup_recovery_path")
    assert raised.value.publication_state_uncertain is True


@pytest.mark.parametrize(
    "scenario",
    [
        "initial-destination-collision",
        "source-swap",
        "restore-destination-collision",
    ],
)
def test_cleanup_rename_interference_preserves_replacements(
    tmp_path,
    monkeypatch,
    scenario,
):
    target, expected_identity = make_owned_cleanup_target(tmp_path)
    state = install_cleanup_rename_interference(
        monkeypatch,
        target,
        scenario,
    )

    if scenario == "initial-destination-collision":
        assert unlink_owned_target(target, expected_identity) is True
        assert not target.exists()
        assert state["collision"].read_text(encoding="utf-8") == "collision"
        return

    with pytest.raises(
        artifact_reservation_module._ArtifactEntryIdentityError
    ) as raised:
        unlink_owned_target(target, expected_identity)

    restored = scenario == "source-swap"
    assert target.read_text(encoding="utf-8") == (
        "replacement" if restored else "restore collision"
    )
    quarantine_path = None if restored else state["quarantine"]
    if quarantine_path is not None:
        assert quarantine_path.read_text(encoding="utf-8") == "replacement"
    assert_cleanup_evidence(
        raised.value,
        target,
        restored=restored,
        recovery_path=quarantine_path,
    )


@pytest.mark.parametrize(
    "unsupported_error",
    [None, errno.ENOSYS, errno.EOPNOTSUPP],
    ids=["libc-symbol", "kernel-syscall", "filesystem-flag"],
)
def test_cleanup_unsupported_preserves_original_without_recovery_path(
    tmp_path,
    monkeypatch,
    unsupported_error,
):
    target, expected_identity = make_owned_cleanup_target(tmp_path)
    if unsupported_error is None:
        operation = None
    else:
        def operation(*_args):
            artifact_reservation_module.ctypes.set_errno(unsupported_error)
            return -1

    monkeypatch.setattr(
        artifact_reservation_module,
        "_LIBC_RENAMEAT2",
        operation,
    )
    with pytest.raises(
        artifact_reservation_module.ArtifactFilesystemUnsupportedError
    ) as raised:
        unlink_owned_target(target, expected_identity)

    assert target.read_text(encoding="utf-8") == "owned"
    assert not list(tmp_path.glob(".artifact-cleanup-*.quarantine"))
    assert_cleanup_evidence(
        raised.value,
        target,
        restored=False,
        unsupported=True,
    )


def test_cleanup_quarantine_mutation_is_not_reported_as_success(
    tmp_path,
    monkeypatch,
):
    target, expected_identity = make_owned_cleanup_target(tmp_path)
    replacement = tmp_path / "replacement.json"
    replacement.write_text("quarantine replacement", encoding="utf-8")
    real_stat = artifact_reservation_module.os.stat
    real_replace = artifact_reservation_module.os.replace
    mutated = False

    def mutate_before_quarantine_stat(path, *args, **kwargs):
        nonlocal mutated
        if (
            not mutated
            and str(path).endswith(".quarantine")
            and kwargs.get("dir_fd") is not None
        ):
            real_replace(replacement, tmp_path / path)
            mutated = True
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(
        artifact_reservation_module.os,
        "stat",
        mutate_before_quarantine_stat,
    )
    with pytest.raises(
        artifact_reservation_module._ArtifactEntryIdentityError
    ) as raised:
        unlink_owned_target(target, expected_identity)

    assert mutated is True
    assert target.read_text(encoding="utf-8") == "quarantine replacement"
    assert_cleanup_evidence(raised.value, target, restored=True)


def test_forged_owner_cannot_commit_async_csv(tmp_path):
    results_path = tmp_path / "results.csv"
    reservation = reserve_run_artifacts(
        results_path=results_path,
        run_id="fixed123",
    )
    forged = replace(reservation, owner_token="0" * 64)

    with pytest.raises(ValueError, match="owner token"):
        save_minimal_result(
            results_path,
            run_id="fixed123",
            inference_mode="async_queue",
            reservation=forged,
        )

    assert not results_path.exists()
    assert not reservation.consumed_path.exists()


def test_multiprocess_mixed_owners_cannot_split_artifact_bundle(tmp_path):
    results_path = tmp_path / "results.csv"
    reservation = reserve_run_artifacts(
        results_path=results_path,
        run_id="shared123",
    )
    context = multiprocessing.get_context("spawn")
    start = context.Event()
    outcomes = context.Queue()
    processes = [
        context.Process(
            target=mixed_owner_bundle_process,
            args=(reservation, forged, start, outcomes),
        )
        for forged in (False, True)
    ]
    for process in processes:
        process.start()
    start.set()
    for process in processes:
        process.join(10.0)
        assert process.exitcode == 0

    results = [outcomes.get(timeout=1.0) for _ in processes]
    assert sorted(result[0] for result in results) == ["error", "ok"]
    assert next(result for result in results if result[0] == "error")[2] == (
        "ValueError"
    )
    assert json.loads(reservation.details_path.read_text())["owner"] == "marker"
    assert not results_path.exists()

    save_minimal_result(
        results_path,
        run_id=reservation.run_id,
        inference_mode="async_queue",
        reservation=reservation,
    )
    assert load_results(results_path=results_path)[0]["run_id"] == "shared123"


def test_sidecar_holds_run_lease_until_active_postverification(
    tmp_path,
    monkeypatch,
):
    results_path = tmp_path / "results.csv"
    reservation = reserve_run_artifacts(
        results_path=results_path,
        run_id="fixed123",
    )
    sidecar_at_link = threading.Event()
    release_sidecar = threading.Event()
    csv_waiting_for_lease = threading.Event()
    real_link = artifact_reservation_module.os.link
    real_flock = artifact_reservation_module.fcntl.flock

    def block_sidecar_link(source, target, *args, **kwargs):
        if target == reservation.details_path.name:
            sidecar_at_link.set()
            assert release_sidecar.wait(2.0)
        return real_link(source, target, *args, **kwargs)

    def observe_run_lease(file_descriptor, operation):
        target = os.readlink(f"/proc/self/fd/{file_descriptor}")
        if (
            target.endswith(f"/{reservation.run_id}.lock")
            and operation & fcntl.LOCK_EX
            and sidecar_at_link.is_set()
        ):
            csv_waiting_for_lease.set()
        return real_flock(file_descriptor, operation)

    monkeypatch.setattr(artifact_reservation_module.os, "link", block_sidecar_link)
    monkeypatch.setattr(
        artifact_reservation_module.fcntl,
        "flock",
        observe_run_lease,
    )
    with ThreadPoolExecutor(max_workers=2) as executor:
        sidecar_future = executor.submit(
            save_async_details,
            reservation.run_id,
            {"value": 1},
            results_dir=reservation.results_root,
            reservation=reservation,
        )
        assert sidecar_at_link.wait(2.0)
        csv_future = executor.submit(
            save_minimal_result,
            results_path,
            run_id=reservation.run_id,
            inference_mode="async_queue",
            reservation=reservation,
        )
        try:
            assert csv_waiting_for_lease.wait(2.0)
            assert not results_path.exists()
            assert not reservation.consumed_path.exists()
        finally:
            release_sidecar.set()

        assert sidecar_future.result(timeout=2.0) == reservation.details_path
        assert csv_future.result(timeout=2.0) == reservation.run_id


def test_sidecar_marker_swap_after_link_rolls_back_final(tmp_path, monkeypatch):
    reservation = reserve_run_artifacts(
        results_path=tmp_path / "results.csv",
        run_id="fixed123",
    )
    real_link = artifact_reservation_module.os.link

    def link_then_corrupt_marker(source, target, *args, **kwargs):
        result = real_link(source, target, *args, **kwargs)
        if target == reservation.details_path.name:
            reservation.marker_path.write_text("{}\n", encoding="utf-8")
        return result

    monkeypatch.setattr(
        artifact_reservation_module.os,
        "link",
        link_then_corrupt_marker,
    )

    with pytest.raises(ValueError, match="owner token|marker binding"):
        save_async_details(
            reservation.run_id,
            {"value": 1},
            results_dir=reservation.results_root,
            reservation=reservation,
        )

    assert not reservation.details_path.exists()


def test_sidecar_exact_marker_replacement_after_link_rolls_back_final(
    tmp_path,
    monkeypatch,
):
    reservation = reserve_run_artifacts(
        results_path=tmp_path / "results.csv",
        run_id="fixed123",
    )
    marker_bytes = reservation.marker_path.read_bytes()
    real_link = artifact_reservation_module.os.link

    def link_then_replace_marker(source, target, *args, **kwargs):
        result = real_link(source, target, *args, **kwargs)
        if target == reservation.details_path.name:
            replacement = reservation.marker_path.with_suffix(".replacement")
            replacement.write_bytes(marker_bytes)
            os.replace(replacement, reservation.marker_path)
        return result

    monkeypatch.setattr(
        artifact_reservation_module.os,
        "link",
        link_then_replace_marker,
    )

    with pytest.raises(ValueError, match="marker identity"):
        save_async_details(
            reservation.run_id,
            {"value": 1},
            results_dir=reservation.results_root,
            reservation=reservation,
        )

    assert not reservation.details_path.exists()


def test_sidecar_run_lock_replacement_after_link_rolls_back_final(
    tmp_path,
    monkeypatch,
):
    reservation = reserve_run_artifacts(
        results_path=tmp_path / "results.csv",
        run_id="fixed123",
    )
    lock_path = reservation.marker_path.with_suffix(".lock")
    real_link = artifact_reservation_module.os.link

    def link_then_replace_lock(source, target, *args, **kwargs):
        result = real_link(source, target, *args, **kwargs)
        if target == reservation.details_path.name:
            replacement = lock_path.with_suffix(".replacement")
            replacement.write_bytes(b"")
            os.replace(replacement, lock_path)
        return result

    monkeypatch.setattr(
        artifact_reservation_module.os,
        "link",
        link_then_replace_lock,
    )

    with pytest.raises(ValueError, match="lock identity"):
        save_async_details(
            reservation.run_id,
            {"value": 1},
            results_dir=reservation.results_root,
            reservation=reservation,
        )

    assert not reservation.details_path.exists()


def test_sidecar_final_inode_replacement_is_detected_without_unlinking_replacement(
    tmp_path,
    monkeypatch,
):
    reservation = reserve_run_artifacts(
        results_path=tmp_path / "results.csv",
        run_id="fixed123",
    )
    real_link = artifact_reservation_module.os.link

    def link_then_replace_final(source, target, *args, **kwargs):
        result = real_link(source, target, *args, **kwargs)
        if target == reservation.details_path.name:
            replacement = reservation.details_path.with_suffix(".replacement")
            replacement.write_text("replacement", encoding="utf-8")
            os.replace(replacement, reservation.details_path)
        return result

    monkeypatch.setattr(
        artifact_reservation_module.os,
        "link",
        link_then_replace_final,
    )

    with pytest.raises(OSError, match="final entry identity"):
        save_async_details(
            reservation.run_id,
            {"value": 1},
            results_dir=reservation.results_root,
            reservation=reservation,
        )

    assert reservation.details_path.read_text(encoding="utf-8") == "replacement"


def test_sidecar_outer_postverify_failure_rolls_back_and_fsyncs_final(
    tmp_path,
    monkeypatch,
):
    reservation = reserve_run_artifacts(
        results_path=tmp_path / "results.csv",
        run_id="fixed123",
    )
    real_revalidate = artifact_reservation_module.revalidate_reservation
    real_unlink = result_store_module.os.unlink
    real_fsync = result_store_module.os.fsync
    verify_calls = 0
    rollback_events = []

    def fail_outer_postverify(verified, *, require_active):
        nonlocal verify_calls
        verify_calls += 1
        if verify_calls == 2:
            raise ValueError("outer sidecar postverify failed")
        return real_revalidate(verified, require_active=require_active)

    def observe_rollback_unlink(target, *args, **kwargs):
        result = real_unlink(target, *args, **kwargs)
        if str(target).endswith(".quarantine"):
            rollback_events.append("unlink")
        return result

    def observe_rollback_fsync(file_descriptor):
        if rollback_events == ["unlink"]:
            opened = os.fstat(file_descriptor)
            if stat.S_ISDIR(opened.st_mode):
                rollback_events.append("fsync")
        return real_fsync(file_descriptor)

    monkeypatch.setattr(
        artifact_reservation_module,
        "revalidate_reservation",
        fail_outer_postverify,
    )
    monkeypatch.setattr(result_store_module.os, "unlink", observe_rollback_unlink)
    monkeypatch.setattr(result_store_module.os, "fsync", observe_rollback_fsync)

    with pytest.raises(ValueError, match="outer sidecar postverify failed"):
        save_async_details(
            reservation.run_id,
            {"value": 1},
            results_dir=reservation.results_root,
            reservation=reservation,
        )

    assert verify_calls == 2
    assert rollback_events == ["unlink", "fsync"]
    assert not reservation.details_path.exists()


def test_sidecar_outer_postverify_preserves_primary_on_rollback_failure(
    tmp_path,
    monkeypatch,
):
    reservation = reserve_run_artifacts(
        results_path=tmp_path / "results.csv",
        run_id="fixed123",
    )
    primary = ValueError("outer sidecar postverify failed")
    real_revalidate = artifact_reservation_module.revalidate_reservation
    real_unlink = result_store_module.os.unlink
    verify_calls = 0

    def fail_outer_postverify(verified, *, require_active):
        nonlocal verify_calls
        verify_calls += 1
        if verify_calls == 2:
            raise primary
        return real_revalidate(verified, require_active=require_active)

    def fail_final_rollback(target, *args, **kwargs):
        if str(target).endswith(".quarantine"):
            raise OSError("outer sidecar rollback failed")
        return real_unlink(target, *args, **kwargs)

    monkeypatch.setattr(
        artifact_reservation_module,
        "revalidate_reservation",
        fail_outer_postverify,
    )
    monkeypatch.setattr(result_store_module.os, "unlink", fail_final_rollback)

    with pytest.raises(ValueError, match="outer sidecar postverify failed") as raised:
        save_async_details(
            reservation.run_id,
            {"value": 1},
            results_dir=reservation.results_root,
            reservation=reservation,
        )

    assert raised.value is primary
    assert len(raised.value.persistence_secondary_errors) == 1
    secondary = raised.value.persistence_secondary_errors[0]
    assert secondary == {
        "phase": "rollback_final",
        "error_type": "OSError",
        "error_message": "outer sidecar rollback failed",
        "publication_state_uncertain": True,
        "final_file_may_remain": True,
        "final_path": str(reservation.details_path),
        "cleanup_original_path": str(reservation.details_path),
        "cleanup_original_restored": True,
        "cleanup_original_preserved": True,
    }
    assert reservation.details_path.exists()
    assert "cleanup_recovery_path" not in secondary


def test_sidecar_rollback_quarantines_stat_unlink_swap(tmp_path, monkeypatch):
    reservation = reserve_run_artifacts(
        results_path=tmp_path / "results.csv",
        run_id="fixed123",
    )
    reservation.details_path.parent.mkdir()
    primary = ValueError("outer sidecar postverify failed")
    real_revalidate = artifact_reservation_module.revalidate_reservation
    verify_calls = 0
    rollback_active = False

    def fail_outer_postverify(verified, *, require_active):
        nonlocal verify_calls, rollback_active
        verify_calls += 1
        if verify_calls == 2:
            rollback_active = True
            raise primary
        return real_revalidate(verified, require_active=require_active)

    state = install_cleanup_swap(
        monkeypatch,
        result_store_module,
        reservation.details_path,
        active=lambda: rollback_active,
    )
    monkeypatch.setattr(
        artifact_reservation_module,
        "revalidate_reservation",
        fail_outer_postverify,
    )

    with pytest.raises(ValueError, match="outer sidecar postverify") as raised:
        save_async_details(
            reservation.run_id,
            {"value": 1},
            results_dir=reservation.results_root,
            reservation=reservation,
        )

    assert raised.value is primary
    assert state["swapped"] is True
    assert reservation.details_path.read_text(encoding="utf-8") == "replacement"
    secondary = raised.value.persistence_secondary_errors[0]
    assert secondary["cleanup_original_path"] == str(reservation.details_path)
    assert secondary["cleanup_original_preserved"] is True
    assert secondary["cleanup_original_restored"] is True
    assert "cleanup_recovery_path" not in secondary
    assert secondary["publication_state_uncertain"] is True


def test_sidecar_retained_directory_close_failure_reports_certain_commit(
    tmp_path,
    monkeypatch,
):
    reservation = reserve_run_artifacts(
        results_path=tmp_path / "results.csv",
        run_id="fixed123",
    )
    real_close = result_store_module.os.close
    real_close_publication = result_store_module._close_sidecar_publication
    failed = False
    closing_committed_publication = False

    def close_details_then_raise(file_descriptor):
        nonlocal failed
        try:
            target = os.readlink(f"/proc/self/fd/{file_descriptor}")
        except OSError:
            target = ""
        if (
            closing_committed_publication
            and target == str(reservation.results_root / "details")
            and not failed
        ):
            failed = True
            real_close(file_descriptor)
            raise OSError("details directory close failed after commit")
        return real_close(file_descriptor)

    def close_committed_publication(publication, primary=None):
        nonlocal closing_committed_publication
        closing_committed_publication = True
        try:
            return real_close_publication(publication, primary)
        finally:
            closing_committed_publication = False

    monkeypatch.setattr(
        result_store_module.os,
        "close",
        close_details_then_raise,
    )
    monkeypatch.setattr(
        result_store_module,
        "_close_sidecar_publication",
        close_committed_publication,
    )

    with pytest.raises(OSError, match="close failed after commit") as raised:
        save_async_details(
            reservation.run_id,
            {"value": 1},
            results_dir=reservation.results_root,
            reservation=reservation,
        )

    assert raised.value.final_file_committed is True
    assert raised.value.publication_state_uncertain is False
    assert raised.value.final_path == str(reservation.details_path)
    assert json.loads(reservation.details_path.read_text())["value"] == 1
    with pytest.raises(FileExistsError):
        save_async_details(
            reservation.run_id,
            {"value": 2},
            results_dir=reservation.results_root,
            reservation=reservation,
        )
    assert json.loads(reservation.details_path.read_text())["value"] == 1


def test_create_run_id_has_stable_path_safe_shape():
    run_ids = {create_run_id() for _ in range(32)}

    assert len(run_ids) == 32
    assert all(re.fullmatch(r"[0-9a-f]{8}", run_id) for run_id in run_ids)


def test_e2e_save_rejects_active_async_reservation_id(tmp_path):
    results_path = tmp_path / "results.csv"
    reservation = reserve_run_artifacts(
        results_path=results_path,
        run_id="fixed123",
    )

    with pytest.raises(ValueError, match="artifact authority|reserved"):
        save_minimal_result(results_path, run_id=reservation.run_id)

    assert not results_path.exists()


def test_e2e_save_rejects_consumed_async_id_after_csv_delete(tmp_path):
    results_path = tmp_path / "results.csv"
    reservation = reserve_run_artifacts(
        results_path=results_path,
        run_id="fixed123",
    )
    save_minimal_result(
        results_path,
        run_id=reservation.run_id,
        inference_mode="async_queue",
        reservation=reservation,
    )
    assert delete_result(reservation.run_id, results_path=results_path) is True

    with pytest.raises(ValueError, match="artifact authority|reserved"):
        save_minimal_result(results_path, run_id=reservation.run_id)

    assert load_results(results_path=results_path) == []


def test_generated_e2e_id_retries_async_artifact_authority_collision(
    tmp_path,
    monkeypatch,
):
    results_path = tmp_path / "results.csv"
    reserve_run_artifacts(results_path=results_path, run_id="owned123")
    candidates = iter(["owned123", "fresh123"])
    monkeypatch.setattr(
        result_store_module,
        "create_run_id",
        lambda: next(candidates),
    )

    assert save_minimal_result(results_path) == "fresh123"
    assert load_results(results_path=results_path)[0]["run_id"] == "fresh123"


def test_e2e_save_preserves_symlinked_parent_path_compatibility(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(target, target_is_directory=True)
    results_path = alias / "results.csv"

    assert save_minimal_result(results_path, run_id="fixed123") == "fixed123"
    assert load_results(results_path=results_path)[0]["run_id"] == "fixed123"


def test_e2e_save_preserves_csv_file_symlink_and_writes_canonical_target(
    tmp_path,
):
    target = tmp_path / "canonical.csv"
    alias = tmp_path / "alias.csv"
    alias.symlink_to(target)

    assert save_minimal_result(alias, run_id="fixed123") == "fixed123"

    assert alias.is_symlink()
    assert load_results(results_path=target)[0]["run_id"] == "fixed123"


def test_e2e_alias_and_canonical_saves_share_one_csv_lock_domain(
    tmp_path,
    monkeypatch,
):
    target = tmp_path / "canonical.csv"
    save_minimal_result(target, run_id="seed123")
    alias = tmp_path / "alias.csv"
    alias.symlink_to(target)
    first_entered_read = threading.Event()
    second_entered_read = threading.Event()
    release_first = threading.Event()
    real_read = result_store_module._read_csv_structure_at
    errors = []

    def observe_read(root_fd, results_name):
        if threading.current_thread().name == "alias-save":
            first_entered_read.set()
            assert release_first.wait(2.0)
        elif threading.current_thread().name == "canonical-save":
            second_entered_read.set()
        return real_read(root_fd, results_name)

    def save_in_thread(path, run_id):
        try:
            save_minimal_result(path, run_id=run_id)
        except BaseException as exc:
            errors.append(exc)

    monkeypatch.setattr(
        result_store_module,
        "_read_csv_structure_at",
        observe_read,
    )
    first = threading.Thread(
        target=save_in_thread,
        args=(alias, "alias123"),
        name="alias-save",
    )
    second = threading.Thread(
        target=save_in_thread,
        args=(target, "direct123"),
        name="canonical-save",
    )
    first.start()
    assert first_entered_read.wait(2.0)
    second.start()
    try:
        assert not second_entered_read.wait(0.2)
    finally:
        release_first.set()
        first.join(2.0)
        second.join(2.0)

    assert not first.is_alive()
    assert not second.is_alive()
    assert errors == []
    assert alias.is_symlink()
    assert {row["run_id"] for row in load_results(results_path=target)} == {
        "seed123",
        "alias123",
        "direct123",
    }


def test_e2e_parent_relocation_after_lease_fails_closed(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "root"
    relocated = tmp_path / "relocated"
    results_path = root / "results.csv"
    real_authority_exists = (
        result_store_module._run_artifact_authority_exists
    )
    replacement_reservation = None

    def relocate_then_check_old_authority(marker_directory, run_id):
        nonlocal replacement_reservation
        if replacement_reservation is None:
            root.rename(relocated)
            root.mkdir()
            replacement_reservation = reserve_run_artifacts(
                results_path=results_path,
                run_id=run_id,
            )
        return real_authority_exists(marker_directory, run_id)

    monkeypatch.setattr(
        result_store_module,
        "_run_artifact_authority_exists",
        relocate_then_check_old_authority,
    )

    with pytest.raises(ValueError, match="path identity|directory identity"):
        save_minimal_result(results_path, run_id="fixed123")

    assert replacement_reservation.marker_path.exists()
    assert not results_path.exists()
    assert not (relocated / "results.csv").exists()


def test_save_result_accepts_exact_preallocated_id_and_protects_async_metadata(
    tmp_path,
):
    csv_path = tmp_path / "results.csv"
    reservation = reserve_run_artifacts(
        results_path=csv_path,
        run_id="fixed123",
    )

    run_id = save_minimal_result(
        csv_path,
        run_id="fixed123",
        metrics={
            "accuracy": 1.0,
            "scenario": "metric-collision",
            "async_run_status": "metric-collision",
        },
        inference_mode="async_queue",
        scenario="offline",
        queue_capacity=8,
        worker_count=2,
        batch_timeout_ms=1.5,
        target_qps=None,
        schedule_seed=7,
        async_run_status="valid",
        async_invalid_reasons="",
        details_path="results/details/fixed123.json",
        request_trace_path="results/traces/fixed123.jsonl",
        reservation=reservation,
    )

    assert run_id == "fixed123"
    row = load_results(results_path=csv_path)[0]
    assert row["inference_mode"] == "async_queue"
    assert row["scenario"] == "offline"
    assert row["queue_capacity"] == "8"
    assert row["target_qps"] == ""
    assert row["async_run_status"] == "valid"
    assert row["request_trace_path"] == "results/traces/fixed123.jsonl"


def test_save_result_rejects_supplied_duplicate_run_id(tmp_path):
    csv_path = tmp_path / "results.csv"
    save_minimal_result(csv_path, run_id="fixed123")

    with pytest.raises(ValueError, match="already exists"):
        save_minimal_result(csv_path, run_id="fixed123", model_name="duplicate")

    rows = load_results(results_path=csv_path)
    assert [row["run_id"] for row in rows] == ["fixed123"]


def test_save_result_retries_generated_run_id_collision(tmp_path, monkeypatch):
    csv_path = tmp_path / "results.csv"
    save_minimal_result(csv_path, run_id="fixed123")
    generated = iter(("fixed123", "fresh123"))
    monkeypatch.setattr(result_store_module, "create_run_id", lambda: next(generated))

    run_id = save_minimal_result(csv_path)

    assert run_id == "fresh123"
    assert {row["run_id"] for row in load_results(results_path=csv_path)} == {
        "fixed123",
        "fresh123",
    }


def test_save_result_preserves_legacy_duplicate_ids_but_adds_no_duplicate(tmp_path):
    csv_path = tmp_path / "results.csv"
    csv_path.write_text(
        "run_id,model_name,accuracy\n"
        "legacy01,first,0.1\n"
        "legacy01,second,0.2\n",
        encoding="utf-8",
    )

    save_minimal_result(csv_path, run_id="fresh123")

    rows = list(reversed(load_results(results_path=csv_path)))
    assert [row["run_id"] for row in rows] == [
        "legacy01",
        "legacy01",
        "fresh123",
    ]


def test_interprocess_supplied_run_id_has_exactly_one_winner(tmp_path):
    csv_path = tmp_path / "results.csv"
    context = multiprocessing.get_context("spawn")
    start = context.Event()
    outcomes = context.Queue()
    processes = [
        context.Process(
            target=save_duplicate_result_process,
            args=(str(csv_path), start, outcomes),
        )
        for _ in range(2)
    ]
    for process in processes:
        process.start()
    start.set()
    for process in processes:
        process.join(10.0)
        assert process.exitcode == 0

    results = sorted(outcomes.get(timeout=1.0) for _ in processes)
    assert results == [("error", "ValueError"), ("ok", "shared123")]
    assert [row["run_id"] for row in load_results(results_path=csv_path)] == [
        "shared123"
    ]


def test_save_result_migrates_legacy_header_without_reordering_data(tmp_path):
    csv_path = tmp_path / "results.csv"
    csv_path.write_text(
        "run_id,model_name,legacy_metric,accuracy\n"
        "old00001,first,keep-a,0.1\n"
        "old00002,second,keep-b,0.2\n",
        encoding="utf-8",
    )
    reservation = reserve_run_artifacts(
        results_path=csv_path,
        run_id="fresh123",
    )

    save_minimal_result(
        csv_path,
        run_id=reservation.run_id,
        inference_mode="async_queue",
        scenario="offline",
        metrics={"accuracy": 1.0, "new_metric": 2.0},
        reservation=reservation,
    )

    with open(csv_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    assert reader.fieldnames[:4] == [
        "run_id",
        "model_name",
        "legacy_metric",
        "accuracy",
    ]
    assert reader.fieldnames[-1] == "new_metric"
    assert [(row["model_name"], row["legacy_metric"]) for row in rows[:2]] == [
        ("first", "keep-a"),
        ("second", "keep-b"),
    ]
    assert rows[0]["inference_mode"] == ""
    assert rows[2]["scenario"] == "offline"


@pytest.mark.parametrize(
    "content",
    [
        "run_id,model_name,model_name\nold0001,first,second\n",
        "run_id,,accuracy\nold0001,tiny,1.0\n",
        "run_id,model_name,accuracy\nold0001,tiny,1.0,extra\n",
        "run_id,model_name,accuracy\nold0001,tiny\n",
        'run_id,model_name,accuracy\nold0001,"unterminated,1.0\n',
    ],
)
def test_save_result_rejects_malformed_legacy_csv_without_mutation(
    tmp_path,
    content,
):
    csv_path = tmp_path / "results.csv"
    csv_path.write_text(content, encoding="utf-8")
    original = csv_path.read_bytes()

    with pytest.raises(ValueError, match="CSV"):
        save_minimal_result(csv_path, run_id="fresh123")

    assert csv_path.read_bytes() == original
    assert not list(tmp_path.glob("*.tmp"))


def test_delete_and_load_reject_malformed_csv_without_mutation(tmp_path):
    csv_path = tmp_path / "results.csv"
    csv_path.write_text(
        "run_id,model_name,accuracy\nold0001,tiny,1.0,extra\n",
        encoding="utf-8",
    )
    original = csv_path.read_bytes()

    with pytest.raises(ValueError, match="CSV"):
        load_results(results_path=csv_path)
    with pytest.raises(ValueError, match="CSV"):
        delete_result("old0001", results_path=csv_path)

    assert csv_path.read_bytes() == original


def test_valid_positional_csv_migration_preserves_every_legacy_cell(tmp_path):
    csv_path = tmp_path / "results.csv"
    header = ["run_id", "model_name", "notes", "accuracy"]
    legacy_rows = [
        ["old0001", "tiny,one", "first line\nsecond line", "1.0"],
        ["old0002", "", "", "0.5"],
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(legacy_rows)

    save_minimal_result(
        csv_path,
        run_id="fresh123",
        metrics={"accuracy": 0.9, "new_metric": 7},
    )

    with open(csv_path, newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle, strict=True))
    migrated_header = rows[0]
    assert migrated_header[: len(header)] == header
    for migrated, original in zip(rows[1:3], legacy_rows):
        assert migrated[: len(header)] == original
        assert migrated[len(header) :] == [""] * (
            len(migrated_header) - len(header)
        )


def test_save_result_preserves_legacy_positional_results_path(tmp_path):
    csv_path = tmp_path / "positional.csv"

    run_id = save_result(
        {"accuracy": 1.0},
        "tiny",
        "IMAGE_CLASSIFICATION",
        "onnxruntime",
        "cpu",
        1,
        0,
        None,
        "target",
        "vendor",
        "accelerator",
        "runtime",
        "compiler",
        "onnx",
        csv_path,
    )

    result = load_results(results_path=csv_path)[0]
    assert result["run_id"] == run_id
    assert result["inference_mode"] == "e2e"


@pytest.mark.parametrize(
    "run_id",
    ["", ".", "..", "../escape", "a/b", r"a\b", "/absolute", "bad\nline", "한글"],
)
def test_artifact_apis_reject_unsafe_run_ids_without_creating_files(
    tmp_path,
    run_id,
):
    csv_path = tmp_path / "results.csv"
    with pytest.raises(ValueError, match="run_id"):
        save_minimal_result(csv_path, run_id=run_id)
    with pytest.raises(ValueError, match="run_id"):
        save_async_details(run_id, {}, results_dir=tmp_path)

    assert not csv_path.exists()
    assert not (tmp_path / "details").exists()


def test_save_result_validates_run_id_before_directory_or_lock_creation(tmp_path):
    csv_path = tmp_path / "missing" / "results.csv"

    with pytest.raises(ValueError, match="run_id"):
        save_minimal_result(csv_path, run_id="../escape")

    assert not csv_path.parent.exists()
    assert not csv_path.with_suffix(".csv.lock").exists()


def test_csv_migration_replace_failure_preserves_legacy_bytes(
    tmp_path,
    monkeypatch,
):
    csv_path = tmp_path / "results.csv"
    csv_path.write_text(
        "run_id,model_name,accuracy\nold00001,tiny,1.0\n",
        encoding="utf-8",
    )
    original = csv_path.read_bytes()

    def fail_replace(source, target, *args, **kwargs):
        raise OSError("replace failed")

    monkeypatch.setattr(result_store_module.os, "replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        save_minimal_result(csv_path, scenario="offline")

    assert csv_path.read_bytes() == original
    assert not list(tmp_path.glob("*.tmp"))


def test_csv_rewrite_preserves_existing_file_mode(tmp_path):
    csv_path = tmp_path / "results.csv"
    csv_path.write_text(
        "run_id,model_name,accuracy\nold00001,tiny,1.0\n",
        encoding="utf-8",
    )
    csv_path.chmod(0o640)

    save_minimal_result(csv_path, run_id="fresh123", scenario="offline")

    assert stat.S_IMODE(csv_path.stat().st_mode) == 0o640


def test_csv_delete_rewrite_preserves_existing_file_mode(tmp_path):
    csv_path = tmp_path / "results.csv"
    save_minimal_result(csv_path, run_id="first123")
    save_minimal_result(csv_path, run_id="second123")
    csv_path.chmod(0o640)

    assert delete_result("first123", results_path=csv_path) is True

    assert stat.S_IMODE(csv_path.stat().st_mode) == 0o640


def test_new_csv_has_sensible_default_mode(tmp_path):
    csv_path = tmp_path / "results.csv"

    save_minimal_result(csv_path, run_id="fresh123")

    assert stat.S_IMODE(csv_path.stat().st_mode) == 0o644


def test_csv_mode_failure_preserves_original_and_removes_temp(
    tmp_path,
    monkeypatch,
):
    csv_path = tmp_path / "results.csv"
    csv_path.write_text(
        "run_id,model_name,accuracy\nold00001,tiny,1.0\n",
        encoding="utf-8",
    )
    csv_path.chmod(0o640)
    original = csv_path.read_bytes()

    monkeypatch.setattr(
        result_store_module.os,
        "fchmod",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("chmod failed")),
    )
    with pytest.raises(OSError, match="chmod failed"):
        save_minimal_result(csv_path, run_id="fresh123", scenario="offline")

    assert csv_path.read_bytes() == original
    assert stat.S_IMODE(csv_path.stat().st_mode) == 0o640
    assert not list(tmp_path.glob("*.tmp"))


def test_csv_primary_replace_error_survives_cleanup_error(
    tmp_path,
    monkeypatch,
):
    csv_path = tmp_path / "results.csv"
    csv_path.write_text(
        "run_id,model_name,accuracy\nold00001,tiny,1.0\n",
        encoding="utf-8",
    )
    original = csv_path.read_bytes()

    monkeypatch.setattr(
        result_store_module.os,
        "replace",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("replace primary")),
    )
    real_unlink = result_store_module.os.unlink

    def fail_temporary_cleanup(path, *args, **kwargs):
        if str(path).startswith(".results.csv.") and str(path).endswith(".tmp"):
            raise OSError("cleanup secondary")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(result_store_module.os, "unlink", fail_temporary_cleanup)
    with pytest.raises(OSError, match="replace primary") as raised:
        save_minimal_result(csv_path, run_id="fresh123")

    assert csv_path.read_bytes() == original
    assert raised.value.persistence_secondary_errors == [
        {
            "phase": "cleanup_temp",
            "error_type": "OSError",
            "error_message": "cleanup secondary",
            "temporary_file_may_remain": True,
            "temporary_path": str(next(tmp_path.glob("*.tmp"))),
        }
    ]
    assert list(tmp_path.glob("*.tmp"))


def test_csv_directory_fsync_error_survives_directory_close_error(
    tmp_path,
    monkeypatch,
):
    csv_path = tmp_path / "results.csv"
    csv_path.write_text(
        "run_id,model_name,accuracy\nold00001,tiny,1.0\n",
        encoding="utf-8",
    )
    real_fsync = result_store_module.os.fsync
    real_close = result_store_module.os.close
    real_replace = result_store_module.os.replace
    csv_replaced = False
    failed_directory_fd = None

    def observe_csv_replace(source, target, *args, **kwargs):
        nonlocal csv_replaced
        result = real_replace(source, target, *args, **kwargs)
        target_fd = kwargs.get("dst_dir_fd")
        if (
            target == csv_path.name
            and target_fd is not None
            and os.path.samefile(
                f"/proc/self/fd/{target_fd}",
                csv_path.parent,
            )
        ):
            csv_replaced = True
        return result

    def fail_directory_fsync(file_descriptor):
        nonlocal failed_directory_fd
        if csv_replaced and failed_directory_fd is None:
            opened = os.fstat(file_descriptor)
            if stat.S_ISDIR(opened.st_mode):
                failed_directory_fd = file_descriptor
                raise OSError("directory fsync primary")
        return real_fsync(file_descriptor)

    def fail_directory_close(file_descriptor):
        if file_descriptor == failed_directory_fd:
            real_close(file_descriptor)
            raise OSError("directory close secondary")
        return real_close(file_descriptor)

    monkeypatch.setattr(result_store_module.os, "replace", observe_csv_replace)
    monkeypatch.setattr(result_store_module.os, "fsync", fail_directory_fsync)
    monkeypatch.setattr(result_store_module.os, "close", fail_directory_close)
    with pytest.raises(OSError, match="directory fsync primary") as raised:
        save_minimal_result(csv_path, run_id="fresh123", scenario="offline")

    assert raised.value.persistence_secondary_errors == [
        {
            "phase": "close_parent_directory",
                "error_type": "OSError",
                "error_message": "directory close secondary",
                "descriptor_close_state_uncertain": True,
            }
        ]
    with open(csv_path, newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle, strict=True))
    assert len(rows) == 3
    assert not list(tmp_path.glob("*.tmp"))


def test_verify_reservation_preserves_primary_and_closes_root_after_marker_close(
    tmp_path,
    monkeypatch,
):
    reservation = reserve_run_artifacts(
        results_path=tmp_path / "results.csv",
        run_id="fixed123",
    )
    captured = {}
    primary = RuntimeError("verify body primary")
    state, real_close = install_effective_close_failure(
        monkeypatch,
        artifact_reservation_module,
        lambda: captured.get("marker_fd"),
        "marker close secondary",
    )

    try:
        with pytest.raises(RuntimeError, match="verify body primary") as raised:
            with artifact_reservation_module.verify_reservation(
                reservation,
                reservation.run_id,
                results_path=reservation.results_path,
                require_active=False,
            ) as verified:
                captured["root"] = verified.root
                captured["root_fd"] = verified.root.file_descriptor
                captured["marker"] = verified.marker_directory
                captured["marker_fd"] = (
                    verified.marker_directory.file_descriptor
                )
                raise primary

        assert raised.value is primary
        assert raised.value.persistence_secondary_errors == [
            {
                "phase": "close_marker_directory",
                "error_type": "OSError",
                "error_message": "marker close secondary",
                "descriptor_close_state_uncertain": True,
            }
        ]
        assert captured["marker"].file_descriptor is None
        assert captured["root"].file_descriptor is None
        with pytest.raises(OSError):
            os.fstat(captured["root_fd"])
        assert stat.S_ISCHR(os.fstat(state["sentinel_fd"]).st_mode)
    finally:
        if "sentinel_fd" in state:
            try:
                real_close(state["sentinel_fd"])
            except OSError:
                pass


def test_open_trusted_directory_take_before_close_preserves_reused_fd_and_closes_next(
    tmp_path,
    monkeypatch,
):
    captured = {}
    real_open = artifact_reservation_module.os.open

    def record_open(path, *args, **kwargs):
        file_descriptor = real_open(path, *args, **kwargs)
        if path == tmp_path.anchor:
            captured["anchor_fd"] = file_descriptor
        elif path == tmp_path.parts[1] and "next_fd" not in captured:
            captured["next_fd"] = file_descriptor
        return file_descriptor

    monkeypatch.setattr(artifact_reservation_module.os, "open", record_open)
    state, real_close = install_effective_close_failure(
        monkeypatch,
        artifact_reservation_module,
        lambda: captured.get("anchor_fd"),
        "trusted transition close failure",
    )

    try:
        with pytest.raises(OSError, match="trusted transition close failure"):
            artifact_reservation_module.open_trusted_directory(
                tmp_path,
                create=False,
            )

        assert stat.S_ISCHR(os.fstat(state["sentinel_fd"]).st_mode)
        with pytest.raises(OSError):
            os.fstat(captured["next_fd"])
    finally:
        for name in ("sentinel_fd",):
            if name in state:
                try:
                    real_close(state[name])
                except OSError:
                    pass
        if "next_fd" in captured:
            try:
                real_close(captured["next_fd"])
            except OSError:
                pass


@pytest.mark.parametrize(
    ("resource", "close_phase"),
    [
        ("trusted", "close_trusted_directory"),
        ("marker", "close_marker_directory_after_open"),
    ],
)
def test_open_directory_fstat_primary_survives_close_reuse(
    tmp_path,
    monkeypatch,
    resource,
    close_phase,
):
    root = (
        artifact_reservation_module.open_trusted_directory(
            tmp_path,
            create=False,
        )
        if resource == "marker"
        else None
    )
    captured = {}
    primary = RuntimeError(f"{resource} fstat primary")
    close_message = f"{resource} close secondary"
    watched_name = (
        artifact_reservation_module._MARKER_DIRECTORY
        if resource == "marker"
        else "."
    )
    real_open = artifact_reservation_module.os.open
    real_fstat = artifact_reservation_module.os.fstat

    def record_open(path, *args, **kwargs):
        file_descriptor = real_open(path, *args, **kwargs)
        if path == watched_name:
            captured["directory_fd"] = file_descriptor
        return file_descriptor

    def fail_directory_fstat(file_descriptor):
        if file_descriptor == captured.get("directory_fd"):
            raise primary
        return real_fstat(file_descriptor)

    monkeypatch.setattr(artifact_reservation_module.os, "open", record_open)
    monkeypatch.setattr(artifact_reservation_module.os, "fstat", fail_directory_fstat)
    state, real_close = install_effective_close_failure(
        monkeypatch,
        artifact_reservation_module,
        lambda: captured.get("directory_fd"),
        close_message,
    )

    try:
        with pytest.raises(RuntimeError, match=f"{resource} fstat primary") as raised:
            if resource == "marker":
                artifact_reservation_module.open_marker_directory(
                    root,
                    create=True,
                )
            else:
                artifact_reservation_module.open_trusted_directory(
                    ".",
                    create=False,
                )

        assert raised.value is primary
        assert_descriptor_close_secondary(
            raised.value,
            close_phase,
            close_message,
        )
        assert stat.S_ISCHR(real_fstat(state["sentinel_fd"]).st_mode)
    finally:
        if root is not None:
            root.close()
        if "sentinel_fd" in state:
            try:
                real_close(state["sentinel_fd"])
            except OSError:
                pass
        if "directory_fd" in captured:
            try:
                real_close(captured["directory_fd"])
            except OSError:
                pass


@pytest.mark.parametrize("artifact", ["marker", "state"])
def test_publication_temp_close_effect_does_not_reclose_reused_fd(
    tmp_path,
    monkeypatch,
    artifact,
):
    results_path = tmp_path / "results.csv"
    reservation = (
        reserve_run_artifacts(results_path=results_path, run_id="fixed123")
        if artifact == "state"
        else None
    )
    suffix = "pending" if reservation is not None else "json"
    final_path = (
        reservation.pending_path
        if reservation is not None
        else tmp_path / ".run_artifacts" / "fixed123.json"
    )
    temporary_prefix = f".fixed123.{suffix}."
    captured = {}
    real_open = artifact_reservation_module.os.open

    def record_temporary_open(path, *args, **kwargs):
        file_descriptor = real_open(path, *args, **kwargs)
        if (
            str(path).startswith(temporary_prefix)
            and str(path).endswith(".tmp")
        ):
            captured["temporary_fd"] = file_descriptor
        return file_descriptor

    monkeypatch.setattr(
        artifact_reservation_module.os,
        "open",
        record_temporary_open,
    )
    close_state, real_close = install_effective_close_failure(
        monkeypatch,
        artifact_reservation_module,
        lambda: captured.get("temporary_fd"),
        f"{artifact} temporary close primary",
    )

    try:
        with pytest.raises(
            OSError,
            match=f"{artifact} temporary close primary",
        ) as raised:
            if reservation is None:
                reserve_run_artifacts(results_path=results_path, run_id="fixed123")
            else:
                with artifact_reservation_module.verify_reservation(
                    reservation,
                    reservation.run_id,
                    results_path=reservation.results_path,
                    require_active=False,
                ) as verified:
                    artifact_reservation_module.publish_pending(
                        verified,
                        "a" * 64,
                        "transaction-time",
                    )

        assert getattr(raised.value, "persistence_secondary_errors", []) == []
        assert not final_path.exists()
        assert not list(final_path.parent.glob(f"{temporary_prefix}*.tmp"))
        assert stat.S_ISCHR(os.fstat(close_state["sentinel_fd"]).st_mode)
    finally:
        if "sentinel_fd" in close_state:
            try:
                real_close(close_state["sentinel_fd"])
            except OSError:
                pass


@pytest.mark.parametrize("operation", ["reserve", "e2e"])
def test_result_marker_close_failure_preserves_body_primary(
    tmp_path,
    monkeypatch,
    operation,
):
    results_path = tmp_path / "results.csv"
    captured = {}
    primary = RuntimeError(f"{operation} body primary")
    real_open_root = result_store_module.open_results_root
    real_open_marker = result_store_module.open_marker_directory

    @contextmanager
    def record_root(*args, **kwargs):
        with real_open_root(*args, **kwargs) as opened:
            captured["root"] = opened.root
            captured["root_fd"] = opened.root.file_descriptor
            yield opened

    def record_marker(*args, **kwargs):
        marker = real_open_marker(*args, **kwargs)
        captured["marker"] = marker
        captured["marker_fd"] = marker.file_descriptor
        return marker

    monkeypatch.setattr(result_store_module, "open_results_root", record_root)
    monkeypatch.setattr(
        result_store_module,
        "open_marker_directory",
        record_marker,
    )
    if operation == "reserve":
        monkeypatch.setattr(
            result_store_module,
            "_read_csv_structure_at",
            lambda *args, **kwargs: (_ for _ in ()).throw(primary),
        )
    else:
        monkeypatch.setattr(
            result_store_module,
            "_run_artifact_authority_exists",
            lambda *args, **kwargs: (_ for _ in ()).throw(primary),
        )
    state, real_close = install_effective_close_failure(
        monkeypatch,
        result_store_module,
        lambda: captured.get("marker_fd"),
        "result marker close secondary",
    )

    try:
        with pytest.raises(RuntimeError, match=f"{operation} body primary") as raised:
            if operation == "reserve":
                reserve_run_artifacts(results_path=results_path, run_id="fixed123")
            else:
                save_minimal_result(results_path, run_id="fixed123")

        assert raised.value is primary
        assert raised.value.persistence_secondary_errors == [
            {
                "phase": "close_marker_directory",
                "error_type": "OSError",
                "error_message": "result marker close secondary",
                "descriptor_close_state_uncertain": True,
            }
        ]
        assert captured["marker"].file_descriptor is None
        assert captured["root"].file_descriptor is None
        with pytest.raises(OSError):
            os.fstat(captured["root_fd"])
        assert stat.S_ISCHR(os.fstat(state["sentinel_fd"]).st_mode)
    finally:
        if "sentinel_fd" in state:
            try:
                real_close(state["sentinel_fd"])
            except OSError:
                pass


def test_interprocess_result_writes_preserve_every_row(tmp_path):
    csv_path = tmp_path / "results.csv"
    context = multiprocessing.get_context("spawn")
    start = context.Event()
    processes = [
        context.Process(
            target=save_result_process,
            args=(str(csv_path), prefix, start, 12),
        )
        for prefix in ("aa", "bb", "cc")
    ]
    for process in processes:
        process.start()
    start.set()
    for process in processes:
        process.join(10.0)
        assert process.exitcode == 0

    rows = load_results(results_path=csv_path)
    assert len(rows) == 36
    assert len({row["run_id"] for row in rows}) == 36


class DetailStatus(Enum):
    READY = "ready"


def test_save_async_details_is_deterministic_strict_json(tmp_path):
    path = save_async_details(
        "fixed123",
        {
            "run_id": "cannot-overwrite",
            "schema_version": "cannot-overwrite",
            "quality_metrics": {"accuracy": np.float32(1.0)},
            "array": np.asarray([1, 2], dtype=np.int64),
            "path": Path("models/tiny.onnx"),
            "state": DetailStatus.READY,
            "nested": (np.int64(3), {"x", "y"}),
        },
        results_dir=tmp_path,
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert path == tmp_path / "details" / "fixed123.json"
    assert payload["schema_version"] == "1.0"
    assert payload["run_id"] == "fixed123"
    assert payload["quality_metrics"]["accuracy"] == 1.0
    assert payload["array"] == [1, 2]
    assert payload["path"] == "models/tiny.onnx"
    assert payload["state"] == "ready"
    assert payload["nested"] == [3, ["x", "y"]]
    assert not list(path.parent.glob("*.tmp"))


def test_save_async_failure_details_uses_deterministic_no_overwrite_path(
    tmp_path,
):
    reservation = reserve_run_artifacts(
        results_path=tmp_path / "results.csv",
        run_id="fixed123",
    )

    path = save_async_failure_details(
        reservation.run_id,
        {"failure": {"phase": "csv_save"}, "value": np.int64(1)},
        results_dir=reservation.results_root,
        reservation=reservation,
    )
    original = path.read_bytes()

    assert path == tmp_path / "details" / "fixed123.failure.json"
    assert reservation.failure_details_path == path
    assert json.loads(original) == {
        "failure": {"phase": "csv_save"},
        "run_id": "fixed123",
        "schema_version": "1.0",
        "value": 1,
    }
    with pytest.raises(FileExistsError):
        save_async_failure_details(
            reservation.run_id,
            {"failure": {"phase": "runtime_unload"}},
            results_dir=reservation.results_root,
            reservation=reservation,
        )
    assert path.read_bytes() == original
    assert not list(path.parent.glob("*.tmp"))


def test_save_async_failure_details_accepts_consumed_reservation(tmp_path):
    results_path = tmp_path / "results.csv"
    reservation = reserve_run_artifacts(
        results_path=results_path,
        run_id="fixed123",
    )
    save_minimal_result(
        results_path,
        run_id=reservation.run_id,
        inference_mode="async_queue",
        reservation=reservation,
    )

    path = save_async_failure_details(
        reservation.run_id,
        {"failure": {"phase": "runtime_unload"}},
        results_dir=reservation.results_root,
        reservation=reservation,
    )

    assert json.loads(path.read_text(encoding="utf-8"))["failure"] == {
        "phase": "runtime_unload"
    }


def test_save_async_failure_details_accepts_and_preserves_pending_authority(
    tmp_path,
):
    reservation = reserve_run_artifacts(
        results_path=tmp_path / "results.csv",
        run_id="fixed123",
    )
    with artifact_reservation_module.verify_reservation(
        reservation,
        reservation.run_id,
        results_path=reservation.results_path,
        require_active=False,
    ) as verified:
        artifact_reservation_module.publish_pending(
            verified,
            "a" * 64,
            "transaction-time",
        )

    assert get_reserved_result_state(reservation) == "pending"
    path = save_async_failure_details(
        reservation.run_id,
        {"failure": {"phase": "csv_save"}},
        results_dir=reservation.results_root,
        reservation=reservation,
    )

    assert json.loads(path.read_text(encoding="utf-8"))["failure"] == {
        "phase": "csv_save"
    }
    assert get_reserved_result_state(reservation) == "pending"
    assert reservation.pending_path.exists()
    assert not reservation.consumed_path.exists()


def test_save_async_failure_details_rejects_unsafe_payload_without_artifact(
    tmp_path,
):
    reservation = reserve_run_artifacts(
        results_path=tmp_path / "results.csv",
        run_id="fixed123",
    )

    with pytest.raises(TypeError, match="HostileDetail"):
        save_async_failure_details(
            reservation.run_id,
            {"hostile": HostileDetail()},
            results_dir=reservation.results_root,
            reservation=reservation,
        )

    assert not reservation.failure_details_path.exists()


def test_save_async_failure_details_requires_matching_reservation_root(
    tmp_path,
):
    reservation = reserve_run_artifacts(
        results_path=tmp_path / "owner" / "results.csv",
        run_id="fixed123",
    )

    with pytest.raises(ValueError, match="results root"):
        save_async_failure_details(
            reservation.run_id,
            {"failure": {"phase": "csv_save"}},
            results_dir=tmp_path / "other",
            reservation=reservation,
        )

    assert not reservation.failure_details_path.exists()


def test_save_async_failure_details_requires_hard_link_publication(
    tmp_path,
    monkeypatch,
):
    reservation = reserve_run_artifacts(
        results_path=tmp_path / "results.csv",
        run_id="fixed123",
    )
    monkeypatch.setattr(
        artifact_reservation_module.os,
        "link",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            OSError(errno.EXDEV, "cross-device link")
        ),
    )

    with pytest.raises(
        result_store_module.ArtifactFilesystemUnsupportedError,
        match="POSIX.*hard-link",
    ):
        save_async_failure_details(
            reservation.run_id,
            {"failure": {"phase": "csv_save"}},
            results_dir=reservation.results_root,
            reservation=reservation,
        )

    assert not reservation.failure_details_path.exists()
    assert not list((reservation.results_root / "details").glob("*.tmp"))


def test_save_async_details_repeated_output_is_byte_deterministic(tmp_path):
    details = {"z": {"beta", "alpha"}, "a": np.float64(1.25)}

    first_path = save_async_details(
        "fixed123",
        details,
        results_dir=tmp_path / "first",
    )
    second_path = save_async_details(
        "fixed123",
        details,
        results_dir=tmp_path / "second",
    )

    assert first_path.read_bytes() == second_path.read_bytes()


def test_save_async_details_preserves_task7_result_sections(tmp_path):
    details = {
        "config": {"scenario": "offline", "queue_capacity": 8},
        "measurement_duration_sec": 1.25,
        "invalid_reasons": [],
        "warnings": ["tail_percentile_low_sample_count"],
        "counter_invariants": {"valid": True},
        "timing_ms": {"queue_wait": {"p99": 2.0}},
        "queue": {"depth_max": 4},
        "workers": {"utilization": 0.5},
        "batch_size": {"max": 2},
        "failure_types": {},
        "quality_metrics": {"accuracy": 1.0},
        "hardware_metrics": {"hw_power_watts": 12.5},
        "status": "valid",
    }

    path = save_async_details("fixed123", details, results_dir=tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert all(section in payload for section in details)
    assert payload["config"]["scenario"] == "offline"
    assert payload["timing_ms"]["queue_wait"]["p99"] == 2.0
    assert payload["hardware_metrics"]["hw_power_watts"] == 12.5


class HostileDetail:
    def __iter__(self):
        raise AssertionError("must not iterate")

    def item(self):
        raise AssertionError("must not call item")

    def __str__(self):
        raise AssertionError("must not stringify")


def test_save_async_details_rejects_unsupported_objects_without_callbacks(tmp_path):
    with pytest.raises(TypeError, match="HostileDetail"):
        save_async_details(
            "fixed123",
            {"hostile": HostileDetail()},
            results_dir=tmp_path,
        )

    assert not (tmp_path / "details" / "fixed123.json").exists()


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
def test_save_async_details_rejects_nonfinite_values(tmp_path, bad_value):
    with pytest.raises(ValueError, match="non-finite"):
        save_async_details(
            "fixed123",
            {"bad": bad_value},
            results_dir=tmp_path,
        )

    assert not (tmp_path / "details" / "fixed123.json").exists()


def test_save_async_details_rejects_cycles(tmp_path):
    cycle = []
    cycle.append(cycle)

    with pytest.raises(ValueError, match="cycles"):
        save_async_details(
            "fixed123",
            {"cycle": cycle},
            results_dir=tmp_path,
        )

    assert not (tmp_path / "details" / "fixed123.json").exists()


def test_save_async_details_rejects_object_array_cycles(tmp_path):
    cycle = np.empty(1, dtype=object)
    cycle[0] = cycle

    with pytest.raises(ValueError, match="cycles"):
        save_async_details(
            "fixed123",
            {"cycle": cycle},
            results_dir=tmp_path,
        )

    assert not (tmp_path / "details" / "fixed123.json").exists()


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (list(range(10_001)), "item budget"),
        (np.arange(4_097), "array budget"),
        ("x" * 1_000_001, "string budget"),
    ],
)
def test_save_async_details_rejects_values_over_normalization_budgets(
    tmp_path,
    value,
    message,
):
    with pytest.raises(ValueError, match=message):
        save_async_details(
            "fixed123",
            {"value": value},
            results_dir=tmp_path,
        )

    assert not (tmp_path / "details").exists()


def test_save_async_details_rejects_depth_over_budget(tmp_path):
    value = "leaf"
    for _ in range(34):
        value = [value]

    with pytest.raises(ValueError, match="depth budget"):
        save_async_details(
            "fixed123",
            {"value": value},
            results_dir=tmp_path,
        )

    assert not (tmp_path / "details").exists()


def test_save_async_details_rejects_enum_identity_cycle_without_recursion(tmp_path):
    class CyclicStatus(Enum):
        READY = "ready"

    object.__setattr__(CyclicStatus.READY, "_value_", CyclicStatus.READY)

    with pytest.raises(ValueError, match="cycle"):
        save_async_details(
            "fixed123",
            {"status": CyclicStatus.READY},
            results_dir=tmp_path,
        )

    assert not (tmp_path / "details").exists()


def test_save_async_details_publish_failure_keeps_previous_complete_file(
    tmp_path,
    monkeypatch,
):
    path = save_async_details(
        "fixed123",
        {"generation": 1},
        results_dir=tmp_path,
    )
    original = path.read_bytes()

    def fail_publish(*args, **kwargs):
        raise OSError("publish failed")

    monkeypatch.setattr(result_store_module.os, "link", fail_publish)
    with pytest.raises(OSError, match="publish failed"):
        save_async_details(
            "fixed123",
            {"generation": 2},
            results_dir=tmp_path,
        )

    assert path.read_bytes() == original
    assert not list(path.parent.glob("*.tmp"))


def test_sidecar_primary_publish_error_survives_cleanup_error(
    tmp_path,
    monkeypatch,
):
    reservation = reserve_run_artifacts(
        results_path=tmp_path / "results.csv",
        run_id="fixed123",
    )
    monkeypatch.setattr(
        result_store_module.os,
        "link",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("publish primary")),
    )
    monkeypatch.setattr(
        result_store_module.os,
        "unlink",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("cleanup secondary")),
    )

    with pytest.raises(OSError, match="publish primary") as raised:
        save_async_details(
            reservation.run_id,
            {"value": 1},
            results_dir=reservation.results_root,
            reservation=reservation,
        )

    errors = raised.value.persistence_secondary_errors
    assert len(errors) == 1
    assert errors[0]["phase"] == "cleanup_temp"
    assert errors[0]["error_type"] == "OSError"
    assert errors[0]["error_message"] == "cleanup secondary"
    assert errors[0]["temporary_file_may_remain"] is True
    assert errors[0]["temporary_path"].endswith(".tmp")
    assert list((tmp_path / "details").glob("*.tmp"))


def test_sidecar_rollback_fsync_failure_is_secondary_uncertain_evidence(
    tmp_path,
    monkeypatch,
):
    reservation = reserve_run_artifacts(
        results_path=tmp_path / "results.csv",
        run_id="fixed123",
    )
    match_results = iter((True, True, False))
    monkeypatch.setattr(
        result_store_module,
        "_sidecar_directories_match",
        lambda *args, **kwargs: next(match_results),
    )
    fsync_calls = 0
    real_fsync = result_store_module.os.fsync

    def fail_rollback_fsync(file_descriptor):
        nonlocal fsync_calls
        fsync_calls += 1
        if fsync_calls == 3:
            raise OSError("rollback fsync failed")
        return real_fsync(file_descriptor)

    monkeypatch.setattr(result_store_module.os, "fsync", fail_rollback_fsync)
    with pytest.raises(OSError, match="changed during publication") as raised:
        save_async_details(
            reservation.run_id,
            {"value": 1},
            results_dir=reservation.results_root,
            reservation=reservation,
        )

    assert raised.value.persistence_secondary_errors == [
        {
            "phase": "rollback_directory_fsync",
            "error_type": "OSError",
            "error_message": "rollback fsync failed",
            "publication_state_uncertain": True,
        }
    ]
    assert not reservation.details_path.exists()


def test_sidecar_reports_unsupported_hard_link_filesystem(tmp_path, monkeypatch):
    reservation = reserve_run_artifacts(
        results_path=tmp_path / "results.csv",
        run_id="fixed123",
    )
    monkeypatch.setattr(
        result_store_module.os,
        "link",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            OSError(errno.EXDEV, "cross-device link")
        ),
    )

    with pytest.raises(
        result_store_module.ArtifactFilesystemUnsupportedError,
        match="POSIX.*hard-link",
    ):
        save_async_details(
            reservation.run_id,
            {"value": 1},
            results_dir=reservation.results_root,
            reservation=reservation,
        )

    assert not reservation.details_path.exists()


def test_sidecar_maps_hard_link_eperm_to_capability_error(tmp_path, monkeypatch):
    reservation = reserve_run_artifacts(
        results_path=tmp_path / "results.csv",
        run_id="fixed123",
    )
    monkeypatch.setattr(
        artifact_reservation_module.os,
        "link",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            PermissionError(errno.EPERM, "hard link denied")
        ),
    )

    with pytest.raises(
        result_store_module.ArtifactFilesystemUnsupportedError,
        match="permission.*capability|capability.*permission",
    ):
        save_async_details(
            reservation.run_id,
            {"value": 1},
            results_dir=reservation.results_root,
            reservation=reservation,
        )

    assert not reservation.details_path.exists()


def test_sidecar_reports_missing_posix_hard_link_support(tmp_path, monkeypatch):
    reservation = reserve_run_artifacts(
        results_path=tmp_path / "results.csv",
        run_id="fixed123",
    )
    link_called = False

    def unexpected_link(*args, **kwargs):
        nonlocal link_called
        link_called = True

    monkeypatch.setattr(
        artifact_reservation_module,
        "_HARD_LINK_PUBLICATION_SUPPORTED",
        False,
    )
    monkeypatch.setattr(artifact_reservation_module.os, "link", unexpected_link)

    with pytest.raises(
        result_store_module.ArtifactFilesystemUnsupportedError,
        match="POSIX.*hard-link",
    ):
        save_async_details(
            reservation.run_id,
            {"value": 1},
            results_dir=reservation.results_root,
            reservation=reservation,
        )

    assert link_called is False
    assert not reservation.details_path.exists()


def test_save_async_details_fsync_failure_never_publishes_partial_file(
    tmp_path,
    monkeypatch,
):
    reservation = reserve_run_artifacts(
        results_path=tmp_path / "results.csv",
        run_id="fixed123",
    )

    def fail_fsync(file_descriptor):
        raise OSError("fsync failed")

    monkeypatch.setattr(result_store_module.os, "fsync", fail_fsync)
    with pytest.raises(OSError, match="fsync failed"):
        save_async_details(
            reservation.run_id,
            {"generation": 1},
            results_dir=reservation.results_root,
            reservation=reservation,
        )

    details_dir = reservation.results_root / "details"
    assert not reservation.details_path.exists()
    assert not list(details_dir.glob("*.tmp"))


def test_save_async_details_rejects_symlinked_details_directory(tmp_path):
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (root / "details").symlink_to(outside, target_is_directory=True)

    with pytest.raises(OSError):
        save_async_details("fixed123", {"value": 1}, results_dir=root)

    assert list(outside.iterdir()) == []


def test_save_async_details_rejects_symlinked_results_root(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    root = tmp_path / "root"
    root.symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        save_async_details("fixed123", {"value": 1}, results_dir=root)

    assert list(outside.iterdir()) == []


def test_save_async_details_does_not_follow_existing_final_symlink(tmp_path):
    details = tmp_path / "details"
    details.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text("sentinel", encoding="utf-8")
    final_path = details / "fixed123.json"
    final_path.symlink_to(outside)

    with pytest.raises(FileExistsError):
        save_async_details("fixed123", {"value": 1}, results_dir=tmp_path)

    assert final_path.is_symlink()
    assert outside.read_text(encoding="utf-8") == "sentinel"


def test_save_async_details_detects_details_symlink_swap_at_publish(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "root"
    details = root / "details"
    outside = tmp_path / "outside"
    details.mkdir(parents=True)
    outside.mkdir()
    reservation = reserve_run_artifacts(
        results_path=root / "results.csv",
        run_id="fixed123",
    )
    real_link = result_store_module.os.link
    swapped = False

    def swap_then_link(source, target, *args, **kwargs):
        nonlocal swapped
        if not swapped:
            swapped = True
            details.rename(root / "relocated")
            details.symlink_to(outside, target_is_directory=True)
        return real_link(source, target, *args, **kwargs)

    monkeypatch.setattr(result_store_module.os, "link", swap_then_link)
    with pytest.raises(OSError, match="changed during publication"):
        save_async_details(
            reservation.run_id,
            {"value": 1},
            results_dir=reservation.results_root,
            reservation=reservation,
        )

    assert list(outside.iterdir()) == []
    assert not (root / "relocated" / "fixed123.json").exists()


def test_concurrent_sidecar_writers_have_exactly_one_winner(tmp_path):
    barrier = threading.Barrier(3)

    def write_sidecar(generation):
        barrier.wait()
        return save_async_details(
            "fixed123",
            {"generation": generation, "values": list(range(100))},
            results_dir=tmp_path,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(write_sidecar, value) for value in (1, 2)]
        barrier.wait()
        outcomes = []
        for future in futures:
            try:
                outcomes.append(("ok", future.result(timeout=5.0)))
            except FileExistsError:
                outcomes.append(("error", None))

    assert sorted(outcome for outcome, _ in outcomes) == ["error", "ok"]
    path = tmp_path / "details" / "fixed123.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["generation"] in (1, 2)
    assert payload["values"] == list(range(100))
    assert payload["run_id"] == "fixed123"
    assert not list(path.parent.glob("*.tmp"))


def test_multiprocess_sidecar_writers_have_exactly_one_winner(tmp_path):
    context = multiprocessing.get_context("spawn")
    start = context.Event()
    outcomes = context.Queue()
    processes = [
        context.Process(
            target=save_sidecar_process,
            args=(str(tmp_path), start, generation, outcomes),
        )
        for generation in (1, 2)
    ]
    for process in processes:
        process.start()
    start.set()
    for process in processes:
        process.join(10.0)
        assert process.exitcode == 0

    results = sorted(outcomes.get(timeout=1.0) for _ in processes)
    assert results[0] == ("error", "FileExistsError")
    assert results[1][0] == "ok"
    payload = json.loads(
        (tmp_path / "details" / "shared123.json").read_text(encoding="utf-8")
    )
    assert payload["generation"] == results[1][1]


def test_trace_requires_matching_reservation_before_parent_side_effects(tmp_path):
    missing_root = tmp_path / "missing"
    with pytest.raises(ValueError, match="reservation"):
        RequestTraceWriter(missing_root / "traces" / "fixed123.jsonl")
    assert not missing_root.exists()

    reservation = reserve_run_artifacts(
        results_path=tmp_path / "owner" / "results.csv",
        run_id="fixed123",
    )
    with pytest.raises(ValueError, match="trace path"):
        RequestTraceWriter(
            reservation.results_root / "traces" / "other.jsonl",
            reservation=reservation,
        )
    assert not (reservation.results_root / "traces").exists()


def test_trace_final_publication_runs_only_in_close_caller(tmp_path, monkeypatch):
    reservation = reserve_run_artifacts(
        results_path=tmp_path / "results.csv",
        run_id="fixed123",
    )
    publication_threads = []
    real_link = trace_module.os.link

    def record_link(*args, **kwargs):
        publication_threads.append(threading.current_thread())
        return real_link(*args, **kwargs)

    monkeypatch.setattr(trace_module.os, "link", record_link)
    writer = RequestTraceWriter(
        reservation.trace_path,
        reservation=reservation,
    )
    writer.start()
    writer.write(make_trace())

    assert writer.close(timeout=1.0) is True
    assert publication_threads == [threading.current_thread()]


def test_trace_timeout_abandonment_never_allows_late_worker_publication(
    tmp_path,
    monkeypatch,
):
    reservation = reserve_run_artifacts(
        results_path=tmp_path / "results.csv",
        run_id="fixed123",
    )
    entered = threading.Event()
    release = threading.Event()
    publication_called = threading.Event()
    real_dumps = trace_module.json.dumps

    def blocked_dumps(*args, **kwargs):
        entered.set()
        assert release.wait(1.0)
        return real_dumps(*args, **kwargs)

    def record_link(*args, **kwargs):
        publication_called.set()
        raise AssertionError("abandoned trace must never publish")

    monkeypatch.setattr(trace_module.json, "dumps", blocked_dumps)
    monkeypatch.setattr(trace_module.os, "link", record_link)
    writer = RequestTraceWriter(
        reservation.trace_path,
        reservation=reservation,
    )
    writer.start()
    writer.write(make_trace())
    assert entered.wait(1.0)

    assert writer.close(timeout=0.0) is False
    release.set()
    writer._thread.join(1.0)

    assert not publication_called.is_set()
    assert not reservation.trace_path.exists()
    assert not list((reservation.results_root / "traces").glob("*.tmp"))


def test_trace_link_entered_before_deadline_cannot_finalize_after_deadline(
    tmp_path,
    monkeypatch,
):
    reservation = reserve_run_artifacts(
        results_path=tmp_path / "results.csv",
        run_id="fixed123",
    )
    clock = [100.0]
    real_link = trace_module.os.link

    def link_then_expire(*args, **kwargs):
        result = real_link(*args, **kwargs)
        clock[0] = 102.0
        return result

    monkeypatch.setattr(trace_module.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(trace_module.os, "link", link_then_expire)
    writer = RequestTraceWriter(
        reservation.trace_path,
        reservation=reservation,
    )
    writer.start()
    writer.write(make_trace())

    assert writer.close(timeout=1.0) is False
    assert writer.error["phase"] == "close"
    assert writer.error["error_type"] == "TimeoutError"
    assert not reservation.trace_path.exists()
    assert not list((reservation.results_root / "traces").glob("*.tmp"))


def test_trace_cleanup_crossing_deadline_reports_committed_final(
    tmp_path,
    monkeypatch,
):
    writer, path, _ = make_trace_writer(tmp_path)
    clock = [100.0]
    real_close = trace_module.os.close
    monkeypatch.setattr(trace_module.time, "monotonic", lambda: clock[0])
    writer.start()
    parent_fd = writer._parent_fd
    writer.write(make_trace())

    def close_parent_after_deadline(file_descriptor):
        result = real_close(file_descriptor)
        if file_descriptor == parent_fd:
            clock[0] = 102.0
        return result

    monkeypatch.setattr(trace_module.os, "close", close_parent_after_deadline)

    assert writer.close(timeout=1.0) is False
    assert path.exists()
    assert writer.error["phase"] == "close"
    assert writer.error["error_type"] == "TimeoutError"
    assert writer.error["final_file_committed"] is True
    assert writer.error["publication_state_uncertain"] is False
    assert writer.error["final_path"] == str(path)


def test_trace_parent_relocation_after_publish_is_rolled_back(
    tmp_path,
    monkeypatch,
):
    reservation = reserve_run_artifacts(
        results_path=tmp_path / "root" / "results.csv",
        run_id="fixed123",
    )
    traces = reservation.results_root / "traces"
    relocated = reservation.results_root / "relocated-traces"
    real_link = trace_module.os.link
    swapped = False

    def link_then_relocate(*args, **kwargs):
        nonlocal swapped
        result = real_link(*args, **kwargs)
        if not swapped:
            swapped = True
            traces.rename(relocated)
            traces.mkdir()
        return result

    monkeypatch.setattr(trace_module.os, "link", link_then_relocate)
    writer = RequestTraceWriter(
        reservation.trace_path,
        reservation=reservation,
    )
    writer.start()
    writer.write(make_trace())

    assert writer.close(timeout=1.0) is False
    assert writer.error["phase"] == "validate_parent_after_publish"
    assert not reservation.trace_path.exists()
    assert not (relocated / "fixed123.jsonl").exists()
    assert not list(relocated.glob("*.tmp"))


def test_trace_marker_swap_after_link_rolls_back_final(tmp_path, monkeypatch):
    reservation = reserve_run_artifacts(
        results_path=tmp_path / "results.csv",
        run_id="fixed123",
    )
    real_link = artifact_reservation_module.os.link

    def link_then_corrupt_marker(source, target, *args, **kwargs):
        result = real_link(source, target, *args, **kwargs)
        if target == reservation.trace_path.name:
            reservation.marker_path.write_text("{}\n", encoding="utf-8")
        return result

    monkeypatch.setattr(
        artifact_reservation_module.os,
        "link",
        link_then_corrupt_marker,
    )
    writer = RequestTraceWriter(
        reservation.trace_path,
        reservation=reservation,
    )
    writer.start()
    writer.write(make_trace())

    assert writer.close(timeout=1.0) is False
    assert writer.error["phase"] == "validate_reservation_after_fsync"
    assert not reservation.trace_path.exists()


def test_trace_final_inode_replacement_is_detected_without_unlinking_replacement(
    tmp_path,
    monkeypatch,
):
    writer, path, _ = make_trace_writer(tmp_path)
    real_link = artifact_reservation_module.os.link

    def link_then_replace_final(source, target, *args, **kwargs):
        result = real_link(source, target, *args, **kwargs)
        if target == path.name:
            replacement = path.with_suffix(".replacement")
            replacement.write_text("replacement", encoding="utf-8")
            os.replace(replacement, path)
        return result

    monkeypatch.setattr(
        artifact_reservation_module.os,
        "link",
        link_then_replace_final,
    )
    writer.start()
    writer.write(make_trace())

    assert writer.close(timeout=1.0) is False
    assert writer.error["phase"] == "validate_final_after_publish"
    assert writer.error["secondary_errors"][0]["phase"] == (
        "rollback_final_identity"
    )
    assert path.read_text(encoding="utf-8") == "replacement"


def test_trace_marker_swap_during_context_postverify_rolls_back_final(
    tmp_path,
    monkeypatch,
):
    writer, path, reservation = make_trace_writer(tmp_path)
    writer.start()
    writer.write(make_trace())
    real_revalidate = artifact_reservation_module.revalidate_reservation
    calls = 0

    def corrupt_marker_on_context_postverify(verified, *, require_active):
        nonlocal calls
        calls += 1
        if calls == 2:
            reservation.marker_path.write_text("{}\n", encoding="utf-8")
        return real_revalidate(verified, require_active=require_active)

    monkeypatch.setattr(
        artifact_reservation_module,
        "revalidate_reservation",
        corrupt_marker_on_context_postverify,
    )

    assert writer.close(timeout=1.0) is False
    assert calls == 2
    assert not path.exists()


def test_trace_rollback_quarantines_stat_unlink_swap(tmp_path, monkeypatch):
    writer, path, _reservation = make_trace_writer(tmp_path)
    writer.start()
    writer.write(make_trace())
    primary = ValueError("trace context postverify failed")
    real_revalidate = artifact_reservation_module.revalidate_reservation
    calls = 0
    rollback_active = False

    def fail_context_postverify(verified, *, require_active):
        nonlocal calls, rollback_active
        calls += 1
        if calls == 2:
            rollback_active = True
            raise primary
        return real_revalidate(verified, require_active=require_active)

    state = install_cleanup_swap(
        monkeypatch,
        trace_module,
        path,
        active=lambda: rollback_active,
    )
    monkeypatch.setattr(
        artifact_reservation_module,
        "revalidate_reservation",
        fail_context_postverify,
    )

    assert writer.close(timeout=1.0) is False

    assert state["swapped"] is True
    assert path.read_text(encoding="utf-8") == "replacement"
    secondary = writer.error["secondary_errors"][0]
    assert secondary["cleanup_original_path"] == str(path)
    assert secondary["cleanup_original_preserved"] is True
    assert secondary["cleanup_original_restored"] is True
    assert "cleanup_recovery_path" not in secondary
    assert secondary["publication_state_uncertain"] is True


def test_trace_rollback_fsync_failure_reports_uncertain_final_state(
    tmp_path,
    monkeypatch,
):
    writer, path, _ = make_trace_writer(tmp_path)
    path.parent.mkdir()
    fsync_calls = 0
    real_fsync = trace_module.os.fsync

    def fail_publish_and_rollback_fsync(file_descriptor):
        nonlocal fsync_calls
        fsync_calls += 1
        if fsync_calls == 2:
            raise OSError("publication fsync failed")
        if fsync_calls == 3:
            raise OSError("rollback fsync failed")
        return real_fsync(file_descriptor)

    monkeypatch.setattr(
        trace_module.os,
        "fsync",
        fail_publish_and_rollback_fsync,
    )
    writer.start()
    writer.write(make_trace())

    assert writer.close(timeout=1.0) is False
    assert writer.error["phase"] == "directory_fsync"
    assert writer.error["secondary_errors"] == [
        {
            "phase": "rollback_directory_fsync",
            "error_type": "OSError",
            "error_message": "rollback fsync failed",
            "publication_state_uncertain": True,
            "final_file_may_remain": True,
            "final_path": str(path),
        }
    ]
    assert not path.exists()


def test_trace_rollback_unlink_failure_reports_leaked_final(
    tmp_path,
    monkeypatch,
):
    writer, path, _ = make_trace_writer(tmp_path)
    path.parent.mkdir()
    fsync_calls = 0
    real_fsync = trace_module.os.fsync
    real_unlink = trace_module.os.unlink

    def fail_publication_fsync(file_descriptor):
        nonlocal fsync_calls
        fsync_calls += 1
        if fsync_calls == 2:
            raise OSError("publication fsync failed")
        return real_fsync(file_descriptor)

    def fail_final_rollback(target, *args, **kwargs):
        if str(target).endswith(".quarantine"):
            raise OSError("rollback unlink failed")
        return real_unlink(target, *args, **kwargs)

    monkeypatch.setattr(trace_module.os, "fsync", fail_publication_fsync)
    monkeypatch.setattr(trace_module.os, "unlink", fail_final_rollback)
    writer.start()
    writer.write(make_trace())

    assert writer.close(timeout=1.0) is False
    assert writer.error["phase"] == "directory_fsync"
    assert len(writer.error["secondary_errors"]) == 1
    secondary = writer.error["secondary_errors"][0]
    assert secondary == {
        "phase": "rollback_final",
        "error_type": "OSError",
        "error_message": "rollback unlink failed",
        "publication_state_uncertain": True,
        "final_file_may_remain": True,
        "final_path": str(path),
        "cleanup_original_path": str(path),
        "cleanup_original_restored": True,
        "cleanup_original_preserved": True,
    }
    assert path.exists()
    assert "cleanup_recovery_path" not in secondary


def test_trace_reports_unsupported_hard_link_filesystem(tmp_path, monkeypatch):
    reservation = reserve_run_artifacts(
        results_path=tmp_path / "results.csv",
        run_id="fixed123",
    )
    monkeypatch.setattr(
        trace_module.os,
        "link",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            OSError(errno.EXDEV, "cross-device link")
        ),
    )
    writer = RequestTraceWriter(
        reservation.trace_path,
        reservation=reservation,
    )
    writer.start()
    writer.write(make_trace())

    assert writer.close(timeout=1.0) is False
    assert writer.error["phase"] == "publish"
    assert writer.error["error_type"] == "ArtifactFilesystemUnsupportedError"
    assert "POSIX" in writer.error["error_message"]
    assert "hard-link" in writer.error["error_message"]
    assert not reservation.trace_path.exists()


def test_get_result_returns_newest_preserved_legacy_duplicate(tmp_path):
    results_path = tmp_path / "results.csv"
    results_path.write_text(
        "run_id,model_name,value\n"
        "duplicate,old,1\n"
        "duplicate,new,2\n",
        encoding="utf-8",
    )

    result = get_result("duplicate", results_path=results_path)

    assert result["model_name"] == "new"
    assert result["value"] == "2"


def test_trace_writer_publishes_only_exact_trace_fields(tmp_path):
    writer, path, _ = make_trace_writer(tmp_path, capacity=2)
    writer.start()
    writer.write(
        make_trace(
            status=TerminalStatus.FAILED,
            error_type="RuntimeError",
            error_message="failed",
        )
    )

    assert writer.close(timeout=1.0) is True
    row = json.loads(path.read_text(encoding="utf-8"))
    assert row == {
        "request_id": 1,
        "sample_index": 2,
        "status": "failed",
        "scheduled_ns": 1,
        "issued_ns": 2,
        "enqueued_ns": 3,
        "runtime_started_ns": 4,
        "runtime_finished_ns": 5,
        "completed_ns": 6,
        "worker_id": 0,
        "batch_size": 1,
        "timed_out": False,
        "sample_count": 3,
        "error_type": "RuntimeError",
        "error_message": "failed",
    }
    assert writer.dropped == 0
    assert writer.error is None
    assert not list(tmp_path.glob("*.tmp"))
    assert writer._queue.unfinished_tasks == 0


@pytest.mark.parametrize("capacity", [0, -1, True, 1.5])
def test_trace_writer_requires_positive_integral_capacity(tmp_path, capacity):
    with pytest.raises(ValueError, match="capacity"):
        RequestTraceWriter(tmp_path / "trace.jsonl", capacity=capacity)


def test_trace_close_timeout_abandons_full_queue_and_cleans_temp(
    tmp_path,
    monkeypatch,
):
    entered = threading.Event()
    release = threading.Event()
    real_dumps = trace_module.json.dumps

    def blocked_dumps(*args, **kwargs):
        entered.set()
        assert release.wait(1.0)
        return real_dumps(*args, **kwargs)

    writer, path, _ = make_trace_writer(tmp_path, capacity=1)
    monkeypatch.setattr(trace_module.json, "dumps", blocked_dumps)
    writer.start()
    writer.write(make_trace(request_id=1))
    assert entered.wait(1.0)
    writer.write(make_trace(request_id=2))

    assert writer.close(timeout=0.0) is False
    release.set()
    writer._thread.join(1.0)

    assert not writer._thread.is_alive()
    assert not path.exists()
    assert not list(path.parent.glob("*.tmp"))
    assert writer.error["phase"] == "close"


def test_trace_writer_lifecycle_contract_is_explicit_and_close_is_idempotent(
    tmp_path,
):
    writer, _, _ = make_trace_writer(tmp_path)
    with pytest.raises(RuntimeError, match="running state"):
        writer.write(make_trace())
    with pytest.raises(RuntimeError, match="requires start"):
        writer.close(timeout=1.0)

    writer.start()
    with pytest.raises(RuntimeError, match="created state"):
        writer.start()
    writer.write(make_trace())
    assert writer.close(timeout=1.0) is True
    assert writer.close(timeout=0.0) is True
    with pytest.raises(RuntimeError, match="running state"):
        writer.write(make_trace())
    with pytest.raises(RuntimeError, match="created state"):
        writer.start()


def test_trace_writer_rejects_payload_like_objects_and_publishes_no_payload(
    tmp_path,
):
    writer, path, _ = make_trace_writer(tmp_path)
    writer.start()

    with pytest.raises(TypeError, match="only exact RequestTrace"):
        writer.write({"input": [1], "output": [2], "sample": "secret"})
    assert writer.close(timeout=1.0) is True

    assert path.read_text(encoding="utf-8") == ""


def test_trace_writer_saturation_is_nonblocking_and_counts_drops(
    tmp_path,
    monkeypatch,
):
    entered = threading.Event()
    release = threading.Event()
    real_dumps = trace_module.json.dumps

    def blocked_dumps(*args, **kwargs):
        entered.set()
        assert release.wait(1.0)
        return real_dumps(*args, **kwargs)

    writer, path, _ = make_trace_writer(tmp_path, capacity=1)
    monkeypatch.setattr(trace_module.json, "dumps", blocked_dumps)
    writer.start()
    writer.write(make_trace(request_id=1))
    assert entered.wait(1.0)
    writer.write(make_trace(request_id=2))
    writer.write(make_trace(request_id=3))

    assert writer.dropped == 1
    release.set()
    assert writer.close(timeout=1.0) is True
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["request_id"] for row in rows] == [1, 2]
    assert writer._queue.unfinished_tasks == 0


def test_trace_writer_accepts_concurrent_writes_without_accounting_loss(tmp_path):
    writer, path, _ = make_trace_writer(tmp_path, capacity=64)
    writer.start()
    barrier = threading.Barrier(5)

    def write_group(group):
        barrier.wait()
        for offset in range(10):
            writer.write(make_trace(request_id=group * 10 + offset))

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(write_group, group) for group in range(4)]
        barrier.wait()
        for future in futures:
            future.result(timeout=5.0)

    assert writer.close(timeout=1.0) is True
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert sorted(row["request_id"] for row in rows) == list(range(40))
    assert writer.dropped == 0
    assert writer._queue.unfinished_tasks == 0


def test_concurrent_trace_writers_have_exactly_one_winner(tmp_path):
    reservation = reserve_run_artifacts(
        results_path=tmp_path / "results.csv",
        run_id="fixed123",
    )
    path = reservation.trace_path
    writers = [
        RequestTraceWriter(path, reservation=reservation),
        RequestTraceWriter(path, reservation=reservation),
    ]
    for writer in writers:
        writer.start()
        writer.write(make_trace(request_id=writers.index(writer) + 1))

    barrier = threading.Barrier(3)

    def close_writer(writer):
        barrier.wait()
        return writer.close(timeout=1.0)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(close_writer, writer) for writer in writers]
        barrier.wait()
        results = [future.result(timeout=5.0) for future in futures]

    assert sorted(results) == [False, True]
    loser = writers[results.index(False)]
    assert loser.error["error_type"] == "FileExistsError"
    assert json.loads(path.read_text())["request_id"] in (1, 2)


def test_trace_start_rejects_existing_final_target(tmp_path):
    reservation = reserve_run_artifacts(
        results_path=tmp_path / "results.csv",
        run_id="fixed123",
    )
    path = reservation.trace_path
    path.parent.mkdir()
    path.write_text("existing\n", encoding="utf-8")
    writer = RequestTraceWriter(path, reservation=reservation)

    with pytest.raises(FileExistsError):
        writer.start()

    assert path.read_text(encoding="utf-8") == "existing\n"
    assert writer.error["error_type"] == "FileExistsError"


def test_trace_serialization_failure_is_diagnostic_and_never_published(
    tmp_path,
    monkeypatch,
):
    writer, path, _ = make_trace_writer(tmp_path)

    def fail_dumps(*args, **kwargs):
        raise ValueError("serialization failed")

    monkeypatch.setattr(trace_module.json, "dumps", fail_dumps)
    writer.start()
    writer.write(make_trace())

    assert writer.close(timeout=1.0) is False
    assert writer.error == {
        "phase": "serialize",
        "error_type": "ValueError",
        "error_message": "serialization failed",
    }
    assert not path.exists()
    assert not list(path.parent.glob("*.tmp"))
    assert writer._queue.unfinished_tasks == 0


class FailingWriteHandle:
    def __init__(self, handle):
        self.handle = handle

    def write(self, text):
        raise OSError("write failed")

    def flush(self):
        return self.handle.flush()

    def fileno(self):
        return self.handle.fileno()

    def close(self):
        return self.handle.close()


class FailingWriteAndCloseHandle(FailingWriteHandle):
    def close(self):
        self.handle.close()
        raise OSError("close secondary")


def test_trace_start_cleanup_retains_identity_and_secondary_failures(
    tmp_path,
    monkeypatch,
):
    writer, path, _ = make_trace_writer(tmp_path)
    real_close = trace_module.os.close
    real_unlink = trace_module.os.unlink

    def fail_thread_start():
        raise RuntimeError("thread start failed")

    def fail_temp_close(file_descriptor):
        try:
            target = os.readlink(f"/proc/self/fd/{file_descriptor}")
        except OSError:
            target = ""
        if target.endswith(".tmp"):
            real_close(file_descriptor)
            raise OSError("temp descriptor close failed")
        return real_close(file_descriptor)

    def fail_temp_unlink(target, *args, **kwargs):
        if str(target).endswith(".tmp"):
            raise OSError("temp unlink failed")
        return real_unlink(target, *args, **kwargs)

    monkeypatch.setattr(writer._thread, "start", fail_thread_start)
    monkeypatch.setattr(trace_module.os, "close", fail_temp_close)
    monkeypatch.setattr(trace_module.os, "unlink", fail_temp_unlink)

    with pytest.raises(RuntimeError, match="thread start failed"):
        writer.start()

    assert writer.error["phase"] == "start"
    assert writer.error["secondary_errors"] == [
        {
            "phase": "close_descriptor",
            "error_type": "OSError",
            "error_message": "temp descriptor close failed",
            "descriptor_close_state_uncertain": True,
        },
        {
            "phase": "cleanup_temp",
            "error_type": "OSError",
            "error_message": "temp unlink failed",
            "temporary_file_may_remain": True,
            "temporary_path": str(
                path.parent / writer._temporary_name
            ),
        },
    ]
    assert writer._temporary_fd is None
    assert writer._temporary_name is not None
    assert writer._parent_fd is not None
    assert list(path.parent.glob("*.tmp"))

    monkeypatch.setattr(trace_module.os, "close", real_close)
    monkeypatch.setattr(trace_module.os, "unlink", real_unlink)
    writer._cleanup_caller_resources()
    assert writer._temporary_fd is None
    assert writer._temporary_name is None
    assert writer._parent_fd is None
    assert not list(path.parent.glob("*.tmp"))


@pytest.mark.parametrize("descriptor_owner", ["temporary", "parent"])
def test_trace_close_error_never_retries_reused_descriptor(
    tmp_path,
    monkeypatch,
    descriptor_owner,
):
    writer, path, _ = make_trace_writer(tmp_path)
    real_close = trace_module.os.close
    sentinel = {}
    start_failed = False

    def fail_thread_start():
        nonlocal start_failed
        start_failed = True
        raise RuntimeError("thread start failed")

    def close_then_reuse(file_descriptor):
        target = os.readlink(f"/proc/self/fd/{file_descriptor}")
        expected = str(path.parent)
        if descriptor_owner == "temporary" and writer._temporary_name:
            expected = str(path.parent / writer._temporary_name)
        if (
            start_failed
            and target == expected
            and "fd" not in sentinel
        ):
            real_close(file_descriptor)
            source = os.open("/dev/null", os.O_RDONLY)
            if source != file_descriptor:
                os.dup2(source, file_descriptor)
                real_close(source)
            sentinel["fd"] = file_descriptor
            raise OSError(
                f"{descriptor_owner} descriptor close reported failure"
            )
        return real_close(file_descriptor)

    monkeypatch.setattr(writer._thread, "start", fail_thread_start)
    monkeypatch.setattr(trace_module.os, "close", close_then_reuse)

    with pytest.raises(RuntimeError, match="thread start failed"):
        writer.start()

    assert getattr(writer, f"_{descriptor_owner}_fd") is None
    writer._cleanup_caller_resources()
    try:
        assert stat.S_ISCHR(os.fstat(sentinel["fd"]).st_mode)
    finally:
        monkeypatch.setattr(trace_module.os, "close", real_close)
        try:
            real_close(sentinel["fd"])
        except OSError:
            pass


def test_trace_write_failure_is_diagnostic_and_cleans_temp(tmp_path, monkeypatch):
    writer, path, _ = make_trace_writer(tmp_path)
    real_fdopen = trace_module.os.fdopen

    def failing_fdopen(*args, **kwargs):
        return FailingWriteHandle(real_fdopen(*args, **kwargs))

    monkeypatch.setattr(trace_module.os, "fdopen", failing_fdopen)
    writer.start()
    writer.write(make_trace())

    assert writer.close(timeout=1.0) is False
    assert writer.error["phase"] == "write"
    assert writer.error["error_message"] == "write failed"
    assert not path.exists()
    assert not list(path.parent.glob("*.tmp"))


def test_trace_primary_write_error_retains_secondary_close_error(
    tmp_path,
    monkeypatch,
):
    writer, _, _ = make_trace_writer(tmp_path)
    real_fdopen = trace_module.os.fdopen

    def failing_fdopen(*args, **kwargs):
        return FailingWriteAndCloseHandle(real_fdopen(*args, **kwargs))

    monkeypatch.setattr(trace_module.os, "fdopen", failing_fdopen)
    writer.start()
    writer.write(make_trace())

    assert writer.close(timeout=1.0) is False
    assert writer.error["phase"] == "write"
    assert writer.error["error_message"] == "write failed"
    assert writer.error["secondary_errors"] == [
        {
            "phase": "close_file",
            "error_type": "OSError",
            "error_message": "close secondary",
        }
    ]


def test_trace_primary_publish_error_retains_cleanup_error_and_temp_path(
    tmp_path,
    monkeypatch,
):
    writer, path, _ = make_trace_writer(tmp_path)
    monkeypatch.setattr(
        trace_module.os,
        "link",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("publish primary")),
    )
    real_unlink = trace_module.os.unlink

    def fail_temp_cleanup(target, *args, **kwargs):
        if str(target).endswith(".tmp"):
            raise OSError("cleanup secondary")
        return real_unlink(target, *args, **kwargs)

    monkeypatch.setattr(
        trace_module.os,
        "unlink",
        fail_temp_cleanup,
    )
    writer.start()
    writer.write(make_trace())

    assert writer.close(timeout=1.0) is False
    assert writer.error["phase"] == "publish"
    assert writer.error["error_message"] == "publish primary"
    secondary = writer.error["secondary_errors"]
    assert len(secondary) == 1
    assert secondary[0]["phase"] == "cleanup_temp"
    assert secondary[0]["error_type"] == "OSError"
    assert secondary[0]["error_message"] == "cleanup secondary"
    assert secondary[0]["temporary_file_may_remain"] is True
    assert secondary[0]["temporary_path"].endswith(".tmp")
    assert list(path.parent.glob("*.tmp"))


def test_trace_fdopen_failure_closes_descriptor_and_cleans_temp(
    tmp_path,
    monkeypatch,
):
    writer, path, _ = make_trace_writer(tmp_path)
    descriptors = []

    def fail_fdopen(file_descriptor, *args, **kwargs):
        descriptors.append(file_descriptor)
        raise OSError("fdopen failed")

    monkeypatch.setattr(trace_module.os, "fdopen", fail_fdopen)
    writer.start()

    assert writer.close(timeout=1.0) is False
    assert writer.error["phase"] == "open"
    assert len(descriptors) == 1
    with pytest.raises(OSError):
        trace_module.os.fstat(descriptors[0])
    assert not path.exists()
    assert not list(path.parent.glob("*.tmp"))


def test_trace_fdopen_failure_cannot_reclose_reused_descriptor(
    tmp_path,
    monkeypatch,
):
    writer, path, _ = make_trace_writer(tmp_path)
    real_close = trace_module.os.close
    observed_closefd = []
    sentinel = {}

    def fail_fdopen(file_descriptor, *args, **kwargs):
        closefd = kwargs.get("closefd", True)
        observed_closefd.append(closefd)
        if closefd:
            real_close(file_descriptor)
            source = os.open("/dev/null", os.O_RDONLY)
            if source != file_descriptor:
                os.dup2(source, file_descriptor)
                real_close(source)
            sentinel["fd"] = file_descriptor
        raise OSError("fdopen failed after ownership transfer")

    monkeypatch.setattr(trace_module.os, "fdopen", fail_fdopen)
    writer.start()

    assert writer.close(timeout=1.0) is False
    assert writer.error["phase"] == "open"
    try:
        if "fd" in sentinel:
            assert stat.S_ISCHR(os.fstat(sentinel["fd"]).st_mode)
        assert observed_closefd == [False]
    finally:
        if "fd" in sentinel:
            try:
                real_close(sentinel["fd"])
            except OSError:
                pass
    assert not path.exists()


@pytest.mark.parametrize(
    ("operation", "expected_phase"),
    [("fsync", "flush"), ("link", "publish")],
)
def test_trace_finalize_failure_is_diagnostic_and_cleans_temp(
    tmp_path,
    monkeypatch,
    operation,
    expected_phase,
):
    writer, path, _ = make_trace_writer(tmp_path)
    path.parent.mkdir()

    def fail(*args, **kwargs):
        raise OSError(f"{operation} failed")

    monkeypatch.setattr(trace_module.os, operation, fail)
    writer.start()
    writer.write(make_trace())

    assert writer.close(timeout=1.0) is False
    assert writer.error["phase"] == expected_phase
    assert writer.error["error_message"] == f"{operation} failed"
    assert not path.exists()
    assert not list(path.parent.glob("*.tmp"))


def test_trace_directory_fsync_failure_rolls_back_atomic_file(
    tmp_path,
    monkeypatch,
):
    writer, path, _ = make_trace_writer(tmp_path)
    path.parent.mkdir()
    calls = 0
    real_fsync = trace_module.os.fsync

    def fail_directory_fsync(file_descriptor):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("directory fsync failed")
        return real_fsync(file_descriptor)

    monkeypatch.setattr(trace_module.os, "fsync", fail_directory_fsync)
    writer.start()
    writer.write(make_trace())

    assert writer.close(timeout=1.0) is False
    assert writer.error["phase"] == "directory_fsync"
    assert not path.exists()
    assert not list(path.parent.glob("*.tmp"))


def test_trace_directory_fsync_error_retains_directory_close_error(
    tmp_path,
    monkeypatch,
):
    writer, path, _ = make_trace_writer(tmp_path)
    path.parent.mkdir()
    writer.start()
    parent_fd = writer._parent_fd
    fsync_calls = 0
    real_fsync = trace_module.os.fsync
    real_close = trace_module.os.close

    def fail_directory_fsync(file_descriptor):
        nonlocal fsync_calls
        fsync_calls += 1
        if fsync_calls == 2:
            raise OSError("directory fsync primary")
        return real_fsync(file_descriptor)

    def fail_directory_close(file_descriptor):
        result = real_close(file_descriptor)
        if file_descriptor == parent_fd:
            raise OSError("directory close secondary")
        return result

    monkeypatch.setattr(trace_module.os, "fsync", fail_directory_fsync)
    monkeypatch.setattr(trace_module.os, "close", fail_directory_close)
    writer.write(make_trace())

    assert writer.close(timeout=1.0) is False
    assert writer.error["phase"] == "directory_fsync"
    assert writer.error["error_message"] == "directory fsync primary"
    assert writer.error["secondary_errors"] == [
        {
            "phase": "close_directory",
            "error_type": "OSError",
            "error_message": "directory close secondary",
                "descriptor_close_state_uncertain": True,
        }
    ]
    assert not path.exists()


def test_trace_close_uses_one_remaining_absolute_deadline(tmp_path, monkeypatch):
    class ControlledThread:
        def __init__(self, **kwargs):
            self.join_timeouts = []
            self.alive = False

        def start(self):
            self.alive = True

        def join(self, timeout):
            self.join_timeouts.append(timeout)

        def is_alive(self):
            return self.alive

    reservation = reserve_run_artifacts(
        results_path=tmp_path / "results.csv",
        run_id="fixed123",
    )
    ticks = iter((100.0, 100.25))
    monkeypatch.setattr(trace_module.threading, "Thread", ControlledThread)
    monkeypatch.setattr(trace_module.time, "monotonic", lambda: next(ticks))
    writer = RequestTraceWriter(
        reservation.trace_path,
        reservation=reservation,
    )
    writer.start()

    assert writer.close(timeout=1.0) is False
    assert writer._thread.join_timeouts == [0.75]
    writer._cleanup_caller_resources()


def test_delete_result_rewrite_failure_preserves_original_file(
    tmp_path,
    monkeypatch,
):
    csv_path = tmp_path / "results.csv"
    run_id = save_minimal_result(csv_path, run_id="fixed123")
    original = csv_path.read_bytes()

    def fail_replace(source, target):
        raise OSError("replace failed")

    monkeypatch.setattr(result_store_module.os, "replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        delete_result(run_id, results_path=csv_path)

    assert csv_path.read_bytes() == original
    assert not list(tmp_path.glob("*.tmp"))
