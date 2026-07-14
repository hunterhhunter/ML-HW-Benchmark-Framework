"""Durable ownership and trusted-path primitives for async run artifacts."""

import ctypes
import errno
import fcntl
import json
import os
import secrets
import stat
import sys
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path


_MARKER_DIRECTORY = ".run_artifacts"
_DIRECTORY_MODE = 0o755
_MARKER_MODE = 0o600
_MAX_MARKER_BYTES = 4096
_CLEANUP_QUARANTINE_ATTEMPTS = 16
_RENAME_NOREPLACE = 1
_POSIX_DIRECTORY_OPERATIONS_SUPPORTED = (
    os.name == "posix"
    and hasattr(os, "O_DIRECTORY")
    and hasattr(os, "O_NOFOLLOW")
    and all(
        operation in os.supports_dir_fd
        for operation in (os.open, os.mkdir, os.stat, os.unlink)
    )
)
_HARD_LINK_PUBLICATION_SUPPORTED = (
    os.name == "posix"
    and hasattr(os, "link")
    and os.link in os.supports_dir_fd
    and os.link in os.supports_follow_symlinks
)


class ArtifactFilesystemUnsupportedError(RuntimeError):
    """Raised when required POSIX no-overwrite primitives are unavailable."""


class _ArtifactEntryIdentityError(ValueError):
    """Raised when a scoped pathname no longer names its expected inode."""


def _load_libc_renameat2():
    if not sys.platform.startswith("linux"):
        return None
    try:
        operation = ctypes.CDLL(None, use_errno=True).renameat2
    except (AttributeError, OSError):
        return None
    operation.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    operation.restype = ctypes.c_int
    return operation


_LIBC_RENAMEAT2 = _load_libc_renameat2()


def _rename_noreplace(
    source_directory_fd: int,
    source_name: str,
    target_directory_fd: int,
    target_name: str,
) -> None:
    """Move one scoped entry atomically without replacing the destination."""
    operation = _LIBC_RENAMEAT2
    if operation is None:
        raise ArtifactFilesystemUnsupportedError(
            "artifact cleanup requires Linux libc renameat2 with "
            "RENAME_NOREPLACE"
        )
    ctypes.set_errno(0)
    result = operation(
        source_directory_fd,
        os.fsencode(source_name),
        target_directory_fd,
        os.fsencode(target_name),
        _RENAME_NOREPLACE,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise FileExistsError(
            error_number,
            os.strerror(error_number),
            target_name,
        )
    if error_number == errno.ENOENT:
        raise FileNotFoundError(
            error_number,
            os.strerror(error_number),
            source_name,
        )
    unsupported_errors = {
        errno.EINVAL,
        errno.ENOSYS,
        errno.EXDEV,
        errno.EOPNOTSUPP,
        getattr(errno, "ENOTSUP", errno.EOPNOTSUPP),
    }
    if error_number in unsupported_errors:
        raise ArtifactFilesystemUnsupportedError(
            "artifact cleanup requires Linux libc renameat2 and filesystem "
            "RENAME_NOREPLACE support"
        )
    raise OSError(error_number, os.strerror(error_number), source_name)


@dataclass(frozen=True)
class RunArtifactReservation:
    """Durable authority for every artifact belonging to one async run."""

    run_id: str
    results_root: Path
    results_path: Path
    owner_token: str = field(repr=False)
    root_device: int = field(repr=False)
    root_inode: int = field(repr=False)
    marker_device: int = field(repr=False)
    marker_inode: int = field(repr=False)
    marker_file_device: int = field(repr=False)
    marker_file_inode: int = field(repr=False)
    lease_device: int | None = field(default=None, repr=False)
    lease_inode: int | None = field(default=None, repr=False)

    @property
    def marker_path(self) -> Path:
        return self.results_root / _MARKER_DIRECTORY / f"{self.run_id}.json"

    @property
    def consumed_path(self) -> Path:
        return (
            self.results_root
            / _MARKER_DIRECTORY
            / f"{self.run_id}.consumed"
        )

    @property
    def pending_path(self) -> Path:
        return (
            self.results_root
            / _MARKER_DIRECTORY
            / f"{self.run_id}.pending"
        )

    @property
    def details_path(self) -> Path:
        return self.results_root / "details" / f"{self.run_id}.json"

    @property
    def trace_path(self) -> Path:
        return self.results_root / "traces" / f"{self.run_id}.jsonl"


@dataclass
class OpenedDirectory:
    path: Path
    file_descriptor: int
    device: int
    inode: int

    def close(self) -> None:
        file_descriptor = self.file_descriptor
        self.file_descriptor = None
        if file_descriptor is not None:
            os.close(file_descriptor)


def _close_taken_file_descriptor(
    file_descriptor: int,
    phase: str,
    primary: BaseException | None,
) -> BaseException | None:
    """Close a descriptor already removed from its owner exactly once."""
    try:
        os.close(file_descriptor)
    except BaseException as exc:
        if primary is None:
            primary = exc
            try:
                setattr(primary, "descriptor_close_state_uncertain", True)
            except BaseException:
                pass
        else:
            _attach_artifact_secondary(
                primary,
                phase,
                exc,
                descriptor_close_state_uncertain=True,
            )
    return primary


@dataclass
class OpenedResultsRoot:
    root: OpenedDirectory
    results_name: str

    @property
    def results_path(self) -> Path:
        return self.root.path / self.results_name


@dataclass
class VerifiedReservation:
    reservation: RunArtifactReservation
    root: OpenedDirectory
    marker_directory: OpenedDirectory
    results_name: str
    lock_file_descriptor: int


def _require_posix_directory_operations() -> None:
    if not _POSIX_DIRECTORY_OPERATIONS_SUPPORTED:
        raise ArtifactFilesystemUnsupportedError(
            "async artifact persistence requires POSIX O_DIRECTORY, "
            "O_NOFOLLOW, dirfd operations, and hard-link publication"
        )


def _directory_flags() -> int:
    _require_posix_directory_operations()
    return os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW


def _symlink_path_error(path: Path, component: str, exc: OSError) -> ValueError:
    return ValueError(
        f"artifact path contains a symlink or non-directory component: "
        f"{path!s} ({component})"
    )


def open_trusted_directory(path, *, create: bool) -> OpenedDirectory:
    """Open every directory component from a trusted anchor without symlinks."""
    requested = Path(path)
    if any(component == ".." for component in requested.parts):
        raise ValueError("artifact path cannot contain parent traversal")

    flags = _directory_flags()
    if requested.is_absolute():
        anchor_path = Path(requested.anchor)
        components = requested.parts[1:]
        current_fd = os.open(requested.anchor, flags)
    else:
        anchor_path = Path.cwd()
        components = requested.parts
        current_fd = os.open(".", flags)

    current_path = anchor_path
    primary = None
    result = None
    try:
        for component in components:
            if component in ("", "."):
                continue
            try:
                next_fd = os.open(component, flags, dir_fd=current_fd)
            except FileNotFoundError:
                if not create:
                    raise
                try:
                    os.mkdir(
                        component,
                        mode=_DIRECTORY_MODE,
                        dir_fd=current_fd,
                    )
                except FileExistsError:
                    pass
                else:
                    os.fsync(current_fd)
                try:
                    next_fd = os.open(component, flags, dir_fd=current_fd)
                except OSError as exc:
                    if exc.errno in (errno.ELOOP, errno.ENOTDIR):
                        raise _symlink_path_error(
                            requested,
                            component,
                            exc,
                        ) from exc
                    raise
            except OSError as exc:
                if exc.errno in (errno.ELOOP, errno.ENOTDIR):
                    raise _symlink_path_error(
                        requested,
                        component,
                        exc,
                    ) from exc
                raise
            owned_current_fd = current_fd
            current_fd = None
            close_error = _close_taken_file_descriptor(
                owned_current_fd,
                "close_trusted_directory_component",
                None,
            )
            if close_error is not None:
                owned_next_fd = next_fd
                next_fd = None
                close_error = _close_taken_file_descriptor(
                    owned_next_fd,
                    "close_next_trusted_directory",
                    close_error,
                )
                raise close_error
            current_fd = next_fd
            next_fd = None
            current_path = current_path / component

        opened = os.fstat(current_fd)
        if not stat.S_ISDIR(opened.st_mode):
            raise ValueError(f"artifact path is not a directory: {requested!s}")
        result = OpenedDirectory(
            path=current_path,
            file_descriptor=current_fd,
            device=opened.st_dev,
            inode=opened.st_ino,
        )
        current_fd = None
    except BaseException as exc:
        primary = exc
    finally:
        if current_fd is not None:
            owned_current_fd = current_fd
            current_fd = None
            primary = _close_taken_file_descriptor(
                owned_current_fd,
                "close_trusted_directory",
                primary,
            )
    if primary is not None:
        raise primary
    return result


@contextmanager
def open_results_root(results_path, *, create: bool):
    requested = Path(results_path)
    results_name = requested.name
    if results_name in ("", ".", ".."):
        raise ValueError("results_path must name a CSV file")
    root = open_trusted_directory(requested.parent, create=create)
    with _close_opened_directories(
        [(root, "close_parent_directory")]
    ):
        yield OpenedResultsRoot(root=root, results_name=results_name)


@contextmanager
def _close_opened_directories(resources):
    primary = None
    try:
        try:
            yield
        except BaseException as exc:
            primary = exc
    finally:
        for resource, phase in reversed(resources):
            try:
                resource.close()
            except BaseException as exc:
                if primary is None:
                    primary = exc
                    try:
                        setattr(
                            primary,
                            "descriptor_close_state_uncertain",
                            True,
                        )
                    except BaseException:
                        pass
                else:
                    _attach_artifact_secondary(
                        primary,
                        phase,
                        exc,
                        descriptor_close_state_uncertain=True,
                    )
    if primary is not None:
        raise primary


def open_marker_directory(
    root: OpenedDirectory,
    *,
    create: bool,
) -> OpenedDirectory:
    flags = _directory_flags()
    marker_fd = None
    try:
        try:
            marker_fd = os.open(
                _MARKER_DIRECTORY,
                flags,
                dir_fd=root.file_descriptor,
            )
        except FileNotFoundError:
            if not create:
                raise
            try:
                os.mkdir(
                    _MARKER_DIRECTORY,
                    mode=_DIRECTORY_MODE,
                    dir_fd=root.file_descriptor,
                )
            except FileExistsError:
                pass
            else:
                os.fsync(root.file_descriptor)
            marker_fd = os.open(
                _MARKER_DIRECTORY,
                flags,
                dir_fd=root.file_descriptor,
            )
        opened = os.fstat(marker_fd)
        if not stat.S_ISDIR(opened.st_mode):
            raise ValueError("artifact marker path is not a directory")
        result = OpenedDirectory(
            path=root.path / _MARKER_DIRECTORY,
            file_descriptor=marker_fd,
            device=opened.st_dev,
            inode=opened.st_ino,
        )
        marker_fd = None
        return result
    except BaseException as primary:
        if marker_fd is not None:
            owned_marker_fd = marker_fd
            marker_fd = None
            primary = _close_taken_file_descriptor(
                owned_marker_fd,
                "close_marker_directory_after_open",
                primary,
            )
        raise primary


def _lease_path_matches(
    marker_directory: OpenedDirectory,
    run_id: str,
    lock_file_descriptor: int,
    expected_identity: tuple[int, int] | None,
) -> bool:
    try:
        path_stat = os.stat(
            f"{run_id}.lock",
            dir_fd=marker_directory.file_descriptor,
            follow_symlinks=False,
        )
        held_stat = os.fstat(lock_file_descriptor)
    except OSError:
        return False
    held_identity = (held_stat.st_dev, held_stat.st_ino)
    return (
        stat.S_ISREG(path_stat.st_mode)
        and stat.S_ISREG(held_stat.st_mode)
        and (path_stat.st_dev, path_stat.st_ino) == held_identity
        and (
            expected_identity is None
            or held_identity == expected_identity
        )
    )


@contextmanager
def reservation_lock(
    marker_directory: OpenedDirectory,
    run_id: str,
    *,
    expected_identity: tuple[int, int] | None = None,
):
    """Hold the persistent cross-process lease for one reserved run ID."""
    lock_name = f"{run_id}.lock"
    lock_fd = None
    created = False
    locked = False
    primary = None
    try:
        try:
            lock_fd = os.open(
                lock_name,
                os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                _MARKER_MODE,
                dir_fd=marker_directory.file_descriptor,
            )
            created = True
        except FileExistsError:
            lock_fd = os.open(
                lock_name,
                os.O_RDWR | os.O_NOFOLLOW,
                dir_fd=marker_directory.file_descriptor,
            )
        opened = os.fstat(lock_fd)
        if not stat.S_ISREG(opened.st_mode):
            raise ValueError("run artifact reservation lock is not regular")
        if not _lease_path_matches(
            marker_directory,
            run_id,
            lock_fd,
            expected_identity,
        ):
            raise ValueError(
                "reservation lock identity changed (lease identity) before acquire"
            )
        if created:
            os.fsync(lock_fd)
            os.fsync(marker_directory.file_descriptor)
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        locked = True
        if not _lease_path_matches(
            marker_directory,
            run_id,
            lock_fd,
            expected_identity,
        ):
            raise ValueError(
                "reservation lock identity changed (lease identity) after acquire"
            )
    except BaseException as exc:
        primary = exc
    if primary is None:
        try:
            yield lock_fd
        except BaseException as exc:
            primary = exc
        if not _lease_path_matches(
            marker_directory,
            run_id,
            lock_fd,
            expected_identity,
        ):
            changed = ValueError(
                "reservation lock identity changed (lease identity) before release"
            )
            if primary is None:
                primary = changed
            else:
                _attach_artifact_secondary(
                    primary,
                    "validate_lease_before_release",
                    changed,
                )
    if locked:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
        except BaseException as exc:
            if primary is None:
                primary = exc
            else:
                _attach_artifact_secondary(primary, "release_run_lease", exc)
    if lock_fd is not None:
        try:
            os.close(lock_fd)
        except BaseException as exc:
            if primary is None:
                primary = exc
            else:
                _attach_artifact_secondary(
                    primary,
                    "close_run_lease_descriptor",
                    exc,
                )
    if primary is not None:
        raise primary


def create_reservation_marker(
    opened_root: OpenedResultsRoot,
    marker_directory: OpenedDirectory,
    lock_file_descriptor: int,
    run_id: str,
) -> RunArtifactReservation:
    owner_token = secrets.token_hex(32)
    canonical_results_path = opened_root.results_path
    opened_lease = os.fstat(lock_file_descriptor)
    if not stat.S_ISREG(opened_lease.st_mode) or not _lease_path_matches(
        marker_directory,
        run_id,
        lock_file_descriptor,
        (opened_lease.st_dev, opened_lease.st_ino),
    ):
        raise ValueError(
            "reservation lock identity changed (lease identity) "
            "before marker create"
        )
    payload = {
        "lease_device": opened_lease.st_dev,
        "lease_inode": opened_lease.st_ino,
        "owner_token": owner_token,
        "results_path": str(canonical_results_path),
        "results_root": str(opened_root.root.path),
        "run_id": run_id,
        "schema_version": "1.0",
    }
    encoded = (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")
    marker_name = f"{run_id}.json"
    temporary_name = f".{marker_name}.{uuid.uuid4().hex}.tmp"
    marker_fd = None
    temporary_identity = None
    final_published = False
    primary = None
    reservation = None
    try:
        try:
            os.stat(
                marker_name,
                dir_fd=marker_directory.file_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError(f"run_id is already reserved: {run_id}")
        marker_fd = os.open(
                temporary_name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                _MARKER_MODE,
                dir_fd=marker_directory.file_descriptor,
            )
        _write_all(marker_fd, encoded)
        os.fsync(marker_fd)
        opened_temporary = os.fstat(marker_fd)
        temporary_identity = _regular_file_identity(
            marker_fd,
            "reservation marker temporary artifact",
        )
        reservation = RunArtifactReservation(
            run_id=run_id,
            results_root=opened_root.root.path,
            results_path=canonical_results_path,
            owner_token=owner_token,
            root_device=opened_root.root.device,
            root_inode=opened_root.root.inode,
            marker_device=marker_directory.device,
            marker_inode=marker_directory.inode,
            marker_file_device=opened_temporary.st_dev,
            marker_file_inode=opened_temporary.st_ino,
            lease_device=opened_lease.st_dev,
            lease_inode=opened_lease.st_ino,
        )
        owned_marker_fd = marker_fd
        marker_fd = None
        os.close(owned_marker_fd)
        if not _creation_directories_match(opened_root, marker_directory):
            raise ValueError("reservation directory identity changed before publish")
        link_no_overwrite(
            temporary_name,
            marker_name,
            source_directory_fd=marker_directory.file_descriptor,
            target_directory_fd=marker_directory.file_descriptor,
        )
        final_published = True
        _validate_published_marker(
            opened_root,
            marker_directory,
            reservation,
            temporary_identity,
        )
        os.unlink(
            temporary_name,
            dir_fd=marker_directory.file_descriptor,
        )
        temporary_name = None
        os.fsync(marker_directory.file_descriptor)
        _validate_published_marker(
            opened_root,
            marker_directory,
            reservation,
            temporary_identity,
        )
    except BaseException as exc:
        primary = exc
    finally:
        if marker_fd is not None:
            owned_marker_fd = marker_fd
            marker_fd = None
            try:
                os.close(owned_marker_fd)
            except BaseException as exc:
                if primary is None:
                    primary = exc
                else:
                    _attach_artifact_secondary(
                        primary,
                        "close_marker_temp_descriptor",
                        exc,
                    )
        if final_published and primary is not None:
            rollback_succeeded = False
            try:
                _unlink_owned_entry(
                    marker_directory.file_descriptor,
                    marker_name,
                    temporary_identity,
                    "reservation marker",
                    directory_path=marker_directory.path,
                )
                rollback_succeeded = True
            except _ArtifactEntryIdentityError as exc:
                _attach_artifact_secondary(
                    primary,
                    "rollback_reservation_marker_identity",
                    exc,
                    publication_state_uncertain=True,
                    marker_file_may_remain=True,
                    marker_path=str(marker_directory.path / marker_name),
                    **_artifact_cleanup_recovery_evidence(exc),
                )
                _mark_uncertain_reservation(primary, reservation)
            except BaseException as exc:
                _attach_artifact_secondary(
                    primary,
                    "rollback_reservation_marker",
                    exc,
                    publication_state_uncertain=True,
                    marker_file_may_remain=True,
                    marker_path=str(marker_directory.path / marker_name),
                    **_artifact_cleanup_recovery_evidence(exc),
                )
                _mark_uncertain_reservation(primary, reservation)
            try:
                os.fsync(marker_directory.file_descriptor)
            except BaseException as exc:
                _attach_artifact_secondary(
                    primary,
                    "rollback_marker_directory_fsync",
                    exc,
                    publication_state_uncertain=True,
                    marker_file_may_remain=not rollback_succeeded,
                )
                _mark_uncertain_reservation(primary, reservation)
        if temporary_name is not None:
            try:
                os.unlink(
                    temporary_name,
                    dir_fd=marker_directory.file_descriptor,
                )
            except FileNotFoundError:
                pass
            except BaseException as exc:
                if primary is None:
                    primary = exc
                else:
                    _attach_artifact_secondary(
                        primary,
                        "cleanup_marker_temp",
                        exc,
                        temporary_file_may_remain=True,
                        temporary_path=str(marker_directory.path / temporary_name),
                    )
            else:
                try:
                    os.fsync(marker_directory.file_descriptor)
                except BaseException as exc:
                    if primary is None:
                        primary = exc
                    else:
                        _attach_artifact_secondary(
                            primary,
                            "cleanup_marker_directory_fsync",
                            exc,
                            publication_state_uncertain=True,
                        )
    if primary is not None:
        raise primary
    return reservation


def _creation_directories_match(
    opened_root: OpenedResultsRoot,
    marker_directory: OpenedDirectory,
) -> bool:
    return directory_binding_matches(
        opened_root.root.path,
        opened_root.root.file_descriptor,
    ) and directory_binding_matches(
        marker_directory.path,
        marker_directory.file_descriptor,
    )


def _validate_published_marker(
    opened_root: OpenedResultsRoot,
    marker_directory: OpenedDirectory,
    reservation: RunArtifactReservation,
    temporary_identity: tuple[int, int],
) -> None:
    if not _creation_directories_match(opened_root, marker_directory):
        raise ValueError("reservation directory identity changed during publish")
    opened = os.stat(
        f"{reservation.run_id}.json",
        dir_fd=marker_directory.file_descriptor,
        follow_symlinks=False,
    )
    if not stat.S_ISREG(opened.st_mode) or (
        opened.st_dev,
        opened.st_ino,
    ) != temporary_identity:
        raise ValueError("reservation marker identity changed during publish")
    if _read_marker(marker_directory, reservation.run_id) != {
        "lease_device": reservation.lease_device,
        "lease_inode": reservation.lease_inode,
        "owner_token": reservation.owner_token,
        "results_path": str(reservation.results_path),
        "results_root": str(reservation.results_root),
        "run_id": reservation.run_id,
        "schema_version": "1.0",
    }:
        raise ValueError("reservation marker binding changed during publish")


def _mark_uncertain_reservation(
    primary: BaseException,
    reservation: RunArtifactReservation,
) -> None:
    try:
        setattr(primary, "publication_state_uncertain", True)
        setattr(primary, "marker_file_may_remain", True)
        setattr(primary, "reservation_recovery", reservation)
    except BaseException:
        pass


def _attach_reservation_recovery(
    primary: BaseException,
    reservation: RunArtifactReservation,
) -> None:
    """Expose owner-bound recovery when cleanup fails after marker publish."""
    _mark_uncertain_reservation(primary, reservation)


def _absolute_lexical_path(path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _write_all(file_descriptor: int, value: bytes) -> None:
    position = 0
    while position < len(value):
        written = os.write(file_descriptor, value[position:])
        if written <= 0:
            raise OSError("artifact marker write made no progress")
        position += written


def _regular_file_identity(
    file_descriptor: int,
    description: str,
) -> tuple[int, int]:
    opened = os.fstat(file_descriptor)
    if not stat.S_ISREG(opened.st_mode):
        raise ValueError(f"{description} is not regular")
    return opened.st_dev, opened.st_ino


def _scoped_entry_matches_identity(
    directory_file_descriptor: int,
    name: str,
    expected_identity: tuple[int, int],
    *,
    require_regular: bool = True,
) -> bool:
    try:
        opened = os.stat(
            name,
            dir_fd=directory_file_descriptor,
            follow_symlinks=False,
        )
    except OSError:
        return False
    return (
        (not require_regular or stat.S_ISREG(opened.st_mode))
        and (opened.st_dev, opened.st_ino) == expected_identity
    )


def _scoped_entry_exists(
    directory_file_descriptor: int,
    name: str,
) -> bool:
    try:
        os.stat(
            name,
            dir_fd=directory_file_descriptor,
            follow_symlinks=False,
        )
    except OSError:
        return False
    return True


@dataclass(frozen=True)
class _CleanupQuarantine:
    directory_file_descriptor: int
    name: str
    path: Path
    original_name: str
    original_path: Path

    def stat(self):
        return os.stat(
            self.name,
            dir_fd=self.directory_file_descriptor,
            follow_symlinks=False,
        )

    def mark(
        self,
        primary: BaseException,
        *,
        restored: bool,
        preserved: bool,
        unsupported: bool = False,
    ) -> None:
        _mark_cleanup_recovery(
            primary,
            self.original_path,
            restored=restored,
            preserved=preserved,
            unsupported=unsupported,
            quarantine_path=(
                self.path
                if _scoped_entry_exists(
                    self.directory_file_descriptor,
                    self.name,
                )
                else None
            ),
        )

    def restore(
        self,
        primary: BaseException,
        quarantined_identity: tuple[int, int],
    ) -> bool:
        restored = False
        try:
            _rename_noreplace(
                self.directory_file_descriptor,
                self.name,
                self.directory_file_descriptor,
                self.original_name,
            )
            restored = _scoped_entry_matches_identity(
                self.directory_file_descriptor,
                self.original_name,
                quarantined_identity,
                require_regular=False,
            )
            if not restored:
                raise _ArtifactEntryIdentityError(
                    "quarantined artifact recovery identity changed after restore"
                )
        except BaseException as restore_exc:
            self.mark(
                restore_exc,
                restored=False,
                preserved=_scoped_entry_exists(
                    self.directory_file_descriptor,
                    self.original_name,
                ),
            )
            _attach_artifact_secondary(
                primary,
                "restore_quarantined_entry",
                restore_exc,
                **_artifact_cleanup_recovery_evidence(restore_exc),
            )
        try:
            os.fsync(self.directory_file_descriptor)
        except BaseException as fsync_exc:
            self.mark(
                fsync_exc,
                restored=restored,
                preserved=(
                    restored
                    or _scoped_entry_exists(
                        self.directory_file_descriptor,
                        self.original_name,
                    )
                ),
            )
            _attach_artifact_secondary(
                primary,
                "fsync_cleanup_quarantine",
                fsync_exc,
                **_artifact_cleanup_recovery_evidence(fsync_exc),
            )
        self.mark(
            primary,
            restored=restored,
            preserved=(
                restored
                or _scoped_entry_exists(
                    self.directory_file_descriptor,
                    self.original_name,
                )
            ),
        )
        return restored


def _move_to_cleanup_quarantine(
    directory_file_descriptor: int,
    directory_path: Path,
    original_name: str,
) -> _CleanupQuarantine | None:
    original_path = directory_path / original_name
    for _attempt in range(_CLEANUP_QUARANTINE_ATTEMPTS):
        name = f".artifact-cleanup-{uuid.uuid4().hex}.quarantine"
        quarantine = _CleanupQuarantine(
            directory_file_descriptor,
            name,
            directory_path / name,
            original_name,
            original_path,
        )
        try:
            _rename_noreplace(
                directory_file_descriptor,
                original_name,
                directory_file_descriptor,
                name,
            )
        except FileExistsError:
            continue
        except FileNotFoundError:
            return None
        except ArtifactFilesystemUnsupportedError as exc:
            quarantine.mark(
                exc,
                restored=False,
                preserved=True,
                unsupported=True,
            )
            raise
        except BaseException as exc:
            quarantine.mark(
                exc,
                restored=False,
                preserved=_scoped_entry_exists(
                    directory_file_descriptor,
                    original_name,
                ),
            )
            raise
        return quarantine
    primary = FileExistsError("unable to reserve artifact cleanup quarantine")
    _mark_cleanup_recovery(
        primary,
        original_path,
        restored=False,
        preserved=_scoped_entry_exists(
            directory_file_descriptor,
            original_name,
        ),
    )
    raise primary


def _unlink_owned_entry(
    directory_file_descriptor: int,
    name: str,
    expected_identity: tuple[int, int],
    description: str,
    *,
    directory_path: Path,
) -> bool:
    quarantine = _move_to_cleanup_quarantine(
        directory_file_descriptor,
        directory_path,
        name,
    )
    if quarantine is None:
        return False

    try:
        moved = quarantine.stat()
    except BaseException as exc:
        quarantine.mark(
            exc,
            restored=False,
            preserved=False,
        )
        raise
    moved_identity = moved.st_dev, moved.st_ino
    if stat.S_ISREG(moved.st_mode) and moved_identity == expected_identity:
        try:
            os.unlink(quarantine.name, dir_fd=directory_file_descriptor)
        except BaseException as exc:
            quarantine.restore(exc, moved_identity)
            raise
        return True

    primary = _ArtifactEntryIdentityError(
        f"{description} identity changed before quarantine cleanup"
    )
    quarantine.restore(primary, moved_identity)
    raise primary


def _mark_cleanup_recovery(
    primary: BaseException,
    original_path: Path,
    *,
    restored: bool,
    preserved: bool,
    unsupported: bool = False,
    quarantine_path: Path | None = None,
) -> None:
    try:
        setattr(primary, "publication_state_uncertain", True)
        setattr(primary, "cleanup_original_path", str(original_path))
        setattr(primary, "cleanup_original_restored", restored)
        setattr(primary, "cleanup_original_preserved", preserved)
        if unsupported:
            setattr(primary, "cleanup_operation_unsupported", True)
        if quarantine_path is None:
            try:
                delattr(primary, "cleanup_recovery_path")
            except AttributeError:
                pass
        else:
            setattr(primary, "cleanup_recovery_path", str(quarantine_path))
    except BaseException:
        pass


def _artifact_cleanup_recovery_evidence(exc: BaseException) -> dict:
    evidence = {}
    for name, expected_type in (
        ("cleanup_recovery_path", str),
        ("cleanup_original_path", str),
        ("cleanup_original_restored", bool),
        ("cleanup_original_preserved", bool),
        ("cleanup_operation_unsupported", bool),
    ):
        value = getattr(exc, name, None)
        if type(value) is expected_type:
            evidence[name] = value
    return evidence


def _read_marker_bytes(file_descriptor: int) -> bytes:
    chunks = []
    total = 0
    while True:
        chunk = os.read(
            file_descriptor,
            min(1024, _MAX_MARKER_BYTES + 1 - total),
        )
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)
        total += len(chunk)
        if total > _MAX_MARKER_BYTES:
            raise ValueError("run artifact reservation marker is invalid")


def _read_marker(marker_directory: OpenedDirectory, run_id: str) -> dict:
    marker_name = f"{run_id}.json"
    try:
        marker_fd = os.open(
            marker_name,
            os.O_RDONLY | os.O_NOFOLLOW,
            dir_fd=marker_directory.file_descriptor,
        )
    except FileNotFoundError as exc:
        raise ValueError("run artifact reservation marker is missing") from exc
    try:
        opened = os.fstat(marker_fd)
        if not stat.S_ISREG(opened.st_mode):
            raise ValueError("run artifact reservation marker is not regular")
        encoded = _read_marker_bytes(marker_fd)
        try:
            value = json.loads(encoded.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeError) as exc:
            raise ValueError("run artifact reservation marker is invalid") from exc
    finally:
        if marker_fd is not None:
            os.close(marker_fd)
    if type(value) is not dict:
        raise ValueError("run artifact reservation marker is invalid")
    return value


def _state_payload(
    reservation: RunArtifactReservation,
    state: str,
    *,
    row_fingerprint: str,
    row_timestamp: str | None = None,
) -> dict:
    payload = {
        "owner_token": reservation.owner_token,
        "row_fingerprint": row_fingerprint,
        "run_id": reservation.run_id,
        "schema_version": "1.0",
        "state": state,
    }
    if row_timestamp is not None:
        payload["row_timestamp"] = row_timestamp
    return payload


def _read_state_artifact(
    marker_directory: OpenedDirectory,
    reservation: RunArtifactReservation,
    suffix: str,
    state: str,
    *,
    with_identity: bool = False,
) -> dict | tuple[dict, tuple[int, int]] | None:
    name = f"{reservation.run_id}.{suffix}"
    try:
        file_fd = os.open(
            name,
            os.O_RDONLY | os.O_NOFOLLOW,
            dir_fd=marker_directory.file_descriptor,
        )
    except FileNotFoundError:
        return None
    try:
        opened = os.fstat(file_fd)
        if not stat.S_ISREG(opened.st_mode):
            raise ValueError(f"run artifact {suffix} state is not regular")
        encoded = _read_marker_bytes(file_fd)
        try:
            value = json.loads(encoded.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeError) as exc:
            raise ValueError(f"run artifact {suffix} state is invalid") from exc
    finally:
        os.close(file_fd)
    expected_base = {
        "owner_token": reservation.owner_token,
        "run_id": reservation.run_id,
        "schema_version": "1.0",
        "state": state,
    }
    if type(value) is not dict or any(
        value.get(key) != expected
        for key, expected in expected_base.items()
    ):
        raise ValueError(f"run artifact {suffix} state binding does not match")
    fingerprint = value.get("row_fingerprint")
    if (
        type(fingerprint) is not str
        or len(fingerprint) != 64
        or any(character not in "0123456789abcdef" for character in fingerprint)
    ):
        raise ValueError(
            f"run artifact {suffix} transaction provenance is invalid"
        )
    if state == "csv_committing":
        timestamp = value.get("row_timestamp")
        if type(timestamp) is not str or not timestamp:
            raise ValueError(
                f"run artifact {suffix} transaction provenance is invalid"
            )
        allowed = {*expected_base, "row_fingerprint", "row_timestamp"}
    else:
        allowed = {*expected_base, "row_fingerprint"}
    if set(value) != allowed:
        raise ValueError(
            f"run artifact {suffix} transaction provenance is invalid"
        )
    if with_identity:
        return value, (opened.st_dev, opened.st_ino)
    return value


def reservation_transaction_state(
    verified: VerifiedReservation,
) -> tuple[dict | None, dict | None]:
    reservation = verified.reservation
    pending = _read_state_artifact(
        verified.marker_directory,
        reservation,
        "pending",
        "csv_committing",
    )
    consumed = _read_state_artifact(
        verified.marker_directory,
        reservation,
        "consumed",
        "consumed",
    )
    return pending, consumed


def _run_artifact_authority_exists(
    marker_directory: OpenedDirectory,
    run_id: str,
) -> bool:
    """Return whether any durable async authority occupies this run ID."""
    for suffix in ("json", "pending", "consumed"):
        try:
            os.stat(
                f"{run_id}.{suffix}",
                dir_fd=marker_directory.file_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            continue
        else:
            return True
    return False


def _safe_artifact_error(phase: str, exc: BaseException) -> dict:
    try:
        error_type = type.__getattribute__(type(exc), "__name__")
    except BaseException:
        error_type = "<unknown>"
    if type(error_type) is not str:
        error_type = "<unknown>"
    message = f"<{error_type}>"
    try:
        args = BaseException.args.__get__(exc, type(exc))
    except BaseException:
        args = ()
    if type(args) is tuple and len(args) == 1 and type(args[0]) is str:
        message = args[0]
    return {
        "phase": phase,
        "error_type": error_type,
        "error_message": message,
    }


def _attach_artifact_secondary(
    primary: BaseException,
    phase: str,
    secondary: BaseException,
    **evidence,
) -> None:
    diagnostic = {
        **_safe_artifact_error(phase, secondary),
        **evidence,
    }
    errors = getattr(primary, "persistence_secondary_errors", None)
    if type(errors) is not list:
        errors = []
        try:
            setattr(primary, "persistence_secondary_errors", errors)
        except BaseException:
            pass
    errors.append(diagnostic)
    try:
        primary.add_note(
            f"secondary persistence failure during {phase}: "
            f"{diagnostic['error_type']}: {diagnostic['error_message']}"
        )
    except BaseException:
        pass


def _mark_uncertain_state(primary: BaseException, state_path: Path) -> None:
    try:
        setattr(primary, "publication_state_uncertain", True)
        setattr(primary, "state_file_may_remain", True)
        setattr(primary, "state_path", str(state_path))
    except BaseException:
        pass


def publish_reservation_state(
    verified: VerifiedReservation,
    *,
    suffix: str,
    state: str,
    row_fingerprint: str,
    row_timestamp: str | None = None,
) -> bool:
    """Durably publish an owner-bound state file without overwrite."""
    reservation = verified.reservation
    payload = _state_payload(
        reservation,
        state,
        row_fingerprint=row_fingerprint,
        row_timestamp=row_timestamp,
    )
    existing = _read_state_artifact(
        verified.marker_directory,
        reservation,
        suffix,
        state,
    )
    if existing is not None:
        if existing != payload:
            raise ValueError(
                f"run artifact {suffix} transaction provenance does not match"
            )
        return False
    final_name = f"{reservation.run_id}.{suffix}"
    temporary_name = f".{final_name}.{uuid.uuid4().hex}.tmp"
    encoded = (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    file_fd = None
    temporary_identity = None
    final_published = False
    primary = None
    try:
        file_fd = os.open(
            temporary_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            _MARKER_MODE,
            dir_fd=verified.marker_directory.file_descriptor,
        )
        _write_all(file_fd, encoded)
        os.fsync(file_fd)
        temporary_identity = _regular_file_identity(
            file_fd,
            f"run artifact {suffix} temporary state",
        )
        owned_file_fd = file_fd
        file_fd = None
        os.close(owned_file_fd)
        if not reservation_binding_matches(verified):
            raise ValueError("reservation path identity changed before state publish")
        link_no_overwrite(
            temporary_name,
            final_name,
            source_directory_fd=verified.marker_directory.file_descriptor,
            target_directory_fd=verified.marker_directory.file_descriptor,
        )
        final_published = True
        if not _scoped_entry_matches_identity(
            verified.marker_directory.file_descriptor,
            final_name,
            temporary_identity,
        ):
            raise ValueError(
                f"run artifact {suffix} state identity changed during publication"
            )
        os.unlink(
            temporary_name,
            dir_fd=verified.marker_directory.file_descriptor,
        )
        temporary_name = None
        os.fsync(verified.marker_directory.file_descriptor)
        if not _scoped_entry_matches_identity(
            verified.marker_directory.file_descriptor,
            final_name,
            temporary_identity,
        ):
            raise ValueError(
                f"run artifact {suffix} state identity changed during publication"
            )
        published = _read_state_artifact(
            verified.marker_directory,
            reservation,
            suffix,
            state,
        )
        if not reservation_binding_matches(verified) or published != payload:
            raise ValueError("reservation state changed during publication")
    except BaseException as exc:
        primary = exc
    finally:
        if final_published and primary is not None:
            try:
                _unlink_owned_entry(
                    verified.marker_directory.file_descriptor,
                    final_name,
                    temporary_identity,
                    f"run artifact {suffix} state",
                    directory_path=verified.marker_directory.path,
                )
            except _ArtifactEntryIdentityError as exc:
                _attach_artifact_secondary(
                    primary,
                    "rollback_state_identity",
                    exc,
                    publication_state_uncertain=True,
                    state_file_may_remain=True,
                    state_path=str(
                        verified.marker_directory.path / final_name
                    ),
                    **_artifact_cleanup_recovery_evidence(exc),
                )
                _mark_uncertain_state(
                    primary,
                    verified.marker_directory.path / final_name,
                )
            except BaseException as exc:
                _attach_artifact_secondary(
                    primary,
                    "rollback_state",
                    exc,
                    publication_state_uncertain=True,
                    state_file_may_remain=True,
                    state_path=str(verified.marker_directory.path / final_name),
                    **_artifact_cleanup_recovery_evidence(exc),
                )
                _mark_uncertain_state(
                    primary,
                    verified.marker_directory.path / final_name,
                )
            else:
                try:
                    os.fsync(verified.marker_directory.file_descriptor)
                except BaseException as exc:
                    _attach_artifact_secondary(
                        primary,
                        "rollback_state_directory_fsync",
                        exc,
                        publication_state_uncertain=True,
                    )
        if file_fd is not None:
            owned_file_fd = file_fd
            file_fd = None
            try:
                os.close(owned_file_fd)
            except BaseException as exc:
                if primary is None:
                    primary = exc
                else:
                    _attach_artifact_secondary(primary, "close_state_descriptor", exc)
        if temporary_name is not None:
            try:
                os.unlink(
                    temporary_name,
                    dir_fd=verified.marker_directory.file_descriptor,
                )
            except FileNotFoundError:
                pass
            except BaseException as exc:
                if primary is None:
                    primary = exc
                else:
                    _attach_artifact_secondary(
                        primary,
                        "cleanup_state_temp",
                        exc,
                        temporary_file_may_remain=True,
                        temporary_path=str(
                            verified.marker_directory.path / temporary_name
                        ),
                    )
            else:
                try:
                    os.fsync(verified.marker_directory.file_descriptor)
                except BaseException as exc:
                    if primary is None:
                        primary = exc
                    else:
                        _attach_artifact_secondary(
                            primary,
                            "cleanup_state_directory_fsync",
                            exc,
                            publication_state_uncertain=True,
                        )
    if primary is not None:
        raise primary
    return True


def publish_pending(
    verified: VerifiedReservation,
    row_fingerprint: str,
    row_timestamp: str,
) -> bool:
    return publish_reservation_state(
        verified,
        suffix="pending",
        state="csv_committing",
        row_fingerprint=row_fingerprint,
        row_timestamp=row_timestamp,
    )


def publish_consumed(
    verified: VerifiedReservation,
    row_fingerprint: str,
) -> bool:
    return publish_reservation_state(
        verified,
        suffix="consumed",
        state="consumed",
        row_fingerprint=row_fingerprint,
    )


def clear_pending(
    verified: VerifiedReservation,
    row_fingerprint: str,
) -> None:
    reservation = verified.reservation
    pending_record = _read_state_artifact(
        verified.marker_directory,
        reservation,
        "pending",
        "csv_committing",
        with_identity=True,
    )
    if pending_record is None:
        return
    pending, pending_identity = pending_record
    if pending["row_fingerprint"] != row_fingerprint:
        raise ValueError(
            "run artifact pending transaction provenance does not match"
        )
    pending_name = f"{reservation.run_id}.pending"
    try:
        _unlink_owned_entry(
            verified.marker_directory.file_descriptor,
            pending_name,
            pending_identity,
            "run artifact pending state",
            directory_path=verified.marker_directory.path,
        )
    except _ArtifactEntryIdentityError as exc:
        _mark_uncertain_state(primary=exc, state_path=reservation.pending_path)
        raise
    os.fsync(verified.marker_directory.file_descriptor)
    if not reservation_binding_matches(verified):
        raise ValueError("reservation path identity changed during pending cleanup")


def revalidate_reservation(
    verified: VerifiedReservation,
    *,
    require_active: bool,
) -> None:
    """Revalidate owner, trusted paths, and active state under the run lease."""
    reservation = verified.reservation
    if not reservation_binding_matches(verified):
        raise ValueError("reservation path identity changed")
    if not reservation_lock_binding_matches(verified):
        raise ValueError("reservation lock identity changed")
    marker_stat = os.stat(
        f"{reservation.run_id}.json",
        dir_fd=verified.marker_directory.file_descriptor,
        follow_symlinks=False,
    )
    if not stat.S_ISREG(marker_stat.st_mode) or (
        marker_stat.st_dev,
        marker_stat.st_ino,
    ) != (
        reservation.marker_file_device,
        reservation.marker_file_inode,
    ):
        raise ValueError("reservation marker identity changed")
    marker = _read_marker(verified.marker_directory, reservation.run_id)
    expected = {
        "lease_device": reservation.lease_device,
        "lease_inode": reservation.lease_inode,
        "owner_token": reservation.owner_token,
        "results_path": str(reservation.results_path),
        "results_root": str(reservation.results_root),
        "run_id": reservation.run_id,
        "schema_version": "1.0",
    }
    if marker.get("owner_token") != reservation.owner_token:
        raise ValueError("reservation owner token does not match")
    if marker != expected:
        raise ValueError("reservation marker binding does not match")
    if require_active:
        pending, consumed = reservation_transaction_state(verified)
        if consumed:
            raise ValueError("run artifact reservation has been consumed")
        if pending:
            raise ValueError("run artifact CSV commit recovery is pending")


@contextmanager
def verify_reservation(
    reservation,
    run_id: str,
    *,
    results_root=None,
    results_path=None,
    require_active: bool = True,
):
    """Verify a reservation against its durable marker and trusted paths."""
    if type(reservation) is not RunArtifactReservation:
        raise ValueError("a valid RunArtifactReservation is required")
    lease_device = getattr(reservation, "lease_device", None)
    lease_inode = getattr(reservation, "lease_inode", None)
    if type(lease_device) is not int or type(lease_inode) is not int:
        raise ValueError(
            "legacy reservation marker has no durable lease identity; "
            "allocate a new run_id"
        )
    if type(run_id) is not str or run_id != reservation.run_id:
        raise ValueError("reservation run_id does not match")
    if results_root is not None:
        requested_root = _absolute_lexical_path(results_root)
        if requested_root != reservation.results_root:
            raise ValueError("reservation results root does not match")
    if results_path is not None:
        requested_results = _absolute_lexical_path(results_path)
        if requested_results != reservation.results_path:
            raise ValueError("reservation results_path does not match")

    root = open_trusted_directory(reservation.results_root, create=False)
    resources = [(root, "close_parent_directory")]
    with _close_opened_directories(resources):
        if (root.device, root.inode) != (
            reservation.root_device,
            reservation.root_inode,
        ):
            raise ValueError("reservation results root identity changed")
        marker_directory = open_marker_directory(root, create=False)
        resources.append((marker_directory, "close_marker_directory"))
        if (marker_directory.device, marker_directory.inode) != (
            reservation.marker_device,
            reservation.marker_inode,
        ):
            raise ValueError("reservation marker directory identity changed")
        with reservation_lock(
            marker_directory,
            run_id,
            expected_identity=(lease_device, lease_inode),
        ) as lock_fd:
            verified = VerifiedReservation(
                reservation=reservation,
                root=root,
                marker_directory=marker_directory,
                results_name=reservation.results_path.name,
                lock_file_descriptor=lock_fd,
            )
            revalidate_reservation(
                verified,
                require_active=require_active,
            )
            yield verified
            revalidate_reservation(
                verified,
                require_active=require_active,
            )


def recover_run_artifact_reservation(
    reservation: RunArtifactReservation,
) -> RunArtifactReservation:
    """Explicitly recover a valid marker left by an uncertain create result."""
    if type(reservation) is not RunArtifactReservation:
        raise ValueError("a valid RunArtifactReservation recovery value is required")
    with verify_reservation(
        reservation,
        reservation.run_id,
        results_path=reservation.results_path,
        require_active=True,
    ):
        pass
    return reservation


def directory_binding_matches(path: Path, file_descriptor: int) -> bool:
    try:
        current = open_trusted_directory(path, create=False)
    except (OSError, ValueError):
        return False
    try:
        pinned = os.fstat(file_descriptor)
        return (current.device, current.inode) == (pinned.st_dev, pinned.st_ino)
    finally:
        current.close()


def reservation_binding_matches(verified: VerifiedReservation) -> bool:
    return directory_binding_matches(
        verified.root.path,
        verified.root.file_descriptor,
    ) and directory_binding_matches(
        verified.marker_directory.path,
        verified.marker_directory.file_descriptor,
    )


def reservation_lock_binding_matches(verified: VerifiedReservation) -> bool:
    """Return whether the lease pathname still names the flocked inode."""
    return _lease_path_matches(
        verified.marker_directory,
        verified.reservation.run_id,
        verified.lock_file_descriptor,
        (
            verified.reservation.lease_device,
            verified.reservation.lease_inode,
        ),
    )


def link_no_overwrite(
    source: str,
    target: str,
    *,
    source_directory_fd: int,
    target_directory_fd: int,
) -> None:
    """Publish one same-directory file without overwrite or fallback."""
    if not _HARD_LINK_PUBLICATION_SUPPORTED:
        raise ArtifactFilesystemUnsupportedError(
            "async artifact persistence requires POSIX same-filesystem "
            "hard-link no-overwrite publication"
        )
    try:
        os.link(
            source,
            target,
            src_dir_fd=source_directory_fd,
            dst_dir_fd=target_directory_fd,
            follow_symlinks=False,
        )
    except (NotImplementedError, TypeError) as exc:
        raise ArtifactFilesystemUnsupportedError(
            "async artifact persistence requires POSIX same-filesystem "
            "hard-link no-overwrite publication"
        ) from exc
    except OSError as exc:
        if exc.errno == errno.EPERM:
            raise ArtifactFilesystemUnsupportedError(
                "async artifact persistence requires POSIX hard-link "
                "permission and filesystem capability for no-overwrite "
                "publication"
            ) from exc
        unsupported_errors = {
            errno.EXDEV,
            errno.ENOSYS,
            getattr(errno, "ENOTSUP", errno.EOPNOTSUPP),
            errno.EOPNOTSUPP,
        }
        if exc.errno in unsupported_errors:
            raise ArtifactFilesystemUnsupportedError(
                "async artifact persistence requires POSIX same-filesystem "
                "hard-link no-overwrite publication"
            ) from exc
        raise
