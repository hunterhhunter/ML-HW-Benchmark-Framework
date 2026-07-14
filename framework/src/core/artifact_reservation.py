"""Durable ownership and trusted-path primitives for async run artifacts."""

import errno
import json
import os
import secrets
import stat
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path


_MARKER_DIRECTORY = ".run_artifacts"
_DIRECTORY_MODE = 0o755
_MARKER_MODE = 0o600
_MAX_MARKER_BYTES = 4096
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
        if self.file_descriptor is not None:
            os.close(self.file_descriptor)
            self.file_descriptor = None


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
            os.close(current_fd)
            current_fd = next_fd
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
        return result
    finally:
        if current_fd is not None:
            os.close(current_fd)


@contextmanager
def open_results_root(results_path, *, create: bool):
    requested = Path(results_path)
    results_name = requested.name
    if results_name in ("", ".", ".."):
        raise ValueError("results_path must name a CSV file")
    root = open_trusted_directory(requested.parent, create=create)
    try:
        yield OpenedResultsRoot(root=root, results_name=results_name)
    finally:
        root.close()


def _open_marker_directory(
    root: OpenedDirectory,
    *,
    create: bool,
) -> OpenedDirectory:
    flags = _directory_flags()
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
    return OpenedDirectory(
        path=root.path / _MARKER_DIRECTORY,
        file_descriptor=marker_fd,
        device=opened.st_dev,
        inode=opened.st_ino,
    )


def create_reservation_marker(
    opened_root: OpenedResultsRoot,
    run_id: str,
) -> RunArtifactReservation:
    marker_directory = _open_marker_directory(opened_root.root, create=True)
    owner_token = secrets.token_hex(32)
    canonical_results_path = opened_root.results_path
    payload = {
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
    marker_fd = None
    try:
        try:
            marker_fd = os.open(
                marker_name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                _MARKER_MODE,
                dir_fd=marker_directory.file_descriptor,
            )
        except FileExistsError as exc:
            raise FileExistsError(f"run_id is already reserved: {run_id}") from exc
        _write_all(marker_fd, encoded)
        os.fsync(marker_fd)
        os.close(marker_fd)
        marker_fd = None
        os.fsync(marker_directory.file_descriptor)
        return RunArtifactReservation(
            run_id=run_id,
            results_root=opened_root.root.path,
            results_path=canonical_results_path,
            owner_token=owner_token,
            root_device=opened_root.root.device,
            root_inode=opened_root.root.inode,
            marker_device=marker_directory.device,
            marker_inode=marker_directory.inode,
        )
    finally:
        if marker_fd is not None:
            os.close(marker_fd)
        marker_directory.close()


def _absolute_lexical_path(path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _write_all(file_descriptor: int, value: bytes) -> None:
    position = 0
    while position < len(value):
        written = os.write(file_descriptor, value[position:])
        if written <= 0:
            raise OSError("artifact marker write made no progress")
        position += written


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


def _reservation_is_consumed(
    marker_directory: OpenedDirectory,
    run_id: str,
) -> bool:
    try:
        consumed = os.stat(
            f"{run_id}.consumed",
            dir_fd=marker_directory.file_descriptor,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        return False
    if not stat.S_ISREG(consumed.st_mode):
        raise ValueError("run artifact reservation consumed marker is invalid")
    return True


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
    marker_directory = None
    try:
        if (root.device, root.inode) != (
            reservation.root_device,
            reservation.root_inode,
        ):
            raise ValueError("reservation results root identity changed")
        marker_directory = _open_marker_directory(root, create=False)
        if (marker_directory.device, marker_directory.inode) != (
            reservation.marker_device,
            reservation.marker_inode,
        ):
            raise ValueError("reservation marker directory identity changed")
        marker = _read_marker(marker_directory, run_id)
        expected = {
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
        if require_active and _reservation_is_consumed(marker_directory, run_id):
            raise ValueError("run artifact reservation has been consumed")
        yield VerifiedReservation(
            reservation=reservation,
            root=root,
            marker_directory=marker_directory,
            results_name=reservation.results_path.name,
        )
    finally:
        if marker_directory is not None:
            marker_directory.close()
        root.close()


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


def consume_reservation(verified: VerifiedReservation) -> None:
    """Permanently consume a verified reservation before its CSV commit."""
    if not reservation_binding_matches(verified):
        raise ValueError("reservation path identity changed before consumption")
    name = f"{verified.reservation.run_id}.consumed"
    payload = (
        json.dumps(
            {
                "owner_token": verified.reservation.owner_token,
                "run_id": verified.reservation.run_id,
                "state": "consumed",
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    try:
        file_fd = os.open(
            name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            _MARKER_MODE,
            dir_fd=verified.marker_directory.file_descriptor,
        )
    except FileExistsError as exc:
        raise ValueError("run artifact reservation has been consumed") from exc
    try:
        _write_all(file_fd, payload)
        os.fsync(file_fd)
    finally:
        os.close(file_fd)
    os.fsync(verified.marker_directory.file_descriptor)
    if not reservation_binding_matches(verified):
        raise ValueError("reservation path identity changed during consumption")


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
