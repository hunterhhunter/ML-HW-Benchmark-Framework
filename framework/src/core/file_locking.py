"""Preserve POSIX flock semantics on Linux Yocto Python builds without fcntl."""

import ctypes
import errno
import sys


try:
    import fcntl as fcntl
except ModuleNotFoundError as exc:
    if exc.name != "fcntl" or not sys.platform.startswith("linux"):
        raise

    class _LinuxFcntlFallback:
        LOCK_SH = 1
        LOCK_EX = 2
        LOCK_NB = 4
        LOCK_UN = 8

        def __init__(self):
            self._flock = ctypes.CDLL(None, use_errno=True).flock
            self._flock.argtypes = (ctypes.c_int, ctypes.c_int)
            self._flock.restype = ctypes.c_int

        def flock(self, file_descriptor, operation: int) -> None:
            fd = file_descriptor if isinstance(file_descriptor, int) else file_descriptor.fileno()
            while True:
                if self._flock(fd, operation) == 0:
                    return
                error = ctypes.get_errno()
                if error != errno.EINTR:
                    raise OSError(error, "flock failed")

    fcntl = _LinuxFcntlFallback()
