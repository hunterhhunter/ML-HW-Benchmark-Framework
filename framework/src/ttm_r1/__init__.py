"""Fixed-shape TTM-R1 host and compiler interfaces."""

from .contracts import TTMR1Contract

__all__ = ["TTMR1Contract", "TTMR1HostAdapter"]


def __getattr__(name: str):
    if name == "TTMR1HostAdapter":
        from .host_adapter import TTMR1HostAdapter

        return TTMR1HostAdapter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
