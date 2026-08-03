"""Core contracts for strict Furiosa RNGD compile-failure reproduction."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class StageResult:
    name: str
    status: str
    detail: str | None = None


@dataclass(frozen=True)
class CaseResult:
    case: str
    status: str
    stages: tuple[StageResult, ...]
    output_shapes: tuple[tuple[int, ...], ...] = ()
    error_type: str | None = None
    error_line: str | None = None
    matched_known_signature: str | None = None


KNOWN_SIGNATURES = (
    "align_up_required (true) != false (false)",
    "EinsumByDpe should be given only a single pass",
    "called `Option::unwrap()` on a `None` value",
    "EdgeIndex(162) has empty transition cost table",
    "mutable op violation",
    "aten._native_batch_norm_legit",
    "Tensor device mismatch! Expected: furiosa:0, Got: cpu",
    "Cannot view a tensor with shape torch.Size([7, 512, 16, 64])",
)


def match_known_signature(text: str) -> str | None:
    """Return the first stable historical Furiosa error signature in text."""
    return next((signature for signature in KNOWN_SIGNATURES if signature in text), None)


def safe_error_line(exc: BaseException) -> str:
    """Return a bounded one-line exception summary suitable for JSON reports."""
    message = next(
        (line.strip() for line in str(exc).splitlines() if line.strip()),
        "",
    )
    summary = type(exc).__name__
    if message:
        summary = f"{summary}: {message}"
    return summary[:500]


def _json_default(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_json(path: Path, payload: Any) -> None:
    """Atomically enough for diagnostics: create the directory and write JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, default=_json_default, indent=2, sort_keys=True) + "\n"
    )


__all__ = [
    "CaseResult",
    "KNOWN_SIGNATURES",
    "StageResult",
    "match_known_signature",
    "safe_error_line",
    "write_json",
]
