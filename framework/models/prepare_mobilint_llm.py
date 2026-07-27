#!/usr/bin/env python3
"""Download complete official Mobilint Llama Model Zoo repositories."""

from __future__ import annotations

import argparse
from numbers import Integral
from pathlib import Path
from typing import Callable


_REPOSITORIES = {
    "llama-3.1-8b": {
        1: "mobilint/Llama-3.1-8B-Instruct",
        16: "mobilint/Llama-3.1-8B-Instruct-Batch16",
        32: "mobilint/Llama-3.1-8B-Instruct-Batch32",
    },
    "llama-3.2-3b": {
        1: "mobilint/Llama-3.2-3B-Instruct",
        16: "mobilint/Llama-3.2-3B-Instruct-Batch16",
        32: "mobilint/Llama-3.2-3B-Instruct-Batch32",
    },
}


def _resolve_download(model: str, batch_capacity: int) -> tuple[str, str]:
    if model not in _REPOSITORIES:
        raise ValueError(
            f"model must be one of {', '.join(sorted(_REPOSITORIES))}."
        )
    if (
        isinstance(batch_capacity, bool)
        or not isinstance(batch_capacity, Integral)
        or batch_capacity not in _REPOSITORIES[model]
    ):
        raise ValueError("batch_capacity must be one of 1, 16, or 32.")
    capacity = int(batch_capacity)
    variant = "standard" if capacity == 1 else f"batch{capacity}"
    return _REPOSITORIES[model][capacity], variant


def download_model(
    *,
    model: str,
    batch_capacity: int,
    output_root: str | Path,
    revision: str = "main",
    snapshot_download: Callable | None = None,
) -> Path:
    repo_id, variant = _resolve_download(model, batch_capacity)
    destination = (Path(output_root) / model / variant).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    if snapshot_download is None:
        try:
            from huggingface_hub import snapshot_download as hub_download
        except (ImportError, ModuleNotFoundError) as exc:
            raise ImportError(
                "Mobilint Llama download requires the optional huggingface_hub package."
            ) from exc
        snapshot_download = hub_download
    snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=str(destination),
    )
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download an official Mobilint ARIES Llama repository.",
    )
    parser.add_argument("--model", choices=sorted(_REPOSITORIES), required=True)
    parser.add_argument(
        "--batch-capacity",
        type=int,
        choices=(1, 16, 32),
        default=1,
        help=(
            "Compiled maximum batch capacity. This is not the mandatory runtime "
            "batch size."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parent / "mobilint",
    )
    parser.add_argument("--revision", default="main")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    destination = download_model(
        model=args.model,
        batch_capacity=args.batch_capacity,
        output_root=args.output_root,
        revision=args.revision,
    )
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
