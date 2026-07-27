"""Compile and run TorchVision ImageNet ResNet50 strictly on Furiosa RNGD."""

from __future__ import annotations

import argparse
import copy
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np


@dataclass(frozen=True)
class CompileCheckConfig:
    device: str = "furiosa:0"
    seed: int = 0
    torch_home: Path | None = None


@dataclass(frozen=True)
class CompileCheckResult:
    first_call_seconds: float
    warm_call_seconds: float
    cpu_top1: int
    npu_top1: int
    max_abs_diff: float
    output_shape: tuple[int, ...]


@dataclass(frozen=True)
class _Dependencies:
    torch: Any
    furiosa_torch: Any
    CompilerConfig: Any
    TacticHintConfig: Any
    resnet50: Any
    imagenet_v2_weights: Any


def _load_dependencies() -> _Dependencies:
    import torch
    import furiosa.torch as furiosa_torch
    from furiosa.torch.config import CompilerConfig, TacticHintConfig
    from torchvision.models import ResNet50_Weights, resnet50

    return _Dependencies(
        torch=torch,
        furiosa_torch=furiosa_torch,
        CompilerConfig=CompilerConfig,
        TacticHintConfig=TacticHintConfig,
        resnet50=resnet50,
        imagenet_v2_weights=ResNet50_Weights.IMAGENET1K_V2,
    )


def _print_stage(message: str) -> None:
    print(message, flush=True)


def _completed_cpu_output(output: Any):
    return output.detach().cpu().float()


def _as_numpy(output: Any, *, label: str) -> np.ndarray:
    value = np.asarray(output.numpy())
    if tuple(value.shape) != (1, 1000):
        raise RuntimeError(
            f"{label} output shape mismatch: expected (1, 1000), "
            f"got {tuple(value.shape)}"
        )
    if not np.isfinite(value).all():
        raise RuntimeError(f"{label} output contains non-finite values")
    return value


def _validate_outputs(cpu_output: Any, npu_output: Any) -> tuple[int, int, float]:
    cpu_array = _as_numpy(cpu_output, label="CPU")
    npu_array = _as_numpy(npu_output, label="RNGD")
    cpu_top1 = int(cpu_array.argmax(axis=1).item())
    npu_top1 = int(npu_array.argmax(axis=1).item())
    if cpu_top1 != npu_top1:
        raise RuntimeError(
            f"CPU/RNGD Top-1 mismatch: cpu={cpu_top1}, rngd={npu_top1}"
        )
    max_abs_diff = float(np.max(np.abs(cpu_array - npu_array)))
    return cpu_top1, npu_top1, max_abs_diff


def run_compile_check(
    config: CompileCheckConfig,
    *,
    dependencies: Any | None = None,
    timer: Callable[[], float] = time.perf_counter,
    emit: Callable[[str], None] = _print_stage,
) -> CompileCheckResult:
    if config.torch_home is not None:
        torch_home = config.torch_home.expanduser().resolve()
        os.environ["TORCH_HOME"] = str(torch_home)
        emit(f"[Furiosa ResNet50] TORCH_HOME={torch_home}")

    dependencies = dependencies or _load_dependencies()
    torch = dependencies.torch

    emit("[Furiosa ResNet50] ImageNet V2 weights/model load: START")
    torch.manual_seed(config.seed)
    cpu_model = dependencies.resnet50(
        weights=dependencies.imagenet_v2_weights
    ).eval()
    emit("[Furiosa ResNet50] ImageNet V2 weights/model load: COMPLETE")

    cpu_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)
    emit("[Furiosa ResNet50] CPU reference inference: START")
    with torch.inference_mode():
        cpu_output = _completed_cpu_output(cpu_model(cpu_input))
    _as_numpy(cpu_output, label="CPU")
    emit("[Furiosa ResNet50] CPU reference inference: COMPLETE")

    torch_device = torch.device(config.device)
    npu_model = copy.deepcopy(cpu_model).to(torch_device)
    npu_input = cpu_input.to(torch_device)
    compiler_config = dependencies.CompilerConfig(
        tactic_hint=dependencies.TacticHintConfig.Default
    )
    backend = dependencies.furiosa_torch.backend.with_config(
        compiler_config,
        eager_fallback=False,
    )
    compiled = torch.compile(
        npu_model,
        backend=backend,
        fullgraph=True,
        dynamic=False,
    )

    emit("[Furiosa ResNet50] strict compile + first inference: START")
    first_started = timer()
    with torch.inference_mode():
        first_output = _completed_cpu_output(compiled(npu_input))
    first_call_seconds = timer() - first_started
    _as_numpy(first_output, label="RNGD first call")
    emit(
        "[Furiosa ResNet50] strict compile + first inference: "
        f"COMPLETE ({first_call_seconds:.6f}s)"
    )

    emit("[Furiosa ResNet50] warm inference: START")
    warm_started = timer()
    with torch.inference_mode():
        warm_output = _completed_cpu_output(compiled(npu_input))
    warm_call_seconds = timer() - warm_started
    cpu_top1, npu_top1, max_abs_diff = _validate_outputs(
        cpu_output,
        warm_output,
    )
    emit(
        "[Furiosa ResNet50] warm inference: "
        f"COMPLETE ({warm_call_seconds:.6f}s)"
    )

    return CompileCheckResult(
        first_call_seconds=first_call_seconds,
        warm_call_seconds=warm_call_seconds,
        cpu_top1=cpu_top1,
        npu_top1=npu_top1,
        max_abs_diff=max_abs_diff,
        output_shape=tuple(warm_output.shape),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download TorchVision ImageNet V2 ResNet50 and strictly compile/run "
            "it on Furiosa RNGD."
        )
    )
    parser.add_argument("--device", default="furiosa:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--torch-home", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_compile_check(
        CompileCheckConfig(
            device=args.device,
            seed=args.seed,
            torch_home=args.torch_home,
        )
    )
    print("=" * 60)
    print("Furiosa RNGD TorchVision ResNet50 compile check: PASS")
    print(f"output_shape: {result.output_shape}")
    print(f"CPU/RNGD Top-1: {result.cpu_top1}/{result.npu_top1}")
    print(f"max_abs_diff: {result.max_abs_diff:.8f}")
    print(f"compile + first inference: {result.first_call_seconds:.6f}s")
    print(f"warm inference: {result.warm_call_seconds:.6f}s")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
