"""Compile the pinned YOLOv5m raw prediction head for RBLN-CA22."""

import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from tools.rbln_compile_recipes.common import (
    RecipeContract,
    TensorContract,
    create_parser,
    emit_description_or_require_output,
    save_and_validate,
)


EXPECTED_YOLOV5_REVISION = "86fd1ab270cb2f7e53ee7412cd4a0650bf4bcc51"
INPUT_SHAPE = (1, 3, 640, 640)
OUTPUT_SHAPE = (1, 25200, 85)
CONTRACT = RecipeContract(
    recipe="yolov5m",
    model_id="ultralytics/yolov5m",
    inputs=(TensorContract("input_np", INPUT_SHAPE, "float32"),),
    outputs=(TensorContract("output", OUTPUT_SHAPE, "float32"),),
    allow_unnamed_outputs=True,
    notes=(
        EXPECTED_YOLOV5_REVISION,
        "raw prediction head only; no NMS or AutoShape",
    ),
)


def validate_sources(yolov5_root: str | Path, weights: str | Path) -> tuple[Path, Path]:
    """Validate the exact local YOLOv5 checkout and non-empty weight file."""
    root = Path(yolov5_root).expanduser().resolve()
    weight_path = Path(weights).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"YOLOv5 source root does not exist: {root}")
    for required_file in ("models/experimental.py", "models/yolo.py"):
        if not (root / required_file).is_file():
            raise FileNotFoundError(
                f"YOLOv5 source root is missing required file: {root / required_file}"
            )
    if not weight_path.is_file():
        raise FileNotFoundError(f"YOLOv5 weight file does not exist: {weight_path}")
    if weight_path.stat().st_size == 0:
        raise ValueError(f"YOLOv5 weight file is empty: {weight_path}")

    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        text=True,
        capture_output=True,
    )
    revision = result.stdout.strip()
    if revision != EXPECTED_YOLOV5_REVISION:
        raise RuntimeError(
            f"YOLOv5 source revision {revision} does not match validated "
            f"{EXPECTED_YOLOV5_REVISION}; run git -C {root} checkout "
            f"{EXPECTED_YOLOV5_REVISION}"
        )
    return root, weight_path


def compile_model(yolov5_root: str | Path, weights: str | Path):
    """Load the raw YOLOv5 prediction head and compile its fixed batch-one ABI."""
    root, weight_path = validate_sources(yolov5_root, weights)
    sys.path.insert(0, str(root))

    import rebel
    import torch
    from models.experimental import attempt_load

    class YoloV5Raw(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, input_np):
            output = self.model(input_np)
            if isinstance(output, (tuple, list)):
                output = output[0]
            return output

    model = attempt_load(str(weight_path), map_location="cpu").fuse().eval()
    model.requires_grad_(False)
    raw_model = YoloV5Raw(model).eval()
    raw_model.requires_grad_(False)
    with torch.no_grad():
        cpu_output = raw_model(torch.zeros(INPUT_SHAPE, dtype=torch.float32))
    if tuple(cpu_output.shape) != OUTPUT_SHAPE:
        raise ValueError(
            f"YOLOv5 raw output shape mismatch: expected {OUTPUT_SHAPE}, "
            f"got {tuple(cpu_output.shape)}"
        )
    return rebel.compile_from_torch(
        raw_model,
        [("input_np", [1, 3, 640, 640], "float32")],
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = create_parser()
    parser.add_argument("--yolov5-root", default=None)
    parser.add_argument("--weights", default=None)
    args = parser.parse_args(argv)
    output = emit_description_or_require_output(args, CONTRACT)
    if output is None:
        return 0
    if args.yolov5_root is None:
        raise ValueError("YOLOv5 compilation requires --yolov5-root")
    if args.weights is None:
        raise ValueError("YOLOv5 compilation requires --weights")

    report = save_and_validate(
        compile_model(args.yolov5_root, args.weights), output, CONTRACT
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
