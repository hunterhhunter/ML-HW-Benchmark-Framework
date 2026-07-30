"""Compile a fixed-batch ResNet50 RBLN artifact."""

import json
from collections.abc import Sequence

from tools.rbln_compile_recipes.common import (
    RecipeContract,
    TensorContract,
    create_parser,
    emit_description_or_require_output,
    save_and_validate,
)


CONTRACT = RecipeContract(
    recipe="resnet50",
    model_id="torchvision/resnet50-imagenet1k-v2",
    inputs=(TensorContract("input_np", (1, 3, 224, 224), "float32"),),
    outputs=(TensorContract("output", (1, 1000), "float32"),),
    allow_unnamed_outputs=True,
    notes=("weights=IMAGENET1K_V2",),
)


def compile_model():
    import rebel
    from torchvision.models import ResNet50_Weights, resnet50

    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).eval()
    model.requires_grad_(False)
    return rebel.compile_from_torch(
        model,
        [("input_np", [1, 3, 224, 224], "float32")],
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)
    output = emit_description_or_require_output(args, CONTRACT)
    if output is None:
        return 0

    report = save_and_validate(compile_model(), output, CONTRACT)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
