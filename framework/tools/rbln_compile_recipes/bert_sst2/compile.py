"""Compile a fixed-batch BERT SST-2 RBLN artifact."""

import json
from collections.abc import Sequence
from dataclasses import replace

from tools.rbln_compile_recipes.common import (
    RecipeContract,
    TensorContract,
    create_parser,
    emit_description_or_require_output,
    save_and_validate,
)


DEFAULT_MODEL_ID = "textattack/bert-base-uncased-SST-2"
CONTRACT = RecipeContract(
    recipe="bert_sst2",
    model_id=DEFAULT_MODEL_ID,
    inputs=(
        TensorContract("input_ids", (1, 128), "int64"),
        TensorContract("attention_mask", (1, 128), "int64"),
    ),
    outputs=(TensorContract("logits", (1, 2), "float32"),),
)


def compile_model(model_id: str):
    import rebel
    import torch
    from transformers import AutoModelForSequenceClassification

    class BertSst2(torch.nn.Module):
        def __init__(self, model_id):
            super().__init__()
            self.model = AutoModelForSequenceClassification.from_pretrained(model_id).eval()

        def forward(self, input_ids, attention_mask):
            return self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            ).logits

    model = BertSst2(model_id).eval()
    model.requires_grad_(False)
    return rebel.compile_from_torch(
        model,
        [
            ("input_ids", [1, 128], "int64"),
            ("attention_mask", [1, 128], "int64"),
        ],
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = create_parser()
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    args = parser.parse_args(argv)
    output = emit_description_or_require_output(args, CONTRACT)
    if output is None:
        return 0

    selected_contract = replace(CONTRACT, model_id=args.model_id)
    report = save_and_validate(compile_model(args.model_id), output, selected_contract)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
