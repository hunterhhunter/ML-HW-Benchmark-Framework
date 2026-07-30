"""Compile a fixed-batch, three-input BERT SQuAD RBLN artifact."""

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


DEFAULT_MODEL_ID = "csarron/bert-base-uncased-squad-v1"
SEQUENCE_LENGTH = 384
CONTRACT = RecipeContract(
    recipe="bert_squad",
    model_id=DEFAULT_MODEL_ID,
    inputs=(
        TensorContract("input_ids", (1, SEQUENCE_LENGTH), "int64"),
        TensorContract("attention_mask", (1, SEQUENCE_LENGTH), "int64"),
        TensorContract("token_type_ids", (1, SEQUENCE_LENGTH), "int64"),
    ),
    outputs=(
        TensorContract("start_logits", (1, SEQUENCE_LENGTH), "float32"),
        TensorContract("end_logits", (1, SEQUENCE_LENGTH), "float32"),
    ),
    allow_unnamed_outputs=True,
    notes=(
        "output[0]=start_logits; output[1]=end_logits",
        "run a real CPU/NPU output mapping check before creating model.rbln.json",
    ),
)


def compile_model(model_id: str):
    import rebel
    import torch
    from transformers import AutoModelForQuestionAnswering

    class BertSquad(torch.nn.Module):
        def __init__(self, model_id):
            super().__init__()
            self.model = AutoModelForQuestionAnswering.from_pretrained(model_id).eval()

        def forward(self, input_ids, attention_mask, token_type_ids):
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                return_dict=True,
            )
            return outputs.start_logits, outputs.end_logits

    model = BertSquad(model_id).eval()
    model.requires_grad_(False)
    return rebel.compile_from_torch(
        model,
        [
            ("input_ids", [1, SEQUENCE_LENGTH], "int64"),
            ("attention_mask", [1, SEQUENCE_LENGTH], "int64"),
            ("token_type_ids", [1, SEQUENCE_LENGTH], "int64"),
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
