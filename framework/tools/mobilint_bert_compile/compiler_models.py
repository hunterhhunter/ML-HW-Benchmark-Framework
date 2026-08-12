"""PyTorch wrappers imported only when a compiler model is constructed."""

from __future__ import annotations

import torch


class BertQuestionAnsweringForCompiler(torch.nn.Module):
    """Avoid Tensor.split(int), which qbcompiler 1.2 cannot lower."""

    def __init__(self, source_model: object):
        super().__init__()
        self.bert = source_model.bert
        self.qa_outputs = source_model.qa_outputs

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
    ):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=False,
        )
        logits = self.qa_outputs(outputs[0])
        return {
            "start_logits": logits[..., 0].contiguous(),
            "end_logits": logits[..., 1].contiguous(),
        }
