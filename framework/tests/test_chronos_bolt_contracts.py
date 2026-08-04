import json

import pytest

from chronos_bolt.contracts import (
    ChronosBoltContract,
    CompileStatus,
    TensorContract,
)
from chronos_bolt.evidence import write_result


def test_tiny_contract_exposes_fixed_external_and_core_abi():
    """Catches an ABI change that would make vendor artifacts incomparable."""
    contract = ChronosBoltContract.tiny(d_model=128)

    assert contract.external_input == TensorContract("context", (1, 512), "float32")
    assert contract.external_output == TensorContract(
        "quantile_preds", (1, 9, 64), "float32"
    )
    assert contract.core_inputs == (
        TensorContract("input_embeds", (1, 33, 128), "float32"),
        TensorContract("attention_mask", (1, 33), "float32"),
        TensorContract("decoder_input_embeds", (1, 1, 128), "float32"),
    )
    assert contract.core_output == contract.external_output
    assert contract.quantile_levels == (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def test_contract_can_represent_a_checkpoint_without_a_reg_token():
    """Catches hard-coding Tiny's learned REG token into every checkpoint variant."""
    contract = ChronosBoltContract.tiny(d_model=128, use_reg_token=False)

    assert contract.core_inputs[:2] == (
        TensorContract("input_embeds", (1, 32, 128), "float32"),
        TensorContract("attention_mask", (1, 32), "float32"),
    )


def test_contract_rejects_non_positive_model_dimension():
    """Catches a malformed checkpoint-derived core ABI before compiler invocation."""
    with pytest.raises(ValueError, match="d_model"):
        ChronosBoltContract.tiny(d_model=0)


def test_result_writer_preserves_terminal_status_and_refuses_overwrite(tmp_path):
    """Catches loss of compile evidence or accidental replacement of a prior run."""
    destination = tmp_path / "result.json"
    payload = {
        "status": CompileStatus.COMPILED.value,
        "artifact": {"path": "tiny.rbln", "size_bytes": 1},
    }

    written = write_result(destination, payload)

    assert written == destination
    assert json.loads(destination.read_text()) == payload
    with pytest.raises(FileExistsError, match="already exists"):
        write_result(destination, payload)


def test_compile_status_distinguishes_artifact_and_device_evidence():
    """Catches a report that promotes compile-only evidence to device verification."""
    assert CompileStatus.COMPILED.value == "compiled"
    assert CompileStatus.DEVICE_VERIFIED.value == "device_verified"
    assert CompileStatus.COMPILED is not CompileStatus.DEVICE_VERIFIED
