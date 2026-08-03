import json


def test_known_signature_prefers_specific_compiler_message():
    from tools import furiosa_compile_repro as repro

    text = """
    furiosa.UnsupportedOpError: failed to compile the graph
    EdgeIndex(162) has empty transition cost table
    """

    assert repro.match_known_signature(text) == (
        "EdgeIndex(162) has empty transition cost table"
    )


def test_safe_error_line_keeps_only_exception_type_and_first_line():
    from tools import furiosa_compile_repro as repro

    exc = RuntimeError("first line\nsecret prompt and path")

    assert repro.safe_error_line(exc) == "RuntimeError: first line"


def test_write_json_serializes_result_contract(tmp_path):
    from tools import furiosa_compile_repro as repro

    result = repro.CaseResult(
        case="resnet50",
        status="failed",
        stages=(repro.StageResult("rngd_first_inference", "failed"),),
        error_type="RuntimeError",
        error_line="RuntimeError: compiler panic",
    )
    output_path = tmp_path / "nested" / "result.json"

    repro.write_json(output_path, result)

    assert json.loads(output_path.read_text()) == {
        "case": "resnet50",
        "status": "failed",
        "stages": [
            {
                "name": "rngd_first_inference",
                "status": "failed",
                "detail": None,
            }
        ],
        "output_shapes": [],
        "error_type": "RuntimeError",
        "error_line": "RuntimeError: compiler panic",
        "matched_known_signature": None,
    }
