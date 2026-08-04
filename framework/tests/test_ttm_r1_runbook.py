from pathlib import Path


def test_ttm_r1_runbook_mentions_all_vendor_modes():
    """Catches a compiler path landing without a copyable operator command."""
    runbook = Path(__file__).parents[1] / "docs" / "ttm-r1-cross-vendor.md"
    text = runbook.read_text(encoding="utf-8")

    assert "--vendor rbln" in text
    assert "--vendor furiosa" in text
    assert "--vendor mobilint" in text
    assert "[1,512,1]" in text
    assert "[1,96,1]" in text
