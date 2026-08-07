import pytest

from ml.reporting.tone import coach_tone_instruction


def test_extreme_tone_stays_evidence_grounded():
    instruction = coach_tone_instruction("extreme")
    assert "high-intensity" in instruction
    assert "cited evidence" in instruction
    assert "never insult" in instruction


def test_unknown_tone_is_rejected():
    with pytest.raises(ValueError, match="Unknown coach tone"):
        coach_tone_instruction("unhinged")
