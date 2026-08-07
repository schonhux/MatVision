import pytest

from ml.reporting.llm import (
    ReportGenerationError,
    build_system_prompt,
    generate_report_content,
    validate_report,
)


def test_build_system_prompt_embeds_tone_instruction():
    prompt = build_system_prompt("extreme")
    assert "high-intensity" in prompt
    assert "cited evidence" in prompt
    assert "never insult" in prompt


def test_build_system_prompt_rejects_unknown_tone():
    with pytest.raises(ValueError, match="Unknown coach tone"):
        build_system_prompt("unhinged")


def test_generate_report_content_requires_api_key():
    with pytest.raises(ReportGenerationError, match="ANTHROPIC_API_KEY"):
        generate_report_content({"events": []}, "balanced", api_key=None)


# --- validate_report: the anti-hallucination gate ---------------------------
# BUILD_PLAN.md Layer 6 acceptance criteria: "Validation pass provably rejects a
# planted unsupported sentence (unit test)." This is that test.

def test_planted_unsupported_statement_is_dropped():
    content = {
        "summary": "Overview.",
        "statements": [
            {"text": "Grounded claim.", "kind": "observation", "evidence_event_ids": ["evt-1"]},
            {"text": "Fabricated claim with no basis.", "kind": "interpretation", "evidence_event_ids": []},
            {"text": "Claim citing a fake event.", "kind": "observation", "evidence_event_ids": ["does-not-exist"]},
        ],
        "priority": {"text": "Work on finishing shots.", "evidence_event_ids": ["evt-1"]},
    }
    result = validate_report(content, valid_event_ids={"evt-1", "evt-2"})

    assert len(result["statements"]) == 1
    assert result["statements"][0]["text"] == "Grounded claim."
    assert result["dropped_statement_count"] == 2
    assert result["priority"]["text"] == "Work on finishing shots."


def test_ungrounded_priority_is_dropped_not_kept_silently():
    content = {
        "summary": "Overview.",
        "statements": [],
        "priority": {"text": "Do something vague.", "evidence_event_ids": []},
    }
    result = validate_report(content, valid_event_ids={"evt-1"})
    assert result["priority"] is None
    assert result["dropped_statement_count"] == 1


def test_partially_grounded_statement_keeps_only_valid_ids():
    content = {
        "summary": "",
        "statements": [
            {
                "text": "Mixed citations.",
                "kind": "observation",
                "evidence_event_ids": ["evt-1", "fake-id", "evt-2"],
            }
        ],
        "priority": None,
    }
    result = validate_report(content, valid_event_ids={"evt-1", "evt-2"})
    assert result["statements"][0]["evidence_event_ids"] == ["evt-1", "evt-2"]
    assert result["dropped_statement_count"] == 0


def test_empty_content_produces_empty_but_valid_result():
    result = validate_report({}, valid_event_ids=set())
    assert result["statements"] == []
    assert result["priority"] is None
    assert result["dropped_statement_count"] == 0
    assert result["summary"] == ""


def test_all_statements_ungrounded_yields_empty_report_without_crashing():
    content = {
        "statements": [
            {"text": "No evidence at all.", "kind": "observation", "evidence_event_ids": []}
            for _ in range(5)
        ],
        "priority": {"text": "Also ungrounded.", "evidence_event_ids": []},
    }
    result = validate_report(content, valid_event_ids={"evt-1"})
    assert result["statements"] == []
    assert result["priority"] is None
    assert result["dropped_statement_count"] == 6
