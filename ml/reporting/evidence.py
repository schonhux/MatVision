"""Assembles the structured evidence JSON that is the *only* thing the report LLM
ever sees (SPEC.md 'Report grounding contract': "LLM receives only structured
evidence JSON ... LLM never sees raw video"). Keeping this assembly in one pure
function makes the contract auditable — the exact payload sent to Claude is
whatever this function returns, nothing more.
"""

from __future__ import annotations

EVENT_FIELDS = (
    "id", "type", "start_ms", "peak_ms", "end_ms",
    "initiator", "outcome", "confidence", "measurements",
)


def build_evidence_json(
    match: dict,
    events: list[dict],
    stats: dict,
    observations: list[dict],
) -> dict:
    return {
        "match": {
            "title": match.get("title", "Untitled match"),
            "style": match.get("style", "folkstyle"),
            "duration_seconds": match.get("duration_seconds"),
        },
        "stats": stats,
        "observations": [
            {
                "type": observation["type"],
                "summary": observation["summary"],
                "evidence_event_ids": observation.get("evidence_event_ids", []),
            }
            for observation in observations
        ],
        "events": [
            {field: event.get(field) for field in EVENT_FIELDS}
            for event in events
        ],
    }
