"""REPORT stage core: builds the coaching prompt, calls Claude, and — most
importantly — validates the response against the evidence before anything is
shown to the athlete.

BUILD_PLAN.md Layer 6 acceptance criteria this file exists to satisfy:
  - "0 unsupported claims across 5 test matches (manual review)."
  - "Validation pass provably rejects a planted unsupported sentence (unit test)."

`generate_report_content()` is the only function here that touches the network;
everything else (prompt construction, JSON parsing, validation) is pure and
tested without a live API key. The Anthropic import is lazy so importing this
module — and every module that imports it — never requires the `anthropic`
package to be installed, matching the lazy-heavy-import pattern the CV stages
already use for torch (ADR-007).
"""

from __future__ import annotations

import json

from ml.reporting.tone import coach_tone_instruction

DEFAULT_MODEL = "claude-sonnet-5"

SYSTEM_PROMPT_TEMPLATE = """You are a wrestling coach reviewing a folkstyle match on film with the athlete who wrestled it.

You will receive a JSON evidence object with match stats, detected patterns \
(observations), and a list of events with IDs, types, timestamps, and \
measurements. This evidence is the ONLY thing you know about the match — you did \
not watch the video.

Hard rules, never break these:
1. Every statement you make must cite at least one real "evidence_event_id" from \
the evidence you were given. Never invent an event ID.
2. Never invent a technique, score, opponent detail, or outcome that is not in \
the evidence.
3. Clearly separate objective observation ("kind": "observation") from your \
interpretation of it ("kind": "interpretation").
4. If the evidence is sparse or a pattern is ambiguous, say so — do not guess.
5. Give exactly ONE practical priority for what to work on next, grounded in \
cited evidence.
6. Never diagnose an injury or make a medical claim.
7. Push hard on technique and effort, never on the athlete's identity or worth \
as a person.

Tone for this report: {tone_instruction}

Respond with ONLY a JSON object (no markdown fences, no commentary) matching \
exactly this shape:
{{
  "summary": "one or two sentence overview",
  "statements": [
    {{"text": "...", "kind": "observation", "evidence_event_ids": ["..."]}},
    {{"text": "...", "kind": "interpretation", "evidence_event_ids": ["..."]}}
  ],
  "priority": {{"text": "...", "evidence_event_ids": ["..."]}}
}}
"""


class ReportGenerationError(Exception):
    """Raised when the report can't be produced — missing API key, network
    failure, or a response that isn't valid JSON. The worker stage turns this
    into a StageError with a user-safe message.
    """


def build_system_prompt(tone: str) -> str:
    return SYSTEM_PROMPT_TEMPLATE.format(tone_instruction=coach_tone_instruction(tone))


def generate_report_content(
    evidence: dict,
    tone: str,
    api_key: str | None,
    model: str = DEFAULT_MODEL,
) -> dict:
    if not api_key:
        raise ReportGenerationError(
            "ANTHROPIC_API_KEY is not configured. Set it in the worker environment "
            "to enable report generation (see SPEC.md — Claude API is the one paid "
            "dependency)."
        )

    import anthropic  # lazy: report generation is the only thing that needs this

    client = anthropic.Anthropic(api_key=api_key)
    try:
        response = client.messages.create(
            model=model,
            max_tokens=2000,
            system=build_system_prompt(tone),
            messages=[{"role": "user", "content": json.dumps(evidence)}],
        )
    except Exception as exc:  # noqa: BLE001 — surface any SDK/network error uniformly
        raise ReportGenerationError(f"Claude API request failed: {exc}") from exc

    text = "".join(block.text for block in response.content if getattr(block, "type", None) == "text")
    return _parse_json_response(text)


def _parse_json_response(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ReportGenerationError(f"Model response was not valid JSON: {exc}") from exc


def validate_report(content: dict, valid_event_ids: set[str]) -> dict:
    """The anti-hallucination gate. Any statement — including the priority — that
    cites zero valid event IDs is dropped rather than shown to the athlete. IDs
    that don't belong to this match are silently stripped from the ones that do
    have at least one real citation, rather than failing the whole statement.
    """
    kept = []
    dropped = 0
    for statement in content.get("statements", []):
        ids = [
            event_id for event_id in statement.get("evidence_event_ids", [])
            if event_id in valid_event_ids
        ]
        if not ids:
            dropped += 1
            continue
        kept.append({
            "text": statement.get("text", ""),
            "kind": statement.get("kind", "observation"),
            "evidence_event_ids": ids,
        })

    priority = content.get("priority")
    validated_priority = None
    if priority:
        ids = [
            event_id for event_id in priority.get("evidence_event_ids", [])
            if event_id in valid_event_ids
        ]
        if ids:
            validated_priority = {"text": priority.get("text", ""), "evidence_event_ids": ids}
        else:
            dropped += 1

    return {
        "summary": content.get("summary", ""),
        "statements": kept,
        "priority": validated_priority,
        "dropped_statement_count": dropped,
    }
