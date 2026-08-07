"""Match statistics — SPEC.md item 8: attempts, conversion, control time, scramble
duration. Pure math over already-persisted events + state segments, so it's fully
unit-testable without a database or the LLM (see ADR-007 testing philosophy: keep
logic separable from I/O and heavy dependencies).

This is the base layer the OBSERVATIONS stage and the REPORT prompt both build on
— everything the coach's note says about volume/conversion/control ultimately
traces back to a number computed here, not a model guess.
"""

from __future__ import annotations

from ml.states import STATE_LABELS

ATHLETES = ("user", "opponent")


def compute_match_stats(events: list[dict], state_segments: list[dict]) -> dict:
    """events: dicts with at least type/initiator/start_ms/end_ms.
    state_segments: dicts with at least state/start_ms/end_ms.
    """
    segments = sorted(state_segments, key=lambda item: item["start_ms"])

    duration_by_state = {state: 0 for state in STATE_LABELS}
    for segment in segments:
        state = segment["state"]
        span = max(0, int(segment["end_ms"]) - int(segment["start_ms"]))
        duration_by_state[state] = duration_by_state.get(state, 0) + span

    control_time_ms = {
        "user": duration_by_state.get("top", 0),
        "opponent": duration_by_state.get("bottom", 0),
    }

    scramble_durations = [
        int(segment["end_ms"]) - int(segment["start_ms"])
        for segment in segments
        if segment["state"] == "scramble"
    ]

    total_duration_ms = 0
    if segments:
        total_duration_ms = max(int(s["end_ms"]) for s in segments) - min(
            int(s["start_ms"]) for s in segments
        )

    by_athlete = {athlete: _athlete_stats(events, athlete) for athlete in ATHLETES}
    restarts = sum(1 for event in events if event["type"] == "restart")

    return {
        "total_duration_ms": int(total_duration_ms),
        "duration_ms_by_state": {k: int(v) for k, v in duration_by_state.items()},
        "control_time_ms": {k: int(v) for k, v in control_time_ms.items()},
        "scramble_count": len(scramble_durations),
        "longest_scramble_ms": int(max(scramble_durations, default=0)),
        "restarts": restarts,
        "by_athlete": by_athlete,
    }


def _athlete_stats(events: list[dict], athlete: str) -> dict:
    opponent = "opponent" if athlete == "user" else "user"

    shot_attempts = [e for e in events if e["type"] == "shot_attempt" and e.get("initiator") == athlete]
    takedowns = [e for e in events if e["type"] == "takedown" and e.get("initiator") == athlete]
    defended = [e for e in events if e["type"] == "defended_shot" and e.get("initiator") == athlete]
    escapes = [e for e in events if e["type"] == "escape" and e.get("initiator") == athlete]
    conceded = [e for e in events if e["type"] == "takedown" and e.get("initiator") == opponent]

    attempts = len(shot_attempts)
    converted = len(takedowns)
    conversion_rate = round(converted / attempts, 4) if attempts else None

    return {
        "shot_attempts": attempts,
        "takedowns": converted,
        "defended_shots": len(defended),
        "conversion_rate": conversion_rate,
        "escapes": len(escapes),
        "takedowns_conceded": len(conceded),
    }
