"""Rule-based pattern detection over events + stats — the OBSERVATIONS stage from
BUILD_PLAN.md Layer 6. Each observation attaches the real event IDs it's grounded
in wherever the pattern is directly traceable to specific events (SPEC.md 'Report
grounding contract'). Patterns computed purely from control-time distribution
(no single event backs them) are still produced for context, but carry no
evidence IDs — the REPORT stage's validation pass will drop any LLM statement
that doesn't cite a real event ID, so an ungrounded observation can inform tone
and framing but can never become an unsupported claim in the final report.
"""

from __future__ import annotations

MAX_EVIDENCE_IDS = 6


def detect_observations(events: list[dict], stats: dict) -> list[dict]:
    observations: list[dict] = []
    by_athlete = stats.get("by_athlete", {})

    for athlete in ("user", "opponent"):
        athlete_stats = by_athlete.get(athlete, {})
        observations.extend(_conversion_observations(events, athlete, athlete_stats))
        observations.extend(_defense_observations(events, athlete, athlete_stats))

    observations.extend(_control_time_observation(events, stats))
    observations.extend(_scramble_observation(stats))
    return observations


def _conversion_observations(events: list[dict], athlete: str, athlete_stats: dict) -> list[dict]:
    attempts = athlete_stats.get("shot_attempts", 0)
    conversion_rate = athlete_stats.get("conversion_rate")
    if attempts < 2 or conversion_rate is None:
        return []

    relevant_types = {"shot_attempt", "defended_shot", "takedown"}
    evidence = _event_ids(events, athlete, relevant_types)

    if attempts >= 3 and conversion_rate < 0.34:
        return [{
            "type": "low_conversion",
            "summary": (
                f"{athlete}: {attempts} shot attempts, "
                f"{athlete_stats.get('takedowns', 0)} finished "
                f"({round(conversion_rate * 100)}% conversion)."
            ),
            "evidence_event_ids": evidence,
            "stats": {"attempts": attempts, "conversion_rate": conversion_rate},
        }]
    if attempts >= 2 and conversion_rate >= 0.6:
        return [{
            "type": "strong_finishing",
            "summary": (
                f"{athlete}: {athlete_stats.get('takedowns', 0)} of {attempts} shots "
                f"finished ({round(conversion_rate * 100)}% conversion)."
            ),
            "evidence_event_ids": evidence,
            "stats": {"attempts": attempts, "conversion_rate": conversion_rate},
        }]
    return []


def _defense_observations(events: list[dict], athlete: str, athlete_stats: dict) -> list[dict]:
    conceded = athlete_stats.get("takedowns_conceded", 0)
    if conceded < 2:
        return []
    opponent = "opponent" if athlete == "user" else "user"
    evidence = _event_ids(events, opponent, {"takedown"})
    return [{
        "type": "defense_leak",
        "summary": f"{athlete}: conceded {conceded} takedowns this match.",
        "evidence_event_ids": evidence,
        "stats": {"takedowns_conceded": conceded},
    }]


def _control_time_observation(events: list[dict], stats: dict) -> list[dict]:
    control = stats.get("control_time_ms", {})
    user_ms = control.get("user", 0)
    opponent_ms = control.get("opponent", 0)
    total = user_ms + opponent_ms
    if total < 10_000:
        return []
    share_diff = abs(user_ms - opponent_ms) / total
    if share_diff < 0.3:
        return []
    dominant = "user" if user_ms > opponent_ms else "opponent"
    evidence = _event_ids(events, None, {"takedown", "escape"})
    return [{
        "type": "control_imbalance",
        "summary": (
            f"Control time split {round(user_ms / 1000)}s user / "
            f"{round(opponent_ms / 1000)}s opponent — {dominant} controlled the majority."
        ),
        "evidence_event_ids": evidence,
        "stats": {"user_control_ms": user_ms, "opponent_control_ms": opponent_ms},
    }]


def _scramble_observation(stats: dict) -> list[dict]:
    longest = stats.get("longest_scramble_ms", 0)
    if longest < 8_000:
        return []
    return [{
        "type": "long_scramble",
        "summary": f"Longest scramble ran {round(longest / 1000, 1)}s with no clean position established.",
        "evidence_event_ids": [],
        "stats": {"longest_scramble_ms": longest},
    }]


def _event_ids(events: list[dict], initiator: str | None, types: set[str]) -> list[str]:
    matches = [
        event for event in events
        if event["type"] in types and (initiator is None or event.get("initiator") == initiator)
    ]
    matches.sort(key=lambda event: event["start_ms"])
    return [event["id"] for event in matches[:MAX_EVIDENCE_IDS]]
