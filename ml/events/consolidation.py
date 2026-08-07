from __future__ import annotations

from dataclasses import replace

from ml.events.rules import EventCandidate


def consolidate_events(candidates: list[EventCandidate], merge_gap_ms: int = 800) -> list[EventCandidate]:
    valid = [candidate for candidate in candidates if _valid(candidate)]
    merged: list[EventCandidate] = []

    for candidate in sorted(valid, key=lambda item: (item.type, item.initiator or "", item.start_ms)):
        if merged and _mergeable(merged[-1], candidate, merge_gap_ms):
            merged[-1] = _merge(merged[-1], candidate)
        else:
            merged.append(candidate)

    return sorted(_deduplicate_transitions(merged), key=lambda item: (item.start_ms, item.type))


def temporal_iou(left: EventCandidate, right: EventCandidate) -> float:
    overlap = max(0, min(left.end_ms, right.end_ms) - max(left.start_ms, right.start_ms))
    union = max(left.end_ms, right.end_ms) - min(left.start_ms, right.start_ms)
    return overlap / union if union else 0.0


def _valid(candidate: EventCandidate) -> bool:
    if candidate.start_ms < 0 or candidate.end_ms <= candidate.start_ms:
        return False
    allowed = {
        "shot_attempt": candidate.state_before in {"neutral", "scramble"},
        "defended_shot": candidate.state_before in {"neutral", "scramble"}
        and candidate.state_after in {"neutral", "scramble", "stopped"},
        "takedown": candidate.state_before in {"neutral", "scramble"}
        and candidate.state_after in {"top", "bottom"},
        "escape": candidate.state_before in {"top", "bottom"}
        and candidate.state_after == "neutral",
        "restart": candidate.state_before == "stopped"
        and candidate.state_after != "stopped",
    }
    return allowed.get(candidate.type, False)


def _mergeable(left: EventCandidate, right: EventCandidate, gap_ms: int) -> bool:
    return (
        left.type == right.type
        and left.initiator == right.initiator
        and right.start_ms <= left.end_ms + gap_ms
    )


def _merge(left: EventCandidate, right: EventCandidate) -> EventCandidate:
    strongest = left if left.confidence >= right.confidence else right
    return replace(
        strongest,
        start_ms=min(left.start_ms, right.start_ms),
        end_ms=max(left.end_ms, right.end_ms),
        confidence=round(max(left.confidence, right.confidence), 4),
    )


def _deduplicate_transitions(events: list[EventCandidate]) -> list[EventCandidate]:
    result = []
    for candidate in sorted(events, key=lambda item: item.confidence, reverse=True):
        duplicate = any(
            existing.type == candidate.type
            and existing.initiator == candidate.initiator
            and abs(existing.peak_ms - candidate.peak_ms) < 2000
            for existing in result
        )
        if not duplicate:
            result.append(candidate)
    return result
