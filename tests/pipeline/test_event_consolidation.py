from ml.events.consolidation import consolidate_events
from ml.events.rules import EventCandidate


def candidate(event_type, start, end, before, after, confidence=0.7):
    return EventCandidate(
        type=event_type,
        start_ms=start,
        peak_ms=(start + end) // 2,
        end_ms=end,
        initiator="user",
        outcome="successful",
        confidence=confidence,
        state_before=before,
        state_after=after,
    )


def test_consolidation_merges_duplicates_and_keeps_best_confidence():
    events = consolidate_events([
        candidate("takedown", 1000, 2600, "neutral", "top", 0.65),
        candidate("takedown", 1200, 2800, "neutral", "top", 0.88),
    ])
    assert len(events) == 1
    assert events[0].start_ms == 1000
    assert events[0].end_ms == 2800
    assert events[0].confidence == 0.88


def test_consolidation_rejects_impossible_transition():
    events = consolidate_events([
        candidate("takedown", 1000, 2000, "top", "neutral"),
        candidate("escape", 3000, 4000, "neutral", "top"),
        candidate("restart", 5000, 6000, "neutral", "neutral"),
    ])
    assert events == []
