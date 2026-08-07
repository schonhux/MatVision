from ml.reporting.observations import detect_observations
from ml.reporting.stats import compute_match_stats


def _event(event_type, start, end, initiator, event_id):
    return {"id": event_id, "type": event_type, "start_ms": start, "end_ms": end, "initiator": initiator}


def test_low_conversion_observation_cites_relevant_events():
    events = [
        _event("shot_attempt", 0, 500, "user", "s1"),
        _event("defended_shot", 0, 2000, "user", "s1-def"),
        _event("shot_attempt", 3000, 3500, "user", "s2"),
        _event("defended_shot", 3000, 5000, "user", "s2-def"),
        _event("shot_attempt", 6000, 6500, "user", "s3"),
        _event("defended_shot", 6000, 8000, "user", "s3-def"),
    ]
    stats = {"by_athlete": {
        "user": {"shot_attempts": 3, "takedowns": 0, "conversion_rate": 0.0, "takedowns_conceded": 0},
        "opponent": {"shot_attempts": 0, "takedowns": 0, "conversion_rate": None, "takedowns_conceded": 0},
    }, "control_time_ms": {"user": 0, "opponent": 0}, "longest_scramble_ms": 0}

    observations = detect_observations(events, stats)
    types = [o["type"] for o in observations]
    assert "low_conversion" in types
    low_conversion = next(o for o in observations if o["type"] == "low_conversion")
    assert set(low_conversion["evidence_event_ids"]) <= {e["id"] for e in events}
    assert low_conversion["evidence_event_ids"], "must cite at least one real event"


def test_strong_finishing_observation_for_high_conversion():
    events = [
        _event("shot_attempt", 0, 500, "user", "s1"),
        _event("takedown", 500, 1000, "user", "t1"),
        _event("shot_attempt", 2000, 2500, "user", "s2"),
        _event("takedown", 2500, 3000, "user", "t2"),
    ]
    stats = {"by_athlete": {
        "user": {"shot_attempts": 2, "takedowns": 2, "conversion_rate": 1.0, "takedowns_conceded": 0},
        "opponent": {"shot_attempts": 0, "takedowns": 0, "conversion_rate": None, "takedowns_conceded": 0},
    }, "control_time_ms": {"user": 0, "opponent": 0}, "longest_scramble_ms": 0}

    observations = detect_observations(events, stats)
    types = [o["type"] for o in observations]
    assert "strong_finishing" in types
    assert "low_conversion" not in types


def test_defense_leak_observation_when_conceding_multiple_takedowns():
    events = [
        _event("takedown", 0, 500, "opponent", "t1"),
        _event("takedown", 5000, 5500, "opponent", "t2"),
    ]
    stats = {"by_athlete": {
        "user": {"shot_attempts": 0, "takedowns": 0, "conversion_rate": None, "takedowns_conceded": 2},
        "opponent": {"shot_attempts": 0, "takedowns": 2, "conversion_rate": None, "takedowns_conceded": 0},
    }, "control_time_ms": {"user": 0, "opponent": 0}, "longest_scramble_ms": 0}

    observations = detect_observations(events, stats)
    leak = next(o for o in observations if o["type"] == "defense_leak")
    assert leak["evidence_event_ids"] == ["t1", "t2"]


def test_no_defense_leak_below_threshold():
    stats = {"by_athlete": {
        "user": {"shot_attempts": 0, "takedowns": 0, "conversion_rate": None, "takedowns_conceded": 1},
        "opponent": {"shot_attempts": 0, "takedowns": 1, "conversion_rate": None, "takedowns_conceded": 0},
    }, "control_time_ms": {"user": 0, "opponent": 0}, "longest_scramble_ms": 0}
    observations = detect_observations([], stats)
    assert not any(o["type"] == "defense_leak" for o in observations)


def test_control_imbalance_flagged_when_lopsided():
    stats = {
        "by_athlete": {"user": {"shot_attempts": 0, "conversion_rate": None, "takedowns_conceded": 0},
                        "opponent": {"shot_attempts": 0, "conversion_rate": None, "takedowns_conceded": 0}},
        "control_time_ms": {"user": 40000, "opponent": 5000},
        "longest_scramble_ms": 0,
    }
    observations = detect_observations([], stats)
    assert any(o["type"] == "control_imbalance" for o in observations)


def test_control_imbalance_not_flagged_when_close():
    stats = {
        "by_athlete": {"user": {"shot_attempts": 0, "conversion_rate": None, "takedowns_conceded": 0},
                        "opponent": {"shot_attempts": 0, "conversion_rate": None, "takedowns_conceded": 0}},
        "control_time_ms": {"user": 20000, "opponent": 18000},
        "longest_scramble_ms": 0,
    }
    observations = detect_observations([], stats)
    assert not any(o["type"] == "control_imbalance" for o in observations)


def test_long_scramble_observation_has_no_evidence_by_design():
    stats = {
        "by_athlete": {"user": {"shot_attempts": 0, "conversion_rate": None, "takedowns_conceded": 0},
                        "opponent": {"shot_attempts": 0, "conversion_rate": None, "takedowns_conceded": 0}},
        "control_time_ms": {"user": 0, "opponent": 0},
        "longest_scramble_ms": 12000,
    }
    observations = detect_observations([], stats)
    scramble = next(o for o in observations if o["type"] == "long_scramble")
    assert scramble["evidence_event_ids"] == []


def test_end_to_end_with_real_compute_match_stats():
    """Sanity check that observations wire up cleanly against real stats output,
    not just hand-built stat dicts.
    """
    events = [
        _event("shot_attempt", 0, 500, "opponent", "s1"),
        _event("takedown", 500, 1000, "opponent", "t1"),
        _event("shot_attempt", 5000, 5500, "opponent", "s2"),
        _event("takedown", 5500, 6000, "opponent", "t2"),
    ]
    segments = [
        {"state": "neutral", "start_ms": 0, "end_ms": 1000, "controlling": None},
        {"state": "bottom", "start_ms": 1000, "end_ms": 5000, "controlling": "opponent"},
        {"state": "neutral", "start_ms": 5000, "end_ms": 6000, "controlling": None},
        {"state": "bottom", "start_ms": 6000, "end_ms": 10000, "controlling": "opponent"},
    ]
    stats = compute_match_stats(events, segments)
    observations = detect_observations(events, stats)
    assert any(o["type"] == "defense_leak" for o in observations)
