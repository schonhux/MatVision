from ml.reporting.stats import compute_match_stats


def _segment(state, start, end, controlling=None):
    return {"state": state, "start_ms": start, "end_ms": end, "controlling": controlling}


def _event(event_type, start, end, initiator=None, event_id=None):
    return {
        "id": event_id or f"{event_type}-{start}",
        "type": event_type,
        "start_ms": start,
        "end_ms": end,
        "initiator": initiator,
    }


def test_duration_by_state_sums_segments():
    segments = [
        _segment("neutral", 0, 5000),
        _segment("top", 5000, 8000, controlling="user"),
        _segment("scramble", 8000, 9000),
    ]
    stats = compute_match_stats([], segments)
    assert stats["duration_ms_by_state"]["neutral"] == 5000
    assert stats["duration_ms_by_state"]["top"] == 3000
    assert stats["duration_ms_by_state"]["scramble"] == 1000
    assert stats["duration_ms_by_state"]["bottom"] == 0
    assert stats["total_duration_ms"] == 9000


def test_control_time_maps_top_to_user_bottom_to_opponent():
    segments = [
        _segment("top", 0, 4000, controlling="user"),
        _segment("bottom", 4000, 7000, controlling="opponent"),
    ]
    stats = compute_match_stats([], segments)
    assert stats["control_time_ms"] == {"user": 4000, "opponent": 3000}


def test_conversion_rate_computed_per_athlete():
    events = [
        _event("shot_attempt", 0, 1000, initiator="user"),
        _event("shot_attempt", 2000, 3000, initiator="user"),
        _event("shot_attempt", 4000, 5000, initiator="user"),
        _event("takedown", 5000, 5500, initiator="user"),
    ]
    segments = [_segment("neutral", 0, 6000)]
    stats = compute_match_stats(events, segments)
    user = stats["by_athlete"]["user"]
    assert user["shot_attempts"] == 3
    assert user["takedowns"] == 1
    assert user["conversion_rate"] == round(1 / 3, 4)


def test_conversion_rate_is_none_with_no_attempts():
    stats = compute_match_stats([], [_segment("neutral", 0, 1000)])
    assert stats["by_athlete"]["user"]["conversion_rate"] is None
    assert stats["by_athlete"]["opponent"]["conversion_rate"] is None


def test_takedowns_conceded_counts_opponent_initiated_takedowns():
    events = [_event("takedown", 1000, 1500, initiator="opponent")]
    stats = compute_match_stats(events, [_segment("neutral", 0, 2000)])
    assert stats["by_athlete"]["user"]["takedowns_conceded"] == 1
    assert stats["by_athlete"]["opponent"]["takedowns_conceded"] == 0


def test_longest_scramble_and_scramble_count():
    segments = [
        _segment("scramble", 0, 2000),
        _segment("neutral", 2000, 3000),
        _segment("scramble", 3000, 9000),
    ]
    stats = compute_match_stats([], segments)
    assert stats["scramble_count"] == 2
    assert stats["longest_scramble_ms"] == 6000


def test_restarts_counted_regardless_of_initiator():
    events = [_event("restart", 0, 500), _event("restart", 1000, 1500)]
    stats = compute_match_stats(events, [_segment("stopped", 0, 2000)])
    assert stats["restarts"] == 2


def test_empty_inputs_do_not_crash():
    stats = compute_match_stats([], [])
    assert stats["total_duration_ms"] == 0
    assert stats["scramble_count"] == 0
    assert stats["by_athlete"]["user"]["shot_attempts"] == 0
