import pandas as pd

from ml.events.rules import detect_events


def test_transition_rules_detect_takedown_escape_and_restart():
    features = pd.DataFrame({
        "timestamp_ms": list(range(0, 7000, 125)),
        "closing_speed": [0.0] * 56,
        "bbox_closing_speed": [0.0] * 56,
    })
    states = [
        {"state": "neutral", "start_ms": 0, "end_ms": 2000, "controlling": None, "confidence": 0.8},
        {"state": "top", "start_ms": 2000, "end_ms": 4000, "controlling": "user", "confidence": 0.9},
        {"state": "neutral", "start_ms": 4000, "end_ms": 5000, "controlling": None, "confidence": 0.85},
        {"state": "stopped", "start_ms": 5000, "end_ms": 6000, "controlling": None, "confidence": 0.95},
        {"state": "neutral", "start_ms": 6000, "end_ms": 7000, "controlling": None, "confidence": 0.9},
    ]

    events = detect_events(features, states)
    types = [event.type for event in events]
    assert "takedown" in types
    assert "escape" in types
    assert "restart" in types
    assert "shot_attempt" in types
    assert next(event for event in events if event.type == "takedown").initiator == "user"
    assert next(event for event in events if event.type == "escape").initiator == "opponent"


def test_motion_rule_detects_a_defended_user_shot():
    timestamps = [index * 125 for index in range(24)]
    closing = [0.0] * 24
    hip_height = [0.6] * 24
    closing[8:11] = [0.12, 0.28, 0.18]
    hip_height[9] = 0.56
    hip_height[10] = 0.54
    features = pd.DataFrame({
        "timestamp_ms": timestamps,
        "closing_speed": closing,
        "bbox_closing_speed": closing,
        "user_hip_height": hip_height,
        "opponent_hip_height": [0.6] * 24,
        "user_level_change_rate": [0.0] * 9 + [0.32, 0.16] + [0.0] * 13,
    })
    states = [{
        "state": "neutral", "start_ms": 0, "end_ms": 5000,
        "controlling": None, "confidence": 0.8,
    }]

    events = detect_events(features, states)
    shot = next(event for event in events if event.type == "shot_attempt")
    defended = next(event for event in events if event.type == "defended_shot")
    assert shot.initiator == "user"
    assert defended.outcome == "failed"
    assert shot.measurements["closing_speed"] == 0.28
