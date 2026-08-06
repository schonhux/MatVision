import numpy as np
import pandas as pd
import pytest

from ml.states.baseline import BBoxStateFallback
from ml.states.inference import predictions_to_segments, smooth_short_runs


def test_bbox_fallback_handles_neutral_and_missing_pose():
    features = pd.DataFrame([
        {
            "user_bbox_detected": True,
            "opponent_bbox_detected": True,
            "bbox_distance": 0.3,
            "bbox_overlap": 0.0,
            "user_visibility": 0.0,
            "opponent_visibility": 0.0,
        }
    ])
    probabilities = BBoxStateFallback().predict_proba(features)
    assert probabilities.shape == (1, 5)
    assert probabilities[0].argmax() == 0
    assert probabilities[0].sum() == pytest.approx(1.0)


def test_bbox_fallback_marks_single_track_as_scramble():
    features = pd.DataFrame([{"user_bbox_detected": True, "opponent_bbox_detected": False}])
    assert BBoxStateFallback().predict_proba(features)[0].argmax() == 3


def test_short_prediction_run_is_smoothed():
    labels = ["neutral", "neutral", "scramble", "neutral", "neutral"]
    smoothed = smooth_short_runs(labels, np.array([0.8, 0.8, 0.4, 0.8, 0.8]), min_frames=2)
    assert smoothed == ["neutral"] * 5


def test_predictions_merge_into_segments_and_set_control():
    segments = predictions_to_segments(
        [0, 125, 250, 375],
        ["neutral", "neutral", "top", "top"],
        np.array([0.8, 0.7, 0.6, 0.5]),
        duration_ms=500,
    )
    assert [(s.state, s.start_ms, s.end_ms) for s in segments] == [
        ("neutral", 0, 250),
        ("top", 250, 500),
    ]
    assert segments[1].controlling == "user"
