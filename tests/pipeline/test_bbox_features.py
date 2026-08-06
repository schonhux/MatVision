import pytest

from ml.features.bbox_features import compute_bbox_features


def test_bbox_features_are_normalized():
    result = compute_bbox_features(
        (100, 100, 300, 500),
        (500, 120, 700, 520),
        frame_width=1000,
        frame_height=800,
    )
    assert result["user_bbox_center_x"] == pytest.approx(0.2)
    assert result["user_bbox_height"] == pytest.approx(0.5)
    assert result["bbox_distance"] > 0
    assert result["bbox_overlap"] == 0


def test_bbox_features_keep_missing_tracks_explicit():
    result = compute_bbox_features((0, 0, 100, 100), None, 1000, 800)
    assert result["user_bbox_detected"] is True
    assert result["opponent_bbox_detected"] is False
    assert result["bbox_distance"] is None


def test_bbox_motion_uses_previous_sample():
    first = compute_bbox_features((100, 100, 200, 300), (500, 100, 600, 300), 1000, 800)
    second = compute_bbox_features(
        (120, 100, 220, 300),
        (480, 100, 580, 300),
        1000,
        800,
        prev=first,
        dt_seconds=0.1,
    )
    assert second["user_bbox_speed"] == pytest.approx(0.2)
    assert second["bbox_closing_speed"] > 0

