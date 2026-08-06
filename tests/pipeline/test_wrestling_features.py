"""
Wrestling feature math tests.

Two things are being verified here, and the second matters more:
  1. The geometry is correct (angles, distances, velocities).
  2. **Occlusion is handled honestly** — features return None, never a fabricated
     number, when the keypoints they need aren't visible. Pose collapses during
     scrambles, so a silently-wrong 0.0 would teach downstream models that occlusion
     means "hips on the floor," which is exactly backwards.
"""

import math

import numpy as np
import pytest

from ml.features.wrestling_features import (
    hip_height, torso_angle, stance_width, knee_bend, center_of_mass,
    keypoint_visibility, athlete_distance, relative_hip_height,
    head_position_relative, velocity, closing_speed, level_change_rate,
    compute_frame_features, interpolate_short_gaps,
    NOSE, L_SHOULDER, R_SHOULDER, L_HIP, R_HIP, L_KNEE, R_KNEE,
    L_ANKLE, R_ANKLE,
)

FRAME_W, FRAME_H = 1280, 720


def make_kpts(points: dict[int, tuple[float, float]], conf: float = 0.9) -> np.ndarray:
    """Builds a (17,3) keypoint array; unspecified joints get confidence 0 (invisible)."""
    kpts = np.zeros((17, 3), dtype=float)
    for idx, (x, y) in points.items():
        kpts[idx] = [x, y, conf]
    return kpts


def standing_wrestler(cx: float = 640, hip_y: float = 400) -> np.ndarray:
    """An upright wrestler with all relevant joints visible."""
    return make_kpts({
        NOSE: (cx, hip_y - 200),
        L_SHOULDER: (cx - 40, hip_y - 150), R_SHOULDER: (cx + 40, hip_y - 150),
        L_HIP: (cx - 30, hip_y), R_HIP: (cx + 30, hip_y),
        L_KNEE: (cx - 30, hip_y + 100), R_KNEE: (cx + 30, hip_y + 100),
        L_ANKLE: (cx - 50, hip_y + 200), R_ANKLE: (cx + 50, hip_y + 200),
    })


# --- individual features -------------------------------------------------------

def test_hip_height_is_normalized_from_floor():
    """Higher hips (smaller y) must yield a LARGER value — the number should read
    intuitively as 'how upright is this athlete'.
    """
    high = hip_height(standing_wrestler(hip_y=200), FRAME_H)
    low = hip_height(standing_wrestler(hip_y=600), FRAME_H)
    assert high > low
    assert high == pytest.approx(1 - 200 / 720)
    assert 0 <= low <= 1


def test_hip_height_none_when_hips_occluded():
    kpts = standing_wrestler()
    kpts[L_HIP, 2] = 0.0
    kpts[R_HIP, 2] = 0.0
    assert hip_height(kpts, FRAME_H) is None


def test_hip_height_uses_single_visible_hip():
    """One hip occluded by the opponent is the common case in a scramble — a
    one-hip estimate is much better than discarding the frame.
    """
    kpts = standing_wrestler()
    kpts[R_HIP, 2] = 0.0
    assert hip_height(kpts, FRAME_H) is not None


def test_torso_angle_upright_is_near_zero():
    assert torso_angle(standing_wrestler()) == pytest.approx(0.0, abs=1.0)


def test_torso_angle_bent_over_is_large():
    """Shoulders pushed forward of the hips = bent posture = large angle."""
    kpts = make_kpts({
        L_SHOULDER: (500, 300), R_SHOULDER: (560, 300),
        L_HIP: (630, 320), R_HIP: (690, 320),
    })
    angle = torso_angle(kpts)
    assert angle is not None and angle > 45


def test_torso_angle_none_without_shoulders():
    kpts = standing_wrestler()
    kpts[L_SHOULDER, 2] = kpts[R_SHOULDER, 2] = 0.0
    assert torso_angle(kpts) is None


def test_stance_width_normalized():
    kpts = standing_wrestler()  # ankles at cx-50 and cx+50 -> 100px
    assert stance_width(kpts, FRAME_W) == pytest.approx(100 / 1280)


def test_stance_width_requires_both_ankles():
    kpts = standing_wrestler()
    kpts[L_ANKLE, 2] = 0.0
    assert stance_width(kpts, FRAME_W) is None


def test_knee_bend_straight_leg_is_180():
    kpts = make_kpts({L_HIP: (600, 300), L_KNEE: (600, 400), L_ANKLE: (600, 500)})
    assert knee_bend(kpts, side="left") == pytest.approx(180.0, abs=1.0)


def test_knee_bend_deep_bend_is_small():
    """Ankle tucked back under the hip = deep bend = small interior angle."""
    kpts = make_kpts({L_HIP: (600, 300), L_KNEE: (600, 400), L_ANKLE: (600, 310)})
    angle = knee_bend(kpts, side="left")
    assert angle is not None and angle < 45


def test_knee_bend_none_when_incomplete():
    kpts = make_kpts({L_HIP: (600, 300), L_KNEE: (600, 400)})  # no ankle
    assert knee_bend(kpts, side="left") is None


def test_center_of_mass_between_shoulders_and_hips():
    com = center_of_mass(standing_wrestler(cx=640, hip_y=400))
    assert com is not None
    assert com[0] == pytest.approx(640, abs=1)
    assert 250 < com[1] < 400  # between shoulder line (250) and hip line (400)


def test_keypoint_visibility_fraction():
    kpts = np.zeros((17, 3))
    kpts[:8, 2] = 0.9  # 8 of 17 confident
    assert keypoint_visibility(kpts) == pytest.approx(8 / 17)
    assert keypoint_visibility(np.zeros((17, 3))) == 0.0


def test_low_confidence_keypoints_are_treated_as_missing():
    """A confidence below threshold must be ignored even though coordinates exist —
    the coordinates in that case are essentially noise.
    """
    kpts = standing_wrestler()
    kpts[L_HIP, 2] = 0.05
    kpts[R_HIP, 2] = 0.05
    assert hip_height(kpts, FRAME_H) is None


# --- relational features -------------------------------------------------------

def test_athlete_distance_scales_with_separation():
    a = standing_wrestler(cx=400)
    near = standing_wrestler(cx=500)
    far = standing_wrestler(cx=900)
    assert athlete_distance(a, far, FRAME_W) > athlete_distance(a, near, FRAME_W)


def test_athlete_distance_none_if_either_missing():
    a = standing_wrestler()
    blank = np.zeros((17, 3))
    assert athlete_distance(a, blank, FRAME_W) is None


def test_relative_hip_height_sign_indicates_who_is_higher():
    higher = standing_wrestler(hip_y=200)
    lower = standing_wrestler(hip_y=600)
    assert relative_hip_height(higher, lower, FRAME_H) > 0
    assert relative_hip_height(lower, higher, FRAME_H) < 0


def test_head_position_relative_sign():
    a = standing_wrestler(cx=400)   # head to the left of B
    b = standing_wrestler(cx=800)
    assert head_position_relative(a, b, FRAME_W) < 0


# --- temporal features ---------------------------------------------------------

def test_velocity_basic():
    v = velocity((700, 400), (640, 400), dt_seconds=0.1, frame_width=1280)
    assert v is not None
    assert v[0] == pytest.approx((60 / 1280) / 0.1)
    assert v[1] == pytest.approx(0.0)


def test_velocity_none_across_a_detection_gap():
    """If the athlete wasn't visible last frame, velocity is unknowable — returning
    a number here would invent motion.
    """
    assert velocity((700, 400), None, 0.1, 1280) is None
    assert velocity(None, (640, 400), 0.1, 1280) is None


def test_velocity_none_for_zero_dt():
    assert velocity((700, 400), (640, 400), 0.0, 1280) is None


def test_closing_speed_positive_when_closing():
    assert closing_speed(0.2, 0.4, 0.1) > 0   # distance shrank -> closing
    assert closing_speed(0.5, 0.3, 0.1) < 0   # distance grew -> separating


def test_level_change_rate_positive_when_dropping():
    """Hip height falling => positive rate. This is the primary shot signal."""
    assert level_change_rate(0.4, 0.6, 0.1) > 0
    assert level_change_rate(0.6, 0.4, 0.1) < 0


def test_temporal_features_none_when_inputs_missing():
    assert closing_speed(None, 0.4, 0.1) is None
    assert level_change_rate(0.4, None, 0.1) is None


# --- full frame assembly -------------------------------------------------------

def test_compute_frame_features_shape_is_rectangular():
    """Every key must exist in every frame regardless of what was visible, so the
    output can go straight into a dataframe/parquet without ragged columns.
    """
    full = compute_frame_features(standing_wrestler(400), standing_wrestler(800), FRAME_W, FRAME_H)
    empty = compute_frame_features(None, None, FRAME_W, FRAME_H)
    assert set(full.keys()) == set(empty.keys())


def test_compute_frame_features_flags_missing_athletes():
    f = compute_frame_features(standing_wrestler(), None, FRAME_W, FRAME_H)
    assert f["user_detected"] is True
    assert f["opponent_detected"] is False
    assert f["both_athletes_visible"] is False
    assert f["athlete_distance"] is None
    assert f["opponent_visibility"] == 0.0


def test_compute_frame_features_relational_when_both_present():
    f = compute_frame_features(standing_wrestler(400), standing_wrestler(800), FRAME_W, FRAME_H)
    assert f["both_athletes_visible"] is True
    assert f["athlete_distance"] is not None
    assert f["relative_hip_height"] is not None


def test_compute_frame_features_temporal_needs_prev():
    first = compute_frame_features(standing_wrestler(400), standing_wrestler(800), FRAME_W, FRAME_H)
    assert first["closing_speed"] is None  # no previous frame

    second = compute_frame_features(
        standing_wrestler(500), standing_wrestler(750), FRAME_W, FRAME_H, prev=first
    )
    assert second["closing_speed"] is not None
    assert second["closing_speed"] > 0  # they moved toward each other


def test_shot_signature_is_detectable_from_features():
    """End-to-end sanity: a level change + closing distance — the actual signature
    of a shot attempt — should show up as a positive level_change_rate and positive
    closing_speed. This is what Layer 5's rules engine will key on.
    """
    prev = compute_frame_features(
        standing_wrestler(cx=400, hip_y=350), standing_wrestler(cx=800, hip_y=350),
        FRAME_W, FRAME_H,
    )
    now = compute_frame_features(
        standing_wrestler(cx=500, hip_y=500),  # dropped level AND moved in
        standing_wrestler(cx=800, hip_y=350),
        FRAME_W, FRAME_H, prev=prev, dt_seconds=0.1,
    )
    assert now["user_level_change_rate"] > 0, "hip drop should register"
    assert now["closing_speed"] > 0, "closing distance should register"


# --- gap interpolation ---------------------------------------------------------

def test_interpolate_fills_short_gap():
    filled = interpolate_short_gaps([1.0, None, 3.0], max_gap=3)
    assert filled == [1.0, pytest.approx(2.0), 3.0]


def test_interpolate_fills_multi_frame_gap():
    filled = interpolate_short_gaps([0.0, None, None, None, 4.0], max_gap=3)
    assert filled[1] == pytest.approx(1.0)
    assert filled[2] == pytest.approx(2.0)
    assert filled[3] == pytest.approx(3.0)


def test_interpolate_leaves_long_gaps_alone():
    """A long dropout means genuine occlusion. Filling it would fabricate motion
    that never happened — the opposite of what this project is for.
    """
    values = [1.0] + [None] * 10 + [5.0]
    filled = interpolate_short_gaps(values, max_gap=3)
    assert filled[1:11] == [None] * 10


def test_interpolate_leaves_leading_and_trailing_gaps():
    """No anchor on one side means nothing to interpolate between."""
    assert interpolate_short_gaps([None, None, 3.0], max_gap=3)[:2] == [None, None]
    assert interpolate_short_gaps([1.0, None, None], max_gap=3)[1:] == [None, None]


def test_interpolate_all_none_is_unchanged():
    assert interpolate_short_gaps([None] * 5, max_gap=3) == [None] * 5


def test_interpolate_no_gaps_is_unchanged():
    assert interpolate_short_gaps([1.0, 2.0, 3.0], max_gap=3) == [1.0, 2.0, 3.0]
