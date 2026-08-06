import pandas as pd
import pytest

from ml.features.tracking_metrics import (
    dominant_track_id,
    id_hold_fraction,
    overall_id_hold,
    count_identity_switches,
    lost_track_duration_frames,
    summarize_track_quality,
    classify_actor_type,
    find_reid_candidate,
)


def make_track_df(rows):
    """rows: list of (frame, track_id, identity, x1, y1, x2, y2, confidence)"""
    return pd.DataFrame(
        rows,
        columns=["frame", "track_id", "identity", "x1", "y1", "x2", "y2", "confidence"],
    )


def active_mask(n_frames, active_range=None):
    active_range = active_range or range(n_frames)
    s = pd.Series(False, index=range(n_frames))
    s.loc[list(active_range)] = True
    return s


# --- id_hold_fraction / dominant_track_id -----------------------------------

def test_perfect_hold_is_1_0():
    df = make_track_df(
        [(f, 1, "wrestler_a", 0, 0, 10, 10, 0.9) for f in range(30)]
    )
    mask = active_mask(30)
    assert id_hold_fraction(df, "wrestler_a", mask) == pytest.approx(1.0)


def test_single_switch_drops_hold_below_gate():
    # 100 active frames; wrestler_a is track 1 for first 50, track 2 for last 50.
    rows = [(f, 1, "wrestler_a", 0, 0, 10, 10, 0.9) for f in range(50)]
    rows += [(f, 2, "wrestler_a", 0, 0, 10, 10, 0.9) for f in range(50, 100)]
    df = make_track_df(rows)
    mask = active_mask(100)
    frac = id_hold_fraction(df, "wrestler_a", mask)
    assert frac == pytest.approx(0.5)
    assert frac < 0.80


def test_missing_identity_returns_zero():
    df = make_track_df([(0, 1, "wrestler_b", 0, 0, 10, 10, 0.9)])
    mask = active_mask(10)
    assert id_hold_fraction(df, "wrestler_a", mask) == 0.0
    assert dominant_track_id(df, "wrestler_a") is None


def test_overall_id_hold_averages_and_gates():
    rows = [(f, 1, "wrestler_a", 0, 0, 10, 10, 0.9) for f in range(100)]
    rows += [(f, 2, "wrestler_b", 0, 0, 10, 10, 0.9) for f in range(100)]
    df = make_track_df(rows)
    mask = active_mask(100)
    result = overall_id_hold(df, mask)
    assert result["overall"] == pytest.approx(1.0)
    assert result["passes_gate"] is True

    # Now break wrestler_b halfway through -> overall should drop and fail gate.
    rows_b_broken = rows[:100] + [
        (f, 2, "wrestler_b", 0, 0, 10, 10, 0.9) for f in range(50)
    ] + [
        (f, 3, "wrestler_b", 0, 0, 10, 10, 0.9) for f in range(50, 100)
    ]
    df2 = make_track_df(rows_b_broken)
    result2 = overall_id_hold(df2, mask)
    assert result2["per_identity"]["wrestler_a"] == pytest.approx(1.0)
    assert result2["per_identity"]["wrestler_b"] == pytest.approx(0.5)
    assert result2["overall"] == pytest.approx(0.75)
    assert result2["passes_gate"] is False  # 0.75 < 0.80 gate


# --- identity switches / lost track ------------------------------------------

def test_count_identity_switches():
    rows = [(f, 1, "wrestler_a", 0, 0, 1, 1, 0.9) for f in range(10)]
    rows += [(f, 2, "wrestler_a", 0, 0, 1, 1, 0.9) for f in range(10, 15)]
    rows += [(f, 1, "wrestler_a", 0, 0, 1, 1, 0.9) for f in range(15, 20)]
    df = make_track_df(rows)
    # switches: 1->2 at frame 10, 2->1 at frame 15 = 2 switches
    assert count_identity_switches(df, "wrestler_a") == 2


def test_count_identity_switches_no_detections():
    df = make_track_df([])
    assert count_identity_switches(df, "wrestler_a") == 0


def test_lost_track_duration_counts_absent_active_frames():
    # Present frames 0-49, absent (lost) 50-79, active range is 0-99.
    rows = [(f, 1, "wrestler_a", 0, 0, 1, 1, 0.9) for f in range(50)]
    rows += [(f, 1, "wrestler_a", 0, 0, 1, 1, 0.9) for f in range(80, 100)]
    df = make_track_df(rows)
    mask = active_mask(100)
    lost = lost_track_duration_frames(df, "wrestler_a", mask)
    assert lost == 30  # frames 50-79


def test_lost_track_ignores_inactive_frames():
    # Wrestler present 0-19; frames 20-29 don't exist in the df but are marked inactive.
    rows = [(f, 1, "wrestler_a", 0, 0, 1, 1, 0.9) for f in range(20)]
    df = make_track_df(rows)
    mask = active_mask(30, active_range=range(0, 20))  # only 0-19 are "active"
    assert lost_track_duration_frames(df, "wrestler_a", mask) == 0


# --- summarize_track_quality (the full report) --------------------------------

def test_summarize_track_quality_shape_and_values():
    rows = [(f, 1, "wrestler_a", 0, 0, 10, 10, 0.9) for f in range(60)]
    rows += [(f, 2, "wrestler_b", 0, 0, 10, 10, 0.9) for f in range(60)]
    df = make_track_df(rows)
    mask = active_mask(60)
    report = summarize_track_quality(df, mask, fps=30.0)

    assert report["passes_gate"] is True
    assert report["overall_id_hold"] == pytest.approx(1.0)
    assert set(report["per_identity"].keys()) == {"wrestler_a", "wrestler_b"}
    for ident_stats in report["per_identity"].values():
        assert ident_stats["identity_switches"] == 0
        assert ident_stats["lost_track_frames"] == 0
        assert ident_stats["lost_track_seconds"] == 0.0


# --- referee / actor-type classification --------------------------------------

def test_classify_striped_uniform_is_always_referee():
    assert classify_actor_type(0.5, 0.1, 5.0, is_striped_uniform=True) == "referee"


def test_classify_small_peripheral_slow_track_is_referee():
    result = classify_actor_type(
        avg_bbox_area_ratio=0.1,
        avg_dist_from_mat_center_ratio=0.7,
        avg_motion_speed=0.2,
        is_striped_uniform=False,
    )
    assert result == "referee"


def test_classify_large_central_fast_track_is_wrestler():
    result = classify_actor_type(
        avg_bbox_area_ratio=0.45,
        avg_dist_from_mat_center_ratio=0.2,
        avg_motion_speed=2.5,
        is_striped_uniform=False,
    )
    assert result == "wrestler"


def test_classify_ambiguous_case_is_unknown():
    result = classify_actor_type(
        avg_bbox_area_ratio=0.2,
        avg_dist_from_mat_center_ratio=0.3,
        avg_motion_speed=0.3,
        is_striped_uniform=False,
    )
    assert result == "unknown"


# --- re-identification stitching ----------------------------------------------

def test_find_reid_candidate_picks_nearest_in_time_and_space():
    candidates = pd.DataFrame(
        [
            # a plausible reappearance: close in space, soon after loss
            {"track_id": 5, "frame": 105, "x1": 100, "y1": 100, "x2": 120, "y2": 120},
            # a decoy: appears in time window but far away spatially
            {"track_id": 6, "frame": 106, "x1": 900, "y1": 900, "x2": 920, "y2": 920},
        ]
    )
    result = find_reid_candidate(
        lost_identity_last_bbox=(95, 95, 115, 115),
        lost_at_frame=100,
        candidate_tracks=candidates,
        max_gap_frames=45,
        max_center_dist_px=150.0,
    )
    assert result == 5


def test_find_reid_candidate_returns_none_when_too_far_or_late():
    candidates = pd.DataFrame(
        [{"track_id": 7, "frame": 300, "x1": 900, "y1": 900, "x2": 920, "y2": 920}]
    )
    result = find_reid_candidate(
        lost_identity_last_bbox=(0, 0, 20, 20),
        lost_at_frame=100,
        candidate_tracks=candidates,
        max_gap_frames=45,
        max_center_dist_px=150.0,
    )
    assert result is None
