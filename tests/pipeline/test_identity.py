"""
Identity binding tests — deciding which track is the user, the opponent, and the
referee. Getting this wrong silently mislabels an entire match (every feature would
be attributed to the wrong wrestler), so the rules are tested explicitly rather than
trusted.
"""

import pytest

from ml.features.identity import (
    iou, denormalize_bbox, bind_identity_from_seed, classify_referee,
    select_wrestler_tracks, resolve_identities, compute_track_stats,
)


# --- geometry ------------------------------------------------------------------

def test_iou_identical_boxes():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)


def test_iou_no_overlap():
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_iou_partial_overlap():
    # 10x10 boxes offset by 5 -> intersection 25, union 175
    assert iou((0, 0, 10, 10), (5, 5, 15, 15)) == pytest.approx(25 / 175)


def test_denormalize_bbox():
    box = denormalize_bbox({"x1": 0.25, "y1": 0.5, "x2": 0.75, "y2": 1.0}, 1280, 720)
    assert box == (320.0, 360.0, 960.0, 720.0)


# --- seed binding ------------------------------------------------------------

def test_bind_identity_picks_best_overlap():
    detections = [
        {"track_id": 1, "bbox": (0, 0, 100, 200)},
        {"track_id": 2, "bbox": (300, 0, 400, 200)},
    ]
    assert bind_identity_from_seed(detections, (10, 10, 105, 195)) == 1


def test_bind_identity_returns_none_when_nothing_overlaps():
    """A misclick or drifted frame must fail loudly rather than bind the wrong
    wrestler and mislabel the whole match.
    """
    detections = [{"track_id": 1, "bbox": (0, 0, 100, 200)}]
    assert bind_identity_from_seed(detections, (900, 900, 1000, 1000)) is None


def test_bind_identity_with_no_detections():
    assert bind_identity_from_seed([], (0, 0, 10, 10)) is None


# --- referee heuristics ---------------------------------------------------------

def test_classify_referee_typical_referee():
    """Small in frame, near the periphery, steady motion."""
    assert classify_referee(
        bbox_area_ratio=0.05, dist_from_center_ratio=0.7, motion_variance=0.1
    ) is True


def test_classify_referee_rejects_wrestler_profile():
    """Large in frame, central, erratic motion — a wrestler, not an official."""
    assert classify_referee(
        bbox_area_ratio=0.30, dist_from_center_ratio=0.15, motion_variance=1.2
    ) is False


def test_classify_referee_requires_all_conditions():
    # Small and peripheral, but moving erratically -> not a referee.
    assert classify_referee(0.05, 0.7, motion_variance=2.0) is False
    # Small and steady, but central -> not a referee.
    assert classify_referee(0.05, 0.1, motion_variance=0.1) is False


# --- wrestler selection ----------------------------------------------------------

def test_select_wrestler_tracks_picks_longest_lived():
    """Wrestlers are on camera the whole match; spurious detections are brief."""
    stats = {
        1: {"frame_count": 500},
        2: {"frame_count": 480},
        3: {"frame_count": 12},   # a spectator walking past
    }
    assert set(select_wrestler_tracks(stats)) == {1, 2}


def test_select_wrestler_tracks_honors_exclusions():
    stats = {1: {"frame_count": 500}, 2: {"frame_count": 480}, 3: {"frame_count": 300}}
    assert set(select_wrestler_tracks(stats, exclude={1})) == {2, 3}


# --- full identity resolution -----------------------------------------------------

def test_resolve_identities_seed_wins_over_heuristics():
    """An explicit click must never be overridden — even if the track looks
    referee-ish by the heuristics.
    """
    stats = {
        7: {"frame_count": 400, "bbox_area_ratio": 0.05,
            "dist_from_center_ratio": 0.8, "motion_variance": 0.1},
        8: {"frame_count": 400, "bbox_area_ratio": 0.25,
            "dist_from_center_ratio": 0.2, "motion_variance": 1.0},
    }
    roles = resolve_identities(stats, user_seed_track_id=7)
    assert roles[7] == "user"


def test_resolve_identities_labels_referee_and_wrestlers():
    stats = {
        1: {"frame_count": 500, "bbox_area_ratio": 0.28,
            "dist_from_center_ratio": 0.2, "motion_variance": 1.5},
        2: {"frame_count": 495, "bbox_area_ratio": 0.26,
            "dist_from_center_ratio": 0.25, "motion_variance": 1.4},
        3: {"frame_count": 480, "bbox_area_ratio": 0.04,
            "dist_from_center_ratio": 0.75, "motion_variance": 0.1},
    }
    roles = resolve_identities(stats)
    assert roles[3] == "referee"
    assert {roles[1], roles[2]} == {"user", "opponent"}


def test_resolve_identities_assigns_both_roles_without_seeds():
    stats = {
        1: {"frame_count": 500, "bbox_area_ratio": 0.3,
            "dist_from_center_ratio": 0.2, "motion_variance": 1.0},
        2: {"frame_count": 490, "bbox_area_ratio": 0.3,
            "dist_from_center_ratio": 0.2, "motion_variance": 1.0},
    }
    roles = resolve_identities(stats)
    assert sorted(roles.values()) == ["opponent", "user"]


def test_resolve_identities_every_track_gets_a_role():
    stats = {i: {"frame_count": 10 * i, "bbox_area_ratio": 0.2,
                 "dist_from_center_ratio": 0.3, "motion_variance": 0.8}
             for i in range(1, 6)}
    roles = resolve_identities(stats)
    assert set(roles.keys()) == set(stats.keys())
    assert all(r in ("user", "opponent", "referee", "unknown") for r in roles.values())


# --- track statistics -------------------------------------------------------------

def test_compute_track_stats_basic():
    rows = [
        {"frame": 0, "track_id": 1, "x1": 100, "y1": 100, "x2": 200, "y2": 400},
        {"frame": 1, "track_id": 1, "x1": 110, "y1": 100, "x2": 210, "y2": 400},
        {"frame": 0, "track_id": 2, "x1": 600, "y1": 100, "x2": 700, "y2": 400},
    ]
    stats = compute_track_stats(rows, 1280, 720)

    assert stats[1]["frame_count"] == 2
    assert stats[2]["frame_count"] == 1
    assert stats[1]["first_frame"] == 0 and stats[1]["last_frame"] == 1
    assert 0 < stats[1]["bbox_area_ratio"] < 1


def test_compute_track_stats_distinguishes_central_from_peripheral():
    """A track hugging the frame edge should score a higher distance-from-center
    ratio than one in the middle — this is what separates the referee.
    """
    central = [{"frame": i, "track_id": 1, "x1": 600, "y1": 320, "x2": 680, "y2": 400}
               for i in range(5)]
    peripheral = [{"frame": i, "track_id": 2, "x1": 10, "y1": 10, "x2": 90, "y2": 90}
                  for i in range(5)]
    stats = compute_track_stats(central + peripheral, 1280, 720)
    assert stats[2]["dist_from_center_ratio"] > stats[1]["dist_from_center_ratio"]


def test_compute_track_stats_motion_variance_reflects_movement():
    still = [{"frame": i, "track_id": 1, "x1": 100, "y1": 100, "x2": 200, "y2": 400}
             for i in range(10)]
    erratic = [{"frame": i, "track_id": 2,
                "x1": 100 + (i % 2) * 300, "y1": 100,
                "x2": 200 + (i % 2) * 300, "y2": 400}
               for i in range(10)]
    stats = compute_track_stats(still + erratic, 1280, 720)
    assert stats[2]["motion_variance"] > stats[1]["motion_variance"]


def test_compute_track_stats_empty_input():
    assert compute_track_stats([], 1280, 720) == {}
