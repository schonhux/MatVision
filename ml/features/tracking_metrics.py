from __future__ import annotations

import pandas as pd
import numpy as np


WRESTLER_IDENTITIES = ("wrestler_a", "wrestler_b")


def dominant_track_id(df: pd.DataFrame, identity: str) -> int | None:
    """The most frequent raw track id for a resolved identity."""
    subset = df[df["identity"] == identity]
    if subset.empty:
        return None
    return int(subset["track_id"].value_counts().idxmax())


def id_hold_fraction(
    df: pd.DataFrame,
    identity: str,
    active_frames: pd.Series,
) -> float:
    """Fraction of active frames where an identity stays on its main track id."""
    active_frame_numbers = set(active_frames[active_frames].index)
    if not active_frame_numbers:
        return 0.0

    canonical = dominant_track_id(df, identity)
    if canonical is None:
        return 0.0

    held = df[
        (df["identity"] == identity)
        & (df["track_id"] == canonical)
        & (df["frame"].isin(active_frame_numbers))
    ]["frame"].nunique()

    return held / len(active_frame_numbers)


def overall_id_hold(
    df: pd.DataFrame,
    active_frames: pd.Series,
    identities: tuple[str, ...] = WRESTLER_IDENTITIES,
) -> dict:
    """Headline metric averaged across both wrestlers, plus per-identity breakdown.

    Returns:
        {
            "overall": float,             # average across identities — compare to 0.80 gate
            "per_identity": {identity: float, ...},
            "passes_gate": bool,
        }
    """
    per_identity = {ident: id_hold_fraction(df, ident, active_frames) for ident in identities}
    overall = float(np.mean(list(per_identity.values()))) if per_identity else 0.0
    return {
        "overall": overall,
        "per_identity": per_identity,
        "passes_gate": overall >= 0.80,
    }


def count_identity_switches(df: pd.DataFrame, identity: str) -> int:
    """Count how many times `identity`'s assigned track_id changes across consecutive
    frames where that identity has a detection. Each change is one 'switch' — the
    tracker briefly lost the wrestler and (correctly or incorrectly) picked up a
    different underlying track.
    """
    subset = df[df["identity"] == identity].sort_values("frame")
    if subset.empty:
        return 0
    track_ids = subset["track_id"].to_numpy()
    return int(np.sum(track_ids[1:] != track_ids[:-1]))


def lost_track_duration_frames(
    df: pd.DataFrame,
    identity: str,
    active_frames: pd.Series,
) -> int:
    """Count of active frames where `identity` has NO detection at all (fully lost,
    as opposed to an identity switch where a detection exists but under the wrong id).
    """
    active_frame_numbers = set(active_frames[active_frames].index)
    present_frames = set(df[df["identity"] == identity]["frame"].unique())
    return len(active_frame_numbers - present_frames)


def summarize_track_quality(
    df: pd.DataFrame,
    active_frames: pd.Series,
    fps: float,
    identities: tuple[str, ...] = WRESTLER_IDENTITIES,
) -> dict:
    """Build a per-match tracking quality report."""
    hold = overall_id_hold(df, active_frames, identities)
    report = {
        "overall_id_hold": hold["overall"],
        "passes_gate": hold["passes_gate"],
        "per_identity": {},
    }
    for ident in identities:
        lost_frames = lost_track_duration_frames(df, ident, active_frames)
        report["per_identity"][ident] = {
            "id_hold_fraction": hold["per_identity"].get(ident, 0.0),
            "identity_switches": count_identity_switches(df, ident),
            "lost_track_frames": lost_frames,
            "lost_track_seconds": round(lost_frames / fps, 2) if fps else None,
        }
    return report


# ---------------------------------------------------------------------------
# Referee filtering — pure decision logic over precomputed per-track summary stats.
# The stats themselves (color histogram distance, position, motion) are computed
# from real video frames in l0_tracer_bullet.py; this function is the testable
# classification rule on top of them.
# ---------------------------------------------------------------------------

def classify_actor_type(
    avg_bbox_area_ratio: float,
    avg_dist_from_mat_center_ratio: float,
    avg_motion_speed: float,
    is_striped_uniform: bool,
    thresholds: dict | None = None,
) -> str:
    """Heuristic referee/wrestler classification for a single track's summary stats.

    Args:
        avg_bbox_area_ratio: track's average bbox area / frame area (referees tend
            to be smaller in frame — they stay back from the action).
        avg_dist_from_mat_center_ratio: average distance from mat center, normalized
            by mat radius (referees circle the action rather than being in the center).
        avg_motion_speed: average per-frame displacement (referees move less abruptly
            than scrambling wrestlers).
        is_striped_uniform: True if the track's dominant color pattern matches a
            referee uniform (black-and-white stripes), from a simple color-histogram
            check upstream.
        thresholds: override defaults for testing.

    Returns:
        One of 'referee', 'wrestler', 'unknown'.
    """
    t = {
        "max_wrestler_bbox_ratio": 0.35,   # referees rarely fill >35% of frame area
        "min_referee_dist_ratio": 0.55,    # referees tend to stay toward the mat edge
        "max_referee_motion": 0.4,          # normalized motion units; referees move less
    }
    if thresholds:
        t.update(thresholds)

    if is_striped_uniform:
        return "referee"

    if (
        avg_bbox_area_ratio < t["max_wrestler_bbox_ratio"]
        and avg_dist_from_mat_center_ratio > t["min_referee_dist_ratio"]
        and avg_motion_speed < t["max_referee_motion"]
    ):
        return "referee"

    if avg_bbox_area_ratio >= t["max_wrestler_bbox_ratio"]:
        return "wrestler"

    return "unknown"


# ---------------------------------------------------------------------------
# Lightweight re-identification: stitch a newly-appearing track back onto a
# recently-lost identity if it reappears nearby in space and time. This lets us
# survive brief occlusions without a full deep re-ID model — appropriate for the
# tracer-bullet's scope.
# ---------------------------------------------------------------------------

def find_reid_candidate(
    lost_identity_last_bbox: tuple[float, float, float, float],
    lost_at_frame: int,
    candidate_tracks: pd.DataFrame,
    max_gap_frames: int = 45,
    max_center_dist_px: float = 150.0,
) -> int | None:
    """Given where an identity was last seen, find a plausible new track_id to
    re-attach it to, among tracks that first appear shortly after and nearby.

    Returns the candidate track_id, or None if nothing plausible is found.
    """
    lx1, ly1, lx2, ly2 = lost_identity_last_bbox
    last_cx, last_cy = (lx1 + lx2) / 2, (ly1 + ly2) / 2

    first_seen = candidate_tracks.groupby("track_id")["frame"].min()
    nearby_in_time = first_seen[
        (first_seen > lost_at_frame) & (first_seen <= lost_at_frame + max_gap_frames)
    ]
    if nearby_in_time.empty:
        return None

    best_id, best_dist = None, float("inf")
    for track_id, first_frame in nearby_in_time.items():
        row = candidate_tracks[
            (candidate_tracks["track_id"] == track_id)
            & (candidate_tracks["frame"] == first_frame)
        ].iloc[0]
        cx, cy = (row["x1"] + row["x2"]) / 2, (row["y1"] + row["y2"]) / 2
        dist = ((cx - last_cx) ** 2 + (cy - last_cy) ** 2) ** 0.5
        if dist < max_center_dist_px and dist < best_dist:
            best_id, best_dist = int(track_id), dist

    return best_id
