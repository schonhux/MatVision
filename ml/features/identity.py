"""
ml/features/identity.py — binding tracker IDs to wrestler identities.

The tracker gives us anonymous, unstable track IDs. This module decides which track
is "the user," which is "the opponent," and which is the referee — the step that
makes everything downstream meaningful, since "hip height of track 7" is useless
without knowing whose hips those are.

Two inputs make this tractable rather than a research problem:
  1. The user's click-to-identify seed box from Layer 2 (MatchAthlete.seed_bbox),
     which pins one identity at one known frame.
  2. Simple, checkable heuristics for the referee (PROJECT_GUIDE.md Layer 3), who
     is genuinely easier to separate than the two wrestlers are from each other:
     smaller in frame, further from mat center, and moving less erratically.

Pure logic — no torch, no video — so the binding rules are actually unit-tested
rather than assumed correct.
"""

from __future__ import annotations

from collections import defaultdict


def iou(box_a: tuple[float, ...], box_b: tuple[float, ...]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def denormalize_bbox(
    norm_bbox: dict, frame_width: int, frame_height: int
) -> tuple[float, float, float, float]:
    """Layer 2 stores seed boxes normalized 0-1 so they survive resolution changes
    (the analysis copy is 720p regardless of what was uploaded). Convert back to
    pixels for IoU against detections.
    """
    return (
        norm_bbox["x1"] * frame_width,
        norm_bbox["y1"] * frame_height,
        norm_bbox["x2"] * frame_width,
        norm_bbox["y2"] * frame_height,
    )


def bind_identity_from_seed(
    detections: list[dict],
    seed_bbox: tuple[float, float, float, float],
    min_iou: float = 0.1,
) -> int | None:
    """Find which tracked detection the user actually clicked on.

    `detections`: [{'track_id': int, 'bbox': (x1,y1,x2,y2)}, ...] for the seed frame.
    Returns the best-overlapping track_id, or None if nothing meaningfully overlaps
    (a bad click, or the frame drifted) — better to report failure than to bind the
    wrong wrestler and silently mislabel the entire match.
    """
    best_id, best_score = None, min_iou
    for det in detections:
        score = iou(det["bbox"], seed_bbox)
        if score > best_score:
            best_id, best_score = det["track_id"], score
    return best_id


def classify_referee(
    bbox_area_ratio: float,
    dist_from_center_ratio: float,
    motion_variance: float,
    thresholds: dict | None = None,
) -> bool:
    """Heuristic referee test for one track's aggregate stats over the match.

    Referees are separable from wrestlers because they behave differently over time,
    not because any single frame looks distinctive:
      - they stay smaller in frame (they keep their distance)
      - they orbit the periphery rather than occupying mat center
      - their motion is steadier — they walk, they don't scramble
    """
    t = {
        "max_area_ratio": 0.12,
        "min_dist_ratio": 0.45,
        "max_motion_variance": 0.35,
    }
    if thresholds:
        t.update(thresholds)

    return (
        bbox_area_ratio < t["max_area_ratio"]
        and dist_from_center_ratio > t["min_dist_ratio"]
        and motion_variance < t["max_motion_variance"]
    )


def select_wrestler_tracks(
    track_stats: dict[int, dict],
    exclude: set[int] | None = None,
) -> list[int]:
    """Pick the two most likely wrestler tracks by total frames present.

    Used when there's no seed box to work from. Wrestlers are on camera essentially
    the whole match; spurious detections (spectators, coaches, partial bodies at the
    frame edge) appear briefly. Longevity is the most robust available signal.
    """
    exclude = exclude or set()
    candidates = [
        (tid, stats) for tid, stats in track_stats.items() if tid not in exclude
    ]
    candidates.sort(key=lambda kv: kv[1].get("frame_count", 0), reverse=True)
    return [tid for tid, _ in candidates[:2]]


def resolve_identities(
    track_stats: dict[int, dict],
    user_seed_track_id: int | None = None,
    opponent_seed_track_id: int | None = None,
) -> dict[int, str]:
    """Assign every track a role: 'user' | 'opponent' | 'referee' | 'unknown'.

    Seed-bound IDs always win — the user explicitly told us who they are, and no
    heuristic should override that. Referee detection runs next, then the two
    longest-lived remaining tracks fill any unassigned wrestler slots.
    """
    roles: dict[int, str] = {}

    if user_seed_track_id is not None:
        roles[user_seed_track_id] = "user"
    if opponent_seed_track_id is not None and opponent_seed_track_id != user_seed_track_id:
        roles[opponent_seed_track_id] = "opponent"

    for tid, stats in track_stats.items():
        if tid in roles:
            continue
        if classify_referee(
            stats.get("bbox_area_ratio", 1.0),
            stats.get("dist_from_center_ratio", 0.0),
            stats.get("motion_variance", 1.0),
        ):
            roles[tid] = "referee"

    assigned_wrestlers = {tid for tid, role in roles.items() if role in ("user", "opponent")}
    if len(assigned_wrestlers) < 2:
        excluded = set(roles.keys())
        for tid in select_wrestler_tracks(track_stats, exclude=excluded):
            if "user" not in roles.values():
                roles[tid] = "user"
            elif "opponent" not in roles.values():
                roles[tid] = "opponent"

    for tid in track_stats:
        roles.setdefault(tid, "unknown")

    return roles


def compute_track_stats(
    rows: list[dict],
    frame_width: int,
    frame_height: int,
) -> dict[int, dict]:
    """Aggregate per-frame detections into the per-track statistics the identity
    heuristics need.

    `rows`: [{'frame': int, 'track_id': int, 'x1','y1','x2','y2': float}, ...]
    """
    frame_area = frame_width * frame_height
    cx_frame, cy_frame = frame_width / 2, frame_height / 2
    max_dist = ((cx_frame ** 2) + (cy_frame ** 2)) ** 0.5

    by_track: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        by_track[r["track_id"]].append(r)

    stats: dict[int, dict] = {}
    for tid, track_rows in by_track.items():
        areas, dists, centers = [], [], []
        for r in track_rows:
            w = max(0.0, r["x2"] - r["x1"])
            h = max(0.0, r["y2"] - r["y1"])
            areas.append((w * h) / frame_area if frame_area else 0.0)

            cx, cy = (r["x1"] + r["x2"]) / 2, (r["y1"] + r["y2"]) / 2
            centers.append((cx, cy))
            dists.append((((cx - cx_frame) ** 2 + (cy - cy_frame) ** 2) ** 0.5) / max_dist
                         if max_dist else 0.0)

        # Motion variance = mean frame-to-frame displacement, normalized. Scrambling
        # wrestlers produce large erratic values; a walking referee doesn't.
        motion = []
        for i in range(1, len(centers)):
            dx = centers[i][0] - centers[i - 1][0]
            dy = centers[i][1] - centers[i - 1][1]
            motion.append(((dx ** 2 + dy ** 2) ** 0.5) / frame_width if frame_width else 0.0)

        stats[tid] = {
            "frame_count": len(track_rows),
            "bbox_area_ratio": sum(areas) / len(areas) if areas else 0.0,
            "dist_from_center_ratio": sum(dists) / len(dists) if dists else 0.0,
            "motion_variance": (sum(motion) / len(motion) * 100) if motion else 0.0,
            "first_frame": min(r["frame"] for r in track_rows),
            "last_frame": max(r["frame"] for r in track_rows),
        }
    return stats
