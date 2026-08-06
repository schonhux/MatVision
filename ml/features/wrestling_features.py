"""
ml/features/wrestling_features.py — turn raw pose keypoints into wrestling-meaningful
numbers.

From PROJECT_GUIDE.md Layer 3: "Raw pose is not enough. Convert keypoints into
wrestling-relevant features." A model can't learn much from 17 raw (x, y) pairs, but
it can learn a lot from "hips dropped fast while closing distance from a neutral
stance."

**The single most important design constraint here** (PROJECT_GUIDE.md, repeated in
every layer): pose estimation degrades badly during ground contact and scrambles —
exactly the moments that matter most in wrestling. So every function in this module:

  1. Accepts a per-keypoint confidence array and treats low-confidence points as
     missing rather than trusting garbage coordinates.
  2. Returns None (not 0.0, not a guess) when the inputs it needs aren't visible.
  3. Emits an explicit `*_available` / mask flag so downstream models can learn to
     distrust those frames instead of silently training on fabricated values.

Returning 0.0 for "unknown hip height" would be a silent correctness bug: the model
would learn that occlusion means "hips on the floor," which is exactly backwards for
a scramble.

This module is deliberately torch-free and operates on numpy arrays, so it's fully
unit-testable without a GPU (see ADR-007 in dev/DECISIONS.md).

COCO-17 keypoint indices, which is what YOLOv8-pose emits:
    0 nose        1 l_eye      2 r_eye      3 l_ear      4 r_ear
    5 l_shoulder  6 r_shoulder 7 l_elbow    8 r_elbow
    9 l_wrist    10 r_wrist   11 l_hip     12 r_hip
   13 l_knee     14 r_knee    15 l_ankle   16 r_ankle
"""

from __future__ import annotations

import math

import numpy as np

NOSE = 0
L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW, R_ELBOW = 7, 8
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANKLE, R_ANKLE = 15, 16

DEFAULT_CONF_THRESHOLD = 0.3


def _visible(kpts: np.ndarray, idx: int, conf_threshold: float) -> bool:
    """A keypoint counts as usable only if the model was reasonably confident about it."""
    return bool(kpts[idx, 2] >= conf_threshold)


def _midpoint(
    kpts: np.ndarray, a: int, b: int, conf_threshold: float
) -> tuple[float, float] | None:
    """Midpoint of two keypoints. Falls back to whichever single point is visible —
    during a scramble one hip is frequently occluded by the opponent's body, and a
    one-hip estimate is far better than no estimate.
    """
    a_ok = _visible(kpts, a, conf_threshold)
    b_ok = _visible(kpts, b, conf_threshold)
    if a_ok and b_ok:
        return float((kpts[a, 0] + kpts[b, 0]) / 2), float((kpts[a, 1] + kpts[b, 1]) / 2)
    if a_ok:
        return float(kpts[a, 0]), float(kpts[a, 1])
    if b_ok:
        return float(kpts[b, 0]), float(kpts[b, 1])
    return None


# --- individual athlete features ------------------------------------------------

def hip_height(
    kpts: np.ndarray,
    frame_height: int,
    conf_threshold: float = DEFAULT_CONF_THRESHOLD,
) -> float | None:
    """Hip height normalized to frame height, measured from the BOTTOM of the frame
    (1.0 = top, 0.0 = floor) so the number reads intuitively: bigger = more upright.

    This is the single most informative wrestling feature — a level change (the start
    of nearly every shot) is a rapid drop in this value.
    """
    hip = _midpoint(kpts, L_HIP, R_HIP, conf_threshold)
    if hip is None or frame_height <= 0:
        return None
    return float(1.0 - (hip[1] / frame_height))


def torso_angle(
    kpts: np.ndarray,
    conf_threshold: float = DEFAULT_CONF_THRESHOLD,
) -> float | None:
    """Angle of the shoulder→hip axis from vertical, in degrees (0 = fully upright).

    Distinguishes a squared-up neutral stance from the bent-over posture of someone
    defending or finishing a shot.
    """
    shoulder = _midpoint(kpts, L_SHOULDER, R_SHOULDER, conf_threshold)
    hip = _midpoint(kpts, L_HIP, R_HIP, conf_threshold)
    if shoulder is None or hip is None:
        return None

    dx = hip[0] - shoulder[0]
    dy = hip[1] - shoulder[1]
    if dx == 0 and dy == 0:
        return None
    # atan2(dx, dy): 0 when the torso is vertical (dy dominant), grows as it tilts.
    return float(abs(math.degrees(math.atan2(dx, dy))))


def stance_width(
    kpts: np.ndarray,
    frame_width: int,
    conf_threshold: float = DEFAULT_CONF_THRESHOLD,
) -> float | None:
    """Horizontal ankle separation, normalized to frame width. A wide base means a
    defensive/settled posture; a narrow one often precedes movement.
    """
    if not (_visible(kpts, L_ANKLE, conf_threshold) and _visible(kpts, R_ANKLE, conf_threshold)):
        return None
    if frame_width <= 0:
        return None
    return float(abs(kpts[L_ANKLE, 0] - kpts[R_ANKLE, 0]) / frame_width)


def knee_bend(
    kpts: np.ndarray,
    conf_threshold: float = DEFAULT_CONF_THRESHOLD,
    side: str = "left",
) -> float | None:
    """Interior knee angle in degrees (180 = straight leg, smaller = deeper bend).
    Deep bend + low hips is the signature of a level change.
    """
    hip, knee, ankle = (
        (L_HIP, L_KNEE, L_ANKLE) if side == "left" else (R_HIP, R_KNEE, R_ANKLE)
    )
    if not all(_visible(kpts, i, conf_threshold) for i in (hip, knee, ankle)):
        return None

    thigh = np.array([kpts[hip, 0] - kpts[knee, 0], kpts[hip, 1] - kpts[knee, 1]])
    shin = np.array([kpts[ankle, 0] - kpts[knee, 0], kpts[ankle, 1] - kpts[knee, 1]])
    n1, n2 = np.linalg.norm(thigh), np.linalg.norm(shin)
    if n1 == 0 or n2 == 0:
        return None

    cos_angle = float(np.clip(np.dot(thigh, shin) / (n1 * n2), -1.0, 1.0))
    return float(math.degrees(math.acos(cos_angle)))


def center_of_mass(
    kpts: np.ndarray,
    conf_threshold: float = DEFAULT_CONF_THRESHOLD,
) -> tuple[float, float] | None:
    """Rough COM as the centroid of shoulders and hips — the torso carries most of
    the mass, and those four points are the most reliably visible during contact.
    """
    shoulder = _midpoint(kpts, L_SHOULDER, R_SHOULDER, conf_threshold)
    hip = _midpoint(kpts, L_HIP, R_HIP, conf_threshold)
    if shoulder is None or hip is None:
        return None
    return ((shoulder[0] + hip[0]) / 2, (shoulder[1] + hip[1]) / 2)


def keypoint_visibility(
    kpts: np.ndarray,
    conf_threshold: float = DEFAULT_CONF_THRESHOLD,
) -> float:
    """Fraction of keypoints the model was confident about. This is the honest
    'how much should you trust this frame' signal — during a tight scramble it
    collapses, and downstream layers are expected to react to that rather than
    treating every frame as equally reliable.
    """
    if kpts.size == 0:
        return 0.0
    return float(np.mean(kpts[:, 2] >= conf_threshold))


# --- relational (two-athlete) features -------------------------------------------

def athlete_distance(
    kpts_a: np.ndarray,
    kpts_b: np.ndarray,
    frame_width: int,
    conf_threshold: float = DEFAULT_CONF_THRESHOLD,
) -> float | None:
    """Distance between the two athletes' centers of mass, normalized to frame width.
    Open-distance vs. hand-contact range is a core distinction in the Layer 6
    coaching analysis ("attacks after establishing contact converted better").
    """
    com_a = center_of_mass(kpts_a, conf_threshold)
    com_b = center_of_mass(kpts_b, conf_threshold)
    if com_a is None or com_b is None or frame_width <= 0:
        return None
    dist = math.dist(com_a, com_b)
    return float(dist / frame_width)


def relative_hip_height(
    kpts_a: np.ndarray,
    kpts_b: np.ndarray,
    frame_height: int,
    conf_threshold: float = DEFAULT_CONF_THRESHOLD,
) -> float | None:
    """Athlete A's hip height minus B's. Strongly signals top/bottom: positive means
    A is above B. Being *relative* makes it robust to camera distance in a way that
    absolute height isn't.
    """
    ha = hip_height(kpts_a, frame_height, conf_threshold)
    hb = hip_height(kpts_b, frame_height, conf_threshold)
    if ha is None or hb is None:
        return None
    return float(ha - hb)


def head_position_relative(
    kpts_a: np.ndarray,
    kpts_b: np.ndarray,
    frame_width: int,
    conf_threshold: float = DEFAULT_CONF_THRESHOLD,
) -> float | None:
    """Horizontal offset from A's head to B's center of mass, normalized. Head
    position (inside vs. outside) is a real coaching cue on shots and defense.
    """
    if not _visible(kpts_a, NOSE, conf_threshold) or frame_width <= 0:
        return None
    com_b = center_of_mass(kpts_b, conf_threshold)
    if com_b is None:
        return None
    return float((kpts_a[NOSE, 0] - com_b[0]) / frame_width)


# --- temporal features (need two frames) -------------------------------------------

def velocity(
    point_now: tuple[float, float] | None,
    point_prev: tuple[float, float] | None,
    dt_seconds: float,
    frame_width: int,
) -> tuple[float, float] | None:
    """(vx, vy) in normalized frame-widths per second. Returns None if either frame
    lacked the point — a velocity computed across a gap where the athlete was
    invisible would be meaningless and potentially huge.
    """
    if point_now is None or point_prev is None or dt_seconds <= 0 or frame_width <= 0:
        return None
    vx = (point_now[0] - point_prev[0]) / frame_width / dt_seconds
    vy = (point_now[1] - point_prev[1]) / frame_width / dt_seconds
    return (float(vx), float(vy))


def closing_speed(
    distance_now: float | None,
    distance_prev: float | None,
    dt_seconds: float,
) -> float | None:
    """Rate the athletes are closing. Positive = closing, negative = separating.
    A shot attempt is typically a sharp positive spike alongside a hip-height drop.
    """
    if distance_now is None or distance_prev is None or dt_seconds <= 0:
        return None
    return float((distance_prev - distance_now) / dt_seconds)


def level_change_rate(
    hip_now: float | None,
    hip_prev: float | None,
    dt_seconds: float,
) -> float | None:
    """Rate of hip drop. Positive = dropping (level change). Combined with closing
    speed, this is the primary rules-based signal for the Layer 5 shot detector.
    """
    if hip_now is None or hip_prev is None or dt_seconds <= 0:
        return None
    return float((hip_prev - hip_now) / dt_seconds)


# --- per-frame assembly ----------------------------------------------------------

def compute_frame_features(
    kpts_user: np.ndarray | None,
    kpts_opponent: np.ndarray | None,
    frame_width: int,
    frame_height: int,
    prev: dict | None = None,
    dt_seconds: float = 1 / 30,
    conf_threshold: float = DEFAULT_CONF_THRESHOLD,
) -> dict:
    """Build the full feature dict for one frame.

    Every key is present in every frame (so the output is a rectangular table),
    but values are None wherever the underlying keypoints weren't visible. The
    `*_valid` flags and `both_athletes_visible` make missingness explicit rather
    than something downstream code has to infer from a sentinel value.

    Args:
        kpts_user / kpts_opponent: (17, 3) arrays of (x, y, confidence), or None if
            the athlete had no detection at all in this frame.
        prev: the previous frame's returned dict, for temporal features.
    """
    f: dict = {}

    for label, kpts in (("user", kpts_user), ("opponent", kpts_opponent)):
        if kpts is None:
            f[f"{label}_hip_height"] = None
            f[f"{label}_torso_angle"] = None
            f[f"{label}_stance_width"] = None
            f[f"{label}_knee_bend_l"] = None
            f[f"{label}_knee_bend_r"] = None
            f[f"{label}_com_x"] = None
            f[f"{label}_com_y"] = None
            f[f"{label}_visibility"] = 0.0
            f[f"{label}_detected"] = False
            continue

        com = center_of_mass(kpts, conf_threshold)
        f[f"{label}_hip_height"] = hip_height(kpts, frame_height, conf_threshold)
        f[f"{label}_torso_angle"] = torso_angle(kpts, conf_threshold)
        f[f"{label}_stance_width"] = stance_width(kpts, frame_width, conf_threshold)
        f[f"{label}_knee_bend_l"] = knee_bend(kpts, conf_threshold, side="left")
        f[f"{label}_knee_bend_r"] = knee_bend(kpts, conf_threshold, side="right")
        f[f"{label}_com_x"] = com[0] if com else None
        f[f"{label}_com_y"] = com[1] if com else None
        f[f"{label}_visibility"] = keypoint_visibility(kpts, conf_threshold)
        f[f"{label}_detected"] = True

    both = kpts_user is not None and kpts_opponent is not None
    f["both_athletes_visible"] = both

    if both:
        f["athlete_distance"] = athlete_distance(
            kpts_user, kpts_opponent, frame_width, conf_threshold
        )
        f["relative_hip_height"] = relative_hip_height(
            kpts_user, kpts_opponent, frame_height, conf_threshold
        )
        f["head_position_relative"] = head_position_relative(
            kpts_user, kpts_opponent, frame_width, conf_threshold
        )
    else:
        f["athlete_distance"] = None
        f["relative_hip_height"] = None
        f["head_position_relative"] = None

    # Temporal features.
    if prev is not None:
        com_now = (
            (f["user_com_x"], f["user_com_y"])
            if f["user_com_x"] is not None and f["user_com_y"] is not None
            else None
        )
        com_prev = (
            (prev.get("user_com_x"), prev.get("user_com_y"))
            if prev.get("user_com_x") is not None and prev.get("user_com_y") is not None
            else None
        )
        v = velocity(com_now, com_prev, dt_seconds, frame_width)
        f["user_velocity_x"] = v[0] if v else None
        f["user_velocity_y"] = v[1] if v else None

        f["closing_speed"] = closing_speed(
            f["athlete_distance"], prev.get("athlete_distance"), dt_seconds
        )
        f["user_level_change_rate"] = level_change_rate(
            f["user_hip_height"], prev.get("user_hip_height"), dt_seconds
        )
    else:
        f["user_velocity_x"] = None
        f["user_velocity_y"] = None
        f["closing_speed"] = None
        f["user_level_change_rate"] = None

    return f


def interpolate_short_gaps(
    values: list[float | None],
    max_gap: int = 3,
) -> list[float | None]:
    """Linearly fill runs of None no longer than `max_gap`, leaving longer runs alone.

    A one- or two-frame dropout is almost always a momentary detection miss and is
    safe to bridge. A long run means the athlete was genuinely occluded or out of
    frame, and inventing values across it would fabricate motion that never happened
    — exactly the kind of quiet dishonesty this project is built to avoid.
    """
    out = list(values)
    n = len(out)
    i = 0
    while i < n:
        if out[i] is not None:
            i += 1
            continue

        gap_start = i
        while i < n and out[i] is None:
            i += 1
        gap_end = i  # exclusive

        gap_len = gap_end - gap_start
        has_both_anchors = gap_start > 0 and gap_end < n
        if has_both_anchors and gap_len <= max_gap:
            before = out[gap_start - 1]
            after = out[gap_end]
            if before is not None and after is not None:
                step = (after - before) / (gap_len + 1)
                for k in range(gap_len):
                    out[gap_start + k] = before + step * (k + 1)
    return out
