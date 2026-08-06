from __future__ import annotations

import math


def compute_bbox_features(
    user_box: tuple[float, float, float, float] | None,
    opponent_box: tuple[float, float, float, float] | None,
    frame_width: int,
    frame_height: int,
    prev: dict | None = None,
    dt_seconds: float = 1 / 8,
) -> dict:
    result: dict = {}

    for role, box in (("user", user_box), ("opponent", opponent_box)):
        prefix = f"{role}_bbox"
        if box is None or frame_width <= 0 or frame_height <= 0:
            result[f"{prefix}_detected"] = False
            for name in ("center_x", "center_y", "width", "height", "area", "speed"):
                result[f"{prefix}_{name}"] = None
            continue

        x1, y1, x2, y2 = box
        center_x = ((x1 + x2) / 2) / frame_width
        center_y = ((y1 + y2) / 2) / frame_height
        width = max(0.0, x2 - x1) / frame_width
        height = max(0.0, y2 - y1) / frame_height

        result[f"{prefix}_detected"] = True
        result[f"{prefix}_center_x"] = center_x
        result[f"{prefix}_center_y"] = center_y
        result[f"{prefix}_width"] = width
        result[f"{prefix}_height"] = height
        result[f"{prefix}_area"] = width * height

        old_x = (prev or {}).get(f"{prefix}_center_x")
        old_y = (prev or {}).get(f"{prefix}_center_y")
        if old_x is None or old_y is None or dt_seconds <= 0:
            result[f"{prefix}_speed"] = None
        else:
            result[f"{prefix}_speed"] = math.hypot(center_x - old_x, center_y - old_y) / dt_seconds

    if user_box is None or opponent_box is None:
        result.update({
            "bbox_distance": None,
            "bbox_overlap": None,
            "bbox_vertical_gap": None,
            "bbox_height_ratio": None,
            "bbox_closing_speed": None,
        })
        return result

    ux = result["user_bbox_center_x"]
    uy = result["user_bbox_center_y"]
    ox = result["opponent_bbox_center_x"]
    oy = result["opponent_bbox_center_y"]
    distance = math.hypot(ux - ox, uy - oy)

    result["bbox_distance"] = distance
    result["bbox_overlap"] = _iou(user_box, opponent_box)
    result["bbox_vertical_gap"] = uy - oy
    opponent_height = result["opponent_bbox_height"]
    result["bbox_height_ratio"] = (
        result["user_bbox_height"] / opponent_height if opponent_height else None
    )

    old_distance = (prev or {}).get("bbox_distance")
    result["bbox_closing_speed"] = (
        (old_distance - distance) / dt_seconds
        if old_distance is not None and dt_seconds > 0
        else None
    )
    return result


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0
