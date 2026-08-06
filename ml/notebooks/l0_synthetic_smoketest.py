"""
ml/notebooks/l0_synthetic_smoketest.py — Layer 0 PLUMBING smoke test (sandbox-runnable)

*** This does NOT validate the Layer 0 acceptance gate. ***
The gate ("wrestlers hold stable IDs through >=80% of active-wrestling time") can only be
measured on real match footage with real YOLO detections — that requires torch/ultralytics
and belongs on the Mac (see l0_tracer_bullet.py).

What THIS script validates: that the pipeline mechanics work end to end without crashing —
video I/O, a detector-shaped interface, ByteTrack integration, identity seeding/re-ID,
overlay rendering, and parquet export — using a synthetic video of colored blobs and a
plain OpenCV color-threshold "detector" (no torch dependency at all). It's a fast,
deterministic CI-style check that catches integration bugs before we ever touch real
footage or burn Mac time.

Run with:  python3 ml/notebooks/l0_synthetic_smoketest.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import supervision as sv

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ml.features.tracking_metrics import summarize_track_quality, find_reid_candidate

WIDTH, HEIGHT, N_FRAMES, FPS = 480, 320, 150, 30.0

# Colors (BGR) for synthetic actors: two "wrestlers" + a "referee".
COLORS = {
    "wrestler_a": (255, 60, 60),   # blue
    "wrestler_b": (60, 60, 255),   # red
    "referee": (200, 200, 200),    # light gray
}


def generate_synthetic_frames():
    """Yields (frame_idx, frame) for a synthetic match: two blobs circle/overlap
    each other (simulating a scramble with occlusion) while a third smaller,
    slower blob (the referee) orbits the perimeter.
    """
    t = np.arange(N_FRAMES)
    cx, cy = WIDTH / 2, HEIGHT / 2

    # Wrestler A: orbits with radius that shrinks into overlap mid-match (occlusion stress).
    a_radius = 60 - 40 * np.abs(np.sin(t / N_FRAMES * np.pi))
    ax = cx + a_radius * np.cos(t * 0.15)
    ay = cy + a_radius * np.sin(t * 0.15)

    # Wrestler B: opposite phase, same shrinking radius -> the two collide around the midpoint.
    bx = cx - a_radius * np.cos(t * 0.15)
    by = cy - a_radius * np.sin(t * 0.15)

    # Referee: large slow orbit near the edge.
    rx = cx + 140 * np.cos(t * 0.04)
    ry = cy + 140 * np.sin(t * 0.04)

    for i in range(N_FRAMES):
        frame = np.full((HEIGHT, WIDTH, 3), (30, 90, 30), dtype=np.uint8)  # green mat
        cv2.circle(frame, (int(rx[i]), int(ry[i])), 14, COLORS["referee"], -1)
        cv2.circle(frame, (int(ax[i]), int(ay[i])), 20, COLORS["wrestler_a"], -1)
        cv2.circle(frame, (int(bx[i]), int(by[i])), 20, COLORS["wrestler_b"], -1)
        yield i, frame


def color_blob_detector(frame: np.ndarray) -> list[tuple]:
    """A trivial stand-in for a real detector: threshold each known color and
    return bounding boxes. This lets us test the tracker/pipeline plumbing
    without needing YOLO or torch.
    """
    boxes = []
    for color in COLORS.values():
        mask = cv2.inRange(frame, np.array(color) - 10, np.array(color) + 10)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            if cv2.contourArea(c) < 30:
                continue
            x, y, w, h = cv2.boundingRect(c)
            boxes.append((x, y, x + w, y + h))
    return boxes


def iou(box_a, box_b) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def main() -> int:
    out_dir = Path(__file__).parent / "output_smoketest"
    out_dir.mkdir(parents=True, exist_ok=True)

    tracker = sv.ByteTrack(frame_rate=int(FPS))
    writer = cv2.VideoWriter(
        str(out_dir / "smoketest_overlay.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT),
    )

    identity_to_track_id = {"wrestler_a": None, "wrestler_b": None}
    identity_last_bbox = {"wrestler_a": None, "wrestler_b": None}
    identity_lost_at = {"wrestler_a": None, "wrestler_b": None}
    track_rows = []

    for frame_idx, frame in generate_synthetic_frames():
        boxes = color_blob_detector(frame)
        if boxes:
            xyxy = np.array(boxes, dtype=float)
            confidence = np.ones(len(boxes))
            class_id = np.zeros(len(boxes), dtype=int)
            detections = sv.Detections(xyxy=xyxy, confidence=confidence, class_id=class_id)
        else:
            detections = sv.Detections.empty()

        detections = tracker.update_with_detections(detections)

        # Seed identities on frame 0 using ground-truth-ish nearest box to each
        # actor's known starting position (stand-in for the real click-to-identify).
        if frame_idx == 0 and detections.tracker_id is not None:
            seed_targets = {
                "wrestler_a": (WIDTH / 2 + 60, HEIGHT / 2),
                "wrestler_b": (WIDTH / 2 - 60, HEIGHT / 2),
            }
            for ident, (sx, sy) in seed_targets.items():
                best_idx, best_dist = None, float("inf")
                for i, box in enumerate(detections.xyxy):
                    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                    d = (cx - sx) ** 2 + (cy - sy) ** 2
                    if d < best_dist:
                        best_idx, best_dist = i, d
                if best_idx is not None:
                    identity_to_track_id[ident] = int(detections.tracker_id[best_idx])

        frame_dets = []
        present_tids = set()
        if detections.tracker_id is not None:
            for i, box in enumerate(detections.xyxy):
                tid = int(detections.tracker_id[i])
                present_tids.add(tid)
                identity = "unknown"
                for ident, canonical in identity_to_track_id.items():
                    if tid == canonical:
                        identity = ident
                        identity_last_bbox[ident] = tuple(box)
                        identity_lost_at[ident] = None
                frame_dets.append({"track_id": tid, "bbox": tuple(box), "identity": identity})

        for ident, canonical in identity_to_track_id.items():
            if canonical is not None and canonical not in present_tids and identity_lost_at[ident] is None:
                identity_lost_at[ident] = frame_idx

        # Periodic re-ID attempt after occlusion.
        if track_rows and frame_idx % 5 == 0:
            recent_df = pd.DataFrame(track_rows[-200:])
            for ident in ("wrestler_a", "wrestler_b"):
                if identity_lost_at[ident] is not None and identity_last_bbox[ident] is not None:
                    candidate = find_reid_candidate(
                        lost_identity_last_bbox=identity_last_bbox[ident],
                        lost_at_frame=identity_lost_at[ident],
                        candidate_tracks=recent_df,
                        max_gap_frames=15,
                        max_center_dist_px=60.0,
                    )
                    if candidate is not None and candidate != identity_to_track_id[ident]:
                        identity_to_track_id[ident] = candidate
                        identity_lost_at[ident] = None

        for d in frame_dets:
            x1, y1, x2, y2 = d["bbox"]
            track_rows.append({
                "frame": frame_idx, "track_id": d["track_id"], "identity": d["identity"],
                "x1": x1, "y1": y1, "x2": x2, "y2": y2, "confidence": 1.0,
            })
            color = COLORS.get(d["identity"], (0, 220, 220))
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 1)
            cv2.putText(frame, f"{d['identity']}#{d['track_id']}", (int(x1), max(0, int(y1) - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

        writer.write(frame)

    writer.release()

    tracks_df = pd.DataFrame(track_rows)
    tracks_df.to_parquet(out_dir / "smoketest_tracks.parquet", index=False)
    active_mask = pd.Series(True, index=range(N_FRAMES))
    report = summarize_track_quality(tracks_df, active_mask, fps=FPS)

    print("=== LAYER 0 SMOKE TEST (synthetic — plumbing check only, NOT the real gate) ===")
    print(f"Frames processed: {N_FRAMES}")
    print(f"Overlay video:    {out_dir / 'smoketest_overlay.mp4'}")
    print(f"Tracks parquet:   {out_dir / 'smoketest_tracks.parquet'}")
    print(f"Overall ID-hold:  {report['overall_id_hold']:.1%}")
    for ident, stats in report["per_identity"].items():
        print(f"  {ident}: hold={stats['id_hold_fraction']:.1%} "
              f"switches={stats['identity_switches']} lost_frames={stats['lost_track_frames']}")

    # Sanity assertions: the plumbing itself must produce sane, non-degenerate output.
    assert len(tracks_df) > 0, "No tracks were produced at all — pipeline is broken."
    assert report["overall_id_hold"] > 0.0, "Zero ID-hold — identity seeding/tracking is broken."
    assert (out_dir / "smoketest_overlay.mp4").exists()
    assert (out_dir / "smoketest_tracks.parquet").exists()
    print("\nPLUMBING OK — pipeline mechanics work end to end.")
    print("Reminder: run l0_tracer_bullet.py on a REAL match on the Mac for the actual gate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
