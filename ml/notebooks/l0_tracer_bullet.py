"""
ml/notebooks/l0_tracer_bullet.py — Layer 0: Tracer Bullet

Runs the full de-risking pipeline on a real wrestling match video:
    video -> YOLO person detection -> ByteTrack -> YOLOv8-pose -> overlay + parquet

This is the ONE thing we must prove before building any product code: that off-the-shelf
detection/tracking survives real wrestling footage (two entangled bodies + a referee).

*** This script requires torch + ultralytics and real compute/time. Run it on the Mac: ***

    cd ml
    python3 -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    python notebooks/l0_tracer_bullet.py --video /path/to/match.mp4 --out notebooks/output

First run downloads ~12MB (yolov8n.pt) and ~7MB (yolov8n-pose.pt) automatically.

Identity seeding: since Layer 0 predates the click-to-identify UI (that's Layer 2),
identity is seeded manually — point at a frame near the start where both wrestlers are
clearly separated and give their bounding boxes. See --wrestler-a-seed / --wrestler-b-seed.

Outputs (written to --out):
    overlay.mp4        — boxes, track ids, skeletons, colored by resolved identity
    tracks.parquet      — frame, track_id, identity, bbox, confidence
    poses.parquet       — frame, track_id, identity, 17 COCO keypoints (x, y, conf)
    quality_report.json — the Layer 0 acceptance-gate numbers (see tracking_metrics.py)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# Allow running as a script from anywhere in the repo.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ml.features.tracking_metrics import (
    summarize_track_quality,
    find_reid_candidate,
)

COCO_SKELETON = [
    (5, 7), (7, 9), (6, 8), (8, 10),          # arms
    (5, 6), (5, 11), (6, 12), (11, 12),        # torso
    (11, 13), (13, 15), (12, 14), (14, 16),    # legs
    (0, 5), (0, 6),                            # head to shoulders (approx)
]

IDENTITY_COLORS = {
    "wrestler_a": (255, 80, 40),   # blue-ish (BGR)
    "wrestler_b": (40, 40, 255),   # red
    "referee": (160, 160, 160),    # gray
    "unknown": (0, 220, 220),      # yellow
}


def parse_bbox(s: str) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = map(float, s.split(","))
    return x1, y1, x2, y2


def iou(box_a, box_b) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def run(args: argparse.Namespace) -> None:
    # Imports here (not at module top) so this file can still be syntax-checked /
    # imported for its helper functions in environments without torch installed.
    from ultralytics import YOLO
    import supervision as sv

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[l0] video: {width}x{height} @ {fps:.1f}fps, {total_frames} frames "
          f"({total_frames / fps:.1f}s)")

    detector = YOLO(args.detect_model)
    pose_model = YOLO(args.pose_model)
    tracker = sv.ByteTrack(frame_rate=int(round(fps)))

    writer = cv2.VideoWriter(
        str(out_dir / "overlay.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    track_rows: list[dict] = []
    pose_rows: list[dict] = []

    # Identity seeding state: canonical track_id per identity, resolved at the seed
    # frame and re-attached via find_reid_candidate() whenever a track is lost.
    identity_to_track_id: dict[str, int | None] = {"wrestler_a": None, "wrestler_b": None}
    identity_last_bbox: dict[str, tuple | None] = {"wrestler_a": None, "wrestler_b": None}
    identity_lost_at: dict[str, int | None] = {"wrestler_a": None, "wrestler_b": None}

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        det_result = detector(frame, classes=[0], conf=args.conf, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(det_result)
        detections = tracker.update_with_detections(detections)

        # Seed identities at the requested frame.
        if frame_idx == args.seed_frame:
            for ident, seed_box in (
                ("wrestler_a", args.wrestler_a_seed),
                ("wrestler_b", args.wrestler_b_seed),
            ):
                if seed_box is None or detections.tracker_id is None:
                    continue
                best_idx, best_iou = None, 0.0
                for i, box in enumerate(detections.xyxy):
                    score = iou(tuple(box), seed_box)
                    if score > best_iou:
                        best_idx, best_iou = i, score
                if best_idx is not None and best_iou > 0.1:
                    identity_to_track_id[ident] = int(detections.tracker_id[best_idx])
                    print(f"[l0] seeded {ident} -> track_id "
                          f"{identity_to_track_id[ident]} (iou={best_iou:.2f})")

        # Resolve each detection's identity for this frame + attempt re-ID recovery.
        frame_dets = []
        if detections.tracker_id is not None:
            for i, box in enumerate(detections.xyxy):
                tid = int(detections.tracker_id[i])
                conf = float(detections.confidence[i]) if detections.confidence is not None else 0.0
                identity = "unknown"
                for ident, canonical_tid in identity_to_track_id.items():
                    if tid == canonical_tid:
                        identity = ident
                        identity_last_bbox[ident] = tuple(box)
                        identity_lost_at[ident] = None
                        break
                frame_dets.append({"track_id": tid, "bbox": tuple(box), "conf": conf, "identity": identity})

        # Check for newly-lost identities and try to re-attach via nearby new tracks.
        present_tids = {d["track_id"] for d in frame_dets}
        for ident, canonical_tid in list(identity_to_track_id.items()):
            if canonical_tid is not None and canonical_tid not in present_tids:
                if identity_lost_at[ident] is None:
                    identity_lost_at[ident] = frame_idx
            elif canonical_tid in present_tids:
                identity_lost_at[ident] = None

        # Attempt re-ID stitching using unknown/newly-seen tracks each frame.
        if frame_idx > 0 and frame_idx % 5 == 0:  # cheap: check periodically, not every frame
            recent_df = pd.DataFrame(track_rows[-500:]) if track_rows else pd.DataFrame(
                columns=["frame", "track_id", "x1", "y1", "x2", "y2"]
            )
            for ident in ("wrestler_a", "wrestler_b"):
                if identity_lost_at[ident] is not None and identity_last_bbox[ident] is not None:
                    candidate = find_reid_candidate(
                        lost_identity_last_bbox=identity_last_bbox[ident],
                        lost_at_frame=identity_lost_at[ident],
                        candidate_tracks=recent_df.rename(
                            columns={"track_id": "track_id"}
                        )[["track_id", "frame", "x1", "y1", "x2", "y2"]] if not recent_df.empty else recent_df,
                        max_gap_frames=args.reid_max_gap,
                        max_center_dist_px=args.reid_max_dist,
                    )
                    if candidate is not None and candidate != identity_to_track_id[ident]:
                        identity_to_track_id[ident] = candidate
                        identity_lost_at[ident] = None
                        print(f"[l0] frame {frame_idx}: re-attached {ident} -> track_id {candidate}")

        for d in frame_dets:
            x1, y1, x2, y2 = d["bbox"]
            track_rows.append({
                "frame": frame_idx, "track_id": d["track_id"], "identity": d["identity"],
                "x1": x1, "y1": y1, "x2": x2, "y2": y2, "confidence": d["conf"],
            })

        # Pose (run on full frame; matched to detections by IoU — simplest correct approach).
        if frame_idx % args.pose_every == 0 and frame_dets:
            pose_result = pose_model(frame, conf=args.conf, verbose=False)[0]
            if pose_result.keypoints is not None:
                kpts_all = pose_result.keypoints.data.cpu().numpy()  # (n, 17, 3)
                boxes_all = pose_result.boxes.xyxy.cpu().numpy() if pose_result.boxes is not None else []
                for kpts, pbox in zip(kpts_all, boxes_all):
                    best_d, best_iou = None, 0.0
                    for d in frame_dets:
                        score = iou(d["bbox"], tuple(pbox))
                        if score > best_iou:
                            best_d, best_iou = d, score
                    if best_d is not None and best_iou > 0.3:
                        pose_rows.append({
                            "frame": frame_idx, "track_id": best_d["track_id"],
                            "identity": best_d["identity"],
                            "keypoints": kpts.tolist(),
                        })
                        _draw_skeleton(frame, kpts)

        _draw_boxes(frame, frame_dets)
        writer.write(frame)

        if frame_idx % 200 == 0:
            print(f"[l0] frame {frame_idx}/{total_frames}")
        frame_idx += 1

    cap.release()
    writer.release()

    tracks_df = pd.DataFrame(track_rows)
    poses_df = pd.DataFrame(pose_rows)
    tracks_df.to_parquet(out_dir / "tracks.parquet", index=False)
    if not poses_df.empty:
        poses_df.to_parquet(out_dir / "poses.parquet", index=False)

    # Active-frame mask: for the tracer bullet, treat every processed frame as
    # "active" unless --active-range is given (a real match will refine this in Layer 3/4).
    active_mask = pd.Series(True, index=range(frame_idx))
    if args.active_range:
        start, end = args.active_range
        active_mask[:] = False
        active_mask.loc[start:end] = True

    report = summarize_track_quality(tracks_df, active_mask, fps=fps)
    report["video"] = args.video
    report["frame_count"] = frame_idx
    report["duration_sec"] = round(frame_idx / fps, 1)

    with open(out_dir / "quality_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("\n=== LAYER 0 ACCEPTANCE GATE ===")
    print(f"Overall ID-hold: {report['overall_id_hold']:.1%}  "
          f"(gate: >=80%)  ->  {'PASS' if report['passes_gate'] else 'FAIL'}")
    for ident, stats in report["per_identity"].items():
        print(f"  {ident}: hold={stats['id_hold_fraction']:.1%}  "
              f"switches={stats['identity_switches']}  "
              f"lost={stats['lost_track_seconds']}s")
    print(f"\nOverlay video: {out_dir / 'overlay.mp4'}")
    print(f"Full report:   {out_dir / 'quality_report.json'}")


def _draw_boxes(frame: np.ndarray, dets: list[dict]) -> None:
    for d in dets:
        x1, y1, x2, y2 = map(int, d["bbox"])
        color = IDENTITY_COLORS.get(d["identity"], IDENTITY_COLORS["unknown"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{d['identity']} #{d['track_id']}"
        cv2.putText(frame, label, (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def _draw_skeleton(frame: np.ndarray, kpts: np.ndarray, conf_thresh: float = 0.3) -> None:
    for i, j in COCO_SKELETON:
        if kpts[i, 2] > conf_thresh and kpts[j, 2] > conf_thresh:
            pt1 = tuple(kpts[i, :2].astype(int))
            pt2 = tuple(kpts[j, :2].astype(int))
            cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
    for x, y, c in kpts:
        if c > conf_thresh:
            cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="MatVision Layer 0 — Tracer Bullet")
    p.add_argument("--video", required=True, help="Path to a real match video")
    p.add_argument("--out", default="notebooks/output", help="Output directory")
    p.add_argument("--detect-model", default="yolov8n.pt")
    p.add_argument("--pose-model", default="yolov8n-pose.pt")
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--pose-every", type=int, default=1,
                    help="Run pose every N frames (1 = every frame)")
    p.add_argument("--seed-frame", type=int, default=0,
                    help="Frame index at which to seed wrestler identities")
    p.add_argument("--wrestler-a-seed", type=parse_bbox, default=None,
                    help="x1,y1,x2,y2 bbox of wrestler A at --seed-frame")
    p.add_argument("--wrestler-b-seed", type=parse_bbox, default=None,
                    help="x1,y1,x2,y2 bbox of wrestler B at --seed-frame")
    p.add_argument("--reid-max-gap", type=int, default=45,
                    help="Max frames a wrestler can be lost before re-ID gives up")
    p.add_argument("--reid-max-dist", type=float, default=150.0,
                    help="Max pixel distance for re-ID re-attachment")
    p.add_argument("--active-range", type=int, nargs=2, default=None, metavar=("START", "END"),
                    help="Optional: restrict the acceptance-gate calc to this frame range")
    return p


if __name__ == "__main__":
    run(build_arg_parser().parse_args())
