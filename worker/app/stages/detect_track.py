"""
DETECT+TRACK stage — productionizes the Layer 0 tracer bullet.

Reads the 720p analysis copy, runs person detection + ByteTrack, resolves which
track is the user / opponent / referee (using the Layer 2 click-to-identify seed
when available), and writes tracks.parquet plus a track-quality report.

Coarse sampling (PROJECT_GUIDE.md: "Do not run the most expensive model on every
full-resolution frame") — we process ~8fps here rather than 30. That's enough to
follow position and identity; Layer 5 can re-run densely around candidate events
if finer temporal resolution turns out to matter.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import cv2
import pandas as pd
from sqlalchemy.orm import Session

from app.models import Match, MatchAthlete
from app.stages.base import StageError
from app import storage

logger = logging.getLogger(__name__)

SAMPLE_FPS = 8.0


def run(match: Match, db: Session) -> dict:
    # Imported here, not at module top: torch/ultralytics can't be installed in the
    # test sandbox (ADR-007), and this keeps the module importable everywhere.
    from app.vision.models import detect_people, make_tracker
    from ml.features.identity import (
        bind_identity_from_seed, denormalize_bbox, compute_track_stats, resolve_identities,
    )
    import supervision as sv
    import numpy as np

    source_key = match.video_keys.get("analysis_720p")
    if not source_key:
        raise StageError("No transcoded video available — transcode stage must run first")

    athletes = db.query(MatchAthlete).filter(MatchAthlete.match_id == match.id).all()
    seeds = {a.role: a for a in athletes if a.seed_bbox}

    with tempfile.TemporaryDirectory() as tmp:
        local_video = str(Path(tmp) / "analysis.mp4")
        storage.download_to_path(source_key, local_video)

        cap = cv2.VideoCapture(local_video)
        if not cap.isOpened():
            raise StageError("Could not open the transcoded video for tracking")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        stride = max(1, int(round(fps / SAMPLE_FPS)))

        tracker = make_tracker(fps / stride)
        rows: list[dict] = []
        seed_frame_detections: dict[str, list[dict]] = {}

        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % stride != 0:
                frame_idx += 1
                continue

            people = detect_people(frame)
            if people:
                detections = sv.Detections(
                    xyxy=np.array([p.bbox for p in people], dtype=float),
                    confidence=np.array([p.confidence for p in people], dtype=float),
                    class_id=np.zeros(len(people), dtype=int),
                )
            else:
                detections = sv.Detections.empty()

            tracked = tracker.update_with_detections(detections)
            timestamp_ms = int(frame_idx / fps * 1000)

            frame_dets = []
            if tracked.tracker_id is not None:
                for i, box in enumerate(tracked.xyxy):
                    row = {
                        "frame": frame_idx,
                        "timestamp_ms": timestamp_ms,
                        "track_id": int(tracked.tracker_id[i]),
                        "x1": float(box[0]), "y1": float(box[1]),
                        "x2": float(box[2]), "y2": float(box[3]),
                        "confidence": float(tracked.confidence[i]) if tracked.confidence is not None else 0.0,
                    }
                    rows.append(row)
                    frame_dets.append({"track_id": row["track_id"],
                                       "bbox": (row["x1"], row["y1"], row["x2"], row["y2"])})

            # Capture detections at each athlete's seed frame for identity binding.
            for role, athlete in seeds.items():
                if athlete.seed_frame_ms is None:
                    continue
                seed_frame = int(athlete.seed_frame_ms / 1000 * fps)
                if abs(frame_idx - seed_frame) < stride and role not in seed_frame_detections:
                    seed_frame_detections[role] = frame_dets

            frame_idx += 1

        cap.release()

        if not rows:
            raise StageError(
                "No people detected anywhere in this video. Check that the footage "
                "actually shows a wrestling match with the mat in frame."
            )

        # Bind identities: explicit user seeds first, heuristics for the rest.
        seed_track_ids: dict[str, int | None] = {}
        for role, athlete in seeds.items():
            dets = seed_frame_detections.get(role, [])
            if dets:
                seed_track_ids[role] = bind_identity_from_seed(
                    dets, denormalize_bbox(athlete.seed_bbox, width, height)
                )

        track_stats = compute_track_stats(rows, width, height)
        roles = resolve_identities(
            track_stats,
            user_seed_track_id=seed_track_ids.get("user"),
            opponent_seed_track_id=seed_track_ids.get("opponent"),
        )

        df = pd.DataFrame(rows)
        df["identity"] = df["track_id"].map(roles).fillna("unknown")

        artifact_key = storage.object_key(match.user_id, match.id, "artifacts", "tracks.parquet")
        local_parquet = str(Path(tmp) / "tracks.parquet")
        df.to_parquet(local_parquet, index=False)
        storage.upload_from_path(local_parquet, artifact_key)

        quality = _track_quality(df, roles, fps / stride)

        return {
            "tracks_key": artifact_key,
            "frames_processed": frame_idx,
            "frames_sampled": len(df["frame"].unique()),
            "sample_fps": SAMPLE_FPS,
            "track_count": int(df["track_id"].nunique()),
            "identities": {str(k): v for k, v in roles.items()},
            "seed_bound": {k: v for k, v in seed_track_ids.items() if v is not None},
            "quality": quality,
            "frame_width": width,
            "frame_height": height,
        }


def _track_quality(df: pd.DataFrame, roles: dict[int, str], effective_fps: float) -> dict:
    """Per-identity tracking quality. Stored so a bad track is visible up front
    rather than discovered later as inexplicably poor model performance — this is
    the Layer 3 acceptance criterion "low-confidence intervals are flagged."
    """
    total_frames = df["frame"].nunique()
    quality: dict = {"total_sampled_frames": int(total_frames)}

    for role in ("user", "opponent"):
        track_ids = [tid for tid, r in roles.items() if r == role]
        if not track_ids:
            quality[role] = {"present": False}
            continue

        sub = df[df["track_id"].isin(track_ids)]
        frames_present = sub["frame"].nunique()
        quality[role] = {
            "present": True,
            "track_ids": [int(t) for t in track_ids],
            "frames_present": int(frames_present),
            "presence_ratio": round(frames_present / total_frames, 4) if total_frames else 0.0,
            "identity_switches": max(0, len(track_ids) - 1),
            "mean_confidence": round(float(sub["confidence"].mean()), 4) if len(sub) else 0.0,
            "lost_seconds": round((total_frames - frames_present) / effective_fps, 2)
            if effective_fps else None,
        }

    ratios = [
        quality[r]["presence_ratio"] for r in ("user", "opponent")
        if quality.get(r, {}).get("present")
    ]
    quality["overall_presence_ratio"] = round(sum(ratios) / len(ratios), 4) if ratios else 0.0
    # The Layer 0 / BUILD_PLAN gate: wrestlers held through >=80% of active time.
    quality["meets_gate"] = quality["overall_presence_ratio"] >= 0.80

    return quality
