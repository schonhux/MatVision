"""
POSE stage — extracts skeleton keypoints for the two tracked wrestlers.

Runs only on frames the tracker already sampled, and matches pose outputs back to
tracked boxes by IoU so every keypoint set carries a known identity. Poses that
don't match any tracked wrestler (referee, spectators) are dropped rather than
stored — they'd be noise in every downstream feature.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import cv2
import pandas as pd
from sqlalchemy.orm import Session

from app.models import Match
from app.stages.base import StageError
from app import storage

logger = logging.getLogger(__name__)

MIN_POSE_IOU = 0.3


def run(match: Match, db: Session) -> dict:
    from app.vision.models import estimate_pose  # lazy: see ADR-007
    from ml.features.identity import iou

    source_key = match.video_keys.get("analysis_720p")
    if not source_key:
        raise StageError("No transcoded video available")

    tracks_key = storage.object_key(match.user_id, match.id, "artifacts", "tracks.parquet")
    if not storage.object_exists(tracks_key):
        raise StageError("tracks.parquet missing — the detect_track stage must run first")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        local_video = str(tmp_path / "analysis.mp4")
        local_tracks = str(tmp_path / "tracks.parquet")
        storage.download_to_path(source_key, local_video)
        storage.download_to_path(tracks_key, local_tracks)

        tracks = pd.read_parquet(local_tracks)
        # Only the two wrestlers matter — skip referee/unknown tracks entirely.
        wrestler_tracks = tracks[tracks["identity"].isin(["user", "opponent"])]
        if wrestler_tracks.empty:
            raise StageError(
                "No wrestler identities were resolved during tracking, so there is "
                "nothing to extract pose for."
            )

        frames_needed = set(wrestler_tracks["frame"].unique())
        by_frame = {
            frame: group.to_dict("records")
            for frame, group in wrestler_tracks.groupby("frame")
        }

        cap = cv2.VideoCapture(local_video)
        if not cap.isOpened():
            raise StageError("Could not open the transcoded video for pose extraction")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        rows: list[dict] = []
        frame_idx = 0
        matched, unmatched = 0, 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx not in frames_needed:
                frame_idx += 1
                continue

            poses = estimate_pose(frame)
            tracked_here = by_frame.get(frame_idx, [])

            for tr in tracked_here:
                tr_box = (tr["x1"], tr["y1"], tr["x2"], tr["y2"])
                best, best_iou = None, MIN_POSE_IOU
                for p in poses:
                    score = iou(tr_box, p.bbox)
                    if score > best_iou:
                        best, best_iou = p, score

                if best is None or best.keypoints is None:
                    unmatched += 1
                    continue

                matched += 1
                rows.append({
                    "frame": frame_idx,
                    "timestamp_ms": int(frame_idx / fps * 1000),
                    "track_id": tr["track_id"],
                    "identity": tr["identity"],
                    "keypoints": best.keypoints.flatten().tolist(),  # 17*3, flat for parquet
                    "pose_iou": round(best_iou, 4),
                    "mean_kp_confidence": float(best.keypoints[:, 2].mean()),
                })

            frame_idx += 1

        cap.release()

        if not rows:
            raise StageError("Pose estimation produced no usable keypoints for either wrestler")

        df = pd.DataFrame(rows)
        artifact_key = storage.object_key(match.user_id, match.id, "artifacts", "poses.parquet")
        local_parquet = str(tmp_path / "poses.parquet")
        df.to_parquet(local_parquet, index=False)
        storage.upload_from_path(local_parquet, artifact_key)

        total_attempts = matched + unmatched
        return {
            "poses_key": artifact_key,
            "pose_rows": len(df),
            "matched": matched,
            "unmatched": unmatched,
            # How often a tracked wrestler had no usable pose. Expected to be
            # meaningfully non-zero during scrambles — that's the known hard case,
            # and it's recorded rather than hidden.
            "match_rate": round(matched / total_attempts, 4) if total_attempts else 0.0,
            "mean_keypoint_confidence": round(float(df["mean_kp_confidence"].mean()), 4),
        }
