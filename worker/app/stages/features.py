"""
FEATURES stage — poses.parquet -> features.parquet.

Converts raw keypoints into the wrestling-meaningful numbers Layer 4's state
classifier and Layer 5's event rules actually consume. All the math lives in
ml/features/wrestling_features.py (torch-free and unit-tested); this stage is just
I/O and assembly.

Missingness is preserved deliberately: frames where a wrestler wasn't visible get
None values plus explicit `*_detected` flags, and only short gaps are interpolated.
See the module docstring in wrestling_features.py for why fabricating values here
would be actively harmful.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from app.models import Match
from app.stages.base import StageError
from app import storage

logger = logging.getLogger(__name__)

INTERPOLATE_MAX_GAP = 3  # frames; ~0.4s at the 8fps sampling rate


def run(match: Match, db: Session) -> dict:
    from ml.features.wrestling_features import compute_frame_features, interpolate_short_gaps

    poses_key = storage.object_key(match.user_id, match.id, "artifacts", "poses.parquet")
    tracks_key = storage.object_key(match.user_id, match.id, "artifacts", "tracks.parquet")
    if not storage.object_exists(poses_key):
        raise StageError("poses.parquet missing — the pose stage must run first")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        local_poses = str(tmp_path / "poses.parquet")
        local_tracks = str(tmp_path / "tracks.parquet")
        storage.download_to_path(poses_key, local_poses)
        storage.download_to_path(tracks_key, local_tracks)

        poses = pd.read_parquet(local_poses)
        tracks = pd.read_parquet(local_tracks)

        # Frame dimensions come from the actual box extents rather than being
        # assumed — the transcode stage preserves aspect ratio, so width varies.
        frame_width = int(max(tracks["x2"].max(), 1))
        frame_height = int(max(tracks["y2"].max(), 1))
        # Round up to the standard 720p-height analysis copy.
        frame_height = max(frame_height, 720)

        by_frame: dict[int, dict[str, np.ndarray]] = {}
        for row in poses.itertuples():
            kpts = np.array(row.keypoints, dtype=float).reshape(17, 3)
            by_frame.setdefault(row.frame, {})[row.identity] = kpts

        timestamps = dict(zip(poses["frame"], poses["timestamp_ms"]))
        ordered_frames = sorted(by_frame.keys())
        if not ordered_frames:
            raise StageError("No pose frames available to compute features from")

        # dt between consecutive *sampled* frames, not raw video frames.
        if len(ordered_frames) > 1 and len(timestamps) > 1:
            deltas = [
                (timestamps[ordered_frames[i]] - timestamps[ordered_frames[i - 1]]) / 1000
                for i in range(1, len(ordered_frames))
            ]
            dt = float(np.median([d for d in deltas if d > 0]) or (1 / 8))
        else:
            dt = 1 / 8

        rows: list[dict] = []
        prev = None
        for frame in ordered_frames:
            entry = by_frame[frame]
            feats = compute_frame_features(
                kpts_user=entry.get("user"),
                kpts_opponent=entry.get("opponent"),
                frame_width=frame_width,
                frame_height=frame_height,
                prev=prev,
                dt_seconds=dt,
            )
            feats["frame"] = frame
            feats["timestamp_ms"] = int(timestamps.get(frame, 0))
            rows.append(feats)
            prev = feats

        df = pd.DataFrame(rows)

        # Bridge only brief dropouts; long occlusions stay None on purpose.
        interpolated_counts = {}
        for col in ("user_hip_height", "opponent_hip_height", "athlete_distance",
                    "relative_hip_height"):
            if col not in df.columns:
                continue
            before = int(df[col].isna().sum())
            df[col] = interpolate_short_gaps(df[col].tolist(), max_gap=INTERPOLATE_MAX_GAP)
            after = int(df[col].isna().sum())
            interpolated_counts[col] = before - after

        artifact_key = storage.object_key(match.user_id, match.id, "artifacts", "features.parquet")
        local_features = str(tmp_path / "features.parquet")
        df.to_parquet(local_features, index=False)
        storage.upload_from_path(local_features, artifact_key)

        both_visible = int(df["both_athletes_visible"].sum())
        return {
            "features_key": artifact_key,
            "feature_rows": len(df),
            "feature_columns": len(df.columns),
            "dt_seconds": round(dt, 4),
            "frames_both_visible": both_visible,
            # The honest quality signal: what fraction of analyzed frames had both
            # wrestlers posed well enough to compute relational features.
            "both_visible_ratio": round(both_visible / len(df), 4) if len(df) else 0.0,
            "interpolated_values": interpolated_counts,
            "mean_user_visibility": round(float(df["user_visibility"].mean()), 4),
            "mean_opponent_visibility": round(float(df["opponent_visibility"].mean()), 4),
        }
