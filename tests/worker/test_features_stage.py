import shutil

import numpy as np
import pandas as pd


def test_features_stage_keeps_frames_when_pose_is_missing(merged_worker_app, tmp_path, monkeypatch):
    from app import models, storage
    from app.database import SessionLocal
    from app.stages import features

    pose_path = tmp_path / "poses.parquet"
    track_path = tmp_path / "tracks.parquet"
    keypoints = np.zeros((17, 3), dtype=float)
    keypoints[:, 0] = 300
    keypoints[:, 1] = 300
    keypoints[:, 2] = 0.9
    pd.DataFrame([
        {"frame": 0, "timestamp_ms": 0, "identity": "user", "keypoints": keypoints.flatten().tolist()},
        {"frame": 0, "timestamp_ms": 0, "identity": "opponent", "keypoints": keypoints.flatten().tolist()},
    ]).to_parquet(pose_path, index=False)
    pd.DataFrame([
        {"frame": 0, "timestamp_ms": 0, "identity": "user", "confidence": 0.9, "x1": 100, "y1": 100, "x2": 250, "y2": 500},
        {"frame": 0, "timestamp_ms": 0, "identity": "opponent", "confidence": 0.9, "x1": 500, "y1": 100, "x2": 650, "y2": 500},
        {"frame": 4, "timestamp_ms": 125, "identity": "user", "confidence": 0.9, "x1": 120, "y1": 100, "x2": 270, "y2": 500},
        {"frame": 4, "timestamp_ms": 125, "identity": "opponent", "confidence": 0.9, "x1": 480, "y1": 100, "x2": 630, "y2": 500},
    ]).to_parquet(track_path, index=False)

    files = {"poses.parquet": pose_path, "tracks.parquet": track_path}
    uploaded = {}
    monkeypatch.setattr(storage, "object_exists", lambda key: True)
    monkeypatch.setattr(
        storage,
        "download_to_path",
        lambda key, path: shutil.copy(files["poses.parquet" if key.endswith("poses.parquet") else "tracks.parquet"], path),
    )
    monkeypatch.setattr(storage, "upload_from_path", lambda path, key: uploaded.setdefault(key, shutil.copy(path, tmp_path / "output.parquet")))

    db = SessionLocal()
    match = models.Match(id="m1", user_id="u1", title="Match")
    db.add(match)
    db.add(models.Job(
        match_id="m1",
        stage="detect_track",
        status=models.JobStageStatus.COMPLETE,
        artifacts={"frame_width": 1280, "frame_height": 720},
    ))
    db.commit()

    result = features.run(match, db)
    output = pd.read_parquet(next(iter(uploaded.values())))
    assert len(output) == 2
    assert bool(output.loc[1, "both_athletes_visible"]) is False
    assert output.loc[1, "bbox_distance"] > 0
    assert result["frames_with_bbox_fallback"] == 1
