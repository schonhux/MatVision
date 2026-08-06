import shutil

import pandas as pd


def test_states_stage_replaces_model_predictions_but_keeps_human_labels(
    merged_worker_app, tmp_path, monkeypatch
):
    from app import models, storage
    from app.config import settings
    from app.database import SessionLocal
    from app.stages import states

    feature_path = tmp_path / "features.parquet"
    pd.DataFrame({
        "frame": range(8),
        "timestamp_ms": [i * 125 for i in range(8)],
        "user_bbox_detected": [True] * 8,
        "opponent_bbox_detected": [True] * 8,
        "bbox_distance": [0.3] * 8,
        "bbox_overlap": [0.0] * 8,
        "bbox_vertical_gap": [0.0] * 8,
        "relative_hip_height": [None] * 8,
        "user_visibility": [0.0] * 8,
        "opponent_visibility": [0.0] * 8,
    }).to_parquet(feature_path, index=False)

    monkeypatch.setattr(storage, "object_exists", lambda key: True)
    monkeypatch.setattr(storage, "download_to_path", lambda key, path: shutil.copy(feature_path, path))
    monkeypatch.setattr(settings, "state_model_path", str(tmp_path / "missing.joblib"))

    db = SessionLocal()
    match = models.Match(id="m1", user_id="u1", title="Match", duration_seconds=1)
    db.add(match)
    db.add(models.StateSegment(
        match_id="m1", state=models.MatchState.NEUTRAL, start_ms=0, end_ms=100,
        source="human",
    ))
    db.add(models.StateSegment(
        match_id="m1", state=models.MatchState.SCRAMBLE, start_ms=0, end_ms=1000,
        source="model:old",
    ))
    db.commit()

    result = states.run(match, db)
    states.run(match, db)

    human = db.query(models.StateSegment).filter(models.StateSegment.source == "human").all()
    predicted = db.query(models.StateSegment).filter(models.StateSegment.source.like("model:%")).all()
    assert len(human) == 1
    assert len(predicted) == result["segment_count"]
    assert all(segment.source == "model:bbox-fallback-v1" for segment in predicted)
    assert result["used_fallback"] is True

