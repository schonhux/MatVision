import json
import shutil
from pathlib import Path

import pandas as pd


def test_events_and_consolidate_stages_preserve_reviewed_predictions(
    merged_worker_app, tmp_path, monkeypatch
):
    from app import models, storage
    from app.database import SessionLocal
    from app.stages import consolidate, events

    features_path = tmp_path / "features.parquet"
    candidate_path = tmp_path / "event_candidates.json"
    pd.DataFrame({
        "timestamp_ms": list(range(0, 5000, 125)),
        "closing_speed": [0.0] * 40,
        "bbox_closing_speed": [0.0] * 40,
    }).to_parquet(features_path, index=False)

    monkeypatch.setattr(storage, "object_exists", lambda key: True)

    def download(key, path):
        source = candidate_path if key.endswith("event_candidates.json") else features_path
        shutil.copy(source, path)

    def upload(path, key, content_type=None):
        if key.endswith("event_candidates.json"):
            candidate_path.write_bytes(Path(path).read_bytes())

    monkeypatch.setattr(storage, "download_to_path", download)
    monkeypatch.setattr(storage, "upload_from_path", upload)

    db = SessionLocal()
    match = models.Match(id="m1", user_id="u1", title="Match", duration_seconds=5)
    db.add(match)
    db.add_all([
        models.StateSegment(
            match_id="m1", state=models.MatchState.NEUTRAL, start_ms=0, end_ms=2000,
            source="model:test", confidence=0.8,
        ),
        models.StateSegment(
            match_id="m1", state=models.MatchState.TOP, start_ms=2000, end_ms=5000,
            controlling="user", source="model:test", confidence=0.9,
        ),
        models.Event(
            match_id="m1", type="restart", start_ms=50, peak_ms=100, end_ms=500,
            source="model:old", review_status="confirmed", state_before="stopped",
            state_after="neutral",
        ),
        models.Event(
            match_id="m1", type="escape", start_ms=500, peak_ms=800, end_ms=1200,
            source="model:old", review_status="unreviewed", state_before="top",
            state_after="neutral",
        ),
    ])
    db.commit()

    event_result = events.run(match, db)
    first = consolidate.run(match, db)
    second = consolidate.run(match, db)

    predicted = db.query(models.Event).filter(models.Event.source.like("model:%")).all()
    assert event_result["candidate_count"] >= 2
    assert first["created_count"] >= 1
    assert second["created_count"] == first["created_count"]
    assert sum(event.review_status == "confirmed" for event in predicted) == 1
    assert not any(event.type == "escape" and event.source == "model:old" for event in predicted)
    assert json.loads(candidate_path.read_text())


def test_clips_stage_cuts_each_unrejected_model_event(merged_worker_app, monkeypatch):
    from app import models, storage
    from app.database import SessionLocal
    from app.stages import clips

    db = SessionLocal()
    match = models.Match(id="m1", user_id="u1", title="Match")
    db.add(match)
    db.add_all([
        models.Event(
            id="e1", match_id="m1", type="takedown", start_ms=100, end_ms=500,
            source="model:rules-v1", review_status="unreviewed",
        ),
        models.Event(
            id="e2", match_id="m1", type="escape", start_ms=600, end_ms=900,
            source="model:rules-v1", review_status="rejected",
        ),
    ])
    db.commit()

    cut = []
    monkeypatch.setattr(storage, "object_exists", lambda key: False)
    monkeypatch.setattr(clips, "cut_clip_for_event", lambda match_id, event_id, db: cut.append(event_id))
    result = clips.run(match, db)

    assert cut == ["e1"]
    assert result["clips_created"] == 1
