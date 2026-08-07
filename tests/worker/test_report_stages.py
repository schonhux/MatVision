import pytest


def _seed_match(models, db, match_id="m1"):
    match = models.Match(id=match_id, user_id="u1", title="Match", coach_tone="hard")
    db.add(match)
    db.add_all([
        models.StateSegment(
            match_id=match_id, state=models.MatchState.NEUTRAL, start_ms=0, end_ms=1000,
            source="model:test", confidence=0.9,
        ),
        models.StateSegment(
            match_id=match_id, state=models.MatchState.BOTTOM, start_ms=1000, end_ms=6000,
            controlling="opponent", source="model:test", confidence=0.85,
        ),
        models.Event(
            id="e-shot-1", match_id=match_id, type="shot_attempt", start_ms=200, end_ms=700,
            source="model:rules-v1", review_status="unreviewed", initiator="opponent",
        ),
        models.Event(
            id="e-td-1", match_id=match_id, type="takedown", start_ms=900, peak_ms=1000, end_ms=1100,
            source="model:rules-v1", review_status="unreviewed", initiator="opponent",
        ),
    ])
    db.commit()
    return match


def test_stats_stage_computes_and_caches_stats(merged_worker_app):
    from app import models
    from app.database import SessionLocal
    from app.stages import stats

    db = SessionLocal()
    match = _seed_match(models, db)

    result = stats.run(match, db)

    assert result["event_count"] == 2
    assert match.stats_summary["total_duration_ms"] == 6000
    assert match.stats_summary["by_athlete"]["opponent"]["takedowns"] == 1

    # Persisted, not just returned — a fresh query sees the same cached value.
    reloaded = db.get(models.Match, match.id)
    assert reloaded.stats_summary["by_athlete"]["opponent"]["shot_attempts"] == 1


def test_observations_stage_requires_stats_first(merged_worker_app):
    from app import models
    from app.database import SessionLocal
    from app.stages import observations
    from app.stages.base import StageError

    db = SessionLocal()
    match = _seed_match(models, db)

    with pytest.raises(StageError, match="stats stage"):
        observations.run(match, db)


def test_observations_stage_persists_and_replaces_model_observations(merged_worker_app):
    from app import models
    from app.database import SessionLocal
    from app.stages import observations, stats

    db = SessionLocal()
    match = _seed_match(match_id="m2", models=models, db=db)
    stats.run(match, db)

    first = observations.run(match, db)
    second = observations.run(match, db)

    rows = db.query(models.Observation).filter(models.Observation.match_id == match.id).all()
    assert len(rows) == first["observation_count"] == second["observation_count"]
    assert all(row.source == "model:rules-v1" for row in rows)


def test_report_stage_raises_clear_error_without_api_key(merged_worker_app, monkeypatch):
    from app import models
    from app.config import settings
    from app.database import SessionLocal
    from app.stages import observations, report, stats
    from app.stages.base import StageError

    monkeypatch.setattr(settings, "anthropic_api_key", None)

    db = SessionLocal()
    match = _seed_match(models=models, db=db, match_id="m3")
    stats.run(match, db)
    observations.run(match, db)

    with pytest.raises(StageError, match="ANTHROPIC_API_KEY"):
        report.run(match, db)


def test_report_stage_validates_and_persists_report(merged_worker_app, monkeypatch):
    from app import models
    from app.config import settings
    from app.database import SessionLocal
    from app.stages import observations, report, stats

    monkeypatch.setattr(settings, "anthropic_api_key", "test-key")

    db = SessionLocal()
    match = _seed_match(models=models, db=db, match_id="m4")
    stats.run(match, db)
    observations.run(match, db)

    def fake_generate(evidence, tone, api_key, model=None):
        assert tone == "hard"
        assert api_key == "test-key"
        real_id = evidence["events"][0]["id"]
        return {
            "summary": "Rough night on top control.",
            "statements": [
                {"text": "Grounded.", "kind": "observation", "evidence_event_ids": [real_id]},
                {"text": "Fabricated.", "kind": "interpretation", "evidence_event_ids": ["fake-id"]},
            ],
            "priority": {"text": "Tighten sprawl.", "evidence_event_ids": [real_id]},
        }

    import ml.reporting.llm as llm_module
    monkeypatch.setattr(llm_module, "generate_report_content", fake_generate)

    result = report.run(match, db)

    assert result["statement_count"] == 1
    assert result["dropped_statement_count"] == 1
    assert result["has_priority"] is True
    assert result["coach_tone"] == "hard"

    stored = db.query(models.Report).filter(models.Report.match_id == match.id).one()
    assert stored.coach_tone == "hard"
    assert len(stored.content["statements"]) == 1
    assert stored.content["priority"]["text"] == "Tighten sprawl."


def test_report_stage_upserts_a_single_row_per_match(merged_worker_app, monkeypatch):
    from app import models
    from app.config import settings
    from app.database import SessionLocal
    from app.stages import observations, report, stats

    monkeypatch.setattr(settings, "anthropic_api_key", "test-key")

    def fake_generate(evidence, tone, api_key, model=None):
        real_id = evidence["events"][0]["id"]
        return {
            "summary": "s",
            "statements": [{"text": "t", "kind": "observation", "evidence_event_ids": [real_id]}],
            "priority": None,
        }

    import ml.reporting.llm as llm_module
    monkeypatch.setattr(llm_module, "generate_report_content", fake_generate)

    db = SessionLocal()
    match = _seed_match(models=models, db=db, match_id="m5")
    stats.run(match, db)
    observations.run(match, db)

    report.run(match, db)
    report.run(match, db)

    rows = db.query(models.Report).filter(models.Report.match_id == match.id).all()
    assert len(rows) == 1
