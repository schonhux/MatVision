"""
Dataset export — the payoff of Layer 2. These tests verify the export is
self-describing, correctly split, and (critically) that it refuses to ship a
dataset with leakage.
"""



def _mock_storage(monkeypatch):
    import app.storage as storage_module
    monkeypatch.setattr(storage_module, "ensure_bucket", lambda: None)
    monkeypatch.setattr(storage_module, "presigned_put_url", lambda *a, **kw: "https://fake/x")


def _make_annotated_match(client, headers, title, athlete_name, venue=None, mark_complete=True):
    """Creates a match with the minimum annotation needed to be exportable."""
    match_id = client.post(
        "/matches",
        json={"title": title, "filename": "m.mp4", "content_type": "video/mp4", "size_bytes": 1000},
        headers=headers,
    ).json()["match_id"]

    client.put(f"/matches/{match_id}/athletes",
               json={"role": "user", "athlete_name": athlete_name}, headers=headers)
    client.post(f"/matches/{match_id}/states",
                json={"state": "neutral", "start_ms": 0, "end_ms": 30000}, headers=headers)
    client.post(f"/matches/{match_id}/events",
                json={"type": "shot_attempt", "start_ms": 5000, "end_ms": 7000,
                      "initiator": "user", "outcome": "successful"},
                headers=headers)

    if venue:
        client.patch(f"/matches/{match_id}/annotation", json={"venue": venue}, headers=headers)
    if mark_complete:
        client.patch(f"/matches/{match_id}/annotation",
                     json={"annotation_complete": True}, headers=headers)
    return match_id


def test_export_with_no_matches_returns_404(signed_up_user, monkeypatch):
    client, headers = signed_up_user
    _mock_storage(monkeypatch)
    assert client.get("/datasets/export", headers=headers).status_code == 404


def test_export_excludes_incomplete_matches_by_default(signed_up_user, monkeypatch):
    client, headers = signed_up_user
    _mock_storage(monkeypatch)
    _make_annotated_match(client, headers, "done", "Schon", mark_complete=True)
    _make_annotated_match(client, headers, "wip", "Other", mark_complete=False)

    body = client.get("/datasets/export", headers=headers).json()
    assert body["summary"]["match_count"] == 1
    assert body["matches"][0]["title"] == "done"

    # ...but can be included explicitly.
    all_body = client.get("/datasets/export?only_complete=false", headers=headers).json()
    assert all_body["summary"]["match_count"] == 2


def test_export_is_self_describing(signed_up_user, monkeypatch):
    client, headers = signed_up_user
    _mock_storage(monkeypatch)
    _make_annotated_match(client, headers, "m1", "Schon", venue="Hilton")

    body = client.get("/datasets/export", headers=headers).json()
    assert body["schema_version"] == "1.1"
    assert "exported_at" in body
    assert body["split_config"]["grouped_by"] == ["match_id", "athlete", "venue"]
    assert body["split_config"]["leakage_verified"] is True
    assert body["summary"]["event_count"] == 1
    assert body["summary"]["state_segment_count"] == 1
    assert body["summary"]["correction_count"] == 0


def test_export_contains_full_annotation_payload(signed_up_user, monkeypatch):
    client, headers = signed_up_user
    _mock_storage(monkeypatch)
    match_id = _make_annotated_match(client, headers, "m1", "Schon Huxley", venue="Hilton")

    body = client.get("/datasets/export", headers=headers).json()
    m = body["matches"][0]
    assert m["match_id"] == match_id
    assert m["venue"] == "Hilton"
    assert m["split"] in ("train", "val", "test")
    assert m["athletes"][0]["name"] == "Schon Huxley"
    assert m["state_segments"][0]["state"] == "neutral"
    assert m["events"][0]["initiator"] == "user"
    assert m["events"][0]["outcome"] == "successful"


def test_export_includes_model_corrections(signed_up_user, monkeypatch):
    client, headers = signed_up_user
    _mock_storage(monkeypatch)
    match_id = _make_annotated_match(client, headers, "m1", "Schon")

    from app.database import SessionLocal
    from app.models import Event
    db = SessionLocal()
    event = Event(
        match_id=match_id,
        type="shot_attempt",
        start_ms=8000,
        end_ms=9500,
        source="model:rules-v1",
        review_status="unreviewed",
    )
    db.add(event)
    db.commit()
    event_id = event.id
    db.close()

    client.patch(
        f"/matches/{match_id}/events/{event_id}",
        json={"type": "defended_shot", "outcome": "failed"},
        headers=headers,
    )
    body = client.get("/datasets/export", headers=headers).json()
    model_event = next(event for event in body["matches"][0]["events"] if event["source"].startswith("model:"))
    assert model_event["review_status"] == "corrected"
    assert {item["field"] for item in model_event["corrections"]} == {"type", "outcome"}
    assert body["summary"]["correction_count"] == 2


def test_export_split_is_leakage_free_for_shared_athlete(signed_up_user, monkeypatch):
    """The whole point: several matches with the same wrestler must all land in the
    same split, and the export must confirm it verified this.
    """
    client, headers = signed_up_user
    _mock_storage(monkeypatch)
    for i in range(6):
        _make_annotated_match(client, headers, f"m{i}", "Schon Huxley", venue=f"gym_{i}")

    body = client.get("/datasets/export", headers=headers).json()
    splits = {m["split"] for m in body["matches"]}
    assert len(splits) == 1, f"same athlete was split across {splits}"


def test_export_split_is_deterministic(signed_up_user, monkeypatch):
    client, headers = signed_up_user
    _mock_storage(monkeypatch)
    for i in range(5):
        _make_annotated_match(client, headers, f"m{i}", f"Athlete {i}", venue=f"gym_{i}")

    first = client.get("/datasets/export?seed=7", headers=headers).json()
    second = client.get("/datasets/export?seed=7", headers=headers).json()
    assert {m["match_id"]: m["split"] for m in first["matches"]} == \
           {m["match_id"]: m["split"] for m in second["matches"]}


def test_export_rejects_invalid_ratios(signed_up_user, monkeypatch):
    client, headers = signed_up_user
    _mock_storage(monkeypatch)
    _make_annotated_match(client, headers, "m1", "Schon")

    resp = client.get("/datasets/export?train_ratio=0.9&val_ratio=0.2", headers=headers)
    assert resp.status_code == 422


def test_export_is_scoped_to_owner(client, monkeypatch):
    _mock_storage(monkeypatch)
    r1 = client.post("/auth/signup", json={"email": "a@test.com", "password": "hunter2222"})
    r2 = client.post("/auth/signup", json={"email": "b@test.com", "password": "hunter2222"})
    ha = {"Authorization": f"Bearer {r1.json()['access_token']}"}
    hb = {"Authorization": f"Bearer {r2.json()['access_token']}"}

    _make_annotated_match(client, ha, "a-match", "Athlete A")
    assert client.get("/datasets/export", headers=ha).json()["summary"]["match_count"] == 1
    assert client.get("/datasets/export", headers=hb).status_code == 404


# --- dataset stats -------------------------------------------------------------

def test_dataset_stats_tracks_progress(signed_up_user, monkeypatch):
    client, headers = signed_up_user
    _mock_storage(monkeypatch)
    _make_annotated_match(client, headers, "m1", "Schon")

    stats = client.get("/datasets/stats", headers=headers).json()
    assert stats["total_matches"] == 1
    assert stats["annotated_matches"] == 1
    assert stats["total_events"] == 1
    assert stats["total_state_segments"] == 1
    assert stats["total_corrections"] == 0
    assert stats["events_by_type"]["shot_attempt"] == 1
    assert stats["states_by_type"]["neutral"] == 1
    assert stats["labeled_minutes"] == 0.5  # 30000ms


def test_dataset_stats_m2_gate(signed_up_user, monkeypatch):
    """BUILD_PLAN.md M2 gate is 5 labeled matches — surfaced so progress is visible."""
    client, headers = signed_up_user
    _mock_storage(monkeypatch)

    stats = client.get("/datasets/stats", headers=headers).json()
    assert stats["m2_gate"] == {"target_matches": 5, "current": 0, "met": False}

    for i in range(5):
        _make_annotated_match(client, headers, f"m{i}", f"Athlete {i}")

    stats2 = client.get("/datasets/stats", headers=headers).json()
    assert stats2["m2_gate"]["current"] == 5
    assert stats2["m2_gate"]["met"] is True
