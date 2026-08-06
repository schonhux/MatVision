"""
Layer 2 annotation endpoints: state segments, athlete identification, match
annotation metadata, and the enhanced event labeling/editing.
"""

import pytest


@pytest.fixture()
def match(signed_up_user, monkeypatch):
    """A match owned by the signed-up user, with storage mocked out."""
    client, headers = signed_up_user
    import app.storage as storage_module
    monkeypatch.setattr(storage_module, "ensure_bucket", lambda: None)
    monkeypatch.setattr(storage_module, "presigned_put_url", lambda *a, **kw: "https://fake/x")

    resp = client.post(
        "/matches",
        json={"title": "Semifinal", "filename": "m.mp4", "content_type": "video/mp4", "size_bytes": 1000},
        headers=headers,
    )
    return client, headers, resp.json()["match_id"]


# --- state segments -----------------------------------------------------------

def test_create_and_list_state_segments(match):
    client, headers, match_id = match

    resp = client.post(
        f"/matches/{match_id}/states",
        json={"state": "neutral", "start_ms": 0, "end_ms": 30000},
        headers=headers,
    )
    assert resp.status_code == 201, resp.text
    assert resp.json()["state"] == "neutral"
    assert resp.json()["source"] == "human"
    assert resp.json()["annotator_id"] is not None

    listed = client.get(f"/matches/{match_id}/states", headers=headers).json()
    assert len(listed) == 1


def test_state_segments_ordered_by_start(match):
    client, headers, match_id = match
    client.post(f"/matches/{match_id}/states",
                json={"state": "scramble", "start_ms": 30000, "end_ms": 40000}, headers=headers)
    client.post(f"/matches/{match_id}/states",
                json={"state": "neutral", "start_ms": 0, "end_ms": 30000}, headers=headers)

    states = client.get(f"/matches/{match_id}/states", headers=headers).json()
    assert [s["state"] for s in states] == ["neutral", "scramble"]


def test_top_and_bottom_require_controlling(match):
    """'top' without knowing who's on top is an unusable training label."""
    client, headers, match_id = match
    for state in ("top", "bottom"):
        resp = client.post(
            f"/matches/{match_id}/states",
            json={"state": state, "start_ms": 0, "end_ms": 1000},
            headers=headers,
        )
        assert resp.status_code == 422, f"{state} should require controlling"

    ok = client.post(
        f"/matches/{match_id}/states",
        json={"state": "top", "start_ms": 0, "end_ms": 1000, "controlling": "user"},
        headers=headers,
    )
    assert ok.status_code == 201


def test_overlapping_state_segments_rejected(match):
    """A wrestler can't be in two positions at once — overlapping ground truth
    would make Layer 4's training targets ambiguous.
    """
    client, headers, match_id = match
    client.post(f"/matches/{match_id}/states",
                json={"state": "neutral", "start_ms": 0, "end_ms": 30000}, headers=headers)

    overlapping = client.post(
        f"/matches/{match_id}/states",
        json={"state": "scramble", "start_ms": 20000, "end_ms": 40000},
        headers=headers,
    )
    assert overlapping.status_code == 409

    # Exactly adjacent (touching but not overlapping) is fine.
    adjacent = client.post(
        f"/matches/{match_id}/states",
        json={"state": "scramble", "start_ms": 30000, "end_ms": 40000},
        headers=headers,
    )
    assert adjacent.status_code == 201


def test_invalid_state_name_rejected(match):
    client, headers, match_id = match
    resp = client.post(
        f"/matches/{match_id}/states",
        json={"state": "flying", "start_ms": 0, "end_ms": 1000},
        headers=headers,
    )
    assert resp.status_code == 422


def test_update_state_segment_boundaries(match):
    client, headers, match_id = match
    created = client.post(
        f"/matches/{match_id}/states",
        json={"state": "neutral", "start_ms": 0, "end_ms": 30000},
        headers=headers,
    ).json()

    resp = client.patch(
        f"/matches/{match_id}/states/{created['id']}",
        json={"end_ms": 25000},
        headers=headers,
    )
    assert resp.status_code == 200
    assert resp.json()["end_ms"] == 25000
    assert resp.json()["start_ms"] == 0  # untouched


def test_update_state_segment_rejects_inverted_range(match):
    client, headers, match_id = match
    created = client.post(
        f"/matches/{match_id}/states",
        json={"state": "neutral", "start_ms": 10000, "end_ms": 30000},
        headers=headers,
    ).json()

    # Only start_ms sent, but the merged result would be invalid.
    resp = client.patch(
        f"/matches/{match_id}/states/{created['id']}",
        json={"start_ms": 40000},
        headers=headers,
    )
    assert resp.status_code == 422


def test_delete_state_segment(match):
    client, headers, match_id = match
    created = client.post(
        f"/matches/{match_id}/states",
        json={"state": "neutral", "start_ms": 0, "end_ms": 1000},
        headers=headers,
    ).json()

    assert client.delete(f"/matches/{match_id}/states/{created['id']}", headers=headers).status_code == 204
    assert client.get(f"/matches/{match_id}/states", headers=headers).json() == []


def test_state_segment_scoped_to_match(client, monkeypatch):
    import app.storage as storage_module
    monkeypatch.setattr(storage_module, "ensure_bucket", lambda: None)
    monkeypatch.setattr(storage_module, "presigned_put_url", lambda *a, **kw: "https://fake/x")

    r1 = client.post("/auth/signup", json={"email": "a@test.com", "password": "hunter2222"})
    r2 = client.post("/auth/signup", json={"email": "b@test.com", "password": "hunter2222"})
    ha = {"Authorization": f"Bearer {r1.json()['access_token']}"}
    hb = {"Authorization": f"Bearer {r2.json()['access_token']}"}

    mid = client.post("/matches", json={"title": "x", "filename": "m.mp4",
                                        "content_type": "video/mp4", "size_bytes": 1}, headers=ha).json()["match_id"]
    assert client.get(f"/matches/{mid}/states", headers=hb).status_code == 404


def _add_model_state(match_id, state="neutral", start_ms=0, end_ms=1000, confidence=0.8):
    from app.database import SessionLocal
    from app.models import MatchState, StateSegment

    db = SessionLocal()
    segment = StateSegment(
        match_id=match_id,
        state=MatchState(state),
        start_ms=start_ms,
        end_ms=end_ms,
        confidence=confidence,
        source="model:test-v1",
    )
    db.add(segment)
    db.commit()
    db.refresh(segment)
    segment_id = segment.id
    db.close()
    return segment_id


def test_state_source_filters_and_preferred_model(match):
    client, headers, match_id = match
    client.post(
        f"/matches/{match_id}/states",
        json={"state": "scramble", "start_ms": 0, "end_ms": 1000},
        headers=headers,
    )
    _add_model_state(match_id, state="neutral")

    human = client.get(f"/matches/{match_id}/states?source=human", headers=headers).json()
    model = client.get(f"/matches/{match_id}/states?source=model", headers=headers).json()
    preferred = client.get(f"/matches/{match_id}/states?source=preferred", headers=headers).json()
    assert [item["source"] for item in human] == ["human"]
    assert [item["source"] for item in model] == ["model:test-v1"]
    assert preferred == model


def test_state_summary_reports_duration_and_confidence(match):
    client, headers, match_id = match
    _add_model_state(match_id, state="neutral", start_ms=0, end_ms=3000, confidence=0.8)
    _add_model_state(match_id, state="scramble", start_ms=3000, end_ms=4000, confidence=0.4)

    summary = client.get(f"/matches/{match_id}/states/summary", headers=headers)
    assert summary.status_code == 200, summary.text
    body = summary.json()
    assert body["source"] == "model:test-v1"
    assert body["duration_ms_by_state"]["neutral"] == 3000
    assert body["percentage_by_state"]["scramble"] == 25.0
    assert body["mean_confidence"] == 0.6
    assert body["low_confidence_count"] == 1


def test_model_state_segments_are_read_only(match):
    client, headers, match_id = match
    segment_id = _add_model_state(match_id)
    patched = client.patch(
        f"/matches/{match_id}/states/{segment_id}",
        json={"end_ms": 2000},
        headers=headers,
    )
    deleted = client.delete(f"/matches/{match_id}/states/{segment_id}", headers=headers)
    assert patched.status_code == 409
    assert deleted.status_code == 409


def test_model_state_does_not_make_annotation_complete(match):
    client, headers, match_id = match
    client.put(
        f"/matches/{match_id}/athletes",
        json={"role": "user", "athlete_name": "Schon"},
        headers=headers,
    )
    _add_model_state(match_id)

    response = client.patch(
        f"/matches/{match_id}/annotation",
        json={"annotation_complete": True},
        headers=headers,
    )
    assert response.status_code == 422
    assert "state" in response.json()["detail"].lower()


# --- athlete identification ----------------------------------------------------

def test_set_and_list_athletes(match):
    client, headers, match_id = match

    resp = client.put(
        f"/matches/{match_id}/athletes",
        json={
            "role": "user",
            "athlete_name": "Schon Huxley",
            "singlet_color": "blue",
            "seed_frame_ms": 1500,
            "seed_bbox": {"x1": 0.3, "y1": 0.2, "x2": 0.5, "y2": 0.8},
        },
        headers=headers,
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["athlete_name"] == "Schon Huxley"
    assert resp.json()["seed_bbox"]["x1"] == 0.3

    listed = client.get(f"/matches/{match_id}/athletes", headers=headers).json()
    assert len(listed) == 1


def test_setting_same_role_twice_upserts(match):
    """One 'user' per match — re-identifying replaces rather than duplicating."""
    client, headers, match_id = match
    client.put(f"/matches/{match_id}/athletes",
               json={"role": "user", "athlete_name": "First"}, headers=headers)
    client.put(f"/matches/{match_id}/athletes",
               json={"role": "user", "athlete_name": "Corrected"}, headers=headers)

    listed = client.get(f"/matches/{match_id}/athletes", headers=headers).json()
    assert len(listed) == 1
    assert listed[0]["athlete_name"] == "Corrected"


def test_invalid_bbox_rejected(match):
    client, headers, match_id = match
    # x2 < x1 -> zero/negative area
    resp = client.put(
        f"/matches/{match_id}/athletes",
        json={"role": "user", "seed_bbox": {"x1": 0.8, "y1": 0.2, "x2": 0.3, "y2": 0.8}},
        headers=headers,
    )
    assert resp.status_code == 422

    # out of normalized range
    resp2 = client.put(
        f"/matches/{match_id}/athletes",
        json={"role": "user", "seed_bbox": {"x1": 0.1, "y1": 0.2, "x2": 1.5, "y2": 0.8}},
        headers=headers,
    )
    assert resp2.status_code == 422


def test_invalid_role_rejected(match):
    client, headers, match_id = match
    resp = client.put(f"/matches/{match_id}/athletes",
                      json={"role": "referee"}, headers=headers)
    assert resp.status_code == 422


# --- match annotation metadata --------------------------------------------------

def test_set_venue(match):
    client, headers, match_id = match
    resp = client.patch(f"/matches/{match_id}/annotation",
                        json={"venue": "Hilton Coliseum"}, headers=headers)
    assert resp.status_code == 200
    assert resp.json()["venue"] == "Hilton Coliseum"


def test_cannot_mark_complete_without_annotations(match):
    """Guards the dataset: an unlabeled match must not be able to enter training."""
    client, headers, match_id = match
    resp = client.patch(f"/matches/{match_id}/annotation",
                        json={"annotation_complete": True}, headers=headers)
    assert resp.status_code == 422
    assert "athlete" in resp.json()["detail"].lower()


def test_can_mark_complete_once_annotated(match):
    client, headers, match_id = match
    client.put(f"/matches/{match_id}/athletes",
               json={"role": "user", "athlete_name": "Schon"}, headers=headers)
    client.post(f"/matches/{match_id}/states",
                json={"state": "neutral", "start_ms": 0, "end_ms": 30000}, headers=headers)

    resp = client.patch(f"/matches/{match_id}/annotation",
                        json={"annotation_complete": True}, headers=headers)
    assert resp.status_code == 200, resp.text
    assert resp.json()["annotation_complete"] is True


# --- enhanced event labeling ------------------------------------------------------

def test_event_accepts_full_annotation_fields(match):
    client, headers, match_id = match
    resp = client.post(
        f"/matches/{match_id}/events",
        json={
            "type": "shot_attempt",
            "start_ms": 1000,
            "end_ms": 3000,
            "initiator": "user",
            "outcome": "successful",
            "state_before": "neutral",
            "state_after": "top",
            "opponent_response": "sprawl",
            "technique": "single_leg",
            "detail": {"setup": "wrist_control", "entry_distance": "close"},
        },
        headers=headers,
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["initiator"] == "user"
    assert body["outcome"] == "successful"
    assert body["technique"] == "single_leg"
    assert body["detail"]["setup"] == "wrist_control"
    assert body["annotator_id"] is not None


def test_event_rejects_invalid_enum_values(match):
    client, headers, match_id = match
    for bad in ({"initiator": "referee"}, {"outcome": "kinda_worked"}, {"state_before": "flying"}):
        resp = client.post(
            f"/matches/{match_id}/events",
            json={"type": "shot_attempt", "start_ms": 0, "end_ms": 1000, **bad},
            headers=headers,
        )
        assert resp.status_code == 422, f"{bad} should be rejected"


def test_update_event_boundaries_and_labels(match):
    client, headers, match_id = match
    created = client.post(
        f"/matches/{match_id}/events",
        json={"type": "shot_attempt", "start_ms": 1000, "end_ms": 3000},
        headers=headers,
    ).json()

    resp = client.patch(
        f"/matches/{match_id}/events/{created['id']}",
        json={"start_ms": 1200, "outcome": "failed", "technique": "double_leg"},
        headers=headers,
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["start_ms"] == 1200
    assert body["end_ms"] == 3000  # untouched
    assert body["outcome"] == "failed"
    assert body["technique"] == "double_leg"
    assert body["type"] == "shot_attempt"  # untouched


def test_update_event_rejects_inverted_merged_range(match):
    client, headers, match_id = match
    created = client.post(
        f"/matches/{match_id}/events",
        json={"type": "shot_attempt", "start_ms": 1000, "end_ms": 3000},
        headers=headers,
    ).json()

    resp = client.patch(
        f"/matches/{match_id}/events/{created['id']}",
        json={"start_ms": 5000},  # would make start > existing end
        headers=headers,
    )
    assert resp.status_code == 422


def test_update_event_from_other_match_rejected(match):
    client, headers, match_id = match
    other = client.post(
        "/matches",
        json={"title": "other", "filename": "m.mp4", "content_type": "video/mp4", "size_bytes": 1},
        headers=headers,
    ).json()["match_id"]

    created = client.post(
        f"/matches/{match_id}/events",
        json={"type": "shot_attempt", "start_ms": 1000, "end_ms": 3000},
        headers=headers,
    ).json()

    resp = client.patch(f"/matches/{other}/events/{created['id']}",
                        json={"outcome": "failed"}, headers=headers)
    assert resp.status_code == 404


def test_delete_event(match):
    client, headers, match_id = match
    created = client.post(
        f"/matches/{match_id}/events",
        json={"type": "shot_attempt", "start_ms": 1000, "end_ms": 3000},
        headers=headers,
    ).json()

    assert client.delete(f"/matches/{match_id}/events/{created['id']}", headers=headers).status_code == 204
    assert client.get(f"/matches/{match_id}/events", headers=headers).json() == []
