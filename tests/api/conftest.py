"""
Shared pytest fixtures for the API test suite.

IMPORTANT: the test DB lives outside the repo folder (a plain tmp_path, not a
relative path under the project directory) — see dev/DECISIONS.md ADR-010.
SQLite over a synced/bridged folder can throw spurious "disk I/O error"s because
those bridges don't reliably support the file locking SQLite needs; a real
Postgres in CI/Docker doesn't have this problem, but local ad hoc test runs
should still point DATABASE_URL outside any cloud-synced project folder.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "api"))

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")

    # Reload settings-dependent modules fresh so the env vars above take effect
    # even if another test already imported the app with different settings.
    for mod in list(sys.modules):
        if mod.startswith("app."):
            del sys.modules[mod]
    if "app" in sys.modules:
        del sys.modules["app"]

    from app.database import Base, engine
    from app import models  # noqa: F401 — registers tables on Base.metadata
    Base.metadata.create_all(bind=engine)

    from app.main import app
    return TestClient(app)


@pytest.fixture()
def signed_up_user(client):
    """Returns (client, auth_headers) for a fresh, signed-up user."""
    resp = client.post("/auth/signup", json={"email": "wrestler@test.com", "password": "hunter2222"})
    assert resp.status_code == 201, resp.text
    token = resp.json()["access_token"]
    return client, {"Authorization": f"Bearer {token}"}
