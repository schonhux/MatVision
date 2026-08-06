"""
Worker tests import against a MERGED app package — mirroring exactly what
worker/Dockerfile produces at build time (api/app's database/models/config/storage
+ worker/app's stages/tasks, minus the API-only modules). See dev/DECISIONS.md
ADR-009. We build that merge into a tmp dir here rather than trusting a
pre-merged directory exists, so these tests catch a broken merge the same way
`docker build` would.
"""

import shutil
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture()
def merged_worker_app(tmp_path, monkeypatch):
    merge_dir = tmp_path / "worker_merged"
    app_dir = merge_dir / "app"
    shutil.copytree(REPO_ROOT / "api" / "app", app_dir)
    # worker/app's stages/ + tasks.py overlay onto the copied api/app, exactly as
    # the two `COPY` instructions in worker/Dockerfile do.
    shutil.copytree(REPO_ROOT / "worker" / "app" / "stages", app_dir / "stages", dirs_exist_ok=True)
    shutil.copy(REPO_ROOT / "worker" / "app" / "tasks.py", app_dir / "tasks.py")
    # API-only modules the worker image deliberately excludes (Dockerfile `rm -rf`).
    for name in ["routers", "main.py", "security.py", "schemas.py", "deps.py", "queue.py"]:
        target = app_dir / name
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()

    db_path = tmp_path / "worker_test.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")

    for mod in list(sys.modules):
        if mod == "app" or mod.startswith("app."):
            del sys.modules[mod]
    sys.path.insert(0, str(merge_dir))

    from app.database import Base, engine
    from app import models
    Base.metadata.create_all(bind=engine)

    yield

    sys.path.remove(str(merge_dir))
