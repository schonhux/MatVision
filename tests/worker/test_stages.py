"""
Real ffmpeg tests for VALIDATE and TRANSCODE — see BUILD_PLAN.md Layer 1 'how we
verify': 'FFmpeg transcode unit test on a 10s clip.' Storage (boto3/MinIO) is
mocked to read/write local files instead, so this tests the actual subprocess
logic without needing a live MinIO.
"""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture()
def sample_video(tmp_path):
    """A tiny real H.264 video generated with ffmpeg's test source — no binary
    fixture file committed to the repo, and it's disposable/regenerable.
    """
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is not installed")

    path = tmp_path / "sample.mp4"
    result = subprocess.run(
        [
            "ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=duration=3:size=640x360:rate=25",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
            "-c:v", "libx264", "-c:a", "aac", "-shortest", str(path),
        ],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    return path


@pytest.fixture()
def fake_storage(monkeypatch, tmp_path, sample_video):
    """Mocks app.storage's download/upload to operate on local files, and seeds
    the fake 'original' object with our synthetic video.
    """
    import app.storage as storage_module

    uploaded = {}
    original_key = "users/u1/matches/m1/original/video.mp4"
    files = {original_key: sample_video}

    def download_to_path(key, local_path):
        shutil.copy(files[key], local_path)

    def upload_from_path(local_path, key, content_type=None):
        dest = tmp_path / f"uploaded_{key.replace('/', '_')}"
        shutil.copy(local_path, dest)
        files[key] = dest
        uploaded[key] = dest

    monkeypatch.setattr(storage_module, "download_to_path", download_to_path)
    monkeypatch.setattr(storage_module, "upload_from_path", upload_from_path)

    return {"original_key": original_key, "files": files, "uploaded": uploaded}


def test_validate_stage_reads_real_video_metadata(merged_worker_app, fake_storage):
    from app.database import SessionLocal
    from app import models
    from app.stages import validate

    db = SessionLocal()
    match = models.Match(id="m1", user_id="u1", title="T", video_keys={"original": fake_storage["original_key"]})
    db.add(match)
    db.commit()

    result = validate.run(match, db)

    assert result["width"] == 640
    assert result["height"] == 360
    assert 2.9 <= result["duration_seconds"] <= 3.1
    assert match.duration_seconds is not None


def test_validate_stage_rejects_video_exceeding_duration_limit(merged_worker_app, fake_storage, monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "max_duration_seconds", 1)  # our sample is ~3s

    from app.database import SessionLocal
    from app import models
    from app.stages import validate
    from app.stages.base import StageError

    db = SessionLocal()
    match = models.Match(id="m1", user_id="u1", title="T", video_keys={"original": fake_storage["original_key"]})
    db.add(match)
    db.commit()

    with pytest.raises(StageError, match="longer than"):
        validate.run(match, db)


def test_transcode_stage_produces_720p_output_and_thumbnail(merged_worker_app, fake_storage):
    from app.database import SessionLocal
    from app import models
    from app.stages import transcode

    db = SessionLocal()
    match = models.Match(id="m1", user_id="u1", title="T", video_keys={"original": fake_storage["original_key"]})
    db.add(match)
    db.commit()

    result = transcode.run(match, db)

    assert "analysis_720p" in match.video_keys
    assert "thumbnail" in match.video_keys
    assert result["output_size_bytes"] > 0

    output_path = fake_storage["uploaded"][match.video_keys["analysis_720p"]]
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,r_frame_rate", "-of", "csv=p=0", str(output_path)],
        capture_output=True, text=True,
    )
    assert probe.returncode == 0
    width, height, _fps = probe.stdout.strip().split(",")
    assert int(height) == 720, f"expected 720p output, got {width}x{height}"


def test_transcode_stage_fails_cleanly_on_missing_source(merged_worker_app, fake_storage):
    from app.database import SessionLocal
    from app import models
    from app.stages import transcode
    from app.stages.base import StageError

    db = SessionLocal()
    match = models.Match(id="m1", user_id="u1", title="T", video_keys={})  # no "original" key at all
    db.add(match)
    db.commit()

    with pytest.raises(StageError):
        transcode.run(match, db)
