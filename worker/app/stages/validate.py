import json
import subprocess
import tempfile
from pathlib import Path

from sqlalchemy.orm import Session

from app.config import settings
from app.models import Match
from app.stages.base import StageError
from app import storage


def probe(path: str) -> dict:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-print_format", "json",
            "-show_format", "-show_streams", path,
        ],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise StageError(f"Could not read video file: {result.stderr.strip()[:300]}")
    return json.loads(result.stdout)


def run(match: Match, db: Session) -> dict:
    original_key = match.video_keys.get("original")
    if not original_key:
        raise StageError("No uploaded video found for this match")

    with tempfile.TemporaryDirectory() as tmp:
        local_path = str(Path(tmp) / "input")
        storage.download_to_path(original_key, local_path)

        info = probe(local_path)
        video_streams = [s for s in info.get("streams", []) if s.get("codec_type") == "video"]
        if not video_streams:
            raise StageError("File has no video stream. Is this actually a video?")

        duration = float(info.get("format", {}).get("duration", 0))
        if duration <= 0:
            raise StageError("Could not determine video duration")
        if duration > settings.max_duration_seconds:
            raise StageError(
                f"Video is {duration:.0f}s, longer than the {settings.max_duration_seconds}s limit "
                "for uploads"
            )

        stream = video_streams[0]
        width, height = stream.get("width"), stream.get("height")

        match.duration_seconds = duration
        db.commit()

        return {
            "duration_seconds": duration,
            "width": width,
            "height": height,
            "codec": stream.get("codec_name"),
        }
