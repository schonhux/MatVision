import subprocess
import tempfile
from pathlib import Path

from sqlalchemy.orm import Session

from app.models import Match, Event
from app.stages.base import StageError
from app import storage


def cut_clip_for_event(match_id: str, event_id: str, db: Session) -> str:
    match = db.get(Match, match_id)
    event = db.get(Event, event_id)
    if match is None or event is None:
        raise StageError("Match or event not found")

    source_key = match.video_keys.get("analysis_720p") or match.video_keys.get("original")
    if not source_key:
        raise StageError("No processed video available to cut a clip from yet")

    start_s = max(0, event.start_ms / 1000 - 1.0)
    duration_s = (event.end_ms - event.start_ms) / 1000 + 2.0

    with tempfile.TemporaryDirectory() as tmp:
        local_input = str(Path(tmp) / "source.mp4")
        local_output = str(Path(tmp) / "clip.mp4")
        storage.download_to_path(source_key, local_input)

        cmd = [
            "ffmpeg", "-y", "-ss", f"{start_s:.2f}", "-i", local_input,
            "-t", f"{duration_s:.2f}", "-c", "copy", "-movflags", "+faststart",
            local_output,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise StageError(f"ffmpeg clip cut failed: {result.stderr.strip()[-500:]}")

        clip_key = storage.object_key(match.user_id, match.id, "clips", f"event-{event.id}.mp4")
        storage.upload_from_path(local_output, clip_key, content_type="video/mp4")

        event.clip_key = clip_key
        db.commit()
        return clip_key
