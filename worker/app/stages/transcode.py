import subprocess
import tempfile
from pathlib import Path

from sqlalchemy.orm import Session

from app.models import Match
from app.stages.base import StageError
from app import storage


def run(match: Match, db: Session) -> dict:
    original_key = match.video_keys.get("original")
    if not original_key:
        raise StageError("No uploaded video found for this match")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        local_input = str(tmp_path / "input")
        local_output = str(tmp_path / "analysis-720p.mp4")
        thumb_path = str(tmp_path / "thumbnail.jpg")

        storage.download_to_path(original_key, local_input)

        transcode_cmd = [
            "ffmpeg", "-y", "-i", local_input,
            "-vf", "scale=-2:720,fps=30",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart",
            local_output,
        ]
        result = subprocess.run(transcode_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise StageError(f"ffmpeg transcode failed: {result.stderr.strip()[-500:]}")

        thumb_cmd = [
            "ffmpeg", "-y", "-ss", "1", "-i", local_output,
            "-frames:v", "1", "-q:v", "3", thumb_path,
        ]
        thumb_result = subprocess.run(thumb_cmd, capture_output=True, text=True)
        if thumb_result.returncode != 0:
            # Non-fatal: a missing thumbnail shouldn't fail the whole stage.
            thumb_path = None

        analysis_key = storage.object_key(match.user_id, match.id, "processed", "analysis-720p.mp4")
        storage.upload_from_path(local_output, analysis_key, content_type="video/mp4")

        keys = dict(match.video_keys)
        keys["analysis_720p"] = analysis_key

        if thumb_path:
            thumb_key = storage.object_key(match.user_id, match.id, "thumbnails", "000.jpg")
            storage.upload_from_path(thumb_path, thumb_key, content_type="image/jpeg")
            keys["thumbnail"] = thumb_key

        match.video_keys = keys
        db.commit()

        return {
            "analysis_key": analysis_key,
            "thumbnail_key": keys.get("thumbnail"),
            "output_size_bytes": Path(local_output).stat().st_size,
        }
