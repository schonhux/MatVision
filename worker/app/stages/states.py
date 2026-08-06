from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
from sqlalchemy.orm import Session

from app import storage
from app.config import settings
from app.models import Match, MatchState, StateSegment
from app.stages.base import StageError


def run(match: Match, db: Session) -> dict:
    from ml.states.baseline import BBoxStateFallback, LightGBMStateModel
    from ml.states.inference import predict_frames, predictions_to_segments

    features_key = storage.object_key(match.user_id, match.id, "artifacts", "features.parquet")
    if not storage.object_exists(features_key):
        raise StageError("features.parquet missing - the features stage must run first")

    with tempfile.TemporaryDirectory() as tmp:
        local_features = str(Path(tmp) / "features.parquet")
        storage.download_to_path(features_key, local_features)
        features = pd.read_parquet(local_features).sort_values("timestamp_ms").reset_index(drop=True)

    if features.empty:
        raise StageError("No feature rows available for state classification")

    model_path = Path(settings.state_model_path)
    if model_path.exists():
        try:
            model = _load_model(model_path, LightGBMStateModel)
        except Exception as exc:
            raise StageError(f"Could not load state model: {exc}") from exc
    else:
        model = BBoxStateFallback()

    labels, confidence = predict_frames(model, features)
    duration_ms = int(match.duration_seconds * 1000) if match.duration_seconds else None
    segments = predictions_to_segments(
        features["timestamp_ms"].astype(int).tolist(),
        labels,
        confidence,
        duration_ms=duration_ms,
    )
    if not segments:
        raise StageError("State classification produced no segments")

    source = f"model:{model.version}"
    db.query(StateSegment).filter(
        StateSegment.match_id == match.id,
        StateSegment.source.like("model:%"),
    ).delete(synchronize_session=False)
    db.add_all([
        StateSegment(
            match_id=match.id,
            state=MatchState(segment.state),
            start_ms=segment.start_ms,
            end_ms=segment.end_ms,
            controlling=segment.controlling,
            confidence=segment.confidence,
            source=source,
        )
        for segment in segments
    ])
    db.commit()

    low_confidence = [
        {"start_ms": segment.start_ms, "end_ms": segment.end_ms, "confidence": segment.confidence}
        for segment in segments
        if segment.confidence < settings.state_confidence_threshold
    ]
    durations: dict[str, int] = {}
    for segment in segments:
        durations[segment.state] = durations.get(segment.state, 0) + segment.end_ms - segment.start_ms

    return {
        "model_version": model.version,
        "source": source,
        "used_fallback": isinstance(model, BBoxStateFallback),
        "segment_count": len(segments),
        "mean_confidence": round(float(confidence.mean()), 4),
        "low_confidence_intervals": low_confidence,
        "duration_ms_by_state": durations,
    }


def _load_model(path: Path, lightgbm_model):
    if path.suffix == ".pt":
        from app.vision.models import get_device
        from ml.states.tcn import TCNStateModel

        return TCNStateModel.load(path, device=get_device())
    return lightgbm_model.load(path)

