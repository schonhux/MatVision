from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ml.states import STATE_LABELS


@dataclass(frozen=True)
class PredictedSegment:
    state: str
    start_ms: int
    end_ms: int
    confidence: float
    controlling: str | None


def predict_frames(model, features: pd.DataFrame) -> tuple[list[str], np.ndarray]:
    probabilities = np.asarray(model.predict_proba(features), dtype=float)
    if probabilities.shape != (len(features), len(STATE_LABELS)):
        raise ValueError("State model returned an invalid probability matrix")
    class_ids = probabilities.argmax(axis=1)
    labels = [STATE_LABELS[index] for index in class_ids]
    confidence = probabilities[np.arange(len(probabilities)), class_ids]
    return smooth_short_runs(labels, confidence), confidence


def smooth_short_runs(
    labels: list[str],
    confidence: np.ndarray,
    min_frames: int = 3,
) -> list[str]:
    smoothed = list(labels)
    if len(smoothed) < 3 or min_frames <= 1:
        return smoothed

    runs = _runs(smoothed)
    for index, (start, end, label) in enumerate(runs):
        if end - start >= min_frames or index == 0 or index == len(runs) - 1:
            continue
        left = runs[index - 1]
        right = runs[index + 1]
        if left[2] == right[2]:
            smoothed[start:end] = [left[2]] * (end - start)
            continue
        left_conf = float(np.mean(confidence[left[0]:left[1]]))
        right_conf = float(np.mean(confidence[right[0]:right[1]]))
        replacement = left[2] if left_conf >= right_conf else right[2]
        smoothed[start:end] = [replacement] * (end - start)
    return smoothed


def predictions_to_segments(
    timestamps_ms: list[int] | np.ndarray,
    labels: list[str],
    confidence: np.ndarray,
    duration_ms: int | None = None,
) -> list[PredictedSegment]:
    timestamps = np.asarray(timestamps_ms, dtype=np.int64)
    if len(timestamps) != len(labels) or len(labels) != len(confidence):
        raise ValueError("timestamps, labels, and confidence must have equal lengths")
    if len(timestamps) == 0:
        return []
    if np.any(np.diff(timestamps) < 0):
        raise ValueError("timestamps must be sorted")

    if len(timestamps) > 1:
        step = max(1, int(np.median(np.diff(timestamps))))
    else:
        step = 125
    final_end = max(int(timestamps[-1]) + step, int(duration_ms or 0))

    segments = []
    for start, end, label in _runs(labels):
        start_ms = int(timestamps[start])
        end_ms = int(timestamps[end]) if end < len(timestamps) else final_end
        controlling = "user" if label == "top" else "opponent" if label == "bottom" else None
        segments.append(PredictedSegment(
            state=label,
            start_ms=start_ms,
            end_ms=end_ms,
            confidence=round(float(np.mean(confidence[start:end])), 4),
            controlling=controlling,
        ))
    return segments


def _runs(labels: list[str]) -> list[tuple[int, int, str]]:
    if not labels:
        return []
    runs = []
    start = 0
    for index in range(1, len(labels)):
        if labels[index] != labels[start]:
            runs.append((start, index, labels[start]))
            start = index
    runs.append((start, len(labels), labels[start]))
    return runs

