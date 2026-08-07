from __future__ import annotations

from dataclasses import asdict, dataclass, field
from itertools import pairwise

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EventCandidate:
    type: str
    start_ms: int
    peak_ms: int
    end_ms: int
    initiator: str | None
    outcome: str | None
    confidence: float
    state_before: str | None
    state_after: str | None
    measurements: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> EventCandidate:
        return cls(**value)


def detect_events(features: pd.DataFrame, state_segments: list[dict]) -> list[EventCandidate]:
    if features.empty or not state_segments:
        return []

    features = features.sort_values("timestamp_ms").reset_index(drop=True)
    segments = sorted(state_segments, key=lambda item: item["start_ms"])
    candidates = _transition_events(segments)
    shots = _shot_attempts(features, segments)

    takedowns = [candidate for candidate in candidates if candidate.type == "takedown"]
    for takedown in takedowns:
        nearby = any(
            shot.initiator == takedown.initiator
            and 0 <= takedown.peak_ms - shot.peak_ms <= 3000
            for shot in shots
        )
        if not nearby:
            shots.append(EventCandidate(
                type="shot_attempt",
                start_ms=max(0, takedown.peak_ms - 1500),
                peak_ms=max(0, takedown.peak_ms - 500),
                end_ms=takedown.peak_ms,
                initiator=takedown.initiator,
                outcome="successful",
                confidence=round(takedown.confidence * 0.82, 4),
                state_before=takedown.state_before,
                state_after=takedown.state_after,
                measurements={"inferred_from_takedown": 1.0},
            ))

    candidates.extend(shots)
    candidates.extend(_defended_shots(shots, takedowns, segments))
    return sorted(candidates, key=lambda item: (item.start_ms, item.type))


def _transition_events(segments: list[dict]) -> list[EventCandidate]:
    events = []
    for before, after in pairwise(segments):
        old_state = _state_value(before["state"])
        new_state = _state_value(after["state"])
        transition_ms = int(after["start_ms"])
        confidence = _transition_confidence(before, after)

        if old_state in {"neutral", "scramble"} and new_state in {"top", "bottom"}:
            controlling = after.get("controlling")
            events.append(EventCandidate(
                type="takedown",
                start_ms=max(int(before["start_ms"]), transition_ms - 1800),
                peak_ms=transition_ms,
                end_ms=min(int(after["end_ms"]), transition_ms + 750),
                initiator=controlling,
                outcome="successful",
                confidence=confidence,
                state_before=old_state,
                state_after=new_state,
            ))
        elif old_state in {"top", "bottom"} and new_state == "neutral":
            controlling = before.get("controlling")
            escaping = "opponent" if controlling == "user" else "user" if controlling == "opponent" else None
            events.append(EventCandidate(
                type="escape",
                start_ms=max(int(before["start_ms"]), transition_ms - 1200),
                peak_ms=transition_ms,
                end_ms=min(int(after["end_ms"]), transition_ms + 500),
                initiator=escaping,
                outcome="successful",
                confidence=confidence,
                state_before=old_state,
                state_after=new_state,
            ))
        elif old_state == "stopped" and new_state != "stopped":
            events.append(EventCandidate(
                type="restart",
                start_ms=max(0, transition_ms - 300),
                peak_ms=transition_ms,
                end_ms=min(int(after["end_ms"]), transition_ms + 700),
                initiator=None,
                outcome=None,
                confidence=confidence,
                state_before=old_state,
                state_after=new_state,
            ))
    return events


def _shot_attempts(features: pd.DataFrame, segments: list[dict]) -> list[EventCandidate]:
    timestamps = features["timestamp_ms"].astype(int).to_numpy()
    opponent_drop = _level_change(features.get("opponent_hip_height"), timestamps)
    scored = []

    for index, row in features.iterrows():
        timestamp = int(row["timestamp_ms"])
        state = _segment_at(segments, timestamp)
        if state is None or _state_value(state["state"]) not in {"neutral", "scramble"}:
            continue

        closing = max(_number(row.get("closing_speed")), _number(row.get("bbox_closing_speed")))
        user_drop = _number(row.get("user_level_change_rate"))
        other_drop = opponent_drop[index]
        drop = max(user_drop, other_drop)
        score = min(1.0, max(0.0, closing) / 0.35) * 0.55
        score += min(1.0, max(0.0, drop) / 0.20) * 0.45
        if score < 0.52 or closing < 0.08:
            continue

        initiator = "user" if user_drop >= other_drop else "opponent"
        scored.append((index, score, initiator, closing, drop, state))

    shots = []
    for group in _contiguous_groups(scored):
        index, score, initiator, closing, drop, state = max(group, key=lambda item: item[1])
        peak_ms = int(timestamps[index])
        shots.append(EventCandidate(
            type="shot_attempt",
            start_ms=max(0, peak_ms - 500),
            peak_ms=peak_ms,
            end_ms=min(int(state["end_ms"]), peak_ms + 1000),
            initiator=initiator,
            outcome=None,
            confidence=round(min(0.95, 0.35 + score * 0.6), 4),
            state_before=_state_value(state["state"]),
            state_after=_state_value((_segment_at(segments, peak_ms + 1000) or state)["state"]),
            measurements={
                "closing_speed": round(closing, 4),
                "level_change_rate": round(drop, 4),
            },
        ))
    return shots


def _defended_shots(
    shots: list[EventCandidate],
    takedowns: list[EventCandidate],
    segments: list[dict],
) -> list[EventCandidate]:
    defended = []
    for shot in shots:
        converted = any(
            takedown.initiator == shot.initiator
            and 0 <= takedown.peak_ms - shot.peak_ms <= 3500
            for takedown in takedowns
        )
        if converted:
            continue
        final_state = _segment_at(segments, shot.peak_ms + 2500)
        state_after = _state_value(final_state["state"]) if final_state else shot.state_after
        if state_after not in {"neutral", "scramble", "stopped"}:
            continue
        defended.append(EventCandidate(
            type="defended_shot",
            start_ms=shot.start_ms,
            peak_ms=shot.peak_ms,
            end_ms=min(int(segments[-1]["end_ms"]), shot.peak_ms + 2000),
            initiator=shot.initiator,
            outcome="failed",
            confidence=round(shot.confidence * 0.88, 4),
            state_before=shot.state_before,
            state_after=state_after,
            measurements=shot.measurements,
        ))
    return defended


def _level_change(series: pd.Series | None, timestamps: np.ndarray) -> np.ndarray:
    result = np.zeros(len(timestamps), dtype=float)
    if series is None or len(series) != len(timestamps):
        return result
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    for index in range(1, len(values)):
        dt = (timestamps[index] - timestamps[index - 1]) / 1000
        if dt > 0 and np.isfinite(values[index]) and np.isfinite(values[index - 1]):
            result[index] = (values[index - 1] - values[index]) / dt
    return result


def _contiguous_groups(values: list[tuple]) -> list[list[tuple]]:
    if not values:
        return []
    groups = [[values[0]]]
    for value in values[1:]:
        if value[0] - groups[-1][-1][0] <= 2:
            groups[-1].append(value)
        else:
            groups.append([value])
    return groups


def _segment_at(segments: list[dict], timestamp_ms: int) -> dict | None:
    return next(
        (segment for segment in segments if segment["start_ms"] <= timestamp_ms < segment["end_ms"]),
        None,
    )


def _transition_confidence(before: dict, after: dict) -> float:
    values = [value for value in (before.get("confidence"), after.get("confidence")) if value is not None]
    return round(float(np.mean(values)) if values else 0.65, 4)


def _state_value(value) -> str:
    return value.value if hasattr(value, "value") else str(value)


def _number(value, default: float = 0.0) -> float:
    try:
        number = float(value)
        return number if np.isfinite(number) else default
    except (TypeError, ValueError):
        return default
