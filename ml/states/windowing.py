from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from ml.states import STATE_LABELS

META_COLUMNS = {"frame", "timestamp_ms", "state", "split", "match_id"}


def numeric_feature_columns(features: pd.DataFrame) -> list[str]:
    columns = []
    for name in features.columns:
        if name in META_COLUMNS:
            continue
        if pd.api.types.is_numeric_dtype(features[name]) or pd.api.types.is_bool_dtype(features[name]):
            columns.append(name)
    return columns


def label_frames(features: pd.DataFrame, segments: Iterable[dict]) -> pd.Series:
    timestamps = features["timestamp_ms"].to_numpy(dtype=np.int64)
    labels = np.full(len(features), None, dtype=object)

    for segment in sorted(segments, key=lambda item: item["start_ms"]):
        state = segment["state"]
        if state not in STATE_LABELS:
            raise ValueError(f"Unknown state label: {state}")
        start_ms = int(segment["start_ms"])
        end_ms = int(segment["end_ms"])
        if end_ms <= start_ms:
            raise ValueError("State segment end must be after its start")
        mask = (timestamps >= start_ms) & (timestamps < end_ms)
        if np.any(pd.notna(labels[mask])):
            raise ValueError("State segments overlap")
        labels[mask] = state

    return pd.Series(labels, index=features.index, name="state", dtype=object)


def build_windows(
    features: pd.DataFrame,
    radius_frames: int = 4,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    if radius_frames < 0:
        raise ValueError("radius_frames must be non-negative")
    if "timestamp_ms" not in features:
        raise ValueError("features must include timestamp_ms")

    source_columns = columns or numeric_feature_columns(features)
    window_size = radius_frames * 2 + 1
    output: dict[str, pd.Series] = {}

    for name in source_columns:
        values = pd.to_numeric(features[name], errors="coerce").astype(float)
        rolling = values.rolling(window_size, center=True, min_periods=1)
        output[f"{name}__current"] = values
        output[f"{name}__mean"] = rolling.mean()
        output[f"{name}__std"] = rolling.std(ddof=0)
        output[f"{name}__min"] = rolling.min()
        output[f"{name}__max"] = rolling.max()
        output[f"{name}__missing"] = values.isna().astype(float).rolling(
            window_size, center=True, min_periods=1
        ).mean()

    return pd.DataFrame(output, index=features.index)


def build_labeled_windows(
    features: pd.DataFrame,
    segments: Iterable[dict],
    radius_frames: int = 4,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    labels = label_frames(features, segments)
    keep = labels.notna()
    windows = build_windows(features, radius_frames=radius_frames)
    return (
        windows.loc[keep].reset_index(drop=True),
        labels.loc[keep].reset_index(drop=True),
        features.loc[keep, "timestamp_ms"].reset_index(drop=True),
    )


def build_sequences(
    features: pd.DataFrame,
    sequence_length: int = 17,
    columns: list[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    if sequence_length < 1 or sequence_length % 2 == 0:
        raise ValueError("sequence_length must be a positive odd number")

    names = columns or numeric_feature_columns(features)
    values = features[names].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    missing = np.isnan(values).astype(np.float32)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    combined = np.concatenate([values, missing], axis=1)

    radius = sequence_length // 2
    padded = np.pad(combined, ((radius, radius), (0, 0)), mode="edge")
    sequences = np.stack([padded[i:i + sequence_length].T for i in range(len(features))])
    channel_names = names + [f"{name}__missing" for name in names]
    return sequences, channel_names
