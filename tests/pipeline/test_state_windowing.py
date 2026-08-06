import numpy as np
import pandas as pd
import pytest

from ml.states.windowing import (
    build_labeled_windows,
    build_sequences,
    build_windows,
    label_frames,
)


@pytest.fixture()
def features():
    return pd.DataFrame({
        "frame": range(6),
        "timestamp_ms": [0, 125, 250, 375, 500, 625],
        "distance": [0.4, 0.3, np.nan, 0.1, 0.2, 0.4],
        "visible": [True, True, False, True, True, True],
    })


def test_frame_labels_use_half_open_intervals(features):
    labels = label_frames(features, [
        {"state": "neutral", "start_ms": 0, "end_ms": 250},
        {"state": "scramble", "start_ms": 250, "end_ms": 500},
    ])
    assert labels.tolist() == ["neutral", "neutral", "scramble", "scramble", None, None]


def test_frame_labels_reject_overlap(features):
    with pytest.raises(ValueError, match="overlap"):
        label_frames(features, [
            {"state": "neutral", "start_ms": 0, "end_ms": 400},
            {"state": "top", "start_ms": 300, "end_ms": 600},
        ])


def test_windows_include_current_summary_and_missingness(features):
    windows = build_windows(features, radius_frames=1)
    assert windows.loc[2, "distance__current"] != windows.loc[2, "distance__current"]
    assert windows.loc[2, "distance__mean"] == pytest.approx(0.2)
    assert windows.loc[2, "distance__missing"] == pytest.approx(1 / 3)


def test_labeled_windows_drop_uncovered_frames(features):
    windows, labels, timestamps = build_labeled_windows(
        features, [{"state": "neutral", "start_ms": 125, "end_ms": 500}], radius_frames=1
    )
    assert len(windows) == 3
    assert labels.tolist() == ["neutral"] * 3
    assert timestamps.tolist() == [125, 250, 375]


def test_sequences_add_missing_value_channels(features):
    sequences, channels = build_sequences(features, sequence_length=3, columns=["distance"])
    assert sequences.shape == (6, 2, 3)
    assert channels == ["distance", "distance__missing"]
    assert sequences[2, 1, 1] == 1.0


def test_sequence_length_must_be_odd(features):
    with pytest.raises(ValueError, match="odd"):
        build_sequences(features, sequence_length=4)

