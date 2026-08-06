import numpy as np
import pandas as pd

from ml.states.baseline import LightGBMStateModel
from ml.states.windowing import build_labeled_windows


def test_lightgbm_state_model_round_trip(tmp_path):
    raw = pd.DataFrame({
        "frame": range(40),
        "timestamp_ms": [index * 125 for index in range(40)],
        "distance": np.r_[np.full(20, 0.35), np.full(20, 0.05)],
        "overlap": np.r_[np.zeros(20), np.full(20, 0.5)],
    })
    segments = [
        {"state": "neutral", "start_ms": 0, "end_ms": 2500},
        {"state": "scramble", "start_ms": 2500, "end_ms": 5000},
    ]
    windows, labels, _ = build_labeled_windows(raw, segments, radius_frames=2)
    model = LightGBMStateModel.fit(windows, labels, "test-v1", radius_frames=2)

    before = model.predict_proba(raw)
    path = tmp_path / "state.joblib"
    model.save(path)
    loaded = LightGBMStateModel.load(path)
    after = loaded.predict_proba(raw)

    assert before.shape == (40, 5)
    assert np.allclose(before, after)
    assert loaded.version == "test-v1"
