import json
from types import SimpleNamespace

import numpy as np
import pandas as pd

from ml.training.train_states import evaluate, train_lightgbm


def test_training_harness_uses_exported_splits(tmp_path):
    features_dir = tmp_path / "features"
    features_dir.mkdir()
    matches = []

    for split in ("train", "val", "test"):
        match_id = f"{split}-match"
        pd.DataFrame({
            "frame": range(40),
            "timestamp_ms": [index * 125 for index in range(40)],
            "bbox_distance": np.r_[np.full(20, 0.35), np.full(20, 0.04)],
            "bbox_overlap": np.r_[np.zeros(20), np.full(20, 0.5)],
        }).to_parquet(features_dir / f"{match_id}.parquet", index=False)
        matches.append({
            "match_id": match_id,
            "split": split,
            "state_segments": [
                {"state": "neutral", "start_ms": 0, "end_ms": 2500},
                {"state": "scramble", "start_ms": 2500, "end_ms": 5000},
            ],
        })

    dataset = tmp_path / "dataset.json"
    dataset.write_text(json.dumps({"matches": matches}))
    args = SimpleNamespace(
        dataset=dataset,
        features_dir=features_dir,
        output=tmp_path / "state.joblib",
        version="test-v1",
        radius=2,
    )

    model = train_lightgbm(args)
    metrics = evaluate(model, args, "test")
    assert args.output.exists()
    assert metrics["macro_f1"] >= 0.9
