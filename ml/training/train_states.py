from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ml.evaluation.state_metrics import evaluate_state_predictions
from ml.states import STATE_LABELS
from ml.states.baseline import LightGBMStateModel
from ml.states.inference import predict_frames
from ml.states.windowing import (
    build_labeled_windows,
    build_sequences,
    label_frames,
    numeric_feature_columns,
)


def load_split(dataset_path: Path, features_dir: Path, split: str, radius_frames: int = 4):
    dataset = json.loads(dataset_path.read_text())
    windows, labels, timestamps, match_ids = [], [], [], []
    raw_by_match = []

    for match in dataset["matches"]:
        if match["split"] != split:
            continue
        feature_path = features_dir / f"{match['match_id']}.parquet"
        if not feature_path.exists():
            raise FileNotFoundError(f"Missing features for {match['match_id']}: {feature_path}")
        raw = pd.read_parquet(feature_path).sort_values("timestamp_ms").reset_index(drop=True)
        x, y, t = build_labeled_windows(raw, match["state_segments"], radius_frames)
        windows.append(x)
        labels.append(y)
        timestamps.extend(t.astype(int).tolist())
        match_ids.extend([match["match_id"]] * len(y))
        raw_by_match.append((match["match_id"], raw, match["state_segments"]))

    if not windows:
        raise ValueError(f"No labeled matches found in the {split} split")
    return pd.concat(windows, ignore_index=True), pd.concat(labels, ignore_index=True), timestamps, match_ids, raw_by_match


def train_lightgbm(args):
    train_x, train_y, _, _, _ = load_split(args.dataset, args.features_dir, "train", args.radius)
    model = LightGBMStateModel.fit(train_x, train_y, args.version, args.radius)
    model.save(args.output)
    return model


def train_tcn(args):
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    from ml.states.tcn import StateTCN, TCNStateModel

    _, _, _, _, matches = load_split(args.dataset, args.features_dir, "train", args.radius)
    feature_columns = numeric_feature_columns(matches[0][1])
    all_x, all_y = [], []
    for _, raw, segments in matches:
        sequences, _ = build_sequences(raw, args.sequence_length, feature_columns)
        labels = label_frames(raw, segments)
        keep = labels.notna().to_numpy()
        all_x.append(sequences[keep])
        all_y.extend(STATE_LABELS.index(value) for value in labels[keep])

    x = torch.from_numpy(np.concatenate(all_x)).float()
    y = torch.tensor(all_y, dtype=torch.long)
    loader = DataLoader(TensorDataset(x, y), batch_size=args.batch_size, shuffle=True)
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    network = StateTCN(x.shape[1]).to(device)
    optimizer = torch.optim.AdamW(network.parameters(), lr=args.learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    network.train()
    for _ in range(args.epochs):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            loss = loss_fn(network(batch_x.to(device)), batch_y.to(device))
            loss.backward()
            optimizer.step()

    model = TCNStateModel(network, feature_columns, args.version, args.sequence_length)
    model.save(args.output)
    return model


def evaluate(model, args, split: str) -> dict:
    _, _, _, _, matches = load_split(args.dataset, args.features_dir, split, args.radius)
    truth, guessed, timestamps, match_ids = [], [], [], []
    for match_id, raw, segments in matches:
        frame_truth = label_frames(raw, segments)
        keep = frame_truth.notna().to_numpy()
        labels, _ = predict_frames(model, raw)
        truth.extend(frame_truth[keep].tolist())
        guessed.extend(np.asarray(labels)[keep].tolist())
        timestamps.extend(raw.loc[keep, "timestamp_ms"].astype(int).tolist())
        match_ids.extend([match_id] * int(keep.sum()))
    return evaluate_state_predictions(truth, guessed, timestamps, match_ids)


def log_run(args, metrics: dict) -> None:
    import mlflow

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment("matvision-state-classifier")
    with mlflow.start_run(run_name=args.version):
        mlflow.log_params({
            "model": args.model,
            "version": args.version,
            "radius_frames": args.radius,
            "sequence_length": args.sequence_length,
        })
        flat = {
            f"{split}.{name}": value
            for split, split_metrics in metrics.items()
            for name, value in split_metrics.items()
            if isinstance(value, (int, float)) and value is not None
        }
        mlflow.log_metrics(flat)
        mlflow.log_artifact(str(args.output))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the MatVision match-state classifier")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", choices=("lightgbm", "tcn"), default="lightgbm")
    parser.add_argument("--version", default=f"state-{datetime.now(timezone.utc):%Y%m%d}")
    parser.add_argument("--radius", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=17)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--tracking-uri", default="mlruns")
    parser.add_argument("--skip-mlflow", action="store_true")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    model = train_lightgbm(args) if args.model == "lightgbm" else train_tcn(args)
    metrics = {split: evaluate(model, args, split) for split in ("val", "test")}
    metrics_path = args.output.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    if not args.skip_mlflow:
        log_run(args, metrics)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

