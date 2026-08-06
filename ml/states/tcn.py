from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from ml.states import STATE_LABELS
from ml.states.windowing import build_sequences


class TemporalBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.norm = nn.BatchNorm1d(channels)

    def forward(self, values):
        return self.norm(values + self.block(values))


class StateTCN(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int = 64, dropout: float = 0.15):
        super().__init__()
        self.input = nn.Conv1d(input_channels, hidden_channels, 1)
        self.temporal = nn.Sequential(
            TemporalBlock(hidden_channels, 3, 1, dropout),
            TemporalBlock(hidden_channels, 3, 2, dropout),
            TemporalBlock(hidden_channels, 3, 4, dropout),
        )
        self.output = nn.Linear(hidden_channels, len(STATE_LABELS))

    def forward(self, values):
        encoded = self.temporal(self.input(values))
        center = encoded[:, :, encoded.shape[-1] // 2]
        return self.output(center)


class TCNStateModel:
    def __init__(self, model, feature_columns: list[str], version: str, sequence_length: int = 17):
        self.model = model
        self.feature_columns = feature_columns
        self.version = version
        self.sequence_length = sequence_length

    def predict_proba(self, raw_features) -> np.ndarray:
        sequences, _ = build_sequences(
            raw_features,
            sequence_length=self.sequence_length,
            columns=self.feature_columns,
        )
        device = next(self.model.parameters()).device
        self.model.eval()
        with torch.no_grad():
            logits = self.model(torch.from_numpy(sequences).to(device))
            return torch.softmax(logits, dim=1).cpu().numpy()

    def save(self, path: str | Path) -> None:
        torch.save({
            "kind": "tcn_state",
            "version": self.version,
            "sequence_length": self.sequence_length,
            "feature_columns": self.feature_columns,
            "state_dict": self.model.state_dict(),
            "input_channels": len(self.feature_columns) * 2,
        }, path)

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> TCNStateModel:
        bundle = torch.load(path, map_location=device, weights_only=True)
        if bundle.get("kind") != "tcn_state":
            raise ValueError("Not a MatVision TCN state model")
        model = StateTCN(bundle["input_channels"])
        model.load_state_dict(bundle["state_dict"])
        model.to(device)
        return cls(
            model,
            bundle["feature_columns"],
            bundle["version"],
            int(bundle.get("sequence_length", 17)),
        )

