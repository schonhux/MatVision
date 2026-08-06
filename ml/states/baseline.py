from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ml.states import STATE_LABELS
from ml.states.windowing import build_windows


@dataclass
class LightGBMStateModel:
    estimator: object
    feature_columns: list[str]
    version: str
    radius_frames: int = 4

    @classmethod
    def fit(
        cls,
        features: pd.DataFrame,
        labels: pd.Series,
        version: str,
        radius_frames: int = 4,
    ) -> LightGBMStateModel:
        from lightgbm import LGBMClassifier

        encoded = np.array([STATE_LABELS.index(value) for value in labels], dtype=np.int64)
        if len(np.unique(encoded)) < 2:
            raise ValueError("State training requires at least two classes")

        model = LGBMClassifier(
            n_estimators=250,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.85,
            colsample_bytree=0.85,
            class_weight="balanced",
            random_state=42,
            verbosity=-1,
        )
        model.fit(features, encoded)
        return cls(model, list(features.columns), version, radius_frames)

    def predict_proba(self, raw_features: pd.DataFrame) -> np.ndarray:
        windows = build_windows(raw_features, radius_frames=self.radius_frames)
        aligned = windows.reindex(columns=self.feature_columns, fill_value=np.nan)
        probabilities = self.estimator.predict_proba(aligned)

        result = np.zeros((len(aligned), len(STATE_LABELS)), dtype=float)
        for index, class_id in enumerate(self.estimator.classes_):
            result[:, int(class_id)] = probabilities[:, index]
        return result

    def save(self, path: str | Path) -> None:
        import joblib

        joblib.dump(
            {
                "kind": "lightgbm_state",
                "version": self.version,
                "radius_frames": self.radius_frames,
                "feature_columns": self.feature_columns,
                "estimator": self.estimator,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> LightGBMStateModel:
        import joblib

        bundle = joblib.load(path)
        if bundle.get("kind") != "lightgbm_state":
            raise ValueError("Not a MatVision LightGBM state model")
        return cls(
            estimator=bundle["estimator"],
            feature_columns=bundle["feature_columns"],
            version=bundle["version"],
            radius_frames=int(bundle.get("radius_frames", 4)),
        )


class BBoxStateFallback:
    version = "bbox-fallback-v1"

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        return np.vstack([self._row_probabilities(row) for _, row in features.iterrows()])

    def _row_probabilities(self, row: pd.Series) -> np.ndarray:
        scores = dict.fromkeys(STATE_LABELS, 0.04)
        user_seen = bool(row.get("user_bbox_detected", False))
        opponent_seen = bool(row.get("opponent_bbox_detected", False))

        if not user_seen and not opponent_seen:
            scores["stopped"] = 0.64
            return _normalize(scores)
        if not user_seen or not opponent_seen:
            scores["scramble"] = 0.52
            return _normalize(scores)

        distance = _number(row.get("bbox_distance"))
        overlap = _number(row.get("bbox_overlap"))
        hip_gap = _number(row.get("relative_hip_height"))
        vertical_gap = _number(row.get("bbox_vertical_gap"))
        visibility = min(
            _number(row.get("user_visibility"), 0.0),
            _number(row.get("opponent_visibility"), 0.0),
        )

        if distance > 0.18 and overlap < 0.08:
            scores["neutral"] = 0.70
        elif visibility < 0.30 or overlap > 0.30:
            scores["scramble"] = 0.58
        else:
            position_signal = hip_gap if abs(hip_gap) >= 0.035 else -vertical_gap
            if position_signal > 0.035:
                scores["top"] = 0.58
            elif position_signal < -0.035:
                scores["bottom"] = 0.58
            else:
                scores["scramble"] = 0.48
        return _normalize(scores)


def _number(value, default: float = 0.0) -> float:
    try:
        number = float(value)
        return number if np.isfinite(number) else default
    except (TypeError, ValueError):
        return default


def _normalize(scores: dict[str, float]) -> np.ndarray:
    values = np.array([scores[label] for label in STATE_LABELS], dtype=float)
    return values / values.sum()
