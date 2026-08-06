from __future__ import annotations

from collections import defaultdict
from itertools import pairwise

import numpy as np

from ml.states import STATE_LABELS


def evaluate_state_predictions(
    truth: list[str],
    predicted: list[str],
    timestamps_ms: list[int] | None = None,
    match_ids: list[str] | None = None,
) -> dict:
    if len(truth) != len(predicted):
        raise ValueError("truth and predicted must have equal lengths")
    if not truth:
        raise ValueError("at least one prediction is required")

    labels = list(STATE_LABELS)
    index = {label: i for i, label in enumerate(labels)}
    confusion = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for actual, guess in zip(truth, predicted):
        if actual not in index or guess not in index:
            raise ValueError(f"Unknown state label: {actual if actual not in index else guess}")
        confusion[index[actual], index[guess]] += 1

    per_class = {}
    f1_values = []
    for label, i in index.items():
        tp = int(confusion[i, i])
        fp = int(confusion[:, i].sum() - tp)
        fn = int(confusion[i, :].sum() - tp)
        support = int(confusion[i, :].sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support,
        }
        if support:
            f1_values.append(f1)

    groups = match_ids or ["match"] * len(truth)
    if len(groups) != len(truth):
        raise ValueError("match_ids must have the same length as predictions")
    times = timestamps_ms or list(range(len(truth)))
    if len(times) != len(truth):
        raise ValueError("timestamps_ms must have the same length as predictions")

    transition = _transition_metrics(truth, predicted, times, groups)
    return {
        "accuracy": round(float(np.trace(confusion) / confusion.sum()), 4),
        "macro_f1": round(float(np.mean(f1_values)) if f1_values else 0.0, 4),
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
        **transition,
    }


def _transition_metrics(
    truth: list[str],
    predicted: list[str],
    timestamps_ms: list[int],
    match_ids: list[str],
) -> dict:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, match_id in enumerate(match_ids):
        grouped[match_id].append(index)

    comparisons = 0
    correct = 0
    boundary_errors = []
    unmatched = 0

    for indices in grouped.values():
        actual_transitions = _transitions(truth, timestamps_ms, indices)
        predicted_transitions = _transitions(predicted, timestamps_ms, indices)

        for left, right in pairwise(indices):
            actual_changed = truth[left] != truth[right]
            predicted_changed = predicted[left] != predicted[right]
            comparisons += 1
            correct += int(actual_changed == predicted_changed)

        remaining = list(predicted_transitions)
        for actual in actual_transitions:
            candidates = [
                (abs(actual[2] - item[2]), pos, item)
                for pos, item in enumerate(remaining)
                if item[:2] == actual[:2]
            ]
            if not candidates:
                unmatched += 1
                continue
            error, pos, _ = min(candidates, key=lambda item: item[0])
            boundary_errors.append(error)
            remaining.pop(pos)
        unmatched += len(remaining)

    return {
        "transition_accuracy": round(correct / comparisons, 4) if comparisons else 1.0,
        "median_boundary_error_ms": (
            round(float(np.median(boundary_errors)), 1) if boundary_errors else None
        ),
        "unmatched_transitions": unmatched,
    }


def _transitions(
    labels: list[str], timestamps_ms: list[int], indices: list[int]
) -> list[tuple[str, str, int]]:
    result = []
    for left, right in pairwise(indices):
        if labels[left] != labels[right]:
            result.append((labels[left], labels[right], int(timestamps_ms[right])))
    return result
