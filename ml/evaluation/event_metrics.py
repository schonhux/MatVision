from __future__ import annotations

from collections import defaultdict
from statistics import median


def evaluate_events(truth: list[dict], predictions: list[dict]) -> dict:
    classes = sorted({item["type"] for item in truth + predictions})
    per_class = {}
    start_errors = []
    total_tp = total_fp = total_fn = 0

    for event_type in classes:
        expected = [item for item in truth if item["type"] == event_type]
        predicted = [item for item in predictions if item["type"] == event_type]
        matches = _match_events(expected, predicted)
        tp = len(matches)
        fp = len(predicted) - tp
        fn = len(expected) - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[event_type] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": len(expected),
        }
        start_errors.extend(abs(expected[a]["start_ms"] - predicted[b]["start_ms"]) for a, b in matches)
        total_tp += tp
        total_fp += fp
        total_fn += fn

    duplicate_count = _duplicate_count(predictions)
    return {
        "macro_f1": round(sum(item["f1"] for item in per_class.values()) / len(per_class), 4)
        if per_class else 0.0,
        "per_class": per_class,
        "true_positives": total_tp,
        "false_positives": total_fp,
        "false_negatives": total_fn,
        "median_start_error_ms": int(median(start_errors)) if start_errors else None,
        "duplicate_rate": round(duplicate_count / len(predictions), 4) if predictions else 0.0,
    }


def _match_events(expected: list[dict], predicted: list[dict]) -> list[tuple[int, int]]:
    candidates = []
    for truth_index, truth in enumerate(expected):
        for pred_index, pred in enumerate(predicted):
            if truth.get("match_id", "__all__") != pred.get("match_id", "__all__"):
                continue
            overlap = _temporal_iou(truth, pred)
            start_gap = abs(truth["start_ms"] - pred["start_ms"])
            if overlap >= 0.2 or start_gap <= 1500:
                candidates.append((max(overlap, 1 - start_gap / 3000), truth_index, pred_index))

    matched_truth = set()
    matched_predictions = set()
    matches = []
    for _, truth_index, pred_index in sorted(candidates, reverse=True):
        if truth_index in matched_truth or pred_index in matched_predictions:
            continue
        matched_truth.add(truth_index)
        matched_predictions.add(pred_index)
        matches.append((truth_index, pred_index))
    return matches


def _temporal_iou(left: dict, right: dict) -> float:
    overlap = max(0, min(left["end_ms"], right["end_ms"]) - max(left["start_ms"], right["start_ms"]))
    union = max(left["end_ms"], right["end_ms"]) - min(left["start_ms"], right["start_ms"])
    return overlap / union if union else 0.0


def _duplicate_count(events: list[dict]) -> int:
    by_type = defaultdict(list)
    duplicates = 0
    for event in sorted(events, key=lambda item: item["start_ms"]):
        prior = by_type[event["type"]]
        if any(_temporal_iou(event, other) >= 0.5 for other in prior):
            duplicates += 1
        prior.append(event)
    return duplicates
