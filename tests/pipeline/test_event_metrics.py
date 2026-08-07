from ml.evaluation.event_metrics import evaluate_events


def test_event_metrics_match_by_type_and_time():
    truth = [
        {"type": "shot_attempt", "start_ms": 1000, "end_ms": 2400},
        {"type": "takedown", "start_ms": 2500, "end_ms": 3800},
    ]
    predictions = [
        {"type": "shot_attempt", "start_ms": 1200, "end_ms": 2500},
        {"type": "takedown", "start_ms": 2900, "end_ms": 3900},
        {"type": "restart", "start_ms": 7000, "end_ms": 7600},
    ]
    metrics = evaluate_events(truth, predictions)
    assert metrics["true_positives"] == 2
    assert metrics["false_positives"] == 1
    assert metrics["median_start_error_ms"] == 300
    assert metrics["per_class"]["takedown"]["f1"] == 1.0


def test_event_metrics_do_not_match_across_matches():
    truth = [{
        "match_id": "match-a", "type": "takedown", "start_ms": 1000, "end_ms": 2000,
    }]
    predictions = [{
        "match_id": "match-b", "type": "takedown", "start_ms": 1000, "end_ms": 2000,
    }]
    metrics = evaluate_events(truth, predictions)
    assert metrics["true_positives"] == 0
    assert metrics["false_positives"] == 1
    assert metrics["false_negatives"] == 1
