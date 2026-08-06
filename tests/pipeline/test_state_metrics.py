import pytest

from ml.evaluation.state_metrics import evaluate_state_predictions


def test_perfect_state_predictions():
    labels = ["neutral", "neutral", "scramble", "top", "top"]
    metrics = evaluate_state_predictions(labels, labels, [0, 100, 200, 300, 400])
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["transition_accuracy"] == 1.0
    assert metrics["median_boundary_error_ms"] == 0


def test_boundary_error_matches_transition_type():
    truth = ["neutral", "neutral", "scramble", "scramble"]
    predicted = ["neutral", "neutral", "neutral", "scramble"]
    metrics = evaluate_state_predictions(truth, predicted, [0, 100, 200, 300])
    assert metrics["median_boundary_error_ms"] == 100


def test_match_boundaries_are_not_counted_as_transitions():
    metrics = evaluate_state_predictions(
        ["neutral", "neutral", "top", "top"],
        ["neutral", "neutral", "top", "top"],
        [0, 100, 0, 100],
        ["a", "a", "b", "b"],
    )
    assert metrics["unmatched_transitions"] == 0


def test_metrics_reject_unknown_labels():
    with pytest.raises(ValueError, match="Unknown"):
        evaluate_state_predictions(["neutral"], ["flying"])

